"""
Text-to-speech that never blocks the camera loop.

Windows: a persistent PowerShell process using the built-in System.Speech
         voices (no extra packages needed).
macOS:   the built-in `say` command.
Other:   pyttsx3 if installed, otherwise espeak.
"""

import queue
import shutil
import subprocess
import sys
import threading

_PS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Volume = {volume}; "
    "while ($true) {{ $l = [Console]::In.ReadLine(); "
    "if ($l -eq $null) {{ break }}; if ($l.Length -gt 0) {{ $s.Speak($l) }} }}"
)


def _clean(text):
    """Keep speech input to simple characters (labels are already restricted)."""
    return "".join(c for c in text if c.isalnum() or c in " '-").strip().lower()


class Speaker:
    def __init__(self, enabled=True, volume=100, backend=None):
        self.enabled = enabled
        self.available = True
        self.volume = volume
        self._backend = backend              # injectable for tests: callable(text)
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text):
        text = _clean(text)
        if self.enabled and self.available and text:
            self._queue.put(text)

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def close(self):
        self._queue.put(None)

    # -- worker thread -----------------------------------------------------------
    def _make_backend(self):
        if self._backend:
            return self._backend
        if sys.platform == "win32":
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 _PS_SCRIPT.format(volume=int(self.volume))],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

            def speak(text):
                proc.stdin.write(text + "\n")
                proc.stdin.flush()
            return speak
        if sys.platform == "darwin" and shutil.which("say"):
            return lambda text: subprocess.run(["say", text])
        try:
            import pyttsx3
            engine = pyttsx3.init()

            def speak(text):
                engine.say(text)
                engine.runAndWait()
            return speak
        except Exception:
            pass
        if shutil.which("espeak"):
            return lambda text: subprocess.run(["espeak", text])
        return None

    def _run(self):
        try:
            speak = self._make_backend()
        except Exception as e:
            print(f"[voice] unavailable: {e}")
            speak = None
        if speak is None:
            self.available = False
            print("[voice] No text-to-speech engine found - voice disabled.")
            return
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                speak(text)
            except Exception as e:
                print(f"[voice] error: {e}")
                self.available = False
                break
