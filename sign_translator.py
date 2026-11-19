"""
Custom Sign Language Translator
===============================
Built for people with any hand shape, including missing fingers: you record
your OWN signs and the app learns them. Instead of checking which fingers
are up, it compares the whole shape (and movement) of your hand(s).

  - Still signs and moving signs, with one or two hands
  - Hold a sign to add it to a sentence; single letters are fingerspelled
  - Spoken out loud (text-to-speech)
  - Several samples per sign, look-alike warnings, practice mode
  - Separate profiles per person, export / import

Run:   python sign_translator.py        (press H in the app for all controls)
"""

import time
from collections import deque

import cv2

from signlang import config as cfg
from signlang import storage, ui
from signlang.features import frame_features
from signlang.matcher import Recognition, Recognizer, find_lookalikes
from signlang.motion import make_entry, recent_speed
from signlang.practice import PracticeSession
from signlang.recorder import RecordingSession
from signlang.sentence import Sentence, Suggester
from signlang.speech import Speaker
from signlang.tracker import HandTracker

WINDOW    = "Sign Language Translator"
ESC       = 27
ENTER     = (13, 10)
BACKSPACE = (8, 127)

GREEN, RED, YELLOW, GREY = (80, 255, 120), (80, 80, 255), (255, 220, 60), (170, 170, 170)


class App:
    def __init__(self):
        self.threshold = cfg.MATCH_THRESHOLD
        self.profile = storage.load_profile(storage.load_last_profile_name())
        self.recognizer = Recognizer()
        self.sentence = Sentence()
        self.suggester = Suggester()
        self.suggestions = []
        self.speaker = Speaker(enabled=cfg.VOICE_ENABLED)

        self.state = "DETECT"       # DETECT | TYPING | RECORDING | PRACTICE | SUMMARY
        self.prompt = None          # (purpose, title, hint) while TYPING
        self.text = ""
        self.session = None         # RecordingSession
        self.rec_buffer = deque()
        self.practice = None
        self.summary = []
        self.show_help = False
        self.status = ("", YELLOW, 0.0)
        self.lookalikes = set()
        self.refresh_lookalikes()

        print(f"Profile {self.profile['name']}: "
              f"{[s['label'] for s in self.signs] or 'no signs yet'}")
        print("Press H in the window for controls.\n")

    @property
    def signs(self):
        return self.profile["signs"]

    # -- helpers ---------------------------------------------------------------------
    def set_status(self, msg, color=YELLOW, secs=2.5):
        self.status = (msg, color, time.monotonic() + secs)
        print(msg)

    def refresh_lookalikes(self):
        pairs = find_lookalikes(self.signs, self.threshold)
        self.lookalikes = {label for a, b, _ in pairs for label in (a, b)}

    def update_suggestions(self):
        labels = [s["label"] for s in self.signs]
        self.suggestions = self.suggester.suggest(self.sentence.spelling, labels,
                                                  self.profile["history"])

    def remember(self, word):
        self.profile["history"][word] = self.profile["history"].get(word, 0) + 1

    def save(self):
        storage.save_profile(self.profile)

    # -- sentence -----------------------------------------------------------------------
    def on_token(self, label):
        action, text = self.sentence.add(label, self.suggestions)
        if action == "word":
            self.remember(text)
            if cfg.SPEAK_EACH_WORD:
                self.speaker.say(text)
        elif action == "speak":
            self.speak_sentence()
        self.update_suggestions()

    def speak_sentence(self):
        self.sentence.commit_spelling()
        text = self.sentence.text()
        if not text:
            self.set_status("Nothing to say yet", GREY)
        elif not self.speaker.available:
            self.set_status("No text-to-speech voice available", RED)
        else:
            self.speaker.enabled = True if not cfg.SPEAK_EACH_WORD else self.speaker.enabled
            self.speaker.say(text)
            self.set_status(f"Speaking: {text}", GREEN)
        self.update_suggestions()

    def accept_suggestion(self, index):
        if index < len(self.suggestions):
            word = self.sentence.accept(self.suggestions[index])
            self.remember(word)
            if cfg.SPEAK_EACH_WORD:
                self.speaker.say(word)
            self.update_suggestions()

    # -- per frame ------------------------------------------------------------------------
    def step(self, img, hands, now):
        rec = Recognition()
        if self.state == "DETECT":
            rec = self.recognizer.update(hands, now, self.signs, self.threshold)
            if rec.token:
                self.on_token(rec.token)

        elif self.state == "PRACTICE":
            rec = self.recognizer.update(hands, now, self.signs, self.threshold,
                                         hold_secs=cfg.PRACTICE_HOLD_SECS)
            self.practice.update(rec, now)

        elif self.state == "RECORDING":
            entry = make_entry(hands, now) if hands else {"t": now, "hands": []}
            self.rec_buffer.append(entry)
            while self.rec_buffer and self.rec_buffer[0]["t"] < now - 1.0:
                self.rec_buffer.popleft()
            phase = self.session.update(frame_features(hands), entry,
                                        recent_speed(self.rec_buffer, now), now)
            if phase == "done":
                self.finish_recording()
            elif phase == "failed":
                self.set_status(self.session.error, RED, 4)
                self.state = "DETECT"

        self.draw(img, hands, rec, now)

    def draw(self, img, hands, rec, now):
        color = ui.label_color(rec.shown) if rec.shown else (0, 200, 255)
        for hand in hands:
            ui.draw_hand(img, hand, color)

        ui.draw_sidebar(img, self.state, self.profile["name"], self.signs, self.threshold,
                        self.speaker.enabled and self.speaker.available, self.lookalikes)

        if self.state in ("DETECT", "TYPING"):
            if self.state == "DETECT":
                ui.draw_detected(img, rec.shown, rec.confidence)
            ui.draw_bottom_bar(img, self.sentence, self.suggestions,
                               rec.shown if rec.hold_progress > 0 else None,
                               rec.hold_progress, rec.moving)
            if self.state == "TYPING":
                _, title, hint = self.prompt
                ui.draw_text_prompt(img, title, self.text, hint)
        elif self.state == "RECORDING":
            ui.draw_recording(img, self.session, now, len(hands))
        elif self.state == "PRACTICE":
            ui.draw_practice(img, self.practice, rec, now)
        elif self.state == "SUMMARY":
            ui.draw_summary(img, self.summary)

        if self.show_help:
            ui.draw_help(img)
        msg, col, until = self.status
        if now < until:
            ui.draw_status(img, msg, col)

    # -- recording --------------------------------------------------------------------------
    def start_recording(self, label, kind, now):
        if label.startswith("#") and label not in cfg.CONTROL_SIGNS:
            self.set_status("Control signs must be one of: " + " ".join(cfg.CONTROL_SIGNS), RED, 5)
            return
        self.session = RecordingSession(label, kind, now)
        self.rec_buffer.clear()
        self.state = "RECORDING"
        print(f"[*] Recording {kind} sign '{label}'")

    def finish_recording(self):
        sign = self.session.result()
        label = sign["label"]
        result = storage.add_sign(self.profile, sign)
        self.save()
        self.refresh_lookalikes()
        self.recognizer.reset()
        self.state = "DETECT"

        total = len(storage.find_sign(self.profile, label)["samples"])
        kind = "moving" if sign["kind"] == "motion" else "still"
        verb = {"added": "Saved", "merged": "Added samples to", "replaced": "Replaced"}[result]
        warn = find_lookalikes(self.signs, self.threshold, only=label)
        if warn:
            a, b, sim = warn[0]
            other = b if a == label else a
            self.set_status(f"Saved '{label}' - but it looks a lot like '{other}' "
                            f"({int(sim * 100)}% similar). A more distinct pose will work better.",
                            RED, 7)
        else:
            self.set_status(f"[OK] {verb} '{label}' ({kind}, {sign['n_hands']}-hand, "
                            f"{total} samples)", GREEN, 4)

    # -- practice ------------------------------------------------------------------------------
    def start_practice(self, now):
        if not self.signs:
            self.set_status("Record some signs first (N)", GREY)
            return
        self.practice = PracticeSession([s["label"] for s in self.signs],
                                        self.profile["stats"], now)
        self.recognizer.reset()
        self.state = "PRACTICE"

    def end_practice(self):
        self.summary = self.practice.summary_lines()
        self.save()
        self.state = "SUMMARY"

    # -- profiles, export, import ------------------------------------------------------------
    def switch_profile(self, name):
        self.save()
        self.profile = storage.load_profile(name)
        self.save()
        storage.save_last_profile_name(self.profile["name"])
        self.recognizer.reset()
        self.refresh_lookalikes()
        self.update_suggestions()
        self.set_status(f"Profile {self.profile['name']} ({len(self.signs)} signs)", GREEN)

    def export(self):
        if not self.signs:
            self.set_status("No signs to export", GREY)
            return
        path = storage.export_profile(self.profile)
        self.set_status(f"Exported {len(self.signs)} signs to {path}", GREEN, 4)

    def import_from(self, text):
        path = storage.resolve_import_path(text)
        if not path:
            self.set_status(f"No file found for '{text}'", RED, 4)
            return
        try:
            added, merged, skipped = storage.import_signs(self.profile, path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.set_status(f"Couldn't import {path}: {e}", RED, 5)
            return
        self.save()
        self.refresh_lookalikes()
        msg = f"Imported from {path}: {added} new, {merged} merged"
        if skipped:
            msg += f", {skipped} skipped (name used by a different sign)"
        self.set_status(msg, GREEN, 5)

    # -- keyboard ------------------------------------------------------------------------------
    def open_prompt(self, purpose, title, hint=""):
        self.prompt, self.text, self.state = (purpose, title, hint), "", "TYPING"

    def on_typing_key(self, key, now):
        purpose = self.prompt[0]
        if key == ESC:
            self.state = "DETECT"
            self.set_status("Cancelled", GREY)
        elif key in ENTER:
            value = self.text.strip()
            if not value:
                self.set_status("Type something first", RED)
                return
            self.state = "DETECT"
            if purpose in ("static", "motion"):
                self.start_recording(value, purpose, now)
            elif purpose == "delete":
                if storage.delete_sign(self.profile, value):
                    self.save()
                    self.refresh_lookalikes()
                    self.set_status(f"Deleted '{value}'", RED)
                else:
                    self.set_status(f"No sign called '{value}'", GREY)
            elif purpose == "profile":
                self.switch_profile(value)
            elif purpose == "import":
                self.import_from(value)
        elif key in BACKSPACE:
            self.text = self.text[:-1]
        elif 32 <= key < 127:
            ch = chr(key)
            if purpose == "import":
                ok, limit = ch.isalnum() or ch in " -_./\\:", 120
            elif purpose == "profile":
                ch, limit = ch.upper(), 24
                ok = ch.isalnum() or ch in " _-"
            else:
                ch, limit = ch.upper(), cfg.MAX_LABEL_LEN
                ok = ch.isalnum() or ch in " '-" or (ch == "#" and not self.text)
            if ok and len(self.text) < limit:
                self.text += ch

    def on_key(self, key, now):
        """Handle a key press. Returns False to quit."""
        if key == 255:
            return True
        if self.state == "TYPING":
            self.on_typing_key(key, now)
            return True
        if self.state == "SUMMARY":
            self.state = "DETECT"
            return True
        if key == ESC:
            if self.show_help:
                self.show_help = False
            elif self.state == "RECORDING":
                self.state = "DETECT"
                self.set_status("Recording cancelled", GREY)
            elif self.state == "PRACTICE":
                self.end_practice()
            else:
                return False
            return True

        ch = chr(key).lower() if key < 128 else ""
        if ch == "q":
            return False
        if ch == "h":
            self.show_help = not self.show_help
        elif ch in ("+", "="):
            self.threshold = min(cfg.THRESHOLD_MAX, round(self.threshold + cfg.THRESHOLD_STEP, 2))
            self.refresh_lookalikes()
            self.set_status(f"More lenient ({self.threshold:.2f})", GREEN)
        elif ch in ("-", "_"):
            self.threshold = max(cfg.THRESHOLD_MIN, round(self.threshold - cfg.THRESHOLD_STEP, 2))
            self.refresh_lookalikes()
            self.set_status(f"Stricter ({self.threshold:.2f})", YELLOW)

        if self.state != "DETECT":
            return True

        if ch == "n":
            self.open_prompt("static", "Name the new still sign:",
                             "Existing name = add more samples.  Single letters = spelling.")
        elif ch == "m":
            self.open_prompt("motion", "Name the new moving sign:",
                             "You'll do the movement 3 times.")
        elif ch == "x":
            if self.signs:
                self.open_prompt("delete", "Delete which sign?")
            else:
                self.set_status("No signs to delete", GREY)
        elif ch == "p":
            names = ", ".join(storage.list_profiles()) or "none"
            self.open_prompt("profile", "Switch to profile (new name = new profile):",
                             f"Existing: {names}"[:70])
        elif ch == "e":
            self.export()
        elif ch == "i":
            self.open_prompt("import", "Import signs from file:",
                             "A file name in exports/ (start of the name is enough) or a full path")
        elif ch == "l":
            self.start_practice(now)
        elif ch == "s":
            self.speak_sentence()
        elif ch == "v":
            on = self.speaker.toggle()
            self.set_status(f"Voice {'on' if on else 'off'}", GREY)
        elif ch == "c":
            self.sentence.clear()
            self.update_suggestions()
        elif key in BACKSPACE:
            self.sentence.delete_last()
            self.update_suggestions()
        elif ch in ("1", "2", "3"):
            self.accept_suggestion(int(ch) - 1)
        return True

    def close(self):
        self.save()
        storage.save_last_profile_name(self.profile["name"])
        self.speaker.close()


def main():
    cap = cv2.VideoCapture(cfg.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cfg.CAM_INDEX}.")

    tracker = HandTracker()
    app = App()
    cv2.namedWindow(WINDOW)
    prev = time.monotonic()
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                continue
            if cfg.MIRROR:
                img = cv2.flip(img, 1)
            now = time.monotonic()
            hands = tracker.detect(img)
            app.step(img, hands, now)

            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            ui.draw_fps(img, fps)
            cv2.imshow(WINDOW, img)

            key = cv2.waitKey(1) & 0xFF
            if not app.on_key(key, now):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # closed with X
                break
    finally:
        app.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
