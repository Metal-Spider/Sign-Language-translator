"""
Recording a sign: countdown, then several samples in a row with a short
rest between them. Still signs average a burst of frames per sample;
moving signs capture from when the hand starts moving until it stops.
"""

from collections import Counter, deque

from . import config as cfg
from .features import average_static
from .motion import build_motion_sample


class RecordingSession:
    """
    Drive with update() once per frame. `phase` is one of:
      countdown, capture (still), wait_move / moving (motion), rest, done, failed
    """

    def __init__(self, label, kind, now, samples_needed=cfg.SAMPLES_PER_SESSION):
        self.label = label
        self.kind = kind                     # "static" or "motion"
        self.samples_needed = samples_needed
        self.samples = []
        self.n_hands = None                  # locked on the first sample
        self.phase = "countdown"
        self.phase_end = now + cfg.COUNTDOWN_SECS
        self.phase_start = now
        self.frames = []
        self.preroll = deque()
        self.move_start = 0.0
        self.still_since = None
        self.message = ""                    # hint for the UI
        self.error = ""

    # -- helpers ----------------------------------------------------------------
    def _begin_capture(self, now):
        self.frames = []
        self.phase_start = now
        self.phase = "capture" if self.kind == "static" else "wait_move"
        self.message = ""

    def _sample_done(self, sample, now):
        self.samples.append(sample)
        if len(self.samples) >= self.samples_needed:
            self.phase = "done"
        else:
            self.phase, self.phase_end = "rest", now + cfg.REST_SECS
            self.message = "Relax, then do it again"

    def _fail(self, reason):
        self.phase, self.error = "failed", reason

    @property
    def sample_number(self):
        return min(len(self.samples) + 1, self.samples_needed)

    def progress(self):
        """0..1 progress of the current sample (for the progress bar)."""
        if self.phase == "capture":
            return len(self.frames) / cfg.RECORD_FRAMES
        if self.phase == "moving":
            return 1.0
        return 0.0

    # -- per frame ---------------------------------------------------------------
    def update(self, feat, entry, speed, now):
        if self.phase in ("done", "failed"):
            return self.phase

        if self.phase in ("countdown", "rest"):
            if now >= self.phase_end:
                self._begin_capture(now)
            return self.phase

        n = len(feat["hands"]) if feat else 0

        if self.phase == "capture":
            if not n:
                self.message = "Show your hand(s) to the camera"
            else:
                if self.n_hands is None:
                    self.n_hands = n
                if n == self.n_hands:
                    self.frames.append(feat)
                    self.message = ""
                else:
                    self.message = f"Keep {self.n_hands} hand(s) in view"
            if len(self.frames) >= cfg.RECORD_FRAMES:
                self._sample_done(average_static(self.frames), now)

        elif self.phase == "wait_move":
            self.preroll.append(entry)
            while self.preroll and self.preroll[0]["t"] < now - cfg.MOTION_PREROLL_SECS:
                self.preroll.popleft()
            self.message = "Do the movement now"
            if speed >= cfg.MOTION_START_SPEED and entry["hands"]:
                self.phase, self.move_start, self.still_since = "moving", now, None
                self.frames = list(self.preroll)
            elif now - self.phase_start > cfg.MOTION_WAIT_SECS:
                self._fail("No movement detected - try a bigger movement")

        elif self.phase == "moving":
            self.frames.append(entry)
            self.message = "Recording movement..."
            if speed < cfg.STILL_SPEED:
                self.still_since = self.still_since or now
            else:
                self.still_since = None
            elapsed = now - self.move_start
            stopped = (self.still_since is not None
                       and now - self.still_since >= cfg.MOTION_STOP_SECS
                       and elapsed >= cfg.MOTION_MIN_SECS)
            if stopped or elapsed >= cfg.MOTION_MAX_SECS:
                self._finish_motion(now)

        return self.phase

    def _finish_motion(self, now):
        frames = self.frames
        if self.still_since is not None:
            frames = [e for e in frames if e["t"] <= self.still_since]
        counts = Counter(len(e["hands"]) for e in frames if e["hands"])
        if not counts:
            self.phase, self.phase_end = "rest", now + cfg.REST_SECS
            self.message = "Lost your hands - let's try that sample again"
            return
        n = counts.most_common(1)[0][0]
        if self.n_hands is not None and n != self.n_hands:
            self.phase, self.phase_end = "rest", now + cfg.REST_SECS
            self.message = f"Use {self.n_hands} hand(s) like the first sample"
            return
        sample = build_motion_sample(frames, n)
        if sample is None:
            self.phase, self.phase_end = "rest", now + cfg.REST_SECS
            self.message = "Too short - try that sample again"
            return
        self.n_hands = n
        self._sample_done(sample, now)

    def result(self):
        """The sign dict built from this session (only valid when done)."""
        return {"label": self.label, "kind": self.kind,
                "n_hands": self.n_hands, "samples": self.samples}
