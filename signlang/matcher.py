"""
Recognising signs from live hands.

A saved sign looks like:
    {"label": "HELLO", "kind": "static" | "motion", "n_hands": 1 | 2,
     "samples": [sample, ...]}
Several samples per sign are kept; the closest one counts.
"""

import itertools
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

from . import config as cfg
from .features import frame_features, hand_distance, static_distance
from .motion import (make_entry, motion_distance_between, motion_sample_distance,
                     movement_extent, recent_speed)

INF = float("inf")


# -- Still signs -----------------------------------------------------------------

def match_static(feat, signs, threshold):
    """
    Return (label, distance) of the best still sign under `threshold`, or
    (None, best_distance). With two hands visible, two-hand signs are tried
    first, then one-hand signs against either hand.
    """
    if feat is None:
        return None, INF
    statics = [s for s in signs if s["kind"] == "static"]
    best_label, best_dist = None, INF

    if len(feat["hands"]) == 2:
        for s in statics:
            if s["n_hands"] == 2:
                for sample in s["samples"]:
                    d = static_distance(feat, sample)
                    if d < best_dist:
                        best_label, best_dist = s["label"], d
        if best_dist <= threshold:
            return best_label, best_dist

    for s in statics:
        if s["n_hands"] == 1:
            for sample in s["samples"]:
                for shape in feat["hands"]:
                    d = hand_distance(shape, sample["hands"][0])
                    if d < best_dist:
                        best_label, best_dist = s["label"], d

    if best_dist <= threshold:
        return best_label, best_dist
    return None, best_dist


# -- Moving signs ----------------------------------------------------------------

def match_motion(entries, now, signs, threshold):
    best_label, best_dist = None, INF
    for s in signs:
        if s["kind"] != "motion":
            continue
        for sample in s["samples"]:
            d = motion_sample_distance(entries, now, sample, s["n_hands"])
            if d < best_dist:
                best_label, best_dist = s["label"], d
    if best_dist <= threshold:
        return best_label, best_dist
    return None, best_dist


# -- Live recogniser ---------------------------------------------------------------

@dataclass
class Recognition:
    shown: Optional[str] = None      # sign currently recognised (for display)
    confidence: float = 0.0
    hold_progress: float = 0.0       # 0..1 towards adding a still sign
    token: Optional[str] = None      # sign to add to the sentence this frame
    source: Optional[str] = None     # "static" or "motion"
    moving: bool = False
    speed: float = 0.0


class Recognizer:
    """
    Smooths detections, handles hold-to-add for still signs (only while the
    hands are still) and fires moving signs as soon as they're completed.
    The same still sign is only added again after it's released.
    """

    def __init__(self):
        self.buffer = deque()
        self.reset()

    def reset(self):
        self.buffer.clear()
        self.history = deque(maxlen=cfg.SMOOTH_FRAMES)
        self.last_dist = {}
        self.hold_label = None
        self.hold_start = 0.0
        self.last_added = None
        self.cooldown_until = 0.0
        self.flash = (None, 0.0, 0.0)   # (label, confidence, until)
        self.frame_no = 0

    def _moved_recently(self, now):
        recent = [e for e in self.buffer if e["t"] >= now - 1.5]
        if not recent or not recent[-1]["hands"]:
            return False
        return movement_extent(recent, len(recent[-1]["hands"])) >= 0.5

    def update(self, hands, now, signs, threshold, hold_secs=cfg.HOLD_TO_ADD_SECS):
        self.frame_no += 1
        self.buffer.append(make_entry(hands, now) if hands else {"t": now, "hands": []})
        while self.buffer and self.buffer[0]["t"] < now - cfg.MOTION_BUFFER_SECS:
            self.buffer.popleft()

        speed = recent_speed(self.buffer, now)
        res = Recognition(speed=speed, moving=speed >= cfg.STILL_SPEED)

        # Moving signs fire immediately when a recorded movement is completed
        motion_threshold = threshold * cfg.MOTION_THRESHOLD_SCALE
        if (any(s["kind"] == "motion" for s in signs)
                and now >= self.cooldown_until
                and self.frame_no % cfg.MOTION_CHECK_EVERY == 0
                and self._moved_recently(now)):
            label, dist = match_motion(list(self.buffer), now, signs, motion_threshold)
            if label:
                conf = max(0.0, 1.0 - dist / motion_threshold)
                self.buffer.clear()
                self.history.clear()
                self.hold_label, self.last_added = None, None
                self.cooldown_until = now + cfg.MOTION_COOLDOWN_SECS
                self.flash = (label, conf, now + 1.0)
                res.shown, res.confidence = label, conf
                res.token, res.source = label, "motion"
                return res

        flash_label, flash_conf, flash_until = self.flash
        if flash_label and now < flash_until:
            res.shown, res.confidence = flash_label, flash_conf
            return res

        # Still signs: smoothed, and added after being held steady
        label, dist = match_static(frame_features(hands), signs, threshold)
        self.history.append(label)
        if label:
            self.last_dist[label] = dist
        top, count = Counter(self.history).most_common(1)[0]
        stable = top if top is not None and count >= cfg.SMOOTH_FRAMES // 2 + 1 else None

        if stable != self.last_added:
            self.last_added = None            # released -> can be added again
        res.shown = stable
        if stable:
            res.confidence = max(0.0, 1.0 - self.last_dist[stable] / threshold)

        if stable is None or res.moving:
            self.hold_label = None
        elif stable != self.hold_label:
            self.hold_label, self.hold_start = stable, now

        if self.hold_label and self.last_added is None:
            res.hold_progress = min(1.0, (now - self.hold_start) / max(hold_secs, 1e-6))
            if res.hold_progress >= 1.0:
                res.token, res.source = stable, "static"
                self.last_added = stable
        return res


# -- Look-alike detection -----------------------------------------------------------

def sign_distance(a, b):
    """Smallest distance between any samples of two signs of the same type."""
    if a["kind"] != b["kind"] or a["n_hands"] != b["n_hands"]:
        return INF
    best = INF
    for sa in a["samples"]:
        for sb in b["samples"]:
            if a["kind"] == "static":
                d = static_distance({"hands": sa["hands"], "rel": sa.get("rel")}, sb)
            else:
                d = motion_distance_between(sa, sb)
            best = min(best, d)
    return best


def find_lookalikes(signs, threshold, only=None):
    """
    Return [(label_a, label_b, similarity 0..1)] for signs that are close
    enough to be confused. If `only` is given, only pairs involving it.
    """
    pairs = []
    for a, b in itertools.combinations(signs, 2):
        if only and only not in (a["label"], b["label"]):
            continue
        limit = threshold * cfg.LOOKALIKE_RATIO
        if a["kind"] == "motion":
            limit *= cfg.MOTION_THRESHOLD_SCALE
        d = sign_distance(a, b)
        if d < limit:
            pairs.append((a["label"], b["label"], 1.0 - d / limit))
    return sorted(pairs, key=lambda p: -p[2])
