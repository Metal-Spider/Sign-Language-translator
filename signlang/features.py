"""
Turning hand landmarks into comparable features, and measuring how
different two hand shapes are.

A hand "shape" is a dict:
    {"xy": 42 floats, "z": 21 floats or None, "side": "Left"/"Right"/None}
with every point relative to the wrist and divided by the palm size, so
position in the frame and distance from the camera don't matter.
"""

import math

import numpy as np

from . import config as cfg

_ANGLES = np.radians(cfg.ROTATION_ANGLES)
# (K, 2, 2) rotation matrices used for wrist-tilt tolerance
_ROTATIONS = np.stack([
    np.stack([np.cos(_ANGLES), -np.sin(_ANGLES)], axis=1),
    np.stack([np.sin(_ANGLES),  np.cos(_ANGLES)], axis=1),
], axis=1)


def palm_size(lm):
    return math.hypot(lm[9][0] - lm[0][0], lm[9][1] - lm[0][1])


def hand_shape(hand):
    """Hand -> normalised shape dict, or None for a degenerate frame."""
    lm = hand.lm
    ps = palm_size(lm)
    if ps < 1e-6:
        return None
    ox, oy, oz = lm[0][0], lm[0][1], lm[0][2]
    xy, z = [], []
    for p in lm:
        xy.append((p[0] - ox) / ps)
        xy.append((p[1] - oy) / ps)
        z.append((p[2] - oz) / ps)
    side = hand.side if hand.side in ("Left", "Right") else None
    return {"xy": xy, "z": z, "side": side}


def order_hands(hands):
    """
    Put two hands in a consistent order: the person's left hand first when
    handedness is clear, otherwise left-to-right on screen.
    """
    if len(hands) == 2 and {h.side for h in hands} == {"Left", "Right"}:
        return sorted(hands, key=lambda h: 0 if h.side == "Left" else 1)
    return sorted(hands, key=lambda h: h.lm[0][0])


def frame_features(hands):
    """
    Features for one frame: {"hands": [shape, ...], "rel": [dx, dy] or None}.
    "rel" is the offset from the first wrist to the second in palm sizes.
    """
    if not hands:
        return None
    ordered = order_hands(hands)
    shapes = [hand_shape(h) for h in ordered]
    if any(s is None for s in shapes):
        return None
    rel = None
    if len(ordered) == 2:
        a, b = ordered[0].lm, ordered[1].lm
        scale = (palm_size(a) + palm_size(b)) / 2
        rel = [(b[0][0] - a[0][0]) / scale, (b[0][1] - a[0][1]) / scale]
    return {"hands": shapes, "rel": rel}


# -- Distances -----------------------------------------------------------------

def xy_distance(a, b):
    """Rotation-tolerant distance between two 42-float xy vectors."""
    pa = np.asarray(a, dtype=float).reshape(-1, 2)
    pb = np.asarray(b, dtype=float).reshape(-1, 2)
    rotated = np.einsum("kij,nj->kni", _ROTATIONS, pa)        # (K, 21, 2)
    d = np.sqrt(((rotated - pb) ** 2).sum(axis=(1, 2)))
    return float(d.min())


def hand_distance(live, saved):
    """Distance between two hand shapes: outline + depth + which hand."""
    d = xy_distance(live["xy"], saved["xy"])
    if live.get("z") is not None and saved.get("z") is not None:
        d += cfg.Z_WEIGHT * float(np.linalg.norm(np.subtract(live["z"], saved["z"])))
    if live.get("side") and saved.get("side") and live["side"] != saved["side"]:
        d += cfg.SIDE_PENALTY
    return d


def static_distance(feat, sample):
    """
    Distance between live features and one saved still-sign sample.
    Two-hand signs use the WORSE hand (both must match) plus the hand
    spacing; both pairings are tried so crossed or mislabelled hands
    still work.
    """
    live, saved = feat["hands"], sample["hands"]
    if len(live) != len(saved):
        return float("inf")
    if len(saved) == 1:
        return hand_distance(live[0], saved[0])

    best = float("inf")
    for swap in (False, True):
        a, b = (live[1], live[0]) if swap else (live[0], live[1])
        d = max(hand_distance(a, saved[0]), hand_distance(b, saved[1]))
        rel, srel = feat.get("rel"), sample.get("rel")
        if rel is not None and srel is not None:
            r = (-rel[0], -rel[1]) if swap else rel
            d += cfg.REL_WEIGHT * math.hypot(r[0] - srel[0], r[1] - srel[1])
        best = min(best, d)
    return best


# -- Averaging (recording) ------------------------------------------------------

def _mean(vectors):
    return np.mean(np.asarray(vectors, dtype=float), axis=0).tolist()


def average_static(frames):
    """Average feature frames (same hand count) into one still-sign sample."""
    n = len(frames[0]["hands"])
    hands = []
    for i in range(n):
        shapes = [f["hands"][i] for f in frames]
        zs = [s["z"] for s in shapes if s.get("z") is not None]
        sides = [s["side"] for s in shapes if s.get("side")]
        hands.append({
            "xy": _mean([s["xy"] for s in shapes]),
            "z": _mean(zs) if len(zs) == len(shapes) else None,
            "side": max(set(sides), key=sides.count) if sides else None,
        })
    rel = None
    if all(f.get("rel") is not None for f in frames) and n == 2:
        rel = _mean([f["rel"] for f in frames])
    return {"hands": hands, "rel": rel}
