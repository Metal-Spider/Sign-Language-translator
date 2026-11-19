"""
Moving signs: a short movement (e.g. a wave) recorded as a sequence of
frames and matched with Dynamic Time Warping (DTW), which compares two
movements even when one is done faster or slower than the other.

Live frames are kept as "entries":
    {"t": seconds, "hands": [{"xy": 42 floats, "wrist": (x, y),
                              "palm": float, "side": str or None}, ...]}
"""

import math
from collections import Counter

import numpy as np

from . import config as cfg
from .features import hand_shape, order_hands, palm_size


def make_entry(hands, t):
    """Build a live-buffer entry from tracked hands."""
    out = []
    for h in order_hands(hands):
        shape = hand_shape(h)
        if shape is None:
            return {"t": t, "hands": []}
        out.append({"xy": shape["xy"], "wrist": (h.lm[0][0], h.lm[0][1]),
                    "palm": palm_size(h.lm), "side": shape["side"]})
    return {"t": t, "hands": out}


def recent_speed(entries, now, window=0.25):
    """
    How fast the hands are moving, in palm sizes per second, over the last
    `window` seconds (the fastest hand counts).
    """
    recent = [e for e in entries if e["t"] >= now - window and e["hands"]]
    if len(recent) < 2:
        return 0.0
    first, last = recent[0], recent[-1]
    dt = last["t"] - first["t"]
    if dt <= 1e-6 or len(first["hands"]) != len(last["hands"]):
        return 0.0
    speed = 0.0
    for a, b in zip(first["hands"], last["hands"]):
        palm = (a["palm"] + b["palm"]) / 2
        dist = math.hypot(b["wrist"][0] - a["wrist"][0], b["wrist"][1] - a["wrist"][1])
        speed = max(speed, dist / palm / dt)
    return speed


def _frames_with(entries, n_hands):
    return [e for e in entries if len(e["hands"]) == n_hands]


def movement_extent(entries, n_hands):
    """Largest distance (palm sizes) any wrist travels from where it started."""
    es = _frames_with(entries, n_hands)
    if len(es) < 2:
        return 0.0
    palm = np.mean([h["palm"] for e in es for h in e["hands"]])
    extent = 0.0
    for k in range(n_hands):
        x0, y0 = es[0]["hands"][k]["wrist"]
        for e in es:
            x, y = e["hands"][k]["wrist"]
            extent = max(extent, math.hypot(x - x0, y - y0) / palm)
    return float(extent)


def build_sequence(entries, n_hands):
    """
    Entries -> (times, vectors). Each vector holds, per hand, the hand shape
    and the wrist's path relative to where the movement started.
    """
    es = _frames_with(entries, n_hands)
    if len(es) < 6:
        return None
    palm = float(np.mean([h["palm"] for e in es for h in e["hands"]]))
    start = [h["wrist"] for h in es[0]["hands"]]
    times, vecs = [], []
    for e in es:
        v = []
        for k, h in enumerate(e["hands"]):
            v.extend(np.asarray(h["xy"]) * cfg.MOTION_SHAPE_WEIGHT)
            v.append((h["wrist"][0] - start[k][0]) / palm * cfg.MOTION_TRAJ_WEIGHT)
            v.append((h["wrist"][1] - start[k][1]) / palm * cfg.MOTION_TRAJ_WEIGHT)
        times.append(e["t"])
        vecs.append(v)
    return np.asarray(times), np.asarray(vecs, dtype=float)


def resample(times, vecs, n=cfg.MOTION_RESAMPLE):
    """Resample a sequence to n evenly-timed points."""
    t = times - times[0]
    if t[-1] <= 1e-6:
        return None
    grid = np.linspace(0.0, t[-1], n)
    return np.stack([np.interp(grid, t, vecs[:, d]) for d in range(vecs.shape[1])], axis=1)


def dtw_distance(a, b, band=cfg.MOTION_BAND):
    """Average per-step DTW cost between two (n, d) sequences."""
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)).tolist()
    inf = float("inf")
    D = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    band = max(band, abs(n - m))
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            D[i][j] = cost[i - 1][j - 1] + min(D[i - 1][j - 1], D[i - 1][j], D[i][j - 1])
    return D[n][m] / n


def _majority_sides(entries, n_hands):
    sides = []
    for k in range(n_hands):
        votes = [e["hands"][k]["side"] for e in entries
                 if len(e["hands"]) == n_hands and e["hands"][k]["side"]]
        sides.append(Counter(votes).most_common(1)[0][0] if votes else None)
    return sides


def build_motion_sample(entries, n_hands):
    """Recorded entries -> a moving-sign sample dict, or None if too short."""
    seq = build_sequence(entries, n_hands)
    if seq is None:
        return None
    times, vecs = seq
    frames = resample(times, vecs)
    if frames is None:
        return None
    return {
        "frames": frames.tolist(),
        "duration": float(times[-1] - times[0]),
        "extent": movement_extent(entries, n_hands),
        "sides": _majority_sides(entries, n_hands),
    }


def motion_sample_distance(entries, now, sample, n_hands):
    """
    Best DTW distance between the most recent live movement and a sample,
    trying a few signing speeds. Returns inf when the live movement can't
    be the sign (wrong hand count, too small, too short).
    """
    template = np.asarray(sample["frames"], dtype=float)
    best = float("inf")
    for scale in cfg.MOTION_SPEED_SCALES:
        dur = sample["duration"] * scale
        window = [e for e in entries if e["t"] >= now - dur]
        if not window or window[-1]["t"] - window[0]["t"] < dur * 0.7:
            continue
        matching = _frames_with(window, n_hands)
        if len(matching) < 0.6 * len(window):
            continue
        if movement_extent(matching, n_hands) < sample["extent"] * cfg.MOTION_MIN_EXTENT:
            continue
        seq = build_sequence(matching, n_hands)
        if seq is None:
            continue
        live = resample(*seq)
        if live is None:
            continue
        d = dtw_distance(live, template) / math.sqrt(n_hands)
        live_sides = _majority_sides(matching, n_hands)
        for ls, ss in zip(live_sides, sample.get("sides") or []):
            if ls and ss and ls != ss:
                d += cfg.SIDE_PENALTY
                break
        best = min(best, d)
    return best


def motion_distance_between(sample_a, sample_b):
    """DTW distance between two stored samples (for look-alike checks)."""
    a = np.asarray(sample_a["frames"], dtype=float)
    b = np.asarray(sample_b["frames"], dtype=float)
    if a.shape[1] != b.shape[1]:
        return float("inf")
    n_hands = max(1, a.shape[1] // 44)
    return dtw_distance(a, b) / math.sqrt(n_hands)
