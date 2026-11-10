"""
Custom Sign Language Translator
===============================
Designed for users with any hand shape - including missing fingers.
Instead of detecting "fingers up", it records the full hand shape(s)
and matches against them, so any pose can be assigned any word.

Signs can use ONE or TWO hands. The number of hands is detected
automatically while recording, so both kinds of sign coexist.

CONTROLS
--------
  N          New sign: type a word, press ENTER, then hold the pose
  X          Delete a sign: type its word, press ENTER
  + / -      More / less lenient matching
  ESC        Cancel a recording (or quit when idle)
  Q          Quit

  While typing a word:  ENTER = confirm, BACKSPACE = erase, ESC = cancel
"""

import cv2
import json
import math
import os
import time
import zlib
from collections import Counter, deque

from cvzone.HandTrackingModule import HandDetector

# -- Config -------------------------------------------------------------------
CAM_INDEX       = 0
CAM_WIDTH       = 1280
CAM_HEIGHT      = 720
DETECT_CON      = 0.7
GESTURES_FILE   = "gestures.json"

# How many frames to average when recording (more = smoother, slower)
RECORD_FRAMES   = 25

# Seconds to get your hand(s) into position after naming a sign
COUNTDOWN_SECS  = 3

# Max normalised distance to count as a match (higher = more lenient)
MATCH_THRESHOLD = 1.40

# How much the position of one hand relative to the other matters
# for two-hand signs (0 = ignore it entirely)
REL_WEIGHT      = 0.5

# Frames used to smooth detection so the result doesn't flicker
SMOOTH_FRAMES   = 5

MAX_LABEL_LEN   = 16
SIDEBAR_W       = 260
# -----------------------------------------------------------------------------

SKELETON_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# BGR colours; each word gets a stable colour from this palette
PALETTE = [
    (255,  80,  80), ( 80, 200, 255), ( 80, 255, 160), (255, 200,  60),
    (200,  80, 255), (255, 120, 200), (120, 255, 255), (180, 255,  80),
]

STATE_BADGES = {
    "DETECT":    (( 40, 150,  40), "DETECTING"),
    "TYPING":    (( 20, 140, 200), "TYPE A WORD"),
    "COUNTDOWN": ((180, 110,  20), "GET READY"),
    "RECORDING": (( 40,  40, 200), "RECORDING"),
}


def label_color(label):
    return PALETTE[zlib.crc32(label.encode()) % len(PALETTE)]


# -- Landmark features --------------------------------------------------------

def normalize_landmarks(lm_list):
    """
    Turn raw pixel landmarks into a position- and scale-invariant vector.

    Steps:
      1. Translate all points so wrist (LM 0) is the origin.
      2. Scale by the wrist -> middle-MCP (LM 9) distance so hand size
         doesn't matter (works at any distance from camera).
      3. Flatten to a list of 42 floats: [x0,y0, x1,y1, ... x20,y20].

    Returns None when the reference distance is too small (bad frame).
    """
    ox, oy    = lm_list[0][0], lm_list[0][1]
    ref_dist  = math.hypot(lm_list[9][0] - ox, lm_list[9][1] - oy)
    if ref_dist < 1e-6:
        return None
    vec = []
    for lm in lm_list:
        vec.append((lm[0] - ox) / ref_dist)
        vec.append((lm[1] - oy) / ref_dist)
    return vec


def palm_size(lm_list):
    return math.hypot(lm_list[9][0] - lm_list[0][0], lm_list[9][1] - lm_list[0][1])


def extract_features(hands):
    """
    Build features from the detected hands.

    Hands are ordered left-to-right on screen so two-hand signs are
    compared consistently. For two hands we also store "rel": the offset
    from the first wrist to the second, measured in palm sizes, so the
    spacing/arrangement of the hands is part of the sign.

    Returns {"hands": [vec, ...], "rel": [dx, dy] or None}, or None.
    """
    if not hands:
        return None
    lms  = sorted((h["lmList"] for h in hands), key=lambda lm: lm[0][0])
    vecs = [normalize_landmarks(lm) for lm in lms]
    if any(v is None for v in vecs):
        return None
    rel = None
    if len(lms) == 2:
        scale = (palm_size(lms[0]) + palm_size(lms[1])) / 2
        rel = [(lms[1][0][0] - lms[0][0][0]) / scale,
               (lms[1][0][1] - lms[0][0][1]) / scale]
    return {"hands": vecs, "rel": rel}


def average_vectors(vecs):
    """Element-wise average of a list of equal-length float lists."""
    return [sum(col) / len(col) for col in zip(*vecs)]


def average_features(samples):
    """Average a list of feature dicts that all have the same hand count."""
    n_hands = len(samples[0]["hands"])
    hands = [average_vectors([s["hands"][i] for s in samples])
             for i in range(n_hands)]
    rel = None
    if samples[0]["rel"] is not None:
        rel = average_vectors([s["rel"] for s in samples])
    return {"hands": hands, "rel": rel}


def vec_distance(a, b):
    """
    Rotation-tolerant Euclidean distance between two 42-float landmark vectors.
    Checks wrist tilt angles from -30 to +30 degrees so the user doesn't have
    to hold their wrist at the exact same angle as when they recorded the sign.
    """
    best = float("inf")
    pairs_a = [(a[i], a[i + 1]) for i in range(0, len(a), 2)]
    pairs_b = [(b[i], b[i + 1]) for i in range(0, len(b), 2)]

    for deg in (-30, -20, -10, -5, 0, 5, 10, 20, 30):
        rad = math.radians(deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        total = 0.0
        for (ax, ay), (bx, by) in zip(pairs_a, pairs_b):
            rx = ax * cos_a - ay * sin_a
            ry = ax * sin_a + ay * cos_a
            total += (rx - bx) ** 2 + (ry - by) ** 2
        d = math.sqrt(total)
        if d < best:
            best = d
    return best


# -- Matching -----------------------------------------------------------------

def sign_distance(hand_vecs, rel, sign):
    """
    Distance between live hands and a saved sign with the same hand count.
    For two-hand signs the WORSE of the two hands is used, so both hands
    must match (one good hand can't hide a wrong one). This keeps the same
    threshold meaningful for 1- and 2-hand signs. The hand-spacing term
    is then added on top.
    """
    d = max(vec_distance(a, b) for a, b in zip(hand_vecs, sign["hands"]))
    if rel is not None and sign.get("rel") is not None:
        d += REL_WEIGHT * math.hypot(rel[0] - sign["rel"][0],
                                     rel[1] - sign["rel"][1])
    return d


def match_sign(feat, signs):
    """
    Return (label, distance) of the best sign under MATCH_THRESHOLD,
    or (None, best_distance).

    With two hands visible, two-hand signs are tried first; if none match,
    one-hand signs are tried against either hand (so a one-hand sign still
    works when the other hand happens to be in view).
    """
    best_label, best_dist = None, float("inf")
    if feat is None:
        return None, best_dist

    if len(feat["hands"]) == 2:
        for s in signs:
            if len(s["hands"]) == 2:
                d = sign_distance(feat["hands"], feat["rel"], s)
                if d < best_dist:
                    best_label, best_dist = s["label"], d
        if best_dist <= MATCH_THRESHOLD:
            return best_label, best_dist

    for s in signs:
        if len(s["hands"]) == 1:
            for hv in feat["hands"]:
                d = vec_distance(hv, s["hands"][0])
                if d < best_dist:
                    best_label, best_dist = s["label"], d

    if best_dist <= MATCH_THRESHOLD:
        return best_label, best_dist
    return None, best_dist


# -- Persistence --------------------------------------------------------------

def load_signs(path):
    """
    Load saved signs. Also accepts the old A-E format ({"A": [42 floats]})
    and converts it to one-hand signs automatically.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "signs" in data:
        return data["signs"]
    return [{"label": k, "hands": [v], "rel": None} for k, v in data.items()]


def save_signs(path, signs):
    with open(path, "w") as f:
        json.dump({"version": 2, "signs": signs}, f, indent=2)


# -- Drawing ------------------------------------------------------------------

def draw_skeleton(img, lm_list, color=(0, 200, 255)):
    for a, b in SKELETON_CONNECTIONS:
        cv2.line(img, tuple(lm_list[a][:2]), tuple(lm_list[b][:2]), color, 2)
    for lm in lm_list:
        cv2.circle(img, (lm[0], lm[1]), 5, (255, 255, 255), cv2.FILLED)


def draw_sidebar(img, signs, state):
    """Translucent left sidebar with state badge and saved-sign list."""
    h       = img.shape[0]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (SIDEBAR_W, h), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

    badge_col, badge_txt = STATE_BADGES[state]
    y = 38
    cv2.rectangle(img, (8, y - 24), (SIDEBAR_W - 8, y + 8), badge_col, cv2.FILLED)
    cv2.putText(img, "  " + badge_txt, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    y += 44

    cv2.putText(img, f"Saved signs ({len(signs)})", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
    y += 26

    hints = ["N      new sign",
             "X      delete sign",
             "+ / -  more/less lenient",
             "ESC    cancel recording",
             "Q      quit"]
    hints_top = h - len(hints) * 19 - 40
    max_rows  = max(0, (hints_top - y - 10) // 26)

    ordered = sorted(signs, key=lambda s: s["label"])
    if not ordered:
        cv2.putText(img, "  none yet - press N", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 90, 90), 1)
    for i, s in enumerate(ordered):
        if i == max_rows - 1 and len(ordered) > max_rows:
            cv2.putText(img, f"  ... +{len(ordered) - i} more", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 120), 1)
            break
        col = label_color(s["label"])
        cv2.putText(img, "  " + s["label"][:12], (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, col, 1)
        cv2.putText(img, f"{len(s['hands'])}H", (SIDEBAR_W - 45, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)
        y += 26

    y = hints_top
    cv2.putText(img, f"Leniency: {MATCH_THRESHOLD:.2f}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 180), 1)
    y += 24
    for hint in hints:
        cv2.putText(img, hint, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 110, 110), 1)
        y += 19


def draw_centered(img, text, y, scale, color, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw text centred in the camera area (right of the sidebar)."""
    w  = img.shape[1]
    cx = SIDEBAR_W + (w - SIDEBAR_W) // 2
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, (cx - tw // 2, y), font, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, (cx - tw // 2, y), font, scale, color, thick)


def draw_text_prompt(img, purpose, text):
    """Centered text-entry box for naming / deleting a sign."""
    h, w   = img.shape[:2]
    cx     = SIDEBAR_W + (w - SIDEBAR_W) // 2
    bw, bh = 560, 170
    x0, y0 = cx - bw // 2, h // 2 - bh // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (200, 200, 200), 1)

    title  = "Name the new sign:" if purpose == "record" else "Delete which sign?"
    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    draw_centered(img, title, y0 + 40, 0.75, (220, 220, 220))
    draw_centered(img, text + cursor, y0 + 100, 1.4, (80, 255, 200), 2)
    draw_centered(img, "ENTER confirm  |  BACKSPACE erase  |  ESC cancel",
                  y0 + 145, 0.48, (150, 150, 150))


def draw_countdown(img, label, secs_left, n_hands):
    h = img.shape[0]
    draw_centered(img, f"Get ready to sign '{label}'", h // 2 - 120, 0.9, (255, 220, 120), 2)
    draw_centered(img, str(secs_left), h // 2 + 60, 6, (255, 255, 255), 10,
                  cv2.FONT_HERSHEY_DUPLEX)
    hint = (f"{n_hands} hand(s) visible - will record a {n_hands}-hand sign"
            if n_hands else "Show the hand(s) you want to use")
    draw_centered(img, hint, h // 2 + 130, 0.65, (200, 200, 200))


def draw_record_progress(img, label, buf_len, locked_hands):
    """Progress bar at the bottom of the camera area while recording."""
    h      = img.shape[0]
    bx, by = SIDEBAR_W + 20, h - 60
    bw     = 320
    filled = int(bw * buf_len / RECORD_FRAMES)
    col    = label_color(label)
    hands_txt = f"{locked_hands}-hand" if locked_hands else "waiting for hands"

    cv2.rectangle(img, (bx, by), (bx + bw, by + 20), (40, 40, 40), cv2.FILLED)
    cv2.rectangle(img, (bx, by), (bx + filled, by + 20), col, cv2.FILLED)
    cv2.putText(img, f"Recording '{label}' ({hands_txt})  {buf_len}/{RECORD_FRAMES}",
                (bx, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def draw_detected(img, label, dist):
    """Big centred word with a confidence bar, auto-sized to fit."""
    if label is None:
        return
    h, w   = img.shape[:2]
    area_w = w - SIDEBAR_W
    cx     = SIDEBAR_W + area_w // 2
    conf   = max(0.0, 1.0 - dist / MATCH_THRESHOLD)
    col    = label_color(label)

    font = cv2.FONT_HERSHEY_DUPLEX
    (tw1, _), _ = cv2.getTextSize(label, font, 1.0, 2)
    scale = min(9.0, (area_w * 0.85) / max(tw1, 1))
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    tx, ty = cx - tw // 2, h // 2 + th // 2
    cv2.putText(img, label, (tx + 5, ty + 5), font, scale, (0, 0, 0), thick + 6)
    cv2.putText(img, label, (tx, ty), font, scale, col, thick)

    bar_w  = 320
    bx, by = cx - bar_w // 2, ty + 28
    cv2.rectangle(img, (bx, by), (bx + bar_w, by + 20), (40, 40, 40), cv2.FILLED)
    cv2.rectangle(img, (bx, by), (bx + int(bar_w * conf), by + 20), col, cv2.FILLED)
    cv2.putText(img, f"{int(conf * 100)}% confidence",
                (bx, by + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 1)


# -- Main ---------------------------------------------------------------------

def main():
    global MATCH_THRESHOLD
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAM_INDEX}.")

    detector = HandDetector(detectionCon=DETECT_CON, maxHands=2)
    signs    = load_signs(GESTURES_FILE)

    state          = "DETECT"   # DETECT | TYPING | COUNTDOWN | RECORDING
    prompt_purpose = None       # "record" | "delete"
    text_buffer    = ""
    record_label   = None
    record_hands   = None       # hand count locked in on first recorded frame
    record_buffer  = []
    countdown_end  = 0.0

    history   = deque(maxlen=SMOOTH_FRAMES)
    last_dist = {}

    status_msg, status_color, status_until = "", (255, 220, 60), 0.0
    prev_time = time.time()

    def set_status(msg, col=(255, 220, 60), duration=2.0):
        nonlocal status_msg, status_color, status_until
        status_msg, status_color, status_until = msg, col, time.time() + duration

    print(f"Loaded signs: {[s['label'] for s in signs] or 'none'}")
    print("N=new sign | X=delete | +/-=leniency | Q=quit\n")

    while True:
        success, img = cap.read()
        if not success:
            continue

        hands, img = detector.findHands(img, draw=False, flipType=True)
        feat = extract_features(hands)

        # -- Detection (smoothed) ---------------------------------------------
        shown_label = None
        if state == "DETECT":
            label, dist = match_sign(feat, signs)
            history.append(label)
            if label:
                last_dist[label] = dist
            top, count = Counter(history).most_common(1)[0]
            if top is not None and count >= SMOOTH_FRAMES // 2 + 1:
                shown_label = top
        else:
            history.clear()

        skel_col = label_color(shown_label) if shown_label else (0, 200, 255)
        for hand in hands:
            draw_skeleton(img, hand["lmList"], skel_col)

        # -- Countdown -> recording -------------------------------------------
        if state == "COUNTDOWN" and time.time() >= countdown_end:
            state, record_buffer, record_hands = "RECORDING", [], None

        if state == "RECORDING":
            if feat is None:
                set_status("Show your hand(s) to the camera!", (60, 60, 255), 0.5)
            else:
                n = len(feat["hands"])
                if record_hands is None:
                    record_hands = n
                if n == record_hands:
                    record_buffer.append(feat)
                else:
                    set_status(f"Keep {record_hands} hand(s) in view", (60, 60, 255), 0.5)

            if len(record_buffer) >= RECORD_FRAMES:
                avg   = average_features(record_buffer)
                signs = [s for s in signs if s["label"] != record_label]
                signs.append({"label": record_label, **avg})
                save_signs(GESTURES_FILE, signs)
                set_status(f"[OK] Saved '{record_label}' ({record_hands}-hand sign)",
                           (80, 255, 120))
                print(f"[OK] Saved '{record_label}' ({record_hands}-hand sign)")
                state = "DETECT"

        # -- Draw UI ----------------------------------------------------------
        draw_sidebar(img, signs, state)

        if state == "DETECT":
            draw_detected(img, shown_label, last_dist.get(shown_label, 0.0))
        elif state == "TYPING":
            draw_text_prompt(img, prompt_purpose, text_buffer)
        elif state == "COUNTDOWN":
            secs_left = max(1, math.ceil(countdown_end - time.time()))
            draw_countdown(img, record_label, secs_left, len(feat["hands"]) if feat else 0)
        elif state == "RECORDING":
            draw_record_progress(img, record_label, len(record_buffer), record_hands)

        if time.time() < status_until:
            draw_centered(img, status_msg, 36, 0.65, status_color)

        curr_time = time.time()
        fps       = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        cv2.putText(img, f"FPS {int(fps)}", (img.shape[1] - 100, 36),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        cv2.imshow("Custom Sign Language Translator", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 255:          # no key pressed
            continue

        # -- Key handling: typing a word --------------------------------------
        if state == "TYPING":
            if key == 27:                               # ESC
                state = "DETECT"
                set_status("Cancelled", (160, 160, 160))
            elif key in (13, 10):                       # ENTER
                label = text_buffer.strip()
                if not label:
                    set_status("Type a word first", (60, 60, 255))
                elif prompt_purpose == "record":
                    record_label  = label
                    countdown_end = time.time() + COUNTDOWN_SECS
                    state         = "COUNTDOWN"
                    print(f"[*] Recording '{label}' - get into position...")
                else:
                    if any(s["label"] == label for s in signs):
                        signs = [s for s in signs if s["label"] != label]
                        save_signs(GESTURES_FILE, signs)
                        set_status(f"Deleted '{label}'", (60, 60, 255))
                        print(f"[X] Deleted '{label}'")
                    else:
                        set_status(f"No sign called '{label}'", (140, 140, 140))
                    state = "DETECT"
            elif key in (8, 127):                       # BACKSPACE
                text_buffer = text_buffer[:-1]
            elif 32 <= key < 127:
                ch = chr(key).upper()
                if (ch.isalnum() or ch in " '-") and len(text_buffer) < MAX_LABEL_LEN:
                    text_buffer += ch
            continue

        # -- Key handling: everything else ------------------------------------
        if key == 27 and state in ("COUNTDOWN", "RECORDING"):
            state = "DETECT"
            set_status("Recording cancelled", (160, 160, 160))
            print("Recording cancelled")

        elif key in (ord('q'), ord('Q'), 27):
            break

        elif key in (ord('n'), ord('N')) and state == "DETECT":
            state, prompt_purpose, text_buffer = "TYPING", "record", ""

        elif key in (ord('x'), ord('X')) and state == "DETECT":
            if signs:
                state, prompt_purpose, text_buffer = "TYPING", "delete", ""
            else:
                set_status("No signs to delete", (140, 140, 140))

        elif key in (ord('+'), ord('=')):
            MATCH_THRESHOLD = min(3.50, round(MATCH_THRESHOLD + 0.15, 2))
            set_status(f"More lenient (threshold: {MATCH_THRESHOLD:.2f})", (120, 255, 120))
            print(f"[+] Leniency threshold increased to {MATCH_THRESHOLD:.2f}")

        elif key in (ord('-'), ord('_')):
            MATCH_THRESHOLD = max(0.30, round(MATCH_THRESHOLD - 0.15, 2))
            set_status(f"Stricter (threshold: {MATCH_THRESHOLD:.2f})", (255, 180, 80))
            print(f"[-] Leniency threshold decreased to {MATCH_THRESHOLD:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
