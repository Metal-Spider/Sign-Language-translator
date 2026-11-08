"""
Custom Sign Language Alphabet
==============================
Designed for users with any hand shape — including missing fingers.
Instead of detecting "fingers up", it records the full 3D hand shape
and matches against it, so any pose can be assigned any letter.

CONTROLS
--------
  SPACE      Toggle between RECORD and DETECT mode
  A/B/C/D/E  (in RECORD mode) Record current pose for that letter
  X + letter Delete saved gesture for that letter  (within 1.5 s)
  Q / ESC    Quit
"""

import cv2
import json
import math
import os
import time

from cvzone.HandTrackingModule import HandDetector

# ── Config ───────────────────────────────────────────────────────────────────
CAM_INDEX       = 0
CAM_WIDTH       = 1280
CAM_HEIGHT      = 720
DETECT_CON      = 0.7
GESTURES_FILE   = "gestures.json"
LETTERS         = list("ABCDE")

# How many frames to average when recording (more = smoother, slower)
RECORD_FRAMES   = 25

# Max normalised distance to count as a match (higher = more lenient)
MATCH_THRESHOLD = 1.40
# ─────────────────────────────────────────────────────────────────────────────

SKELETON_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

LETTER_COLORS = {
    "A": (255,  80,  80),
    "B": ( 80, 200, 255),
    "C": ( 80, 255, 160),
    "D": (255, 200,  60),
    "E": (200,  80, 255),
}


# ── Landmark normalisation ────────────────────────────────────────────────────

def normalize_landmarks(lm_list):
    """
    Turn raw pixel landmarks into a position- and scale-invariant vector.

    Steps:
      1. Translate all points so wrist (LM 0) is the origin.
      2. Scale by the wrist → middle-MCP (LM 9) distance so hand size
         doesn't matter (works at any distance from camera).
      3. Flatten to a list of 42 floats: [x0,y0, x1,y1, … x20,y20].

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


def average_vectors(vecs):
    """Element-wise average of a list of equal-length float lists."""
    return [sum(col) / len(col) for col in zip(*vecs)]


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


# ── Gesture persistence ───────────────────────────────────────────────────────

def load_gestures(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_gestures(path, gestures):
    with open(path, "w") as f:
        json.dump(gestures, f, indent=2)


# ── Matching ──────────────────────────────────────────────────────────────────

def match_gesture(vec, gestures):
    """Return (letter, distance) of closest saved gesture, or (None, inf)."""
    best_letter, best_dist = None, float("inf")
    for letter, saved_vec in gestures.items():
        d = vec_distance(vec, saved_vec)
        if d < best_dist:
            best_dist  = d
            best_letter = letter
    return best_letter, best_dist


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_skeleton(img, lm_list, color=(0, 200, 255)):
    for a, b in SKELETON_CONNECTIONS:
        cv2.line(img, tuple(lm_list[a][:2]), tuple(lm_list[b][:2]), color, 2)
    for lm in lm_list:
        cv2.circle(img, (lm[0], lm[1]), 5, (255, 255, 255), cv2.FILLED)


def draw_sidebar(img, gestures, mode):
    """Translucent left sidebar with mode badge and gesture list."""
    h          = img.shape[0]
    panel_w    = 230
    overlay    = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

    y = 38
    if mode == "RECORD":
        badge_col = (180, 30, 30)
        badge_txt = "  REC MODE"
    else:
        badge_col = (30, 140, 30)
        badge_txt = "  DETECT MODE"
    cv2.rectangle(img, (8, y - 24), (panel_w - 8, y + 8), badge_col, cv2.FILLED)
    cv2.putText(img, badge_txt, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    y += 44

    cv2.putText(img, "Saved gestures", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
    y += 26
    for letter in LETTERS:
        saved = letter in gestures
        col   = LETTER_COLORS[letter] if saved else (70, 70, 70)
        icon  = "[SAVED]" if saved else "[ -- ]"
        cv2.putText(img, f"  {letter}  {icon}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, col, 1)
        y += 28

    y += 12
    cv2.putText(img, f"Leniency: {MATCH_THRESHOLD:.2f}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 180), 1)
    y += 24
    for hint in ["SPACE  toggle mode",
                 "A-E    record pose",
                 "+ / -  more/less lenient",
                 "X+ltr  delete pose",
                 "Q/ESC  quit"]:
        cv2.putText(img, hint, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 110, 110), 1)
        y += 19


def draw_record_progress(img, recording_letter, buf_len):
    """Progress bar at the bottom while recording."""
    h, w   = img.shape[:2]
    bx, by = 10, h - 55
    bw     = 220
    filled = int(bw * buf_len / RECORD_FRAMES)
    col    = LETTER_COLORS.get(recording_letter, (200, 200, 200))

    cv2.rectangle(img, (bx, by),         (bx + bw, by + 20), (40, 40, 40), cv2.FILLED)
    cv2.rectangle(img, (bx, by),         (bx + filled, by + 20), col, cv2.FILLED)
    cv2.putText(img, f"Recording  '{recording_letter}'  {buf_len}/{RECORD_FRAMES}",
                (bx, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)


def draw_detected(img, letter, dist, panel_w):
    """Big centred letter with confidence bar on the right side of frame."""
    if letter is None:
        return
    h, w   = img.shape[:2]
    conf   = max(0.0, 1.0 - dist / MATCH_THRESHOLD)
    col    = LETTER_COLORS.get(letter, (255, 255, 255))
    cx     = panel_w + (w - panel_w) // 2

    # Shadow + letter
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 11, 22
    (tw, th), _ = cv2.getTextSize(letter, font, scale, thick)
    tx, ty = cx - tw // 2, h // 2 + th // 2
    cv2.putText(img, letter, (tx + 5, ty + 5), font, scale, (0, 0, 0), thick + 6)
    cv2.putText(img, letter, (tx,     ty),     font, scale, col,       thick)

    # Confidence bar
    bar_w  = 320
    bx     = cx - bar_w // 2
    by     = ty + 28
    filled = int(bar_w * conf)
    cv2.rectangle(img, (bx, by),         (bx + bar_w, by + 20), (40, 40, 40), cv2.FILLED)
    cv2.rectangle(img, (bx, by),         (bx + filled, by + 20), col, cv2.FILLED)
    cv2.putText(img, f"{int(conf * 100)}% confidence",
                (bx, by + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 1)


def draw_status(img, message, color=(255, 220, 60)):
    """Temporary status message at the top-centre."""
    h, w   = img.shape[:2]
    (tw, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cx = w // 2
    cv2.putText(img, message, (cx - tw // 2, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(img, message, (cx - tw // 2, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global MATCH_THRESHOLD
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAM_INDEX}.")

    detector = HandDetector(detectionCon=DETECT_CON, maxHands=1)
    gestures = load_gestures(GESTURES_FILE)

    mode             = "RECORD"
    recording_letter = None
    record_buffer    = []
    detected_letter  = None
    detected_dist    = float("inf")

    # Temporary status message
    status_msg       = ""
    status_color     = (255, 220, 60)
    status_until     = 0.0

    # X + letter delete combo
    delete_armed     = False
    delete_arm_time  = 0.0

    prev_time = time.time()
    SIDEBAR_W = 230

    print(f"Loaded gestures: {list(gestures.keys()) or 'none'}")
    print("SPACE=toggle mode | A-E=record | X+letter=delete | Q=quit\n")

    def set_status(msg, col=(255, 220, 60), duration=2.0):
        nonlocal status_msg, status_color, status_until
        status_msg   = msg
        status_color = col
        status_until = time.time() + duration

    while True:
        success, img = cap.read()
        if not success:
            continue

        hands, img = detector.findHands(img, draw=False, flipType=True)

        # ── Landmark extraction ───────────────────────────────────────────────
        normalized = None
        if hands:
            lm_list    = hands[0]["lmList"]
            normalized = normalize_landmarks(lm_list)
            skel_color = (LETTER_COLORS.get(detected_letter, (0, 200, 255))
                          if mode == "DETECT" and detected_letter else (0, 200, 255))
            draw_skeleton(img, lm_list, skel_color)

        # ── Recording ─────────────────────────────────────────────────────────
        if recording_letter:
            if normalized:
                record_buffer.append(normalized)
                if len(record_buffer) >= RECORD_FRAMES:
                    avg = average_vectors(record_buffer)
                    gestures[recording_letter] = avg
                    save_gestures(GESTURES_FILE, gestures)
                    set_status(f"[OK] Saved '{recording_letter}'", (80, 255, 120))
                    print(f"[OK] Saved gesture for '{recording_letter}'")
                    recording_letter = None
                    record_buffer    = []
            else:
                set_status("Show your hand to the camera!", (60, 60, 255))
                record_buffer = []  # reset if hand disappears mid-recording

        # ── Detection ─────────────────────────────────────────────────────────
        if mode == "DETECT":
            if normalized and gestures:
                detected_letter, detected_dist = match_gesture(normalized, gestures)
                if detected_dist > MATCH_THRESHOLD:
                    detected_letter = None
            else:
                detected_letter = None
        else:
            detected_letter = None

        # ── Draw UI ───────────────────────────────────────────────────────────
        draw_sidebar(img, gestures, mode)

        if mode == "DETECT":
            draw_detected(img, detected_letter, detected_dist, SIDEBAR_W)

        if recording_letter:
            draw_record_progress(img, recording_letter, len(record_buffer))

        if time.time() < status_until:
            draw_status(img, status_msg, status_color)

        # FPS
        curr_time = time.time()
        fps       = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        cv2.putText(img, f"FPS {int(fps)}", (CAM_WIDTH - 100, 36),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        cv2.imshow("Custom Sign Language Alphabet", img)
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ──────────────────────────────────────────────────────
        if key in (ord('q'), ord('Q'), 27):
            break

        elif key == ord(' '):
            mode             = "DETECT" if mode == "RECORD" else "RECORD"
            recording_letter = None
            record_buffer    = []
            detected_letter  = None
            set_status(f"Switched to {mode} mode")
            print(f"-> {mode} mode")

        elif (chr(key).upper() in LETTERS) if 0 <= key < 128 else False:
            letter = chr(key).upper()

            if delete_armed and (time.time() - delete_arm_time < 1.5):
                # X + letter = delete gesture
                if letter in gestures:
                    del gestures[letter]
                    save_gestures(GESTURES_FILE, gestures)
                    set_status(f"Deleted '{letter}'", (60, 60, 255))
                    print(f"[X] Deleted gesture for '{letter}'")
                else:
                    set_status(f"'{letter}' not saved yet", (140, 140, 140))
                delete_armed = False

            elif mode == "RECORD":
                if normalized is None:
                    set_status("No hand detected - show your hand first!", (60, 60, 255))
                    print("[!] No hand in frame.")
                else:
                    recording_letter = letter
                    record_buffer    = []
                    set_status(f"Hold pose for '{letter}'...", (255, 200, 60), 3.0)
                    print(f"[*] Recording '{letter}' - hold pose steady...")
            else:
                set_status("Switch to REC mode (SPACE) to record")

        elif key in (ord('x'), ord('X')):
            delete_armed    = True
            delete_arm_time = time.time()
            set_status("Delete armed - press A-E to delete that gesture", (255, 100, 100))
            print("Delete armed - press a letter within 1.5 s")

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
