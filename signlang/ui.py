"""All on-screen drawing (OpenCV). Only ASCII text - cv2.putText can't draw anything else."""

import time
import zlib

import cv2

from . import config as cfg

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOLD = cv2.FONT_HERSHEY_DUPLEX

SKELETON = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8), (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16), (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17),
]

PALETTE = [  # BGR
    (255,  80,  80), ( 80, 200, 255), ( 80, 255, 160), (255, 200,  60),
    (200,  80, 255), (255, 120, 200), (120, 255, 255), (180, 255,  80),
]

BADGES = {
    "DETECT":    (( 40, 150,  40), "TRANSLATING"),
    "TYPING":    (( 20, 140, 200), "TYPE A NAME"),
    "RECORDING": (( 40,  40, 200), "RECORDING"),
    "PRACTICE":  ((160,  60, 160), "PRACTICE"),
    "SUMMARY":   ((160,  60, 160), "PRACTICE RESULTS"),
}

HELP_LINES = [
    ("N", "record a new still sign (or add samples to one)"),
    ("M", "record a new moving sign"),
    ("X", "delete a sign"),
    ("L", "practice mode"),
    ("P", "switch / create profile"),
    ("E / I", "export / import signs"),
    ("S", "speak the sentence"),
    ("V", "voice on / off"),
    ("BACKSPACE", "delete last letter / word"),
    ("C", "clear the sentence"),
    ("1 2 3", "accept a spelling suggestion"),
    ("+ / -", "more / less lenient matching"),
    ("H", "show / hide this help"),
    ("ESC", "cancel / leave mode (quits when idle)"),
    ("Q", "quit"),
    ("", ""),
    ("Signs named", "#SPACE #DEL #CLEAR #SPEAK #OK work as controls"),
    ("Single letters", "A, B, 7 ... are fingerspelled into words"),
]


def label_color(label):
    return PALETTE[zlib.crc32(label.encode()) % len(PALETTE)] if label else (0, 200, 255)


def camera_center_x(img):
    return cfg.SIDEBAR_W + (img.shape[1] - cfg.SIDEBAR_W) // 2


def text_with_shadow(img, text, org, scale, color, thick=1, font=FONT):
    cv2.putText(img, text, (org[0] + 2, org[1] + 2), font, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, org, font, scale, color, thick)


def draw_centered(img, text, y, scale, color, thick=1, font=FONT):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    text_with_shadow(img, text, (camera_center_x(img) - tw // 2, y), scale, color, thick, font)


def fit_scale(text, max_w, max_scale, thick_ratio=2, font=BOLD):
    (tw, _), _ = cv2.getTextSize(text, font, 1.0, 2)
    return min(max_scale, max_w / max(tw, 1))


def panel(img, x0, y0, x1, y1, alpha=0.8, color=(20, 20, 20), border=(170, 170, 170)):
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border:
        cv2.rectangle(img, (x0, y0), (x1, y1), border, 1)


def bar(img, x, y, w, h, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), cv2.FILLED)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), color, cv2.FILLED)


# -- Hands --------------------------------------------------------------------------

def draw_hand(img, hand, color):
    lm = hand.lm
    for a, b in SKELETON:
        cv2.line(img, tuple(lm[a][:2]), tuple(lm[b][:2]), color, 2)
    for p in lm:
        cv2.circle(img, (p[0], p[1]), 4, (255, 255, 255), cv2.FILLED)
    tag = {"Left": "L", "Right": "R"}.get(hand.side, "?")
    text_with_shadow(img, tag, (lm[0][0] - 8, lm[0][1] + 30), 0.8, color, 2)


# -- Sidebar ------------------------------------------------------------------------

def draw_sidebar(img, state, profile_name, signs, threshold, voice_on, lookalike_labels):
    h = img.shape[0]
    panel(img, 0, 0, cfg.SIDEBAR_W, h, alpha=0.72, color=(15, 15, 15), border=None)

    col, text = BADGES[state]
    cv2.rectangle(img, (8, 12), (cfg.SIDEBAR_W - 8, 44), col, cv2.FILLED)
    cv2.putText(img, "  " + text, (12, 35), FONT, 0.58, (255, 255, 255), 1)

    y = 70
    cv2.putText(img, f"Profile: {profile_name}", (12, y), FONT, 0.5, (220, 220, 220), 1)
    y += 22
    voice = "ON" if voice_on else "OFF"
    cv2.putText(img, f"Voice: {voice}   Leniency: {threshold:.2f}", (12, y), FONT, 0.42,
                (180, 220, 180), 1)
    y += 30
    cv2.putText(img, f"Signs ({len(signs)})", (12, y), FONT, 0.48, (160, 160, 160), 1)
    y += 24

    bottom = h - 70
    max_rows = max(1, (bottom - y) // 23)
    ordered = sorted(signs, key=lambda s: (s["label"].startswith("#"), s["label"]))
    if not ordered:
        cv2.putText(img, "  none yet - press N", (12, y), FONT, 0.5, (90, 90, 90), 1)
    for i, s in enumerate(ordered):
        if i == max_rows - 1 and len(ordered) > max_rows:
            cv2.putText(img, f"  ... +{len(ordered) - i} more", (12, y), FONT, 0.48,
                        (120, 120, 120), 1)
            break
        warn = s["label"] in lookalike_labels
        cv2.putText(img, ("! " if warn else "  ") + s["label"][:13], (12, y), FONT, 0.52,
                    (60, 60, 255) if warn else label_color(s["label"]), 1)
        tags = f"{'~' if s['kind'] == 'motion' else ''}{s['n_hands']}H x{len(s['samples'])}"
        cv2.putText(img, tags, (cfg.SIDEBAR_W - 78, y), FONT, 0.42, (150, 150, 150), 1)
        y += 23

    cv2.putText(img, "~ = moving   ! = look-alike", (12, h - 48), FONT, 0.38, (110, 110, 110), 1)
    cv2.putText(img, "H = help     Q = quit", (12, h - 24), FONT, 0.45, (170, 170, 170), 1)


# -- Translating view ------------------------------------------------------------------

def draw_detected(img, label, confidence):
    if not label:
        return
    max_w = (img.shape[1] - cfg.SIDEBAR_W) * 0.8
    scale = fit_scale(label, max_w, 3.5)
    thick = max(2, int(scale * 2))
    col = label_color(label)
    draw_centered(img, label, 120, scale, col, thick, BOLD)
    cx = camera_center_x(img)
    bar(img, cx - 120, 140, 240, 10, confidence, col)


def draw_bottom_bar(img, sentence, suggestions, hold_label, hold_progress, moving):
    h, w = img.shape[:2]
    x0, y0 = cfg.SIDEBAR_W, h - cfg.BOTTOM_H
    panel(img, x0, y0, w, h, alpha=0.8, color=(25, 25, 25), border=None)

    words = " ".join(sentence.words)
    text = words + (" " if words and sentence.spelling else "")
    x = x0 + 20
    y = y0 + 45
    scale = 1.0
    max_w = w - x0 - 40

    def width(s):
        return cv2.getTextSize(s, BOLD, scale, 2)[0][0]

    if width(text + sentence.spelling + "_") > max_w:      # long sentence: show the end
        while text and width("..." + text + sentence.spelling + "_") > max_w:
            text = text[1:]
        text = "..." + text
    cv2.putText(img, text, (x, y), BOLD, scale, (240, 240, 240), 2)
    (tw, _), _ = cv2.getTextSize(text, BOLD, scale, 2)
    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    cv2.putText(img, sentence.spelling + cursor, (x + tw, y), BOLD, scale, (80, 255, 200), 2)

    y2 = y0 + 85
    if suggestions:
        sx = x
        for i, s in enumerate(suggestions, 1):
            chip = f"{i}: {s}"
            (cw, _), _ = cv2.getTextSize(chip, FONT, 0.55, 1)
            cv2.rectangle(img, (sx - 6, y2 - 20), (sx + cw + 6, y2 + 8), (70, 70, 70), cv2.FILLED)
            cv2.putText(img, chip, (sx, y2), FONT, 0.55, (220, 220, 220), 1)
            sx += cw + 24
    elif not sentence.words and not sentence.spelling:
        cv2.putText(img, "Hold a sign steady to add it.  S = speak   C = clear   BACKSPACE = undo",
                    (x, y2), FONT, 0.5, (130, 130, 130), 1)

    if hold_label and hold_progress > 0:
        bx = w - 260
        cv2.putText(img, f"adding {hold_label}", (bx, y0 + 22), FONT, 0.45,
                    label_color(hold_label), 1)
        bar(img, bx, y0 + 30, 230, 8, hold_progress, label_color(hold_label))
    elif moving:
        cv2.putText(img, "moving...", (w - 260, y0 + 22), FONT, 0.45, (150, 150, 150), 1)


# -- Prompts & overlays -------------------------------------------------------------------

def draw_text_prompt(img, title, text, hint=""):
    h = img.shape[0]
    cx = camera_center_x(img)
    x0, y0 = cx - 300, h // 2 - 100
    panel(img, x0, y0, x0 + 600, y0 + 190, alpha=0.88)
    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    draw_centered(img, title, y0 + 40, 0.75, (220, 220, 220))
    draw_centered(img, text + cursor, y0 + 100, 1.4, (80, 255, 200), 2)
    if hint:
        draw_centered(img, hint, y0 + 140, 0.45, (170, 170, 170))
    draw_centered(img, "ENTER confirm  |  BACKSPACE erase  |  ESC cancel", y0 + 172, 0.45,
                  (130, 130, 130))


def draw_recording(img, session, now, n_hands_visible):
    h = img.shape[0]
    kind = "moving" if session.kind == "motion" else "still"
    sample = f"sample {session.sample_number}/{session.samples_needed}"
    draw_centered(img, f"Recording {kind} sign '{session.label}'  ({sample})", 80, 0.8,
                  (255, 220, 120), 2)

    if session.phase in ("countdown", "rest"):
        secs = max(1, int(session.phase_end - now + 0.999))
        draw_centered(img, str(secs), h // 2 + 40, 5, (255, 255, 255), 8, BOLD)
        if session.phase == "countdown":
            hint = (f"{n_hands_visible} hand(s) visible" if n_hands_visible
                    else "Show the hand(s) you want to use")
            if session.kind == "motion":
                hint += " - start still, then do the movement"
        else:
            hint = session.message
        draw_centered(img, hint, h // 2 + 110, 0.65, (210, 210, 210))
    else:
        col = label_color(session.label)
        cx = camera_center_x(img)
        if session.phase == "capture":
            bar(img, cx - 200, h - 80, 400, 20, session.progress(), col)
            draw_centered(img, "Hold the pose steady", h - 95, 0.65, (255, 255, 255))
        elif session.phase == "wait_move":
            draw_centered(img, "GO - do the movement", h // 2, 1.3, col, 3)
        elif session.phase == "moving":
            cv2.circle(img, (cx, h // 2), 18, (40, 40, 230), cv2.FILLED)
            draw_centered(img, "Recording movement... stop to finish", h // 2 + 60, 0.75,
                          (255, 255, 255), 2)
        if session.message and session.phase != "moving":
            draw_centered(img, session.message, h - 120, 0.6, (120, 200, 255))
    draw_centered(img, "ESC = cancel", h - 20, 0.45, (150, 150, 150))


def draw_practice(img, session, recognition, now):
    h = img.shape[0]
    cx = camera_center_x(img)
    hits, total = session.score
    cv2.putText(img, f"Score {hits}/{total}", (img.shape[1] - 190, 70), FONT, 0.7,
                (255, 255, 255), 2)
    draw_centered(img, "Sign this:", 70, 0.7, (200, 200, 200))
    col = label_color(session.target)
    scale = fit_scale(session.target, (img.shape[1] - cfg.SIDEBAR_W) * 0.7, 3.0)
    draw_centered(img, session.target, 150, scale, col, max(2, int(scale * 2)), BOLD)

    if session.phase == "prompt":
        bar(img, cx - 200, 175, 400, 10, session.time_left(now) / cfg.PRACTICE_TIMEOUT_SECS, col)
        if recognition.shown and recognition.shown != session.target:
            draw_centered(img, f"I see: {recognition.shown}", h - 60, 0.7, (120, 160, 255), 2)
        elif recognition.shown == session.target:
            bar(img, cx - 100, h - 70, 200, 12, recognition.hold_progress, (80, 255, 120))
    else:
        kind, info = session.feedback
        if kind == "hit":
            draw_centered(img, f"Correct!  ({info:.1f}s)", h // 2, 1.4, (80, 255, 120), 3)
        else:
            msg = "Time's up" + (f" - looked like {info}" if info else "")
            draw_centered(img, msg, h // 2, 1.1, (80, 80, 255), 3)
    draw_centered(img, "ESC = finish", h - 20, 0.45, (150, 150, 150))


def draw_summary(img, lines):
    h = img.shape[0]
    cx = camera_center_x(img)
    x0, y0 = cx - 330, 110
    panel(img, x0, y0, x0 + 660, y0 + 60 + 32 * len(lines), alpha=0.9)
    draw_centered(img, "Practice results", y0 + 38, 0.9, (255, 220, 120), 2)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x0 + 40, y0 + 80 + 32 * i), FONT, 0.65, (230, 230, 230), 1)
    draw_centered(img, "Press any key to continue", h - 30, 0.55, (170, 170, 170))


def draw_help(img):
    cx = camera_center_x(img)
    x0, y0 = cx - 360, 60
    panel(img, x0, y0, x0 + 720, y0 + 70 + 28 * len(HELP_LINES), alpha=0.92)
    draw_centered(img, "Controls", y0 + 38, 0.9, (255, 220, 120), 2)
    for i, (key, desc) in enumerate(HELP_LINES):
        y = y0 + 75 + 28 * i
        cv2.putText(img, key, (x0 + 30, y), FONT, 0.55, (120, 255, 200), 1)
        cv2.putText(img, desc, (x0 + 210, y), FONT, 0.55, (230, 230, 230), 1)


def draw_status(img, message, color):
    draw_centered(img, message, 36, 0.65, color)


def draw_fps(img, fps):
    cv2.putText(img, f"FPS {int(fps)}", (img.shape[1] - 110, 30), cv2.FONT_HERSHEY_PLAIN, 1.4,
                (0, 255, 0), 2)
