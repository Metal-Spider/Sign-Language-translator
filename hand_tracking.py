"""
Hand Tracking - Real-time hand landmark detection
Uses cvzone + MediaPipe to detect hands, draw landmarks,
count fingers, and display hand info on webcam feed.
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import time


# ── Config ──────────────────────────────────────────────────────────────────
CAM_INDEX   = 0       # 0 = default webcam
CAM_WIDTH   = 1280
CAM_HEIGHT  = 720
MAX_HANDS   = 2
DETECT_CON  = 0.8     # detection confidence threshold
TRACK_CON   = 0.8     # tracking confidence threshold
DRAW_BONES  = True    # draw skeleton + bounding box
FLIP_IMAGE  = True    # mirror like a selfie cam
# ────────────────────────────────────────────────────────────────────────────


def draw_fps(img, fps):
    """Render a clean FPS counter in the top-left corner."""
    cv2.rectangle(img, (10, 10), (120, 50), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f"FPS: {int(fps)}", (18, 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


def draw_hand_info(img, hand, index):
    """
    Overlay hand metadata (type, finger count, landmark count)
    next to the hand's bounding box.
    """
    bbox        = hand["bbox"]          # (x, y, w, h)
    hand_type   = hand["type"]          # 'Left' or 'Right'
    lm_list     = hand["lmList"]        # 21 landmarks [(x,y,z), ...]
    fingers_up  = hand["fingersUp"]     # [thumb, index, middle, ring, pinky]
    finger_count = sum(fingers_up)

    x, y, w, h = bbox

    # label above the bounding box
    label_color = (255, 165, 0) if hand_type == "Right" else (0, 165, 255)
    label = f"#{index + 1} {hand_type}  Fingers: {finger_count}"
    cv2.putText(img, label, (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    # small dot + coords for each landmark
    for lm in lm_list:
        cx, cy = lm[0], lm[1]
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}. "
                           "Check your webcam connection.")

    detector = HandDetector(
        detectionCon=DETECT_CON,
        maxHands=MAX_HANDS,
    )

    prev_time = time.time()
    print("Hand Tracking started. Press 'q' or ESC to quit.\n")

    while True:
        success, img = cap.read()
        if not success:
            print("Empty frame — skipping.")
            continue

        # ── Detection ────────────────────────────────────────────────────────
        hands, img = detector.findHands(img, draw=DRAW_BONES, flipType=FLIP_IMAGE)

        # ── Per-hand overlay ─────────────────────────────────────────────────
        if hands:
            for i, hand in enumerate(hands):
                draw_hand_info(img, hand, i)

            # Distance between index fingertips when both hands are visible
            if len(hands) == 2:
                lm0 = hands[0]["lmList"]
                lm1 = hands[1]["lmList"]
                dist, _, img = detector.findDistance(lm0[8][0:2], lm1[8][0:2], img)
                mid_x = (lm0[8][0] + lm1[8][0]) // 2
                mid_y = (lm0[8][1] + lm1[8][1]) // 2
                cv2.putText(img, f"{int(dist)}px", (mid_x - 20, mid_y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

        # ── FPS ───────────────────────────────────────────────────────────────
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        draw_fps(img, fps)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Hand Tracking — press Q to quit", img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # q / Q / ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Hand Tracking stopped.")


if __name__ == "__main__":
    main()
