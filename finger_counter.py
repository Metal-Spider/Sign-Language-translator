"""
Finger Counter — bonus demo
Shows a big digit overlay for how many fingers are held up,
across up to 2 hands (max 10 fingers total).
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import time


CAM_INDEX  = 0
CAM_WIDTH  = 1280
CAM_HEIGHT = 720
MAX_HANDS  = 2
DETECT_CON = 0.8


def draw_count(img, count):
    """Big centred finger-count display at the bottom of the frame."""
    h, w = img.shape[:2]
    text  = str(count)
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 8
    thick = 12
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = h - 30
    # shadow
    cv2.putText(img, text, (x + 4, y + 4), font, scale, (0, 0, 0),   thick)
    # foreground
    cv2.putText(img, text, (x,     y),     font, scale, (57, 255, 20), thick)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    detector = HandDetector(detectionCon=DETECT_CON, maxHands=MAX_HANDS)
    prev_time = time.time()

    print("Finger Counter started. Press 'q' or ESC to quit.\n")

    while True:
        success, img = cap.read()
        if not success:
            continue

        hands, img = detector.findHands(img, draw=True, flipType=True)

        total_fingers = 0
        if hands:
            for hand in hands:
                total_fingers += sum(hand["fingersUp"])

        draw_count(img, total_fingers)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        cv2.putText(img, f"FPS {int(fps)}", (10, 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Finger Counter — press Q to quit", img)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
