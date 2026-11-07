# ✋ Sign Language Translator — Hand Tracking

Real-time hand tracking demos using **OpenCV**, **MediaPipe**, and **cvzone**.

## Files

| File | Description |
|------|-------------|
| `hand_tracking.py` | Main hand tracking demo — landmarks, bounding boxes, finger count, inter-hand distance |
| `finger_counter.py` | Bonus demo — big on-screen digit showing total fingers held up (up to 10) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Main hand tracking
python hand_tracking.py

# Finger counter
python finger_counter.py
```

Press **Q** or **ESC** to quit.

## How it works

```
Webcam frame
     │
     ▼
MediaPipe Hands (via cvzone.HandDetector)
     │   detects up to 2 hands
     │   returns 21 landmarks per hand (x, y, z)
     ▼
Per-frame overlay:
  • Skeleton + bounding box drawn by cvzone
  • Hand type (Left / Right) label
  • Finger-up count per hand
  • Inter-hand index-fingertip distance (when both hands visible)
  • Live FPS counter
     │
     ▼
cv2.imshow → display window
```

## Landmark Map

MediaPipe assigns **21 landmarks** to each hand:

```
                 8   12  16  20
                 |   |   |   |
              7  |11  |15  |19
              |  |  | |  | |  |
           6  | 10  |14  |18
           | ---|---|-|--|--|
        5  |  9  | 13  | 17
        |  |  |  |  |  |  |
        4  |  |  |  |  |  |
        |  |  |  |  |  |  |
        3  |  |  |  |  |  |
  THUMB-2  INDEX MIDDLE RING PINKY
        1
        |
        0 ← WRIST
```

| ID | Landmark         |
|----|-----------------|
| 0  | Wrist           |
| 4  | Thumb tip       |
| 8  | Index tip       |
| 12 | Middle tip      |
| 16 | Ring tip        |
| 20 | Pinky tip       |
