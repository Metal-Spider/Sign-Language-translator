# ✋ Sign Language Translator — Hand Tracking

Real-time hand tracking + a custom accessible sign language alphabet, using **OpenCV**, **MediaPipe**, and **cvzone**.

## Files

| File | Description |
|------|-------------|
| `custom_alphabet.py` | **Custom alphabet system** — record your own A-E poses, works with missing fingers |
| `hand_tracking.py` | Hand tracking demo — landmarks, bounding boxes, finger count, inter-hand distance |
| `finger_counter.py` | Finger count overlay (0–10) |
| `requirements.txt` | Python dependencies |
| `gestures.json` | Auto-created when you save your first gesture |

## Setup

```bash
pip install -r requirements.txt
```

## Custom Alphabet (main feature)

Designed for users with **any hand shape**, including missing fingers.  
Instead of detecting "which fingers are up", it records the full hand shape and matches against it — so any pose maps to any letter.

```bash
python custom_alphabet.py
```

### How to use

| Key | Action |
|-----|--------|
| `SPACE` | Toggle between **RECORD** and **DETECT** mode |
| `A` `B` `C` `D` `E` | (in RECORD mode) Hold pose → press letter to save it |
| `X` then a letter | Delete that gesture (within 1.5 s) |
| `Q` / `ESC` | Quit |

**Step-by-step:**
1. Run the script — left sidebar shows which letters are saved
2. Hold your hand in the pose you want for letter **A**
3. Press **A** — a progress bar fills over 40 frames, then saves
4. Repeat for B–E
5. Press **SPACE** to switch to DETECT mode
6. Hold any saved pose — the matching letter appears big on screen with a confidence bar

Gestures are saved to `gestures.json` and reload automatically next run.

---

## Other demos

```bash
python hand_tracking.py   # skeleton + finger count + inter-hand distance
python finger_counter.py  # big digit showing total fingers up
```

Press **Q** or **ESC** to quit any demo.

---

## How matching works

```
Raw webcam frame
      │
      ▼
MediaPipe → 21 hand landmarks (pixel coords)
      │
      ▼
Normalize:
  • Translate so wrist = origin
  • Scale by wrist→middle-MCP distance  ← scale-invariant
  • Flatten to 42-float vector
      │
      ▼
Euclidean distance vs. each saved gesture vector
      │
      ▼
Best match below threshold → show letter + confidence %
```

## Landmark map

```
      8   12  16  20
      |   |   |   |
   7  11  15  19
   |  |   |   |
   6  10  14  18
   5   9  13  17   ← MCPs
    \ / \ / \ /
      0 ← WRIST
      |
     (4 = thumb tip)
```
