# ✋ Sign Language Translator — Hand Tracking

Real-time hand tracking + a custom accessible sign language alphabet, using **OpenCV**, **MediaPipe**, and **cvzone**.

## Files

| File | Description |
|------|-------------|
| `custom_alphabet.py` | **Custom sign system**: record your own one- or two-hand signs for any word. Works with missing fingers |
| `hand_tracking.py` | Hand tracking demo — landmarks, bounding boxes, finger count, inter-hand distance |
| `finger_counter.py` | Finger count overlay (0–10) |
| `requirements.txt` | Python dependencies |
| `gestures.json` | Your saved signs (created automatically) |

## Setup

```bash
pip install -r requirements.txt
```

## Custom Signs (main feature)

Designed for users with **any hand shape**, including missing fingers.  
Instead of detecting "which fingers are up", it records the full shape of your hand(s) and matches against it, so any pose can mean any word.

Signs can use **one or two hands**. The number of hands is detected automatically while recording.

```bash
python custom_alphabet.py
```

### Controls

| Key | Action |
|-----|--------|
| `N` | New sign: type a word, press `ENTER`, then hold the pose |
| `X` | Delete a sign: type its word, press `ENTER` |
| `+` / `-` | More / less lenient matching |
| `ESC` | Cancel a recording (quits when idle) |
| `Q` | Quit |

While typing a word: `ENTER` confirms, `BACKSPACE` erases, `ESC` cancels.

**Step-by-step:**
1. Press **N** and type a word, e.g. `HELLO` (letters, numbers, spaces, `'` and `-`, up to 16 characters)
2. Press **ENTER**. You get a 3-second countdown to move your hands into position
3. Show **one hand** for a one-hand sign, or **both hands** for a two-hand sign, and hold it until the progress bar fills
4. Hold any saved sign and the word appears on screen with a confidence bar

Recording a word that already exists replaces it. Signs are saved to `gestures.json` and load automatically next time. Signs saved by the older A–E version are converted automatically.

### How two-hand signs are matched
- Hands are ordered left-to-right on screen, so it doesn't matter which hand the camera sees first
- **Both** hands must match their recorded shape; one correct hand can't make up for a wrong one
- The spacing between your hands counts too (adjust `REL_WEIGHT` in the config to change how much)
- If a two-hand pose doesn't match any two-hand sign, each hand is checked against your one-hand signs, so a one-hand sign still works when your other hand is in view

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
