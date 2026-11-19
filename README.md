# Sign Language Translator

A real-time, webcam-based sign recognition system built on OpenCV and MediaPipe. It translates user-defined hand signs into text and speech.

Most sign recognition tools rely on a fixed vocabulary and assume a standard hand anatomy. This project takes a different approach: each user records their own signs, and recognition compares the full geometry and movement of the hands rather than checking which fingers are extended. As a result, the system works for people with limb differences, including missing or partially formed fingers, and supports any personal or regional sign vocabulary.

## Features

- **User-defined vocabulary.** Any static pose or movement can be assigned to any word.
- **Static and dynamic signs.** Supports held poses and motion-based signs, each with one or two hands.
- **Sentence construction.** Recognised words are assembled into sentences. Single-character signs are treated as fingerspelling, with word completion suggestions.
- **Text-to-speech output.** Words and sentences can be spoken aloud using the platform's native speech engine.
- **Multiple samples per sign.** Several recordings are kept for each sign to improve robustness, and the system warns when two signs are too similar to tell apart reliably.
- **Practice mode.** Prompts the user to perform signs and tracks recognition accuracy for each sign.
- **Profiles.** Each user keeps an independent sign set. Profiles can be exported and imported.

## Requirements

- Python 3.9 or later
- A webcam
- Windows, macOS, or Linux

Speech output uses Windows SAPI on Windows and `say` on macOS. On Linux, install `pyttsx3` or `espeak`. If no speech engine is available, the application keeps working without voice output.

## Installation

```bash
git clone https://github.com/Metal-Spider/Sign-Language-translator.git
cd Sign-Language-translator
pip install -r requirements.txt
```

The MediaPipe hand landmark model (`hand_landmarker.task`) is included in the repository. If it is missing, it is downloaded automatically the first time you run the application.

## Usage

```bash
python sign_translator.py
```

Press `H` in the application window to show the full list of controls.

### Recording a sign

1. Press `N` for a static sign or `M` for a moving sign.
2. Enter a label (up to 16 characters) and press `ENTER`.
3. After a three-second countdown, perform the sign. Each recording session captures three samples in a row:
   - **Static signs:** hold the pose until the progress bar completes.
   - **Moving signs:** perform the movement and then hold still briefly. Repeat for each sample.
4. The number of hands (one or two) is detected automatically during recording.

Recording an existing label with the same type and number of hands adds samples to that sign (up to six are kept). Recording it with a different type or number of hands replaces it.

### Recognition and sentence construction

- **Static signs** are added to the sentence after being held steady for one second. The same sign is not added again until it has been released.
- **Moving signs** are added as soon as the movement is completed.
- **Single-character labels** (for example `A` or `7`) are treated as fingerspelling and are joined into a word. Up to three completion suggestions are shown, which can be accepted with `1`, `2`, or `3`.

Signs with the following reserved labels act as controls instead of words, so the sentence can be edited without using the keyboard:

| Label    | Function                                 |
|----------|------------------------------------------|
| `#SPACE` | Finish the word currently being spelled  |
| `#DEL`   | Delete the last letter or word           |
| `#CLEAR` | Clear the sentence                       |
| `#SPEAK` | Speak the sentence aloud                 |
| `#OK`    | Accept the top suggestion                |

### Keyboard controls

| Key           | Action                                        |
|---------------|-----------------------------------------------|
| `N`           | Record a new static sign, or add samples to an existing one |
| `M`           | Record a new moving sign                      |
| `X`           | Delete a sign                                 |
| `L`           | Start practice mode                           |
| `P`           | Switch to or create a profile                 |
| `E` / `I`     | Export / import signs                         |
| `S`           | Speak the current sentence                    |
| `V`           | Turn voice output on or off                   |
| `BACKSPACE`   | Delete the last letter or word                |
| `C`           | Clear the sentence                            |
| `1` `2` `3`   | Accept a spelling suggestion                  |
| `+` / `-`     | Increase / decrease matching tolerance        |
| `H`           | Show or hide help                             |
| `ESC`         | Cancel the current action (quits when idle)   |
| `Q`           | Quit                                          |

## Data storage

| Path                | Contents                                                          |
|---------------------|-------------------------------------------------------------------|
| `profiles/<NAME>.json` | Signs, practice statistics, and word history for each profile |
| `profiles/_last.txt`   | Name of the most recently used profile                         |
| `exports/`             | Timestamped profile exports                                    |
| `gestures.json`        | Signs recorded with the legacy `custom_alphabet.py` tool       |

On first launch, signs in `gestures.json` are imported into the `DEFAULT` profile automatically. Profile files are written atomically to prevent corruption if the application is interrupted.

## Configuration

All tunable parameters are defined in [`signlang/config.py`](signlang/config.py). The most commonly adjusted settings are:

| Setting            | Default | Description                                                       |
|--------------------|---------|-------------------------------------------------------------------|
| `CAM_INDEX`        | `0`     | Index of the camera device                                        |
| `MIRROR`           | `True`  | Display the camera feed mirrored                                  |
| `MATCH_THRESHOLD`  | `1.40`  | Matching tolerance; higher values are more lenient                |
| `HOLD_TO_ADD_SECS` | `1.0`   | How long a static sign must be held before it is added            |
| `SWAP_HANDEDNESS`  | `False` | Set to `True` if left/right hand labels appear reversed           |
| `SPEAK_EACH_WORD`  | `True`  | Speak each word as it is added, in addition to full sentences     |

Words used for fingerspelling suggestions are listed in [`signlang/data/common_words.txt`](signlang/data/common_words.txt) and can be extended.

## Recognition method

1. **Landmark detection.** MediaPipe Hand Landmarker produces 21 three-dimensional landmarks and a left/right classification for each visible hand.
2. **Normalisation.** Landmarks are translated so that the wrist is the origin and scaled by palm size (the distance from the wrist to the middle-finger knuckle). This makes features independent of hand position and distance from the camera.
3. **Static matching.** The live hand shape is compared to each stored sample using Euclidean distance. The comparison is evaluated across a range of wrist rotations (±30°), and includes weighted terms for depth and handedness. For two-hand signs, the distance is taken from the worse-matching hand, so both hands must match, plus a term for the relative position of the hands.
4. **Dynamic matching.** Movements are encoded as sequences of hand shape and wrist trajectory, resampled to a fixed length, and compared with dynamic time warping (DTW). Several playback speeds are evaluated to allow for variation in signing speed.
5. **Decision.** The closest sign below the matching threshold is accepted. Results are smoothed over consecutive frames to suppress flicker.

## Project structure

```
sign_translator.py        Application entry point
signlang/
  config.py               Configuration parameters
  tracker.py              MediaPipe hand tracking
  features.py             Landmark normalisation and static distance metrics
  motion.py               Movement encoding and DTW matching
  matcher.py              Live recognition, smoothing, and look-alike detection
  recorder.py             Sign recording sessions
  sentence.py             Sentence construction and word suggestions
  speech.py               Text-to-speech output
  practice.py             Practice mode
  storage.py              Profiles, persistence, export, and import
  ui.py                   On-screen rendering
  data/common_words.txt   Vocabulary for fingerspelling suggestions
hand_landmarker.task      MediaPipe hand landmark model
```

### Legacy tools

The following standalone scripts predate the main application. They are kept for reference and testing, and require `cvzone`.

| Script               | Description                                                              |
|----------------------|--------------------------------------------------------------------------|
| `custom_alphabet.py` | Earlier single-file version supporting static one- and two-hand signs    |
| `hand_tracking.py`   | Hand tracking demonstration: landmarks, bounding boxes, finger count     |
| `finger_counter.py`  | Displays the total number of extended fingers (0–10)                     |

## Limitations

- Recognition is user-dependent. Signs recorded by one person may not be recognised reliably for another, which is why separate profiles are provided.
- Performance depends on lighting, camera quality, and hands being clearly visible in the frame.
- The system recognises discrete signs and does not interpret the grammar of any natural sign language, such as ASL or BSL. Facial expressions and body posture are not considered.
