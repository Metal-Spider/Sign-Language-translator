"""
All tunable settings in one place.
Distances are measured in "palm sizes" (wrist -> middle knuckle), so they
don't depend on how far the hand is from the camera.
"""

# -- Camera ---------------------------------------------------------------------
CAM_INDEX   = 0
CAM_WIDTH   = 1280
CAM_HEIGHT  = 720
MIRROR      = True      # show the camera like a mirror (also makes Left/Right correct)

# -- Hand tracking (MediaPipe) ----------------------------------------------------
MODEL_PATH       = "hand_landmarker.task"
MODEL_URL        = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                    "hand_landmarker/float16/1/hand_landmarker.task")
MAX_HANDS        = 2
DETECTION_CONF   = 0.6
PRESENCE_CONF    = 0.6
TRACKING_CONF    = 0.5
# If the L/R tags drawn on your hands are the wrong way round, set this to True
SWAP_HANDEDNESS  = False

# -- Storage ----------------------------------------------------------------------
PROFILES_DIR     = "profiles"
EXPORTS_DIR      = "exports"
LEGACY_FILE      = "gestures.json"     # file used by the older versions
DEFAULT_PROFILE  = "DEFAULT"

# -- Matching (still signs) ---------------------------------------------------------
MATCH_THRESHOLD  = 1.40     # higher = more lenient (adjust live with + / -)
THRESHOLD_MIN    = 0.30
THRESHOLD_MAX    = 3.50
THRESHOLD_STEP   = 0.15
ROTATION_ANGLES  = (-30, -20, -10, -5, 0, 5, 10, 20, 30)   # wrist-tilt tolerance
REL_WEIGHT       = 0.5      # how much the spacing between two hands matters
Z_WEIGHT         = 0.35     # how much depth (fingers toward/away from camera) matters
SIDE_PENALTY     = 0.6      # extra distance when the wrong hand (L vs R) is used
LOOKALIKE_RATIO  = 1.0      # warn when two signs are closer than threshold * this

# -- Recording -------------------------------------------------------------------
RECORD_FRAMES        = 20   # frames averaged into one sample of a still sign
SAMPLES_PER_SESSION  = 3    # each recording captures this many samples in a row
MAX_SAMPLES          = 6    # samples kept per sign (oldest dropped first)
COUNTDOWN_SECS       = 3
REST_SECS            = 1.5  # pause between samples

# -- Moving signs ------------------------------------------------------------------
MOTION_RESAMPLE        = 16          # points each movement is resampled to
MOTION_BAND            = 4           # DTW band width (how much timing may warp)
MOTION_SHAPE_WEIGHT    = 0.5         # hand shape vs. path of the hand
MOTION_TRAJ_WEIGHT     = 1.0
MOTION_THRESHOLD_SCALE = 0.8         # moving-sign threshold = MATCH_THRESHOLD * this
MOTION_SPEED_SCALES    = (0.75, 1.0, 1.33)   # allow signing faster / slower
MOTION_START_SPEED     = 2.0         # palm sizes per second that counts as "moving"
MOTION_MIN_SECS        = 0.4
MOTION_MAX_SECS        = 3.5
MOTION_STOP_SECS       = 0.35        # still this long = movement finished
MOTION_WAIT_SECS       = 6.0         # give up recording if no movement starts
MOTION_PREROLL_SECS    = 0.2
MOTION_COOLDOWN_SECS   = 1.0
MOTION_MIN_EXTENT      = 0.4         # live movement must be >= 40% as big as recorded
MOTION_BUFFER_SECS     = 5.0
MOTION_CHECK_EVERY     = 2           # run moving-sign matching every N frames

# -- Detection & sentence -------------------------------------------------------------
SMOOTH_FRAMES     = 5
STILL_SPEED       = 1.5     # palm sizes per second; still signs only count when slower
HOLD_TO_ADD_SECS  = 1.0     # hold a still sign this long to add it to the sentence
NUM_SUGGESTIONS   = 3
VOICE_ENABLED     = True
SPEAK_EACH_WORD   = True    # say each word as it's added (plus whole sentence on SPEAK)

# Saved signs with these names act as controls instead of words
CONTROL_SIGNS = {
    "#SPACE": "finish the word being spelled",
    "#DEL":   "delete last letter / word",
    "#CLEAR": "clear the sentence",
    "#SPEAK": "speak the sentence",
    "#OK":    "accept the top suggestion",
}

# -- Practice ------------------------------------------------------------------------
PRACTICE_TIMEOUT_SECS  = 8.0
PRACTICE_HOLD_SECS     = 0.4
PRACTICE_FEEDBACK_SECS = 1.2

# -- UI ------------------------------------------------------------------------------
SIDEBAR_W     = 270
BOTTOM_H      = 110
MAX_LABEL_LEN = 16
