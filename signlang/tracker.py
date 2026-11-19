"""
Hand tracking using MediaPipe directly (instead of the cvzone wrapper),
so we keep real depth (z) and Left/Right handedness for every hand.
"""

import os
import time
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from . import config as cfg


@dataclass
class Hand:
    lm: list            # 21 x [x_px, y_px, z_px]; z uses the same pixel scale as x
    side: str           # "Left", "Right" or "Unknown" (the person's own hand)
    score: float        # handedness confidence


class HandTracker:
    def __init__(self, model_path=cfg.MODEL_PATH, max_hands=cfg.MAX_HANDS):
        if not os.path.exists(model_path):
            print("Downloading MediaPipe hand model...")
            urllib.request.urlretrieve(cfg.MODEL_URL, model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=cfg.DETECTION_CONF,
            min_hand_presence_confidence=cfg.PRESENCE_CONF,
            min_tracking_confidence=cfg.TRACKING_CONF,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, img_bgr):
        """Return a list of Hand objects for a BGR frame."""
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int(time.monotonic() * 1000)           # VIDEO mode needs rising timestamps
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._landmarker.detect_for_video(image, ts)
        hands = []
        for i, landmarks in enumerate(result.hand_landmarks):
            lm = [[int(p.x * w), int(p.y * h), p.z * w] for p in landmarks]
            side, score = "Unknown", 0.0
            if i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                side, score = cat.category_name, cat.score
                if cfg.SWAP_HANDEDNESS and side in ("Left", "Right"):
                    side = "Right" if side == "Left" else "Left"
            hands.append(Hand(lm=lm, side=side, score=score))
        return hands

    def close(self):
        self._landmarker.close()
