"""
augmentations.py -- train-only augmentation on the cached uint8 RGB image.

Everything keeps the black outside-FOV region black (constant border, masked
colour ops) so augmentation cannot introduce border artefacts that differ by
dataset. Resolution jitter deliberately blurs per-camera sharpness cues.
"""
import random

import cv2
import numpy as np

import config


class FundusAugmentor:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < config.AUG_HFLIP_PROB:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img)
        img = self._affine(img)
        img = self._color(img)
        if random.random() < config.AUG_RESOLUTION_JITTER_PROB:
            h, w = img.shape[:2]
            f = random.uniform(0.5, 0.8)
            small = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if random.random() < config.AUG_BLUR_PROB:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    @staticmethod
    def _affine(img):
        h, w = img.shape[:2]
        angle = random.uniform(-config.AUG_ROTATION_DEGREES, config.AUG_ROTATION_DEGREES)
        scale = random.uniform(*config.AUG_SCALE_RANGE)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += random.uniform(-config.AUG_TRANSLATE, config.AUG_TRANSLATE) * w
        M[1, 2] += random.uniform(-config.AUG_TRANSLATE, config.AUG_TRANSLATE) * h
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    @staticmethod
    def _color(img):
        fov = img.max(axis=2) > 0
        x = img.astype(np.float32) / 255.0
        x *= 1.0 + random.uniform(-config.AUG_BRIGHTNESS, config.AUG_BRIGHTNESS)
        if fov.any():
            m = x[fov].mean()
            x = (x - m) * (1.0 + random.uniform(-config.AUG_CONTRAST, config.AUG_CONTRAST)) + m
        x = np.clip(x, 0, 1)
        x = x ** (1.0 + random.uniform(-config.AUG_GAMMA, config.AUG_GAMMA))
        hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + random.uniform(-config.AUG_SATURATION, config.AUG_SATURATION)), 0, 255)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-config.AUG_HUE, config.AUG_HUE) * 179) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        out[~fov] = 0
        return out
