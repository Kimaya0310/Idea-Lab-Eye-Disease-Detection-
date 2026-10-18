"""
preprocessing.py -- deterministic fundus preprocessing with on-disk caching
(no torch dependency).

Pipeline:  FOV crop -> pad to square (no stretching) -> resize -> optional
gray-world colour norm -> optional CLAHE -> [cached uint8] -> ImageNet norm.

The cached uint8 image is what augmentation operates on, so the (slow) 8-MP
decode + resize happens once per image, not once per epoch.
"""
import hashlib
import os
from typing import Optional

import cv2
import numpy as np

import config
from disc_crop import crop_disc_roi


def crop_to_fov(img: np.ndarray, thresh: int = config.FOV_THRESHOLD) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else img
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    mask = (gray > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(ys) < 0.05 * mask.size:
        return img
    y0, y1 = int(ys.min() / scale), int(min(h, (ys.max() + 1) / scale + 1))
    x0, x1 = int(xs.min() / scale), int(min(w, (xs.max() + 1) / scale + 1))
    crop = img[y0:y1, x0:x1]
    if crop.shape[0] * crop.shape[1] < 0.2 * h * w:
        return img
    return crop


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def gray_world(img: np.ndarray) -> np.ndarray:
    mask = img.max(axis=2) > config.FOV_THRESHOLD
    if mask.sum() < 100:
        return img
    f = img.astype(np.float32)
    means = np.array([f[..., c][mask].mean() for c in range(3)]) + 1e-6
    gains = np.clip(means.mean() / means, 0.5, 3.0)
    out = np.clip(f * gains, 0, 255)
    out[~mask] = 0
    return out.astype(np.uint8)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


class FundusPreprocessor:
    def __init__(self, image_size: int = config.IMAGE_SIZE, use_cache: bool = config.USE_CACHE):
        self.image_size = image_size
        self.use_cache = use_cache
        self.mean = np.array(config.IMAGENET_MEAN, np.float32)
        self.std = np.array(config.IMAGENET_STD, np.float32)
        tag = f"s{image_size}_fov{config.FOV_THRESHOLD}_cl{int(config.ENABLE_CLAHE)}_cn{config.COLOR_NORM}_dc{int(config.DISC_CROP)}{config.DISC_ROI_FRAC}"
        self.cache_dir = os.path.join(config.CACHE_DIR, tag)

    # ---- pure image -> image ------------------------------------------------
    def process(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        img = pad_to_square(crop_to_fov(rgb))
        if config.DISC_CROP:
            img = crop_disc_roi(img, config.DISC_ROI_FRAC, config.FOV_THRESHOLD)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if config.COLOR_NORM == "gray_world":
            img = gray_world(img)
        if config.ENABLE_CLAHE:
            img = apply_clahe(img)
        return img

    # ---- path -> cached uint8 ------------------------------------------------
    def cache_path(self, image_path: str) -> str:
        h = hashlib.sha1(os.path.abspath(image_path).lower().encode()).hexdigest()[:20]
        return os.path.join(self.cache_dir, h + ".png")

    def load_processed(self, image_path: str) -> Optional[np.ndarray]:
        """Returns uint8 RGB (S,S,3), or None if the source cannot be read."""
        cp = self.cache_path(image_path)
        if self.use_cache and os.path.exists(cp):
            bgr = cv2.imread(cp, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        out = self.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = f"{cp}.{os.getpid()}.tmp.png"
            cv2.imwrite(tmp, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            os.replace(tmp, cp)
        return out

    # ---- uint8 -> normalized float ---------------------------------------------
    def normalize(self, img_uint8: np.ndarray) -> np.ndarray:
        x = img_uint8.astype(np.float32) / 255.0
        return ((x - self.mean) / self.std).astype(np.float32)   # HWC

    def preprocess_for_display(self, rgb: np.ndarray) -> np.ndarray:
        return self.process(rgb)
