"""
disc_crop.py -- optic-disc-centred region of interest (no torch).

Why: glaucoma is judged on the optic disc / cup / rim. At 384 px the whole
fundus puts the disc in ~50 px, so the network barely sees the cup. Cropping a
window around the disc (taken from the ORIGINAL high-res image) gives the CNN
~2.5x more pixels on the structure that actually carries the label.

Localisation is a simple heuristic (brightest large blob in the smoothed
red+green image, inside the FOV). It is NOT a trained detector: check a
montage of crops (python disc_crop.py --check <folder>) before trusting it,
and it falls back to the image centre if nothing sensible is found.
"""
import cv2
import numpy as np


def find_disc_center(img: np.ndarray, fov_thresh: int = 12):
    """img: uint8 RGB (square, FOV-cropped). Returns (cx, cy) in img pixels."""
    h, w = img.shape[:2]
    s = 256.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    fov = (small.max(axis=2) > fov_thresh).astype(np.uint8)
    fov = cv2.erode(fov, np.ones((15, 15), np.uint8))   # keep away from rim glare
    if fov.sum() < 0.1 * fov.size:
        return w / 2.0, h / 2.0
    # disc is bright in red AND green; vessels/glare are less consistent
    g = (small[..., 0].astype(np.float32) + small[..., 1].astype(np.float32)) / 2.0
    k = int(sw * 0.12) | 1
    g = cv2.GaussianBlur(g, (k, k), 0)
    g[fov == 0] = 0
    _, _, _, loc = cv2.minMaxLoc(g)
    return loc[0] / s, loc[1] / s


def crop_disc_roi(img: np.ndarray, roi_frac: float = 0.40, fov_thresh: int = 12) -> np.ndarray:
    """Square crop of side roi_frac*min(h,w) centred on the disc, clamped inside the image."""
    h, w = img.shape[:2]
    cx, cy = find_disc_center(img, fov_thresh)
    side = int(round(roi_frac * min(h, w)))
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return img[y0:y0 + side, x0:x0 + side]
