"""
feature_extractor.py
--------------------
HOG feature extraction for Bird-vs-Drone patch classification.

All patches are assumed to be uint8 RGB arrays of shape (H, W, 3).
The public API:

    extract_hog(image)          -> 1-D float32 feature vector
    extract_hog_batch(images)   -> 2-D float32 array  [N, feature_dim]
    get_feature_dim()           -> int  (cached after first call)

HOG configuration
-----------------
  window_size      : 64 × 64  pixels
  orientations     : 9
  pixels_per_cell  : (8, 8)
  cells_per_block  : (2, 2)
  channel_axis     : -1   (colour HOG, last axis = channels)
  transform_sqrt   : True  (Gamma correction for illumination robustness)
  feature_vector   : True  (return flat array)

Expected feature_dim = 9 * ((64/8 - 1)^2) * 4  orientations * 4 blocks
  = 9 * 49 * 4 = 1764  (for single-channel)
  × 3 channels → 7 * 7 * (2*2) * 9 * 3 = 9 * 7 * 7 * 4 * 3 = 5292
  (exact value is computed at runtime via skimage and cached)
"""

from __future__ import annotations

from typing import Optional, Callable

import cv2
import numpy as np
from skimage.feature import hog

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

WINDOW_SIZE: tuple[int, int] = (64, 64)   # (width, height)

HOG_PARAMS: dict = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "channel_axis": -1,      # colour HOG; skimage uses last axis for channels
    "transform_sqrt": True,
    "feature_vector": True,
}

# ---------------------------------------------------------------------------
# Module-level cache so get_feature_dim() is only computed once per process
# ---------------------------------------------------------------------------
_FEATURE_DIM_CACHE: Optional[int] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_hog(image: np.ndarray) -> np.ndarray:
    """
    Compute the HOG feature vector for a single image patch.

    The function is robust to:
      - Any spatial size  → resized to WINDOW_SIZE before extraction.
      - Grayscale input   → duplicated to 3 channels so channel_axis=-1 works.
      - float or uint8    → always converted to float32 in [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image.  Expected shape (H, W) or (H, W, 1) or (H, W, 3/4).
        dtype may be uint8 or float.

    Returns
    -------
    np.ndarray
        1-D float32 HOG feature vector of length ``get_feature_dim()``.
    """
    # ---- ensure 3-channel uint8 ----
    img = _to_3ch_uint8(image)

    # ---- resize to window size if necessary ----
    h, w = img.shape[:2]
    win_w, win_h = WINDOW_SIZE
    if w != win_w or h != win_h:
        img = cv2.resize(img, WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)

    # ---- convert to float32 [0, 1] ----
    img_f = img.astype(np.float32) / 255.0

    # ---- extract HOG ----
    feat = hog(img_f, **HOG_PARAMS)
    return feat.astype(np.float32)


def extract_hog_batch(
    images: np.ndarray,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """
    Compute HOG features for an array of patches.

    Parameters
    ----------
    images : np.ndarray
        Shape (N, H, W, 3) or (N, H, W) — array of image patches.
    progress_callback : optional callable
        Called as ``callback(percent: int)`` every 200 images and at the end.
        *percent* is an integer in [0, 100].

    Returns
    -------
    np.ndarray
        Float32 array of shape (N, feature_dim).
    """
    n = len(images)
    if n == 0:
        return np.empty((0, get_feature_dim()), dtype=np.float32)

    features = np.empty((n, get_feature_dim()), dtype=np.float32)

    for i, img in enumerate(images):
        features[i] = extract_hog(img)

        # Report progress every 200 images and on the last one
        if progress_callback is not None:
            if i % 200 == 0 or i == n - 1:
                pct = int((i + 1) / n * 100)
                progress_callback(pct)

    return features


def get_feature_dim() -> int:
    """
    Return the length of the HOG feature vector produced by ``extract_hog``.

    The value is computed once by running HOG on a blank window and then
    cached in a module-level variable so subsequent calls are O(1).

    Returns
    -------
    int
        Number of elements in the HOG descriptor for WINDOW_SIZE.
    """
    global _FEATURE_DIM_CACHE

    if _FEATURE_DIM_CACHE is not None:
        return _FEATURE_DIM_CACHE

    # Create a blank 3-channel float32 image of the target window size
    win_w, win_h = WINDOW_SIZE
    blank = np.zeros((win_h, win_w, 3), dtype=np.float32)
    feat = hog(blank, **HOG_PARAMS)
    _FEATURE_DIM_CACHE = int(feat.shape[0])
    return _FEATURE_DIM_CACHE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_3ch_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert any image array to a 3-channel uint8 array.

    Conversion rules
    ----------------
    - float images in [0, 1]   → multiply by 255, cast to uint8
    - float images in (1, 255] → cast directly to uint8
    - grayscale (H, W)          → stack 3 copies along channel axis
    - grayscale (H, W, 1)       → repeat channel axis to get (H, W, 3)
    - 4-channel RGBA            → drop alpha, keep RGB
    """
    img = np.asarray(image)

    # Normalise float to uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0 and img.dtype.kind == "f":
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    # Ensure 3 channels
    if img.ndim == 2:
        # Pure grayscale (H, W) → (H, W, 3)
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        c = img.shape[2]
        if c == 1:
            # (H, W, 1) → (H, W, 3)
            img = np.concatenate([img, img, img], axis=-1)
        elif c == 4:
            # RGBA → RGB
            img = img[:, :, :3]
        elif c == 3:
            pass  # already correct
        else:
            raise ValueError(f"Unsupported number of channels: {c}")
    else:
        raise ValueError(f"Expected 2-D or 3-D image array, got shape {img.shape}")

    return img
