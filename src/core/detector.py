"""
Sliding-window + image-pyramid detector for Bird vs Drone classification.

Pipeline per frame
------------------
1. Build an image pyramid (up to 6 levels, scaling down by scale_factor).
2. For each pyramid level and each window size, tile sliding windows over
   the scaled image.
3. Filter blank/sky patches via HOG energy (_has_content).
4. Batch-extract HOG features for all surviving windows in one call.
5. Batch-predict with the model (predict_proba_all + predict).
6. Apply confidence threshold AND margin threshold to filter uncertain windows.
7. Collect detections above threshold, map coords back to original scale.
8. Apply per-class Non-Maximum Suppression (IoU=0.3) across all pyramid levels
   and window sizes.
9. Return (x, y, w, h, confidence, label_str) tuples.
"""

from __future__ import annotations

import logging
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from .feature_extractor import WINDOW_SIZE, extract_hog_batch

logger = logging.getLogger(__name__)

# Type aliases
Box = List[float]          # [x1, y1, x2, y2]
Detection = Tuple[int, int, int, int, float, str]   # x, y, w, h, conf, label

# 3-class label mapping: 0 = background, 1 = Bird, 2 = Drone
_LABEL_MAP = {0: "Background", 1: "Bird", 2: "Drone"}


# ---------------------------------------------------------------------------
# Background content filter
# ---------------------------------------------------------------------------

def _has_content(patch: np.ndarray, min_energy: float = 0.01) -> bool:
    """Return False for patches that are mostly blank sky/uniform regions.
    Uses mean HOG image energy as a proxy for visual complexity."""
    from skimage.feature import hog as _hog
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    gray_f = gray.astype(np.float32) / 255.0
    _, hog_img = _hog(
        gray_f, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), visualize=True, feature_vector=True,
    )
    return float(np.mean(hog_img)) >= min_energy


# ---------------------------------------------------------------------------
# Image pyramid
# ---------------------------------------------------------------------------

def image_pyramid(
    image: np.ndarray,
    scale_factor: float = 0.8,
    min_size: Tuple[int, int] = (64, 64),
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """Yield (scaled_image, current_scale_ratio) tuples.

    Starts at (image, 1.0) and multiplies scale by scale_factor each step.
    Stops when either dimension would fall below min_size, or after 6 levels.

    Parameters
    ----------
    image : np.ndarray
        Input BGR or grayscale image.
    scale_factor : float
        Factor < 1.0 by which the image shrinks each step.
    min_size : tuple[int, int]
        (min_width, min_height) stopping condition.
    """
    current = image.copy()
    current_scale = 1.0
    max_levels = 6

    for _ in range(max_levels):
        yield current, current_scale

        # Compute next size
        next_w = int(current.shape[1] * scale_factor)
        next_h = int(current.shape[0] * scale_factor)

        if next_w < min_size[0] or next_h < min_size[1]:
            break

        current = cv2.resize(current, (next_w, next_h), interpolation=cv2.INTER_AREA)
        current_scale *= scale_factor


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def sliding_window(
    image: np.ndarray,
    stride: int,
    win_size: Tuple[int, int] = WINDOW_SIZE,
) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """Yield (x, y, patch) for every sliding window position.

    Parameters
    ----------
    image : np.ndarray
        Image at the current pyramid level.
    stride : int
        Step size in pixels.
    win_size : tuple[int, int]
        (width, height) of each patch.
    """
    win_w, win_h = win_size
    img_h, img_w = image.shape[:2]

    y = 0
    while y + win_h <= img_h:
        x = 0
        while x + win_w <= img_w:
            yield x, y, image[y : y + win_h, x : x + win_w]
            x += stride
        y += stride


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------

def non_max_suppression(
    boxes: List[Box],
    scores: List[float],
    overlap_thresh: float = 0.3,
) -> Tuple[List[Box], List[float]]:
    """Greedy IoU-based NMS.

    Parameters
    ----------
    boxes : list of [x1, y1, x2, y2]
    scores : list of float
    overlap_thresh : float
        IoU threshold above which the lower-scoring box is suppressed.

    Returns
    -------
    (kept_boxes, kept_scores)
    """
    if len(boxes) == 0:
        return [], []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)

    # Sort indices by descending score
    order = np.argsort(scores_arr)[::-1]

    kept_indices: List[int] = []

    while order.size > 0:
        i = int(order[0])
        kept_indices.append(i)

        if order.size == 1:
            break

        rest = order[1:]

        # Intersection
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])

        iw = np.maximum(0.0, ix2 - ix1 + 1.0)
        ih = np.maximum(0.0, iy2 - iy1 + 1.0)
        intersection = iw * ih

        iou = intersection / (areas[i] + areas[rest] - intersection)

        # Keep boxes whose IoU with current is below threshold
        order = rest[iou <= overlap_thresh]

    kept_boxes = boxes_arr[kept_indices].tolist()
    kept_scores = scores_arr[kept_indices].tolist()
    return kept_boxes, kept_scores


# ---------------------------------------------------------------------------
# Main detect function
# ---------------------------------------------------------------------------

def detect(
    image: np.ndarray,
    model,
    seq_length: int = 8,
    confidence_threshold: float = 0.5,
    window_sizes: list | None = None,   # default: [(64,64), (96,96), (128,128)]
) -> List[Detection]:
    """Run multi-scale sliding-window detection on a single frame.

    Parameters
    ----------
    image : np.ndarray
        BGR image (as returned by cv2.imread or cv2.VideoCapture.read).
    model : BirdDroneModel
        A trained model exposing predict_proba_all(X) and predict(X).
    seq_length : int
        Controls sliding stride: stride = max(4, int(32 - seq_length * 2.0)).
        Range 1-16 maps stride from ~30 down to 4.
    confidence_threshold : float
        Minimum object confidence (1 - P(background)) to consider a detection.
    window_sizes : list or None
        List of (width, height) tuples to use as sliding window sizes.
        Defaults to [(64, 64), (96, 96), (128, 128)].

    Returns
    -------
    list of (x, y, w, h, confidence, label_str)
        Coordinates in original image space, after per-class NMS.
        Returns [] if the model is not trained or no detections survive NMS.
    """
    if not model.is_trained:
        logger.debug("detect() called on untrained model — returning empty list.")
        return []

    if window_sizes is None:
        window_sizes = [(64, 64), (96, 96), (128, 128)]

    stride = max(4, int(32 - seq_length * 2.0))
    win_w, win_h = WINDOW_SIZE   # HOG extraction target size (64, 64)

    all_boxes: List[Box] = []
    all_scores: List[float] = []
    all_labels: List[int] = []

    for scaled_img, scale_ratio in image_pyramid(image):
        for ws in window_sizes:
            # Collect raw positions and patches for this pyramid level + window size
            raw_positions: List[Tuple[int, int]] = []
            raw_patches: List[np.ndarray] = []

            for x, y, patch in sliding_window(scaled_img, stride=stride, win_size=ws):
                raw_positions.append((x, y))
                raw_patches.append(patch)

            if len(raw_patches) == 0:
                continue

            # Filter blank/sky patches before HOG extraction
            positions: List[Tuple[int, int]] = []
            patches: List[np.ndarray] = []
            for pos, patch in zip(raw_positions, raw_patches):
                if _has_content(patch):
                    positions.append(pos)
                    patches.append(patch)

            if len(patches) == 0:
                continue

            # Batch HOG + predict — one call per (pyramid level, window size)
            features = extract_hog_batch(patches)           # shape [N, F]
            probas_all = model.predict_proba_all(features)  # shape [N, 3]

            # confidence = P(any object) = 1 - P(background)
            object_conf = 1.0 - probas_all[:, 0]

            # margin = difference between top-2 class probabilities
            sorted_p = np.sort(probas_all, axis=1)
            margin = sorted_p[:, -1] - sorted_p[:, -2]     # shape [N]

            preds = model.predict(features)                 # shape [N] — 0/1/2

            inv_scale = 1.0 / scale_ratio
            ws_w, ws_h = ws

            for idx, (px, py) in enumerate(positions):
                cls_id_for_thresh = int(preds[idx])
                # Drone gets a lower bar to compensate for sparse training data
                _thresh = (confidence_threshold if cls_id_for_thresh != 2
                           else max(0.1, confidence_threshold - 0.15))
                if object_conf[idx] < _thresh:
                    continue
                # Drone class: accept lower margin — model is uncertain on sparse class
                _margin_thresh = 0.15 if int(preds[idx]) != 2 else 0.05
                if margin[idx] < _margin_thresh:
                    continue
                if int(preds[idx]) == 0:
                    continue

                # Map window top-left + size to original image space
                orig_x1 = int(round(px * inv_scale))
                orig_y1 = int(round(py * inv_scale))
                orig_x2 = int(round((px + ws_w) * inv_scale))
                orig_y2 = int(round((py + ws_h) * inv_scale))

                all_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
                all_scores.append(float(object_conf[idx]))
                all_labels.append(int(preds[idx]))

    if len(all_boxes) == 0:
        return []

    # Per-class NMS — Bird=1, Drone=2 (Background=0 already filtered above)
    final_boxes: List[Box] = []
    final_scores: List[float] = []
    final_labels: List[int] = []

    for cls_id in [1, 2]:
        idxs = [i for i, l in enumerate(all_labels) if l == cls_id]
        if not idxs:
            continue
        cls_boxes  = [all_boxes[i]  for i in idxs]
        cls_scores = [all_scores[i] for i in idxs]
        kept_b, kept_s = non_max_suppression(cls_boxes, cls_scores, overlap_thresh=0.3)
        final_boxes.extend(kept_b)
        final_scores.extend(kept_s)
        final_labels.extend([cls_id] * len(kept_b))

    results: List[Detection] = []
    for box, score, label_id in zip(final_boxes, final_scores, final_labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        w = x2 - x1
        h = y2 - y1
        label_str = _LABEL_MAP.get(label_id, f"Class{label_id}")
        results.append((x1, y1, w, h, float(score), label_str))

    return results
