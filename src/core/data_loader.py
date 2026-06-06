"""
data_loader.py
--------------
Loads Bird-vs-Drone YOLO-format data into patch arrays for classical ML.

Label file format (one annotation per line):
    line_number<TAB>class_id x1 y1 x2 y2 ...

  - 4 coordinate floats  → standard YOLO bbox:  x_center y_center width height
  - >=8 coordinate floats → polygon: interleaved (x y) pairs normalised [0,1]

All coordinates are normalised.

⚠  BUG DOCUMENTED (FIX_PLAN 2026-05-25)
-----------------------------------------
The comment below was WRONG. Verified from actual label files:
  - YOLO class 0 = DRONE  (DT* image files — confirmed: "0 cx cy w h")
  - YOLO class 1 = BIRD   (BT* image files — confirmed: "1 cx cy w h")

The old comment claimed the opposite. This reversed mapping caused the
"Drone → Bird 100%" classification bug:
  load_dataset() maps class_id==0 → y=1 (Bird) and class_id==1 → y=2 (Drone)
  But since class 0 IS the drone in the data, every drone was trained as Bird.

THIS FILE IS KEPT FOR BACKWARD COMPATIBILITY with the legacy HOG+MLP pipeline.
New code should use the YOLOv10 pipeline:
  - datasets/data.yaml     — correct class mapping (0=drone, 1=bird)
  - src/core/yolov10_detector.py — inference
  - scripts/train_yolov10.py     — training

If you must use this classical ML loader, fix load_dataset() below:
  class_id == 0 → y=2 (Drone)   ← was y=1 (Bird)  ← THIS IS THE BUG
  class_id == 1 → y=1 (Bird)    ← was y=2 (Drone)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
BBox = Tuple[int, int, int, int]          # (x1, y1, x2, y2) in pixel space
PatchArray = np.ndarray                   # uint8 RGB image patch


# ---------------------------------------------------------------------------
# 1. Label parser
# ---------------------------------------------------------------------------

def parse_yolo_label(
    label_path: str | Path,
    img_w: int,
    img_h: int,
) -> List[Tuple[BBox, int]]:
    """
    Parse one YOLO v7 label file and return pixel-space bounding boxes with
    their class IDs.

    Handles two annotation styles present in this dataset:
      • Standard bbox  : class_id  cx cy  w h          (5 tokens per line)
      • Polygon        : class_id  x1 y1 x2 y2 …      (>=9 tokens per line)

    Note: each line is prefixed by a 1-based line number followed by a TAB,
    so raw token layout is: line_no class_id coord …

    Parameters
    ----------
    label_path : path to the .txt annotation file
    img_w, img_h : pixel dimensions of the matching image

    Returns
    -------
    List of ((x1, y1, x2, y2), class_id) tuples.
    Boxes are clipped to [0, img_w/h].
    Degenerate boxes (zero area) are excluded.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    results: List[Tuple[BBox, int]] = []

    with label_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()

            # Standard YOLO format: class_id coord0 coord1 …
            # tokens[0] = class_id (int), tokens[1:] = normalised coords
            if len(tokens) < 5:
                continue   # need at least class + 4 bbox coords

            try:
                class_id = int(tokens[0])
            except ValueError:
                continue  # malformed class_id

            coord_tokens = tokens[1:]
            try:
                coords = [float(v) for v in coord_tokens]
            except ValueError:
                continue  # malformed line

            n_coords = len(coords)

            if n_coords == 4:
                # Standard YOLO bbox: cx cy w h (all normalised)
                cx, cy, bw, bh = coords
                x1 = (cx - bw / 2) * img_w
                y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w
                y2 = (cy + bh / 2) * img_h

            elif n_coords >= 8 and n_coords % 2 == 0:
                # Polygon: interleaved x y pairs, normalised
                xs = [coords[i] * img_w for i in range(0, n_coords, 2)]
                ys = [coords[i] * img_h for i in range(1, n_coords, 2)]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)

            else:
                # Unrecognised format — skip
                continue

            # Clip to image bounds
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(img_w), x2)
            y2 = min(float(img_h), y2)

            # Discard degenerate boxes
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            results.append(((int(x1), int(y1), int(x2), int(y2)), class_id))

    return results


# ---------------------------------------------------------------------------
# 2. IoU utility
# ---------------------------------------------------------------------------

def iou(box1: BBox, box2: BBox) -> float:
    """
    Intersection-over-Union for two (x1, y1, x2, y2) boxes.

    Returns a float in [0, 1].  Returns 0 when both boxes have zero area.
    """
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# ---------------------------------------------------------------------------
# 3. Negative patch sampler
# ---------------------------------------------------------------------------

def get_negative_patches(
    image: np.ndarray,
    bboxes: List[BBox],
    n: int,
    window_size: Tuple[int, int] = (64, 64),
) -> List[np.ndarray]:
    """
    Sample random crops from *image* that do not significantly overlap any
    positive bounding box.

    A candidate crop is accepted when its IoU with every ground-truth box
    is below 0.3.  Up to 100 random positions are tried; if fewer than *n*
    valid negatives are found, the available ones are returned.

    Parameters
    ----------
    image       : H×W×3 uint8 BGR/RGB array (as loaded by cv2).
    bboxes      : list of (x1, y1, x2, y2) positive boxes (pixel coords).
    n           : number of negative patches to collect.
    window_size : (width, height) of each returned patch.

    Returns
    -------
    List of uint8 patches, each resized to window_size (W×H×3).
    """
    img_h, img_w = image.shape[:2]
    win_w, win_h = window_size
    patches: List[np.ndarray] = []

    if img_w < win_w or img_h < win_h:
        return patches

    max_attempts = 100
    attempts = 0

    while len(patches) < n and attempts < max_attempts:
        attempts += 1

        # Random top-left corner
        x1 = random.randint(0, img_w - win_w)
        y1 = random.randint(0, img_h - win_h)
        x2 = x1 + win_w
        y2 = y1 + win_h
        candidate: BBox = (x1, y1, x2, y2)

        # Reject if it overlaps any positive box
        overlaps = any(iou(candidate, gt) >= 0.3 for gt in bboxes)
        if overlaps:
            continue

        crop = image[y1:y2, x1:x2]
        patch = cv2.resize(crop, window_size, interpolation=cv2.INTER_LINEAR)
        patches.append(patch)

    return patches


# ---------------------------------------------------------------------------
# 4. Single-directory dataset loader
# ---------------------------------------------------------------------------

def _image_extensions() -> Tuple[str, ...]:
    return (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def load_dataset(
    data_dir: str | Path,
    max_images: int = 3000,
    neg_ratio: int = 1,   # 1:1 balance — was 3, caused positive class to be swamped
    window_size: Tuple[int, int] = (64, 64),
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load positive and negative patches from a single split directory.

    Directory layout expected::

        data_dir/
            images/   ← .jpg / .jpeg / .png files
            labels/   ← matching .txt annotation files

    The label file for ``images/foo.jpg`` must be ``labels/foo.txt``.

    Label scheme (3-class):
      - y=0 → background (negative crops)
      - y=1 → Bird  (YOLO class_id == 0)
      - y=2 → Drone (YOLO class_id == 1)

    Drone class imbalance is handled by SMOTE in model.py — raw patches are
    appended once each.

    Parameters
    ----------
    data_dir        : root of the split (e.g. ``Data/train``).
    max_images      : upper bound on images to process (shuffled first).
    neg_ratio       : negatives extracted per positive patch.
    window_size     : (W, H) to which every patch is resized.
    progress_callback : called as ``callback(percent: int, msg: str)`` at
                       regular intervals.  May be None.

    Returns
    -------
    X : uint8 ndarray of shape (N, H, W, 3)
    y : int   ndarray of shape (N,)  — 0=background, 1=bird, 2=drone
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Collect image paths
    all_images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in _image_extensions()
    )

    if not all_images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Shuffle and truncate
    random.shuffle(all_images)
    all_images = all_images[:max_images]

    bird_patches: List[np.ndarray] = []
    drone_patches: List[np.ndarray] = []
    negatives: List[np.ndarray] = []
    total = len(all_images)

    for idx, img_path in enumerate(all_images):
        # Progress reporting
        if progress_callback is not None and (idx % 50 == 0 or idx == total - 1):
            pct = int((idx + 1) / total * 100)
            progress_callback(pct, f"Loading {idx + 1}/{total}: {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # Find matching label file
        label_path = labels_dir / (img_path.stem + ".txt")
        annotated = parse_yolo_label(label_path, img_w, img_h)

        # Extract just the BBox list for negative sampling (IoU checks)
        bboxes: List[BBox] = [box for box, _ in annotated]

        # Positive patches — crop each annotated object, assign per-class label
        for (x1, y1, x2, y2), class_id in annotated:
            # Guard against zero-size crops after int conversion
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            patch = cv2.resize(crop, window_size, interpolation=cv2.INTER_LINEAR)

            if class_id == 0:
                # DRONE (class 0 in actual label files — DT* images)
                # BUG FIX (FIX_PLAN 2026-05-25): was incorrectly mapped to Bird (y=1)
                drone_patches.append(patch)
            elif class_id == 1:
                # BIRD (class 1 in actual label files — BT* images)
                # BUG FIX (FIX_PLAN 2026-05-25): was incorrectly mapped to Drone (y=2)
                bird_patches.append(patch)
            # Unknown class_id: skip

        # Negative patches
        n_neg = max(1, len(bboxes)) * neg_ratio if bboxes else neg_ratio
        neg_patches = get_negative_patches(img_rgb, bboxes, n_neg, window_size)
        negatives.extend(neg_patches)

    if not bird_patches and not drone_patches and not negatives:
        raise RuntimeError(f"No patches could be extracted from {data_dir}")

    win_h, win_w = window_size[1], window_size[0]
    empty = np.empty((0, win_h, win_w, 3), dtype=np.uint8)

    bird_arr  = np.array(bird_patches,  dtype=np.uint8) if bird_patches  else empty
    drone_arr = np.array(drone_patches, dtype=np.uint8) if drone_patches else empty
    neg_arr   = np.array(negatives,     dtype=np.uint8) if negatives     else empty

    X = np.concatenate([bird_arr, drone_arr, neg_arr], axis=0)
    y = np.concatenate(
        [
            np.ones(len(bird_arr),  dtype=int),       # y=1 → Bird
            np.full(len(drone_arr), 2, dtype=int),     # y=2 → Drone
            np.zeros(len(neg_arr),  dtype=int),        # y=0 → Background
        ],
        axis=0,
    )

    # Shuffle combined dataset
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# 5. Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir_for_split(data_root: Path, split: str) -> Path:
    return data_root / ".cache" / split


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / "meta.json"


_CACHE_VERSION = 4


def _cache_valid(cache_dir: Path, max_images: int) -> bool:
    """Return True only when cached files exist, version==3, and max_images matches."""
    meta = _meta_path(cache_dir)
    if not meta.exists():
        return False
    if not (cache_dir / "X.npy").exists():
        return False
    if not (cache_dir / "y.npy").exists():
        return False
    try:
        with meta.open("r") as fh:
            stored = json.load(fh)
        return (
            stored.get("version") == _CACHE_VERSION
            and stored.get("max_images") == max_images
        )
    except (json.JSONDecodeError, KeyError):
        return False


def _save_cache(
    cache_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    max_images: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_dir / "X.npy"), X)
    np.save(str(cache_dir / "y.npy"), y)
    with _meta_path(cache_dir).open("w") as fh:
        json.dump(
            {
                "version": _CACHE_VERSION,
                "max_images": max_images,
            },
            fh,
        )


def _load_cache(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(str(cache_dir / "X.npy"))
    y = np.load(str(cache_dir / "y.npy"))
    return X, y


# ---------------------------------------------------------------------------
# 6. High-level train/test loader with caching
# ---------------------------------------------------------------------------

def load_train_test(
    data_root: str | Path = "Data",
    max_train: int = 3000,
    max_test: int = 500,
    window_size: Tuple[int, int] = (64, 64),
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train, validation, and test patch arrays, with on-disk numpy caching.

    Cache is stored under::

        data_root/.cache/train/   (X.npy, y.npy, meta.json)
        data_root/.cache/test/

    The cache is reused when version and max_images match the stored meta
    values.  Changing either invalidates the cache.

    Test split is taken from ``data_root/test/``; if that is missing, the
    loader falls back to ``data_root/valid/``.

    An 80/20 stratified split is applied to the raw training data to produce
    a held-out validation set before returning.

    Parameters
    ----------
    data_root       : top-level data directory (contains train/, test/, valid/).
    max_train       : max images used from the training split.
    max_test        : max images used from the test/valid split.
    window_size     : patch size passed to load_dataset.
    progress_callback : ``callback(percent: int, msg: str)``; may be None.

    Returns
    -------
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test
      — uint8 / int ndarrays.
    Labels: 0=background, 1=bird, 2=drone.
    X_train_raw and X_val_raw are the 80 % / 20 % stratified split of the
    raw training patches (before any SMOTE augmentation).
    """
    data_root = Path(data_root)

    # ------------------------------------------------------------------
    # Training split
    # ------------------------------------------------------------------
    train_cache = _cache_dir_for_split(data_root, "train")

    if _cache_valid(train_cache, max_train):
        if progress_callback:
            progress_callback(0, "Loading train set from cache…")
        X_train_raw, y_train = _load_cache(train_cache)
        if progress_callback:
            progress_callback(50, f"Train cache loaded — {len(X_train_raw)} patches")
    else:
        if progress_callback:
            progress_callback(0, "Building train set from images…")

        def train_cb(pct: int, msg: str) -> None:
            if progress_callback:
                # Map train progress to 0–50 %
                progress_callback(pct // 2, msg)

        X_train_raw, y_train = load_dataset(
            data_dir=data_root / "train",
            max_images=max_train,
            window_size=window_size,
            progress_callback=train_cb,
        )
        _save_cache(train_cache, X_train_raw, y_train, max_train)
        if progress_callback:
            progress_callback(50, f"Train set built — {len(X_train_raw)} patches, cache saved")

    # The provided dataset stores Bird samples in train and Drone samples in
    # test/valid. If the train split is missing Drone entirely, supplement it
    # so the classifier can actually learn class 2.
    train_labels = set(np.unique(y_train).tolist())
    if 2 not in train_labels:
        supplemental_x: list[np.ndarray] = []
        supplemental_y: list[np.ndarray] = []

        for split_name in ["valid", "test"]:
            split_dir = data_root / split_name
            if not split_dir.exists() or not (split_dir / "images").exists():
                continue

            if progress_callback:
                progress_callback(52, f"Supplementing drone samples from {split_name}…")

            X_extra, y_extra = load_dataset(
                data_dir=split_dir,
                max_images=max_test,
                window_size=window_size,
                progress_callback=None,
            )

            drone_mask = y_extra == 2
            if np.any(drone_mask):
                supplemental_x.append(X_extra[drone_mask])
                supplemental_y.append(y_extra[drone_mask])

        if supplemental_x:
            X_train_raw = np.concatenate([X_train_raw, *supplemental_x], axis=0)
            y_train = np.concatenate([y_train, *supplemental_y], axis=0)
            if progress_callback:
                progress_callback(55, f"Added {sum(len(arr) for arr in supplemental_y)} drone patches to training")
        elif progress_callback:
            progress_callback(55, "No drone samples found outside train split")

    # 80/20 stratified split into train and validation
    counts = np.bincount(y_train.astype(int), minlength=3)
    stratify_y = y_train if np.count_nonzero(counts) > 1 and np.all(counts[counts > 0] >= 2) else None
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42, stratify=stratify_y
    )

    # ------------------------------------------------------------------
    # Test / validation split
    # ------------------------------------------------------------------
    test_split_dir = data_root / "test"
    if not test_split_dir.exists() or not (test_split_dir / "images").exists():
        test_split_dir = data_root / "valid"

    if not test_split_dir.exists():
        raise FileNotFoundError(
            f"Neither {data_root / 'test'} nor {data_root / 'valid'} exist."
        )

    split_name = test_split_dir.name          # "test" or "valid"
    test_cache = _cache_dir_for_split(data_root, split_name)

    if _cache_valid(test_cache, max_test):
        if progress_callback:
            progress_callback(50, "Loading test set from cache…")
        X_test_raw, y_test = _load_cache(test_cache)
        if progress_callback:
            progress_callback(100, f"Test cache loaded — {len(X_test_raw)} patches")
    else:
        if progress_callback:
            progress_callback(50, f"Building {split_name} set from images…")

        def test_cb(pct: int, msg: str) -> None:
            if progress_callback:
                # Map test progress to 50–100 %
                progress_callback(50 + pct // 2, msg)

        X_test_raw, y_test = load_dataset(
            data_dir=test_split_dir,
            max_images=max_test,
            window_size=window_size,
            progress_callback=test_cb,
        )
        _save_cache(test_cache, X_test_raw, y_test, max_test)
        if progress_callback:
            progress_callback(100, f"{split_name.capitalize()} set built — {len(X_test_raw)} patches, cache saved")

    return X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test


# ---------------------------------------------------------------------------
# 7. Dataset statistics (label-only, no image loading)
# ---------------------------------------------------------------------------

def dataset_stats(data_root: str | Path = "Data") -> dict:
    """
    Count annotations per class across all splits without loading images.

    Returns dict like::

        {'train': {'Bird': 1200, 'Drone': 80}, 'test': {'Bird': 300, 'Drone': 20}}

    Fast — reads label files only.

    Parameters
    ----------
    data_root : top-level data directory (contains train/, test/, valid/).

    Returns
    -------
    dict mapping split name → dict of class name → annotation count.
    """
    # BUG FIX (FIX_PLAN 2026-05-25): class 0=Drone, 1=Bird (verified from actual labels)
    _CLASS_NAMES = {0: "Drone", 1: "Bird"}
    result = {}
    for split in ["train", "test", "valid"]:
        lab_dir = Path(data_root) / split / "labels"
        if not lab_dir.exists():
            continue
        counts: dict[str, int] = {}
        for f in lab_dir.iterdir():
            if f.suffix != ".txt":
                continue
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                tokens = line.strip().split()
                if len(tokens) >= 5:
                    try:
                        cls = int(tokens[0])
                        name = _CLASS_NAMES.get(cls, f"Class{cls}")
                        counts[name] = counts.get(name, 0) + 1
                    except ValueError:
                        pass
        if counts:
            result[split] = counts
    return result
