"""
metrics.py — Evaluation utilities for Bird vs Drone Detector.

Provides classification metrics (accuracy, F1, precision, recall,
confusion matrix) and detection-level IoU-based metrics (precision,
recall, AP) compatible with YOLO-style bounding-box outputs.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels (0 = background/negative, 1 = object/positive).
    y_pred : np.ndarray
        Predicted integer labels, same shape as ``y_true``.

    Returns
    -------
    dict
        Keys and types:

        * ``accuracy``        – float, fraction of correct predictions.
        * ``f1``              – float, binary F1 for positive class (label=1).
        * ``precision``       – float, binary precision for positive class.
        * ``recall``          – float, binary recall for positive class.
        * ``confusion_matrix``– np.ndarray, shape (2, 2); [[TN, FP], [FN, TP]].
        * ``report``          – str, full sklearn classification_report.
        * ``n_samples``       – int, total number of samples.
        * ``n_positive``      – int, number of predicted positives.
        * ``n_true_positive`` – int, number of ground-truth positives.

    Notes
    -----
    * Works correctly when only one class is present in ``y_true`` (e.g. all
      negatives in a tiny test split).  All sklearn calls use
      ``zero_division=0`` to avoid divide-by-zero warnings/errors.
    * Both arrays are cast to ``int`` internally; passing float probabilities
      will raise ``ValueError``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays.")

    # Determine which labels actually appear.
    present_labels = np.unique(np.concatenate([y_true, y_pred])).tolist()
    n_classes = len(present_labels)

    # Use 'weighted' for multiclass (3 classes: bg/bird/drone),
    # 'binary' only when the labels are exactly {0, 1}.
    # If labels are e.g. [0, 2] (background + drone, no bird in split),
    # binary mode with pos_label=1 would crash — fall back to weighted.
    if set(present_labels) == {0, 1}:
        avg      = "binary"
        pos_kw   = {"pos_label": 1}
    else:
        avg      = "weighted"
        pos_kw   = {}          # pos_label not valid for multiclass/non-standard averages

    acc = float(accuracy_score(y_true, y_pred))

    f1 = float(
        f1_score(y_true, y_pred, average=avg, zero_division=0,
                 labels=present_labels, **pos_kw)
    )

    prec = float(
        precision_score(y_true, y_pred, average=avg, zero_division=0,
                        labels=present_labels, **pos_kw)
    )

    rec = float(
        recall_score(y_true, y_pred, average=avg, zero_division=0,
                     labels=present_labels, **pos_kw)
    )

    # Confusion matrix — always square over all known classes [0..n-1]
    all_labels = list(range(max(present_labels) + 1))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    label_names = {0: "background", 1: "bird", 2: "drone"}
    target_names = [label_names.get(l, str(l)) for l in all_labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=target_names,
        zero_division=0,
    )

    per_class_p, per_class_r, per_class_f, per_class_s = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )

    _class_names = {0: "Background", 1: "Bird", 2: "Drone"}
    per_class = {}
    for i, lbl in enumerate(all_labels):
        name = _class_names.get(lbl, str(lbl))
        per_class[name] = {
            "precision": float(per_class_p[i]),
            "recall":    float(per_class_r[i]),
            "f1":        float(per_class_f[i]),
            "support":   int(per_class_s[i]),
        }

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "report": report,
        "n_samples": int(y_true.size),
        "n_positive": int(np.sum(y_pred >= 1)),      # any non-background prediction
        "n_true_positive": int(np.sum(y_true >= 1)), # any non-background ground truth
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Union for two axis-aligned boxes.

    Parameters
    ----------
    box_a, box_b : array-like of shape (4,)
        Boxes in (x1, y1, x2, y2) format.

    Returns
    -------
    float
        IoU in [0, 1].  Returns 0.0 if both boxes have zero area.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision via 11-point interpolation (VOC style).

    Parameters
    ----------
    recalls, precisions : np.ndarray
        Parallel arrays of recall and precision values, *already sorted by
        ascending recall*.

    Returns
    -------
    float
        AP in [0, 1].
    """
    ap = 0.0
    for thr in np.linspace(0.0, 1.0, 11):
        mask = recalls >= thr
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


# ---------------------------------------------------------------------------
# Detection-level metrics
# ---------------------------------------------------------------------------

def compute_iou_metrics(
    pred_boxes: list,
    gt_boxes: list,
    iou_thresh: float = 0.5,
) -> dict:
    """Compute detection-level precision, recall, AP, and mean IoU.

    Each predicted box is matched to at most one ground-truth box using a
    greedy algorithm sorted by descending confidence.  A prediction is a
    True Positive (TP) if its IoU with an unmatched GT box exceeds
    ``iou_thresh``, otherwise it is a False Positive (FP).  Unmatched GT
    boxes count as False Negatives (FN).

    Parameters
    ----------
    pred_boxes : list of (x1, y1, x2, y2, conf)
        Predicted bounding boxes with confidence score.
    gt_boxes : list of (x1, y1, x2, y2)
        Ground-truth bounding boxes.
    iou_thresh : float, optional
        IoU threshold for a match to count as TP (default 0.5).

    Returns
    -------
    dict
        Keys:

        * ``precision``  – float, TP / (TP + FP). 0 when no predictions.
        * ``recall``     – float, TP / (TP + FN). 0 when no GT boxes.
        * ``ap``         – float, Average Precision (11-point VOC).
        * ``iou_mean``   – float, mean IoU over matched (TP) pairs. 0 if none.
        * ``tp``         – int, true positives.
        * ``fp``         – int, false positives.
        * ``fn``         – int, false negatives.
        * ``n_pred``     – int, total predictions.
        * ``n_gt``       – int, total ground-truth boxes.

    Notes
    -----
    * When ``pred_boxes`` or ``gt_boxes`` is empty the function returns
      gracefully with all numeric values set to 0.
    * Boxes should be in pixel (x1, y1, x2, y2) format; the function does
      *not* require x2 > x1 or y2 > y1 but degenerate boxes will have 0
      area and can only match another 0-area box at exactly the same
      location.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    # --- trivial edge cases -------------------------------------------
    if n_pred == 0 and n_gt == 0:
        return {
            "precision": 0.0, "recall": 0.0, "ap": 0.0,
            "iou_mean": 0.0, "tp": 0, "fp": 0, "fn": 0,
            "n_pred": 0, "n_gt": 0,
        }
    if n_pred == 0:
        return {
            "precision": 0.0, "recall": 0.0, "ap": 0.0,
            "iou_mean": 0.0, "tp": 0, "fp": 0, "fn": n_gt,
            "n_pred": 0, "n_gt": n_gt,
        }
    if n_gt == 0:
        return {
            "precision": 0.0, "recall": 0.0, "ap": 0.0,
            "iou_mean": 0.0, "tp": 0, "fp": n_pred, "fn": 0,
            "n_pred": n_pred, "n_gt": 0,
        }

    # Convert to numpy for vectorised IoU later.
    preds = np.array(pred_boxes, dtype=float)          # (N, 5)
    gts   = np.array(gt_boxes,   dtype=float)          # (M, 4)

    # Sort predictions by confidence descending.
    order     = np.argsort(-preds[:, 4])
    preds_sorted = preds[order]

    gt_matched = np.zeros(n_gt, dtype=bool)
    tp_flags   = np.zeros(n_pred, dtype=bool)
    iou_vals   = []

    for i, pred in enumerate(preds_sorted):
        best_iou  = 0.0
        best_j    = -1
        for j, gt in enumerate(gts):
            if gt_matched[j]:
                continue
            iou_val = _iou(pred[:4], gt[:4])
            if iou_val > best_iou:
                best_iou = iou_val
                best_j   = j

        if best_iou >= iou_thresh and best_j >= 0:
            tp_flags[i]         = True
            gt_matched[best_j]  = True
            iou_vals.append(best_iou)

    tp = int(tp_flags.sum())
    fp = n_pred - tp
    fn = n_gt   - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou_mean  = float(np.mean(iou_vals)) if iou_vals else 0.0

    # Build PR curve for AP calculation (cumulative over confidence-sorted preds).
    cum_tp  = np.cumsum(tp_flags.astype(int))
    cum_fp  = np.cumsum((~tp_flags).astype(int))
    pr_prec = cum_tp / (cum_tp + cum_fp + 1e-9)
    pr_rec  = cum_tp / (n_gt + 1e-9)

    # Prepend (0, 1) sentinel so AP is well-defined at recall=0.
    pr_rec  = np.concatenate([[0.0], pr_rec])
    pr_prec = np.concatenate([[1.0], pr_prec])

    ap = _compute_ap(pr_rec, pr_prec)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "ap":        ap,
        "iou_mean":  iou_mean,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "n_pred":    n_pred,
        "n_gt":      n_gt,
    }
