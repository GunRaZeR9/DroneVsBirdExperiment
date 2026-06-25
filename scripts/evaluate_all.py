"""
evaluate_all.py — Evaluate all trained models on the test set.

Models evaluated:
  - HOG+MLP  (models/hog_mlp.pkl)
  - HOG+SVM  (models/hog_svm.pkl)
  - YOLOv8n  (runs/train/drone_v8/weights/best.pt)
  - YOLOv10  (runs/detect/runs/train/drone_v10_cpu/weights/best.pt
              or runs/train/drone_v10_cpu/weights/best.pt)

Results saved to results/all_metrics.json.

Usage:
    python scripts/evaluate_all.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must come before any local imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Standard imports (after path setup)
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_record() -> dict:
    return {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "map50": None,
        "train_time": None,
    }


def _safe_float(value) -> float | None:
    """Convert a value to Python float, returning None for NaN/inf/None."""
    try:
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):  # NaN or inf
            return None
        return v
    except (TypeError, ValueError):
        return None


def _f1_from_pr(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom <= 0.0:
        return None
    return 2.0 * precision * recall / denom


def _print_table(all_metrics: dict) -> None:
    """Print a formatted summary table to stdout."""
    header = f"{'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mAP@50':>10}"
    separator = "-" * len(header)
    print()
    print("=" * len(header))
    print("  EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print(separator)
    for model_name, m in all_metrics.items():
        def fmt(v):
            return f"{v:.4f}" if v is not None else "   N/A  "
        print(
            f"{model_name:<14}"
            f" {fmt(m['accuracy']):>10}"
            f" {fmt(m['precision']):>10}"
            f" {fmt(m['recall']):>10}"
            f" {fmt(m['f1']):>10}"
            f" {fmt(m['map50']):>10}"
        )
    print(separator)
    print()


# ---------------------------------------------------------------------------
# HOG model evaluation (shared by MLP and SVM)
# ---------------------------------------------------------------------------

def _evaluate_hog_model(weights_path: Path, label: str) -> dict:
    """Load a HOG+sklearn model and evaluate on the test patch set."""
    from src.core.data_loader import load_train_test
    from src.core.feature_extractor import extract_hog_batch
    from src.core.model import BirdDroneModel
    from src.utils.metrics import compute_metrics

    print(f"  Loading {label} from {weights_path} ...")
    model = BirdDroneModel()
    model.load(str(weights_path))

    print("  Loading test patches ...")
    _, _, _, _, X_test_raw, y_test = load_train_test(
        data_root=str(ROOT / "Data"),
        max_train=100,
        max_test=500,
        window_size=(64, 64),
    )

    print(f"  Extracting HOG features for {len(X_test_raw)} test patches ...")
    X_test_hog = extract_hog_batch(X_test_raw)

    print("  Running predictions ...")
    y_pred = model.predict(X_test_hog)

    metrics = compute_metrics(y_test, y_pred)
    print(
        f"  {label}: acc={metrics['accuracy']:.4f}  "
        f"f1={metrics['f1']:.4f}  "
        f"prec={metrics['precision']:.4f}  "
        f"rec={metrics['recall']:.4f}"
    )

    return {
        "accuracy": _safe_float(metrics["accuracy"]),
        "precision": _safe_float(metrics["precision"]),
        "recall": _safe_float(metrics["recall"]),
        "f1": _safe_float(metrics["f1"]),
        "map50": None,
        "train_time": None,
    }


# ---------------------------------------------------------------------------
# YOLO model evaluation (shared by v8 and v10)
# ---------------------------------------------------------------------------

def _evaluate_yolo_model(weights_path: Path, label: str) -> dict:
    """Load a YOLO model and evaluate via model.val()."""
    from ultralytics import YOLO

    data_yaml = str(ROOT / "datasets" / "merged" / "data.yaml")

    print(f"  Loading {label} from {weights_path} ...")
    model = YOLO(str(weights_path))

    print("  Running validation on test split ...")
    results = model.val(
        data=data_yaml,
        split="test",
        device="cpu",
        verbose=False,
    )

    precision = _safe_float(results.box.mp)
    recall = _safe_float(results.box.mr)
    map50 = _safe_float(results.box.map50)
    f1 = _f1_from_pr(precision, recall)

    print(
        f"  {label}: mAP@50={map50}  "
        f"prec={precision}  "
        f"rec={recall}  "
        f"f1={f1}"
    )

    return {
        "accuracy": None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "train_time": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_metrics: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # HOG+MLP
    # -----------------------------------------------------------------------
    mlp_path = ROOT / "models" / "hog_mlp.pkl"
    print(f"\n[1/4] HOG+MLP — {'found' if mlp_path.exists() else 'NOT FOUND'}")
    if mlp_path.exists():
        try:
            all_metrics["HOG+MLP"] = _evaluate_hog_model(mlp_path, "HOG+MLP")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_metrics["HOG+MLP"] = _null_record()
    else:
        print(f"  Skipping: {mlp_path} does not exist.")
        all_metrics["HOG+MLP"] = _null_record()

    # -----------------------------------------------------------------------
    # HOG+SVM
    # -----------------------------------------------------------------------
    svm_path = ROOT / "models" / "hog_svm.pkl"
    print(f"\n[2/4] HOG+SVM — {'found' if svm_path.exists() else 'NOT FOUND'}")
    if svm_path.exists():
        try:
            all_metrics["HOG+SVM"] = _evaluate_hog_model(svm_path, "HOG+SVM")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_metrics["HOG+SVM"] = _null_record()
    else:
        print(f"  Skipping: {svm_path} does not exist.")
        all_metrics["HOG+SVM"] = _null_record()

    # -----------------------------------------------------------------------
    # YOLOv8n
    # -----------------------------------------------------------------------
    v8_path = ROOT / "runs" / "train" / "drone_v8" / "weights" / "best.pt"
    print(f"\n[3/4] YOLOv8n — {'found' if v8_path.exists() else 'NOT FOUND'}")
    if v8_path.exists():
        try:
            all_metrics["YOLOv8n"] = _evaluate_yolo_model(v8_path, "YOLOv8n")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_metrics["YOLOv8n"] = _null_record()
    else:
        print(f"  Skipping: {v8_path} does not exist.")
        all_metrics["YOLOv8n"] = _null_record()

    # -----------------------------------------------------------------------
    # YOLOv10
    # -----------------------------------------------------------------------
    # Clean path is primary; nested path (legacy task="detect" bug) is fallback.
    v10_path_primary = ROOT / "runs" / "train" / "drone_v10_cpu" / "weights" / "best.pt"
    v10_path_alt = (
        ROOT / "runs" / "detect" / "runs" / "train" / "drone_v10_cpu" / "weights" / "best.pt"
    )

    if v10_path_primary.exists():
        v10_path = v10_path_primary
    elif v10_path_alt.exists():
        v10_path = v10_path_alt
    else:
        v10_path = None

    print(f"\n[4/4] YOLOv10 — {'found at ' + str(v10_path) if v10_path else 'NOT FOUND'}")
    if v10_path is not None:
        try:
            all_metrics["YOLOv10"] = _evaluate_yolo_model(v10_path, "YOLOv10")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_metrics["YOLOv10"] = _null_record()
    else:
        print(
            f"  Skipping: neither\n"
            f"    {v10_path_primary}\n"
            f"    {v10_path_alt}\n"
            f"  exists."
        )
        all_metrics["YOLOv10"] = _null_record()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "all_metrics.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)

    print(f"\nMetrics saved to {output_path}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    _print_table(all_metrics)


if __name__ == "__main__":
    main()
