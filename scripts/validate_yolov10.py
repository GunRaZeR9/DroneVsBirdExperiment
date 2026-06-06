"""
validate_yolov10.py — Phase 4 of FIX_PLAN
==========================================
Full validation of a trained YOLOv10 model including:
  - mAP50 / mAP50-95 (overall and per-class)
  - Confusion matrix (confusion_matrix_normalized.png)
  - PR curve, F1 curve
  - Edge-case testing

Usage:
    # Standard validation on test split:
    python scripts/validate_yolov10.py

    # Validate on val split:
    python scripts/validate_yolov10.py --split val

    # With custom weights + data yaml:
    python scripts/validate_yolov10.py ^
        --weights runs/train/drone_v10/weights/best.pt ^
        --data    datasets/data.yaml ^
        --split   test

    # Edge-case batch test:
    python scripts/validate_yolov10.py --edge-cases test_edge_cases/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate YOLOv10 drone/bird model")
    p.add_argument(
        "--weights",
        default=str(ROOT / "runs" / "train" / "drone_v10" / "weights" / "best.pt"),
    )
    p.add_argument(
        "--data",
        default=str(ROOT / "datasets" / "data.yaml"),
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to validate on",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for validation (low = measure full PR curve)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for mAP matching",
    )
    p.add_argument(
        "--device",
        default="0",
    )
    p.add_argument(
        "--edge-cases",
        default="",
        help="Directory of edge-case images to test separately",
    )
    p.add_argument(
        "--project",
        default=str(ROOT / "runs" / "val"),
    )
    p.add_argument("--name", default="drone_v10_val")
    return p.parse_args()


def print_metrics(metrics) -> None:
    """Print formatted validation metrics."""
    print()
    print("─" * 55)
    print("  VALIDATION RESULTS")
    print("─" * 55)

    try:
        map50    = metrics.box.map50
        map5095  = metrics.box.map
        print(f"  mAP@50     : {map50:.4f}  (target > 0.80)")
        print(f"  mAP@50-95  : {map5095:.4f}  (target > 0.55)")
    except AttributeError:
        print("  (metrics not fully available — check results manually)")

    try:
        # Per-class AP
        names = metrics.names  # {0: 'drone', 1: 'bird'}
        ap_per_class = metrics.box.ap_class_index
        aps = metrics.box.ap50
        print()
        print("  Per-class AP@50:")
        for idx, class_idx in enumerate(ap_per_class):
            name = names.get(int(class_idx), f"class{class_idx}")
            target = "target > 0.75" if name == "drone" else ""
            print(f"    {name:10s}: {aps[idx]:.4f}  {target}")
    except (AttributeError, IndexError):
        pass

    print("─" * 55)


def check_confusion_matrix(save_dir: Path) -> None:
    """Print guidance on reading the confusion matrix."""
    cm_file = save_dir / "confusion_matrix_normalized.png"
    cm_raw  = save_dir / "confusion_matrix.png"

    print()
    print("  CONFUSION MATRIX")
    if cm_file.exists():
        print(f"  → Saved at: {cm_file}")
    elif cm_raw.exists():
        print(f"  → Saved at: {cm_raw}")
    else:
        print("  → Confusion matrix not found (run with plots=True)")

    print()
    print("  Reading the matrix (rows=actual, cols=predicted):")
    print("  ┌──────────────────────────────────────┐")
    print("  │              Predicted               │")
    print("  │         drone    bird   background   │")
    print("  │ drone  [HIGH]   [LOW]    [LOW]        │")
    print("  │ bird   [LOW]   [HIGH]   [LOW]         │")
    print("  └──────────────────────────────────────┘")
    print()
    print("  If drone→bird cell is still HIGH after retraining:")
    print("  → Dataset still imbalanced. Return to Phase 1.")
    print("  → Check: python scripts/audit_labels.py")


def run_edge_case_test(model, edge_dir: Path, conf: float) -> None:
    """Run batch prediction on edge-case images and print per-file results."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    edge_images = [p for p in edge_dir.iterdir() if p.suffix.lower() in image_exts]

    if not edge_images:
        print(f"  No images found in {edge_dir}")
        return

    print()
    print("─" * 55)
    print(f"  EDGE CASE TEST  ({len(edge_images)} images from {edge_dir.name})")
    print("─" * 55)

    results = model.predict(
        source=[str(p) for p in edge_images],
        conf=conf,
        verbose=False,
        save=True,
        save_txt=True,
    )

    for img_path, result in zip(edge_images, results):
        n_boxes  = len(result.boxes) if result.boxes is not None else 0
        labels   = []
        if result.boxes is not None:
            names = result.names
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i])
                c      = float(result.boxes.conf[i])
                labels.append(f"{names.get(cls_id, str(cls_id))}({c:.2f})")

        status = "✓" if n_boxes > 0 else "✗"
        label_str = ", ".join(labels) if labels else "(none)"
        print(f"  {status} {img_path.name:40s}  {label_str}")


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        print("  Train first: python scripts/train_yolov10.py", file=sys.stderr)
        sys.exit(1)

    print("=" * 55)
    print("  YOLOv10 Validation — Drone/Bird Detection")
    print("=" * 55)
    print(f"  Weights : {weights}")
    print(f"  Data    : {args.data}")
    print(f"  Split   : {args.split}")
    print(f"  Device  : {args.device}")
    print()

    model = YOLO(str(weights))

    # ------------------------------------------------------------------
    # Full validation run
    # ------------------------------------------------------------------
    metrics = model.val(
        data=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,          # generates confusion matrix, PR curve, F1 curve
        save_json=True,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    print_metrics(metrics)

    save_dir = Path(args.project) / args.name
    check_confusion_matrix(save_dir)

    # ------------------------------------------------------------------
    # Edge case test (optional)
    # ------------------------------------------------------------------
    if args.edge_cases:
        edge_dir = Path(args.edge_cases)
        if edge_dir.exists():
            run_edge_case_test(model, edge_dir, conf=0.45)
        else:
            print(f"\n  ⚠  Edge-case directory not found: {edge_dir}")
            print("     Create it and add test images:")
            print("       mkdir test_edge_cases")
            print("       # Add: drone_sunset.jpg, drone_backlit.jpg, small_drone.jpg,")
            print("       #      bird_and_drone.jpg, multi_drone.jpg, occluded_drone.jpg")

    # ------------------------------------------------------------------
    # Summary checklist
    # ------------------------------------------------------------------
    print()
    print("─" * 55)
    print("  PHASE 4 CHECKLIST")
    print("─" * 55)
    try:
        map50 = metrics.box.map50
        ok_map50 = "✓" if map50 > 0.80 else "✗"
        print(f"  {ok_map50} mAP@50 > 0.80  (got {map50:.3f})")
    except Exception:
        print("  ? mAP@50 > 0.80  (check results.csv manually)")

    cm_path = save_dir / "confusion_matrix_normalized.png"
    print(f"  {'✓' if cm_path.exists() else '✗'} confusion_matrix_normalized.png saved")
    print(f"  ? Drone→Bird cell near 0  (open {cm_path})")
    print(f"  ? 6 edge case scenarios tested")
    print()
    print("  If mAP < 0.60 at epoch 80 → dataset still imbalanced.")
    print("  If drone AP low but bird AP high → add more drone images.")
    print()
    print("  NEXT: python scripts/export_yolov10.py")


if __name__ == "__main__":
    main()
