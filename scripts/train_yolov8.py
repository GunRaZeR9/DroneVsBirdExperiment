"""
train_yolov8.py
===============
Train a YOLOv8 model on the drone/bird dataset.

Prerequisites:
    pip install ultralytics

Usage:
    # Quick start (CPU-friendly defaults):
    python scripts/train_yolov8.py

    # Custom run:
    python scripts/train_yolov8.py ^
        --model yolov8s.pt ^
        --data  datasets/data.yaml ^
        --epochs 50 ^
        --batch 16 ^
        --device 0

    # Use merged dataset:
    python scripts/train_yolov8.py --merged

    # GPU run with larger model:
    python scripts/train_yolov8.py --model yolov8m.pt --device 0 --batch 16 --epochs 50

Notes:
    - YOLOv8 uses standard NMS-based detection.
    - workers=0 is hardcoded — required on Windows to avoid multiprocessing deadlocks.
    - Checkpoints saved every 5 epochs (save_period=5).
    - Results auto-saved to runs/train/drone_v8/
    - Tensorboard: tensorboard --logdir runs/train/drone_v8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 for drone/bird detection")
    p.add_argument(
        "--model",
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help=(
            "Pretrained YOLOv8 checkpoint. "
            "yolov8n is fastest (nano) — good for CPU. "
            "yolov8s for better accuracy. "
            "yolov8m for best accuracy/speed balance."
        ),
    )
    p.add_argument(
        "--data",
        default=str(ROOT / "datasets" / "data.yaml"),
        help="Path to data.yaml. Defaults to datasets/data.yaml.",
    )
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--imgsz",       type=int,   default=416)
    p.add_argument("--batch",       type=int,   default=8)
    p.add_argument("--device",      default="cpu", help="CUDA device id or 'cpu'")
    p.add_argument(
        "--project",
        default=str(ROOT / "runs" / "train"),
        help="Directory where training runs are saved",
    )
    p.add_argument("--name",        default="drone_v8")
    p.add_argument(
        "--hyp",
        default=str(ROOT / "hyp.drone.yaml"),
        help="Custom hyperparameter YAML (hyp.drone.yaml by default)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted training from last checkpoint",
    )
    p.add_argument(
        "--merged",
        action="store_true",
        help=(
            "Use datasets/merged/data.yaml (after running merge_datasets.py) "
            "instead of the default Data/ yaml"
        ),
    )
    return p.parse_args()


def check_prerequisites() -> None:
    """Verify ultralytics is importable."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print(
            "ERROR: ultralytics not installed.\n"
            "Run: pip install ultralytics\n"
            "     (or pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 first for CUDA)",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    args = parse_args()
    check_prerequisites()

    from ultralytics import YOLO

    # Override data path if --merged flag set
    data_path = args.data
    if args.merged:
        data_path = str(ROOT / "datasets" / "merged" / "data.yaml")
        print(f"Using merged dataset: {data_path}")

    # Verify data.yaml exists
    if not Path(data_path).exists():
        print(f"ERROR: data.yaml not found at {data_path}", file=sys.stderr)
        print("  Run scripts/merge_datasets.py first, or check --data path.", file=sys.stderr)
        sys.exit(1)

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"

    print("=" * 60)
    print("  YOLOv8 Training — Drone/Bird Detection")
    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Data       : {data_path}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch      : {args.batch}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Device     : {args.device}")
    print(f"  Workers    : 0 (hardcoded — Windows multiprocessing safety)")
    print(f"  Save every : 5 epochs")
    print(f"  Output     : {args.project}/{args.name}")
    print(f"  Best weights will be at: {best_weights}")
    print()
    print("  To run inference after training:")
    print(f"    python scripts/run_inference.py --weights {best_weights}")
    print("=" * 60)
    print()

    # Load model — when resuming, last.pt must be the model (not the base weights)
    if args.resume:
        candidate_paths = [
            Path(args.project) / args.name / "weights" / "last.pt",
        ]
        last_pt = next((p for p in candidate_paths if p.exists()), None)
        if last_pt is None:
            print(
                f"ERROR: --resume set but last.pt not found in any of:\n"
                + "\n".join(f"  {p}" for p in candidate_paths),
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        model = YOLO(args.model)

    # Build kwargs dict — only pass hyp if the file exists
    train_kwargs: dict = dict(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        task="detect",          # explicit — prevents ultralytics prepending runs/detect/
        project=args.project,
        name=args.name,
        exist_ok=True,          # reuse runs/train/drone_v8 — keep weights path stable
        workers=0,              # hardcoded: Windows multiprocessing safety
        save_period=5,          # checkpoint every 5 epochs
        cos_lr=True,
        multi_scale=False,      # OFF: prevents 1x1 feature maps that crash BatchNorm
                                #      on the size-1 last batch (16761 % 8 == 1)
        # Augmentation / regularisation tweaks
        label_smoothing=0.1,   # prevents Bird 100% overconfidence
        copy_paste=0.1,        # copies drone crops into bird-heavy images
        mixup=0.15,            # class boundary learning
        scale=0.9,             # wide scale range — drones at many distances
        cls=0.3,               # lower cls weight → more careful class learning
        resume=args.resume,
        verbose=True,
    )

    if Path(args.hyp).exists():
        train_kwargs["cfg"] = args.hyp
        print(f"Loaded hyperparameters from {args.hyp}")
    else:
        print(f"hyp.drone.yaml not found at {args.hyp} — using inline kwargs")

    # Train
    results = model.train(**train_kwargs)

    # Summary
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best weights: {best_weights}")
    print()
    print("  NEXT STEPS")
    print("  1. Validate:  python scripts/validate_yolov10.py --weights", best_weights)
    print("  2. Inference: python scripts/run_inference.py --weights", best_weights)
    print()
    print(f"  Tensorboard: tensorboard --logdir runs/train/{args.name}")
    print()
    print("  KEY METRICS TO CHECK (results.csv):")
    print("    metrics/mAP50        target > 0.80")
    print("    metrics/mAP50-95     target > 0.55")
    print("    val/cls_loss         should decrease steadily")
    print("    Per-class AP drone   target > 0.75")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
