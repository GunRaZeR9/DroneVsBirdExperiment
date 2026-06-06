"""
run_inference.py — Phase 3 CLI for FIX_PLAN
============================================
Run YOLOv10 inference on images, video, or webcam.

Usage:
    # Single image:
    python scripts/run_inference.py --source path/to/drone.jpg

    # Video file:
    python scripts/run_inference.py --source path/to/video.mp4 --show

    # Webcam:
    python scripts/run_inference.py --source 0 --show

    # Custom weights + lower confidence:
    python scripts/run_inference.py ^
        --weights runs/train/drone_v10/weights/best.pt ^
        --source  path/to/image.jpg ^
        --conf    0.35 ^
        --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv10 drone/bird inference")
    p.add_argument(
        "--weights",
        default=str(ROOT / "runs" / "train" / "drone_v10" / "weights" / "best.pt"),
        help="Path to best.pt weights file",
    )
    p.add_argument(
        "--source",
        required=True,
        help="Image path, video path, or webcam index (0)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence threshold (0.45 recommended per FIX_PLAN)",
    )
    p.add_argument(
        "--device",
        default="0",
        help="Device: '0' (GPU), 'cpu', 'mps'",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to runs/predict/",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display live annotated frames (video/webcam only)",
    )
    p.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results as YOLO .txt files",
    )
    return p.parse_args()


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main() -> None:
    args = parse_args()

    # Try to use Ultralytics CLI-style directly for maximum compatibility
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: weights not found: {weights_path}", file=sys.stderr)
        print("  Train first: python scripts/train_yolov10.py", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(weights_path))

    print(f"Model  : {weights_path}")
    print(f"Source : {args.source}")
    print(f"Conf   : {args.conf}")
    print(f"Device : {args.device}")
    print()

    # For images — use predict() and print results
    source = args.source
    try:
        source_int = int(source)
        source = source_int   # webcam index
    except ValueError:
        pass

    if isinstance(source, str) and is_image(source):
        # Image inference
        results = model.predict(
            source=source,
            conf=args.conf,
            device=args.device,
            save=args.save,
            save_txt=args.save_txt,
            show_labels=True,
            show_conf=True,
            verbose=True,
        )

        print()
        print("─" * 50)
        print("DETECTIONS:")
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                print("  (none above confidence threshold)")
                continue
            names = result.names   # {0: 'drone', 1: 'bird'}
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i])
                conf   = float(result.boxes.conf[i])
                x1, y1, x2, y2 = [int(v) for v in result.boxes.xyxy[i].tolist()]
                label = names.get(cls_id, f"class{cls_id}")
                print(f"  {label:10s}  conf={conf:.3f}  box=[{x1},{y1},{x2},{y2}]  size={x2-x1}x{y2-y1}px")

        if args.save:
            print()
            print("Saved annotated image to runs/predict/")

    else:
        # Video / webcam — use detect_video from the module
        from src.core.yolov10_detector import YOLOv10Detector

        detector = YOLOv10Detector(
            weights=str(weights_path),
            conf=args.conf,
            device=args.device,
        )

        save_path = None
        if args.save and isinstance(source, str):
            save_path = str(ROOT / "runs" / "predict" / Path(source).stem) + "_out.mp4"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving annotated video to: {save_path}")

        detector.detect_video(
            source=source,
            display=args.show,
            save_path=save_path,
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
