"""
export_yolov10.py — Phase 5 of FIX_PLAN
=========================================
Export a trained YOLOv10 model to deployment formats.

Supported formats:
  - onnx     : Universal CPU/GPU (best portability)
  - engine   : TensorRT (NVIDIA, fastest inference, requires TensorRT install)
  - openvino : Intel CPU optimised
  - coreml   : Apple Silicon / iPhone

Usage:
    # Export to ONNX (recommended default):
    python scripts/export_yolov10.py

    # Export to TensorRT FP16 (fastest on NVIDIA GPU):
    python scripts/export_yolov10.py --format engine --half

    # Export to multiple formats at once:
    python scripts/export_yolov10.py --format onnx engine openvino

    # Custom weights:
    python scripts/export_yolov10.py --weights runs/train/my_run/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLOv10 model for deployment")
    p.add_argument(
        "--weights",
        default=str(ROOT / "runs" / "train" / "drone_v10" / "weights" / "best.pt"),
        help="Path to best.pt weights",
    )
    p.add_argument(
        "--format",
        nargs="+",
        default=["onnx"],
        choices=["onnx", "engine", "openvino", "coreml", "tflite", "torchscript"],
        help="Export format(s). Multiple formats can be specified.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for export",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 (half precision). Supported for: engine, onnx.",
    )
    p.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify ONNX graph (default True)",
    )
    p.add_argument(
        "--device",
        default="0",
        help="Device for TensorRT export (must be GPU). Use 'cpu' for ONNX/OpenVINO.",
    )
    return p.parse_args()


FORMAT_NOTES = {
    "onnx": (
        "ONNX — universal format. Works on CPU and GPU via onnxruntime.\n"
        "  Use for: deployment on any hardware, server inference, ONNX runtime integration."
    ),
    "engine": (
        "TensorRT — NVIDIA GPU only. Fastest possible inference speed.\n"
        "  Requires: TensorRT SDK installed, NVIDIA GPU.\n"
        "  Use for: real-time drone detection on Jetson or server with NVIDIA GPU."
    ),
    "openvino": (
        "OpenVINO — Intel CPU optimised. Good for edge devices with Intel CPU.\n"
        "  Use for: deployment on Intel NUC, laptop CPU, or Raspberry Pi with Intel NCS."
    ),
    "coreml": (
        "CoreML — Apple Silicon (M1/M2/M3) and iPhone/iPad.\n"
        "  Use for: iOS app integration, macOS inference."
    ),
}


def export_format(model, fmt: str, imgsz: int, half: bool, simplify: bool, device: str) -> Path | None:
    """Export to one format and return the output path."""
    kwargs: dict = dict(format=fmt, imgsz=imgsz)

    if fmt == "onnx":
        kwargs["simplify"] = simplify
        kwargs["half"] = half
    elif fmt == "engine":
        kwargs["half"] = half
        kwargs["device"] = device
    elif fmt == "openvino":
        kwargs["half"] = half

    print(f"\n  Exporting to {fmt.upper()}…")
    try:
        out_path = model.export(**kwargs)
        return Path(out_path)
    except Exception as exc:
        print(f"  ✗ Export failed: {exc}", file=sys.stderr)
        return None


def print_deployment_snippet(fmt: str, out_path: Path) -> None:
    """Print the inference code snippet for the exported model."""
    print()
    print(f"  Deployment snippet ({fmt.upper()}):")
    if fmt == "onnx":
        print(f"""
    from ultralytics import YOLO

    model = YOLO(r"{out_path}")
    results = model.predict(source="drone.jpg", conf=0.45)
    # Or use onnxruntime directly for zero-Ultralytics-dependency deployment:
    #   import onnxruntime as ort
    #   session = ort.InferenceSession(r"{out_path}")
""")
    elif fmt == "engine":
        print(f"""
    from ultralytics import YOLO

    model = YOLO(r"{out_path}")
    results = model.predict(source="drone.jpg", conf=0.45, device=0)
""")
    else:
        print(f"""
    from ultralytics import YOLO
    model = YOLO(r"{out_path}")
    results = model.predict(source="drone.jpg", conf=0.45)
""")


def update_repo_inference(weights_path: Path) -> None:
    """Print instructions for swapping weights in the repo's inference code."""
    print()
    print("─" * 55)
    print("  HOW TO USE IN YOUR REPO")
    print("─" * 55)
    print("""
  Replace the old HOG+MLP detector with YOLOv10:

  # OLD (src/core/detector.py — HOG sliding window):
  from src.core.detector import detect
  results = detect(image, model, confidence_threshold=0.5)

  # NEW (src/core/yolov10_detector.py — drop-in compatible):
  from src.core.yolov10_detector import YOLOv10Detector, detect
  detector = YOLOv10Detector(weights=r"{weights}")
  results = detect(image, detector, confidence_threshold=0.45)

  # Or direct API:
  detections = detector.detect(image)
  for x, y, w, h, conf, label in detections:
      print(f"{{label}} @ ({{x}},{{y}}) conf={{conf:.2f}}")
""".format(weights=weights_path))


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
    print("  YOLOv10 Model Export — Drone/Bird Detection")
    print("=" * 55)
    print(f"  Weights  : {weights}")
    print(f"  Formats  : {', '.join(args.format)}")
    print(f"  Image sz : {args.imgsz}")
    print(f"  FP16     : {args.half}")
    print()

    for fmt in args.format:
        note = FORMAT_NOTES.get(fmt, "")
        if note:
            print(f"  {note}")

    model = YOLO(str(weights))

    exported: dict[str, Path | None] = {}
    for fmt in args.format:
        out = export_format(
            model,
            fmt=fmt,
            imgsz=args.imgsz,
            half=args.half,
            simplify=args.simplify,
            device=args.device,
        )
        exported[fmt] = out
        if out:
            print(f"  ✓ {fmt.upper():12s} → {out}")

    # Print deployment snippets
    for fmt, out_path in exported.items():
        if out_path:
            print_deployment_snippet(fmt, out_path)

    # Print repo integration instructions
    update_repo_inference(weights)

    print()
    print("─" * 55)
    print("  PHASE 5 CHECKLIST")
    print("─" * 55)
    for fmt in args.format:
        out = exported.get(fmt)
        status = "✓" if out else "✗"
        print(f"  {status} Exported to {fmt.upper()}")
    print("  ? Smoke test deployed model on drone_sunset.jpg")
    print("  ? Swap weights path in existing repo inference code")
    print()
    print("  FIX_PLAN complete. All 5 phases implemented.")


if __name__ == "__main__":
    main()
