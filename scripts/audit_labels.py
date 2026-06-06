"""
audit_labels.py — Phase 1.1 of FIX_PLAN
=========================================
Audit class distribution in YOLO-format label files across all splits.

Class mapping (verified from actual DT/BT label files):
  0 = drone
  1 = bird

Usage:
    python scripts/audit_labels.py
    python scripts/audit_labels.py --data-root D:/Repos/LiceentaBogdan/Data
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import sys
from pathlib import Path


CLASS_NAMES = {0: "drone", 1: "bird"}
SPLITS = ["train", "valid", "test"]


def audit_split(label_dir: Path) -> dict[int, int]:
    """Count annotations per class in one label directory."""
    counter: dict[int, int] = collections.defaultdict(int)
    label_files = list(label_dir.glob("*.txt"))

    for label_file in label_files:
        try:
            text = label_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(parts[0])
                counter[cls_id] += 1
            except ValueError:
                continue

    return dict(counter)


def format_ratio(counts: dict[int, int]) -> str:
    """Format drone:bird ratio string."""
    drone = counts.get(0, 0)
    bird = counts.get(1, 0)
    if bird == 0:
        return "∞:1" if drone > 0 else "N/A"
    ratio = drone / bird
    return f"{ratio:.2f}:1 (drone:bird)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO label class distribution")
    parser.add_argument(
        "--data-root",
        default=r"D:\Repos\LiceentaBogdan\Data",
        help="Root data directory containing train/, valid/, test/ splits",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.2,
        help="Warn if drone:bird ratio is below this value (default 0.2 = 1:5)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  YOLO Label Audit — Drone vs Bird Detection")
    print("=" * 60)
    print(f"  Data root : {data_root}")
    print(f"  Classes   : 0=drone, 1=bird")
    print()

    total_all: dict[int, int] = collections.defaultdict(int)

    for split in SPLITS:
        label_dir = data_root / split / "labels"
        if not label_dir.exists():
            print(f"  [{split:6s}]  labels dir not found — skipping")
            continue

        n_files = len(list(label_dir.glob("*.txt")))
        counts = audit_split(label_dir)

        drone = counts.get(0, 0)
        bird = counts.get(1, 0)
        total = drone + bird

        for k, v in counts.items():
            total_all[k] += v

        # Build bar representation (ASCII for Windows cp1252 compatibility)
        if total > 0:
            drone_bar = "#" * min(40, int(40 * drone / total))
            bird_bar  = "#" * min(40, int(40 * bird  / total))
        else:
            drone_bar = bird_bar = ""

        print(f"  [{split:6s}]  {n_files} label files")
        print(f"           drone : {drone:6,}  {drone_bar}")
        print(f"           bird  : {bird:6,}  {bird_bar}")
        print(f"           ratio : {format_ratio(counts)}")

        drone_frac = drone / total if total > 0 else 0.0
        if drone_frac < args.warn_threshold and total > 0:
            print(f"  ⚠  WARNING: drone is only {drone_frac*100:.1f}% of {split} annotations.")
            print(f"     Target ≥ {args.warn_threshold*100:.0f}% drone. Add more drone images!")
        print()

    # Totals
    drone_total = total_all.get(0, 0)
    bird_total  = total_all.get(1, 0)
    grand_total = drone_total + bird_total
    drone_pct   = 100 * drone_total / grand_total if grand_total > 0 else 0.0

    print("=" * 60)
    print("  TOTALS (all splits)")
    print(f"  drone : {drone_total:8,}  ({drone_pct:.1f}%)")
    print(f"  bird  : {bird_total:8,}  ({100 - drone_pct:.1f}%)")
    print(f"  ratio : {format_ratio(dict(total_all))}")
    print()

    # Diagnosis
    bird_total = total_all.get(1, 0)
    print("  DIAGNOSIS")
    if grand_total == 0:
        print("  ✗ No annotations found. Check data-root path.")
    elif bird_total == 0:
        print("  ✗ CRITICAL: Zero bird annotations in training data!")
        print("    The model has NEVER seen a bird during training.")
        print("    At inference, any non-drone object defaults to drone/background.")
        print("    Root cause of the class confusion bug.")
        print("    Action: Download DroneVsBird or add Roboflow bird images.")
        print("    Target: drone:bird ≈ 1:1 in the TRAIN split.")
    elif drone_pct < 20:
        print("  ✗ CRITICAL: Severe class imbalance — drone < 20% of data.")
        print("    Root cause of 'Drone → Bird 100%' misclassification.")
        print("    Action: Download VisDrone2019 or DroneVsBird dataset.")
        print("    Target: drone:bird ≈ 1:1 (or at least 1:3)")
    elif drone_pct > 80:
        print("  ✗ CRITICAL: Severe class imbalance — bird < 20% of data.")
        print("    Model will rarely predict 'bird' — class barely seen in training.")
        print("    Action: Add more bird images. Target drone:bird ≈ 1:1.")
    elif drone_pct < 33:
        print("  ! Moderate imbalance — consider adding more drone images.")
        print("    Or use class_weights / copy_paste augmentation in training.")
    else:
        print("  ✓ Class balance looks acceptable (both classes well represented).")

    print()
    print("  NEXT STEP")
    if bird_total < drone_total * 0.33:
        print("  Bird class is under-represented.")
        print("  Download DroneVsBird or any bird image dataset, then:")
        print("    python scripts/merge_datasets.py \\")
        print("        --sources Data D:/Downloads/BirdDataset \\")
        print("        --output datasets/merged")
        print("  Then: python scripts/train_yolov10.py --merged")
    else:
        print("  Dataset looks balanced enough to train.")
        print("  Run: python scripts/train_yolov10.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
