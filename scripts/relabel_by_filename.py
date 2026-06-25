"""
relabel_by_filename.py — Fix corrupted class IDs in YOLO label files.

ROOT CAUSE (discovered 2026-06-07)
----------------------------------
The dataset's label *class IDs* are unreliable: ~98.8% of bird images carry
class 0 (drone) instead of class 1 (bird). The bounding-box coordinates are
correct — only the leading class_id token is wrong.

The reliable ground truth is the **filename prefix**:
    B*  (BT/BV) → bird   → class 1
    D*  (DT/DV) → drone  → class 0

This script rewrites the first token of every annotation line to the class
implied by the filename's first letter. Coordinates are preserved exactly.
Polygon lines (class + many coords) are handled the same way — only the class
token changes.

The operation is IDEMPOTENT: running it twice produces the same result.
Original labels are backed up to labels_backup.zip before the first change.

Usage:
    python scripts/relabel_by_filename.py                 # relabel both datasets
    python scripts/relabel_by_filename.py --merged-only    # only datasets/merged
    python scripts/relabel_by_filename.py --no-backup       # skip the zip backup
    python scripts/relabel_by_filename.py --dry-run         # report only, no writes
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# label dirs to process, relative to ROOT
_DATA_DIRS = [
    "Data/train/labels", "Data/valid/labels", "Data/test/labels",
]
_MERGED_DIRS = [
    "datasets/merged/train/labels", "datasets/merged/val/labels", "datasets/merged/test/labels",
]


def class_for_filename(name: str) -> int | None:
    """Return target class from filename's first letter, or None if unknown."""
    first = name[:1].upper()
    if first == "B":
        return 1   # bird
    if first == "D":
        return 0   # drone
    return None


def relabel_line(line: str, target_cls: int) -> str:
    """Rewrite the class token (first token) of a single annotation line."""
    stripped = line.rstrip("\n")
    if not stripped.strip():
        return line  # blank line — leave as-is
    tokens = stripped.split()
    tokens[0] = str(target_cls)
    return " ".join(tokens) + "\n"


def collect_label_files(dirs: list[str]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        p = ROOT / d
        if p.is_dir():
            files.extend(sorted(p.glob("*.txt")))
    return files


def backup(files: list[Path], zip_path: Path) -> None:
    """Zip all label files (preserving paths relative to ROOT) before editing."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=str(f.relative_to(ROOT)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Relabel YOLO class IDs from filename prefix")
    ap.add_argument("--merged-only", action="store_true", help="Only relabel datasets/merged/")
    ap.add_argument("--no-backup", action="store_true", help="Skip the labels_backup.zip backup")
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = ap.parse_args()

    dirs = list(_MERGED_DIRS)
    if not args.merged_only:
        dirs = _DATA_DIRS + dirs

    files = collect_label_files(dirs)
    if not files:
        print("No label files found. Check dataset paths.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} label files across {len(dirs)} directories.")

    # Backup first (unless skipped or dry-run)
    if not args.no_backup and not args.dry_run:
        zip_path = ROOT / "labels_backup.zip"
        print(f"Backing up originals to {zip_path} ...")
        backup(files, zip_path)
        print(f"  Backup complete ({zip_path.stat().st_size / 1e6:.1f} MB).")

    # Relabel
    changed_files = 0
    changed_lines = 0
    skipped_unknown = 0

    for f in files:
        target = class_for_filename(f.name)
        if target is None:
            skipped_unknown += 1
            continue

        original = f.read_text(encoding="utf-8", errors="ignore")
        new_lines = []
        file_changed = False
        for line in original.splitlines(keepends=True):
            new_line = relabel_line(line, target)
            if new_line != line:
                file_changed = True
                changed_lines += 1
            new_lines.append(new_line)

        if file_changed:
            changed_files += 1
            if not args.dry_run:
                f.write_text("".join(new_lines), encoding="utf-8")

    verb = "would change" if args.dry_run else "changed"
    print(f"\nDone. {verb} {changed_lines} lines in {changed_files} files.")
    if skipped_unknown:
        print(f"Skipped {skipped_unknown} files with unrecognized prefix (not B*/D*).")

    # Verify result distribution
    print("\nVerifying class distribution after relabel:")
    import collections
    for d in dirs:
        p = ROOT / d
        if not p.is_dir():
            continue
        bird = drone = 0
        for f in p.glob("*.txt"):
            tcls = class_for_filename(f.name)
            if tcls == 1:
                bird += 1
            elif tcls == 0:
                drone += 1
        print(f"  {d:38s} bird-images={bird:5d}  drone-images={drone:5d}")


if __name__ == "__main__":
    main()
