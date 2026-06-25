# Handoff — LiceentaBogdan (2026-06-07)

## Goal
Bring the Bird vs Drone detector to a presentation-ready state with 3 methods
(HOG+MLP, HOG+SVM, YOLOv8n) + existing YOLOv10, a unified "Train All" button,
a 4-way live detection view, and a results comparison page.

## ROOT CAUSE FOUND (the real bug)
The dataset **labels were corrupted**: ~98.8% of bird images were labeled
class 0 (drone). Filename prefix is the true label (B*=bird, D*=drone). This —
not the detection code — is why birds were detected as drones and YOLO detected
nothing useful. Fixed by `scripts/relabel_by_filename.py` (rewrites class IDs
from filename; originals backed up to `labels_backup.zip`, 34.9 MB).

## Current State

### DONE & VERIFIED
- **Labels relabeled** in both `Data/` and `datasets/merged/` (B→1 bird, D→0 drone).
  Caches cleared (HOG `.cache`, YOLO `labels.cache`).
- **HOG+MLP / HOG+SVM trained** on corrected labels → `models/hog_mlp.pkl`,
  `models/hog_svm.pkl`. Birds now detected:
  - MLP: acc 0.797, bird recall 79% (bird→drone confusion down to 10%)
  - SVM: acc 0.849, bird recall 85% (bird→drone confusion down to 6%)
- **New files**: `src/core/yolov8_detector.py`, `scripts/train_yolov8.py`,
  `scripts/evaluate_all.py`, `scripts/train_hog.py`, `scripts/relabel_by_filename.py`.
- **app.py rewritten**: unified "Train All Models" button, 4-method Live Detection
  (HOG+MLP | HOG+SVM | YOLOv8n | YOLOv10 simultaneously), Results comparison table
  reading `results/all_metrics.json`. Subprocess output → log files (no PIPE deadlock).

### BUG FIXED MID-RUN: multi_scale crash
First YOLO run crashed: `ValueError: Expected more than 1 value per channel ...
torch.Size([1, 256, 1, 1])`. Cause: `multi_scale=True` shrank an image until a
feature map hit 1x1, and the dataset's size-1 last batch (16761 % 8 == 1) made
BatchNorm fail. Fix: set `multi_scale=False` in both `train_yolov8.py` and
`train_yolov10.py`. Also speeds up CPU training. Both relaunched fresh.

### ALL TRAINING COMPLETE (2026-06-07 ~20:30)
Both YOLO models finished cleanly after the multi_scale fix.
- YOLOv8n  → `runs/train/drone_v8/weights/best.pt`
- YOLOv10  → `runs/train/drone_v10_cpu/weights/best.pt`
Stale May-25 nested run `runs/detect/` was DELETED (held a broken 2-epoch model
that both app + eval were wrongly picking up). Path priority flipped in app.py
and evaluate_all.py to prefer the clean `runs/train/...` location.

### FINAL TEST-SET METRICS (results/all_metrics.json)
| Model   | Accuracy | Precision | Recall | F1    | mAP50 |
|---------|----------|-----------|--------|-------|-------|
| HOG+MLP | 0.806    | 0.817     | 0.806  | 0.807 | —     |
| HOG+SVM | 0.827    | 0.833     | 0.827  | 0.826 | —     |
| YOLOv8n | —        | 0.668     | 0.512  | 0.579 | 0.601 |
| YOLOv10 | —        | 0.694     | 0.503  | 0.583 | 0.578 |

## TASK COMPLETE — presentation ready
Just run: `.venv\Scripts\streamlit.exe run app.py`
- Results tab → 4-way comparison table (from results/all_metrics.json)
- Live Detection tab → upload an image, all 4 methods run side-by-side
- Do NOT click "Train All" again unless you want to retrain (it overwrites).

## To regenerate metrics later
`python scripts/evaluate_all.py`  (re-reads saved models/weights)

## Key Commands
```
# Re-relabel (idempotent, already done):
python scripts/relabel_by_filename.py

# Retrain HOG only (fast):
python scripts/train_hog.py --max-train 1500 --max-test 400 --mlp-epochs 40

# Retrain a YOLO model:
python scripts/train_yolov8.py  --merged --epochs 10 --device cpu
python scripts/train_yolov10.py --merged --epochs 10 --device cpu --batch 8 --model yolov10n.pt --name drone_v10_cpu

# Evaluate all 4 → results/all_metrics.json:
python scripts/evaluate_all.py
```

## Environment
- `.venv` at `D:\Repos\LiceentaBogdan\.venv`, ultralytics 8.4.54, torch CPU-only.
- Windows 11, no GPU. YOLO `workers=0` (Windows), `exist_ok=True` (stable run dirs).

## Gotchas
- HOG live detection (sliding window) is CPU-heavy; very slow while YOLO trains.
- To revert labels: unzip `labels_backup.zip` over the repo root.
