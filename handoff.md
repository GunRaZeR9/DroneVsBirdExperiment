# Handoff — LiceentaBogdan (2026-05-25 23:47 +03:00)

## Goal
Migrate drone/bird detector from YOLOv7 + HOG/MLP to YOLOv10, fixing two bugs:
1. **Class confusion** — drone predicted as bird 100% of the time
2. **Duplicate bounding boxes** — ~10 boxes per object

## Current State — COMPLETE except waiting for training

All code is done. Training is running in the background on CPU.

---

## What Was Done (all phases complete)

### Root Cause Fixed
- `src/core/data_loader.py` — class IDs were **inverted** (class 0 mapped to Bird, class 1 to Drone). Fixed to match actual label files: 0=drone, 1=bird.
- `dataset_stats()` — same inversion, fixed.
- Training split had **0 bird annotations** (58,937 drone / 0 bird). Fixed by merge_datasets.py pulling in bird images.

### Files Created
| File | Purpose |
|------|---------|
| `scripts/audit_labels.py` | Audits YOLO label class distribution per split |
| `scripts/merge_datasets.py` | Merges YOLO datasets, writes data.yaml |
| `scripts/train_yolov10.py` | YOLOv10 training entry point |
| `scripts/run_inference.py` | CLI inference: image / video / webcam |
| `scripts/validate_yolov10.py` | Full validation with mAP, confusion matrix |
| `scripts/export_yolov10.py` | Export to ONNX / TensorRT / OpenVINO / CoreML |
| `src/core/yolov10_detector.py` | Drop-in YOLOv10 detector class |
| `datasets/data.yaml` | Correct class mapping for raw Data/ directory |
| `datasets/merged/data.yaml` | Merged dataset (drone + bird), absolute path |
| `hyp.drone.yaml` | YOLOv10 hyperparameters |

### app.py — Fully Updated
- **Live Test tab** has a backend radio button: `HOG + MLP (classical)` | `YOLOv10 (deep learning)`
- YOLOv10 path: loads weights, caches detector in session state, runs `detect()` + `annotate()`, BGR to RGB conversion for display
- YOLOv10 path with no weights: shows warning + training command to start/resume
- HOG+MLP path: unchanged behavior, fixed broken `detect()` call (was calling removed name, now calls `hog_detect()`)
- Sidebar shows YOLOv10 weight status

---

## Training — In Progress

**Started:** 2026-05-25 23:29 local time  
**Command used:**
```
python scripts/train_yolov10.py --device cpu --batch 8 --epochs 50 --model yolov10n.pt --merged --name drone_v10_cpu
```

**Args saved to:** `runs/detect/runs/train/drone_v10_cpu/args.yaml`  
**Weights will appear at:** `runs/detect/runs/train/drone_v10_cpu/weights/best.pt`  
(nested path due to ultralytics bug — `task="detect"` fix was added to the script but current run was already launched before the fix)

**Estimated completion:** ~18 hours from start (50 epochs x ~22 min/epoch on CPU)  
**Estimated finish time:** 2026-05-26 ~17:30 local time

**Note:** `save_period=-1` — weights only saved at the very end. `weights/` directory remains empty until training completes.

**To resume if interrupted:**
```
python scripts/train_yolov10.py --device cpu --batch 8 --epochs 50 --model yolov10n.pt --merged --name drone_v10_cpu --resume
```

---

## Next Step (when training finishes)

1. **Verify weights exist:**
   ```
   runs/detect/runs/train/drone_v10_cpu/weights/best.pt
   ```

2. **Run Streamlit app:**
   ```
   .venv\Scripts\streamlit.exe run app.py
   ```
   Browser -> Live Test tab -> select "YOLOv10 (deep learning)" -> weights path auto-fills -> upload image -> Run Detection

3. **Validate metrics (optional):**
   ```
   python scripts/validate_yolov10.py --weights runs/detect/runs/train/drone_v10_cpu/weights/best.pt
   ```
   Targets: mAP50 > 0.80, mAP50-95 > 0.55, per-class AP drone > 0.75

4. **Export for deployment (optional):**
   ```
   python scripts/export_yolov10.py --weights runs/detect/runs/train/drone_v10_cpu/weights/best.pt --format onnx
   ```

---

## Environment
- Python `.venv` at `D:\Repos\LiceentaBogdan\.venv`
- `ultralytics` installed, `torch` CPU-only
- No NVIDIA GPU — all training/inference on CPU
- Windows 11, PowerShell

## Known Issues / Gotchas
- **Nested runs path** — existing training run saves to `runs/detect/runs/train/drone_v10_cpu/` (not `runs/train/drone_v10_cpu/`). app.py checks both paths automatically. Future runs (with fixed train script) will use the clean path.
- **label_smoothing deprecation** — ultralytics 8.4.54 warns `'label_smoothing' is deprecated`. Non-fatal.
- **Windows workers=0** — DataLoader must use `workers=0` on Windows to avoid multiprocessing hang. Already set in train script.
- **save_period=-1** — no intermediate checkpoints. If training is interrupted mid-run, resume with `--resume` flag.

---

## Resume Prompt
Training is running on CPU for YOLOv10 drone/bird detector. Weights will appear at `runs/detect/runs/train/drone_v10_cpu/weights/best.pt` when done (~18h from 2026-05-25 23:29). app.py is fully updated with dual-backend Live Test tab (HOG+MLP and YOLOv10). When training is done: run streamlit, pick YOLOv10 backend, test. If training stopped, resume with `python scripts/train_yolov10.py --device cpu --batch 8 --epochs 50 --model yolov10n.pt --merged --name drone_v10_cpu --resume`.
