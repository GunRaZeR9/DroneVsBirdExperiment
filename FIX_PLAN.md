# 🛠️ YOLOv7 → YOLOv10 Drone Detection — Full Fix Plan (v2)
**Project:** `D:\Repos\LiceentaBogdan`  
**Date:** 2026-05-25  
**Primary fix:** Migrate to YOLOv10 + dataset surgery  
**Team:** Research Lead · Dataset Engineer · Training Engineer · Inference Engineer

---

## 🔍 Root Cause Summary

| Bug | Symptom | Root cause | Fix strategy |
|---|---|---|---|
| Class confusion | Drone → Bird 100% | Dataset imbalance (18k birds vs ~1k drones in OpenImages/COCO) | Dataset surgery + class weighting |
| Duplicate boxes | 10 detections on 1 drone | NMS IoU threshold too high; 3 scale heads each fire independently | **Architecturally eliminated by YOLOv10** |

---

## 🏗️ Why Switch to YOLOv10

YOLOv10 (2024) introduces a **dual-head, NMS-free architecture**:

- One head optimises for training (with NMS-like supervision)
- One head optimises for inference (produces exactly **one prediction per object**, no NMS step)

Bug 2 — your 10 duplicate boxes — becomes **architecturally impossible** in YOLOv10. You do not tune `iou-thres`. You do not add `--agnostic-nms`. The model simply does not produce duplicates.

Additionally, the **Ultralytics API** (which YOLOv10 uses) gives you:
- Built-in confusion matrix, per-class mAP, and training curves with zero extra code
- Auto-anchor computation built into the training loop
- One-line export to ONNX, TensorRT, CoreML

Your existing annotation files (`.txt` in YOLO format) work **unchanged**. Only the weights and training command change.

---

## Phase 0 — Environment Setup

```bash
# Create a clean environment (recommended)
python -m venv venv_yolov10
venv_yolov10\Scripts\activate   # Windows

# Install Ultralytics (includes YOLOv10)
pip install ultralytics

# Verify
python -c "from ultralytics import YOLO; print('OK')"
```

> **Note:** If you have a CUDA GPU, ensure `torch` with CUDA is installed first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Phase 1 — Dataset Surgery

This is the highest-impact phase. No model switch fixes class confusion — that is always a data problem.

### 1.1 — Audit Existing Labels

```python
# audit_labels.py — run this first
import os, glob, collections

label_dir = r"D:\Repos\LiceentaBogdan\datasets\train\labels"
class_counts = collections.Counter()

for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_counts[int(parts[0])] += 1

print("Class distribution:", dict(class_counts))
# Healthy target: {0: ~3000, 1: ~3000}  (drone : bird ≈ 1:1)
# Problem signal: {0: 500, 1: 8000}     (severely imbalanced)
```

Verify your `data.yaml` class order — this is a common silent bug:

```yaml
# data.yaml
path: D:/Repos/LiceentaBogdan/datasets
train: train/images
val: val/images

nc: 2
names:
  0: drone   # MUST be 0 — if reversed, the model learns backwards
  1: bird
```

### 1.2 — Add Drone-Specific Datasets

Your ~3,000 training images likely contain fewer than 700 drone samples. You need at least 2,500 more. Download from these sources:

| Dataset | Images | Where | Notes |
|---|---|---|---|
| **VisDrone2019** | ~6,500 | https://github.com/VisDrone/VisDrone-Dataset | Best quality, diverse altitudes and distances |
| **DroneVsBird** | ~3,500 | Search GitHub: `drone-vs-bird-detection` | Exact dual-class use case |
| **MAV-VID** | ~1,500 | https://github.com/animetauren/MAV-VID | Moving drones, varied backgrounds |
| **Roboflow Universe** | Variable | https://universe.roboflow.com → search "UAV detection" | Filter for YOLO format export |

**Target dataset composition after merge:**

```
Drone images:  3,000 – 4,000
Bird images:   3,000 – 4,000
Ratio:         1:1 to 1:1.5 (drone:bird)

Split:
  train/  → 80%
  val/    → 15%
  test/   → 5%
```

### 1.3 — Merge and Re-split (Helper Script)

```python
# merge_datasets.py
import os, shutil, random, glob

SOURCES = [
    r"D:\Repos\LiceentaBogdan\datasets\original",
    r"D:\Downloads\VisDrone\converted",
    r"D:\Downloads\DroneVsBird",
]
OUTPUT = r"D:\Repos\LiceentaBogdan\datasets\merged"

for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT}/{split}/labels", exist_ok=True)

all_images = []
for src in SOURCES:
    all_images += glob.glob(f"{src}/**/*.jpg", recursive=True)
    all_images += glob.glob(f"{src}/**/*.png", recursive=True)

random.shuffle(all_images)
n = len(all_images)
splits = {"train": all_images[:int(n*0.8)],
          "val":   all_images[int(n*0.8):int(n*0.95)],
          "test":  all_images[int(n*0.95):]}

for split, imgs in splits.items():
    for img in imgs:
        label = img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        shutil.copy(img, f"{OUTPUT}/{split}/images/")
        if os.path.exists(label):
            shutil.copy(label, f"{OUTPUT}/{split}/labels/")

print(f"Done. Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
```

---

## Phase 2 — Training with YOLOv10

### 2.1 — Model Size Selection

| Model | Params | mAP (COCO) | Speed (T4) | Recommendation |
|---|---|---|---|---|
| yolov10n | 2.3M | 38.5 | 1.84ms | Too small for 2-class fine-tuning |
| **yolov10s** | 7.2M | 46.3 | 2.49ms | **Good for fast iteration / testing** |
| **yolov10m** | 15.4M | 51.1 | 4.74ms | **Best balance — recommended** |
| yolov10l | 24.4M | 53.2 | 7.28ms | Use if GPU allows and accuracy still lacking |

Start with `yolov10m`. If training takes too long, drop to `yolov10s`.

### 2.2 — Custom Hyperparameter File

Create `D:\Repos\LiceentaBogdan\hyp.drone.yaml`:

```yaml
# hyp.drone.yaml — tuned for 2-class drone/bird problem
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Loss weights
box: 7.5
cls: 0.3          # lowered from default 0.5 — forces more careful class learning
dfl: 1.5

# Augmentation — important for drone variety
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 5.0
translate: 0.1
scale: 0.9        # large range — drones appear at many distances
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.15       # helps with class boundary learning
copy_paste: 0.1   # copies drone instances into bird-heavy images during training
label_smoothing: 0.1   # CRITICAL — prevents "Bird 100%" overconfidence
```

### 2.3 — Training Command

```bash
cd D:\Repos\LiceentaBogdan

yolo detect train \
  model=yolov10m.pt \
  data=datasets/merged/data.yaml \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=drone_v10 \
  cos_lr=True \
  multi_scale=True \
  cfg=hyp.drone.yaml \
  workers=8
```

Or as Python:

```python
from ultralytics import YOLO

model = YOLO("yolov10m.pt")

model.train(
    data=r"D:\Repos\LiceentaBogdan\datasets\merged\data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/train",
    name="drone_v10",
    cos_lr=True,
    multi_scale=True,
    label_smoothing=0.1,
    copy_paste=0.1,
    mixup=0.15,
    scale=0.9,
    cls=0.3,
)
```

### 2.4 — Key Difference from YOLOv7 Training

With YOLOv7 you had to manually configure NMS thresholds, anchor files, and loss weights via separate YAML files. In YOLOv10 / Ultralytics:

- **Anchors:** Computed automatically at the start of training. No `autoanchor.py` step.
- **NMS:** Not needed at inference — the dual-head eliminates duplicates by design.
- **Metrics:** Printed to console and saved to `runs/train/drone_v10/results.csv` automatically, including per-class AP.

### 2.5 — Metrics to Watch During Training

Monitor `runs/train/drone_v10/results.csv`:

| Metric | Target by epoch 150 | Warning sign |
|---|---|---|
| `metrics/mAP50` | > 0.80 | < 0.60 at epoch 80 → dataset still imbalanced |
| `metrics/mAP50-95` | > 0.55 | — |
| `val/cls_loss` | Steadily decreasing | Flat after epoch 30 → class weighting needed |
| Per-class AP (drone) | > 0.75 | If bird AP is high but drone AP is low → add more drone data |
| `train/cls_loss` | < 0.05 by epoch 100 | High → raise `cls` weight in hyps |

Tensorboard is available automatically:
```bash
tensorboard --logdir runs/train/drone_v10
```

---

## Phase 3 — Inference (No NMS Tuning Required)

```python
from ultralytics import YOLO

model = YOLO(r"runs/train/drone_v10/weights/best.pt")

# YOLOv10 NMS-free inference — no iou_thres needed
results = model.predict(
    source=r"D:\path\to\your\drone_sunset.jpg",
    conf=0.45,     # confidence threshold — only tune this
    save=True,
    show_labels=True,
    show_conf=True,
)
```

**Notice:** No `iou_thres`, no `agnostic_nms`. The model produces exactly one box per object. If you still see duplicates (edge case: very low overlap between predictions), add `iou=0.35` — but you likely won't need it.

### CLI inference:

```bash
yolo detect predict \
  model=runs/train/drone_v10/weights/best.pt \
  source="D:\path\to\drone_sunset.jpg" \
  conf=0.45
```

---

## Phase 4 — Validation

### 4.1 — Full Validation Run

```python
from ultralytics import YOLO

model = YOLO(r"runs/train/drone_v10/weights/best.pt")

metrics = model.val(
    data=r"D:\Repos\LiceentaBogdan\datasets\merged\data.yaml",
    split="test",   # run on held-out test split
    plots=True,     # generates confusion matrix, PR curve, F1 curve
    save_json=True,
)

print(f"mAP50:    {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Per-class AP: {metrics.box.ap_class_index}")
```

### 4.2 — Reading the Confusion Matrix

Open `runs/val/exp/confusion_matrix_normalized.png`:

```
              Predicted
              drone    bird     background
Actual drone  [HIGH]   [LOW]    [LOW]
Actual bird   [LOW]    [HIGH]   [LOW]
```

If the drone→bird cell is still high after retraining, go back to Phase 1 — there is still label contamination or too few drone training images.

### 4.3 — Edge Case Test Set

Manually test on these specific scenarios and log results:

```bash
# Batch test on a folder of edge cases
yolo detect predict \
  model=runs/train/drone_v10/weights/best.pt \
  source="D:\Repos\LiceentaBogdan\test_edge_cases\" \
  conf=0.45 \
  save=True \
  save_txt=True
```

Edge cases to include in that folder:
- Drone against bright sky (the sunset photo)
- Drone as dark silhouette (backlit)
- Small drone far away (< 60px in frame)
- Bird in same frame as drone
- Multiple drones
- Drone partially occluded

---

## Phase 5 — Export for Deployment

Once validation passes, export to a deployment format:

```python
from ultralytics import YOLO

model = YOLO(r"runs/train/drone_v10/weights/best.pt")

# ONNX (universal, CPU/GPU)
model.export(format="onnx", imgsz=640, simplify=True)

# TensorRT (NVIDIA GPU, fastest inference)
model.export(format="engine", imgsz=640, half=True, device=0)

# OpenVINO (Intel CPU)
model.export(format="openvino", imgsz=640)
```

For the existing Python app in your repo, swap the weights path and update the inference call per Phase 3.

---

## 📋 Master Checklist

```
PHASE 0 — ENVIRONMENT (20 min)
[ ] pip install ultralytics
[ ] Verify CUDA torch if GPU available
[ ] python -c "from ultralytics import YOLO; print('OK')"

PHASE 1 — DATASET (1–2 days)
[ ] Run audit_labels.py — check class distribution
[ ] Verify data.yaml: names[0] = drone, names[1] = bird
[ ] Download VisDrone2019 or DroneVsBird
[ ] Run merge_datasets.py — target 3k+ drone images
[ ] Check final ratio: drone:bird ≈ 1:1

PHASE 2 — TRAINING (6–12h GPU)
[ ] Create hyp.drone.yaml with label_smoothing=0.1
[ ] Run yolo detect train with yolov10m.pt
[ ] Monitor drone per-class AP — should grow each epoch
[ ] Confirm val/cls_loss is decreasing after epoch 30
[ ] Save best.pt path

PHASE 3 — INFERENCE CHECK
[ ] Run predict on original drone sunset image
[ ] Confirm: 1 box, labelled "drone", conf 70–90%
[ ] No NMS tuning needed

PHASE 4 — VALIDATION
[ ] Run model.val(split="test", plots=True)
[ ] Open confusion_matrix_normalized.png
[ ] Confirm drone→bird cell is near 0
[ ] Test all 6 edge case scenarios

PHASE 5 — EXPORT
[ ] Export to ONNX or TensorRT
[ ] Swap weights in existing repo inference code
[ ] Smoke test deployed model
```

---

## ⚡ Quick Comparison: Before vs After

| | Before (YOLOv7) | After (YOLOv10) |
|---|---|---|
| Duplicate boxes | 10 per drone | 1 per drone (by design) |
| NMS tuning required | Yes (`iou-thres`, `agnostic-nms`) | No |
| Class label | Bird 100% | Drone 70–90% |
| Training command | Long, multi-file config | Single `yolo train` call |
| Anchor config | Manual `autoanchor.py` | Automatic |
| Metrics | Manual CSV parsing | Built-in Tensorboard + plots |
| Export | Manual scripts | `model.export(format=...)` |

---

*Generated by professional-dev-team · LiceentaBogdan fix plan v2.0 — YOLOv10 edition*
