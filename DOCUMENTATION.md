# Bird vs Drone Detection — Technical Documentation

**Licence Thesis Project**
A comparative study of classical machine-learning and modern deep-learning object
detectors for the binary discrimination of **birds** and **drones** in aerial/ground
imagery.

---

## Table of Contents
1. [Objective and Scope](#1-objective-and-scope)
2. [System Architecture](#2-system-architecture)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Hyperparameters and Training Configuration](#5-hyperparameters-and-training-configuration)
6. [Experimental Setup](#6-experimental-setup)
7. [Results](#7-results)
8. [Findings and Analysis](#8-findings-and-analysis)
9. [Limitations](#9-limitations)
10. [Future Work](#10-future-work)
11. [Reproducibility](#11-reproducibility)
12. [Conclusion](#12-conclusion)

---

## 1. Objective and Scope

The goal of this project is to detect and classify two visually similar but
semantically distinct classes — **birds** and **drones** — and to **compare three
detection methodologies** spanning two paradigms:

| Paradigm | Method | Family |
|----------|--------|--------|
| Classical ML | **HOG + MLP** | Hand-crafted features + neural classifier |
| Classical ML | **HOG + SVM** | Hand-crafted features + kernel classifier |
| Deep learning | **YOLOv8n** | Single-stage CNN detector (anchor-based, NMS) |
| Deep learning | **YOLOv10n** | Single-stage CNN detector (NMS-free, dual-head) |

The deliverable is an interactive **Streamlit** application that allows:
- training all models from a single action ("Train All Models"),
- running **all four detectors simultaneously** on an uploaded image (live test),
- viewing a **side-by-side evaluation comparison** on the held-out test set.

The discrimination problem is motivated by real-world counter-UAS (Unmanned Aerial
System) applications, where a perception system must avoid raising false alarms on
birds while reliably detecting drones.

---

## 2. System Architecture

```
LiceentaBogdan/
├── app.py                       # Streamlit UI: Train All | Results | Live Detection
├── src/core/
│   ├── data_loader.py           # YOLO-label parsing → patch arrays (HOG pipeline)
│   ├── feature_extractor.py     # HOG descriptor computation
│   ├── model.py                 # BirdDroneModel: MLP/SVM wrapper (3-class)
│   ├── detector.py              # Multi-scale sliding-window detector (HOG)
│   ├── yolov8_detector.py       # YOLOv8 inference wrapper (Ultralytics)
│   └── yolov10_detector.py      # YOLOv10 inference wrapper (Ultralytics)
├── src/utils/metrics.py         # Classification + IoU/AP metrics
├── scripts/
│   ├── relabel_by_filename.py   # Label-corruption fix (B→bird, D→drone)
│   ├── train_hog.py             # CLI training for HOG+MLP and HOG+SVM
│   ├── train_yolov8.py          # YOLOv8 training entry point
│   ├── train_yolov10.py         # YOLOv10 training entry point
│   ├── evaluate_all.py          # Evaluates all 4 models → results/all_metrics.json
│   └── merge_datasets.py        # Builds the balanced merged dataset
├── datasets/merged/data.yaml    # YOLO dataset config (nc=2; 0=drone, 1=bird)
├── models/                      # hog_mlp.pkl, hog_svm.pkl (saved classical models)
├── runs/train/                  # YOLO training runs (weights + results.csv)
└── results/all_metrics.json     # Persisted 4-way comparison
```

Two parallel pipelines share one dataset:

- **Classical pipeline** (`data_loader → feature_extractor → model → detector`)
  operates on **image patches**: objects are cropped from annotations, resized to
  64×64, and described with HOG. Inference uses a multi-scale **sliding window**.
- **Deep pipeline** (`yolov8_detector` / `yolov10_detector` via Ultralytics)
  operates **end-to-end** on full images, predicting boxes and classes directly.

Both detector wrappers return an identical 6-tuple
`(x, y, w, h, confidence, label)` so the UI can treat them interchangeably.

---

## 3. Dataset

### 3.1 Source and Structure

The dataset follows the YOLO convention: each image has a matching `.txt` label
file with one annotation per line, `class_id  x_center  y_center  width  height`
(normalised). Both standard bounding boxes and polygon annotations are present.

Images follow a strict filename convention that encodes the ground-truth class:

| Prefix | Meaning | Split |
|--------|---------|-------|
| `BT*`  | **B**ird, **T**rain/Test | train, test |
| `BV*`  | **B**ird, **V**alidation | valid |
| `DT*`  | **D**rone, **T**rain/Test | train, test |
| `DV*`  | **D**rone, **V**alidation | valid |

Two dataset roots are used:
- `Data/` — raw split (`train/`, `valid/`, `test/`) consumed by the classical
  (HOG) pipeline.
- `datasets/merged/` — a class-balanced merge (`train/`, `val/`, `test/`) used for
  YOLO training, declared in `datasets/merged/data.yaml`
  (`nc: 2`, `names: {0: drone, 1: bird}`).

### 3.2 Critical Finding — Label Corruption (root cause)

A central finding of this work is that **the dataset's annotation class IDs were
systematically corrupted**. A cross-tabulation of *filename prefix* (true class)
versus *annotated class ID* revealed:

| Split | Bird images (`B*`) labelled class 0 (drone) | Bird images labelled class 1 (bird) | % bird images mislabelled |
|-------|---------------------------------------------|-------------------------------------|---------------------------|
| train | 6 119 | 73 | **98.8 %** |
| val   | 1 142 | 12 | 98.9 % |
| test  | 385   | 4  | 99.0 % |

In other words, **almost every bird was annotated as a drone (class 0)**. The
bounding-box *coordinates* were correct — only the class token was wrong. This
single defect explains every observed failure mode prior to the fix:

- **HOG+MLP/SVM** were trained on data where the "Bird" class essentially did not
  exist (`Bird` patch count ≈ 0), so they could never predict it — every bird was
  reported as a drone.
- **YOLO** models learned a near-degenerate single-class ("everything is a drone")
  distribution, producing meaningless detections.

This is a textbook illustration that **data quality dominates model quality**: no
amount of architecture choice or hyperparameter tuning can compensate for
corrupted labels.

### 3.3 Relabeling Methodology

The fix (`scripts/relabel_by_filename.py`) rewrites the **class token of every
annotation line** using the reliable filename prefix:

```
filename starts with 'B'  →  class 1 (bird)
filename starts with 'D'  →  class 0 (drone)
```

Properties of the procedure:
- **Coordinate-preserving** — only the first token per line is changed; bounding
  boxes and polygons are untouched.
- **Idempotent** — running it repeatedly yields the same result.
- **Reversible** — all original labels are archived to `labels_backup.zip`
  (34.9 MB) before any write.
- Applied to **both** `Data/` and `datasets/merged/` (41 858 files, 74 652 lines
  rewritten).

After relabeling, all HOG patch caches (`*/.cache`) and YOLO label caches
(`*/labels.cache`) were invalidated and rebuilt so the corrected labels propagate.

### 3.4 Final Class Distribution (post-relabel)

**Image-level counts (merged dataset):**

| Split | Bird images | Drone images | Total |
|-------|-------------|--------------|-------|
| train | 6 759 | 10 002 | 16 761 |
| val   | 1 265 | 1 877  | 3 142  |
| test  | 427   | 622    | 1 049  |

**Patch-level distribution used to train/test the HOG classifiers** (3-class:
Background / Bird / Drone), `max_train=1500`, `max_test=400` images sampled:

| Split | Background | Bird | Drone |
|-------|-----------|------|-------|
| train | 3 861 | 1 574 | 2 286 |
| test  | 409   | 172   | 235   |

> The classical pipeline is a **3-class** problem (it must additionally reject the
> "Background" of empty sliding windows), whereas YOLO is **2-class** (it localises
> objects directly and has no explicit background class).

---

## 4. Methodology

### 4.1 HOG Feature Extraction (`feature_extractor.py`)

Each patch is resized to a fixed **64×64** window and described with the
**Histogram of Oriented Gradients** descriptor:

| HOG parameter | Value |
|---------------|-------|
| Window size | 64 × 64 px |
| Orientations | 9 |
| Pixels per cell | 8 × 8 |
| Cells per block | 2 × 2 |
| Block normalisation | L2-Hys (skimage default) |
| Colour | RGB (`channel_axis = -1`) |
| Gamma correction | `transform_sqrt = True` |
| Resulting descriptor length | **5 292** (9 orient × 7×7 blocks × 2×2 cells × 3 channels) |

HOG encodes local edge/gradient orientation structure and is robust to modest
illumination changes — historically effective for rigid-shape detection.

### 4.2 Classical Classifier (`model.py` — `BirdDroneModel`)

A common preprocessing and balancing stack precedes both classifiers:

1. **StandardScaler** — zero-mean/unit-variance normalisation of the 5 292-D HOG
   vector.
2. **PCA → 256 components** — dimensionality reduction for class separability and
   speed.
3. **SMOTE** — synthetic minority oversampling to balance Background/Bird/Drone
   before fitting.

#### 4.2.1 HOG + MLP

A multi-layer perceptron trained epoch-by-epoch via `partial_fit`:

| Parameter | Value |
|-----------|-------|
| Hidden layers | 3 × 128 units |
| Activation | ReLU |
| Solver | Adam |
| Initial learning rate | 0.001 (adaptive) |
| Epochs | 30–40 |
| Batch size | 32 |
| Output classes | 3 (Background, Bird, Drone) |

#### 4.2.2 HOG + SVM

A kernel support-vector classifier with probability calibration:

| Parameter | Value |
|-----------|-------|
| Kernel | RBF |
| C | 10.0 |
| Class weighting | `balanced` |
| Probability | Platt scaling (`probability=True`) |
| Output classes | 3 (Background, Bird, Drone) |

### 4.3 Sliding-Window Detector (`detector.py`)

Because the classical models classify fixed patches, detection on a full image is
performed by a **multi-scale sliding window**:

| Component | Configuration |
|-----------|---------------|
| Image pyramid | scale factor 0.8, up to 6 levels, min side 64 px |
| Window sizes | 64×64, 96×96, 128×128 |
| Stride | `max(4, 32 − seq_length·2)` (UI `seq_length` ∈ [1,16]) |
| Background filter | HOG-energy gate rejects blank/sky windows |
| Confidence | object score = `1 − P(background)`, thresholded (UI default 0.72) |
| Duplicate removal | per-class **Non-Maximum Suppression**, IoU = 0.3 |

### 4.4 YOLOv8n (`yolov8_detector.py`, `train_yolov8.py`)

YOLOv8 nano — a single-stage anchor-based CNN detector (Ultralytics). The wrapper
loads `best.pt`, runs `model.predict`, and converts results to the common 6-tuple.
Inference confidence default: **0.35** (tunable in the UI). 2-class head
(`0=drone, 1=bird`). ~3.0 M parameters.

### 4.5 YOLOv10n (`yolov10_detector.py`, `train_yolov10.py`)

YOLOv10 nano — a more recent single-stage detector featuring an **NMS-free
dual-head** design (consistent one-to-one + one-to-many label assignment), so it
emits at most one box per object without a post-hoc NMS step. ~2.27 M parameters,
6.5 GFLOPs. Inference confidence default: 0.45 (tunable).

---

## 5. Hyperparameters and Training Configuration

### 5.1 Classical Models

See §4.2. Both models share StandardScaler → PCA(256) → SMOTE; they differ only in
the final estimator (MLP vs SVC).

### 5.2 YOLO Training (`hyp.drone.yaml` + script defaults)

Both YOLO models were trained with identical settings on the merged dataset:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image size | 416 | Smaller input → tractable CPU training |
| Batch size | 8 | Memory-bounded on CPU |
| Epochs | 10 | Time-bounded (presentation deadline) |
| Device | CPU | No GPU available |
| Workers | 0 | Required to avoid Windows DataLoader deadlock |
| Optimizer | auto → AdamW (lr≈0.00167) | Ultralytics auto-selection |
| `lr0` / `lrf` | 0.01 / 0.01 | Base/final LR schedule |
| Scheduler | cosine (`cos_lr=True`) | Smooth decay |
| `momentum` | 0.937 | — |
| `weight_decay` | 0.0005 | — |
| `warmup_epochs` | 3.0 | — |
| Box / cls / dfl loss weights | 7.5 / **0.3** / 1.5 | Lowered `cls` to discourage overconfidence |
| `label_smoothing` | 0.1 | Mitigates 100%-confidence overfitting |
| `mixup` / `copy_paste` | 0.15 / 0.1 | Augmentation to help the minority (bird) class |
| `scale` | 0.9 | Wide scale range (objects appear at many distances) |
| `fliplr` / HSV | 0.5 / (0.015, 0.7, 0.4) | Standard photometric/geometric augmentation |
| **`multi_scale`** | **False** | **Disabled after a crash — see §8.2** |
| `save_period` | 5 | Checkpoint every 5 epochs |
| `exist_ok` | True | Stable run directory (weights path does not drift) |

---

## 6. Experimental Setup

| Component | Detail |
|-----------|--------|
| OS | Windows 11 Pro (10.0.26200) |
| Hardware | CPU-only (no NVIDIA GPU) |
| Python env | local `.venv` |
| Deep-learning stack | Ultralytics 8.4.54, PyTorch (CPU build) |
| Classical stack | scikit-learn, scikit-image, imbalanced-learn (SMOTE), OpenCV, joblib |
| UI | Streamlit |

**Observed training wall-clock (CPU, parallel runs):**
- YOLOv8n, 10 epochs: ≈ 13 147 s (≈ 3 h 39 m)
- YOLOv10n, 10 epochs: ≈ 23 782 s (≈ 6 h 36 m)
- HOG+MLP and HOG+SVM (combined): a few minutes.

---

## 7. Results

### 7.1 Quantitative Comparison (held-out test set)

Source: `results/all_metrics.json`, produced by `scripts/evaluate_all.py`.
Classical models are scored at **patch level** (3-class classification); YOLO
models are scored with **detection metrics** (`model.val`, mAP@0.50 IoU).

| Model | Accuracy | Precision | Recall | F1 | mAP@50 |
|-------|----------|-----------|--------|------|--------|
| **HOG + MLP** | 0.806 | 0.817 | 0.806 | 0.807 | — |
| **HOG + SVM** | **0.827** | **0.833** | **0.827** | **0.826** | — |
| **YOLOv8n** | — | 0.668 | 0.512 | 0.579 | **0.601** |
| **YOLOv10n** | — | 0.694 | 0.503 | 0.583 | 0.578 |

> ⚠️ **Metrics are not directly comparable across paradigms.** The classical
> accuracy/F1 are computed on **pre-cropped object patches** (an easier, localised
> task), whereas YOLO mAP@50 measures **full detection** (localisation *and*
> classification over the whole image). The two numbers answer different
> questions; see §8.3.

### 7.2 YOLO Training Metrics (final epoch, validation split)

From `runs/train/*/results.csv`, epoch 10:

| Model | Precision(B) | Recall(B) | mAP@50 | mAP@50–95 |
|-------|--------------|-----------|--------|-----------|
| YOLOv8n | 0.716 | 0.531 | 0.623 | 0.405 |
| YOLOv10n | 0.710 | 0.521 | 0.597 | 0.388 |

Both loss components (box/cls/dfl) decreased monotonically across the 10 epochs,
indicating healthy convergence; mAP was still rising at epoch 10 (i.e., the models
are **under-trained**, bounded by the CPU time budget — see §9).

### 7.3 HOG Classifier Confusion Matrices (patch test set)

Rows = ground truth, columns = prediction; order **[Background, Bird, Drone]**.
Test patch support: 409 Background, 172 Bird, 235 Drone.

**HOG + MLP** (accuracy 0.806):
```
            pred Bg  pred Bird  pred Drone
true Bg   [   307        43         59   ]
true Bird [    18       136         18   ]   → bird recall = 136/172 = 79.1 %
true Drone[    12        16        207   ]   → drone recall = 207/235 = 88.1 %
```

**HOG + SVM** (accuracy 0.827):
```
            pred Bg  pred Bird  pred Drone
true Bg   [   327        42         40   ]
true Bird [    15       146         11   ]   → bird recall = 146/172 = 84.9 %
true Drone[     9         6        220   ]   → drone recall = 220/235 = 93.6 %
```

**Interpretation.** On tightly-cropped object patches, both classical models
discriminate birds and drones well. Crucially, the **Bird→Drone confusion that
plagued the corrupted dataset is resolved**: only 18/172 (MLP) and 11/172 (SVM)
birds are mislabelled as drones, versus ~100 % before the relabeling fix. SVM is
the stronger classical model on every axis.

### 7.4 Qualitative Live Detection (full-image, sliding window vs end-to-end)

Two representative full images were processed by all four detectors in the UI.

**Image A — a blue wren (true class: bird), foliage background:**

| Detector | Top label | # boxes | Verdict |
|----------|-----------|---------|---------|
| HOG + MLP | **Bird 100 %** | 40 | ✅ correct class, very noisy (many overlapping boxes) |
| HOG + SVM | Drone 99.8 % | 34 | ❌ wrong class, noisy |
| YOLOv8n | **bird 0.42** | 2 | ✅ correct, clean |
| YOLOv10n | drone 0.40 | 1 | ❌ wrong class, but single clean box |

**Image B — a quadcopter over a beach (true class: drone), sky background:**

| Detector | Top label | # boxes | Verdict |
|----------|-----------|---------|---------|
| HOG + MLP | Bird 100 % | 14 | ❌ wrong class, noisy |
| HOG + SVM | **Drone 99.4 %** | 9 | ✅ correct top box, noisy |
| YOLOv8n | **drone 0.85** | 1 | ✅ correct, clean, high confidence |
| YOLOv10n | **drone 0.87** | 1 | ✅ correct, clean, high confidence |

**Observations.**
- **YOLOv8n was the most consistent** detector across both images (bird→bird,
  drone→drone, single high-confidence box). YOLOv10n matched it on the drone but
  mislabelled the bird (consistent with its slightly lower mAP at 10 epochs).
- **YOLO produces clean, single boxes**; the classical detectors emit **dozens of
  overlapping boxes** even after per-class NMS, because every grid window over a
  textured object can fire.
- The HOG models exhibit a **patch↔full-image performance gap** (see §8.3): their
  per-patch accuracy is high, but on loose sliding windows the dominant label can
  flip (e.g., SVM confidently reports "Drone" on a bird image).

---

## 8. Findings and Analysis

### 8.1 Label corruption was the dominant defect (§3.2)
The most important empirical result of the project is methodological: a single
data-quality defect (≈99 % of birds mislabelled as drones) accounted for *all*
pre-fix failures across *all four* models. Fixing the labels — not changing any
model — restored bird detectability everywhere. **Garbage labels in, garbage
models out.**

### 8.2 `multi_scale` training crash (a reproducible deep-learning pitfall)
The first YOLO training runs crashed with:
```
ValueError: Expected more than 1 value per channel when training,
            got input size torch.Size([1, 256, 1, 1])
```
**Cause:** with `multi_scale=True`, Ultralytics randomly rescales each batch; an
aggressive downscale can shrink a feature map to **1×1** spatially. Simultaneously,
the training set size (16 761) is **not divisible by the batch size (8)** — it
leaves a final batch of **one** image (`16761 mod 8 = 1`). BatchNorm cannot compute
statistics from a single value per channel → it raises. The crash is intermittent
because it requires both conditions (size-1 last batch *and* an unlucky 1×1
rescale) to coincide.
**Fix:** set `multi_scale=False` in both training scripts. This removes the 1×1
feature-map condition entirely (at 416 px the deepest map is 13×13, giving 169
values per channel even for a batch of one) and, as a side benefit, **speeds up CPU
training**. After the fix, both models trained to completion.

### 8.3 Patch-level vs sliding-window discrepancy
The headline tension in the results is that the HOG models score **0.81–0.83**
accuracy on patches yet behave **erratically** in full-image live detection
(§7.4). This is expected and instructive:
- The patch metrics evaluate the classifier on **tight, well-centred crops** — the
  task it was trained for.
- The sliding window presents the classifier with thousands of **arbitrary,
  loosely-aligned windows** (partial objects, object+background mixtures, multiple
  scales). Out-of-distribution windows can trigger confident wrong predictions,
  and the densest-firing class wins the image. Classical HOG features have no
  mechanism to *localise* — they only classify whatever window they are given.
- YOLO, by contrast, is trained **end-to-end for localisation**: it learns where
  objects are and emits one calibrated box. Hence its clean output.
This gap is itself a key thesis finding: **strong patch-classification accuracy
does not imply strong detection** — the detection protocol matters as much as the
classifier.

### 8.4 Stale-weights trap (engineering reproducibility)
An earlier (pre-fix) YOLOv10 run had written weights to a *nested* legacy path
(`runs/detect/runs/train/...`) due to an Ultralytics `task` quirk. Both the app and
the evaluator preferred that path, so they silently scored a **broken 2-epoch
model** (mAP@50 ≈ 0.09) instead of the freshly trained one. Removing the stale
directory and flipping the path priority to the clean
`runs/train/drone_v10_cpu/...` location raised the measured YOLOv10 mAP@50 from
**0.09 → 0.58**. Lesson: deterministic, unambiguous artifact paths are essential
for trustworthy evaluation.

### 8.5 Classical vs deep — summary judgement
- **Best patch classifier:** HOG + SVM (acc 0.827).
- **Best practical detector:** YOLOv8n (cleanest, most consistent full-image
  results; mAP@50 0.601).
- **Most modern architecture:** YOLOv10n (NMS-free; competitive but slightly
  behind v8n at this short training budget).
- **Classical methods** remain useful as **interpretable baselines** and are cheap
  to train, but they do not localise and are brittle under the sliding-window
  protocol. Deep detectors are markedly superior for the end-to-end task.

---

## 9. Limitations

1. **CPU-only, short training budget.** YOLO models trained for only 10 epochs and
   were still improving; mAP would rise substantially with a GPU and more epochs.
2. **Class imbalance.** Drones outnumber birds ~1.5:1; combined with the historical
   label corruption, the bird class remains the harder one (lower recall).
3. **Cross-paradigm metric mismatch.** Classical accuracy (patch) and YOLO mAP
   (detection) cannot be averaged or ranked on a single scale (§7.1).
4. **Sliding-window cost and noise.** The HOG detector is slow (seconds per image)
   and produces many redundant boxes; it is a baseline, not a production detector.
5. **Annotation heterogeneity.** Mixed bbox/polygon labels; Ultralytics discards
   the polygon segments and uses bounding boxes only.

---

## 10. Future Work

- **GPU training** with longer schedules (100–300 epochs) and early stopping.
- **Larger backbones** (YOLOv8s/m, YOLOv10s/m) for an accuracy/speed trade-off
  study.
- **Bird-class rebalancing** via targeted augmentation or additional bird imagery.
- **Replace the HOG sliding window** with a proper region proposal, or frame the
  classical baseline as *classification-only* on YOLO crops for a fairer
  comparison.
- **mAP@50–95 and per-class AP** reporting for the classical pipeline using
  `compute_iou_metrics` for a fully aligned detection comparison.
- **Confidence calibration** study (reliability diagrams) across all four models.

---

## 11. Reproducibility

### 11.1 One-time data fix
```bash
python scripts/relabel_by_filename.py          # relabel both datasets (+backup)
# caches are rebuilt automatically on next load
```

### 11.2 Train
```bash
# Classical (fast):
python scripts/train_hog.py --max-train 1500 --max-test 400 --mlp-epochs 40

# Deep (CPU, ~hours):
python scripts/train_yolov8.py  --merged --epochs 10 --device cpu
python scripts/train_yolov10.py --merged --epochs 10 --device cpu --batch 8 \
       --model yolov10n.pt --name drone_v10_cpu
```
Or, from the UI: **`streamlit run app.py` → "Train All Models"** (one click runs
HOG synchronously, then launches both YOLO trainings in the background).

### 11.3 Evaluate and inspect
```bash
python scripts/evaluate_all.py                 # writes results/all_metrics.json
streamlit run app.py                           # Results tab = comparison table
                                               # Live Detection tab = 4-way demo
```

### 11.4 Key artifacts
| Artifact | Path |
|----------|------|
| HOG models | `models/hog_mlp.pkl`, `models/hog_svm.pkl` |
| YOLOv8 weights | `runs/train/drone_v8/weights/best.pt` |
| YOLOv10 weights | `runs/train/drone_v10_cpu/weights/best.pt` |
| Comparison metrics | `results/all_metrics.json` |
| Label backup | `labels_backup.zip` |
| Training logs | `logs/train_yolov8.log`, `logs/train_yolov10.log` |

---

## 12. Conclusion

This project delivered a working four-way comparison of bird-vs-drone detectors
spanning classical (HOG+MLP, HOG+SVM) and deep (YOLOv8n, YOLOv10n) approaches,
integrated into a single interactive application.

The decisive result was not an architectural one but a **data-quality** one:
correcting a systematic label corruption (≈99 % of birds mislabelled as drones)
restored functionality across every model — a concrete demonstration that data
integrity is a precondition for, and often more impactful than, model selection.

On the corrected data, the **classical SVM** achieved the best patch-level accuracy
(0.827), while **YOLOv8n** delivered the most reliable end-to-end detection
(mAP@50 0.601, clean single-box predictions). The **patch-vs-detection performance
gap** observed for the HOG pipeline highlights that classification accuracy and
detection quality are distinct properties — the evaluation protocol is as
important as the model. With GPU training and longer schedules, the deep detectors
are expected to widen their lead substantially.

---

*Document generated from the project's source code, training logs, and evaluation
outputs. All metrics are reproducible via the commands in §11.*
