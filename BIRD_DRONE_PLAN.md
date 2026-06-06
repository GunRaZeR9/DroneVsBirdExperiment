# 🦅 Bird vs Drone Detector — Full Project Plan

> **Architecture:** HOG Feature Extraction → sklearn Pipeline (SVM / MLP) → Sliding Window + NMS Detection  
> **GUI:** PyQt5 — Left params panel, Center Results tab, Live Test tab  
> **Runtime:** CPU-only | No pretrained weights | Fully trainable from scratch

---

## 1. Architecture Decision (Research-Backed)

### Why HOG + sklearn SVM/MLP?

| Criterion | HOG+SVM | YOLO (from scratch) | HOG+MLP |
|---|---|---|---|
| CPU-only viable | ✅ | ❌ very slow | ✅ |
| Fully trainable from scratch | ✅ | ✅ but complex | ✅ |
| sklearn-native | ✅ | ❌ | ✅ |
| Bounding boxes | ✅ sliding window+NMS | ✅ native | ✅ sliding window+NMS |
| GUI sliders apply | SVM: partial | ❌ | ✅ all sliders |

**Chosen:** **Dual-model pipeline** — user picks SVM **or** MLP from the GUI:
- **SVM mode:** HOG → `sklearn.svm.LinearSVC` (fast, high accuracy)
- **MLP mode:** HOG → `sklearn.neural_network.MLPClassifier` (exposes all sliders: epochs, LR, layers, hidden size, batch size)

**Detection loop:** Image pyramid + sliding window → HOG features → classifier → NMS → bounding boxes  
**Sequence Length slider:** Controls the image pyramid window size (meaningful for small/far objects like drones)

---

## 2. Folder Structure

```
BirdDroneDetector/
│
├── Data/                          # ← YOUR EXISTING DATA FOLDER
│   ├── train/
│   │   ├── bird/                  # bird training images
│   │   └── drone/                 # drone training images
│   └── test/
│       ├── bird/
│       └── drone/
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Load images, parse annotations, train/test split
│   │   ├── feature_extractor.py   # HOG extraction (skimage), normalization
│   │   ├── model.py               # sklearn SVM + MLP wrapper, save/load
│   │   └── detector.py            # Sliding window, image pyramid, NMS
│   │
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py         # Root PyQt5 QMainWindow, tab orchestration
│   │   ├── panels/
│   │   │   ├── params_panel.py    # LEFT: all sliders, model selector, Train/Test buttons
│   │   │   ├── results_panel.py   # CENTER TAB 1: metrics, confusion matrix, loss curve
│   │   │   └── live_test_panel.py # CENTER TAB 2: image upload, bounding box render
│   │   └── workers/
│   │       ├── train_worker.py    # QThread: runs training without freezing GUI
│   │       └── test_worker.py     # QThread: runs inference without freezing GUI
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py             # Accuracy, F1, confusion matrix, IoU
│       ├── visualizer.py          # Draw bounding boxes + confidence on images
│       └── logger.py             # Training log to file + GUI console
│
├── models/                        # Saved .pkl model files
│   └── .gitkeep
│
├── outputs/                       # Processed/annotated output images
│   └── .gitkeep
│
├── logs/                          # Training run logs (.txt)
│   └── .gitkeep
│
├── requirements.txt
├── main.py                        # Entry point
└── README.md
```

---

## 3. GUI Layout (PyQt5)

```
┌─────────────────────────────────────────────────────────────────┐
│  🦅 Bird vs Drone Detector                            [─][□][✕] │
├──────────────────┬──────────────────────────────────────────────┤
│  PARAMETERS      │  [  Results Tab  ] [  Live Test Tab  ]       │
│  ─────────────── │  ────────────────────────────────────────    │
│  Model:          │                                              │
│  ○ SVM  ● MLP    │  RESULTS TAB:                                │
│                  │  ┌──────────────┐  ┌──────────────────────┐ │
│  Epochs     [50] │  │ Accuracy     │  │  Loss / Accuracy     │ │
│  ━━━━━━━━━━━━━━  │  │ F1 Score     │  │  curve (matplotlib   │ │
│  Learning Rate   │  │ Precision    │  │  embedded plot)      │ │
│  [0.001]━━━━━━━  │  │ Recall       │  │                      │ │
│                  │  │ Confusion    │  │                      │ │
│  Num Layers  [3] │  │ Matrix       │  └──────────────────────┘ │
│  ━━━━━━━━━━━━━━  │  └──────────────┘                           │
│  Hidden Size[128]│  ─────────────────────────────────────────  │
│  ━━━━━━━━━━━━━━  │                                              │
│  Seq Length  [8] │  LIVE TEST TAB:                              │
│  ━━━━━━━━━━━━━━  │  [ Upload Image ]   [ Run Detection ]        │
│  Batch Size [32] │  ┌────────────────────────────────────────┐  │
│  ━━━━━━━━━━━━━━  │  │                                        │  │
│                  │  │   [Uploaded image with bounding boxes] │  │
│  [ 🚀 TRAIN ]    │  │   ┌─────────────┐                      │  │
│  [ 🧪 TEST  ]    │  │   │ BIRD  0.94  │                      │  │
│  [ 💾 SAVE  ]    │  │   └─────────────┘                      │  │
│  [ 📂 LOAD  ]    │  │                                        │  │
│                  │  └────────────────────────────────────────┘  │
│  ── Log ──────   │  Class: BIRD | Confidence: 94.2%             │
│  [console log]   │  Bounding Box: (x=120, y=45, w=80, h=60)     │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## 4. Module Breakdown

### `src/core/data_loader.py`
- Scans `Data/train/bird`, `Data/train/drone` folders
- Reads images (JPG/PNG), resizes to fixed window size
- Returns `X_train, y_train, X_test, y_test`
- Handles label encoding (bird=0, drone=1)

### `src/core/feature_extractor.py`
- `extract_hog(image)` → feature vector using `skimage.feature.hog`
- Parameters: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)
- Returns normalized feature vector

### `src/core/model.py`
- `BirdDroneModel` class wrapping sklearn:
  - Mode `SVM`: `sklearn.svm.LinearSVC`
  - Mode `MLP`: `sklearn.neural_network.MLPClassifier`
    - `max_iter` ← Epochs slider
    - `learning_rate_init` ← LR slider
    - `hidden_layer_sizes` ← (HiddenSize,) × NumLayers
    - `batch_size` ← Batch Size slider
- `train(X, y)`, `predict(X)`, `save(path)`, `load(path)` methods
- Emits progress signals for GUI progress bar

### `src/core/detector.py`
- `detect(image, model, extractor)` → list of `(x, y, w, h, confidence, label)`
- Image pyramid: scales image down iteratively
- Sliding window: steps across each pyramid level
- HOG extracted per window → model prediction + confidence
- NMS: `imutils.object_detection.non_max_suppression` or manual implementation
- Sequence Length slider → controls window stride / pyramid scale step

### `src/gui/workers/train_worker.py`
- `QThread` subclass — runs `model.train()` in background
- Emits: `progress(int)`, `log(str)`, `finished(dict)` signals
- Prevents GUI freezing during long training

### `src/utils/visualizer.py`
- `draw_boxes(image, detections)` → annotated PIL/numpy image
- Color: green for bird, red for drone
- Draws label + confidence score above each box

---

## 5. Sklearn Slider Mapping

| GUI Slider | Parameter | Applies to |
|---|---|---|
| Epochs | `MLPClassifier.max_iter` | MLP only |
| Learning Rate | `MLPClassifier.learning_rate_init` | MLP only |
| Number of Layers | `len(hidden_layer_sizes)` | MLP only |
| Hidden Size | `hidden_layer_sizes[i]` value | MLP only |
| Sequence Length | Window stride / pyramid steps | Both (detection loop) |
| Batch Size | `MLPClassifier.batch_size` | MLP only |

> **Note:** SVM sliders (Epochs, LR, Layers, Hidden Size, Batch) are grayed out when SVM is selected. Sequence Length remains active for both.

---

## 6. Technology Stack

```
Python 3.10+
├── scikit-learn        # SVM, MLP, metrics
├── scikit-image        # HOG feature extraction
├── opencv-python       # Image I/O, sliding window, drawing
├── PyQt5               # GUI framework
├── matplotlib          # Embedded training curves
├── numpy               # Array ops
├── Pillow              # Image handling in GUI
├── joblib              # Model save/load (.pkl)
└── imutils             # NMS utility (optional)
```

---

## 7. Implementation Phases

### Phase 1 — Project Scaffold (Day 1)
- [ ] Create full folder structure
- [ ] `requirements.txt`
- [ ] `main.py` entry point
- [ ] Skeleton classes for all modules

### Phase 2 — Core ML Pipeline (Days 2–3)
- [ ] `data_loader.py` — scan Data folder, load images, split
- [ ] `feature_extractor.py` — HOG pipeline
- [ ] `model.py` — SVM + MLP wrapper
- [ ] CLI test: train and evaluate without GUI

### Phase 3 — Detection Engine (Days 3–4)
- [ ] `detector.py` — sliding window + pyramid
- [ ] NMS implementation
- [ ] `visualizer.py` — bounding box drawing
- [ ] Test on sample images

### Phase 4 — GUI (Days 5–7)
- [ ] `main_window.py` — layout skeleton
- [ ] `params_panel.py` — all sliders + buttons
- [ ] `results_panel.py` — metrics display + matplotlib plot
- [ ] `live_test_panel.py` — image upload + annotated result
- [ ] `train_worker.py` + `test_worker.py` — background threads

### Phase 5 — Integration & Polish (Days 8–9)
- [ ] Wire GUI ↔ core pipeline
- [ ] Model save/load from GUI
- [ ] Log window in params panel
- [ ] Error handling + input validation

### Phase 6 — Testing (Day 10)
- [ ] Train on full dataset
- [ ] Evaluate metrics
- [ ] Live test with unseen images
- [ ] Fix edge cases

---

## 8. Data Structure Expected

The Kaggle dataset (stealthknight/bird-vs-drone) should be placed as:

```
Data/
├── train/
│   ├── bird/     ← bird training images (JPG/PNG)
│   └── drone/    ← drone training images (JPG/PNG)
└── test/
    ├── bird/
    └── drone/
```

If annotations (CSV/XML/YOLO TXT) are included, `data_loader.py` will parse them for ground-truth bounding boxes used in IoU evaluation. If no annotations exist, the model trains as a **classifier** and detection is done via sliding window.

---

## 9. Key Metrics Displayed

- **Training:** Accuracy per epoch (MLP), final accuracy (SVM), training time
- **Testing:** Accuracy, F1, Precision, Recall, Confusion Matrix
- **Detection:** Confidence score, bounding box coordinates, IoU vs ground truth (if annotations available)

---

## 10. Files to Build (Priority Order)

```
1. requirements.txt
2. main.py
3. src/core/data_loader.py
4. src/core/feature_extractor.py
5. src/core/model.py
6. src/core/detector.py
7. src/utils/visualizer.py
8. src/utils/metrics.py
9. src/gui/main_window.py
10. src/gui/panels/params_panel.py
11. src/gui/panels/results_panel.py
12. src/gui/panels/live_test_panel.py
13. src/gui/workers/train_worker.py
14. src/gui/workers/test_worker.py
15. README.md
```

---

*Plan version 1.0 — Ready for implementation. Say "build phase 1" to start scaffolding.*
