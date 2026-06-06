# Bird vs Drone Detector

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

Desktop application that classifies aerial objects as **birds** or **drones** using
classical computer-vision features (HOG) fed into either a **Support Vector Machine** or
a **Multi-Layer Perceptron** trained entirely on-device — no cloud, no GPU required.

---

## What it does

| Stage | Detail |
|---|---|
| Feature extraction | Histogram of Oriented Gradients (HOG) via `scikit-image` |
| Classifier | SVM (RBF / linear kernel) **or** MLP with configurable hidden layers |
| Detection | Sliding-window scan with multi-scale pyramid + Non-Maximum Suppression |
| GUI | PyQt5 desktop app — train, evaluate, and run live detection in one window |

The full pipeline is a `scikit-learn` `Pipeline` object (scaler → PCA → classifier),
serialised with `joblib` so models are portable across machines.

---

## Requirements

- Python **3.10** or newer
- All Python dependencies listed in `requirements.txt`

```
pip install -r requirements.txt
```

On Windows, if OpenCV installation fails, try:

```
pip install opencv-python-headless>=4.8.0
```

---

## Data setup

The app expects images and labels in **YOLO format** (one `.txt` per image, bounding
boxes as `class cx cy w h` in normalised coordinates, class `0` = bird, class `1` = drone).

Arrange your dataset like this:

```
Data/
  train/
    images/       ← .jpg / .png training images
    labels/       ← matching .txt YOLO label files
  test/
    images/
    labels/
  valid/          ← optional validation split
    images/
    labels/
```

The trainer auto-crops each labelled bounding box, resizes to the HOG window size, and
builds positive/negative samples.  No manual pre-processing required.

**Recommended source:** Roboflow *Drone, Birds* dataset (CC BY 4.0) — see Dataset Credit
section below.  Export in YOLO format and drop the unzipped folder as `Data/` in the
project root.

---

## How to run

```
python main.py
```

The GUI opens immediately.  No CLI flags needed.

---

## GUI usage guide

### Train tab (left panel)

| Control | Purpose |
|---|---|
| **Classifier** combo | Choose `SVM` or `MLP` |
| **Kernel** (SVM only) | `rbf` gives best accuracy; `linear` trains faster |
| **C** | SVM regularisation — higher = tighter fit, risk of overfit |
| **HOG orientations** | Number of gradient direction bins (6–12 recommended) |
| **HOG pixels/cell** | Cell size in pixels — smaller = more detail, slower |
| **HOG cells/block** | Block normalisation window (2 is standard) |
| **Window size** | Sliding-window patch size in pixels (e.g. 64) |
| **PCA components** | Dimensions kept after PCA whitening; 0 = disabled |
| **Test split** | Fraction of data held out for evaluation |
| **Train** button | Starts background training thread — progress bar updates live |
| **Save model** | Serialises trained pipeline to a `.joblib` file |
| **Load model** | Restores a previously saved `.joblib` pipeline |

Progress and log messages appear in the log area at the bottom of the left panel.

### Results tab (right panel, auto-selected after training)

- **Metrics card** — accuracy, precision, recall, F1 per class
- **Learning curves** — training vs validation accuracy / loss by epoch (MLP) or
  cross-validation fold score (SVM)
- **Confusion matrix** — colour-coded heat-map of true vs predicted labels

### Live Test tab

1. Click **Open Image** and choose a `.jpg` or `.png` file.
2. Adjust **Confidence threshold** (0–100 %) — detections below this score are suppressed.
3. Click **Detect** — the sliding-window scan runs in a background thread.
4. Bounding boxes are drawn on the image; each box is labelled *Bird* or *Drone* with
   the confidence score.

---

## Architecture overview

```
Image
  └─► HOG feature vector  (scikit-image.feature.hog)
        └─► StandardScaler
              └─► PCA  (optional dimensionality reduction)
                    └─► SVM (sklearn.svm.SVC)
                        or MLP (sklearn.neural_network.MLPClassifier)
                              └─► class label + decision score

Live detection:
  Image pyramid  →  sliding window  →  HOG  →  pipeline.predict_proba
                                                   └─► NMS  →  bounding boxes
```

All heavy computation runs in `QThread` workers so the GUI stays responsive.

---

## Slider / spin-box descriptions

| Parameter | Range | Default | Effect |
|---|---|---|---|
| HOG orientations | 6 – 18 | 9 | More bins capture finer gradient directions; diminishing returns above 12 |
| HOG pixels/cell | 4 – 16 | 8 | Smaller cells = richer local texture; also larger feature vector |
| HOG cells/block | 1 – 4 | 2 | Bigger block normalisation reduces lighting sensitivity |
| Window size | 32 – 128 | 64 | Must be divisible by `pixels/cell × cells/block` |
| Stride | 4 – 32 | 8 | Smaller stride = denser scan = slower but fewer missed detections |
| Scale factor | 1.05 – 1.5 | 1.25 | Pyramid downscale ratio; smaller = more scales checked |
| NMS IoU threshold | 0.1 – 0.9 | 0.3 | Higher = more overlapping boxes kept |
| Confidence threshold | 0 – 100 % | 50 % | Minimum classifier probability to display a detection |
| PCA components | 0 – 512 | 128 | 0 disables PCA; tune to balance speed vs accuracy |
| Test split | 0.10 – 0.40 | 0.20 | Fraction of labelled data withheld for evaluation |

---

## Dataset credit

**Drone, Birds** dataset assembled and hosted on
[Roboflow Universe](https://roboflow.com) by the Roboflow community.

> License: **CC BY 4.0** — attribution required.
> Original images sourced from publicly available aerial photography datasets.

If you use this dataset in academic work, cite the Roboflow dataset page:

```
@dataset{drone-birds,
  title  = {Drone, Birds Dataset},
  author = {Roboflow Universe Contributors},
  year   = {2023},
  url    = {https://universe.roboflow.com},
  license= {CC BY 4.0}
}
```

---

## Project structure

```
LiceentaBogdan/
  main.py                        ← entry point
  requirements.txt
  README.md
  Data/                          ← dataset (not tracked by git)
  models/                        ← saved .joblib files
  src/
    core/
      model.py                   ← BirdDroneModel (pipeline wrapper)
      feature_extractor.py       ← HOG extraction helpers
      detector.py                ← sliding window + NMS
    gui/
      main_window.py             ← QMainWindow orchestrator
      panels/
        params_panel.py          ← left sidebar (hyperparameters + log)
        results_panel.py         ← metrics, curves, confusion matrix
        live_test_panel.py       ← image viewer + detection overlay
      workers/
        train_worker.py          ← QThread — data loading + training
        test_worker.py           ← QThread — sliding-window detection
```

---

## License

MIT — see `LICENSE` for details.
