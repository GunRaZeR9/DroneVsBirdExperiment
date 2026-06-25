"""
train_hog.py — Train HOG+MLP and HOG+SVM classifiers from the CLI.

Mirrors the "Train All Models" HOG steps in app.py, but headless so models can
be produced/verified without the Streamlit UI. Saves:
    models/hog_mlp.pkl
    models/hog_svm.pkl

Usage:
    python scripts/train_hog.py --max-train 1500 --max-test 400 --mlp-epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.core.data_loader import load_train_test          # noqa: E402
from src.core.feature_extractor import extract_hog_batch  # noqa: E402
from src.core.model import BirdDroneModel                 # noqa: E402
from src.utils.metrics import compute_metrics             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train HOG+MLP and HOG+SVM")
    ap.add_argument("--data-root", default="Data")
    ap.add_argument("--max-train", type=int, default=1500)
    ap.add_argument("--max-test", type=int, default=400)
    ap.add_argument("--mlp-epochs", type=int, default=40)
    args = ap.parse_args()

    print(f"Loading patches from {args.data_root} ...")
    Xtr_raw, ytr, Xv_raw, yv, Xte_raw, yte = load_train_test(
        data_root=str(ROOT / args.data_root),
        max_train=args.max_train,
        max_test=args.max_test,
        window_size=(64, 64),
    )
    print(f"  train dist (Bg/Bird/Drone): {np.bincount(ytr.astype(int), minlength=3)}")
    print(f"  test  dist (Bg/Bird/Drone): {np.bincount(yte.astype(int), minlength=3)}")

    print("Extracting HOG features ...")
    Xtr = extract_hog_batch(Xtr_raw)
    Xte = extract_hog_batch(Xte_raw)

    (ROOT / "models").mkdir(exist_ok=True)

    # ── MLP ──────────────────────────────────────────────────────────────────
    print("\nTraining HOG+MLP ...")
    mlp = BirdDroneModel(mode="MLP", epochs=args.mlp_epochs, lr=0.001,
                         num_layers=3, hidden_size=128, batch_size=32)
    mlp.train(Xtr, ytr)
    mlp_metrics = compute_metrics(yte, mlp.predict(Xte))
    mlp.save(str(ROOT / "models" / "hog_mlp.pkl"))
    print(f"  MLP  acc={mlp_metrics['accuracy']:.4f}  f1={mlp_metrics['f1']:.4f}  "
          f"prec={mlp_metrics['precision']:.4f}  rec={mlp_metrics['recall']:.4f}")
    print(f"  confusion (rows=true Bg/Bird/Drone):\n{mlp_metrics['confusion_matrix']}")

    # ── SVM ──────────────────────────────────────────────────────────────────
    print("\nTraining HOG+SVM ...")
    svm = BirdDroneModel(mode="SVM", epochs=1, lr=0.001,
                         num_layers=1, hidden_size=128, batch_size=32)
    svm.train(Xtr, ytr)
    svm_metrics = compute_metrics(yte, svm.predict(Xte))
    svm.save(str(ROOT / "models" / "hog_svm.pkl"))
    print(f"  SVM  acc={svm_metrics['accuracy']:.4f}  f1={svm_metrics['f1']:.4f}  "
          f"prec={svm_metrics['precision']:.4f}  rec={svm_metrics['recall']:.4f}")
    print(f"  confusion (rows=true Bg/Bird/Drone):\n{svm_metrics['confusion_matrix']}")

    print("\nSaved: models/hog_mlp.pkl, models/hog_svm.pkl")


if __name__ == "__main__":
    main()
