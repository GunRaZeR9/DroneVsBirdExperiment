"""
BirdDroneModel — sklearn-based classifier with epoch-by-epoch tracking.

Supports two modes:
  - 'MLP' : MLPClassifier with SMOTE balancing + partial-fit loop
  - 'SVM' : SVC(rbf, probability=True) with balanced class weights

Supports 3-class classification:
  - 0 = Background
  - 1 = Bird
  - 2 = Drone
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# Fixed class set expected by the 3-class model
_ALL_CLASSES = np.array([0, 1, 2])


class BirdDroneModel:
    """3-class classifier: Background (0), Bird (1), Drone (2).

    Parameters
    ----------
    mode : str
        'MLP' or 'SVM'.
    epochs : int
        Number of training epochs (MLP only; SVM ignores this).
    lr : float
        Learning rate (MLP only).
    num_layers : int
        Number of hidden layers (MLP only).
    hidden_size : int
        Number of units per hidden layer (MLP only).
    batch_size : int
        Mini-batch size for MLP partial_fit loop.
    """

    def __init__(
        self,
        mode: str = "MLP",
        epochs: int = 50,
        lr: float = 0.001,
        num_layers: int = 3,
        hidden_size: int = 128,
        batch_size: int = 32,
    ) -> None:
        self.mode = mode.upper()
        self.epochs = epochs
        self.lr = lr
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.is_trained: bool = False
        self.training_history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }
        self.class_names: list[str] = ["Background", "Bird", "Drone"]

        # Set after train()
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._clf = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, List[float]]:
        """Fit scaler + classifier on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape [N, F]
        y : np.ndarray, shape [N]
        progress_callback : callable(percent: float) | None
        log_callback : callable(message: str) | None

        Returns
        -------
        dict
            self.training_history
        """
        # Reset history each call
        self.training_history = {"loss": [], "accuracy": [], "val_accuracy": []}

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            warnings.warn(
                f"Training data contains only class(es) {unique_classes}. "
                "Model will predict a single class for all inputs.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Single-class training data detected: %s", unique_classes
            )

        # Log class distribution
        counts = np.bincount(y.astype(int))
        self._emit_log(
            log_callback,
            f"Classes in training data: {unique_classes}  counts: {counts}",
        )

        # Fit scaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # PCA: reduce HOG features (1764-dim) → 256 dims for better class separation
        _n_components = min(256, X_scaled.shape[1], X_scaled.shape[0] - 1)
        self._pca = PCA(n_components=_n_components, random_state=42)
        X_scaled = self._pca.fit_transform(X_scaled)
        self._emit_log(
            log_callback,
            f"PCA: {_n_components} components (from {X.shape[1]} HOG dims, {X_scaled.shape[0]} samples)",
        )

        if self.mode == "SVM":
            self._train_svm(X_scaled, y, unique_classes, progress_callback, log_callback)
        else:
            self._train_mlp(X_scaled, y, unique_classes, progress_callback, log_callback)

        self.is_trained = True
        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels.

        Parameters
        ----------
        X : np.ndarray, shape [N, F]

        Returns
        -------
        np.ndarray, shape [N]
        """
        self._require_trained()
        X_scaled = self._scaler.transform(X)
        if self._pca is not None:
            X_scaled = self._pca.transform(X_scaled)
        return self._clf.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return object confidence score per sample.

        For binary models (2 classes): returns P(class=1).
        For 3-class models: returns 1 - P(class=0), i.e. the probability of
        being ANY object (Bird or Drone), used for confidence thresholding.

        Parameters
        ----------
        X : np.ndarray, shape [N, F]

        Returns
        -------
        np.ndarray, shape [N]  (probabilities in [0, 1])
        """
        self._require_trained()
        X_scaled = self._scaler.transform(X)
        if self._pca is not None:
            X_scaled = self._pca.transform(X_scaled)
        proba = self._clf.predict_proba(X_scaled)
        # proba has shape [N, n_classes]
        if proba.shape[1] == 1:
            # Single-class degenerate case — return the only column
            return proba[:, 0]
        if proba.shape[1] == 2:
            # Binary model: column 1 is positive class
            return proba[:, 1]
        # 3-class model: return P(any object) = 1 - P(background)
        return 1.0 - proba[:, 0]

    def predict_proba_all(self, X: np.ndarray) -> np.ndarray:
        """Return full probability matrix for all classes.

        Parameters
        ----------
        X : np.ndarray, shape [N, F]

        Returns
        -------
        np.ndarray, shape [N, n_classes]
            Column order matches the classifier's internal class ordering
            (Background=0, Bird=1, Drone=2 for the 3-class model).
        """
        self._require_trained()
        X_scaled = self._scaler.transform(X)
        if self._pca is not None:
            X_scaled = self._pca.transform(X_scaled)
        return self._clf.predict_proba(X_scaled)

    def save(self, path: str) -> None:
        """Persist model to disk via joblib.

        Parameters
        ----------
        path : str
            File path (e.g. 'model.pkl').
        """
        payload = {
            "scaler": self._scaler,
            "pca": self._pca,
            "clf": self._clf,
            "mode": self.mode,
            "params": self.get_params(),
            "history": self.training_history,
            "class_names": self.class_names,
        }
        joblib.dump(payload, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Restore model from disk.

        Parameters
        ----------
        path : str
            File path written by :meth:`save`.
        """
        payload = joblib.load(path)
        self._scaler = payload["scaler"]
        self._pca = payload.get("pca", None)
        self._clf = payload["clf"]
        self.mode = payload["mode"]
        self.training_history = payload.get("history", self.training_history)
        self.class_names = payload.get("class_names", ["Background", "Bird", "Drone"])

        params = payload.get("params", {})
        self.epochs = params.get("epochs", self.epochs)
        self.lr = params.get("lr", self.lr)
        self.num_layers = params.get("num_layers", self.num_layers)
        self.hidden_size = params.get("hidden_size", self.hidden_size)
        self.batch_size = params.get("batch_size", self.batch_size)

        self.is_trained = True
        logger.info("Model loaded from %s", path)

    def get_params(self) -> Dict:
        """Return current hyper-parameters as a plain dict."""
        return {
            "mode": self.mode,
            "epochs": self.epochs,
            "lr": self.lr,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "batch_size": self.batch_size,
        }

    def get_feature_dim(self) -> Optional[int]:
        """Return expected input feature dimension, or None if not trained."""
        if self._scaler is None:
            return None
        return self._scaler.n_features_in_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if not self.is_trained or self._clf is None:
            raise RuntimeError(
                "Model is not trained yet. Call train() or load() first."
            )

    def _emit_progress(
        self, callback: Optional[Callable[[float], None]], percent: float
    ) -> None:
        if callback is not None:
            try:
                callback(float(percent))
            except Exception:
                pass

    def _emit_log(
        self, callback: Optional[Callable[[str], None]], message: str
    ) -> None:
        logger.debug(message)
        if callback is not None:
            try:
                callback(message)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Mode-specific trainers
    # ------------------------------------------------------------------

    def _train_mlp(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        unique_classes: np.ndarray,
        progress_callback: Optional[Callable],
        log_callback: Optional[Callable],
    ) -> None:
        """Epoch-by-epoch MLP training with SMOTE balancing + partial_fit."""
        hidden_layers = tuple(self.hidden_size for _ in range(self.num_layers))

        # Apply SMOTE to balance all 3 classes before training
        class_counts = np.bincount(y.astype(int), minlength=len(_ALL_CLASSES))
        present_counts = class_counts[class_counts > 0]
        min_class_count = int(np.min(present_counts)) if present_counts.size else 0
        self._emit_log(
            log_callback,
            f"Applying SMOTE: {dict(zip(*np.unique(y, return_counts=True)))} →",
        )
        if min_class_count < 2:
            X_bal, y_bal = X_scaled, y
            self._emit_log(
                log_callback,
                "  skipping SMOTE: not enough samples per class",
            )
        else:
            sm = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
            X_bal, y_bal = sm.fit_resample(X_scaled, y)
            self._emit_log(
                log_callback,
                f"  balanced: {dict(zip(*np.unique(y_bal, return_counts=True)))}",
            )

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            learning_rate_init=self.lr,
            learning_rate="adaptive",
            # warm_start removed — incompatible with partial_fit when a shuffled
            # epoch batch happens to miss one class (e.g. rare Drone samples).
            # partial_fit is already incremental; warm_start is not needed.
            max_iter=1,
            random_state=42,
            batch_size=min(self.batch_size, len(y_bal)),
        )

        n_samples = X_bal.shape[0]

        for epoch in range(1, self.epochs + 1):
            perm = np.random.permutation(n_samples)
            X_ep = X_bal[perm]
            y_ep = y_bal[perm]

            # Always pass all 3 class labels so the MLP reserves 3 output nodes,
            # even when a class is absent from this batch.
            clf.partial_fit(X_ep, y_ep, classes=_ALL_CLASSES)

            loss = getattr(clf, "loss_", float("nan"))
            acc  = clf.score(X_bal, y_bal)

            self.training_history["loss"].append(float(loss))
            self.training_history["accuracy"].append(float(acc))
            self.training_history["val_accuracy"].append(float(acc))

            percent = epoch / self.epochs * 100.0
            self._emit_progress(progress_callback, percent)
            self._emit_log(
                log_callback,
                f"Epoch {epoch}/{self.epochs} loss={loss:.4f} acc={acc:.4f}",
            )

        self._clf = clf

    def _train_svm(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        unique_classes: np.ndarray,
        progress_callback: Optional[Callable],
        log_callback: Optional[Callable],
    ) -> None:
        """SVM training — SVC(rbf) with SMOTE + Platt scaling for real probabilities."""
        # Apply SMOTE to balance drone vs bird vs background before fitting
        class_counts = np.bincount(y.astype(int), minlength=len(_ALL_CLASSES))
        present_counts = class_counts[class_counts > 0]
        min_class_count = int(np.min(present_counts)) if present_counts.size else 0
        self._emit_log(
            log_callback,
            f"Applying SMOTE (SVM): {dict(zip(*np.unique(y, return_counts=True)))} →",
        )
        if min_class_count < 2:
            X_bal, y_bal = X_scaled, y
            self._emit_log(log_callback, "  skipping SMOTE: not enough samples per class")
        else:
            sm = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
            X_bal, y_bal = sm.fit_resample(X_scaled, y)
            self._emit_log(
                log_callback,
                f"  balanced: {dict(zip(*np.unique(y_bal, return_counts=True)))}",
            )

        self._emit_log(
            log_callback,
            "Training 3-class SVC(rbf, C=10) — class_weight='balanced' …",
        )

        clf = SVC(
            kernel='rbf',
            C=10.0,
            class_weight='balanced',
            probability=True,
            random_state=42,
        )
        clf.fit(X_bal, y_bal)

        # SVM has no epoch loop; fill history with a single entry
        acc = clf.score(X_bal, y_bal)
        self.training_history["loss"].append(float("nan"))
        self.training_history["accuracy"].append(float(acc))
        self.training_history["val_accuracy"].append(float(acc))

        self._emit_progress(progress_callback, 100.0)
        self._emit_log(log_callback, f"SVM training complete. Train acc={acc:.4f}")

        self._clf = clf
