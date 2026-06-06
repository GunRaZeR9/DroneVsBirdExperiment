"""
yolov10_detector.py — Phase 3 of FIX_PLAN
==========================================
YOLOv10 inference module replacing the legacy HOG+MLP sliding-window detector.

Key differences from the old detector.py:
  - No sliding window. No image pyramid. No HOG feature extraction.
  - NMS-free by architecture — exactly 1 box per object, no duplicate suppression needed.
  - No iou_thres tuning. Only ``conf`` threshold matters.
  - 10–100× faster inference per frame.
  - Uses Ultralytics API for one-line model loading.

Class mapping (verified from actual label files):
  0 = drone
  1 = bird

Usage:
    from src.core.yolov10_detector import YOLOv10Detector

    detector = YOLOv10Detector("runs/train/drone_v10/weights/best.pt")
    detections = detector.detect(image)   # image: BGR np.ndarray (cv2 convention)
    for x, y, w, h, conf, label in detections:
        print(f"{label} @ ({x},{y}) size {w}x{h} conf={conf:.2f}")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias — matches the return type of the legacy detector for drop-in compatibility
Detection = Tuple[int, int, int, int, float, str]   # x, y, w, h, conf, label


class YOLOv10Detector:
    """
    Drop-in replacement for the legacy ``detect()`` function in detector.py.

    Parameters
    ----------
    weights : str | Path
        Path to the YOLOv10 best.pt checkpoint produced by training.
    conf : float
        Confidence threshold (default 0.45). Only tune this — no iou_thres needed.
    device : str
        PyTorch device string: '0' (first GPU), 'cpu', 'mps' (Apple Silicon).
    """

    CLASS_NAMES = {0: "drone", 1: "bird"}

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.45,
        device: str = "0",
    ) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self.device = device
        self._model = None

        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        conf: float | None = None,
    ) -> List[Detection]:
        """
        Run YOLOv10 inference on a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image as returned by ``cv2.imread`` or ``cv2.VideoCapture.read``.
        conf : float | None
            Override the instance confidence threshold for this call only.

        Returns
        -------
        list of (x, y, w, h, confidence, label_str)
            * Coordinates are in original image pixel space.
            * Exactly one box per detected object (NMS-free architecture).
            * Returns [] when no objects detected above threshold.
        """
        if self._model is None:
            logger.warning("detect() called but model is not loaded.")
            return []

        threshold = conf if conf is not None else self.conf

        results = self._model.predict(
            source=image,
            conf=threshold,
            verbose=False,
            # No iou parameter needed — YOLOv10 is NMS-free
        )

        detections: List[Detection] = []

        if not results:
            return detections

        result = results[0]   # single image → single result

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            # xyxy format → convert to (x, y, w, h) for legacy compatibility
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            w = x2 - x1
            h = y2 - y1
            conf_val = float(boxes.conf[i])
            cls_id   = int(boxes.cls[i])
            label    = self.CLASS_NAMES.get(cls_id, f"class{cls_id}")

            detections.append((x1, y1, w, h, conf_val, label))

        return detections

    def detect_batch(
        self,
        images: List[np.ndarray],
        conf: float | None = None,
    ) -> List[List[Detection]]:
        """
        Run YOLOv10 inference on a batch of BGR images.

        Parameters
        ----------
        images : list of np.ndarray
            BGR images (need not be same size).
        conf : float | None
            Override confidence threshold.

        Returns
        -------
        list of detection lists — one per input image.
        """
        if self._model is None:
            return [[] for _ in images]

        threshold = conf if conf is not None else self.conf

        results = self._model.predict(
            source=images,
            conf=threshold,
            verbose=False,
        )

        batch_detections: List[List[Detection]] = []

        for result in results:
            detections: List[Detection] = []
            if result.boxes is not None:
                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = [int(v) for v in result.boxes.xyxy[i].tolist()]
                    w = x2 - x1
                    h = y2 - y1
                    conf_val = float(result.boxes.conf[i])
                    cls_id   = int(result.boxes.cls[i])
                    label    = self.CLASS_NAMES.get(cls_id, f"class{cls_id}")
                    detections.append((x1, y1, w, h, conf_val, label))
            batch_detections.append(detections)

        return batch_detections

    def detect_video(
        self,
        source: str | int,
        conf: float | None = None,
        display: bool = False,
        save_path: str | None = None,
    ) -> None:
        """
        Run detection on a video file or webcam stream.

        Parameters
        ----------
        source : str | int
            Video file path or webcam index (0 = first webcam).
        conf : float | None
            Confidence threshold override.
        display : bool
            Show live annotated frames (requires display).
        save_path : str | None
            If set, write annotated video to this path.
        """
        if self._model is None:
            logger.error("Model not loaded.")
            return

        threshold = conf if conf is not None else self.conf
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error(f"Cannot open video source: {source}")
            return

        writer: cv2.VideoWriter | None = None
        if save_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detect(frame, conf=threshold)
                annotated = self._draw_detections(frame, detections)

                if writer:
                    writer.write(annotated)

                if display:
                    cv2.imshow("YOLOv10 Drone/Bird Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Convenience: annotate image
    # ------------------------------------------------------------------

    def annotate(self, image: np.ndarray, conf: float | None = None) -> np.ndarray:
        """Return a copy of ``image`` with detection boxes drawn."""
        detections = self.detect(image, conf=conf)
        return self._draw_detections(image.copy(), detections)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load YOLOv10 checkpoint via Ultralytics API."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(
                "ultralytics not installed. Run: pip install ultralytics"
            )
            return

        if not self.weights.exists():
            logger.error(
                f"Weights file not found: {self.weights}\n"
                "Train first with: python scripts/train_yolov10.py"
            )
            return

        try:
            self._model = YOLO(str(self.weights))
            # Warm-up inference (suppresses first-call latency)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(source=dummy, conf=0.99, verbose=False)
            logger.info(f"YOLOv10 model loaded from {self.weights}")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            self._model = None

    @staticmethod
    def _draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on image (in-place)."""
        color_map = {
            "drone": (0, 255, 0),    # green
            "bird":  (0, 165, 255),  # orange
        }
        for x, y, w, h, conf, label in detections:
            color = color_map.get(label, (255, 255, 255))
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(
                image, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )
        return image

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"YOLOv10Detector("
            f"weights='{self.weights}', conf={self.conf}, loaded={loaded})"
        )


# ---------------------------------------------------------------------------
# Convenience function — drop-in compatible with legacy detect() signature
# ---------------------------------------------------------------------------

def detect(
    image: np.ndarray,
    model: "YOLOv10Detector",
    confidence_threshold: float = 0.45,
    **_kwargs,
) -> List[Detection]:
    """
    Legacy-compatible wrapper around YOLOv10Detector.detect().

    Accepts and silently ignores ``seq_length`` and ``window_sizes`` kwargs
    that the old HOG-based detector required.  This allows existing call sites
    to switch detectors without code changes.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    model : YOLOv10Detector
        Loaded detector instance.
    confidence_threshold : float
        Confidence threshold (default 0.45).

    Returns
    -------
    list of (x, y, w, h, confidence, label_str)
    """
    return model.detect(image, conf=confidence_threshold)
