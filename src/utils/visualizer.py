"""
visualizer.py — Drawing utilities for Bird vs Drone Detector.

Provides OpenCV-based bounding-box rendering, PIL conversion helpers,
and a PyQt5 QPixmap converter for GUI display.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
# Each detection is: (x, y, w, h, confidence, label_str)
Detection = Tuple[int, int, int, int, float, str]

# Colour used for all object boxes (BGR for OpenCV, RGB for PIL).
_BOX_COLOR_BGR = (0, 200, 0)
_BOX_COLOR_RGB = (0, 200, 0)

# Thickness of the bounding-box rectangle (pixels).
_BOX_THICKNESS = 2

# Background colour for the label chip (dark, semi-opaque when using PIL).
_LABEL_BG_BGR = (20, 20, 20)
_LABEL_BG_RGB = (20, 20, 20)

# Text colour – white for maximum contrast on dark background.
_TEXT_COLOR_BGR = (255, 255, 255)
_TEXT_COLOR_RGB = (255, 255, 255)

# Font scale & thickness for OpenCV text.
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1
_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _format_label(label_str: str, confidence: float, show_conf: bool) -> str:
    """Build the label text shown above the bounding box.

    Examples
    --------
    >>> _format_label("Bird", 0.942, True)
    'Bird 94.2%'
    >>> _format_label("Drone", 0.5, False)
    'Drone'
    """
    label_str = label_str.strip() or "Object"
    if show_conf:
        return f"{label_str} {confidence * 100:.1f}%"
    return label_str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_boxes(
    image: np.ndarray,
    detections: List[Detection],
    show_conf: bool = True,
) -> np.ndarray:
    """Draw bounding boxes with labels on a BGR image.

    Parameters
    ----------
    image : np.ndarray
        Input image in **BGR** format (as returned by ``cv2.imread``).
        The array is *copied* internally; the original is not mutated.
    detections : list of (x, y, w, h, confidence, label_str)
        Each detection tuple describes one object:

        * ``x``, ``y`` – top-left corner in pixels (int).
        * ``w``, ``h`` – box width and height in pixels (int).
        * ``confidence`` – score in [0, 1] (float).
        * ``label_str``  – human-readable class name, e.g. ``"Bird"`` or
          ``"Drone"`` (str).
    show_conf : bool, optional
        Whether to append the confidence percentage to the label text
        (default ``True``).

    Returns
    -------
    np.ndarray
        Copy of *image* with bounding boxes and label chips rendered in BGR.

    Notes
    -----
    * Boxes are clamped to the image boundary so that partially out-of-frame
      detections are still drawn correctly.
    * If *detections* is empty the function returns a copy of the input image
      unchanged.
    * The label chip is drawn *above* the box top edge.  When the box is at
      the very top of the image the chip is rendered *inside* the box instead
      so it remains visible.
    """
    if image is None or image.size == 0:
        raise ValueError("draw_boxes: received an empty image array.")

    canvas = image.copy()
    h_img, w_img = canvas.shape[:2]

    for det in detections:
        if len(det) != 6:
            # Skip malformed detections rather than crashing.
            continue

        x, y, w, h, conf, label_str = det

        # Convert to int; handle numpy scalar types.
        x, y, w, h = int(x), int(y), int(w), int(h)
        conf = float(conf)

        # Clamp box coordinates to image bounds.
        x1 = _clamp(x, 0, w_img - 1)
        y1 = _clamp(y, 0, h_img - 1)
        x2 = _clamp(x + w, 0, w_img - 1)
        y2 = _clamp(y + h, 0, h_img - 1)

        # Skip degenerate boxes.
        if x2 <= x1 or y2 <= y1:
            continue

        # --- draw rectangle -------------------------------------------------
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _BOX_COLOR_BGR, _BOX_THICKNESS)

        # --- build label text -----------------------------------------------
        text = _format_label(label_str, conf, show_conf)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, _FONT_FACE, _FONT_SCALE, _FONT_THICKNESS
        )
        padding = 3  # px around text inside chip
        chip_h = text_h + baseline + padding * 2

        # Position chip above the box; fall inside if not enough room.
        chip_y1 = y1 - chip_h
        chip_y2 = y1
        if chip_y1 < 0:
            chip_y1 = y1
            chip_y2 = y1 + chip_h

        chip_x2 = min(x1 + text_w + padding * 2, w_img - 1)

        # --- draw label background chip ------------------------------------
        cv2.rectangle(
            canvas,
            (x1, chip_y1),
            (chip_x2, chip_y2),
            _LABEL_BG_BGR,
            cv2.FILLED,
        )

        # --- draw label text -----------------------------------------------
        text_x = x1 + padding
        text_y = chip_y2 - baseline - padding
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            _FONT_FACE,
            _FONT_SCALE,
            _TEXT_COLOR_BGR,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return canvas


def detections_to_pil(
    image: np.ndarray,
    detections: List[Detection],
    show_conf: bool = True,
) -> Image.Image:
    """Convert a NumPy image + detections to a PIL Image with boxes drawn.

    Parameters
    ----------
    image : np.ndarray
        Input image.  Accepted colour orders:

        * 3-channel BGR (OpenCV default) — converted to RGB internally.
        * 3-channel RGB — used as-is.
        * 1-channel or grayscale — converted to RGB.
    detections : list of (x, y, w, h, confidence, label_str)
        Same format as :func:`draw_boxes`.
    show_conf : bool, optional
        Append confidence percentage to label text (default ``True``).

    Returns
    -------
    PIL.Image.Image
        RGB PIL image with bounding boxes and labels rendered.

    Notes
    -----
    The function uses PIL's :class:`ImageDraw` for rendering which gives
    anti-aliased text on platforms that support it.  Falls back to a
    built-in bitmap font if no TrueType fonts are available so it never
    raises ``OSError``.
    """
    if image is None or image.size == 0:
        raise ValueError("detections_to_pil: received an empty image array.")

    # Normalise to RGB uint8.
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        # Grayscale → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # BGRA → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        # Assume BGR → RGB (standard OpenCV output).
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(rgb)
    draw    = ImageDraw.Draw(pil_img)

    # Try to load a reasonable font; fall back to PIL default.
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    h_img, w_img = image.shape[:2]

    for det in detections:
        if len(det) != 6:
            continue

        x, y, w, h, conf, label_str = det
        x, y, w, h = int(x), int(y), int(w), int(h)
        conf = float(conf)

        x1 = _clamp(x, 0, w_img - 1)
        y1 = _clamp(y, 0, h_img - 1)
        x2 = _clamp(x + w, 0, w_img - 1)
        y2 = _clamp(y + h, 0, h_img - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw box.
        for t in range(_BOX_THICKNESS):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=_BOX_COLOR_RGB,
            )

        text = _format_label(label_str, conf, show_conf)

        # Measure text size.
        try:
            bbox = font.getbbox(text)          # Pillow >= 9
            tw   = bbox[2] - bbox[0]
            th   = bbox[3] - bbox[1]
        except AttributeError:
            tw, th = font.getsize(text)        # Pillow < 9

        padding  = 3
        chip_h   = th + padding * 2
        chip_y1  = y1 - chip_h
        chip_y2  = y1
        if chip_y1 < 0:
            chip_y1 = y1
            chip_y2 = y1 + chip_h

        chip_x2 = min(x1 + tw + padding * 2, w_img - 1)

        draw.rectangle([x1, chip_y1, chip_x2, chip_y2], fill=_LABEL_BG_RGB)
        draw.text(
            (x1 + padding, chip_y1 + padding),
            text,
            fill=_TEXT_COLOR_RGB,
            font=font,
        )

    return pil_img


def numpy_to_pixmap(image: np.ndarray):
    """Convert a NumPy RGB array to a ``QPixmap`` for PyQt5 display.

    Parameters
    ----------
    image : np.ndarray
        Image in **RGB** uint8 format, shape ``(H, W, 3)``.  If the array is
        not contiguous it is made contiguous before conversion.

    Returns
    -------
    PyQt5.QtGui.QPixmap
        QPixmap ready to be assigned to a ``QLabel`` or painted in a widget.

    Raises
    ------
    ImportError
        If PyQt5 is not installed in the current environment.
    ValueError
        If *image* is not a 3-channel uint8 array.

    Notes
    -----
    This function imports PyQt5 lazily so the rest of the module can be used
    in headless (server-side) environments without triggering an ImportError.
    """
    from PyQt5.QtGui import QImage, QPixmap  # lazy import — GUI only

    if image is None or image.size == 0:
        raise ValueError("numpy_to_pixmap: received an empty image array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"numpy_to_pixmap expects shape (H, W, 3), got {image.shape}."
        )
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # QImage requires a C-contiguous buffer.
    if not image.data.contiguous:
        image = np.ascontiguousarray(image)

    h, w, ch = image.shape
    bytes_per_line = ch * w
    qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_image)
