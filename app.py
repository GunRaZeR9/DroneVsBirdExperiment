"""
app.py — Bird vs Drone Detector (Streamlit UI)
----------------------------------------------
Run:  streamlit run app.py

Replaces the PyQt5 desktop GUI. All ML logic (data_loader, feature_extractor,
model, detector, metrics) is unchanged — only the presentation layer is swapped.
"""

from __future__ import annotations

import io
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.core.data_loader import load_train_test, dataset_stats
from src.core.detector import detect as hog_detect
from src.core.feature_extractor import extract_hog_batch
from src.core.model import BirdDroneModel
from src.utils.metrics import compute_metrics

# YOLOv10 — imported lazily (only when ultralytics is available)
try:
    from src.core.yolov10_detector import YOLOv10Detector
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# Default weights path (produced by training script)
_DEFAULT_WEIGHTS = Path("runs/detect/runs/train/drone_v10_cpu/weights/best.pt")

# Also check the non-nested path (task="detect" fix)
_DEFAULT_WEIGHTS_ALT = Path("runs/train/drone_v10_cpu/weights/best.pt")

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bird vs Drone Detector",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark card theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Metric card tweaks */
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #cdd6f4; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #89dceb; font-size: 1.6rem; font-weight: 700; }

    /* Tab bar */
    button[data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }

    /* Log textarea */
    .log-area textarea { font-family: monospace; font-size: 0.78rem; background: #1e1e2e; color: #a6e3a1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "model": None,
    "metrics": None,
    "val_metrics": None,
    "history": None,
    "y_test": None,
    "y_pred": None,
    "training_logs": [],
    "is_training": False,
    "detection_result": None,   # (pil_image, detections) from last run
    "train_elapsed": None,
    "train_start": None,
    "_yolo_detector": None,
    "_yolo_weights_loaded": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Helper — colored bounding boxes
# ─────────────────────────────────────────────────────────────────────────────
def draw_colored_boxes(bgr_image, detections):
    """Draw Bird=green, Drone=red boxes on image."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    _COLORS = {"Bird": "#00c853", "Drone": "#ff1744", "Background": "#aaaaaa"}

    for x, y, w, h, conf, label in detections:
        color = _COLORS.get(label, "#ffffff")
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        text = f"{label} {conf * 100:.1f}%"
        draw.text((x + 3, max(0, y - 16)), text, fill=color, font=font)
    return pil_img


def _format_dataset_stats(stats: dict) -> str:
    """Return a compact dataset summary string for the sidebar."""
    parts: list[str] = []
    for split_name in ("train", "test", "valid"):
        split_stats = stats.get(split_name)
        if not split_stats:
            continue
        label_text = " · ".join(f"{label}: {count:,}" for label, count in split_stats.items())
        parts.append(f"{split_name}: {label_text}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Parameters + actions
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Parameters")

    # --- Model ---
    st.subheader("Model")
    model_type = st.radio("Classifier", ["MLP", "SVM"], horizontal=True)

    if model_type == "MLP":
        epochs      = st.slider("Epochs",        5, 200,  50, step=5)
        lr          = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.001, 0.01, 0.1],
            value=0.001,
            format_func=lambda x: str(x),
        )
        num_layers  = st.slider("Hidden Layers",  1, 5,    3)
        hidden_size = st.select_slider(
            "Units / Layer",
            options=[32, 64, 128, 256, 512],
            value=128,
        )
        batch_size  = st.select_slider(
            "Batch Size",
            options=[16, 32, 64, 128],
            value=32,
        )
    else:
        epochs, lr, num_layers, hidden_size, batch_size = 1, 0.001, 1, 128, 32

    st.divider()

    # --- Data ---
    st.subheader("Data")
    max_train = st.slider("Max Train Images", 100, 3000, 1000, step=100)
    max_test  = st.slider("Max Test Images",   50, 1000,  300, step=50)

    try:
        _stats = dataset_stats("Data")
        _stats_text = _format_dataset_stats(_stats)
        if _stats_text:
            st.caption(_stats_text)
    except Exception:
        pass

    st.divider()

    # --- Detection ---
    st.subheader("Detection (HOG)")
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.10, 0.99, 0.72, step=0.05
    )
    seq_length = st.slider(
        "Stride Control (seq_length)", 1, 16, 8,
        help="Higher = finer stride = slower but more thorough scan",
    )

    st.divider()

    # --- Train button ---
    train_clicked = st.button("Train HOG Model", use_container_width=True, type="primary")

    st.divider()

    # --- Save / Load ---
    st.subheader("Save / Load")
    model_path = st.text_input("Model file", value="bird_drone_model.pkl")

    col_save, col_load = st.columns(2)
    with col_save:
        save_clicked = st.button("Save", use_container_width=True)
    with col_load:
        load_clicked = st.button("Load", use_container_width=True)

    # Model status indicator
    if st.session_state.model is not None:
        m = st.session_state.model
        class_text = "/".join(getattr(m, "class_names", ["Background", "Bird", "Drone"]))
        st.success(f"HOG {m.mode} ready: {class_text}")
    else:
        st.warning("No HOG model loaded")

    if _YOLO_AVAILABLE:
        # Show YOLOv10 weight status in sidebar
        yolo_weights_exist = _DEFAULT_WEIGHTS.exists() or _DEFAULT_WEIGHTS_ALT.exists()
        if yolo_weights_exist:
            found = _DEFAULT_WEIGHTS if _DEFAULT_WEIGHTS.exists() else _DEFAULT_WEIGHTS_ALT
            st.success(f"YOLOv10 weights found: {found.name}")
        else:
            st.info("YOLOv10: training in progress (no weights yet)")
    else:
        st.warning("ultralytics not installed — YOLOv10 unavailable")


# ─────────────────────────────────────────────────────────────────────────────
# Main area — header + tabs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Bird vs Drone Detector")
st.caption("HOG + SVM/MLP  |  YOLOv10 deep learning  |  Streamlit UI")

tab_train, tab_results, tab_live = st.tabs(
    ["Training (HOG)", "Results", "Live Test"]
)


# =============================================================================
# TAB 1 — Training (HOG+MLP/SVM)
# =============================================================================
with tab_train:
    st.subheader("Training Pipeline (HOG + MLP/SVM)")

    # Persistent log from previous run
    if st.session_state.training_logs and not train_clicked:
        st.text_area(
            "Previous run log",
            value="\n".join(st.session_state.training_logs[-80:]),
            height=400,
            disabled=True,
        )

    if train_clicked:
        st.session_state.training_logs = []
        st.session_state.is_training   = True
        st.session_state.train_start   = time.time()

        params = dict(
            mode=model_type,
            epochs=epochs,
            lr=lr,
            num_layers=num_layers,
            hidden_size=hidden_size,
            batch_size=batch_size,
        )

        # live UI widgets (updated via callbacks during training)
        status_ph  = st.empty()
        pbar_ph    = st.empty()
        log_ph     = st.empty()

        live_logs: list[str] = []

        def _add_log(msg: str) -> None:
            live_logs.append(msg)
            log_ph.text_area(
                "Live log",
                value="\n".join(live_logs[-80:]),
                height=400,
                disabled=True,
            )

        _pbar = pbar_ph.progress(0, text="Initialising...")

        def _progress(pct: int, msg: str = "") -> None:
            label = (msg[:120] if msg else f"{pct}%")
            _pbar.progress(min(int(pct), 100), text=label)

        try:
            # 1. Load / cache patches
            status_ph.info("Step 1/4 — Loading dataset...")
            _add_log("--- Step 1: Loading dataset ---")

            def _data_cb(pct: int, msg: str) -> None:
                _progress(pct // 4, msg)
                _add_log(msg)

            X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = load_train_test(
                data_root="Data",
                max_train=max_train,
                max_test=max_test,
                window_size=(64, 64),
                progress_callback=_data_cb,
            )
            _progress(25, f"Dataset ready — {len(X_train_raw)} train, {len(X_test_raw)} test patches")
            _add_log(
                f"  Train: {len(X_train_raw)} patches  "
                f"(pos={int(y_train.sum())}  neg={int((y_train == 0).sum())})"
            )
            _add_log(
                f"  Test:  {len(X_test_raw)} patches  "
                f"(pos={int(y_test.sum())}  neg={int((y_test == 0).sum())})"
            )

            # 2. HOG train
            status_ph.info("Step 2/4 — Extracting HOG features (train set)...")
            _add_log("--- Step 2: HOG extraction (train) ---")

            X_train = extract_hog_batch(
                X_train_raw,
                progress_callback=lambda p: _progress(25 + p // 4, f"HOG train {p}%"),
            )
            _progress(50, f"HOG train done — shape {X_train.shape}")
            _add_log(f"  Train HOG shape: {X_train.shape}")

            # 2b. HOG val
            X_val = extract_hog_batch(
                X_val_raw,
                progress_callback=lambda p: _progress(50 + p // 10, f"HOG val {p}%"),
            )
            _add_log(f"  Val HOG shape:   {X_val.shape}")

            # 3. HOG test
            status_ph.info("Step 3/4 — Extracting HOG features (test set)...")
            _add_log("--- Step 3: HOG extraction (test) ---")

            X_test = extract_hog_batch(
                X_test_raw,
                progress_callback=lambda p: _progress(50 + p // 10, f"HOG test {p}%"),
            )
            _progress(60, f"HOG test done — shape {X_test.shape}")
            _add_log(f"  Test HOG shape:  {X_test.shape}")

            # 4. Fit model
            status_ph.info(f"Step 4/4 — Training {model_type} classifier...")
            _add_log(f"--- Step 4: Training {model_type} ---")

            model = BirdDroneModel(**params)
            model.train(
                X_train, y_train,
                progress_callback=lambda p: _progress(60 + int(p * 0.35), f"Train {p:.0f}%"),
                log_callback=_add_log,
            )
            _progress(95, "Evaluating on test set...")

            # 5. Evaluate
            _add_log("--- Evaluating ---")
            y_pred  = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            y_val_pred  = model.predict(X_val)
            val_metrics = compute_metrics(y_val, y_val_pred)

            elapsed = time.time() - st.session_state.get("train_start", time.time())

            _progress(100, "Complete!")

            _add_log(
                f"  Accuracy:  {metrics['accuracy']:.4f}\n"
                f"  F1:        {metrics['f1']:.4f}\n"
                f"  Precision: {metrics['precision']:.4f}\n"
                f"  Recall:    {metrics['recall']:.4f}\n"
            )
            _add_log(
                f"  Val Accuracy: {val_metrics['accuracy']:.4f}  "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            # Store results
            st.session_state.model         = model
            st.session_state.metrics       = metrics
            st.session_state.val_metrics   = val_metrics
            st.session_state.history       = model.training_history
            st.session_state.y_test        = y_test
            st.session_state.y_pred        = y_pred
            st.session_state.training_logs = live_logs
            st.session_state.train_elapsed = elapsed

            status_ph.success(
                f"Training complete!  "
                f"Accuracy={metrics['accuracy']:.4f}  "
                f"F1={metrics['f1']:.4f}"
            )
            st.balloons()

        except Exception as exc:
            status_ph.error(f"Training failed: {exc}")
            _add_log("--- ERROR ---")
            _add_log(traceback.format_exc())
            st.session_state.training_logs = live_logs

        finally:
            st.session_state.is_training = False


# =============================================================================
# TAB 2 — Results
# =============================================================================
with tab_results:
    st.subheader("Evaluation Results")

    metrics  = st.session_state.metrics
    history  = st.session_state.history

    if metrics is None:
        st.info("Train a model first to see results here.")
    else:
        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
        c2.metric("F1 Score",  f"{metrics['f1']:.4f}")
        c3.metric("Precision", f"{metrics['precision']:.4f}")
        c4.metric("Recall",    f"{metrics['recall']:.4f}")

        if st.session_state.val_metrics:
            vm = st.session_state.val_metrics
            st.caption(f"Validation set — Accuracy: {vm['accuracy']:.4f}  F1: {vm['f1']:.4f}")

        if st.session_state.get("train_elapsed"):
            st.caption(f"Training time: {st.session_state.train_elapsed:.1f}s")

        st.divider()

        col_curves, col_cm = st.columns([3, 2])

        with col_curves:
            st.markdown("**Training Curves**")
            if history and history.get("loss") and len(history["loss"]) > 1:
                fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.5))
                fig.patch.set_facecolor("#0e1117")
                for ax in (ax_loss, ax_acc):
                    ax.set_facecolor("#1e1e2e")
                    ax.tick_params(colors="#cdd6f4")
                    ax.xaxis.label.set_color("#cdd6f4")
                    ax.yaxis.label.set_color("#cdd6f4")
                    ax.title.set_color("#cdd6f4")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#313244")

                ax_loss.plot(history["loss"], color="#f38ba8", lw=2, label="Loss")
                ax_loss.set_title("Loss")
                ax_loss.set_xlabel("Epoch")
                ax_loss.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")

                ax_acc.plot(history["accuracy"], color="#a6e3a1", lw=2, label="Train Acc")
                if history.get("val_accuracy"):
                    ax_acc.plot(
                        history["val_accuracy"], "--", color="#89dceb", lw=1.5, label="Val Acc"
                    )
                ax_acc.set_title("Accuracy")
                ax_acc.set_xlabel("Epoch")
                ax_acc.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info(
                    "Single-step training (SVM) — no epoch curves. "
                    f"Train accuracy: {metrics['accuracy']:.4f}"
                )

        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm   = metrics["confusion_matrix"]
            n_cl = cm.shape[0]
            _tick_names = {0: "Bg", 1: "Bird", 2: "Drone"}
            tick_labels = [_tick_names.get(i, str(i)) for i in range(n_cl)]
            fig_sz = max(4.0, n_cl * 1.5)
            fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.95))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#1e1e2e")
            ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ticks = list(range(n_cl))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(tick_labels, color="#cdd6f4")
            ax.set_yticklabels(tick_labels, color="#cdd6f4")
            ax.set_xlabel("Predicted", color="#cdd6f4")
            ax.set_ylabel("True",      color="#cdd6f4")
            ax.set_title("Confusion Matrix", color="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_edgecolor("#313244")
            thresh = cm.max() / 2.0
            for i in range(n_cl):
                for j in range(n_cl):
                    ax.text(
                        j, i, f"{cm[i, j]:,}",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "#1e1e2e",
                    )
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

        s1, s2, s3 = st.columns(3)
        s1.metric("Total Samples",     f"{metrics['n_samples']:,}")
        s2.metric("True Positives",    f"{metrics['n_true_positive']:,}")
        s3.metric("Pred Positives",    f"{metrics['n_positive']:,}")

        with st.expander("Full Classification Report"):
            st.caption("Labels: 0=Background  1=Bird  2=Drone")
            st.code(metrics["report"], language=None)


# =============================================================================
# TAB 3 — Live Test
# =============================================================================
with tab_live:
    st.subheader("Live Detection Test")

    # Backend selector
    backend_options = ["HOG + MLP (classical)"]
    if _YOLO_AVAILABLE:
        backend_options.append("YOLOv10 (deep learning)")

    backend = st.radio(
        "Detection backend",
        backend_options,
        horizontal=True,
        key="live_backend",
    )
    use_yolo = _YOLO_AVAILABLE and backend.startswith("YOLOv10")

    st.divider()

    # ── YOLOv10 path ──────────────────────────────────────────────────────────
    if use_yolo:
        # Auto-select existing weights path
        _auto_weights = (
            str(_DEFAULT_WEIGHTS) if _DEFAULT_WEIGHTS.exists()
            else str(_DEFAULT_WEIGHTS_ALT) if _DEFAULT_WEIGHTS_ALT.exists()
            else str(_DEFAULT_WEIGHTS)
        )

        weights_input = st.text_input(
            "YOLOv10 weights (.pt)",
            value=_auto_weights,
            key="yolo_weights_path",
        )
        weights_path = Path(weights_input.strip())

        yolo_conf = st.slider(
            "Confidence threshold (YOLOv10)",
            min_value=0.10, max_value=0.95, value=0.45, step=0.05,
            key="yolo_conf",
        )

        if not weights_path.exists():
            st.warning(
                f"Weights not found: `{weights_path}`\n\n"
                "Training is still running (or has not started yet). "
                "Once best.pt appears, refresh and run detection."
            )
            st.info(
                "**Start/resume CPU training:**\n"
                "```\n"
                "python scripts/train_yolov10.py --device cpu --batch 4 --epochs 50\n"
                "```\n"
                "Checkpoints saved every epoch to `runs/train/drone_v10_cpu/weights/`.\n"
                "Estimated time: ~17 h on CPU."
            )
        else:
            st.success(f"Weights loaded from: `{weights_path}`")

            # Load/cache detector in session state
            if st.session_state.get("_yolo_weights_loaded") != str(weights_path):
                with st.spinner("Loading YOLOv10 model..."):
                    try:
                        st.session_state["_yolo_detector"] = YOLOv10Detector(
                            weights=str(weights_path), conf=yolo_conf, device="cpu"
                        )
                        st.session_state["_yolo_weights_loaded"] = str(weights_path)
                    except Exception as exc:
                        st.error(f"Failed to load model: {exc}")
                        st.code(traceback.format_exc())
                        st.session_state["_yolo_detector"] = None

            yolo_detector = st.session_state.get("_yolo_detector")

            uploaded_file = st.file_uploader(
                "Upload a test image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="live_uploader_yolo",
            )

            if uploaded_file is not None:
                file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
                bgr_image  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if bgr_image is None:
                    st.error("Could not decode image.")
                elif yolo_detector is None:
                    st.error("Model failed to load — see error above.")
                else:
                    h_img, w_img = bgr_image.shape[:2]
                    st.caption(f"Image size: {w_img} x {h_img} px")

                    col_orig, col_result = st.columns(2)
                    with col_orig:
                        st.markdown("**Original**")
                        st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), use_container_width=True)

                    run_det = st.button("Run YOLOv10 Detection", type="primary", use_container_width=True)

                    if run_det:
                        with st.spinner("Running YOLOv10 inference..."):
                            try:
                                detections = yolo_detector.detect(bgr_image, conf=yolo_conf)
                                # annotate() returns BGR numpy — convert to RGB for st.image
                                annotated_bgr = yolo_detector.annotate(bgr_image, conf=yolo_conf)
                                annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                                st.session_state.detection_result = (annotated, detections)
                            except Exception as exc:
                                st.error(f"Detection error: {exc}")
                                st.code(traceback.format_exc())
                                st.session_state.detection_result = None

                    if st.session_state.detection_result is not None:
                        pil_result, detections = st.session_state.detection_result

                        with col_result:
                            st.markdown(f"**Result — {len(detections)} detection(s)**")
                            st.image(pil_result, use_container_width=True)

                        if detections:
                            label_counts: Counter = Counter(d[5] for d in detections)
                            label_conf: dict = {}
                            for d in detections:
                                label_conf.setdefault(d[5], []).append(d[4])

                            summary_parts = []
                            for label in ["drone", "bird"]:
                                if label in label_counts:
                                    avg_conf = sum(label_conf[label]) / len(label_conf[label])
                                    color = "🔴" if label == "drone" else "🟢"
                                    summary_parts.append(
                                        f"{color} **{label_counts[label]}x {label}** "
                                        f"(avg {avg_conf * 100:.1f}%)"
                                    )
                            if summary_parts:
                                st.markdown("  |  ".join(summary_parts))

                        st.divider()

                        if detections:
                            rows = [
                                {
                                    "#":          i + 1,
                                    "Label":      d[5],
                                    "Confidence": f"{d[4] * 100:.1f}%",
                                    "x": d[0], "y": d[1], "w": d[2], "h": d[3],
                                }
                                for i, d in enumerate(detections)
                            ]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        else:
                            st.info("No objects detected above threshold. Try lowering confidence.")

    # ── HOG + MLP path ────────────────────────────────────────────────────────
    else:
        current_model = st.session_state.model

        if current_model is None:
            st.warning(
                "No trained HOG+MLP model available. "
                "Train or load a model in the Training tab first."
            )
        else:
            st.info(
                f"Model: **{current_model.mode}** | "
                f"Classes: Bird / Drone / Background | "
                f"Confidence >= {confidence_threshold:.2f}"
            )

            uploaded_file = st.file_uploader(
                "Upload a test image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="live_uploader_hog",
            )

            if uploaded_file is not None:
                file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
                bgr_image  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if bgr_image is None:
                    st.error("Could not decode image. Try a different file.")
                else:
                    h_img, w_img = bgr_image.shape[:2]
                    st.caption(f"Image size: {w_img} x {h_img} px")

                    col_orig, col_result = st.columns(2)

                    with col_orig:
                        st.markdown("**Original**")
                        rgb_orig = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                        st.image(rgb_orig, use_container_width=True)

                    run_det = st.button("Run HOG+MLP Detection", type="primary", use_container_width=True)

                    if run_det:
                        with st.spinner("Running multi-scale sliding-window detection..."):
                            try:
                                detections = hog_detect(
                                    bgr_image,
                                    current_model,
                                    seq_length=seq_length,
                                    confidence_threshold=confidence_threshold,
                                )
                                st.session_state.detection_result = (
                                    draw_colored_boxes(bgr_image, detections),
                                    detections,
                                )
                            except Exception as exc:
                                st.error(f"Detection error: {exc}")
                                st.code(traceback.format_exc())
                                st.session_state.detection_result = None

                    if st.session_state.detection_result is not None:
                        pil_result, detections = st.session_state.detection_result

                        with col_result:
                            st.markdown(f"**Result — {len(detections)} detection(s)**")
                            st.image(pil_result, use_container_width=True)

                        if detections:
                            label_counts: Counter = Counter(d[5] for d in detections)
                            label_conf: dict = {}
                            for d in detections:
                                label_conf.setdefault(d[5], []).append(d[4])

                            summary_parts = []
                            for label in ["Bird", "Drone", "Background"]:
                                if label in label_counts:
                                    avg_conf = sum(label_conf[label]) / len(label_conf[label])
                                    color = "🟢" if label == "Bird" else "🔴" if label == "Drone" else "⚪"
                                    summary_parts.append(
                                        f"{color} **{label_counts[label]}x {label}** "
                                        f"(avg {avg_conf * 100:.1f}%)"
                                    )
                            if summary_parts:
                                st.markdown("  |  ".join(summary_parts))

                        st.divider()

                        if detections:
                            rows = [
                                {
                                    "#":          i + 1,
                                    "Label":      d[5],
                                    "Confidence": f"{d[4] * 100:.1f}%",
                                    "x": d[0], "y": d[1], "w": d[2], "h": d[3],
                                }
                                for i, d in enumerate(detections)
                            ]
                            st.dataframe(
                                pd.DataFrame(rows),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info(
                                "No objects detected above the confidence threshold. "
                                "Try lowering the threshold in the sidebar."
                            )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Save / Load actions (evaluated after all tabs are rendered)
# ─────────────────────────────────────────────────────────────────────────────
if save_clicked:
    if st.session_state.model is None:
        st.sidebar.error("Nothing to save — train a model first.")
    else:
        try:
            st.session_state.model.save(model_path)
            st.sidebar.success(f"Saved: {model_path}")
        except Exception as exc:
            st.sidebar.error(f"Save failed: {exc}")

if load_clicked:
    try:
        loaded = BirdDroneModel()
        loaded.load(model_path)
        st.session_state.model   = loaded
        st.session_state.history = loaded.training_history
        # Metrics are not serialised separately — clear to avoid stale display
        if not loaded.training_history.get("loss"):
            st.session_state.metrics = None
        st.sidebar.success(f"Loaded {loaded.mode} model from {model_path}")
        st.rerun()
    except FileNotFoundError:
        st.sidebar.error(f"File not found: {model_path}")
    except Exception as exc:
        st.sidebar.error(f"Load failed: {exc}")
