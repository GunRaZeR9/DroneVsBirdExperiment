"""
app.py — Bird vs Drone Detector (Streamlit UI)
----------------------------------------------
Run:  streamlit run app.py

All ML logic (data_loader, feature_extractor, model, detector, metrics) is
unchanged — only the presentation layer is swapped.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
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

# Project root (one level above this file)
ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Optional deep-learning detectors
# ─────────────────────────────────────────────────────────────────────────────
try:
    from src.core.yolov10_detector import YOLOv10Detector
    _YOLO10_AVAILABLE = True
except ImportError:
    _YOLO10_AVAILABLE = False

try:
    from src.core.yolov8_detector import YOLOv8Detector
    _YOLO8_AVAILABLE = True
except ImportError:
    _YOLO8_AVAILABLE = False

# Keep legacy alias so any downstream code still referencing _YOLO_AVAILABLE works
_YOLO_AVAILABLE = _YOLO10_AVAILABLE

# Default weights paths — clean path is primary; nested path is legacy fallback
_YOLO10_WEIGHTS     = Path("runs/train/drone_v10_cpu/weights/best.pt")
_YOLO10_WEIGHTS_ALT = Path("runs/detect/runs/train/drone_v10_cpu/weights/best.pt")
_YOLO8_WEIGHTS      = Path("runs/train/drone_v8/weights/best.pt")

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
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #cdd6f4; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #89dceb; font-size: 1.6rem; font-weight: 700; }
    button[data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    .log-area textarea { font-family: monospace; font-size: 0.78rem; background: #1e1e2e; color: #a6e3a1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    # legacy / backward-compat
    "model": None,
    "metrics": None,
    "val_metrics": None,
    "history": None,
    "y_test": None,
    "y_pred": None,
    "training_logs": [],
    "is_training": False,
    "detection_result": None,
    "train_elapsed": None,
    "train_start": None,
    # new per-model state
    "hog_mlp_model": None,
    "hog_svm_model": None,
    "hog_mlp_metrics": None,
    "hog_svm_metrics": None,
    # YOLO detector cache
    "_yolo8_detector": None,
    "_yolo8_weights_loaded": None,
    "_yolo10_detector": None,
    "_yolo_weights_loaded": None,   # tracks loaded path for yolov10
    # subprocess handles for background YOLO training
    "yolo_train_procs": {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# Auto-load HOG models from disk on startup
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.hog_mlp_model is None and Path("models/hog_mlp.pkl").exists():
    try:
        _m = BirdDroneModel()
        _m.load("models/hog_mlp.pkl")
        st.session_state.hog_mlp_model = _m
        st.session_state.model = _m
    except Exception:
        pass

if st.session_state.hog_svm_model is None and Path("models/hog_svm.pkl").exists():
    try:
        _m = BirdDroneModel()
        _m.load("models/hog_svm.pkl")
        st.session_state.hog_svm_model = _m
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helper — colored bounding boxes
# ─────────────────────────────────────────────────────────────────────────────
def draw_colored_boxes(bgr_image, detections):
    """Draw Bird=green, Drone=red boxes on image. Returns PIL Image (RGB)."""
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
    parts: list[str] = []
    for split_name in ("train", "test", "valid"):
        split_stats = stats.get(split_name)
        if not split_stats:
            continue
        label_text = " · ".join(f"{label}: {count:,}" for label, count in split_stats.items())
        parts.append(f"{split_name}: {label_text}")
    return "\n".join(parts)


def _yolo_results_csv(run_name: str) -> Path | None:
    """Return path to results.csv for a YOLO run, or None if not found."""
    candidates = [
        Path(f"runs/train/{run_name}/results.csv"),
        Path(f"runs/detect/runs/train/{run_name}/results.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_yolo_latest_epoch(run_name: str) -> str:
    """Return a one-line summary of the latest epoch from results.csv."""
    csv_path = _yolo_results_csv(run_name)
    if csv_path is None:
        return "No results.csv yet"
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if df.empty:
            return "results.csv is empty"
        last = df.iloc[-1]
        epoch_col = next((c for c in df.columns if "epoch" in c.lower()), None)
        map_col   = next((c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()), None)
        loss_col  = next((c for c in df.columns if "box_loss" in c.lower() or "train/box" in c.lower()), None)
        parts = []
        if epoch_col:
            parts.append(f"Epoch {int(last[epoch_col])}/{len(df)}")
        if map_col:
            parts.append(f"mAP50={float(last[map_col]):.4f}")
        if loss_col:
            parts.append(f"box_loss={float(last[loss_col]):.4f}")
        return "  |  ".join(parts) if parts else f"Row {len(df)} parsed"
    except Exception as exc:
        return f"Parse error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Parameters")

    # ── HOG Parameters ────────────────────────────────────────────────────────
    st.subheader("HOG Parameters")
    max_train_images = st.slider("Max Train Images", 100, 3000, 1000, step=100)
    max_test_images  = st.slider("Max Test Images",   50, 1000,  300, step=50)
    mlp_epochs       = st.slider("MLP Epochs",         5,  100,   30, step=5)
    conf_threshold   = st.slider(
        "Confidence Threshold (HOG)",
        min_value=0.10, max_value=0.99, value=0.72, step=0.05,
    )
    seq_length = st.slider(
        "Stride Control (seq_length)", 1, 16, 8,
        help="Higher = finer stride = slower but more thorough scan",
    )

    try:
        _stats = dataset_stats("Data")
        _stats_text = _format_dataset_stats(_stats)
        if _stats_text:
            st.caption(_stats_text)
    except Exception:
        pass

    st.divider()

    # ── YOLO Parameters ───────────────────────────────────────────────────────
    st.subheader("YOLO Parameters")
    yolo_v8_epochs  = st.slider("YOLOv8 Epochs",  5, 30, 10, step=1)
    yolo_v10_epochs = st.slider("YOLOv10 Epochs", 5, 50, 15, step=1)
    yolo_conf       = st.slider(
        "Conf Threshold (YOLO)",
        min_value=0.10, max_value=0.95, value=0.25, step=0.05,
    )

    st.divider()

    # ── Train All Models button ───────────────────────────────────────────────
    train_all_clicked = st.button("Train All Models", use_container_width=True, type="primary")

    st.divider()

    # ── Save / Load (HOG model) ───────────────────────────────────────────────
    st.subheader("Save / Load")
    model_path = st.text_input("Model file", value="bird_drone_model.pkl")
    col_save, col_load = st.columns(2)
    with col_save:
        save_clicked = st.button("Save", use_container_width=True)
    with col_load:
        load_clicked = st.button("Load", use_container_width=True)

    st.divider()

    # ── Status ────────────────────────────────────────────────────────────────
    st.subheader("Model Status")

    def _status_line(label: str, ok: bool, detail: str = "") -> None:
        icon = "✅" if ok else "⬜"
        txt  = f"{icon} **{label}**"
        if detail:
            txt += f" — {detail}"
        st.markdown(txt)

    _mlp_ok  = st.session_state.hog_mlp_model is not None
    _svm_ok  = st.session_state.hog_svm_model is not None
    _v8_ok   = _YOLO8_WEIGHTS.exists()
    _v10_ok  = _YOLO10_WEIGHTS.exists() or _YOLO10_WEIGHTS_ALT.exists()

    _status_line("HOG+MLP",  _mlp_ok,  "loaded" if _mlp_ok else "not trained")
    _status_line("HOG+SVM",  _svm_ok,  "loaded" if _svm_ok else "not trained")
    _status_line("YOLOv8n",  _v8_ok,   "weights found" if _v8_ok else "no weights")
    _status_line("YOLOv10",  _v10_ok,  "weights found" if _v10_ok else "no weights")

    if not _YOLO8_AVAILABLE:
        st.caption("ultralytics not installed — YOLO unavailable")


# ─────────────────────────────────────────────────────────────────────────────
# Main header + tabs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Bird vs Drone Detector")
st.caption("HOG+MLP  |  HOG+SVM  |  YOLOv8n  |  YOLOv10  |  Streamlit UI")

tab_train, tab_results, tab_live = st.tabs(
    ["Train All Models", "Results", "Live Detection"]
)


# =============================================================================
# TAB 1 — Train All Models
# =============================================================================
with tab_train:
    st.subheader("Unified Training Pipeline")

    # Show previous log if idle
    if st.session_state.training_logs and not train_all_clicked:
        st.text_area(
            "Previous run log",
            value="\n".join(st.session_state.training_logs[-80:]),
            height=300,
            disabled=True,
        )

    if train_all_clicked:
        st.session_state.training_logs = []
        st.session_state.is_training   = True
        st.session_state.train_start   = time.time()

        # ── 4 status columns ─────────────────────────────────────────────────
        col_mlp_s, col_svm_s, col_v8_s, col_v10_s = st.columns(4)
        ph_mlp  = col_mlp_s.empty()
        ph_svm  = col_svm_s.empty()
        ph_v8   = col_v8_s.empty()
        ph_v10  = col_v10_s.empty()

        with col_mlp_s:
            st.markdown("**HOG+MLP**")
        with col_svm_s:
            st.markdown("**HOG+SVM**")
        with col_v8_s:
            st.markdown("**YOLOv8n**")
        with col_v10_s:
            st.markdown("**YOLOv10**")

        ph_mlp.info("Waiting...")
        ph_svm.info("Waiting...")
        ph_v8.info("Waiting...")
        ph_v10.info("Waiting...")

        # ── Live progress widgets ─────────────────────────────────────────────
        status_ph = st.empty()
        pbar_ph   = st.empty()
        log_ph    = st.empty()

        live_logs: list[str] = []

        def _add_log(msg: str) -> None:
            live_logs.append(msg)
            log_ph.text_area(
                "Live log",
                value="\n".join(live_logs[-80:]),
                height=300,
                disabled=True,
            )

        _pbar = pbar_ph.progress(0, text="Initialising...")

        def _progress(pct: int, msg: str = "") -> None:
            label = (msg[:120] if msg else f"{pct}%")
            _pbar.progress(min(int(pct), 100), text=label)

        try:
            # ── Step 1: Load dataset ──────────────────────────────────────────
            status_ph.info("Step 1/5 — Loading dataset patches...")
            _add_log("--- Step 1: Loading dataset ---")

            def _data_cb(pct: int, msg: str) -> None:
                _progress(pct // 5, msg)
                _add_log(msg)

            X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = load_train_test(
                data_root="Data",
                max_train=max_train_images,
                max_test=max_test_images,
                window_size=(64, 64),
                progress_callback=_data_cb,
            )
            _progress(20, f"Dataset ready — {len(X_train_raw)} train, {len(X_test_raw)} test patches")
            _add_log(
                f"  Train: {len(X_train_raw)} patches  "
                f"(pos={int(y_train.sum())}  neg={int((y_train == 0).sum())})"
            )
            _add_log(
                f"  Test:  {len(X_test_raw)} patches  "
                f"(pos={int(y_test.sum())}  neg={int((y_test == 0).sum())})"
            )

            # ── Step 2: Extract HOG features ──────────────────────────────────
            status_ph.info("Step 2/5 — Extracting HOG features (train)...")
            _add_log("--- Step 2: HOG extraction ---")

            X_train_hog = extract_hog_batch(
                X_train_raw,
                progress_callback=lambda p: _progress(20 + p // 5, f"HOG train {p}%"),
            )
            _add_log(f"  Train HOG shape: {X_train_hog.shape}")

            X_val_hog = extract_hog_batch(
                X_val_raw,
                progress_callback=lambda p: _progress(30 + p // 20, f"HOG val {p}%"),
            )
            _add_log(f"  Val HOG shape:   {X_val_hog.shape}")

            X_test_hog = extract_hog_batch(
                X_test_raw,
                progress_callback=lambda p: _progress(35 + p // 20, f"HOG test {p}%"),
            )
            _progress(40, f"HOG extraction done — test shape {X_test_hog.shape}")
            _add_log(f"  Test HOG shape:  {X_test_hog.shape}")

            # ── Step 3: Train HOG+MLP ─────────────────────────────────────────
            status_ph.info("Step 3/5 — Training HOG+MLP...")
            _add_log("--- Step 3: Training HOG+MLP ---")
            ph_mlp.warning("Training...")

            mlp_model = BirdDroneModel(
                mode="MLP",
                epochs=mlp_epochs,
                lr=0.001,
                num_layers=3,
                hidden_size=128,
                batch_size=32,
            )
            mlp_model.train(
                X_train_hog, y_train,
                progress_callback=lambda p: _progress(40 + int(p * 0.20), f"MLP {p:.0f}%"),
                log_callback=_add_log,
            )

            y_pred_mlp  = mlp_model.predict(X_test_hog)
            mlp_metrics = compute_metrics(y_test, y_pred_mlp)

            Path("models").mkdir(exist_ok=True)
            mlp_model.save("models/hog_mlp.pkl")

            st.session_state.hog_mlp_model  = mlp_model
            st.session_state.hog_mlp_metrics = mlp_metrics
            # backward compat
            st.session_state.model       = mlp_model
            st.session_state.metrics     = mlp_metrics
            st.session_state.val_metrics = compute_metrics(y_val, mlp_model.predict(X_val_hog))
            st.session_state.history     = mlp_model.training_history
            st.session_state.y_test      = y_test
            st.session_state.y_pred      = y_pred_mlp

            _progress(60, "HOG+MLP done")
            _add_log(
                f"  MLP — Accuracy: {mlp_metrics['accuracy']:.4f}  "
                f"F1: {mlp_metrics['f1']:.4f}"
            )
            ph_mlp.success(
                f"Acc={mlp_metrics['accuracy']:.3f}\nF1={mlp_metrics['f1']:.3f}"
            )

            # ── Step 4: Train HOG+SVM ─────────────────────────────────────────
            status_ph.info("Step 4/5 — Training HOG+SVM...")
            _add_log("--- Step 4: Training HOG+SVM ---")
            ph_svm.warning("Training...")

            svm_model = BirdDroneModel(
                mode="SVM",
                epochs=1,
                lr=0.001,
                num_layers=1,
                hidden_size=128,
                batch_size=32,
            )
            svm_model.train(
                X_train_hog, y_train,
                progress_callback=lambda p: _progress(60 + int(p * 0.15), f"SVM {p:.0f}%"),
                log_callback=_add_log,
            )

            y_pred_svm  = svm_model.predict(X_test_hog)
            svm_metrics = compute_metrics(y_test, y_pred_svm)

            svm_model.save("models/hog_svm.pkl")
            st.session_state.hog_svm_model   = svm_model
            st.session_state.hog_svm_metrics = svm_metrics

            _progress(75, "HOG+SVM done")
            _add_log(
                f"  SVM — Accuracy: {svm_metrics['accuracy']:.4f}  "
                f"F1: {svm_metrics['f1']:.4f}"
            )
            ph_svm.success(
                f"Acc={svm_metrics['accuracy']:.3f}\nF1={svm_metrics['f1']:.3f}"
            )

            # ── Step 5: Launch YOLO training (non-blocking subprocesses) ──────
            status_ph.info("Step 5/5 — Launching YOLO training in background...")
            _add_log("--- Step 5: Launching YOLO subprocesses ---")

            python_exe = sys.executable

            # Redirect subprocess output to log files — NEVER use subprocess.PIPE
            # without draining it, or the OS pipe buffer (~64KB) fills and the
            # YOLO training process blocks (deadlocks) on its next print.
            _log_dir = ROOT / "logs"
            _log_dir.mkdir(exist_ok=True)
            _v8_log  = _log_dir / "train_yolov8.log"
            _v10_log = _log_dir / "train_yolov10.log"

            v8_cmd = [
                python_exe, "scripts/train_yolov8.py",
                "--merged",
                "--epochs", str(yolo_v8_epochs),
                "--device", "cpu",
            ]
            v10_cmd = [
                python_exe, "scripts/train_yolov10.py",
                "--merged",
                "--epochs", str(yolo_v10_epochs),
                "--device", "cpu",
                "--batch", "8",
                "--model", "yolov10n.pt",
                "--name", "drone_v10_cpu",
            ]

            try:
                _v8_fh = open(_v8_log, "w", encoding="utf-8")
                v8_proc = subprocess.Popen(
                    v8_cmd,
                    cwd=str(ROOT),
                    stdout=_v8_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                ph_v8.info("Training in background...")
                _add_log(f"  YOLOv8 launched (PID {v8_proc.pid}) → logs/train_yolov8.log")
            except Exception as exc:
                v8_proc = None
                ph_v8.error(f"Launch failed: {exc}")
                _add_log(f"  YOLOv8 launch error: {exc}")

            try:
                _v10_fh = open(_v10_log, "w", encoding="utf-8")
                v10_proc = subprocess.Popen(
                    v10_cmd,
                    cwd=str(ROOT),
                    stdout=_v10_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                ph_v10.info("Training in background...")
                _add_log(f"  YOLOv10 launched (PID {v10_proc.pid}) → logs/train_yolov10.log")
            except Exception as exc:
                v10_proc = None
                ph_v10.error(f"Launch failed: {exc}")
                _add_log(f"  YOLOv10 launch error: {exc}")

            st.session_state.yolo_train_procs = {
                k: p for k, p in [("YOLOv8n", v8_proc), ("YOLOv10", v10_proc)]
                if p is not None
            }

            elapsed = time.time() - st.session_state.get("train_start", time.time())
            st.session_state.train_elapsed = elapsed
            st.session_state.training_logs = live_logs

            _progress(100, "HOG training complete. YOLO running in background.")
            status_ph.success(
                f"HOG training complete in {elapsed:.1f}s — "
                f"MLP Acc={mlp_metrics['accuracy']:.4f}  "
                f"SVM Acc={svm_metrics['accuracy']:.4f}. "
                f"YOLO training launched in background."
            )
            st.balloons()

        except Exception as exc:
            status_ph.error(f"Training failed: {exc}")
            _add_log("--- ERROR ---")
            _add_log(traceback.format_exc())
            st.session_state.training_logs = live_logs

        finally:
            st.session_state.is_training = False

    # ── YOLO background status section ───────────────────────────────────────
    st.divider()
    st.subheader("YOLO Training Status")

    _yolo_procs: dict = st.session_state.get("yolo_train_procs", {})

    refresh_yolo = st.button("Refresh YOLO Status", key="refresh_yolo_top")

    _V8_RUN_NAME  = "drone_v8"
    _V10_RUN_NAME = "drone_v10_cpu"

    sc1, sc2 = st.columns(2)

    for col, name, run_name, weights_path in [
        (sc1, "YOLOv8n",  _V8_RUN_NAME,  _YOLO8_WEIGHTS),
        (sc2, "YOLOv10",  _V10_RUN_NAME, _YOLO10_WEIGHTS_ALT),
    ]:
        with col:
            with st.expander(f"{name} — details", expanded=True):
                proc = _yolo_procs.get(name)
                if proc is None:
                    st.info("Not started this session.")
                else:
                    ret = proc.poll()
                    if ret is None:
                        st.warning("Running...")
                    elif ret == 0:
                        st.success("Complete (exit 0)")
                    else:
                        st.error(f"Exited with code {ret}")

                latest = _read_yolo_latest_epoch(run_name)
                st.caption(f"Latest: {latest}")

                # Tail the training log file (last 15 lines)
                _logf = ROOT / "logs" / (
                    "train_yolov8.log" if name == "YOLOv8n" else "train_yolov10.log"
                )
                if _logf.exists():
                    try:
                        _tail = _logf.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:]
                        if _tail:
                            st.code("\n".join(_tail), language=None)
                    except Exception:
                        pass

                if weights_path.exists():
                    st.success(f"Weights: {weights_path}")
                else:
                    # Also check nested path for v10
                    if name == "YOLOv10" and _YOLO10_WEIGHTS.exists():
                        st.success(f"Weights: {_YOLO10_WEIGHTS}")
                    else:
                        st.info("No weights file yet.")


# =============================================================================
# TAB 2 — Results
# =============================================================================
with tab_results:
    st.subheader("Evaluation Results")

    metrics = st.session_state.metrics
    history = st.session_state.history

    if metrics is None:
        st.info("Train a model first to see results here.")
    else:
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
            cm    = metrics["confusion_matrix"]
            n_cl  = cm.shape[0]
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
        s1.metric("Total Samples",  f"{metrics['n_samples']:,}")
        s2.metric("True Positives", f"{metrics['n_true_positive']:,}")
        s3.metric("Pred Positives", f"{metrics['n_positive']:,}")

        with st.expander("Full Classification Report"):
            st.caption("Labels: 0=Background  1=Bird  2=Drone")
            st.code(metrics["report"], language=None)

    # ── Model Comparison section ──────────────────────────────────────────────
    st.divider()
    st.subheader("Model Comparison")

    _METRICS_PATH = Path("results/all_metrics.json")
    if _METRICS_PATH.exists():
        try:
            all_metrics = json.loads(_METRICS_PATH.read_text())
            rows = []
            for model_name, m in all_metrics.items():
                rows.append({
                    "Model":     model_name,
                    "Accuracy":  f"{m['accuracy']:.4f}"  if m.get("accuracy")  else "—",
                    "Precision": f"{m['precision']:.4f}" if m.get("precision") else "—",
                    "Recall":    f"{m['recall']:.4f}"    if m.get("recall")    else "—",
                    "F1":        f"{m['f1']:.4f}"        if m.get("f1")        else "—",
                    "mAP50":     f"{m['map50']:.4f}"     if m.get("map50")     else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Generated by scripts/evaluate_all.py")
        except Exception as exc:
            st.warning(f"Could not parse {_METRICS_PATH}: {exc}")
    else:
        st.info(
            "No comparison data yet. "
            "Run `python scripts/evaluate_all.py` after training all models."
        )

    # Current session HOG metrics
    if st.session_state.get("hog_mlp_metrics") or st.session_state.get("hog_svm_metrics"):
        st.markdown("**Current Session HOG Metrics**")
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            st.markdown("*HOG+MLP*")
            m = st.session_state.get("hog_mlp_metrics")
            if m:
                st.metric("Accuracy", f"{m['accuracy']:.4f}")
                st.metric("F1",       f"{m['f1']:.4f}")
        with hcol2:
            st.markdown("*HOG+SVM*")
            m = st.session_state.get("hog_svm_metrics")
            if m:
                st.metric("Accuracy", f"{m['accuracy']:.4f}")
                st.metric("F1",       f"{m['f1']:.4f}")


# =============================================================================
# TAB 3 — Live Detection (all 4 models simultaneously)
# =============================================================================
with tab_live:
    st.subheader("Live Detection — All Models")
    st.info(
        "Upload an image to run all 4 detection methods simultaneously. "
        "Models not yet trained/loaded will show a placeholder."
    )

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="live_uploader_all",
    )

    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        bgr_image  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if bgr_image is None:
            st.error("Could not decode image.")
        else:
            h_img, w_img = bgr_image.shape[:2]
            st.caption(f"Image size: {w_img} x {h_img} px")

            # Show original thumbnail
            thumb_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            st.image(thumb_rgb, caption="Original", width=300)

            run_all = st.button("Run All Detections", type="primary", use_container_width=True)

            if run_all:
                col_mlp, col_svm, col_v8, col_v10 = st.columns(4)

                # ── HOG+MLP ──────────────────────────────────────────────────
                with col_mlp:
                    st.markdown("**HOG+MLP**")
                    hog_mlp_model = st.session_state.hog_mlp_model
                    if hog_mlp_model is None:
                        st.warning("Not trained yet")
                    else:
                        with st.spinner("Running..."):
                            try:
                                mlp_dets = hog_detect(
                                    bgr_image,
                                    hog_mlp_model,
                                    seq_length=seq_length,
                                    confidence_threshold=conf_threshold,
                                )
                                st.image(
                                    draw_colored_boxes(bgr_image, mlp_dets),
                                    use_container_width=True,
                                )
                                st.caption(f"{len(mlp_dets)} detection(s)")
                                if mlp_dets:
                                    top = max(mlp_dets, key=lambda d: d[4])
                                    st.caption(f"Top: {top[5]} {top[4]*100:.1f}%")
                            except Exception as exc:
                                st.error(f"Error: {exc}")

                # ── HOG+SVM ──────────────────────────────────────────────────
                with col_svm:
                    st.markdown("**HOG+SVM**")
                    hog_svm_model = st.session_state.hog_svm_model
                    if hog_svm_model is None:
                        st.warning("Not trained yet")
                    else:
                        with st.spinner("Running..."):
                            try:
                                svm_dets = hog_detect(
                                    bgr_image,
                                    hog_svm_model,
                                    seq_length=seq_length,
                                    confidence_threshold=conf_threshold,
                                )
                                st.image(
                                    draw_colored_boxes(bgr_image, svm_dets),
                                    use_container_width=True,
                                )
                                st.caption(f"{len(svm_dets)} detection(s)")
                                if svm_dets:
                                    top = max(svm_dets, key=lambda d: d[4])
                                    st.caption(f"Top: {top[5]} {top[4]*100:.1f}%")
                            except Exception as exc:
                                st.error(f"Error: {exc}")

                # ── YOLOv8n ──────────────────────────────────────────────────
                with col_v8:
                    st.markdown("**YOLOv8n**")
                    if not _YOLO8_AVAILABLE:
                        st.warning("ultralytics not installed")
                    elif not _YOLO8_WEIGHTS.exists():
                        st.warning(f"No weights:\n`{_YOLO8_WEIGHTS}`")
                    else:
                        # Load / cache detector
                        w8_str = str(_YOLO8_WEIGHTS)
                        if st.session_state.get("_yolo8_weights_loaded") != w8_str:
                            with st.spinner("Loading YOLOv8..."):
                                try:
                                    st.session_state["_yolo8_detector"] = YOLOv8Detector(
                                        weights=w8_str, conf=yolo_conf, device="cpu"
                                    )
                                    st.session_state["_yolo8_weights_loaded"] = w8_str
                                except Exception as exc:
                                    st.error(f"Load failed: {exc}")
                                    st.session_state["_yolo8_detector"] = None

                        det8 = st.session_state.get("_yolo8_detector")
                        if det8 is None:
                            st.error("Model not loaded")
                        else:
                            with st.spinner("Running..."):
                                try:
                                    dets8 = det8.detect(bgr_image, conf=yolo_conf)
                                    ann8_bgr = det8.annotate(bgr_image, conf=yolo_conf)
                                    ann8_rgb = cv2.cvtColor(ann8_bgr, cv2.COLOR_BGR2RGB)
                                    st.image(ann8_rgb, use_container_width=True)
                                    st.caption(f"{len(dets8)} detection(s)")
                                    if dets8:
                                        top = max(dets8, key=lambda d: d[4])
                                        st.caption(f"Top: {top[5]} {top[4]*100:.1f}%")
                                except Exception as exc:
                                    st.error(f"Error: {exc}")

                # ── YOLOv10 ──────────────────────────────────────────────────
                with col_v10:
                    st.markdown("**YOLOv10**")
                    if not _YOLO10_AVAILABLE:
                        st.warning("ultralytics not installed")
                    else:
                        # Resolve weights path
                        _w10 = (
                            _YOLO10_WEIGHTS if _YOLO10_WEIGHTS.exists()
                            else _YOLO10_WEIGHTS_ALT if _YOLO10_WEIGHTS_ALT.exists()
                            else None
                        )
                        if _w10 is None:
                            st.warning("No weights file yet")
                        else:
                            w10_str = str(_w10)
                            if st.session_state.get("_yolo_weights_loaded") != w10_str:
                                with st.spinner("Loading YOLOv10..."):
                                    try:
                                        st.session_state["_yolo10_detector"] = YOLOv10Detector(
                                            weights=w10_str, conf=yolo_conf, device="cpu"
                                        )
                                        st.session_state["_yolo_weights_loaded"] = w10_str
                                    except Exception as exc:
                                        st.error(f"Load failed: {exc}")
                                        st.session_state["_yolo10_detector"] = None

                            det10 = st.session_state.get("_yolo10_detector")
                            if det10 is None:
                                st.error("Model not loaded")
                            else:
                                with st.spinner("Running..."):
                                    try:
                                        dets10 = det10.detect(bgr_image, conf=yolo_conf)
                                        ann10_bgr = det10.annotate(bgr_image, conf=yolo_conf)
                                        ann10_rgb = cv2.cvtColor(ann10_bgr, cv2.COLOR_BGR2RGB)
                                        st.image(ann10_rgb, use_container_width=True)
                                        st.caption(f"{len(dets10)} detection(s)")
                                        if dets10:
                                            top = max(dets10, key=lambda d: d[4])
                                            st.caption(f"Top: {top[5]} {top[4]*100:.1f}%")
                                    except Exception as exc:
                                        st.error(f"Error: {exc}")


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
        if not loaded.training_history.get("loss"):
            st.session_state.metrics = None
        st.sidebar.success(f"Loaded {loaded.mode} model from {model_path}")
        st.rerun()
    except FileNotFoundError:
        st.sidebar.error(f"File not found: {model_path}")
    except Exception as exc:
        st.sidebar.error(f"Load failed: {exc}")
