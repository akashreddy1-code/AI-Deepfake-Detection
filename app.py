"""
app.py - STREAMLIT DEMO FOR THE 3-CLASS DEEPFAKE DETECTOR.

    streamlit run app.py

- Upload an image containing a face (jpg / jpeg / png / webp).
- The image is shown, its face is detected and cropped, and the SAME
  preprocessing pipeline used during training (preprocess.py) prepares it
  for the model (the same pipeline predict.py uses).
- Displays the three probabilities (REAL / AI-GENERATED / FAKE/DEEPFAKE),
  the final prediction and the confidence.
- Shows a low-confidence warning when the top probability is below 70%.
- Shows the REAL test metrics measured on the held-out test split, only if
  results/test_metrics.json (or models/test_metrics.json per config.json)
  exists. No accuracy is invented.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException
from tensorflow import keras
from tensorflow.config.optimizer import set_jit
from tensorflow.config.threading import (set_intra_op_parallelism_threads,
                                         set_inter_op_parallelism_threads)

try:
    set_jit(True)
    set_intra_op_parallelism_threads(8)
    set_inter_op_parallelism_threads(2)
except Exception:
    pass

import calibrate
import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

ALLOWED = ("jpg", "jpeg", "png", "webp", "bmp")

SMOOTH_WINDOW = 5
CONFIDENCE_WARNING = 0.60
LOW_CONFIDENCE_MESSAGE = "LOW CONFIDENCE - MANUAL REVIEW RECOMMENDED"

CLASS_EXPLANATIONS = {
    "real": ("REAL", "The image appears consistent with the real-image "
                    "patterns learned by the model."),
    "ai_generated": ("AI-GENERATED", "The image contains patterns that the "
                                      "model associates with AI-generated "
                                      "imagery."),
    "fake_deepfake": ("FAKE-DEEPFAKE", "The image contains patterns that the "
                                        "model associates with "
                                        "manipulated / deepfake imagery "
                                        "(e.g. FaceSwap, FaceForensics++, Celeb-DF)."),
}

DISCLAIMER_TEXT = ("Model predictions are not forensic proof and may be "
                   "uncertain.")


@st.cache_resource
def load_config(config_path=CONFIG_PATH):
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_models(config):
    rel_paths = list(config.get("model_ensemble") or [config["model_path"]])
    loaded = []
    for rel in rel_paths:
        model_path = os.path.join(BASE_DIR, rel)
        if not os.path.exists(model_path):
            return None, "Model file not found: `%s`" % rel
        try:
            model = keras.models.load_model(model_path)
            calib = calibrate.load_calibration(model_path)
        except Exception as exc:
            return None, "Failed to load model from `%s` (%s)." % (rel, exc)
        loaded.append((model, calib))
    return loaded, ""


def predict_probs(models, batch):
    total = None
    n = 0
    for entry in models:
        model = _unwrap_model(entry)
        if model is None:
            continue
        calib = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else None
        raw = np.asarray(model.predict(batch, verbose=0))[0]
        flipped = np.asarray(model.predict(batch[:, :, ::-1, :], verbose=0))[0]
        raw = 0.5 * raw + 0.5 * flipped
        p = calibrate.apply_calibration(raw, calib) if calib else np.asarray(raw, dtype=np.float64)
        total = p if total is None else total + p
        n += 1
    return np.asarray(total) / max(1, n)


def _unwrap_model(entry):
    if isinstance(entry, (tuple, list)):
        if not entry:
            return None
        entry = entry[0]
    if hasattr(entry, "output_shape") or hasattr(entry, "outputs"):
        return entry
    return None


def _model_n_classes(model):
    try:
        shape = getattr(model, "output_shape", None)
        if shape:
            return int(shape[-1])
    except (TypeError, ValueError, AttributeError):
        pass
    try:
        outputs = getattr(model, "outputs", None)
        if outputs:
            return int(outputs[0].shape[-1])
    except (TypeError, ValueError, AttributeError, IndexError):
        pass
    return None


def find_metrics_path(config):
    primary = os.path.join(BASE_DIR, config["metrics_path"])
    if os.path.exists(primary):
        return primary
    for fallback in [
        os.path.join(BASE_DIR, "results", "test_metrics.json"),
        os.path.join(BASE_DIR, "models", "test_metrics.json"),
    ]:
        if os.path.exists(fallback):
            return fallback
    return None


@st.cache_data
def load_test_metrics(config):
    metrics_path = find_metrics_path(config)
    if metrics_path is None:
        return None
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def render_probabilities(classes, labels, probs):
    for cls in classes:
        pct = float(probs[classes.index(cls)]) * 100
        st.write(f"**{labels[cls]}** - {pct:.1f}%")
        st.progress(min(1.0, float(probs[classes.index(cls)])))


def prediction_card_html(labels, best_cls, conf, title="Final Prediction"):
    return (
        "<div style='padding:0.8rem 1rem;border-radius:0.5rem;"
        "background:rgba(70,140,255,0.15);border:1px solid "
        "rgba(70,140,255,0.45);'>"
        "<div style='font-size:0.85rem;opacity:0.75'>{title}</div>"
        "<div style='font-size:1.6rem;font-weight:700'>"
        "{cls}</div>"
        "<div style='font-size:0.95rem;'>Confidence: "
        "{conf}</div></div>"
    ).format(title=title, cls=labels[best_cls], conf="%.2f%%" % (conf * 100))


def detect_input_size(models):
    m = _unwrap_model(models[0])
    inp = None
    try:
        inp = m.input_shape
    except (AttributeError, TypeError, ValueError):
        try:
            inp = tuple(t.shape for t in m.inputs)[0]
        except Exception:
            inp = None
    if inp is not None and len(inp) >= 4:
        try:
            h, w = int(inp[1]), int(inp[2])
            if h > 0 and w > 0:
                return (h, w)
        except (TypeError, ValueError):
            pass
    return tuple(preprocess.IMAGE_SIZE)


def classify_bytes(data, models, input_size=preprocess.IMAGE_SIZE):
    img = preprocess.decode_from_bytes(data)
    if img is None:
        return (None, False, None, "Unable to read this image.")
    err = preprocess.validate_image(img)
    if err:
        return None, False, None, err
    rgb, face_found, crop_bgr, _box = preprocess.predict_face(img, target_size=input_size)
    batch = preprocess.normalize_batch(rgb)
    probs = predict_probs(models, batch)
    cfg = load_config()
    model_output_order = cfg.get("model_output_order", cfg["classes"])
    if model_output_order != cfg["classes"]:
        order_map = [model_output_order.index(cls) for cls in cfg["classes"]]
        probs = probs[order_map]
    return probs, face_found, crop_bgr, ""


def classify_frame(img_bgr, models, input_size=preprocess.IMAGE_SIZE):
    if img_bgr is None:
        return None, False, None, None, None, 0, "No frame captured."
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    err = preprocess.validate_image(img_bgr)
    if err:
        return None, False, None, None, None, 0, err
    faces = preprocess.detect_faces(img_bgr)
    n_faces = len(faces)
    box = faces[0] if faces else None
    crop_bgr = preprocess.crop_face(img_bgr, box) if box else img_bgr
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, input_size)
    batch = preprocess.normalize_batch(rgb)
    if not faces:
        return None, False, crop_bgr, box, batch, 0, ""
    probs = predict_probs(models, batch)
    cfg = load_config()
    model_output_order = cfg.get("model_output_order", cfg["classes"])
    if model_output_order != cfg["classes"]:
        order_map = [model_output_order.index(cls) for cls in cfg["classes"]]
        probs = probs[order_map]
    return probs, True, crop_bgr, box, batch, n_faces, ""


def render_upload(classes, labels, models,
                  conf_threshold=CONFIDENCE_WARNING,
                  input_size=preprocess.IMAGE_SIZE):
    uploaded = st.file_uploader("Upload an image containing a face.", type=list(ALLOWED))
    if uploaded is None:
        st.write("Waiting for an image...")
        return
    data = uploaded.read()
    try:
        probs, face_found, crop_bgr, err = classify_bytes(data, models, input_size)
    except Exception as exc:
        probs, err = None, "Prediction failed (%s)." % exc
    if err or probs is None:
        st.error(err)
        return
    img_bgr = preprocess.decode_from_bytes(data)
    display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.subheader("Uploaded image")
        st.image(display_rgb, width="stretch")
        face_status = "YES" if face_found else "NO"
        st.write(f"**Face Detected:** {face_status}")
        if face_found:
            st.success("Face detected - the face region was cropped (same preprocessing as training).")
        else:
            st.warning("No face detected. Please upload a clear image containing a visible face.")
            st.write("The whole image was used - result may be unreliable.")
        if crop_bgr is not None:
            st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                     caption="Input actually fed to the model (resized to 224x224)", width=224)
    idx = int(np.argmax(probs))
    best_cls = classes[idx]
    conf = float(probs[idx])
    with col_res:
        st.subheader("Prediction")
        st.markdown(prediction_card_html(labels, best_cls, conf), unsafe_allow_html=True)
        if conf < conf_threshold:
            st.warning(LOW_CONFIDENCE_MESSAGE)
        else:
            st.success(f"Confidence {conf * 100:.1f}% - prediction shown normally.")
        st.markdown("**Probabilities**")
        render_probabilities(classes, labels, probs)
    with st.expander("Debug information", expanded=False):
        img_bgr = preprocess.decode_from_bytes(data)
        box = preprocess.detect_face(img_bgr) if img_bgr is not None else None
        st.write("**Detected face bounding box (x, y, w, h):** %s" % (box if box is not None else "none - full image used",))
        if crop_bgr is not None:
            st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption="Face region actually fed to the model", width=224)
            st.write("**Processed (crop) image size (H, W, C):** %s" % (crop_bgr.shape,))
        st.write("**Model input shape:** (batch, %d, %d, %d)" % (input_size[0], input_size[1], 3))
        st.write("**Preprocessing target size:** %s" % (input_size,))
        st.write("**Class mapping (index -> class -> display name):**")
        for i, cls in enumerate(classes):
            st.write("  - %d -> `%s` -> **%s**" % (i, cls, labels[cls]))
        st.write("**Prediction probabilities (per class):**")
        for i, cls in enumerate(classes):
            st.write("  - %s: %.4f (%.2f%%)" % (labels[cls], float(probs[i]), float(probs[i]) * 100))
        st.write("**Final prediction:** **%s**" % labels[best_cls])
        st.write("**Confidence:** %.2f%%" % (conf * 100))
        st.write("**Confidence threshold:** %.0f%%" % (conf_threshold * 100))


_VALIDATION_CACHE = {}


def model_cache_key(config):
    rel_paths = list(config.get("model_ensemble") or [config["model_path"]])
    key = []
    for rel in rel_paths:
        p = os.path.join(BASE_DIR, rel)
        try:
            key.append((rel, os.path.getmtime(p)))
        except OSError:
            key.append((rel, None))
    return tuple(key)


def _compute_validation_metrics(models):
    t0 = time.time()
    import train_model as tm
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score)
    cfg = tm.load_config()
    classes = list(cfg["classes"])
    _, _, test_data, _, _ = tm.collect_split(cfg, cfg["seed"])
    ds = tm.make_dataset(test_data, augment=False, shuffle=False)
    entries = []
    for entry in models:
        m = _unwrap_model(entry)
        calib = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else None
        entries.append((m, calib))
    y_true, prob_blocks = [], []
    for xb, yb in ds:
        y_true.extend(np.argmax(yb.numpy(), axis=1).tolist())
        avg = None
        for m, calib in entries:
            raw = np.asarray(m.predict(xb, verbose=0))
            flipped = np.asarray(m.predict(xb[:, :, ::-1, :], verbose=0))
            raw = 0.5 * raw + 0.5 * flipped
            p = calibrate.apply_calibration(raw, calib)
            avg = p if avg is None else avg + p
        prob_blocks.append(np.asarray(avg) / max(1, len(entries)))
    probs = np.vstack(prob_blocks)
    y_true = np.asarray(y_true)
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    return {
        "classes": classes, "n_test": int(len(y_true)),
        "elapsed_s": round(time.time() - t0, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "report_text": classification_report(y_true, y_pred, target_names=classes, zero_division=0),
        "per_class": {cls: {"precision": float(report_dict[cls]["precision"]),
                            "recall": float(report_dict[cls]["recall"]),
                            "f1-score": float(report_dict[cls]["f1-score"]),
                            "support": int(report_dict[cls]["support"])}
                      for cls in classes},
    }


def get_validation_metrics(models):
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    key = model_cache_key(cfg)
    if key not in _VALIDATION_CACHE:
        _VALIDATION_CACHE[key] = _compute_validation_metrics(models)
    return _VALIDATION_CACHE[key]


def plot_confusion_matrix(cm, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([labels_short(c) for c in classes], rotation=20, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([labels_short(c) for c in classes])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - held-out test split")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def labels_short(cls):
    return {"real": "REAL", "ai_generated": "AI-GEN", "fake_deepfake": "FAKE-DF"}.get(cls, cls)


def list_test_images(cls):
    folder = os.path.join(BASE_DIR, "processed_dataset", "test", cls)
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in ALLOWED)


def render_comprehensive_validation():
    val_path = os.path.join(BASE_DIR, "results", "validation_summary.txt")
    val_json = os.path.join(BASE_DIR, "results", "validation_results.json")
    if not os.path.exists(val_path):
        st.info("Comprehensive validation not yet run. Run `python validate_comprehensive.py` to generate results.")
        return
    with open(val_path, encoding="utf-8") as f:
        val_text = f.read()
    st.markdown("### Comprehensive Validation Summary")
    st.text(val_text)
    if os.path.exists(val_json):
        import json
        with open(val_json, encoding="utf-8") as f:
            vdata = json.load(f)
        dv = vdata.get("dataset_validation")
        if dv:
            st.subheader("Dataset Validation Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", "%.2f%%" % (dv["accuracy"] * 100))
            c2.metric("Precision (macro)", "%.4f" % dv["precision_macro"])
            c3.metric("Recall (macro)", "%.4f" % dv["recall_macro"])
            c4.metric("F1-score (macro)", "%.4f" % dv["f1_macro"])
            st.write("**Samples tested:** %d | **Correct:** %d | **Incorrect:** %d" % (dv["total_samples"], dv["correct"], dv["incorrect"]))
            cm = np.array(dv["confusion_matrix"])
            st.write("**Confusion Matrix:**")
            st.table(pd.DataFrame(cm, index=["REAL", "AI-GENERATED", "FAKE-DEEPFAKE"], columns=["REAL", "AI-GENERATED", "FAKE-DEEPFAKE"]))


def render_validation(classes, labels, models, conf_threshold, input_size=preprocess.IMAGE_SIZE):
    st.subheader("TESTING & VALIDATION")
    st.caption("Evaluation runs on `processed_dataset/test` - the held-out 15% split that was NEVER used for training or validation.")
    test_root = os.path.join(BASE_DIR, "processed_dataset", "test")
    if not os.path.isdir(test_root):
        st.error("`processed_dataset/test` not found. Build it first with `python clean_preprocess_dataset.py`.")
        return
    col_run, col_info = st.columns([1, 2])
    with col_run:
        rerun = st.button("Re-run evaluation", help="The result is computed once and cached. Press this to force a fresh measurement.")
    with col_info:
        st.write("First run scores every held-out test image once (about a minute on CPU); afterwards it is instant.")
    if rerun:
        _VALIDATION_CACHE.clear()
    try:
        with st.spinner("Evaluating the held-out test split..."):
            vm = get_validation_metrics(models)
    except Exception as exc:
        st.error("Evaluation failed: %s" % exc)
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "%.2f%%" % (vm["accuracy"] * 100))
    c2.metric("Precision (macro)", "%.4f" % vm["precision_macro"])
    c3.metric("Recall (macro)", "%.4f" % vm["recall_macro"])
    c4.metric("F1-score (macro)", "%.4f" % vm["f1_macro"])
    st.caption("Measured on %d held-out test images in %.1f s." % (vm["n_test"], vm["elapsed_s"]))
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Classification report**")
        st.code(vm["report_text"], language="text")
    with right:
        st.markdown("**Confusion matrix** (rows = true, cols = predicted)")
        cm = np.asarray(vm["confusion_matrix"])
        st.pyplot(plot_confusion_matrix(cm, classes), width="stretch")
        st.markdown("**Per-class accuracy**")
        for cls in classes:
            pc = vm["per_class"][cls]
            st.write("- **%s**: accuracy %.2f%% (%d/%d correct) - precision %.3f - F1 %.3f" % (labels[cls], pc["recall"] * 100, round(pc["recall"] * pc["support"]), pc["support"], pc["precision"], pc["f1-score"]))
    st.markdown("---")
    st.subheader("Test individual images")
    st.caption("Pick any image from the held-out TEST split of each class and run it through exactly the same pipeline as the Image Upload tab.")
    sel_cls = st.selectbox("Class", classes, format_func=lambda c: labels[c], key="val_class_select")
    files = list_test_images(sel_cls)
    if not files:
        st.warning("No test images found for %s." % labels[sel_cls])
        return
    sel_file = st.selectbox("Test image (%d available)" % len(files), files, key="val_file_select")
    if st.button("Run prediction"):
        path = os.path.join(test_root, sel_cls, sel_file)
        try:
            with open(path, "rb") as f:
                data = f.read()
            probs, face_found, crop_bgr, err = classify_bytes(data, models, input_size)
        except Exception as exc:
            probs, face_found, crop_bgr, err = None, False, None, "Prediction failed (%s)." % exc
        if err or probs is None:
            st.error(err or "Prediction failed.")
            return
        idx = int(np.argmax(probs))
        best_cls = classes[idx]
        conf = float(probs[idx])
        cc1, cc2 = st.columns(2)
        with cc1:
            if crop_bgr is not None:
                st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption="Face fed to the model", width=224)
            st.write("**True class:** %s" % labels[sel_cls])
            st.write("**Face detected:** %s" % ("YES" if face_found else "NO"))
        with cc2:
            st.markdown(prediction_card_html(labels, best_cls, conf, title="Model Prediction"), unsafe_allow_html=True)
            if conf < conf_threshold:
                st.warning(LOW_CONFIDENCE_MESSAGE)
            ok = best_cls == sel_cls
            (st.success if ok else st.error)("CORRECT - matches the true class" if ok else "WRONG - true class is %s" % labels[sel_cls])
            render_probabilities(classes, labels, probs)
    render_comprehensive_validation()


def _render_debug_info():
    st.markdown("---")
    st.markdown("### Debug info")
    c1, c2, c3, c4 = st.columns(4)
    c1.write("**Camera opened:** %s" % ("YES" if st.session_state.camera_opened else "NO"))
    c2.write("**Frame received:** %s" % ("YES" if st.session_state.frame_received else "NO"))
    c3.write("**Face detected:** %s" % ("YES" if st.session_state.face_detected else "NO"))
    c4.write("**Prediction completed:** %s" % ("YES" if st.session_state.prediction_completed else "NO"))
    st.write("**Frame number:** %d" % st.session_state.frame_count)
    st.write("**Camera process PID:** %s" % (st.session_state.pid if st.session_state.pid else "N/A"))
    st.write("**Recent log:** %s" % st.session_state.recent_log)


def render_live_camera(classes, labels, models, conf_threshold, input_size):
    if 'camera_worker_proc' not in st.session_state:
        st.session_state.camera_worker_proc = None
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'frame_path' not in st.session_state:
        st.session_state.frame_path = None
    if 'stop_file' not in st.session_state:
        st.session_state.stop_file = None
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'pred_history' not in st.session_state:
        st.session_state.pred_history = deque(maxlen=SMOOTH_WINDOW)
    if 'camera_opened' not in st.session_state:
        st.session_state.camera_opened = False
    if 'frame_received' not in st.session_state:
        st.session_state.frame_received = False
    if 'face_detected' not in st.session_state:
        st.session_state.face_detected = False
    if 'prediction_completed' not in st.session_state:
        st.session_state.prediction_completed = False
    if 'recent_log' not in st.session_state:
        st.session_state.recent_log = ""
    if 'pid' not in st.session_state:
        st.session_state.pid = None
    if 'tmp_dir' not in st.session_state:
        st.session_state.tmp_dir = None

    st.subheader("LIVE CAMERA")
    st.caption("Webcam → detect face → preprocessing → model → smoothing")

    show_debug = st.checkbox("✓ Show live debug info")

    col_start, col_stop = st.columns([1, 1])
    with col_start:
        if st.button("▶ Start Camera"):
            if st.session_state.camera_worker_proc is None or st.session_state.camera_worker_proc.poll() is not None:
                tmp_dir = tempfile.mkdtemp()
                frame_path = os.path.join(tmp_dir, "frame.jpg")
                stop_file = os.path.join(tmp_dir, "stop.txt")
                proc = subprocess.Popen(
                    [sys.executable, "camera_worker.py",
                     "--out", frame_path,
                     "--stop", stop_file],
                    cwd=BASE_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                st.session_state.camera_worker_proc = proc
                st.session_state.frame_path = frame_path
                st.session_state.stop_file = stop_file
                st.session_state.tmp_dir = tmp_dir
                st.session_state.camera_running = True
                st.session_state.camera_opened = True
                st.session_state.frame_count = 0
                st.session_state.pred_history = deque(maxlen=SMOOTH_WINDOW)
                st.session_state.pid = proc.pid
                st.session_state.recent_log = "Camera opened: YES (PID: %d)" % proc.pid

    with col_stop:
        if st.button("⏹ Stop Camera"):
            if st.session_state.stop_file and os.path.exists(st.session_state.stop_file):
                try:
                    with open(st.session_state.stop_file, 'w') as f:
                        f.write("stop")
                except Exception:
                    pass
            if st.session_state.camera_worker_proc:
                try:
                    st.session_state.camera_worker_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        st.session_state.camera_worker_proc.terminate()
                    except Exception:
                        pass
            st.session_state.camera_running = False
            st.session_state.camera_opened = False
            st.session_state.frame_received = False
            st.session_state.recent_log = "Camera stopped"
            if st.session_state.tmp_dir and os.path.exists(st.session_state.tmp_dir):
                try:
                    shutil.rmtree(st.session_state.tmp_dir)
                except Exception:
                    pass
            st.session_state.tmp_dir = None

    if st.session_state.camera_running:
        frame_path = st.session_state.frame_path

        if st.session_state.camera_worker_proc and st.session_state.camera_worker_proc.poll() is not None:
            st.session_state.camera_running = False
            st.session_state.camera_opened = False
            st.session_state.recent_log = "Camera process terminated unexpectedly"

        frame = None
        if frame_path and os.path.exists(frame_path):
            try:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    st.session_state.frame_received = True
            except Exception:
                st.session_state.frame_received = False
        else:
            st.session_state.frame_received = False

        probs = None
        face_found = False

        if frame is not None:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            st.session_state.frame_count += 1

            if st.session_state.frame_count % 5 == 0:
                result = classify_frame(frame, models, input_size)
                probs, face_found, _, _, _, _, err = result
                if probs is not None:
                    st.session_state.pred_history.append(probs)
                    st.session_state.prediction_completed = True
                else:
                    st.session_state.prediction_completed = False
                    st.session_state.recent_log = "Frame %d: %s" % (st.session_state.frame_count, err)
                st.session_state.face_detected = face_found
                if probs is not None:
                    st.session_state.recent_log = "Frame %d: prediction completed" % st.session_state.frame_count
                else:
                    st.session_state.recent_log = "Frame %d: %s" % (st.session_state.frame_count, err)

        smoothed = None
        best_cls = None
        conf = 0.0

        if len(st.session_state.pred_history) > 0:
            smoothed = np.mean(np.asarray(list(st.session_state.pred_history)), axis=0)
            idx = int(np.argmax(smoothed))
            best_cls = classes[idx]
            conf = float(smoothed[idx])

        left_column, right_column = st.columns([1, 1])

        with left_column:
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, width=640, use_container_width=False)
            st.write("Live webcam frame")
            if st.session_state.face_detected:
                st.markdown(
                    "<div style='padding:0.5rem;border-radius:0.5rem;"
                    "background:rgba(0,255,0,0.15);border:1px solid "
                    "rgba(0,255,0,0.45);'>"
                    "<div style='font-size:1.1rem;font-weight:700;color:green;'>"
                    "✓ Face detected: YES</div></div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='padding:0.5rem;border-radius:0.5rem;"
                    "background:rgba(255,0,0,0.15);border:1px solid "
                    "rgba(255,0,0,0.45);'>"
                    "<div style='font-size:1.1rem;font-weight:700;color:red;'>"
                    "Face detected: NO</div></div>",
                    unsafe_allow_html=True)

        with right_column:
            st.markdown("### LIVE PREDICTION")
            if best_cls is not None:
                st.markdown(
                    "<div style='padding:0.8rem 1rem;border-radius:0.5rem;"
                    "background:rgba(70,140,255,0.15);border:1px solid "
                    "rgba(70,140,255,0.45);'>"
                    "<div style='font-size:0.85rem;opacity:0.75'>"
                    "Smoothed Prediction</div>"
                    "<div style='font-size:1.6rem;font-weight:700'>"
                    "%s</div>"
                    "<div style='font-size:0.95rem;'>Confidence: %.2f%%</div>"
                    "</div>" % (labels[best_cls], conf * 100),
                    unsafe_allow_html=True)
                if conf < conf_threshold:
                    st.markdown(
                        "<div style='padding:0.5rem;border-radius:0.5rem;"
                        "background:rgba(255,255,0,0.15);border:1px solid "
                        "rgba(255,255,0,0.45);'>"
                        "<div style='font-size:1rem;font-weight:700;color:orange;'>"
                        "⚠ LOW CONFIDENCE – MANUAL REVIEW RECOMMENDED</div>"
                        "</div>",
                        unsafe_allow_html=True)
                smoothed_probs = [float(smoothed[i]) for i in range(len(classes))]
                st.markdown("**Smoothed probabilities** (average of latest 5 frames)")
                for i, cls in enumerate(classes):
                    pct = smoothed_probs[i] * 100
                    st.markdown("**%s — %.1f%%**" % (labels[cls], pct))
                    st.progress(min(1.0, smoothed_probs[i]))
            else:
                st.markdown("### LIVE PREDICTION")
                st.write("Waiting for prediction...")
                for cls in classes:
                    st.markdown("**%s — 0.0%%**" % labels[cls])
                    st.progress(0.0)

        if show_debug:
            _render_debug_info()

        time.sleep(0.05)
        st.rerun()
    else:
        if show_debug:
            _render_debug_info()


def main():
    st.set_page_config(page_title="AI Deepfake Detection System", layout="centered")
    config = load_config()
    models, model_err = load_models(config)
    metrics = load_test_metrics(config)
    classes = config["classes"]
    labels = config["class_labels"]
    conf_threshold = float(config.get("uncertainty_threshold", CONFIDENCE_WARNING))
    model_output_order = config.get("model_output_order", classes)
    if model_output_order != classes:
        st.warning("Model output order differs from config class order.")
    st.session_state["model_output_order"] = model_output_order
    st.session_state["config_classes"] = classes
    st.title("AI Deepfake Detection System")
    st.caption("Real / AI-Generated / Fake-Deepfake - 3-class MobileNetV2 classifier")
    if metrics:
        acc = float(metrics.get("accuracy", 0.0))
        p = float(metrics.get("precision_macro", 0.0))
        r = float(metrics.get("recall_macro", 0.0))
        f1 = float(metrics.get("f1_macro", 0.0))
        ece = float(metrics.get("ece_after", float("nan")))
        st.info("**Real test metrics** (measured on the held-out 15% test split): accuracy **{:.2f}%** - macro F1 **{:.4f}** - macro precision **{:.4f}** - macro recall **{:.4f}**".format(acc * 100, f1, p, r) + (f"\n\nCalibration error (ECE): **{ece:.4f}**" if ece == ece else "") + "\n\nThis model is **not** 100% accurate. Treat low-confidence predictions with caution.")
    if models is None:
        st.error(model_err)
        st.stop()
    input_size = detect_input_size(models)
    for _entry in models:
        _m = _unwrap_model(_entry)
        if _m is None:
            st.error("A loaded model entry is not a Keras model.")
            st.stop()
        n_out = _model_n_classes(_m)
        if n_out is None:
            st.error("Could not determine the model's output shape.")
            st.stop()
        if n_out != len(classes):
            st.error("Model has %d outputs but config.json lists %d classes." % (n_out, len(classes)))
            st.stop()
    st.markdown("---")
    tab_upload, tab_live, tab_val = st.tabs(["Image Upload", "Live Camera", "Testing & Validation"])
    with tab_upload:
        render_upload(classes, labels, models, conf_threshold, input_size)
    with tab_live:
        render_live_camera(classes, labels, models, conf_threshold, input_size)
    with tab_val:
        render_validation(classes, labels, models, conf_threshold, input_size)
    st.markdown("---")
    st.subheader("What does each class mean?")
    for cls in classes:
        name, text = CLASS_EXPLANATIONS[cls]
        st.write(f"**{name}:** {text}")
    st.write(f"*{DISCLAIMER_TEXT}*")
    st.markdown("---")
    st.subheader("Model Information")
    info_cols = st.columns(2)
    with info_cols[0]:
        st.write("**Model:** MobileNetV2 (transfer learning)")
        st.write("**Task:** 3-Class Image Classification")
    with info_cols[1]:
        st.write("**Input Size:** %d x %d pixels" % (input_size[0], input_size[1]))
        st.write(f"**Classes:** {' / '.join(labels[cls] for cls in classes)}")
    if metrics:
        st.write(f"**Test Accuracy:** {float(metrics['accuracy']) * 100:.2f}%")
    else:
        st.write("**Test Accuracy:** not available")
    st.markdown("---")
    st.subheader("How it works")
    st.write("1. Upload an image")
    st.write("2. Detect the face")
    st.write("3. Crop and preprocess the face (224x224, normalized)")
    st.write("4. MobileNetV2 analyzes the image")
    st.write("5. Generate the three class probabilities")
    st.write("6. Display the final classification")
    st.markdown("---")
    st.caption("Class order comes from config.json; model output is mapped through it.")


if __name__ == "__main__":
    main()
