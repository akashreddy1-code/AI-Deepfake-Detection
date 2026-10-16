"""
calibrate.py - PROBABILITY CALIBRATION (honest, validation-fitted).

Why this file exists
====================
A deep network's softmax output is often OVERCONFIDENT: for the 3-class
detector this showed up as real photographs being predicted "AI-GENERATED"
with ~85% "confidence" even though the model is wrong about a large share of
them. The numbers a user sees should match how often the model is actually
right, so every displayed probability is calibrated BEFORE it is shown.

Method (calibrated on VALIDATION split ONLY - test split is never touched
for fitting):
    Vector scaling. Given raw softmax probabilities p (or, equivalently, the
    logits z before softmax) the calibrated probability is

        q_i = softmax(s_i * logit_i + b_i)

    with per-class scale s_i and bias b_i fitted by minimizing the negative
    log-likelihood of the validation split (scipy L-BFGS-B). This is a
    standard, cheap, 6-parameter post-hoc calibration (Guo et al., 2017,
    "On Calibration of Modern Neural Networks").

    We fit on log(p + eps) as the "logit" representation, which is exact for
    softmax outputs and works without modifying the saved model.

What is stored
==============
    results/calibration.json:
        {
          "type": "vector_scaling",
          "fit_split": "validation",
          "scale": [s0, s1, s2],
          "bias":   [b0, b1, b2],
          "ece_before": ..., "ece_after": ...,   # 10-bin ECE, val split
          "n_fit": ...,                          # validation image count
          "note": "..."
        }

Anything that shows probabilities (predict.py, app.py, test_unseen.py,
train_model.py metrics, evaluate_model.py) reads this file through
load_calibration() and applies apply_calibration() so every number a user
sees is the calibrated, honest one.
"""

import json
import os

import numpy as np
from scipy.optimize import minimize

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH  = os.path.join(BASE_DIR, "results", "calibration.json")

# The legacy results/calibration.json was fitted for THIS model (the model
# that config.json pointed at when it was created). Every other model gets
# its own file: results/calibration_<model_stem>.json.
LEGACY_CALIB_MODEL_STEM = "deepfake_detector"

EPS = 1e-12


def calibration_path_for(model_path):
    """Per-model calibration file path:
    results/calibration_<keras file stem>.json."""
    stem = os.path.splitext(os.path.basename(str(model_path)))[0]
    return os.path.join(BASE_DIR, "results", "calibration_%s.json" % stem)


# ---------------------------------------------------------------------------
# ECE (expected calibration error, 10 bins) - standard measure
# ---------------------------------------------------------------------------

def ece(probs, y_true, n_bins=10):
    """probs: (n, C) probabilities; y_true: (n,) class indices."""
    probs   = np.asarray(probs, dtype=np.float64)
    y_true  = np.asarray(y_true)
    conf    = np.max(probs, axis=1)
    pred    = np.argmax(probs, axis=1)
    acc     = (pred == y_true).astype(np.float64)
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    err     = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        err += (mask.sum() / len(conf)) * abs(acc[mask].mean() - conf[mask].mean())
    return float(err)


def softmax_rows(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Fit / apply
# ---------------------------------------------------------------------------

def logit_rep(probs):
    """log(p + eps) is an exact logit representation of a softmax output."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0))


def apply_params(logit_rep, scale, bias):
    """q = softmax(scale * logit + bias)  (vector scaling)."""
    return softmax_rows(logit_rep * np.asarray(scale) + np.asarray(bias))


def fit_calibration(val_probs, val_true):
    """Fit vector scaling on the VALIDATION split.

    val_probs : (n, C) raw softmax probabilities
    val_true  : (n,) integer class indices
    Returns the calibration dict (with ECE before/after)."""
    val_probs = np.asarray(val_probs, dtype=np.float64)
    val_true  = np.asarray(val_true)
    L = val_logits = logit_rep(val_probs)
    n_classes = L.shape[1]
    y_oh = np.zeros_like(val_probs)
    y_oh[np.arange(len(val_true)), val_true] = 1.0

    def nll(theta):
        scale = theta[:n_classes]
        bias  = theta[n_classes:]
        q = apply_params(L, scale, bias)
        q = np.clip(q, EPS, 1.0)
        return float(-np.mean(np.sum(y_oh * np.log(q), axis=1)))

    theta0 = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
    bounds = [(1e-3, 1e3)] * n_classes + [(-50.0, 50.0)] * n_classes
    res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000})
    scale = list(np.asarray(res.x[:n_classes]).astype(float))
    bias  = list(np.asarray(res.x[n_classes:]).astype(float))

    # ---- SAFETY GUARD -----------------------------------------------------
    # A calibration is only kept when it genuinely IMPROVES the validation
    # split on BOTH the log-loss (NLL) and the ECE. An earlier fit actually
    # made ECE worse (0.041 -> 0.047) while suppressing the real class
    # (scale 0.58 / bias -0.99), producing "REAL with low confidence"
    # predictions. If the fit does not improve both metrics, it is marked
    # disabled and every consumer falls back to the RAW softmax.
    ece_before = ece(val_probs, val_true)
    calibrated = apply_params(L, scale, bias)
    ece_after = ece(calibrated, val_true)
    nll_before = float(nll(np.concatenate(
        [np.ones(n_classes), np.zeros(n_classes)])))
    improved = (ece_after < ece_before) and (float(res.fun) < nll_before)

    return {
        "type": "vector_scaling",
        "fit_split": "validation",
        "enabled": bool(improved),
        "scale": scale,
        "bias": bias,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "nll_before": nll_before,
        "nll_after": float(res.fun),
        "n_fit": int(len(val_true)),
        "note": ("Probability calibration fitted on the validation split "
                 "(never on test). When enabled=false the raw softmax is "
                 "used unchanged because fitting did not improve either "
                 "ECE or log-loss."),
    }


def save_calibration(calib, path=CALIB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)


def load_calibration(model_path=None, path=None):
    """Load the calibration fitted for the given model.

    Order:
      1. explicit `path` (tests / smoke scripts)
      2. results/calibration_<model_stem>.json for the given model
      3. legacy results/calibration.json - ONLY for the model it was
         originally fitted for (deepfake_detector.keras)
      4. None (raw softmax is shown unchanged - nothing invented)
    """
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if model_path:
        per_model = calibration_path_for(model_path)
        if os.path.exists(per_model):
            with open(per_model, encoding="utf-8") as f:
                return json.load(f)
        stem = os.path.splitext(os.path.basename(str(model_path)))[0]
        if stem != LEGACY_CALIB_MODEL_STEM:
            return None   # no calibration for this model: use raw softmax
    if not os.path.exists(CALIB_PATH):
        return None
    with open(CALIB_PATH, encoding="utf-8") as f:
        return json.load(f)


def apply_calibration(raw_probs, calib=None):
    """Calibrate raw softmax probabilities (n, C) or (C,).

    Without a calibration file the raw probabilities are returned unchanged.
    A calibration whose "enabled" flag is false (it failed the improvement
    guard in fit_calibration) is also ignored — the raw probabilities are
    returned unchanged (safe default: nothing is ever invented)."""
    if calib is None:
        calib = load_calibration()
    if calib is not None and not calib.get("enabled", True):
        calib = None   # disabled fit: never distort the raw softmax
    raw = np.asarray(raw_probs, dtype=np.float64)
    if calib is None:
        return raw
    single = (raw.ndim == 1)
    if single:
        raw = raw[None, :]
    scale = np.asarray(calib["scale"], dtype=np.float64)
    bias  = np.asarray(calib["bias"],  dtype=np.float64)
    out = apply_params(logit_rep(raw), scale, bias)
    return out[0] if single else out


def fit_for_model(model_path):
    """Fit calibration for ONE saved .keras model on the validation split
    (evaluation only - no training, no dataset changes) and save it to the
    per-model file results/calibration_<stem>.json."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras
    import train_model as tm

    if not os.path.exists(model_path):
        raise SystemExit("ERROR: model not found: %s" % model_path)

    cfg = tm.load_config()
    _, val_data, _, _, _ = tm.collect_split(cfg, cfg["seed"])
    model = keras.models.load_model(model_path)
    val_ds = tm.make_dataset(val_data, augment=False, shuffle=False)
    print("Predicting %d validation images with %s ..."
          % (len(val_data), os.path.basename(model_path)), flush=True)
    val_raw = tm.tta_predict(model, val_ds)
    val_true = np.array([c for _, c in val_data])
    calib = fit_calibration(val_raw, val_true)
    out_path = calibration_path_for(model_path)
    save_calibration(calib, out_path)
    print("Saved %s (ECE %.4f -> %.4f, n=%d)"
          % (out_path, calib["ece_before"], calib["ece_after"], calib["n_fit"]))
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Probability calibration utilities.")
    parser.add_argument(
        "--fit-for", metavar="MODEL.keras",
        help="fit calibration for this model on the validation split and "
             "save results/calibration_<stem>.json")
    args = parser.parse_args()

    if args.fit_for:
        mp = args.fit_for if os.path.isabs(args.fit_for) \
            else os.path.join(BASE_DIR, args.fit_for)
        fit_for_model(mp)
    else:
        # smoke test on random data
        rng = np.random.default_rng(0)
        probs = softmax_rows(rng.normal(size=(500, 3)))
        y = np.argmax(probs, axis=1)
        c = fit_calibration(probs, y)
        print("smoke test OK: ECE before=%.4f after=%.4f" %
              (c["ece_before"], c["ece_after"]))