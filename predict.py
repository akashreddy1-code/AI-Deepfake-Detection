"""
predict.py  —  CLASSIFY A SINGLE IMAGE FROM THE COMMAND LINE  (Sprint 7)

Usage:
    python predict.py "path/to/image.jpg"

Output format (Task 11):
    ================================
    AI DEEPFAKE DETECTION
    ================================

    Prediction: REAL / AI-GENERATED / FAKE-DEEPFAKE

    REAL: XX.XX%
    AI-GENERATED: XX.XX%
    FAKE-DEEPFAKE: XX.XX%

    Confidence: XX.XX%

    LOW CONFIDENCE — MANUAL REVIEW RECOMMENDED   (only when confidence is low)
    ================================

Rules:
  - Class mapping is NEVER hard-coded. Read from config.json.
  - Preprocessing is IDENTICAL to training (shared preprocess.py).
  - Predictions are purely from image pixel data. Filename is never used.
  - No 100%% accuracy claimed.
"""

import argparse
import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import calibrate
import preprocess
from tensorflow import keras

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_model(config):
    model_path = os.path.join(BASE_DIR, config["model_path"])
    if not os.path.exists(model_path):
        sys.exit(
            "ERROR: Model not found at %s\n"
            "Run 'python train_model.py' to train the model first." % model_path)
    return keras.models.load_model(model_path)


def classify_image(path, config, model, calib=None):
    """Run the shared preprocessing pipeline + model.
    Returns a result dict. Never uses filename in the prediction logic.

    Test-time augmentation (TTA): the probability is the average of the
    original image and its horizontal flip (same preprocessing for both).
    The averaged softmax is then CALIBRATED with the validation-fitted
    vector scaling so displayed percentages are honest."""
    batch, face_found, _crop, err = preprocess.preprocess_from_path(path)
    if err:
        return {"error": err}

    flipped       = batch[:, :, ::-1, :]                     # horizontal flip
    raw           = 0.5 * model.predict(batch, verbose=0)
    raw          += 0.5 * model.predict(flipped, verbose=0)
    raw           = raw[0]
    probs         = calibrate.apply_calibration(raw, calib)
    calib_active  = bool(calib and calib.get("enabled", True))
    classes       = config["classes"]
    labels        = config["class_labels"]

    prob_map   = {cls: float(p) for cls, p in zip(classes, probs)}
    idx        = int(np.argmax(probs))
    best_cls   = classes[idx]
    confidence = float(probs[idx])
    threshold  = float(config.get("uncertainty_threshold", 0.70))

    return {
        "path"                : path,
        "face_detected"       : face_found,
        "probabilities"       : prob_map,
        "final_prediction"    : labels[best_cls],
        "predicted_class"     : best_cls,
        "confidence"          : confidence,
        "uncertainty_threshold": threshold,
        "uncertain"           : confidence < threshold,
        "calibration_applied" : calib_active,
        "calibration_note"    : ("Probabilities are calibration-corrected "
                                 "(fitted on the validation split)."
                                 if calib_active else
                                 "Raw softmax probabilities (calibration "
                                 "disabled: it did not improve validation "
                                 "ECE/log-loss)."),
    }


def print_result(result, config):
    classes = config["classes"]
    labels  = config["class_labels"]

    print()
    print("================================")
    print("AI DEEPFAKE DETECTION")
    print("================================")
    print()
    print("Image     : %s" % result["path"])
    face_str = "YES (face region cropped)" if result["face_detected"] else "NO (full image used)"
    print("Face      : %s" % face_str)
    print()
    print("Prediction: %s" % result["final_prediction"])
    print()
    for cls in classes:
        lbl = labels[cls]
        pct = result["probabilities"][cls] * 100
        print("%-20s: %6.2f%%" % (lbl, pct))
    print()
    print("Confidence: %.2f%%" % (result["confidence"] * 100))
    print(result["calibration_note"])
    print()
    if result["uncertain"]:
        print("LOW CONFIDENCE — MANUAL REVIEW RECOMMENDED")
        print("(Confidence below %.0f%% threshold)" % (result["uncertainty_threshold"] * 100))
        print()
    print("================================")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Classify an image as REAL / AI-GENERATED / FAKE-DEEPFAKE")
    parser.add_argument("image", help="Path to an image file (JPG/JPEG/PNG/WEBP/BMP)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        sys.exit("ERROR: File not found: %s" % args.image)

    config = load_config()
    model  = load_model(config)

    # Sanity: model output width must match class list in config.json
    n_model   = model.output_shape[-1]
    n_classes = len(config["classes"])
    if n_model != n_classes:
        sys.exit(
            "ERROR: Model has %d outputs but config.json lists %d classes.\n"
            "Re-train the model with 'python train_model.py'." % (n_model, n_classes))

    result = classify_image(args.image, config, model, calibrate.load_calibration(os.path.join(BASE_DIR, config["model_path"])))
    if "error" in result:
        sys.exit("ERROR: %s" % result["error"])

    print_result(result, config)


if __name__ == "__main__":
    main()