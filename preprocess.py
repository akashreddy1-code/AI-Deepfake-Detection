"""
preprocess.py - THE SINGLE SHARED PREPROCESSING PIPELINE.

Every part of the project (prepare_dataset.py, train_model.py, predict.py,
app.py) goes through exactly the same steps:

    validate image -> detect face -> crop face (with margin)
    -> resize 224x224 -> BGR->RGB (OpenCV reads BGR, the model expects RGB)
    -> MobileNetV2 normalize (preprocess_input, values in [-1, 1])

This keeps training-time and prediction-time preprocessing identical.
"""

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_SIZE = (224, 224)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FACE_MARGIN = 0.25

# Improved detection parameters for webcam frames
DETECT_PARAMS = {
    "scaleFactor": 1.05,
    "minNeighbors": 8,
    "minSize": (30, 30),
}
FALLBACK_PARAMS = {
    "scaleFactor": 1.1,
    "minNeighbors": 5,
    "minSize": (40, 40),
}


def apply_adaptive_preprocessing(img_bgr):
    """Enhance webcam frame quality for face detection only.

    Applies CLAHE contrast limiting adaptive histogram equalization
    on the luminance channel and mild bilateral filtering to reduce
    sensor noise while preserving edges. This improves face detection
    on low-quality webcam frames WITHOUT changing the model input
    distribution (the model pipeline uses the original frame).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    return denoised


def load_image(path):
    """Decode an image from disk as BGR. Returns None if it cannot be read."""
    return cv2.imread(path)


def decode_from_bytes(data):
    """Decode raw file bytes (Streamlit upload) as BGR. None on failure."""
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def validate_image(img):
    """Return a human-readable error string if the image is unusable, else ''."""
    if img is None:
        return "Could not read / decode the image."
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        return "Image is too small (%dx%d)." % (w, h)
    if len(img.shape) != 3 or img.shape[2] != 3:
        return "Image is not a 3-channel color image."
    return ""


def detect_faces(img_bgr):
    """Return ALL face bounding boxes, LARGEST FIRST (x, y, w, h tuples).

    Uses multi-scale detection with sensitive parameters (optimized for
    webcam frames), with a fallback to relaxed parameters if no faces
    are found. Applies histogram equalization for low-contrast frames.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Histogram equalization improves detection on low-contrast webcam frames
    gray_eq = cv2.equalizeHist(gray)
    faces = FACE_CASCADE.detectMultiScale(
        gray_eq, **DETECT_PARAMS
    )
    if len(faces) == 0:
        faces = FACE_CASCADE.detectMultiScale(
            gray, **FALLBACK_PARAMS
        )
    if len(faces) == 0:
        return []
    boxes = [tuple(f) for f in faces]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def detect_face(img_bgr):
    """Return the largest face bounding box (x, y, w, h) or None."""
    faces = detect_faces(img_bgr)
    return faces[0] if faces else None


def crop_face(img_bgr, box):
    """Crop around the face box with a margin, clamped to image bounds."""
    h, w = img_bgr.shape[:2]
    x, y, bw, bh = box
    mx = int(bw * FACE_MARGIN)
    my = int(bh * FACE_MARGIN)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    return img_bgr[y0:y1, x0:x1]


def normalize_batch(rgb_uint8):
    """RGB uint8 -> MobileNetV2-normalized (1, 224, 224, 3) float32 batch.
    The exact normalization the training pipeline uses (preprocess_input)."""
    normalized = preprocess_input(np.asarray(rgb_uint8).astype(np.float32))
    return np.expand_dims(normalized, axis=0)


def prepare_from_box(img_bgr, box, target_size=IMAGE_SIZE):
    """Crop a KNOWN face box (+margin), convert to RGB and resize.

    Lets callers that already ran face detection (e.g. the live camera,
    which also needs the face COUNT) reuse this ONE pipeline without
    detecting twice. Returns (rgb_uint8, crop_bgr)."""
    crop_bgr = crop_face(img_bgr, box) if box is not None else img_bgr
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, target_size)
    return rgb, crop_bgr


def _prepare_image(img_bgr, target_size=IMAGE_SIZE):
    """Shared internal implementation: face-crop + RGB + resize.

    Returns (rgb_uint8, face_found, crop_bgr, box). Every public entry point
    (uploads, webcam, dataset prep) goes through this ONE function so the
    preprocessing can never diverge between training and prediction."""
    face_found = False
    crop_bgr = img_bgr
    box = detect_face(img_bgr)
    if box is not None:
        face_found = True
        crop_bgr = crop_face(img_bgr, box)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, target_size)
    return rgb, face_found, crop_bgr, box


def prepare_image(img_bgr, target_size=IMAGE_SIZE):
    """Face-crop + RGB + resize. Returns (rgb_uint8, face_found, crop_bgr).

    This is the shape/colour part of the pipeline. No normalization is done
    here, so the same output can feed training (augmentation + normalize)
    and prediction (normalize).
    """
    rgb, face_found, crop_bgr, _box = _prepare_image(img_bgr, target_size)
    return rgb, face_found, crop_bgr


def preprocess_for_model(img_bgr, target_size=IMAGE_SIZE):
    """Turn a decoded BGR image into a model-ready float32 batch.

    Returns:
        batch       : (1, 224, 224, 3) normalized to [-1, 1]
        face_found  : True if a face was detected and cropped
        crop_bgr    : the (face-cropped or full) BGR image used, or None on error
    """
    rgb, face_found, crop_bgr, _box = _prepare_image(img_bgr, target_size)
    normalized = preprocess_input(rgb.astype(np.float32))
    batch = np.expand_dims(normalized, axis=0)
    return batch, face_found, crop_bgr


def predict_face(img_bgr, target_size=IMAGE_SIZE):
    """Common face pipeline for BOTH image upload and live webcam.

    Uses detect_face() on the ORIGINAL BGR frame — exactly the same
    as classify_bytes(). This guarantees identical preprocessing for
    upload and webcam predictions.

    Returns (rgb_uint8, face_found, crop_bgr, box)."""
    return _prepare_image(img_bgr, target_size)


def preprocess_for_model_debug(img_bgr, target_size=IMAGE_SIZE):
    """Identical to preprocess_for_model, additionally returns the detected
    face bounding box (x, y, w, h) or None.

    Used by the live camera so its debug panel can show the exact box that
    produced the crop. The face is detected ONCE (in the shared
    _prepare_image) — no second pipeline, no double detection."""
    rgb, face_found, crop_bgr, box = _prepare_image(img_bgr, target_size)
    normalized = preprocess_input(rgb.astype(np.float32))
    batch = np.expand_dims(normalized, axis=0)
    return batch, face_found, crop_bgr, box


def preprocess_from_path(path):
    """Convenience: load + validate + preprocess in one call.

    Returns (batch, face_found, crop_bgr, error_string).
    """
    img = load_image(path)
    err = validate_image(img)
    if err:
        return None, False, None, err
    batch, face_found, crop_bgr = preprocess_for_model(img)
    return batch, face_found, crop_bgr, ""


def preprocess_from_bytes(data):
    """Convenience: decode bytes (uploaded file) + validate + preprocess."""
    img = decode_from_bytes(data)
    err = validate_image(img)
    if err:
        return None, False, None, err
    batch, face_found, crop_bgr = preprocess_for_model(img)
    return batch, face_found, crop_bgr, ""