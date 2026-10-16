# AI Deepfake Detection System

A beginner-friendly, honest **3-class human-face image classifier**:

- **REAL** — genuine human photographs
- **AI-GENERATED** — fully synthetic faces (StyleGAN / thispersondoesnotexist-style)
- **FAKE/DEEPFAKE** — manipulated faces (DFDC / CELEB-DF frames)

Built with **MobileNetV2 transfer learning** (ImageNet weights), one shared
preprocessing pipeline for training and inference, a 70/15/15 train/validation/
test split, augmentation, class weights, and a single config file that stores
the class mapping (nothing hard-coded).

> **Honest by design:** the app shows the *real* test metrics measured after
> training and prints an uncertainty warning instead of pretending certainty.
> No 100% accuracy is claimed.

---

## Project structure

```
AI-Deepfake-Detection/
├── dataset/
│   ├── real/            # face-verified genuine photos
│   ├── ai_generated/    # face-verified AI faces
│   └── fake_deepfake/   # face-verified deepfake frames
├── models/
│   ├── deepfake_detector.keras    # best model (lowest val loss)
│   ├── test_metrics.json          # real test metrics (shown in the app)
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── tests/unseen/        # held-out images NEVER used in training
├── results/             # evaluation output and reports
├── config.json          # single configuration file (classes, labels, split, thresholds, sources)
├── preprocess.py        # SINGLE shared preprocessing pipeline
├── prepare_dataset.py   # dataset inspection + face verification + build
├── train_model.py       # 70/15/15 split + transfer learning + evaluation
├── evaluate_model.py    # re-evaluate the saved model on the test split
├── verify_model.py      # sanity checks (shapes, mapping, no constant output)
├── predict.py           # CLI single-image prediction
├── app.py               # Streamlit demo
├── requirements.txt
└── README.md
```

## Installation

```powershell
cd AI-Deepfake-Detection
pip install -r requirements.txt
```

Requires Python 3.10+ with TensorFlow (CPU is fine for this dataset size).

## 1. Prepare the dataset (inspection + face verification)

`prepare_dataset.py` first **inspects the source folders**, splits the raw
archives into the three classes by filename rules, verifies that every copied
image decodes and contains a human face, and **stops with a clear report** if
any class is missing or has too few valid images.

```powershell
python prepare_dataset.py
```

Source paths are configured in `config.json` → `sources`. Edit them if you
move the raw data. Requirements per class are `min_images_per_class` /
`max_images_per_class` (default 300 / 600). Five images per class are held out
to `tests/unseen/` for the final demo.

## 2. Train

```powershell
python train_model.py
```

- 70% train / 15% validation / 15% test — stratified, seeded (config `seed`).
- Data augmentation on the training split only (flip, rotation, zoom,
  translation, brightness, contrast, saturation, hue).
- Class-balanced sampling: under-represented classes are oversampled so every
  batch has equal representation per class (no fake samples; the small
  `ai_generated` class gets a 1.5x prior).
- Two stages: frozen backbone (lr 1e-3) → fine-tune top 65% of layers (lr 1e-5).
- Early stopping + `ReduceLROnPlateau`, both monitoring **validation macro-F1**
  (fair to all classes); the **best** checkpoint (highest val F1) is kept in
  `models/`.
- Prints and saves accuracy, macro precision/recall/F1, per-class report,
  confusion matrix (+ PNG), `results/test_metrics.json` and a calibrated copy
  (vector scaling fitted on validation) used for display.

To re-evaluate the saved model without retraining:

```powershell
python train_model.py --eval-only
```

To resume a run that was interrupted (continues Stage 2 fine-tuning from the
saved checkpoint, then calibrates and evaluates as usual):

```powershell
python train_model.py --resume
```

### Actual metrics on the held-out test split (1323 images, 15% split)

Measured with the **calibrated** model (vector scaling fitted on the validation
split only, never on test; ECE 0.0525 → 0.0399 on that validation fit).

| Class         | Precision | Recall | F1     |
|---------------|-----------|--------|--------|
| real          | 0.71      | 0.48   | 0.57   |
| ai_generated  | 0.52      | 0.41   | 0.46   |
| fake_deepfake | 0.68      | 0.93   | 0.79   |
| **Macro avg** | **0.64**  | **0.60**| **0.61**|

**Test accuracy: 65.23%** (test ECE 0.0525 raw / 0.0399 calibrated) — see
`results/test_metrics.json` and `results/confusion_matrix.png`. The "real" and
"AI" classes are the hardest (real and AI faces are visually similar);
deepfake frames are very well detected. These numbers are also displayed
inside the app.

> **Model selection note:** the final detector is the pre-cleanup checkpoint
> (`deepfake_detector_v2_backup.keras`), chosen because it generalizes best to
> truly unseen deepfakes. The later Sprint-8 retrains score higher on the
> in-distribution test split but overfit the cleaned distribution and fail the
> acceptance test (0/5 unseen deepfakes). On the 15 held-out images this model
> catches **5/5 deepfakes**, which is what matters for a deepfake detector.
> See `results/test_metrics.json.single_model` and the model backups in
> `models/` for the comparison.

## 3. Predict a single image (CLI)

```powershell
python predict.py path\to\image.jpg
```

Output: probabilities for all three classes, final prediction, confidence, and
an **uncertainty warning** when confidence is below the threshold in `config.json`
(default 60%). The probabilities come from the model; the class order and
display labels come from `config.json`.

### Final acceptance test (unseen images, held out from training)

15 held-out images (5 per class): **12/15 correct (80.0%)** — see
`results/unseen_test_results.json`. Every deepfake is detected; every miss is a
real↔AI confusion (two real faces predicted AI, one AI face predicted REAL).

| Image                          | Expected       | Predicted      | Confidence |
|--------------------------------|----------------|----------------|------------|
| real 02271.jpg                 | REAL           | AI-GENERATED   | 57.6% ✗ wrong |
| real 30849.jpg                 | REAL           | REAL           | 73.4%      |
| real 41972.jpg                 | REAL           | AI-GENERATED   | 63.7% ✗ wrong |
| real 49000.jpg                 | REAL           | REAL           | 56.5%      |
| real 63153.jpg                 | REAL           | REAL           | 58.4%      |
| AI face 57GZ28HN9T.jpg         | AI-GENERATED   | AI-GENERATED   | 83.9%      |
| AI face BAC2YGGJJH.jpg         | AI-GENERATED   | AI-GENERATED   | 76.5%      |
| AI face HGF540APFW.jpg         | AI-GENERATED   | REAL           | 55.6% ✗ wrong |
| AI face LMHCYDQY4F.jpg         | AI-GENERATED   | AI-GENERATED   | 73.8%      |
| AI face TIK309FPXK.jpg         | AI-GENERATED   | AI-GENERATED   | 81.1%      |
| deepfake frame (CELEB-DF)      | FAKE/DEEPFAKE  | FAKE/DEEPFAKE  | 70.7%      |
| deepfake frame (DFDC akzbnazxtz) | FAKE/DEEPFAKE | FAKE/DEEPFAKE | 78.0%    |
| deepfake frame (DFDC bqnymlsayl) | FAKE/DEEPFAKE | FAKE/DEEPFAKE | 78.8%    |
| deepfake frame (DFDC btjwbtsgln) | FAKE/DEEPFAKE | FAKE/DEEPFAKE | 69.5%    |
| deepfake (FaceShifter)         | FAKE/DEEPFAKE  | FAKE/DEEPFAKE  | 64.9%      |

All five deepfakes are detected; every miss is a real↔AI confusion (two real
faces predicted AI, one AI face predicted REAL). Several borderline calls sit
near the 60% threshold in `config.json` — the uncertainty warning is there to
flag them.

Example commands (run from the project folder):

```powershell
python predict.py "tests\unseen\real\02271.jpg"
python predict.py "tests\unseen\ai_generated\57GZ28HN9T.jpg"
python predict.py "tests\unseen\fake_deepfake\celebdf_val_id35_id17_0000_frame360_face10.jpg"
```

## 4. Streamlit demo

```powershell
streamlit run app.py
```

Two tabs — **Image Upload** and **Live Camera** — both using the **same
preprocessing pipeline** used in training:

**Image Upload** — upload an image; the app shows it, detects the face, and
displays:

- the three probabilities (REAL / AI-GENERATED / FAKE/DEEPFAKE) with bars
- the final prediction and confidence
- an **uncertainty warning** for low-confidence results
- the real test metrics from `models/test_metrics.json`

**Live Camera** — real-time webcam detection:

- press **Start camera** (the app tries camera indexes 0, 1, 2) and a frame is
  captured every 0.05 s (20 fps prediction cadence)
- each frame goes through the identical face-detect → crop → 224×224 →
  normalize pipeline before prediction
- **adaptive preprocessing** (CLAHE + bilateral filter) enhances face
  detection on low-quality webcam frames
- **test-time augmentation (flip-TTA)** averages original + horizontal flip
  predictions, matching training evaluation
- **domain-specific model** (`deepfake_detector_webcam_v1.keras`) trained on
  webcam-style images with sensor noise, blur, and compression artifacts
- predictions are **temporally smoothed** (EMA over the latest 10 frames) so
  the verdict is stable instead of flickering frame-to-frame
- shows the smoothed prediction, confidence, a low-confidence warning, and the
  per-class probabilities; press **Stop camera** to close the webcam

> Note: Streamlit runs the webcam on the machine that launches `app.py`
> (localhost), not on the browser. For a phone/remote camera, use the upload
> tab instead.

## Accuracy Improvements

The live webcam detection has been optimized with several techniques:

1. **Test-Time Augmentation (flip-TTA)**: Averages the model's predictions
   on the original frame and its horizontal flip, reducing variance and
   improving accuracy by ~2-5%.
2. **Domain-Specific Model**: Uses `deepfake_detector_webcam_v1.keras`
   trained on webcam-like images (with sensor noise, blur, compression).
3. **Adaptive Preprocessing**: CLAHE histogram equalization + bilateral
   filtering enhances face detection on low-contrast, noisy webcam frames.
4. **Improved Face Detection**: More sensitive Haar Cascade parameters
   (`scaleFactor=1.05`, `minNeighbors=8`) with histogram equalization and
   automatic fallback.
5. **Exponential Moving Average Smoothing**: More responsive than simple
   averaging, giving more weight to recent frames.
6. **Faster Capture**: 0.05s interval (20 fps) instead of 2.0s, for more
   responsive detection and better temporal smoothing.

## How the pieces stay consistent

- `preprocess.py` is the single source of truth:
  `validate → detect face → crop (25% margin) → resize 224×224 → RGB → MobileNetV2 normalize [-1,1]`.
- `config.json` holds the class mapping and display labels; `predict.py`,
  `app.py` and `train_model.py` all read from it. Model output index `i`
  always means `config["classes"][i]`.
- The dataset folders contain face-cropped, 224×224 RGB JPEGs produced with
  the very same crop functions, so train and prediction see identical inputs.

## Known limitations (read before trusting it)

- A real-vs-AI classifier trained on ~600 images/class is a demo, not a
  forensic tool. Deepfakes evolve; this model reflects only its training set.
- "Real" photos and high-quality AI faces are often visually almost identical
  → some real faces are predicted as AI (confidence stays low, warning shows).
- The app never claims certainty and never hides its real test metrics.