### [← Back to README](../README.md)

**Sections:** [Overview](#overview) • [Module: src.hand_landmarker](#module-srchand_landmarker) • [Module: src.feature_extractor](#module-srcfeature_extractor) • [Module: src.__main__](#module-src__main__) • [Module: scripts.extract_landmarks](#module-scriptsextract_landmarks) • [Module: scripts.train](#module-scriptstrain)

---

<br/>

# API Reference

## Overview

This document provides a comprehensive technical reference for the public interfaces, classes, functions, and configuration parameters across the codebase.

## Module: src.hand_landmarker

Defined in [`src/hand_landmarker.py`](../src/hand_landmarker.py).

### Class: `HandLandmarker`

Wrapper class managing the initialization, inference, and resource teardown of the Google MediaPipe Tasks Hand Landmarker.

#### `__init__(self, model_path: str, num_hands: int = 2)`
Initializes the MediaPipe vision detector from a serialized task bundle.

- **Parameters:**
  - `model_path` (*str*): Path to the `hand_landmarker.task` file.
  - `num_hands` (*int*, optional): Maximum number of hands to detect concurrently. Default is `2`.
- **Attributes Initialized:**
  - `detector` (*vision.HandLandmarker*): Instantiated MediaPipe landmarker execution engine.

#### `grab_hand_landmarks(self, image: cv2.typing.MatLike) -> vision.HandLandmarkerResult`
Executes synchronous landmark detection on a standard BGR OpenCV frame.

- **Parameters:**
  - `image` (*cv2.typing.MatLike* / *np.ndarray*): Input image in BGR format (height, width, 3).
- **Returns:**
  - `vision.HandLandmarkerResult`: MediaPipe detection result object containing:
    - `hand_landmarks`: List of 21 normalized image-coordinate landmarks per hand.
    - `hand_world_landmarks`: List of 21 metric 3D world-coordinate landmarks per hand.
    - `handedness`: List of handedness classification scores per detected hand.

#### `close(self) -> None`
Releases native C++ MediaPipe detector resources and closes underlying handles.

---

### Function: `overlay_hand_landmarks`

#### `overlay_hand_landmarks(image: np.ndarray, results: mp.Image) -> np.ndarray`
Renders skeletal landmark connections and joint keypoints directly onto the provided image frame in-place.

- **Parameters:**
  - `image` (*np.ndarray*): BGR image frame to draw onto (modified in-place).
  - `results` (*vision.HandLandmarkerResult*): Detection result from `grab_hand_landmarks`.
- **Returns:**
  - `np.ndarray`: The modified image frame with rendered lines and joints.
- **Rendering Specifications:**
  - Skeletal Bonds: Gold/cyan lines (`BGR: (255, 215, 0)`), thickness 2 px.
  - Joints: Concentric circles with inner white core (`radius = 2`) and outer green ring (`BGR: (0, 255, 65)`, `radius = 4`), plus fingertip target rings (`radius = 7`).
  - Reticles: Bounding corner brackets around detected hands with dynamic handedness tags.

---

## Module: src.feature_extractor

Defined in [`src/feature_extractor.py`](../src/feature_extractor.py).

### Function: `normalize_hand`

#### `normalize_hand(landmarks) -> Optional[np.ndarray]`
Normalizes an individual 21-landmark array to achieve translation and scale invariance.

- **Parameters:**
  - `landmarks`: Iterable collection of 21 landmark objects possessing `.x`, `.y`, and `.z` attributes.
- **Returns:**
  - `Optional[np.ndarray]`: A 1D NumPy array of shape `(63,)` and `dtype=float32`, or `None` if the scale is less than $10^{-6}$.
- **Mathematical Steps:**
  1. Translates all landmarks relative to the wrist: $\mathbf{p}_i' = \mathbf{p}_i - \mathbf{p}_0$.
  2. Computes the maximum Euclidean distance: $s = \max_i \|\mathbf{p}_i'\|_2$.
  3. Rejects degenerate configurations if $s < 10^{-6}$.
  4. Scales points by $s$: $\hat{\mathbf{p}}_i = \mathbf{p}_i' / s$.
  5. Flattens array into 63 continuous coordinates: $[\hat{x}_0, \hat{y}_0, \hat{z}_0, \dots, \hat{x}_{20}, \hat{y}_{20}, \hat{z}_{20}]$.

---

### Function: `extract_features` (Dual Hand)

#### `extract_features(result, world=True) -> Optional[np.ndarray]`
Extracts and canonicalizes landmarks from both hands into a standardized 126-dimensional feature vector.

- **Parameters:**
  - `result`: MediaPipe `HandLandmarkerResult` containing detected hand structures.
  - `world` (*bool*): If `True`, processes `hand_world_landmarks`. If `False`, processes `hand_landmarks`.
- **Returns:**
  - `Optional[np.ndarray]`: A 1D NumPy array of shape `(126,)` and `dtype=float32`, or `None` if no landmarks are present in `result`.
- **Hand Slot Mapping:**
  - Indices `[0:63]`: Left hand normalized features (or zeros if absent).
  - Indices `[63:126]`: Right hand normalized features (or zeros if absent).
- **Assignment Logic:**
  - Identifies hand category via `result.handedness[i][0].category_name`.
  - Falls back to horizontal centroid division (`center_x < 0.5` is Left, else Right) if handedness metadata is unavailable or ambiguous.

---

## Module: src.__main__

Defined in [`src/__main__.py`](../src/__main__.py).

### Module Constants

| Constant | Type | Value | Description |
| :--- | :---: | :--- | :--- |
| `MODEL_PATH` | `str` | `"models/isl_classifier.pkl"` | Path to serialized Random Forest classifier. |
| `ENCODER_PATH` | `str` | `"models/label_encoder.pkl"` | Path to serialized `LabelEncoder` instance. |
| `LANDMARK_MODEL` | `str` | `"models/hand_landmarker.task"` | Path to MediaPipe task asset. |
| `CONFIDENCE_THRESHOLD` | `float` | `0.75` | Minimum posterior probability required to accept a prediction. |

### Function: `main`

#### `main() -> None`
Main entrypoint for the interactive webcam recognition application.
- Instantiates [`HandLandmarker`](../src/hand_landmarker.py).
- Loads serialized classification artifacts via `joblib`.
- Opens `cv2.VideoCapture(0)`.
- Runs the frame processing, inference, HUD rendering, and keyboard event loop.
- Manages sentence buffer accumulation (`Space`, `Backspace`, `c`).
- Ensures clean shutdown and release of camera and UI resources in `finally` block.

---

## Module: scripts.extract_landmarks

Defined in [`scripts/extract_landmarks.py`](../scripts/extract_landmarks.py).

### Script Constants

| Constant | Type | Value | Description |
| :--- | :---: | :--- | :--- |
| `DATASET` | `Path` | `Path("datasets/isl_dataset")` | Source directory containing raw training images. |
| `OUTPUT` | `Path` | `Path("datasets/processed")` | Destination directory for `.npy` feature files. |
| `MODEL_PATH` | `str` | `"models/hand_landmarker.task"` | Path to MediaPipe task asset. |
| `IMAGE_EXTENSIONS` | `set` | `{".jpg", ".jpeg", ".png", ".bmp", ".webp"}` | Valid file extensions for image ingestion. |

### Function: `main`

#### `main() -> None`
Batch landmark extraction pipeline:
- Scans `datasets/isl_dataset` for all subfolders representing classes.
- Processes each image using `HandLandmarker`.
- Normalizes features into 126D vectors via `extract_features(world=True)`.
- Compiles matrices and writes `features.npy`, `labels.npy`, and `classes.npy`.
- Logs extraction summary and failure count per class.

---

## Module: scripts.train

Defined in [`scripts/train.py`](../scripts/train.py).

### Training Script Workflow

1. Loads binary feature matrix `X` and target vector `y` from `datasets/processed/`.
2. Encodes alphanumeric string labels using `sklearn.preprocessing.LabelEncoder`.
3. Performs stratified 80/20 train/test partition using `train_test_split`.
4. Computes balanced class weights via `compute_class_weight("balanced")`.
5. Fits `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)`.
6. Evaluates test set predictions using `accuracy_score` and `classification_report`.
7. Computes and saves a 36-class confusion matrix heatmap to `confusion_matrix.png`.
8. Serializes `isl_classifier.pkl` and `label_encoder.pkl` to `models/`.
