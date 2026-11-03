### [← Back to README](../README.md)

**Sections:** [Overview](#overview) • [MediaPipe Hand Topology](#mediapipe-hand-topology) • [Landmark Coordinate Representations](#landmark-coordinate-representations) • [Mathematical Normalization Formulation](#mathematical-normalization-formulation) • [Dual-Hand Feature Alignment (126D Vector)](#dual-hand-feature-alignment-126d-vector) • [Handedness Disambiguation and Spatial Fallback](#handedness-disambiguation-and-spatial-fallback) • [Invariance Properties and Environmental Robustness](#invariance-properties-and-environmental-robustness) • [Implementation Reference](#implementation-reference)

---

<br/>

# Feature Extraction and Geometric Normalization

## Overview

Feature extraction translates variable, high-dimensional visual data into a compact, deterministic, and invariant numerical representation suitable for machine learning classification.

Raw image pixels suffer from high variance caused by changes in lighting, background clutter, skin tone variations, camera perspective, and subject distance. The feature extraction pipeline implemented in [`src/feature_extractor.py`](../src/feature_extractor.py) abstracts away these variances by extracting 3D hand skeletal topologies and projecting them into a canonical 126-dimensional feature vector.

## MediaPipe Hand Topology

Google MediaPipe Hand Landmarker identifies 21 spatial landmarks per hand. Each landmark corresponds to an anatomical joint or landmark on the human hand:

| Index | Landmark Name | Anatomical Region |
| :---: | :--- | :--- |
| `0` | `WRIST` | Wrist origin joint |
| `1` | `THUMB_CMC` | Thumb carpometacarpal joint |
| `2` | `THUMB_MCP` | Thumb metacarpophalangeal joint |
| `3` | `THUMB_IP` | Thumb interphalangeal joint |
| `4` | `THUMB_TIP` | Thumb fingertip |
| `5` | `INDEX_FINGER_MCP` | Index finger metacarpophalangeal joint |
| `6` | `INDEX_FINGER_PIP` | Index finger proximal interphalangeal joint |
| `7` | `INDEX_FINGER_DIP` | Index finger distal interphalangeal joint |
| `8` | `INDEX_FINGER_TIP` | Index fingertip |
| `9` | `MIDDLE_FINGER_MCP` | Middle finger metacarpophalangeal joint |
| `10` | `MIDDLE_FINGER_PIP` | Middle finger proximal interphalangeal joint |
| `11` | `MIDDLE_FINGER_DIP` | Middle finger distal interphalangeal joint |
| `12` | `MIDDLE_FINGER_TIP` | Middle fingertip |
| `13` | `RING_FINGER_MCP` | Ring finger metacarpophalangeal joint |
| `14` | `RING_FINGER_PIP` | Ring finger proximal interphalangeal joint |
| `15` | `RING_FINGER_DIP` | Ring finger distal interphalangeal joint |
| `16` | `RING_FINGER_TIP` | Ring fingertip |
| `17` | `PINKY_MCP` | Little finger metacarpophalangeal joint |
| `18` | `PINKY_PIP` | Little finger proximal interphalangeal joint |
| `19` | `PINKY_DIP` | Little finger distal interphalangeal joint |
| `20` | `PINKY_TIP` | Little fingertip |

## Landmark Coordinate Representations

The detector provides two distinct coordinate systems:
1. **Screen Coordinates (`hand_landmarks`):**
   - Coordinates normalized to $[0.0, 1.0]$ across frame width $w$ and height $h$.
   - Subject to focal length and perspective skew.
2. **World Coordinates (`hand_world_landmarks`):**
   - 3D metric coordinates expressed in meters.
   - Origin centered at the hand geometric root.
   - Preserves true Euclidean angles and inter-joint distances regardless of the subject's distance from the camera.

The pipeline explicitly utilizes `world=True` by default in both offline extraction and online inference.

## Mathematical Normalization Formulation

The function [`normalize_hand`](../src/feature_extractor.py) maps 21 landmark vertices into an invariant feature vector:

Let $\mathbf{P} = \{\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_{20}\} \subset \mathbb{R}^3$ represent the raw coordinates of a detected hand, where $\mathbf{p}_i = [x_i, y_i, z_i]^\top$.

### Step 1: Wrist-Relative Translation
To achieve translation invariance, the wrist position $\mathbf{p}_0$ is defined as the coordinate frame origin:
$$\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0, \quad \forall i \in \{0, 1, \dots, 20\}$$

Following this transformation:
$$\mathbf{p}'_0 = [0, 0, 0]^\top$$

### Step 2: Scale Invariance via Maximum L2 Norm
Different users have varying hand dimensions, and users may position their hands at varying distances from the camera. To achieve scale invariance, the maximum Euclidean distance from the wrist origin across all landmarks is computed:

$$s = \max_{0 \le i \le 20} \|\mathbf{p}'_i\|_2 = \max_{0 \le i \le 20} \sqrt{(x'_i)^2 + (y'_i)^2 + (z'_i)^2}$$

A numerical stability threshold $\epsilon = 10^{-6}$ prevents division by zero in degraded detections:
- If $s < \epsilon$, the detection is rejected as degenerate and returns `None`.
- If $s \ge \epsilon$, all coordinates are scaled:
  $$\hat{\mathbf{p}}_i = \frac{\mathbf{p}'_i}{s}, \quad \forall i \in \{0, 1, \dots, 20\}$$

### Step 3: Vector Serialization
The scaled matrix $\hat{\mathbf{P}} \in \mathbb{R}^{21 \times 3}$ is flattened into a 63-dimensional single-hand vector:
$$\mathbf{v} = \text{vec}(\hat{\mathbf{P}}) = [\hat{x}_0, \hat{y}_0, \hat{z}_0, \hat{x}_1, \hat{y}_1, \hat{z}_1, \dots, \hat{x}_{20}, \hat{y}_{20}, \hat{z}_{20}]^\top \in \mathbb{R}^{63}$$

## Dual-Hand Feature Alignment (126D Vector)

Indian Sign Language (ISL) uses both single-hand and two-hand gestures. For instance, signs such as 'A', 'B', 'C' are often single-handed or two-handed depending on dialect, while signs such as 'W', 'X', and numbers may require coordination between both hands.

To support both single-handed and two-handed gestures without changing the input dimension expected by the classifier, the pipeline maintains a fixed 126-dimensional vector:

$$\mathbf{X} = [\mathbf{v}_{\text{Left}}^\top, \; \mathbf{v}_{\text{Right}}^\top]^\top \in \mathbb{R}^{126}$$

Where:
- $\mathbf{v}_{\text{Left}} \in \mathbb{R}^{63}$ occupies indices `[0:63]`.
- $\mathbf{v}_{\text{Right}} \in \mathbb{R}^{63}$ occupies indices `[63:126]`.
- If a hand is not detected or absent, its corresponding 63 dimensions are zero-padded: $\mathbf{0}_{63}$.

```
Feature Vector Structure (126 Dimensions):
┌────────────────────────────────────────┬────────────────────────────────────────┐
│           Left Hand (63 dims)          │          Right Hand (63 dims)          │
├───────────────────┬────────────────────┼───────────────────┬────────────────────┤
│ Landmarks 0 to 20 │   x, y, z floats   │ Landmarks 0 to 20 │   x, y, z floats   │
│ Indices [0:62]    │   Zeros if absent  │ Indices [63:125]  │   Zeros if absent  │
└───────────────────┴────────────────────┴───────────────────┴────────────────────┘
```

## Handedness Disambiguation and Spatial Fallback

Assigning a detected hand correctly to either the `Left` or `Right` slot is critical for classification consistency.

### Primary Strategy: MediaPipe Handedness
MediaPipe provides handedness classification based on visual palm-versus-back appearance:
```python
if (
    hasattr(result, "handedness")
    and i < len(result.handedness)
    and result.handedness[i]
):
    side = result.handedness[i][0].category_name  # "Left" or "Right"
```

### Secondary Fallback: Spatial Centroid Partitioning
In real-time video feeds with rapid motion, extreme angles, or partial occlusion, MediaPipe's handedness probability may be unavailable or ambiguous. The extractor implements a spatial fallback heuristic:
```python
# Compute horizontal centroid across all landmarks of the hand
center_x = np.mean([lm.x for lm in result.hand_landmarks[i]])

# Assign Left if situated on the left half of the sensor frame, else Right
side = "Left" if center_x < 0.5 else "Right"
```

This prevents feature slot collision and guarantees deterministic feature alignment.

## Invariance Properties and Environmental Robustness

The mathematical transformations provide multiple geometric guarantees:

1. **Translation Invariance:**
   Shifting the hand to any position within the camera view field does not alter $\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0$.
2. **Scale Invariance:**
   Moving the hand closer to or further from the camera changes raw coordinates proportionally, but dividing by $s = \max_i \|\mathbf{p}'_i\|$ maintains constant feature values.
3. **Lighting and Background Invariance:**
   Because classification operates exclusively on landmark coordinates rather than raw pixel textures, changes in ambient lighting, shadows, and complex backgrounds do not degrade classifier accuracy.
4. **Color and Skin Invariance:**
   The perception model operates on structural hand morphology, insulating the system from skin pigmentation or glove color biases.

## Implementation Reference

The feature extraction routine is implemented in [`src/feature_extractor.py`](../src/feature_extractor.py):
- [`normalize_hand`](../src/feature_extractor.py): Performs wrist translation, scale normalization, and flattening for an individual hand.
- [`extract_features`](../src/feature_extractor.py): Coordinates multi-hand sorting, handedness identification, fallback ordering, zero-padding, and 126D concatenation.
