
import numpy as np


def normalize_hand(landmarks):

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    # Wrist-relative coordinates
    points -= points[0]

    # Scale normalization
    scale = np.linalg.norm(
        points,
        axis=1
    ).max()

    if scale < 1e-6:
        return None

    points /= scale

    return points.flatten()


def extract_features(result, world=True):

    landmark_data = (
        result.hand_world_landmarks
        if world
        else result.hand_landmarks
    )

    if not landmark_data:
        return None

    # Two fixed slots
    hands = {
        "Left": np.zeros(63, dtype=np.float32),
        "Right": np.zeros(63, dtype=np.float32)
    }

    unknown_hands = []

    for i, landmarks in enumerate(landmark_data):

        feature = normalize_hand(landmarks)

        if feature is None:
            continue

        side = None

        # Get handedness corresponding to this hand
        if (
            hasattr(result, "handedness")
            and i < len(result.handedness)
            and result.handedness[i]
        ):
            side = result.handedness[i][0].category_name

        if side in hands and not np.any(hands[side]):
            hands[side] = feature
        else:
            unknown_hands.append((i, feature))

    # Fallback: spatial ordering if handedness is unavailable
    for i, feature in unknown_hands:

        if i < len(result.hand_landmarks):
            center_x = np.mean([
                lm.x for lm in result.hand_landmarks[i]
            ])
        else:
            center_x = 0.5

        side = "Left" if center_x < 0.5 else "Right"

        if not np.any(hands[side]):
            hands[side] = feature

    return np.concatenate([
        hands["Left"],
        hands["Right"]
    ]).astype(np.float32)
