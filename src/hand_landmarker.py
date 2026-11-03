import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


class HandLandmarker:
    def __init__(self, model_path: str, num_hands: int = 2):
        base_opt = python.BaseOptions(model_asset_path=model_path)
        opt = vision.HandLandmarkerOptions(base_options=base_opt, num_hands=num_hands)
        self.detector = vision.HandLandmarker.create_from_options(opt)

    def grab_hand_landmarks(self, image: cv2.typing.MatLike):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect(mp_image)

    def close(self):
        self.detector.close()


def overlay_hand_landmarks(image: np.ndarray, results: mp.Image) -> np.ndarray:
    if results is None or not results.hand_landmarks:
        return image

    h, w, _ = image.shape
    fingertip_indices = {4, 8, 12, 16, 20}

    for idx, hand in enumerate(results.hand_landmarks):
        points = []
        for lm in hand:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        if not points:
            continue

        # Cyber skeletal connections
        for connection in CONNECTIONS:
            start = points[connection.start]
            end   = points[connection.end]
            cv2.line(image, start, end, (255, 215, 0), 2, cv2.LINE_AA)

        # Tactical joint points: white center core and outer cyber-green ring
        for i, point in enumerate(points):
            if i in fingertip_indices:
                cv2.circle(image, point, 7, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.circle(image, point, 4, (0, 255, 65), -1, cv2.LINE_AA)
            cv2.circle(image, point, 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Bounding reticle around hand
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        pad = 22
        min_x = max(0, min(xs) - pad)
        max_x = min(w - 1, max(xs) + pad)
        min_y = max(0, min(ys) - pad)
        max_y = min(h - 1, max(ys) + pad)

        c_len = max(8, min(24, (max_x - min_x) // 4, (max_y - min_y) // 4))
        c_color = (0, 255, 65)

        # Reticle corner brackets
        cv2.line(image, (min_x, min_y), (min_x + c_len, min_y), c_color, 2, cv2.LINE_AA)
        cv2.line(image, (min_x, min_y), (min_x, min_y + c_len), c_color, 2, cv2.LINE_AA)
        cv2.line(image, (max_x, min_y), (max_x - c_len, min_y), c_color, 2, cv2.LINE_AA)
        cv2.line(image, (max_x, min_y), (max_x, min_y + c_len), c_color, 2, cv2.LINE_AA)
        cv2.line(image, (min_x, max_y), (min_x + c_len, max_y), c_color, 2, cv2.LINE_AA)
        cv2.line(image, (min_x, max_y), (min_x, max_y - c_len), c_color, 2, cv2.LINE_AA)
        cv2.line(image, (max_x, max_y), (max_x - c_len, max_y), c_color, 2, cv2.LINE_AA)
        cv2.line(image, (max_x, max_y), (max_x, max_y - c_len), c_color, 2, cv2.LINE_AA)

        # Handedness label tag
        side = "HAND"
        if (
            hasattr(results, "handedness")
            and idx < len(results.handedness)
            and results.handedness[idx]
        ):
            side = results.handedness[idx][0].category_name.upper()

        tag = f"[{side}_LOCKED]"
        cv2.putText(
            image,
            tag,
            (min_x, max(14, min_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 65),
            1,
            cv2.LINE_AA
        )

    return image
