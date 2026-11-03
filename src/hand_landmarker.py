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


def overlay_hand_landmarks(image: np.ndarray, results: mp.Image):
    h, w, _ = image.shape

    for hand in results.hand_landmarks:
        points = []
        for lm in hand:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        for connection in CONNECTIONS:
            start = points[connection.start]
            end   = points[connection.end]

            cv2.line(image, start, end, (0, 0, 255), 2)
        
        for point in points:
            cv2.circle(image, point, 4, (0, 255, 0), -1)

    return image
