import cv2
import numpy as np
import joblib

from src.hand_landmarker import (
    HandLandmarker, overlay_hand_landmarks
)

from src.feature_extractor import extract_features


MODEL_PATH     = "models/isl_classifier.pkl"
ENCODER_PATH   = "models/label_encoder.pkl"
LANDMARK_MODEL = "models/hand_landmarker.task"

CONFIDENCE_THRESHOLD = 0.75

def main():
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    detector = HandLandmarker(LANDMARK_MODEL)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    sentence = ""
    prediction = "Waiting..."
    confidence = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            result = detector.grab_hand_landmarks(frame)

            if result.hand_landmarks is None:
                return
                
            overlay_hand_landmarks(frame, result)

            # Extract features
            features = extract_features(result,world=True)

            if features is not None:
                features = features.reshape(1, -1)
                probabilities = model.predict_proba(features)[0]
                index = np.argmax(probabilities)
                confidence = probabilities[index]
                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction = encoder.inverse_transform(
                        [model.classes_[index]]
                    )[0]
                else:
                    prediction = "Unknown"

            else:
                prediction = "No hand detected"
                confidence = 0.0

            # UI
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (30, 30, 30), -1)

            cv2.putText(frame, f"Prediction: {prediction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("ISL Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                if prediction not in (
                    "Unknown",
                    "No hand detected",
                    "Waiting..."
                ):
                    sentence += prediction

            elif key == 8:
                sentence = sentence[:-1]

            elif key == ord('c'):
                sentence = ""

            print_text = f"Text: {sentence}"

            print(print_text, end="\r")

    except Exception as e:
        print("ERR: ", e)

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
