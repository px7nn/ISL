import time
import cv2
import numpy as np
import joblib

from src.hand_landmarker import (
    HandLandmarker,
    overlay_hand_landmarks
)
from src.feature_extractor import extract_features


MODEL_PATH = "models/isl_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
LANDMARK_MODEL = "models/hand_landmarker.task"

CONFIDENCE_THRESHOLD = 0.75


def draw_hud(frame, prediction, confidence, sentence, fps, hands_count):
    h, w, _ = frame.shape

    top_h = max(80, min(100, int(h * 0.14)))
    bot_h = max(45, min(60, int(h * 0.08)))

    # Semi-transparent HUD glass panels
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_h), (10, 14, 16), -1)
    cv2.rectangle(overlay, (0, h - bot_h), (w, h), (10, 14, 16), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Accent boundary lines
    cv2.line(frame, (0, top_h), (w, top_h), (0, 255, 65), 1, cv2.LINE_AA)
    cv2.line(frame, (0, h - bot_h), (w, h - bot_h), (0, 255, 65), 1, cv2.LINE_AA)

    # Tactical corner crosshairs
    c_len = 10
    corners = [(15, top_h + 15), (w - 15, top_h + 15), (15, h - bot_h - 15), (w - 15, h - bot_h - 15)]
    for cx, cy in corners:
        cv2.line(frame, (cx - c_len, cy), (cx + c_len, cy), (60, 120, 60), 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - c_len), (cx, cy + c_len), (60, 120, 60), 1, cv2.LINE_AA)

    # Top Header Telemetry
    cv2.putText(
        frame,
        "// SYS.ISL_RECOG :: TERMINAL ONLINE // FEED: ACTIVE // PROTOCOL: v1.0",
        (16, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (80, 150, 80),
        1,
        cv2.LINE_AA
    )

    # Left: Target Prediction
    cv2.putText(frame, "TARGET >>", (16, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 215, 0), 1, cv2.LINE_AA)
    if prediction in ("Unknown", "No hand detected", "Waiting..."):
        pred_color = (0, 180, 255) if prediction == "Unknown" else (140, 150, 140)
        display_pred = f"[{prediction.upper()}]"
        cv2.putText(frame, display_pred, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"[ {prediction} ]", (16, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 65), 2, cv2.LINE_AA)

    # Center: Confidence Gauge
    is_locked = confidence >= CONFIDENCE_THRESHOLD
    gauge_color = (0, 255, 65) if is_locked else (0, 180, 255)
    gauge_status = "[LOCKED]" if is_locked else "[ACQUIRING]"

    bar_w = max(140, min(220, int(w * 0.22)))
    bar_h = 12
    bar_x = max(180, w // 2 - bar_w // 2)
    bar_y = 52

    cv2.putText(
        frame,
        f"CONF: {confidence * 100:.1f}% {gauge_status}",
        (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        gauge_color,
        1,
        cv2.LINE_AA
    )
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 90, 50), 1)
    fill_w = int((bar_w - 4) * max(0.0, min(1.0, confidence)))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x + 2, bar_y + 2), (bar_x + 2 + fill_w, bar_y + bar_h - 2), gauge_color, -1)

    # Right: Telemetry / FPS / Hands
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 65), 1, cv2.LINE_AA)
    hands_text = f"HANDS: {hands_count}/2 [{'ACTIVE' if hands_count > 0 else 'STANDBY'}]"
    cv2.putText(frame, hands_text, (w - 150, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 215, 0), 1, cv2.LINE_AA)

    # Bottom: Terminal Sentence Prompt
    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    cv2.putText(frame, "TERMINAL >>", (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 215, 0), 1, cv2.LINE_AA)
    
    # Render sentence text with cursor
    sentence_display = f"{sentence}{cursor}"
    cv2.putText(frame, sentence_display, (135, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 255, 240), 2, cv2.LINE_AA)

    # Bottom Right: Key bindings
    shortcuts = "[SPC] PUSH | [BKSP] POP | [C] CLR | [Q] EXIT"
    cv2.putText(frame, shortcuts, (max(16, w - 340), h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 150, 120), 1, cv2.LINE_AA)


def main():
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    detector = HandLandmarker(LANDMARK_MODEL)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera device at index 0")

    sentence = ""
    prediction = "Waiting..."
    confidence = 0.0

    prev_time = time.time()
    fps = 30.0

    print("========================================")
    print("  ISL RECOGNITION TERMINAL ENGAGED")
    print("  CONTROLS: [SPC] PUSH | [BKSP] POP | [C] CLR | [Q] EXIT")
    print("========================================")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Frame timing
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            result = detector.grab_hand_landmarks(frame)
            hands_count = len(result.hand_landmarks) if (result and result.hand_landmarks) else 0

            if hands_count > 0:
                overlay_hand_landmarks(frame, result)
                features = extract_features(result, world=True)
            else:
                features = None

            if features is not None:
                features = features.reshape(1, -1)
                probabilities = model.predict_proba(features)[0]
                index = np.argmax(probabilities)
                confidence = float(probabilities[index])

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction = encoder.inverse_transform([model.classes_[index]])[0]
                else:
                    prediction = "Unknown"
            else:
                prediction = "No hand detected"
                confidence = 0.0

            # Render Cyber HUD
            draw_hud(frame, prediction, confidence, sentence, fps, hands_count)

            cv2.imshow("ISL Tactical Recognition Terminal", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                if prediction not in ("Unknown", "No hand detected", "Waiting..."):
                    sentence += prediction
                    print(f"\r[TERMINAL BUFFER] >> {sentence} ", end="", flush=True)

            elif key == 8:  # Backspace
                sentence = sentence[:-1]
                print(f"\r[TERMINAL BUFFER] >> {sentence} ", end="", flush=True)

            elif key == ord('c'):
                sentence = ""
                print(f"\r[TERMINAL BUFFER] >> [CLEARED] ", end="", flush=True)

    except Exception as e:
        print("\nERR: ", e)

    finally:
        print("\nReleasing resources and closing terminal...")
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
