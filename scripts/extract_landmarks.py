
import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from src.hand_landmarker   import HandLandmarker
from src.feature_extractor import extract_features


DATASET = Path("datasets/isl_dataset")
OUTPUT = Path("datasets/processed")

MODEL_PATH = "models/hand_landmarker.task"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    detector = HandLandmarker(MODEL_PATH)

    features = []
    labels = []
    failures = {}

    classes = sorted(
        folder.name
        for folder in DATASET.iterdir()
        if folder.is_dir()
        and not folder.name.startswith(".")
    )

    try:
        for label in classes:
            folder = DATASET / label

            images = [
                p for p in folder.rglob("*")
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
                and ".ipynb_checkpoints" not in p.parts
            ]

            failed = 0

            for path in tqdm(images, desc=f"Class {label}"):
                image = cv2.imread(str(path))

                if image is None:
                    failed += 1
                    continue

                result = detector.grab_hand_landmarks(image)

                vector = extract_features(result, world=True)

                if vector is None:
                    failed += 1
                    continue

                features.append(vector)
                labels.append(label)

            failures[label] = failed

    finally:
        detector.close()

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    np.save(OUTPUT / "features.npy", X)
    np.save(OUTPUT / "labels.npy", y)
    np.save(OUTPUT / "classes.npy", np.asarray(classes))

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Features:", X.shape)
    print("Labels:", y.shape)

    for label, count in failures.items():
        print(f"{label}: {count} failures")


if __name__ == "__main__":
    main()
