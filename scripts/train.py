
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns


DATA = Path("datasets/processed")
MODEL_DIR = Path("models")

X = np.load(DATA / "features.npy")
y = np.load(DATA / "labels.npy")

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Calculate balanced class weights explicitly
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
    class_weight=class_weights
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)

print(
    classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_
    )
)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Confusion matrix
classes = sorted(np.unique(y))

cm = confusion_matrix(
    y_test,
    predictions,
    labels=classes
)

plt.figure(figsize=(18, 15))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes,
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ISL Alphabet and Digit Recognition")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Save model
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_DIR / "isl_classifier.pkl")
joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

print("\nModel saved successfully!")
