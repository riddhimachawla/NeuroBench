import os
import joblib
import numpy as np
import pandas as pd

from src.visualization import Visualizer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.dataset_loader import EEGDataset
from src.preprocessing import bandpass_filter, notch_filter
from src.feature_extraction import CSPExtractor
from src.classical_models import get_models
from src.hyperparameter_tuning import HyperparameterTuner

# ==========================================================
# Configuration
# ==========================================================

DATASET = "datasets/A02T.gdf"

os.makedirs("models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading EEG Dataset...")
print("=" * 60)

loader = EEGDataset(DATASET)

raw = loader.load_raw()

print("Applying Notch Filter...")
raw = notch_filter(raw)

print("Applying Bandpass Filter...")
raw = bandpass_filter(raw)

print("Extracting Events...")
events = loader.get_events(raw)

print("Creating Epochs...")
X, y = loader.create_epochs(raw, events)

print("\nDataset Loaded Successfully")
print(f"EEG Shape : {X.shape}")
print(f"Labels    : {y.shape}")

# ==========================================================
# Feature Extraction
# ==========================================================

print("\n" + "=" * 60)
print("Running CSP Feature Extraction...")
print("=" * 60)

csp = CSPExtractor(n_components=8)

X = csp.fit_transform(X, y)

print("Feature Shape:", X.shape)

# ==========================================================
# Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ==========================================================
# Model Training
# ==========================================================

models = get_models()

visualizer = Visualizer()

tuner = HyperparameterTuner()

results = []

best_accuracy = 0
best_model = None
best_name = None

print("\n" + "=" * 60)
print("Training Classical ML Models")
print("=" * 60)

best_accuracy = 0

best_model = None

best_name = None

for name, model in models.items():

    print("\n")

    print("=" * 60)

    print(name)

    print("=" * 60)

    if name == "SVM (RBF)":

        model = tuner.tune_svm(

            model,

            X_train,

            y_train

        )

    cv = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="accuracy"

    )

    model.fit(

        X_train,

        y_train

    )

    predictions = model.predict(

        X_test

    )

    accuracy = accuracy_score(

        y_test,

        predictions

    )

    print()

    print("Accuracy")

    print(f"{accuracy:.4f}")

    print()

    print("Cross Validation")

    print(f"{cv.mean():.4f}")

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Cross Validation": cv.mean()

    })

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_name = name
# ==========================================================
# Save Best Model
# ==========================================================

print("\n" + "=" * 60)
print("Best Model")
print("=" * 60)

print(f"Model    : {best_name}")
print(f"Accuracy : {best_accuracy:.4f}")

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(csp, "models/csp.pkl")

print("\nSaved:")
print("models/best_model.pkl")
print("models/csp.pkl")

# ==========================================================
# Evaluation
# ==========================================================

predictions = best_model.predict(X_test)

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    "results/metrics/classification_report.csv",
    index=True,
)

cm = confusion_matrix(
    y_test,
    predictions,
)

np.savetxt(
    "results/metrics/confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d",
)

# ==========================================================
# Model Comparison
# ==========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/metrics/model_comparison.csv",
    index=False,
)
class_names = [
    "Left",
    "Right",
    "Feet",
    "Tongue"
]

visualizer.plot_confusion_matrix(
    cm,
    class_names
)

visualizer.plot_accuracy(
    results_df
)

visualizer.plot_cross_validation(
    results_df
)

# ==========================================================
# Print Results
# ==========================================================

print("\n")
print("=" * 60)
print("Model Comparison")
print("=" * 60)

print(results_df)

print("\n")
print("=" * 60)
print("Classification Report")
print("=" * 60)

print(report_df)

print("\n")
print("=" * 60)
print("Project Completed Successfully!")
print("=" * 60)