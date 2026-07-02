from load_data import load_subject
from preprocess import create_epochs, bandpass_filter
from train_svm import train_svm

# Load data
raw = load_subject("datasets/A01T.gdf")

# Preprocess
epochs = create_epochs(raw)
epochs = bandpass_filter(epochs)

# Convert to NumPy
X = epochs.get_data()
y = epochs.events[:, -1]

print("Dataset Shape:", X.shape)

# Train baseline model
train_svm(X, y)