from load_data import load_subject
from preprocess import create_epochs, bandpass_filter
from train_svm import train_svm

raw = load_subject(1)

epochs = create_epochs(raw)
epochs = bandpass_filter(epochs)

X = epochs.get_data()
y = epochs.events[:, -1]

print("Dataset:", X.shape)

model = train_svm(X, y)