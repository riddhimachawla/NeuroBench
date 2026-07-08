import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from braindecode.models import EEGNet

from src.dataset_loader import EEGDataset as Loader
from src.preprocessing import notch_filter, bandpass_filter
from src.dataset import EEGDataset
from src.trainer import Trainer

# ==========================================================
# Configuration
# ==========================================================

DATASET = "datasets/A02T.gdf"

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Using Device:", DEVICE)
print("=" * 60)

# ==========================================================
# Load Dataset
# ==========================================================

loader = Loader(DATASET)

raw = loader.load_raw()

raw = notch_filter(raw)

raw = bandpass_filter(raw)

events = loader.get_events(raw)

X, y = loader.create_epochs(raw, events)

print("Dataset Shape:", X.shape)

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# PyTorch Dataset
# ==========================================================

train_dataset = EEGDataset(
    X_train,
    y_train
)

test_dataset = EEGDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================================
# EEGNet
# ==========================================================

model = EEGNet(
    n_chans=22,
    n_outputs=4,
    n_times=1001,
    final_conv_length="auto",
    drop_prob=0.25,
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
    epochs=EPOCHS,
)

trainer.train()

history = trainer.get_history()

print()

print("=" * 60)

print("Training Complete")

print("=" * 60)

print("Best Model Saved To:")

print("models/eegnet_best.pt")