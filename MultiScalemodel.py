import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import math
# -------------------------
# Configuration
# -------------------------
class Config:
    # Data
    data_path = "processed_subjects"
    batch_size = 32
    num_workers = 2

    # Model
    input_size = 6000
    num_classes = 5

    # Training
    epochs = 50
    lr = 3e-4
    weight_decay = 1e-3
    early_stop_patience = 5
    use_amp = True
    valid_ratio = 0.1
    test_ratio = 0.1

    # System
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = True

# -------------------------
# Dataset Class
# -------------------------
class SleepECGDataset(Dataset):
    def __init__(self, npz_folder, max_subjects=100):
        self.epochs = []
        self.labels = []
        self.label_map = {0:1, 1:2, 2:3, 3:4, 5:0}

        files = sorted(f for f in os.listdir(npz_folder) if f.endswith('.npz'))
        if max_subjects:
            files = files[:max_subjects]

        for fname in tqdm(files, desc="Loading data"):
            try:
                with np.load(os.path.join(npz_folder, fname)) as data:
                    ecg = data["ecg"].astype(np.float32)
                    labels = data["labels"].astype(np.int64)
                    mask = labels != 4
                    self.epochs.append(ecg[mask])
                    self.labels.append(np.vectorize(self.label_map.get)(labels[mask]))
            except Exception as e:
                print(f"Error loading {fname}: {str(e)}")
                continue

        self.epochs = np.concatenate(self.epochs)
        self.labels = np.concatenate(self.labels)
        self._check_label_distribution()

    def _check_label_distribution(self):
        print("\n📊 Label Distribution:")
        for i, name in enumerate(['Wake', 'N1', 'N2', 'N3', 'REM']):
            print(f"{name}: {np.sum(self.labels == i):,} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.epochs[idx], dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(self.labels[idx], dtype=torch.long)
        )

# -------------------------
# CNN Model
# -------------------------
class SleepECGNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=51, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),


            nn.Conv1d(32, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),


            nn.Conv1d(64, 64, kernel_size=13, padding=6),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,  # Matches CNN output channels
                nhead=4,
                dim_feedforward=128,
                dropout=0.2,
                batch_first=True),
            num_layers=2)


        self.lstm = nn.LSTM(64, 64, bidirectional=True, batch_first=True)


        self.fc = nn.Sequential(
            nn.Linear(128, 64),  # 128 because of bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 5))

    def forward(self, x):
        x = x.float()
        # CNN feature extraction
        cnn_out = self.cnn(x)
        # Prepare for transformer (batch, seq_len, channels)
        cnn_out = cnn_out.permute(0, 2, 1)
        # Transformer
        attn_out = self.transformer(cnn_out)
        # LSTM
        lstm_out, _ = self.lstm(attn_out)
        # Global average pooling and classification
        return self.fc(torch.mean(lstm_out, dim=1))

# -------------------------
# Training Functions
# -------------------------
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(X)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred):
    plt.figure(figsize=(8, 6))  # Adjust the size as needed
    sns.heatmap(confusion_matrix(y_true, y_pred),
                annot=True, fmt='d', cmap='viridis',
                xticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'],
                yticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# -------------------------
# Training Function
# -------------------------
def train_model(config):
    # Initialize
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # Data
    print("Loading dataset...")
    dataset = SleepECGDataset(config.data_path)

    # Compute class weights for imbalance handling
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(dataset.labels),
        y=dataset.labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.device)

    print(f"\nClass Weights: {class_weights}")

    # Split
    train_size = int((1 - config.valid_ratio - config.test_ratio) * len(dataset))
    val_size = int(config.valid_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42))

    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory)

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size*2,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory)

    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size*2,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory)

    # Model
    model = SleepECGNet().to(config.device)
    optimizer = optim.AdamW(model.parameters(),
                          lr=config.lr,
                          weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing = 0.05)


    # Training
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print("\nStarting Training...")
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            X, y = X.to(config.device, non_blocking=True), y.to(config.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=config.use_amp):
                outputs = model(X)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(config.device), y.to(config.device)
                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    val_loss += criterion(model(X), y).item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'ms_model.pth')

        print(f"\nEpoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {time.time()-start_time:.1f}s")

    # Evaluation
    model.load_state_dict(torch.load('ms_model.pth'))
    y_true, y_pred = evaluate_model(model, test_loader, config.device)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                              target_names=['Wake', 'N1', 'N2', 'N3', 'REM'],
                              digits=4))

    plot_confusion_matrix(y_true, y_pred)

    torch.save(model.state_dict(), 'ms_model.pth')
    print("\nModels saved successfully")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    config = Config()

    print("\n🛠️ System Diagnostics:")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print(f"CPU Cores: {os.cpu_count()}")
    print(f"Data Samples: {len([f for f in os.listdir(config.data_path) if f.endswith('.npz')])}")

    train_model(config)
