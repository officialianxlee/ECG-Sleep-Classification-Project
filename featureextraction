import os
import numpy as np
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
INPUT_FOLDER = "processed_subjects"  
OUTPUT_FOLDER = "subject_features_clean"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(epoch):
    diff = np.diff(epoch)
    return [
        np.mean(epoch),
        np.std(epoch),
        np.min(epoch),
        np.max(epoch),
        np.median(epoch),
        np.percentile(epoch, 25),
        np.percentile(epoch, 75),
        np.var(epoch),
        np.ptp(epoch),
        np.mean(diff),
        np.std(diff)
    ]

# -----------------------------
# Loop Over Subjects
# -----------------------------
saved_files = []

for fname in tqdm(sorted(os.listdir(INPUT_FOLDER))):
    if fname.endswith(".npz"):
        raw = np.load(os.path.join(INPUT_FOLDER, fname))
        signals = raw["ecg"]
        labels = raw["labels"]

        # Drop undefined (label 4)
        mask = labels != 4
        signals = signals[mask]
        labels = labels[mask]

        # Extract features per epoch
        features = np.array([extract_features(epoch) for epoch in signals])

        # Save to clean folder
        save_path = os.path.join(OUTPUT_FOLDER, fname)
        np.savez(save_path, features=features, labels=labels)
        saved_files.append(fname)


