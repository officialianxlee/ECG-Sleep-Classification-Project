import os
import numpy as np
import pywt
from scipy.stats import entropy
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
INPUT_FOLDER = "subject_features_clean"
OUTPUT_FOLDER = "subject_features_cwt"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

wavelet = 'morl'
scales = np.arange(1, 32)  # adjust as needed

def extract_cwt_features(signal, wavelet='morl', scales=np.arange(1, 32)):
    """
    Apply CWT and extract summary features for each scale.
    Returns a flat feature vector: [mean, std, energy, entropy per scale]
    """
    coeffs, _ = pywt.cwt(signal, scales, wavelet)
    features = []

    for scale_coeffs in coeffs:
        power = np.abs(scale_coeffs) ** 2
        scale_entropy = entropy(power / np.sum(power) + 1e-8)
        features.extend([
            np.mean(scale_coeffs),
            np.std(scale_coeffs),
            np.sum(power),
            scale_entropy
        ])
    return features  # (num_scales × 4)

# -----------------------
# Loop through each subject file
# -----------------------
for fname in tqdm(os.listdir(INPUT_FOLDER)):
    if not fname.endswith(".npz"):
        continue

    # Load feature file
    feature_path = os.path.join(INPUT_FOLDER, fname)
    feat_data = np.load(feature_path)
    base_features = feat_data["features"]
    labels = feat_data["labels"]

    # Load matching ECG file
    ecg_path = os.path.join("processed_subjects", fname)
    ecg_data = np.load(ecg_path)
    ecg = ecg_data["ecg"]
    raw_labels = ecg_data["labels"]

    # Mask to drop undefined epochs (label == 4)
    mask = raw_labels != 4
    signals = ecg[mask]

    # Sanity check
    assert len(signals) == len(base_features) == len(labels), f"{fname} mismatch: signals={len(signals)}, features={len(base_features)}, labels={len(labels)}"

    # Compute CWT
    all_cwt_features = []
    for epoch in signals:
        cwt_feats = extract_cwt_features(epoch)
        all_cwt_features.append(cwt_feats)

    all_cwt_features = np.array(all_cwt_features)
    combined = np.concatenate([base_features, all_cwt_features], axis=1)

    save_path = os.path.join(OUTPUT_FOLDER, fname)
    np.savez(save_path, features=combined, labels=labels)
    print(f"Saved enriched CWT features: {save_path}")
