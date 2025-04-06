# Code to convert the .mat files from the training folder of the CinC 2018 Sleep Dataset to .npz for ease of use
import os
import numpy as np
import h5py
import scipy.io

# Constants
SAMPLE_RATE = 200
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLE_RATE * EPOCH_SECONDS
RAW_DATA_PATH = "training"
SAVE_DIR = "processed_subjects"

os.makedirs(SAVE_DIR, exist_ok=True)

def convert_to_npz(subject_id):
    subject_path = os.path.join(RAW_DATA_PATH, subject_id)
    mat_path = os.path.join(subject_path, f"{subject_id}.mat")
    arousal_path = os.path.join(subject_path, f"{subject_id}-arousal.mat")

    try:
        ecg_mat = scipy.io.loadmat(mat_path)
        ecg_signal = ecg_mat['val'][-1]

        with h5py.File(arousal_path, 'r') as f:
            stages = f['data']['sleep_stages']
            stage_array = np.zeros(ecg_signal.shape[0], dtype=np.int64)
            for i, key in enumerate(['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']):
                mask = stages[key][:].flatten()
                stage_array[mask == 1] = i

        n_epochs = ecg_signal.shape[0] // SAMPLES_PER_EPOCH
        ecg_epochs = ecg_signal[:n_epochs * SAMPLES_PER_EPOCH].reshape(n_epochs, SAMPLES_PER_EPOCH)
        label_epochs = stage_array[:n_epochs * SAMPLES_PER_EPOCH].reshape(n_epochs, SAMPLES_PER_EPOCH)
        majority_labels = np.array([np.bincount(epoch).argmax() for epoch in label_epochs])

        save_path = os.path.join(SAVE_DIR, f"{subject_id}.npz")
        np.savez(save_path, ecg=ecg_epochs, labels=majority_labels)
        print(f" Saved: {save_path}")

    except Exception as e:
        print(f" Failed: {subject_id} — {e}")

# Batch process all subjects
for subject_id in os.listdir(RAW_DATA_PATH):
    if os.path.isdir(os.path.join(RAW_DATA_PATH, subject_id)):
        convert_to_npz(subject_id)
