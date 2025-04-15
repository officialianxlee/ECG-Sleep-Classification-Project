# Sleep Stage Classification from ECG  
**A Comparative Study of Traditional Machine Learning and Deep Learning Models**

## 🧠 Overview
This project evaluates and compares machine learning (ML) and deep learning (DL) models for classifying sleep stages using ECG signals. The analysis is based on a subset of the PhysioNet/Computing in Cardiology Challenge 2018 dataset. We extracted statistical and wavelet-based features for classical ML models and trained DL models on raw ECG signals to analyze performance under consistent experimental conditions.

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `saveasnpz.py` | Converts PhysioNet `.mat` files to `.npz` format for easier handling in Python. |
| `featureextraction.py` | Extracts basic statistical features (mean, std, etc.) from 30-second ECG epochs. |
| `cwtfeatureextraction.py` | Applies Continuous Wavelet Transform (CWT) using the Morlet wavelet and extracts mean, std, energy, and entropy from each scale. |
| `MLmodels.py` | Contains implementations of Random Forest, XGBoost, and Multilayer Perceptron using engineered features. |
| `CNNonlymodel.py` | Implements a 1D Convolutional Neural Network for sleep staging using raw ECG input. |
| `CNN+BiLSTMmodel.py` | Combines a CNN with a Bidirectional LSTM to capture both spatial and temporal patterns. |
| `MultiScalemodel.py` | Stacked model that integrates CNN, Transformer Encoder, and BiLSTM for multi-scale temporal learning. |

## 🧪 Dataset
- **PhysioNet Challenge 2018**: A large polysomnographic dataset including ECG, EEG, EOG, EMG, and annotations for sleep staging.
- This project uses a **subset of 100 subjects**, and only the ECG channel is used.

## ⚙️ Feature Engineering
- **Statistical features**: Mean, standard deviation, min, max, percentiles, and differences across 30-second epochs.
- **Wavelet features**: CWT using Morlet wavelet at scales 1–31. Extracted mean, std, energy, and entropy from each scale.

## 🧠 Model Architectures
### Machine Learning
- **Random Forest**
- **XGBoost**
- **Multilayer Perceptron (MLP)**

### Deep Learning
- **CNN**: Basic 1D CNN with 3 convolutional layers.
- **CNN + BiLSTM**: Combines spatial and temporal feature extraction.
- **Multi-Scaled Model**: CNN → Transformer Encoder → BiLSTM → Classifier head.

## 📊 Evaluation Metrics
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

All models were evaluated under the same conditions using 30-second ECG segments and stratified train-validation-test splits.

## 🚀 Future Directions
- Incorporate multimodal signals (e.g., respiration).
- Explore deeper architectures and self-supervised pretraining.
- Address class imbalance using advanced resampling or loss weighting strategies.

## 📌 Citation
If you use this codebase, please cite or acknowledge:
**Ian Lee & Aarushi Bhardwaj – Sleep Stage Classification from ECG: A Comparative Study of Traditional Machine Learning and Deep Learning Models**

## 📬 Contact
For questions or collaboration:
- [GitHub](https://github.com/officialianxlee)
- Email: ianx.lee@mail.utoronto.ca
