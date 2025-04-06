import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------
# Configuration
# -----------------------------
DATA_FOLDER = FEATURE_FOLDER
MAX_SUBJECTS = 100
SEED = 42

# -----------------------------
# Load Preprocessed Feature Files
# -----------------------------
all_features, all_labels = [], []
loaded = 0

for fname in sorted(os.listdir(DATA_FOLDER)):
    if fname.endswith(".npz"):
        path = os.path.join(DATA_FOLDER, fname)
        data = np.load(path)
        X = data["features"]
        y = data["labels"]
        all_features.append(X)
        all_labels.append(y)
        loaded += 1
        if loaded >= MAX_SUBJECTS:
            break

X_all = np.concatenate(all_features, axis=0)
y_all = np.concatenate(all_labels, axis=0)

# -----------------------------
# Encode Labels
# -----------------------------
le = LabelEncoder()
y_all = le.fit_transform(y_all)
print(f"Loaded: {X_all.shape}, Labels: {y_all.shape}, Classes: {le.classes_}")

# -----------------------------
# Normalize Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# -----------------------------
# Train-Test Split (before SMOTE)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.2, stratify=y_all, random_state=SEED)

# -----------------------------
# SMOTE for Class Balancing (on Training Set Only)
# -----------------------------
smote = SMOTE(random_state=SEED)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("After SMOTE:", Counter(y_train_resampled))

# -----------------------------
# Train Models
# -----------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=SEED
    ),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, alpha=0.001,
                         early_stopping=True, random_state=SEED)
}

results = {}

for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="viridis")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    results[name] = classification_report(y_test, y_pred, output_dict=True)

# Convert results to DataFrame
import pandas as pd
df_results = pd.DataFrame({name: res['weighted avg'] for name, res in results.items()}).T
df_results = df_results[['precision', 'recall', 'f1-score', 'support']]
df_results
