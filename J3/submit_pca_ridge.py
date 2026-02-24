"""
Submission rapide — PCA(150) + LogisticRegression L2 (C=0.01)
Entraîné sur TOUT le dataset train, pas de CV.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

# ── Config ──
DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
TRAIN_FEATURES_DIR = DATA_DIR / "train_input" / "moco_features"
TEST_FEATURES_DIR = DATA_DIR / "test_input" / "moco_features"
N_COMPONENTS = 150
C = 0.01

# ── Chargement ──
print("Chargement des métadonnées...")
df_train = pd.read_csv(DATA_DIR / "supplementary_data" / "train_metadata.csv")
df_test = pd.read_csv(DATA_DIR / "supplementary_data" / "test_metadata.csv")
y_train_df = pd.read_csv(DATA_DIR / "train_output.csv")
df_train = df_train.merge(y_train_df, on="Sample ID")


def process_sample(sample_info, features_dir, is_train=True):
    if is_train:
        sample, label = sample_info
    else:
        sample = sample_info
    _features = np.load(features_dir / sample)
    features = _features[:, 3:]
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    max_feat = np.max(features, axis=0)
    concatenated = np.concatenate([mean_feat, std_feat, max_feat])
    if is_train:
        return concatenated, label
    return concatenated


# ── Feature extraction ──
print("Extraction des features (Train)...")
train_results = Parallel(n_jobs=-1)(
    delayed(process_sample)(row, TRAIN_FEATURES_DIR, True)
    for row in tqdm(df_train[["Sample ID", "Target"]].values, desc="Train")
)
X_train = np.array([r[0] for r in train_results])
y_train = np.array([r[1] for r in train_results], dtype=int)

print("Extraction des features (Test)...")
X_test = np.array(Parallel(n_jobs=-1)(
    delayed(process_sample)(s, TEST_FEATURES_DIR, False)
    for s in tqdm(df_test["Sample ID"].values, desc="Test")
))

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ── StandardScaler + PCA ──
print(f"StandardScaler + PCA(n_components={N_COMPONENTS})...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

var_explained = np.sum(pca.explained_variance_ratio_)
print(f"  Variance expliquée avec {N_COMPONENTS} composantes : {var_explained:.2%}")

# ── Entraînement sur tout le train ──
print(f"Entraînement LogisticRegression(C={C}, L2) sur tout le train...")
clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42)
clf.fit(X_train_pca, y_train)

# ── Prédiction ──
preds_test = clf.predict_proba(X_test_pca)[:, 1]

# ── Submission ──
submission = pd.DataFrame({
    "Sample ID": df_test["Sample ID"].values,
    "Target": preds_test
}).sort_values("Sample ID")

assert all(submission["Target"].between(0, 1))
assert submission.shape == (149, 2)
assert list(submission.columns) == ["Sample ID", "Target"]

output_path = DATA_DIR / "pca_ridge_submission.csv"
submission.to_csv(output_path, index=None)

print(f"\n✓ Soumission sauvegardée : {output_path}")
print(f"  Target mean = {preds_test.mean():.4f}")
print(f"  Target std  = {preds_test.std():.4f}")
print(submission.head(10).to_string(index=False))
