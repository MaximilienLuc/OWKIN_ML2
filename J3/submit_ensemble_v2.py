"""
OWKIN ML2 — Submission améliorée v2
====================================
Stratégie : Ensemble de plusieurs approches pour stabiliser les prédictions.

Ensembles :
  A) Mean pooling (2048) → PCA → LogReg L2 (baseline amélioré)  
  B) Mean pooling (2048) → LogReg L2 (sans PCA, comme le baseline original)
  C) Mean+Std+Max (6144) → PCA → LogReg L2
  D) Mean pooling (2048) → PCA → LogReg L1

Chaque modèle est un ensemble 5-fold × 5 repeats (25 modèles).
La prédiction finale est la moyenne de tous les modèles.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

# ── Config ──
DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
TRAIN_FEATURES_DIR = DATA_DIR / "train_input" / "moco_features"
TEST_FEATURES_DIR = DATA_DIR / "test_input" / "moco_features"
N_SPLITS = 5
N_REPEATS = 5

# ── Chargement ──
print("=" * 60)
print("Chargement des données")
print("=" * 60)
df_train = pd.read_csv(DATA_DIR / "supplementary_data" / "train_metadata.csv")
df_test = pd.read_csv(DATA_DIR / "supplementary_data" / "test_metadata.csv")
y_train_df = pd.read_csv(DATA_DIR / "train_output.csv")
df_train = df_train.merge(y_train_df, on="Sample ID")


def extract_features(sample, features_dir):
    _features = np.load(features_dir / sample)
    features = _features[:, 3:]
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    max_feat = np.max(features, axis=0)
    q25 = np.percentile(features, 25, axis=0)
    q75 = np.percentile(features, 75, axis=0)
    return mean_feat, std_feat, max_feat, q25, q75


print("Extraction des features (Train)...")
train_results = Parallel(n_jobs=-1)(
    delayed(extract_features)(s, TRAIN_FEATURES_DIR)
    for s in tqdm(df_train["Sample ID"].values, desc="Train")
)
y_train = df_train["Target"].values.astype(int)
patients_train = df_train["Patient ID"].values

mean_train = np.array([r[0] for r in train_results])
std_train = np.array([r[1] for r in train_results])
max_train = np.array([r[2] for r in train_results])
q25_train = np.array([r[3] for r in train_results])
q75_train = np.array([r[4] for r in train_results])

print("Extraction des features (Test)...")
test_results = Parallel(n_jobs=-1)(
    delayed(extract_features)(s, TEST_FEATURES_DIR)
    for s in tqdm(df_test["Sample ID"].values, desc="Test")
)

mean_test = np.array([r[0] for r in test_results])
std_test = np.array([r[1] for r in test_results])
max_test = np.array([r[2] for r in test_results])
q25_test = np.array([r[3] for r in test_results])
q75_test = np.array([r[4] for r in test_results])

# Patient-level setup
patients_unique = np.unique(patients_train)
y_unique = np.array([int(np.mean(y_train[patients_train == p])) for p in patients_unique])

print(f"Mean: {mean_train.shape}, Std: {std_train.shape}")


def run_pipeline(name, X_tr, X_te, y, patients, patients_uniq, y_uniq,
                 use_pca=True, n_components=100, C=0.01, penalty="l2"):
    """Entraîne un ensemble 5-fold × 5-repeats et retourne les prédictions test."""
    print(f"\n  [{name}] X_train={X_tr.shape}, PCA={use_pca}({n_components}), "
          f"C={C}, penalty={penalty}")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    all_preds = np.zeros(len(X_te))
    all_aucs = []
    n_models = 0

    for k in range(N_REPEATS):
        kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=k)
        for train_idx_, val_idx_ in kfold.split(patients_uniq, y_uniq):
            train_mask = np.isin(patients, patients_uniq[train_idx_])
            val_mask = np.isin(patients, patients_uniq[val_idx_])

            Xf_tr = X_tr_s[train_mask]
            yf_tr = y[train_mask]
            Xf_val = X_tr_s[val_mask]
            yf_val = y[val_mask]
            Xf_te = X_te_s.copy()

            if use_pca:
                nc = min(n_components, Xf_tr.shape[0] - 1, Xf_tr.shape[1])
                pca = PCA(n_components=nc, random_state=42)
                Xf_tr = pca.fit_transform(Xf_tr)
                Xf_val = pca.transform(Xf_val)
                Xf_te = pca.transform(Xf_te)

            solver = "lbfgs" if penalty == "l2" else "liblinear"
            clf = LogisticRegression(C=C, penalty=penalty, solver=solver,
                                     max_iter=3000, random_state=42)
            clf.fit(Xf_tr, yf_tr)

            preds_val = clf.predict_proba(Xf_val)[:, 1]
            all_aucs.append(roc_auc_score(yf_val, preds_val))
            all_preds += clf.predict_proba(Xf_te)[:, 1]
            n_models += 1

    all_preds /= n_models
    mean_auc = np.mean(all_aucs)
    print(f"  [{name}] CV AUC = {mean_auc:.4f} ± {np.std(all_aucs):.4f}")
    return all_preds, mean_auc


# ============================================================
# PIPELINES
# ============================================================
print("\n" + "=" * 60)
print("Entraînement des pipelines")
print("=" * 60)

# Construire différentes matrices de features
X_mean_tr, X_mean_te = mean_train, mean_test
X_full_tr = np.hstack([mean_train, std_train, max_train])
X_full_te = np.hstack([mean_test, std_test, max_test])
X_extended_tr = np.hstack([mean_train, std_train, max_train, q25_train, q75_train])
X_extended_te = np.hstack([mean_test, std_test, max_test, q25_test, q75_test])

pipelines = []

# A) Baseline amélioré : Mean → PCA → LogReg L2
for C in [0.005, 0.01, 0.05, 0.1]:
    for nc in [50, 100, 150]:
        pipelines.append((
            f"Mean_PCA{nc}_L2_C{C}",
            X_mean_tr, X_mean_te, True, nc, C, "l2"
        ))

# B) Mean → pas de PCA → LogReg L2 (baseline original)
for C in [0.005, 0.01, 0.05, 0.1]:
    pipelines.append((
        f"Mean_noPCA_L2_C{C}",
        X_mean_tr, X_mean_te, False, 0, C, "l2"
    ))

# C) Mean+Std+Max → PCA → LogReg L2
for C in [0.005, 0.01, 0.05]:
    for nc in [100, 150, 200]:
        pipelines.append((
            f"Full_PCA{nc}_L2_C{C}",
            X_full_tr, X_full_te, True, nc, C, "l2"
        ))

# D) Mean → PCA → LogReg L1
for C in [0.01, 0.05, 0.1]:
    for nc in [50, 100]:
        pipelines.append((
            f"Mean_PCA{nc}_L1_C{C}",
            X_mean_tr, X_mean_te, True, nc, C, "l1"
        ))

# E) Extended (Mean+Std+Max+Q25+Q75) → PCA → L2
for C in [0.01, 0.05]:
    for nc in [100, 150]:
        pipelines.append((
            f"Ext_PCA{nc}_L2_C{C}",
            X_extended_tr, X_extended_te, True, nc, C, "l2"
        ))

# Run all pipelines
results = {}
for name, Xtr, Xte, use_pca, nc, C, pen in pipelines:
    preds, auc = run_pipeline(
        name, Xtr, Xte, y_train, patients_train,
        patients_unique, y_unique, use_pca, nc, C, pen
    )
    results[name] = (preds, auc)

# ============================================================
# ENSEMBLE : Moyenne pondérée par AUC des top-K modèles
# ============================================================
print("\n" + "=" * 60)
print("Ensembling")
print("=" * 60)

# Trier par AUC décroissant
sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)

print("\nTop 10 modèles :")
for i, (name, (_, auc)) in enumerate(sorted_results[:10]):
    print(f"  {i+1:2d}. {name:35s}  AUC = {auc:.4f}")

# Ensemble des top-K modèles, pondéré par AUC
for TOP_K in [3, 5, 8, 10, 15]:
    top_models = sorted_results[:TOP_K]
    weights = np.array([auc for _, (_, auc) in top_models])
    weights = weights / weights.sum()

    preds_ensemble = np.zeros(len(mean_test))
    for (name, (preds, auc)), w in zip(top_models, weights):
        preds_ensemble += w * preds

    print(f"\n  Ensemble Top-{TOP_K}: mean={preds_ensemble.mean():.4f}, "
          f"std={preds_ensemble.std():.4f}")

    # Save
    sub = pd.DataFrame({
        "Sample ID": df_test["Sample ID"].values,
        "Target": preds_ensemble
    }).sort_values("Sample ID")
    assert all(sub["Target"].between(0, 1))
    assert sub.shape == (149, 2)
    path = DATA_DIR / f"pca_ensemble_top{TOP_K}_submission.csv"
    sub.to_csv(path, index=None)
    print(f"  → {path}")

# Aussi sauver le meilleur modèle seul
best_name, (best_preds, best_auc) = sorted_results[0]
sub_best = pd.DataFrame({
    "Sample ID": df_test["Sample ID"].values,
    "Target": best_preds
}).sort_values("Sample ID")
sub_best.to_csv(DATA_DIR / "pca_best_single_submission.csv", index=None)
print(f"\n  Meilleur single : {best_name} (AUC CV = {best_auc:.4f})")
print(f"  → {DATA_DIR / 'pca_best_single_submission.csv'}")

print("\n✅ Terminé !")
