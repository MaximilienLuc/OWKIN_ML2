"""
OWKIN ML2 Challenge — Iteration 3: PCA + Regularization
========================================================
Objectif : Réduire la dimensionnalité des features Mean/Std/Max (6144 dims)
           via PCA, puis comparer des classifieurs régularisés.

Pipeline :
  1. Chargement des données (moco_features .npy)
  2. Feature engineering : Mean + Std + Max pooling → (344, 6144)
  3. StandardScaler (fit sur train uniquement)
  4. PCA : recherche du nb optimal de composantes
  5. Comparaison de modèles régularisés (Ridge, Lasso, ElasticNet)
  6. Cross-validation patient-level 5-fold × 5 repeats
  7. Ensembling + Submission
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed


# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
TRAIN_FEATURES_DIR = DATA_DIR / "train_input" / "moco_features"
TEST_FEATURES_DIR = DATA_DIR / "test_input" / "moco_features"
OUTPUT_DIR = DATA_DIR
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# PCA: candidats pour le nombre de composantes
PCA_CANDIDATES = [50, 100, 150, 200, 250]

# Régularisation: valeurs de C à tester
C_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

# CV
N_SPLITS = 5
N_REPEATS = 5


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================
print("=" * 60)
print("ÉTAPE 1 : Chargement des données")
print("=" * 60)

df_train = pd.read_csv(DATA_DIR / "supplementary_data" / "train_metadata.csv")
df_test = pd.read_csv(DATA_DIR / "supplementary_data" / "test_metadata.csv")
y_train_df = pd.read_csv(DATA_DIR / "train_output.csv")
df_train = df_train.merge(y_train_df, on="Sample ID")

print(f"  Train samples : {len(df_train)}")
print(f"  Test samples  : {len(df_test)}")


# ============================================================
# 3. FEATURE ENGINEERING : Mean + Std + Max pooling
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 2 : Feature Engineering (Mean + Std + Max)")
print("=" * 60)


def process_sample(sample_info, features_dir, is_train=True):
    """Extrait Mean, Std, Max des features MoCo pour une slide."""
    if is_train:
        sample, label, center, patient = sample_info
    else:
        sample = sample_info

    _features = np.load(features_dir / sample)
    coordinates, features = _features[:, :3], _features[:, 3:]

    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    max_feat = np.max(features, axis=0)

    concatenated = np.concatenate([mean_feat, std_feat, max_feat])
    if is_train:
        return concatenated, label, center, patient
    return concatenated


# Extraction parallèle — Train
print("  Extraction des features (Train)...")
train_results = Parallel(n_jobs=-1)(
    delayed(process_sample)(row, TRAIN_FEATURES_DIR, True)
    for row in tqdm(
        df_train[["Sample ID", "Target", "Center ID", "Patient ID"]].values,
        desc="  Train"
    )
)
X_train = np.array([r[0] for r in train_results])
y_train = np.array([r[1] for r in train_results], dtype=int)
centers_train = np.array([r[2] for r in train_results])
patients_train = np.array([r[3] for r in train_results])

# Extraction parallèle — Test
print("  Extraction des features (Test)...")
X_test_list = Parallel(n_jobs=-1)(
    delayed(process_sample)(sample, TEST_FEATURES_DIR, False)
    for sample in tqdm(df_test["Sample ID"].values, desc="  Test")
)
X_test = np.array(X_test_list)

print(f"  X_train shape : {X_train.shape}")  # (344, 6144)
print(f"  X_test shape  : {X_test.shape}")    # (149, 6144)


# ============================================================
# 4. PATIENT-LEVEL SPLIT SETUP
# ============================================================
patients_unique = np.unique(patients_train)
y_unique = np.array(
    [int(np.mean(y_train[patients_train == p])) for p in patients_unique]
)
centers_unique = np.array(
    [centers_train[patients_train == p][0] for p in patients_unique]
)
print(f"\n  Patients uniques  : {len(patients_unique)}")
print(f"  Centres uniques   : {len(np.unique(centers_unique))}")


# ============================================================
# 5. PCA — ANALYSE DE LA VARIANCE EXPLIQUÉE
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 3 : Analyse PCA — Variance expliquée")
print("=" * 60)

# StandardScaler fit sur train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA complet pour analyser la variance
pca_full = PCA(n_components=min(X_train_scaled.shape))
pca_full.fit(X_train_scaled)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

# Trouver les seuils
for threshold in [0.90, 0.95, 0.99]:
    n = np.argmax(cumulative_var >= threshold) + 1
    print(f"  {threshold*100:.0f}% de variance → {n} composantes")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(cumulative_var, linewidth=2, color="#2196F3")
ax.axhline(y=0.90, color="#FF5722", linestyle="--", alpha=0.7, label="90%")
ax.axhline(y=0.95, color="#4CAF50", linestyle="--", alpha=0.7, label="95%")
ax.axhline(y=0.99, color="#9C27B0", linestyle="--", alpha=0.7, label="99%")
ax.set_xlabel("Nombre de composantes")
ax.set_ylabel("Variance expliquée cumulée")
ax.set_title("PCA — Variance expliquée vs. Nombre de composantes")
ax.legend()
ax.set_xlim(0, 344)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "pca_variance_explained.png", dpi=150)
print(f"  → Figure sauvegardée : {FIGURES_DIR / 'pca_variance_explained.png'}")
plt.close(fig)


# ============================================================
# 6. RECHERCHE DU MEILLEUR N_COMPONENTS PAR CV
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 4 : Recherche du meilleur n_components par CV")
print("=" * 60)


def evaluate_pca_cv(n_components, X_scaled, y, patients, patients_uniq, y_uniq,
                    C=0.01, n_repeats=3):
    """Évalue un nombre de composantes PCA avec LogisticRegression L2."""
    aucs = []
    for k in range(n_repeats):
        kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=k)
        for train_idx_, val_idx_ in kfold.split(patients_uniq, y_uniq):
            train_mask = np.isin(patients, patients_uniq[train_idx_])
            val_mask = np.isin(patients, patients_uniq[val_idx_])

            X_fold_train = X_scaled[train_mask]
            y_fold_train = y[train_mask]
            X_fold_val = X_scaled[val_mask]
            y_fold_val = y[val_mask]

            # PCA fit sur le fold train uniquement
            pca = PCA(n_components=n_components, random_state=42)
            X_pca_train = pca.fit_transform(X_fold_train)
            X_pca_val = pca.transform(X_fold_val)

            # Logistic Ridge
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                     random_state=42)
            clf.fit(X_pca_train, y_fold_train)
            preds = clf.predict_proba(X_pca_val)[:, 1]
            aucs.append(roc_auc_score(y_fold_val, preds))

    return np.mean(aucs), np.std(aucs)


pca_results = {}
for nc in PCA_CANDIDATES:
    mean_auc, std_auc = evaluate_pca_cv(
        nc, X_train_scaled, y_train, patients_train,
        patients_unique, y_unique, C=0.01, n_repeats=3
    )
    pca_results[nc] = (mean_auc, std_auc)
    print(f"  n_components={nc:3d}  →  AUC = {mean_auc:.4f} ± {std_auc:.4f}")

best_nc = max(pca_results, key=lambda k: pca_results[k][0])
print(f"\n  ★ Meilleur n_components = {best_nc} "
      f"(AUC = {pca_results[best_nc][0]:.4f})")


# ============================================================
# 7. COMPARAISON DE MODÈLES RÉGULARISÉS
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 5 : Comparaison des modèles régularisés")
print("=" * 60)


def evaluate_model(model_name, model_factory, X_scaled, y, patients,
                   patients_uniq, y_uniq, n_components, n_repeats=5):
    """Évalue un modèle avec PCA + CV patient-level."""
    aucs = []
    for k in range(n_repeats):
        kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=k)
        for train_idx_, val_idx_ in kfold.split(patients_uniq, y_uniq):
            train_mask = np.isin(patients, patients_uniq[train_idx_])
            val_mask = np.isin(patients, patients_uniq[val_idx_])

            X_fold_train = X_scaled[train_mask]
            y_fold_train = y[train_mask]
            X_fold_val = X_scaled[val_mask]
            y_fold_val = y[val_mask]

            # PCA
            pca = PCA(n_components=n_components, random_state=42)
            X_pca_train = pca.fit_transform(X_fold_train)
            X_pca_val = pca.transform(X_fold_val)

            # Model
            clf = model_factory()
            clf.fit(X_pca_train, y_fold_train)

            # Predictions — handle RidgeClassifier (no predict_proba)
            if hasattr(clf, "predict_proba"):
                preds = clf.predict_proba(X_pca_val)[:, 1]
            else:
                preds = clf.decision_function(X_pca_val)

            aucs.append(roc_auc_score(y_fold_val, preds))

    return np.mean(aucs), np.std(aucs)


# Définition des modèles à comparer
models_to_compare = {}

# Pour chaque valeur de C, tester Ridge Logistic, Lasso Logistic, ElasticNet
for C in C_VALUES:
    models_to_compare[f"LogReg_L2_C={C}"] = lambda c=C: LogisticRegression(
        C=c, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42
    )
    models_to_compare[f"LogReg_L1_C={C}"] = lambda c=C: LogisticRegression(
        C=c, penalty="l1", solver="liblinear", max_iter=2000, random_state=42
    )
    alpha = 1.0 / (2.0 * C)
    models_to_compare[f"Ridge_alpha={alpha:.3f}"] = lambda a=alpha: RidgeClassifier(
        alpha=a, random_state=42
    )

# ElasticNet via SGDClassifier
for C in [0.001, 0.01, 0.1, 1.0]:
    alpha_sgd = 1.0 / C
    models_to_compare[f"ElasticNet_alpha={alpha_sgd:.1f}"] = lambda a=alpha_sgd: SGDClassifier(
        loss="log_loss", penalty="elasticnet", alpha=a, l1_ratio=0.5,
        max_iter=2000, random_state=42
    )

# Évaluation
all_results = {}
best_auc = 0
best_model_name = ""

print(f"  Évaluation de {len(models_to_compare)} configurations "
      f"avec n_components={best_nc}...\n")

for name, factory in models_to_compare.items():
    mean_auc, std_auc = evaluate_model(
        name, factory, X_train_scaled, y_train, patients_train,
        patients_unique, y_unique, best_nc, n_repeats=N_REPEATS
    )
    all_results[name] = (mean_auc, std_auc)
    marker = " ★" if mean_auc > best_auc else ""
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model_name = name
    print(f"    {name:35s}  AUC = {mean_auc:.4f} ± {std_auc:.4f}{marker}")

print(f"\n  ★★★ MEILLEUR MODÈLE : {best_model_name} "
      f"(AUC = {best_auc:.4f}) ★★★")


# ============================================================
# 8. PLOT: COMPARAISON DES MODÈLES
# ============================================================
# Top 15 modèles pour le plot
sorted_results = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)[:15]
names = [r[0] for r in sorted_results]
means = [r[1][0] for r in sorted_results]
stds = [r[1][1] for r in sorted_results]

fig, ax = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(names))
bars = ax.barh(y_pos, means, xerr=stds, align="center", color="#2196F3", alpha=0.8,
               edgecolor="#1565C0")
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("AUC ROC (CV)")
ax.set_title(f"Top 15 modèles — PCA({best_nc}) + Régularisation")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
print(f"\n  → Figure sauvegardée : {FIGURES_DIR / 'model_comparison.png'}")
plt.close(fig)


# ============================================================
# 9. ENTRAÎNEMENT FINAL + ENSEMBLING + SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 6 : Entraînement final — Ensemble 5-fold × 5 repeats")
print("=" * 60)

# Recréer la factory du meilleur modèle
best_factory = models_to_compare[best_model_name]

final_models = []
final_pcas = []
final_aucs = []

for k in range(N_REPEATS):
    kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=k)
    fold = 0
    for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):
        train_mask = np.isin(patients_train, patients_unique[train_idx_])
        val_mask = np.isin(patients_train, patients_unique[val_idx_])

        X_fold_train = X_train_scaled[train_mask]
        y_fold_train = y_train[train_mask]
        X_fold_val = X_train_scaled[val_mask]
        y_fold_val = y_train[val_mask]

        # PCA
        pca = PCA(n_components=best_nc, random_state=42)
        X_pca_train = pca.fit_transform(X_fold_train)
        X_pca_val = pca.transform(X_fold_val)

        # Model
        clf = best_factory()
        clf.fit(X_pca_train, y_fold_train)

        # Score
        if hasattr(clf, "predict_proba"):
            preds_val = clf.predict_proba(X_pca_val)[:, 1]
        else:
            preds_val = clf.decision_function(X_pca_val)

        auc = roc_auc_score(y_fold_val, preds_val)
        final_aucs.append(auc)
        final_models.append(clf)
        final_pcas.append(pca)

        print(f"  Repeat {k} Fold {fold}: AUC = {auc:.4f}")
        fold += 1
    print(f"  ---")

print(f"\n  Moyenne AUC (5×5 folds) = {np.mean(final_aucs):.4f} "
      f"± {np.std(final_aucs):.4f}")


# ============================================================
# 10. INFERENCE SUR LE TEST + SOUMISSION
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 7 : Inférence sur le test set")
print("=" * 60)

preds_test = np.zeros(len(X_test))
for pca, clf in zip(final_pcas, final_models):
    X_test_pca = pca.transform(X_test_scaled)
    if hasattr(clf, "predict_proba"):
        preds_test += clf.predict_proba(X_test_pca)[:, 1]
    else:
        # Normaliser decision_function dans [0, 1] pour moyenner
        df_scores = clf.decision_function(X_test_pca)
        # Sigmoid
        preds_test += 1.0 / (1.0 + np.exp(-df_scores))

preds_test /= len(final_models)

# Submission
submission = pd.DataFrame({
    "Sample ID": df_test["Sample ID"].values,
    "Target": preds_test
}).sort_values("Sample ID")

# Sanity checks
assert all(submission["Target"].between(0, 1)), "`Target` values must be in [0, 1]"
assert submission.shape == (149, 2), "Submission must be (149, 2)"
assert list(submission.columns) == ["Sample ID", "Target"]

output_path = OUTPUT_DIR / "pca_ridge_submission.csv"
submission.to_csv(output_path, index=None)

print(f"  ✓ Soumission sauvegardée : {output_path}")
print(f"  ✓ Shape : {submission.shape}")
print(f"  ✓ Target mean : {submission['Target'].mean():.4f}")
print(f"  ✓ Target std  : {submission['Target'].std():.4f}")
print(f"\n  Preview :")
print(submission.head(10).to_string(index=False))


# ============================================================
# RÉSUMÉ
# ============================================================
print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)
print(f"  Features brutes          : {X_train.shape[1]}")
print(f"  Composantes PCA          : {best_nc}")
print(f"  Meilleur modèle          : {best_model_name}")
print(f"  AUC CV (5×5)             : {np.mean(final_aucs):.4f} ± {np.std(final_aucs):.4f}")
print(f"  Fichier de soumission    : {output_path}")
print("=" * 60)
