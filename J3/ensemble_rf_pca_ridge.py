"""
OWKIN ML2 — RF + PCA-Ridge Stacking Ensemble
=============================================
Combine Random Forest (non-linear, on Mean features) with
PCA-Ridge (linear, on Mean+Std+Max → PCA space) via stacking.

Strategy:
  1. Extract Mean / Std / Max features from MoCo .npy files
  2. Model A: Random Forest + Optuna on Mean (2048 dims)
  3. Model B: PCA(150) + LogReg L2 C=0.01 on Mean+Std+Max (6144 dims)
  4. Stacking: OOF predictions → LogReg meta-learner
  5. Also output simple avg and weighted avg as baselines
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
TRAIN_FEATURES_DIR = DATA_DIR / "train_input" / "moco_features"
TEST_FEATURES_DIR = DATA_DIR / "test_input" / "moco_features"
OUTPUT_DIR = DATA_DIR

N_SPLITS = 5
N_REPEATS = 3           # repeats for final ensemble
OPTUNA_TRIALS = 20       # tuning budget for RF
PCA_COMPONENTS = 150
RIDGE_C = 0.01


# ============================================================
# 2. DATA LOADING
# ============================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df_train = pd.read_csv(DATA_DIR / "supplementary_data" / "train_metadata.csv")
df_test = pd.read_csv(DATA_DIR / "supplementary_data" / "test_metadata.csv")
y_train_df = pd.read_csv(DATA_DIR / "train_output.csv")
df_train = df_train.merge(y_train_df, on="Sample ID")

print(f"  Train samples: {len(df_train)}")
print(f"  Test samples : {len(df_test)}")


# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Feature extraction (Mean + Std + Max)")
print("=" * 60)


def extract_features(sample, features_dir):
    _features = np.load(features_dir / sample)
    features = _features[:, 3:]
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    max_feat = np.max(features, axis=0)
    return mean_feat, std_feat, max_feat


print("  Extracting train features...")
train_results = Parallel(n_jobs=-1)(
    delayed(extract_features)(s, TRAIN_FEATURES_DIR)
    for s in tqdm(df_train["Sample ID"].values, desc="  Train")
)
y_train = df_train["Target"].values.astype(int)
patients_train = df_train["Patient ID"].values

mean_train = np.array([r[0] for r in train_results])
std_train = np.array([r[1] for r in train_results])
max_train = np.array([r[2] for r in train_results])

print("  Extracting test features...")
test_results = Parallel(n_jobs=-1)(
    delayed(extract_features)(s, TEST_FEATURES_DIR)
    for s in tqdm(df_test["Sample ID"].values, desc="  Test")
)
mean_test = np.array([r[0] for r in test_results])
std_test = np.array([r[1] for r in test_results])
max_test = np.array([r[2] for r in test_results])

# Feature matrices
X_mean_train = mean_train                                          # (344, 2048)
X_mean_test = mean_test
X_full_train = np.hstack([mean_train, std_train, max_train])       # (344, 6144)
X_full_test = np.hstack([mean_test, std_test, max_test])

print(f"  X_mean: {X_mean_train.shape}, X_full: {X_full_train.shape}")

# Patient-level setup
patients_unique = np.unique(patients_train)
y_unique = np.array(
    [int(np.mean(y_train[patients_train == p])) for p in patients_unique]
)
print(f"  Unique patients: {len(patients_unique)}")


# ============================================================
# 4. MODEL A — RANDOM FOREST + OPTUNA
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Random Forest — Optuna hyperparameter tuning")
print("=" * 60)


def rf_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.1, 0.2, 0.3])

    aucs = []
    kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=42)
    for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):
        train_mask = np.isin(patients_train, patients_unique[train_idx_])
        val_mask = np.isin(patients_train, patients_unique[val_idx_])

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_mean_train[train_mask], y_train[train_mask])
        preds = rf.predict_proba(X_mean_train[val_mask])[:, 1]
        aucs.append(roc_auc_score(y_train[val_mask], preds))

    return np.mean(aucs)


study = optuna.create_study(direction="maximize")
study.optimize(rf_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

best_rf_params = study.best_trial.params
print(f"  Best RF AUC (tuning): {study.best_trial.value:.4f}")
print(f"  Best params: {best_rf_params}")


# ============================================================
# 5. GENERATE OUT-OF-FOLD PREDICTIONS (STACKING)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Generating out-of-fold predictions for stacking")
print("=" * 60)

# We use a SINGLE fixed split for OOF to get aligned predictions
oof_rf = np.zeros(len(y_train))
oof_ridge = np.zeros(len(y_train))
oof_counts = np.zeros(len(y_train))

test_preds_rf = np.zeros(len(X_mean_test))
test_preds_ridge = np.zeros(len(X_full_test))
n_models = 0

for repeat in range(N_REPEATS):
    kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=repeat)
    fold = 0

    for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):
        train_mask = np.isin(patients_train, patients_unique[train_idx_])
        val_mask = np.isin(patients_train, patients_unique[val_idx_])

        # ------- Model A: Random Forest on Mean features -------
        rf = RandomForestClassifier(
            **best_rf_params,
            random_state=42 + repeat * 100 + fold,
            n_jobs=-1,
        )
        rf.fit(X_mean_train[train_mask], y_train[train_mask])
        rf_val_preds = rf.predict_proba(X_mean_train[val_mask])[:, 1]
        oof_rf[val_mask] += rf_val_preds
        test_preds_rf += rf.predict_proba(X_mean_test)[:, 1]

        # ------- Model B: PCA-Ridge on Full features -------
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_full_train[train_mask])
        X_val_s = scaler.transform(X_full_train[val_mask])
        X_te_s = scaler.transform(X_full_test)

        nc = min(PCA_COMPONENTS, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
        pca = PCA(n_components=nc, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_s)
        X_val_pca = pca.transform(X_val_s)
        X_te_pca = pca.transform(X_te_s)

        lr = LogisticRegression(C=RIDGE_C, penalty="l2", solver="lbfgs",
                                max_iter=2000, random_state=42)
        lr.fit(X_tr_pca, y_train[train_mask])
        lr_val_preds = lr.predict_proba(X_val_pca)[:, 1]
        oof_ridge[val_mask] += lr_val_preds
        test_preds_ridge += lr.predict_proba(X_te_pca)[:, 1]

        oof_counts[val_mask] += 1
        n_models += 1

        print(f"  Repeat {repeat} Fold {fold}: "
              f"RF AUC={roc_auc_score(y_train[val_mask], rf_val_preds):.4f}  "
              f"Ridge AUC={roc_auc_score(y_train[val_mask], lr_val_preds):.4f}")
        fold += 1

# Average the OOF predictions (each sample seen N_REPEATS times)
oof_rf /= oof_counts
oof_ridge /= oof_counts
test_preds_rf /= n_models
test_preds_ridge /= n_models

# Individual model CV scores
auc_rf = roc_auc_score(y_train, oof_rf)
auc_ridge = roc_auc_score(y_train, oof_ridge)
print(f"\n  RF OOF AUC:    {auc_rf:.4f}")
print(f"  Ridge OOF AUC: {auc_ridge:.4f}")


# ============================================================
# 6. META-LEARNER (STACKING)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training meta-learner (stacking)")
print("=" * 60)

# Stack the OOF predictions as meta-features
meta_train = np.column_stack([oof_rf, oof_ridge])
meta_test = np.column_stack([test_preds_rf, test_preds_ridge])

# CV the meta-learner to estimate stacked performance
meta_aucs = []
meta_test_preds = np.zeros(len(meta_test))
meta_n = 0

for repeat in range(5):
    kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=repeat + 100)
    for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):
        train_mask = np.isin(patients_train, patients_unique[train_idx_])
        val_mask = np.isin(patients_train, patients_unique[val_idx_])

        meta_clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                       max_iter=1000, random_state=42)
        meta_clf.fit(meta_train[train_mask], y_train[train_mask])

        meta_val_preds = meta_clf.predict_proba(meta_train[val_mask])[:, 1]
        meta_aucs.append(roc_auc_score(y_train[val_mask], meta_val_preds))

# Train final meta-learner on all OOF data
meta_clf_final = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                     max_iter=1000, random_state=42)
meta_clf_final.fit(meta_train, y_train)
preds_stacking = meta_clf_final.predict_proba(meta_test)[:, 1]

print(f"  Meta-learner coefficients: {meta_clf_final.coef_[0]}")
print(f"  Meta-learner intercept:    {meta_clf_final.intercept_[0]:.4f}")
print(f"  Stacking CV AUC:           {np.mean(meta_aucs):.4f} ± {np.std(meta_aucs):.4f}")


# ============================================================
# 7. SIMPLE & WEIGHTED AVERAGES
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Simple & weighted averages")
print("=" * 60)

# Simple average
preds_avg = 0.5 * test_preds_rf + 0.5 * test_preds_ridge
auc_avg_oof = roc_auc_score(y_train, 0.5 * oof_rf + 0.5 * oof_ridge)
print(f"  Simple avg OOF AUC: {auc_avg_oof:.4f}")

# Weighted average (weight by individual OOF AUC)
w_rf = auc_rf / (auc_rf + auc_ridge)
w_ridge = auc_ridge / (auc_rf + auc_ridge)
preds_weighted = w_rf * test_preds_rf + w_ridge * test_preds_ridge
auc_weighted_oof = roc_auc_score(y_train, w_rf * oof_rf + w_ridge * oof_ridge)
print(f"  Weighted avg OOF AUC: {auc_weighted_oof:.4f}")
print(f"    RF weight={w_rf:.3f}, Ridge weight={w_ridge:.3f}")

# Sweep blending weights for best alpha
print("\n  Blending weight sweep (α × RF + (1−α) × Ridge):")
best_alpha, best_blend_auc = 0, 0
for alpha in np.arange(0.0, 1.05, 0.05):
    blend_oof = alpha * oof_rf + (1 - alpha) * oof_ridge
    blend_auc = roc_auc_score(y_train, blend_oof)
    if blend_auc > best_blend_auc:
        best_alpha = alpha
        best_blend_auc = blend_auc
    if alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"    α={alpha:.2f}  →  OOF AUC = {blend_auc:.4f}")

print(f"\n  ★ Best blend: α={best_alpha:.2f}, OOF AUC={best_blend_auc:.4f}")
preds_best_blend = best_alpha * test_preds_rf + (1 - best_alpha) * test_preds_ridge


# ============================================================
# 8. SAVE SUBMISSIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Saving submissions")
print("=" * 60)

submissions = {
    "ensemble_stacking.csv": preds_stacking,
    "ensemble_avg.csv": preds_avg,
    "ensemble_weighted.csv": preds_weighted,
    "ensemble_best_blend.csv": preds_best_blend,
}

for fname, preds in submissions.items():
    sub = pd.DataFrame({
        "Sample ID": df_test["Sample ID"].values,
        "Target": preds,
    }).sort_values("Sample ID")

    assert all(sub["Target"].between(0, 1)), f"{fname}: Target not in [0,1]"
    assert sub.shape == (149, 2), f"{fname}: Wrong shape {sub.shape}"

    path = OUTPUT_DIR / fname
    sub.to_csv(path, index=None)
    print(f"  ✓ {fname}: mean={preds.mean():.4f}, std={preds.std():.4f} → {path}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Model A (RF):           OOF AUC = {auc_rf:.4f}")
print(f"  Model B (PCA-Ridge):    OOF AUC = {auc_ridge:.4f}")
print(f"  Simple average:         OOF AUC = {auc_avg_oof:.4f}")
print(f"  Weighted average:       OOF AUC = {auc_weighted_oof:.4f}")
print(f"  Best blend (α={best_alpha:.2f}):   OOF AUC = {best_blend_auc:.4f}")
print(f"  Stacking meta-learner:  CV AUC  = {np.mean(meta_aucs):.4f}")
print(f"\n  Baseline RF:            Test AUC = 0.6554  (to beat)")
print("=" * 60)
print("✅ Done! Submit the best .csv to the challenge platform.")
