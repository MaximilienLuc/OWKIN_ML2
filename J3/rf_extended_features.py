"""
OWKIN ML2 — Extended Features + Conservative Random Forest
==========================================================
Strategy: Richer slide statistics + RF with anti-overfitting safeguards.

Features (9 × 2048 = 18,432 dims):
  Mean, Std, Max, Min, Q25, Median, Q75, Skew, Kurtosis

Anti-overfitting:
  1. Constrained RF search space (shallow trees, high regularization)
  2. Few Optuna trials (15)
  3. Adversarial validation → remove drifting features
  4. Generate submissions with vs without feature selection
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import optuna
from scipy import stats as sp_stats

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
N_REPEATS = 3
OPTUNA_TRIALS = 15
TOP_K_DRIFT = 50  # number of drifting features to remove


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
# 3. EXTENDED FEATURE EXTRACTION
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Extended feature extraction")
print("=" * 60)


def extract_extended_features(sample, features_dir):
    """Extract 9 statistics per feature dimension."""
    _features = np.load(features_dir / sample)
    features = _features[:, 3:]  # drop coordinates

    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    max_feat = np.max(features, axis=0)
    min_feat = np.min(features, axis=0)
    q25 = np.percentile(features, 25, axis=0)
    median_feat = np.median(features, axis=0)
    q75 = np.percentile(features, 75, axis=0)
    skew_feat = sp_stats.skew(features, axis=0)
    kurtosis_feat = sp_stats.kurtosis(features, axis=0)

    return np.concatenate([
        mean_feat, std_feat, max_feat, min_feat,
        q25, median_feat, q75, skew_feat, kurtosis_feat
    ])


print("  Extracting train features...")
X_train_ext = np.array(Parallel(n_jobs=-1)(
    delayed(extract_extended_features)(s, TRAIN_FEATURES_DIR)
    for s in tqdm(df_train["Sample ID"].values, desc="  Train")
))
y_train = df_train["Target"].values.astype(int)
patients_train = df_train["Patient ID"].values

print("  Extracting test features...")
X_test_ext = np.array(Parallel(n_jobs=-1)(
    delayed(extract_extended_features)(s, TEST_FEATURES_DIR)
    for s in tqdm(df_test["Sample ID"].values, desc="  Test")
))

# Also keep mean-only for comparison
DIM = 2048
X_train_mean = X_train_ext[:, :DIM]
X_test_mean = X_test_ext[:, :DIM]

print(f"  X_train extended: {X_train_ext.shape}")  # (344, 18432)
print(f"  X_train mean:     {X_train_mean.shape}")  # (344, 2048)

# Feature names for interpretability
stat_names = ["mean", "std", "max", "min", "q25", "median", "q75", "skew", "kurtosis"]
feature_names = [f"{stat}_{i}" for stat in stat_names for i in range(DIM)]

# Patient-level setup
patients_unique = np.unique(patients_train)
y_unique = np.array(
    [int(np.mean(y_train[patients_train == p])) for p in patients_unique]
)
print(f"  Unique patients: {len(patients_unique)}")


# ============================================================
# 4. ADVERSARIAL VALIDATION — DETECT DRIFTING FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Adversarial validation — detecting drifting features")
print("=" * 60)

# Label: 1 = train, 0 = test
X_adv = np.vstack([X_train_ext, X_test_ext])
y_adv = np.array([1] * len(X_train_ext) + [0] * len(X_test_ext))

# Replace NaN/Inf from skew/kurtosis
X_adv = np.nan_to_num(X_adv, nan=0.0, posinf=0.0, neginf=0.0)
X_train_ext = np.nan_to_num(X_train_ext, nan=0.0, posinf=0.0, neginf=0.0)
X_test_ext = np.nan_to_num(X_test_ext, nan=0.0, posinf=0.0, neginf=0.0)

# Train a simple RF to distinguish train from test
adv_rf = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
)
adv_rf.fit(X_adv, y_adv)
adv_importances = adv_rf.feature_importances_

# Identify top drifting features
drift_ranking = np.argsort(adv_importances)[::-1]
top_drift_idx = drift_ranking[:TOP_K_DRIFT]
top_drift_names = [feature_names[i] for i in top_drift_idx[:10]]
print(f"  Top 10 drifting features: {top_drift_names}")
print(f"  Removing top {TOP_K_DRIFT} drifting features...")

# Create filtered feature matrices
keep_mask = np.ones(X_train_ext.shape[1], dtype=bool)
keep_mask[top_drift_idx] = False

X_train_filtered = X_train_ext[:, keep_mask]
X_test_filtered = X_test_ext[:, keep_mask]
print(f"  X_train filtered: {X_train_filtered.shape}")


# ============================================================
# 5. RF + OPTUNA (CONSERVATIVE SEARCH SPACE)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: RF + Optuna (constrained search space)")
print("=" * 60)


def create_objective(X_data, label):
    """Create an Optuna objective for a given feature matrix."""
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 8)           # SHALLOW
        min_samples_split = trial.suggest_int("min_samples_split", 5, 30)  # HIGH
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 25)    # HIGH
        max_features = trial.suggest_float("max_features", 0.05, 0.3)     # LOW

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
            rf.fit(X_data[train_mask], y_train[train_mask])
            preds = rf.predict_proba(X_data[val_mask])[:, 1]
            aucs.append(roc_auc_score(y_train[val_mask], preds))

        return np.mean(aucs)

    return objective


# --- Config A: Mean only (2048d) - the known good baseline ---
print("\n  [A] Tuning RF on Mean features (2048d)...")
study_a = optuna.create_study(direction="maximize")
study_a.optimize(create_objective(X_train_mean, "mean"), n_trials=OPTUNA_TRIALS,
                 show_progress_bar=True)
print(f"  [A] Best CV AUC: {study_a.best_trial.value:.4f}")
print(f"  [A] Best params: {study_a.best_trial.params}")

# --- Config B: Extended filtered (after drift removal) ---
print(f"\n  [B] Tuning RF on Extended filtered features ({X_train_filtered.shape[1]}d)...")
study_b = optuna.create_study(direction="maximize")
study_b.optimize(create_objective(X_train_filtered, "ext_filtered"), n_trials=OPTUNA_TRIALS,
                 show_progress_bar=True)
print(f"  [B] Best CV AUC: {study_b.best_trial.value:.4f}")
print(f"  [B] Best params: {study_b.best_trial.params}")

# --- Config C: Full extended (no drift removal, for comparison) ---
print(f"\n  [C] Tuning RF on Full extended features ({X_train_ext.shape[1]}d)...")
study_c = optuna.create_study(direction="maximize")
study_c.optimize(create_objective(X_train_ext, "ext_full"), n_trials=OPTUNA_TRIALS,
                 show_progress_bar=True)
print(f"  [C] Best CV AUC: {study_c.best_trial.value:.4f}")
print(f"  [C] Best params: {study_c.best_trial.params}")


# ============================================================
# 6. TRAIN FINAL ENSEMBLES & GENERATE PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training final ensembles (5-fold × 3 repeats)")
print("=" * 60)


def train_ensemble(name, X_tr, X_te, best_params):
    """Train a 5-fold × N_REPEATS ensemble and return test predictions + CV AUC."""
    print(f"\n  [{name}] Training ensemble...")
    all_preds = np.zeros(len(X_te))
    all_aucs = []
    n_models = 0

    for repeat in range(N_REPEATS):
        kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=repeat)
        fold = 0
        for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):
            train_mask = np.isin(patients_train, patients_unique[train_idx_])
            val_mask = np.isin(patients_train, patients_unique[val_idx_])

            rf = RandomForestClassifier(
                **best_params,
                random_state=42 + repeat * 100 + fold,
                n_jobs=-1,
            )
            rf.fit(X_tr[train_mask], y_train[train_mask])

            preds_val = rf.predict_proba(X_tr[val_mask])[:, 1]
            auc = roc_auc_score(y_train[val_mask], preds_val)
            all_aucs.append(auc)

            all_preds += rf.predict_proba(X_te)[:, 1]
            n_models += 1
            fold += 1

    all_preds /= n_models
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    print(f"  [{name}] CV AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    return all_preds, mean_auc, std_auc


# A: Mean only
preds_a, auc_a, std_a = train_ensemble(
    "A: Mean RF", X_train_mean, X_test_mean, study_a.best_trial.params
)

# B: Extended filtered
preds_b, auc_b, std_b = train_ensemble(
    "B: Ext Filtered RF", X_train_filtered, X_test_filtered, study_b.best_trial.params
)

# C: Full extended
preds_c, auc_c, std_c = train_ensemble(
    "C: Ext Full RF", X_train_ext, X_test_ext, study_c.best_trial.params
)


# ============================================================
# 7. SAVE SUBMISSIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Saving submissions")
print("=" * 60)

submissions = {
    "rf_mean_conservative.csv": preds_a,
    "rf_ext_filtered.csv": preds_b,
    "rf_ext_full.csv": preds_c,
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
    print(f"  ✓ {fname}: mean={preds.mean():.4f}, std={preds.std():.4f}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  [A] Mean RF (2048d):            CV AUC = {auc_a:.4f} ± {std_a:.4f}")
print(f"      Params: {study_a.best_trial.params}")
print(f"  [B] Ext Filtered RF ({X_train_filtered.shape[1]}d):  CV AUC = {auc_b:.4f} ± {std_b:.4f}")
print(f"      Params: {study_b.best_trial.params}")
print(f"  [C] Ext Full RF ({X_train_ext.shape[1]}d):     CV AUC = {auc_c:.4f} ± {std_c:.4f}")
print(f"      Params: {study_c.best_trial.params}")
print(f"\n  Previous best leaderboard: 0.6554 (RF+Optuna baseline)")
print(f"\n  ⚠️  Remember: high CV AUC ≠ high leaderboard score!")
print(f"  ⚠️  Submit the variant with the LOWEST CV AUC gap risk.")
print("=" * 60)
print("✅ Done!")
