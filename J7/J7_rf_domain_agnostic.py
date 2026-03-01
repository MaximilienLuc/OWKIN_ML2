import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from scipy import stats as sp_stats
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1. PARAMETRES
# ============================================================
DATA_DIR = "Data"
TRAIN_FEATURES_DIR = os.path.join(DATA_DIR, "train_input", "moco_features")
TEST_FEATURES_DIR = os.path.join(DATA_DIR, "test_input", "moco_features")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

OPTUNA_TRIALS = 25
TOP_K_DRIFT = 2000 # On bannit les 2000 features les plus corrélées au Domaine (Train vs Test)

# ============================================================
# 2. CHARGEMENT DES METADONNEES
# ============================================================
print("=" * 60)
print("ÉTAPE 1 : Chargement des métadonnées (Domain Shift Awareness)")
print("=" * 60)

df_train = pd.read_csv(os.path.join(DATA_DIR, "supplementary_data", "train_metadata.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "supplementary_data", "test_metadata.csv"))
y_train_df = pd.read_csv(os.path.join(DATA_DIR, "train_output.csv"))

df_train = df_train.merge(y_train_df, on="Sample ID")

print(f"Centres d'entraînement : {df_train['Center ID'].unique()}")
print(f"Centres de test : {df_test['Center ID'].unique()}")

# ============================================================
# 3. EXTRACTION ROBUSTE DES FEATURES (18432 dimensions)
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 2 : Extraction des Statistiques Étendues (Caching en cours...)")
print("=" * 60)

def extract_extended_features(sample, features_dir):
    _features = np.load(os.path.join(features_dir, sample))
    features = _features[:, 3:]  # On retire les coordonnées (x, y, zoom)
    
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

CACHE_TRAIN = os.path.join(DATA_DIR, "cache_train_ext.npy")
CACHE_TEST = os.path.join(DATA_DIR, "cache_test_ext.npy")

if os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST):
    print("  Chargement depuis le cache local...")
    X_train_ext = np.load(CACHE_TRAIN)
    X_test_ext = np.load(CACHE_TEST)
else:
    print("  Extraction en cours (Train)...")
    X_train_ext = np.array(Parallel(n_jobs=-1)(
        delayed(extract_extended_features)(s, TRAIN_FEATURES_DIR)
        for s in tqdm(df_train["Sample ID"].values)
    ))
    
    print("  Extraction en cours (Test)...")
    X_test_ext = np.array(Parallel(n_jobs=-1)(
        delayed(extract_extended_features)(s, TEST_FEATURES_DIR)
        for s in tqdm(df_test["Sample ID"].values)
    ))
    
    np.save(CACHE_TRAIN, X_train_ext)
    np.save(CACHE_TEST, X_test_ext)

y_train = df_train["Target"].values
groups = df_train["Center ID"].values # <--- LA CLÉ DU SUCCÈS EST ICI

print(f"  Shape Train : {X_train_ext.shape}")
print(f"  Shape Test  : {X_test_ext.shape}")

# Preprocessing des NaN/Inf générés par le skew/kurtosis
X_train_ext = np.nan_to_num(X_train_ext, nan=0.0, posinf=0.0, neginf=0.0)
X_test_ext = np.nan_to_num(X_test_ext, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================
# 4. ADVERSARIAL VALIDATION AGRESSIVE (Anti-Domain Shift)
# ============================================================
print("\n" + "=" * 60)
print(f"ÉTAPE 3 : Adversarial Validation (Drop Top {TOP_K_DRIFT} features)")
print("=" * 60)

# Labels: 1 pour Train, 0 pour Test
X_adv = np.vstack([X_train_ext, X_test_ext])
y_adv = np.array([1] * len(X_train_ext) + [0] * len(X_test_ext))

adv_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
adv_rf.fit(X_adv, y_adv)

adv_importances = adv_rf.feature_importances_
drift_ranking = np.argsort(adv_importances)[::-1]
top_drift_idx = drift_ranking[:TOP_K_DRIFT]

# Création des matrices d'entraînement filtrées (Domain-Agnostic)
keep_mask = np.ones(X_train_ext.shape[1], dtype=bool)
keep_mask[top_drift_idx] = False

X_train_filtered = X_train_ext[:, keep_mask]
X_test_filtered = X_test_ext[:, keep_mask]

print(f"  Dimensions après filtrage de l'hôpital : {X_train_filtered.shape[1]} features restantes.")

# ============================================================
# 5. OPTUNA AVEC GROUP K-FOLD CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 4 : Optimisation Optuna avec GroupKFold (par 'Center ID')")
print("=" * 60)

def objective(trial):
    # Paramètres conservateurs pour petits volumes / Random Forest
    n_estimators = trial.suggest_int("n_estimators", 100, 400)
    max_depth = trial.suggest_int("max_depth", 3, 7)
    min_samples_split = trial.suggest_int("min_samples_split", 10, 40)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 20)
    max_features = trial.suggest_float("max_features", 0.05, 0.25)
    
    aucs = []
    # /!\ GROUP K-FOLD : Le set de validation sera toujours un Hôpital que le modèle ne connaît pas
    gkf = GroupKFold(n_splits=3)
    
    for train_idx, val_idx in gkf.split(X_train_filtered, y_train, groups):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_filtered[train_idx], y_train[train_idx])
        preds = rf.predict_proba(X_train_filtered[val_idx])[:, 1]
        
        auc = roc_auc_score(y_train[val_idx], preds)
        aucs.append(auc)
        
    # Le score Optuna est la MOYENNE de l'AUC sur ces hôpitaux inconnus
    return np.mean(aucs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

print(f"  Meilleure CV AUC 'Domain-Agnostic' : {study.best_value:.4f}")
print(f"  Meilleurs paramètres : {study.best_params}")

# ============================================================
# 6. ENTRAÎNEMENT FINAL & INFERENCE
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 5 : Entraînement Final & Prédiction")
print("=" * 60)

best_params = study.best_params

# Pour une inférence ultra robuste, on en fait un bag de plusieurs modèles
print("  Entraînement de l'ensemble Random Forest...")
final_preds = np.zeros(len(X_test_filtered))
NUM_MODELS = 5

for seed in range(NUM_MODELS):
    rf = RandomForestClassifier(
        **best_params,
        random_state=42 + seed,
        n_jobs=-1
    )
    rf.fit(X_train_filtered, y_train)
    final_preds += rf.predict_proba(X_test_filtered)[:, 1]

final_preds /= NUM_MODELS

# Sauvegarde de la soumission
os.makedirs(OUTPUT_DIR, exist_ok=True)
submission = pd.DataFrame({
    "Sample ID": df_test["Sample ID"].values,
    "Target": final_preds
}).sort_values("Sample ID")

output_path = os.path.join(DATA_DIR, "submission_j7_rf_domain_agnostic.csv")
submission.to_csv(output_path, index=False)

print(f"✅ Soumission sauvegardée dans : {output_path}")
print(f"   Ce fichier est 100% calibré pour faire face au nouveau centre de données !")
