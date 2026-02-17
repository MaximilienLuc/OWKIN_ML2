import json
from pathlib import Path

# Use baseline_optuna_optim.ipynb as the base
nb_path = Path("baseline_optuna_optim.ipynb") 
try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
except FileNotFoundError:
    print(f"Error: {nb_path} not found. Trying tile_level.ipynb")
    nb_path = Path("tile_level.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

print(f"Reading notebook: {nb_path}")

# --- 1. PREPARE ALL CODE BLOCKS ---

header_code = [
    "!pip install xgboost joblib",
    "",
    "from pathlib import Path",
    "from tqdm import tqdm",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "from sklearn.metrics import roc_auc_score",
    "from sklearn.model_selection import StratifiedKFold, cross_val_score",
    "import xgboost as xgb",
    "from xgboost import XGBClassifier",
    "import optuna",
    "from joblib import Parallel, delayed"
]

feature_loading_code = [
    "def process_sample(sample_info, features_dir, is_train=True):",
    "    if is_train:",
    "        sample, label, center, patient = sample_info",
    "    else:",
    "        sample = sample_info",
    "    ",
    "    _features = np.load(features_dir / sample)",
    "    coordinates, features = _features[:, :3], _features[:, 3:]",
    "    ",
    "    # Enhanced Pooling: Mean + Std + Max",
    "    mean_feat = np.mean(features, axis=0)",
    "    std_feat = np.std(features, axis=0)",
    "    max_feat = np.max(features, axis=0)",
    "    ",
    "    concatenated_features = np.concatenate([mean_feat, std_feat, max_feat])",
    "    if is_train:",
    "        return concatenated_features, label, center, patient",
    "    return concatenated_features",
    "",
    "print(\"Starting parallel feature extraction (Train)...\")",
    "train_results = Parallel(n_jobs=-1)(",
    "    delayed(process_sample)(row, train_features_dir, True) ",
    "    for row in tqdm(df_train[[\"Sample ID\", \"Target\", \"Center ID\", \"Patient ID\"]].values)",
    ")",
    "X_train = np.array([r[0] for r in train_results])",
    "y_train = np.array([r[1] for r in train_results])",
    "centers_train = np.array([r[2] for r in train_results])",
    "patients_train = np.array([r[3] for r in train_results])",
    "",
    "print(\"Starting parallel feature extraction (Test)...\")",
    "X_test_list = Parallel(n_jobs=-1)(",
    "    delayed(process_sample)(sample, test_features_dir, False)",
    "    for sample in tqdm(df_test[\"Sample ID\"].values)",
    ")",
    "X_test = np.array(X_test_list)",
    "print(f\"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}\")"
]

drift_removal_code = [
    "# --- ADVERSARIAL VALIDATION TO REMOVE DRIFT ---",
    "print(\"Running Adversarial Validation to detect Covariate Shift...\")",
    "",
    "y_av_train = np.zeros(len(X_train))",
    "y_av_test = np.ones(len(X_test))",
    "X_av = np.vstack([X_train, X_test])",
    "y_av = np.concatenate([y_av_train, y_av_test])",
    "",
    "model_av = XGBClassifier(",
    "    n_estimators=50, max_depth=4, learning_rate=0.1,",
    "    eval_metric='auc', use_label_encoder=False, n_jobs=-1, random_state=42",
    ")",
    "model_av.fit(X_av, y_av)",
    "",
    "# Identify top 20 drifting features",
    "importance = model_av.feature_importances_",
    "drifting_indices = np.argsort(importance)[::-1][:20]",
    "print(f\"Removing Top 20 Drifting Features: {drifting_indices}\")",
    "",
    "# Remove them from datasets",
    "X_train = np.delete(X_train, drifting_indices, axis=1)",
    "X_test = np.delete(X_test, drifting_indices, axis=1)",
    "print(f\"New X_train shape: {X_train.shape}\")",
    "print(f\"New X_test shape: {X_test.shape}\")",
    "# ------------------------------------------------"
]

xgb_optuna_code = [
    "def objective(trial):",
    "    params = {",
    "        'n_estimators': trial.suggest_int('n_estimators', 50, 300),",
    "        'max_depth': trial.suggest_int('max_depth', 3, 8),",
    "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),",
    "        'subsample': trial.suggest_float('subsample', 0.6, 1.0),",
    "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),",
    "        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),",
    "        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),",
    "        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),",
    "        'use_label_encoder': False,",
    "        'eval_metric': 'auc',",
    "        'random_state': 42,",
    "        'n_jobs': -1,",
    "        'verbosity': 0",
    "    }",
    "    ",
    "    aucs = []",
    "    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "    for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):",
    "        train_idx = np.arange(len(X_train))[pd.Series(patients_train).isin(patients_unique[train_idx_])]",
    "        val_idx = np.arange(len(X_train))[pd.Series(patients_train).isin(patients_unique[val_idx_])]",
    "        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]",
    "        X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]",
    "        model = XGBClassifier(**params)",
    "        model.fit(X_fold_train, y_fold_train)",
    "        preds_val = model.predict_proba(X_fold_val)[:, 1]",
    "        auc = roc_auc_score(y_fold_val, preds_val)",
    "        aucs.append(auc)",
    "    return np.mean(aucs)",
    "",
    "print(\"Starting Optuna optimization...\")",
    "study = optuna.create_study(direction='maximize')",
    "study.optimize(objective, n_trials=30)",
    "print(f\"Best params: {study.best_params}\")",
    "",
    "best_params = study.best_params",
    "best_params['use_label_encoder'] = False",
    "best_params['eval_metric'] = 'auc'",
    "best_params['random_state'] = 42",
    "best_params['n_jobs'] = -1",
    "",
    "lrs = []",
    "aucs = []",
    "print(\"Retraining ensemble with best params...\")",
    "kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):",
    "    train_idx = np.arange(len(X_train))[pd.Series(patients_train).isin(patients_unique[train_idx_])]",
    "    val_idx = np.arange(len(X_train))[pd.Series(patients_train).isin(patients_unique[val_idx_])]",
    "    X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]",
    "    X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]",
    "    model = XGBClassifier(**best_params)",
    "    model.fit(X_fold_train, y_fold_train)",
    "    preds_val = model.predict_proba(X_fold_val)[:, 1]",
    "    auc = roc_auc_score(y_fold_val, preds_val)",
    "    aucs.append(auc)",
    "    lrs.append(model)",
    "print(f\"Mean AUC with Best XGBoost: {np.mean(aucs):.3f}\")"
]

inference_code = [
    "preds_test = 0",
    "for lr in lrs:",
    "    preds_test += lr.predict_proba(X_test)[:, 1]",
    "preds_test = preds_test / len(lrs)"
]

submission_code = [
    "submission = pd.DataFrame(",
    "    {\"Sample ID\": df_test[\"Sample ID\"].values, \"Target\": preds_test}",
    ").sort_values(\"Sample ID\")",
    "submission.to_csv(data_dir / \"xgb_robust_submission.csv\", index=None)",
    "submission.head()"
]

# --- 2. APPLY TRANSFORMATIONS ---

# Delete redundant cells (we will consolidate)
# We want to remove the old test processing and old submission cells to avoid confusion
new_cells = []
for cell in nb["cells"]:
    source = "".join(cell["source"])
    
    # 1. Imports
    if "from sklearn.linear_model import LogisticRegression" in source or "import pandas as pd" in source:
        cell["source"] = [line + "\n" for line in header_code]
        new_cells.append(cell)
        print("Replaced Imports.")
    
    # 2. Replaced Data Processing (consolidated Train + Test + Drift Removal)
    elif "X_train.append(np.mean(features, axis=0))" in source:
        cell["source"] = [line + "\n" for line in feature_loading_code]
        new_cells.append(cell)
        
        # INSERT DRIFT REMOVAL IMMEDIATELY AFTER LOADING
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in drift_removal_code]
        })
        print("Replaced Feature Extraction & Inserted Drift Removal.")

    # 3. Optimization
    elif "LogisticRegression(C=0.01" in source or "def objective(trial):" in source:
        cell["source"] = [line + "\n" for line in xgb_optuna_code]
        new_cells.append(cell)
        print("Replaced Optimization.")

    # 4. Inference (Replace the averaging loop)
    elif "preds_test += lr.predict_proba(X_test)[:, 1]" in source or "lr.predict_proba" in source:
        if "submission" not in source: # Avoid doubling with next check
            cell["source"] = [line + "\n" for line in inference_code]
            new_cells.append(cell)
            print("Replaced Inference.")

    # 5. Submission
    elif "submission.to_csv" in source:
        cell["source"] = [line + "\n" for line in submission_code]
        new_cells.append(cell)
        print("Replaced Submission.")

    # Skip old test processing cells (we moved them up)
    elif "X_test.append(np.mean(features, axis=0))" in source or "process_test_sample" in source:
        print("Skipping redundant Test processing cell.")
        continue
    
    else:
        new_cells.append(cell)

nb["cells"] = new_cells

output_path = "baseline_xgb_robust.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Successfully fixed {output_path}")
