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

# 1. Update Imports & Install XGBoost/Joblib
header_code = [
    "!pip install xgboost joblib",
    "",
    "from pathlib import Path",
    "from tqdm import tqdm",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "from sklearn.metrics import roc_auc_score",
    "from sklearn.model_selection import StratifiedKFold",
    "import xgboost as xgb",
    "from xgboost import XGBClassifier",
    "import optuna",
    "from joblib import Parallel, delayed"
]

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "from sklearn.linear_model import LogisticRegression" in source or "import pandas as pd" in source:
             cell["source"] = [line + "\n" for line in header_code]
             print("Updated imports cell.")
             break

# 2. Parallel Feature Extraction (Mean + Std + Max)
feature_extraction_code = [
    "def process_sample(sample_info, features_dir):",
    "    sample, label, center, patient = sample_info",
    "    _features = np.load(features_dir / sample)",
    "    coordinates, features = _features[:, :3], _features[:, 3:]",
    "    ",
    "    # Enhanced Pooling: Mean + Std + Max",
    "    mean_feat = np.mean(features, axis=0)",
    "    std_feat = np.std(features, axis=0)",
    "    max_feat = np.max(features, axis=0)",
    "    ",
    "    concatenated_features = np.concatenate([mean_feat, std_feat, max_feat])",
    "    return concatenated_features, label, center, patient",
    "",
    "print(\"Starting parallel feature extraction (Train)...\")",
    "# Parallel processing",
    "results = Parallel(n_jobs=-1)(",
    "    delayed(process_sample)(row, train_features_dir) ",
    "    for row in tqdm(df_train[[\"Sample ID\", \"Target\", \"Center ID\", \"Patient ID\"]].values)",
    ")",
    "",
    "X_train = np.array([r[0] for r in results])",
    "y_train = np.array([r[1] for r in results])",
    "centers_train = np.array([r[2] for r in results])",
    "patients_train = np.array([r[3] for r in results])",
    "",
    "print(f\"X_train shape: {X_train.shape}\")"
]

# Replace the training data loading cell
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "X_train.append(np.mean(features, axis=0))" in source:
            cell["source"] = [line + "\n" for line in feature_extraction_code]
            print("Updated TRAINING feature extraction loop with Joblib & Enhanced Features.")
            break

# 3. XGBoost Optimization Logic
xgb_optuna_code = [
    "def objective(trial):",
    "    params = {",
    "        'n_estimators': trial.suggest_int('n_estimators', 50, 300),",
    "        'max_depth': trial.suggest_int('max_depth', 3, 8),",
    "        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),",
    "        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),",
    "        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 0.5), # Small colsample for high dim",
    "        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),",
    "        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),",
    "        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),",
    "        'use_label_encoder': False,",
    "        'eval_metric': 'auc',",
    "        'random_state': 42,",
    "        'n_jobs': -1,",
    "        'verbosity': 0",
    "    }",
    "    ",
    "    aucs = []",
    "    # 5-fold CV",
    "    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "    ",
    "    for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):",
    "        # Map patient split to sample split",
    "        train_idx = np.arange(len(X_train))[",
    "            pd.Series(patients_train).isin(patients_unique[train_idx_])",
    "        ]",
    "        val_idx = np.arange(len(X_train))[",
    "            pd.Series(patients_train).isin(patients_unique[val_idx_])",
    "        ]",
    "        ",
    "        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]",
    "        X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]",
    "        ",
    "        model = XGBClassifier(**params)",
    "        model.fit(X_fold_train, y_fold_train)",
    "        preds_val = model.predict_proba(X_fold_val)[:, 1]",
    "        auc = roc_auc_score(y_fold_val, preds_val)",
    "        aucs.append(auc)",
    "        ",
    "    return np.mean(aucs)",
    "",
    "print(\"Starting Optuna optimization...\")",
    "study = optuna.create_study(direction='maximize')",
    "study.optimize(objective, n_trials=20) # 20 trials to save time, increase if needed",
    "",
    "print(f\"Best params: {study.best_params}\")",
    "",
    "# Retrain with best parameters",
    "best_params = study.best_params",
    "best_params['use_label_encoder'] = False",
    "best_params['eval_metric'] = 'auc'",
    "best_params['random_state'] = 42",
    "best_params['n_jobs'] = -1",
    "",
    "lrs = [] # Reuse variable name for compatibility",
    "aucs = []",
    "",
    "print(\"Retraining ensemble with best params...\")",
    "kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "",
    "for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):",
    "    train_idx = np.arange(len(X_train))[",
    "        pd.Series(patients_train).isin(patients_unique[train_idx_])",
    "    ]",
    "    val_idx = np.arange(len(X_train))[",
    "        pd.Series(patients_train).isin(patients_unique[val_idx_])",
    "    ]",
    "    ",
    "    X_fold_train = X_train[train_idx]",
    "    y_fold_train = y_train[train_idx]",
    "    X_fold_val = X_train[val_idx]",
    "    y_fold_val = y_train[val_idx]",
    "    ",
    "    model = XGBClassifier(**best_params)",
    "    model.fit(X_fold_train, y_fold_train)",
    "    ",
    "    preds_val = model.predict_proba(X_fold_val)[:, 1]",
    "    auc = roc_auc_score(y_fold_val, preds_val)",
    "    aucs.append(auc)",
    "    lrs.append(model)",
    "",
    "print(f\"Mean AUC with Best XGBoost: {np.mean(aucs):.3f}\")"
]

# Replace the training/optimization cell
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "LogisticRegression(C=0.01" in source or "def objective(trial):" in source:
             cell["source"] = [line + "\n" for line in xgb_optuna_code]
             print("Updated Model Optimization loop with XGBoost.")
             break

# 4. Test Data Processing (Parallel)
test_processing_code = [
    "def process_test_sample(sample, features_dir):",
    "    _features = np.load(features_dir / sample)",
    "    coordinates, features = _features[:, :3], _features[:, 3:]",
    "    ",
    "    mean_feat = np.mean(features, axis=0)",
    "    std_feat = np.std(features, axis=0)",
    "    max_feat = np.max(features, axis=0)",
    "    ",
    "    return np.concatenate([mean_feat, std_feat, max_feat])",
    "",
    "print(\"Starting parallel feature extraction (Test)...\")",
    "X_test_list = Parallel(n_jobs=-1)(",
    "    delayed(process_test_sample)(sample, test_features_dir)",
    "    for sample in tqdm(df_test[\"Sample ID\"].values)",
    ")",
    "X_test = np.array(X_test_list)",
    "print(f\"X_test shape: {X_test.shape}\")"
]

# Replace test processing cell
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "X_test.append(np.mean(features, axis=0))" in source:
            cell["source"] = [line + "\n" for line in test_processing_code]
            print("Updated TEST feature extraction loop with Joblib.")
            break

# 5. Fix Submission Filename
submission_code = [
    "submission = pd.DataFrame(",
    "    {\"Sample ID\": df_test[\"Sample ID\"].values, \"Target\": preds_test}",
    ").sort_values(",
    "    \"Sample ID\"",
    ")",
    "",
    "# save the submission as a csv file",
    "submission.to_csv(data_dir / \"xgb_enhanced_submission.csv\", index=None)",
    "submission.head()"
]

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "submission.to_csv" in source:
            cell["source"] = [line + "\n" for line in submission_code]
            print("Updated submission filename.")
            break


output_path = "baseline_xgb_enhanced.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Successfully created {output_path}")
