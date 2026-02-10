import nbformat as nbf
from pathlib import Path

nb_path = Path("baseline_max.ipynb")
nb = nbf.read(nb_path, as_version=4)

# 1. Update Imports
for cell in nb.cells:
    if cell.cell_type == "code":
        if "from sklearn.linear_model import LogisticRegression" in cell.source:
            # Add new imports if not already present
            if "RandomForestClassifier" not in cell.source:
                cell.source = cell.source.replace(
                    "from sklearn.linear_model import LogisticRegression",
                    "from sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nimport optuna"
                )
                print("Updated imports.")

# 2. Replace CV Loop with Optuna
optuna_code = """
def objective(trial):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    aucs = []
    # 5-fold CV (single repeat for speed in optimization)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):
        train_idx = np.arange(len(X_train))[
            pd.Series(patients_train).isin(patients_unique[train_idx_])
        ]
        val_idx = np.arange(len(X_train))[
            pd.Series(patients_train).isin(patients_unique[val_idx_])
        ]
        
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        rf.fit(X_fold_train, y_fold_train)
        preds_val = rf.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, preds_val)
        aucs.append(auc)
        
    return np.mean(aucs)

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best trial: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")

# Retrain with best parameters
best_params = study.best_trial.params
lrs = [] # Reuse 'lrs' variable for compatibility with downstream code

print("Retraining with best parameters...")
aucs = []

# We perform 5-fold CV with best params to get 5 models for the ensemble
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):
    train_idx = np.arange(len(X_train))[
        pd.Series(patients_train).isin(patients_unique[train_idx_])
    ]
    val_idx = np.arange(len(X_train))[
        pd.Series(patients_train).isin(patients_unique[val_idx_])
    ]
    
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]
    
    rf = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_fold_train, y_fold_train)
    
    preds_val = rf.predict_proba(X_fold_val)[:, 1]
    auc = roc_auc_score(y_fold_val, preds_val)
    aucs.append(auc)
    lrs.append(rf)

print(f"Mean AUC with best params: {np.mean(aucs):.3f}")
"""

# Replace the training cell
for cell in nb.cells:
    if cell.cell_type == "code":
        # Check for characteristic lines of the old training loop
        if "lrs = []" in cell.source and "LogisticRegression(C=0.01" in cell.source:
             cell.source = optuna_code
             print("Replaced training loop with Optuna optimization.")
             break

nbf.write(nb, nb_path)
print("Notebook modification complete.")
