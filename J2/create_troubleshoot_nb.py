import json
from pathlib import Path

# Create a new notebook dictionary
nb = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_code_cell(source_lines):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    })

def add_markdown_cell(source_lines):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    })

# 1. Imports
add_markdown_cell(["# Troubleshooting: Adversarial Validation & Distribution Shift Analysis",
                   "This notebook diagnoses why the CV score (0.68) is much higher than Test score (0.61).",
                   "We perform **Adversarial Validation** to check if Train and Test data distributions are different."])

add_code_cell([
    "!pip install xgboost joblib scikit-learn matplotlib pandas tqdm",
    "",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "from pathlib import Path",
    "from tqdm import tqdm",
    "from joblib import Parallel, delayed",
    "from sklearn.model_selection import StratifiedKFold, cross_val_score",
    "from sklearn.metrics import roc_auc_score",
    "from xgboost import XGBClassifier"
])

# 2. Data Loading (Same as before)
add_markdown_cell(["## Data Loading & Feature Extraction",
                   "We reuse the efficient Joblib loading developed earlier."])

add_code_cell([
    "data_dir = Path(\"Data\")",
    "train_features_dir = data_dir / \"train_input\" / \"moco_features\"",
    "test_features_dir = data_dir / \"test_input\" / \"moco_features\"",
    "df_train = pd.read_csv(data_dir  / \"supplementary_data\" / \"train_metadata.csv\")",
    "df_test = pd.read_csv(data_dir  / \"supplementary_data\" / \"test_metadata.csv\")",
    "",
    "def process_sample(sample_info, features_dir, is_train=True):",
    "    if is_train:",
    "        sample = sample_info[0]",
    "    else:",
    "        sample = sample_info",
    "        ",
    "    _features = np.load(features_dir / sample)",
    "    # Discard coordinates, keep features",
    "    features = _features[:, 3:]",
    "    ",
    "    # Standard Pooling (Mean, Std, Max)",
    "    mean_feat = np.mean(features, axis=0)",
    "    std_feat = np.std(features, axis=0)",
    "    max_feat = np.max(features, axis=0)",
    "    ",
    "    return np.concatenate([mean_feat, std_feat, max_feat])",
    "",
    "print(\"Loading Train Data...\")",
    "X_train_list = Parallel(n_jobs=-1)(",
    "    delayed(process_sample)(row, train_features_dir, is_train=True) ",
    "    for row in tqdm(df_train[[\"Sample ID\"]].values)",
    ")",
    "X_train = np.array(X_train_list)",
    "",
    "print(\"Loading Test Data...\")",
    "X_test_list = Parallel(n_jobs=-1)(",
    "    delayed(process_sample)(sample, test_features_dir, is_train=False)",
    "    for sample in tqdm(df_test[\"Sample ID\"].values)",
    ")",
    "X_test = np.array(X_test_list)",
    "",
    "print(f\"Train shape: {X_train.shape}, Test shape: {X_test.shape}\")"
])

# 3. Adversarial Validation logic
add_markdown_cell(["## Adversarial Validation",
                   "We try to distinguish Train vs Test samples. If the model can easily do this (AUC > 0.5), it means the features have a different distribution (Covariate Shift)."])

add_code_cell([
    "# Create dataset for Adversarial Validation",
    "# 0 = Train, 1 = Test",
    "y_av_train = np.zeros(len(X_train))",
    "y_av_test = np.ones(len(X_test))",
    "",
    "X_av = np.vstack([X_train, X_test])",
    "y_av = np.concatenate([y_av_train, y_av_test])",
    "",
    "print(f\"Adversarial dataset shape: {X_av.shape}\")",
    "",
    "# Train XGBoost to distinguish Train from Test",
    "model_av = XGBClassifier(",
    "    n_estimators=50,",
    "    max_depth=4,",
    "    learning_rate=0.1,",
    "    eval_metric='auc',",
    "    use_label_encoder=False,",
    "    random_state=42,",
    "    n_jobs=-1",
    ")",
    "",
    "# 5-Fold Stratified CV",
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "scores = cross_val_score(model_av, X_av, y_av, cv=cv, scoring='roc_auc', n_jobs=-1)",
    "",
    "print(f\"Adversarial Validation AUC: {np.mean(scores):.3f} +/- {np.std(scores):.3f}\")",
    "",
    "if np.mean(scores) > 0.70:",
    "    print(\"WARNING: Significant distribution shift detected! The model can easily tell Train from Test.\")",
    "else:",
    "    print(\"Distribution shift seems moderate.\")"
])

# 4. Calibration Analysis
add_markdown_cell(["## Feature Selection based on Drift",
                   "If drift is high, we can find the features responsible for it and remove them."])

add_code_cell([
    "model_av.fit(X_av, y_av)",
    "importance = model_av.feature_importances_",
    "indices = np.argsort(importance)[::-1]",
    "",
    "print(\"Top 20 Drifting Features (Indices):\")",
    "for f in range(20):",
    "    print(f\"{f+1}. Feature {indices[f]} ({importance[indices[f]]:.4f})\")",
    "",
    "# Suggest filtering",
    "drift_features = indices[:50] # Top 50 most different features",
    "print(f\"\\nSuggested action: Try removing these top drifting features from the main model.\")"
])

output_path = "troubleshoot.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Successfully created {output_path}")
