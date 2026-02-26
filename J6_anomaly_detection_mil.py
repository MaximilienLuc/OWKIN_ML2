import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib

# Paths
DATA_PATH = "Data"
TRAIN_LABELS_PATH = os.path.join(DATA_PATH, "train_output.csv")
TRAIN_MOCO_DIR = os.path.join(DATA_PATH, "train_input", "moco_features")
TEST_MOCO_DIR = os.path.join(DATA_PATH, "test_input", "moco_features")
OUTPUT_DIR = os.path.join(DATA_PATH, "output")

# Dimensions
NUM_TILES_PER_PATIENT = 1000
FEATURE_DIM = 2051
TOP_K_ANOMALIES = 20  # Number of "most anomalous" tiles to keep per positive patient

def load_patient_moco(moco_dir, sample_id):
    """Loads the 1000x2051 numpy array for a given patient."""
    file_path = os.path.join(moco_dir, sample_id)
    return np.load(file_path)

def extract_normal_tiles(df, max_patients_to_use=50):
    """
    Extract all tiles from random NEGATIVE patients to build the "normal" distribution.
    We subsample to avoid memory issues with Isolation Forest.
    """
    df_neg = df[df['Target'] == 0]
    
    # Subsample negative patients if there are too many
    if len(df_neg) > max_patients_to_use:
        df_neg = df_neg.sample(n=max_patients_to_use, random_state=42)
        
    print(f"Extracting {len(df_neg) * NUM_TILES_PER_PATIENT} normal tiles from {len(df_neg)} negative patients...")
    all_normal_tiles = []
    
    for sample_id in df_neg['Sample ID']:
        tiles = load_patient_moco(TRAIN_MOCO_DIR, sample_id)
        all_normal_tiles.append(tiles)
        
    return np.vstack(all_normal_tiles)

def build_tile_level_dataset(df, iso_forest):
    """
    Uses the trained Isolation Forest to assign pseudo-labels to tiles.
    - Negative patient: Take TOP_K random tiles, label = 0
    - Positive patient: Take TOP_K most anomalous tiles, label = 1
    Returns X (features) and y (tile labels), plus patient mapping for validation.
    """
    X_tiles = []
    y_tiles = []
    patient_ids = []  # To track which tile belongs to which patient (useful for GroupKFold or grouping predictions)
    
    print(f"Pseudo-labeling tiles for {len(df)} patients...")
    for idx, row in df.iterrows():
        sample_id = row['Sample ID']
        target = row['Target']
        
        tiles = load_patient_moco(TRAIN_MOCO_DIR, sample_id) # [1000, 2051]
        
        if target == 0:
            # For negative patients, all 1000 tiles are normal. 
            # We pick TOP_K random tiles to balance the dataset.
            indices = np.random.choice(NUM_TILES_PER_PATIENT, TOP_K_ANOMALIES, replace=False)
            selected_tiles = tiles[indices]
            labels = np.zeros(TOP_K_ANOMALIES)
        else:
            # For positive patients, we score all 1000 tiles with Isolation Forest
            # IsolationForest returns anomaly score (lower means more anomalous)
            # Actually decision_function returns > 0 for normal, < 0 for anomalies.
            scores = iso_forest.decision_function(tiles)
            
            # We want the MOST anomalous tiles (lowest scores)
            # argsort sorts ascending, so the first K elements are the most negative (most anomalous)
            anomaly_indices = np.argsort(scores)[:TOP_K_ANOMALIES]
            selected_tiles = tiles[anomaly_indices]
            labels = np.ones(TOP_K_ANOMALIES)
            
        X_tiles.append(selected_tiles)
        y_tiles.append(labels)
        patient_ids.extend([sample_id] * TOP_K_ANOMALIES)
        
    X_tiles = np.vstack(X_tiles)
    y_tiles = np.concatenate(y_tiles)
    
    return X_tiles, y_tiles, np.array(patient_ids)

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_LABELS_PATH)
    
    # ---------------------------------------------------------
    # STEP 1: Train Anomaly Detector on NORMAL data
    # ---------------------------------------------------------
    print("--- STEP 1: Training Isolation Forest on Normal Data ---")
    normal_tiles = extract_normal_tiles(df, max_patients_to_use=30) # ~30,000 tiles
    
    # Train Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(normal_tiles)
    print("Isolation Forest training complete.\n")
    
    # ---------------------------------------------------------
    # STEP 2: Cross Validation for LightGBM Classifier
    # ---------------------------------------------------------
    print("--- STEP 2: Training LightGBM on Pseudo-labeled Tiles ---")
    
    # Stratified KFold on PATIENT LEVEL
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(df))
    
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df, df['Target'])):
        print(f"\n[Fold {fold+1}/5]")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        
        # Build tile-level dataset for training
        X_train_tiles, y_train_tiles, _ = build_tile_level_dataset(df_train, iso_forest)
        
        # Train LightGBM on tiles
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        clf.fit(X_train_tiles, y_train_tiles)
        models.append(clf)
        
        # Validation: For each patient in validation, predict all their 1000 tiles
        # and take the mean of the top K highest probabilities
        for i, row in df_val.iterrows():
            sample_id = row['Sample ID']
            val_idx_in_original = val_idx[i]
            
            tiles = load_patient_moco(TRAIN_MOCO_DIR, sample_id)
            # Predict probability of being cancerous for all 1000 tiles
            tile_probs = clf.predict_proba(tiles)[:, 1]
            
            # The patient-level probability is the AVERAGE of the TOP_K most cancerous tiles
            top_k_probs = np.sort(tile_probs)[-TOP_K_ANOMALIES:]
            patient_prob = np.mean(top_k_probs)
            
            oof_predictions[val_idx_in_original] = patient_prob
            
        # Fold AUC
        fold_auc = roc_auc_score(df_val['Target'], oof_predictions[val_idx])
        print(f"Fold {fold+1} Val AUC: {fold_auc:.4f}")
        
    overall_auc = roc_auc_score(df['Target'], oof_predictions)
    print(f"\n====================================")
    print(f"OVERALL OOF AUC: {overall_auc:.4f}")
    print(f"====================================\n")
    
    # ---------------------------------------------------------
    # STEP 3: Inference on Test Set
    # ---------------------------------------------------------
    print("--- STEP 3: Inference on Test Set ---")
    test_files = [f for f in os.listdir(TEST_MOCO_DIR) if f.endswith('.npy')]
    test_df = pd.DataFrame({'Sample ID': test_files})
    
    final_preds = np.zeros(len(test_df))
    
    for i, row in test_df.iterrows():
        sample_id = row['Sample ID']
        tiles = load_patient_moco(TEST_MOCO_DIR, sample_id)
        
        # Average predictions from the 5 fold models
        fold_probs = []
        for clf in models:
            tile_probs = clf.predict_proba(tiles)[:, 1]
            top_k_probs = np.sort(tile_probs)[-TOP_K_ANOMALIES:]
            patient_prob = np.mean(top_k_probs)
            fold_probs.append(patient_prob)
            
        final_preds[i] = np.mean(fold_probs)
        
    test_df['Target'] = final_preds
    sub_path = os.path.join(OUTPUT_DIR, "submission_anomaly_mil.csv")
    test_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

if __name__ == "__main__":
    run_pipeline()
