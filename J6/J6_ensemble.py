import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

# Paths
DATA_PATH = "Data"
OUTPUT_DIR = os.path.join(DATA_PATH, "output")

# Liste des fichiers de soumissions candidats (à adapter si certains manquent)
SUBMISSION_FILES = [
    os.path.join(DATA_PATH, "rf_test_output.csv"),              # ~0.68 + (Le meilleur modèle RF de base)
    os.path.join(DATA_PATH, "xgb_robust_submission.csv"),         # Modèle XGBoost robuste
    os.path.join(DATA_PATH, "pca_ridge_submission.csv"),          # Modèle Ridge + PCA
    os.path.join(OUTPUT_DIR, "submission_moco_attention.csv"),    # Nouveau AB-MIL (Deep Learning)
    os.path.join(OUTPUT_DIR, "submission_anomaly_mil.csv")        # Nouveau Anomaly Detection
]

def load_submissions():
    dfs = []
    names = []
    
    for file_path in SUBMISSION_FILES:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # S'assurer que les Sample IDs sont triés pareil
            df = df.sort_values(by="Sample ID").reset_index(drop=True)
            dfs.append(df)
            names.append(os.path.basename(file_path))
        else:
            print(f"ATTENTION: Fichier manquant ignoré -> {file_path}")
            
    return dfs, names

def rank_ensemble(dfs):
    print("--- Rank Ensembling ---")
    n_samples = len(dfs[0])
    
    ranks_matrix = np.zeros((len(dfs), n_samples))
    
    for i, df in enumerate(dfs):
        ranks = rankdata(df["Target"]) / n_samples
        ranks_matrix[i, :] = ranks
        
    weight_map = {
        "rf_test_output.csv": 4.0,
        "xgb_robust_submission.csv": 3.0,
        "pca_ridge_submission.csv": 2.0,
        "submission_moco_attention.csv": 1.5,
        "submission_anomaly_mil.csv": 0.5
    }
    
    active_weights = []
    file_names = [os.path.basename(f) for f in SUBMISSION_FILES if os.path.exists(f)]
    for name in file_names:
        w = weight_map.get(name, 1.0)
        active_weights.append(w)
        
    active_weights = np.array(active_weights)
    active_weights = active_weights / np.sum(active_weights) # Normalize to sum to 1
    
    print("Poids utilisés pour le Rank Ensembling :")
    for name, w in zip(file_names, active_weights):
        print(f"- {name}: {w:.3f}")

    final_ranks = np.average(ranks_matrix, axis=0, weights=active_weights)
    
    return final_ranks

if __name__ == "__main__":
    dfs, names = load_submissions()
    
    if len(dfs) < 2:
        print("Erreur: Au moins 2 fichiers de soumission sont requis pour l'ensemble.")
        exit(1)
        
    final_probs = rank_ensemble(dfs)
    
    final_sub = dfs[0].copy()
    final_sub["Target"] = final_probs
    
    sub_path = os.path.join(OUTPUT_DIR, "j6_final_ensemble_submission.csv")
    final_sub.to_csv(sub_path, index=False)
    print(f"\nSoumission ensemble validée et sauvegardée dans : {sub_path}")

