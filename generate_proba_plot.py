import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
import pickle

train_features_dir = Path("/Users/enfants/Code/OWKIN_ML2/data/train_features/")
df_train = pd.read_csv("/Users/enfants/Code/OWKIN_ML2/data/train_metadata.csv")
train_output = pd.read_csv("/Users/enfants/Code/OWKIN_ML2/data/train_output.csv")
df_train = df_train.merge(train_output, on="Sample ID")

df_tiles_train = pd.read_csv("/Users/enfants/Code/OWKIN_ML2/tiles_predictions.csv")

def plot_patient_distributions(df_tiles):
    bins = np.linspace(0, 1, 51)
    
    patient_hists = []
    patient_targets = []
    
    print("Computing histograms per patient...")
    for patient, group in df_tiles.groupby("Patient ID"):
        target = group["Target"].iloc[0]
        # Calculate density histogram for each patient
        hist, _ = np.histogram(group["Prediction"], bins=bins, density=True)
        patient_hists.append(hist)
        patient_targets.append(target)
        
    patient_hists = np.array(patient_hists)
    patient_targets = np.array(patient_targets)
    
    # Average histograms per group
    mean_hist_mutated = np.mean(patient_hists[patient_targets == 1], axis=0)
    mean_hist_non_mutated = np.mean(patient_hists[patient_targets == 0], axis=0)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, mean_hist_mutated, label="Mutated (Target=1)", color="red", lw=2)
    plt.plot(bin_centers, mean_hist_non_mutated, label="Non-Mutated (Target=0)", color="blue", lw=2)
    plt.fill_between(bin_centers, mean_hist_mutated, alpha=0.3, color="red")
    plt.fill_between(bin_centers, mean_hist_non_mutated, alpha=0.3, color="blue")
    
    plt.title("Average Tile-Level Probability Distributions per Patient Group")
    plt.xlabel("Tile-Level Probability (Mutated)")
    plt.ylabel("Average Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("/Users/enfants/Code/OWKIN_ML2/report/tile_proba_distribution.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_patient_distributions(df_tiles_train)
