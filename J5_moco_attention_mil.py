import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Paramètres
EPOCHS = 30
BATCH_SIZE = 4
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7 # Early stopping patience
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Chemins
DATA_PATH = "Data"
TRAIN_LABELS = os.path.join(DATA_PATH, "train_output.csv")
TRAIN_MOCO_DIR = os.path.join(DATA_PATH, "train_input", "moco_features")
TEST_MOCO_DIR = os.path.join(DATA_PATH, "test_input", "moco_features")

# ==========================================
# 1. DATASET
# ==========================================
class MoCoBagDataset(Dataset):
    def __init__(self, df, moco_dir, is_train=True):
        self.df = df
        self.moco_dir = moco_dir
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['Sample ID']
        
        # Load the 1000x2048 numpy array
        file_path = os.path.join(self.moco_dir, file_name)
        moco_features = np.load(file_path) # [1000, 2048]
        
        # Convert to tensor
        bag = torch.tensor(moco_features, dtype=torch.float32)
        
        if self.is_train:
            label = torch.tensor(row['Target'], dtype=torch.float32)
            return bag, label, file_name
        else:
            return bag, file_name

# ==========================================
# 2. MODEL : Attention-Based MIL
# ==========================================
class MoCoAttentionMIL(nn.Module):
    def __init__(self, feature_dim=2051, hidden_dim=256):
        super(MoCoAttentionMIL, self).__init__()
        
        # Réduction de dimensionnalité optionnelle pour features (ex: 2048 -> 512)
        # Mais ici on garde direct l'Attention pour être simple.
        
        # 1. Mécanisme d'Attention (Ilse et al.)
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # 2. Classifieur Final
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, num_instances, feature_dim] -> e.g. [B, 1000, 2048]
        
        # Astuce : On reshape pour appliquer les couches Linear sur toutes les tuiles
        B, N, D = x.size()
        x_flat = x.view(B * N, D) # [B*1000, 2048]
        
        # Calcul de l'Attention avec Gated Attention (plus performant que standard)
        A_V = self.attention_V(x_flat) # [B*1000, hidden_dim]
        A_U = self.attention_U(x_flat) # [B*1000, hidden_dim]
        A = self.attention_weights(A_V * A_U) # [B*1000, 1]
        
        A = A.view(B, N) # [B, 1000]
        # Poids de l'attention entre 0 et 1 (Softmax sur la dimension des 1000 tuiles)
        A = F.softmax(A, dim=1) # [B, 1000]
        
        # Agrégation : Somme pondérée
        # A.unsqueeze(1) : [B, 1, 1000]
        # x : [B, 1000, 2048]
        # torch.bmm : [B, 1, 1000] x [B, 1000, 2048] -> [B, 1, 2048]
        M = torch.bmm(A.unsqueeze(1), x).squeeze(1) # [B, 2048]
        
        # Prédiction Finale avec le vecteur patient agrégé "M"
        y_prob = self.classifier(M) # [B, 1]
        
        return y_prob.squeeze(1), A

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model():
    print(f"--- Entraînement MoCo AB-MIL sur {DEVICE} ---")
    df = pd.read_csv(TRAIN_LABELS)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(df))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df, df['Target'])):
        print(f"\n--- Fold {fold+1}/5 ---")
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = MoCoBagDataset(df_train, TRAIN_MOCO_DIR, is_train=True)
        val_dataset = MoCoBagDataset(df_val, TRAIN_MOCO_DIR, is_train=True)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = MoCoAttentionMIL().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        criterion = nn.BCELoss()
        
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss = 0.0
            for bags, labels, _ in train_loader:
                bags, labels = bags.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                probs, att_weights = model(bags)
                loss = criterion(probs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for bags, labels, _ in val_loader:
                    bags, labels = bags.to(DEVICE), labels.to(DEVICE)
                    probs, _ = model(bags)
                    loss = criterion(probs, labels)
                    val_loss += loss.item()
                    
                    val_preds.extend(probs.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_auc = roc_auc_score(val_targets, val_preds)
            scheduler.step(val_auc)
            
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
                
        # Store OOF predictions
        model.load_state_dict(best_model_state)
        # save model
        torch.save(model.state_dict(), f"moco_abmil_fold{fold}.pt")
        
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for bags, _, _ in val_loader:
                bags = bags.to(DEVICE)
                probs, _ = model(bags)
                fold_preds.extend(probs.cpu().numpy())
        oof_predictions[val_idx] = fold_preds
        
        print(f"Fold {fold+1} Best Val AUC: {best_val_auc:.4f}")
        
    overall_auc = roc_auc_score(df['Target'], oof_predictions)
    print(f"\n====================================")
    print(f"Overall OOF AUC Score: {overall_auc:.4f}")
    print(f"====================================")

# ==========================================
# 4. INFERENCE
# ==========================================
def run_inference():
    print(f"--- Inférence sur Test Set ---")
    test_files = os.listdir(TEST_MOCO_DIR)
    test_df = pd.DataFrame({'Sample ID': test_files})
    
    test_dataset = MoCoBagDataset(test_df, TEST_MOCO_DIR, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_fold_preds = []
    
    for fold in range(5):
        model = MoCoAttentionMIL().to(DEVICE)
        model.load_state_dict(torch.load(f"moco_abmil_fold{fold}.pt", map_location=DEVICE, weights_only=True))
        model.eval()
        
        preds = []
        with torch.no_grad():
            for bags, _ in test_loader:
                bags = bags.to(DEVICE)
                probs, _ = model(bags)
                preds.extend(probs.cpu().numpy())
        all_fold_preds.append(preds)
        
    # Mean across folds
    final_preds = np.mean(all_fold_preds, axis=0)
    
    test_df['Target'] = final_preds
    
    # Save submission
    os.makedirs(os.path.join(DATA_PATH, "output"), exist_ok=True)
    sub_path = os.path.join(DATA_PATH, "output", "submission_moco_attention.csv")
    test_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

if __name__ == '__main__':
    train_model()
    run_inference()
