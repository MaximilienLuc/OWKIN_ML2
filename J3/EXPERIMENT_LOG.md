# 🧪 ML Experiment Log: PCA + Regularization
Date: 2026-02-24

Owner: @maximilienlucille

Status: 🔴 Failed (below Random Forest Optimized baseline)

## 1. Context & Hypothesis
**Objective**: Reduce dimensionality of the Mean/Std/Max pooling features (6,144 dims) via PCA, then apply regularized linear classifiers to combat the curse of dimensionality observed in J2.

**Hypothesis**:
1. PCA will compress the 6,144 raw features into a compact, denoised representation, removing center-specific noise and irrelevant variance.
2. A regularized Logistic Regression (L2/Ridge) on the PCA-reduced space will generalize better than tree-based models on such a small dataset (n=344).
3. The combination of dimensionality reduction + regularization will avoid the overfitting trap seen with XGBoost in J2.

**Baseline**: 0.6554 (Random Forest + Optuna, from J2).

## 2. Configuration (The "How")
**Architecture**: Mean/Std/Max Pooling → StandardScaler → PCA(150) → LogisticRegression L2

**Pipeline**:
1. Feature extraction: Mean + Std + Max pooling on MoCo features → (344, 6144)
2. StandardScaler fitted on train only
3. PCA: grid search over candidates [50, 100, 150, 200, 250]
4. Model comparison: LogReg L2, LogReg L1, RidgeClassifier, ElasticNet (SGD) across 9 C values
5. Cross-validation: Patient-level stratified 5-fold × 5 repeats

**Best Hyperparameters**:
- PCA components: **150**
- Model: **LogisticRegression L2, C=0.01**
- Solver: `lbfgs`, max_iter=2000

**Ensembling**: 25 models (5 folds × 5 repeats), averaged predictions on test set.

## 3. Results & Metrics

| Iteration | Description | CV Score | Test Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Baseline (RF + Optuna) | ~0.65 | **0.6554** | ✅ Best |
| 1 | PCA(150) + LogReg L2 C=0.01 | — | **0.6434** | 🔴 Failed |
| 2 | RF + PCA-Ridge Ensemble (blend α=0.55) | 0.6975 | **0.6306** | 🔴 Failed (overfit) |
| **3a** | **Conservative RF — Mean only (2048d)** | **0.6713** | **TBD** | 🟡 Pending |
| **3b** | **Conservative RF — Ext Filtered (18382d)** | **0.6865** | **TBD** | 🟡 Pending |
| **3c** | **Conservative RF — Ext Full (18432d)** | **0.6878** | **TBD** | 🟡 Pending |

### Iteration 2 — Ensemble Breakdown

| Method | OOF AUC |
| :--- | :--- |
| RF alone (Optuna-tuned, Mean 2048d) | 0.6774 |
| PCA-Ridge alone (Full 6144d → PCA 150) | 0.6932 |
| Simple average (0.5 / 0.5) | 0.6972 |
| Weighted average (by AUC) | 0.6969 |
| **Best blend (α=0.55 RF + 0.45 Ridge)** | **0.6975** |
| Stacking meta-learner (LogReg) | 0.6934 |

**RF Optuna best params**: n_estimators=57, max_depth=20, min_samples_split=15, min_samples_leaf=10, max_features=0.1

## 4. Observations & Learnings

### Iteration 1 — PCA-Ridge
- **PCA crushes useful signal**: Reducing from 6,144 to 150 components explains ~95% of variance, but the discarded 5% likely contains discriminative biological features that tree-based models can exploit non-linearly.
- **Linear models are too simplistic**: Even with optimal regularization, Logistic Regression cannot capture the non-linear interactions between pathology features that Random Forest handles naturally.
- **Regularization search was thorough**: 9 values of C tested across 4 model families (L2, L1, Ridge, ElasticNet). All performed comparably, suggesting the bottleneck is the linear assumption, not the regularization strength.
- **Patient-level CV is robust**: The 5-fold × 5-repeat patient-level split avoids data leakage between slides of the same patient.

### Iteration 2 — RF + PCA-Ridge Ensemble
- **Diversity pays off**: The two models have partially uncorrelated errors — RF captures non-linear patterns on raw features, PCA-Ridge captures linear signal in a denoised compressed space. Their blend outperforms both individually.
- **Simple averaging beats stacking**: With only 2 meta-features, the LogReg meta-learner slightly overfits (0.6934 < 0.6975). Simple blending avoids this.
- **Meta-learner coefficients** [0.997, 1.748] confirm Ridge gets more weight — consistent with its higher individual OOF AUC (0.6932 > 0.6774).
- **Optimal blend α=0.55** is close to 50/50, suggesting both models contribute meaningfully.
- **OOF AUC 0.6975 is promising** but may not fully translate to leaderboard score. The key test is the actual submission.

## 5. Next Steps
- [x] **Ensemble methods**: Combine RF + PCA-Ridge predictions (stacking/blending) to leverage both linear and non-linear strengths.
- [ ] **Submit and evaluate**: Submit `ensemble_best_blend.csv` and `ensemble_avg.csv` to leaderboard.
- [ ] **Non-linear on PCA space**: Try SVM-RBF or small MLP on the PCA-reduced features.
- [ ] **Feature selection instead of PCA**: Use mutual information or LASSO-based feature selection to preserve discriminative features rather than maximizing variance.
- [ ] **Quantile features**: Add 25/50/75 percentile pooling for richer slide representation.
