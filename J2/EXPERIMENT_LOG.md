# 🧪 ML Experiment Log: OWKIN Feature Engineering & Robustness
Date: 2026-02-16

Owner: @maximilienlucille

Status: 🔴 Failed (no improvement from Baseline)

## 1. Context & Hypothesis
**Objective**: Improve the ROC AUC score in the OWKIN ML2 challenge by moving beyond a simple Mean Pooling / Logistic Regression approach.

**Hypothesis**: 
1. Adding higher-order statistics (Standard Deviation and Maximum) to the Mean pooling will capture slide heterogeneity and "hotspots".
2. Switching to XGBoost will improve accuracy on tabular features.
3. Removing features that differ significantly between Train and Test (Covariate Shift) will reduce the generalization gap.

**Baseline**: 0.650 (Random Forest + Optuna benchmark).

## 2. Configuration (The "How")
**Architecture**: Enhanced Statistics (Mean, Std, Max) -> Feature Slicing -> XGBoost + Optuna.

**Tooling**: `joblib` for parallel feature extraction (5x speedup), `optuna` for hyperparameter tuning.

**Hyperparameters**:
- `learning_rate`: log-uniform [0.01, 0.2] (fixed to `suggest_float` in v2)
- `max_depth`: [3, 8]
- `colsample_bytree`: [0.1, 0.5]

## 3. Results & Metrics

| Iteration | Description | CV Score | Test Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Baseline (RF/LR) | 0.65 | 0.65 | - |
| 1 | XGB + Mean/Std/Max | **0.680** | **0.610** | 🔴 Overfit |
| **2** | **XGB + Drift Removal** | **0.680** | **0.6346** | 🔴 Failed |

## 4. Observations & Learning
- **The Overfitting Trap**: Adding variance and max features tripled our dimensions (6144 features). With only 344 samples, the model "memorized" noise, leading to the 0.61 crash on the test set.
- **Covariate Shift**: Adversarial validation confirmed that certain features are "dead giveaways" of the training set (likely center-specific noise).
- **The Fix**: Removing the **top 20 drifting features** before training forced the model to rely on more stable biological signals.
- **Optuna**: Upgraded to `suggest_float` to avoid deprecation warnings.

## 5. Next Steps (Future Pragmatic Goals)
- [ ] **Distribution Features**: Add 25/50/75 quantiles to refine the histogram representation.
- [ ] **Compression**: Use **PCA** on the 10k+ features to extract the top ~200 most robust components.
- [ ] **Ensembling**: Blend XGBoost, Random Forest, and Ridge to stabilize predictions.
