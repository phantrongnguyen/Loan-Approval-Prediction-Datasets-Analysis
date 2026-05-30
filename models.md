# Model Implementation & Analysis

## 1. Overview

| Item | Description |
|------|-------------|
| **Task** | Binary classification: Loan Approval (Approved / Rejected) |
| **Dataset** | 4,269 records, 12 original features → 11 final features |
| **Target** | `loan_status` — 62.2% Approved (1), 37.8% Rejected (0) |
| **Models** | Logistic Regression (baseline) → Random Forest (tuned) |
| **Best Model** | Random Forest — **97.42% Accuracy**, **0.9984 ROC AUC** |

---

## 2. Data Preprocessing & Feature Engineering

### 2.1 Raw Data Cleaning

- Stripped leading/trailing whitespace from all column names and string values
- Label-encoded categorical variables:
  - `education`: `{'Graduate': 1, 'Not Graduate': 0}`
  - `self_employed`: `{'Yes': 1, 'No': 0}`
  - `loan_status`: `{'Approved': 1, 'Rejected': 0}`

### 2.2 Engineered Features (5 features — later removed)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `total_assets_value` | `residential + commercial + luxury + bank_asset` | Total wealth proxy |
| `loan_to_income_ratio` | `loan_amount / income_annum` | Debt burden |
| `assets_to_loan_ratio` | `total_assets / loan_amount` | Collateral coverage |
| `income_per_dependent` | `income_annum / (dependents + 1)` | Per-capita income |
| `loan_per_term` | `loan_amount / loan_term` | Annual repayment pressure |

**All 5 were removed** after identifying them as potential data leakage sources (derived from input features already present).

### 2.3 Final Feature Set (11 features)

```
['no_of_dependents', 'education', 'self_employed', 'income_annum',
 'loan_amount', 'loan_term', 'cibil_score',
 'residential_assets_value', 'commercial_assets_value',
 'luxury_assets_value', 'bank_asset_value']
```

---

## 3. Model 1: Logistic Regression (Baseline)

### 3.1 Implementation

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

baseline_pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), numeric_features)
    ])),
    ("model", LogisticRegression(max_iter=1000))
])
```

### 3.2 Training Configuration

- **Scaler**: `StandardScaler` (all 16 numeric features, including engineered)
- **Split**: 80/20 stratified (`test_size=0.2`, `random_state=42`)
- **Solver**: L-BFGS (default), max_iter=1000

### 3.3 Results

| Metric | Train | Test |
|--------|:----:|:----:|
| Accuracy | 0.9280 | **0.9251** |
| Precision | 0.9400 | 0.9332 |
| Recall | 0.9445 | 0.9473 |
| F1 Score | 0.9423 | 0.9402 |
| ROC AUC | 0.9723 | 0.9789 |
| Log Loss | 0.2075 | 0.1833 |

**Cross-Validation (5-fold):** Mean accuracy = 0.9276 (±0.0074)

### 3.4 Confusion Matrix (Test)

| | Pred: Rejected (0) | Pred: Approved (1) |
|---|---|---|
| **Actual: Rejected (0)** | 288 (TN) | 35 (FP) |
| **Actual: Approved (1)** | 28 (FN) | 503 (TP) |

- Sensitivity: 94.7%
- Specificity: 89.2%

### 3.5 Analysis

Logistic Regression provides a solid baseline. The model generalizes well (train/test gap ≈ 0.3%), but leaves room for improvement with a non-linear model.

---

## 4. Model 2: Random Forest (Default)

### 4.1 Implementation (Pre-tuning)

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_split=15,
    min_samples_leaf=8,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### 4.2 Training Configuration

- **Features**: 11 (removed `loan_id` + 5 leakage features)
- **Split**: 70/30 stratified (`test_size=0.3`, `random_state=42`)
- **Train size**: 2,988 | **Test size**: 1,281

### 4.3 Results

| Metric | Train | Test |
|--------|:----:|:----:|
| Accuracy | 0.9639 | **0.9688** |
| Precision | 0.9944 | 0.9961 |
| Recall | 0.9473 | 0.9536 |
| F1 Score | 0.9702 | 0.9744 |
| ROC AUC | 0.9990 | 0.9982 |
| Log Loss | 0.1294 | 0.1314 |

**Overfitting gap**: -0.0049 (test accuracy > train accuracy)

### 4.4 Cross-Validation (5-fold)

```
Scores: [0.9599, 0.9599, 0.9565, 0.9698, 0.9514]
Mean CV Accuracy: 0.9595 (±0.0060)
```

---

## 5. Model 3: Random Forest (Tuned — Best Model)

### 5.1 Hyperparameter Tuning via GridSearchCV

```python
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [5, 6],
    'min_samples_split': [15, 20],
    'min_samples_leaf': [8, 10],
    'max_features': ['sqrt', 0.6],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid_small,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
```

**Search space**: 32 combinations × 3 folds = 96 fits

### 5.2 Best Parameters

| Parameter | Value |
|-----------|:-----:|
| `class_weight` | `'balanced'` |
| `max_depth` | 6 |
| `max_features` | 0.6 |
| `min_samples_leaf` | 8 |
| `min_samples_split` | 15 |
| `n_estimators` | 150 |

**Best CV Score**: 0.9675

### 5.3 Final Model Architecture

```python
RandomForestClassifier(
    class_weight='balanced',
    max_depth=6,
    max_features=0.6,
    min_samples_leaf=8,
    min_samples_split=15,
    n_estimators=150,
    n_jobs=-1,
    random_state=42
)
```

### 5.4 Results

| Metric | Train | Test |
|--------|:----:|:----:|
| Accuracy | 0.9732 | **0.9742** |
| Precision | 0.9944 | **0.9935** |
| Recall | 0.9623 | **0.9649** |
| F1 Score | 0.9781 | **0.9790** |
| ROC AUC | 0.9993 | **0.9984** |
| Log Loss | 0.0551 | **0.0573** |

**Overfitting gap**: -0.0010 (test accuracy exceeds train — excellent generalization)

### 5.5 Confusion Matrix (Test Set)

| | Pred: Rejected (0) | Pred: Approved (1) |
|---|---|---|
| **Actual: Rejected (0)** | 481 (TN) | 3 (FP) |
| **Actual: Approved (1)** | 30 (FN) | 767 (TP) |

- **Specificity**: 99.4% (excellent at rejecting bad loans)
- **Sensitivity/Recall**: 96.2%
- **False Positive Rate**: 0.6%
- **False Negative Rate**: 3.8%

### 5.6 Classification Report

```
              precision    recall  f1-score   support
           0       0.94      0.99      0.97       484
           1       0.99      0.96      0.98       797
    accuracy                           0.97      1281
   macro avg       0.97      0.98      0.97      1281
weighted avg       0.98      0.97      0.97      1281
```

### 5.7 Cross-Validation (5-fold)

```
Mean CV Accuracy: 0.9645 (±0.0067)
```

---

## 6. Feature Importance Analysis

### 6.1 Top Features (Tuned Random Forest)

| Rank | Feature | Importance | Cumulative |
|:----:|---------|:----------:|:----------:|
| 1 | **cibil_score** | **91.32%** | 91.32% |
| 2 | loan_term | 4.65% | 95.97% |
| 3 | loan_amount | 1.71% | 97.68% |
| 4 | income_annum | 0.73% | 98.41% |
| 5 | luxury_assets_value | 0.44% | 98.85% |
| 6 | commercial_assets_value | 0.38% | 99.24% |
| 7 | residential_assets_value | 0.34% | 99.58% |
| 8 | bank_asset_value | 0.25% | 99.83% |
| 9 | no_of_dependents | 0.11% | 99.94% |
| 10 | self_employed | 0.04% | 99.97% |
| 11 | education | 0.03% | 100.00% |

### 6.2 Key Insights

- **`cibil_score` dominates** with 91.32% importance — the CIBIL credit score is overwhelmingly the strongest predictor of loan approval
- **`loan_term`** (4.65%) and **`loan_amount`** (1.71%) provide moderate contribution
- **`education`** and **`self_employed`** are nearly irrelevant (< 0.05% combined), confirmed by Chi-Square tests (p=0.772 and p=1.000 respectively)
- All 4 asset features combined contribute only ~1.4%

### 6.3 Model Without `cibil_score` (Recommendation)

Since `cibil_score` dominates 91%, the model effectively reduces to a single-variable decision. For a more robust model, consider training without `cibil_score` to understand how well other features can predict loan approval independently.

---

## 7. Model Comparison Summary

| Metric | LR Baseline | RF Default | RF Tuned (Best) |
|--------|:----------:|:----------:|:--------------:|
| Test Accuracy | 0.9251 | 0.9688 | **0.9742** |
| Test Precision | 0.9332 | 0.9961 | **0.9935** |
| Test Recall | 0.9473 | 0.9536 | **0.9649** |
| Test F1 | 0.9402 | 0.9744 | **0.9790** |
| Test ROC AUC | 0.9789 | 0.9982 | **0.9984** |
| Test Log Loss | 0.1833 | 0.1314 | **0.0573** |
| Overfitting Gap | +0.003 | -0.005 | -0.001 |
| CV Mean | 0.9276 | 0.9595 | **0.9645** |

**Improvement from LR → Best RF**: +4.91% accuracy, +3.88% F1, -0.1260 log loss

---

## 8. Statistical Validation

| Test | Variables | p-value | Conclusion |
|------|-----------|:-------:|------------|
| Chi-Square | education vs loan_status | 0.772 | Not significant |
| Chi-Square | self_employed vs loan_status | 1.000 | Not significant |
| Pearson | income_annum vs loan_amount | <0.001 | Strong positive correlation |
| Pearson | total_assets vs bank_asset | <0.001 | Strong positive correlation |
| ANOVA | education vs loan_amount | 0.487 | Not significant |
| ANOVA | loan_status vs loan_amount | 0.291 | Not significant |

---

## 9. Prediction Demo

```python
sample = {
    "no_of_dependents": 2,
    "education": 1,
    "self_employed": 0,
    "income_annum": 500000,
    "loan_amount": 200000,
    "loan_term": 10,
    "cibil_score": 750,
    "residential_assets_value": 300000,
    "commercial_assets_value": 100000,
    "luxury_assets_value": 50000,
    "bank_asset_value": 200000
}
# Prediction: 1 (Approved)
# Approval Probability: 0.908
```

---

## 10. Saved Model Files

| File | Path |
|------|------|
| Logistic Regression (baseline) | `models/logistic_regression_baseline.pkl` |
| Best Random Forest (tuned) | `models/random_forest_best_fixed.pkl` |
| Feature names (RF) | `models/feature_names_fixed.pkl` |
| Feature columns (LR) | `models/feature_columns.pkl` |

---

## 11. Conclusions & Recommendations

1. **Random Forest with tuning achieved 97.42% accuracy** with minimal overfitting
2. **CIBIL score is the dominant predictor** (91.3%) — recommend investigating model performance without it
3. **Education and self-employment status are irrelevant** for loan approval prediction
4. **Consider additional models** (XGBoost, LightGBM) to compare performance
5. **Consider SMOTE / threshold tuning** to further improve recall on the minority class (Rejected)
