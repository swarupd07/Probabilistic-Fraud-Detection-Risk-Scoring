# Probabilistic Fraud Detection & Risk Scoring System

A machine learning project that predicts credit card fraud **as a probability**, then converts that probability into a business decision — **Allow, Manual Review, or Block** — using expected financial loss.

Instead of a model that just says *Fraud* or *Not Fraud*, this project asks: **given the evidence, what is the probability this transaction is fraudulent, and what does the business lose if we get it wrong?** The pipeline was built, tested, and revised through several rounds of real experimentation — including catching and fixing a data-leakage bug — documented below alongside the final results.

---

## Key Concepts

### Statistics & Decision Theory
- Probability Estimation & Conditional Probability
- Class Imbalance Handling (class-weighting vs. resampling)
- Probability Calibration (Platt Scaling vs. Isotonic Regression)
- Expected Value & Cost-Sensitive Decision Making
- Population Stability Index (distribution drift detection)
- Precision@K (operational/business-relevant evaluation)

### Machine Learning Used
- Logistic Regression, Random Forest, XGBoost
- Model Selection via **PR-AUC** (not ROC-AUC — see below)
- Probability Calibration (`CalibratedClassifierCV`, Isotonic Regression)
- SMOTE / Balanced Random Forest (evaluated, not deployed — see below)
- Feature Scaling, Temporal Train/Validation/Test Splitting
- SHAP-based Model Explainability

---

## Dataset

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Download `creditcard.csv` and place it at:
```
data/creditcard.csv
```

30 features (`Time`, PCA-anonymized `V1`–`V28`, `Amount`) and a binary `Class` label. Extremely imbalanced — roughly **0.17% of transactions are fraud**.

---

## Project Files

| File | Purpose |
|---|---|
| `train_model.py` | Loads data, splits it temporally, trains Logistic Regression / Random Forest / XGBoost, selects the best by PR-AUC, calibrates it, and finds a cost-optimal decision threshold |
| `model_evaluation.py` | Calibration comparison (Platt vs. Isotonic), Precision@K, cost-based threshold search, and PSI drift-check utilities |
| `test_evaluation.py` | Final, one-time evaluation on the held-out test set — never used for any tuning decision |
| `decision_engine.py` | Converts a fraud probability into Risk Level, Expected Loss, and a Recommended Action, using the saved thresholds |
| `shap_explain.py` | Generates per-transaction SHAP explanations (which features pushed the score up or down) |
| `utils.py` | Shared helpers to load data, save/load the model, scaler, and thresholds |
| `app.py` | Interactive Streamlit dashboard for scoring transactions and viewing model performance |
| `requirements.txt` | Python dependencies |

---

## How to Run

```bash
pip install -r requirements.txt

# place creditcard.csv in data/ first, then:
python3 train_model.py        # trains, calibrates, and saves the model + thresholds
python3 test_evaluation.py    # reports final, unbiased performance on the held-out test set
streamlit run app.py          # launches the interactive dashboard
```

---

## Design Decisions & Experimentation

This project went through several real rounds of experimentation rather than a single train-and-ship pass. The key decisions, what was tried, and what the evidence showed:

### 1. Fixing a data-leakage bug: 75/25 split → 70/15/15 temporal split
The original pipeline used only a train/test split, meaning the calibration method and the cost-optimal threshold were both being chosen using the *same* data later used to report final performance — a leakage issue that made the "final" metrics optimistic. This was fixed by moving to a proper three-way split:

- **Train (70%)** — model fitting only
- **Validation (15%)** — model selection, calibration-method choice, threshold optimization
- **Test (15%)** — touched exactly once, for final reporting only (`test_evaluation.py`)

The split is **temporal** (sorted by `Time`, no shuffling), so the model is always evaluated on transactions that happened *after* the ones it was trained on — mirroring how it would actually be deployed.

### 2. Model selection: PR-AUC instead of ROC-AUC
With fraud occurring in only ~0.17% of transactions, ROC-AUC is dominated by the (easy) majority class and can look nearly identical across models that behave very differently on the rare fraud class. PR-AUC is far more sensitive to minority-class performance, so it was used as the selection criterion instead:

| Model | Precision (fraud) | Recall (fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.053 | 0.929 | 0.9828 | 0.8394 |
| **Random Forest (selected)** | 0.522 | 0.839 | 0.9772 | **0.8513** |
| XGBoost | 0.454 | 0.786 | 0.9816 | 0.8068 |

XGBoost actually had the higher raw ROC-AUC — but Random Forest's clear PR-AUC edge is the more meaningful signal at this level of class imbalance, so **Random Forest was selected**.

### 3. Probability calibration: Platt Scaling vs. Isotonic Regression
A raw model's "90% fraud" prediction doesn't necessarily mean 90 out of 100 similar transactions really are fraud. Both calibration methods were tested and compared by Brier score (lower = better-calibrated):

| Method | Brier Score |
|---|---|
| Sigmoid (Platt) | ~0.00036 |
| **Isotonic (selected)** | **~0.00034** |

Isotonic Regression consistently won and was used to calibrate the final model.

### 4. Cost-based threshold optimization
Rather than using an arbitrary cutoff like 0.5, the decision threshold was chosen by minimizing an explicit business cost function:
```
Total Cost = Σ(Amount of missed fraud) + false_positive_cost × (# false alarms)
```
Searching thresholds from 0.01 to 0.99 against this cost function — instead of optimizing for accuracy or F1 — surfaced a **cost-optimal threshold of 0.15**, with an estimated total test-set cost of **€2,502.40** (€2,372.40 from missed fraud, €130 from false alarms). A sensitivity check across a range of plausible false-positive-cost assumptions confirmed 0.15 is optimal in the realistic €10–100 friction-cost range, with a break-even point (where the optimal threshold jumps to 0.43) around €165/case.

### 5. EDA: Population Stability Index (drift check)
Before trusting the new train/test split, a **Population Stability Index (PSI)** check was run on `Amount` between the train and test sets to rule out distributional drift as an explanation for any performance gap:

```
Amount: PSI = 0.0228   (stable — rule of thumb: <0.1 = stable, 0.1–0.25 = moderate, >0.25 = major drift)
```

No meaningful drift was found — the training and test transaction-amount distributions are consistent.

### 6. Handling class imbalance: SMOTE vs. simple class-weighting
Given how rare fraud is, resampling techniques were evaluated as an alternative to simple class-weighting:

| Configuration | Precision @ deployed threshold | Recall @ deployed threshold |
|---|---|---|
| **Class-weighted, no resampling (deployed)** | **0.750** | **0.750** |
| Balanced Random Forest | Precision collapsed (~0.05 @0.5 cutoff) | — |
| SMOTE (1:1) + class-weighting (double-corrected) | 0.672 | 0.750 |
| SMOTE (sampling_strategy=0.3) | 0.383 | 0.788 |

Every resampling variant either double-corrected for imbalance (when combined with class-weighting) or clearly underperformed the simple weighted baseline on precision at the actual deployed threshold. **The final model uses class-weighting only — no resampling.**

### 7. Final result (held-out test set, never used for any tuning)

| Split | Rows | Fraud | ROC-AUC | PR-AUC | Precision@0.5 | Recall@0.5 | Precision@Block(0.15) | Recall@Block(0.15) |
|---|---|---|---|---|---|---|---|---|
| Train | 199,364 | 384 | 0.9997 | 0.8797 | 0.876 | 0.938 | 0.816 | 0.969 |
| Val | 42,721 | 56 | 0.9816 | 0.8498 | 1.000 | 0.750 | 0.849 | 0.804 |
| **Test** | 42,722 | 52 | **0.9823** | **0.756** | 0.947 | 0.692 | **0.750** | **0.750** |

At the deployed threshold, the model catches **75% of fraud** with **75% precision**, at an estimated total cost of **€2,502.40** on the test set — and every number above comes from data the model never touched during any tuning step.

---

## Business Decision Logic

`decision_engine.py` applies threshold rules **derived from the cost-optimal search above**, not fixed guesses:

| Fraud Probability | Risk Level | Action |
|---|---|---|
| ≥ Block Threshold | High Risk | Block Transaction |
| Review Threshold – Block Threshold | Medium Risk | Send for Manual Review |
| < Review Threshold | Low Risk | Allow |

`BLOCK_THRESHOLD` is the cost-optimal threshold found by `train_model.py`; `REVIEW_THRESHOLD` sits 5 percentage points below it, giving a manual-review buffer zone instead of a single hard cutoff. Both are saved to `models/optimal_threshold.json` and loaded automatically — they can be re-derived any time the cost assumptions change, without retraining the model.

It also computes **Expected Loss = Fraud Probability × Amount**, so a large low-risk transaction can be fairly compared against a small high-risk one — the same cost-sensitive decision-making used in real fraud systems.

---

## Explainability

- **Global feature importance** — which features matter most to the model overall (from the underlying Random Forest).
- **Local SHAP explanations** (`shap_explain.py`) — for any individual transaction, which specific feature values pushed its score toward or away from fraud, surfaced live in the Streamlit dashboard to support the Manual Review workflow.

---

## Skills Demonstrated

- End-to-end ML pipeline design, including diagnosing and fixing a real data-leakage bug
- Model selection and evaluation under extreme class imbalance (PR-AUC vs. ROC-AUC)
- Probability calibration (Platt Scaling vs. Isotonic Regression, compared empirically)
- Cost-sensitive decision theory and threshold optimization tied to business cost
- Distribution drift detection (Population Stability Index)
- Disciplined experimentation with negative results reported honestly (SMOTE/Balanced RF tested and not adopted)
- Model explainability (SHAP) for auditable, transaction-level decisions
- Streamlit dashboard development for interactive model deployment
