# Bayesian Fraud Detection & Risk Scoring System

A simple ML project that predicts credit card fraud **as a probability**, then converts that probability into a business decision — **Allow, Manual Review, or Block** — using expected financial loss.

Instead of a model that just says *Fraud* or *Not Fraud*, this project asks: **given the evidence, what is the probability this transaction is fraudulent?** That's Bayesian thinking — starting from a prior belief (fraud is rare) and updating it as evidence comes in.

---

## Highlights

### Statistics Used
- Probability
- Conditional Probability
- Bayes' Theorem
- Prior & Posterior
- Likelihood
- Class Imbalance
- Calibration
- Expected Value
- Cost-sensitive Decision Making

### Machine Learning Used
- Logistic Regression
- Random Forest
- XGBoost
- Probability Calibration
- Feature Engineering
- Cross Validation

---

## Dataset

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Download `creditcard.csv` and place it at:
```
data/creditcard.csv
```

---

## Project Files

| File | Purpose |
|---|---|
| `train_model.py` | Trains Logistic Regression, Random Forest, and XGBoost, picks the best by ROC-AUC, then calibrates its probabilities |
| `decision_engine.py` | Converts a fraud probability into Risk Level, Expected Loss, and a Recommended Action |
| `utils.py` | Shared helpers to load data/model and save the trained model |
| `app.py` | Interactive Streamlit dashboard for scoring transactions |
| `requirements.txt` | Python dependencies |

---

## How to Run

```bash
pip install -r requirements.txt

# place creditcard.csv in data/ first, then:
python3 train_model.py

streamlit run app.py
```

---

## The Core Idea: Probability Calibration

A raw model can be overconfident — a "90% fraud" prediction should mean that, out of 100 similar transactions, about 90 really are fraud. Most raw models don't satisfy this by default.

`train_model.py` fixes this with `CalibratedClassifierCV`, which corrects the model's raw output using the class balance in the data (**prior**) and the model's own confidence (**likelihood**), producing a trustworthy **posterior** probability — the same idea behind Bayes' Theorem, applied through scikit-learn.

---

## Business Decision Logic

`decision_engine.py` applies simple, explainable threshold rules:

| Fraud Probability | Risk Level | Action |
|---|---|---|
| ≥ 75% | High Risk | Block Immediately |
| 40% – 74% | Medium Risk | Send for Manual Review |
| < 40% | Low Risk | Allow |

BLOCK_THRESHOLD & REVIEW_THRESHOLD are thresholds for risk levels and recommended actions. These can be tuned based on business needs.

It also computes **Expected Loss = Fraud Probability × Amount**, so a large low-risk transaction can be fairly compared against a small high-risk one — the same cost-sensitive decision-making used in real fraud systems.

---

