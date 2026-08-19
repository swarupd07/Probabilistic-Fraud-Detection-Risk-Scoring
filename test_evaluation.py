# test_evaluation.py

# FINAL, UNBIASED evaluation on the held-out TEST set (15% of the data).
# This split is NEVER touched during model selection, calibration-method comparison,
# or cost-threshold optimization.

import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from utils import load_data, load_model, load_thresholds
from train_model import prepare_data
from model_evaluation import print_precision_at_k_table


def evaluate_on_test_set(block_threshold=None, review_threshold=None, false_positive_cost=10):
    df = load_data()

    ( _, _, X_test_scaled, y_train, y_val, y_test, _, X_train_raw, X_val_raw, X_test_raw) = prepare_data(df)

    model, _ = load_model()
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Load thresholds dynamically if not provided
    if block_threshold is None or review_threshold is None:
        saved_block, saved_review = load_thresholds(default_block=0.65, default_review=0.60)
        if block_threshold is None:
            block_threshold = saved_block
        if review_threshold is None:
            review_threshold = saved_review

    print("=" * 60)
    print("FINAL TEST SET EVALUATION (held-out, never used for tuning)")
    print("=" * 60)
    print(f"Test set size: {len(y_test)} transactions ({int(y_test.sum())} fraud)")

    preds_05 = (probs >= 0.5).astype(int)
    print("\n--- Classification Report (@ 0.5 cutoff) ---")
    print(classification_report(y_test, preds_05, digits=3, zero_division=0))

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    # Precision@K: the business-relevant metric, on the true holdout
    print_precision_at_k_table(y_test.values, probs)

    print(f"\n--- At Deployed Block Threshold ({block_threshold:.2f}) ---")
    block_preds = (probs >= block_threshold).astype(int)
    print(
        classification_report(
            y_test,
            block_preds,
            digits=3,
            zero_division=0,
            target_names=["Not Blocked", "Blocked"],
        )
    )

    amounts_test = X_test_raw["Amount"].values
    false_negatives = (probs < block_threshold) & (y_test.values == 1)
    false_positives = (probs >= block_threshold) & (y_test.values == 0)
    fn_cost = float(amounts_test[false_negatives].sum())
    fp_cost = float(false_positives.sum() * false_positive_cost)
    total_cost = fn_cost + fp_cost

    print(f"Estimated cost on TEST set at deployed Block threshold: € {total_cost:,.2f}")
    print(f"  (missed-fraud cost: € {fn_cost:,.2f}, false-alarm cost: € {fp_cost:,.2f})")

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "test_cost": round(float(total_cost), 2),
        "fn_cost": round(float(fn_cost), 2),
        "fp_cost": round(float(fp_cost), 2),
        "probs": probs,
        "y_test": y_test,
        "block_threshold": block_threshold,
        "review_threshold": review_threshold,
    }


if __name__ == "__main__":
    evaluate_on_test_set()
