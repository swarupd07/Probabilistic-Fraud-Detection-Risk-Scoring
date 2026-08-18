# model_evaluation.py

# Advanced evaluation tools:
# 1. Calibration comparison (Platt vs Isotonic vs Uncalibrated)
# 2. Precision@K (Operational business metric)
# 3. Cost-based threshold optimization (Dollar/Rupee cost minimization)
# 4. Population Stability Index (PSI)

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, precision_recall_curve


# ---------------------------------------------------------------------
# 1. Calibration comparison: Platt (sigmoid) vs Isotonic
# ---------------------------------------------------------------------
def compare_calibration_methods(base_model, X_train, y_train, X_val, y_val):
    print("\nEXPERIMENT: Platt (sigmoid) vs Isotonic Calibration")
    results = {}

    # Check uncalibrated baseline Brier score
    base_probs = base_model.predict_proba(X_val)[:, 1]
    base_brier = brier_score_loss(y_val, base_probs)
    print(f"Uncalibrated base -> Brier score: {base_brier:.5f}")

    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(clone(base_model), method=method, cv=3)
        calibrated.fit(X_train, y_train)
        probs = calibrated.predict_proba(X_val)[:, 1]
        brier = brier_score_loss(y_val, probs)
        results[method] = {"model": calibrated, "brier": brier}
        print(f"{method.capitalize():>9} calibration -> Brier score: {brier:.5f} (lower is better)")

    best_method = min(results, key=lambda m: results[m]["brier"])
    print(f"Chosen method: {best_method} (lowest Brier score)")

    return results[best_method]["model"], best_method, results


# ---------------------------------------------------------------------
# 2. Precision@K
# ---------------------------------------------------------------------
def precision_at_k(y_true, y_scores, k_percent):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    n = len(y_true)
    k = max(1, int(np.ceil(n * (k_percent / 100))))

    top_k_indices = np.argsort(y_scores)[::-1][:k]
    return float(y_true[top_k_indices].mean())


def print_precision_at_k_table(y_true, y_scores, k_values=(0.5, 1, 2, 5, 10)):
    print("\n Precision@K (top K% riskiest transactions reviewed)")
    for k in k_values:
        p = precision_at_k(y_true, y_scores, k_percent=k)
        print(f"Top {k:>4}% flagged -> Precision: {p:.3f}")


# ---------------------------------------------------------------------
# 3. Cost-based threshold optimization
# ---------------------------------------------------------------------
def find_cost_optimal_threshold(y_true, y_scores, amounts, false_positive_cost=10, verbose=True):
    """
    Finds the decision threshold that minimizes total business cost:
    Total Cost = False Negatives Cost (sum of missed fraud transaction amounts)
               + False Positives Cost (count of false alarms * friction cost per case)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    amounts = np.asarray(amounts)

    thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold = 0.5
    best_cost = np.inf
    cost_curve = []  # (threshold, total_cost) pairs -> for plotting

    for t in thresholds:
        predicted_fraud = y_scores >= t

        false_negatives = (~predicted_fraud) & (y_true == 1)
        false_positives = predicted_fraud & (y_true == 0)

        fn_cost = float(amounts[false_negatives].sum())
        fp_cost = float(false_positives.sum() * false_positive_cost)
        total_cost = fn_cost + fp_cost

        cost_curve.append((round(float(t), 2), round(float(total_cost), 2)))

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t

    if verbose:
        print("\n--- Cost-Based Threshold Optimization ---")
        print(f"False positive friction cost assumed: Rs {false_positive_cost} per case")
        print(f"Cost-optimal Block threshold: {best_threshold:.2f}")
        print(f"Estimated total cost at that threshold: Rs {best_cost:,.2f}")

    return round(float(best_threshold), 2), round(float(best_cost), 2), cost_curve



def get_precision_recall_curve(y_true, y_scores):
    """Thin wrapper so app.py doesn't need to import sklearn directly."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall, thresholds


'''# ---------------------------------------------------------------------
# 4. Population Stability Index (PSI) - quick drift check between splits
# ---------------------------------------------------------------------
def calculate_psi(expected, actual, buckets=10):
    """
    PSI compares the distribution of a feature between two samples
    (e.g. train vs test) to flag drift.
    Rule of thumb: <0.1 = stable, 0.1-0.25 = moderate drift, >0.25 = major drift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Handle case where all values are identical
    if np.all(expected == expected[0]) and np.all(actual == actual[0]):
        return 0.0

    percentiles = np.linspace(0, 100, buckets + 1)
    raw_breakpoints = np.percentile(expected, percentiles)
    breakpoints = np.unique(raw_breakpoints)

    # If unique breakpoints are fewer than 2, feature has too few distinct values
    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid divide-by-zero / log(0)
    eps = 1e-4
    expected_pct = np.where(expected_pct <= 0, eps, expected_pct)
    actual_pct = np.where(actual_pct <= 0, eps, actual_pct)

    # Normalize after epsilon adjustment
    expected_pct = expected_pct / np.sum(expected_pct)
    actual_pct = actual_pct / np.sum(actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def print_psi_report(X_train_raw, X_test_raw, columns=None):
    """Prints PSI for each given column, comparing train vs test distributions."""
    if columns is None:
        columns = X_train_raw.columns.tolist()

    print("\n--- Drift Check: PSI (Train vs Test) ---")
    print("(<0.1 = stable, 0.1-0.25 = moderate drift, >0.25 = major drift)")
    for col in columns:
        psi = calculate_psi(X_train_raw[col].values, X_test_raw[col].values)
        flag = "OK" if psi < 0.1 else ("WATCH" if psi < 0.25 else "DRIFT")
        print(f"{col:>10}: PSI = {psi:.4f}  [{flag}]")
'''