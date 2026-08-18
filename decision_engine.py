# decision_engine.py

from utils import load_thresholds

DEFAULT_BLOCK_THRESHOLD = 0.65
REVIEW_RANGE = 0.05  # 5% range below block threshold for manual review

# Initial load from disk
BLOCK_THRESHOLD, REVIEW_THRESHOLD = load_thresholds(
    default_block=DEFAULT_BLOCK_THRESHOLD,
    default_review=max(0.0, DEFAULT_BLOCK_THRESHOLD - REVIEW_RANGE),
)


def get_thresholds():
    """Fetches the latest thresholds saved on disk."""
    return load_thresholds(
        default_block=DEFAULT_BLOCK_THRESHOLD,
        default_review=max(0.0, DEFAULT_BLOCK_THRESHOLD - REVIEW_RANGE),
    )


def get_risk_level(fraud_probability, block_threshold=None, review_threshold=None):
    if block_threshold is None:
        block_threshold = BLOCK_THRESHOLD
    if review_threshold is None:
        review_threshold = REVIEW_THRESHOLD

    if fraud_probability >= block_threshold:
        return "High Risk"
    elif fraud_probability >= review_threshold:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_recommended_action(fraud_probability, block_threshold=None, review_threshold=None):
    if block_threshold is None:
        block_threshold = BLOCK_THRESHOLD
    if review_threshold is None:
        review_threshold = REVIEW_THRESHOLD

    if fraud_probability >= block_threshold:
        return "Block Transaction"
    elif fraud_probability >= review_threshold:
        return "Send for Manual Review"
    else:
        return "Allow"


def get_expected_loss(fraud_probability, amount):
    return round(float(fraud_probability * amount), 2)


def evaluate_transaction(fraud_probability, amount, block_threshold=None, review_threshold=None):
    """
    Evaluates a transaction given its estimated fraud probability and amount.
    Returns fraud probability %, risk level, expected monetary loss, and action.
    """
    return {
        "fraud_probability": round(float(fraud_probability * 100), 2),
        "risk_level": get_risk_level(fraud_probability, block_threshold, review_threshold),
        "expected_loss": get_expected_loss(fraud_probability, amount),
        "recommended_action": get_recommended_action(fraud_probability, block_threshold, review_threshold),
    }

