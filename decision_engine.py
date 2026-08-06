# decision_engine.py

# Thresholds
BLOCK_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.40


def get_risk_level(fraud_probability):
    # Turn a probability into a human-readable risk level.
 
    if fraud_probability >= BLOCK_THRESHOLD:
        return "High Risk"
    elif fraud_probability >= REVIEW_THRESHOLD:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_recommended_action(fraud_probability):
    # Turn a probability into a recommended action.
    
    if fraud_probability >= BLOCK_THRESHOLD:
        return "Block Immediately"
    elif fraud_probability >= REVIEW_THRESHOLD:
        return "Send for Manual Review"
    else:
        return "Allow"


def get_expected_loss(fraud_probability, amount):
    
    # This is basic expected value from probability theory:
    #    E[Loss] = P(Fraud) * Amount

    return round(fraud_probability * amount, 2)


def evaluate_transaction(fraud_probability, amount):
    # Single function the Streamlit app will call.
    
    return {
        "fraud_probability": round(fraud_probability * 100, 2),  # as %
        "risk_level": get_risk_level(fraud_probability),
        "expected_loss": get_expected_loss(fraud_probability, amount),
        "recommended_action": get_recommended_action(fraud_probability),
    }


"""if __name__ == "__main__":
    # quick manual test
    examples = [
        (0.97, 100000),
        (0.55, 5000),
        (0.10, 500),
    ]
    for prob, amt in examples:
        result = evaluate_transaction(prob, amt)
        print(f"Probability={prob}, Amount={amt} -> {result}")"""
