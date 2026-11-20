
import joblib
import pandas as pd

from pathlib import Path

# Import the cleaning function
try:
    from .preprocessing import clean_data
except ImportError:
    from preprocessing import clean_data


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the trained model
MODEL_PATH = PROJECT_ROOT / "Model" / "best_model.pkl"


# Load the trained model
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Please run train.py first."
        )

    return joblib.load(MODEL_PATH)


# Load model once and reuse it
model = load_model()


def prepare_data(data):
    """
    Clean and prepare new transaction data
    using the same preprocessing function used in training.
    """

    df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()

    if "transaction_datetime" not in df.columns:
        raise ValueError(
            "transaction_datetime is required to generate time features."
        )

    # Apply the same cleaning and feature engineering
    df_clean = clean_data(df, drop_duplicates=False)

    # Get the exact feature columns used during training
    feature_columns = list(model.feature_names_in_)

    # Check for missing features
    missing_columns = [
        col for col in feature_columns
        if col not in df_clean.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    # Keep the same features and order used during training
    X = df_clean[feature_columns]

    return X


def predict(transaction, threshold=0.5):
    """
    Predict whether a single credit card transaction is fraudulent.

    Args:
        transaction: Dictionary containing transaction details.
        threshold: Fraud probability threshold.

    Returns:
        Dictionary containing prediction and fraud probability.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    X = prepare_data(transaction)

    # Get fraud probabilities
    probabilities = model.predict_proba(X)[0]

    # Find the probability corresponding to class 1 (Fraud)
    fraud_index = list(model.classes_).index(1)
    fraud_probability = float(probabilities[fraud_index])

    # Apply the selected threshold
    is_fraud = fraud_probability >= threshold

    return {
        "prediction": int(is_fraud),
        "label": "Fraud" if is_fraud else "Legitimate",
        "is_fraud": bool(is_fraud),
        "fraud_probability": round(fraud_probability, 4)
    }


def predict_batch(data, threshold=0.5):
    """
    Predict fraud for multiple transactions.

    Args:
        data: Pandas DataFrame containing transactions.
        threshold: Fraud probability threshold.

    Returns:
        DataFrame containing original data and predictions.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    X = prepare_data(data)

    # Get probabilities for all transactions
    fraud_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, fraud_index]

    # Add prediction results
    results = data.copy()

    results["fraud_probability"] = probabilities
    results["prediction"] = (
        results["fraud_probability"] >= threshold
    ).astype(int)

    results["label"] = results["prediction"].map({
        0: "Legitimate",
        1: "Fraud"
    })

    results["is_fraud"] = results["prediction"].astype(bool)

    return results
