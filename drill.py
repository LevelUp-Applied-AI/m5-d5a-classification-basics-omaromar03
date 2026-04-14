import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(df):
    """
    Split dataset into train/test sets using the target column 'churned'.

    Requirements:
    - 80% train / 20% test
    - random_state=42
    - stratify on target
    - verify split sizes
    - verify churn rate preserved within 2 percentage points

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if "churned" not in df.columns:
        raise ValueError("DataFrame must contain a 'churned' column.")

    X = df.drop(columns=["churned"]).copy()
    y = df["churned"].copy()

    if "customer_id" in X.columns:
        X = X.drop(columns=["customer_id"])

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    total_rows = len(df)
    expected_test_size = int(round(total_rows * 0.2))
    expected_train_size = total_rows - expected_test_size

    if len(X_train) != expected_train_size:
        raise ValueError("Training set size is incorrect.")
    if len(X_test) != expected_test_size:
        raise ValueError("Test set size is incorrect.")

    original_rate = y.mean()
    train_rate = y_train.mean()
    test_rate = y_test.mean()

    if abs(train_rate - original_rate) > 0.02:
        raise ValueError("Training set churn rate differs from original by more than 2 percentage points.")

    if abs(test_rate - original_rate) > 0.02:
        raise ValueError("Test set churn rate differs from original by more than 2 percentage points.")

    return X_train, X_test, y_train, y_test


def compute_classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.

    Returns:
        dict with keys:
        - accuracy
        - precision
        - recall
        - f1
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_cross_validation(X_train, y_train):
    """
    Run 5-fold stratified cross-validation using LogisticRegression.

    Model settings:
    - random_state=42
    - max_iter=1000
    - class_weight='balanced'

    Scoring:
    - accuracy

    Returns:
        dict with keys:
        - scores
        - mean
        - std
    """
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    X_train_encoded = pd.get_dummies(X_train, drop_first=True)

    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train_encoded,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores))
    }