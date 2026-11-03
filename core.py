"""
core.py - shared logic for the Loan Agent project.

Everything that touches data preparation or model training lives here so that
the Streamlit app and the training script can never drift apart.

The steps mirror the original notebook (Loan_Agent.ipynb) exactly:
  1. Load loan_data_new.csv
  2. Loan Status -> 1/0 (it is already numeric in the CSV)
  3. pd.get_dummies(..., drop_first=True) on the 5 categorical columns
  4. Drop 'Loan percentage' and 'Employee Experience'
  5. train_test_split(test_size=0.2, random_state=42)
  6. RandomForestClassifier(random_state=42)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_data_new.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "loan_rf_model.joblib"
ARTIFACTS_PATH = MODEL_DIR / "loan_artifacts.joblib"

# --------------------------------------------------------------------------- #
# Constants taken from the notebook (column names kept EXACTLY as in the CSV,
# including the 'Home Onwership' spelling)
# --------------------------------------------------------------------------- #
TARGET = "Loan Status"
CAT_COLS = ["Gender", "Education", "Home Onwership", "Loan Intent", "Previous Loan"]
DROP_FEATURES = ["Loan percentage", "Employee Experience"]
NUMERIC_INPUTS = [
    "Age",
    "Person Income",
    "Loan Amount",
    "Loan interest Rate",
    "Credit History",
    "Credit Score",
]
# Raw (pre-encoding) columns the user must supply for a prediction
RAW_INPUT_COLUMNS = [
    "Age", "Gender", "Education", "Person Income", "Home Onwership",
    "Loan Amount", "Loan Intent", "Loan interest Rate", "Credit History",
    "Credit Score", "Previous Loan",
]
# Class 1 = approved, class 0 = rejected (the notebook maps Approved -> 1, Rejected -> 0)
LABELS = {1: "Approved", 0: "Rejected"}

TEST_SIZE = 0.2
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data loading and the notebook's preprocessing, step for step
# --------------------------------------------------------------------------- #
def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def notebook_preprocessing(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Literal re-implementation of notebook cells 5-7. Returns (X_clean, y)."""
    df = raw.copy()
    if df[TARGET].dtype == "object":  # same guard as the notebook
        df[TARGET] = df[TARGET].map(
            {"Y": 1, "N": 0, "Yes": 1, "No": 0, "Approved": 1, "Rejected": 0}
        )
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    y = df[TARGET]
    X = df.drop(TARGET, axis=1)
    X_clean = X.drop(DROP_FEATURES, axis=1)
    return X_clean, y


def get_categories(raw: pd.DataFrame) -> dict[str, list[str]]:
    """Sorted category levels per categorical column (the order get_dummies uses)."""
    return {c: sorted(raw[c].unique().tolist()) for c in CAT_COLS}


def encode_applicants(
    df_raw: pd.DataFrame,
    categories: dict[str, list[str]],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Encode one or many applicants exactly as the training data was encoded.

    A plain pd.get_dummies on a single row would produce a different column set
    (and drop_first would drop the only level), so we fix the category levels to
    the ones seen in training, apply the same get_dummies(drop_first=True), then
    drop the same two columns and re-order to the training feature order.
    """
    df = df_raw[RAW_INPUT_COLUMNS].copy()
    for col in CAT_COLS:
        df[col] = pd.Categorical(df[col], categories=categories[col])
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    return df.reindex(columns=feature_columns, fill_value=False)


def original_feature_name(encoded_name: str) -> str:
    """Map a one-hot column ('Education_Master') back to its source column."""
    for col in CAT_COLS:
        if encoded_name.startswith(col + "_"):
            return col
    return encoded_name


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_and_save(verbose: bool = True) -> dict:
    raw = load_raw_data()
    X, y = notebook_preprocessing(raw)
    feature_columns = X.columns.tolist()
    categories = get_categories(raw)

    # Consistency check: the app's encoder must reproduce the notebook's X exactly.
    app_X = encode_applicants(raw, categories, feature_columns)
    pd.testing.assert_frame_equal(
        app_X.astype(int).reset_index(drop=True),
        X.astype(int).reset_index(drop=True),
        check_dtype=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    imp = pd.DataFrame(
        {"Feature": feature_columns, "Importance": model.feature_importances_}
    )
    # Light-weight on purpose (single process, 2,000-row sample) so the app can
    # retrain on small free servers without running out of memory.
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
    perm = permutation_importance(
        model, sample, y_test.loc[sample.index],
        n_repeats=3, random_state=RANDOM_STATE, n_jobs=1,
    )
    imp["Permutation Importance"] = perm.importances_mean
    imp["Permutation Std"] = perm.importances_std
    imp["Source Column"] = imp["Feature"].map(original_feature_name)
    imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)

    numeric_stats = {
        c: {
            "min": float(raw[c].min()),
            "max": float(raw[c].max()),
            "median": float(raw[c].median()),
            "q01": float(raw[c].quantile(0.01)),
            "q99": float(raw[c].quantile(0.99)),
        }
        for c in NUMERIC_INPUTS
    }
    modes = {c: raw[c].mode().iloc[0] for c in CAT_COLS}

    artifacts = {
        "feature_columns": feature_columns,
        "categories": categories,
        "numeric_stats": numeric_stats,
        "modes": modes,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_approved": precision_score(y_test, y_pred, pos_label=1),
            "recall_approved": recall_score(y_test, y_pred, pos_label=1),
            "f1_approved": f1_score(y_test, y_pred, pos_label=1),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "majority_baseline": float(max(y_test.mean(), 1 - y_test.mean())),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "report_text": classification_report(y_test, y_pred),
        "roc": {"fpr": fpr, "tpr": tpr},
        "y_test": y_test.to_numpy(),
        "y_proba": y_proba,
        "importance": imp,
        "sklearn_version": sklearn.__version__,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(artifacts, ARTIFACTS_PATH, compress=3)
    if verbose:
        print(f"Train/test sizes: {X_train.shape} / {X_test.shape}")
        print(f"Accuracy: {artifacts['metrics']['accuracy']:.6f}")
        print(artifacts["report_text"])
        print(f"Saved model     -> {MODEL_PATH}")
        print(f"Saved artifacts -> {ARTIFACTS_PATH}")
    return artifacts


def load_model_and_artifacts():
    """Load saved files; (re)train automatically if missing or incompatible."""
    try:
        model = joblib.load(MODEL_PATH)
        artifacts = joblib.load(ARTIFACTS_PATH)
        if artifacts.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("scikit-learn version changed since training")
        return model, artifacts
    except Exception:
        train_and_save(verbose=False)
        return joblib.load(MODEL_PATH), joblib.load(ARTIFACTS_PATH)


def predict_applicant(model, artifacts: dict, applicant: dict) -> dict:
    """Run one applicant (dict of raw inputs) through the training pipeline."""
    row = pd.DataFrame([applicant])
    X = encode_applicants(row, artifacts["categories"], artifacts["feature_columns"])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    p_approved = float(proba[classes.index(1)])
    p_rejected = float(proba[classes.index(0)])
    label = LABELS[1] if p_approved >= p_rejected else LABELS[0]
    return {"label": label, "p_approved": p_approved, "p_rejected": p_rejected}
