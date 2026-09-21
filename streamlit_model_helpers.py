"""
Utility functions for loading TRAUMALOS deployment bundles in Streamlit.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def load_traumalos_bundle(path):
    return joblib.load(Path(path))


def predict_with_ci(bundle, patient_data, ci_level=None):
    if isinstance(patient_data, dict):
        X_new = pd.DataFrame([patient_data])
    else:
        X_new = pd.DataFrame(patient_data).copy()

    required = bundle["features"]
    missing = [c for c in required if c not in X_new.columns]
    if missing:
        raise KeyError(f"Missing variables: {missing}")

    X_new = X_new[required].copy()

    for col in bundle["categorical_cols"]:
        X_new[col] = X_new[col].astype("object")

    point = bundle["final_calibrated_model"].predict_proba(X_new)[:, 1]

    ci = bundle["ci_level"] if ci_level is None else ci_level
    alpha = 1.0 - ci

    boot = np.column_stack([
        model.predict_proba(X_new)[:, 1]
        for model in bundle["bootstrap_models"]
    ])

    lower = np.quantile(boot, alpha / 2.0, axis=1)
    upper = np.quantile(boot, 1.0 - alpha / 2.0, axis=1)

    return pd.DataFrame({
        "predicted_probability": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "predicted_class": (
            point >= bundle["decision_threshold"]
        ).astype(int),
    })
