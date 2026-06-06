"""Random Forest uncertainty helpers.

A Random Forest gives many tree predictions. The mean is the point prediction;
the standard deviation across trees is a simple epistemic-uncertainty proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def tree_prediction_matrix(model: RandomForestRegressor, X) -> np.ndarray:
    """Return shape (n_trees, n_samples) tree-level predictions."""
    if not hasattr(model, "estimators_"):
        raise ValueError("The RandomForestRegressor must be fitted before calling this function.")
    return np.vstack([tree.predict(X) for tree in model.estimators_])


def predict_mean_std(model: RandomForestRegressor, X) -> tuple[np.ndarray, np.ndarray]:
    """Return prediction mean and std across forest trees."""
    tree_preds = tree_prediction_matrix(model, X)
    return tree_preds.mean(axis=0), tree_preds.std(axis=0)


def add_uncertainty_columns(df: pd.DataFrame, pred_mean, pred_std, target_col: str = "RUL") -> pd.DataFrame:
    """Attach prediction, uncertainty, and error columns to a copy of df."""
    out = df.copy().reset_index(drop=True)
    out["pred_mean"] = pred_mean
    out["pred_std"] = pred_std
    if target_col in out.columns:
        out["error"] = out[target_col] - out["pred_mean"]
        out["abs_error"] = out["error"].abs()
        out["overestimate"] = out["error"] < 0
    return out


def uncertainty_error_bins(
    df: pd.DataFrame,
    uncertainty_col: str = "pred_std",
    error_col: str = "abs_error",
    bins: int = 6,
) -> pd.DataFrame:
    """Bucket rows by uncertainty and summarize error/risk behavior."""
    out = df.copy()
    out["uncertainty_bin"] = pd.qcut(out[uncertainty_col], q=bins, duplicates="drop")
    agg = out.groupby("uncertainty_bin", observed=True).agg(
        count=(uncertainty_col, "size"),
        mean_uncertainty=(uncertainty_col, "mean"),
        mean_abs_error=(error_col, "mean"),
        median_abs_error=(error_col, "median"),
        dangerous_rate=("overestimate", "mean"),
        mean_overestimate=("error", lambda x: (-x[x < 0]).mean() if (x < 0).any() else 0),
    )
    return agg.reset_index()


def conservative_rul(pred_mean, pred_std, k: float = 1.64):
    """Lower confidence-style conservative RUL estimate.

    k=1.64 is roughly a one-sided 90% normal-approximation factor.
    """
    return np.asarray(pred_mean) - k * np.asarray(pred_std)
