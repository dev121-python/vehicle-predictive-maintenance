"""Maintenance-policy evaluation under asymmetric risk."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_policy_frame(
    df: pd.DataFrame,
    true_col: str = "RUL",
    pred_col: str = "pred_mean",
    uncertainty_col: str = "pred_std",
    maintenance_threshold: float = 20,
    k: float = 1.64,
) -> pd.DataFrame:
    """Add point and uncertainty-aware maintenance decisions.

    Decision convention: True = maintain now.
    """
    out = df.copy()
    out["actual_rul"] = out[true_col]
    out["predicted_rul"] = out[pred_col]
    out["uncertainty_std"] = out[uncertainty_col]
    out["conservative_rul"] = out["predicted_rul"] - k * out["uncertainty_std"]
    out["true_need_maintenance"] = out["actual_rul"] <= maintenance_threshold
    out["decision_point_policy"] = out["predicted_rul"] <= maintenance_threshold
    out["decision_uncertainty_policy"] = out["conservative_rul"] <= maintenance_threshold
    return out


def evaluate_policy(
    df: pd.DataFrame,
    decision_col: str,
    true_col: str = "true_need_maintenance",
) -> dict[str, float | int | str]:
    """Return confusion-style policy metrics."""
    decision = df[decision_col].astype(bool)
    truth = df[true_col].astype(bool)

    tp = int((decision & truth).sum())
    fp = int((decision & ~truth).sum())
    tn = int((~decision & ~truth).sum())
    fn = int((~decision & truth).sum())

    return {
        "policy": decision_col,
        "TP_maintained_risky": tp,
        "FP_unnecessary_maintenance": fp,
        "TN_correct_no_maintenance": tn,
        "FN_missed_risky_cases": fn,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
    }


def risk_cost(
    df: pd.DataFrame,
    decision_col: str,
    true_col: str = "true_need_maintenance",
    cost_fp: float = 1,
    cost_fn: float = 5,
) -> dict[str, float | int | str]:
    """Asymmetric policy cost: false negatives are more expensive than false positives."""
    metrics = evaluate_policy(df, decision_col, true_col=true_col)
    fp = metrics["FP_unnecessary_maintenance"]
    fn = metrics["FN_missed_risky_cases"]
    total_cost = cost_fp * fp + cost_fn * fn
    return {
        "policy": decision_col,
        "FP_unnecessary_maintenance": fp,
        "FN_missed_risky_cases": fn,
        "total_cost": float(total_cost),
        "average_cost_per_sample": float(total_cost / len(df)),
    }


def compare_policies(
    df: pd.DataFrame,
    decision_cols=("decision_point_policy", "decision_uncertainty_policy"),
    cost_fp: float = 1,
    cost_fn: float = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return metrics and cost tables for multiple policy decision columns."""
    metrics = pd.DataFrame([evaluate_policy(df, col) for col in decision_cols])
    costs = pd.DataFrame([risk_cost(df, col, cost_fp=cost_fp, cost_fn=cost_fn) for col in decision_cols])
    return metrics, costs


def sweep_maintenance_thresholds(
    df: pd.DataFrame,
    thresholds=range(10, 61, 5),
    k: float = 1.64,
    cost_fp: float = 1,
    cost_fn: float = 5,
) -> pd.DataFrame:
    """Evaluate point vs uncertainty policy across maintenance thresholds."""
    rows = []
    for threshold in thresholds:
        temp = build_policy_frame(df, maintenance_threshold=threshold, k=k)
        for policy_col, label in [
            ("decision_point_policy", "Point Prediction"),
            ("decision_uncertainty_policy", "Uncertainty-Aware"),
        ]:
            metrics = evaluate_policy(temp, policy_col)
            costs = risk_cost(temp, policy_col, cost_fp=cost_fp, cost_fn=cost_fn)
            rows.append({
                "threshold": threshold,
                "policy": label,
                "fn_rate": metrics["false_negative_rate"],
                "fp_rate": metrics["false_positive_rate"],
                "cost": costs["average_cost_per_sample"],
                "fn_count": metrics["FN_missed_risky_cases"],
                "fp_count": metrics["FP_unnecessary_maintenance"],
            })
    return pd.DataFrame(rows)


def sweep_cost_ratios(
    df: pd.DataFrame,
    fixed_threshold: float = 20,
    cost_ratios=(5, 10, 20, 50, 100),
    k: float = 1.64,
    cost_fp: float = 1,
) -> pd.DataFrame:
    """Evaluate point vs uncertainty policy as missed-failure cost increases."""
    rows = []
    base = build_policy_frame(df, maintenance_threshold=fixed_threshold, k=k)
    for cost_fn in cost_ratios:
        for policy_col, label in [
            ("decision_point_policy", "Point Prediction"),
            ("decision_uncertainty_policy", "Uncertainty-Aware"),
        ]:
            metrics = evaluate_policy(base, policy_col)
            costs = risk_cost(base, policy_col, cost_fp=cost_fp, cost_fn=cost_fn)
            rows.append({
                "failure_cost": cost_fn,
                "policy": label,
                "fn_rate": metrics["false_negative_rate"],
                "fp_rate": metrics["false_positive_rate"],
                "cost": costs["average_cost_per_sample"],
                "fn_count": metrics["FN_missed_risky_cases"],
                "fp_count": metrics["FP_unnecessary_maintenance"],
            })
    return pd.DataFrame(rows)



def sweep_k_values(
    actual_rul,
    predicted_rul,
    uncertainty_std,
    k_values=None,
    maintenance_threshold=20,
    cost_fp=1.0,
    cost_fn=5.0,
):
    """
    Sweep different k values for conservative RUL:

        conservative_rul = predicted_rul - k * uncertainty_std

    For each k, evaluate:
    - false positives: unnecessary maintenance
    - false negatives: missed risky cases
    - total asymmetric cost
    """

    import numpy as np
    import pandas as pd

    if k_values is None:
        k_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.64, 2.0]

    rows = []

    actual_rul = np.asarray(actual_rul)
    predicted_rul = np.asarray(predicted_rul)
    uncertainty_std = np.asarray(uncertainty_std)

    true_need_maintenance = actual_rul <= maintenance_threshold

    for k in k_values:
        conservative_rul = predicted_rul - k * uncertainty_std
        decision_maintenance = conservative_rul <= maintenance_threshold

        tp = np.sum(decision_maintenance & true_need_maintenance)
        fp = np.sum(decision_maintenance & ~true_need_maintenance)
        tn = np.sum(~decision_maintenance & ~true_need_maintenance)
        fn = np.sum(~decision_maintenance & true_need_maintenance)

        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        total_cost = cost_fp * fp + cost_fn * fn
        average_cost_per_sample = total_cost / len(actual_rul)

        rows.append(
            {
                "k": k,
                "TP_maintained_risky": tp,
                "FP_unnecessary_maintenance": fp,
                "TN_correct_no_maintenance": tn,
                "FN_missed_risky_cases": fn,
                "false_negative_rate": false_negative_rate,
                "false_positive_rate": false_positive_rate,
                "total_cost": total_cost,
                "average_cost_per_sample": average_cost_per_sample,
            }
        )

    return pd.DataFrame(rows)