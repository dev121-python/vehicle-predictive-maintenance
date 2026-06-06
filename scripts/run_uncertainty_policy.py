"""Run Random Forest uncertainty + maintenance policy analysis.

Usage:
    python scripts/run_uncertainty_policy.py
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR, PROCESSED_DIR, DEFAULT_RUL_CAP, RANDOM_STATE, SELECTED_SENSORS
from src.data.loading import load_train
from src.features.build_features import add_rul_target, add_rolling_features, train_val_split_by_unit, make_feature_list, prepare_xy
from src.models.train_baselines import fit_models, transform_features, get_model_step
from src.uncertainty.random_forest import predict_mean_std, add_uncertainty_columns, uncertainty_error_bins
from src.evaluation.policy import build_policy_frame, compare_policies, sweep_maintenance_thresholds, sweep_cost_ratios


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_train(DATA_DIR, "FD001")
    df = add_rul_target(df, cap=DEFAULT_RUL_CAP)
    df = add_rolling_features(df, SELECTED_SENSORS, window_size=5)

    train_df, val_df = train_val_split_by_unit(df, random_state=RANDOM_STATE)
    features = make_feature_list(SELECTED_SENSORS, rolling_window=5)
    X_train, y_train, X_val, y_val = prepare_xy(train_df, val_df, features)

    fitted = fit_models(X_train, y_train, random_state=RANDOM_STATE)
    rf_pipe = fitted["Random Forest"]
    rf_model = get_model_step(rf_pipe)
    X_val_scaled = transform_features(rf_pipe, X_val)

    pred_mean, pred_std = predict_mean_std(rf_model, X_val_scaled)
    results_df = add_uncertainty_columns(val_df[["unit", "cycle", "RUL"]], pred_mean, pred_std)
    results_df.to_csv(PROCESSED_DIR / "fd001_uncertainty_results.csv", index=False)

    print("\nUncertainty bins")
    print(uncertainty_error_bins(results_df).to_string(index=False))

    policy_df = build_policy_frame(results_df, maintenance_threshold=20, k=1.64)
    policy_metrics, policy_costs = compare_policies(policy_df, cost_fp=1, cost_fn=5)
    print("\nPolicy metrics")
    print(policy_metrics.to_string(index=False))
    print("\nPolicy costs")
    print(policy_costs.to_string(index=False))

    threshold_df = sweep_maintenance_thresholds(results_df, cost_fp=1, cost_fn=5)
    cost_ratio_df = sweep_cost_ratios(results_df, fixed_threshold=20)
    threshold_df.to_csv(PROCESSED_DIR / "threshold_sensitivity_results.csv", index=False)
    cost_ratio_df.to_csv(PROCESSED_DIR / "cost_ratio_sensitivity_results.csv", index=False)
    print(f"\nSaved outputs to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
