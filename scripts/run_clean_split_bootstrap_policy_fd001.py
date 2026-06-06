import numpy as np
import pandas as pd

from src.config import DATA_DIR, PROCESSED_DIR, DEFAULT_RUL_CAP, SELECTED_SENSORS
from src.data.loading import load_train
from src.features.build_features import (
    add_rul_target,
    add_rolling_features,
    make_feature_list,
    prepare_xy,
    train_cal_eval_split_by_unit,
)
from src.models.train_baselines import fit_models, get_model_step, transform_features
from src.uncertainty.random_forest import predict_mean_std, uncertainty_error_bins
from src.evaluation.policy import sweep_k_values


DATASETS = ["FD001", "FD002", "FD003", "FD004"]

K_VALUES_FINE = np.round(np.arange(0.0, 3.01, 0.05), 2)

COST_FN_VALUES = [2, 5, 10, 20, 50, 100]

MAINTENANCE_THRESHOLD = 20
COST_FP = 1.0
ROLLING_WINDOW = 5


def make_uncertainty_result_frame(df_split, pred_mean, pred_std):
    """
    Attach predictions, uncertainty, and error columns to a dataframe.
    """

    result_df = df_split.copy()

    result_df["predicted_rul"] = pred_mean
    result_df["uncertainty_std"] = pred_std
    result_df["error"] = result_df["RUL"] - result_df["predicted_rul"]
    result_df["abs_error"] = result_df["error"].abs()
    result_df["overestimate"] = result_df["predicted_rul"] > result_df["RUL"]

    return result_df


def choose_best_k_on_calibration(cal_df, cost_fn):
    """
    Use calibration data to choose the best k.
    """

    cal_sweep_df = sweep_k_values(
        actual_rul=cal_df["RUL"],
        predicted_rul=cal_df["predicted_rul"],
        uncertainty_std=cal_df["uncertainty_std"],
        k_values=K_VALUES_FINE,
        maintenance_threshold=MAINTENANCE_THRESHOLD,
        cost_fp=COST_FP,
        cost_fn=cost_fn,
    )

    best_row = cal_sweep_df.loc[cal_sweep_df["total_cost"].idxmin()].copy()

    return best_row, cal_sweep_df


def evaluate_fixed_k(eval_df, k, cost_fn):
    """
    Evaluate one fixed k on final evaluation data.
    This is the clean final result.
    """

    eval_result_df = sweep_k_values(
        actual_rul=eval_df["RUL"],
        predicted_rul=eval_df["predicted_rul"],
        uncertainty_std=eval_df["uncertainty_std"],
        k_values=[k],
        maintenance_threshold=MAINTENANCE_THRESHOLD,
        cost_fp=COST_FP,
        cost_fn=cost_fn,
    )

    return eval_result_df.iloc[0].copy()


def run_one_dataset(dataset_name):
    print("\n==============================")
    print(f"Running clean split experiment: {dataset_name}")
    print("==============================")

    # 1. Load data
    df = load_train(DATA_DIR, dataset_name)

    # 2. Add capped RUL target
    df = add_rul_target(df, cap=DEFAULT_RUL_CAP)

    # 3. Select sensors
    selected_sensors = [sensor for sensor in SELECTED_SENSORS if sensor in df.columns]

    # 4. Add rolling features
    df_features = add_rolling_features(
        df=df,
        sensor_cols=selected_sensors,
        window_size=ROLLING_WINDOW,
    )

    # 5. Build feature columns
    operation_cols = ["op_1", "op_2", "op_3"]

    feature_cols = operation_cols + make_feature_list(
        base_sensors=selected_sensors,
        rolling_window=ROLLING_WINDOW,
    )

    feature_cols = [col for col in feature_cols if col in df_features.columns]

    print(f"Selected sensors: {selected_sensors}")
    print(f"Number of features: {len(feature_cols)}")

    # 6. Clean engine-wise split
    train_df, cal_df, eval_df = train_cal_eval_split_by_unit(
        df_features,
        train_size=0.60,
        cal_size=0.20,
        eval_size=0.20,
        unit_col="unit",
        random_state=42,
    )

    print(f"Train engines: {train_df['unit'].nunique()}, rows: {len(train_df)}")
    print(f"Calibration engines: {cal_df['unit'].nunique()}, rows: {len(cal_df)}")
    print(f"Evaluation engines: {eval_df['unit'].nunique()}, rows: {len(eval_df)}")

    # 7. Prepare train/cal/eval matrices
    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]

    X_cal = cal_df[feature_cols]
    y_cal = cal_df["RUL"]

    X_eval = eval_df[feature_cols]
    y_eval = eval_df["RUL"]

    # 8. Train baseline models on train only
    fitted_models = fit_models(X_train, y_train)

    # 9. Use Random Forest for uncertainty
    rf_pipe = fitted_models["Random Forest"]
    rf_model = get_model_step(rf_pipe)

    # 10. Transform cal/eval features through the same pipeline scaler
    X_cal_scaled = transform_features(rf_pipe, X_cal)
    X_eval_scaled = transform_features(rf_pipe, X_eval)

    # 11. RF tree uncertainty predictions
    cal_pred_mean, cal_pred_std = predict_mean_std(rf_model, X_cal_scaled)
    eval_pred_mean, eval_pred_std = predict_mean_std(rf_model, X_eval_scaled)

    # 12. Build result frames
    cal_result_df = make_uncertainty_result_frame(
        cal_df,
        pred_mean=cal_pred_mean,
        pred_std=cal_pred_std,
    )

    eval_result_df = make_uncertainty_result_frame(
        eval_df,
        pred_mean=eval_pred_mean,
        pred_std=eval_pred_std,
    )

    # 13. Save uncertainty result frames
    cal_result_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_clean_cal_uncertainty_results.csv",
        index=False,
    )

    eval_result_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_clean_eval_uncertainty_results.csv",
        index=False,
    )

    # 14. Uncertainty bins on final eval set
    eval_bins_input = eval_result_df.rename(
        columns={
            "predicted_rul": "pred_mean",
            "uncertainty_std": "pred_std",
        }
    )

    eval_uncertainty_bins = uncertainty_error_bins(eval_bins_input)

    eval_uncertainty_bins.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_clean_eval_uncertainty_bins.csv",
        index=False,
    )

    # 15. Choose k on calibration, report on evaluation
    all_cal_sweeps = []
    clean_summary_rows = []

    for cost_fn in COST_FN_VALUES:
        best_cal_row, cal_sweep_df = choose_best_k_on_calibration(
            cal_result_df,
            cost_fn=cost_fn,
            )
        best_k = best_cal_row["k"]

    # Point policy baseline: k = 0 means no uncertainty adjustment
        point_policy_row = evaluate_fixed_k(
            eval_result_df,
            k=0.0,
            cost_fn=cost_fn,
    )

    # Uncertainty-aware policy: k chosen on calibration
        eval_fixed_row = evaluate_fixed_k(
        eval_result_df,
        k=best_k,
        cost_fn=cost_fn,
    )

        cal_sweep_df["dataset"] = dataset_name
        cal_sweep_df["cost_fn"] = cost_fn
        cal_sweep_df["cost_fp"] = COST_FP
        cal_sweep_df["split_used_for_k_selection"] = "calibration"

        all_cal_sweeps.append(cal_sweep_df)

        point_cost = point_policy_row["total_cost"]
        uncertainty_cost = eval_fixed_row["total_cost"]

        cost_reduction_percent = (
        100 * (point_cost - uncertainty_cost) / point_cost
        if point_cost > 0
        else 0.0
    )

        point_fn = point_policy_row["FN_missed_risky_cases"]
        uncertainty_fn = eval_fixed_row["FN_missed_risky_cases"]

        fn_reduction_percent = (
        100 * (point_fn - uncertainty_fn) / point_fn
        if point_fn > 0
        else 0.0
    )

        clean_summary_rows.append(
        {
            "dataset": dataset_name,
            "cost_fn": cost_fn,
            "cost_fp": COST_FP,
            "best_k_chosen_on_calibration": best_k,

            "point_eval_FP": point_policy_row["FP_unnecessary_maintenance"],
            "point_eval_FN": point_policy_row["FN_missed_risky_cases"],
            "point_eval_false_negative_rate": point_policy_row["false_negative_rate"],
            "point_eval_false_positive_rate": point_policy_row["false_positive_rate"],
            "point_eval_total_cost": point_policy_row["total_cost"],

            "uncertainty_eval_FP": eval_fixed_row["FP_unnecessary_maintenance"],
            "uncertainty_eval_FN": eval_fixed_row["FN_missed_risky_cases"],
            "uncertainty_eval_false_negative_rate": eval_fixed_row["false_negative_rate"],
            "uncertainty_eval_false_positive_rate": eval_fixed_row["false_positive_rate"],
            "uncertainty_eval_total_cost": eval_fixed_row["total_cost"],

            "cost_reduction_percent": cost_reduction_percent,
            "fn_reduction_percent": fn_reduction_percent,

            "cal_FP": best_cal_row["FP_unnecessary_maintenance"],
            "cal_FN": best_cal_row["FN_missed_risky_cases"],
            "cal_total_cost": best_cal_row["total_cost"],
        }
    )
    all_cal_sweeps_df = pd.concat(all_cal_sweeps, ignore_index=True)
    clean_summary_df = pd.DataFrame(clean_summary_rows)

    # 16. Save outputs
    all_cal_sweeps_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_clean_calibration_k_sweeps.csv",
        index=False,
    )

    clean_summary_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_clean_cal_to_eval_summary.csv",
        index=False,
    )

    print("\nClean calibration-to-evaluation summary:")
    print(clean_summary_df)

    return clean_summary_df


def main():
    all_clean_results = []

    for dataset_name in DATASETS:
        dataset_result = run_one_dataset(dataset_name)
        all_clean_results.append(dataset_result)

    all_clean_results_df = pd.concat(all_clean_results, ignore_index=True)

    all_clean_results_df.to_csv(
        PROCESSED_DIR / "all_datasets_clean_cal_to_eval_summary.csv",
        index=False,
    )

    print("\n\n==============================")
    print("FINAL CLEAN CALIBRATION → EVALUATION RESULTS")
    print("==============================")
    print(all_clean_results_df)

    print("\nSaved to:")
    print(PROCESSED_DIR / "all_datasets_clean_cal_to_eval_summary.csv")


if __name__ == "__main__":
    main()