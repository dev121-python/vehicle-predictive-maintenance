import numpy as np
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR, PROCESSED_DIR, DEFAULT_RUL_CAP, SELECTED_SENSORS
from src.data.loading import load_train
from src.features.build_features import (
    add_rul_target,
    add_rolling_features,
    make_feature_list,
    train_cal_eval_split_by_unit,
)
from src.uncertainty.quantile import (
    fit_quantile_models,
    predict_quantile_interval,
)
from src.uncertainty.random_forest import uncertainty_error_bins
from src.evaluation.policy import sweep_k_values


DATASETS = ["FD001", "FD002", "FD003", "FD004"]

METHOD_NAME = "quantile_gbr"

K_VALUES_FINE = np.round(np.arange(0.0, 6.01, 0.05), 2)
COST_FN_VALUES = [2, 5, 10, 20, 50, 100]

MAINTENANCE_THRESHOLD = 20
COST_FP = 1.0
ROLLING_WINDOW = 5

LOWER_ALPHA = 0.10
MEDIAN_ALPHA = 0.50
UPPER_ALPHA = 0.90


def make_quantile_result_frame(
    df_split,
    pred_median,
    uncertainty_proxy,
    pred_lower,
    pred_upper,
    interval_width,
):
    result_df = df_split.copy()

    result_df["predicted_rul"] = pred_median
    result_df["uncertainty_std"] = uncertainty_proxy

    result_df["pred_lower"] = pred_lower
    result_df["pred_upper"] = pred_upper
    result_df["interval_width"] = interval_width

    result_df["error"] = result_df["RUL"] - result_df["predicted_rul"]
    result_df["abs_error"] = result_df["error"].abs()
    result_df["overestimate"] = result_df["predicted_rul"] > result_df["RUL"]

    result_df["covered_by_interval"] = (
        (result_df["RUL"] >= result_df["pred_lower"])
        & (result_df["RUL"] <= result_df["pred_upper"])
    )

    return result_df


def choose_best_k_on_calibration(cal_df, cost_fn):
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
    print(f"Running {METHOD_NAME} on {dataset_name}")
    print("==============================")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

    # 5. Build feature list
    operation_cols = ["op_1", "op_2", "op_3"]

    feature_cols = operation_cols + make_feature_list(
        base_sensors=selected_sensors,
        rolling_window=ROLLING_WINDOW,
    )

    feature_cols = [col for col in feature_cols if col in df_features.columns]

    print(f"Selected sensors: {selected_sensors}")
    print(f"Number of features: {len(feature_cols)}")

    # 6. Clean split
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

    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]

    X_cal = cal_df[feature_cols]
    X_eval = eval_df[feature_cols]

    # 7. Train quantile models
    quantile_models = fit_quantile_models(
        X_train=X_train,
        y_train=y_train,
        lower_alpha=LOWER_ALPHA,
        median_alpha=MEDIAN_ALPHA,
        upper_alpha=UPPER_ALPHA,
        random_state=42,
    )

    # 8. Predict intervals
    cal_pred_median, cal_uncertainty, cal_lower, cal_upper, cal_width = predict_quantile_interval(
        quantile_models,
        X_cal,
    )

    eval_pred_median, eval_uncertainty, eval_lower, eval_upper, eval_width = predict_quantile_interval(
        quantile_models,
        X_eval,
    )

    # 9. Build result frames
    cal_result_df = make_quantile_result_frame(
        cal_df,
        pred_median=cal_pred_median,
        uncertainty_proxy=cal_uncertainty,
        pred_lower=cal_lower,
        pred_upper=cal_upper,
        interval_width=cal_width,
    )

    eval_result_df = make_quantile_result_frame(
        eval_df,
        pred_median=eval_pred_median,
        uncertainty_proxy=eval_uncertainty,
        pred_lower=eval_lower,
        pred_upper=eval_upper,
        interval_width=eval_width,
    )

    # 10. Save uncertainty results
    cal_result_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_{METHOD_NAME}_clean_cal_uncertainty_results.csv",
        index=False,
    )

    eval_result_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_{METHOD_NAME}_clean_eval_uncertainty_results.csv",
        index=False,
    )

    # 11. Save uncertainty bins
    eval_bins_input = eval_result_df.rename(
        columns={
            "predicted_rul": "pred_mean",
            "uncertainty_std": "pred_std",
        }
    )

    eval_uncertainty_bins = uncertainty_error_bins(eval_bins_input)

    eval_uncertainty_bins.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_{METHOD_NAME}_clean_eval_uncertainty_bins.csv",
        index=False,
    )

    # 12. Save interval quality metrics
    interval_quality = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "method": METHOD_NAME,
                "coverage": eval_result_df["covered_by_interval"].mean(),
                "mean_interval_width": eval_result_df["interval_width"].mean(),
                "median_interval_width": eval_result_df["interval_width"].median(),
                "mean_abs_error": eval_result_df["abs_error"].mean(),
            }
        ]
    )

    interval_quality.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_{METHOD_NAME}_interval_quality.csv",
        index=False,
    )

    # 13. Choose k on calibration and evaluate on evaluation
    all_cal_sweeps = []
    clean_summary_rows = []

    for cost_fn in COST_FN_VALUES:
        best_cal_row, cal_sweep_df = choose_best_k_on_calibration(
            cal_result_df,
            cost_fn=cost_fn,
        )

        best_k = best_cal_row["k"]

        point_policy_row = evaluate_fixed_k(
            eval_result_df,
            k=0.0,
            cost_fn=cost_fn,
        )

        eval_fixed_row = evaluate_fixed_k(
            eval_result_df,
            k=best_k,
            cost_fn=cost_fn,
        )

        cal_sweep_df["dataset"] = dataset_name
        cal_sweep_df["method"] = METHOD_NAME
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
                "method": METHOD_NAME,
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

                "eval_interval_coverage": eval_result_df["covered_by_interval"].mean(),
                "eval_mean_interval_width": eval_result_df["interval_width"].mean(),
            }
        )

    all_cal_sweeps_df = pd.concat(all_cal_sweeps, ignore_index=True)
    clean_summary_df = pd.DataFrame(clean_summary_rows)

    # 14. Save per-dataset outputs
    all_cal_sweeps_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_{METHOD_NAME}_clean_calibration_k_sweeps.csv",
        index=False,
    )

    clean_summary_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_{METHOD_NAME}_clean_cal_to_eval_summary.csv",
        index=False,
    )

    print("\nClean quantile calibration-to-evaluation summary:")
    print(clean_summary_df)

    return clean_summary_df, interval_quality


def main():
    all_results = []
    all_interval_quality = []

    for dataset_name in DATASETS:
        dataset_result, dataset_interval_quality = run_one_dataset(dataset_name)
        all_results.append(dataset_result)
        all_interval_quality.append(dataset_interval_quality)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_interval_quality_df = pd.concat(all_interval_quality, ignore_index=True)

    all_results_df.to_csv(
        PROCESSED_DIR / f"all_datasets_{METHOD_NAME}_clean_cal_to_eval_summary.csv",
        index=False,
    )

    all_interval_quality_df.to_csv(
        PROCESSED_DIR / f"all_datasets_{METHOD_NAME}_interval_quality.csv",
        index=False,
    )

    print("\n\n==============================")
    print(f"FINAL ALL-DATASET RESULTS: {METHOD_NAME}")
    print("==============================")
    print(all_results_df)

    print("\nInterval quality:")
    print(all_interval_quality_df)

    print("\nSaved to:")
    print(PROCESSED_DIR / f"all_datasets_{METHOD_NAME}_clean_cal_to_eval_summary.csv")


if __name__ == "__main__":
    main()