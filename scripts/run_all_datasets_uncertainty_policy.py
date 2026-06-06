import numpy as np
import pandas as pd

from src.config import DATA_DIR, PROCESSED_DIR, DEFAULT_RUL_CAP, SELECTED_SENSORS
from src.data.loading import load_train
from src.features.build_features import (
    add_rul_target,
    train_val_split_by_unit,
    add_rolling_features,
    make_feature_list,
    prepare_xy,
)
from src.models.train_baselines import fit_models, get_model_step, transform_features
from src.uncertainty.random_forest import predict_mean_std, uncertainty_error_bins
from src.evaluation.policy import sweep_k_values


DATASETS = ["FD001", "FD002", "FD003", "FD004"]

K_VALUES_FINE = np.round(np.arange(0.0, 2.01, 0.05), 2)

COST_FN_VALUES = [2, 5, 10, 20, 50, 100]

MAINTENANCE_THRESHOLD = 20
COST_FP = 1.0
ROLLING_WINDOW = 5


def run_one_dataset(dataset_name: str):
    print(f"\n==============================")
    print(f"Running dataset: {dataset_name}")
    print(f"==============================")

    # 1. Load training data
    df = load_train(DATA_DIR, dataset_name)

    # 2. Add RUL target
    df = add_rul_target(df, cap=DEFAULT_RUL_CAP)

    # 3. Select sensors that exist in this dataset
    selected_sensors = [sensor for sensor in SELECTED_SENSORS if sensor in df.columns]

    print(f"Selected sensors: {selected_sensors}")

    # 4. Add rolling features
    df_features = add_rolling_features(
        df=df,
        sensor_cols=selected_sensors,
        window_size=ROLLING_WINDOW,
    )

    # 5. Build feature list
    # Include operational settings because FD002 and FD004 have multiple operating conditions.
    operation_cols = ["op_1", "op_2", "op_3"]

    feature_cols = operation_cols + make_feature_list(
        base_sensors=selected_sensors,
        rolling_window=ROLLING_WINDOW,
    )

    # Keep only columns that actually exist
    feature_cols = [col for col in feature_cols if col in df_features.columns]

    print(f"Number of features: {len(feature_cols)}")

    # 6. Engine-wise train/validation split
    train_df, val_df = train_val_split_by_unit(df_features)

    # 7. Prepare X and y
    X_train, y_train, X_val, y_val = prepare_xy(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
    )

    print(f"Train rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")

    # 8. Train baseline models
    fitted_models = fit_models(X_train, y_train)

    # 9. Use Random Forest for uncertainty
    rf_pipe = fitted_models["Random Forest"]

    rf_model = get_model_step(rf_pipe)
    X_val_scaled = transform_features(rf_pipe, X_val)

    pred_mean, pred_std = predict_mean_std(rf_model, X_val_scaled)

    # 10. Save uncertainty bin analysis
    uncertainty_df = val_df.copy()
    uncertainty_df["predicted_rul"] = pred_mean
    uncertainty_df["uncertainty_std"] = pred_std
    uncertainty_df["error"] = uncertainty_df["RUL"] - uncertainty_df["predicted_rul"]
    uncertainty_df["abs_error"] = uncertainty_df["error"].abs()
    uncertainty_df["overestimate"] = uncertainty_df["predicted_rul"] > uncertainty_df["RUL"]

    uncertainty_bins = uncertainty_error_bins(
        uncertainty_df.rename(
            columns={
                "predicted_rul": "pred_mean",
                "uncertainty_std": "pred_std",
            }
        )
    )

    uncertainty_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_uncertainty_results.csv",
        index=False,
    )

    uncertainty_bins.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_uncertainty_bins.csv",
        index=False,
    )

    # 11. Fine k-sweep for multiple missed-failure costs
    dataset_sweep_results = []

    for cost_fn in COST_FN_VALUES:
        temp_df = sweep_k_values(
            actual_rul=y_val,
            predicted_rul=pred_mean,
            uncertainty_std=pred_std,
            k_values=K_VALUES_FINE,
            maintenance_threshold=MAINTENANCE_THRESHOLD,
            cost_fp=COST_FP,
            cost_fn=cost_fn,
        )

        temp_df["dataset"] = dataset_name
        temp_df["cost_fp"] = COST_FP
        temp_df["cost_fn"] = cost_fn
        temp_df["maintenance_threshold"] = MAINTENANCE_THRESHOLD

        dataset_sweep_results.append(temp_df)

    dataset_sweep_df = pd.concat(dataset_sweep_results, ignore_index=True)

    # 12. Best k for each cost_fn in this dataset
    best_k_df = (
        dataset_sweep_df
        .loc[dataset_sweep_df.groupby("cost_fn")["total_cost"].idxmin()]
        .sort_values("cost_fn")
        .reset_index(drop=True)
    )

    # 13. Save per-dataset results
    dataset_sweep_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_fine_k_cost_sweep.csv",
        index=False,
    )

    best_k_df.to_csv(
        PROCESSED_DIR / f"{dataset_name.lower()}_best_k_by_cost.csv",
        index=False,
    )

    print("\nBest k by missed-failure cost:")
    print(
        best_k_df[
            [
                "dataset",
                "cost_fn",
                "k",
                "TP_maintained_risky",
                "FP_unnecessary_maintenance",
                "FN_missed_risky_cases",
                "false_negative_rate",
                "false_positive_rate",
                "total_cost",
            ]
        ]
    )

    return dataset_sweep_df, best_k_df


def main():
    all_sweeps = []
    all_best = []

    for dataset_name in DATASETS:
        sweep_df, best_df = run_one_dataset(dataset_name)
        all_sweeps.append(sweep_df)
        all_best.append(best_df)

    all_sweeps_df = pd.concat(all_sweeps, ignore_index=True)
    all_best_df = pd.concat(all_best, ignore_index=True)

    all_sweeps_df.to_csv(
        PROCESSED_DIR / "all_datasets_fine_k_cost_sweep.csv",
        index=False,
    )

    all_best_df.to_csv(
        PROCESSED_DIR / "all_datasets_best_k_by_cost.csv",
        index=False,
    )

    print("\n\n==============================")
    print("FINAL COMBINED BEST-K TABLE")
    print("==============================")
    print(
        all_best_df[
            [
                "dataset",
                "cost_fn",
                "k",
                "TP_maintained_risky",
                "FP_unnecessary_maintenance",
                "FN_missed_risky_cases",
                "false_negative_rate",
                "false_positive_rate",
                "total_cost",
            ]
        ]
    )

    print("\nSaved combined outputs to:")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    main()