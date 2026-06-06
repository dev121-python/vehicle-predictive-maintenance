import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def make_bootstrap_base_model(
    model_type="gbr",
    random_state=42,
):
    """
    Create one base model for the bootstrap ensemble.

    model_type:
        "gbr" = Gradient Boosting Regressor
        "rf"  = Random Forest Regressor
    """

    if model_type == "gbr":
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )

    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    else:
        raise ValueError("model_type must be either 'gbr' or 'rf'")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    return pipeline


def sample_bootstrap_units(
    train_df,
    unit_col="unit",
    sample_frac=1.0,
    random_state=42,
):
    """
    Bootstrap sample engines with replacement.

    Important:
    We sample whole engines, not rows.
    This preserves time-series grouping and avoids weird cycle leakage.

    If an engine is sampled multiple times, its rows are repeated.
    """

    rng = np.random.default_rng(random_state)

    unique_units = train_df[unit_col].unique()
    n_units_to_sample = int(len(unique_units) * sample_frac)

    sampled_units = rng.choice(
        unique_units,
        size=n_units_to_sample,
        replace=True,
    )

    sampled_parts = []

    for unit in sampled_units:
        unit_df = train_df[train_df[unit_col] == unit].copy()
        sampled_parts.append(unit_df)

    boot_df = pd.concat(sampled_parts, ignore_index=True)

    return boot_df


def fit_bootstrap_ensemble(
    train_df,
    feature_cols,
    target_col="RUL",
    unit_col="unit",
    n_models=20,
    sample_frac=1.0,
    model_type="gbr",
    random_state=42,
):
    """
    Train a bootstrap ensemble.

    Each model is trained on a different bootstrap sample of engines.

    Returns:
        ensemble: list of fitted sklearn pipelines
    """

    ensemble = []

    for i in range(n_models):
        seed = random_state + i

        boot_df = sample_bootstrap_units(
            train_df=train_df,
            unit_col=unit_col,
            sample_frac=sample_frac,
            random_state=seed,
        )

        X_boot = boot_df[feature_cols]
        y_boot = boot_df[target_col]

        model = make_bootstrap_base_model(
            model_type=model_type,
            random_state=seed,
        )

        model.fit(X_boot, y_boot)

        ensemble.append(model)

        print(f"Trained bootstrap model {i + 1}/{n_models}")

    return ensemble


def bootstrap_prediction_matrix(
    ensemble,
    X,
):
    """
    Return predictions from every model.

    Shape:
        n_models x n_samples
    """

    preds = []

    for model in ensemble:
        pred = model.predict(X)
        preds.append(pred)

    pred_matrix = np.vstack(preds)

    return pred_matrix


def predict_bootstrap_mean_std(
    ensemble,
    X,
):
    """
    Use bootstrap ensemble to estimate prediction mean and uncertainty.

    pred_mean:
        average prediction across bootstrap models

    pred_std:
        standard deviation across bootstrap models
    """

    pred_matrix = bootstrap_prediction_matrix(ensemble, X)

    pred_mean = pred_matrix.mean(axis=0)
    pred_std = pred_matrix.std(axis=0, ddof=1)

    return pred_mean, pred_std