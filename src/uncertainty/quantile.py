import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def make_quantile_model(
    alpha,
    random_state=42,
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
):
    """
    Create one Gradient Boosting quantile regression model.

    alpha:
        Quantile level.
        Example:
            0.10 = lower quantile
            0.50 = median
            0.90 = upper quantile
    """

    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    return pipeline


def fit_quantile_models(
    X_train,
    y_train,
    lower_alpha=0.10,
    median_alpha=0.50,
    upper_alpha=0.90,
    random_state=42,
):
    """
    Fit lower, median, and upper quantile models.
    """

    lower_model = make_quantile_model(
        alpha=lower_alpha,
        random_state=random_state,
    )

    median_model = make_quantile_model(
        alpha=median_alpha,
        random_state=random_state + 1,
    )

    upper_model = make_quantile_model(
        alpha=upper_alpha,
        random_state=random_state + 2,
    )

    print("Training lower quantile model...")
    lower_model.fit(X_train, y_train)

    print("Training median quantile model...")
    median_model.fit(X_train, y_train)

    print("Training upper quantile model...")
    upper_model.fit(X_train, y_train)

    return {
        "lower": lower_model,
        "median": median_model,
        "upper": upper_model,
        "lower_alpha": lower_alpha,
        "median_alpha": median_alpha,
        "upper_alpha": upper_alpha,
    }


def predict_quantile_interval(
    quantile_models,
    X,
):
    """
    Predict lower, median, and upper quantiles.

    Returns:
        pred_median
        uncertainty_proxy
        pred_lower
        pred_upper
        interval_width

    We define:
        uncertainty_proxy = max(pred_median - pred_lower, 0)

    This means:
        conservative_rul = pred_median - k * uncertainty_proxy

    When k = 1, the conservative prediction is approximately the lower quantile.
    """

    pred_lower = quantile_models["lower"].predict(X)
    pred_median = quantile_models["median"].predict(X)
    pred_upper = quantile_models["upper"].predict(X)

    # Handle quantile crossing safely
    pred_lower_fixed = np.minimum(pred_lower, pred_median)
    pred_upper_fixed = np.maximum(pred_upper, pred_median)

    interval_width = pred_upper_fixed - pred_lower_fixed

    uncertainty_proxy = np.maximum(pred_median - pred_lower_fixed, 0.0)

    return pred_median, uncertainty_proxy, pred_lower_fixed, pred_upper_fixed, interval_width