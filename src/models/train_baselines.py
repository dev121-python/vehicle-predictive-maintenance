"""Baseline model training helpers."""
from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def make_baseline_models(random_state: int = 42) -> dict[str, object]:
    """Return the baseline regressors used in the first experiment."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.07, max_depth=3, random_state=random_state
        ),
    }


def make_pipeline(model: object) -> Pipeline:
    """Scale numeric inputs, then fit the supplied regressor."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def fit_models(
    X_train,
    y_train,
    models: dict[str, object] | None = None,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Fit all baseline models and return fitted sklearn pipelines."""
    if models is None:
        models = make_baseline_models(random_state=random_state)
    fitted = {}
    for name, model in models.items():
        pipe = make_pipeline(clone(model))
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted


def predict_models(fitted_models: dict[str, Pipeline], X) -> dict[str, object]:
    """Predict with every fitted model."""
    return {name: model.predict(X) for name, model in fitted_models.items()}


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Return standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mse ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model_predictions(y_true, predictions: dict[str, object]) -> pd.DataFrame:
    """Evaluate a dictionary of model predictions."""
    rows = []
    for name, y_pred in predictions.items():
        rows.append({"Model": name, **evaluate_regression(y_true, y_pred)})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def get_model_step(pipeline: Pipeline):
    """Return the final regressor from a fitted pipeline."""
    return pipeline.named_steps["model"]


def transform_features(pipeline: Pipeline, X):
    """Apply the pipeline scaler to X without predicting."""
    return pipeline.named_steps["scaler"].transform(X)
