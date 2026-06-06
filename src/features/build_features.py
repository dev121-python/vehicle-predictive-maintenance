"""Feature engineering utilities for RUL modeling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def add_rul_target(
    df: pd.DataFrame,
    id_col: str = "unit",
    time_col: str = "cycle",
    cap: int | None = 140,
    drop_max_cycle: bool = False,
) -> pd.DataFrame:
    """Add train-set RUL target using each engine's last observed cycle as failure.

    In the C-MAPSS train set, every engine trajectory ends at failure, so:
    RUL = max_cycle_for_engine - current_cycle.
    """
    out = df.copy()
    max_cycle = out.groupby(id_col, as_index=False)[time_col].max()
    max_cycle = max_cycle.rename(columns={time_col: "max_cycle"})
    out = out.merge(max_cycle, on=id_col, how="left")
    out["RUL"] = out["max_cycle"] - out[time_col]
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=cap)
    if drop_max_cycle:
        out = out.drop(columns=["max_cycle"])
    return out


def add_test_rul_at_final_cycle(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    id_col: str = "unit",
    time_col: str = "cycle",
    cap: int | None = 140,
) -> pd.DataFrame:
    """Attach true RUL labels to the final row of each test engine.

    The C-MAPSS test target is defined only for the final observed cycle of each engine.
    """
    final_rows = test_df.sort_values([id_col, time_col]).groupby(id_col, as_index=False).tail(1).copy()
    final_rows = final_rows.sort_values(id_col).reset_index(drop=True)
    if len(final_rows) != len(rul_df):
        raise ValueError(
            f"Number of test engines ({len(final_rows)}) does not match RUL rows ({len(rul_df)})"
        )
    final_rows["RUL"] = rul_df["final_rul"].to_numpy()
    if cap is not None:
        final_rows["RUL"] = final_rows["RUL"].clip(upper=cap)
    return final_rows


def train_val_split_by_unit(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    id_col: str = "unit",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by engine id so cycles from the same engine never leak across sets."""
    engine_ids = df[id_col].unique()
    train_ids, val_ids = train_test_split(
        engine_ids, test_size=test_size, random_state=random_state
    )
    train_df = df[df[id_col].isin(train_ids)].copy()
    val_df = df[df[id_col].isin(val_ids)].copy()
    return train_df, val_df


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_size: int = 5,
    id_col: str = "unit",
    time_col: str = "cycle",
) -> pd.DataFrame:
    """Add rolling mean and std for selected sensors within each engine trajectory."""
    out = df.copy().sort_values([id_col, time_col])
    for col in sensor_cols:
        if col not in out.columns:
            raise KeyError(f"Missing sensor column: {col}")
        grouped = out.groupby(id_col)[col]
        out[f"{col}_roll_mean_{window_size}"] = grouped.transform(
            lambda s: s.rolling(window=window_size, min_periods=1).mean()
        )
        out[f"{col}_roll_std_{window_size}"] = grouped.transform(
            lambda s: s.rolling(window=window_size, min_periods=1).std().fillna(0)
        )
    return out


def _slope(values: np.ndarray) -> float:
    """Fast least-squares slope for a 1-D array."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    denom = np.dot(x_centered, x_centered)
    if denom == 0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def add_recent_slope_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_size: int = 30,
    id_col: str = "unit",
    time_col: str = "cycle",
) -> pd.DataFrame:
    """Add rolling recent-slope features for selected sensors."""
    out = df.copy().sort_values([id_col, time_col])
    for col in sensor_cols:
        if col not in out.columns:
            raise KeyError(f"Missing sensor column: {col}")
        out[f"{col}_slope_{window_size}"] = out.groupby(id_col)[col].transform(
            lambda s: s.rolling(window=window_size, min_periods=2).apply(_slope, raw=True).fillna(0)
        )
    return out


def make_feature_list(
    base_sensors: list[str],
    rolling_window: int | None = None,
    include_raw: bool = True,
) -> list[str]:
    """Build the feature names used after rolling-feature creation."""
    features: list[str] = []
    if include_raw:
        features.extend(base_sensors)
    if rolling_window is not None:
        features.extend([f"{s}_roll_mean_{rolling_window}" for s in base_sensors])
        features.extend([f"{s}_roll_std_{rolling_window}" for s in base_sensors])
    return features


def prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "RUL",
):
    """Return X_train, y_train, X_val, y_val from prepared DataFrames."""
    missing = [c for c in feature_cols if c not in train_df.columns or c not in val_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return (
        train_df[feature_cols],
        train_df[target_col],
        val_df[feature_cols],
        val_df[target_col],
    )

from sklearn.model_selection import train_test_split


def train_cal_eval_split_by_unit(
    df,
    train_size=0.60,
    cal_size=0.20,
    eval_size=0.20,
    unit_col="unit",
    random_state=42,
):
    """
    Engine-wise split into train / calibration / evaluation sets.

    Important:
    We split by engine ID, not by rows.
    This prevents cycles from the same engine appearing in multiple splits.

    train:
        Used to train the RUL model.

    calibration:
        Used to choose best k / safety margin.

    evaluation:
        Used only for final reported results.
    """

    if abs((train_size + cal_size + eval_size) - 1.0) > 1e-8:
        raise ValueError("train_size + cal_size + eval_size must equal 1.0")

    units = df[unit_col].unique()

    train_units, temp_units = train_test_split(
        units,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )

    relative_cal_size = cal_size / (cal_size + eval_size)

    cal_units, eval_units = train_test_split(
        temp_units,
        train_size=relative_cal_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df = df[df[unit_col].isin(train_units)].copy()
    cal_df = df[df[unit_col].isin(cal_units)].copy()
    eval_df = df[df[unit_col].isin(eval_units)].copy()

    return train_df, cal_df, eval_df