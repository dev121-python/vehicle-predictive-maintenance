def get_sensor_features(df):
    return [c for c in df.columns if c.startswith("s_")]


def prepare_features(train_df, val_df):
    features = get_sensor_features(train_df)

    X_train = train_df[features]
    y_train = train_df["RUL"]

    X_val = val_df[features]
    y_val = val_df["RUL"]

    return X_train, y_train, X_val, y_val


# 🔥 rolling features (your strongest part)
def add_rolling_features(df, sensor_cols, window_size=5, id_col="unit", time_col="cycle"):
    df = df.copy()
    df = df.sort_values(by=[id_col, time_col])

    for col in sensor_cols:
        df[f"{col}_roll_mean"] = (
            df.groupby(id_col)[col]
            .rolling(window=window_size, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{col}_roll_std"] = (
            df.groupby(id_col)[col]
            .rolling(window=window_size, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    return df