import pandas as pd
from scipy.stats import linregress
import numpy as np


def load_cmapss(path):
    cols = (
        ["unit", "cycle"] +
        [f"op_{i}" for i in range(1, 4)] +
        [f"s_{i}" for i in range(1, 22)]
    )
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols)


def add_rul(df, cap=140):
    max_cycle = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    df = df.merge(max_cycle, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=cap)
    return df

from sklearn.model_selection import train_test_split

def split_by_engine(df, test_size=0.2, random_state=42):
    engine_ids = df['unit'].unique()

    train_ids, val_ids = train_test_split(
        engine_ids,
        test_size=test_size,
        random_state=random_state
    )

    train_df = df[df['unit'].isin(train_ids)]
    val_df = df[df['unit'].isin(val_ids)]

    return train_df, val_df


def get_features_targets(train_df, val_df):
    features = [c for c in train_df.columns if c.startswith("s_")]

    X_train = train_df[features]
    y_train = train_df["RUL"]

    X_val = val_df[features]
    y_val = val_df["RUL"]

    return X_train, y_train, X_val, y_val



def compute_slope(series):
    x = np.arange(len(series))
    slope, _, _, _, _ = linregress(x,series)
    return slope


def recent_slope(series, window = 30):
    series = series[-window:]
    x = np.arange(len(series))
    slope, _, _, _, _ = linregress(x,series)
    return slope
import pandas as pd


def add_rolling_features(df, sensor_cols, window_size=5, id_col="unit", time_col="cycle"):
   
    """
    Adds rolling mean and std for given sensor columns.
    
    Args:
        df: pandas DataFrame
        sensor_cols: list of sensor column names
        window_size: int
        
    Returns:
        df with new features
    """
    
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