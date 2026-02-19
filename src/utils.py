import pandas as pd
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
