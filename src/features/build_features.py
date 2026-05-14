"""feature engineering utilities for Rul moeling """
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def  add_rul_target(
        df: pd.DataFrame,
        id_col:str = "unit",
        time_col : str = 'cycle',
        cap: int | None =140,
        drop_max_cycle : bool = False )-> pd.DataFrame:
    """Add train-set Rul target usinf each engines last observed cycle as failure
    
    in the c-mapss train set , every engine trajectory ends at failure , so 
    RUL = max_cycle_for_engine - current _cycle
    """
    out  = df.copy()
    max_cycle = out.groupby(id_col,as_index=False)[time_col].max()
    max_cycle = max_cycle.rename(columns= {time_col:"max_cycle"})
    out = out.merge(max_cycle,on = id_col,how="left")
    out['RUL'] = out["max_cycle"] - out[time_col]
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=cap)
    if drop_max_cycle:
        out = out.drop(columns=["max_cycle"])
    return out 


def add_test_rul_at_final_cycle(
        test_df : pd.DataFrame ,
        rul_df : pd.DataFrame,
        id_col : str = "unit",
        time_col: str = "cycle",
        cap : int | None  = 140
        
)-> pd.DataFrame:
    """attacj true rul labels to the final row of each test engine 
    
    the cmapss test target is defined only for the final observed cycle of each engine 
    """
    final_rows = test_df.sort_values([id_col,time_col]).groupby(id_col ,as_index=False).tail(1).copy()
    
    final_rows = final_rows.sort_values(id_col).reset_index(drop=True)
    if len(final_rows)!=len(rul_df):
        raise ValueError(f"number of test engines ({len(final_rows)})does not match rul rows ({len(rul_df)})")
    final_rows["RUL"] = rul_df["final_rul"].to_numpy()
    if cap is not None :
        final_rows["RUL"] = final_rows["RUL"].clip(upper=cap)

    return final_rows




def train_val_split_by_unit(
        df: pd.DataFrame,
        test_size : float = 0.2
        ,random_state : int = 42,
        id_col : str = "unit"
) -> tuple[pd.DataFrame,pd.DataFrame]:
    """split engine by id so cycles from the same engine never leak across sets
    """
    engine_ids = df[id_col].unique()
    train_ids,val_ids = train_test_split(engine_ids,test_size=test_size,random_state=random_state)
    train_df = df[df[id_col].isin(train_ids)].copy()
    val_df = df[df[id_col].isin(val_ids)].copy()
    return train_df , val_df




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
def add_rolling_features(
        df : pd.DataFrame, 
        sensor_cols : list[str], 
        window_size:int = 5, 
        id_col: str = "unit", 
        time_col: str = "cycle"
        )-> pd.DataFrame:
    """add rolling mean and std for selected sensors within each engine trajectory
    """
    
    
    out = df.copy().sort_values(by=[id_col, time_col])

    for col in sensor_cols:
        if col not in out.columns:
            raise KeyError(f"missing sensor columns: {col}")
        grouped = out.groupby(id_col)[col]
        out[f"{col}_roll_mean_{window_size}"] = grouped.transform(
            lambda s: s.rolling(window=window_size, min_period = 1).mean()
              )
        out[f"{col}_roll_std_{window_size}"] = grouped.transform(
            lambda s : s.rolling(window = window_size , min_periods  = 1).std().fillna(0)

        )
    return out

def _slope (values: np.ndarray)-> float:
    """fast least squares slope for 1d array"""
    values = np.asarray(values, dtype=float)
    if len(values)<2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    denom = np.dot(x_centered , y_centered)
    if denom == 0:
        return 0.0
    
    return float(np.dot(x_centered,y_centered)/denom)

def add_recent_slope_features(
        df:pd.DataFrame,
        sensor_cols : list[str],
        window_size : int = 30
        ,id_col : str = "unit"
        ,time_col :str = "cycle"
)->pd.DataFrame:
    """add rolling recent slope features for selected sensors"""
    out = df.copy().sort_values([id_col,time_col])
    for col in sensor_cols:
        if col in out.columns:
            raise KeyError(f"missing sensor columns :{col}")
        out[f"{col}_slope_{window_size}"] = out.groupby(id_col)[col].transform(
            lambda s: s.rolling(window = window_size, min_periods = 2).apply(_slope,raw=True).fillna(0)

        )
    return out





def prepare_xy(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols:list[str],
        target_col:str = "RUL"):
    """return xtrain ytrain xval and yval from prepared dataframes"""
    missing = [c for c in feature_cols if c not in train_df.columns or c not in val_df.columns]
    if missing:
        raise KeyError(f"missing features columns:{missing}")
    return(
        train_df[feature_cols],
        train_df[target_col],
        val_df[feature_cols]
        ,val_df[target_col],
    )

def train_cal_eval_split_by_unit(
        df,
        train_size = 0.60,
        cal_size = 0.20,
        eval_size = 0.20,
        unit_col = "unit",
        random_state = 42
):
    """engine wise split into train / calibration / evaluation sets
    
    important:
    we split by engine id , not by rows ,
    this prevents cycles from the same engine appearing in multiple splits 
    
    train:
          used to train the rul model
          
    calibration:
          used to choose best k value / safety margin.
          
    evaluation :
          used only for final reported results """
    

    if abs ((train_size + cal_size +eval_size )- 1.0)> 1e-8:
        raise ValueError("train_size + cal_size + eval_size must equal 1.0")
    units = df[unit_col].unique()

    train_units , temp_units = train_test_split(
        units,
        train_size = train_size,
        random_state=random_state,
        shuffle=True
    )

    relative_cal_size = cal_size/(cal_size + eval_size)
    cal_units , eval_units = train_test_split(
        temp_units,
        train_size=relative_cal_size,
        random_state=random_state,
        shuffle = True

    )
    train_df = df[df[unit_col].isin(train_units)].copy()
    cal_df = df[df[unit_col].isin(cal_units)].copy()
    eval_df = df[df[unit_col].isin(eval_units)].copy()

    return train_df ,eval_df , cal_df

