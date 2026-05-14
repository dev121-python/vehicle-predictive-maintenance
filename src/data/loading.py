""" Data Loading helpers for NASA C-MAPSS turbofan RUL data.


THE raw text files contain 26 columns :
unit ,cycle ,3 operational settings and 21 sensor measurements 
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd 

CMAPSS_COLUMNS = (["unit","cycle"] 
                  + [f"op_{i}" for i in range(1,4)]
                  + [f"s_{i}" for i in range (1,22)])

VALID_DATASETS = {"FD001","FD002","FD003","FD004"}

def _validate_dataset(dataset: str):
    dataset = dataset.upper()
    if dataset not in VALID_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(VALID_DATASETS)}, got {dataset!r}")
    return dataset


def load_cmapss_file(path: str | Path):
    """Load one c mapss train/test txt file with standard column names """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"could not find file :{path}")
    return pd.read_csv(path,sep="\s+" , header=None , names = CMAPSS_COLUMNS)
    


def load_train(data_dir: str | Path, dataset : str = "FD001"):
    """load train__fdxxx.txt"""
    dataset = _validate_dataset(dataset)
    return load_cmapss_file(Path(data_dir) / f"train_{dataset}.txt")



def load_test( data_dir: str | Path , dataset: str = "FD001"):
    """load train__fdxxx.txt"""
    dataset = _validate_dataset(dataset)
    return load_cmapss_file(Path(data_dir)/ f"test{dataset}.txt")

def load_rul(data_dir: str | Path , dataset : str = "FD001"):
    """load rul_fdxxx.txt as a one column dataframe named final_rul"""
    dataset = _validate_dataset(dataset)
    path = Path(data_dir) / f"RUL_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"could not find RUL file: {path}")
    return pd.read_csv(path , sep = r"\s+", header = None , names = ['final_rul'])


def load_readme(data_dir:str|Path):
    """read the dataset readme text if present """
    path = Path(data_dir)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")

def sensor_columns(df: pd.dataframe |None = None) -> list[str]:
    """return sensor columns if df is supplied return only columns present in it  """
    cols  = [f"s_{i}" for i in range(1,22)]
    if df is None:
        return cols
    return [ c for c in cols if c in df.columns]

def operation_columns(df:pd.DataFrame|None = None):
    "return operational setting columns if df is supplied return only columns present "
    cols = [f"op_{i}"for i in range (1,4)]
    if df is None:
        return cols
    return [c for c in cols if c in df.columns]