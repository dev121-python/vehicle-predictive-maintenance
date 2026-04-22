import pandas as pd
import numpy as np

from utils import load_cmapss ,add_rul

import pandas as pd

def load_data(path):
    return load_cmapss(path)

def add_target(df):
    return add_rul(df, cap=140)