"""Safety aware metrics for RUL prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def residuals(y_true , y_pred):
    """residual = y_true - y_pred. negative residual mean overestimation"""
    return np.asarray(y_true) - np.asarray(y_pred)



def safety_mae(y_true,y_pred, overestimate_weight : float = 2.0)->float:
    """MAE WITH HEAVIER PENALTY FOR OVERESTIMATING RUL
    
    OBVERESTIMATION IS DANGEROUS BECAUSE PREDICTED RUL > TRUE RUL CAN DELAY MAINTENANCE 
    """
    r =  residuals(y_true,y_pred)
    abs_error = np.abs(r)
    weighted = np.where(r<0 , overestimate_weight* abs_error , abs_error)
    return float(np.mean(weighted))

def risk_loss (y_true,y_pred , lambda_: float = 2.0 , gamma : float  = 0.05)-> float:
    """exponential assymetric loss that weights overestimation more neae failure"""
    y_true = np.asarray(y_true)
    r = residuals(y_true,y_pred)
    maintenance = np.maximum(r,0)
    catastrophic = lambda_* np.exp(-gamma*y_true)*np.maximum(-r,0)
    return float (np.mean(maintenance+catastrophic ))

def count_dangerous_overestimates(y_true,y_pred , threshold:float =20)->float:
    """ count overestimates in the low rul region """    
    y_true = np.asarray(y_true)
    r = residuals(y_true,y_pred)
    return int(((y_true < threshold ) &  (r < 0 )).sum())

def avg_overestimate_magnitude(y_true,y_pred,threshold:float = 20)->float:
    """average magnitude of dangerous overestimations in the low RUL region"""
    y_true = np.asarray(y_true)
    r = residuals(y_true,y_pred)
    mask = (y_true<threshold) & (r<0)
    if mask.sum() ==0:
        return 0.0
    return float(np.mean(-r[mask]))



def catastrophic_risk_index(y_true,y_pred , threshold:float = 5)->float:
    """simple frequency x severity score for dangerous overestimation"""
    y_true = np.asarray(y_true)
    r = residuals(y_true,y_pred)
    region = y_true < threshold
    dangerous = region & (r<0)
    if region.sum() == 0:
        return 0.0
    frequency = dangerous.sum() / region.sum()
    severity = np.mean(-r[dangerous]) if dangerous.sum() else 0.0
    return float(frequency*severity)


def make_safety_table(y_true , predictions:dict[str,object],threshold:float = 20)-> pd.DataFrame:
    """combine standard MAE with safety specific metrics"""
    rows = []
    for name, y_pred in predictions.items():
        rows.append({
            "Model": name,
            "MAE": float(mean_absolute_error(y_true,y_pred)),
            "Safety_MAE": safety_mae(y_true,y_pred),
            f"Dangerous_Count_RUL_LT_{threshold}":count_dangerous_overestimates(y_true , y_pred ,threshold),
            f"Avg_Overestimate_RUL_LT_{threshold}": avg_overestimate_magnitude(y_true,y_pred,threshold)

        })
    return pd.DataFrame(rows).sort_values("Safety_MAE").reset_index(drop=True)

def sweep_risk_loss(y_true,predictions:dict[str,object],
                    lambda_values = (1,2,3,5),
                    gamma_values = (0.01,0.05,0.1),)->pd.DataFrame:
    
    """evaluate riskloss over lambda/gamma sensitivity grid"""
    rows = []
    for lambda_ in lambda_values:
        for gamma in gamma_values:
            for model_name , y_pred in predictions.items():
                rows.append({
                    "lambda":lambda_,
                    "gamma": gamma,
                    "Model" : model_name,
                    "Risk_Loss" : risk_loss(y_true,y_pred, lambda_=lambda_ , gamma=gamma)
                })
    return pd.DataFrame(rows).sort_values(["lambda", "gamma", "Risk_Loss"]).reset_index(drop=True)
