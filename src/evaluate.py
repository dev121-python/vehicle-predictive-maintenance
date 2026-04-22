from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from evaluation_utils import safety_mae, catastrophic_risk_index

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
def compute_residuals(y_true, models):
    return {
        name: y_true - y_pred
        for name, y_pred in models.items()
    }
def residual_bias_curve(y_true, y_pred, bins=15):
    import pandas as pd
    
    df = pd.DataFrame({
        "y_true": y_true,
        "residual": y_true - y_pred
    })
    
    df["rul_bin"] = pd.cut(df["y_true"], bins=bins)
    
    return df.groupby("rul_bin")["residual"].mean()
def residual_variance_curve(y_true, y_pred, bins=15):
    import pandas as pd
    
    df = pd.DataFrame({
        "y_true": y_true,
        "residual": y_true - y_pred
    })
    
    df["rul_bin"] = pd.cut(df["y_true"], bins=bins)
    
    return df.groupby("rul_bin")["residual"].std()

def overestimation_rate_curve(y_true, y_pred, bins=15):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        "y_true": y_true,
        "residual": y_true - y_pred
    })
    
    df["rul_bin"] = pd.cut(df["y_true"], bins=bins)
    
    return df.groupby("rul_bin").apply(
        lambda x: np.mean(x["residual"] < 0)
    )

def evaluate_all_models(models, y_true):
    import pandas as pd
    
    results = []

    for name, y_pred in models.items():
        results.append([
            name,
            safety_mae(y_true, y_pred),
            catastrophic_risk_index(y_true, y_pred, threshold=5)
        ])

    return pd.DataFrame(
        results,
        columns=["Model", "Safety-MAE", "CRI@5"]
    ).sort_values("CRI@5")