"""Run the FD001 baseline experiment from the command line.

Usage:
    python scripts/run_fd001_baseline.py
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR, DEFAULT_RUL_CAP, RANDOM_STATE
from src.data.loading import load_train, sensor_columns
from src.features.build_features import add_rul_target, train_val_split_by_unit, prepare_xy
from src.models.train_baselines import fit_models, predict_models, evaluate_model_predictions
from src.evaluation.metrics import make_safety_table


def main():
    df = load_train(DATA_DIR, "FD001")
    df = add_rul_target(df, cap=DEFAULT_RUL_CAP)
    train_df, val_df = train_val_split_by_unit(df, random_state=RANDOM_STATE)
    features = sensor_columns(df)
    X_train, y_train, X_val, y_val = prepare_xy(train_df, val_df, features)

    fitted = fit_models(X_train, y_train, random_state=RANDOM_STATE)
    preds = predict_models(fitted, X_val)

    print("\nStandard regression metrics")
    print(evaluate_model_predictions(y_val, preds).to_string(index=False))
    print("\nSafety-aware metrics")
    print(make_safety_table(y_val, preds, threshold=20).to_string(index=False))


if __name__ == "__main__":
    main()
