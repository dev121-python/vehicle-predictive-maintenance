"""Plotting helpers for notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted RUL"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.2)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_residuals(y_true, residual, title="Residual Plot"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, residual, alpha=0.2)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Residual = y_true - y_pred")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_low_rul_hist(residual, title="Low-RUL Residual Distribution"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residual, bins=30)
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Residual = y_true - y_pred")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_uncertainty_vs_error(df, uncertainty_col="pred_std", error_col="abs_error"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[uncertainty_col], df[error_col], alpha=0.2)
    ax.set_xlabel("Prediction uncertainty (std across trees)")
    ax.set_ylabel("Absolute error")
    ax.set_title("Uncertainty vs Error")
    ax.grid(alpha=0.3)
    return ax


def plot_policy_sweep(sweep_df, y_col="cost", title="Policy Comparison"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in sweep_df["policy"].unique():
        temp = sweep_df[sweep_df["policy"] == policy]
        ax.plot(temp["threshold"], temp[y_col], marker="o", label=policy)
    ax.set_xlabel("Maintenance threshold")
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_cost_ratio_sweep(cost_ratio_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in cost_ratio_df["policy"].unique():
        temp = cost_ratio_df[cost_ratio_df["policy"] == policy]
        ax.plot(temp["failure_cost"], temp["cost"], marker="o", label=policy)
    ax.set_xlabel("False-negative failure cost")
    ax.set_ylabel("Average asymmetric cost")
    ax.set_title("Cost Sensitivity")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax
