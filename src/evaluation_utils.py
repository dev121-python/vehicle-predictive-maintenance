import numpy as np
# safety mae
def safety_mae(y_true, y_pred):
    residual = y_true - y_pred
    abs_error = np.abs(residual)
    safety_error = np.where(residual < 0, 2 * abs_error, abs_error)
    return np.mean(safety_error)

# riskloss
def risk_loss(y_true, y_pred, lambda_, gamma):
    r = y_true - y_pred
    maintenance = np.maximum(r, 0)
    catastrophic = lambda_ * np.exp(-gamma * y_true) * np.maximum(-r, 0)
    return np.mean(maintenance + catastrophic)

# dangerous overestimation count
def count_dangerous_overestimates(y_true, y_pred, threshold):
    r = y_true - y_pred
    near_failure = y_true < threshold
    overestimate = r < 0
    dangerous = near_failure & overestimate
    return np.sum(dangerous)

# avg overestimate magnitudes

def avg_overestimate_magnitude(y_true, y_pred, threshold):
    r = y_true - y_pred
    mask = (y_true < threshold) & (r < 0)

    if np.sum(mask) == 0:
        return 0.0

    return np.mean(-r[mask])

# catastrophic risk index

def catastrophic_risk_index(y_true, y_pred, threshold):
    r = y_true - y_pred
    mask_region = y_true < threshold
    mask_danger = mask_region & (r < 0)

    total_region = np.sum(mask_region)
    dangerous_count = np.sum(mask_danger)

    if dangerous_count == 0 or total_region == 0:
        return 0.0

    avg_magnitude = np.mean(-r[mask_danger])
    frequency_ratio = dangerous_count / total_region

    return frequency_ratio * avg_magnitude


