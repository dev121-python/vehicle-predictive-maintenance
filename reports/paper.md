# Uncertainty-Aware Remaining Useful Life Prediction for Risk-Sensitive Maintenance Decisions

## Abstract

Predicting Remaining Useful Life (RUL) is central to predictive maintenance. However, in safety-critical settings, prediction accuracy alone is insufficient—decision-making must account for asymmetric risk, where late maintenance (overestimating RUL) can lead to catastrophic failure. This work investigates whether incorporating predictive uncertainty improves maintenance decisions. Using the NASA C-MAPSS Dataset, we train a Random Forest regressor with engineered features and estimate uncertainty via ensemble disagreement. We show that uncertainty correlates strongly with prediction error and catastrophic risk. Incorporating uncertainty into a conservative decision policy significantly reduces false negatives (missed failures) and yields lower asymmetric cost across thresholds, at the expense of increased preventive maintenance. The results demonstrate that even simple ensemble-based uncertainty can transform predictive models into effective risk-aware decision systems.

---

## 1. Introduction

Predictive maintenance aims to anticipate failures and schedule interventions optimally. A common approach is to estimate Remaining Useful Life (RUL) from sensor data. While modern machine learning models achieve strong predictive performance, decision-making based solely on point estimates can be unsafe in real-world deployments.

In many applications, costs are asymmetric:

* Overestimating RUL (late maintenance) → **high-risk failure**
* Underestimating RUL (early maintenance) → **lower cost inefficiency**

Therefore, the key problem is not only prediction accuracy, but **risk-aware decision-making**.

This paper investigates:

> Can uncertainty-aware RUL predictions improve maintenance decisions under asymmetric risk?

---

## 2. Related Work

Predictive maintenance has been widely studied using machine learning models such as linear regression, tree-based ensembles, and deep learning. Traditional approaches focus on minimizing prediction error (e.g., RMSE, MAE), but often neglect decision-level consequences.

Recent work has explored uncertainty estimation methods, including:

* Bayesian neural networks
* Monte Carlo Dropout
* Ensemble methods

Ensemble-based uncertainty, such as variance across Random Forest trees, provides a practical and computationally efficient proxy for predictive uncertainty.

However, fewer studies explicitly connect uncertainty to **decision-making under asymmetric cost**, which is the focus of this work.

---

## 3. Methodology

### 3.1 Dataset

Experiments are conducted on the NASA C-MAPSS Dataset, which contains simulated engine degradation data with multiple sensor measurements across operational cycles.

Each engine instance is treated independently, and data is split by engine ID to prevent leakage.

---

### 3.2 Feature Engineering

We select a subset of informative sensors:

* s_7, s_12, s_14, s_20, s_21

To capture temporal degradation trends, rolling statistics are computed:

* Rolling mean
* Rolling standard deviation (window size = 5)

---

### 3.3 Predictive Model

A Random Forest Regressor is trained to predict RUL using the engineered features.

---

### 3.4 Uncertainty Estimation

Uncertainty is approximated using ensemble disagreement:

* For each sample, predictions from all trees are collected
* Mean prediction:
  $$
  \hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(x)
  $$
* Uncertainty (standard deviation):
  $$
  \sigma = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (f_t(x) - \hat{y})^2}
  $$

Prediction intervals are constructed assuming approximate normality:

$$
\hat{y} \pm 1.64\sigma
$$

---

### 3.5 Decision Policies

We define a maintenance threshold (\tau).

#### Point Prediction Policy

$$
\text{Maintain if } \hat{y} \leq \tau
$$

#### Uncertainty-Aware Policy

$$
\text{Maintain if } \hat{y} - 1.64\sigma \leq \tau
$$

This introduces a conservative adjustment to account for uncertainty.

---

### 3.6 Evaluation Metrics

We evaluate both prediction quality and decision performance:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Correlation between uncertainty and error
* Prediction interval coverage
* False Negative Rate (missed failures)
* False Positive Rate (unnecessary maintenance)
* Asymmetric cost:
  $$
  \text{Cost} = c_{fn} \cdot FN + c_{fp} \cdot FP
  $$

where (c_{fn} \gg c_{fp}).

---

## 4. Results

### 4.1 Uncertainty Calibration

* Correlation (uncertainty vs error): **0.585**
* 90% interval coverage: **0.937**

Uncertainty increases monotonically with error and risk:

* Mean error rises significantly across uncertainty bins
* Dangerous overestimate rate increases sharply

#### Key Insight

> Predictive uncertainty is strongly aligned with both error magnitude and catastrophic risk.

---

### 4.2 Safety vs Maintenance Trade-off

The trade-off between false positives (maintenance) and false negatives (failures) shows:

* Point prediction policy:

  * Low false positive rate
  * High false negative rate (~15–27%)

* Uncertainty-aware policy:

  * Moderate false positive rate
  * Near-zero false negative rate (~0–2%)

#### Core Result

> The uncertainty-aware policy shifts the system from a high-risk regime to a low-risk regime.

---

### 4.3 Cost Analysis under Asymmetric Risk

Under increasing failure cost:

* Point prediction cost grows rapidly
* Uncertainty-aware cost remains stable

#### Interpretation

> Uncertainty-aware policies are robust to increasing failure penalties.

---

### 4.4 Threshold Sensitivity

Across maintenance thresholds:

* Uncertainty-aware policy consistently achieves lower cost
* Benefits are strongest at low and high thresholds
* Policies converge at moderate thresholds

---

## 5. Discussion

The results highlight a fundamental trade-off:

* Point prediction models are efficient but unsafe
* Uncertainty-aware models are safer but conservative

In high-risk environments, reducing false negatives is critical. The uncertainty-aware policy effectively captures high-risk cases and avoids catastrophic outcomes.

---

## 6. Limitations

* Uncertainty is approximated via ensemble variance
* Gaussian assumption for intervals may not hold
* Only one model (Random Forest) is evaluated

---

## 7. Conclusion

This work demonstrates that:

* Ensemble-based uncertainty is a strong proxy for prediction risk
* Incorporating uncertainty significantly reduces catastrophic failures
* Decision-level improvements can be achieved without complex probabilistic models

---

## 8. Future Work

Future directions include:

* Neural network uncertainty (MC Dropout)
* Quantile-based prediction intervals
* Improved calibration methods
* Multi-model comparison

---

## One-Line Takeaway

> Uncertainty transforms predictive models into risk-aware decision systems under asymmetric cost.
