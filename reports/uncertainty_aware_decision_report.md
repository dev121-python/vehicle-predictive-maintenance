# 📊 Results & Discussion — Uncertainty-Aware Maintenance Decisions

## 🔹 Overview

This section evaluates whether incorporating uncertainty into Remaining Useful Life (RUL) predictions improves maintenance decisions under asymmetric risk.

Two policies are compared:

1. **Point Prediction Policy**

   * Maintenance decision based only on predicted RUL:

   ```
   Maintain if: ŷ ≤ τ
   ```

2. **Uncertainty-Aware Policy**

   * Uses a conservative estimate incorporating predictive uncertainty:

   ```
   Maintain if: ŷ − 1.64σ ≤ τ
   ```

---

## 🔹 Uncertainty as a Risk Indicator

Uncertainty was estimated using ensemble disagreement from a Random Forest model.

Key observations:

* Correlation between uncertainty and absolute error: **0.585**
* 90% prediction interval coverage: **0.937**

Additionally, analysis across uncertainty bins showed:

* Mean absolute error increases significantly with uncertainty
* Dangerous overestimation rate increases sharply with uncertainty

👉 **Interpretation:**

> Predictive uncertainty is strongly associated with both prediction error and catastrophic risk.

---

## 🔹 Safety vs Maintenance Trade-off

The trade-off between safety and maintenance efficiency is illustrated below:

### Observations:

* **Point Prediction Policy (blue):**

  * Very low false positive rate (efficient)
  * High false negative rate (~15–27%) ❌
  * Frequently misses high-risk engines

* **Uncertainty-Aware Policy (orange):**

  * Slightly higher false positive rate (more maintenance)
  * Near-zero false negative rate (~0–2%) ✅
  * Successfully identifies almost all risky cases

---

## 🔥 Core Result

> The uncertainty-aware policy shifts the operating point from a high-risk, low-maintenance regime to a low-risk, moderate-maintenance regime.

---

## 🔹 Cost Analysis under Asymmetric Risk

Under asymmetric cost assumptions:

* Early maintenance cost = 1
* Failure cost = variable (5–100)

### Key findings:

* Point Prediction Policy:

  * Cost increases rapidly with failure cost
  * Sensitive to missed failures

* Uncertainty-Aware Policy:

  * Cost remains relatively stable
  * Robust to increasing failure penalties

👉 **Interpretation:**

> Uncertainty-aware decisions provide cost stability under extreme asymmetric risk conditions.

---

## 🔹 Trade-off Analysis

| Policy            | False Negatives (Risk) | False Positives (Maintenance) |
| ----------------- | ---------------------- | ----------------------------- |
| Point Prediction  | High ❌                 | Very Low ✅                    |
| Uncertainty-Aware | Near Zero ✅            | Moderate ⚠️                   |

---

## 🔹 Discussion

The results reveal a fundamental trade-off:

* **Point prediction models are efficient but unsafe**
* **Uncertainty-aware models are safer but more conservative**

In safety-critical systems, such as predictive maintenance:

```text
Cost of failure >> cost of early maintenance
```

Therefore:

> Reducing false negatives is significantly more important than minimizing false positives.

---

## 🔹 Key Insight

> Predictive uncertainty acts as a reliable proxy for risk and enables safer decision-making.

---

## 🔹 Limitations

* Uncertainty is approximated using ensemble variance (not fully probabilistic)
* Gaussian assumption for prediction intervals may not always hold
* Results are based on a single model (Random Forest)

---

## 🔹 Conclusion

This study demonstrates that:

* Uncertainty-aware decision policies significantly reduce catastrophic failures
* The increase in maintenance actions is justified under asymmetric cost settings
* Ensemble-based uncertainty is sufficient to achieve strong decision improvements

---

## 🔥 One-Line Takeaway

> Incorporating uncertainty into maintenance decisions transforms predictive models from accuracy-focused systems into risk-aware decision tools.
