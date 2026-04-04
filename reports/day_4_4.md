# 📊 Experiment Log — RUL Prediction under Asymmetric Risk

## 🧠 Objective

The goal of this project is to predict Remaining Useful Life (RUL) of engines using the NASA C-MAPSS Dataset and evaluate model performance under **asymmetric risk**.

In this setting:

* Late predictions (overestimating RUL near failure) are **dangerous**
* Early predictions (underestimating RUL) are **less costly**

---

## ⚠️ Definition of Dangerous Predictions

A **dangerous overestimation** occurs when:

* Actual RUL is low (near failure)
* Model predicts a high RUL (false sense of safety)

We evaluate this in three regions:

* RUL < 20 → Warning zone
* RUL < 10 → High risk
* RUL < 5 → Critical zone

---

## 📊 Baseline Model Results (No Rolling Features)

### Dangerous Overestimations

#### RUL < 20

* Linear Regression: 239
* Decision Tree: 302
* Random Forest: 315
* Gradient Boosting: 281

#### RUL < 10

* Linear Regression: 101
* Decision Tree: 180
* Random Forest: 177
* Gradient Boosting: 163

#### RUL < 5

* Linear Regression: **43**
* Decision Tree: 100
* Random Forest: 96
* Gradient Boosting: 89

---

## 🧠 Observation (Baseline)

* Linear Regression is the **safest model** (least dangerous errors)
* More complex models (Random Forest, Gradient Boosting) are:

  * More accurate overall
  * But **more dangerous near failure**

👉 This highlights a key trade-off:

> Higher accuracy does not guarantee safer predictions.

---

## 🔧 Feature Engineering: Rolling Features

To improve temporal understanding, rolling features were added:

### Selected Sensors

* s_7, s_12, s_14, s_20, s_21

### Features Used

* Raw sensor values
* Rolling mean
* Rolling standard deviation

Window size = 5

---

## 🌲 Random Forest (Selected Sensors + Rolling)

* R²: 0.807
* MAE: 14.94
* RMSE: 20.52
* CRI: 2.38
* Dangerous overestimates: **86**

---

## 🌿 Gradient Boosting (Selected Sensors + Rolling)

* R²: 0.607
* MAE: 19.53
* RMSE: 29.33
* CRI: 2.20
* Dangerous overestimates: **68**

---

## 📊 Comparison Summary

| Model             | Accuracy | Dangerous Errors | Risk     |
| ----------------- | -------- | ---------------- | -------- |
| Random Forest     | High     | 86               | Moderate |
| Gradient Boosting | Lower    | **68**           | Lower    |

---

## 🔥 Key Insights

1. **Feature selection significantly impacts safety**

   * Using selected sensors reduced dangerous errors

2. **Rolling features help capture degradation trends**

   * Improved model stability

3. **Accuracy and safety are not aligned**

   * Models with lower RMSE can still be unsafe
   * Simpler models may behave more conservatively

4. **Trade-off identified**

   * High accuracy ↔ High risk
   * Lower accuracy ↔ Lower risk

---

## 🚨 Key Finding

> Complex models can become overconfident and produce dangerous predictions near failure.

---

## 🎯 Next Step

Even after feature engineering:

* Dangerous overestimations still exist

This suggests:

* Deterministic models lack awareness of uncertainty

### Next Direction:

* Introduce **uncertainty-aware models (MC Dropout)**
* Use uncertainty to improve safety decisions

---

## 🧠 Working Hypothesis

> Uncertainty-aware models will reduce dangerous overestimations while maintaining reasonable predictive accuracy.

---

## 🚀 Progress Summary

* ✔ Baseline models implemented
* ✔ Risk-based evaluation designed
* ✔ Feature engineering completed
* ✔ Trade-off between accuracy and safety identified

➡️ Next: Uncertainty modeling + decision optimization
