# DATASET Notes - CMAPS5 (_FD0001)

## Dataset Overview
the NASA CMAPS5 turbofan dataset stimulates engine degradation using multi variate timeseries data.
it is commonly used for predictive maintenance and Remaining Useful Life(RUL) prediction.

## DATA STRUCTURE
Each row shows one operational cycle of an engine and contains:
- Engine ID
- cycle number
- 3 operational settings
- 21 sensor measurments

the datset does not include column header : column semantics are defined in the official documentation

## Prediction Task
the Primary task is remaining useful life (RUL) prediction, where the goal is to predict how many cycle remain before failure.


## Subset Selection
initial experiments will focus on FD001 subset, which represents a single operating condition and fault mode.

## NOTE
this dataset serves as a proxy for real world automotive engine sensor data used in predictive maintenence systems