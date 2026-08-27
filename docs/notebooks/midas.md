```python
"""
This script illustrates the main functionality of the MIDAS class with synthetic data.
It generates data from a MIDAS DGP with known weights, lags, AR terms, and an outlier:
y[t+h] = alpha + beta * X[t]'w + sum_{k=1..p} phi_k * y[t+h-k] + delta * D[t] + eps
where D[t] is an outlier dummy for one quarter. Here, h = 4 and p = 1.
The model uses the same lags, AR terms, and outlier dummy for estimation.
"""
```



    '\nThis script illustrates the main functionality of the MIDAS class with synthetic data.\nIt generates data from a MIDAS DGP with known weights, lags, AR terms, and an outlier:\ny[t+h] = alpha + beta * X[t]\'w + sum_{k=1..p} phi_k * y[t+h-k] + delta * D[t] + eps\nwhere D[t] is an outlier dummy for one quarter. Here, h = 4 and p = 1.\nThe model uses the same lags, AR terms, and outlier dummy for estimation.\n'




```python
import matplotlib.pyplot as plt
import pandas as pd

from sc_midas.midas import MIDAS
from sc_midas.utils import sample_data
```


```python
target, regressors = sample_data(n_obs=500, n_lags=6, seed=0, horizon=4, n_ar_lags=1)

# Inject a large positive outlier at one quarter and model it explicitly
# with ``dummy_periods`` so it does not contaminate the weight and AR estimates.
outlier_date = pd.Timestamp(target["date"].iloc[300])
target.loc[target["date"] == outlier_date, "value"] += 10.0

# Train/test split (hold out last 12 quarters for forecast evaluation)
T_train = len(target) - 12
target_train = target.iloc[:T_train]
regressors_train = regressors[regressors["date"] <= target_train["date"].iloc[-1]]
```


```python
# Create a MIDAS model with multiple direct-forecast horizons.
# Pass ``n_ar_lags`` to additionally regress on lags of the dependent variable,
# turning the model into an AR-augmented MIDAS:
#   y[t+h] = alpha + beta * X[t]'w + sum_{k=1..p} phi_k * y[t+h-k]
#            + delta * D[t+h] + eps
model = MIDAS(
    method="almon",
    n_lags=6,
    estimator="ols",
    horizons=[0, 4],
    n_ar_lags=1,  # include y[t+h-1] as extra regressor
    dummy_periods=[outlier_date],  # model the outlier as a one-off dummy
)
```


```python
# Fit the model; it shifts each horizon internally.
model.fit(target_train, regressors_train)
```




    <sc_midas.midas.MIDAS at 0x1bec29e74d0>




```python
# Build a multi-horizon forecast table with one row per horizon.
# fc_table = model.forecast(regressors)
# print(fc_table)
```


```python
# Plot the fitted h=0 model
model.plot_fit()
model.plot_weights()

plt.show()
```




```python
# Print a formatted summary of every fitted horizon.
model.summary()
```

    ====================================================
             MIDAS Regression Results
    ====================================================
      Method / Estimator : almon / OLS
      Lags (n_lags)      : 6
      Horizon            : 0
      Observations       : 487
    ----------------------------------------------------
      alpha :  2.725338
      beta  :  1.000000
      theta[0] = -0.051580
      theta[1] =  0.014342
      w[ 0]    = -0.051580
      w[ 1]    = -0.037238
      w[ 2]    = -0.022896
      w[ 3]    = -0.008555
      w[ 4]    =  0.005787
      w[ 5]    =  0.020128
      gamma[0] =  8.278060
      phi[1]   = -0.008524
    ----------------------------------------------------
      SSE  :  316.607858
      RMSE :  0.806299
    ====================================================
    ====================================================
             MIDAS Regression Results
    ====================================================
      Method / Estimator : almon / OLS
      Lags (n_lags)      : 6
      Horizon            : 4
      Observations       : 483
    ----------------------------------------------------
      alpha :  2.204835
      beta  :  1.000000
      theta[0] =  0.393128
      theta[1] = -0.103609
      w[ 0]    =  0.393128
      w[ 1]    =  0.289519
      w[ 2]    =  0.185909
      w[ 3]    =  0.082300
      w[ 4]    = -0.021309
      w[ 5]    = -0.124918
      gamma[0] =  9.607761
      phi[1]   =  0.208125
    ----------------------------------------------------
      SSE  :  153.158357
      RMSE :  0.563115
    ====================================================
