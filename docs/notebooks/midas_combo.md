```python
"""MIDAS combination pipeline example.

Demonstrates the full workflow:
  1. Simulate quarterly target + monthly / quarterly indicators via
     :func:`~sc_midas.utils.sample_combo_data` (with a one-off outlier).
  2. Define ``MidasSpec`` (with optional period dummies) for the monthly
     indicators, and ``OLSSpec`` for the quarterly regressor.
  3. Build a two-level ``ComboSpec`` hierarchy:
        soft_combo  = MSE-weighted average of monthly indicators
        final_combo = constrained regression of (soft_combo, hard).
  4. Fit with ``MidasCombo``, generate out-of-sample forecasts via
    ``model.forecast()``, and inspect ``fits_df_``, ``forecasts_df_``, and
    ``combo_weights_``.
"""
```

    'MIDAS combination pipeline example.\n\nDemonstrates the full workflow:\n  1. Simulate quarterly target + monthly / quarterly indicators via\n     :func:`~sc_midas.utils.sample_combo_data` (with a one-off outlier).\n  2. Define ``MidasSpec`` (with optional period dummies) for the monthly\n     indicators, and ``OLSSpec`` for the quarterly regressor.\n  3. Build a two-level ``ComboSpec`` hierarchy:\n        soft_combo  = MSE-weighted average of monthly indicators\n        final_combo = constrained regression of (soft_combo, hard).\n  4. Fit with ``MidasCombo``, generate out-of-sample forecasts via\n     ``model.forecast()``, and inspect ``fits_df_``, ``forecasts_df_``, and\n     ``combo_weights_``.\n'




```python
import pandas as pd
import matplotlib.pyplot as plt

from sc_midas.midas_combo import MidasCombo
from sc_midas.specs import ComboSpec, MidasSpec, OLSSpec
from sc_midas.utils import sample_combo_data
```


```python
# 1.  Simulate data
# ---------------------------------------------------------------------
target, regressors, INFO = sample_combo_data(n_quarters=60, seed=42)
OUTLIER_DATE = INFO["outlier_date"]
```


```python
# 2.  Define specs
# ---------------------------------------------------------------------
# Period dummies for the outlier quarter — both the MIDAS and OLS legs
# absorb the shock through their respective `gamma` coefficients.
DUMMIES = [OUTLIER_DATE]

# Indicator specs — passed directly as ComboSpec sources, no need to
# also list them on MidasCombo.
midas_monthly_1 = MidasSpec(
    "monthly_1", method="almon", n_lags=5, dummy_periods=DUMMIES
)
midas_monthly_2 = MidasSpec(
    "monthly_2", method="almon", n_lags=5, dummy_periods=DUMMIES
)
midas_monthly_3 = MidasSpec(
    "monthly_3", method="unrestricted", n_lags=3, dummy_periods=DUMMIES
)
ols_quarterly_1 = OLSSpec("quarterly_1", n_lags=1, dummy_periods=DUMMIES)

# Layer 1 — MSE-weighted combination of the soft indicators.
soft_combo = ComboSpec(
    name="soft_combo",
    sources=[midas_monthly_1, midas_monthly_2, midas_monthly_3],
    method="mse",
    window=8,
    discount_rate=0.95,
)

# Layer 2 — constrained regression of (soft_combo, quarterly_1) onto the target.
final_combo = ComboSpec(
    name="final_combo",
    sources=[soft_combo, ols_quarterly_1],
    method="regression",
    window=None,
)
```


```python
# 3.  Fit
# ---------------------------------------------------------------------
model = MidasCombo(combo_specs=final_combo, horizons=1)
model.fit(target=target, regressors=regressors)
```




    <sc_midas.midas_combo.MidasCombo at 0x1c7edf3b620>




```python
# 4.  Forecast and inspect
# ---------------------------------------------------------------------
oos = model.forecast()
print(oos)

print(model.summary(horizon=0))
```

       horizon         spec     value                 date
    2        0     monthly_3  0.727442  2025-03-31 00:00:00
    1        0     monthly_2  1.747390  2025-03-31 00:00:00
    0        0     monthly_1  2.314327  2025-03-31 00:00:00
    3        0    quarterly_1  0.921705  2025-03-31 00:00:00
    5        0  final_combo  1.803456  2025-03-31 00:00:00
    4        0   soft_combo  1.803456  2025-03-31 00:00:00
    ============================================================
           MIDAS Combination Pipeline Results - Horizon 0
    ============================================================

      Individual MIDAS Models:
    ------------------------------------------------------------
        monthly_1             method=almon         n_lags=5  RMSE=0.9546
        monthly_2             method=almon         n_lags=5  RMSE=1.0420
        monthly_3             method=unrestricted  n_lags=3  RMSE=1.1236

      Combinations:
    ------------------------------------------------------------
        soft_combo            method=mse           sources=[monthly_1, monthly_2, monthly_3]  RMSE=0.9527
          Latest weights:
            monthly_1: 0.4724
            monthly_2: 0.3200
            monthly_3: 0.2076
          final_combo           method=regression    sources=[soft_combo, quarterly_1]  RMSE=0.9627
          Latest weights:
            soft_combo: 1.0000
            quarterly_1: 0.0000
    ============================================================



```python
# 5.  Plot the fitted values, forecast, and weights.
# ---------------------------------------------------------------------
HORIZON = 0
dates_q = target["date"].to_numpy()
gdp = target["value"].to_numpy()
T = len(dates_q)

insample_soft = model.fitted_["soft_combo"][HORIZON][:T]
insample_final = model.fitted_["final_combo"][HORIZON][:T]

# `oos` is long-format [date, horizon, value, spec]: y[T+h+1] is predicted
# by the horizon-`h` model (the only diagonal cell that doesn't require
# future regressor values).
final_oos = oos[oos["spec"] == "final_combo"].sort_values("horizon")
forecast_dates = pd.to_datetime(final_oos["date"]).to_list()
forecast_vals = final_oos["value"].to_list()

soft_weights = model.combo_weights_["soft_combo"][HORIZON]

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# --- Plot 1: in-sample fits ---
ax = axes[0]
ax.plot(dates_q, gdp, "k.-", lw=1.3, label="GDP (actual)")
ax.plot(dates_q, insample_soft, "--", lw=1.2, label="soft_combo (MSE)")
ax.plot(dates_q, insample_final, "-", lw=2.0, label="final_combo (regression)")
ax.axvline(OUTLIER_DATE, color="red", linestyle=":", lw=1.0, label="outlier dummy")
ax.set_title("In-sample: GDP vs Combination Fitted Values")
ax.set_ylabel("GDP")
ax.legend()

# --- Plot 2: in-sample + OOS forecast ---
ax = axes[1]
ax.plot(dates_q, gdp, "k.-", lw=1.3, label="GDP (actual)")
ax.plot(dates_q, insample_final, "-", lw=2.0, label="final_combo (in-sample)")
ax.plot(
    forecast_dates,
    forecast_vals,
    "o--",
    lw=2.0,
    color="red",
    label="final_combo (forecast)",
)
ax.axvline(dates_q[-1], color="grey", linestyle=":", lw=1.0)
ax.set_title("In-sample + Forecast: final_combo")
ax.set_ylabel("GDP")
ax.legend()

# --- Plot 3: soft_combo weights over time ---
ax = axes[2]
for src, w in soft_weights.items():
    ax.plot(dates_q, w[:T], label=src)
ax.set_title("soft_combo: MSE-Weighted Indicator Weights (h=0)")
ax.set_ylabel("Weight")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()
```
