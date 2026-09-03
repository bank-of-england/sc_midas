# nowcast-midas

**`nowcast-midas`** — a Python implementation of the Staggered-Combination MIDAS
(SC-MIDAS) model.

The package combines:

* **Mixed-Frequency Data Sampling (MIDAS)** regressions for monthly
  indicators of a quarterly target.
* **Plain OLS** regressions for quarterly regressors.
* A **hierarchical combination layer** that pools indicator forecasts
  through error-weighted averaging and a constrained regression step.

## What is MIDAS?

MIDAS regression is a framework for regressing a low-frequency variable
$y_t$ on high-frequency predictors **without** aggregating them down to
the low frequency.  Each of the $K$ high-frequency lags receives its own
weight $w(j;\theta)$ controlled by a small number of shape parameters,
so the lag profile is learnt from the data:

$$
y_t \;=\; \alpha \;+\; \beta \sum_{j=0}^{K-1} w(j;\theta)\, x_{t,j}
   \;+\; \varepsilon_t
$$

## Why combine?

A single indicator rarely captures the full picture.  Soft survey data
moves early, hard activity data is more accurate but lags.  SC-MIDAS
addresses this with a two-layer combination (the methods pages call these
Layer 1 and Layer 2):

1. A **soft combo** (Layer 1) pools many MIDAS indicators with error-weighted
   weights — fast-moving signal.
2. A **final combo** (Layer 2) uses constrained regression to merge the soft
   combo with a quarterly regressor (typically a partial release of GDP),
   yielding a single best-estimate nowcast for each forecast horizon.

## Quick start

```python
from nowcast_midas import ComboSpec, MidasCombo, MidasSpec, OLSSpec
from nowcast_midas.utils import sample_combo_data

# Simulated mixed-frequency data: three monthly series, one quarterly
# regressor, one quarterly target, plus one injected outlier.
target_df, regressors_df, info = sample_combo_data(n_quarters=60, seed=42)
outlier = info["outlier_date"]

midas_monthly_1 = MidasSpec(
    "monthly_1", method="almon", n_lags=6, dummy_periods=[outlier]
)
midas_monthly_2 = MidasSpec(
    "monthly_2", method="almon", n_lags=6, dummy_periods=[outlier]
)
midas_monthly_3 = MidasSpec(
    "monthly_3", method="unrestricted", n_lags=3, dummy_periods=[outlier]
)
ols_quarterly_1 = OLSSpec("quarterly_1", n_lags=1, dummy_periods=[outlier])

soft = ComboSpec(
    "soft",
    sources=[midas_monthly_1, midas_monthly_2, midas_monthly_3],
    method="mse",  # inverse mean-squared-error weights — see methods/combo.md
    window=8,
    discount_rate=0.95,
)
final = ComboSpec("final", sources=[soft, ols_quarterly_1], method="regression")

model = MidasCombo(combo_specs=final, horizons=3)  # horizons is a COUNT, not an index
model.fit(target=target_df, regressors=regressors_df)

forecasts = model.forecast()  # long format: one row per (spec, horizon step)
print(forecasts.head())
model.summary(horizon=0)  # prints and returns the text
```

> This is the same example as the project README. Runnable end-to-end scripts are
> in [Worked examples](examples/core_models.md).

## Where to go next

Working with a **single indicator**? Start with
[User Guide — MIDAS model](guide/midas_model.md). Building the **full
combination pipeline**? Go to [User Guide — SC-MIDAS pipeline](guide/sc_midas.md),
then [Worked examples](examples/core_models.md) for runnable end-to-end scripts.

* **[User Guide — MIDAS model](guide/midas_model.md)** — the
  single-indicator estimator: weighting schemes, OLS/NLS, AR lags,
  dummies, multi-horizon estimation and forecasting.
* **[User Guide — SC-MIDAS pipeline](guide/sc_midas.md)** — the
  high-level `MidasCombo`: input layout, spec tree, fitting,
  combination weights and the long-format OOS forecast table.
* **[User Guide — Real-time analysis](guide/realtime.md)** — running
  the pipeline on vintage data without look-ahead bias.
* **[User Guide — Interpreting decompositions](guide/decomposition.md)** —
  reading the `forecast_decomp()` output and the real-time news tables.
* **[Methods — SC-MIDAS framework](methods/sc_midas_framework.md)** —
  the maths behind the two-layer combination.
* **[Methods — Weighting schemes](methods/combo.md)** — equal, inverse-
  error, and constrained-regression combination weights.
* **[API Reference](api.md)** — every public symbol.
