# Real-time analysis

Real-time forecasting uses the `forecast-realtime` package to refit a model
at each historical information date. Each fit uses only observations
available at that date, so the resulting forecast sequence avoids look-ahead
bias.

This guide uses sampled data. It assigns a synthetic release date to each
observation, then passes the resulting vintage table to
`forecast_realtime`. Replace the sampled frames with a real-time database
when applying the workflow to published data.

Install the integration packages before running the examples:

```bash
pip install -e ".[realtime]"
```

## Generate sampled vintage data

`sample_combo_data()` returns a quarterly target and monthly and quarterly
regressors. The code below adds a one-month publication lag to create a
repeatable synthetic vintage schedule.

```python
import pandas as pd

import forecast_evaluation as fe
import forecast_realtime as rt
import news_decomp as nd
from nowcast_midas.utils import sample_combo_data

target, regressors, _ = sample_combo_data(
    n_quarters=60,
    n_lags=6,
    seed=42,
)

target["vintage_date"] = target["date"] + pd.offsets.MonthEnd(1)
target["metric"] = "pop"
regressors["vintage_date"] = regressors["date"] + pd.offsets.MonthEnd(1)
regressors["metric"] = "pop"

outturns_data = pd.concat([target, regressors], ignore_index=True)
outturns_data["frequency"] = outturns_data["frequency"].replace({"QE": "Q", "ME": "M"})
data = fe.NowcastData(outturns_data=outturns_data)
```

The sampler uses `QE` and `ME` frequency codes. The real-time integration
expects the equivalent `Q` and `M` codes, so the adapter translates them in
the copied vintage table.

The returned frames use these columns:

| frame | columns |
|---|---|
| `target` | `date`, `variable`, `frequency`, `value`, `vintage_date`, `metric` |
| `regressors` | `date`, `variable`, `frequency`, `value`, `vintage_date`, `metric` |
| `outturns_data` | both frames concatenated into one vintage table |

## Define the model

Define the same MIDAS specification for every vintage. The example forecasts
the quarterly target from one monthly indicator.

## Refit at each vintage

`ForecastMIDAS` wraps `MIDAS` for the `forecast_realtime.RealTimeModel`
runner. The runner filters the vintage table, refits the model, and stores each
forecast under `label`.

```python
model = rt.models.ForecastMIDAS(
    formula="quarterly_target ~ monthly_1",
    method="almon",
    n_lags=6,
    estimator="ols",
    horizons=[0, 1],
)
rt_model = rt.RealTimeModel(models=model, data=data)

rt_model.forecast(
    y_variables=["quarterly_target"],
    X_variables=["monthly_1"],
    data_transformation={
        "quarterly_target": "pop",
        "monthly_1": "pop",
    },
    label="sampled sc-midas",
    first_vintage=target["vintage_date"].iloc[24].strftime("%Y-%m-%d"),
    last_vintage=target["vintage_date"].iloc[-2].strftime("%Y-%m-%d"),
    reconstruct_levels=False,
    decomp=True,
)
```

The runner stores one forecast for the MIDAS model at each sampled information
date. The `date` column identifies the forecast target, while
`vintage_date` identifies the information date used for the fit.

Pass the resulting decomposition table to `news_decomp` for a news summary:

```python
news = nd.NewsData(rt_model.decompositions)
news.summary()
```

Use the same sampled frames to compare model specifications, horizons, and
combination methods without relying on external data or services.
