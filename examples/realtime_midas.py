"""Run the SC-MIDAS combination through ``forecast-realtime``.

This example requires the ``forecast-realtime`` package and its supporting
integration packages. Install them with ``pip install -e '.[realtime]'``
before running the example. It creates synthetic vintage metadata from sampled
data and does not require a real-time database.
"""

import forecast_evaluation as fe
import forecast_realtime as rt
import news_decomp as nd
import pandas as pd

from sc_midas.utils import sample_combo_data


def main() -> None:
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
    outturns_data["frequency"] = outturns_data["frequency"].replace(
        {"QE": "Q", "ME": "M"}
    )
    data = fe.NowcastData(outturns_data=outturns_data)

    model = rt.models.ForecastMIDAS(
        formula="quarterly_target ~ monthly_1",
        method="almon",
        n_lags=6,
        estimator="ols",
        horizons=[0, 1],
    )
    rt_model = rt.RealTimeModel(models=model, data=data)
    first_vintage = target["vintage_date"].iloc[24].strftime("%Y-%m-%d")
    last_vintage = target["vintage_date"].iloc[-2].strftime("%Y-%m-%d")

    rt_model.forecast(
        y_variables=["quarterly_target"],
        X_variables=["monthly_1"],
        data_transformation={
            "quarterly_target": "pop",
            "monthly_1": "pop",
        },
        label="sampled sc-midas",
        first_vintage=first_vintage,
        last_vintage=last_vintage,
        reconstruct_levels=False,
        decomp=True,
    )

    news = nd.NewsData(rt_model.decompositions)
    news.summary()


if __name__ == "__main__":
    main()
