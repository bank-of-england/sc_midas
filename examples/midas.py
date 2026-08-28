"""Fit the core SC-MIDAS models with sampled mixed-frequency data."""

from nowcast_midas.midas import MIDAS
from nowcast_midas.midas_combo import MidasCombo
from nowcast_midas.multi_midas import MultiMIDAS
from nowcast_midas.specs import ComboSpec, MidasSpec, OLSSpec, VariableSpec
from nowcast_midas.utils import sample_combo_data


def main() -> None:
    target, regressors, info = sample_combo_data(
        n_quarters=60,
        n_lags=6,
        seed=42,
    )
    outlier_date = info["outlier_date"]
    target_train = target.iloc[:-8]
    regressors_train = regressors[regressors["date"] <= target_train["date"].iloc[-1]]

    midas = MIDAS(
        method="almon",
        n_lags=6,
        estimator="ols",
        horizons=[0, 1],
        dummy_periods=[outlier_date],
    )
    monthly_1 = regressors_train[regressors_train["variable"] == "monthly_1"]
    midas.fit(target_train[["date", "value"]], monthly_1[["date", "value"]])
    print("MIDAS forecast:")
    print(midas.forecast(monthly_1[["date", "value"]]))

    multi = MultiMIDAS(
        variables=[
            VariableSpec("monthly_1", method="exp_almon", n_lags=6),
            VariableSpec("monthly_2", method="almon", n_lags=6),
            VariableSpec("quarterly_1", frequency="QE", n_lags=1),
        ],
        horizons=[0, 1],
    )
    multi.fit(
        target_train[["date", "value"]],
        regressors_train[["date", "variable", "value"]],
    )
    print("\nMultiMIDAS forecast:")
    print(multi.forecast(regressors[["date", "variable", "value"]]))

    soft_combo = ComboSpec(
        name="soft_combo",
        sources=[
            MidasSpec("monthly_1", method="almon", n_lags=6),
            MidasSpec("monthly_2", method="almon", n_lags=6),
        ],
        method="mse",
        window=8,
        discount_rate=0.95,
    )
    final_combo = ComboSpec(
        name="final_combo",
        sources=[
            soft_combo,
            OLSSpec("quarterly_1", n_lags=1, dummy_periods=[outlier_date]),
        ],
        method="regression",
    )
    combo = MidasCombo(combo_specs=final_combo, horizons=1)
    combo.fit(target_train, regressors_train)
    print("\nMidasCombo forecast:")
    print(combo.forecast())


if __name__ == "__main__":
    main()
