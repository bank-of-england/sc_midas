"""Tests for the quarterly OLS class."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sc_midas.ols import OLS, FittedOLS

ATOL = 0.05


def _simulate_ols(
    n_obs: int = 200,
    alpha: float = 1.5,
    coef: list[float] | None = None,
    noise: float = 0.1,
    seed: int = 0,
    horizon: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, float, np.ndarray]:
    """Generate quarterly target and regressor following

        y[t+h] = alpha + sum_k coef[k] * x[t-k] + noise

    so that the OLS model (which pairs ``y[t+h]`` with ``X[t]`` where
    ``X[t, k] = x[t-k]``) recovers ``(alpha, coef)``.

    Returns target/regressor DataFrames plus the truth ``alpha`` and ``coef``.
    """
    rng = np.random.default_rng(seed)
    coef_arr = np.asarray([2.0] if coef is None else coef, dtype=float)
    n_lags = len(coef_arr)

    dates = pd.date_range("2000-03-31", periods=n_obs, freq="QE")
    x = rng.standard_normal(n_obs)  # contemporaneous regressor at each quarter

    # Fill y with pure noise everywhere first (so target has no NaNs even at
    # the leading rows where X has missing lags / horizon shift makes the
    # signal undefined). The model only uses rows where X is fully populated.
    y = alpha + noise * rng.standard_normal(n_obs)
    for s in range(n_obs):
        t = s - horizon  # corresponding X row index
        if n_lags - 1 <= t < n_obs:
            signal = alpha + float(np.dot(coef_arr, x[t - n_lags + 1 : t + 1][::-1]))
            y[s] = signal + noise * rng.standard_normal()

    target = pd.DataFrame({"date": dates, "value": y})
    regressors = pd.DataFrame({"date": dates, "value": x})
    return target, regressors, alpha, coef_arr


def _quarterly_fixture(n_obs: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-03-31", periods=n_obs, freq="QE")
    x = np.arange(1.0, n_obs + 1.0)
    target = pd.DataFrame({"date": dates, "value": 1.0 + 2.0 * x})
    regressors = pd.DataFrame({"date": dates, "value": x})
    return target, regressors


class TestOLSEstimation:
    """OLS should recover DGP intercept and coefficients."""

    def test_recover_intercept_and_slope_h0(self):
        target, regressors, alpha, coef = _simulate_ols(
            n_obs=400, alpha=1.5, coef=[2.0], noise=0.05, seed=1
        )
        model = OLS(n_lags=1).fit(target, regressors)
        fit = model.fits_[0]

        assert isinstance(fit, FittedOLS)
        np.testing.assert_allclose(fit.intercept, alpha, atol=ATOL)
        np.testing.assert_allclose(fit.coef, coef, atol=ATOL)
        assert fit.nobs == 400

    def test_recover_multi_lag_coefficients(self):
        true_coef = [1.0, -0.5, 0.3]
        target, regressors, alpha, coef = _simulate_ols(
            n_obs=600, alpha=0.7, coef=true_coef, noise=0.05, seed=2
        )
        model = OLS(n_lags=len(true_coef)).fit(target, regressors)
        fit = model.fits_[0]

        np.testing.assert_allclose(fit.intercept, alpha, atol=ATOL)
        np.testing.assert_allclose(fit.coef, coef, atol=ATOL)

    def test_fitted_values_match_design(self):
        target, regressors, _, _ = _simulate_ols(
            n_obs=200, coef=[1.5], noise=0.1, seed=3
        )
        model = OLS(n_lags=1).fit(target, regressors)
        fit = model.fits_[0]

        # residuals + fitted == y_valid
        y_valid = target["value"].to_numpy()[model.valid_mask_]
        np.testing.assert_allclose(fit.fitted_values + fit.residuals, y_valid)


class TestOLSHorizons:
    """Direct-forecasting alignment ``y[t+h] ~ X[t]`` at multiple horizons."""

    def test_horizon_dgp_recovery(self):
        """At h=2 the model should recover the DGP used at lead 2."""
        h = 2
        target, regressors, alpha, coef = _simulate_ols(
            n_obs=400, alpha=2.0, coef=[1.0], noise=0.05, seed=4, horizon=h
        )
        model = OLS(n_lags=1, horizons=[h]).fit(target, regressors)
        fit = model.fits_[h]

        np.testing.assert_allclose(fit.intercept, alpha, atol=ATOL)
        np.testing.assert_allclose(fit.coef, coef, atol=ATOL)

    def test_multi_horizon_fit_lengths(self):
        target, regressors, _, _ = _simulate_ols(n_obs=120, noise=0.1, seed=5)
        horizons = [0, 1, 2, 3]
        model = OLS(n_lags=1, horizons=horizons).fit(target, regressors)

        assert set(model.fits_.keys()) == set(horizons)
        # At horizon h: T_valid - h observations.
        T_valid = int(model.valid_mask_.sum())
        for h in horizons:
            assert model.fits_[h].nobs == T_valid - h

    def test_only_specified_horizons_fitted(self):
        target, regressors, _, _ = _simulate_ols(n_obs=80, seed=6)
        model = OLS(n_lags=1, horizons=[1, 3]).fit(target, regressors)
        assert set(model.fits_.keys()) == {1, 3}


class TestOLSDummies:
    """Outlier dummies should absorb spikes and leave the slope intact."""

    def test_dummy_absorbs_outlier(self):
        target, regressors, alpha, coef = _simulate_ols(
            n_obs=200, alpha=1.0, coef=[2.0], noise=0.05, seed=7
        )
        # Inject a large outlier in the target at one quarter.
        spike_date = pd.Timestamp("2010-03-31")
        spike_idx = int(
            np.where(target["date"].values == spike_date.to_datetime64())[0][0]
        )
        OUTLIER = 50.0
        target.loc[spike_idx, "value"] += OUTLIER

        model = OLS(n_lags=1, dummy_periods=[spike_date]).fit(target, regressors)
        fit = model.fits_[0]

        np.testing.assert_allclose(fit.intercept, alpha, atol=ATOL)
        np.testing.assert_allclose(fit.coef, coef, atol=ATOL)
        # Dummy coefficient should recover the injected magnitude.
        assert len(fit.gamma) == 1
        np.testing.assert_allclose(fit.gamma[0], OUTLIER, atol=ATOL)


class TestOLSValidation:
    """Argument validation and error paths."""

    def test_n_lags_must_be_positive(self):
        with pytest.raises(ValueError, match="n_lags"):
            OLS(n_lags=0)

    def test_horizon_too_large_raises(self):
        target, regressors, _, _ = _simulate_ols(n_obs=10, seed=8)
        with pytest.raises(ValueError, match="horizon"):
            OLS(horizons=[20]).fit(target, regressors)

    def test_target_with_nan_raises(self):
        target, regressors, _, _ = _simulate_ols(n_obs=50, seed=9)
        target.loc[5, "value"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            OLS().fit(target, regressors)

    def test_target_column_check(self):
        target, regressors, _, _ = _simulate_ols(n_obs=50, seed=10)
        with pytest.raises(ValueError, match="'date' and 'value'"):
            OLS().fit(target.rename(columns={"value": "v"}), regressors)

    def test_infinite_target_and_regressor_rows_are_excluded(self):
        target, regressors = _quarterly_fixture()
        target.loc[3, "value"] = np.inf
        regressors.loc[5, "value"] = np.inf

        model = OLS(n_lags=1).fit(target, regressors)

        assert not model.valid_mask_[3]
        assert not model.valid_mask_[5]
        assert model.fits_[0].nobs == len(target) - 2
        assert np.all(np.isfinite(model.fits_[0].fitted_values))
        assert np.all(np.isfinite(model.fits_[0].residuals))

    @pytest.mark.parametrize("replacement", [np.nan, np.inf])
    def test_interior_nonfinite_quarter_is_not_compressed(self, replacement):
        target, regressors = _quarterly_fixture()
        regressors.loc[4, "value"] = replacement

        model = OLS(n_lags=2).fit(target, regressors)

        assert not model.valid_mask_[4]
        assert not model.valid_mask_[5]
        assert model.valid_mask_[6]

    def test_missing_calendar_quarter_remains_an_incomplete_lag(self):
        target, regressors = _quarterly_fixture()
        regressors = regressors.drop(index=4).reset_index(drop=True)

        model = OLS(n_lags=2).fit(target, regressors)

        assert not model.valid_mask_[4]
        assert not model.valid_mask_[5]
        assert model.valid_mask_[6]


class TestOLSForecastDecomp:
    """Forecast decomposition: contributions must sum to the forecast value."""

    def test_columns_and_sum_to_forecast(self):
        target, regressors, _, _ = _simulate_ols(
            n_obs=200, coef=[2.0], noise=0.05, seed=11
        )
        model = OLS(n_lags=1, horizons=[0, 1, 2]).fit(target, regressors)

        fc = model.forecast(regressors)
        decomp = model.forecast_decomp(regressors, regressor_name="x")

        assert list(decomp.columns) == [
            "horizon",
            "date",
            "component",
            "contribution",
            "weight",
        ]
        for h in [0, 1, 2]:
            s = decomp.loc[decomp["horizon"] == h, "contribution"].sum()
            f = fc.loc[fc["horizon"] == h, "forecast"].iloc[0]
            np.testing.assert_allclose(s, f, atol=1e-9)

        # intercept carries weight 1.0; the single regressor carries its coef.
        intercept = decomp.loc[decomp["component"] == "intercept"]
        assert (intercept["weight"] == 1.0).all()
        block = decomp.loc[decomp["component"] == "x"]
        np.testing.assert_allclose(block["weight"].iloc[0], model.fits_[0].coef[0])

    def test_multi_lag_components(self):
        target, regressors, _, _ = _simulate_ols(
            n_obs=300, coef=[1.0, -0.5, 0.3], noise=0.05, seed=12
        )
        model = OLS(n_lags=3, horizons=[0, 1]).fit(target, regressors)

        fc = model.forecast(regressors)
        decomp = model.forecast_decomp(regressors, regressor_name="x")

        components = set(decomp["component"])
        assert {"x_lag0", "x_lag1", "x_lag2"}.issubset(components)
        for h in [0, 1]:
            s = decomp.loc[decomp["horizon"] == h, "contribution"].sum()
            f = fc.loc[fc["horizon"] == h, "forecast"].iloc[0]
            np.testing.assert_allclose(s, f, atol=1e-8)

    def test_sum_to_forecast_with_ar_lags(self):
        target, regressors, _, _ = _simulate_ols(
            n_obs=200, coef=[1.5], noise=0.05, seed=13
        )
        model = OLS(n_lags=1, horizons=[0, 1], n_ar_lags=2).fit(target, regressors)

        fc = model.forecast(regressors)
        decomp = model.forecast_decomp(regressors, regressor_name="x")

        components = set(decomp["component"])
        assert {"ar_lag1", "ar_lag2"}.issubset(components)
        for h in [0, 1]:
            s = decomp.loc[decomp["horizon"] == h, "contribution"].sum()
            f = fc.loc[fc["horizon"] == h, "forecast"].iloc[0]
            np.testing.assert_allclose(s, f, atol=1e-8)

    @pytest.mark.parametrize("trailing_value", [np.nan, np.inf])
    def test_trailing_nonfinite_rows_do_not_advance_forecast_origin(
        self, trailing_value
    ):
        target, regressors = _quarterly_fixture()
        model = OLS(n_lags=2, horizons=[0, 1, 2]).fit(target, regressors)
        extended = pd.concat(
            [
                regressors,
                pd.DataFrame(
                    {
                        "date": [
                            pd.Timestamp("2022-12-31"),
                            pd.Timestamp("2023-03-31"),
                        ],
                        "value": [trailing_value, trailing_value],
                    }
                ),
            ],
            ignore_index=True,
        )

        forecast = model.forecast(extended)
        decomposition = model.forecast_decomp(extended)

        expected_dates = [
            pd.Timestamp("2022-06-30"),
            pd.Timestamp("2022-09-30"),
            pd.Timestamp("2022-12-31"),
        ]
        assert list(pd.to_datetime(forecast["date"])) == expected_dates
        assert set(pd.to_datetime(decomposition["date"])) == set(expected_dates)
        for horizon in [0, 1, 2]:
            decomposition_sum = decomposition.loc[
                decomposition["horizon"] == horizon, "contribution"
            ].sum()
            forecast_value = forecast.loc[
                forecast["horizon"] == horizon, "forecast"
            ].iloc[0]
            np.testing.assert_allclose(decomposition_sum, forecast_value)
