"""Independent real time validation tests for :class:`nowcast_midas.MIDAS`."""

import numpy as np
import pandas as pd
import pytest

import nowcast_midas.midas as midas_module
from nowcast_midas import temporal_weights
from nowcast_midas.midas import MIDAS
from tests.midas.sample_midas import (
    _beta_weights,
    sample_midas,
    sample_vintage_midas,
)

METHODS = ("almon", "unrestricted", "exp_almon", "beta")
PARAMETER_TOLERANCES = {
    "almon": {"alpha": 2e-5, "coefficients": 2e-5},
    "unrestricted": {"alpha": 2e-5, "coefficients": 2e-5},
    "exp_almon": {"alpha": 2e-5, "coefficients": 1e-2},
    "beta": {"alpha": 2e-5, "coefficients": 1e-2},
}
EXPECTED_RELEASE_SCHEDULE = {
    pd.Timestamp("2023-11-30"): pd.Timestamp("2023-12-05"),
    pd.Timestamp("2023-12-31"): pd.Timestamp("2023-12-15"),
    pd.Timestamp("2024-01-31"): pd.Timestamp("2024-01-15"),
    pd.Timestamp("2024-02-29"): pd.Timestamp("2024-02-15"),
    pd.Timestamp("2024-03-31"): pd.Timestamp("2024-03-15"),
    pd.Timestamp("2024-04-30"): pd.Timestamp("2024-05-05"),
}
EXPECTED_PUBLISHED_REFERENCE_DATES = {
    pd.Timestamp("2023-12-20"): pd.date_range("2018-01-31", "2023-12-31", freq="ME"),
    pd.Timestamp("2024-01-15"): pd.date_range("2018-01-31", "2024-01-31", freq="ME"),
    pd.Timestamp("2024-02-15"): pd.date_range("2018-01-31", "2024-02-29", freq="ME"),
    pd.Timestamp("2024-03-15"): pd.date_range("2018-01-31", "2024-03-31", freq="ME"),
}
EXPECTED_NEXT_UNPUBLISHED = {
    pd.Timestamp("2023-12-20"): (
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-01-15"),
    ),
    pd.Timestamp("2024-01-15"): (
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-02-15"),
    ),
    pd.Timestamp("2024-02-15"): (
        pd.Timestamp("2024-03-31"),
        pd.Timestamp("2024-03-15"),
    ),
    pd.Timestamp("2024-03-15"): (
        pd.Timestamp("2024-04-30"),
        pd.Timestamp("2024-05-05"),
    ),
}


def _fit_stage(sample, vintage, method):
    model = MIDAS(
        method=method,
        n_lags=sample.truth.n_lags,
        n_pars_weights=sample.truth.n_pars_weights,
        horizons=[vintage.horizon],
    )
    model.fit(vintage.target_train, vintage.regressors)
    return model


def test_vintage_contains_only_published_observations():
    sample = sample_midas()
    observed_releases = sample.observed.set_index("date")["release_date"]

    for reference_date, release_date in EXPECTED_RELEASE_SCHEDULE.items():
        assert observed_releases.loc[reference_date] == release_date

    for vintage in sample.vintages:
        expected_dates = EXPECTED_PUBLISHED_REFERENCE_DATES[vintage.vintage_date]
        np.testing.assert_array_equal(
            vintage.published["date"].to_numpy(), expected_dates.to_numpy()
        )
        next_reference, next_release = EXPECTED_NEXT_UNPUBLISHED[vintage.vintage_date]
        assert next_reference not in set(vintage.published["date"])
        assert observed_releases.loc[next_reference] == next_release


def test_sampler_covers_survey_lead_and_ordinary_release_lags():
    sample = sample_midas()
    observed_releases = sample.observed.set_index("date")["release_date"]

    survey_dates = pd.to_datetime(
        ["2023-12-31", "2024-01-31", "2024-02-29", "2024-03-31"]
    )
    survey_leads = [
        (observed_releases.loc[reference_date] - reference_date).days
        for reference_date in survey_dates
    ]
    assert all(-16 <= lag <= -14 for lag in survey_leads)
    assert observed_releases.loc[pd.Timestamp("2024-04-30")] > pd.Timestamp(
        "2024-04-30"
    )


def test_beta_truth_metadata_generates_effective_coefficients():
    sample = sample_midas(method="beta")

    for coefficients in sample.truth.coefficients_by_horizon.values():
        expected_weights = _beta_weights(coefficients.theta_true, sample.truth.n_lags)
        np.testing.assert_allclose(
            coefficients.effective_coefficients,
            coefficients.beta * expected_weights,
            atol=1e-12,
        )


@pytest.mark.parametrize("method", METHODS)
def test_generated_target_matches_independent_calendar_truth(method):
    sample = sample_midas(method=method)

    for target_date, target_value in sample.truth.target.itertuples(index=False):
        lag_row = sample.forecast_truth(
            sample.latent_regressors.loc[
                sample.latent_regressors["date"] <= target_date
            ],
            0,
        )
        np.testing.assert_allclose(
            lag_row,
            target_value,
            atol=2e-5 if method in ("almon", "unrestricted") else 1e-2,
        )


def test_vintage_derives_horizon_from_latest_release():
    sample = sample_midas()
    target_period = sample.truth.target_date.to_period("Q")
    expected_latest_dates = pd.to_datetime(
        ["2023-12-31", "2024-01-31", "2024-02-29", "2024-03-31"]
    )
    expected_horizons = [1, 0, 0, 0]

    for vintage, expected_latest, expected_horizon in zip(
        sample.vintages, expected_latest_dates, expected_horizons
    ):
        latest_date = vintage.regressors.loc[
            np.isfinite(vintage.regressors["value"]), "date"
        ].max()
        latest_period = pd.Timestamp(latest_date).to_period("Q")
        expected = target_period.ordinal - latest_period.ordinal
        assert pd.Timestamp(latest_date) == expected_latest
        assert vintage.latest_regressor_date == expected_latest
        assert vintage.horizon == expected
        assert vintage.horizon == expected_horizon


def test_horizon_transitions_for_one_held_out_target():
    sample = sample_midas()

    assert sample.truth.target_date == pd.Timestamp("2024-03-31")
    assert [v.horizon for v in sample.vintages] == [1, 0, 0, 0]
    assert all(
        pd.Timestamp(vintage.target_train["date"].iloc[-1])
        == pd.Timestamp("2023-12-31")
        for vintage in sample.vintages
    )


def test_lag_row_matches_calendar_oracle_at_each_stage():
    sample = sample_midas()

    for vintage in sample.vintages:
        model = _fit_stage(sample, vintage, "unrestricted")
        expected = vintage.expected_lag_rows
        valid = vintage.expected_valid_rows
        horizon = vintage.horizon
        source_expected = expected if horizon == 0 else expected[:-horizon]
        source_valid = valid if horizon == 0 else valid[:-horizon]

        np.testing.assert_array_equal(model.valid_mask_, valid)
        np.testing.assert_allclose(
            model.fits_[horizon].X,
            source_expected[source_valid],
            equal_nan=True,
        )
    assert any(np.isnan(vintage.expected_lag_rows).any() for vintage in sample.vintages)


@pytest.mark.parametrize(
    ("vintage_date", "expected_anchor"),
    [
        ("2020-02-14", pd.Timestamp("2019-10-31")),
        ("2020-03-14", pd.Timestamp("2019-11-30")),
        ("2020-04-14", pd.Timestamp("2019-12-31")),
    ],
)
def test_vintage_sampler_anchors_each_historical_quarter(vintage_date, expected_anchor):
    sample = sample_vintage_midas(vintage_date)
    model = MIDAS(method="unrestricted", n_lags=len(sample.weights)).fit(
        sample.target_train, sample.regressors
    )
    fit = model.fits_[0]
    row_index = np.flatnonzero(fit.dates == np.datetime64("2019-12-31"))[0]

    expected_dates = pd.DatetimeIndex(
        [
            (expected_anchor.to_period("M") - lag).end_time.normalize()
            for lag in range(len(sample.weights))
        ]
    )
    expected_values = sample.regressors.set_index("date").loc[expected_dates, "value"]
    np.testing.assert_array_equal(fit.X[row_index], expected_values.to_numpy())


@pytest.mark.parametrize("month_within_quarter", [1, 2, 3])
def test_vintage_sampler_filters_by_release_date(month_within_quarter):
    vintage_dates = {1: "2020-02-14", 2: "2020-03-14", 3: "2020-04-14"}
    sample = sample_vintage_midas(vintage_dates[month_within_quarter])
    expected = sample.full_regressors.loc[
        sample.full_regressors["release_date"] <= sample.vintage_date,
        ["date", "value"],
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(sample.regressors, expected)
    assert sample.info_date == expected["date"].max()


@pytest.mark.parametrize("vintage_date", ["2020-02-14", "2020-03-14", "2020-04-14"])
def test_vintage_sampler_recovers_from_matching_ragged_edge(vintage_date):
    sample = sample_vintage_midas(vintage_date)
    model = MIDAS(method="unrestricted", n_lags=len(sample.weights)).fit(
        sample.target_train, sample.regressors
    )
    fit = model.fits_[0]

    np.testing.assert_allclose(fit.alpha, sample.alpha, atol=1e-10)
    np.testing.assert_allclose(fit.weights, sample.weights, atol=1e-10)
    forecast = model.forecast(sample.regressors).iloc[0]
    assert pd.Timestamp(forecast["date"]) == pd.Timestamp("2020-03-31")
    np.testing.assert_allclose(forecast["value"], sample.target.iloc[-1]["value"])


def test_vintage_sampler_rejects_unavailable_months():
    sample = sample_vintage_midas("2020-02-14")
    full_regressors = sample.full_regressors[["date", "value"]]
    model = MIDAS(method="unrestricted", n_lags=len(sample.weights)).fit(
        sample.target_train, full_regressors
    )

    assert not np.allclose(model.fits_[0].weights, sample.weights, atol=1e-3)


@pytest.mark.parametrize("method", METHODS)
def test_parameters_match_truth_at_each_stage(method):
    sample = sample_midas(method=method)

    for vintage in sample.vintages:
        model = _fit_stage(sample, vintage, method)
        fit = model.fits_[vintage.horizon]
        expected_coefficients = sample.truth.coefficients_by_horizon[
            vintage.horizon
        ].effective_coefficients

        tolerances = PARAMETER_TOLERANCES[method]
        np.testing.assert_allclose(
            fit.alpha, sample.truth.alpha, atol=tolerances["alpha"]
        )
        np.testing.assert_allclose(
            fit.beta * fit.weights,
            expected_coefficients,
            atol=tolerances["coefficients"],
        )
        valid = vintage.expected_valid_rows
        source_valid = valid if vintage.horizon == 0 else valid[: -vintage.horizon]
        np.testing.assert_array_equal(model.valid_mask_, valid)
        assert fit.nobs == int(source_valid.sum())
        np.testing.assert_array_equal(
            fit.dates,
            vintage.target_train["date"].to_numpy()[vintage.horizon :][source_valid],
        )


@pytest.mark.parametrize("method", METHODS)
def test_held_out_forecast_matches_truth_at_each_stage(method):
    sample = sample_midas(method=method)

    for vintage in sample.vintages:
        model = _fit_stage(sample, vintage, method)
        forecast = model.forecast(vintage.regressors)
        row = forecast.iloc[0]

        assert pd.Timestamp(row["date"]) == sample.truth.target_date
        np.testing.assert_allclose(row["value"], sample.truth.target_value, atol=2e-5)
        np.testing.assert_allclose(
            vintage.independent_forecast, sample.truth.target_value, atol=2e-5
        )
        np.testing.assert_allclose(
            row["value"], vintage.independent_forecast, atol=2e-5
        )


def test_fitted_dummy_gamma_matches_sampler_truth():
    sample = sample_midas(include_dummy=True)
    vintage = sample.vintages[-1]
    model = MIDAS(
        method="unrestricted",
        n_lags=sample.truth.n_lags,
        horizons=[vintage.horizon],
        dummy_periods=list(sample.truth.dummy_periods),
    ).fit(vintage.target_train, vintage.regressors)

    np.testing.assert_allclose(
        model.fits_[vintage.horizon].gamma,
        sample.truth.dummy_coefficients,
        atol=2e-5,
    )


def test_sampler_oracle_does_not_use_production_weight_dispatcher(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("sampler called production get_weights")

    monkeypatch.setattr(temporal_weights, "get_weights", fail_if_called)

    for method in METHODS:
        sample_midas(method=method)


def test_sample_midas_noise_is_configurable():
    noiseless = sample_midas(noise=0.0)
    noisy = sample_midas(noise=0.1)

    assert np.isfinite(noiseless.truth.target.iloc[0]["value"])
    assert not np.array_equal(
        noiseless.truth.target["value"].to_numpy(),
        noisy.truth.target["value"].to_numpy(),
    )


def test_forecast_with_and_without_active_dummy():
    sample = sample_midas(include_dummy=True)
    dummy_date = sample.truth.dummy_periods[0]
    model = MIDAS(
        method="unrestricted",
        n_lags=sample.truth.n_lags,
        horizons=[1],
        dummy_periods=sample.truth.dummy_periods,
    ).fit(sample.truth.target.iloc[:-1], sample.vintages[-1].regressors)

    active_regressors = sample.regressors_before(dummy_date - pd.offsets.QuarterEnd())
    active = model.forecast(active_regressors).iloc[0]
    inactive_regressors = sample.regressors_before(dummy_date + pd.offsets.QuarterEnd())
    inactive = model.forecast(inactive_regressors).iloc[0]

    assert pd.Timestamp(active["date"]) == dummy_date
    assert pd.Timestamp(inactive["date"]) != dummy_date
    active_expected = sample.forecast_truth(active_regressors, 1)
    inactive_expected = sample.forecast_truth(inactive_regressors, 1)
    np.testing.assert_allclose(active["value"], active_expected, atol=2e-5)
    np.testing.assert_allclose(inactive["value"], inactive_expected, atol=2e-5)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("vintage_index", range(4))
def test_forecast_decomposition_sums_to_forecast(method, vintage_index):
    sample = sample_midas(method=method)
    vintage = sample.vintages[vintage_index]
    model = _fit_stage(sample, vintage, method)

    forecast = model.forecast(vintage.regressors)
    decomposition = model.forecast_decomp(vintage.regressors)
    contribution = decomposition["contribution"].sum()

    assert set(decomposition["horizon"]) == {vintage.horizon}
    assert set(pd.to_datetime(decomposition["date"])) == {sample.truth.target_date}
    np.testing.assert_allclose(contribution, forecast.iloc[0]["value"], atol=1e-9)


def test_forecast_uses_latest_finite_observation_after_trailing_missing_value(
    monkeypatch,
):
    sample = sample_midas()
    vintage = sample.vintages[2]
    regressors = pd.concat(
        [
            vintage.regressors,
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-04-30")],
                    "value": [np.nan],
                }
            ),
        ],
        ignore_index=True,
    )
    model = _fit_stage(sample, vintage, "unrestricted")

    production_lag_rows = []
    build_lag_matrix = midas_module._build_lag_matrix

    def capture_production_lag_row(target_dates, regressors, n_lags, start_lag=0):
        rows = build_lag_matrix(target_dates, regressors, n_lags, start_lag)
        production_lag_rows.extend(rows)
        return rows

    monkeypatch.setattr(midas_module, "_build_lag_matrix", capture_production_lag_row)
    forecast = model.forecast(regressors).iloc[0]
    monkeypatch.setattr(midas_module, "_build_lag_matrix", build_lag_matrix)
    decomposition = model.forecast_decomp(regressors)
    fit = model.fits_[vintage.horizon]
    latest_finite = regressors.loc[np.isfinite(regressors["value"]), "date"].max()
    values_by_month = regressors.assign(
        month=pd.to_datetime(regressors["date"]).dt.to_period("M")
    ).set_index("month")["value"]
    expected_lag_row = np.asarray(
        [
            values_by_month.get(pd.Timestamp(latest_finite).to_period("M") - lag)
            for lag in range(model.n_lags)
        ],
        dtype=float,
    )
    expected_contribution = expected_lag_row @ (fit.beta * fit.weights)

    assert pd.Timestamp(latest_finite) == pd.Timestamp("2024-02-29")
    assert len(production_lag_rows) == 1
    np.testing.assert_allclose(production_lag_rows[0], expected_lag_row, atol=2e-5)
    np.testing.assert_allclose(
        decomposition.loc[decomposition["component"] == "X", "contribution"].iloc[0],
        expected_contribution,
        atol=2e-5,
    )
    assert pd.Timestamp(forecast["date"]) == sample.truth.target_date
    np.testing.assert_allclose(forecast["value"], sample.truth.target_value, atol=2e-5)
