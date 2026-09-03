"""Tests for :class:`~nowcast_midas.multi_midas.MultiMIDAS` and
:class:`~nowcast_midas.specs.MultiMidasSpec`.
"""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import nowcast_midas.multi_midas as multi_midas_module
from nowcast_midas.midas import MIDAS
from nowcast_midas.midas_combo import MidasCombo
from nowcast_midas.multi_midas import MultiMIDAS
from nowcast_midas.specs import (
    ComboSpec,
    MidasSpec,
    MultiMidasSpec,
    OLSSpec,
    VariableSpec,
)
from nowcast_midas.utils import _build_quarterly_lag_matrix, sample_combo_data
from tests.midas.sample_midas import sample_vintage_midas
from tests.midas.sample_multi_midas import sample_multi_midas


def _independent_weights(method, theta, n_lags):
    theta = np.asarray(theta, dtype=float)
    lag = np.arange(n_lags, dtype=float)
    if method == "unrestricted":
        return theta.copy()
    if method == "almon":
        return sum(theta[index] * lag**index for index in range(len(theta)))
    if method == "exp_almon":
        unnormalised = np.exp(
            sum(theta[index] * lag ** (index + 1) for index in range(len(theta)))
        )
        return unnormalised / unnormalised.sum()
    if method == "beta":
        grid = np.linspace(0.001, 0.999, n_lags)
        unnormalised = grid ** (theta[0] - 1) * (1 - grid) ** (theta[1] - 1)
        return unnormalised / unnormalised.sum()
    raise AssertionError(f"unsupported test method: {method}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_single_var_data(
    method="almon",
    n_lags=6,
    n_pars_weights=2,
    n_quarters=80,
    noise=1e-10,
    seed=42,
    horizon=0,
):
    """Generate target + single monthly regressor using sample_combo_data."""
    target, regressors, info = sample_combo_data(
        n_quarters=n_quarters + horizon + 1,
        n_lags=n_lags,
        noise=noise,
        seed=seed,
        horizon=horizon,
        method=method,
        monthly_vars=["x_m"],
        quarterly_vars=[],
        outlier_date=None,
    )
    return target, regressors, info


def _make_multi_var_data(
    method="almon",
    n_lags=6,
    n_pars_weights=2,
    n_quarters=80,
    noise=1e-10,
    seed=42,
    horizon=0,
):
    """Generate target + multiple monthly regressors."""
    target, regressors, info = sample_combo_data(
        n_quarters=n_quarters + horizon + 1,
        n_lags=n_lags,
        noise=noise,
        seed=seed,
        horizon=horizon,
        method=method,
        monthly_vars=["PMI", "IP"],
        quarterly_vars=[],
        outlier_date=None,
    )
    return target, regressors, info


def _as_multi_vintage(sample, use_full_regressors=False):
    regressors = (
        sample.full_regressors if use_full_regressors else sample.regressors
    ).copy()
    regressors.insert(1, "variable", "x")
    return sample.target_train, regressors


@pytest.mark.parametrize("vintage_date", ["2020-02-14", "2020-03-14", "2020-04-14"])
def test_multi_midas_recovers_matching_ragged_edge(vintage_date):
    sample = sample_vintage_midas(vintage_date)
    target, regressors = _as_multi_vintage(sample)
    model = MultiMIDAS(["x"], method="unrestricted", n_lags=len(sample.weights))
    model.fit(target, regressors)

    fit = model.fits_[0]
    np.testing.assert_allclose(fit.alpha, sample.alpha, atol=1e-10)
    np.testing.assert_allclose(
        fit.variable_fits["x"].weights, sample.weights, atol=1e-10
    )
    forecast = model.forecast(regressors).iloc[0]
    np.testing.assert_allclose(forecast["value"], sample.target.iloc[-1]["value"])


def test_multi_midas_rejects_unavailable_months():
    sample = sample_vintage_midas("2020-02-14")
    target, regressors = _as_multi_vintage(sample, use_full_regressors=True)
    model = MultiMIDAS(["x"], method="unrestricted", n_lags=len(sample.weights))
    model.fit(target, regressors)

    assert not np.allclose(
        model.fits_[0].variable_fits["x"].weights, sample.weights, atol=1e-3
    )


# ===========================================================================
# Test: MIDAS ≡ MultiMIDAS with one indicator (OLS path)
# ===========================================================================


class TestSingleIndicatorEquivalence:
    """When MultiMIDAS has a single variable and uses the same method /
    settings as MIDAS, estimation and forecasts must be exactly equal.
    """

    @pytest.mark.parametrize("method", ["almon", "unrestricted"])
    @pytest.mark.parametrize("horizon", [0, 1, 2])
    def test_ols_equivalence(self, method, horizon):
        """OLS paths of MIDAS and MultiMIDAS produce identical results."""
        n_lags = 6
        # DGP always uses 'almon'; the estimator method is what we test.
        target, regressors, _ = _make_single_var_data(
            method="almon", n_lags=n_lags, horizon=horizon
        )
        # Trim to fit-only sample (drop last horizon+1 quarters)
        N_FIT = len(target) - (horizon + 1)
        target_fit = target.iloc[:N_FIT].copy()

        # --- MIDAS (single variable) ---
        reg_single = (
            regressors.loc[regressors["variable"] == "x_m", ["date", "value"]]
            .sort_values("date")
            .copy()
        )
        midas = MIDAS(
            method=method,
            n_lags=n_lags,
            n_pars_weights=2,
            estimator="ols",
            horizons=[horizon],
        )
        midas.fit(target_fit[["date", "value"]], reg_single)

        # --- MultiMIDAS (one variable) ---
        reg_multi = (
            regressors.loc[
                regressors["variable"] == "x_m",
                ["date", "variable", "value"],
            ]
            .sort_values("date")
            .copy()
        )
        multi = MultiMIDAS(
            variables=["x_m"],
            method=method,
            n_lags=n_lags,
            n_pars_weights=2,
            horizons=[horizon],
        )
        multi.fit(target_fit[["date", "value"]], reg_multi)

        fit_m = midas.fits_[horizon]
        fit_mm = multi.fits_[horizon]

        # Intercept
        np.testing.assert_allclose(fit_mm.alpha, fit_m.alpha, atol=1e-10)

        # Weights
        vf = fit_mm.variable_fits["x_m"]
        np.testing.assert_allclose(vf.weights, fit_m.weights, atol=1e-10)

        # Fitted values
        np.testing.assert_allclose(
            fit_mm.fitted_values, fit_m.fitted_values, atol=1e-10
        )

        # Residuals
        np.testing.assert_allclose(fit_mm.residuals, fit_m.residuals, atol=1e-10)

        # Forecast
        fc_midas = midas.forecast(reg_single)
        fc_multi = multi.forecast(reg_multi)
        np.testing.assert_allclose(
            fc_multi["value"].values,
            fc_midas["value"].values,
            atol=1e-10,
        )


# ===========================================================================
# Test: MultiMIDAS weight recovery
# ===========================================================================


class TestMultiMIDASWeightRecovery:
    """Check that MultiMIDAS recovers known weights with near-zero noise."""

    ATOL = 0.02

    def test_almon_two_variables(self):
        target, regressors, _ = _make_multi_var_data(method="almon", noise=1e-8)
        multi = MultiMIDAS(
            variables=["PMI", "IP"],
            method="almon",
            n_lags=6,
            n_pars_weights=2,
            horizons=[0],
        )
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "IP"]),
            ["date", "variable", "value"],
        ]
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]

        # Residuals should be very small
        assert np.max(np.abs(fit.residuals)) < 1e-3

    def test_per_variable_specs(self):
        """VariableSpec overrides per-variable settings."""
        target, regressors, _ = _make_multi_var_data(method="almon", noise=1e-8)
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6, n_pars_weights=2),
                VariableSpec("IP", method="almon", n_lags=6, n_pars_weights=3),
            ],
            horizons=[0],
        )
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "IP"]),
            ["date", "variable", "value"],
        ]
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]

        assert len(fit.variable_fits["PMI"].theta) == 2
        assert len(fit.variable_fits["IP"].theta) == 3
        assert np.max(np.abs(fit.residuals)) < 1e-3


# ===========================================================================
# Test: MultiMidasSpec with MidasCombo
# ===========================================================================


class TestMultiMidasSpecInCombo:
    """MultiMidasSpec integrates into the MidasCombo pipeline."""

    def test_multi_midas_spec_in_combo_fit_forecast(self):
        """A MultiMidasSpec can be used as a source in ComboSpec."""
        target, regressors, _ = _make_multi_var_data(
            method="almon", noise=0.5, n_quarters=80
        )

        multi_spec = MultiMidasSpec(
            name="multi_model",
            variables=["PMI", "IP"],
            method="almon",
            n_lags=6,
            n_pars_weights=2,
        )

        # Also add a single MIDAS spec alongside the multi spec
        midas_spec = MidasSpec(
            variable="PMI",
            method="almon",
            n_lags=6,
            n_pars_weights=2,
        )

        combo = ComboSpec(
            name="my_combo",
            sources=[midas_spec, multi_spec],
            method="average",
        )

        mc = MidasCombo(combo_specs=combo, horizons=3)
        mc.fit(target, regressors)

        # Fitted values must be populated for both sources
        assert "PMI" in mc.fitted_
        assert "multi_model" in mc.fitted_
        assert "my_combo" in mc.fitted_

        # Forecast
        fc = mc.forecast()
        assert fc is not None
        assert len(fc) > 0
        # Check for my_combo in long-format spec column with horizon=1 (1qa)
        assert "my_combo" in fc["spec"].values
        assert (fc.loc[fc["spec"] == "my_combo", "horizon"] == 1).any()

    def test_multi_midas_spec_explicit_kwarg(self):
        """Pass MultiMidasSpec via combo_specs sources."""
        target, regressors, _ = _make_multi_var_data(
            method="almon", noise=0.5, n_quarters=80
        )

        multi_spec = MultiMidasSpec(
            name="multi_model",
            variables=["PMI", "IP"],
            method="almon",
            n_lags=6,
        )

        combo = ComboSpec(
            name="my_combo",
            sources=[multi_spec],
            method="average",
        )

        mc = MidasCombo(
            combo_specs=combo,
            horizons=2,
        )
        mc.fit(target, regressors)
        fc = mc.forecast()
        # Check for my_combo in long-format spec column with horizon=1 (1qa)
        assert "my_combo" in fc["spec"].values
        assert (fc.loc[fc["spec"] == "my_combo", "horizon"] == 1).any()

    def test_multi_midas_spec_with_variable_specs(self):
        """MultiMidasSpec with VariableSpec per variable."""
        target, regressors, _ = _make_multi_var_data(
            method="almon", noise=0.5, n_quarters=80
        )

        multi_spec = MultiMidasSpec(
            name="multi_model",
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6),
                VariableSpec("IP", method="almon", n_lags=6, n_pars_weights=3),
            ],
        )

        combo = ComboSpec(
            name="my_combo",
            sources=[multi_spec],
            method="average",
        )

        mc = MidasCombo(combo_specs=combo, horizons=2)
        mc.fit(target, regressors)

        # Check MultiMIDAS model was fitted
        assert "multi_model" in mc.multi_midas_instances_
        mdl = mc.multi_midas_instances_["multi_model"]
        assert len(mdl.fits_[0].variable_fits) == 2

        fc = mc.forecast()
        assert not fc.empty

    def test_multi_midas_spec_exact_recovery(self):
        """With near-zero noise and a single MultiMidasSpec source,
        in-sample fit residuals should be near-zero."""
        target, regressors, _ = _make_multi_var_data(
            method="almon", noise=1e-10, n_quarters=80
        )

        multi_spec = MultiMidasSpec(
            name="multi_model",
            variables=["PMI", "IP"],
            method="almon",
            n_lags=6,
        )

        combo = ComboSpec(
            name="my_combo",
            sources=[multi_spec],
            method="average",
        )

        mc = MidasCombo(combo_specs=combo, horizons=1)
        mc.fit(target, regressors)

        fit = mc.multi_midas_models_["multi_model"][0]
        assert np.max(np.abs(fit.residuals)) < 1e-4


# ===========================================================================
# Test: MultiMidasSpec validation
# ===========================================================================


class TestMultiMidasSpecValidation:
    def test_empty_variables_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            MultiMIDAS(variables=[], method="almon")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            MultiMIDAS(variables=["x"], method="invalid_method")

    def test_variable_not_in_regressors_raises(self):
        target, regressors, _ = _make_single_var_data()
        multi = MultiMIDAS(variables=["nonexistent"], method="almon")
        with pytest.raises(ValueError, match="not found"):
            reg = regressors[["date", "variable", "value"]]
            multi.fit(target[["date", "value"]], reg)


# ===========================================================================
# Test: VariableSpec is not mutated
# ===========================================================================


def test_variable_spec_not_mutated():
    """MultiMIDAS must not mutate user-supplied VariableSpec objects."""
    spec = VariableSpec("x_m", method="almon", estimator=None)
    MultiMIDAS(variables=[spec], method="almon", n_lags=3)
    assert spec.estimator is None, "VariableSpec.estimator was mutated"


# ===========================================================================
# Test: ComboSpec.collect_indicators returns MultiMidasSpec
# ===========================================================================


def test_collect_indicators_returns_multi_midas_spec():
    multi_spec = MultiMidasSpec(name="mm", variables=["A", "B"])
    midas_spec = MidasSpec(variable="C")
    ols_spec = OLSSpec(variable="D")

    combo = ComboSpec(
        name="c",
        sources=[multi_spec, midas_spec, ols_spec],
    )
    midas_list, ols_list, multi_list = combo.collect_indicators()

    assert len(midas_list) == 1
    assert midas_list[0].variable == "C"
    assert len(ols_list) == 1
    assert ols_list[0].variable == "D"
    assert len(multi_list) == 1
    assert multi_list[0].name == "mm"


def test_combo_source_names_with_multi_midas_spec():
    multi_spec = MultiMidasSpec(name="mm", variables=["A", "B"])
    midas_spec = MidasSpec(variable="C")

    combo = ComboSpec(
        name="c",
        sources=[multi_spec, midas_spec, "other"],
    )
    assert combo.source_names == ["mm", "C", "other"]


# ===========================================================================
# Independent real time MultiMIDAS validation
# ===========================================================================


def _fit_sample_vintage(sample, vintage, *, include_dummy=False):
    model = MultiMIDAS(
        variables=list(sample.truth.variables),
        horizons=[vintage.horizon],
        dummy_periods=(list(sample.truth.dummy_periods) if include_dummy else None),
    )
    model.fit(vintage.target_train, vintage.regressors)
    return model


def test_vintage_applies_each_variable_publication_lag():
    sample = sample_multi_midas()
    observed = sample.observed.set_index(["variable", "date"])["release_date"]

    survey_release = observed.loc["survey", pd.Timestamp("2024-01-31")]
    delayed_release = observed.loc["activity", pd.Timestamp("2024-01-31")]
    assert -16 <= (survey_release - pd.Timestamp("2024-01-31")).days <= -14
    assert delayed_release > pd.Timestamp("2024-01-31")

    latest_by_vintage = []
    for vintage in sample.vintages:
        latest_by_vintage.append(
            {
                variable: vintage.regressors.loc[
                    (vintage.regressors["variable"] == variable)
                    & np.isfinite(vintage.regressors["value"]),
                    "date",
                ].max()
                for variable in ("survey", "activity")
            }
        )
    assert latest_by_vintage == [
        {"survey": pd.Timestamp("2023-12-31"), "activity": pd.Timestamp("2023-11-30")},
        {"survey": pd.Timestamp("2024-01-31"), "activity": pd.Timestamp("2023-12-31")},
        {"survey": pd.Timestamp("2024-02-29"), "activity": pd.Timestamp("2024-01-31")},
        {"survey": pd.Timestamp("2024-03-31"), "activity": pd.Timestamp("2024-02-29")},
    ]


def test_common_origin_rule_is_independent_of_variable_latest_dates():
    sample = sample_multi_midas()

    for vintage in sample.vintages:
        latest_dates = [
            vintage.regressors.loc[
                (vintage.regressors["variable"] == variable)
                & np.isfinite(vintage.regressors["value"]),
                "date",
            ].max()
            for variable in ("survey", "activity")
        ]
        expected_origin = max(pd.Timestamp(date) for date in latest_dates)
        expected_horizon = (
            sample.truth.target_date.to_period("Q").ordinal
            - expected_origin.to_period("Q").ordinal
        )
        assert vintage.common_origin_date == expected_origin
        assert vintage.horizon == expected_horizon


def test_sampler_supports_compact_lag_count_and_start_lag_coverage():
    sample = sample_multi_midas(n_lags=3, start_lag=1)

    assert all(
        block.shape[1] == 3 for block in sample.vintages[-1].expected_blocks.values()
    )
    assert all(spec.start_lag == 1 for spec in sample.truth.variables)


@pytest.mark.parametrize("method", ["almon", "unrestricted", "exp_almon", "beta"])
def test_truth_fixture_has_valid_family_coefficients(method):
    sample = sample_multi_midas(method=method, n_lags=4)

    for variable_truth in sample.truth.variables_truth.values():
        if variable_truth.frequency == "QE":
            continue
        expected_weights = _independent_weights(
            variable_truth.method,
            variable_truth.theta_true,
            variable_truth.n_lags,
        )
        np.testing.assert_allclose(
            variable_truth.effective_coefficients[0],
            variable_truth.scale * expected_weights,
            atol=1e-12,
        )
        assert np.isfinite(variable_truth.scale)
        assert np.all(np.isfinite(variable_truth.theta_true))


def test_start_lag_target_and_forecast_recovery_at_latest_vintage():
    sample = sample_multi_midas(method="unrestricted", n_lags=3, start_lag=1)

    for vintage in sample.vintages:
        assert np.isfinite(vintage.independent_forecast)
        model = _fit_sample_vintage(sample, vintage)
        fit = model.fits_[vintage.horizon]
        expected_valid = vintage.expected_valid_rows
        source_valid = expected_valid[: len(expected_valid) - vintage.horizon]
        expected_dates = vintage.target_train["date"].to_numpy()[vintage.horizon :][
            source_valid
        ]

        np.testing.assert_allclose(fit.alpha, sample.truth.alpha, atol=2e-5)
        assert fit.nobs == int(source_valid.sum())
        np.testing.assert_array_equal(fit.dates, expected_dates)
        for variable in ("survey", "activity"):
            variable_fit = fit.variable_fits[variable]
            expected = vintage.expected_coefficients[variable]
            np.testing.assert_allclose(
                variable_fit.beta * variable_fit.weights,
                expected,
                atol=2e-2,
            )

        forecast = model.forecast(vintage.regressors).iloc[0]
        assert pd.Timestamp(forecast["date"]) == sample.truth.target_date
        np.testing.assert_allclose(
            forecast["value"], sample.truth.target_value, atol=2e-5
        )
        np.testing.assert_allclose(
            forecast["value"], vintage.independent_forecast, atol=2e-5
        )
        assert "common origin quarter" in sample.truth.common_origin_equation


@pytest.mark.parametrize("block", ["target", "monthly", "quarterly", "ar"])
def test_fit_excludes_infinite_values_from_complete_cases(block):
    sample = sample_multi_midas(include_quarterly=True)
    vintage = sample.vintages[-1]
    target = vintage.target_train.copy()
    regressors = vintage.regressors.copy()
    n_ar_lags = 1 if block == "ar" else 0

    if block == "target" or block == "ar":
        target.loc[5, "value"] = np.inf
    elif block == "monthly":
        regressors.loc[
            (regressors["variable"] == "survey")
            & (regressors["date"] == pd.Timestamp("2022-02-28")),
            "value",
        ] = np.inf
    else:
        regressors.loc[
            (regressors["variable"] == "quarterly")
            & (regressors["date"] == pd.Timestamp("2022-03-31")),
            "value",
        ] = np.inf

    model = MultiMIDAS(
        variables=list(sample.truth.variables),
        horizons=[0],
        n_ar_lags=n_ar_lags,
    )
    model.fit(target, regressors)

    if block == "target":
        assert not model.valid_mask_[5]
    elif block == "ar":
        assert not model.valid_mask_[5]
        assert not model.valid_mask_[6]
    elif block == "monthly":
        row = int(np.flatnonzero(target["date"] == pd.Timestamp("2022-03-31"))[0])
        assert not model.valid_mask_[row]
    else:
        row = int(np.flatnonzero(target["date"] == pd.Timestamp("2022-03-31"))[0])
        assert not model.valid_mask_[row]


def test_trailing_nonfinite_rows_do_not_advance_common_origin_or_lag_anchor():
    sample = sample_multi_midas()
    vintage = sample.vintages[-1]
    model = _fit_sample_vintage(sample, vintage)
    trailing = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-04-30"), pd.Timestamp("2024-05-31")],
            "variable": ["survey", "activity"],
            "value": [np.nan, np.inf],
        }
    )
    regressors = pd.concat([vintage.regressors, trailing], ignore_index=True)

    forecast = model.forecast(regressors).iloc[0]
    decomposition = model.forecast_decomp(regressors)
    expected = model.forecast(vintage.regressors).iloc[0]

    assert pd.Timestamp(forecast["date"]) == pd.Timestamp(expected["date"])
    np.testing.assert_allclose(forecast["value"], expected["value"])
    assert set(pd.to_datetime(decomposition["date"])) == {
        pd.Timestamp(expected["date"])
    }


def test_quarterly_lag_matrix_preserves_an_interior_calendar_gap():
    target_dates = pd.date_range("2020-03-31", periods=4, freq="QE")
    regressors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-12-31"]),
            "value": [1.0, 2.0, 4.0],
        }
    )
    expected = np.array([[1.0, np.nan], [2.0, 1.0], [np.nan, 2.0], [4.0, np.nan]])

    block = _build_quarterly_lag_matrix(target_dates, regressors, n_lags=2)

    np.testing.assert_allclose(block, expected, equal_nan=True)


def test_decomposition_is_empty_when_fitted_parameters_are_nonfinite():
    sample = sample_multi_midas()
    vintage = sample.vintages[-1]
    model = _fit_sample_vintage(sample, vintage)
    model.fits_[vintage.horizon].alpha = np.inf

    forecast = model.forecast(vintage.regressors).iloc[0]
    decomposition = model.forecast_decomp(vintage.regressors)

    assert np.isnan(forecast["value"])
    assert decomposition.empty


def test_production_design_blocks_match_independent_blocks():
    sample = sample_multi_midas(include_quarterly=True, n_lags=3)
    vintage = sample.vintages[-1]
    captured_monthly = []
    captured_quarterly = []
    original_monthly = multi_midas_module._build_lag_matrix
    original_quarterly = multi_midas_module._build_quarterly_lag_matrix

    def capture_monthly(*args, **kwargs):
        block = original_monthly(*args, **kwargs)
        captured_monthly.append(block)
        return block

    def capture_quarterly(*args, **kwargs):
        block = original_quarterly(*args, **kwargs)
        captured_quarterly.append(block)
        return block

    model = MultiMIDAS(
        variables=list(sample.truth.variables), horizons=[vintage.horizon]
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(multi_midas_module, "_build_lag_matrix", capture_monthly)
    monkeypatch.setattr(
        multi_midas_module, "_build_quarterly_lag_matrix", capture_quarterly
    )
    try:
        model.fit(vintage.target_train, vintage.regressors)
    finally:
        monkeypatch.undo()

    np.testing.assert_allclose(
        captured_monthly[0], vintage.expected_blocks["survey"], equal_nan=True
    )
    np.testing.assert_allclose(
        captured_monthly[1], vintage.expected_blocks["activity"], equal_nan=True
    )
    np.testing.assert_allclose(
        captured_quarterly[0], vintage.expected_blocks["quarterly"], equal_nan=True
    )
    missing_row = np.flatnonzero(~vintage.expected_valid_rows)[0]
    assert np.isnan(captured_monthly[0][missing_row, 1])


def test_incomplete_joint_rows_are_dropped():
    sample = sample_multi_midas()

    for vintage in sample.vintages:
        model = _fit_sample_vintage(sample, vintage)
        expected_valid = vintage.expected_valid_rows
        source_valid = expected_valid[vintage.horizon :]
        np.testing.assert_array_equal(model.valid_mask_, expected_valid)
        assert model.fits_[vintage.horizon].nobs == int(source_valid.sum())


def test_missing_calendar_periods_are_not_compressed():
    sample = sample_multi_midas()
    vintage = sample.vintages[-1]
    target_index = vintage.target_train.index[
        vintage.target_train["date"] == pd.Timestamp("2021-03-31")
    ][0]
    row = vintage.expected_blocks["survey"][target_index]

    assert np.isfinite(row[0])
    assert np.isnan(row[1])
    assert not vintage.expected_valid_rows[target_index]


@pytest.mark.parametrize("method", ["almon", "unrestricted"])
def test_joint_parameters_match_truth_at_each_stage(method):
    sample = sample_multi_midas(method=method)

    for vintage in sample.vintages:
        model = _fit_sample_vintage(sample, vintage)
        fit = model.fits_[vintage.horizon]
        np.testing.assert_allclose(fit.alpha, sample.truth.alpha, atol=2e-5)
        for variable in ("survey", "activity"):
            variable_fit = fit.variable_fits[variable]
            expected = sample.truth.variables_truth[variable].effective_coefficients[
                vintage.horizon
            ]
            np.testing.assert_allclose(
                variable_fit.beta * variable_fit.weights, expected, atol=2e-5
            )


def test_exp_almon_parameters_match_truth_at_each_stage():
    sample = sample_multi_midas(method="exp_almon", n_lags=3)

    for vintage in sample.vintages:
        model = _fit_sample_vintage(sample, vintage)
        fit = model.fits_[vintage.horizon]
        for variable in ("survey", "activity"):
            expected = sample.truth.variables_truth[variable].effective_coefficients[
                vintage.horizon
            ]
            np.testing.assert_allclose(
                fit.variable_fits[variable].beta * fit.variable_fits[variable].weights,
                expected,
                atol=2e-5,
            )


def test_beta_parameters_match_truth_at_each_stage():
    sample = sample_multi_midas(method="beta", n_lags=3)

    for vintage in sample.vintages:
        model = _fit_sample_vintage(sample, vintage)
        fit = model.fits_[vintage.horizon]
        for variable in ("survey", "activity"):
            expected = sample.truth.variables_truth[variable].effective_coefficients[
                vintage.horizon
            ]
            np.testing.assert_allclose(
                fit.variable_fits[variable].beta * fit.variable_fits[variable].weights,
                expected,
                atol=2e-5,
            )


def test_held_out_joint_forecast_matches_truth_at_each_stage():
    sample = sample_multi_midas()

    for vintage in sample.vintages:
        model = _fit_sample_vintage(sample, vintage)
        forecast = model.forecast(vintage.regressors).iloc[0]
        assert pd.Timestamp(forecast["date"]) == sample.truth.target_date
        np.testing.assert_allclose(
            forecast["value"], sample.truth.target_value, atol=2e-5
        )
        np.testing.assert_allclose(
            forecast["value"], vintage.independent_forecast, atol=2e-5
        )


def test_mixed_monthly_and_quarterly_parameters_match_truth():
    sample = sample_multi_midas(include_quarterly=True)
    vintage = sample.vintages[-1]
    model = _fit_sample_vintage(sample, vintage)
    fit = model.fits_[vintage.horizon]

    np.testing.assert_allclose(
        fit.variable_fits["survey"].weights,
        sample.truth.variables_truth["survey"].effective_coefficients[vintage.horizon],
        atol=2e-5,
    )
    np.testing.assert_allclose(
        fit.variable_fits["quarterly"].weights,
        sample.truth.variables_truth["quarterly"].effective_coefficients[
            vintage.horizon
        ],
        atol=2e-5,
    )


def test_mixed_parameters_and_availability_match_at_each_vintage():
    sample = sample_multi_midas(include_quarterly=True, method="almon", n_lags=3)

    for vintage in sample.vintages:
        model = _fit_sample_vintage(sample, vintage)
        fit = model.fits_[vintage.horizon]
        for variable in ("survey", "activity", "quarterly"):
            expected = sample.truth.variables_truth[variable].effective_coefficients[
                vintage.horizon
            ]
            np.testing.assert_allclose(
                fit.variable_fits[variable].weights,
                expected,
                atol=2e-5,
            )

        activity_dates = vintage.regressors.loc[
            (vintage.regressors["variable"] == "activity")
            & np.isfinite(vintage.regressors["value"]),
            "date",
        ]
        assert pd.Timestamp(activity_dates.max()) < sample.truth.target_date
        assert np.all(np.isfinite(vintage.expected_blocks["quarterly"][-1]))


def test_incomplete_forecast_block_returns_nan():
    sample = sample_multi_midas()
    vintage = sample.vintages[-1]
    model = _fit_sample_vintage(sample, vintage)
    incomplete = vintage.regressors.copy()
    latest_activity = incomplete.loc[
        incomplete["variable"] == "activity", "date"
    ].max() - pd.offsets.MonthEnd(1)
    incomplete.loc[
        (incomplete["variable"] == "activity")
        & (incomplete["date"] == latest_activity),
        "value",
    ] = np.nan

    forecast = model.forecast(incomplete).iloc[0]
    assert np.isnan(forecast["value"])


def test_decomposition_is_empty_when_required_block_is_missing_or_infinite():
    sample = sample_multi_midas()
    vintage = sample.vintages[-1]
    model = _fit_sample_vintage(sample, vintage)

    for replacement in (np.nan, np.inf):
        incomplete = vintage.regressors.copy()
        latest_activity = incomplete.loc[
            incomplete["variable"] == "activity", "date"
        ].max() - pd.offsets.MonthEnd(1)
        incomplete.loc[
            (incomplete["variable"] == "activity")
            & (incomplete["date"] == latest_activity),
            "value",
        ] = replacement

        forecast = model.forecast(incomplete).iloc[0]
        decomposition = model.forecast_decomp(incomplete)
        assert np.isnan(forecast["value"])
        assert decomposition.empty


def test_joint_forecast_with_and_without_dummies():
    sample = sample_multi_midas(include_dummy=True)
    vintage = sample.vintages[-1]
    with_dummy = _fit_sample_vintage(sample, vintage, include_dummy=True)
    without_dummy = _fit_sample_vintage(sample, vintage)

    forecast_with = with_dummy.forecast(vintage.regressors).iloc[0]["value"]
    forecast_without = without_dummy.forecast(vintage.regressors).iloc[0]["value"]
    np.testing.assert_allclose(forecast_with, sample.truth.target_value, atol=2e-5)
    assert abs(forecast_without - forecast_with) > 1e-3


def test_joint_forecast_decomposition_sums_to_forecast():
    sample = sample_multi_midas(include_dummy=True)
    vintage = sample.vintages[-1]
    model = _fit_sample_vintage(sample, vintage, include_dummy=True)

    forecast = model.forecast(vintage.regressors).iloc[0]
    decomposition = model.forecast_decomp(vintage.regressors)
    assert set(decomposition["horizon"]) == {vintage.horizon}
    assert set(pd.to_datetime(decomposition["date"])) == {sample.truth.target_date}
    np.testing.assert_allclose(
        decomposition["contribution"].sum(), forecast["value"], atol=1e-9
    )


# ===========================================================================
# Test: Quarterly regressors in MultiMIDAS
# ===========================================================================


def _make_mixed_data(n_quarters=80, noise=1e-10, seed=42, horizon=0):
    """Generate target + monthly PMI + quarterly UNEMP regressors."""
    target, regressors, info = sample_combo_data(
        n_quarters=n_quarters + horizon + 1,
        n_lags=6,
        noise=noise,
        seed=seed,
        horizon=horizon,
        method="almon",
        monthly_vars=["PMI"],
        quarterly_vars=["UNEMP"],
        outlier_date=None,
    )
    return target, regressors, info


class TestQuarterlyRegressors:
    """MultiMIDAS with a mix of monthly (ME) and quarterly (QE) regressors."""

    def test_fit_returns_variable_fits_for_both(self):
        """Both ME and QE variables appear in variable_fits after fitting."""
        target, regressors, _ = _make_mixed_data()
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "UNEMP"]),
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]

        assert "PMI" in fit.variable_fits
        assert "UNEMP" in fit.variable_fits
        assert len(fit.variable_fits["UNEMP"].weights) == 1

    def test_ols_path_low_residuals(self):
        """OLS path (almon + QE) should recover near-zero residuals."""
        target, regressors, _ = _make_mixed_data(noise=1e-10)
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "UNEMP"]),
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]
        assert np.max(np.abs(fit.residuals)) < 1e-4

    def test_nls_path_low_residuals(self):
        """NLS path (exp_almon + QE) should recover near-zero residuals."""
        target, regressors, _ = _make_mixed_data(noise=1e-10)
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "UNEMP"]),
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="exp_almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]
        assert np.max(np.abs(fit.residuals)) < 0.01

    def test_nls_almon_not_degenerate(self):
        """An ``almon`` variable mixed with ``exp_almon`` (forcing NLS) must be
        estimated linearly and not drift to degenerate theta / uniform weights."""
        target, regressors, _ = sample_combo_data(
            n_quarters=81,
            n_lags=6,
            noise=1e-10,
            seed=42,
            horizon=0,
            method="almon",
            monthly_vars=["PMI", "IP"],
            quarterly_vars=["UNEMP"],
            outlier_date=None,
        )
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "IP", "UNEMP"]),
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="exp_almon", n_lags=6),
                VariableSpec("IP", method="almon", n_lags=6, n_pars_weights=2),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]
        ip_fit = fit.variable_fits["IP"]
        # Linear (OLS-style) almon: beta is fixed to 1.0, theta stays finite.
        assert ip_fit.beta == 1.0
        assert np.all(np.isfinite(ip_fit.theta))
        assert np.max(np.abs(ip_fit.theta)) < 1e3
        # Weights must not collapse to a uniform vector.
        assert np.std(ip_fit.weights) > 1e-6

    def test_ols_path_recovers_gamma(self):
        """The quarterly coefficient delta matches the DGP gamma."""
        target, regressors, info = _make_mixed_data(noise=1e-10)
        true_gamma = info["gammas"]["UNEMP"]  # e.g. -0.5

        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "UNEMP"]),
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]
        estimated_delta = fit.variable_fits["UNEMP"].weights[0]
        np.testing.assert_allclose(estimated_delta, true_gamma, atol=1e-3)

    def test_monthly_only_specs_unchanged(self):
        """Existing behaviour with monthly-only specs is unaffected."""
        target, regressors, _ = _make_mixed_data(noise=1e-10)
        reg_df = regressors.loc[
            regressors["variable"] == "PMI",
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(variables=["PMI"], method="almon", n_lags=6)
        multi.fit(target[["date", "value"]], reg_df)
        assert len(multi.quarterly_specs) == 0
        assert len(multi.monthly_specs) == 1

    def test_forecast_returns_finite_value(self):
        """forecast() produces a finite number when all lags are available."""
        target, regressors, _ = _make_mixed_data(noise=1e-10)
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "UNEMP"]),
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fc = multi.forecast(reg_df)
        assert fc is not None
        assert np.isfinite(fc["value"].values[0])

    def test_multiple_quarterly_lags(self):
        """QE variable with n_lags=2 stores 2 delta coefficients."""
        target, regressors, _ = _make_mixed_data(noise=1e-10)
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "UNEMP"]),
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=2),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]
        assert len(fit.variable_fits["UNEMP"].weights) == 2

    def test_quarterly_only_specs(self):
        """MultiMIDAS with only QE regressors (no monthly) works via OLS."""
        target, regressors, _ = _make_mixed_data(noise=1e-10)
        reg_df = regressors.loc[
            regressors["variable"] == "UNEMP",
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]
        assert "UNEMP" in fit.variable_fits
        assert np.isfinite(fit.alpha)

    def test_invalid_frequency_raises(self):
        """VariableSpec with unknown frequency raises ValueError."""
        with pytest.raises(ValueError, match="frequency"):
            MultiMIDAS(
                variables=[VariableSpec("X", frequency="weekly")],
            )

    def test_quarterly_spec_not_mutated(self):
        """MultiMIDAS must not mutate user-supplied QE VariableSpec."""
        spec = VariableSpec("UNEMP", frequency="QE", n_lags=1)
        assert spec.estimator is None
        MultiMIDAS(variables=[spec])
        assert spec.estimator is None


# ===========================================================================
# Test: exact coefficient & forecast recovery (parity with MIDAS tests)
# ===========================================================================


class TestMultiMIDASExactRecovery:
    """Mirror the MIDAS recovery tests at the multi-regressor level.

    With (near) noiseless data, MultiMIDAS must recover the true
    per-variable coefficients *and* produce forecasts that match held-out
    actuals -- not merely drive the in-sample residuals to zero.
    """

    ATOL_W = 0.02  # weight / coefficient tolerance
    ATOL_FC = 0.05  # forecast tolerance

    def test_ols_recovers_weights_betas_and_gamma(self):
        """OLS path: each monthly variable's recovered (unnormalised)
        weights equal ``beta_v * true_weights``, and the quarterly delta
        equals the DGP gamma."""
        target, regressors, info = sample_combo_data(
            n_quarters=120,
            n_lags=6,
            noise=1e-8,
            seed=7,
            method="almon",
            theta_true=[1.0, -0.2],
            monthly_vars=["PMI", "IP"],
            quarterly_vars=["UNEMP"],
            outlier_date=None,
        )
        reg_df = regressors[["date", "variable", "value"]]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6, n_pars_weights=2),
                VariableSpec("IP", method="almon", n_lags=6, n_pars_weights=2),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]

        np.testing.assert_allclose(fit.alpha, info["alpha"], atol=self.ATOL_W)
        for v in ["PMI", "IP"]:
            expected = info["betas"][v] * info["weights"]
            np.testing.assert_allclose(
                fit.variable_fits[v].weights, expected, atol=self.ATOL_W
            )
        np.testing.assert_allclose(
            fit.variable_fits["UNEMP"].weights[0],
            info["gammas"]["UNEMP"],
            atol=self.ATOL_W,
        )

    def test_nls_recovers_weights_beta_and_gamma(self):
        """NLS path: the exp_almon variable recovers its normalised
        weights and a separate slope ``beta``, and the quarterly delta
        equals the DGP gamma."""
        target, regressors, info = sample_combo_data(
            n_quarters=120,
            n_lags=6,
            noise=1e-8,
            seed=7,
            method="exp_almon",
            theta_true=[-0.5, -0.1],
            monthly_vars=["PMI"],
            quarterly_vars=["UNEMP"],
            outlier_date=None,
        )
        reg_df = regressors[["date", "variable", "value"]]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="exp_almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)
        fit = multi.fits_[0]
        pmi = fit.variable_fits["PMI"]

        np.testing.assert_allclose(pmi.beta, info["betas"]["PMI"], atol=self.ATOL_W)
        np.testing.assert_allclose(pmi.weights, info["weights"], atol=self.ATOL_W)
        np.testing.assert_allclose(
            fit.variable_fits["UNEMP"].weights[0],
            info["gammas"]["UNEMP"],
            atol=self.ATOL_W,
        )

    def test_forecast_accuracy_ols(self):
        """Noiseless OLS path: held-out nowcast matches the actual."""
        target, regressors, _ = sample_combo_data(
            n_quarters=120,
            n_lags=6,
            noise=0.0,
            seed=7,
            method="almon",
            theta_true=[1.0, -0.2],
            monthly_vars=["PMI", "IP"],
            quarterly_vars=["UNEMP"],
            outlier_date=None,
        )
        target_train = target.iloc[:-1]
        actual = target.iloc[-1]["value"]
        held_out_date = target.iloc[-1]["date"]

        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6),
                VariableSpec("IP", method="almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        reg_df = regressors[["date", "variable", "value"]]
        multi.fit(target_train[["date", "value"]], reg_df)
        reg_fc = reg_df[reg_df["date"] <= held_out_date]
        fc = multi.forecast(reg_fc)
        predicted = fc.loc[fc["horizon"] == 0, "value"].iloc[0]
        np.testing.assert_allclose(predicted, actual, atol=self.ATOL_FC)

    def test_forecast_accuracy_nls(self):
        """Noiseless mixed/NLS path: held-out nowcast matches the actual."""
        target, regressors, _ = sample_combo_data(
            n_quarters=120,
            n_lags=6,
            noise=0.0,
            seed=7,
            method="exp_almon",
            theta_true=[-0.5, -0.1],
            monthly_vars=["PMI"],
            quarterly_vars=["UNEMP"],
            outlier_date=None,
        )
        target_train = target.iloc[:-1]
        actual = target.iloc[-1]["value"]
        held_out_date = target.iloc[-1]["date"]

        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="exp_almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        reg_df = regressors[["date", "variable", "value"]]
        multi.fit(target_train[["date", "value"]], reg_df)
        reg_fc = reg_df[reg_df["date"] <= held_out_date]
        fc = multi.forecast(reg_fc)
        predicted = fc.loc[fc["horizon"] == 0, "value"].iloc[0]
        np.testing.assert_allclose(predicted, actual, atol=self.ATOL_FC)


# ===========================================================================
# Test: ragged edge (parity with the MIDAS ragged-edge tests)
# ===========================================================================


class TestMultiMIDASRaggedEdge:
    """MultiMIDAS must honour ragged-edge data availability the way single
    MIDAS does, including the realistic case where each regressor has its
    own publication lag (a *heterogeneous* ragged edge).
    """

    N_LAGS = 6
    THETA: ClassVar[list[float]] = [-0.5, -0.1]

    def _true_weights(self):
        import jax.numpy as jnp

        from nowcast_midas.temporal_weights import get_weights

        return np.asarray(get_weights("almon", jnp.array(self.THETA), self.N_LAGS))

    def test_matches_single_midas_under_ragged_edge(self):
        """A one-variable MultiMIDAS equals single MIDAS on ragged data.

        The monthly series ends one month before the last quarter-end
        (``month_within_quarter == 2``), so both models must apply the same
        ragged-edge alignment and yield identical forecasts.
        """
        import pandas as pd

        from nowcast_midas.utils import _build_lag_matrix

        n_lags = self.N_LAGS
        true_w = self._true_weights()

        dates_q = pd.date_range("2005-03-31", periods=100, freq="QE")
        start_m = dates_q[0] - pd.DateOffset(months=n_lags)
        # End monthly data one month before the last quarter-end -> ragged.
        dates_m = pd.date_range(
            start_m, end=dates_q[-1] - pd.DateOffset(months=1), freq="ME"
        )
        rng = np.random.default_rng(3)
        reg = pd.DataFrame(
            {"date": dates_m, "value": rng.standard_normal(len(dates_m))}
        )

        X = _build_lag_matrix(dates_q, reg, n_lags)
        valid = ~np.any(np.isnan(X), axis=1)
        y = 1.5 + 0.9 * (X[valid] @ true_w)
        target = pd.DataFrame({"date": dates_q[valid], "value": y})

        reg_multi = reg.assign(variable="x")[["date", "variable", "value"]]

        midas = MIDAS(method="almon", n_lags=n_lags, horizons=[0, 1]).fit(target, reg)
        multi = MultiMIDAS(
            variables=["x"], method="almon", n_lags=n_lags, horizons=[0, 1]
        ).fit(target[["date", "value"]], reg_multi)

        fc_m = midas.forecast(reg)
        fc_mm = multi.forecast(reg_multi)
        np.testing.assert_allclose(
            fc_mm["value"].to_numpy(),
            fc_m["value"].to_numpy(),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_heterogeneous_ragged_edge_fit_and_forecast(self):
        """Two monthly regressors with *different* publication lags.

        ``PMI`` ends one month before the last quarter-end
        (``month_within_quarter == 2``) while ``IP`` is fully up to date
        (``== 3``).  Because each variable's lag matrix is anchored to its
        own latest month, a noiseless DGP must be recovered exactly and the
        held-out nowcast must match the actual.
        """
        import pandas as pd

        from nowcast_midas.utils import _build_lag_matrix

        n_lags = self.N_LAGS
        true_w = self._true_weights()
        alpha, betas = 2.0, {"PMI": 1.0, "IP": 0.7}

        dates_q = pd.date_range("2000-03-31", periods=120, freq="QE")
        start_m = dates_q[0] - pd.DateOffset(months=n_lags)
        dates_m_pmi = pd.date_range(
            start_m, end=dates_q[-1] - pd.DateOffset(months=1), freq="ME"
        )
        dates_m_ip = pd.date_range(start_m, end=dates_q[-1], freq="ME")

        rng = np.random.default_rng(11)
        reg_pmi = pd.DataFrame(
            {"date": dates_m_pmi, "value": rng.standard_normal(len(dates_m_pmi))}
        )
        reg_ip = pd.DataFrame(
            {"date": dates_m_ip, "value": rng.standard_normal(len(dates_m_ip))}
        )

        X_pmi = _build_lag_matrix(dates_q, reg_pmi, n_lags)
        X_ip = _build_lag_matrix(dates_q, reg_ip, n_lags)
        valid = ~np.any(np.isnan(X_pmi), axis=1) & ~np.any(np.isnan(X_ip), axis=1)
        y = (
            alpha
            + betas["PMI"] * (X_pmi[valid] @ true_w)
            + betas["IP"] * (X_ip[valid] @ true_w)
        )
        target = pd.DataFrame({"date": dates_q[valid], "value": y})

        reg_long = pd.concat(
            [reg_pmi.assign(variable="PMI"), reg_ip.assign(variable="IP")],
            ignore_index=True,
        )[["date", "variable", "value"]]

        target_train = target.iloc[:-1]
        actual = target.iloc[-1]["value"]
        held_out_date = target.iloc[-1]["date"]

        multi = MultiMIDAS(
            variables=["PMI", "IP"], method="almon", n_lags=n_lags, horizons=[0]
        )
        multi.fit(target_train[["date", "value"]], reg_long)

        # Noiseless and well specified -> residuals must vanish.
        assert np.max(np.abs(multi.fits_[0].residuals)) < 1e-6

        reg_fc = reg_long[reg_long["date"] <= held_out_date]
        fc = multi.forecast(reg_fc)
        predicted = fc.loc[fc["horizon"] == 0, "value"].iloc[0]
        np.testing.assert_allclose(predicted, actual, atol=1e-6)


class TestMultiMIDASForecastDecomp:
    """Forecast decomposition: contributions must sum to the forecast value."""

    def test_columns_and_sum_to_forecast_monthly(self):
        """Two monthly regressors: contributions sum to the forecast."""
        target, regressors, _ = _make_multi_var_data(n_quarters=80, noise=1e-10)
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "IP"]),
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=["PMI", "IP"], method="almon", n_lags=6, horizons=[0, 1, 2]
        )
        multi.fit(target[["date", "value"]], reg_df)

        fc = multi.forecast(reg_df)
        decomp = multi.forecast_decomp(reg_df)

        assert list(decomp.columns) == [
            "horizon",
            "date",
            "component",
            "contribution",
            "weight",
        ]
        assert {"intercept", "PMI", "IP"}.issubset(set(decomp["component"]))
        for h in [0, 1, 2]:
            s = decomp.loc[decomp["horizon"] == h, "contribution"].sum()
            f = fc.loc[fc["horizon"] == h, "value"].iloc[0]
            np.testing.assert_allclose(s, f, atol=1e-8)

    def test_sum_to_forecast_mixed_frequency(self):
        """Monthly (MIDAS) + quarterly (linear) regressors: sum holds."""
        target, regressors, _ = _make_mixed_data(noise=1e-10)
        reg_df = regressors.loc[
            regressors["variable"].isin(["PMI", "UNEMP"]),
            ["date", "variable", "value"],
        ]
        multi = MultiMIDAS(
            variables=[
                VariableSpec("PMI", method="almon", n_lags=6),
                VariableSpec("UNEMP", frequency="QE", n_lags=1),
            ],
            horizons=[0],
        )
        multi.fit(target[["date", "value"]], reg_df)

        fc = multi.forecast(reg_df)
        decomp = multi.forecast_decomp(reg_df)

        assert {"intercept", "PMI", "UNEMP"}.issubset(set(decomp["component"]))
        # Monthly/quarterly blocks carry NaN weight; intercept carries 1.0.
        assert decomp.loc[decomp["component"] == "PMI", "weight"].isna().all()
        assert decomp.loc[decomp["component"] == "UNEMP", "weight"].isna().all()
        assert (decomp.loc[decomp["component"] == "intercept", "weight"] == 1.0).all()

        s = decomp.loc[decomp["horizon"] == 0, "contribution"].sum()
        f = fc.loc[fc["horizon"] == 0, "value"].iloc[0]
        np.testing.assert_allclose(s, f, atol=1e-8)
