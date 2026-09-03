"""Smoke tests for :mod:`nowcast_midas.plots.midas_combo`.

`~nowcast_midas.midas_combo.MidasCombo` mixes in the plotting
methods from ``plots/midas_combo.py``; before this file that module was
only imported, never exercised.  These tests fit a small two-level combo
pipeline and check that each plotting entry point returns a populated
figure/axes pair.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for testing

import pytest

from nowcast_midas import ComboSpec, MidasCombo, MidasSpec, OLSSpec
from nowcast_midas.utils import sample_combo_data


@pytest.fixture
def fitted_combo():
    """A fitted, forecasted two-source regression combo of a soft combo + OLS."""
    target, regressors, info = sample_combo_data(n_quarters=48, seed=1)
    outlier = info["outlier_date"]
    soft = ComboSpec(
        "soft",
        sources=[
            MidasSpec("monthly_1", "almon", n_lags=6, dummy_periods=[outlier]),
            MidasSpec("monthly_2", "almon", n_lags=6, dummy_periods=[outlier]),
        ],
        method="mse",
        window=8,
    )
    final = ComboSpec(
        "final",
        sources=[soft, OLSSpec("quarterly_1", n_lags=1)],
        method="regression",
    )
    model = MidasCombo(combo_specs=final, horizons=2).fit(target, regressors)
    model.forecast()
    return model


@pytest.mark.parametrize("method", ["plot_fit", "plot_weights", "plot_forecast"])
def test_combo_plot_smoke_single_horizon(fitted_combo, method):
    """Each plot method draws at least one line/patch on a single-horizon axis."""
    fig, ax = getattr(fitted_combo, method)(horizon=0)
    assert fig is not None and ax is not None
    assert len(ax.lines) > 0 or len(ax.patches) > 0
    assert ax.get_legend() is not None


@pytest.mark.parametrize("method", ["plot_fit", "plot_forecast"])
def test_combo_plot_smoke_all_horizons(fitted_combo, method):
    """Omitting ``horizon`` stacks one populated subplot per combo horizon."""
    _fig, axes = getattr(fitted_combo, method)()
    axes = axes.ravel()
    assert len(axes) == 2
    for ax in axes:
        assert len(ax.lines) > 0 or len(ax.patches) > 0


def test_combo_plot_forecast_requires_forecast():
    """``plot_forecast`` raises before `forecast()` has been called."""
    target, regressors, _ = sample_combo_data(n_quarters=48, seed=2)
    spec = ComboSpec(
        "final",
        sources=[MidasSpec("monthly_1", "almon", n_lags=6), OLSSpec("quarterly_1")],
        method="average",
    )
    model = MidasCombo(combo_specs=spec, horizons=2).fit(target, regressors)
    with pytest.raises(RuntimeError, match="forecast"):
        model.plot_forecast(horizon=0)


def test_combo_plot_unknown_source_raises(fitted_combo):
    """Requesting a source name that was never fitted is an error."""
    with pytest.raises(ValueError, match="not found"):
        fitted_combo.plot_fit(combo_names="does_not_exist", horizon=0)
