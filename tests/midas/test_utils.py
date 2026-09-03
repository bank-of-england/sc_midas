"""Direct tests for the sample-data generators in :mod:`nowcast_midas.utils`.

`~nowcast_midas.utils.sample_data` and
`~nowcast_midas.utils.sample_combo_data` underpin most of the test
suite but were never checked in their own right.  These tests pin down
their output schema, seed reproducibility, horizon validation and the
optional target outlier.
"""

import numpy as np
import pandas as pd
import pytest

from nowcast_midas.utils import sample_combo_data, sample_data

# --------------------------------------------------------------------------- #
#  sample_data                                                                #
# --------------------------------------------------------------------------- #


def test_sample_data_schema_and_dtypes():
    target, regressors = sample_data(n_obs=80, n_lags=6, seed=0)
    assert list(target.columns) == ["date", "value"]
    assert list(regressors.columns) == ["date", "value"]
    assert pd.api.types.is_datetime64_any_dtype(target["date"])
    assert pd.api.types.is_datetime64_any_dtype(regressors["date"])
    assert pd.api.types.is_float_dtype(target["value"])
    assert not target["value"].isna().any()
    # One monthly row per month spanning the (padded) quarterly range.
    assert len(regressors) > 3 * len(target)


def test_sample_data_is_reproducible_for_a_seed():
    a_t, a_r = sample_data(n_obs=60, seed=123)
    b_t, b_r = sample_data(n_obs=60, seed=123)
    pd.testing.assert_frame_equal(a_t, b_t)
    pd.testing.assert_frame_equal(a_r, b_r)


def test_sample_data_seed_changes_values_not_dates():
    a_t, _ = sample_data(n_obs=60, seed=1)
    b_t, _ = sample_data(n_obs=60, seed=2)
    pd.testing.assert_series_equal(a_t["date"], b_t["date"])
    assert not np.allclose(a_t["value"], b_t["value"])


def test_sample_data_horizon_out_of_range_raises():
    with pytest.raises(ValueError, match="horizon must be in"):
        sample_data(n_obs=40, seed=0, horizon=999)


def test_sample_data_negative_ar_lags_raises():
    with pytest.raises(ValueError, match="n_ar_lags must be >= 0"):
        sample_data(n_obs=40, seed=0, n_ar_lags=-1)


def test_sample_data_ar_lags_embed_persistence():
    target_ar, _ = sample_data(n_obs=200, seed=0, n_ar_lags=2, phi_true=[0.6, 0.0])
    target_iid, _ = sample_data(n_obs=200, seed=0, n_ar_lags=0)
    ac_ar = target_ar["value"].autocorr(lag=1)
    ac_iid = target_iid["value"].autocorr(lag=1)
    assert ac_ar > ac_iid


# --------------------------------------------------------------------------- #
#  sample_combo_data                                                          #
# --------------------------------------------------------------------------- #


def test_sample_combo_data_long_format_schema():
    target, regressors, info = sample_combo_data(n_quarters=48, seed=0)
    expected = {"date", "variable", "frequency", "value"}
    assert set(target.columns) == expected
    assert set(regressors.columns) == expected
    assert set(regressors["frequency"].unique()) == {"ME", "QE"}
    assert set(info["monthly_vars"]).issubset(set(regressors["variable"]))
    assert set(info["quarterly_vars"]).issubset(set(regressors["variable"]))
    assert len(info["weights"]) == 6


def test_sample_combo_data_custom_variable_names():
    _target, regressors, info = sample_combo_data(
        n_quarters=48,
        seed=0,
        monthly_vars=["soft_1", "soft_2"],
        quarterly_vars=["hard_1"],
    )
    assert info["monthly_vars"] == ["soft_1", "soft_2"]
    monthly = regressors[regressors["frequency"] == "ME"]
    assert set(monthly["variable"].unique()) == {"soft_1", "soft_2"}


def test_sample_combo_data_is_reproducible_for_a_seed():
    a_t, a_r, _ = sample_combo_data(n_quarters=44, seed=7)
    b_t, b_r, _ = sample_combo_data(n_quarters=44, seed=7)
    pd.testing.assert_frame_equal(a_t, b_t)
    pd.testing.assert_frame_equal(a_r, b_r)


def test_sample_combo_data_outlier_is_injected_at_requested_quarter():
    clean_t, _, _ = sample_combo_data(n_quarters=48, seed=3, outlier_date=None)
    shocked_t, _, info = sample_combo_data(
        n_quarters=48, seed=3, outlier_date="2020-06-30", outlier_size=-25.0
    )

    assert info["outlier_date"] == pd.Timestamp("2020-06-30")
    diff = shocked_t["value"].to_numpy() - clean_t["value"].to_numpy()

    hit = shocked_t["date"] == pd.Timestamp("2020-06-30")
    assert np.isclose(diff[hit.to_numpy()][0], -25.0)
    # Every other quarter is untouched.
    np.testing.assert_allclose(diff[~hit.to_numpy()], 0.0, atol=1e-12)


def test_sample_combo_data_outlier_date_none_disables_shock():
    _, _, info = sample_combo_data(n_quarters=48, seed=3, outlier_date=None)
    assert info["outlier_date"] is None


def test_sample_combo_data_outlier_outside_sample_raises():
    with pytest.raises(ValueError, match="not one of the simulated quarters"):
        sample_combo_data(n_quarters=20, seed=0, outlier_date="1990-06-30")


def test_sample_combo_data_horizon_out_of_range_raises():
    with pytest.raises(ValueError, match="horizon must be in"):
        sample_combo_data(n_quarters=20, seed=0, horizon=999)
