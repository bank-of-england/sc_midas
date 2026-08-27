"""Tests for minimum_sample_size functionality.

Tests that minimum_sample_size parameter in specs works correctly to filter
models based on number of fitted observations.
"""

import numpy as np
import pandas as pd

from sc_midas.midas_combo import MidasCombo
from sc_midas.specs import ComboSpec, MidasSpec


def test_minimum_sample_size_in_nested_combo():
    """Minimum sample size on inner ComboSpec excludes its direct descendants."""
    rng = np.random.RandomState(123)

    # Generate small target (40 quarters = 10 years)
    n_quarters = 40
    target = pd.DataFrame(
        {
            "date": pd.date_range("2014-01-01", periods=n_quarters, freq="QE"),
            "variable": "y",
            "frequency": "QE",
            "value": np.cumsum(rng.standard_normal(n_quarters)) + 100,
        }
    )

    # Generate regressors (all monthly)
    dates = pd.date_range("2014-01-01", periods=n_quarters * 3, freq="ME")
    regressors = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "variable": "x1",
                    "frequency": "ME",
                    "value": np.cumsum(rng.standard_normal(len(dates))) + 50,
                }
            ),
            pd.DataFrame(
                {
                    "date": dates,
                    "variable": "x2",
                    "frequency": "ME",
                    "value": np.cumsum(rng.standard_normal(len(dates))) + 60,
                }
            ),
            pd.DataFrame(
                {
                    "date": dates,
                    "variable": "x3",
                    "frequency": "ME",
                    "value": np.cumsum(rng.standard_normal(len(dates))) + 70,
                }
            ),
        ],
        ignore_index=True,
    )

    # Create three MIDAS specs
    midas_x1 = MidasSpec(variable="x1", method="almon", n_lags=6)
    midas_x2 = MidasSpec(variable="x2", method="almon", n_lags=6)
    midas_x3 = MidasSpec(variable="x3", method="almon", n_lags=6)

    # Inner combo (soft_combo) with minimum_sample_size
    # This combo contains x1 and x2, and has minimum_sample_size=50
    soft_combo = ComboSpec(
        name="soft_combo",
        sources=[midas_x1, midas_x2],
        method="average",
        minimum_sample_size=50,  # High threshold to exclude x1 and x2
    )

    # Outer combo (final_combo) with x3 and soft_combo
    # No elevated minimum_sample_size set on outer combo
    final_combo = ComboSpec(
        name="final_combo",
        sources=[soft_combo, midas_x3],
        method="average",
        minimum_sample_size=100,  # High threshold to exclude midas_x3
    )

    mc = MidasCombo(combo_specs=final_combo, horizons=1)

    mc.fit(target=target, regressors=regressors)

    weights = mc.weights_df_
    weights_softs = weights[weights["source"].isin(["x1", "x2"])]
    weights_softs = weights_softs.dropna(subset=["value"])

    # check that all weights are 0.5
    assert np.allclose(weights_softs["value"], 0.5), (
        "Weights for x1 and x2 should be 0.5"
    )

    weights_x3 = weights[weights["source"] == "x3"]
    weights_x3 = weights_x3.dropna(subset=["value"])

    # check that all weights are 0
    assert np.allclose(weights_x3["value"], 0), "Weights for x3 should be 0"
