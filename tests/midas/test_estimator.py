"""Cross-checks on the OLS estimator's linear-algebra internals.

Historically this file held only a ``TODO`` note about comparing the sum
of squared residuals against an EViews reference run.  We do not have that
reference dataset, but the property it was meant to protect -- that the
reported residuals solve the ordinary-least-squares normal equations for
the design matrix ``A`` -- can be verified directly against
`numpy.linalg.lstsq`.
"""

import numpy as np

from nowcast_midas.midas import MIDAS
from nowcast_midas.utils import sample_data


def test_ols_sse_matches_lstsq_reference():
    """The fitted residuals minimise ``||y - A beta||^2`` for the design ``A``."""
    target, regressors = sample_data(n_obs=120, method="almon", seed=0)
    fit = (
        MIDAS(method="almon", n_lags=6, estimator="ols")
        .fit(target, regressors)
        .fits_[0]
    )

    beta_hat, *_ = np.linalg.lstsq(fit.A, fit.y, rcond=None)
    ref_sse = float(np.sum((fit.y - fit.A @ beta_hat) ** 2))

    assert np.isclose(float(fit.residuals @ fit.residuals), ref_sse, rtol=1e-8)


def test_ols_residuals_orthogonal_to_design():
    """OLS residuals are orthogonal to every column of the design matrix."""
    target, regressors = sample_data(n_obs=120, method="almon", seed=1)
    fit = (
        MIDAS(method="unrestricted", n_lags=6, estimator="ols")
        .fit(target, regressors)
        .fits_[0]
    )

    scale = np.linalg.norm(fit.y) * np.sqrt(fit.nobs)
    np.testing.assert_allclose(fit.A.T @ fit.residuals, 0.0, atol=1e-8 * scale)
