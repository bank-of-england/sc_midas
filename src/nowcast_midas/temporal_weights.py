"""MIDAS lag-weighting functions."""

import jax.numpy as jnp

__all__ = ["almon", "beta", "exp_almon", "get_weights", "unrestricted"]


def exp_almon(theta: jnp.ndarray, n_lags: int) -> jnp.ndarray:
    """Normalised Exponential Almon lag weights.

    Computes ``w_j = exp(sum_i theta_i * j^(i+1))`` for ``j = 0, …, n_lags-1``,
    then normalises so that the weights sum to one.

    Parameters
    ----------
    theta : jnp.ndarray
        Shape parameters of the exponential polynomial.
    n_lags : int
        Number of high-frequency lags.

    Returns
    -------
    weights : jnp.ndarray
        Normalised lag weights.
    """
    theta = jnp.asarray(theta, dtype=float)
    j = jnp.arange(n_lags, dtype=float)
    w = jnp.exp(sum(theta[i] * j ** (i + 1) for i in range(theta.shape[0])))
    return w / w.sum()


def beta(theta: jnp.ndarray, n_lags: int) -> jnp.ndarray:
    """Beta polynomial lag weights.

    Evaluates the Beta density on an equally-spaced grid over ``(0, 1)``
    using shape parameters ``(a, b) = theta``, then normalises the result.
    Both parameters are clipped to a minimum of ``1e-6`` for numerical
    stability.

    Parameters
    ----------
    theta : jnp.ndarray
        Shape parameters ``[a, b]`` of the Beta distribution.
    n_lags : int
        Number of high-frequency lags.

    Returns
    -------
    weights : jnp.ndarray
        Normalised lag weights.
    """
    theta = jnp.asarray(theta, dtype=float)
    a = jnp.clip(theta[0], 1e-6, None)
    b = jnp.clip(theta[1], 1e-6, None)
    grid = jnp.linspace(0.001, 0.999, n_lags)
    w = grid ** (a - 1) * (1 - grid) ** (b - 1)
    return w / w.sum()


def almon(theta: jnp.ndarray, n_lags: int) -> jnp.ndarray:
    """Polynomial Almon lag weights.

    Computes ``w_j = sum_i theta_i * j^i`` for ``j = 0, …, n_lags-1``, then
    normalises so that the weights sum to one.

    Parameters
    ----------
    theta : jnp.ndarray
        Polynomial coefficients, one per degree starting from degree 0.
    n_lags : int
        Number of high-frequency lags.

    Returns
    -------
    weights : jnp.ndarray
        Normalised lag weights.
    """
    theta = jnp.asarray(theta, dtype=float)
    j = jnp.arange(n_lags, dtype=float)
    w = sum(theta[i] * j**i for i in range(theta.shape[0]))
    return w / w.sum()


def unrestricted(theta: jnp.ndarray, n_lags: int) -> jnp.ndarray:
    """Unrestricted (U-MIDAS) lag weights.

    Returns ``theta`` unchanged, so the model estimates one coefficient for
    each lag without a parametric restriction.

    Parameters
    ----------
    theta : jnp.ndarray
        One coefficient for each lag.
    n_lags : int
        Number of high-frequency lags (unused; present for API consistency).

    Returns
    -------
    weights : jnp.ndarray
        The input ``theta``, unmodified.
    """
    return jnp.asarray(theta, dtype=float)


_WEIGHTS = {
    "exp_almon": exp_almon,
    "beta": beta,
    "almon": almon,
    "unrestricted": unrestricted,
}


def get_weights(method: str, theta: jnp.ndarray, n_lags: int) -> jnp.ndarray:
    """Return lag weights for the selected weighting scheme.

    Parameters
    ----------
    method : str
        Name of the weighting scheme.
    theta : jnp.ndarray
        Shape parameters passed to the weight function.
    n_lags : int
        Number of high-frequency lags.

    Returns
    -------
    weights : jnp.ndarray
        Lag weights returned by the selected scheme.
    """
    return _WEIGHTS[method](theta, n_lags)


def _normalised_jacobian(u: jnp.ndarray, du: jnp.ndarray) -> jnp.ndarray:
    """Jacobian of ``w = u / sum(u)`` given ``u`` and ``du/dtheta``.

    Applies the quotient rule
    ``dw/dtheta_m = (du[:, m] - w * sum(du[:, m])) / S`` where ``S = sum(u)``.

    Parameters
    ----------
    u : jnp.ndarray
        Unnormalised weights.
    du : jnp.ndarray
        Derivatives of the unnormalised weights w.r.t. each parameter.

    Returns
    -------
    jnp.ndarray
        Derivatives of the normalised weights w.r.t. each parameter.
    """
    s = u.sum()
    w = u / s
    return (du - jnp.outer(w, du.sum(axis=0))) / s


def _weight_jacobian(method: str, theta: jnp.ndarray, n_lags: int) -> jnp.ndarray:
    """Analytical Jacobian ``dw/dtheta`` of the lag weights.

    Parameters
    ----------
    method : str
        Name of the weighting scheme.
    theta : jnp.ndarray
        Shape parameters.
    n_lags : int
        Number of high-frequency lags.

    Returns
    -------
    jnp.ndarray
        Partial derivatives of each weight w.r.t. each parameter.

    Raises
    ------
    ValueError
        If *method* is not a supported weighting scheme.
    """
    theta = jnp.asarray(theta, dtype=float)
    j = jnp.arange(n_lags, dtype=float)

    if method == "unrestricted":
        return jnp.eye(n_lags)

    if method == "almon":
        u = jnp.asarray(
            sum(theta[i] * j**i for i in range(theta.shape[0])), dtype=float
        )
        du = jnp.column_stack([j**i for i in range(theta.shape[0])])
        return _normalised_jacobian(u, du)

    if method == "exp_almon":
        u = jnp.exp(sum(theta[i] * j ** (i + 1) for i in range(theta.shape[0])))
        du = jnp.column_stack([u * j ** (i + 1) for i in range(theta.shape[0])])
        return _normalised_jacobian(u, du)

    if method == "beta":
        a = jnp.clip(theta[0], 1e-6, None)
        b = jnp.clip(theta[1], 1e-6, None)
        grid = jnp.linspace(0.001, 0.999, n_lags)
        u = grid ** (a - 1) * (1 - grid) ** (b - 1)
        du = jnp.column_stack([u * jnp.log(grid), u * jnp.log(1 - grid)])
        return _normalised_jacobian(u, du)

    raise ValueError(f"Unknown method '{method}'")
