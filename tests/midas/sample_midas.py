"""Small independent real time data generator for MIDAS tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MidasCoefficientTruth:
    """Known parameters for one direct forecasting horizon."""

    theta_true: np.ndarray
    beta: float
    effective_coefficients: np.ndarray


@dataclass(frozen=True)
class MidasTruth:
    """Known values used as an oracle for one sampled MIDAS process."""

    target: pd.DataFrame
    target_date: pd.Timestamp
    target_value: float
    alpha: float
    coefficients_by_horizon: dict[int, MidasCoefficientTruth]
    n_lags: int
    n_pars_weights: int
    dummy_periods: tuple[pd.Timestamp, ...]
    dummy_coefficients: np.ndarray


@dataclass(frozen=True)
class MidasVintage:
    """Data and independent expectations at one information vintage."""

    vintage_date: pd.Timestamp
    published: pd.DataFrame
    regressors: pd.DataFrame
    target_train: pd.DataFrame
    latest_regressor_date: pd.Timestamp
    horizon: int
    expected_lag_rows: np.ndarray
    expected_valid_rows: np.ndarray
    independent_forecast: float


@dataclass(frozen=True)
class MidasSample:
    """Complete sampled process plus its release-filtered vintage views."""

    truth: MidasTruth
    observed: pd.DataFrame
    latent_regressors: pd.DataFrame
    regressors: pd.DataFrame
    vintages: tuple[MidasVintage, ...]

    def regressors_before(self, date: pd.Timestamp) -> pd.DataFrame:
        """Return observed regressors available no later than ``date``."""
        date = pd.Timestamp(date)
        return self.regressors.loc[self.regressors["date"] <= date].reset_index(
            drop=True
        )

    def forecast_truth(self, regressors: pd.DataFrame, horizon: int) -> float:
        """Calculate a forecast from truth and the independent calendar lookup."""
        finite = regressors.loc[np.isfinite(regressors["value"])]
        info_date = pd.Timestamp(finite["date"].max())
        row = _calendar_lag_row(
            info_date,
            regressors,
            self.truth.n_lags,
            _month_within_quarter(regressors),
        )
        forecast_date = (info_date.to_period("Q") + horizon).end_time.normalize()
        coefficients = self.truth.coefficients_by_horizon[horizon]
        value = self.truth.alpha + float(row @ coefficients.effective_coefficients)
        if forecast_date in self.truth.dummy_periods:
            dummy_index = self.truth.dummy_periods.index(forecast_date)
            value += float(self.truth.dummy_coefficients[dummy_index])
        return value


@dataclass(frozen=True)
class VintageMidasSample:
    """A MIDAS process generated from one monthly information vintage."""

    target: pd.DataFrame
    target_train: pd.DataFrame
    regressors: pd.DataFrame
    full_regressors: pd.DataFrame
    vintage_date: pd.Timestamp
    alpha: float
    weights: np.ndarray
    info_date: pd.Timestamp


def _month_within_quarter(regressors: pd.DataFrame) -> int:
    finite = regressors.loc[np.isfinite(regressors["value"])]
    latest = pd.Timestamp(finite["date"].max())
    return ((latest.month - 1) % 3) + 1


def _calendar_lag_row(
    target_date: pd.Timestamp,
    regressors: pd.DataFrame,
    n_lags: int,
    month_within_quarter: int,
) -> np.ndarray:
    """Look up monthly calendar positions without compressing missing months."""
    values = {
        pd.Timestamp(date).to_period("M"): value
        for date, value in zip(regressors["date"], regressors["value"])
    }
    quarter = pd.Timestamp(target_date).to_period("Q")
    anchor = quarter.start_time + pd.DateOffset(months=month_within_quarter - 1)
    anchor = anchor.to_period("M")
    return np.asarray(
        [values.get(anchor - lag, np.nan) for lag in range(n_lags)], dtype=float
    )


def _calendar_lag_rows(
    target_dates: pd.Series,
    regressors: pd.DataFrame,
    n_lags: int,
) -> np.ndarray:
    month_within_quarter = _month_within_quarter(regressors)
    return np.vstack(
        [
            _calendar_lag_row(date, regressors, n_lags, month_within_quarter)
            for date in target_dates
        ]
    )


def _release_dates(monthly_dates: pd.DatetimeIndex) -> pd.Series:
    releases = pd.Series(
        [date + pd.Timedelta(days=5) for date in monthly_dates], index=monthly_dates
    )
    release_months = {
        pd.Timestamp("2023-12-31"): pd.Timestamp("2023-12-15"),
        pd.Timestamp("2024-01-31"): pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-02-29"): pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-03-31"): pd.Timestamp("2024-03-15"),
    }
    for date, release in release_months.items():
        if date in releases.index:
            releases.loc[date] = release
    return releases


def _beta_weights(theta: np.ndarray, n_lags: int) -> np.ndarray:
    """Evaluate normalised beta weights without using production dispatch."""
    grid = np.linspace(0.001, 0.999, n_lags)
    unnormalised = grid ** (theta[0] - 1) * (1 - grid) ** (theta[1] - 1)
    return unnormalised / unnormalised.sum()


def sample_midas(
    method: str = "unrestricted",
    *,
    include_dummy: bool = False,
    noise: float = 0.0,
    seed: int = 2024,
) -> MidasSample:
    """Generate a compact quarterly target and monthly real time regressor.

    The sampler supports both direct relations needed by the release stages:
    the current-quarter relation and the one-quarter-ahead relation each use
    an explicit effective coefficient vector.  Monthly values are
    constructed so that each release within a quarter evaluates the same
    current-quarter signal.  One historical month is absent from the observed
    calendar to make accidental positional compression visible to the tests.
    """
    n_lags = 5
    n_pars_weights = 2
    alpha = 2.5
    if method not in ("almon", "unrestricted", "exp_almon", "beta"):
        raise ValueError(f"unsupported sample method: {method}")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    weights_by_method = {
        "almon": np.array([0.28571429, 0.24285714, 0.2, 0.15714286, 0.11428571]),
        "unrestricted": np.array([0.45, 0.3, 0.15, 0.07, 0.03]),
        "exp_almon": np.array(
            [0.32403955, 0.25746031, 0.19264817, 0.13575687, 0.09009508]
        ),
    }
    horizon_weights_by_method = {
        "almon": np.array([0.3, 0.252, 0.204, 0.156, 0.108]),
        "unrestricted": np.array([0.4, 0.3, 0.2, 0.1, 0.02]),
        "exp_almon": np.array(
            [0.26679189, 0.23662318, 0.20163698, 0.16508639, 0.12986156]
        ),
    }
    theta_by_horizon = {
        "almon": {
            0: np.array([0.28571429, -0.04285714]),
            1: np.array([0.3, -0.048]),
        },
        "unrestricted": {
            0: np.array([0.45, 0.3, 0.15, 0.07, 0.03]),
            1: np.array([0.8, 0.5, 0.3, 0.2, 0.1]),
        },
        "exp_almon": {
            0: np.array([-0.2, -0.03]),
            1: np.array([-0.1, -0.02]),
        },
        "beta": {
            0: np.array([0.8, 3.2]),
            1: np.array([2.0, 5.0]),
        },
    }
    beta_by_horizon = {0: 1.0, 1: 1.0}
    if method in ("exp_almon", "beta"):
        beta_by_horizon[1] = 1.02
    if method == "beta":
        weights_by_method[method] = _beta_weights(theta_by_horizon[method][0], n_lags)
        horizon_weights_by_method[method] = _beta_weights(
            theta_by_horizon[method][1], n_lags
        )
        beta_by_horizon[1] = 1.04
    weights = weights_by_method[method]
    horizon_weights = horizon_weights_by_method[method]
    coefficients_by_horizon = {
        horizon: MidasCoefficientTruth(
            theta_true=theta_by_horizon[method][horizon],
            beta=beta_by_horizon[horizon],
            effective_coefficients=beta_by_horizon[horizon]
            * (weights if horizon == 0 else horizon_weights),
        )
        for horizon in (0, 1)
    }
    horizon_coefficients = coefficients_by_horizon[1].effective_coefficients

    target_dates = pd.date_range("2019-03-31", periods=21, freq="QE")
    monthly_dates = pd.date_range("2018-01-31", "2024-06-30", freq="ME")
    rng = np.random.default_rng(seed)
    monthly_values = []
    signals = {}
    previous_month_three = rng.normal()
    previous_month_two = rng.normal()
    previous_month_one = rng.normal()
    previous_month_zero = rng.normal()
    next_signal = None
    for quarter in monthly_dates.to_period("Q").unique():
        if next_signal is None:
            month_one = rng.normal()
        else:
            month_one = (
                next_signal
                - weights[1] * previous_month_three
                - weights[2] * previous_month_two
                - weights[3] * previous_month_one
                - weights[4] * previous_month_zero
            ) / weights[0]
        signal = (
            weights[0] * month_one
            + weights[1] * previous_month_three
            + weights[2] * previous_month_two
            + weights[3] * previous_month_one
            + weights[4] * previous_month_zero
        )
        month_two = (
            signal
            - weights[1] * month_one
            - weights[2] * previous_month_three
            - weights[3] * previous_month_two
            - weights[4] * previous_month_one
        ) / weights[0]
        month_three = (
            signal
            - weights[1] * month_two
            - weights[2] * month_one
            - weights[3] * previous_month_three
            - weights[4] * previous_month_two
        ) / weights[0]
        signals[quarter] = signal
        monthly_values.extend([month_one, month_two, month_three])
        next_signal = (
            horizon_coefficients[0] * month_three
            + horizon_coefficients[1] * month_two
            + horizon_coefficients[2] * month_one
            + horizon_coefficients[3] * previous_month_three
            + horizon_coefficients[4] * previous_month_two
        )
        previous_month_zero = previous_month_three
        previous_month_two = month_two
        previous_month_three = month_three
        previous_month_one = month_one
    monthly_values = np.asarray(monthly_values[: len(monthly_dates)])
    latent = pd.DataFrame({"date": monthly_dates, "value": monthly_values})

    observed = latent.copy()
    observed.loc[observed["date"] == pd.Timestamp("2021-02-28"), "value"] = np.nan
    releases = _release_dates(monthly_dates)
    observed["release_date"] = observed["date"].map(releases)

    target_values = np.asarray(
        [alpha + signals[date.to_period("Q")] for date in target_dates], dtype=float
    )
    target_values += noise * rng.standard_normal(len(target_values))

    dummy_periods: tuple[pd.Timestamp, ...] = ()
    dummy_coefficients = np.array([], dtype=float)
    if include_dummy:
        dummy_periods = (pd.Timestamp(target_dates[-2]),)
        dummy_coefficients = np.array([4.0])
        target_values[-2] += dummy_coefficients[0]

    target = pd.DataFrame({"date": target_dates, "value": target_values})
    truth = MidasTruth(
        target=target,
        target_date=pd.Timestamp(target_dates[-1]),
        target_value=float(target_values[-1]),
        alpha=alpha,
        coefficients_by_horizon=coefficients_by_horizon,
        n_lags=n_lags,
        n_pars_weights=n_pars_weights,
        dummy_periods=dummy_periods,
        dummy_coefficients=dummy_coefficients,
    )

    vintage_dates = pd.to_datetime(
        ["2023-12-20", "2024-01-15", "2024-02-15", "2024-03-15"]
    )
    vintages = []
    for vintage_date in vintage_dates:
        published = observed.loc[observed["release_date"] <= vintage_date].reset_index(
            drop=True
        )
        regressors = published[["date", "value"]].copy()
        expected = _calendar_lag_rows(target_dates, regressors, n_lags)
        valid = ~np.any(np.isnan(expected), axis=1)
        latest = pd.Timestamp(
            regressors.loc[np.isfinite(regressors["value"]), "date"].max()
        )
        horizon = (
            truth.target_date.to_period("Q").ordinal - latest.to_period("Q").ordinal
        )
        forecast_weights = truth.coefficients_by_horizon[horizon].effective_coefficients
        expected_forecast = alpha + float(
            _calendar_lag_row(
                latest, regressors, n_lags, _month_within_quarter(regressors)
            )
            @ forecast_weights
        )
        vintages.append(
            MidasVintage(
                vintage_date=vintage_date,
                published=published,
                regressors=regressors,
                target_train=target.iloc[:-1].copy(),
                latest_regressor_date=latest,
                horizon=horizon,
                expected_lag_rows=expected[:-1],
                expected_valid_rows=valid[:-1],
                independent_forecast=expected_forecast,
            )
        )

    return MidasSample(
        truth=truth,
        observed=observed.copy(),
        latent_regressors=latent.copy(),
        regressors=observed[["date", "value"]].copy(),
        vintages=tuple(vintages),
    )


def sample_vintage_midas(
    vintage_date: str | pd.Timestamp = "2020-02-14",
) -> VintageMidasSample:
    """Generate a MIDAS process from a release-filtered vintage.

    Monthly observations are released on the first day of the following
    month.  The target is generated from the same calendar-aligned lag
    matrix that is available after filtering releases at ``vintage_date``.
    This fixture is intentionally not invariant to the month within the
    quarter, so using a later vintage provides a meaningful recovery test.
    """
    vintage_date = pd.Timestamp(vintage_date)
    target_dates = pd.date_range("2017-03-31", "2020-03-31", freq="QE")
    monthly_dates = pd.date_range("2015-01-31", "2020-03-31", freq="ME")

    rng = np.random.default_rng(2026)
    full_regressors = pd.DataFrame(
        {
            "date": monthly_dates,
            "value": rng.standard_normal(len(monthly_dates)),
            "release_date": monthly_dates + pd.offsets.MonthBegin(1),
        }
    )
    published = full_regressors.loc[
        full_regressors["release_date"] <= vintage_date
    ].reset_index(drop=True)
    regressors = published[["date", "value"]].copy()
    info_date = pd.Timestamp(regressors["date"].max())

    alpha = 1.75
    weights = np.array([0.55, 0.2, -0.1, 0.15, 0.08])
    lag_rows = _calendar_lag_rows(target_dates, regressors, len(weights))
    target = pd.DataFrame({"date": target_dates, "value": alpha + lag_rows @ weights})

    return VintageMidasSample(
        target=target,
        target_train=target.iloc[:-1].copy(),
        regressors=regressors,
        full_regressors=full_regressors,
        vintage_date=vintage_date,
        alpha=alpha,
        weights=weights,
        info_date=info_date,
    )
