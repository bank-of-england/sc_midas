"""Independent real time data generator for MultiMIDAS tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sc_midas.specs import VariableSpec


@dataclass(frozen=True)
class MultiVariableTruth:
    """Known contribution coefficients for one sampled variable."""

    variable: str
    frequency: str
    method: str
    n_lags: int
    start_lag: int
    theta_true: np.ndarray
    scale: float
    effective_coefficients: dict[int, np.ndarray]


@dataclass(frozen=True)
class MultiTruth:
    """Known values and parameters for a sampled joint process."""

    target: pd.DataFrame
    target_date: pd.Timestamp
    target_value: float
    alpha: float
    variables: tuple[VariableSpec, ...]
    variables_truth: dict[str, MultiVariableTruth]
    dummy_periods: tuple[pd.Timestamp, ...]
    dummy_coefficients: np.ndarray
    common_origin_equation: str


@dataclass(frozen=True)
class MultiVintage:
    """Release-filtered data and independent expectations at one vintage."""

    vintage_date: pd.Timestamp
    published: pd.DataFrame
    regressors: pd.DataFrame
    target_train: pd.DataFrame
    common_origin_date: pd.Timestamp
    horizon: int
    expected_blocks: dict[str, np.ndarray]
    expected_valid_rows: np.ndarray
    expected_coefficients: dict[str, np.ndarray]
    independent_forecast: float


@dataclass(frozen=True)
class MultiSample:
    """Complete sampled process and its release-filtered vintage views."""

    truth: MultiTruth
    observed: pd.DataFrame
    latent_regressors: pd.DataFrame
    vintages: tuple[MultiVintage, ...]


def _family_parameters(method: str, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    if method == "unrestricted":
        theta = np.array([0.75, -0.2] + [0.08] * (n_lags - 2))[:n_lags]
        return theta, theta.copy()
    if method == "almon":
        theta = np.array([0.8, -0.15])
        lag = np.arange(n_lags, dtype=float)
        return theta, theta[0] + theta[1] * lag
    if method == "exp_almon":
        theta = np.array([-0.12, -0.015])
        lag = np.arange(n_lags, dtype=float)
        weights = np.exp(theta[0] * lag + theta[1] * lag**2)
        return theta, weights / weights.sum()
    if method == "beta":
        theta = np.array([1.0, 2.0])
        grid = np.linspace(0.001, 0.999, n_lags)
        weights = grid ** (theta[0] - 1) * (1 - grid) ** (theta[1] - 1)
        return theta, weights / weights.sum()
    raise ValueError(f"unsupported sample method: {method}")


def _calendar_monthly_row(
    target_date: pd.Timestamp,
    data: pd.DataFrame,
    n_lags: int,
    start_lag: int,
) -> np.ndarray:
    finite_dates = data.loc[np.isfinite(data["value"]), "date"]
    if finite_dates.empty:
        return np.full(n_lags, np.nan)
    latest = pd.Timestamp(finite_dates.max())
    month_within_quarter = ((latest.month - 1) % 3) + 1
    quarter_start = pd.Timestamp(target_date).to_period("Q").start_time
    anchor = (quarter_start + pd.DateOffset(months=month_within_quarter - 1)).to_period(
        "M"
    )
    values = {
        pd.Timestamp(date).to_period("M"): value
        for date, value in zip(data["date"], data["value"])
    }
    return np.asarray(
        [
            values.get(anchor - lag, np.nan)
            for lag in range(start_lag, start_lag + n_lags)
        ],
        dtype=float,
    )


def _calendar_quarterly_row(
    target_date: pd.Timestamp, data: pd.DataFrame, n_lags: int, start_lag: int
) -> np.ndarray:
    values = {
        pd.Timestamp(date).to_period("Q"): value
        for date, value in zip(data["date"], data["value"])
    }
    quarter = pd.Timestamp(target_date).to_period("Q")
    return np.asarray(
        [
            values.get(quarter - lag, np.nan)
            for lag in range(start_lag, start_lag + n_lags)
        ],
        dtype=float,
    )


def _monthly_rows(
    dates: pd.Series, data: pd.DataFrame, n_lags: int, start_lag: int
) -> np.ndarray:
    return np.vstack(
        [
            _calendar_monthly_row(pd.Timestamp(date), data, n_lags, start_lag)
            for date in dates
        ]
    )


def _quarterly_rows(
    dates: pd.Series, data: pd.DataFrame, n_lags: int, start_lag: int
) -> np.ndarray:
    return np.vstack(
        [
            _calendar_quarterly_row(pd.Timestamp(date), data, n_lags, start_lag)
            for date in dates
        ]
    )


def _generate_monthly_series(
    dates: pd.DatetimeIndex,
    signals: dict[pd.Period, float],
    coefficients: np.ndarray,
    rng: np.random.Generator,
    start_lag: int = 0,
) -> pd.DataFrame:
    """Create monthly values whose weighted signal is fixed within each quarter."""
    n_lags = len(coefficients)
    if start_lag:
        values: list[float] = []
        previous = list(rng.normal(size=n_lags + start_lag))
        for quarter in dates.to_period("Q").unique():
            signal = signals[quarter]
            quarter_values = list(rng.normal(size=3))
            for included_month in range(2 - start_lag, -1, -1):
                if included_month < 0:
                    break
                prior = quarter_values[:included_month][::-1] + previous
                contribution = float(np.dot(coefficients[1:], prior[: n_lags - 1]))
                quarter_values[included_month] = (signal - contribution) / coefficients[
                    0
                ]
            values.extend(quarter_values)
            previous = (quarter_values[::-1] + previous)[: n_lags + start_lag]
        return pd.DataFrame({"date": dates, "value": np.asarray(values[: len(dates)])})
    values: list[float] = []
    previous = list(rng.normal(size=n_lags))
    for quarter in dates.to_period("Q").unique():
        signal = signals[quarter]
        quarter_values: list[float] = []
        for _ in range(3):
            lagged = quarter_values[::-1] + previous
            current = (
                signal - float(np.dot(coefficients[1:], lagged[: n_lags - 1]))
            ) / (coefficients[0])
            quarter_values.append(current)
        values.extend(quarter_values)
        previous = (quarter_values[::-1] + previous)[:n_lags]
    return pd.DataFrame({"date": dates, "value": np.asarray(values[: len(dates)])})


def _generate_recurrent_monthly_series(
    dates: pd.DatetimeIndex,
    frequency: float,
    alternating: bool,
    sine_scale: float = 0.2,
) -> pd.DataFrame:
    month_index = np.arange(len(dates))
    real_rate = -0.99 if alternating else 0.99
    real_mode = real_rate**month_index
    values = real_mode + np.cos(frequency * month_index)
    values += sine_scale * np.sin(frequency * month_index)
    return pd.DataFrame({"date": dates, "value": values})


def _release_dates(dates: pd.DatetimeIndex, variable: str) -> pd.Series:
    if variable == "survey":
        return pd.Series([date - pd.Timedelta(days=16) for date in dates], index=dates)
    if variable == "activity":
        return pd.Series([date + pd.Timedelta(days=10) for date in dates], index=dates)
    if variable == "quarterly":
        quarter_ends = dates.to_period("Q").end_time.normalize()
        return pd.Series(
            [date - pd.Timedelta(days=80) for date in quarter_ends], index=dates
        )
    raise ValueError(f"unknown sampled variable: {variable}")


def sample_multi_midas(
    method: str = "unrestricted",
    *,
    include_quarterly: bool = False,
    include_dummy: bool = False,
    n_lags: int = 2,
    start_lag: int = 0,
    noise: float = 0.0,
    seed: int = 2024,
) -> MultiSample:
    """Generate a noiseless joint MIDAS process with release calendars."""
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if start_lag < 0:
        raise ValueError("start_lag must be non-negative")

    alpha = 2.5
    target_dates = pd.date_range("2018-03-31", periods=25, freq="QE")
    monthly_dates = pd.date_range("2016-01-31", "2024-03-31", freq="ME")
    rng = np.random.default_rng(seed)
    base_theta, base_weights = _family_parameters(method, n_lags)
    activity_theta = base_theta.copy()
    activity_method = method
    survey_scale = 1.0
    activity_scale = 0.6
    if start_lag == 0 and n_lags == 2 and method in ("almon", "unrestricted"):
        base_theta = np.array([0.75, -0.2])
        base_weights = base_theta.copy()
        if method == "almon":
            activity_theta = np.array([0.4125, -0.1325])
        else:
            activity_theta = np.array([0.4125, 0.28])
        activity_scale = 1.0
        activity_weights = activity_theta.copy()
    else:
        activity_weights = base_weights
    if method == "beta" and start_lag == 0:
        activity_method = "almon"
        activity_theta = np.array([0.8, -0.15])
        lag = np.arange(n_lags, dtype=float)
        activity_weights = activity_theta[0] + activity_theta[1] * lag
    base_coefficients = survey_scale * base_weights
    activity_coefficients = activity_scale * activity_weights
    horizon_coefficients = 0.8 * base_coefficients
    activity_horizon_coefficients = 0.8 * activity_coefficients

    survey_signals = {
        quarter: float(rng.normal())
        for quarter in monthly_dates.to_period("Q").unique()
    }
    survey = _generate_monthly_series(
        monthly_dates, survey_signals, base_coefficients, rng, start_lag
    )
    survey_rows = {
        quarter: _calendar_monthly_row(
            quarter.end_time,
            survey,
            n_lags,
            start_lag,
        )
        for quarter in monthly_dates.to_period("Q").unique()
    }
    survey_h0 = {
        quarter: float(row @ base_coefficients) for quarter, row in survey_rows.items()
    }
    survey_h1 = {
        quarter: float(row @ horizon_coefficients)
        for quarter, row in survey_rows.items()
    }

    activity_signals: dict[pd.Period, float] = {}
    if start_lag == 0:
        previous_total = float(rng.normal())
        for quarter in monthly_dates.to_period("Q").unique():
            activity_signals[quarter] = 0.8 * previous_total - survey_h0[quarter]
            previous_total = survey_h0[quarter] + activity_signals[quarter]
    else:
        activity_signals = {
            quarter: float(rng.normal())
            for quarter in monthly_dates.to_period("Q").unique()
        }
    target_quarter = target_dates[-1].to_period("Q")
    if start_lag == 0:
        activity_signals[target_quarter] = activity_signals[target_quarter - 1]
    activity = _generate_monthly_series(
        monthly_dates,
        activity_signals,
        activity_coefficients,
        rng,
        start_lag,
    )
    start_lag_signals = None
    if start_lag:
        survey = _generate_recurrent_monthly_series(monthly_dates, 0.7, False)
        month_index = np.arange(len(monthly_dates))
        activity_frequency = 1.1
        activity_base = (-0.99) ** month_index + np.cos(
            activity_frequency * month_index
        )
        activity_sine_scale = -(activity_base[95] - activity_base[98]) / (
            np.sin(activity_frequency * 95) - np.sin(activity_frequency * 98)
        )
        activity = _generate_recurrent_monthly_series(
            monthly_dates,
            activity_frequency,
            True,
            activity_sine_scale,
        )
        quarter_index = np.arange(len(monthly_dates.to_period("Q").unique()))
        start_lag_signals = {
            "survey": survey["value"].to_numpy()[3 * quarter_index + 2],
            "activity": activity["value"].to_numpy()[3 * quarter_index + 2],
        }
    activity_rows = {
        quarter: _calendar_monthly_row(quarter.end_time, activity, n_lags, start_lag)
        for quarter in monthly_dates.to_period("Q").unique()
    }
    activity_h0 = {
        quarter: float(row @ activity_coefficients)
        for quarter, row in activity_rows.items()
    }
    activity_target_rows = {
        quarter: _calendar_monthly_row(
            quarter.start_time + pd.DateOffset(months=1),
            activity.loc[activity["date"] <= pd.Timestamp("2024-02-29")],
            n_lags,
            start_lag,
        )
        for quarter in monthly_dates.to_period("Q").unique()
    }
    activity_target_h0 = {
        quarter: float(row @ activity_coefficients)
        for quarter, row in activity_target_rows.items()
    }
    activity_h1 = {
        quarter: float(row @ activity_horizon_coefficients)
        for quarter, row in activity_rows.items()
    }
    if start_lag == 0:
        survey_signals[target_quarter] = (
            0.8 * survey_h0[target_quarter - 1] - 0.2 * activity_h0[target_quarter - 1]
        )
        survey = _generate_monthly_series(
            monthly_dates,
            survey_signals,
            base_coefficients,
            rng,
            start_lag,
        )
        survey_rows = {
            quarter: _calendar_monthly_row(
                quarter.end_time,
                survey,
                n_lags,
                start_lag,
            )
            for quarter in monthly_dates.to_period("Q").unique()
        }
        survey_h0 = {
            quarter: float(row @ base_coefficients)
            for quarter, row in survey_rows.items()
        }
        survey_h1 = {
            quarter: float(row @ horizon_coefficients)
            for quarter, row in survey_rows.items()
        }

    latent_parts = []
    variables = [
        VariableSpec(
            "survey",
            method=method,
            n_lags=n_lags,
            n_pars_weights=2,
            start_lag=start_lag,
        ),
        VariableSpec(
            "activity",
            method=activity_method,
            n_lags=n_lags,
            n_pars_weights=2,
            start_lag=start_lag,
        ),
    ]
    variable_truths = {
        "survey": MultiVariableTruth(
            variable="survey",
            frequency="ME",
            method=method,
            n_lags=n_lags,
            start_lag=start_lag,
            theta_true=base_theta.copy(),
            scale=survey_scale,
            effective_coefficients={
                0: base_coefficients,
                1: horizon_coefficients,
            },
        ),
        "activity": MultiVariableTruth(
            variable="activity",
            frequency="ME",
            method=activity_method,
            n_lags=n_lags,
            start_lag=start_lag,
            theta_true=activity_theta.copy(),
            scale=activity_scale,
            effective_coefficients={
                0: activity_coefficients,
                1: activity_horizon_coefficients,
            },
        ),
    }
    for variable, data in (("survey", survey), ("activity", activity)):
        data = data.copy()
        if variable == "survey":
            data.loc[data["date"] == pd.Timestamp("2021-02-28"), "value"] = np.nan
        data["variable"] = variable
        data["release_date"] = _release_dates(monthly_dates, variable).to_numpy()
        latent_parts.append(data)

    quarterly_values = None
    quarterly_horizon_scale = 0.9
    if include_quarterly:
        quarterly_dates = pd.date_range("2016-03-31", "2024-03-31", freq="QE")
        quarterly_values = np.empty(len(quarterly_dates), dtype=float)
        quarterly_values[0] = rng.normal()
        for index in range(1, len(quarterly_values)):
            quarterly_values[index] = (
                quarterly_horizon_scale * quarterly_values[index - 1]
            )
        quarterly = pd.DataFrame({"date": quarterly_dates, "value": quarterly_values})
        quarterly["variable"] = "quarterly"
        quarterly["release_date"] = _release_dates(
            quarterly_dates, "quarterly"
        ).to_numpy()
        latent_parts.append(quarterly)
        variables.append(VariableSpec("quarterly", frequency="QE", n_lags=1))
        quarterly_truth = np.array([0.6])
        variable_truths["quarterly"] = MultiVariableTruth(
            variable="quarterly",
            frequency="QE",
            method="ols",
            n_lags=1,
            start_lag=0,
            theta_true=np.array([0.6]),
            scale=1.0,
            effective_coefficients={
                0: quarterly_truth,
                1: quarterly_horizon_scale * quarterly_truth,
            },
        )

    latent = pd.concat(latent_parts, ignore_index=True)
    target_values = []
    quarterly_by_period = {}
    if quarterly_values is not None:
        quarterly_by_period = dict(
            zip(
                pd.to_datetime(quarterly_dates).to_period("Q"),
                quarterly_values,
            )
        )
    target_quarters = target_dates.to_period("Q")
    if start_lag:
        all_quarters = monthly_dates.to_period("Q").unique()
        signal_index = {quarter: index for index, quarter in enumerate(all_quarters)}
        target_values = np.asarray(
            [
                alpha
                + start_lag_signals["survey"][signal_index[quarter]]
                + start_lag_signals["activity"][signal_index[quarter]]
                for quarter in target_quarters
            ],
            dtype=float,
        )
    else:
        target_values = []
        for index, quarter in enumerate(target_quarters):
            value = alpha + survey_h0[quarter] + activity_target_h0[quarter]
            if index > 0:
                expected_h1 = survey_h1[quarter - 1] + activity_h1[quarter - 1]
                np.testing.assert_allclose(value, alpha + expected_h1, atol=1e-10)
            target_values.append(value)
        if quarterly_values is not None:
            target_values = np.asarray(target_values) + np.asarray(
                [0.6 * quarterly_by_period[quarter] for quarter in target_quarters]
            )
        target_values = np.asarray(target_values)
    if noise:
        target_values += noise * rng.standard_normal(len(target_values))

    dummy_periods: tuple[pd.Timestamp, ...] = ()
    dummy_coefficients = np.array([], dtype=float)
    if include_dummy:
        dummy_periods = (pd.Timestamp("2022-12-31"),)
        dummy_coefficients = np.array([4.0])
        target_values[target_dates == dummy_periods[0]] += dummy_coefficients[0]

    target = pd.DataFrame({"date": target_dates, "value": target_values})
    truth = MultiTruth(
        target=target,
        target_date=pd.Timestamp(target_dates[-1]),
        target_value=float(target_values[-1]),
        alpha=alpha,
        variables=tuple(variables),
        variables_truth=variable_truths,
        dummy_periods=dummy_periods,
        dummy_coefficients=dummy_coefficients,
        common_origin_equation=(
            "target quarter = common origin quarter + horizon; "
            "each variable block uses its latest finite release"
        ),
    )

    vintage_dates = pd.to_datetime(
        ["2023-12-20", "2024-01-15", "2024-02-15", "2024-03-15"]
    )
    vintages = []
    for vintage_date in vintage_dates:
        published = latent.loc[latent["release_date"] <= vintage_date].reset_index(
            drop=True
        )
        regressors = published[["date", "variable", "value"]].copy()
        expected_blocks = {}
        valid = np.ones(len(target), dtype=bool)
        for spec in variables:
            variable_data = regressors.loc[
                regressors["variable"] == spec.variable,
                ["date", "value"],
            ]
            if spec.frequency == "ME":
                block = _monthly_rows(
                    target["date"], variable_data, spec.n_lags, spec.start_lag
                )
            else:
                block = _quarterly_rows(
                    target["date"], variable_data, spec.n_lags, spec.start_lag
                )
            expected_blocks[spec.variable] = block[:-1]
            valid &= np.all(np.isfinite(block), axis=1)

        latest_dates = [
            pd.Timestamp(
                regressors.loc[
                    (regressors["variable"] == spec.variable)
                    & np.isfinite(regressors["value"]),
                    "date",
                ].max()
            )
            for spec in variables
        ]
        common_origin = max(latest_dates)
        horizon = (
            truth.target_date.to_period("Q").ordinal
            - common_origin.to_period("Q").ordinal
        )
        expected_coefficients = {}
        if start_lag:
            horizon_origins = target_dates[:-2] if horizon == 1 else target_dates[:-1]
            target_start = int(
                np.flatnonzero(all_quarters == target_dates[0].to_period("Q"))[0]
            )
            component_start = target_start + horizon
            component_values = {
                "survey": start_lag_signals["survey"][
                    component_start : component_start + len(horizon_origins)
                ],
                "activity": start_lag_signals["activity"][
                    component_start : component_start + len(horizon_origins)
                ],
            }
            for variable in ("survey", "activity"):
                latest = pd.Timestamp(
                    regressors.loc[
                        (regressors["variable"] == variable)
                        & np.isfinite(regressors["value"]),
                        "date",
                    ].max()
                )
                source = {
                    "survey": survey,
                    "activity": activity,
                }[variable]
                stage_source = source.loc[source["date"] <= latest]
                stage_rows = _monthly_rows(
                    pd.Series(horizon_origins),
                    stage_source,
                    n_lags,
                    start_lag,
                )
                expected_coefficients[variable] = np.linalg.lstsq(
                    stage_rows,
                    component_values[variable],
                    rcond=None,
                )[0]
        else:
            expected_coefficients = {
                spec.variable: variable_truths[spec.variable].effective_coefficients[
                    horizon
                ]
                for spec in variables
            }

        forecast_value = alpha
        for spec in variables:
            data = regressors.loc[
                regressors["variable"] == spec.variable, ["date", "value"]
            ]
            latest_variable_date = pd.Timestamp(
                data.loc[np.isfinite(data["value"]), "date"].max()
            )
            if spec.frequency == "ME":
                block = _calendar_monthly_row(
                    latest_variable_date, data, spec.n_lags, spec.start_lag
                )
            else:
                block = _calendar_quarterly_row(
                    latest_variable_date, data, spec.n_lags, spec.start_lag
                )
            coefficients = expected_coefficients[spec.variable]
            forecast_value += float(block @ coefficients)
        forecast_date = (common_origin.to_period("Q") + horizon).end_time.normalize()
        if forecast_date in dummy_periods:
            forecast_value += float(dummy_coefficients[0])
        vintages.append(
            MultiVintage(
                vintage_date=vintage_date,
                published=published,
                regressors=regressors,
                target_train=target.iloc[:-1].copy(),
                common_origin_date=common_origin,
                horizon=horizon,
                expected_blocks=expected_blocks,
                expected_valid_rows=valid[:-1],
                expected_coefficients=expected_coefficients,
                independent_forecast=forecast_value,
            )
        )

    return MultiSample(
        truth=truth,
        observed=latent.copy(),
        latent_regressors=latent.copy(),
        vintages=tuple(vintages),
    )
