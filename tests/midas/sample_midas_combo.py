"""Independent real time data generator for MidasCombo tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ComboLeafTruth:
    """Independent expectations for one combo leaf."""

    variable: str
    latest_regressor_date: pd.Timestamp
    horizon: int
    coefficients: np.ndarray
    forecast: float


@dataclass(frozen=True)
class MidasComboSample:
    """A release-filtered vintage with independent leaf expectations."""

    target: pd.DataFrame
    target_train: pd.DataFrame
    regressors: pd.DataFrame
    vintage_date: pd.Timestamp
    target_date: pd.Timestamp
    alpha: float
    leaves: dict[str, ComboLeafTruth]


@dataclass(frozen=True)
class MixedModelTruth:
    """Known parameters shared by MIDAS, MultiMIDAS, and OLS leaves."""

    alpha: float
    persistence: float
    weights: np.ndarray
    target_date: pd.Timestamp
    target_value: float


@dataclass(frozen=True)
class MixedModelVintage:
    """Published regressors and expected leaf horizons at one vintage."""

    vintage_date: pd.Timestamp
    target_train: pd.DataFrame
    regressors: pd.DataFrame
    horizons: dict[str, int]


@dataclass(frozen=True)
class MixedModelComboSample:
    """A common process represented by three estimator families."""

    truth: MixedModelTruth
    vintages: tuple[MixedModelVintage, ...]


def _lag_row(
    target_date: pd.Timestamp,
    regressors: pd.DataFrame,
    n_lags: int,
) -> np.ndarray:
    finite = regressors.loc[np.isfinite(regressors["value"])]
    latest = pd.Timestamp(finite["date"].max())
    month_within_quarter = ((latest.month - 1) % 3) + 1
    anchor = (
        pd.Timestamp(target_date).to_period("Q").start_time
        + pd.DateOffset(months=month_within_quarter - 1)
    ).to_period("M")
    values = (
        regressors.assign(month=pd.to_datetime(regressors["date"]).dt.to_period("M"))
        .set_index("month")["value"]
        .sort_index()
    )
    available = values.loc[values.index <= anchor]
    return available.iloc[-n_lags:][::-1].to_numpy(dtype=float)


def _monthly_series_for_signal(
    dates: pd.DatetimeIndex,
    signals: dict[pd.Period, float],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct monthly values whose weighted lag row equals each signal."""
    values: list[float] = []
    previous = list(rng.normal(size=len(weights)))
    for quarter in dates.to_period("Q").unique():
        quarter_values: list[float] = []
        for _ in range(3):
            lagged = quarter_values[::-1] + previous
            prior_contribution = float(np.dot(weights[1:], lagged[: len(weights) - 1]))
            quarter_values.append((signals[quarter] - prior_contribution) / weights[0])
        values.extend(quarter_values)
        previous = (quarter_values[::-1] + previous)[: len(weights)]
    return np.asarray(values[: len(dates)])


def sample_mixed_model_combo(seed: int = 2026) -> MixedModelComboSample:
    """Generate exact MIDAS, MultiMIDAS, and OLS representations of one target."""
    alpha = 2.5
    persistence = 1.01
    weights = np.array([0.5, 0.3, 0.2])
    target_dates = pd.date_range("2010-03-31", "2024-03-31", freq="QE")
    monthly_dates = pd.date_range("2008-01-31", "2024-03-31", freq="ME")
    monthly_quarters = monthly_dates.to_period("Q").unique()
    signal_values = persistence ** np.arange(len(monthly_quarters), dtype=float)
    signals = dict(zip(monthly_quarters, signal_values))
    rng = np.random.default_rng(seed)

    delayed_control = {
        quarter: value
        for quarter, value in zip(monthly_quarters, rng.normal(size=len(signals)))
    }
    monthly_signals = {
        "midas": signals,
        "multi_fast": signals,
        "multi_slow": delayed_control,
    }

    monthly_frames = []
    for variable, variable_signals in monthly_signals.items():
        frame = pd.DataFrame(
            {
                "date": monthly_dates,
                "variable": variable,
                "frequency": "ME",
                "value": _monthly_series_for_signal(
                    monthly_dates, variable_signals, weights, rng
                ),
            }
        )
        if variable == "multi_slow":
            frame["release_date"] = frame["date"] + pd.Timedelta(days=10)
        else:
            frame["release_date"] = frame["date"].dt.to_period(
                "M"
            ).dt.start_time + pd.Timedelta(days=14)
        monthly_frames.append(frame)

    quarterly = pd.DataFrame(
        {
            "date": target_dates,
            "variable": "quarterly",
            "frequency": "QE",
            "value": [signals[date.to_period("Q")] for date in target_dates],
        }
    )
    quarterly["release_date"] = quarterly["date"] + pd.Timedelta(days=10)
    observed = pd.concat([*monthly_frames, quarterly], ignore_index=True)

    target = pd.DataFrame(
        {
            "date": target_dates,
            "variable": "y",
            "frequency": "QE",
            "value": [alpha + signals[date.to_period("Q")] for date in target_dates],
        }
    )
    target_train = target.iloc[:-1].copy()
    target_period = pd.Timestamp(target_dates[-1]).to_period("Q")
    vintages = []
    for vintage_date in pd.to_datetime(
        ["2023-12-20", "2024-01-15", "2024-02-15", "2024-03-15"]
    ):
        published = observed.loc[observed["release_date"] <= vintage_date].copy()
        horizons = {}
        for name, variables in {
            "midas": ["midas"],
            "multi": ["multi_fast", "multi_slow"],
            "quarterly": ["quarterly"],
        }.items():
            latest = published.loc[published["variable"].isin(variables), "date"].max()
            horizons[name] = (
                target_period.ordinal - pd.Timestamp(latest).to_period("Q").ordinal
            )
        vintages.append(
            MixedModelVintage(
                vintage_date=pd.Timestamp(vintage_date),
                target_train=target_train.copy(),
                regressors=published.drop(columns="release_date").reset_index(
                    drop=True
                ),
                horizons=horizons,
            )
        )

    return MixedModelComboSample(
        truth=MixedModelTruth(
            alpha=alpha,
            persistence=persistence,
            weights=weights,
            target_date=pd.Timestamp(target_dates[-1]),
            target_value=float(target["value"].iloc[-1]),
        ),
        vintages=tuple(vintages),
    )


def sample_midas_combo(seed: int = 2024) -> MidasComboSample:
    """Generate two leaves with distinct release horizons for one target.

    Both leaves observe the same latent process.  The fast leaf is released
    through March 2024, while the slow leaf is released only through
    December 2023.  The target is generated so its current-quarter signal
    uses the fast leaf's horizon-zero coefficients and its next-quarter
    signal uses the slow leaf's horizon-one coefficients.
    """
    alpha = 2.25
    h0_coefficients = np.array([0.5, 0.3, 0.2])
    h1_coefficients = np.array([0.7, 0.2, 0.1])
    target_dates = pd.date_range("2019-03-31", "2024-03-31", freq="QE")
    monthly_dates = pd.date_range("2018-01-31", "2024-03-31", freq="ME")
    rng = np.random.default_rng(seed)

    monthly_values: list[float] = []
    previous_row = rng.normal(size=3)
    target_values: list[float] = []
    for quarter in monthly_dates.to_period("Q").unique():
        if not monthly_values:
            signal = float(rng.normal())
        else:
            signal = float(previous_row @ h1_coefficients)

        month_one = float(rng.normal())
        month_two = float(rng.normal())
        month_three = (
            signal - h0_coefficients[1] * month_two - h0_coefficients[2] * month_one
        ) / h0_coefficients[0]
        row = np.array([month_three, month_two, month_one])
        monthly_values.extend([month_one, month_two, month_three])
        previous_row = row

        if quarter in target_dates.to_period("Q"):
            target_values.append(alpha + signal)

    latent = pd.DataFrame({"date": monthly_dates, "value": monthly_values})
    target = pd.DataFrame(
        {
            "date": target_dates,
            "variable": "y",
            "frequency": "QE",
            "value": target_values,
        }
    )
    vintage_date = pd.Timestamp("2024-03-20")
    fast = latent.copy()
    fast["variable"] = "fast"
    fast["frequency"] = "ME"
    fast["release_date"] = fast["date"] + pd.Timedelta(days=5)
    fast.loc[fast["date"] == pd.Timestamp("2024-03-31"), "release_date"] = vintage_date
    fast = fast.loc[fast["release_date"] <= vintage_date].copy()

    slow = fast.loc[fast["date"] <= pd.Timestamp("2023-12-31")].copy()
    slow["variable"] = "slow"
    regressors = pd.concat(
        [fast.drop(columns="release_date"), slow.drop(columns="release_date")],
        ignore_index=True,
    )
    target_date = pd.Timestamp(target_dates[-1])
    leaves: dict[str, ComboLeafTruth] = {}
    for variable, coefficients in (
        ("fast", h0_coefficients),
        ("slow", h1_coefficients),
    ):
        leaf_data = regressors.loc[regressors["variable"] == variable]
        latest = pd.Timestamp(leaf_data["date"].max())
        horizon = target_date.to_period("Q").ordinal - latest.to_period("Q").ordinal
        forecast = alpha + float(_lag_row(target_date, leaf_data, 3) @ coefficients)
        leaves[variable] = ComboLeafTruth(
            variable=variable,
            latest_regressor_date=latest,
            horizon=horizon,
            coefficients=coefficients.copy(),
            forecast=forecast,
        )

    return MidasComboSample(
        target=target.copy(),
        target_train=target.iloc[:-1].copy(),
        regressors=regressors.copy(),
        vintage_date=vintage_date,
        target_date=target_date,
        alpha=alpha,
        leaves=leaves,
    )
