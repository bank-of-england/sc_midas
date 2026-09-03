"""Backward-compatibility shims for deprecated public behaviour."""

from __future__ import annotations

import functools
from collections.abc import Callable

import pandas as pd

__all__ = ["legacy_forecast_alias"]


def legacy_forecast_alias(
    forecast_method: Callable[..., pd.DataFrame],
) -> Callable[..., pd.DataFrame]:
    """Add the deprecated ``forecast`` column to a ``forecast()`` result.

    The long-format output of every model's ``forecast()`` names the point
    forecast column ``value``.  Consumers written against the previous schema
    (e.g. ``forecast-realtime <= 0.5.6``) still read a ``forecast`` column, so
    this decorator appends ``forecast`` as an alias of ``value``.  It is slated
    for removal once those consumers have migrated.

    Parameters
    ----------
    forecast_method : Callable[..., pd.DataFrame]
        An unbound ``forecast`` method returning a long-format DataFrame with
        a ``value`` column.

    Returns
    -------
    Callable[..., pd.DataFrame]
        The wrapped method, whose returned DataFrame also carries a
        ``forecast`` column identical to ``value``.
    """

    @functools.wraps(forecast_method)
    def wrapper(self, *args: object, **kwargs: object) -> pd.DataFrame:
        df = forecast_method(self, *args, **kwargs)
        if "value" in df.columns:
            df["forecast"] = df["value"]
        return df

    return wrapper
