"""Plotting methods for `MIDAS`."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import _ols_fit_se, _ols_weight_se, _residual_std
from .style import _BOE_COLOURS, _apply_boe_style, _thin_xticklabels

__all__ = []


class _MIDASPlots:
    """Mixin that adds plot_fit, plot_weights and plot_forecast to MIDAS."""

    def _plot_fit_single(
        self,
        horizon: int,
        ax: plt.Axes,
    ) -> None:
        """Plot actual vs fitted for one horizon onto an existing Axes."""
        fit = self.fits_[horizon]
        x = fit.dates if fit.dates is not None else np.arange(fit.nobs)
        fv = fit.fitted_values
        y = fit.y

        se = (
            _ols_fit_se(fit, self.estimator)
            if self.estimator == "ols"
            else np.full(fit.nobs, _residual_std(fit, self.estimator))
        )

        ax.plot(x, y, color=_BOE_COLOURS[0], lw=1.2, label="Actual")
        ax.plot(x, fv, color=_BOE_COLOURS[2], lw=1.2, label="Fitted")
        ax.fill_between(
            x,
            fv - 2 * se,
            fv + 2 * se,
            color=_BOE_COLOURS[2],
            alpha=0.15,
            label="±2 SE",
        )
        ax.set_title(f"Actual vs Fitted (h={horizon})")
        ax.legend()
        _thin_xticklabels(ax)

    def plot_fit(
        self,
        horizon: int | None = None,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
        """Plot actual vs fitted values with a ±2 SE band.

        Parameters
        ----------
        horizon : int | None
            Which horizon to plot.  When ``None`` (default) all fitted
            horizons are plotted in a vertically stacked figure.
        ax : plt.Axes | None
            Only used when a single *horizon* is specified.

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes | np.ndarray

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.fits_:
            raise RuntimeError("Model is not fitted.")

        # --- single horizon ---
        if horizon is not None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 4))
            else:
                fig = ax.get_figure()
            self._plot_fit_single(horizon, ax)
            _apply_boe_style(fig, ax)
            return fig, ax

        # --- all horizons, stacked vertically ---
        horizons = sorted(self.fits_.keys())
        n = len(horizons)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)
        if n == 1:
            axes = np.array([axes])
        for h, axis in zip(horizons, axes):
            self._plot_fit_single(h, axis)
        fig.tight_layout()
        _apply_boe_style(fig, axes)
        return fig, axes

    def _plot_weights_single(self, horizon: int, ax: plt.Axes) -> None:
        """Plot lag weights for one horizon onto an existing Axes."""
        fit = self.fits_[horizon]
        lags = np.arange(self.n_lags)
        w = fit.weights
        se = (
            _ols_weight_se(
                fit, self.estimator, self.method, self.n_lags, self.n_pars_weights
            )
            if self.estimator == "ols"
            else None
        )
        ax.bar(
            lags,
            w,
            color=_BOE_COLOURS[0],
            alpha=0.8,
            yerr=2 * se if se is not None else None,
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": _BOE_COLOURS[3]},
            label="Weight ±2 SE" if se is not None else "Weight",
        )
        ax.set_xlabel("Lag")
        ax.set_ylabel("Weight")
        ax.set_title(f"Lag weights  [{self.method}]  (h={horizon})")
        ax.set_xticks(lags)
        ax.legend()

    def plot_weights(
        self,
        horizon: int | None = None,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
        """Plot estimated lag weights with ±2 SE bars (OLS only).

        Parameters
        ----------
        horizon : int | None
            Which horizon to plot.  When ``None`` (default) all fitted
            horizons are plotted in a vertically stacked figure.
        ax : plt.Axes | None
            Only used when a single *horizon* is specified.

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes | np.ndarray

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.fits_:
            raise RuntimeError("Model is not fitted.")

        if horizon is not None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 4))
            else:
                fig = ax.get_figure()
            self._plot_weights_single(horizon, ax)
            _apply_boe_style(fig, ax)
            return fig, ax

        horizons = sorted(self.fits_.keys())
        n = len(horizons)
        fig, axes = plt.subplots(n, 1, figsize=(7, 4 * n), sharex=True)
        if n == 1:
            axes = np.array([axes])
        for h, axis in zip(horizons, axes):
            self._plot_weights_single(h, axis)
        fig.tight_layout()
        _apply_boe_style(fig, axes)
        return fig, axes

    def plot_forecast(
        self,
        horizon: int | None = None,
        start_date: pd.Timestamp | str | None = None,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
        """Plot historical data, in-sample fit and out-of-sample forecast.

        Requires ``forecast()`` to have been called first.

        Parameters
        ----------
        horizon : int | None
            Which horizon to plot.  When ``None`` (default) all fitted
            horizons are plotted in a vertically stacked figure.
        start_date : pd.Timestamp | str | None
            Earliest x value to display (trims the left of the plot).
        ax : plt.Axes | None

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes | np.ndarray

        Raises
        ------
        RuntimeError
            If the model has not been fitted or no forecasts are available.
        """
        if not self.fits_:
            raise RuntimeError("Model is not fitted.")
        if self.forecasts_df_ is None:
            raise RuntimeError("No forecasts available. Call forecast() first.")

        if horizon is None:
            horizons = sorted(self.fits_.keys())
            n = len(horizons)
            fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n), sharex=False)
            if n == 1:
                axes = np.array([axes])
            for h, axis in zip(horizons, axes):
                self._plot_forecast_single(h, start_date, axis)
            fig.tight_layout()
            _apply_boe_style(fig, axes)
            return fig, axes

        if ax is None:
            fig, ax = plt.subplots(figsize=(11, 4))
        else:
            fig = ax.get_figure()
        self._plot_forecast_single(horizon, start_date, ax)
        _apply_boe_style(fig, ax)
        return fig, ax

    def _plot_forecast_single(self, horizon: int, start_date, ax: plt.Axes) -> None:
        """Plot forecast for one horizon onto an existing Axes."""
        fit = self.fits_[horizon]
        fc_row = self.forecasts_df_[self.forecasts_df_["horizon"] == horizon]
        if fc_row.empty:
            raise ValueError(f"No forecast found for horizon {horizon}.")

        fc = fc_row["value"].to_numpy()
        steps = len(fc)
        fc_se = np.full(steps, _residual_std(fit, self.estimator))

        x_hist = fit.dates if fit.dates is not None else np.arange(fit.nobs)
        x_fc = fc_row["date"].to_numpy()

        x_join = np.concatenate([[x_hist[-1]], x_fc])
        y_join = np.concatenate([[fit.fitted_values.iloc[-1]], fc])

        mask = (
            x_hist >= start_date
            if start_date is not None
            else np.ones(fit.nobs, dtype=bool)
        )

        ax.plot(
            x_hist[mask], fit.y[mask], color=_BOE_COLOURS[0], lw=1.2, label="Actual"
        )
        ax.plot(
            x_hist[mask],
            fit.fitted_values[mask],
            color=_BOE_COLOURS[2],
            lw=1.2,
            label="Fitted",
        )
        ax.plot(
            x_join, y_join, color=_BOE_COLOURS[5], lw=1.5, ls="--", label="Forecast"
        )
        ax.fill_between(
            x_fc,
            fc - 2 * fc_se,
            fc + 2 * fc_se,
            color=_BOE_COLOURS[5],
            alpha=0.15,
            label="±2 SE",
        )
        ax.axvline(x_hist[-1], color="#cccccc", lw=0.8, ls=":")
        ax.set_title(f"Data, Fit and Forecast (h={horizon})")
        ax.legend()
        _thin_xticklabels(ax)
