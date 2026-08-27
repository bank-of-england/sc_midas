"""Plotting methods for :class:`~sc_midas.midas_combo.MidasCombo`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..combo_weights import _filter_sources
from .style import _BOE_COLOURS, _apply_boe_style, _thin_xticklabels

__all__ = []


def _parse_xlim(
    xlim: tuple,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Convert x-axis limits to values accepted by Matplotlib.

    String values (e.g. ``'2022-12'``) are parsed via ``pd.Timestamp``;
    values that are already date-like are passed through unchanged.
    ``None`` entries are preserved so matplotlib can auto-scale one side.
    """

    def _convert(v):
        if v is None or isinstance(v, pd.Timestamp):
            return v
        try:
            return pd.Timestamp(v)
        except (TypeError, ValueError, OverflowError):
            return v

    return (_convert(xlim[0]), _convert(xlim[1]))


class _ComboPlots:
    """Mixin that adds plot_combo_fit and plot_combo_weights to MidasCombo."""

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _resolve_combo_name(self, combo_name: str | None) -> str:
        """Return *combo_name* or the root combo spec name when None."""
        if combo_name is not None:
            return combo_name
        if self.combo_specs is None:
            raise RuntimeError("No combo specs configured.")
        return self.combo_specs.name

    def _combo_horizons(self) -> list[int]:
        return list(range(self.horizons))

    def _source_dates(self, combo_name: str, horizon: int) -> pd.DatetimeIndex:
        """Return dates aligned with the stored weights for a combination."""
        spec = next(s for s in self._combo_specs_flat if s.name == combo_name)
        source_df = pd.DataFrame(
            {src: self.fitted_[src][horizon] for src in spec.source_names}
        )
        source_df = _filter_sources(source_df, spec.minimum_sample_size)
        return source_df.index

    # ------------------------------------------------------------------ #
    #  Actual vs Fitted                                                   #
    # ------------------------------------------------------------------ #

    def _plot_fit_on_ax(
        self,
        combo_names: list[str],
        indicator_names: list[str],
        horizon: int,
        ax: plt.Axes,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Plot actual vs fitted for multiple combo/MIDAS sources at *horizon* on *ax*."""
        all_names = combo_names + indicator_names

        # Resolve each name to a Series aligned with the target index.
        series: list[tuple[str, pd.Series]] = []
        for name in all_names:
            if name not in self.fitted_ or horizon not in self.fitted_[name]:
                continue
            raw = self.fitted_[name][horizon]
            fitted = (
                raw
                if isinstance(raw, pd.Series)
                else pd.Series(raw, index=self.target_.index)
            )
            series.append((name, fitted))

        if not series:
            return

        # Plot the actual series once against the union of fitted indices.
        union_idx = series[0][1].index
        for _, s in series[1:]:
            union_idx = union_idx.union(s.index)
        valid_target = self.target_.reindex(union_idx).dropna()
        boe_idx = 0
        ax.plot(
            valid_target.index,
            valid_target.values,
            color=_BOE_COLOURS[boe_idx],
            lw=1.5,
            label="Actual",
        )
        boe_idx += 1

        # Draw indicators in grey and share one legend entry.
        indicator_set = set(indicator_names)
        _indicator_legend_added = False
        for name, fitted in series:
            valid = fitted.notna()
            if name in indicator_set:
                ax.plot(
                    fitted.index[valid],
                    fitted.values[valid],
                    color="grey",
                    alpha=0.4,
                    lw=1.0,
                    label="Indicators" if not _indicator_legend_added else "_nolegend_",
                )
                _indicator_legend_added = True
            else:
                ax.plot(
                    fitted.index[valid],
                    fitted.values[valid],
                    color=_BOE_COLOURS[boe_idx % len(_BOE_COLOURS)],
                    lw=1.5,
                    label=name,
                )
                boe_idx += 1

        ax.set_title(f"{', '.join(all_names)} — Actual vs Fitted (h={horizon})")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        _thin_xticklabels(ax)

    def plot_fit(
        self,
        combo_names: str | list[str] | None = None,
        indicator_names: str | list[str] | None = None,
        horizon: int | None = None,
        ax: plt.Axes | None = None,
        ylim: tuple[float, float] | None = None,
        xlim: tuple | None = None,
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
        """Plot actual vs fitted for one or more combo and/or individual indicator sources.

        Parameters
        ----------
        combo_names : str | list[str] | None
            Name(s) of :class:`~sc_midas.specs.ComboSpec` to plot.  When
            both *combo_names* and *indicator_names* are ``None`` (default),
            the root combo spec is used.
        indicator_names : str | list[str] | None
            Name(s) of individual MIDAS / OLS / MultiMIDAS variable
            sources to overlay on the same chart.
        horizon : int | None
            Horizon to plot.  When ``None`` (default) all horizons are
            plotted in a vertically stacked figure.
        ax : plt.Axes | None
            Only used when a single *horizon* is specified.
        ylim : tuple[float, float] | None
            Fix the y-axis bounds as ``(ymin, ymax)`` on every subplot.
            When ``None`` (default) matplotlib auto-scales.
        xlim : tuple | None
            Fix the x-axis bounds as ``(xmin, xmax)`` on every subplot.
            Accepts any values that matplotlib accepts for ``set_xlim``
            (e.g. ``pd.Timestamp`` objects or strings).  When ``None``
            (default) the axis spans the full data range.

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes | np.ndarray

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If a requested combo or indicator name is unavailable.
        """
        if not self.fitted_:
            raise RuntimeError("Model is not fitted.")

        def _apply_limits(ax: plt.Axes) -> None:
            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(_parse_xlim(xlim))

        # Normalise to lists
        if combo_names is None and indicator_names is None:
            resolved_combos = [self._resolve_combo_name(None)]
            resolved_indicators: list[str] = []
        else:
            resolved_combos = (
                [combo_names] if isinstance(combo_names, str) else (combo_names or [])
            )
            resolved_indicators = (
                [indicator_names]
                if isinstance(indicator_names, str)
                else (indicator_names or [])
            )

        for name in resolved_combos + resolved_indicators:
            if name not in self.fitted_:
                raise ValueError(
                    f"'{name}' not found. Available: {list(self.fitted_.keys())}"
                )

        if horizon is not None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 4))
            else:
                fig = ax.get_figure()
            self._plot_fit_on_ax(
                resolved_combos, resolved_indicators, horizon, ax, ylim
            )
            _apply_limits(ax)
            _apply_boe_style(fig, ax)
            return fig, ax

        horizons = self._combo_horizons()
        n = len(horizons)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)
        if n == 1:
            axes = np.array([axes])
        for h, axis in zip(horizons, axes):
            self._plot_fit_on_ax(resolved_combos, resolved_indicators, h, axis, ylim)
            _apply_limits(axis)
        fig.tight_layout()
        _apply_boe_style(fig, axes)
        return fig, axes

    # ------------------------------------------------------------------ #
    #  Combination weights over time                                      #
    # ------------------------------------------------------------------ #

    def _plot_weights_single(
        self,
        combo_name: str,
        horizon: int,
        ax: plt.Axes,
    ) -> None:
        """Plot per-source combination weights over time for *combo_name* at *horizon*."""
        combo_w = self.combo_weights_[combo_name][horizon]  # dict[str, ndarray]
        dates = self._source_dates(combo_name, horizon)
        T = len(dates)

        for i, (src, w_arr) in enumerate(combo_w.items()):
            # Error-weighted arrays can include OOS application rows. Plot
            # only the in-sample portion so the dates remain aligned.
            w_in_sample = w_arr[:T]
            ax.plot(
                dates,
                w_in_sample,
                color=_BOE_COLOURS[i % len(_BOE_COLOURS)],
                lw=1.2,
                label=src,
            )

        ax.set_title(f"{combo_name} — Combination weights (h={horizon})")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.legend()
        _thin_xticklabels(ax)

    def plot_weights(
        self,
        combo_name: str | None = None,
        horizon: int | None = None,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
        """Plot combination weights for each source variable over time.

        Parameters
        ----------
        combo_name : str | None
            Name of the :class:`~sc_midas.specs.ComboSpec` to plot.
            Defaults to the root combo spec.
        horizon : int | None
            Horizon to plot.  When ``None`` (default) all horizons are
            plotted in a vertically stacked figure.
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
        ValueError
            If the requested combination is unavailable.
        """
        if not self.combo_weights_:
            raise RuntimeError("Model is not fitted.")
        combo_name = self._resolve_combo_name(combo_name)
        if combo_name not in self.combo_weights_:
            raise ValueError(
                f"'{combo_name}' not found. "
                f"Available: {list(self.combo_weights_.keys())}"
            )

        if horizon is not None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 4))
            else:
                fig = ax.get_figure()
            self._plot_weights_single(combo_name, horizon, ax)
            _apply_boe_style(fig, ax)
            return fig, ax

        horizons = self._combo_horizons()
        n = len(horizons)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)
        if n == 1:
            axes = np.array([axes])
        for h, axis in zip(horizons, axes):
            self._plot_weights_single(combo_name, h, axis)
        fig.tight_layout()
        _apply_boe_style(fig, axes)
        return fig, axes

    # ------------------------------------------------------------------ #
    #  Forecast                                                           #
    # ------------------------------------------------------------------ #

    def _plot_forecast_on_ax(
        self,
        combo_names: list[str],
        indicator_names: list[str],
        horizon: int,
        ax: plt.Axes,
        ylim: tuple[float, float] | None = None,
        xlim: tuple | None = None,
    ) -> None:
        """Plot in-sample fit + OOS forecast for multiple sources at *horizon* on *ax*."""
        indicator_set = set(indicator_names)
        all_names = combo_names + indicator_names

        # --- in-sample fitted series ---
        fitted_series: list[tuple[str, pd.Series]] = []
        for name in all_names:
            if name not in self.fitted_ or horizon not in self.fitted_[name]:
                continue
            raw = self.fitted_[name][horizon]
            s = (
                raw
                if isinstance(raw, pd.Series)
                else pd.Series(raw, index=self.target_.index)
            )
            fitted_series.append((name, s))

        if not fitted_series:
            return

        # --- actual target ---
        union_idx = fitted_series[0][1].index
        for _, s in fitted_series[1:]:
            union_idx = union_idx.union(s.index)
        valid_target = self.target_.reindex(union_idx).dropna()

        boe_idx = 0
        ax.plot(
            valid_target.index,
            valid_target.values,
            color=_BOE_COLOURS[boe_idx],
            lw=1.5,
            label="Actual",
        )
        boe_idx += 1

        # --- in-sample fits ---
        _indicator_legend_added = False
        combo_boe: dict[str, str] = {}
        for name, fitted in fitted_series:
            valid = fitted.notna()
            if name in indicator_set:
                ax.plot(
                    fitted.index[valid],
                    fitted.values[valid],
                    color="grey",
                    alpha=0.4,
                    lw=1.0,
                    label="Indicators" if not _indicator_legend_added else "_nolegend_",
                )
                _indicator_legend_added = True
            else:
                colour = _BOE_COLOURS[boe_idx % len(_BOE_COLOURS)]
                combo_boe[name] = colour
                ax.plot(
                    fitted.index[valid],
                    fitted.values[valid],
                    color=colour,
                    lw=1.5,
                    label=name,
                )
                boe_idx += 1

        # --- OOS forecasts (combos + indicators) ---
        if self.forecasts_df_ is not None and len(self.forecasts_df_) > 0:
            fc_h = self.forecasts_df_[self.forecasts_df_["horizon"] == horizon]
            # Draw combo forecasts as dashed continuations of their fit lines.
            for name in combo_names:
                fc_rows = fc_h[fc_h["spec"] == name]
                if fc_rows.empty:
                    continue
                colour = combo_boe.get(name, _BOE_COLOURS[boe_idx % len(_BOE_COLOURS)])
                last_fit = fitted_series[
                    next(i for i, (n, _) in enumerate(fitted_series) if n == name)
                ][1].dropna()
                join_dates = [last_fit.index[-1], *list(fc_rows["date"])]
                join_vals = [last_fit.iloc[-1], *list(fc_rows["value"])]
                ax.plot(
                    join_dates,
                    join_vals,
                    color=colour,
                    lw=1.5,
                    ls="--",
                    label=f"{name} (forecast)",
                )
            # Draw indicator forecasts as grey dashed lines with one legend entry.
            _ind_fc_legend_added = False
            for name in indicator_names:
                fc_rows = fc_h[fc_h["spec"] == name]
                if fc_rows.empty:
                    continue
                # Find the last in-sample point when one exists.
                try:
                    idx = next(i for i, (n, _) in enumerate(fitted_series) if n == name)
                    last_fit = fitted_series[idx][1].dropna()
                    join_dates = [last_fit.index[-1], *list(fc_rows["date"])]
                    join_vals = [last_fit.iloc[-1], *list(fc_rows["value"])]
                except StopIteration:
                    join_dates = list(fc_rows["date"])
                    join_vals = list(fc_rows["value"])
                ax.plot(
                    join_dates,
                    join_vals,
                    color="grey",
                    alpha=0.4,
                    lw=1.0,
                    ls="--",
                    label="Indicators (forecast)"
                    if not _ind_fc_legend_added
                    else "_nolegend_",
                )
                _ind_fc_legend_added = True

        ax.axvline(self.target_.index[-1], color="#cccccc", lw=0.8, ls=":")
        ax.set_title(f"{', '.join(all_names)} — Fit & Forecast (h={horizon})")
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(_parse_xlim(xlim))
        ax.legend()
        _thin_xticklabels(ax)

    def plot_forecast(
        self,
        combo_names: str | list[str] | None = None,
        indicator_names: str | list[str] | None = None,
        horizon: int | None = None,
        ax: plt.Axes | None = None,
        ylim: tuple[float, float] | None = None,
        xlim: tuple | None = None,
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
        """Plot in-sample fit and out-of-sample forecast for combo and indicator sources.

        Requires :meth:`forecast` to have been called first.

        Parameters
        ----------
        combo_names : str | list[str] | None
            Name(s) of :class:`~sc_midas.specs.ComboSpec` to plot.  When
            both *combo_names* and *indicator_names* are ``None`` (default),
            the root combo spec is used.  Combos are shown with a solid line
            for the in-sample fit and a dashed line for the OOS forecast.
        indicator_names : str | list[str] | None
            Name(s) of individual MIDAS / OLS / MultiMIDAS variable sources
            to overlay.  Shown in grey at 60% transparency (in-sample only).
        horizon : int | None
            Horizon to plot.  When ``None`` (default) all horizons are
            plotted in a vertically stacked figure.
        ax : plt.Axes | None
            Only used when a single *horizon* is specified.
        ylim : tuple[float, float] | None
            Fix the y-axis bounds as ``(ymin, ymax)`` on every subplot.
            When ``None`` (default) matplotlib auto-scales.
        xlim : tuple | None
            Fix the x-axis bounds as ``(xmin, xmax)`` on every subplot.
            Accepts any values that matplotlib accepts for ``set_xlim``
            (e.g. ``pd.Timestamp`` objects or strings).  When ``None``
            (default) the axis spans the full data range.

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes | np.ndarray

        Raises
        ------
        RuntimeError
            If the model has not been fitted or no forecasts are available.
        ValueError
            If a requested combo or indicator name is unavailable.
        """
        if not self.fitted_:
            raise RuntimeError("Model is not fitted.")
        if self.forecasts_df_ is None or len(self.forecasts_df_) == 0:
            raise RuntimeError("No forecasts found — call forecast() first.")

        # Normalise to lists
        if combo_names is None and indicator_names is None:
            resolved_combos = [self._resolve_combo_name(None)]
            resolved_indicators: list[str] = []
        else:
            resolved_combos = (
                [combo_names] if isinstance(combo_names, str) else (combo_names or [])
            )
            resolved_indicators = (
                [indicator_names]
                if isinstance(indicator_names, str)
                else (indicator_names or [])
            )

        for name in resolved_combos + resolved_indicators:
            if name not in self.fitted_:
                raise ValueError(
                    f"'{name}' not found. Available: {list(self.fitted_.keys())}"
                )

        if horizon is not None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 4))
            else:
                fig = ax.get_figure()
            self._plot_forecast_on_ax(
                resolved_combos, resolved_indicators, horizon, ax, ylim, xlim
            )
            _apply_boe_style(fig, ax)
            return fig, ax

        horizons = self._combo_horizons()
        n = len(horizons)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)
        if n == 1:
            axes = np.array([axes])
        for h, axis in zip(horizons, axes):
            self._plot_forecast_on_ax(
                resolved_combos, resolved_indicators, h, axis, ylim, xlim
            )
        fig.tight_layout()
        _apply_boe_style(fig, axes)
        return fig, axes
