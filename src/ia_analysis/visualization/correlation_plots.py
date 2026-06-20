"""Correlation-function plotting helpers.

Purpose
-------
Correlation notebooks use compact grid layouts for many statistic keys and
mass/abundance selections.  This module provides reusable axis-level and grid
helpers for those correlation products without assuming a particular HDF5
schema.

Provides
--------
- Single correlation series drawing with optional uncertainty.
- Dictionary and DataFrame grid plotting for repeated statistic keys.
- Common logarithmic radius formatting.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ia_analysis.visualization.figure_io import create_figure_grid, iter_axes
from ia_analysis.visualization.profile_plots import draw_series_with_band


def draw_correlation_series(
    ax: Any,
    radius: Sequence[float],
    values: Sequence[float],
    error: Sequence[float] | None = None,
    *,
    color: str = "0.3",
    label: str | None = None,
    linestyle: str = "-",
    marker: str | None = "o",
    logx: bool = True,
) -> Any:
    """Draw one radial correlation-function series."""
    line = draw_series_with_band(
        ax,
        radius,
        values,
        error,
        color=color,
        label=label,
        linestyle=linestyle,
        marker=marker,
    )
    if logx:
        ax.set_xscale("log")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.set_xlabel("r [Mpc/h]")
    return line


def plot_correlation_key(
    results: Mapping[str, Any],
    key: str,
    *,
    radius_key: str = "r",
    error_key: str | None = None,
    ax: Any | None = None,
    title: str | None = None,
    **draw_kwargs: Any,
) -> tuple[Any, Any]:
    """Plot one named correlation result from a mapping."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
    else:
        fig = ax.figure
    radius = results[radius_key]
    values = results[key]
    error = None if error_key is None or error_key not in results else results[error_key]
    draw_correlation_series(ax, radius, values, error, **draw_kwargs)
    ax.set_ylabel(key)
    ax.set_title(str(title or key))
    return fig, ax


def plot_correlation_grid(
    results: Mapping[str, Any],
    keys: Sequence[str],
    *,
    radius_key: str = "r",
    error_suffix: str | None = "_err",
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    title_func: Any | None = None,
    **draw_kwargs: Any,
) -> tuple[Any, Any]:
    """Plot several correlation statistic keys from one result mapping."""
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = create_figure_grid(nrows, ncols, figsize=figsize or (4.2 * ncols, 3.2 * nrows), squeeze=False)
    flat = list(iter_axes(axes))
    for ax, key in zip(flat, keys):
        err_key = f"{key}{error_suffix}" if error_suffix else None
        if key not in results:
            ax.set_axis_off()
            continue
        plot_correlation_key(
            results,
            key,
            radius_key=radius_key,
            error_key=err_key if err_key in results else None,
            ax=ax,
            title=title_func(key) if title_func else key,
            **draw_kwargs,
        )
    for ax in flat[len(keys):]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes


def plot_tidy_correlation_grid(
    frame: Any,
    *,
    radius_col: str = "r",
    value_col: str = "value",
    statistic_col: str = "statistic",
    hue_col: str | None = None,
    error_col: str | None = "error",
    statistic_order: Sequence[Any] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Plot a tidy DataFrame of correlation-function measurements."""
    stats = list(statistic_order) if statistic_order is not None else list(frame[statistic_col].dropna().unique())
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(stats) / ncols))
    fig, axes = create_figure_grid(nrows, ncols, figsize=figsize or (4.2 * ncols, 3.2 * nrows), squeeze=False)
    flat = list(iter_axes(axes))
    for ax, stat in zip(flat, stats):
        panel = frame.loc[frame[statistic_col] == stat]
        groups = [None] if hue_col is None else list(panel[hue_col].dropna().unique())
        for group in groups:
            subset = panel if hue_col is None else panel.loc[panel[hue_col] == group]
            subset = subset.sort_values(radius_col)
            err = None if error_col is None or error_col not in subset else subset[error_col].to_numpy()
            draw_correlation_series(
                ax,
                subset[radius_col].to_numpy(),
                subset[value_col].to_numpy(),
                err,
                label=None if group is None else str(group),
            )
        ax.set_title(str(stat))
        if hue_col is not None:
            ax.legend()
    for ax in flat[len(stats):]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes


__all__ = [
    "draw_correlation_series",
    "plot_correlation_key",
    "plot_correlation_grid",
    "plot_tidy_correlation_grid",
]
