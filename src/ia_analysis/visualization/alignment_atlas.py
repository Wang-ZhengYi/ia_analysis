"""Reusable alignment-atlas plotting utilities.

Purpose
-------
Exploratory notebooks repeatedly build the same families of IA figures: metric
curves versus one coordinate, redshift-evolution panels, two-dimensional
alignment heatmaps, and property-distribution atlases.  This module provides
DataFrame-oriented plotting utilities for those repeated patterns while the
legacy ``alignment_plots`` facade continues to expose the original paper-figure
implementation.

Provides
--------
- Multi-metric atlas grids from tidy or wide tables.
- Redshift-evolution panels with optional hue grouping.
- Pivot-table heatmaps for alignment diagnostics.
- Distribution atlases for halo, subhalo, and galaxy properties.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ia_analysis.visualization.figure_io import create_figure_grid, iter_axes
from ia_analysis.visualization.legends import add_axis_colorbar
from ia_analysis.visualization.plot_styles import model_color, model_label
from ia_analysis.visualization.profile_plots import draw_series_with_band


def _plt() -> Any:
    """Import matplotlib lazily so the package namespace stays lightweight."""
    import matplotlib.pyplot as plt

    return plt


def _as_list(value: str | Sequence[str]) -> list[str]:
    """Return a list of column names from a string or sequence."""
    if isinstance(value, str):
        return [value]
    return list(value)


def _ordered_unique(frame: Any, column: str, explicit: Sequence[Any] | None = None) -> list[Any]:
    """Return deterministic unique values for a DataFrame-like column."""
    if explicit is not None:
        return list(explicit)
    values = list(frame[column].dropna().unique())
    try:
        return sorted(values)
    except Exception:
        return values


def _error_values(frame: Any, metric: str, yerr_map: Mapping[str, str] | None) -> np.ndarray | None:
    """Return an optional uncertainty column for one metric."""
    if not yerr_map or metric not in yerr_map:
        return None
    err_col = yerr_map[metric]
    if err_col not in frame:
        return None
    return frame[err_col].to_numpy()


def plot_metric_atlas(
    frame: Any,
    *,
    x_col: str,
    metric_cols: str | Sequence[str],
    hue_col: str | None = None,
    hue_order: Sequence[Any] | None = None,
    yerr_map: Mapping[str, str] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharex: bool | str = True,
    sharey: bool | str = False,
    marker: str | None = "o",
    logx: bool = False,
) -> tuple[Any, Any]:
    """Plot several alignment metrics against one x coordinate."""
    metrics = _as_list(metric_cols)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = create_figure_grid(nrows, ncols, figsize=figsize or (4.2 * ncols, 3.2 * nrows), sharex=sharex, sharey=sharey, squeeze=False)
    flat = list(iter_axes(axes))
    hue_values = [None] if hue_col is None else _ordered_unique(frame, hue_col, hue_order)
    for ax, metric in zip(flat, metrics):
        for hue in hue_values:
            subset = frame if hue is None else frame.loc[frame[hue_col] == hue]
            if len(subset) == 0 or metric not in subset:
                continue
            subset = subset.sort_values(x_col)
            draw_series_with_band(
                ax,
                subset[x_col].to_numpy(),
                subset[metric].to_numpy(),
                _error_values(subset, metric, yerr_map),
                color="0.3" if hue is None else model_color(hue),
                label=None if hue is None else model_label(hue),
                marker=marker,
            )
        ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric)
        ax.set_title(metric)
    for ax in flat[len(metrics):]:
        ax.set_axis_off()
    if hue_col is not None:
        handles, labels = [], []
        for ax in flat[: len(metrics)]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(hue_values)), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return fig, axes


def plot_metric_x_atlas(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Compatibility alias for metric-versus-x atlas figures."""
    return plot_metric_atlas(*args, **kwargs)


def plot_redshift_evolution(
    frame: Any,
    *,
    redshift_col: str = "z",
    metric_cols: str | Sequence[str],
    hue_col: str | None = None,
    hue_order: Sequence[Any] | None = None,
    yerr_map: Mapping[str, str] | None = None,
    invert_redshift_axis: bool = True,
    **atlas_kwargs: Any,
) -> tuple[Any, Any]:
    """Plot one or more alignment metrics as functions of redshift."""
    fig, axes = plot_metric_atlas(
        frame,
        x_col=redshift_col,
        metric_cols=metric_cols,
        hue_col=hue_col,
        hue_order=hue_order,
        yerr_map=yerr_map,
        **atlas_kwargs,
    )
    if invert_redshift_axis:
        for ax in iter_axes(axes):
            if ax.has_data():
                ax.invert_xaxis()
    return fig, axes


def plot_alignment_heatmap(
    frame: Any,
    *,
    x_col: str,
    y_col: str,
    value_col: str,
    ax: Any | None = None,
    aggfunc: str = "mean",
    cmap: str = "coolwarm",
    symmetric: bool = True,
    annotate: bool = False,
    colorbar_label: str | None = None,
) -> tuple[Any, Any]:
    """Plot a pivot-table heatmap for one alignment diagnostic."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
    else:
        fig = ax.figure
    matrix = frame.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc=aggfunc)
    values = matrix.to_numpy(dtype=float)
    kwargs: dict[str, Any] = {"origin": "lower", "aspect": "auto", "cmap": cmap}
    if symmetric:
        finite = values[np.isfinite(values)]
        limit = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        kwargs.update(vmin=-limit, vmax=limit)
    image = ax.imshow(values, **kwargs)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels([str(v) for v in matrix.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(v) for v in matrix.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(value_col)
    if annotate:
        for iy in range(matrix.shape[0]):
            for ix in range(matrix.shape[1]):
                val = values[iy, ix]
                if np.isfinite(val):
                    ax.text(ix, iy, f"{val:.2g}", ha="center", va="center", fontsize=8)
    add_axis_colorbar(fig, ax, image, label=colorbar_label or value_col)
    return fig, ax


def plot_alignment_heatmaps(
    frame: Any,
    *,
    x_col: str,
    y_col: str,
    value_cols: str | Sequence[str],
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    **heatmap_kwargs: Any,
) -> tuple[Any, Any]:
    """Plot several alignment heatmaps in one atlas."""
    values = _as_list(value_cols)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(values) / ncols))
    fig, axes = create_figure_grid(nrows, ncols, figsize=figsize or (4.2 * ncols, 3.6 * nrows), squeeze=False)
    flat = list(iter_axes(axes))
    for ax, value_col in zip(flat, values):
        plot_alignment_heatmap(frame, x_col=x_col, y_col=y_col, value_col=value_col, ax=ax, **heatmap_kwargs)
    for ax in flat[len(values):]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes


def plot_property_distribution_atlas(
    frame: Any,
    *,
    value_cols: str | Sequence[str],
    hue_col: str | None = None,
    hue_order: Sequence[Any] | None = None,
    bins: int | Sequence[float] = 32,
    density: bool = True,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    histtype: str = "step",
) -> tuple[Any, Any]:
    """Plot repeated property distributions from a wide analysis table."""
    columns = _as_list(value_cols)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = create_figure_grid(nrows, ncols, figsize=figsize or (4.0 * ncols, 3.0 * nrows), squeeze=False)
    flat = list(iter_axes(axes))
    hues = [None] if hue_col is None else _ordered_unique(frame, hue_col, hue_order)
    for ax, column in zip(flat, columns):
        for hue in hues:
            subset = frame if hue is None else frame.loc[frame[hue_col] == hue]
            values = subset[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            ax.hist(
                values,
                bins=bins,
                density=density,
                histtype=histtype,
                linewidth=1.5,
                color="0.35" if hue is None else model_color(hue),
                label=None if hue is None else model_label(hue),
            )
        ax.set_xlabel(column)
        ax.set_ylabel("density" if density else "count")
        ax.set_title(column)
    for ax in flat[len(columns):]:
        ax.set_axis_off()
    if hue_col is not None:
        handles, labels = [], []
        for ax in flat[: len(columns)]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(hues)), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return fig, axes


__all__ = [
    "plot_metric_atlas",
    "plot_metric_x_atlas",
    "plot_redshift_evolution",
    "plot_alignment_heatmap",
    "plot_alignment_heatmaps",
    "plot_property_distribution_atlas",
]
