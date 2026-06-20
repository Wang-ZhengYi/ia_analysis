"""Generic profile and HOD plotting utilities.

Purpose
-------
HOD, radial-profile, satellite-distribution, and property-atlas notebooks all
use the same plotting pattern: draw one or more curves with optional bands, then
arrange them by model, redshift, sample, or component.  This module provides
the reusable DataFrame-oriented pieces without depending on notebook globals.

Provides
--------
- Binned profile statistics with optional weights.
- Series drawing with symmetric or asymmetric error bands.
- DataFrame grouping helpers for model-comparison and redshift-evolution grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ia_analysis.visualization.figure_io import create_figure_grid, iter_axes
from ia_analysis.visualization.plot_styles import model_color, model_label


@dataclass(frozen=True)
class ProfileGridSpec:
    """Column names and orders for a profile grid."""

    x: str
    y: str
    row: str | None = None
    col: str | None = None
    hue: str | None = None
    yerr: str | None = None
    row_order: tuple[Any, ...] | None = None
    col_order: tuple[Any, ...] | None = None
    hue_order: tuple[Any, ...] | None = None


def binned_profile(
    x: Sequence[float],
    y: Sequence[float],
    *,
    bins: int | Sequence[float] = 12,
    weights: Sequence[float] | None = None,
    statistic: str = "mean",
) -> dict[str, np.ndarray]:
    """Compute a simple binned profile for plotting."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("`x` and `y` must have matching shapes")
    w_arr = None if weights is None else np.asarray(weights, dtype=float)
    if w_arr is not None and w_arr.shape != x_arr.shape:
        raise ValueError("`weights` must match `x`")

    good = np.isfinite(x_arr) & np.isfinite(y_arr)
    if w_arr is not None:
        good &= np.isfinite(w_arr) & (w_arr > 0.0)
    x_arr = x_arr[good]
    y_arr = y_arr[good]
    w_arr = None if w_arr is None else w_arr[good]
    if x_arr.size == 0:
        return {"x": np.array([]), "y": np.array([]), "error": np.array([]), "count": np.array([], dtype=int)}

    edges = np.linspace(np.nanmin(x_arr), np.nanmax(x_arr), int(bins) + 1) if np.isscalar(bins) else np.asarray(bins, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(centers.size, np.nan, dtype=float)
    errors = np.full(centers.size, np.nan, dtype=float)
    counts = np.zeros(centers.size, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x_arr >= lo) & (x_arr < hi if i < centers.size - 1 else x_arr <= hi)
        counts[i] = int(np.count_nonzero(mask))
        if counts[i] == 0:
            continue
        yy = y_arr[mask]
        ww = None if w_arr is None else w_arr[mask]
        if statistic == "median":
            values[i] = float(np.nanmedian(yy))
        elif ww is not None:
            values[i] = float(np.average(yy, weights=ww))
        else:
            values[i] = float(np.nanmean(yy))
        errors[i] = float(np.nanstd(yy) / np.sqrt(max(counts[i], 1)))
    return {"x": centers, "y": values, "error": errors, "count": counts, "edges": edges}


def draw_series_with_band(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    error: Sequence[float] | tuple[Sequence[float], Sequence[float]] | None = None,
    *,
    color: str = "0.3",
    label: str | None = None,
    linestyle: str = "-",
    marker: str | None = None,
    linewidth: float = 1.6,
    markersize: float = 3.0,
    alpha: float = 0.20,
    zorder: int = 3,
    logy: bool = False,
) -> Any:
    """Draw a line with an optional uncertainty band."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    line = ax.plot(
        x_arr,
        y_arr,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=float(linewidth),
        markersize=float(markersize),
        label=label,
        zorder=zorder,
    )[0]
    if error is not None:
        if isinstance(error, tuple):
            lo = np.asarray(error[0], dtype=float)
            hi = np.asarray(error[1], dtype=float)
        else:
            err = np.asarray(error, dtype=float)
            lo = y_arr - err
            hi = y_arr + err
        if logy:
            lo = np.where(lo > 0.0, lo, np.nan)
        ax.fill_between(x_arr, lo, hi, color=color, alpha=float(alpha), linewidth=0.0, zorder=zorder - 1)
    return line


def draw_dataframe_series(
    ax: Any,
    frame: Any,
    *,
    x: str,
    y: str,
    yerr: str | None = None,
    color: str = "0.3",
    label: str | None = None,
    sort_by_x: bool = True,
    **draw_kwargs: Any,
) -> Any:
    """Draw one series from a pandas-like table."""
    data = frame.sort_values(x) if sort_by_x and hasattr(frame, "sort_values") else frame
    err = None if yerr is None or yerr not in data else data[yerr].to_numpy()
    return draw_series_with_band(ax, data[x].to_numpy(), data[y].to_numpy(), err, color=color, label=label, **draw_kwargs)


def _ordered_values(frame: Any, column: str | None, explicit: Sequence[Any] | None) -> list[Any]:
    """Return deterministic panel or hue values from a DataFrame."""
    if column is None:
        return [None]
    if explicit is not None:
        return list(explicit)
    values = list(frame[column].dropna().unique())
    try:
        return sorted(values)
    except Exception:
        return values


def plot_dataframe_grid(
    frame: Any,
    spec: ProfileGridSpec,
    *,
    figsize: tuple[float, float] | None = None,
    sharex: bool | str = True,
    sharey: bool | str = False,
    color_func: Callable[[Any], str] | None = None,
    label_func: Callable[[Any], str] | None = None,
    title_func: Callable[[Any, Any], str] | None = None,
    draw_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Plot a DataFrame as a row/column/hue curve grid."""
    rows = _ordered_values(frame, spec.row, spec.row_order)
    cols = _ordered_values(frame, spec.col, spec.col_order)
    hues = _ordered_values(frame, spec.hue, spec.hue_order)
    fig, axes = create_figure_grid(
        len(rows),
        len(cols),
        figsize=figsize or (4.2 * len(cols), 3.2 * len(rows)),
        sharex=sharex,
        sharey=sharey,
    )
    axes_arr = np.asarray(axes, dtype=object).reshape(len(rows), len(cols))
    kwargs = dict(draw_kwargs or {})
    for ir, row_value in enumerate(rows):
        for ic, col_value in enumerate(cols):
            ax = axes_arr[ir, ic]
            panel = frame
            if spec.row is not None:
                panel = panel.loc[panel[spec.row] == row_value]
            if spec.col is not None:
                panel = panel.loc[panel[spec.col] == col_value]
            for hue_value in hues:
                subset = panel if spec.hue is None else panel.loc[panel[spec.hue] == hue_value]
                if len(subset) == 0:
                    continue
                color = color_func(hue_value) if color_func is not None else model_color(hue_value)
                label = label_func(hue_value) if label_func is not None else model_label(hue_value)
                data = subset.sort_values(spec.x)
                error = None if spec.yerr is None or spec.yerr not in data else data[spec.yerr].to_numpy()
                draw_series_with_band(
                    ax,
                    data[spec.x].to_numpy(),
                    data[spec.y].to_numpy(),
                    error,
                    color=color,
                    label=label,
                    **kwargs,
                )
            if title_func is not None:
                ax.set_title(title_func(row_value, col_value))
            elif spec.row is not None or spec.col is not None:
                bits = []
                if spec.row is not None:
                    bits.append(f"{spec.row}={row_value}")
                if spec.col is not None:
                    bits.append(f"{spec.col}={col_value}")
                ax.set_title(", ".join(bits))
            ax.set_xlabel(spec.x)
            ax.set_ylabel(spec.y)
    for ax in iter_axes(axes_arr):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
            break
    fig.tight_layout()
    return fig, axes_arr


def plot_model_comparison_grid(frame: Any, *, x: str, y: str, model_col: str, panel_col: str | None = None, **kwargs: Any) -> tuple[Any, Any]:
    """Plot model-comparison curves in optional panels."""
    spec = ProfileGridSpec(x=x, y=y, col=panel_col, hue=model_col, yerr=kwargs.pop("yerr", None))
    return plot_dataframe_grid(frame, spec, **kwargs)


def plot_redshift_evolution_grid(frame: Any, *, x: str, y: str, redshift_col: str, model_col: str | None = None, **kwargs: Any) -> tuple[Any, Any]:
    """Plot redshift-evolution curves, optionally one panel per model."""
    spec = ProfileGridSpec(x=x, y=y, col=model_col, hue=redshift_col, yerr=kwargs.pop("yerr", None))
    return plot_dataframe_grid(frame, spec, **kwargs)


__all__ = [
    "ProfileGridSpec",
    "binned_profile",
    "draw_series_with_band",
    "draw_dataframe_series",
    "plot_dataframe_grid",
    "plot_model_comparison_grid",
    "plot_redshift_evolution_grid",
]
