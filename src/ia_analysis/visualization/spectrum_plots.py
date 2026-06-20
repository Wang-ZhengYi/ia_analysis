"""Power-spectrum and AIA plotting helpers.

Purpose
-------
Power-spectrum notebooks repeated the same routines for drawing P(k), AIA(k),
redshift-evolution panels, and model ratios relative to GR.  This module keeps
those reusable pieces independent of the measurement code in
``ia_analysis.spectra``.

Provides
--------
- P(k) and AIA curve drawing with optional uncertainty bands.
- Ratio-to-reference helpers.
- DataFrame grid wrappers for spectrum products.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ia_analysis.visualization.profile_plots import ProfileGridSpec, draw_series_with_band, plot_dataframe_grid


def draw_power_spectrum_series(
    ax: Any,
    k: Sequence[float],
    power: Sequence[float],
    sigma: Sequence[float] | None = None,
    *,
    color: str = "0.3",
    label: str | None = None,
    linestyle: str = "-",
    marker: str | None = "o",
    linewidth: float = 1.4,
    markersize: float = 3.0,
    log_axes: bool = True,
) -> Any:
    """Draw one power-spectrum curve with optional error band."""
    line = draw_series_with_band(
        ax,
        k,
        power,
        sigma,
        color=color,
        label=label,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
        markersize=markersize,
        logy=log_axes,
    )
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P(k) [(Mpc/h)^3]")
    return line


def draw_aia_series(
    ax: Any,
    k: Sequence[float],
    aia: Sequence[float],
    sigma: Sequence[float] | None = None,
    *,
    color: str = "0.3",
    label: str | None = None,
    linestyle: str = "-",
    marker: str | None = "o",
    linewidth: float = 1.4,
    markersize: float = 3.0,
) -> Any:
    """Draw one AIA(k) curve with a horizontal zero reference."""
    line = draw_series_with_band(
        ax,
        k,
        aia,
        sigma,
        color=color,
        label=label,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
        markersize=markersize,
    )
    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("AIA")
    return line


def ratio_to_reference(values: Sequence[float], reference: Sequence[float], *, subtract_one: bool = True) -> np.ndarray:
    """Return y/reference or y/reference - 1 with finite-value masking."""
    y = np.asarray(values, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if y.shape != ref.shape:
        raise ValueError("`values` and `reference` must have matching shapes")
    out = np.full(y.shape, np.nan, dtype=float)
    good = np.isfinite(y) & np.isfinite(ref) & (ref != 0.0)
    out[good] = y[good] / ref[good]
    if subtract_one:
        out[good] -= 1.0
    return out


def draw_ratio_series(
    ax: Any,
    k: Sequence[float],
    values: Sequence[float],
    reference: Sequence[float],
    *,
    color: str = "0.3",
    label: str | None = None,
    subtract_one: bool = True,
    **kwargs: Any,
) -> Any:
    """Draw a ratio or fractional-enhancement curve relative to a reference."""
    ratio = ratio_to_reference(values, reference, subtract_one=subtract_one)
    line = draw_series_with_band(ax, k, ratio, None, color=color, label=label, **kwargs)
    ax.axhline(0.0 if subtract_one else 1.0, color="0.7", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("fractional difference" if subtract_one else "ratio")
    return line


def plot_spectrum_grid(
    frame: Any,
    *,
    k_col: str = "k",
    value_col: str = "P",
    model_col: str = "flag",
    panel_col: str | None = "spectrum",
    yerr_col: str | None = "sigma",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Plot spectrum curves from a tidy DataFrame."""
    spec = ProfileGridSpec(x=k_col, y=value_col, col=panel_col, hue=model_col, yerr=yerr_col)
    fig, axes = plot_dataframe_grid(frame, spec, draw_kwargs={"marker": "o", "logy": True}, **kwargs)
    for ax in np.asarray(axes, dtype=object).ravel():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k [h/Mpc]")
    return fig, axes


def plot_aia_grid(
    frame: Any,
    *,
    k_col: str = "k",
    value_col: str = "AIA",
    model_col: str = "flag",
    panel_col: str | None = "method",
    yerr_col: str | None = "sigma",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Plot AIA curves from a tidy DataFrame."""
    spec = ProfileGridSpec(x=k_col, y=value_col, col=panel_col, hue=model_col, yerr=yerr_col)
    fig, axes = plot_dataframe_grid(frame, spec, draw_kwargs={"marker": "o"}, **kwargs)
    for ax in np.asarray(axes, dtype=object).ravel():
        ax.set_xscale("log")
        ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
        ax.set_xlabel("k [h/Mpc]")
    return fig, axes


__all__ = [
    "draw_power_spectrum_series",
    "draw_aia_series",
    "ratio_to_reference",
    "draw_ratio_series",
    "plot_spectrum_grid",
    "plot_aia_grid",
]
