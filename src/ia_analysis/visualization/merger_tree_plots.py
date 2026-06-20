"""Merger-tree and cross-time plotting helpers.

Purpose
-------
Merger-tree notebooks often combine repeated visual tasks: plotting a subhalo
track in an orbital plane, comparing closure or residual quantities across
snapshots, and showing how shell or particle profiles evolve with time.  This
module collects those reusable figure utilities while leaving data loading,
shape measurement, and shell construction to the dedicated science modules.

Provides
--------
- Orbit-plane tracks for one object followed across snapshots.
- Component-wise time-evolution panels for pi-closure style diagnostics.
- Shell-density and particle-profile comparison plots across time.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ia_analysis.visualization.figure_io import create_figure_grid, iter_axes
from ia_analysis.visualization.legends import add_axis_colorbar
from ia_analysis.visualization.plot_styles import component_color, component_label, redshift_scalar_mappable
from ia_analysis.visualization.profile_plots import draw_series_with_band


def _plt() -> Any:
    """Import matplotlib lazily so package import remains lightweight."""
    import matplotlib.pyplot as plt

    return plt


def _series_from_table(table: Any, column: str) -> np.ndarray:
    """Return one numeric column from a pandas-like table or mapping."""
    values = table[column]
    return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)


def _ordered_unique(frame: Any, column: str, explicit: Sequence[Any] | None = None) -> list[Any]:
    """Return stable unique values from a DataFrame-like column."""
    if explicit is not None:
        return list(explicit)
    values = list(frame[column].dropna().unique()) if hasattr(frame[column], "dropna") else list(dict.fromkeys(frame[column]))
    try:
        return sorted(values)
    except Exception:
        return values


def plot_orbit_plane_track(
    x: Sequence[float],
    y: Sequence[float],
    *,
    color_value: Sequence[float] | None = None,
    ax: Any | None = None,
    cmap: str = "viridis",
    label: str | None = None,
    start_marker: bool = True,
    end_marker: bool = True,
    equal_aspect: bool = True,
    x_label: str = "plane x [ckpc/h]",
    y_label: str = "plane y [ckpc/h]",
    colorbar_label: str | None = "snapshot",
    line_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Plot one cross-time track in an orbital or principal-axis plane."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
    else:
        fig = ax.figure

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    good = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[good]
    y_arr = y_arr[good]
    kwargs = {"linewidth": 1.5, "alpha": 0.85, **dict(line_kwargs or {})}
    if x_arr.size == 0:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return fig, ax

    ax.plot(x_arr, y_arr, color=kwargs.pop("color", "0.35"), label=label, **kwargs)
    if color_value is not None:
        c_arr = np.asarray(color_value, dtype=float)[good]
        sc = ax.scatter(x_arr, y_arr, c=c_arr, cmap=cmap, s=26, zorder=4)
        if colorbar_label:
            add_axis_colorbar(fig, ax, sc, label=colorbar_label)
    else:
        ax.scatter(x_arr, y_arr, color="0.25", s=18, zorder=4)
    if start_marker:
        ax.scatter([x_arr[0]], [y_arr[0]], marker="o", s=58, facecolors="none", edgecolors="tab:green", linewidths=1.4, zorder=5)
    if end_marker:
        ax.scatter([x_arr[-1]], [y_arr[-1]], marker="*", s=86, color="tab:red", zorder=5)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if label:
        ax.legend()
    return fig, ax


def plot_orbit_plane_evolution(
    table: Any,
    *,
    x_col: str = "x",
    y_col: str = "y",
    time_col: str = "SnapNum",
    group_col: str | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    **track_kwargs: Any,
) -> tuple[Any, Any]:
    """Plot orbit-plane tracks from a tidy merger-tree table."""
    if group_col is None:
        fig, ax = plot_orbit_plane_track(
            _series_from_table(table, x_col),
            _series_from_table(table, y_col),
            color_value=_series_from_table(table, time_col),
            colorbar_label=time_col,
            **track_kwargs,
        )
        return fig, ax

    groups = _ordered_unique(table, group_col)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = create_figure_grid(nrows, ncols, figsize=figsize or (4.6 * ncols, 4.0 * nrows), squeeze=False)
    flat = list(iter_axes(axes))
    for ax, group in zip(flat, groups):
        panel = table.loc[table[group_col] == group]
        plot_orbit_plane_track(
            panel[x_col].to_numpy(),
            panel[y_col].to_numpy(),
            color_value=panel[time_col].to_numpy(),
            ax=ax,
            label=str(group),
            colorbar_label=time_col,
            **track_kwargs,
        )
        ax.set_title(str(group))
    for ax in flat[len(groups):]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes


def plot_pi_closure_evolution(
    closure_table: Any,
    *,
    x_col: str = "SnapNum",
    component_col: str = "component",
    value_col: str = "closure",
    error_col: str | None = None,
    components: Sequence[str] | None = None,
    ax: Any | None = None,
    invert_xaxis: bool = True,
    ylabel: str = "closure",
) -> tuple[Any, Any]:
    """Plot component-wise pi-closure or residual evolution across snapshots."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
    else:
        fig = ax.figure
    component_values = components if components is not None else _ordered_unique(closure_table, component_col)
    for comp in component_values:
        panel = closure_table.loc[closure_table[component_col] == comp].sort_values(x_col)
        if len(panel) == 0:
            continue
        error = None if error_col is None or error_col not in panel else panel[error_col].to_numpy()
        draw_series_with_band(
            ax,
            panel[x_col].to_numpy(),
            panel[value_col].to_numpy(),
            error,
            color=component_color(comp),
            label=component_label(comp),
            marker="o",
        )
    ax.axhline(0.0, color="0.72", linewidth=0.8, zorder=0)
    if invert_xaxis:
        ax.invert_xaxis()
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax


def plot_shell_density_summary(
    shell_table: Any,
    *,
    radius_col: str = "radius",
    density_col: str = "density",
    shell_col: str | None = "shell",
    time_col: str | None = "SnapNum",
    ax: Any | None = None,
    log_radius: bool = True,
    log_density: bool = True,
    cmap: str = "viridis",
) -> tuple[Any, Any]:
    """Plot shell-density profiles, optionally colored by snapshot or shell."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
    else:
        fig = ax.figure

    color_values = _ordered_unique(shell_table, time_col) if time_col is not None and time_col in shell_table else [None]
    mappable = None
    if time_col is not None and time_col in shell_table:
        z = np.asarray(shell_table[time_col], dtype=float)
        mappable = redshift_scalar_mappable(z, cmap=cmap)
    for value in color_values:
        panel = shell_table if value is None else shell_table.loc[shell_table[time_col] == value]
        if shell_col is None or shell_col not in panel:
            groups = [(value, panel)]
        else:
            groups = [(shell, panel.loc[panel[shell_col] == shell]) for shell in _ordered_unique(panel, shell_col)]
        for shell, subset in groups:
            if len(subset) == 0:
                continue
            subset = subset.sort_values(radius_col)
            if mappable is not None and value is not None:
                color = mappable.to_rgba(float(value))
                label = f"{time_col}={value}, {shell_col}={shell}" if shell_col else f"{time_col}={value}"
            else:
                color = component_color(shell)
                label = None if shell is None else str(shell)
            ax.plot(subset[radius_col].to_numpy(), subset[density_col].to_numpy(), color=color, linewidth=1.4, label=label)
    if log_radius:
        ax.set_xscale("log")
    if log_density:
        ax.set_yscale("log")
    ax.set_xlabel(radius_col)
    ax.set_ylabel(density_col)
    handles, labels = ax.get_legend_handles_labels()
    if handles and len(handles) <= 12:
        ax.legend(fontsize=8)
    if mappable is not None:
        add_axis_colorbar(fig, ax, mappable, label=time_col)
    return fig, ax


def plot_particle_profile_comparison(
    profile_table: Any,
    *,
    radius_col: str = "radius",
    value_col: str = "value",
    component_col: str = "component",
    time_col: str | None = "SnapNum",
    error_col: str | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    log_radius: bool = True,
    log_value: bool = False,
    cmap: str = "viridis",
) -> tuple[Any, Any]:
    """Plot particle-profile curves by component and optional time grouping."""
    components = _ordered_unique(profile_table, component_col)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(components) / ncols))
    fig, axes = create_figure_grid(nrows, ncols, figsize=figsize or (4.2 * ncols, 3.2 * nrows), squeeze=False)
    flat = list(iter_axes(axes))
    mappable = None
    if time_col is not None and time_col in profile_table:
        times = np.asarray(profile_table[time_col], dtype=float)
        mappable = redshift_scalar_mappable(times, cmap=cmap)
    for ax, component in zip(flat, components):
        panel = profile_table.loc[profile_table[component_col] == component]
        groups = [None] if time_col is None or time_col not in panel else _ordered_unique(panel, time_col)
        for group in groups:
            subset = panel if group is None else panel.loc[panel[time_col] == group]
            subset = subset.sort_values(radius_col)
            error = None if error_col is None or error_col not in subset else subset[error_col].to_numpy()
            if group is None:
                color = component_color(component)
            elif mappable is not None:
                color = mappable.to_rgba(float(group))
            else:
                color = None
            draw_series_with_band(
                ax,
                subset[radius_col].to_numpy(),
                subset[value_col].to_numpy(),
                error,
                color=color or "0.35",
                label=None if group is None else str(group),
                marker="o",
            )
        if log_radius:
            ax.set_xscale("log")
        if log_value:
            ax.set_yscale("log")
        ax.set_title(component_label(component))
        ax.set_xlabel(radius_col)
        ax.set_ylabel(value_col)
        if time_col is not None:
            handles, _ = ax.get_legend_handles_labels()
            if handles and len(handles) <= 8:
                ax.legend(fontsize=8)
    if mappable is not None and flat:
        add_axis_colorbar(fig, flat[0], mappable, label=time_col)
    for ax in flat[len(components):]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes


__all__ = [
    "plot_orbit_plane_track",
    "plot_orbit_plane_evolution",
    "plot_pi_closure_evolution",
    "plot_shell_density_summary",
    "plot_particle_profile_comparison",
]
