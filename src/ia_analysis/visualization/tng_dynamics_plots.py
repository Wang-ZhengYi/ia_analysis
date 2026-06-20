"""TNG halo-dynamics plotting helpers.

Purpose
-------
TNG dynamics notebooks repeatedly plotted Pi closure tables, residual
histograms, Dimroth-Watson alignment distributions, component fractions, and
binding-energy mass histograms.  This module provides reusable versions that
operate on the tidy tables returned by ``ia_analysis.dynamics.hd_tng``.

Provides
--------
- Pi direct-vs-affine closure panels by shell.
- Residual histograms for closure diagnostics.
- Alignment-angle distribution panels.
- Component fraction and binding-energy distribution plots.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ia_analysis.visualization.plot_styles import component_color, component_label

PI_COMPONENTS = ("01", "02", "12")


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_pi_closure_table(
    closure_df: Any,
    *,
    components: Sequence[str] = PI_COMPONENTS,
    x_col: str = "shell",
    direct_prefix: str = "Pi_direct",
    affine_prefix: str = "Pi_aff",
    axes: Any | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Plot direct and affine Pi estimates for each skew component."""
    plt = _plt()
    if axes is None:
        fig, axes = plt.subplots(1, len(components), figsize=figsize or (4.2 * len(components), 3.3), squeeze=False)
        axes_arr = np.asarray(axes, dtype=object).ravel()
    else:
        axes_arr = np.asarray(axes, dtype=object).ravel()
        fig = axes_arr[0].figure

    for ax, comp in zip(axes_arr, components):
        direct_col = f"{direct_prefix}_{comp}"
        affine_col = f"{affine_prefix}_{comp}"
        if direct_col in closure_df:
            ax.plot(closure_df[x_col], closure_df[direct_col], "o-", color="0.2", label="direct")
        if affine_col in closure_df:
            ax.plot(closure_df[x_col], closure_df[affine_col], "s--", color="#1f77b4", label="affine")
        ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
        ax.set_title(f"Pi {comp}")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Gyr^-1")
        ax.legend()
    fig.tight_layout()
    return fig, axes_arr


def plot_pi_residual_histogram(
    closure_df: Any,
    *,
    components: Sequence[str] = PI_COMPONENTS,
    residual_prefix: str = "rel_residual",
    bins: int = 32,
    ax: Any | None = None,
    percent: bool = True,
) -> tuple[Any, Any]:
    """Plot closure residual distributions for all requested Pi components."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
    else:
        fig = ax.figure
    for comp in components:
        col = f"{residual_prefix}_{comp}_pct" if percent else f"residual_{comp}"
        if col not in closure_df:
            continue
        vals = np.asarray(closure_df[col], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=int(bins), histtype="step", linewidth=1.5, label=comp)
    ax.axvline(0.0, color="0.75", linewidth=0.8)
    ax.set_xlabel("relative residual [%]" if percent else "residual [Gyr^-1]")
    ax.set_ylabel("count")
    ax.legend()
    return fig, ax


def plot_dw_alignment_distribution(
    cos_values: Sequence[float],
    *,
    ax: Any | None = None,
    bins: int = 30,
    density: bool = True,
    title: str | None = None,
    fit: bool = False,
) -> tuple[Any, Any]:
    """Plot an absolute-cosine alignment distribution with optional DW fit."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
    else:
        fig = ax.figure
    vals = np.asarray(cos_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=int(bins), range=(0.0, 1.0), density=bool(density), color="0.75", edgecolor="0.25")
    if fit and vals.size:
        try:
            from ia_analysis.visualization.distribution_fits import DimrothWatson

            model = DimrothWatson(name="dimroth_watson")
            fit_result = model.fit(vals)
            x = np.linspace(0.0, 1.0, 256)
            ax.plot(x, model.pdf(x, fit_result["kappa"]), color="#d62728", linewidth=1.5, label="DW fit")
            ax.legend()
        except Exception:
            pass
    ax.set_xlabel("|cos(theta)|")
    ax.set_ylabel("density" if density else "count")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_component_fraction_panel(
    frame: Any,
    *,
    component_col: str = "component",
    value_col: str = "fraction",
    ax: Any | None = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Plot component fractions as a compact violin/strip panel."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
    else:
        fig = ax.figure
    try:
        import seaborn as sns

        sns.violinplot(data=frame, x=component_col, y=value_col, ax=ax, inner="quartile", color="0.85")
        sns.stripplot(data=frame, x=component_col, y=value_col, ax=ax, color="0.25", size=2.5, alpha=0.7)
    except Exception:
        groups = list(frame[component_col].dropna().unique())
        data = [np.asarray(frame.loc[frame[component_col] == group, value_col], dtype=float) for group in groups]
        ax.boxplot(data, labels=[component_label(group) for group in groups])
    ax.set_xlabel("")
    ax.set_ylabel(value_col)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_binding_energy_distribution(
    distribution_df: Any,
    *,
    component_col: str = "component",
    energy_col: str = "binding_energy_center",
    mass_col: str = "mass",
    ax: Any | None = None,
    logx: bool = True,
    logy: bool = True,
) -> tuple[Any, Any]:
    """Plot component mass histograms over specific binding energy."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
    else:
        fig = ax.figure
    if energy_col not in distribution_df and "binding_energy" in distribution_df:
        energy_col = "binding_energy"
    for component in distribution_df[component_col].dropna().unique():
        subset = distribution_df.loc[distribution_df[component_col] == component].sort_values(energy_col)
        ax.plot(
            subset[energy_col],
            subset[mass_col],
            marker="o",
            linewidth=1.4,
            color=component_color(component),
            label=component_label(component),
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("specific binding energy [(km/s)^2]")
    ax.set_ylabel("mass [Msun]")
    ax.legend()
    return fig, ax


__all__ = [
    "PI_COMPONENTS",
    "plot_pi_closure_table",
    "plot_pi_residual_histogram",
    "plot_dw_alignment_distribution",
    "plot_component_fraction_panel",
    "plot_binding_energy_distribution",
]
