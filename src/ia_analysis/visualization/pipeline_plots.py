"""Plots for the layered catalog-to-correlation analysis pipeline."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def plot_catalog_inventory(frame: Any) -> tuple[Any, Any]:
    """Plot catalog row counts and file sizes by model and snapshot."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if frame.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No catalog files", ha="center", va="center")
            axis.set_axis_off()
        return fig, axes
    for label, panel in frame.groupby("label", dropna=False):
        panel = panel.sort_values("snap")
        axes[0].plot(panel["snap"], panel["n_rows_max"], "o-", label=str(label))
        axes[1].plot(panel["snap"], panel["size_bytes"] / 2.0**20, "o-", label=str(label))
    axes[0].set(xlabel="Snapshot", ylabel="Maximum dataset rows", title="Catalog population")
    axes[1].set(xlabel="Snapshot", ylabel="File size [MiB]", title="Catalog product size")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    return fig, axes


def plot_orbit_shape_suite(runs: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[Any, Any]:
    """Plot orbit, stripping, axis-ratio, and alignment histories."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    case_names = list(dict.fromkeys(key[0] for key in runs))
    colors = dict(zip(case_names, plt.cm.tab10(np.linspace(0.0, 0.85, max(len(case_names), 1)))))
    styles = {"conservative": "-", "with_df": "--"}
    for (case_name, mode), product in runs.items():
        orbit = product["orbit"]
        shape = product["shape"]
        style = styles.get(mode, "-")
        color = colors[case_name]
        axes[0, 0].plot(orbit.x, orbit.y, style, color=color, lw=1.2, label=f"{case_name} / {mode}")
        axes[0, 1].plot(orbit.t, orbit.r, style, color=color, lw=1.2)
        axes[0, 2].plot(orbit.t, orbit.M / orbit.M[0], style, color=color, lw=1.2)
        axes[1, 0].plot(orbit.t, shape["q"][:, 0], style, color=color, lw=1.2)
        axes[1, 1].plot(orbit.t, shape["q"][:, 1], style, color=color, lw=1.2)
        axes[1, 2].plot(orbit.t, shape["angle_major_tide_major_deg"], style, color=color, lw=1.2)
    axes[0, 0].set(xlabel="x [ckpc/h]", ylabel="y [ckpc/h]", title="Orbit plane")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 1].set(xlabel="Time [Gyr]", ylabel="r [ckpc/h]", title="Orbital radius")
    axes[0, 2].set(xlabel="Time [Gyr]", ylabel="M/M0", title="Irreversible stripping")
    axes[1, 0].set(xlabel="Time [Gyr]", ylabel="b/a", title="Intermediate-axis ratio")
    axes[1, 1].set(xlabel="Time [Gyr]", ylabel="c/a", title="Minor-axis ratio")
    axes[1, 2].set(
        xlabel="Time [Gyr]",
        ylabel="Angle [deg]",
        title="Major shape--tide misalignment",
        ylim=(0.0, 90.0),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    if runs:
        axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    return fig, axes


def plot_spectrum_ratios(frame: Any) -> tuple[Any, Any]:
    """Plot fractional spectrum differences relative to the selected reference."""
    import matplotlib.pyplot as plt

    panels = list(frame["spectrum"].dropna().unique()) if not frame.empty else []
    count = max(len(panels), 1)
    fig, axes = plt.subplots(1, count, figsize=(5.0 * count, 4.0), squeeze=False)
    for axis, spectrum in zip(axes[0], panels):
        panel = frame.loc[frame["spectrum"] == spectrum]
        for (flag, snap), curve in panel.groupby(["flag", "snap"], dropna=False):
            curve = curve.sort_values("k")
            axis.plot(curve["k"], curve["fractional_difference"], label=f"{flag} s{snap}")
        axis.axhline(0.0, color="0.7", lw=0.8)
        axis.set(xscale="log", xlabel="k [h/Mpc]", ylabel="P/P_ref - 1", title=str(spectrum))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    if not panels:
        axes[0, 0].text(0.5, 0.5, "No spectrum data", ha="center", va="center")
        axes[0, 0].set_axis_off()
    fig.tight_layout()
    return fig, axes


def plot_correlation_quality(frame: Any) -> tuple[Any, Any]:
    """Plot covariance condition and total signal-to-noise diagnostics."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if frame.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No covariance products", ha="center", va="center")
            axis.set_axis_off()
        return fig, axes
    index = np.arange(len(frame))
    axes[0].scatter(index, frame["condition_number_positive"])
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Covariance product", ylabel="Positive condition number", title="Covariance conditioning")
    axes[1].scatter(index, frame["signal_to_noise"])
    axes[1].set(xlabel="Covariance product", ylabel="Total S/N", title="Correlation signal-to-noise")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig, axes


__all__ = [
    "plot_catalog_inventory",
    "plot_orbit_shape_suite",
    "plot_spectrum_ratios",
    "plot_correlation_quality",
]
