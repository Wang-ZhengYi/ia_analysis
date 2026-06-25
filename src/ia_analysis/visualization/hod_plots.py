"""Visualization helpers for halo-occupation measurements."""

from __future__ import annotations

from typing import Any, Iterable


def plot_hod_components(
    frame: Any,
    *,
    components: Iterable[str] = ("Ngal", "Ncen", "Nsat"),
    hue: str | None = None,
    error_column: str | None = "occupation_sem",
) -> tuple[Any, Any]:
    """Plot mean occupation curves for total, central, and satellite samples."""
    import matplotlib.pyplot as plt

    components = tuple(components)
    fig, axes = plt.subplots(1, len(components), figsize=(5.0 * len(components), 4.0), squeeze=False)
    for axis, component in zip(axes[0], components):
        panel = frame.loc[frame["component"] == component]
        groups = [(None, panel)] if hue is None else panel.groupby(hue, dropna=False)
        for label, curve in groups:
            curve = curve.sort_values("mass_center")
            error = None if error_column is None or error_column not in curve else curve[error_column]
            axis.errorbar(
                curve["mass_center"],
                curve["mean_occupation"],
                yerr=error,
                marker="o",
                label=None if label is None else str(label),
            )
        axis.set(xscale="log", xlabel="Halo mass", ylabel=f"<{component}>", title=component)
        axis.grid(alpha=0.2)
        if hue is not None:
            axis.legend(frameon=False)
    fig.tight_layout()
    return fig, axes


__all__ = ["plot_hod_components"]
