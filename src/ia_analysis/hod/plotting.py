"""Matplotlib-only plots for HOD, assembly, phase-space, and IA components."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def plot_hod(measurement: Any, *, ax: Any | None = None, label: str | None = None) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
    else:
        fig = ax.figure
    ax.plot(measurement.mass_centers, measurement.mean_cen, "o-", label=f"{label or ''} central".strip())
    ax.plot(measurement.mass_centers, measurement.mean_sat, "s-", label=f"{label or ''} satellite".strip())
    ax.plot(measurement.mass_centers, measurement.mean_tot, "^-", label=f"{label or ''} total".strip())
    ax.set(xscale="log", xlabel="Halo mass", ylabel="Mean occupation")
    ax.legend(frameon=False)
    return fig, ax


def plot_hod_by_sample(measurements: Mapping[str, Any], *, component: str = "mean_tot") -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for label, measurement in measurements.items():
        ax.plot(measurement.mass_centers, getattr(measurement, component), marker="o", label=label)
    ax.set(xscale="log", xlabel="Halo mass", ylabel=component)
    ax.legend(frameon=False)
    return fig, ax


def plot_assembly_hod(result: Mapping[str, Any], **kwargs: Any) -> tuple[Any, Any]:
    return plot_hod_by_sample(result["measurements"], **kwargs)


plot_environment_hod = plot_assembly_hod
plot_concentration_hod = plot_assembly_hod


def plot_phase_space_hod(frame: Any, *, x: str = "radius_low", y: str = "mean_per_halo", hue: str | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    groups = [(None, frame)] if hue is None else frame.groupby(hue)
    for label, panel in groups:
        ax.plot(panel[x], panel[y], marker="o", label=None if label is None else str(label))
    ax.set(xlabel=x, ylabel=y)
    if hue is not None:
        ax.legend(frameon=False)
    return fig, ax


def plot_alignment_component(measurement: Any, *, ax: Any | None = None, model: Any | None = None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
    else:
        fig = ax.figure
    values = np.asarray(measurement.values)
    if measurement.mass_edges is not None and values.ndim == 1:
        x = np.sqrt(measurement.mass_edges[:-1] * measurement.mass_edges[1:])
        ax.errorbar(x, values, yerr=measurement.errors, marker="o", linestyle="none", label="measurement")
        ax.set_xscale("log")
    elif measurement.radius_edges is not None and values.ndim == 1:
        x = 0.5 * (measurement.radius_edges[:-1] + measurement.radius_edges[1:])
        ax.errorbar(x, values, yerr=measurement.errors, marker="o", linestyle="none", label="measurement")
    else:
        ax.plot(values.ravel(), marker="o", label="measurement")
    if model is not None:
        ax.plot(np.asarray(model).ravel(), label="model")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set(ylabel="Alignment A")
    ax.legend(frameon=False)
    return fig, ax


def plot_mass_radius_alignment_grid(measurement: Any):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    image = ax.imshow(np.asarray(measurement.values).T, origin="lower", aspect="auto")
    fig.colorbar(image, ax=ax, label="Alignment A")
    ax.set(xlabel="Mass bin", ylabel="Radius bin")
    return fig, ax


def plot_assembly_dependent_alignment(measurement: Any):
    return plot_mass_radius_alignment_grid(measurement)


def plot_ia_component_comparison(measurements: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for name, measurement in measurements.items():
        ax.plot(np.asarray(measurement.values).ravel(), marker="o", label=name)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.legend(frameon=False)
    return fig, ax


def plot_pairwise_omega_eta(result: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(result["rmid"], result["omega"], "o-", label="omega")
    ax.plot(result["rmid"], result["eta"], "s-", label="eta")
    ax.set(xlabel="Pair separation", ylabel="Alignment", xscale="log")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.legend(frameon=False)
    return fig, ax


__all__ = [
    "plot_hod", "plot_hod_by_sample", "plot_assembly_hod", "plot_environment_hod",
    "plot_concentration_hod", "plot_phase_space_hod", "plot_alignment_component",
    "plot_mass_radius_alignment_grid", "plot_assembly_dependent_alignment",
    "plot_ia_component_comparison", "plot_pairwise_omega_eta",
]
