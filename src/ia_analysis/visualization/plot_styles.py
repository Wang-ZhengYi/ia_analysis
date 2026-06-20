"""Project-wide plotting style and label helpers.

Purpose
-------
Raw notebooks used many local dictionaries for model colors, component labels,
redshift labels, and Matplotlib style setup.  This module centralizes those
small decisions while keeping plotting libraries imported only when a function
needs them.

Provides
--------
- Matplotlib/seaborn style setup for paper and notebook figures.
- Stable color and label maps for gravity models, particle components, and
  snapshots.
- Redshift colormap construction for evolution panels.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

DEFAULT_MODEL_COLORS = {
    "GR": "#222222",
    "F5": "#1f77b4",
    "F6": "#ff7f0e",
    "F5nonrad": "#2ca02c",
    "F6nonrad": "#d62728",
    "TNG": "#9467bd",
}

DEFAULT_COMPONENT_COLORS = {
    "gas": "#1f77b4",
    "dm": "#444444",
    "dark_matter": "#444444",
    "stars": "#d62728",
    "stellar": "#d62728",
    "bhs": "#9467bd",
    "bh": "#9467bd",
}

DEFAULT_COMPONENT_LABELS = {
    "gas": "Gas",
    "dm": "Dark matter",
    "dark_matter": "Dark matter",
    "stars": "Stars",
    "stellar": "Stars",
    "bhs": "Black holes",
    "bh": "Black holes",
}

DEFAULT_MODEL_LABELS = {
    "GR": "GR",
    "F5": "F5",
    "F6": "F6",
    "F5nonrad": "F5 non-rad",
    "F6nonrad": "F6 non-rad",
    "TNG": "TNG",
}


def set_project_style(
    *,
    context: str = "paper",
    font_scale: float = 1.0,
    use_tex: bool = False,
    rc: Mapping[str, Any] | None = None,
) -> None:
    """Apply a consistent Matplotlib style for IA analysis figures."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(context=context, style="ticks", font_scale=float(font_scale))
    except Exception:
        plt.style.use("default")

    params = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "text.usetex": bool(use_tex),
    }
    if rc:
        params.update(dict(rc))
    plt.rcParams.update(params)


def model_label(flag: Any, labels: Mapping[str, str] | None = None) -> str:
    """Return a display label for a gravity model flag."""
    mapping = DEFAULT_MODEL_LABELS if labels is None else labels
    return str(mapping.get(str(flag), str(flag)))


def model_color(flag: Any, colors: Mapping[str, str] | None = None, fallback: str = "0.35") -> str:
    """Return a display color for a gravity model flag."""
    mapping = DEFAULT_MODEL_COLORS if colors is None else colors
    return str(mapping.get(str(flag), fallback))


def component_label(component: Any, labels: Mapping[str, str] | None = None) -> str:
    """Return a display label for a particle or galaxy component."""
    key = str(component).strip().lower()
    mapping = DEFAULT_COMPONENT_LABELS if labels is None else labels
    return str(mapping.get(key, str(component)))


def component_color(component: Any, colors: Mapping[str, str] | None = None, fallback: str = "0.4") -> str:
    """Return a display color for a particle or galaxy component."""
    key = str(component).strip().lower()
    mapping = DEFAULT_COMPONENT_COLORS if colors is None else colors
    return str(mapping.get(key, fallback))


def snapshot_label(snap: Any, zmap: Mapping[int, float] | None = None, prefix: str = "z") -> str:
    """Return a compact snapshot or redshift label."""
    try:
        snap_i = int(snap)
    except Exception:
        return f"snap {snap}"
    if zmap is not None and snap_i in zmap:
        return f"{prefix}={float(zmap[snap_i]):.2f}"
    return f"snap {snap_i}"


def redshift_scalar_mappable(
    values: Sequence[float],
    *,
    cmap: str = "viridis_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    """Return a Matplotlib ScalarMappable for redshift-colored curves."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    norm = Normalize(
        vmin=float(np.nanmin(finite) if vmin is None else vmin),
        vmax=float(np.nanmax(finite) if vmax is None else vmax),
    )
    return ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))


def value_to_color(value: float, scalar_mappable: Any) -> Any:
    """Map one scalar value through a Matplotlib ScalarMappable."""
    return scalar_mappable.to_rgba(float(value))


__all__ = [
    "DEFAULT_MODEL_COLORS",
    "DEFAULT_COMPONENT_COLORS",
    "DEFAULT_COMPONENT_LABELS",
    "DEFAULT_MODEL_LABELS",
    "set_project_style",
    "model_label",
    "model_color",
    "component_label",
    "component_color",
    "snapshot_label",
    "redshift_scalar_mappable",
    "value_to_color",
]
