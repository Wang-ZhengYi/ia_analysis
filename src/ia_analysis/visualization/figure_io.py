"""Common figure saving and layout helpers.

Purpose
-------
Notebook exports repeatedly defined small ``save_figure`` or ``save_fig``
helpers.  This module provides one reusable implementation for all visualization
layers so figures are saved consistently and output directories are created in
one place.

Provides
--------
- Path creation and filename sanitizing.
- Single-format or multi-format figure export.
- Lightweight grid creation and axis-iteration helpers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sanitize_filename(text: Any, replacement: str = "_") -> str:
    """Return a filesystem-friendly filename stem."""
    stem = re.sub(r"[^A-Za-z0-9_.=-]+", replacement, str(text)).strip(replacement)
    return stem or "figure"


def _normalise_formats(formats: str | Sequence[str] | None, stem: str | Path) -> tuple[str, ...]:
    """Resolve output formats from an explicit list or a path suffix."""
    if formats is None:
        suffix = Path(stem).suffix.lstrip(".")
        return (suffix or "png",)
    if isinstance(formats, str):
        return (formats.lstrip("."),)
    return tuple(str(fmt).lstrip(".") for fmt in formats)


def save_figure(
    fig: Any,
    stem: str | Path,
    *,
    root: str | Path = "plots",
    subdir: str | Path | None = None,
    formats: str | Sequence[str] | None = "png",
    dpi: int = 300,
    close: bool = False,
    bbox_inches: str | None = "tight",
    transparent: bool = False,
    **savefig_kwargs: Any,
) -> list[Path]:
    """Save one Matplotlib figure and return all written paths.

    Parameters are intentionally close to ``Figure.savefig`` while adding the
    project conventions used by the notebooks: directory creation, sanitized
    stems, optional multi-format output, and optional close-after-save.
    """
    output_dir = Path(root)
    if subdir is not None:
        output_dir = output_dir / subdir
    ensure_directory(output_dir)

    stem_path = Path(stem)
    safe_stem = sanitize_filename(stem_path.with_suffix("").as_posix())
    written: list[Path] = []
    for fmt in _normalise_formats(formats, stem_path):
        out = output_dir / f"{safe_stem}.{fmt}"
        fig.savefig(out, dpi=int(dpi), bbox_inches=bbox_inches, transparent=transparent, **savefig_kwargs)
        written.append(out)

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return written


def save_fig(*args: Any, **kwargs: Any) -> list[Path]:
    """Compatibility alias for notebook functions named ``save_fig``."""
    return save_figure(*args, **kwargs)


def create_figure_grid(
    nrows: int,
    ncols: int,
    *,
    figsize: tuple[float, float] | None = None,
    sharex: bool | str = False,
    sharey: bool | str = False,
    squeeze: bool = False,
    **subplots_kwargs: Any,
) -> tuple[Any, Any]:
    """Create a Matplotlib figure grid with a stable non-squeezed axis array."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        int(nrows),
        int(ncols),
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        **subplots_kwargs,
    )
    return fig, axes


def iter_axes(axes: Any) -> Iterable[Any]:
    """Yield axes from a scalar, list, or NumPy axis array."""
    try:
        import numpy as np

        yield from np.asarray(axes, dtype=object).ravel()
    except Exception:
        if isinstance(axes, (list, tuple)):
            for ax in axes:
                yield ax
        else:
            yield axes


__all__ = [
    "ensure_directory",
    "sanitize_filename",
    "save_figure",
    "save_fig",
    "create_figure_grid",
    "iter_axes",
]
