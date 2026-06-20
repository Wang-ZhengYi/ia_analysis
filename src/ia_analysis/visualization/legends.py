"""Legend and colorbar helpers shared by plotting modules.

Purpose
-------
Many notebooks recreated the same model legend, component legend, and one-axis
colorbar helpers.  This module keeps those small figure decorations reusable and
independent of any specific data product.

Provides
--------
- Line-handle construction for custom legends.
- Model, component, and two-row figure legends.
- Axis-attached scalar colorbars.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ia_analysis.visualization.plot_styles import component_color, component_label, model_color, model_label


def make_line_handles(
    labels: Sequence[Any],
    *,
    colors: Sequence[str] | Mapping[Any, str] | None = None,
    linestyles: Sequence[str] | Mapping[Any, str] | None = None,
    markers: Sequence[str | None] | Mapping[Any, str | None] | None = None,
    linewidth: float = 2.0,
) -> list[Any]:
    """Build Matplotlib line handles without drawing data."""
    from matplotlib.lines import Line2D

    handles = []
    for i, label in enumerate(labels):
        if isinstance(colors, Mapping):
            color = colors.get(label, "0.3")
        elif colors is not None:
            color = colors[i]
        else:
            color = "0.3"

        if isinstance(linestyles, Mapping):
            linestyle = linestyles.get(label, "-")
        elif linestyles is not None:
            linestyle = linestyles[i]
        else:
            linestyle = "-"

        if isinstance(markers, Mapping):
            marker = markers.get(label, None)
        elif markers is not None:
            marker = markers[i]
        else:
            marker = None

        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=float(linewidth),
                label=str(label),
            )
        )
    return handles


def add_axis_colorbar(
    fig: Any,
    ax: Any,
    scalar_mappable: Any,
    *,
    label: str | None = None,
    size: str = "3%",
    pad: float = 0.04,
    location: str = "right",
    **colorbar_kwargs: Any,
) -> Any:
    """Add a compact colorbar next to one axis."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    cbar = fig.colorbar(scalar_mappable, cax=cax, **colorbar_kwargs)
    if label:
        cbar.set_label(label)
    return cbar


def add_model_legend(
    fig: Any,
    flags: Sequence[Any],
    *,
    loc: str = "upper center",
    bbox_to_anchor: tuple[float, float] = (0.5, 1.02),
    ncol: int | None = None,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    **legend_kwargs: Any,
) -> Any:
    """Add a figure-level gravity-model legend."""
    display = [model_label(flag, labels) for flag in flags]
    handle_colors = [model_color(flag, colors) for flag in flags]
    handles = make_line_handles(display, colors=handle_colors)
    return fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=int(ncol or max(1, len(flags))),
        **legend_kwargs,
    )


def add_component_legend(
    fig: Any,
    components: Sequence[Any],
    *,
    loc: str = "upper center",
    bbox_to_anchor: tuple[float, float] = (0.5, 0.98),
    ncol: int | None = None,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    linestyles: Sequence[str] | Mapping[Any, str] | None = None,
    **legend_kwargs: Any,
) -> Any:
    """Add a figure-level component legend."""
    display = [component_label(component, labels) for component in components]
    handle_colors = [component_color(component, colors) for component in components]
    handles = make_line_handles(display, colors=handle_colors, linestyles=linestyles)
    return fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=int(ncol or max(1, len(components))),
        **legend_kwargs,
    )


def add_two_row_legend(
    fig: Any,
    *,
    model_flags: Sequence[Any],
    components: Sequence[Any],
    y_model: float = 1.03,
    y_component: float = 0.99,
    model_kwargs: Mapping[str, Any] | None = None,
    component_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Add stacked model and component legends to a figure."""
    leg1 = add_model_legend(
        fig,
        model_flags,
        bbox_to_anchor=(0.5, float(y_model)),
        **dict(model_kwargs or {}),
    )
    leg2 = add_component_legend(
        fig,
        components,
        bbox_to_anchor=(0.5, float(y_component)),
        **dict(component_kwargs or {}),
    )
    return leg1, leg2


__all__ = [
    "make_line_handles",
    "add_axis_colorbar",
    "add_model_legend",
    "add_component_legend",
    "add_two_row_legend",
]
