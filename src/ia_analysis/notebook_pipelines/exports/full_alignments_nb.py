"""Exported code from notebooks/raw_20260618/full_alignments.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # Full alignment comparison: gravity models and redshift evolution This notebook is organized by scientific target: 1. **Galaxy**: stellar galaxy--halo/subhalo alignment. 2. **Halo**: dark-matter halo/subhalo alignment with tidal fields. 3. **Tidal**: stellar alignment with the GR, FoF-group, and GR + fifth-force tidal tensors. 4. **Radial**: stellar/subhalo radial alignment and orbital/figure-rotation diagnostics. 5. **Shape-cluster maps**: shape-dependent radial alignment in the $(b/a,c/a)$ pl

# %% code cell 2
# IPython-only: !pwd

# %% code cell 3

from pathlib import Path
import importlib
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# User configuration
# ------------------------------------------------------------
MG_GLOBAL_DIR = Path("/cosma8/data/dp203/dc-wang17/MG_global")
OUTPUT_ROOT = Path("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/plots/full_alignments_publication")

REQUESTED_FLAGS = ["GR", "F40", "F45", "F50", "F55", "F60"]

# Nine-snapshot grid for clean publication-style comparison.
SNAP_LIST = [1, 3, 6, 8, 10, 12, 15, 18, 21]

# Axes are intentionally separated into different figures.
SHAPE_AXES = ["major", "medium", "minor"]

# Uncertainty display in mu plots.
ERROR_STYLE = "shade"      # publication default: shaded band
ERROR_ALPHA = 0.30

# Redshift colors for same-model evolution.
REDSHIFT_CMAP = "turbo"

# Runtime switches. Start with one chapter if you want a quick test.
RUN_GALAXY = True
RUN_HALO   = True
RUN_TIDAL  = True
RUN_RADIAL = True

# Optional, can be a little heavier.
RUN_SHAPE_CLUSTER_ANALYSIS = False
SHAPE_CLUSTER_FLAG = "GR"
SHAPE_CLUSTER_SNAP = 21
SHAPE_CLUSTER_NCLUSTERS = 30

# Plot display and saving.
SHOW_FIGURES = True
SAVE_FIGURES = False
CONTINUE_ON_ERROR = True

# %% code cell 4

# ------------------------------------------------------------
# Import updated plotting library
# ------------------------------------------------------------
# Preferred use on COSMA:
#   cp arts_IA.py /cosma/home/dp203/dc-wang17/IA_analysis/anaIA/arts_IA.py
# Then this cell will import arts_IA normally.
try:
    import arts_IA
except Exception:
    import arts_publication as arts_IA

importlib.reload(arts_IA)
arts_IA.set_plot_output_root(OUTPUT_ROOT)
arts_IA.set_paper_style()

print("Loaded arts_IA module from:", arts_IA.__file__ if hasattr(arts_IA, "__file__") else arts_IA)
print("Output root:", OUTPUT_ROOT)

# %% code cell 5
# ------------------------------------------------------------
# Patch: HOD_data-style per-panel redshift colorbars
# ------------------------------------------------------------
# This overrides arts_IA.plot_alignment_redshift_evolution at runtime.
# The original arts_IA function is still used to draw all curves; after that,
# the single empty-panel colorbar is removed and each model subplot receives
# its own compact colorbar, following the style used in HOD_data.ipynb.

import inspect
import os
from pathlib import Path

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


if not hasattr(arts_IA, "_plot_alignment_redshift_evolution_original"):
    arts_IA._plot_alignment_redshift_evolution_original = arts_IA.plot_alignment_redshift_evolution


def _get_arts_IA_zmap():
    """Return the active snapshot-redshift map from the arts_IA module."""
    for name in ("zmap", "ZMAP", "ZMAP_ALL"):
        zmap_obj = getattr(arts_IA, name, None)
        if isinstance(zmap_obj, dict) and len(zmap_obj) > 0:
            return zmap_obj
    raise RuntimeError("Could not find a redshift map in arts_IA: expected zmap, ZMAP, or ZMAP_ALL.")


def _active_flags(flags_to_use=None):
    """Return the model flags that should correspond to the redshift-evolution panels."""
    if flags_to_use is None:
        flags_to_use = getattr(arts_IA, "flags", [])
    available = list(getattr(arts_IA, "flags", flags_to_use))
    return [f for f in flags_to_use if (not available or f in available)]


def _remove_existing_redshift_colorbar_axes(fig):
    """
    Remove old single-colorbar axes produced by the original redshift figure.

    The previous implementation put one colorbar in an empty panel.  Its inset
    cbar axis usually carries the label 'Redshift'.  We remove that axis before
    adding the HOD_data-style per-panel colorbars.
    """
    for ax in list(fig.axes):
        ylabel = str(ax.get_ylabel())
        title = str(ax.get_title())
        is_redshift_cbar = (
            "Redshift" in ylabel
            or ylabel.strip() in {r"$\mathrm{z}$", r"$z$", "z"}
            or "Redshift" in title
        )
        if is_redshift_cbar:
            try:
                ax.remove()
            except Exception:
                pass


def _add_axis_redshift_colorbar(fig, ax, cmap, norm, *, label=r"$\mathrm{z}$"):
    """
    Add a compact vertical colorbar directly beside one model subplot.

    Using make_axes_locatable keeps the colorbar height tied to the subplot
    height, so it cannot become longer than the panel itself.
    """
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.2%", pad=0.075)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, labelpad=3)
    cb.ax.tick_params(labelsize=8, direction="in")
    return cb


def _postprocess_redshift_colorbars(fig, *, snap_list, flags_to_use, cmap_name=None):
    """Replace the old single colorbar with one HOD_data-style colorbar per model panel."""
    zmap = _get_arts_IA_zmap()
    zvals = np.array([zmap[int(s)] for s in snap_list if int(s) in zmap], dtype=float)
    zvals = zvals[np.isfinite(zvals)]
    if zvals.size == 0:
        return fig

    cmap = plt.get_cmap(cmap_name or globals().get("REDSHIFT_CMAP", "turbo"))
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(zvals)), vmax=float(np.nanmax(zvals)))

    _remove_existing_redshift_colorbar_axes(fig)

    flags_now = _active_flags(flags_to_use)
    n_model = len(flags_now)

    # In the full_alignments redshift-evolution layout, panel 0 is the schematic
    # and panels 1..n_model are the model panels.  Other panels are kept off.
    # We filter out non-subplot cbar axes first, so this remains stable after reruns.
    subplot_axes = [ax for ax in fig.axes if hasattr(ax, "get_subplotspec")]
    model_axes = subplot_axes[1:1 + n_model]

    for ax in model_axes:
        if ax.get_visible():
            _add_axis_redshift_colorbar(fig, ax, cmap, norm)

    # Leave the old colorbar/extra panels blank; they are useful spacing buffers.
    for ax in subplot_axes[1 + n_model:]:
        ax.axis("off")

    # Extra horizontal space prevents the per-panel colorbars from crowding neighbours.
    fig.subplots_adjust(
        left=0.055,
        right=0.965,
        bottom=0.075,
        top=0.910,
        wspace=0.46,
        hspace=0.43,
    )
    return fig


def _spec_name_for_output(spec):
    """Return the output-safe alignment spec name used in the original save path."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return str(spec.get("name", "alignment"))
    return str(spec)


def _save_patched_redshift_figure(fig, spec):
    """Save the patched redshift-evolution figure using the original directory convention."""
    root = Path(getattr(arts_IA, "PLOT_ROOT", globals().get("OUTPUT_ROOT", ".")))
    outdir = root / "alignment_redshift_evolution"
    outdir.mkdir(parents=True, exist_ok=True)
    fout = outdir / f"{_spec_name_for_output(spec)}_redshift_evolution.png"
    fig.savefig(fout, dpi=220, bbox_inches="tight")
    print("Saved:", fout)


def _call_original_redshift_function(original, spec, call_kwargs):
    """Call the original arts_IA function, filtering kwargs when needed."""
    sig = inspect.signature(original)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        filtered = call_kwargs
    else:
        filtered = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
    return original(spec, **filtered)


def plot_alignment_redshift_evolution_hod_colorbars(
    spec,
    snap_list=None,
    flags_to_use=None,
    save=True,
    show=True,
    xlim=None,
    ylim=None,
    sample_xrange=None,
    bins=None,
    output_root=None,
    cmap_name=None,
    **kwargs,
):
    """
    Same-model redshift evolution with one compact redshift colorbar per model panel.

    This is intentionally a thin wrapper around the original arts_IA implementation:
    it keeps all existing selection, binning, error-band and plotting logic unchanged.
    """
    original = arts_IA._plot_alignment_redshift_evolution_original
    if snap_list is None:
        snap_list = list(getattr(arts_IA, "SNAP_LIST", globals().get("SNAP_LIST", [])))

    call_kwargs = dict(
        snap_list=snap_list,
        flags_to_use=flags_to_use,
        save=False,
        show=False,
        xlim=xlim,
        ylim=ylim,
        sample_xrange=sample_xrange,
        bins=bins,
        output_root=output_root,
        cmap_name=cmap_name,
    )
    call_kwargs.update(kwargs)
    call_kwargs["save"] = False
    call_kwargs["show"] = False

    fig = _call_original_redshift_function(original, spec, call_kwargs)
    fig = _postprocess_redshift_colorbars(
        fig,
        snap_list=snap_list,
        flags_to_use=flags_to_use,
        cmap_name=cmap_name,
    )

    if save:
        _save_patched_redshift_figure(fig, spec)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


arts_IA.plot_alignment_redshift_evolution = plot_alignment_redshift_evolution_hod_colorbars
print("Patched arts_IA.plot_alignment_redshift_evolution: per-panel HOD_data-style redshift colorbars are enabled.")

# %% code cell 6

# ------------------------------------------------------------
# Load all requested model/snapshot files
# ------------------------------------------------------------
MAset, flags, missing_files = arts_IA.load_alignment_maset(
    MG_GLOBAL_DIR,
    requested_flags=REQUESTED_FLAGS,
    snap_list=SNAP_LIST,
    verbose=True,
)

arts_IA.set_alignment_context(
    MAset,
    flags,
    {s: arts_IA.ZMAP_ALL[s] for s in SNAP_LIST},
    snap_list=SNAP_LIST,
)

print("Loaded models:", [arts_IA.flag_label(f) for f in flags])
print("Loaded snapshots:", SNAP_LIST)

# %% [markdown] cell 7
# ## Available alignment families The following cell lists the automatic figure families grouped by chapter.

# %% code cell 8

arts_IA.list_alignment_chapters()

# %% [markdown] cell 9
# # 1. Galaxy chapter Galaxy--halo/subhalo alignment. This section is primarily for checking how stellar shape axes align with their host dark matter shape axes, and how this depends on stellar mass, host mass, baryon fraction, and satellite radius.

# %% code cell 10

if RUN_GALAXY:
    failed_galaxy = arts_IA.plot_alignment_chapter(
        "galaxy",
        shape_axes=SHAPE_AXES,
        mode="both",
        snap_list=SNAP_LIST,
        flags_to_use=flags,
        save=SAVE_FIGURES,
        show=SHOW_FIGURES,
        output_root=OUTPUT_ROOT,
        error_style=ERROR_STYLE,
        error_alpha=ERROR_ALPHA,
        cmap_name=REDSHIFT_CMAP,
        continue_on_error=CONTINUE_ON_ERROR,
    )
else:
    failed_galaxy = []
failed_galaxy

# %% [markdown] cell 11
# # 2. Halo chapter Dark-matter halo/subhalo alignment with the external tidal field. These are the most direct diagnostics for how modified gravity changes the orientation of the dark matter structure itself.

# %% code cell 12

if RUN_HALO:
    failed_halo = arts_IA.plot_alignment_chapter(
        "halo",
        shape_axes=SHAPE_AXES,
        mode="both",
        snap_list=SNAP_LIST,
        flags_to_use=flags,
        save=SAVE_FIGURES,
        show=SHOW_FIGURES,
        output_root=OUTPUT_ROOT,
        error_style=ERROR_STYLE,
        error_alpha=ERROR_ALPHA,
        cmap_name=REDSHIFT_CMAP,
        continue_on_error=CONTINUE_ON_ERROR,
    )
else:
    failed_halo = []
failed_halo

# %% [markdown] cell 13
# # 3. Tidal chapter Stellar alignment with GR, total, and MG-only tidal fields. This is the cleanest chapter for comparing the *galaxy-level response* to the large-scale tidal environment across gravity models and redshift.

# %% code cell 14

if RUN_TIDAL:
    failed_tidal = arts_IA.plot_alignment_chapter(
        "tidal",
        shape_axes=SHAPE_AXES,
        mode="both",
        snap_list=SNAP_LIST,
        flags_to_use=flags,
        save=SAVE_FIGURES,
        show=SHOW_FIGURES,
        output_root=OUTPUT_ROOT,
        error_style=ERROR_STYLE,
        error_alpha=ERROR_ALPHA,
        cmap_name=REDSHIFT_CMAP,
        continue_on_error=CONTINUE_ON_ERROR,
    )
else:
    failed_tidal = []
failed_tidal

# %% [markdown] cell 15
# # 4. Radial chapter Satellite radial alignment and orbital/figure-rotation diagnostics. These plots are important for separating a tidal-field alignment effect from a local host-centric radial-alignment effect.

# %% code cell 16

if RUN_RADIAL:
    failed_radial = arts_IA.plot_alignment_chapter(
        "radial",
        shape_axes=SHAPE_AXES,
        mode="both",
        snap_list=SNAP_LIST,
        flags_to_use=flags,
        save=SAVE_FIGURES,
        show=SHOW_FIGURES,
        output_root=OUTPUT_ROOT,
        error_style=ERROR_STYLE,
        error_alpha=ERROR_ALPHA,
        cmap_name=REDSHIFT_CMAP,
        continue_on_error=CONTINUE_ON_ERROR,
    )
else:
    failed_radial = []
failed_radial

# %% [markdown] cell 17
# # 5. Shape-cluster radial-alignment maps This reproduces and organizes the shape-cluster analysis in the $(b/a,c/a)$ plane. It makes two groups of figures: - **Subhalo / DM** - **Galaxy / stars** Each group contains three subplots, one for each shape axis. The colormap is kept as `bwr`, with the lower triangle, triaxiality curves, and text annotations retained.

# %% code cell 18

if RUN_SHAPE_CLUSTER_ANALYSIS:
    MA = MAset[SHAPE_CLUSTER_FLAG][f"{int(SHAPE_CLUSTER_SNAP):03d}"]
    sat_mask = arts_IA.mask_population(MA, population="satellite")
    shape_cluster_results = arts_IA.plot_shape_cluster_alignment_suite(
        MA,
        mask=sat_mask,
        n_clusters=SHAPE_CLUSTER_NCLUSTERS,
        cmap="bwr",
        output_root=OUTPUT_ROOT,
        prefix=f"{arts_IA.flag_label(SHAPE_CLUSTER_FLAG)}_z{arts_IA.ZMAP_ALL[SHAPE_CLUSTER_SNAP]:.2f}_satellite",
        show=SHOW_FIGURES,
        save=SAVE_FIGURES,
    )
else:
    shape_cluster_results = None
shape_cluster_results

# %% [markdown] cell 19
# # 6. Optional physical sanity checks These plots are not the main science figures, but they help verify that differences between gravity models are not dominated by pathological sample changes.

# %% code cell 20

# Uncomment selected lines if needed.

# arts_IA.plot_physical("Mstar_distribution", snap_list=SNAP_LIST, flags_to_use=flags,
#                    save=SAVE_FIGURES, show=SHOW_FIGURES, output_root=OUTPUT_ROOT)

# arts_IA.plot_physical("M200c_distribution", snap_list=SNAP_LIST, flags_to_use=flags,
#                    save=SAVE_FIGURES, show=SHOW_FIGURES, output_root=OUTPUT_ROOT)

# arts_IA.plot_physical("chi_star_distribution", snap_list=SNAP_LIST, flags_to_use=flags,
#                    save=SAVE_FIGURES, show=SHOW_FIGURES, output_root=OUTPUT_ROOT)

# arts_IA.plot_physical("chi_dm_distribution", snap_list=SNAP_LIST, flags_to_use=flags,
#                    save=SAVE_FIGURES, show=SHOW_FIGURES, output_root=OUTPUT_ROOT)

# %% [markdown] cell 21
# # 7. Failure summary

# %% code cell 22

failure_summary = {
    "galaxy": failed_galaxy,
    "halo": failed_halo,
    "tidal": failed_tidal,
    "radial": failed_radial,
}
failure_summary
