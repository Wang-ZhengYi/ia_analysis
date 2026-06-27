#!/usr/bin/env python3
"""Parallel full-alignment figure production for arts_IA specs."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from ia_analysis.visualization import arts_IA


DEFAULT_ROOT_DIR = "/cosma8/data/dp203/dc-wang17/MG_global"
DEFAULT_FLAGS = ("GR", "F40", "F45", "F50", "F55", "F60")
DEFAULT_SNAPS = (1, 3, 6, 8, 10, 12, 15, 18, 21)

G_MASET = None
G_FLAGS = None
G_ZMAP = None
G_SNAPS = None


def _parse_csv_or_space(values: list[str] | None, cast=str):
    if not values:
        return None
    out = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            out.append(cast(part))
    return out


def _init_worker(flags, zmap, snaps):
    arts_IA.set_alignment_context(G_MASET, flags, zmap, snap_list=snaps)


def _profile_task(task):
    try:
        spec_name, flag, snap, min_count, xlim, sample_xrange, bins = task
        spec = arts_IA.get_alignment_spec_by_name(spec_name)
        spec = arts_IA._copy_spec_with_overrides(spec, xlim=xlim, sample_xrange=sample_xrange, bins=bins)
        out = arts_IA.get_binned_alignment_profile(spec, flag, snap, min_count=min_count)
        if out is None:
            return spec_name, flag, int(snap), None
        xc, yy, ee, nn = out
        return spec_name, flag, int(snap), (xc.copy(), yy.copy(), ee.copy(), nn.copy())
    finally:
        # Batch workers process thousands of profiles.  Keeping tensor eigensystem
        # caches across tasks duplicates large (N, 3, 3) arrays in every worker
        # and can OOM on full 205-spec runs.
        arts_IA.clear_alignment_caches()


def _select_specs(chapters=None, specs=None):
    if specs:
        names = list(specs)
        return [arts_IA.get_alignment_spec_by_name(name) for name in names]
    if chapters:
        chapter_set = set(chapters)
        return [spec for spec in arts_IA.ALIGNMENT_SPECS if spec.chapter in chapter_set]
    return list(arts_IA.ALIGNMENT_SPECS)


def _finite_x_for_spec(spec, flag, snap):
    MA = G_MASET[flag][arts_IA._snap_key(snap)]
    x = np.asarray(spec.xfunc(MA), dtype=float)
    mask = arts_IA.mask_population(MA, population=spec.population, err_field=spec.err_field, err_max=spec.err_max)
    mask &= np.isfinite(x)
    return x[mask]


def _central_range_bins(spec, xlo, xhi):
    if not np.isfinite(xlo) or not np.isfinite(xhi) or xhi <= xlo:
        return spec.xlim, spec.sample_xrange, spec.bins
    nbins = len(spec.bins) - 1 if not np.isscalar(spec.bins) else int(spec.bins)
    nbins = max(1, nbins)
    if spec.logx and xlo > 0:
        bins = np.logspace(np.log10(xlo), np.log10(xhi), nbins + 1)
    else:
        bins = np.linspace(xlo, xhi, nbins + 1)
    return (float(xlo), float(xhi)), (float(xlo), float(xhi)), bins


def _measure_x_distributions(specs, flags, snaps, outdir):
    rows = []
    spec_ranges = {}
    hist_data = {}
    for spec in tqdm(specs, desc="Measuring x distributions"):
        all_x = []
        for flag in flags:
            for snap in snaps:
                if arts_IA._snap_key(snap) not in G_MASET.get(flag, {}):
                    continue
                try:
                    x = _finite_x_for_spec(spec, flag, snap)
                except Exception:
                    continue
                if x.size == 0:
                    rows.append({
                        "spec": spec.name,
                        "chapter": spec.chapter,
                        "flag": flag,
                        "snap": int(snap),
                        "n": 0,
                        "xmin": np.nan,
                        "p05": np.nan,
                        "p50": np.nan,
                        "p95": np.nan,
                        "xmax": np.nan,
                    })
                    continue
                rows.append({
                    "spec": spec.name,
                    "chapter": spec.chapter,
                    "flag": flag,
                    "snap": int(snap),
                    "n": int(x.size),
                    "xmin": float(np.nanmin(x)),
                    "p05": float(np.nanpercentile(x, 5)),
                    "p50": float(np.nanpercentile(x, 50)),
                    "p95": float(np.nanpercentile(x, 95)),
                    "xmax": float(np.nanmax(x)),
                })
                all_x.append(x)
        if all_x:
            xcat = np.concatenate(all_x)
            xlo = float(np.nanpercentile(xcat, 5))
            xhi = float(np.nanpercentile(xcat, 95))
            xlim, sample_xrange, bins = _central_range_bins(spec, xlo, xhi)
            spec_ranges[spec.name] = {"xlim": xlim, "sample_xrange": sample_xrange, "bins": bins, "p05": xlo, "p95": xhi}
            hmask = (xcat >= xlo) & (xcat <= xhi)
            counts, edges = np.histogram(xcat[hmask], bins=bins)
            hist_data[spec.name] = {"counts": counts, "edges": edges}
        else:
            spec_ranges[spec.name] = {"xlim": spec.xlim, "sample_xrange": spec.sample_xrange, "bins": spec.bins, "p05": np.nan, "p95": np.nan}

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "x_distribution_stats.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("spec,chapter,flag,snap,n,xmin,p05,p50,p95,xmax\n")
        for row in rows:
            fh.write(
                f"{row['spec']},{row['chapter']},{row['flag']},{row['snap']},{row['n']},"
                f"{row['xmin']},{row['p05']},{row['p50']},{row['p95']},{row['xmax']}\n"
            )
    return spec_ranges, hist_data, csv_path


def _caption(spec, figure_kind):
    if figure_kind == "hist":
        return (
            f"Distribution of the horizontal variable for {spec.title}. "
            "The shaded sampling range is the central 90 percent of the combined model-redshift sample."
        )
    if figure_kind == "snapshot":
        return (
            f"Snapshot-grid comparison for {spec.title}. "
            "Each panel compares gravity models at fixed redshift; the vertical axis is the Dimroth-Watson fitted mu parameter in bins sampled over the central 90 percent of the horizontal-variable distribution."
        )
    return (
        f"Redshift-evolution comparison for {spec.title}. "
        "Each model panel shows redshift-colored curves of the Dimroth-Watson fitted mu parameter with a panel-height colorbar on the right."
    )


def _draw_profile(ax, profile, *, label=None, color=None, lw=1.8, alpha=0.95, error_style="shade", error_alpha=0.30):
    if profile is None:
        return None
    xc, yy, ee, _nn = profile
    good = np.isfinite(yy)
    if not np.any(good):
        return None
    line, = ax.plot(xc[good], yy[good], color=color, lw=lw, alpha=alpha, label=label)
    if error_style == "shade":
        egood = good & np.isfinite(ee)
        ax.fill_between(xc[egood], yy[egood] - ee[egood], yy[egood] + ee[egood], color=color, alpha=error_alpha, lw=0)
    elif error_style == "errorbar":
        egood = good & np.isfinite(ee)
        ax.errorbar(xc[egood], yy[egood], yerr=ee[egood], fmt="none", ecolor=color, alpha=0.55, lw=0.8, capsize=1.5)
    return line


def _profile_y_values(profile, include_error=True):
    if profile is None:
        return np.array([], dtype=float)
    _xc, yy, ee, _nn = profile
    vals = [np.asarray(yy, dtype=float)]
    if include_error:
        vals.append(np.asarray(yy, dtype=float) - np.asarray(ee, dtype=float))
        vals.append(np.asarray(yy, dtype=float) + np.asarray(ee, dtype=float))
    out = np.concatenate([v.ravel() for v in vals])
    return out[np.isfinite(out)]


def _adaptive_ylim_from_profiles(profile_list, *, logy=False, floor=None):
    arrays = [_profile_y_values(profile) for profile in profile_list if profile is not None]
    arrays = [arr for arr in arrays if arr.size > 0]
    vals = np.concatenate(arrays) if arrays else np.array([], dtype=float)
    vals = vals[np.isfinite(vals)]
    if logy:
        vals = vals[vals > 0]
    if vals.size == 0:
        return None
    lo = float(np.nanpercentile(vals, 2))
    hi = float(np.nanpercentile(vals, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        pad = 0.05 * abs(hi) if hi != 0 else 0.05
        lo -= pad
        hi += pad
    elif logy:
        lo *= 0.92
        hi *= 1.08
    else:
        pad = 0.08 * (hi - lo)
        lo -= pad
        hi += pad
    if floor is not None and not logy:
        lo = max(floor, lo)
    return lo, hi


def _apply_adaptive_ylim(axes, profile_list, *, logy=False):
    ylim = _adaptive_ylim_from_profiles(profile_list, logy=logy)
    if ylim is None:
        return
    for ax in axes:
        if ax.get_visible():
            ax.set_ylim(*ylim)


def _plot_x_distribution_hist(spec, hist_entry, outdir, pdf, *, save_png=True, dpi=220):
    arts_IA.set_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    counts = hist_entry.get("counts")
    edges = hist_entry.get("edges")
    if counts is None or edges is None or len(counts) == 0:
        ax.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=ax.transAxes)
    else:
        widths = np.diff(edges)
        ax.bar(edges[:-1], counts, width=widths, align="edge", color="0.35", alpha=0.78, edgecolor="k", linewidth=0.55)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, color="k", lw=1.2)
        ax.axvspan(edges[0], edges[-1], color="#4c78a8", alpha=0.08, lw=0)
    if spec.logx:
        ax.set_xscale("log")
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel("Objects per bin")
    ax.set_title(f"Central 90% x-distribution: {spec.name}", fontsize=12, weight="bold")
    ax.grid(alpha=0.18)
    fig.tight_layout(rect=(0.04, 0.08, 0.98, 0.94))
    _save_or_pdf(
        fig,
        outdir / "x_distribution_hists" / f"{spec.name}_x_hist.png",
        pdf,
        save_png=save_png,
        dpi=dpi,
        caption=_caption(spec, "hist"),
    )


def _save_or_pdf(fig, png_path, pdf, *, save_png=True, dpi=220, caption=None):
    if caption:
        fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8, wrap=True)
    if save_png:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_snapshot_grid(spec, profiles, flags, snaps, zmap, outdir, pdf, *, save_png=True, dpi=220, error_style="shade", error_alpha=0.30):
    arts_IA.set_paper_style()
    use_schematic = len(snaps) <= 8
    fig, axes = plt.subplots(3, 3, figsize=(15.8, 11.2), sharey=False)
    axes = axes.ravel()
    if use_schematic:
        arts_IA._draw_schematic_for_spec(axes[0], spec)
        data_axes = axes[1:1 + len(snaps)]
    else:
        data_axes = axes[:len(snaps)]

    for ax, snap in zip(data_axes, snaps):
        active = [flag for flag in flags if (spec.name, flag, int(snap)) in profiles]
        drew_any = False
        for flag in active:
            line = _draw_profile(
                ax,
                profiles.get((spec.name, flag, int(snap))),
                label=arts_IA.flag_label(flag),
                color=arts_IA.flag_color(flag),
                lw=1.9,
                alpha=0.95,
                error_style=error_style,
                error_alpha=error_alpha,
            )
            drew_any = drew_any or line is not None
        z = zmap.get(int(snap), arts_IA.ZMAP_ALL.get(int(snap), np.nan))
        ax.set_title(rf"$z={z:.2f}$  (snap={int(snap):03d})")
        arts_IA.apply_alignment_axis_format(ax, spec)
        if not drew_any:
            ax.text(0.5, 0.5, "No valid DWE fits", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.35")

    snapshot_profiles = [
        profiles.get((spec.name, flag, int(snap)))
        for flag in flags
        for snap in snaps
    ]
    _apply_adaptive_ylim(data_axes, snapshot_profiles, logy=spec.logy)

    first_empty = (1 + len(snaps)) if use_schematic else len(snaps)
    if first_empty < len(axes):
        arts_IA._draw_model_legend_in_axis(axes[first_empty], flags)
        for ax in axes[first_empty + 1:]:
            ax.axis("off")
    else:
        fig.legend(
            handles=arts_IA._model_legend_handles(flags),
            loc="upper center",
            ncol=min(len(flags), 7),
            frameon=False,
            bbox_to_anchor=(0.5, 0.925),
            borderaxespad=0.0,
        )

    fig.suptitle(spec.title, fontsize=15, weight="bold", y=0.982)
    fig.tight_layout(rect=(0.025, 0.055, 0.985, 0.875), w_pad=1.05, h_pad=1.2)
    _save_or_pdf(
        fig,
        outdir / "alignment_snapshot_grids" / f"{spec.name}_snapshot_grid.png",
        pdf,
        save_png=save_png,
        dpi=dpi,
        caption=_caption(spec, "snapshot"),
    )


def _plot_redshift_evolution(spec, profiles, flags, snaps, zmap, outdir, pdf, *, save_png=True, dpi=220, error_style="shade", error_alpha=0.30, cmap_name="turbo"):
    arts_IA.set_paper_style()
    n_model = len(flags)
    n_panel = 1 + n_model
    ncols = 3
    nrows = int(math.ceil(n_panel / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.65 * ncols, 3.45 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    arts_IA._draw_schematic_for_spec(axes[0], spec)

    zvals = np.array([zmap.get(int(s), arts_IA.ZMAP_ALL.get(int(s), np.nan)) for s in snaps], dtype=float)
    finite_z = zvals[np.isfinite(zvals)]
    if len(finite_z) == 0:
        zvals = np.arange(len(snaps), dtype=float)
        finite_z = zvals
    norm = plt.Normalize(vmin=np.nanmin(finite_z), vmax=np.nanmax(finite_z))
    cmap = plt.get_cmap(cmap_name)

    for ax, flag in zip(axes[1:1 + n_model], flags):
        model_profiles = []
        drew_any = False
        for snap, z in zip(snaps, zvals):
            profile = profiles.get((spec.name, flag, int(snap)))
            model_profiles.append(profile)
            line = _draw_profile(
                ax,
                profile,
                label=rf"$z={z:.2f}$",
                color=cmap(norm(z)),
                lw=1.7,
                alpha=0.92,
                error_style=error_style,
                error_alpha=error_alpha,
            )
            drew_any = drew_any or line is not None
        ax.set_title(arts_IA.flag_label(flag), color=arts_IA.flag_color(flag), weight="bold")
        arts_IA.apply_alignment_axis_format(ax, spec)
        _apply_adaptive_ylim([ax], model_profiles, logy=spec.logy)
        if not drew_any:
            ax.text(0.5, 0.5, "No valid DWE fits", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.35")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.0%", pad=0.045)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("z", labelpad=3)
        cbar.ax.tick_params(labelsize=8, direction="in")

    for ax in axes[1 + n_model:]:
        ax.axis("off")

    fig.suptitle(spec.title + " - redshift evolution", fontsize=15, weight="bold", y=0.972)
    fig.tight_layout(rect=(0.025, 0.06, 0.985, 0.925), w_pad=1.0, h_pad=1.1)
    _save_or_pdf(
        fig,
        outdir / "alignment_redshift_evolution" / f"{spec.name}_redshift_evolution.png",
        pdf,
        save_png=save_png,
        dpi=dpi,
        caption=_caption(spec, "redshift"),
    )


def build_full_alignment_figures(
    *,
    root_dir=DEFAULT_ROOT_DIR,
    outdir="pipeline/01_alignment/outputs/full_alignments_parallel",
    requested_flags=DEFAULT_FLAGS,
    snap_list=DEFAULT_SNAPS,
    chapters=None,
    specs=None,
    workers=None,
    mode="both",
    save_png=True,
    summary_pdf=True,
    pdf_name="full_alignment_summary.pdf",
    dpi=220,
    min_count=8,
    error_style="shade",
    error_alpha=0.30,
    cmap_name="turbo",
    strict=False,
):
    global G_MASET, G_FLAGS, G_SNAPS, G_ZMAP

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(root_dir)
    snap_list = [int(s) for s in snap_list]
    requested_flags = [str(f) for f in requested_flags]

    G_MASET, flags, missing = arts_IA.load_alignment_maset(
        root_dir,
        requested_flags=requested_flags,
        snap_list=snap_list,
        verbose=True,
        strict=strict,
    )
    zmap = {snap: arts_IA.ZMAP_ALL[snap] for snap in snap_list if snap in arts_IA.ZMAP_ALL}
    arts_IA.set_alignment_context(G_MASET, flags, zmap, snap_list=snap_list)
    G_FLAGS, G_SNAPS, G_ZMAP = flags, snap_list, zmap

    selected_specs = _select_specs(chapters=chapters, specs=specs)
    spec_ranges, hist_data, stats_csv = _measure_x_distributions(selected_specs, flags, snap_list, outdir)
    selected_specs = [
        arts_IA._copy_spec_with_overrides(
            spec,
            xlim=spec_ranges[spec.name]["xlim"],
            sample_xrange=spec_ranges[spec.name]["sample_xrange"],
            bins=spec_ranges[spec.name]["bins"],
        )
        for spec in selected_specs
    ]
    if workers is None:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = max(1, int(workers))

    tasks = [
        (
            spec.name,
            flag,
            snap,
            min_count,
            spec_ranges[spec.name]["xlim"],
            spec_ranges[spec.name]["sample_xrange"],
            spec_ranges[spec.name]["bins"],
        )
        for spec in selected_specs
        for flag in flags
        for snap in snap_list
        if arts_IA._snap_key(snap) in G_MASET.get(flag, {})
    ]
    profiles: dict[tuple[str, str, int], Any] = {}

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(flags, zmap, snap_list),
    ) as executor:
        futures = {executor.submit(_profile_task, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing alignment profiles"):
            spec_name, flag, snap, profile = future.result()
            profiles[(spec_name, flag, snap)] = profile

    pdf = None
    if summary_pdf:
        pdf_path = outdir / pdf_name
        pdf = PdfPages(pdf_path)
    try:
        for spec in tqdm(selected_specs, desc="Writing figures"):
            if spec.name in hist_data:
                _plot_x_distribution_hist(
                    spec,
                    hist_data[spec.name],
                    outdir,
                    pdf,
                    save_png=save_png,
                    dpi=dpi,
                )
            if mode in {"snapshot", "both"}:
                _plot_snapshot_grid(
                    spec,
                    profiles,
                    flags,
                    snap_list,
                    zmap,
                    outdir,
                    pdf,
                    save_png=save_png,
                    dpi=dpi,
                    error_style=error_style,
                    error_alpha=error_alpha,
                )
            if mode in {"redshift", "both"}:
                _plot_redshift_evolution(
                    spec,
                    profiles,
                    flags,
                    snap_list,
                    zmap,
                    outdir,
                    pdf,
                    save_png=save_png,
                    dpi=dpi,
                    error_style=error_style,
                    error_alpha=error_alpha,
                    cmap_name=cmap_name,
                )
    finally:
        if pdf is not None:
            pdf.close()

    if missing:
        print(f"Missing/failed input catalogues: {len(missing)}")
    print("X-distribution statistics:", stats_csv)
    if summary_pdf:
        print("Summary PDF:", outdir / pdf_name)
    print("Output directory:", outdir)
    return outdir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Parallel full arts_IA alignment plotter.")
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR, help="Directory containing global HDF5 catalogues.")
    parser.add_argument("--outdir", default="pipeline/01_alignment/outputs/full_alignments_parallel")
    parser.add_argument("--flags", nargs="+", default=list(DEFAULT_FLAGS), help="Model flags, space or comma separated.")
    parser.add_argument("--snaps", nargs="+", default=[str(s) for s in DEFAULT_SNAPS], help="Snapshot numbers, space or comma separated.")
    parser.add_argument("--chapters", nargs="+", default=None, choices=["galaxy", "halo", "tidal", "radial"], help="Optional chapter subset.")
    parser.add_argument("--specs", nargs="+", default=None, help="Optional explicit spec-name subset.")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes; default SLURM_CPUS_PER_TASK or os.cpu_count.")
    parser.add_argument("--mode", choices=["snapshot", "redshift", "both"], default="both")
    parser.add_argument("--no-png", action="store_true", help="Do not write individual PNG files.")
    parser.add_argument("--no-pdf", action="store_true", help="Do not write the summary PDF.")
    parser.add_argument("--pdf-name", default="full_alignment_summary.pdf")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--error-style", choices=["shade", "errorbar", "none"], default="shade")
    parser.add_argument("--error-alpha", type=float, default=0.30)
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    flags = _parse_csv_or_space(args.flags, str) or list(DEFAULT_FLAGS)
    snaps = _parse_csv_or_space(args.snaps, int) or list(DEFAULT_SNAPS)
    chapters = _parse_csv_or_space(args.chapters, str)
    specs = _parse_csv_or_space(args.specs, str)
    error_style = "shade" if args.error_style == "none" else args.error_style
    build_full_alignment_figures(
        root_dir=args.root_dir,
        outdir=args.outdir,
        requested_flags=flags,
        snap_list=snaps,
        chapters=chapters,
        specs=specs,
        workers=args.workers,
        mode=args.mode,
        save_png=not args.no_png,
        summary_pdf=not args.no_pdf,
        pdf_name=args.pdf_name,
        dpi=args.dpi,
        min_count=args.min_count,
        error_style=error_style,
        error_alpha=0.0 if args.error_style == "none" else args.error_alpha,
        cmap_name=args.cmap,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
