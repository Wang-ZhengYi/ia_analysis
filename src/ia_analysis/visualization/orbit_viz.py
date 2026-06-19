#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_viz.py

Orbit animation utilities for orbit_nfw.py results.

Exports MP4 via OpenCV (no external ffmpeg binary required).

Available functions
-------------------
save_orbit_movie6(...)
    Full 2x3 layout:
    - top row: 3 orbit panels
    - bottom row: 3 time-series panels

save_orbit_movie3(...)
    Top row only:
    - 3 orbit panels

All plotting is in user-facing units:
- Length: ckpc/h
- Time:   Gyr
- Speed:  km/s
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import seaborn as sns

try:
    from tqdm import tqdm
except Exception as exc:
    raise RuntimeError("This module requires tqdm. Install with: pip install tqdm") from exc


DEFAULT_CLIST = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"
]


def _set_user_plot_style():
    sns.set(style='ticks')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['mathtext.fontset'] = 'cm'


def _to_array(x):
    return np.asarray(x, dtype=float)


def _normalize_vec2(vx: float, vy: float):
    n = np.hypot(vx, vy)
    if n <= 0.0:
        return 0.0, 0.0
    return vx / n, vy / n


def _sample_frame_indices(t, fps: int, duration: float):
    t = _to_array(t)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("res.t must be a non-empty 1D array.")
    if duration is None or duration <= 0:
        return np.arange(t.size, dtype=int)
    nframes = max(2, int(round(float(fps) * float(duration))))
    idx = np.linspace(0, t.size - 1, nframes)
    return np.unique(np.round(idx).astype(int))


def _tail_segments_with_alpha(x, y, t, i_now, trail_gyr):
    if i_now < 1:
        return np.empty((0, 2, 2)), np.empty((0, 4))
    t_now = t[i_now]
    t_min = t_now - float(trail_gyr)
    idx = np.where((t[:i_now + 1] >= t_min) & (t[:i_now + 1] <= t_now))[0]
    if idx.size < 2:
        return np.empty((0, 2, 2)), np.empty((0, 4))

    xw = x[idx]
    yw = y[idx]
    tw = t[idx]
    segs = np.stack(
        [np.column_stack([xw[:-1], yw[:-1]]),
         np.column_stack([xw[1:],  yw[1:]])],
        axis=1,
    )

    tmid = 0.5 * (tw[:-1] + tw[1:])
    age = np.clip(t_now - tmid, 0.0, float(trail_gyr))
    alpha = 1.0 - age / max(float(trail_gyr), 1e-12)
    alpha = np.clip(alpha, 0.0, 1.0)

    rgba = np.zeros((alpha.size, 4), dtype=float)
    rgba[:, -1] = alpha
    return segs, rgba


def _prep_series_limits(y, pad_frac=0.05):
    y = _to_array(y)
    finite = np.isfinite(y)
    if not np.any(finite):
        return -1.0, 1.0
    ymin = float(np.min(y[finite]))
    ymax = float(np.max(y[finite]))
    if ymin == ymax:
        dy = 1.0 if ymin == 0.0 else abs(ymin) * 0.2
        return ymin - dy, ymax + dy
    dy = (ymax - ymin) * float(pad_frac)
    return ymin - dy, ymax + dy


def _canvas_to_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf[:, :, :3].copy()


def _set_trail_linecollection(lc, segs, rgba, base_color):
    if segs.shape[0] == 0:
        lc.set_segments([])
        lc.set_color(np.empty((0, 4)))
        return
    rgb = np.array(plt.matplotlib.colors.to_rgb(base_color))
    rgba2 = rgba.copy()
    rgba2[:, :3] = rgb[None, :]
    lc.set_segments(segs)
    lc.set_color(rgba2)


def save_orbit_movie6(
    res,
    r200c,
    *,
    fps=30,
    duration=10.0,
    outfile='orbit6.mp4',
    figsize=(12, 8),
    dpi=180,
    trail_gyr=2.0,
    panel_pad=1.3,
    clist=None,
    bitrate=None,
    codec='mp4v',
    show_progress=True,
    close_fig=True,
    top_only=False,
):
    """
    Export an orbit movie to MP4 via OpenCV.

    Parameters
    ----------
    top_only : bool
        If True, export only the top three orbit panels.
        If False, export the full 2x3 layout.
    """
    _set_user_plot_style()
    clist = DEFAULT_CLIST if clist is None else clist

    t = _to_array(res.t)
    pos = np.asarray(res.pos, dtype=float)
    vel = np.asarray(res.v, dtype=float)

    if pos.ndim != 2 or pos.shape[0] != t.size or pos.shape[1] < 2:
        raise ValueError("res.pos must have shape (N, >=2) and match len(res.t).")
    if vel.ndim != 2 or vel.shape[0] != t.size or vel.shape[1] < 2:
        raise ValueError("res.v must have shape (N, >=2) and match len(res.t).")

    x = pos[:, 0]
    y = pos[:, 1]
    vx = vel[:, 0]
    vy = vel[:, 1]

    M = _to_array(res.M) if hasattr(res, 'M') and (getattr(res, 'M') is not None) else None
    rt = _to_array(res.r_t) if hasattr(res, 'r_t') and (getattr(res, 'r_t') is not None) else None
    rsub = _to_array(res.r_sub) if hasattr(res, 'r_sub') and (getattr(res, 'r_sub') is not None) else None

    if hasattr(res, 'Trr'):
        Trr = _to_array(res.Trr)
    elif hasattr(res, 'T_na') and (getattr(res, 'T_na') is not None):
        Trr = _to_array(res.T_na)[:, 0, 0]
    else:
        Trr = None

    has_M = M is not None
    has_rt = rt is not None
    has_rs = rsub is not None
    has_Trr = Trr is not None

    for name, arr in [('M', M), ('r_t', rt), ('r_sub', rsub), ('Trr', Trr)]:
        if arr is not None and arr.size != t.size:
            raise ValueError(f"res.{name} must have the same length as res.t.")

    frame_idx = _sample_frame_indices(t, fps=fps, duration=duration)
    nframes = frame_idx.size

    lim = float(panel_pad) * float(r200c)
    xlim = (-lim, lim)
    ylim = (-lim, lim)
    arrow_scale = float(r200c) / 10.0
    theta = np.linspace(0.0, 2.0 * np.pi, 500)

    if top_only:
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
        ax4 = ax5 = ax6 = None
    else:
        fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
        ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
        ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]

    # ----------------------------
    # Panel 1: global orbit, growing with time
    # ----------------------------
    p1_line, = ax1.plot([], [], color=clist[9], lw=2.0, label='orbit')
    p1_point, = ax1.plot([], [], marker='o', ms=2.5, color=clist[0], lw=0)
    ax1.scatter(0.0, 0.0, s=20, marker='x', color='k', label='centre')
    ax1.plot(r200c * np.cos(theta), r200c * np.sin(theta), ls='--', color=clist[8], label=r'$R_{200c}$')
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_xlabel(r'$x\;[ckpc/h]$')
    ax1.set_ylabel(r'$y\;[ckpc/h]$')
    ax1.set_aspect('equal')

    # ----------------------------
    # Panel 2: global annotated
    # ----------------------------
    ax2.scatter(0.0, 0.0, s=20, marker='x', color='k')
    ax2.plot(r200c * np.cos(theta), r200c * np.sin(theta), ls='--', color=clist[8])
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_xlabel(r'$x\;[ckpc/h]$')
    ax2.set_ylabel(r'$y\;[ckpc/h]$')
    ax2.set_aspect('equal')

    trail2 = LineCollection([], linewidths=2.0)
    ax2.add_collection(trail2)
    p2_point, = ax2.plot([], [], marker='o', ms=2.5, color='k', lw=0)

    rt2 = Circle((0.0, 0.0), radius=1.0, fill=False, ec=clist[1], lw=2.0, ls='-')
    rs2 = Circle((0.0, 0.0), radius=1.0, fill=False, ec=clist[2], lw=2.0, ls='--')
    ax2.add_patch(rt2)
    ax2.add_patch(rs2)
    rt2.set_visible(False)
    rs2.set_visible(False)

    arr2_c = None
    arr2_v = None

    # ----------------------------
    # Panel 3: subhalo-centred local
    # ----------------------------
    ax3.set_xlabel(r'$\Delta x\;[ckpc/h]$')
    ax3.set_ylabel(r'$\Delta y\;[ckpc/h]$')
    ax3.set_aspect('equal')

    trail3 = LineCollection([], linewidths=2.0)
    ax3.add_collection(trail3)
    ax3_point, = ax3.plot([], [], marker='o', ms=2.5, color='k', lw=0)

    rt3 = Circle((0.0, 0.0), radius=1.0, fill=False, ec=clist[1], lw=2.0, ls='-')
    rs3 = Circle((0.0, 0.0), radius=1.0, fill=False, ec=clist[2], lw=2.0, ls='--')
    ax3.add_patch(rt3)
    ax3.add_patch(rs3)
    rt3.set_visible(False)
    rs3.set_visible(False)

    # ----------------------------
    # Bottom panels
    # ----------------------------
    line4 = line5 = line6a = line6b = None
    vline4 = vline5 = vline6 = None

    if not top_only:
        if has_M:
            line4, = ax4.plot([], [], color=clist[0], lw=2.0)
            y4 = M
            ax4.set_ylabel(r'$M\;[10^{10}M_\odot/h]$')
        else:
            ax4.text(0.5, 0.5, 'M(t) unavailable', transform=ax4.transAxes, ha='center', va='center')
            y4 = np.array([0.0, 1.0])
        ax4.set_xlabel(r'$t\;[Gyr]$')
        ax4.set_xlim(float(t[0]), float(t[-1]))
        ax4.set_ylim(*_prep_series_limits(y4))
        vline4 = ax4.axvline(t[0], color='k', lw=1.5, alpha=0.6)

        if has_Trr:
            line5, = ax5.plot([], [], color=clist[3], lw=2.0)
            y5 = Trr
            ax5.set_ylabel(r'$T_{rr}$')
        else:
            ax5.text(0.5, 0.5, 'Trr unavailable', transform=ax5.transAxes, ha='center', va='center')
            y5 = np.array([0.0, 1.0])
        ax5.set_xlabel(r'$t\;[Gyr]$')
        ax5.set_xlim(float(t[0]), float(t[-1]))
        ax5.set_ylim(*_prep_series_limits(y5))
        vline5 = ax5.axvline(t[0], color='k', lw=1.5, alpha=0.6)

        if has_rt:
            line6a, = ax6.plot([], [], color=clist[1], lw=2.0, label=r'$r_t$')
        if has_rs:
            line6b, = ax6.plot([], [], color=clist[2], lw=2.0, ls='--', label=r'$r_{\rm sub}$')
        if (not has_rt) and (not has_rs):
            ax6.text(0.5, 0.5, 'r_t / r_sub unavailable', transform=ax6.transAxes, ha='center', va='center')
            y6 = np.array([0.0, 1.0])
        else:
            y6_parts = []
            if has_rt:
                y6_parts.append(rt[np.isfinite(rt)])
            if has_rs:
                y6_parts.append(rsub[np.isfinite(rsub)])
            y6 = np.concatenate(y6_parts) if len(y6_parts) else np.array([0.0, 1.0])
        ax6.set_xlabel(r'$t\;[Gyr]$')
        ax6.set_ylabel(r'$r\;[ckpc/h]$')
        ax6.set_xlim(float(t[0]), float(t[-1]))
        ax6.set_ylim(*_prep_series_limits(y6))
        if has_rt or has_rs:
            ax6.legend(frameon=False, fontsize=10, loc='upper right')
        vline6 = ax6.axvline(t[0], color='k', lw=1.5, alpha=0.6)

    title = fig.suptitle('', y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV (cv2) is required to write MP4 without ffmpeg. "
            "Install with: pip install opencv-python"
        ) from exc

    outpath = str(Path(outfile).expanduser().resolve())
    if Path(outpath).suffix.lower() != '.mp4':
        outpath = str(Path(outpath).with_suffix('.mp4'))

    rgb0 = _canvas_to_rgb(fig)
    h, w, _ = rgb0.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(outpath, fourcc, float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError(
            f"OpenCV VideoWriter failed to open '{outpath}'. "
            f"Try codec='avc1' or codec='mp4v', or change output path."
        )

    it = range(nframes)
    if show_progress:
        it = tqdm(it, desc="Encoding video", unit="frame")

    for k in it:
        i = int(frame_idx[k])
        xi, yi = float(x[i]), float(y[i])
        vxi, vyi = float(vx[i]), float(vy[i])

        # Panel 1
        p1_line.set_data(x[:i + 1], y[:i + 1])
        p1_point.set_data([xi], [yi])

        # Panel 2
        segs2, rgba2 = _tail_segments_with_alpha(x, y, t, i, trail_gyr)
        _set_trail_linecollection(trail2, segs2, rgba2, clist[9])
        p2_point.set_data([xi], [yi])

        if has_rt and np.isfinite(rt[i]) and rt[i] > 0.0:
            rt2.center = (xi, yi)
            rt2.set_radius(float(rt[i]))
            rt2.set_visible(True)
        else:
            rt2.set_visible(False)

        if has_rs and np.isfinite(rsub[i]) and rsub[i] > 0.0:
            rs2.center = (xi, yi)
            rs2.set_radius(float(rsub[i]))
            rs2.set_visible(True)
        else:
            rs2.set_visible(False)

        if arr2_c is not None:
            arr2_c.remove()
        if arr2_v is not None:
            arr2_v.remove()

        cx, cy = _normalize_vec2(-xi, -yi)
        ux, uy = _normalize_vec2(vxi, vyi)
        arr2_c = ax2.arrow(
            xi, yi, arrow_scale * cx, arrow_scale * cy,
            length_includes_head=True,
            head_width=0.08 * arrow_scale,
            head_length=0.14 * arrow_scale,
            lw=2.0, color=clist[3], zorder=6,
        )
        arr2_v = ax2.arrow(
            xi, yi, arrow_scale * ux, arrow_scale * uy,
            length_includes_head=True,
            head_width=0.08 * arrow_scale,
            head_length=0.14 * arrow_scale,
            lw=2.0, color=clist[0], zorder=6,
        )

        # Panel 3
        segs3, rgba3 = _tail_segments_with_alpha(x - xi, y - yi, t, i, trail_gyr)
        _set_trail_linecollection(trail3, segs3, rgba3, clist[9])
        ax3_point.set_data([0.0], [0.0])

        local_span = max(0.25 * float(r200c), 2.0 * arrow_scale)
        if has_rt and np.isfinite(rt[i]):
            local_span = max(local_span, 1.25 * float(rt[i]))
        if has_rs and np.isfinite(rsub[i]):
            local_span = max(local_span, 1.25 * float(rsub[i]))
        ax3.set_xlim(-local_span, local_span)
        ax3.set_ylim(-local_span, local_span)

        if has_rt and np.isfinite(rt[i]) and rt[i] > 0.0:
            rt3.center = (0.0, 0.0)
            rt3.set_radius(float(rt[i]))
            rt3.set_visible(True)
        else:
            rt3.set_visible(False)

        if has_rs and np.isfinite(rsub[i]) and rsub[i] > 0.0:
            rs3.center = (0.0, 0.0)
            rs3.set_radius(float(rsub[i]))
            rs3.set_visible(True)
        else:
            rs3.set_visible(False)

        # Bottom panels
        ti = float(t[i])
        if line4 is not None:
            line4.set_data(t[:i + 1], M[:i + 1])
        if line5 is not None:
            line5.set_data(t[:i + 1], Trr[:i + 1])
        if line6a is not None:
            line6a.set_data(t[:i + 1], rt[:i + 1])
        if line6b is not None:
            line6b.set_data(t[:i + 1], rsub[:i + 1])

        if vline4 is not None:
            vline4.set_xdata([ti, ti])
        if vline5 is not None:
            vline5.set_xdata([ti, ti])
        if vline6 is not None:
            vline6.set_xdata([ti, ti])

        title.set_text(fr'$t = {ti:.3f}\,\mathrm{{Gyr}}$')

        rgb = _canvas_to_rgb(fig)
        bgr = rgb[:, :, ::-1]
        vw.write(bgr)

    vw.release()

    if close_fig:
        plt.close(fig)

    return outpath


def save_orbit_movie3(res, r200c, **kwargs):
    """
    Alias for the top-row-only animation.

    Equivalent to:
        save_orbit_movie6(..., top_only=True)
    """
    kwargs = dict(kwargs)
    kwargs['top_only'] = True
    return save_orbit_movie6(res, r200c, **kwargs)
