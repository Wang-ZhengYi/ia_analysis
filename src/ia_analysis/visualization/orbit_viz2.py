#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_viz.py

Utilities for visualizing orbit_nfw.py results and exporting animations.

Main features
-------------
Given an orbit result object `res` (with fields like res.t, res.pos, res.v, res.r,
optionally res.r_t and res.r_sub), this module builds a two-panel animation:

Left panel
----------
- The subhalo point moves along the orbit
- The trajectory "slides down" from the start to the current time
- Only the last `trail_gyr` of trajectory is shown
- Older parts inside that window fade toward zero alpha

Right panel
-----------
- The subhalo centre is marked with an "x"
- Circles for tidal radius r_t and subhalo boundary r_sub (if available)
- Arrow from subhalo centre toward group centre
- Arrow in the velocity direction
- Both arrows have fixed length = r200c / 10

Designed for notebook usage:
    from orbit_viz import make_orbit_animation
    out = make_orbit_animation(res, r200c=..., fps=30, duration=12, figsize=(12, 6))

User-facing units
-----------------
- Length: ckpc/h
- Time:   Gyr
- Speed:  km/s

Dependencies
------------
- numpy
- matplotlib
- seaborn
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import seaborn as sns


# ----------------------------------------------------------------------
# Default palette (fallback if user does not provide clist)
# ----------------------------------------------------------------------

DEFAULT_CLIST = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"
]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _to_array(x):
    """Convert to a 1D float array."""
    return np.asarray(x, dtype=float)


def _normalize_vec2(vx: float, vy: float):
    """Return a normalized 2D vector. Zero vector stays zero."""
    n = np.hypot(vx, vy)
    if n <= 0.0:
        return 0.0, 0.0
    return vx / n, vy / n


def _sample_frame_indices(t, fps: int, duration: float):
    """
    Map output frames to source time indices.

    If duration is None or <= 0, use the native orbit duration:
        nframes = len(t)

    Otherwise create approximately fps * duration frames, spanning the full orbit.
    """
    t = _to_array(t)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("res.t must be a non-empty 1D array.")

    if duration is None or duration <= 0:
        return np.arange(t.size, dtype=int)

    nframes = max(2, int(round(float(fps) * float(duration))))
    idx = np.linspace(0, t.size - 1, nframes)
    return np.unique(np.round(idx).astype(int))


def _tail_segments_with_alpha(x, y, t, i_now, trail_gyr):
    """
    Build line segments for the recent trajectory only.

    The tail includes points with:
        t_now - trail_gyr <= t <= t_now

    Alpha fades linearly from 0 at the oldest retained point to 1 at the newest.
    """
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
        [
            np.column_stack([xw[:-1], yw[:-1]]),
            np.column_stack([xw[1:], yw[1:]]),
        ],
        axis=1,
    )

    # Segment ages measured at the segment midpoint
    tmid = 0.5 * (tw[:-1] + tw[1:])
    age = np.clip(t_now - tmid, 0.0, float(trail_gyr))
    alpha = 1.0 - age / max(float(trail_gyr), 1e-12)
    alpha = np.clip(alpha, 0.0, 1.0)

    rgba = np.zeros((alpha.size, 4), dtype=float)
    rgba[:, -1] = alpha
    return segs, rgba


def _set_user_plot_style():
    """Apply the plotting style requested by the user."""
    sns.set(style='ticks')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['mathtext.fontset'] = 'cm'


# ----------------------------------------------------------------------
# Main animation API
# ----------------------------------------------------------------------

def make_orbit_animation(
    res,
    r200c,
    *,
    fps=30,
    duration=10.0,
    figsize=(12, 6),
    outfile='orbit_animation.mp4',
    dpi=180,
    clist=None,
    trail_gyr=2.0,
    xlim=None,
    ylim=None,
    show_r200c=True,
    panel_pad=1.3,
    arrow_scale=None,
    orbit_label='orbit',
    start_label='start',
    end_label='end',
    centre_label='centre',
    rt_label=r'$r_t$',
    rsub_label=r'$r_{\rm sub}$',
    lw_trail=2.0,
    point_size=28,
    end_point_size=22,
    legend_fontsize=11,
    ncol_legend=5,
    bitrate=2400,
    writer=None,
    close_fig=True,
):
    """
    Create and save a two-panel orbit animation.

    Parameters
    ----------
    res : object
        Orbit result object. Must provide:
            res.t   : (N,)
            res.pos : (N,3) or at least [:,0], [:,1]
            res.v   : (N,3) or at least [:,0], [:,1]

        Optional:
            res.r_t   : (N,)
            res.r_sub : (N,)

    r200c : float
        Host virial radius in ckpc/h.

    fps : int
        Output video frame rate.

    duration : float
        Output video duration in seconds.
        The full orbit is resampled into fps * duration frames.

    figsize : tuple
        Figure size passed to plt.subplots.

    outfile : str or path-like
        Output filename. Usually .mp4 or .gif.

    dpi : int
        Save DPI.

    clist : list or None
        Color list. If None, uses a built-in fallback palette.

    trail_gyr : float
        Maximum time span of the visible trail on the left panel.

    xlim, ylim : tuple or None
        Optional global limits for the left panel. If None, use ±panel_pad*r200c.

    show_r200c : bool
        If True, draw the r200c circle on the left panel.

    panel_pad : float
        Plot limits scale factor if xlim/ylim are not provided.

    arrow_scale : float or None
        Arrow length in ckpc/h. If None, use r200c / 10.

    writer : str or None
        Force a writer: 'ffmpeg' or 'pillow'. If None, infer from suffix.

    close_fig : bool
        Close the figure after saving.

    Returns
    -------
    outpath : str
        Output path as a string.

    fig : matplotlib.figure.Figure
        The created figure.

    anim : matplotlib.animation.FuncAnimation
        The animation object.

    Notes
    -----
    - The left panel follows your requested style and content.
    - The right panel is centred on the instantaneous subhalo position, with circles
      for r_t and r_sub if those arrays exist.
    - The group-centre arrow points from the subhalo centre toward (0,0).
    - The velocity arrow points along the instantaneous velocity direction.
    - Both arrows have length = r200c/10 by default.
    """
    _set_user_plot_style()

    clist = DEFAULT_CLIST if clist is None else clist
    arrow_scale = float(r200c) / 10.0 if arrow_scale is None else float(arrow_scale)

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

    has_rt = hasattr(res, 'r_t') and (getattr(res, 'r_t') is not None)
    has_rsub = hasattr(res, 'r_sub') and (getattr(res, 'r_sub') is not None)

    rt = _to_array(res.r_t) if has_rt else None
    rsub = _to_array(res.r_sub) if has_rsub else None

    if has_rt and rt.size != t.size:
        raise ValueError("res.r_t must have the same length as res.t.")
    if has_rsub and rsub.size != t.size:
        raise ValueError("res.r_sub must have the same length as res.t.")

    frame_idx = _sample_frame_indices(t, fps=fps, duration=duration)
    nframes = frame_idx.size

    theta = np.linspace(0.0, 2.0 * np.pi, 500)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axL, axR = axes

    # ----------------------------
    # Left panel: orbit in group frame
    # ----------------------------
    limx = (-panel_pad * r200c, panel_pad * r200c) if xlim is None else xlim
    limy = (-panel_pad * r200c, panel_pad * r200c) if ylim is None else ylim

    axL.scatter(x[0], y[0], s=20, marker='o', color=clist[4], label=start_label)
    axL.scatter(x[-1], y[-1], s=20, marker='o', color=clist[0], label=end_label)
    axL.scatter(0.0, 0.0, s=20, marker='x', color='k', label=centre_label)

    if show_r200c:
        axL.plot(
            r200c * np.cos(theta),
            r200c * np.sin(theta),
            ls='--',
            color=clist[8],
            label=r'$R_{200c}$'
        )

    orbit_point, = axL.plot([], [], marker='o', ms=np.sqrt(point_size), color=clist[9], lw=0, label=orbit_label)
    trail = LineCollection([], linewidths=lw_trail)
    axL.add_collection(trail)

    axL.set_xlim(*limx)
    axL.set_ylim(*limy)
    axL.set_xlabel(r'$x\;[ckpc/h]$')
    axL.set_ylabel(r'$y\;[ckpc/h]$')
    axL.legend(frameon=False, ncol=ncol_legend, fontsize=legend_fontsize)
    axL.set_aspect('equal')

    # ----------------------------
    # Right panel: local subhalo-centred view
    # ----------------------------
    span_right = max(0.25 * r200c, 2.5 * arrow_scale)

    subhalo_center, = axR.plot([], [], marker='x', ms=np.sqrt(point_size), color='k', lw=0)
    rt_circle = Circle((0.0, 0.0), radius=1.0, fill=False, ec=clist[1], lw=2.0, ls='-')
    rsub_circle = Circle((0.0, 0.0), radius=1.0, fill=False, ec=clist[2], lw=2.0, ls='--')
    axR.add_patch(rt_circle)
    axR.add_patch(rsub_circle)
    rt_circle.set_visible(False)
    rsub_circle.set_visible(False)

    # We redraw arrows every frame for clean updating.
    centre_arrow = None
    vel_arrow = None

    axR.set_xlim(-span_right, span_right)
    axR.set_ylim(-span_right, span_right)
    axR.set_xlabel(r'$\Delta x\;[ckpc/h]$')
    axR.set_ylabel(r'$\Delta y\;[ckpc/h]$')
    axR.set_aspect('equal')

    # A small static legend on the right panel
    proxy_rt, = axR.plot([], [], color=clist[1], lw=2.0, ls='-', label=rt_label)
    proxy_rs, = axR.plot([], [], color=clist[2], lw=2.0, ls='--', label=rsub_label)
    proxy_c, = axR.plot([], [], marker='x', color='k', lw=0, label='subhalo centre')

    handles = [proxy_c]
    if has_rt:
        handles.append(proxy_rt)
    if has_rsub:
        handles.append(proxy_rs)
    axR.legend(handles=handles, frameon=False, fontsize=legend_fontsize, loc='upper right')

    time_text = fig.text(0.50, 0.97, '', ha='center', va='top', fontsize=12)

    def init():
        orbit_point.set_data([], [])
        trail.set_segments([])
        trail.set_color(np.empty((0, 4)))
        subhalo_center.set_data([], [])
        rt_circle.set_visible(False)
        rsub_circle.set_visible(False)
        time_text.set_text('')
        return orbit_point, trail, subhalo_center, rt_circle, rsub_circle, time_text

    def update(k):
        nonlocal centre_arrow, vel_arrow

        i = frame_idx[k]
        xi, yi = x[i], y[i]
        vxi, vyi = vx[i], vy[i]

        # Left panel orbit point + fading trail
        orbit_point.set_data([xi], [yi])

        segs, rgba = _tail_segments_with_alpha(x, y, t, i, trail_gyr)
        if segs.shape[0] > 0:
            # Set all segments to the same RGB, with varying alpha
            rgb = np.array(plt.matplotlib.colors.to_rgb(clist[9]))
            rgba[:, :3] = rgb[None, :]
            trail.set_segments(segs)
            trail.set_color(rgba)
        else:
            trail.set_segments([])
            trail.set_color(np.empty((0, 4)))

        # Right panel: subhalo-centred view
        subhalo_center.set_data([0.0], [0.0])

        # Dynamic window: large enough for circles and arrows
        local_span = span_right
        if has_rt and np.isfinite(rt[i]):
            local_span = max(local_span, 1.25 * rt[i])
        if has_rsub and np.isfinite(rsub[i]):
            local_span = max(local_span, 1.25 * rsub[i])
        local_span = max(local_span, 2.5 * arrow_scale)

        axR.set_xlim(-local_span, local_span)
        axR.set_ylim(-local_span, local_span)

        if has_rt and np.isfinite(rt[i]) and rt[i] > 0.0:
            rt_circle.center = (0.0, 0.0)
            rt_circle.set_radius(rt[i])
            rt_circle.set_visible(True)
        else:
            rt_circle.set_visible(False)

        if has_rsub and np.isfinite(rsub[i]) and rsub[i] > 0.0:
            rsub_circle.center = (0.0, 0.0)
            rsub_circle.set_radius(rsub[i])
            rsub_circle.set_visible(True)
        else:
            rsub_circle.set_visible(False)

        # Remove previous arrows
        if centre_arrow is not None:
            centre_arrow.remove()
        if vel_arrow is not None:
            vel_arrow.remove()

        # Arrow toward group centre: from subhalo to origin, i.e. direction = -(x,y)
        cx, cy = _normalize_vec2(-xi, -yi)
        centre_arrow = axR.arrow(
            0.0, 0.0,
            arrow_scale * cx, arrow_scale * cy,
            length_includes_head=True,
            head_width=0.08 * arrow_scale,
            head_length=0.14 * arrow_scale,
            lw=2.0,
            color=clist[3],
            zorder=5,
        )

        # Velocity direction arrow
        ux, uy = _normalize_vec2(vxi, vyi)
        vel_arrow = axR.arrow(
            0.0, 0.0,
            arrow_scale * ux, arrow_scale * uy,
            length_includes_head=True,
            head_width=0.08 * arrow_scale,
            head_length=0.14 * arrow_scale,
            lw=2.0,
            color=clist[0],
            zorder=5,
        )

        time_text.set_text(fr'$t = {t[i]:.3f}\,\mathrm{{Gyr}}$')
        return orbit_point, trail, subhalo_center, rt_circle, rsub_circle, time_text, centre_arrow, vel_arrow

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=nframes,
        interval=1000.0 / max(int(fps), 1),
        blit=False,
        repeat=False,
    )

    outpath = str(Path(outfile).expanduser().resolve())
    suffix = Path(outpath).suffix.lower()

    if writer is None:
        if suffix == '.gif':
            writer = 'pillow'
        else:
            writer = 'ffmpeg'

    if writer == 'ffmpeg':
        try:
            anim.save(outpath, writer=FFMpegWriter(fps=fps, bitrate=bitrate), dpi=dpi)
        except Exception as exc:
            warnings.warn(
                f"FFmpeg save failed ({exc}). Falling back to GIF via Pillow.",
                RuntimeWarning
            )
            outpath = str(Path(outpath).with_suffix('.gif'))
            anim.save(outpath, writer=PillowWriter(fps=fps), dpi=dpi)
    elif writer == 'pillow':
        if suffix != '.gif':
            outpath = str(Path(outpath).with_suffix('.gif'))
        anim.save(outpath, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        raise ValueError("writer must be None, 'ffmpeg', or 'pillow'.")

    if close_fig:
        plt.close(fig)

    return outpath, fig, anim


def preview_orbit_frame(
    res,
    r200c,
    *,
    i=-1,
    figsize=(12, 6),
    clist=None,
    trail_gyr=2.0,
    xlim=None,
    ylim=None,
    show_r200c=True,
    panel_pad=1.3,
    arrow_scale=None,
):
    """
    Draw a single frame using the same visual design as the animation.

    Useful for checking aesthetics in a notebook before exporting the video.
    """
    _set_user_plot_style()

    clist = DEFAULT_CLIST if clist is None else clist
    arrow_scale = float(r200c) / 10.0 if arrow_scale is None else float(arrow_scale)

    t = _to_array(res.t)
    pos = np.asarray(res.pos, dtype=float)
    vel = np.asarray(res.v, dtype=float)
    x = pos[:, 0]
    y = pos[:, 1]
    vx = vel[:, 0]
    vy = vel[:, 1]

    has_rt = hasattr(res, 'r_t') and (getattr(res, 'r_t') is not None)
    has_rsub = hasattr(res, 'r_sub') and (getattr(res, 'r_sub') is not None)
    rt = _to_array(res.r_t) if has_rt else None
    rsub = _to_array(res.r_sub) if has_rsub else None

    i = int(i)
    if i < 0:
        i = len(t) + i
    i = max(0, min(i, len(t) - 1))

    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axL, axR = axes

    limx = (-panel_pad * r200c, panel_pad * r200c) if xlim is None else xlim
    limy = (-panel_pad * r200c, panel_pad * r200c) if ylim is None else ylim

    axL.scatter(x[0], y[0], s=20, marker='o', color=clist[4], label='start')
    axL.scatter(x[-1], y[-1], s=20, marker='o', color=clist[0], label='end')
    axL.scatter(0.0, 0.0, s=20, marker='x', color='k', label='centre')
    if show_r200c:
        axL.plot(r200c * np.cos(theta), r200c * np.sin(theta), ls='--', color=clist[8], label=r'$R_{200c}$')

    segs, rgba = _tail_segments_with_alpha(x, y, t, i, trail_gyr)
    if segs.shape[0] > 0:
        rgb = np.array(plt.matplotlib.colors.to_rgb(clist[9]))
        rgba[:, :3] = rgb[None, :]
        lc = LineCollection(segs, colors=rgba, linewidths=2.0)
        axL.add_collection(lc)

    axL.plot([x[i]], [y[i]], marker='o', ms=6, color=clist[9], lw=0, label='orbit')
    axL.set_xlim(*limx)
    axL.set_ylim(*limy)
    axL.set_xlabel(r'$x\;[ckpc/h]$')
    axL.set_ylabel(r'$y\;[ckpc/h]$')
    axL.legend(frameon=False, ncol=5, fontsize=11)
    axL.set_aspect('equal')

    xi, yi = x[i], y[i]
    vxi, vyi = vx[i], vy[i]

    local_span = max(0.25 * r200c, 2.5 * arrow_scale)
    if has_rt and np.isfinite(rt[i]):
        local_span = max(local_span, 1.25 * rt[i])
    if has_rsub and np.isfinite(rsub[i]):
        local_span = max(local_span, 1.25 * rsub[i])

    axR.plot(0.0, 0.0, marker='x', color='k')
    if has_rt and np.isfinite(rt[i]) and rt[i] > 0.0:
        axR.add_patch(Circle((0.0, 0.0), radius=rt[i], fill=False, ec=clist[1], lw=2.0, ls='-', label=r'$r_t$'))
    if has_rsub and np.isfinite(rsub[i]) and rsub[i] > 0.0:
        axR.add_patch(Circle((0.0, 0.0), radius=rsub[i], fill=False, ec=clist[2], lw=2.0, ls='--', label=r'$r_{\rm sub}$'))

    cx, cy = _normalize_vec2(-xi, -yi)
    ux, uy = _normalize_vec2(vxi, vyi)

    axR.arrow(0.0, 0.0, arrow_scale * cx, arrow_scale * cy,
              length_includes_head=True, head_width=0.08 * arrow_scale,
              head_length=0.14 * arrow_scale, lw=2.0, color=clist[3])
    axR.arrow(0.0, 0.0, arrow_scale * ux, arrow_scale * uy,
              length_includes_head=True, head_width=0.08 * arrow_scale,
              head_length=0.14 * arrow_scale, lw=2.0, color=clist[0])

    axR.set_xlim(-local_span, local_span)
    axR.set_ylim(-local_span, local_span)
    axR.set_xlabel(r'$\Delta x\;[ckpc/h]$')
    axR.set_ylabel(r'$\Delta y\;[ckpc/h]$')
    axR.set_aspect('equal')
    axR.set_title(fr'$t = {t[i]:.3f}\,\mathrm{{Gyr}}$')

    return fig, axes
