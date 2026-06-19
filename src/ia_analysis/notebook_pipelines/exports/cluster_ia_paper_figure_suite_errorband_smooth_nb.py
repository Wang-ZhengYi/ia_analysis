"""Exported code from notebooks/raw_20260618/cluster_ia_paper_figure_suite_errorband_smooth.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # Cluster intrinsic-alignment paper figure suite — error bands and smooth distributions This notebook is a compact plotting suite for **intrinsic alignment in galaxy clusters** across gravity models and redshifts. Main plotting policy in this version: - Alignment measurements are binned means of the relevant $|\cos heta|$ statistic. - Each binned mean has an associated statistical uncertainty, shown as a **filled error band** rather than an error bar. - The default uncertainty is the standard er

# %% code cell 2
# ============================================================
# Imports
# ============================================================
import os
import pickle
import warnings
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.stats import gaussian_kde
except Exception as exc:
    gaussian_kde = None
    warnings.warn(f"scipy.stats.gaussian_kde is unavailable; falling back to smoothed histograms: {exc}")

try:
    from scipy.ndimage import gaussian_filter1d
except Exception as exc:
    gaussian_filter1d = None
    warnings.warn(f"scipy.ndimage.gaussian_filter1d is unavailable; distribution smoothing will be more basic: {exc}")


try:
    import h5py
except Exception as exc:
    h5py = None
    warnings.warn(f"h5py is unavailable: {exc}")

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

# Project-local modules used by your original notebook.
# The plotting functions below can fall back to internal eigenvector routines if II/VI/VV are unavailable.
try:
    import Iana
    from Iana import *
except Exception as exc:
    warnings.warn(f"Could not import Iana: {exc}")

try:
    import shape
    from shape import *
except Exception as exc:
    warnings.warn(f"Could not import shape: {exc}")

try:
    import arts
    from arts import *
except Exception as exc:
    warnings.warn(f"Could not import arts: {exc}")

try:
    from tidal_field import *
except Exception as exc:
    warnings.warn(f"Could not import tidal_field: {exc}")

# %% code cell 3
# ============================================================
# Global paths and catalogue configuration
# ============================================================
BASE_DIR = Path('/cosma8/data/dp203/dc-wang17/MG_global')
PICKLE_PATH = BASE_DIR / 'MArenew.pkl'
OUTDIR = Path('./plots_cluster_ia_paper')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Snapshot-to-redshift map used in your existing analysis.
zmap = {
    1: 2.00,
    6: 0.97,
    8: 0.80,
    12: 0.51,
    15: 0.33,
    18: 0.16,
    21: 0.00,
}
ALL_SNAP_LIST = sorted(zmap.keys())

# Model sequence. GR is kept last so it can be used as the reference model.
flags = ['F40', 'F45', 'F50', 'F55', 'F60', 'GR']
MG_FLAGS = [f for f in flags if f != 'GR']

FLAG_COLOR = {
    'F40': '#abcdef',
    'F45': '#79B9DC',
    'F50': '#5F81C2',
    'F55': '#687CBC',
    'F60': '#0C52B5',
    'GR': 'k',
}

# A compact linewidth/marker style for dense paper panels.
FLAG_LS = {
    'F40': '-',
    'F45': '-',
    'F50': '-',
    'F55': '-',
    'F60': '-',
    'GR': '-',
}

# %% code cell 4
# ============================================================
# Plot style utilities
# ============================================================
def setup_plot_style(fontsize=7, use_tex=False):
    """Set compact paper-style plotting defaults."""
    sns.set(style='ticks')
    mpl.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize - 1,
        'ytick.labelsize': fontsize - 1,
        'legend.fontsize': fontsize - 1,
        'figure.titlesize': fontsize + 1,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'mathtext.fontset': 'cm',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.usetex': use_tex,
    })


def save_fig(fig, filename, outdir=OUTDIR, dpi=300, close=False):
    """Save both PDF and PNG unless an explicit extension is supplied."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    filename = Path(filename)
    paths = []
    if filename.suffix:
        path = outdir / filename.name
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    else:
        for ext in ['.pdf', '.png']:
            path = outdir / f'{filename.name}{ext}'
            fig.savefig(path, dpi=dpi)
            paths.append(path)
    if close:
        plt.close(fig)
    return paths


def despine_all(fig):
    for ax in fig.axes:
        try:
            sns.despine(ax=ax)
        except Exception:
            pass
    return fig


def format_dense_axis(ax):
    ax.tick_params(which='both', direction='in', top=True, right=True, length=2.5, pad=1.5)
    ax.minorticks_on()
    return ax

setup_plot_style()

# %% [markdown] cell 5
# ## Load or build the analysis dictionary The plotting suite expects a dictionary `MAset[flag][snap_key]`, where `snap_key` is a three-digit string such as `'021'`. The loader first tries to read `/cosma8/data/dp203/dc-wang17/MG_global/MArenew.pkl`. If it is missing, it rebuilds `MAset` from the HDF5 files.

# %% code cell 6
# ============================================================
# Data loading and derived quantities
# ============================================================
def _arr(x):
    """Convert h5py datasets or array-like objects to NumPy arrays."""
    return np.asarray(x[()] if hasattr(x, '__class__') and x.__class__.__name__ == 'Dataset' else x)


def _safe_get_h5(f, key, default=None):
    if key in f:
        return np.asarray(f[key])
    return default


def _safe_group_get(f, group, key, default=None):
    if group in f and key in f[group]:
        return np.asarray(f[group][key])
    return default


def _fallback_chi_q_s(I):
    """Fallback triaxiality/shape proxies from inertia eigenvalues."""
    I = np.asarray(I)
    n = I.shape[0]
    out = {'chi': np.full(n, np.nan), 'q': np.full(n, np.nan), 's': np.full(n, np.nan)}
    try:
        T = np.asarray(I, dtype=float)
        if T.ndim == 2 and T.shape[1] == 6:
            TT = np.zeros((T.shape[0], 3, 3), dtype=float)
            TT[:, 0, 0] = T[:, 0]
            TT[:, 1, 1] = T[:, 1]
            TT[:, 2, 2] = T[:, 2]
            TT[:, 0, 1] = TT[:, 1, 0] = T[:, 3]
            TT[:, 0, 2] = TT[:, 2, 0] = T[:, 4]
            TT[:, 1, 2] = TT[:, 2, 1] = T[:, 5]
            T = TT
        elif not (T.ndim == 3 and T.shape[1:] == (3, 3)):
            raise ValueError(f'Cannot interpret tensor shape {T.shape}')
        T = 0.5 * (T + np.swapaxes(T, -1, -2))
        vals = np.linalg.eigvalsh(T)
        vals = np.sort(np.abs(vals), axis=1)[:, ::-1]
        a = np.sqrt(np.maximum(vals[:, 0], 0))
        b = np.sqrt(np.maximum(vals[:, 1], 0))
        c = np.sqrt(np.maximum(vals[:, 2], 0))
        out['q'] = np.divide(b, a, out=np.full_like(a, np.nan), where=a > 0)
        out['s'] = np.divide(c, a, out=np.full_like(a, np.nan), where=a > 0)
        # chi: rough oblate/prolate indicator; keep it bounded for plotting.
        out['chi'] = np.clip((out['q'] - out['s']) / (1.0 - out['s'] + 1e-12), -1, 1)
    except Exception:
        pass
    return out


def _fallback_omega_fig(I, dI):
    """Very conservative fallback if omega_fig is unavailable."""
    I = np.asarray(I)
    return np.full((I.shape[0], 3), np.nan)


def mkMA_from_h5(path):
    """Read one HDF5 catalogue and construct the flat MA dictionary used by the plotter."""
    if h5py is None:
        raise RuntimeError('h5py is required to build MAset from HDF5 files.')

    path = Path(path)
    with h5py.File(path, 'r') as f:
        MA = {}

        for key in [
            'CenID', 'GroupID', 'Group_M_Crit200', 'Group_M_Crit500',
            'Group_R_Crit200', 'Group_R_Crit500', 'SubhaloBHMass',
            'SubhaloBHMdot', 'SubhaloGasMetallicity', 'SubhaloID',
            'SubhaloMass', 'SubhaloMassInRadType', 'SubhaloSFR',
            'SubhaloVmax', 'SubhaloWindMass'
        ]:
            if key in f:
                MA[key] = np.asarray(f[key])

        pos_rel = np.asarray(f['pos_rel'])
        vel_rel = np.asarray(f['vel_rel'])
        r200 = np.asarray(f['Group_R_Crit200'])
        MA['R'] = pos_rel
        MA['V'] = vel_rel
        MA['R_over_R_200c'] = np.divide(
            np.linalg.norm(pos_rel, axis=1), r200,
            out=np.full(pos_rel.shape[0], np.nan), where=r200 > 0
        )

        for ob in ['DM', 'Star']:
            I = np.asarray(f[ob]['I'])
            dI = np.asarray(f[ob]['dI']) if 'dI' in f[ob] else np.full_like(I, np.nan)
            MA[f'I_{ob}'] = I
            try:
                chis = chiSO(I)
            except Exception:
                chis = _fallback_chi_q_s(I)
            MA[f'chi_{ob}'] = np.asarray(chis.get('chi', np.full(I.shape[0], np.nan)))
            MA[f'q_{ob}'] = np.asarray(chis.get('q', np.full(I.shape[0], np.nan)))
            MA[f's_{ob}'] = np.asarray(chis.get('s', np.full(I.shape[0], np.nan)))
            MA[f'kappa_rot_{ob}'] = _safe_group_get(f, ob, 'kappa_rot', np.full(I.shape[0], np.nan))
            try:
                MA[f'omega_{ob}'] = omega_fig(I, dI)
            except Exception:
                MA[f'omega_{ob}'] = _fallback_omega_fig(I, dI)

            cos_err = _safe_group_get(f, ob, 'cos_err', None)
            axis_relerr = _safe_group_get(f, ob, 'axis_relerr', None)
            if cos_err is not None:
                MA[f'cos_err_max_{ob}'] = np.nanmin(cos_err, axis=1)
            else:
                MA[f'cos_err_max_{ob}'] = np.full(I.shape[0], 0.0)
            if axis_relerr is not None:
                MA[f'axe_err_max_{ob}'] = np.nanmin(axis_relerr, axis=1)
            else:
                MA[f'axe_err_max_{ob}'] = np.full(I.shape[0], 0.0)

        MA['T_grp'] = _safe_get_h5(f, 'Tidal_grp', np.full_like(MA['I_DM'], np.nan))
        MA['T_GR'] = _safe_get_h5(f, 'Tidal_tot', np.full_like(MA['I_DM'], np.nan))
        T_mg_extra = _safe_get_h5(f, 'Tidal_tot_mg', np.zeros_like(MA['T_GR']))
        MA['T_MG'] = MA['T_GR'] + T_mg_extra

    return MA


def load_or_build_MAset(use_pickle=True, save_pickle=True, overwrite=False):
    """Load MAset from pickle, or build it from the HDF5 catalogues."""
    if use_pickle and PICKLE_PATH.exists() and not overwrite:
        with open(PICKLE_PATH, 'rb') as f:
            MAset = pickle.load(f)
        print(f'Loaded MAset from {PICKLE_PATH}')
        return MAset

    MAset = {}
    for flag in tqdm(flags, desc='flags'):
        MAset[flag] = {}
        for snap in tqdm(ALL_SNAP_LIST, desc=f'{flag}', leave=False):
            path = BASE_DIR / f'L302_N1136_{flag}_s{snap:03d}.hdf5'
            if not path.exists():
                warnings.warn(f'Missing catalogue: {path}')
                continue
            MAset[flag][f'{snap:03d}'] = mkMA_from_h5(path)

    if save_pickle:
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(MAset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved MAset to {PICKLE_PATH}')
    return MAset

# Main data object.
MAset = load_or_build_MAset(use_pickle=True, save_pickle=True, overwrite=False)

# %% code cell 7
# ============================================================
# Low-level alignment and physical-variable helpers
# ============================================================
def as_tensor_array(T):
    """Return tensor array with shape (N, 3, 3). Supports (N, 3, 3) and common (N, 6) storage."""
    T = np.asarray(T, dtype=float)
    if T.ndim == 3 and T.shape[1:] == (3, 3):
        return 0.5 * (T + np.swapaxes(T, -1, -2))
    if T.ndim == 2 and T.shape[1] == 6:
        out = np.zeros((T.shape[0], 3, 3), dtype=float)
        out[:, 0, 0] = T[:, 0]
        out[:, 1, 1] = T[:, 1]
        out[:, 2, 2] = T[:, 2]
        out[:, 0, 1] = out[:, 1, 0] = T[:, 3]
        out[:, 0, 2] = out[:, 2, 0] = T[:, 4]
        out[:, 1, 2] = out[:, 2, 1] = T[:, 5]
        return out
    raise ValueError(f'Cannot interpret tensor array with shape {T.shape}')


def norm_vectors(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=1)
    return np.divide(v, n[:, None], out=np.full_like(v, np.nan), where=n[:, None] > 0)


def tensor_axis(T, axis='major'):
    """Principal axis of a tensor array. major=max eigenvalue, minor=min eigenvalue."""
    T = as_tensor_array(T)
    vals, vecs = np.linalg.eigh(T)
    if axis == 'major':
        idx = np.argmax(vals, axis=1)
    elif axis in {'middle', 'intermediate'}:
        idx = np.argsort(vals, axis=1)[:, 1]
    elif axis == 'minor':
        idx = np.argmin(vals, axis=1)
    else:
        raise ValueError(f'Unknown tensor axis: {axis}')
    return np.asarray([vecs[i, :, idx[i]] for i in range(T.shape[0])])


def cos_tensor_tensor(A, B, axis_a='major', axis_b='major', sign_a=1.0, sign_b=1.0, absolute=True):
    a = tensor_axis(sign_a * np.asarray(A), axis=axis_a)
    b = tensor_axis(sign_b * np.asarray(B), axis=axis_b)
    c = np.sum(norm_vectors(a) * norm_vectors(b), axis=1)
    return np.abs(c) if absolute else c


def cos_vector_tensor(v, T, axis_t='major', sign_v=1.0, sign_t=1.0, absolute=True):
    a = norm_vectors(sign_v * np.asarray(v))
    b = tensor_axis(sign_t * np.asarray(T), axis=axis_t)
    c = np.sum(a * norm_vectors(b), axis=1)
    return np.abs(c) if absolute else c


def cos_vector_vector(v1, v2, sign_1=1.0, sign_2=1.0, absolute=True):
    a = norm_vectors(sign_1 * np.asarray(v1))
    b = norm_vectors(sign_2 * np.asarray(v2))
    c = np.sum(a * b, axis=1)
    return np.abs(c) if absolute else c


def alignment_metric(MA, cfg):
    """Evaluate one alignment metric from the metric registry."""
    kind = cfg['kind']
    comp = cfg.get('component', 'major')
    absolute = cfg.get('absolute', True)

    # Prefer project functions II/VI/VV where possible to keep consistency with the old notebook.
    try:
        if kind == 'tensor_tensor' and 'II' in globals():
            A = cfg.get('sign_a', 1.0) * np.asarray(MA[cfg['a']])
            B = cfg.get('sign_b', 1.0) * np.asarray(MA[cfg['b']])
            y = np.asarray(II(A, B)[comp])
            return np.abs(y) if absolute else y
        if kind == 'vector_tensor' and 'VI' in globals():
            v = cfg.get('sign_v', 1.0) * np.asarray(MA[cfg['v']])
            T = cfg.get('sign_t', 1.0) * np.asarray(MA[cfg['t']])
            y = np.asarray(VI(v, T)[comp])
            return np.abs(y) if absolute else y
        if kind == 'vector_vector' and 'VV' in globals():
            y = np.asarray(VV(cfg.get('sign_a', 1.0) * MA[cfg['a']], cfg.get('sign_b', 1.0) * MA[cfg['b']]))
            return np.abs(y) if absolute else y
    except Exception:
        pass

    # Internal fallback.
    if kind == 'tensor_tensor':
        return cos_tensor_tensor(
            MA[cfg['a']], MA[cfg['b']],
            axis_a=cfg.get('axis_a', comp), axis_b=cfg.get('axis_b', comp),
            sign_a=cfg.get('sign_a', 1.0), sign_b=cfg.get('sign_b', 1.0),
            absolute=absolute,
        )
    if kind == 'vector_tensor':
        return cos_vector_tensor(
            MA[cfg['v']], MA[cfg['t']],
            axis_t=cfg.get('axis_t', comp),
            sign_v=cfg.get('sign_v', 1.0), sign_t=cfg.get('sign_t', 1.0),
            absolute=absolute,
        )
    if kind == 'vector_vector':
        return cos_vector_vector(
            MA[cfg['a']], MA[cfg['b']],
            sign_1=cfg.get('sign_a', 1.0), sign_2=cfg.get('sign_b', 1.0),
            absolute=absolute,
        )
    if kind == 'callable':
        return cfg['func'](MA)
    raise ValueError(f'Unknown metric kind: {kind}')


def safe_log10(x, min_positive=1e-300):
    x = np.asarray(x, dtype=float)
    return np.log10(np.where(x > min_positive, x, np.nan))


def orbital_frequency(MA):
    R = np.asarray(MA['R'], dtype=float)
    V = np.asarray(MA['V'], dtype=float)
    R2 = np.sum(R * R, axis=1)
    return np.divide(np.cross(R, V), R2[:, None], out=np.full_like(R, np.nan), where=R2[:, None] > 0)


def period_from_omega(omega):
    w = np.linalg.norm(np.asarray(omega, dtype=float), axis=1)
    return np.divide(2.0 * np.pi, w, out=np.full_like(w, np.nan), where=w > 0)


def t_orb(MA):
    return period_from_omega(orbital_frequency(MA))


def t_fig_star(MA):
    return period_from_omega(MA['omega_Star'])


def t_fig_dm(MA):
    return period_from_omega(MA['omega_DM'])


def safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=np.isfinite(b) & (b != 0))


def dcosdt_star_radial(MA):
    """Proxy used in the original notebook: VI(V, I_star).major + VV(omega_star, R)."""
    try:
        ev = np.asarray(VI(MA['V'], MA['I_Star'])['major'])
        wr = np.asarray(VV(MA['omega_Star'], MA['R']))
        return ev + wr
    except Exception:
        return cos_vector_tensor(MA['V'], MA['I_Star'], axis_t='major', absolute=False) + cos_vector_vector(MA['omega_Star'], MA['R'], absolute=False)

def field_or_nan(MA, key, like='SubhaloID'):
    """Return MA[key] as float array; if the key is absent, return a NaN array with catalogue length."""
    if key in MA:
        return np.asarray(MA[key], dtype=float)
    n = len(np.asarray(MA[like])) if like in MA else 0
    return np.full(n, np.nan, dtype=float)

# %% code cell 8
# ============================================================
# Metric, x-variable and selection registries
# ============================================================
ALIGNMENT_METRICS = {
    # Galaxy/halo shape alignments.
    'CGHA': dict(label=r'Cen. $*{-}$halo', ylabel=r'$\langle |\hat e_{*,1}\!\cdot\!\hat e_{\rm DM,1}|\rangle$', kind='tensor_tensor', a='I_Star', b='I_DM', component='major', sample='central', err_obj='Star', err_limit=0.01),
    'SGHA': dict(label=r'Sat. $*{-}$sub.', ylabel=r'$\langle |\hat e_{*,1}\!\cdot\!\hat e_{\rm DM,1}|\rangle$', kind='tensor_tensor', a='I_Star', b='I_DM', component='major', sample='satellite', err_obj='Star', err_limit=0.01),

    # Radial alignments.
    'SRA_STAR': dict(label=r'Sat. $*{-}r$', ylabel=r'$\langle |\hat r\!\cdot\!\hat e_{*,1}|\rangle$', kind='vector_tensor', v='R', t='I_Star', component='major', sample='satellite', err_obj='Star', err_limit=0.01),
    'SRA_DM': dict(label=r'Sub. DM$ {-}r$', ylabel=r'$\langle |\hat r\!\cdot\!\hat e_{\rm DM,1}|\rangle$', kind='vector_tensor', v='R', t='I_DM', component='major', sample='satellite', err_obj='DM', err_limit=0.01),

    # Tidal tensor versus satellite position vector.
    'TRA_GRP': dict(label=r'$T_{\rm grp}{-}r$', ylabel=r'$\langle |\hat r\!\cdot\!\hat T_{\rm grp,1}|\rangle$', kind='vector_tensor', v='R', t='T_grp', component='major', sample='satellite', err_obj='Star', err_limit=0.01),
    'TRA_GR': dict(label=r'$T_{\rm GR}{-}r$', ylabel=r'$\langle |\hat r\!\cdot\!\hat T_{\rm GR,1}|\rangle$', kind='vector_tensor', v='R', t='T_GR', sign_t=-1.0, component='major', sample='satellite', err_obj='Star', err_limit=0.01),
    'TRA_MG': dict(label=r'$T_{\rm tot}{-}r$', ylabel=r'$\langle |\hat r\!\cdot\!\hat T_{\rm tot,1}|\rangle$', kind='vector_tensor', v='R', t='T_MG', sign_t=-1.0, component='major', sample='satellite', err_obj='Star', err_limit=0.01),

    # Satellite stellar shape versus tidal field.
    'GTA_STAR_GRP': dict(label=r'Sat. $*{-}T_{\rm grp}$', ylabel=r'$\langle |\hat e_{*,1}\!\cdot\!\hat T_{\rm grp,1}|\rangle$', kind='tensor_tensor', a='I_Star', b='T_grp', component='major', sample='satellite', err_obj='Star', err_limit=0.01),
    'GTA_STAR_GR': dict(label=r'Sat. $*{-}T_{\rm GR}$', ylabel=r'$\langle |\hat e_{*,1}\!\cdot\!\hat T_{\rm GR,1}|\rangle$', kind='tensor_tensor', a='I_Star', b='T_GR', sign_b=-1.0, component='major', sample='satellite', err_obj='Star', err_limit=0.01),
    'GTA_STAR_MG': dict(label=r'Sat. $*{-}T_{\rm tot}$', ylabel=r'$\langle |\hat e_{*,1}\!\cdot\!\hat T_{\rm tot,1}|\rangle$', kind='tensor_tensor', a='I_Star', b='T_MG', sign_b=-1.0, component='major', sample='satellite', err_obj='Star', err_limit=0.01),

    # Satellite/subhalo DM shape versus tidal field.
    'HTA_GRP': dict(label=r'Sub. DM$ {-}T_{\rm grp}$', ylabel=r'$\langle |\hat e_{\rm DM,1}\!\cdot\!\hat T_{\rm grp,1}|\rangle$', kind='tensor_tensor', a='I_DM', b='T_grp', component='major', sample='satellite', err_obj='DM', err_limit=0.01),
    'HTA_GR': dict(label=r'Sub. DM$ {-}T_{\rm GR}$', ylabel=r'$\langle |\hat e_{\rm DM,1}\!\cdot\!\hat T_{\rm GR,1}|\rangle$', kind='tensor_tensor', a='I_DM', b='T_GR', sign_b=-1.0, component='major', sample='satellite', err_obj='DM', err_limit=0.01),
    'HTA_MG': dict(label=r'Sub. DM$ {-}T_{\rm tot}$', ylabel=r'$\langle |\hat e_{\rm DM,1}\!\cdot\!\hat T_{\rm tot,1}|\rangle$', kind='tensor_tensor', a='I_DM', b='T_MG', sign_b=-1.0, component='major', sample='satellite', err_obj='DM', err_limit=0.01),

    # Central galaxy/halo tidal alignments.
    'CGTA_STAR_GRP': dict(label=r'Cen. $*{-}T_{\rm grp}$', ylabel=r'$\langle |\hat e_{*,1}\!\cdot\!\hat T_{\rm grp,1}|\rangle$', kind='tensor_tensor', a='I_Star', b='T_grp', component='major', sample='central', err_obj='Star', err_limit=0.01),
    'CGTA_STAR_GR': dict(label=r'Cen. $*{-}T_{\rm GR}$', ylabel=r'$\langle |\hat e_{*,1}\!\cdot\!\hat T_{\rm GR,1}|\rangle$', kind='tensor_tensor', a='I_Star', b='T_GR', sign_b=-1.0, component='major', sample='central', err_obj='Star', err_limit=0.01),
    'CGTA_STAR_MG': dict(label=r'Cen. $*{-}T_{\rm tot}$', ylabel=r'$\langle |\hat e_{*,1}\!\cdot\!\hat T_{\rm tot,1}|\rangle$', kind='tensor_tensor', a='I_Star', b='T_MG', sign_b=-1.0, component='major', sample='central', err_obj='Star', err_limit=0.01),
    'CHTA_GRP': dict(label=r'Cen. halo$ {-}T_{\rm grp}$', ylabel=r'$\langle |\hat e_{\rm DM,1}\!\cdot\!\hat T_{\rm grp,1}|\rangle$', kind='tensor_tensor', a='I_DM', b='T_grp', component='major', sample='central', err_obj='DM', err_limit=0.01),
}

X_VARIABLES = {
    'mstar_log': dict(label=r'$\log_{10}(M_*/M_\odot h^{-1})$', getter=lambda MA: safe_log10(MA['SubhaloMassInRadType'][:, 4]) + 10.0, xlim=(9.8, 12.2), logx=False, nbins=14),
    'mdm_log': dict(label=r'$\log_{10}(M_{\rm DM}/M_\odot h^{-1})$', getter=lambda MA: safe_log10(MA['SubhaloMassInRadType'][:, 1]) + 10.0, xlim=(9.8, 13.0), logx=False, nbins=14),
    'm200_log': dict(label=r'$\log_{10}(M_{200c}/M_\odot h^{-1})$', getter=lambda MA: safe_log10(MA['Group_M_Crit200']) + 10.0, xlim=(10.0, 14.8), logx=False, nbins=14),
    'r200': dict(label=r'$r/r_{200c}$', getter=lambda MA: MA['R_over_R_200c'], xlim=(0.03, 1.50), logx=True, nbins=14),
    'kappa_star': dict(label=r'$\kappa_{\rm rot,*}$', getter=lambda MA: MA['kappa_rot_Star'], xlim=(0.0, 0.9), logx=False, nbins=14),
    'kappa_dm': dict(label=r'$\kappa_{\rm rot,DM}$', getter=lambda MA: MA['kappa_rot_DM'], xlim=(0.0, 0.9), logx=False, nbins=14),
    'q_star': dict(label=r'$q_*$', getter=lambda MA: MA['q_Star'], xlim=(0.0, 1.0), logx=False, nbins=14),
    's_star': dict(label=r'$s_*$', getter=lambda MA: MA['s_Star'], xlim=(0.0, 1.0), logx=False, nbins=14),
    'chi_star': dict(label=r'$\chi_*$', getter=lambda MA: MA['chi_Star'], xlim=(-1.0, 1.0), logx=False, nbins=16),
    'q_dm': dict(label=r'$q_{\rm DM}$', getter=lambda MA: MA['q_DM'], xlim=(0.0, 1.0), logx=False, nbins=14),
    's_dm': dict(label=r'$s_{\rm DM}$', getter=lambda MA: MA['s_DM'], xlim=(0.0, 1.0), logx=False, nbins=14),
    'chi_dm': dict(label=r'$\chi_{\rm DM}$', getter=lambda MA: MA['chi_DM'], xlim=(-1.0, 1.0), logx=False, nbins=16),
    'cos_err_star': dict(label=r'$\epsilon_{\cos,*}$', getter=lambda MA: field_or_nan(MA, 'cos_err_max_Star'), xlim=(1e-5, 3e-1), logx=True, nbins=28, apply_err_cut=False),
    'cos_err_dm': dict(label=r'$\epsilon_{\cos,{\rm DM}}$', getter=lambda MA: field_or_nan(MA, 'cos_err_max_DM'), xlim=(1e-5, 3e-1), logx=True, nbins=28, apply_err_cut=False),
    'tfig_star_torb': dict(label=r'$T_{\rm fig,*}/T_{\rm orb}$', getter=lambda MA: safe_ratio(t_fig_star(MA), t_orb(MA)), xlim=(1e-3, 1e2), logx=True, nbins=16),
    'tfig_dm_torb': dict(label=r'$T_{\rm fig,DM}/T_{\rm orb}$', getter=lambda MA: safe_ratio(t_fig_dm(MA), t_orb(MA)), xlim=(1e-3, 1e2), logx=True, nbins=16),
    'tfig_star_dm': dict(label=r'$T_{\rm fig,*}/T_{\rm fig,DM}$', getter=lambda MA: safe_ratio(t_fig_star(MA), t_fig_dm(MA)), xlim=(1e-2, 1e2), logx=True, nbins=16),
    'dcosdt_star': dict(label=r'$d\cos\theta_{\rm RA,*}/dt$', getter=dcosdt_star_radial, xlim=(-2.0, 2.0), logx=False, nbins=16),
}

PROPERTY_VARIABLES = {
    'mstar_log': X_VARIABLES['mstar_log'],
    'm200_log': X_VARIABLES['m200_log'],
    'r200': X_VARIABLES['r200'],
    'kappa_star': X_VARIABLES['kappa_star'],
    'kappa_dm': X_VARIABLES['kappa_dm'],
    'q_star': X_VARIABLES['q_star'],
    's_star': X_VARIABLES['s_star'],
    'chi_star': X_VARIABLES['chi_star'],
    'q_dm': X_VARIABLES['q_dm'],
    's_dm': X_VARIABLES['s_dm'],
    'chi_dm': X_VARIABLES['chi_dm'],
    'cos_err_star': X_VARIABLES['cos_err_star'],
    'cos_err_dm': X_VARIABLES['cos_err_dm'],
    'tfig_star_torb': X_VARIABLES['tfig_star_torb'],
    'tfig_dm_torb': X_VARIABLES['tfig_dm_torb'],
}

# %% code cell 9
# ============================================================
# Selection masks, binned statistics, error bands, and smooth densities
# ============================================================
# Statistical uncertainty for alignment curves.
# - 'sem': standard error of the mean, std(y)/sqrt(N). Fast and stable.
# - 'bootstrap': percentile bootstrap half-width. Slower but more non-parametric.
ERROR_METHOD = 'sem'
N_BOOTSTRAP = 300
BOOTSTRAP_PERCENTILES = (16.0, 84.0)
RNG_SEED = 12345

# Visual style for error bands and curves.
ERROR_ALPHA = 0.16
DELTA_ERROR_ALPHA = 0.18
LINEWIDTH_MAIN = 1.35
LINEWIDTH_GR = 1.65
DENSITY_LINEWIDTH = 1.25
DENSITY_GRID_SIZE = 256
DENSITY_MIN_COUNT = 12
DENSITY_SMOOTH_SIGMA = 1.1


def sample_mask(MA, sample='satellite', err_obj='Star', err_limit=0.01, require_finite=True):
    sid = np.asarray(MA['SubhaloID'])
    cid = np.asarray(MA['CenID'])
    if sample in {'satellite', 'sat', 'sub'}:
        mask = sid != cid
    elif sample in {'central', 'cen'}:
        mask = sid == cid
    elif sample == 'all':
        mask = np.ones_like(sid, dtype=bool)
    else:
        raise ValueError(f'Unknown sample: {sample}')

    err_key = f'cos_err_max_{err_obj}'
    if err_key in MA and err_limit is not None:
        err = np.asarray(MA[err_key], dtype=float)
        mask &= np.isfinite(err) & (err < err_limit)

    if require_finite:
        mask &= np.isfinite(np.asarray(MA['R_over_R_200c'], dtype=float))
    return mask


def finite_xy_mask(x, y, extra=None):
    mask = np.isfinite(x) & np.isfinite(y)
    if extra is not None:
        mask &= np.asarray(extra, dtype=bool)
    return mask


def bin_edges_from_config(xcfg, x=None, nbins=None):
    nbins = int(nbins or xcfg.get('nbins', 10))
    xmin, xmax = xcfg.get('xlim', (None, None))
    if xmin is None or xmax is None:
        finite = np.isfinite(x)
        if xcfg.get('logx', False):
            finite &= x > 0
        if not np.any(finite):
            return np.linspace(0.0, 1.0, nbins + 1)
        xmin, xmax = np.nanpercentile(x[finite], [2, 98])
    if xcfg.get('logx', False):
        if x is not None and np.any((x > 0) & np.isfinite(x)):
            xmin = max(xmin, np.nanmin(x[(x > 0) & np.isfinite(x)]))
        return np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    return np.linspace(xmin, xmax, nbins + 1)


def mean_error(y, method=None, n_bootstrap=None, rng=None):
    """Mean and symmetric statistical uncertainty for a one-dimensional sample."""
    method = method or ERROR_METHOD
    n_bootstrap = int(n_bootstrap or N_BOOTSTRAP)
    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    n = len(yy)
    if n == 0:
        return np.nan, np.nan, 0
    mu = float(np.nanmean(yy))
    if n < 2:
        return mu, np.nan, n
    if method == 'bootstrap':
        if rng is None:
            rng = np.random.default_rng(RNG_SEED)
        boot = np.empty(n_bootstrap, dtype=float)
        for ib in range(n_bootstrap):
            boot[ib] = np.nanmean(rng.choice(yy, size=n, replace=True))
        lo, hi = np.nanpercentile(boot, BOOTSTRAP_PERCENTILES)
        err = 0.5 * (hi - lo)
    elif method == 'sem':
        err = float(np.nanstd(yy, ddof=1) / np.sqrt(n))
    else:
        raise ValueError("ERROR_METHOD must be either 'sem' or 'bootstrap'.")
    return mu, err, n


def binned_mean(x, y, mask=None, edges=None, min_count=8, error_method=None):
    """Binned mean with a per-bin uncertainty used for filled error bands."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    mask = finite_xy_mask(x, y, mask)
    if edges is None:
        if not np.any(mask):
            edges = np.linspace(0.0, 1.0, 11)
        else:
            edges = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if np.all(edges > 0) and np.nanmax(edges) / np.nanmin(edges) > 20:
        centers = np.sqrt(edges[:-1] * edges[1:])

    mean = np.full(len(centers), np.nan)
    err = np.full(len(centers), np.nan)
    n = np.zeros(len(centers), dtype=int)
    rng = np.random.default_rng(RNG_SEED) if (error_method or ERROR_METHOD) == 'bootstrap' else None
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = mask & (x >= lo) & (x < hi)
        if i == len(centers) - 1:
            m = mask & (x >= lo) & (x <= hi)
        n[i] = int(np.sum(m))
        if n[i] >= min_count:
            mean[i], err[i], _ = mean_error(y[m], method=error_method, rng=rng)
    return centers, mean, err, n


def get_curve(flag, snap, metric_key, x_key, *, edges=None, min_count=8, sample_override=None, err_limit_override=None, error_method=None):
    MA = MAset[flag][f'{snap:03d}']
    mcfg = ALIGNMENT_METRICS[metric_key]
    xcfg = X_VARIABLES[x_key]
    y = alignment_metric(MA, mcfg)
    x = xcfg['getter'](MA)
    mask = sample_mask(
        MA,
        sample=sample_override or mcfg.get('sample', 'satellite'),
        err_obj=mcfg.get('err_obj', 'Star'),
        err_limit=err_limit_override if err_limit_override is not None else mcfg.get('err_limit', 0.01),
    )
    if edges is None:
        edges = bin_edges_from_config(xcfg, x=x)
    return binned_mean(x, y, mask=mask, edges=edges, min_count=min_count, error_method=error_method)


def scalar_summary(flag, snap, metric_key, x_key=None, xlim=None, sample_override=None, err_limit_override=None, error_method=None):
    MA = MAset[flag][f'{snap:03d}']
    mcfg = ALIGNMENT_METRICS[metric_key]
    y = alignment_metric(MA, mcfg)
    mask = sample_mask(
        MA,
        sample=sample_override or mcfg.get('sample', 'satellite'),
        err_obj=mcfg.get('err_obj', 'Star'),
        err_limit=err_limit_override if err_limit_override is not None else mcfg.get('err_limit', 0.01),
    )
    if x_key is not None and xlim is not None:
        x = X_VARIABLES[x_key]['getter'](MA)
        mask &= np.isfinite(x) & (x >= xlim[0]) & (x <= xlim[1])
    mask &= np.isfinite(y)
    if np.sum(mask) == 0:
        return np.nan, np.nan, 0
    return mean_error(y[mask], method=error_method)


def _transformed_domain(x, xcfg):
    """Return x, transformed x, and transformed limits for smooth density estimation."""
    x = np.asarray(x, dtype=float)
    logx = bool(xcfg.get('logx', False))
    xlim = xcfg.get('xlim', None)
    m = np.isfinite(x)
    if logx:
        m &= x > 0
    if xlim is not None:
        lo, hi = xlim
        if lo is not None:
            m &= x >= lo
        if hi is not None:
            m &= x <= hi
    if not np.any(m):
        return np.array([]), np.array([]), (0.0, 1.0)
    xx = x[m]
    if xlim is None or xlim[0] is None or xlim[1] is None:
        lo, hi = np.nanpercentile(xx, [1, 99])
    else:
        lo, hi = xlim
    if logx:
        lo = max(lo, np.nanmin(xx[xx > 0]))
        uu = np.log10(xx)
        ulim = (np.log10(lo), np.log10(hi))
    else:
        uu = xx
        ulim = (lo, hi)
    return xx, uu, ulim


def smooth_density_curve(x, xcfg, *, n_grid=DENSITY_GRID_SIZE, min_count=DENSITY_MIN_COUNT, bw_method='scott'):
    """
    Smooth one-dimensional distribution curve.

    For log-x variables the smoothing is done in log10(x), then plotted against x.
    The density is normalized in the transformed coordinate. This is intentional for
    compact comparison of shapes across gravity models and redshifts.
    """
    xx, uu, (umin, umax) = _transformed_domain(x, xcfg)
    if len(uu) < min_count or not np.isfinite(umin) or not np.isfinite(umax) or umin == umax:
        return np.array([]), np.array([]), len(uu)
    ugrid = np.linspace(umin, umax, int(n_grid))

    if gaussian_kde is not None and len(uu) >= 3:
        try:
            kde = gaussian_kde(uu, bw_method=bw_method)
            dens = kde(ugrid)
        except Exception:
            dens = None
    else:
        dens = None

    if dens is None:
        nbins = max(20, min(80, int(np.sqrt(len(uu)) * 3)))
        hist, edges = np.histogram(uu, bins=nbins, range=(umin, umax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if gaussian_filter1d is not None:
            hist = gaussian_filter1d(hist, sigma=DENSITY_SMOOTH_SIGMA, mode='nearest')
        dens = np.interp(ugrid, centers, hist, left=np.nan, right=np.nan)

    xgrid = 10 ** ugrid if xcfg.get('logx', False) else ugrid
    return xgrid, dens, len(uu)

# %% code cell 10
# ============================================================
# Paper figure engines
# ============================================================
def _axis_array(axes):
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)
    return axes


def _plot_curve_with_band(ax, x, y, e, *, color=None, ls='-', lw=LINEWIDTH_MAIN, label=None, alpha=ERROR_ALPHA, zorder=2):
    """Plot a line with a filled symmetric uncertainty band."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    e = np.asarray(e, dtype=float)
    good_line = np.isfinite(x) & np.isfinite(y)
    good_band = good_line & np.isfinite(e)
    if np.any(good_band):
        ax.fill_between(
            x[good_band], y[good_band] - e[good_band], y[good_band] + e[good_band],
            color=color, alpha=alpha, lw=0, zorder=zorder - 1,
        )
    if np.any(good_line):
        ax.plot(
            x[good_line], y[good_line],
            color=color, ls=ls, lw=lw, label=label,
            solid_capstyle='round', solid_joinstyle='round', zorder=zorder,
        )


def plot_metric_x_atlas(
    row_specs,
    *,
    snap_list=ALL_SNAP_LIST,
    model_flags=flags,
    delta_to_gr=False,
    ylabel_mode='metric',
    outfile='atlas',
    figsize=None,
    min_count=8,
    legend=True,
    ylim=(0.0, 1.0),
    delta_ylim=(-0.25, 0.25),
    show_count=False,
    show_error_band=True,
    error_method=None,
):
    """
    Dense atlas: rows are metric/x combinations, columns are snapshots.

    Alignment uncertainties are shown as filled bands. In delta-to-GR mode, the
    error band is propagated as sqrt(err_model^2 + err_GR^2).
    """
    specs = []
    for item in row_specs:
        if isinstance(item, dict):
            specs.append(item.copy())
        else:
            metric_key, x_key = item
            specs.append({'metric': metric_key, 'x': x_key})

    nrow, ncol = len(specs), len(snap_list)
    if figsize is None:
        figsize = (1.75 * ncol, 1.25 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharey=False, squeeze=False)

    plot_flags = [f for f in model_flags if not (delta_to_gr and f == 'GR')]

    for i, spec in enumerate(specs):
        metric_key = spec['metric']
        x_key = spec['x']
        mcfg = ALIGNMENT_METRICS[metric_key]
        xcfg = X_VARIABLES[x_key]
        edges = bin_edges_from_config(xcfg, nbins=spec.get('nbins', xcfg.get('nbins', 10)))

        for j, snap in enumerate(snap_list):
            ax = axes[i, j]
            format_dense_axis(ax)

            gr_curve = None
            gr_error = None
            if delta_to_gr:
                try:
                    _, gr_curve, gr_error, _ = get_curve('GR', snap, metric_key, x_key, edges=edges, min_count=min_count, error_method=error_method)
                except Exception:
                    gr_curve = None
                    gr_error = None
                ax.axhline(0.0, color='0.30', lw=0.65, ls=':', zorder=0)

            for flag in plot_flags:
                try:
                    xc, yy, ee, nn = get_curve(flag, snap, metric_key, x_key, edges=edges, min_count=min_count, error_method=error_method)
                except Exception:
                    continue
                if delta_to_gr:
                    if gr_curve is None:
                        continue
                    yy = yy - gr_curve
                    if gr_error is not None:
                        ee = np.sqrt(ee**2 + gr_error**2)
                color = FLAG_COLOR.get(flag, None)
                lw = LINEWIDTH_GR if flag == 'GR' else LINEWIDTH_MAIN
                alpha = DELTA_ERROR_ALPHA if delta_to_gr else ERROR_ALPHA
                if show_error_band:
                    _plot_curve_with_band(
                        ax, xc, yy, ee,
                        color=color,
                        ls=FLAG_LS.get(flag, '-'),
                        lw=lw,
                        label=flag,
                        alpha=alpha,
                        zorder=3 if flag == 'GR' else 2,
                    )
                else:
                    ax.plot(xc, yy, color=color, ls=FLAG_LS.get(flag, '-'), lw=lw, label=flag)

            if xcfg.get('logx', False):
                ax.set_xscale('log')
            if xcfg.get('xlim') is not None:
                ax.set_xlim(*xcfg['xlim'])

            this_ylim = spec.get('ylim', delta_ylim if delta_to_gr else ylim)
            if this_ylim is not None:
                ax.set_ylim(*this_ylim)

            if i == 0:
                ax.set_title(rf'$z={zmap[snap]:.2f}$', pad=2)
            if j == 0:
                row_label = spec.get('label', mcfg.get('label', metric_key))
                ylabel = row_label if ylabel_mode == 'short' else mcfg.get('ylabel', row_label)
                if delta_to_gr:
                    ylabel = r'$\Delta$ ' + row_label
                ax.set_ylabel(ylabel)
            if i == nrow - 1:
                ax.set_xlabel(xcfg['label'])
            else:
                ax.set_xticklabels([])

            if show_count and i == 0 and j == 0:
                ax.text(0.02, 0.98, f'N bins', transform=ax.transAxes, va='top', ha='left')

    if legend:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.06, top=0.94, wspace=0.08, hspace=0.08)
    save_fig(fig, outfile)
    return fig, axes


def plot_redshift_evolution(tasks, *, snap_list=ALL_SNAP_LIST, model_flags=flags, outfile='redshift_evolution', figsize=None, error_method=None):
    """Small-multiple redshift evolution summaries with filled uncertainty bands."""
    n = len(tasks)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (3.0 * ncols, 2.1 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    z = np.array([zmap[s] for s in snap_list])
    order = np.argsort(z)
    z_plot = z[order]
    snap_ordered = np.array(snap_list)[order]

    for ax, task in zip(axes, tasks):
        metric_key = task['metric']
        x_key = task.get('x')
        xlim = task.get('xlim')
        label = task.get('label', ALIGNMENT_METRICS[metric_key]['label'])
        for flag in model_flags:
            ys, es = [], []
            for snap in snap_ordered:
                y, e, nobj = scalar_summary(flag, int(snap), metric_key, x_key=x_key, xlim=xlim, error_method=error_method)
                ys.append(y)
                es.append(e)
            ys = np.asarray(ys, dtype=float)
            es = np.asarray(es, dtype=float)
            color = FLAG_COLOR.get(flag, None)
            good = np.isfinite(z_plot) & np.isfinite(ys)
            good_band = good & np.isfinite(es)
            if np.any(good_band):
                ax.fill_between(z_plot[good_band], ys[good_band] - es[good_band], ys[good_band] + es[good_band], color=color, alpha=ERROR_ALPHA, lw=0)
            if np.any(good):
                ax.plot(z_plot[good], ys[good], color=color, lw=LINEWIDTH_GR if flag == 'GR' else LINEWIDTH_MAIN, marker='o', ms=2.7, label=flag)
        ax.set_title(label, pad=2)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\langle |\cos\theta|\rangle$')
        ax.set_ylim(0.0, 1.0)
        ax.invert_xaxis()
        format_dense_axis(ax)
    for ax in axes[len(tasks):]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.09, top=0.88, wspace=0.28, hspace=0.35)
    save_fig(fig, outfile)
    return fig, axes


def plot_alignment_heatmaps(metric_keys, *, snap_list=ALL_SNAP_LIST, model_flags=flags, outfile='alignment_heatmaps', figsize=None):
    """Heatmaps: rows are gravity models, columns are redshifts."""
    n = len(metric_keys)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (2.25 * ncols, 1.95 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    im = None
    for ax, metric_key in zip(axes, metric_keys):
        data = np.full((len(model_flags), len(snap_list)), np.nan)
        for i, flag in enumerate(model_flags):
            for j, snap in enumerate(snap_list):
                data[i, j], _, _ = scalar_summary(flag, snap, metric_key)
        im = ax.imshow(data, aspect='auto', origin='upper', vmin=0.0, vmax=1.0)
        ax.set_title(ALIGNMENT_METRICS[metric_key]['label'], pad=2)
        ax.set_xticks(np.arange(len(snap_list)))
        ax.set_xticklabels([f'{zmap[s]:.2f}' for s in snap_list], rotation=45, ha='right')
        ax.set_yticks(np.arange(len(model_flags)))
        ax.set_yticklabels(model_flags)
        ax.set_xlabel(r'$z$')
        ax.tick_params(length=0, pad=1)
    for ax in axes[len(metric_keys):]:
        ax.axis('off')
    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:len(metric_keys)], shrink=0.75, pad=0.01)
        cbar.set_label(r'$\langle |\cos\theta|\rangle$')
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.12, top=0.90, wspace=0.35, hspace=0.55)
    save_fig(fig, outfile)
    return fig, axes


def plot_property_distribution_atlas(
    prop_keys,
    *,
    sample='satellite',
    err_obj='Star',
    err_limit=0.01,
    snap_list=ALL_SNAP_LIST,
    model_flags=flags,
    outfile='property_distributions',
    figsize=None,
    density_min_count=DENSITY_MIN_COUNT,
):
    """Dense property-distribution atlas drawn as smooth density curves, not histograms."""
    nrow, ncol = len(prop_keys), len(snap_list)
    if figsize is None:
        figsize = (1.75 * ncol, 1.08 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    for i, pkey in enumerate(prop_keys):
        pcfg = PROPERTY_VARIABLES[pkey]
        for j, snap in enumerate(snap_list):
            ax = axes[i, j]
            format_dense_axis(ax)
            ymax = 0.0
            for flag in model_flags:
                try:
                    MA = MAset[flag][f'{snap:03d}']
                    x = pcfg['getter'](MA)
                    this_err_limit = err_limit if pcfg.get('apply_err_cut', True) else None
                    m = sample_mask(MA, sample=sample, err_obj=err_obj, err_limit=this_err_limit)
                    x = np.asarray(x, dtype=float)[m]
                    grid, dens, nobj = smooth_density_curve(x, pcfg, min_count=density_min_count)
                    if len(grid) == 0:
                        continue
                    color = FLAG_COLOR.get(flag)
                    ax.plot(grid, dens, lw=DENSITY_LINEWIDTH if flag != 'GR' else LINEWIDTH_MAIN, color=color, ls=FLAG_LS.get(flag, '-'), label=flag)
                    if np.any(np.isfinite(dens)):
                        ymax = max(ymax, float(np.nanmax(dens)))
                except Exception:
                    continue
            if pcfg.get('logx', False):
                ax.set_xscale('log')
            if pcfg.get('xlim') is not None:
                ax.set_xlim(*pcfg['xlim'])
            if ymax > 0:
                ax.set_ylim(0.0, ymax * 1.08)
            if i == 0:
                ax.set_title(rf'$z={zmap[snap]:.2f}$', pad=2)
            if j == 0:
                ax.set_ylabel(pcfg['label'])
            else:
                ax.set_yticklabels([])
            if i == nrow - 1:
                ax.set_xlabel('value')
            else:
                ax.set_xticklabels([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.055, top=0.94, wspace=0.08, hspace=0.08)
    save_fig(fig, outfile)
    return fig, axes

# %% [markdown] cell 11
# ## Figure definitions The definitions below are intentionally compact. Each list becomes one dense, paper-style multi-panel figure.

# %% code cell 12
# ============================================================
# Curated figure suites
# ============================================================
CORE_MASS_ATLAS = [
    ('CGHA', 'm200_log'),
    ('SGHA', 'm200_log'),
    ('SRA_STAR', 'm200_log'),
    ('SRA_DM', 'm200_log'),
    ('GTA_STAR_GRP', 'm200_log'),
    ('HTA_GRP', 'm200_log'),
]

SATELLITE_RADIAL_TIDAL_ATLAS = [
    ('SRA_STAR', 'r200'),
    ('SRA_DM', 'r200'),
    ('TRA_GRP', 'r200'),
    ('TRA_GR', 'r200'),
    ('TRA_MG', 'r200'),
    ('GTA_STAR_GRP', 'r200'),
    ('GTA_STAR_GR', 'r200'),
    ('GTA_STAR_MG', 'r200'),
    ('HTA_GRP', 'r200'),
    ('HTA_GR', 'r200'),
    ('HTA_MG', 'r200'),
]

CENTRAL_TIDAL_ATLAS = [
    ('CGHA', 'm200_log'),
    ('CGTA_STAR_GRP', 'm200_log'),
    ('CGTA_STAR_GR', 'm200_log'),
    ('CGTA_STAR_MG', 'm200_log'),
    ('CHTA_GRP', 'm200_log'),
]

PHYSICS_ATLAS = [
    {'metric': 'SGHA', 'x': 'kappa_star', 'label': r'SGHA vs $\kappa_{\rm rot,*}$'},
    {'metric': 'SRA_STAR', 'x': 'kappa_star', 'label': r'RA$_*$ vs $\kappa_{\rm rot,*}$'},
    {'metric': 'SRA_STAR', 'x': 'q_star', 'label': r'RA$_*$ vs $q_*$'},
    {'metric': 'SRA_STAR', 'x': 's_star', 'label': r'RA$_*$ vs $s_*$'},
    {'metric': 'SRA_STAR', 'x': 'chi_star', 'label': r'RA$_*$ vs $\chi_*$'},
    {'metric': 'SRA_STAR', 'x': 'tfig_star_torb', 'label': r'RA$_*$ vs $T_{\rm fig,*}/T_{\rm orb}$'},
    {'metric': 'SRA_DM', 'x': 'tfig_dm_torb', 'label': r'RA$_{\rm DM}$ vs $T_{\rm fig,DM}/T_{\rm orb}$'},
    {'metric': 'SGHA', 'x': 'tfig_star_dm', 'label': r'SGHA vs $T_{\rm fig,*}/T_{\rm fig,DM}$'},
    {'metric': 'SRA_STAR', 'x': 'dcosdt_star', 'label': r'RA$_*$ vs $d\cos\theta/dt$'},
    {'metric': 'GTA_STAR_GRP', 'x': 'kappa_star', 'label': r'$*{-}T_{\rm grp}$ vs $\kappa_{\rm rot,*}$'},
]

PROPERTY_ATLAS = [
    'mstar_log', 'm200_log', 'r200',
    'kappa_star', 'kappa_dm',
    'q_star', 's_star', 'chi_star',
    'q_dm', 's_dm', 'chi_dm',
    'cos_err_star', 'cos_err_dm',
    'tfig_star_torb', 'tfig_dm_torb',
]

EVOLUTION_TASKS = [
    {'metric': 'CGHA', 'x': 'm200_log', 'xlim': (12.0, 14.8), 'label': r'CGHA, high $M_{200c}$'},
    {'metric': 'SGHA', 'x': 'm200_log', 'xlim': (12.0, 14.8), 'label': r'SGHA, high $M_{200c}$'},
    {'metric': 'SRA_STAR', 'x': 'r200', 'xlim': (0.05, 0.30), 'label': r'RA$_*$, inner cluster'},
    {'metric': 'SRA_STAR', 'x': 'r200', 'xlim': (0.30, 1.00), 'label': r'RA$_*$, outer cluster'},
    {'metric': 'GTA_STAR_GRP', 'x': 'r200', 'xlim': (0.05, 1.00), 'label': r'Sat. $*{-}T_{\rm grp}$'},
    {'metric': 'HTA_GRP', 'x': 'r200', 'xlim': (0.05, 1.00), 'label': r'Sub. DM$ {-}T_{\rm grp}$'},
]

HEATMAP_METRICS = [
    'CGHA', 'SGHA', 'SRA_STAR', 'SRA_DM',
    'GTA_STAR_GRP', 'GTA_STAR_GR', 'GTA_STAR_MG',
    'HTA_GRP', 'HTA_GR', 'HTA_MG',
    'CGTA_STAR_GRP', 'CGTA_STAR_MG',
]

# %% code cell 13
# ============================================================
# One-command figure production
# ============================================================
def make_all_paper_figures(close=False):
    """
    Generate the full compact figure suite.

    Output files are written to OUTDIR in both PDF and PNG format.
    The function returns a list of created figures so that they remain visible in the notebook.
    """
    figs = []

    # A. Smooth sample/property diagnostics, including axis ratios and cosine-angle errors.
    figs.append(plot_property_distribution_atlas(
        PROPERTY_ATLAS,
        sample='satellite',
        outfile='fig00_satellite_property_distribution_atlas',
        figsize=(12.5, 16.2),
    )[0])

    # B. Core IA versus halo mass.
    figs.append(plot_metric_x_atlas(
        CORE_MASS_ATLAS,
        outfile='fig01_core_alignment_vs_m200_atlas',
        figsize=(12.5, 7.8),
        ylim=(0.35, 1.0),
    )[0])

    # C. Core IA MG-GR difference.
    figs.append(plot_metric_x_atlas(
        CORE_MASS_ATLAS,
        delta_to_gr=True,
        outfile='fig02_core_alignment_vs_m200_delta_to_GR',
        figsize=(12.5, 7.8),
        delta_ylim=(-0.20, 0.20),
    )[0])

    # D. Satellite radial and tidal alignment inside clusters.
    figs.append(plot_metric_x_atlas(
        SATELLITE_RADIAL_TIDAL_ATLAS,
        outfile='fig03_satellite_radial_tidal_alignment_atlas',
        figsize=(12.5, 13.2),
        ylim=(0.25, 1.0),
    )[0])

    # E. MG-GR difference for radial/tidal alignments.
    figs.append(plot_metric_x_atlas(
        SATELLITE_RADIAL_TIDAL_ATLAS,
        delta_to_gr=True,
        outfile='fig04_satellite_radial_tidal_delta_to_GR',
        figsize=(12.5, 13.2),
        delta_ylim=(-0.25, 0.25),
    )[0])

    # F. Central galaxy/halo tidal response.
    figs.append(plot_metric_x_atlas(
        CENTRAL_TIDAL_ATLAS,
        outfile='fig05_central_tidal_alignment_atlas',
        figsize=(12.5, 6.4),
        ylim=(0.25, 1.0),
    )[0])

    # G. Morphology, kinetic support, and rotation-timescale dependence.
    figs.append(plot_metric_x_atlas(
        PHYSICS_ATLAS,
        outfile='fig06_physics_dependence_atlas',
        figsize=(12.5, 12.0),
        ylim=(0.25, 1.0),
    )[0])

    # H. Physics-dependence MG-GR difference.
    figs.append(plot_metric_x_atlas(
        PHYSICS_ATLAS,
        delta_to_gr=True,
        outfile='fig07_physics_dependence_delta_to_GR',
        figsize=(12.5, 12.0),
        delta_ylim=(-0.25, 0.25),
    )[0])

    # I. Redshift evolution in physically interpretable windows.
    figs.append(plot_redshift_evolution(
        EVOLUTION_TASKS,
        outfile='fig08_redshift_evolution_summary',
        figsize=(9.2, 4.6),
    )[0])

    # J. Global metric-by-model-by-redshift heatmaps.
    figs.append(plot_alignment_heatmaps(
        HEATMAP_METRICS,
        outfile='fig09_alignment_model_redshift_heatmaps',
        figsize=(9.5, 6.6),
    )[0])

    if close:
        for fig in figs:
            plt.close(fig)
    return figs

# Run this cell to produce the full figure suite.
figs = make_all_paper_figures(close=False)
print(f'Generated {len(figs)} figures in {OUTDIR.resolve()}')

# %% [markdown] cell 14
# ## Error-band and smoothing controls The most useful controls are in the **Selection masks, binned statistics, error bands, and smooth densities** cell: ```python ERROR_METHOD = 'sem' # or 'bootstrap' N_BOOTSTRAP = 300 # used only for bootstrap ERROR_ALPHA = 0.16 # alignment uncertainty band transparency LINEWIDTH_MAIN = 1.35 # non-GR model line width LINEWIDTH_GR = 1.65 # GR reference line width DENSITY_GRID_SIZE = 256 # smooth distribution resolution DENSITY_MIN_COUNT = 12 # minimum objects re

# %% [markdown] cell 15
# ## How to add a new paper figure Add one entry to `ALIGNMENT_METRICS` if the new diagnostic changes the **alignment definition**, or add one entry to `X_VARIABLES` if it only changes the **horizontal axis**. Then append `('YOUR_METRIC', 'YOUR_X')` to one of the atlas lists and rerun `make_all_paper_figures()`. This keeps all figure style, panel layout, binning, output path and model comparison logic centralized.
