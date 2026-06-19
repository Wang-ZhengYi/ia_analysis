"""Exported code from notebooks/raw_20260618/full.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% code cell 1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import h5py
# import .Mesh
# import powers

from catalog_loader import CSCatalog

# %% code cell 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import h5py
import os
# from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
from collections import defaultdict
from IPython.display import HTML
from IPython.display import Video
from mpl_toolkits.mplot3d import Axes3D
import illustris_python as il
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from numba import njit,set_num_threads
from illustris_python import groupcat, snapshot
from collections import Counter

import pyccl as ccl

import pickle
import importlib


import Iana
importlib.reload(Iana)
from Iana import *

import shape
importlib.reload(shape)
from shape import *
import arts 
importlib.reload(arts )
from arts import *
from tidal_field import *

# %% code cell 3
from functools import partial

# %% code cell 4
plt.clf()
fig = plt.figure(figsize = (18,12))
gs = fig.add_gridspec(2,2, hspace=0.2, wspace=0.2)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax00 = plt.subplot(gs[0,0])
visualize_galaxy_system(ax=ax00, components=['central'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Central galaxy-Halo")

ax01 = plt.subplot(gs[0,1])

visualize_galaxy_system(ax=ax01, components=['satellite','subhalo','satellite_axis','subhalo_axis'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Subhalo")


ax10 = plt.subplot(gs[1,0])
visualize_galaxy_system(ax=ax10, components=['position_vector','satellite','satellite_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Postion")



ax11 = plt.subplot(gs[1,1])
visualize_galaxy_system(ax=ax11, components=['position_vector','subhalo','subhalo_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Subhalo-Postion")


fig.savefig('./plots/IA_sketch.png')

# %% code cell 5
def plot_cosMA(data,ax=None,title=''):
    mu_sym=np.concatenate([data, -data])
    # mu_sym=data.copy()
    fit_res = dw.fit(mu_sym)
    print(fit_res)
    xs = np.linspace(-1,1,300)
    pdfs=dw._pdf( xs, fit_res['kappa'])
    pdf_p=dw._pdf( xs, fit_res['kappa']+fit_res['kappa_error'])
    pdf_m=dw._pdf( xs, fit_res['kappa']-fit_res['kappa_error'])
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    ax.plot(xs,pdfs)
    ax.hist(mu_sym,bins=50,density=True,histtype='step')
    ax.fill_between(xs,pdf_p,pdf_m,color='red',alpha=0.4)
    ax.set(xlim=(-1,1),title=title)

# %% code cell 6
plt.rcParams['animation.embed_limit'] = 100

# %% code cell 7
clist=['#c02c38','#c2c116','#3c9566','#1177b0','#ff7c38','#be8936','#e03e36','#b80d57','#700961','#11659a','#abcdef','#fedcba']

DH=['#A73D30','#C16355','#D77E73','#F0D0C6',
    '#0C52B5','#387CBC','#5F81C2','#79B9DC',
    '#81521D','#C1823E','#DAB25B','#E9D077',
    '#305937','#718A70','#68A270','#8FC198',]

# %% code cell 8
get_colors(clist)

# %% code cell 9
get_colors(DH)

# %% code cell 10
FLAG_COLOR = {
    "F40": '#abcdef',
    "F45": '#79B9DC',
    "F50": '#5F81C2',
    "F55": '#687CBC',
    "F60": '#0C52B5',
    "GR": "k",
}

# %% code cell 11
# zmap = {
#     0: 3.00, 1: 2.00, 2: 1.36, 3: 1.26, 4: 1.15, 5: 1.06,
#     6: 0.97, 7: 0.88, 8: 0.80, 9: 0.73, 10: 0.65, 11: 0.58,
#     12: 0.51, 13: 0.45, 14: 0.39, 15: 0.33, 16: 0.27, 17: 0.21,
#     18: 0.16, 19: 0.10, 20: 0.05, 21: 0.00,
# }
# Map snapshot number to redshift
zmap = {1: 2.00,
        6: 0.97,
        8: 0.80, 
        12: 0.51,
        15: 0.33, 
            18: 0.16, 21: 0.00,
        }

# Model sequence to plot
flags = ['F40', 'F45', 'F50', 'F55', 'F60', 'GR']
res={}
for cosmo_flag in flags:
    res[cosmo_flag]={}
    for snaps in zmap.keys():
        res[cosmo_flag][f'{snaps:03d}']=h5py.File(f'/cosma8/data/dp203/dc-wang17/MG_global/L302_N1136_{cosmo_flag}_s{snaps:03d}.hdf5','r')

# %% code cell 12


def mkMA(cosmo_flag,snap):
    
    the_res = res[cosmo_flag][f'{snap:03d}']
    MA = {}

    # ------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------
    
    for key in ['CenID', 'GroupID', 'Group_M_Crit200', 'Group_M_Crit500', 
                'Group_R_Crit200', 'Group_R_Crit500', 'SubhaloBHMass',
                'SubhaloBHMdot', 'SubhaloGasMetallicity', 'SubhaloID', 'SubhaloMass', 
                'SubhaloMassInRadType', 'SubhaloSFR', 'SubhaloVmax', 
                'SubhaloWindMass']:
        if key in the_res:
            MA[key] = the_res[key]
            
    MA["R_over_R_200c"] = np.linalg.norm(the_res['pos_rel'],axis=1)/the_res['Group_R_Crit200']


    
    for ob in ["DM", "Star"]:
        chis = chiSO(the_res[ob]['I'])
        MA[f"chi_{ob}"] = chis['chi']
        MA[f"q_{ob}"] = chis['q']
        MA[f"s_{ob}"] = chis['s']
        MA[f'kappa_rot_{ob}']=the_res[ob]['kappa_rot'][:]
        MA[f"omega_{ob}"] = omega_fig(the_res[ob]['I'][:],the_res[ob]['dI'][:])
        MA[f'cos_err_max_{ob}']        = np.min(the_res[f'{ob}']['cos_err'][:],axis=1)
        MA[f'axe_err_max_{ob}']        = np.min(the_res[f'{ob}']['axis_relerr'][:],axis=1)
        

    MA['R']           = the_res['pos_rel'][:]
    MA['V']           = the_res['vel_rel'][:]
    MA['I_Star']      = the_res['Star']['I'][:]
    MA['I_DM']        = the_res['DM']['I'][:]
    MA['T_grp']       = the_res['Tidal_grp'][:]
    MA['T_GR']        = the_res['Tidal_tot'][:]
    MA['T_MG']        = the_res['Tidal_tot'][:]+the_res['Tidal_tot_mg'][:]
    
    
    return MA

# %% code cell 13
MAset={}
for flag in tqdm(flags):
    MAset[flag]={}
    for sn in tqdm(zmap.keys()):
        MAset[flag][f'{sn:03d}']=mkMA(cosmo_flag=flag,snap=sn).copy()

# %% code cell 14
def materialize_h5(obj):
    """Recursively convert h5py objects into plain Python / NumPy objects."""
    if isinstance(obj, h5py.Dataset):
        return obj[()]  
    elif isinstance(obj, h5py.Group):
        return {k: materialize_h5(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: materialize_h5(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [materialize_h5(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(materialize_h5(v) for v in obj)
    elif isinstance(obj, set):
        return {materialize_h5(v) for v in obj}
    else:
        return obj

# %% code cell 15
MAset_plain = materialize_h5(MAset)

# %% code cell 16
with open('/cosma8/data/dp203/dc-wang17/MG_global/MArenew.pkl', 'wb') as fs:
    pickle.dump(MAset_plain, fs, protocol=pickle.HIGHEST_PROTOCOL)

# %% code cell 17
# with open('/cosma8/data/dp203/dc-wang17/MG_global/MArenew.pkl', 'rb') as f:
#     MAset = pickle.load(f)

# %% code cell 18
MAset.keys()

# %% code cell 19
# M_star=MAset[flag][f'{snaps:03d}']['SubhaloMassInRadType'][:,4]
# len(M_star),

# %% code cell 20
# len(M_star[:])

# %% code cell 21
from DWE import DimrothWatson

import matplotlib as mpl
def mu_to_kappa(mu):
    """
    Convert alignment strength mu to kappa using the convention in DWE.py:
        mu = -2 * arctan(kappa) / pi
    so the inverse is
        kappa = -tan(pi * mu / 2)
    """
    mu = np.asarray(mu, dtype=float)
    return -np.tan(0.5 * np.pi * mu)


def cos_to_deg(x):
    """
    Convert cos(theta) to theta in degrees.
    """
    x = np.clip(x, -1.0, 1.0)
    return np.degrees(np.arccos(x))


def deg_to_cos(theta_deg):
    """
    Convert theta in degrees to cos(theta).
    """
    return np.cos(np.radians(theta_deg))


def plot_dwe_family(
    mu_min=-1.0,
    mu_max=1.0,
    n_mu=21,
    nx=1200,
    cmap='cool',
    figsize=(5.2, 4.1),
    lw=1.8,
    ylim=(0.0, 1.5),
    top_axis=True,
    colorbar=True,
    outfile=None,
    dpi=300,
):
    """
    Plot a family of Dimroth-Watson distributions p(cos(theta)) for a range of mu.

    Parameters
    ----------
    mu_min, mu_max : float
        Minimum and maximum alignment strength mu.
    n_mu : int
        Number of curves.
    nx : int
        Number of x-grid points in cos(theta).
    cmap : str or Colormap
        Matplotlib colormap name or object.
    figsize : tuple
        Figure size.
    lw : float
        Line width.
    ylim : tuple
        y-axis limits.
    top_axis : bool
        Whether to show the top axis in degrees.
    colorbar : bool
        Whether to show the colorbar for mu.
    outfile : str or None
        If not None, save the figure to this file.
    dpi : int
        Save dpi.
    """
    x = np.linspace(-1.0, 1.0, nx)
    mu_vals = np.linspace(mu_min, mu_max, n_mu)

    dw = DimrothWatson(name="DimrothWatson")

    cmap_obj = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=mu_min, vmax=mu_max)

    fig, ax = plt.subplots(figsize=figsize)

    for mu in mu_vals:
        # Avoid divergence exactly at |mu| = 1
        mu_safe = np.clip(mu, -0.999999, 0.999999)
        kappa = mu_to_kappa(mu_safe)
        y = dw.pdf(x, kappa)
        ax.plot(x, y, lw=lw, color=cmap_obj(norm(mu)))

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\cos(\theta)$ misalignment', fontsize=15)
    ax.set_ylabel(r'$dP/d\cos(\theta)$', fontsize=15)
    ax.tick_params(axis='both', which='major', direction='in', labelsize=12)

    if top_axis:
        secax = ax.secondary_xaxis('top', functions=(cos_to_deg, deg_to_cos))
        secax.set_xticks([0, 45, 90, 135, 180])
        secax.set_xticklabels([
            r'$0^\circ$', r'$45^\circ$', r'$90^\circ$', r'$45^\circ$', r'$0^\circ$'
        ])
        secax.tick_params(axis='x', which='major', direction='in', labelsize=12)

    if colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.03)
        cbar.set_label(r'alignment strength $\mu$', fontsize=14)
        cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()

    if outfile is not None:
        fig.savefig(outfile, dpi=dpi, bbox_inches='tight')

    plt.show()

    return fig, ax

# %% code cell 22
plot_dwe_family(cmap='rainbow', outfile='./plots/misalignment_model.png',dpi=200)

# %% code cell 23

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flat

for ax, snaps in zip(axes, zmap.keys()):
    for ff, flag in enumerate(flags):
        M_star=MAset[flag][f'{snaps:03d}']['SubhaloMassInRadType'][:,4]

        M_star_conv  = M_star[MAset[flag][f'{snaps:03d}']['cos_err_max_Star'][:]<=0.01]
        ax.hist(
            np.log10(M_star)+10,
            bins=50,
            density=True,
            histtype='step',
            color=clist[ff + 1],
            label=flag
        )
        ax.hist(
            np.log10(M_star_conv)+10,
            bins=50,
            density=True,ls=':',
            histtype='step',
            color=clist[ff + 1],
            label=flag+' converged'
        )

    ax.set_title(rf'$z = {zmap[snaps]}$')
    ax.set_xlim(9.5, 12)
    # ax.set_ylim(0, 5)
    # ax.semilogx()

for ax in axes[2:]:
    ax.set_xlabel(r'$lg(M_*/M_\odot)$')

# for ax in [axes[0], axes[2]]:
#     ax.set_ylabel('png',dpi=200)

axes[0].legend(ncol=2, frameon=False)
plt.tight_layout()


plt.show()

# %% code cell 24

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flat

for ax, snaps in zip(axes, zmap.keys()):
    for ff, flag in enumerate(flags):
        M_grp=MAset[flag][f'{snaps:03d}']['Group_M_Crit200'][:]
        ax.hist(
            np.log10(M_grp[M_grp>0])+10,
            bins=50,
            density=True,
            histtype='step',
            color=clist[ff + 1],
            label=flag
        )

    ax.set_title(rf'$z = {zmap[snaps]}$')
    ax.set_xlim(9.5, 15)
    # ax.set_ylim(0, 5)
    # ax.semilogx()

for ax in axes[2:]:
    ax.set_xlabel(r'$lg(M_{\rm 200c})$')


axes[0].legend(ncol=2, frameon=False)
plt.tight_layout()


plt.show()

# %% code cell 25

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flat

for ax, snaps in zip(axes, zmap.keys()):
    for ff, flag in enumerate(flags):
        M_DM=MAset[flag][f'{snaps:03d}']['SubhaloMass'][:]
        mask=(M_DM>0)&(MAset[flag][f'{snaps:03d}']['cos_err_max_DM'][:]<=0.01)
        M_DM_conv  = M_DM[mask]
        ax.hist(
            np.log10(M_DM[(M_DM>0)])+10,
            bins=50,
            density=True,
            histtype='step',
            color=clist[ff + 1],
            label=flag
        )
        ax.hist(
            np.log10(M_DM_conv)+10,
            bins=50,
            density=True,ls=':',
            histtype='step',
            color=clist[ff + 1],
            label=flag+' converged'
        )

    ax.set_title(rf'$z = {zmap[snaps]}$')
    ax.set_xlim(9.5, 13)
    ax.set_ylim(0, 1.2)
    # ax.semilogx()

for ax in axes[2:]:
    ax.set_xlabel(r'$lg(M_{DM}/M_\odot)$')


axes[0].legend(ncol=2, frameon=False)
plt.tight_layout()


plt.show()

# %% code cell 26


sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flat

for ax, snaps in zip(axes, zmap.keys()):
    for ff, flag in enumerate(flags):
        chi_star=MAset[flag][f'{snaps:03d}']['chi_Star']
        chi_valid = chi_star[chi_star > -0.99]
        chi_conv  = chi_star[(chi_star > -0.99)&(MAset[flag][f'{snaps:03d}']['cos_err_max_Star'][:]<=0.01)]
        ax.hist(
            chi_valid,
            bins=50,
            density=True,
            histtype='step',
            color=clist[ff + 1],
            label=flag
        )
        ax.hist(
            chi_conv,
            bins=50,
            density=True,ls=':',
            histtype='step',
            color=clist[ff + 1],
            label=flag+' converged'
        )

    ax.set_title(rf'$z = {zmap[snaps]}$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 5)
    ax.text(-1.0, 4.5, 'Prolate', fontsize=12)
    ax.text(-0.1, 4.5, 'Spheroid', fontsize=12)
    ax.text(0.8, 4.5, 'Oblate', fontsize=12)

for ax in axes[2:]:
    ax.set_xlabel(r'$\chi_*$')


axes[0].legend(ncol=2, frameon=False)
plt.tight_layout()


plt.show()

# %% code cell 27

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flat

for ax, snaps in zip(axes, zmap.keys()):
    for ff, flag in enumerate(flags):
        axe_err=MAset[flag][f'{snaps:03d}']['cos_err_max_Star'][:]

        ax.hist(
            np.log10(axe_err),
            bins=50,
            density=True,
            histtype='step',
            color=clist[ff + 1],
            label=flag
        )


    ax.set_title(rf'$z = {zmap[snaps]}$')
    # ax.set_xlim(-4, 2)
    # ax.semilogx()
    # ax.set_ylim(0, 5)

for ax in axes[2:]:
    ax.set_xlabel(r'$\lg(\sigma_a/a)$')


axes[0].legend(ncol=2, frameon=False)
plt.tight_layout()


plt.show()

# %% code cell 28

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flat

for ax, snaps in zip(axes, zmap.keys()):
    for ff, flag in enumerate(flags):
        axe_err=MAset[flag][f'{snaps:03d}']['cos_err_max_Star'][:]

        ax.hist(
            np.log10(axe_err),
            bins=50,
            density=True,
            histtype='step',
            color=clist[ff + 1],
            label=flag
        )


    ax.set_title(rf'$z = {zmap[snaps]}$')
    ax.set_xlim(-8, 2)
    # ax.semilogx()
    # ax.set_ylim(0, 5)

for ax in axes[2:]:
    ax.set_xlabel(r'$\lg(\sigma_\mu)$')


axes[0].legend(ncol=2, frameon=False)
plt.tight_layout()


plt.show()

# %% code cell 29


sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 12), sharex=True, sharey=True)
axes = axes.flatten()  


for idx, (ax, flag) in enumerate(zip(axes, flags)):
    
    ax.scatter(MAset[flag]['021']['kappa_rot_Star'], MAset[flag]['021']['chi_Star'], s=1)
    

    ax.set_ylabel(r'$\chi_*$')
    ax.set_xlabel(r'$\kappa_{\rm rot}$')
    ax.set_ylim(-0.99, 1)
    ax.set_title(flag)
    

if len(flags) < len(axes):
    axes[len(flags)].axis('off')

plt.tight_layout()

plt.show()

# %% code cell 30


sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()  


for idx, (ax, flag) in enumerate(zip(axes, flags)):
    
    ax.scatter(MAset[flag]['021']['kappa_rot_DM'], MAset[flag]['021']['chi_DM'], s=1)
    

    ax.set_ylabel(r'$\chi_{DM}$')
    ax.set_xlabel(r'$\kappa_{\rm rot,DM}$')
    ax.set_ylim(-0.99, 1)
    ax.set_title(flag)
    

if len(flags) < len(axes):
    axes[len(flags)].axis('off')

plt.tight_layout()

plt.show()

# %% code cell 31


sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()  


for idx, (ax, flag) in enumerate(zip(axes, flags)):
    M_star=MAset[flag][f'{snaps:03d}']['SubhaloMassInRadType'][:,4]
    cos_err=MAset[flag][f'{snaps:03d}']['cos_err_max_Star'][:]
    ax.scatter(M_star*1e10,cos_err, s=1)
    

    ax.set_xlabel(r'$lg(M_*/M_\odot)$')
    ax.set_ylabel(r'$\lg(\sigma_\mu)$')
    # ax.set_ylim(-0.99, 1)
    ax.loglog()
    ax.set_title(flag)
    

if len(flags) < len(axes):
    axes[len(flags)].axis('off')

plt.tight_layout()

plt.show()

# %% code cell 32


sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()  


for idx, (ax, flag) in enumerate(zip(axes, flags)):
    chi_star=MAset[flag][f'{snaps:03d}']['chi_Star']

    mask = (chi_star > -2)#(MAset[flag][f'{snaps:03d}']['axe_err_max_Star']<=0.01)&
    # ax.scatter(MAset[flag]['021']['q_Star'][mask], MAset[flag]['021']['s_Star'][mask], s=1,color='orange',alpha=0.3)
    ax.scatter(MAset[flag]['021']['q_Star'], MAset[flag]['021']['s_Star'], s=1,alpha=0.3)
    ax.set_xlabel(r'$q_*$',fontsize=15)
    ax.set_ylabel(r'$s_*$',fontsize=15)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title(flag)
    

if len(flags) < len(axes):
    axes[len(flags)].axis('off')

plt.tight_layout()

plt.show()

# %% code cell 33


sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()  


for idx, (ax, flag) in enumerate(zip(axes, flags)):
    chi_DM=MAset[flag][f'{snaps:03d}']['chi_DM']
# (MAset[flag][f'{snaps:03d}']['axe_err_max_DM']<=0.01)&
    mask = (chi_DM > -1)
    # ax.scatter(MAset[flag]['021']['q_DM'][mask], MAset[flag]['021']['s_DM'][mask],color='orange', s=1,alpha=0.3)
    ax.scatter(MAset[flag]['021']['q_DM'], MAset[flag]['021']['s_DM'], s=1,alpha=0.3)
    
    

    ax.set_ylabel(r'$s_{DM}$')
    ax.set_xlabel(r'$q_{DM}$')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title(flag)
    

if len(flags) < len(axes):
    axes[len(flags)].axis('off')

plt.tight_layout()

plt.show()

# %% code cell 34

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(
    ax=ax0,
    components=['central'],
    misalignment_angle=30,
    size_factor=1,
    show_dashed_axis=True,
    title="Cen.- Halo"
)


snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])
        M_star = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II(MA['I_Star'], MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M_star) + 10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\lg M_*$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=12.5,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0., 1.0)


#

plt.tight_layout()
fig.savefig('./plots/CGHA_Mstar.png')
plt.show()

# %% code cell 35

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(
    ax=ax0,
    components=['central'],
    misalignment_angle=30,
    size_factor=1,
    show_dashed_axis=True,
    title="Cen.- Halo"
)


snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])&(axe_err<0.01)
        M_DM = MA['SubhaloMassInRadType'][:, 1]
        mu_com = II(MA['I_Star'], MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M_DM) + 10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\lg M_{\rm DM}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=12.5,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0., 1.0)


#

plt.tight_layout()
fig.savefig('./plots/CGHA_MDM.png')
plt.show()

# %% code cell 36

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(
    ax=ax0,
    components=['central'],
    misalignment_angle=30,
    size_factor=1,
    show_dashed_axis=True,
    title="Cen.- Halo"
)


snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])&(axe_err<0.01)
        M_grp = MA['Group_M_Crit200'][:]
        mu_com = II(MA['I_Star'], MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M_grp) + 10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\lg M_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=14.5,
            Halo_info_dlim=12,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0.4, 1.0)


#

plt.tight_layout()
fig.savefig('./plots/CGHA_Mgrp.png')
plt.show()

# %% code cell 37

# %% code cell 38

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['subhalo','satellite','subhalo_axis','satellite_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Sat.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M_star = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II(MA['I_Star'], MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M_star) + 10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\lg M_*$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=12.,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(10, 12)


#

plt.tight_layout()
fig.savefig('./plots/SGHA_Mstar.png',dpi=200)
plt.show()

# %% code cell 39

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['subhalo','satellite','subhalo_axis','satellite_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Sat.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M_DM = MA['SubhaloMassInRadType'][:, 1]
        mu_com = II(MA['I_Star'], MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M_DM) + 10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\lg M_{\rm DM}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=12,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(10, 12)


#

plt.tight_layout()
fig.savefig('./plots/SGHA_MDM.png')
plt.show()

# %% code cell 40

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','satellite','satellite_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Pos.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)
        M_star = MA['SubhaloMassInRadType'][:, 4]
        mu_com = VI( MA['R'],MA['I_Star'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M_star) + 10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\lg M_*$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=11.5,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0., 1.0)
    ax.set_xlim(10., 11.6)


#

plt.tight_layout()
fig.savefig('./plots/GRA_Mstar.png',dpi=200)
plt.show()

# %% code cell 41

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','subhalo','subhalo_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Pos.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)
        M_DM = MA['SubhaloMassInRadType'][:, 1]
        mu_com = VI( MA['R'],MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M_DM ) + 10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\lg M_{\rm DM}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=12.5,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0., 1.0)
    ax.set_xlim(10., 12)


# #

plt.tight_layout()
fig.savefig('./plots/SRA_MDM.png',dpi=200)
plt.show()

# %% code cell 42

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','subhalo','subhalo_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Pos.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])
        # M_DM = MA['SubhaloMassInRadType'][:, 1]
        mu_com = VI( MA['R'],MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=MA['R_over_R_200c'],
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=1.1,
            Halo_info_dlim=0.09,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0.5, 0.8)


# #

plt.tight_layout()
fig.savefig('./plots/SRA_R.png',dpi=200)
plt.show()

# %% code cell 43

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','subvec'],sat_sub_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="total Tidal-Pos.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)
        # M_DM = MA['SubhaloMassInRadType'][:, 1]
        mu_com = VI( MA['R'],MA['T_MG'])

        plot_mu(
            mu_set=mu_com['minor'],
            x_set=MA['R_over_R_200c'],
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=3.1,
            Halo_info_dlim=0.09,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=True,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0.5, 1.0)


# #

plt.tight_layout()
fig.savefig('./plots/TRA_R.png',dpi=200)
plt.show()

# %% code cell 44

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','subvec'],sat_sub_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="GR Tidal-Pos.")


snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)
        # M_DM = MA['SubhaloMassInRadType'][:, 1]
        mu_com = VI( MA['R'],-MA['T_GR'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=MA['R_over_R_200c'],
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=3.1,
            Halo_info_dlim=0.09,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-1, 1.0)


# #

plt.tight_layout()
fig.savefig('./plots/TRA_R_GR.png',dpi=200)
plt.show()

# %% code cell 45

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','subvec'],sat_sub_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Group Tidal-Pos.")


snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)
        # M_DM = MA['SubhaloMassInRadType'][:, 1]
        mu_com = VI( MA['R'],MA['T_grp'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=MA['R_over_R_200c'],
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=3.1,
            Halo_info_dlim=0.001,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-0 ,1.0)


# #

plt.tight_layout()
fig.savefig('./plots/TRA_R_grp.png',dpi=200)
plt.show()

# %% code cell 46

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','satellite','satellite_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sat.-Pos.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        # M_DM = MA['SubhaloMassInRadType'][:, 1]
        mu_com = VI( MA['R'],MA['I_Star'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=MA['R_over_R_200c'],
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=1.1,
            Halo_info_dlim=0.01,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-0., 0.8)


# #

plt.tight_layout()
fig.savefig('./plots/GRA_R.png',dpi=200)
plt.show()

# %% code cell 47

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(
    ax=ax0,
    components=['central'],
    misalignment_angle=30,
    size_factor=1,
    show_dashed_axis=True,
    title="Cen.- Halo"
)


snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])&(axe_err<0.01)
        mu_com = II(MA['I_Star'], MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set='kappa_rot_Star',
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\kappa_{\rm rot,*}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=0.8,
            Halo_info_dlim=0.2,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0., 1)



plt.tight_layout()
fig.savefig('./plots/CGHA_kappa.png',dpi=200)
plt.show()

# %% code cell 48

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['subhalo','satellite','subhalo_axis','satellite_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Sat.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)
        mu_com = II(MA['I_Star'], MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set='kappa_rot_DM',
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\kappa_{\rm rot,DM}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=0.8,
            Halo_info_dlim=0.2,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0.0, 1)



plt.tight_layout()
fig.savefig('./plots/SGHA_kappa.png',dpi=200)
plt.show()

# %% code cell 49

# %% code cell 50

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','satellite','satellite_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Pos.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)
        # M_DM = MA['SubhaloMassInRadType'][:, 1]


        ev_star = VI(MA['V'][:],MA['I_Star'][:])
        wr = VV(MA['omega_Star'][:],MA['R'][:])

        dcosMAdt = ev_star['major'][:]+wr[:]
        mu_com = VI(MA['R'][:],MA['I_Star'][:])
        plot_mu(
            mu_set=mu_com['major'],
            x_set=dcosMAdt,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$d \cos\theta_{\rm MA}/d t [Gyr^{-1}]$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=2,
            Halo_info_dlim=-2,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0.1, 0.7)


# #

plt.tight_layout()
fig.savefig('./plots/GRA_dcosMAdt.png',dpi=200)
plt.show()

# %% code cell 51

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','satellite','satellite_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sat.-Pos.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)
        # M_DM = MA['SubhaloMassInRadType'][:, 1]

        o_orb_Star=np.cross(MA['R'][:], MA['V'][:])/np.sum(MA['R'][:]**2, axis=1, keepdims=True)
        t_orb_Star = 2*np.pi/np.linalg.norm(o_orb_Star,axis=1)
        t_fig_Star = 2*np.pi/np.linalg.norm(MA['omega_Star'][:],axis=1)
        mu_com = VI(MA['R'][:],MA['I_Star'][:])
        
        plot_mu(
            mu_set=mu_com['major'],
            x_set=t_fig_Star/t_orb_Star,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$T_{\rm fig,*}/T_{\rm orb}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=100,
            Halo_info_dlim=0.001,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=True,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-0.1, 0.6)


# #

plt.tight_layout()
fig.savefig('./plots/GRA_fig.png',dpi=200)
plt.show()

# %% code cell 52

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['position_vector','subhalo','subhalo_axis'],
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Pos.")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)
        # M_DM = MA['SubhaloMassInRadType'][:, 1]

        o_orb_DM=np.cross(MA['R'][:], MA['V'][:])/np.sum(MA['R'][:]**2, axis=1, keepdims=True)
        t_orb_DM = 2*np.pi/np.linalg.norm(o_orb_DM,axis=1)
        t_fig_DM = 2*np.pi/np.linalg.norm(MA['omega_DM'][:],axis=1)
        mu_com = VI(MA['R'][:],MA['I_DM'][:])
        
        plot_mu(
            mu_set=mu_com['major'],
            x_set=t_fig_DM/t_orb_DM,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$T_{\rm fig,DM}/T_{\rm orb}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=100,
            Halo_info_dlim=0.001,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=True,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-0.3, 0.8)


# #

plt.tight_layout()
fig.savefig('./plots/HRA_fig.png',dpi=200)
plt.show()

# %% code cell 53

# fig = plt.figure(figsize=(20, 10))
# gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

# sns.set(style='ticks')
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['mathtext.fontset'] = 'cm'

# snap_list=[6, 12, 15, 18, 21]
# ax0 = fig.add_subplot(gs[0, 0])
# visualize_galaxy_system(ax=ax0, components=['subhalo','satellite','subhalo_axis','satellite_axis'],
#                         misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
#                            title="Sub.-Sat.")

# snap_sorted = sorted(snap_list[:5])

# panel_pos = [
#     (0, 1), (0, 2),
#     (1, 0), (1, 1), (1, 2)
# ]

# for (r, c), snap in zip(panel_pos, snap_sorted):
#     ax = fig.add_subplot(gs[r, c])

#     for flag in flags:
#         MA = MAset[flag][f'{snap:03d}']
#         axe_err=MA['cos_err_max_Star'][:]
#         sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.01)

#         t_fig_Star= 2*np.pi/np.linalg.norm(MA['omega_Star'][:],axis=1)
#         t_fig_DM = 2*np.pi/np.linalg.norm(MA['omega_DM'][:],axis=1)

        
#         mu_com = VI(MA['R'][:],MA['I_DM'][:])

        
#         mu_com = II(MA['I_Star'], MA['I_DM'])

#         plot_mu(
#             mu_set=mu_com['major'],
#             x_set=t_fig_Star/t_fig_DM,
#             muind=sat_ind_MA,
#             galaxy_halo_MA=MA,
#             bins=10,
#             xlabel=r'$T_{\rm fig,*}/T_{\rm fig, DM}$',
#             title=rf'$z={zmap[snap]:.2f}$',
#             color=FLAG_COLOR[flag],
#             Halo_info_ulim=0.8,
#             Halo_info_dlim=0.2,
#             ax=ax,
#             label=flag,
#             fmt='-',
#             make_plot=True,
#             return_eb=False,
#             logx=False,
#             logy=False,
#             in_plot='mu',ncol=2
#         )

#         del MA

#     ax.set_ylim(0.4, 1)



# plt.tight_layout()
# fig.savefig('./plots/SGHA_omega.png',dpi=200)
# plt.show()

# %% code cell 54

# fig = plt.figure(figsize=(20, 10))
# gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

# sns.set(style='ticks')
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['mathtext.fontset'] = 'cm'

# snap_list=[6, 12, 15, 18, 21]
# ax0 = fig.add_subplot(gs[0, 0])
# visualize_galaxy_system(
#     ax=ax0,
#     components=['central'],
#     misalignment_angle=30,
#     size_factor=1,
#     show_dashed_axis=True,
#     title="Cen.- Halo"
# )


# snap_sorted = sorted(snap_list[:5])

# panel_pos = [
#     (0, 1), (0, 2),
#     (1, 0), (1, 1), (1, 2)
# ]

# for (r, c), snap in zip(panel_pos, snap_sorted):
#     ax = fig.add_subplot(gs[r, c])

#     for flag in flags:
#         MA = MAset[flag][f'{snap:03d}']
#         axe_err=MA['cos_err_max_Star'][:]
#         sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])&(axe_err<0.01)

#         t_fig_Star= 2*np.pi/np.linalg.norm(MA['omega_Star'][:],axis=1)
#         t_fig_DM = 2*np.pi/np.linalg.norm(MA['omega_DM'][:],axis=1)

        
#         mu_com = VI(MA['R'][:],MA['I_DM'][:])

        
#         mu_com = II(MA['I_Star'], MA['I_DM'])

#         plot_mu(
#             mu_set=mu_com['major'],
#             x_set=t_fig_Star/t_fig_DM,
#             muind=sat_ind_MA,
#             galaxy_halo_MA=MA,
#             bins=10,
#             xlabel=r'$T_{\rm fig,*}/T_{\rm fig, DM}$',
#             title=rf'$z={zmap[snap]:.2f}$',
#             color=FLAG_COLOR[flag],
#             Halo_info_ulim=0.8,
#             Halo_info_dlim=0.2,
#             ax=ax,
#             label=flag,
#             fmt='-',
#             make_plot=True,
#             return_eb=False,
#             logx=False,
#             logy=False,
#             in_plot='mu',ncol=2
#         )

#         del MA

#     ax.set_ylim(0.4, 1)



# plt.tight_layout()
# fig.savefig('./plots/CGHA_omega.png')
# plt.show()

# %% code cell 55

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['satellite','subvec'], sat_sub_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sat.-GR Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( -MA['T_GR'],MA['I_Star'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=M,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=1.1,
            Halo_info_dlim=0.09,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-1., 1)


# #

plt.tight_layout()
fig.savefig('./plots/GTA_GR_Mstar.png',dpi=200)
plt.show()

# %% code cell 56

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['satellite','subvec'], sat_sub_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sat.-Total Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( -MA['T_MG'],MA['I_Star'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=M,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=1.1,
            Halo_info_dlim=0.09,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-1., 1)


# #

plt.tight_layout()
fig.savefig('./plots/GTA_MG_Mstar.png',dpi=200)
plt.show()

# %% code cell 57

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['satellite','subvec'], sat_sub_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Gorup Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( MA['T_grp'],MA['I_Star'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=M,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=1.1,
            Halo_info_dlim=0.09,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-1., 1)


# #

plt.tight_layout()
fig.savefig('./plots/GTA_grp_Mstar.png',dpi=200)
plt.show()

# %% code cell 58

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['subhalo','subvec'], sat_sub_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-Group Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( MA['T_grp'],MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=M,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=1.5,
            Halo_info_dlim=0.1,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-0., 1)


# #

plt.tight_layout()
fig.savefig('./plots/HTA_grp_Mstar.png',dpi=200)
plt.show()

# %% code cell 59

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['subhalo','subvec'], sat_sub_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-GR Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( -MA['T_GR'],MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=M,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=1.5,
            Halo_info_dlim=0.1,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-0., 1)


# #

plt.tight_layout()
fig.savefig('./plots/HTA_GR_Mstar.png',dpi=200)
plt.show()

# %% code cell 60

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['subhalo','subvec'], sat_sub_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Sub.-MG Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( -MA['T_MG'],MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=M,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$r/r_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=1.5,
            Halo_info_dlim=0.1,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-0., 1)


# #

plt.tight_layout()
fig.savefig('./plots/HTA_MG_Mstar.png',dpi=200)
plt.show()

# %% code cell 61

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['central','cenvec'], cen_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Cen.-GR Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( -MA['T_GR'],MA['I_Star'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=M,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$M_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=12,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-1., 1)


# #

plt.tight_layout()
fig.savefig('./plots/CGTA_GR_Mstar.png',dpi=200)
plt.show()

# %% code cell 62

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['central','cenvec'], cen_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Cen.-MG Tidal")


snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])&(axe_err<0.01)
        M_grp = MA['Group_M_Crit200'][:]
        mu_com = II(MA['I_Star'], -MA['T_MG'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M_grp) + 10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$\lg M_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=13.1,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(0.4, 1.0)


#

plt.tight_layout()
fig.savefig('./plots/CGHA_Mgrp.png')
plt.show()

# %% code cell 63

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['central','cenvec'], cen_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Cen.-g Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( MA['T_grp'],MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=M,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$M_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=12,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-1., 1)


# #

plt.tight_layout()
fig.savefig('./plots/CGTA_GR_Mstar.png',dpi=200)
plt.show()

# %% code cell 64

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['central','cenvec'], cen_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Cen.-Total Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( -MA['T_MG'],MA['I_Star'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=M,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$M_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=1.1,
            Halo_info_dlim=0.09,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-1., 1)


# #

plt.tight_layout()
fig.savefig('./plots/CGTA_MG_Mstar.png',dpi=200)
plt.show()

# %% code cell 65

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['central','cenvec'], cen_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Cen.-Group Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 4]
        mu_com = II( MA['T_grp'],MA['I_Star'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M)+10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$M_{\rm 200c}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=12,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-1., 1)


# #

plt.tight_layout()
fig.savefig('./plots/CGTA_grp_Mstar.png',dpi=200)
plt.show()

# %% code cell 66

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

snap_list=[6, 12, 15, 18, 21]
ax0 = fig.add_subplot(gs[0, 0])
visualize_galaxy_system(ax=ax0, components=['cenvec'], cen_vec=r'$\vec{T}$',
                        misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Halo-Group Tidal")

snap_sorted = sorted(snap_list[:5])

panel_pos = [
    (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

for (r, c), snap in zip(panel_pos, snap_sorted):
    ax = fig.add_subplot(gs[r, c])

    for flag in flags:
        MA = MAset[flag][f'{snap:03d}']
        axe_err=MA['cos_err_max_Star'][:]
        sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
        M = MA['SubhaloMassInRadType'][:, 1]
        mu_com = II( MA['T_grp'],MA['I_DM'])

        plot_mu(
            mu_set=mu_com['major'],
            x_set=np.log10(M)+10,
            muind=sat_ind_MA,
            galaxy_halo_MA=MA,
            bins=10,
            xlabel=r'$M_{\rm 200c, DM}$',
            title=rf'$z={zmap[snap]:.2f}$',
            color=FLAG_COLOR[flag],
            Halo_info_ulim=12,
            Halo_info_dlim=10,
            ax=ax,
            label=flag,
            fmt='-',
            make_plot=True,
            return_eb=False,
            logx=False,
            logy=False,
            in_plot='mu',ncol=2
        )

        del MA

    ax.set_ylim(-1., 1)


# #

plt.tight_layout()
fig.savefig('./plots/CHTA_grp_MDM.png',dpi=200)
plt.show()

# %% code cell 67
zmap

# %% code cell 68
MA = MAset['GR']['021']

R = MA['R'][:]
omega_star=MA['omega_Star'][:]
omega_DM=MA['omega_DM'][:]
Mstar = MA['SubhaloMassInRadType'][:,4]
M_DM  = MA['SubhaloMassInRadType'][:,1]



ev_DM = VI(MA['V'][:],MA['I_DM'][:])
wr_DM = VV(omega_DM ,R)
ev_star = VI(MA['V'][:],MA['I_Star'][:])
wr_star = VV(omega_star,R)
sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])

# %% code cell 69
plt.scatter(np.linalg.norm(omega_star[sat_ind_MA],axis=1),np.linalg.norm(omega_DM[sat_ind_MA],axis=1),s=0.1)
plt.xlabel(r'$\omega_*$')
plt.ylabel(r'$\omega_{\rm DM}$')
plt.title(r'$z=0$')
plt.loglog()

# %% code cell 70
plt.scatter(np.linalg.norm(omega_star[sat_ind_MA],axis=1),np.linalg.norm(omega_DM[sat_ind_MA],axis=1),s=0.1)
plt.xlabel(r'$\omega_*$')
plt.ylabel(r'$\omega_{\rm DM}$')
plt.title(r'$z=0$')
plt.loglog()

# %% code cell 71
Mstar[sat_ind_MA]/M_DM[sat_ind_MA]

# %% code cell 72
omega_star[sat_ind_MA]/omega_DM[sat_ind_MA]

# %% code cell 73

# %% code cell 74
MA.keys()

# %% code cell 75
plt.scatter(np.log10(MA['Group_M_Crit200'][sat_ind_MA])+10,ev_star['major'][sat_ind_MA]+wr_star[sat_ind_MA],s=0.1)
plt.xlim(10,12)

# %% code cell 76
plt.scatter(np.log10(MA['SubhaloMassInRadType'][sat_ind_MA,4])+10,ev_star['major'][sat_ind_MA]+wr_star[sat_ind_MA],s=0.1)
plt.xlim(10,12)

# %% code cell 77
plt.scatter(MA['kappa_rot_DM'][sat_ind_MA],ev_DM['major'][sat_ind_MA]+wr_DM[sat_ind_MA],s=0.1)
plt.xlim(0,1)

# %% code cell 78
R.shape

# %% code cell 79
plt.scatter(MA['kappa_rot_Star'][sat_ind_MA],ev_star['major'][sat_ind_MA]+wr_star[sat_ind_MA],s=0.1)
plt.xlim(0,1)

# %% code cell 80
plt.scatter(MA['kappa_rot_Star'][sat_ind_MA],1/np.linalg.norm(omega_star[sat_ind_MA],axis=1),s=0.1)
plt.xlim(0,1)
plt.semilogy()

# %% code cell 81
plt.scatter(np.log10(MA['SubhaloMassInRadType'][sat_ind_MA,4])+10,1/np.linalg.norm(omega_star[sat_ind_MA],axis=1),s=0.1)
plt.xlim(10,12)
plt.semilogy()

# %% code cell 82
o_orb_Star=np.cross(MA['R'][:], MA['V'][:])/np.sum(R**2, axis=1, keepdims=True)
t_orb_Star = 2*np.pi/np.linalg.norm(o_orb_Star,axis=1)
t_fig_Star = 2*np.pi/np.linalg.norm(MA['omega_Star'][:],axis=1)
plt.scatter(np.log10(MA['Group_M_Crit200'][:])+10,t_fig_Star/t_orb_Star,s=0.1)
plt.xlim(10,12.5)
plt.semilogy()

# %% code cell 83
np.cross(MA['R'][:], MA['V'][:])

# %% code cell 84
np.linalg.norm(MA['R'][:],axis=1)**2

# %% code cell 85
plt.hist(np.linalg.norm(R,axis=1),bins=10,density=True)

# %% code cell 86
np.linalg.norm(R,axis=1)

# %% code cell 87
MA['R_over_R_200c'][sat_ind_MA]-np.linalg.norm(R[sat_ind_MA],axis=1)/MA['Group_R_Crit200'][sat_ind_MA]

# %% code cell 88
np.linalg.norm(R[sat_ind_MA],axis=1)/MA['Group_R_Crit200'][sat_ind_MA]

# %% code cell 89
MA['R_over_R_200c'][sat_ind_MA]

# %% code cell 90
def plot_MA_list(tidal_type,obj_type,x_set,Halo_info_ulim=1.1,Halo_info_dlim=0.01,ylim=(-1., 1)):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)
    
    sns.set(style='ticks')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['mathtext.fontset'] = 'cm'
    
    snap_list=[6, 12, 15, 18, 21]
    ax0 = fig.add_subplot(gs[0, 0])

    
    visualize_galaxy_system(ax=ax0, components=[obj_type,'subvec'], sat_sub_vec=r'$\vec{T}$',
                            misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                               title="Sub.-Gorup Tidal")
    
    snap_sorted = sorted(snap_list[:5])
    
    panel_pos = [
        (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2)
    ]

    if obj_type=='sateliite' or obj_type=='central':
        I_type = 'I_Star'
        mass_ptype = 4
    else:
        I_type = 'I_DM'
        mass_ptype = 1
    
    for (r, c), snap in zip(panel_pos, snap_sorted):
        ax = fig.add_subplot(gs[r, c])
    
        for flag in flags:

            if obj_type=='sateliite': 
                sat_ind_MA = (MA['SubhaloID'][:] != MA['CenID'][:])&(axe_err<0.1)
            else:
                sat_ind_MA = (MA['SubhaloID'][:] == MA['CenID'][:])&(axe_err<0.1)
            MA = MAset[flag][f'{snap:03d}']
            
            M = MA['SubhaloMassInRadType'][:, mass_ptype]
            mu_com = II( MA[tidal_type],MA[I_type])
    
            plot_mu(
                mu_set=mu_com['major'],
                x_set=M,
                muind=sat_ind_MA,
                galaxy_halo_MA=MA,
                bins=10,
                xlabel=r'$M_{}$'.format(),
                title=rf'$z={zmap[snap]:.2f}$',
                color=FLAG_COLOR[flag],
                Halo_info_ulim=Halo_info_ulim,
                Halo_info_dlim=Halo_info_ulim,
                ax=ax,
                label=flag,
                fmt='-',
                make_plot=True,
                return_eb=False,
                logx=False,
                logy=False,
                in_plot='mu',ncol=2
            )
    
            del MA
    
        ax.set(ylim=ylim)
    
    
    # #
    
    plt.tight_layout()
    fig.savefig('./plots/GTA_grp_Mstar.png',dpi=200)
    plt.show()

# %% code cell 91

# %% code cell 92

# %% code cell 93

# %% code cell 94
