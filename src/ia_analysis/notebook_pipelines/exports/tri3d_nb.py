"""Exported code from notebooks/raw_20260618/tri3D.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% code cell 1
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
# from numba import njit,set_num_threads
from illustris_python import groupcat, snapshot
from collections import Counter

# %% code cell 2
import pyccl as ccl

import pickle
import shape
import importlib
importlib.reload(shape)
from shape import *
from Iana import *
from arts import *
from tidal_orbit import *
from tidal_field import *

# %% code cell 3
from functools import partial

# %% code cell 4

# %% code cell 5
plt.rcParams['animation.embed_limit'] = 100

# %% code cell 6
clist=['#c02c38','#c2c116','#3c9566','#1177b0','#ff7c38','#bec936','#e03e36','#b80d57','#700961','#11659a','#abcdef','#fedcba']

DH=['#A73D30','#C16355','#D77E73','#F0D0C6',
    '#0C52B5','#387CBC','#5F81C2','#79B9DC',
    '#81521D','#C1823E','#DAB25B','#E9D077',
    '#305937','#718A70','#68A270','#8FC198',]

# %% code cell 7
get_colors(clist)

# %% code cell 8
get_colors(DH)

# %% code cell 9
# PartType0 - GAS
# PartType1 - DM
# PartType2 - (unused)
# PartType3 - TRACERS
# PartType4 - STARS & WIND PARTICLES
# PartType5 - BLACK HOLES

# %% code cell 10
sc_cat = h5py.File('./stellar/stellar_circs.hdf5','r')

# %% code cell 11
h = 0.6774

# %% code cell 12
SubHaloIDs=np.loadtxt('subhalo_cutout.txt')

# %% code cell 13
cosmo = ccl.Cosmology(
    Omega_c=0.26, Omega_b=0.04, h=0.6774, n_s=0.96, sigma8=0.81)

# %% code cell 14
pre_info =open('galaxy_pre_info.pkl','rb')
galaxy_pre_info=pickle.load(pre_info )

# %% code cell 15
galaxy_pre_info.keys()

# %% code cell 16
gala_pro = h5py.File('./merged_tidal_I_dI_s099_m20.hdf5','r')

# %% code cell 17
gala_pro.keys()

# %% code cell 18
gala_pro['tidal_grp']

# %% code cell 19
from collections import Counter
count_HaloIDs = Counter(galaxy_pre_info['CenSubHaloID'])

# %% code cell 20
gal_tidal=h5py.File('./ALL_groups_tidal_s099_m20.hdf5','r')

# %% code cell 21
gal_tidal.keys()

# %% code cell 22
gal_tidal['tidal_tot'][:]

# %% code cell 23
N_gal = len(gal_tidal['CenSubhaloID'][:])
gal_tidal_abc={}
gal_tidal_abc['tidal_abc_grp']=np.zeros((N_gal,3))
gal_tidal_abc['tidal_abc_tot']=np.zeros((N_gal,3))
gal_tidal_abc['tidal_major_grp']=np.zeros((N_gal,3))
gal_tidal_abc['tidal_major_tot']=np.zeros((N_gal,3))
gal_tidal_abc['tidal_medium_grp']=np.zeros((N_gal,3))
gal_tidal_abc['tidal_medium_tot']=np.zeros((N_gal,3))
gal_tidal_abc['tidal_minor_grp']=np.zeros((N_gal,3))
gal_tidal_abc['tidal_minor_tot']=np.zeros((N_gal,3))


for ii in tqdm(range(N_gal)):
    T_grp=compute_hessian_matrix(gal_tidal['tidal_grp'][ii])
    T_tot=compute_hessian_matrix(gal_tidal['tidal_tot'][ii])
    grp_abc,grp_3axes=compute_axis(T_grp)
    tot_abc,tot_3axes=compute_axis(T_tot)

    gal_tidal_abc['tidal_abc_grp'][ii] =np.array([grp_abc['a'],grp_abc['b'],grp_abc['c']])
    gal_tidal_abc['tidal_abc_tot'][ii] =np.array([tot_abc['a'],tot_abc['b'],tot_abc['c']])
    gal_tidal_abc['tidal_major_grp'][ii] =grp_3axes['major']
    gal_tidal_abc['tidal_major_tot'][ii] =tot_3axes['major']
    gal_tidal_abc['tidal_medium_grp'][ii]=grp_3axes['medium']
    gal_tidal_abc['tidal_medium_tot'][ii]=tot_3axes['medium']
    gal_tidal_abc['tidal_minor_grp'][ii] =grp_3axes['minor']
    gal_tidal_abc['tidal_minor_tot'][ii] =tot_3axes['minor']

# %% code cell 24
plt.plot(gal_tidal_abc['tidal_abc_grp'][:,0]/gal_tidal_abc['tidal_abc_tot'][:,0],'+')
plt.semilogy()

# %% code cell 25
T_grp

# %% code cell 26
T_tot

# %% code cell 27
compute_axis(-(T_grp))

# %% code cell 28
compute_axis(T_grp)

# %% code cell 29
def ellipsoidal_ratio(
    r,                   # (N,3) vectors from center to points (global coordinates)
    abc_shape,           # (N,3) shape-only semi-axes (axis ratios): (a,b,c) up to a common scale
    r_over_r_vir,        # (N,) scalar distances divided by virial radius: ||r|| / R_vir
    u_major,             # (N,3) unit vector of major axis
    u_medium,            # (N,3) unit vector of medium axis
    u_minor,             # (N,3) unit vector of minor axis
    *,
    validate=True,       # basic sanity checks
    orthogonalize=False, # per-row Gram–Schmidt if axes are slightly non-orthogonal
    eps=1e-12            # numerical epsilon
):
    """
    Compute the ellipsoidal radius ratio m using 'shape-only' axes (a,b,c) and the
    *distance normalized by the virial radius* r_over_r_vir. No explicit R_vir is needed.

    Theory
    ------
    Equal-volume scaling implies the scaled semi-axes are (A,B,C) = k*(a,b,c), where
        k = R_vir / (a*b*c)^(1/3).
    Then m = sqrt( r^T [U diag(1/A^2,1/B^2,1/C^2) U^T] r ).
    Writing r = (r_over_r_vir * R_vir) * r_hat and substituting k eliminates R_vir:
        m = (a*b*c)^(1/3) * r_over_r_vir * sqrt( r_hat^T [U diag(1/a^2,1/b^2,1/c^2) U^T] r_hat ).

    Parameters
    ----------
    r : (N,3)
        Vectors from center to points in global frame. Only the *direction* is used.
    abc_shape : (N,3)
        Semi-axes for *shape only* (axis ratios). Overall scale is irrelevant but values must be > 0.
    r_over_r_vir : (N,)
        Radial distance divided by the virial radius: ||r|| / R_vir.
    u_major, u_medium, u_minor : (N,3)
        Principal axis direction vectors (global frame). Should be unit and orthogonal.
    validate : bool
        If True, run light checks on inputs.
    orthogonalize : bool
        If True, apply Gram–Schmidt per row to (u_major,u_medium,u_minor) and renormalize.
    eps : float
        Numerical epsilon for guarding divisions and norms.

    Returns
    -------
    m : (N,)
        Ellipsoidal radius ratio. 
        - m == 1 : on the (equal-volume) virial ellipsoid
        - m  < 1 : inside
        - m  > 1 : outside
    abc_over_rvir : (N,3)
        The *scaled* semi-axes expressed in units of R_vir, i.e.
            (A/R_vir, B/R_vir, C/R_vir) = (a,b,c) / (a*b*c)^(1/3).
        Useful for diagnostics/visualization without ever using R_vir explicitly.
    """
    r   = np.asarray(r, dtype=float)
    abc = np.asarray(abc_shape, dtype=float)
    rho = np.asarray(r_over_r_vir, dtype=float).reshape(-1)

    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("r must have shape (N,3)")
    N = r.shape[0]

    def _ensure(arr, name):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape != (N,3):
            raise ValueError(f"{name} must have shape (N,3) matching r")
        return arr

    u1 = _ensure(u_major,  "u_major")
    u2 = _ensure(u_medium, "u_medium")
    u3 = _ensure(u_minor,  "u_minor")
    abc = _ensure(abc, "abc_shape")

    if rho.shape[0] != N:
        raise ValueError("r_over_r_vir must have shape (N,) matching r")

    if validate:
        if np.any(rho < 0):
            raise ValueError("r_over_r_vir must be non-negative.")

    # Normalize axis directions
    def _normalize(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n = np.maximum(n, eps)
        return v / n

    u1 = _normalize(u1)
    u2 = _normalize(u2)
    u3 = _normalize(u3)

    if orthogonalize:
        # Per-row Gram–Schmidt orthogonalization
        def _proj(a, b):
            c = np.einsum("ij,ij->i", a, b)    # (N,)
            return c[:, None] * b
        u2 = _normalize(u2 - _proj(u2, u1))
        u3 = _normalize(u3 - _proj(u3, u1) - _proj(u3, u2))

    # Unit direction of r (use direction only)
    r_norm = np.linalg.norm(r, axis=1)
    # Handle zero-length vectors robustly:
    zero_mask = r_norm < eps
    r_hat = np.zeros_like(r)
    if np.any(~zero_mask):
        r_hat[~zero_mask] = r[~zero_mask] / r_norm[~zero_mask, None]
    # If ||r|| == 0 but r_over_r_vir > 0, input is inconsistent; choose to raise in validate mode.
    if validate and np.any(zero_mask & (rho > eps)):
        raise ValueError("Rows with ||r||≈0 must have r_over_r_vir≈0 for consistency.")

    # Build U (N,3,3) with columns [u1 u2 u3]
    U = np.stack([u1, u2, u3], axis=2)   # (N,3,3)

    # Shape-only inverse-squared axes: diag(1/a^2, 1/b^2, 1/c^2)
    inv_a2b2c2 = 1.0 / np.maximum(abc, eps)**2
    Dinv = np.zeros((N,3,3))
    Dinv[:,0,0] = inv_a2b2c2[:,0]
    Dinv[:,1,1] = inv_a2b2c2[:,1]
    Dinv[:,2,2] = inv_a2b2c2[:,2]

    # Q0 = U diag(1/a^2,1/b^2,1/c^2) U^T  (shape-only metric)
    UD  = np.einsum("nij,njk->nik", U, Dinv)
    Q0  = np.einsum("nij,nkj->nik", UD, U)       # (N,3,3)

    # s_i = r_hat^T Q0 r_hat (dimensionless), vectorized
    Qr  = np.einsum("nij,nj->ni", Q0, r_hat)     # (N,3)
    s   = np.einsum("ni,ni->n", r_hat, Qr)       # (N,)
    s   = np.maximum(s, 0.0)                      # guard tiny negatives

    # Geometric mean g = (abc)^(1/3)
    gmean = (abc[:,0] * abc[:,1] * abc[:,2])**(1.0/3.0)
    gmean = np.maximum(gmean, eps)

    # Final ratio m
    m = gmean * rho * np.sqrt(s)
    # Define (A,B,C)/R_vir = (a,b,c)/gmean for diagnostics
    abc_over_rvir = abc / gmean[:, None]

    # For rows with ||r||==0, force m = 0 when r_over_r_vir≈0
    if np.any(zero_mask):
        z_ok = zero_mask & (rho <= eps)
        m[z_ok] = 0.0

    return m

# %% code cell 30

sub = np.asarray(galaxy_pre_info['SubHaloID'])      # (N,)
cen = np.asarray(galaxy_pre_info['CenSubHaloID'])   # (N,)


def map_central_abc(sub, cen, abc):
    """
    Return abc_main (N,3): for each row i, the abc_dm of its *central* halo.
    A "central" row is defined by SubHaloID == CenSubHaloID.
    
    Parameters
    ----------
    sub : array-like, shape (N,)
        SubHaloID for each row.
    cen : array-like, shape (N,)
        CenSubHaloID for each row.
    abc : array-like, shape (N,3)
        abc_dm for each row.

    Returns
    -------
    abc_main : np.ndarray, shape (N,3), dtype float
        For each i, abc_dm of the row whose SubHaloID equals cen[i]
        (i.e., the group's central). Rows with missing centrals are filled with NaN.
    """
    sub = np.asarray(sub)
    cen = np.asarray(cen)
    abc = np.asarray(abc, dtype=float)

    if abc.ndim != 2 or abc.shape[1] != 3:
        raise ValueError("'abc' must have shape (N,3)")
    if sub.shape[0] != cen.shape[0] or sub.shape[0] != abc.shape[0]:
        raise ValueError("All inputs must have the same length N")

    N = sub.shape[0]

    # 1) Identify central rows: SubHaloID == CenSubHaloID
    is_central = (sub == cen)
    central_ids = sub[is_central]      # (M,)
    central_abc = abc[is_central]      # (M,3)

    # If no centrals exist, return all-NaN
    if central_ids.size == 0:
        return np.full((N, 3), np.nan, dtype=float)

    # 2) De-duplicate centrals (keep first occurrence)
    uniq_ids, first_idx = np.unique(central_ids, return_index=True)
    uniq_abc = central_abc[first_idx]  # (K,3) aligned with uniq_ids

    # 3) Sort unique central IDs for binary search mapping
    order = np.argsort(uniq_ids)
    ids_sorted = uniq_ids[order]       # (K,)
    abc_sorted = uniq_abc[order]       # (K,3)

    # 4) Map each cen[i] to its central abc via searchsorted
    pos = np.searchsorted(ids_sorted, cen)                   # insertion positions
    match = (pos < ids_sorted.size) & (ids_sorted[pos] == cen)  # exact matches

    # 5) Build output: matched rows get their central abc; others are NaN
    abc_main = np.full((N, 3), np.nan, dtype=float)
    abc_main[match] = abc_sorted[pos[match]]

    return abc_main

# %% code cell 31
def get_cosMA(v1, v2):
    """
    Cosine similarity for N pairs of 3D vectors, shape(N,3) each.
    """
    if v1.shape != v2.shape or v1.ndim!=2 or v1.shape[1]!=3:
        raise ValueError("Input arrays must be shape (N,3)")
    dot_products = np.einsum('ij,ij->i', v1, v2)
    norm_v1 = np.sqrt(np.einsum('ij,ij->i', v1, v1))
    norm_v2 = np.sqrt(np.einsum('ij,ij->i', v2, v2))
    epsilon = 1e-8
    zero_mask = (norm_v1 < epsilon) | (norm_v2 < epsilon)
    denominators = norm_v1 * norm_v2 + epsilon
    cos_sim = dot_products / denominators
    cos_sim[zero_mask] = 0
    return np.clip(cos_sim, -1.0, 1.0)

# %% code cell 32
gala_pro.keys()

# %% code cell 33
def mkMA():
    galaxy_halo_MA={}
    abc = np.asarray(gala_pro['abc_dm'])
    galaxy_pre_info['abc_main']=map_central_abc(sub, cen, abc)
    
    
    for copylist in ['SubHaloID', 'HaloID', 'CenSubHaloID', 'R_over_R_vir',"SubhaloVmaxRad","SubhaloVmax",'R_sat2cen',
                     'M_vir_Halo', 'Vel_sat', 'SubhaloVelDisp','SubhaloMassType','SubhaloMass','abc_main']:
        galaxy_halo_MA[copylist]=galaxy_pre_info[copylist].copy()
    
    galaxy_halo_MA['R_L']=get_cosMA(galaxy_pre_info['L_sat_tot'],galaxy_pre_info['R_sat2cen'])
    galaxy_halo_MA['R_Lstar']=get_cosMA(galaxy_pre_info['L_sat_tot'],gala_pro['Lstar'])
    galaxy_halo_MA['R_V']=get_cosMA(galaxy_pre_info['Vel_sat'],galaxy_pre_info['R_sat2cen'])

    for axisax in ['major','medium','minor']:
        galaxy_halo_MA[axisax]=get_cosMA(gala_pro[axisax+'_dm'][:],gala_pro[axisax+'_star'])
        galaxy_halo_MA[axisax+'_main']=map_central_abc(sub, cen, gala_pro[axisax+'_dm'])
        for ob in ['subhalo','galaxy']:
            galaxy_halo_MA['abc_'+ob]=gala_pro['abc_'+ob]
            galaxy_halo_MA['R_'+axisax+'_'+ob]=get_cosMA(gala_pro[axisax+'_'+ob][:],galaxy_pre_info['R_sat2cen'])
            galaxy_halo_MA['R_T_grp_'+axisax]=get_cosMA(galaxy_pre_info['R_sat2cen'][:],gal_tidal_abc['tidal_'+axisax+'_grp'][:])
            galaxy_halo_MA['R_T_tot_'+axisax]=get_cosMA(galaxy_pre_info['R_sat2cen'][:],gal_tidal_abc['tidal_'+axisax+'_tot'][:])
            galaxy_halo_MA['T_T_'+axisax]=get_cosMA(gal_tidal_abc['tidal_'+axisax+'_grp'][:],gal_tidal_abc['tidal_'+axisax+'_tot'][:])
            galaxy_halo_MA['L_'+axisax+'_'+ob]=get_cosMA(gala_pro[axisax+'_'+ob][:],galaxy_pre_info['L_sat_tot'])
            galaxy_halo_MA['T_grp_'+axisax+'_'+ob]=get_cosMA(gala_pro[axisax+'_'+ob][:],gal_tidal_abc['tidal_'+axisax+'_grp'][:])
            galaxy_halo_MA['T_tot_'+axisax+'_'+ob]=get_cosMA(gala_pro[axisax+'_'+ob][:],gal_tidal_abc['tidal_'+axisax+'_tot'][:])
            galaxy_halo_MA['Lstar_'+axisax+'_'+ob]=get_cosMA(gala_pro[axisax+'_'+ob][:],gala_pro['Lstar'])
            galaxy_halo_MA['Ldm_'+axisax+'_'+ob]=get_cosMA(gala_pro[axisax+'_'+ob][:],gala_pro['Ldm'])
            
            galaxy_halo_MA['V_'+axisax+'_'+ob]=get_cosMA(gala_pro[axisax+'_'+ob][:],galaxy_pre_info['Vel_sat'])
            _,_,flat_SO=shape_param_np(gala_pro['abc_'+ob][:,0], gala_pro['abc_'+ob][:,1], gala_pro['abc_'+ob][:,2], return_sorted=False)
            galaxy_halo_MA['flatness_'+ob]=flat_SO
        
    galaxy_halo_MA['S_over_R_vir']=ellipsoidal_ratio(
        r=galaxy_pre_info['R_sat2cen'],            
        abc_shape=galaxy_pre_info['abc_main'],    
        r_over_r_vir=galaxy_pre_info['R_over_R_vir'], 
        u_major=galaxy_halo_MA['major_main'],      
        u_medium=galaxy_halo_MA['medium_main'],     
        u_minor=galaxy_halo_MA['minor_main'],      
        )       
    return galaxy_halo_MA

# %% code cell 34
galaxy_halo_MA=mkMA()

# %% code cell 35

# %% code cell 36
galaxy_halo_MA.keys()

# %% code cell 37
sat_ind=(galaxy_halo_MA['SubHaloID']!=galaxy_halo_MA['CenSubHaloID'])&(galaxy_halo_MA['SubhaloMassType'][:,4]>1)
cen_ind=~sat_ind
inhalo_ind = (galaxy_halo_MA['R_over_R_vir']<1)
outhalo_ind =~inhalo_ind
stellar_mass_8_10 =(galaxy_halo_MA['SubhaloMassType'][:,4]*1e10>8 )&(galaxy_halo_MA['SubhaloMassType'][:,4]*1e10<10)
stellar_mass_10_11=(galaxy_halo_MA['SubhaloMassType'][:,4]*1e10>10)&(galaxy_halo_MA['SubhaloMassType'][:,4]*1e10<11)
stellar_mass_11_12=(galaxy_halo_MA['SubhaloMassType'][:,4]*1e10>11)&(galaxy_halo_MA['SubhaloMassType'][:,4]*1e10<12)

# %% code cell 38
sub_mass_11_115  =(galaxy_halo_MA['SubhaloMass'][:]*1e10/0.6774>10**11  )&(galaxy_halo_MA['SubhaloMass'][:]*1e10/0.6774<10**11.5)
sub_mass_115_12  =(galaxy_halo_MA['SubhaloMass'][:]*1e10/0.6774>10**11.5)&(galaxy_halo_MA['SubhaloMass'][:]*1e10/0.6774<10**12 )
sub_mass_12_13   =(galaxy_halo_MA['SubhaloMass'][:]*1e10/0.6774>10**12  )&(galaxy_halo_MA['SubhaloMass'][:]*1e10/0.6774<10**13)
sub_mass_13_plus =(galaxy_halo_MA['SubhaloMass'][:]*1e10/0.6774>10**13  )

# %% code cell 39
rrvir0001=(galaxy_halo_MA['R_over_R_vir']<0.1)
rrvir0103=(galaxy_halo_MA['R_over_R_vir']>0.1)&(galaxy_halo_MA['R_over_R_vir']<0.3)
rrvir0305=(galaxy_halo_MA['R_over_R_vir']>0.3)&(galaxy_halo_MA['R_over_R_vir']<0.5)
rrvir0508=(galaxy_halo_MA['R_over_R_vir']>0.8)&(galaxy_halo_MA['R_over_R_vir']<0.8)
rrvir0812=(galaxy_halo_MA['R_over_R_vir']>0.8)&(galaxy_halo_MA['R_over_R_vir']<1.2)
rrvir1218=(galaxy_halo_MA['R_over_R_vir']>1.2)&(galaxy_halo_MA['R_over_R_vir']<1.8)
rrvir1899=(galaxy_halo_MA['R_over_R_vir']>1.8)

# %% code cell 40
rrvir_labels = [
    r'$r/r_{\rm vir}<0.1$',r'$0.1<r/r_{\rm vir}<0.3$',
    r'$0.3<r/r_{\rm vir}<0.5$',
    r'$0.5<r/r_{\rm vir}<1.2$',
]

submass_labels=[
    r'$11.5<\lg M_{\rm sub}/M_\odot<12$',
    r'$11.5<\lg M_{\rm sub}/M_\odot<12$',
    r'$12.0<\lg M_{\rm sub}/M_\odot<13.0$',
    r'$\lg M_{\rm sub}/M_\odot>13.0$',
]
rrvir_ind=[
rrvir0001,
rrvir0103, rrvir0305,
(rrvir0508 | rrvir0812)]

submass_inds=[sub_mass_11_115,sub_mass_115_12,sub_mass_12_13,sub_mass_13_plus]

# %% code cell 41
T_fall=np.sqrt(np.linalg.norm(galaxy_pre_info['R_sat2cen'],axis=1)**3/galaxy_pre_info['SubhaloMass']/4.302e-3)*1.022712e-3/0.6774
omega_star_axis=np.linalg.norm(gala_pro['omega_star'][:] * 1.022712e-3,axis=1)
omega_star_vir=np.linalg.norm(gala_pro['omega_star_vir'][:] * 1.022712e-3,axis=1)

# %% code cell 42
omega_kepler = 2*np.pi/T_fall#np.linalg.norm(np.cross(galaxy_pre_info['R_sat2cen'],galaxy_pre_info['Vel_sat']),axis=1)*1.022712e-3/np.linalg.norm(galaxy_pre_info['R_sat2cen'],axis=1)**2

# %% code cell 43

# %% code cell 44
fall2axisrot_full = omega_star_axis/omega_kepler
fall2virrot_full=omega_star_vir/omega_kepler

# %% code cell 45
fall2axisrot=omega_star_axis[sat_ind]/omega_kepler[sat_ind]

# %% code cell 46
len(fall2axisrot)

# %% code cell 47
plt.plot(fall2virrot_full/fall2axisrot_full,'+')
plt.semilogy()

# %% code cell 48
ratiocut=(fall2axisrot_full<5000)&(fall2axisrot_full>0.0001)
cenids = galaxy_pre_info['CenSubHaloID'][ratiocut]
cenmass=np.zeros_like(cenids)

for ii in range(len(galaxy_pre_info['CenSubHaloID'][ratiocut])):
    cenid = cenids[ii]
    try:
        cenmass[ii]=galaxy_pre_info['SubhaloMass'][(galaxy_pre_info['SubHaloID']==cenid)]
    except:
        pass
    
submass=galaxy_pre_info['SubhaloMass'][ratiocut]

# %% code cell 49
cenmass.shape

# %% code cell 50
len(fall2axisrot_full[ratiocut])

# %% code cell 51

cenmass_star=np.zeros_like(cenids)
cenmass_dm  =np.zeros_like(cenids)
for ii in range(len(galaxy_pre_info['CenSubHaloID'][ratiocut])):
    cenid = cenids[ii]
    theind=(galaxy_pre_info['SubHaloID']==cenid)
    try:
        cenmass_star[ii]=galaxy_pre_info['SubhaloMassType'][theind,4]
        cenmass_dm[ii]=galaxy_pre_info['SubhaloMassType'][theind,1]
    except:
        pass
    
submass_star=galaxy_pre_info['SubhaloMassType'][ratiocut,4]
submass_dm  =galaxy_pre_info['SubhaloMassType'][ratiocut,1]

# %% code cell 52
plt.figure(figsize=(8,6))
plt.scatter(galaxy_pre_info['SubhaloMass'][sat_ind],1/fall2virrot_full[sat_ind],s=1)
plt.xlabel('subhalo mass')
plt.ylabel(r'$\tau_{\rm vir}/\hat{\tau}_{\rm orbit}$')
plt.ylim(0.01,1000)
plt.loglog()

# %% code cell 53

rrr = np.linspace(0.01,1500,100)
Rrvir=np.linalg.norm(galaxy_halo_MA['R_sat2cen'][ratiocut],axis=1)/galaxy_halo_MA['R_over_R_vir'][ratiocut]
Rr = np.linalg.norm(galaxy_halo_MA['R_sat2cen'][ratiocut],axis=1)

# %% code cell 54
# # bb,hh=plt.hist((submass_star*Rr**3/np.array(cenmass)/gala_pro['abc_star'][ratiocut,0]**3)**(-0.5),bins=20,log=True)

# # plt.clf()
# # bincenter = 
plt.figure(figsize=(12,8))
plt.scatter((submass_star*Rr**3/np.array(cenmass)/gala_pro['abc_star'][ratiocut,0]**3)**(-0.5)/1.022712e-3*0.6774,
            1/fall2axisrot_full[ratiocut],s=1,alpha=0.5,color=clist[0],label=r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$')

plt.scatter((submass_star*Rr**3/np.array(cenmass)/gala_pro['abc_star'][ratiocut,0]**3)**(-0.5)/1.022712e-3*0.6774,
            1/fall2virrot_full[ratiocut],s=1,alpha=0.5,color=clist[2],label=r'$\tau_{\rm vir}/\hat{\tau}_{\rm orbit}$')
plt.plot(rrr,rrr,color=clist[3],label='y=x')
plt.xlabel(r'$\sqrt{\dfrac{a^3_{\rm gal}}{M_*} /\dfrac{R^3_{\rm vir,grp}}{M_{\rm grp}}}$')
plt.ylabel(r'$\tau_{\rm gal}/\hat{\tau}_{\rm orbit}$')
plt.legend()
plt.ylim(0.01,1500)
plt.xlim(0.01,1500)
plt.loglog()

# %% code cell 55

# %% code cell 56
plt.figure(figsize=(12,8))
plt.scatter(submass_dm,1/fall2axisrot_full[ratiocut],s=1)
plt.xlabel('subhalo mass(DM)')
plt.ylabel(r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$')
plt.ylim(0.001,500)
plt.loglog()

# %% code cell 57
plt.figure(figsize=(12,8))
plt.scatter(submass_star/cenmass_star,1/fall2axisrot_full[ratiocut],s=1)
plt.xlabel('satellite mass(star)/central mass(star)')
plt.ylabel(r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$')
plt.ylim(0.01,5000)
plt.loglog()

# %% code cell 58
plt.figure(figsize=(12,8))
plt.scatter(submass_star/cenmass_star,1/fall2virrot_full[ratiocut],s=1)
plt.xlabel('satellite mass(star)/central mass(star)')
plt.ylabel(r'$\tau_{\rm vir}/\hat{\tau}_{\rm orbit}$')
plt.ylim(0.01,5000)
plt.loglog()

# %% code cell 59
plt.figure(figsize=(12,8))
plt.scatter(submass_star,1/fall2axisrot_full[ratiocut],s=1)
plt.xlabel('satellite mass(star)')
plt.ylabel(r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$')
plt.ylim(0.01,5000)
plt.loglog()

# %% code cell 60
plt.figure(figsize=(12,8))
plt.scatter(submass_star/cenmass,1/fall2axisrot_full[ratiocut],s=1)
plt.xlabel('satellite mass(star)/Halo mass (total)')
plt.ylabel(r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$')
plt.ylim(0.01,5000)
plt.loglog()

# %% code cell 61
len(galaxy_pre_info['CenSubHaloID'][ratiocut])

# %% code cell 62
len(galaxy_pre_info['SubhaloMass'][ratiocut])

# %% code cell 63
rrvir_ind=[
rrvir0001,
(rrvir0103 | rrvir0305),
(rrvir0508 | rrvir0812),
(rrvir1218|rrvir1899)]

# %% code cell 64
# ratiocut=(fall2axisrot_full<500)&(fall2axisrot_full>0.001)
# plt.figure(figsize=(12,8))
# f
# plt.scatter(galaxy_pre_info['SubhaloMass'][ratiocut]*1e10/0.6774,1/fall2axisrot_full[ratiocut],s=1)
# plt.scatter(galaxy_pre_info['SubhaloMass'][ratiocut&rrvircut05]*1e10/0.6774,1/fall2axisrot_full[ratiocut&rrvircut05],s=1,color=clist[0])
# plt.scatter(galaxy_pre_info['SubhaloMass'][ratiocut&rrvircut03]*1e10/0.6774,1/fall2axisrot_full[ratiocut&rrvircut03],s=1,color=clist[2])
# plt.xlabel('subhalo mass'+r"$[M_{\odot}]$")
# plt.ylabel(r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$')
# plt.loglog()

# %% code cell 65
ratiocut=(fall2axisrot<500)&(fall2axisrot>0.001)
plt.scatter(galaxy_pre_info['R_over_R_vir'][sat_ind][ratiocut],1/fall2axisrot[ratiocut],s=1)
plt.xlabel(r"$r/r_{\rm vir}$")
plt.ylabel(r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$')
plt.xlim(0.01,5)
plt.ylim(0.01,1000)
plt.loglog()

# %% code cell 66
plt.scatter(galaxy_pre_info['R_over_R_vir'][sat_ind],1/fall2virrot_full[sat_ind],s=1)
plt.xlabel(r"$r/r_{\rm vir}$")
plt.ylabel(r'$\tau_{\rm vir}/\hat{\tau}_{\rm orbit}$')
plt.xlim(0.01,5)
plt.ylim(0.01,1000)
plt.loglog()

# %% code cell 67
plt.scatter(galaxy_pre_info['R_over_R_vir'],galaxy_halo_MA['flatness_dm'],s=1)
plt.xlabel(r"$r/r_{\rm vir}$")
plt.ylabel('Flatness')
plt.xlim(0.01,5)
plt.ylim(-1,1)
plt.semilogx()

# %% code cell 68
plt.scatter(1/fall2axisrot_full[sat_ind],1/fall2virrot_full[sat_ind],s=1)
plt.xlabel(r"$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$")
plt.ylabel(r'$\tau_{\rm vir}/\hat{\tau}_{\rm orbit}$')
plt.xlim(0.01,1000)
plt.ylim(0.01,1000)
plt.loglog()

# %% code cell 69
from DWE import DimrothWatson

dw = DimrothWatson()

# %% code cell 70
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

# %% code cell 71
plot_cosMA(data=galaxy_halo_MA['major'],ax=None)

# %% code cell 72
vmax,mu_vmax,muerr_vmax=plot_mu(mu_set='major',x_set='SubhaloVmax',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$V_{\rm max}$',title='',color=clist[0],
        Halo_info_ulim=300,Halo_info_dlim=200,
            make_plot=True,return_eb=True,logx=False,in_plot='mu')

# %% code cell 73

ax=plot_mu(mu_set='R_major_dm',x_set='S_over_R_vir',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$r/r_{\rm vir}$',title='R_major_dm',color=clist[0],
        Halo_info_ulim=2,Halo_info_dlim=0.01,
            make_plot=True,return_eb=False,logx=False,in_plot='mu')
ax=plot_mu(mu_set='R_major_dm',x_set='R_over_R_vir',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$r/r_{\rm vir}$',title='R_major_dm',color=clist[0],
        Halo_info_ulim=2,Halo_info_dlim=0.01,
            make_plot=True,return_eb=False,logx=False,in_plot='mu')
fig=ax.figure
fig.savefig('sub_radial.png')

# %% code cell 74
mu_set_label='R_major_star'
plot_mu(mu_set=mu_set_label,x_set='S_over_R_vir',muind=np.ones_like(sat_ind),
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$r/r_{\rm vir}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=2,Halo_info_dlim=-0.001,
            make_plot=True,return_eb=False,logx=False,in_plot='mu')

# %% code cell 75
mu_set_label='R_major_star'
plot_mu(mu_set=mu_set_label,x_set=1/fall2axisrot_full,muind=sat_ind&(fall2axisrot_full>0.001)&(fall2axisrot_full<500),
        galaxy_halo_MA=galaxy_halo_MA,bins=15,xlabel=r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=20,Halo_info_dlim=0.01,
            make_plot=True,return_eb=False,logx=True,in_plot='mu')

# %% code cell 76
mu_set_label='major'
plot_mu(mu_set=mu_set_label,x_set='S_over_R_vir',muind=np.ones_like(sat_ind),
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$r/r_{\rm vir}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=2,Halo_info_dlim=0.001,
            make_plot=True,return_eb=False,logx=False)

# %% code cell 77
mu_set_label='R_T_tot_major'
plot_mu(mu_set=mu_set_label,x_set='S_over_R_vir',muind=np.ones_like(sat_ind),
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$r/r_{\rm vir}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=1,Halo_info_dlim=-0.5,
            make_plot=True,return_eb=False,logx=False,ylim=(0.4,0.95))

# %% code cell 78
mu_set_label='R_T_grp_major'
plot_mu(mu_set=mu_set_label,x_set='S_over_R_vir',muind=np.ones_like(sat_ind),
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$r/r_{\rm vir}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=1,Halo_info_dlim=-0.5,
            make_plot=True,return_eb=False,logx=False,ylim=(0.4,1))

# %% code cell 79

# %% code cell 80
mu_set_label='T_T_major'
plot_mu(mu_set=mu_set_label,x_set='S_over_R_vir',muind=np.ones_like(sat_ind),
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$r/r_{\rm vir}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=1,Halo_info_dlim=-0.5,
            make_plot=True,return_eb=False,logx=False,ylim=(0.4,1))

# %% code cell 81
mu_set_label='R_V'
plot_mu(mu_set=mu_set_label,x_set='S_over_R_vir',muind=np.ones_like(sat_ind),
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$r/r_{\rm vir}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=2,Halo_info_dlim=-0.5,
            make_plot=True,return_eb=False,logx=False)

# %% code cell 82
mu_set_label='R_V'
plot_mu(mu_set=mu_set_label,x_set='SubhaloVmax',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$V_{\rm max}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=300,Halo_info_dlim=160,
            make_plot=True,return_eb=False,logx=False)

# %% code cell 83
mu_set_label='R_major_dm'
plot_mu(mu_set=mu_set_label,x_set='SubhaloVmax',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$V_{\rm max}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=300,Halo_info_dlim=160,
            make_plot=True,return_eb=False,logx=False,ylim=(0.7,1))

# %% code cell 84
mu_set_label='R_major_star'
plot_mu(mu_set=mu_set_label,x_set='SubhaloVmax',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$V_{\rm max}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=300,Halo_info_dlim=160,
            make_plot=True,return_eb=False,logx=False)

# %% code cell 85
plt.scatter(galaxy_halo_MA['SubhaloMassType'][sat_ind,4],galaxy_halo_MA['SubhaloVmax'][sat_ind],s=1)
plt.xlabel(r'$M_*$')
plt.ylabel(r'$V_{\rm max}$')
plt.xlim(1,20)
plt.ylim(100,500)
plt.loglog()

# %% code cell 86
plt.scatter(galaxy_halo_MA['SubhaloMassType'][sat_ind,4],galaxy_halo_MA['SubhaloMassType'][sat_ind,1],s=1)
plt.xlim(1,20)
plt.ylim(3,1000)
plt.xlabel(r'$M_*$')
plt.ylabel(r'$M_{\rm subhalo}$')
plt.loglog()

# %% code cell 87
from arts import KMeans2DClusterer

# %% code cell 88
ba_sub=galaxy_halo_MA['abc_dm'][:,1]/galaxy_halo_MA['abc_dm'][:,0]
ca_sub=galaxy_halo_MA['abc_dm'][:,2]/galaxy_halo_MA['abc_dm'][:,0]

ba_gal=galaxy_halo_MA['abc_star'][:,1]/galaxy_halo_MA['abc_star'][:,0]
ca_gal=galaxy_halo_MA['abc_star'][:,2]/galaxy_halo_MA['abc_star'][:,0]

valid_abc = (ba_sub<1)&(ca_sub<1)&(ba_gal<1)&(ca_gal<1)&(ba_sub>0)&(ca_sub>0)&(ba_gal>0)&(ca_gal>0)

km_abc_sub = KMeans2DClusterer(n_clusters=30, random_state=42).fit(ba_sub[valid_abc], ca_sub[valid_abc])
centers_abc_sub = km_abc_sub.get_centers()        
labels_abc_sub  = km_abc_sub.get_numeric_labels() 
masks_abc_sub   = km_abc_sub.get_boolean_index()  

km_abc_gal = KMeans2DClusterer(n_clusters=30, random_state=42).fit(ba_gal[valid_abc], ca_gal[valid_abc])
centers_abc_gal = km_abc_gal.get_centers()        
labels_abc_gal  = km_abc_gal.get_numeric_labels() 
masks_abc_gal   = km_abc_gal.get_boolean_index()

# %% code cell 89
plt.clf()
fig = plt.figure(figsize = (18,10))
gs = fig.add_gridspec(1,2, hspace=0.2, wspace=0.2)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax00 = plt.subplot(gs[0,0])
ax00.scatter(ba_sub[valid_abc],ca_sub[valid_abc], c=labels_abc_sub, s=1, cmap="tab10", alpha=0.4)
# ax00.scatter(centers_abc_sub[:,0], centers_abc_sub[:,1], c="k", s=80, marker="x", label="centers")
km_abc_sub.plot_boundaries(ax00, mode="hull",   linestyle="--",  linewidth=1.5, color="k", alpha=0.7)  

ax00.legend()
ax00.set_xlabel(r'$b/a$'); ax.set_ylabel(r'$c/a$'); ax.set_title("subhalo")
ax00.grid(True)
ax00.set(xlim=(0,1),ylim=(0,1))

ax01 = plt.subplot(gs[0,1])
ax01.scatter(ba_gal[valid_abc],ca_gal[valid_abc], c=labels_abc_gal, s=1, cmap="tab10", alpha=0.4)
# ax01.scatter(centers_abc_gal[:,0], centers_abc_gal[:,1], c="k", s=80, marker="x", label="centers")
km_abc_gal.plot_boundaries(ax01, mode="hull",   linestyle="--",  linewidth=1.5, color="k", alpha=0.7)  

ax01.legend()
ax01.set_xlabel(r'$b/a$'); ax.set_ylabel(r'$c/a$'); ax.set_title("galaxy")
ax01.grid(True)
ax01.set(xlim=(0,1),ylim=(0,1))

# %% code cell 90
from matplotlib.colors import Normalize, TwoSlopeNorm

def scatter_by_cluster_value(
    ax,
    x, y, labels,
    cluster_values,
    *,
    cmap='bwr',            # continuous colormap (e.g., 'bwr','viridis',...)
    vmin=None, vmax=None,  # color range; default from cluster_values
    center=None,           # if not None, use TwoSlopeNorm with vcenter=center
    s=6, alpha=0.6,
    edgecolors='none',
    cbar_label='cluster value',
    cbar_kwargs=None
):
    """
    Plot (x,y) and color each point by the value assigned to its cluster.
    All points in the same cluster share the same color (continuous colormap).

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : (N,) arrays of coordinates
    labels : (N,) int labels in [0..K-1]
    cluster_values : (K,) array of scalar values, one per cluster
    cmap : str or Colormap
        Continuous colormap used to map cluster_values to colors.
    vmin, vmax : float or None
        Color scale bounds. Defaults to min/max of cluster_values.
    center : float or None
        If set, uses TwoSlopeNorm(vmin, center, vmax) to center the colormap.
    s, alpha, edgecolors : scatter styling
    cbar_label : str
        Label for the colorbar.
    cbar_kwargs : dict
        Extra kwargs forwarded to plt.colorbar.

    Returns
    -------
    sc : PathCollection (the scatter)
    cb : Colorbar
    norm : Normalize (for consistent scaling across plots)
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    labels = np.asarray(labels, int).ravel()
    cv = np.asarray(cluster_values, float).ravel()
    if not (x.size == y.size == labels.size):
        raise ValueError("x, y, labels must have the same length")
    if cv.ndim != 1:
        raise ValueError("cluster_values must be a 1D array of length K")

    K = cv.size
    if labels.min() < 0 or labels.max() >= K:
        raise ValueError("labels contain indices outside [0, K-1] for given cluster_values")

    # Map each point's color value = value of its cluster
    v_point = cv[labels]

    # Build color normalization (from cluster_values, not from points, so legend reflects cluster scale)
    if vmin is None: vmin = float(np.nanmin(cv))
    if vmax is None: vmax = float(np.nanmax(cv))
    if center is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=float(center), vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Scatter with continuous colormap (all points in a cluster have same v_point)
    sc = ax.scatter(x, y, c=v_point, cmap=cmap, norm=norm, s=s, alpha=alpha, edgecolors=edgecolors)

    # Attach colorbar to this axis
    kw = dict(fraction=0.046, pad=0.04)
    if cbar_kwargs:
        kw.update(cbar_kwargs)
    cb = plt.colorbar(sc, ax=ax, **kw)
    if cbar_label:
        cb.set_label(cbar_label)

    return sc, cb, norm

# %% code cell 91
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

def draw_by_cluster_value(
    ax,
    x, y, labels,
    cluster_values,
    *,
    cmap='bwr',
    vmin=None, vmax=None,
    center=None,
    s=6, alpha=0.6,          # polygon face alpha; kept for API compatibility
    edgecolors='none',        # polygon edge color
    cbar_label='cluster value',
    cbar_kwargs=None
):
    """
    Fill each cluster's convex-hull region with a color determined by its cluster value.

    This function does NOT scatter points. For visual boundaries, call your KMeans helper
    (e.g., km.plot_boundaries(ax, mode="hull", ...)) before/after this function.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x, y : array-like of shape (N,)
        Point coordinates.
    labels : array-like of shape (N,)
        Integer cluster labels in [0..K-1] for each point.
    cluster_values : array-like of shape (K,)
        One scalar per cluster. Each cluster k is colored by cluster_values[k].
    cmap : str or Colormap, default 'bwr'
        Colormap used to map values -> colors.
    vmin, vmax : float or None
        Color scale range. Defaults to min/max of cluster_values (ignoring NaNs).
    center : float or None
        If provided, use TwoSlopeNorm(vmin, center, vmax) — useful for diverging maps.
    s : float, default 6
        Unused for points; here reused as polygon alpha default (for backwards compat).
    alpha : float, default 0.6
        Polygon face alpha (transparency).
    edgecolors : color spec or 'none', default 'none'
        Polygon edge color. Keep 'none' if you rely on external boundary lines.
    cbar_label : str, default 'cluster value'
        Colorbar label text. Use '' to hide.
    cbar_kwargs : dict or None
        Extra kwargs forwarded to plt.colorbar.

    Returns
    -------
    patches : list[matplotlib.patches.Polygon]
        The filled hull polygons (clusters with >= 3 unique points).
    cb : matplotlib.colorbar.Colorbar
        Colorbar reflecting the global normalization.
    norm : matplotlib.colors.Normalize
        The normalization used to map values to colors.

    Notes
    -----
    - Convex hulls are computed per cluster using Andrew's monotone chain (no SciPy).
    - If a cluster has < 3 unique points, it is skipped (no filled area).
    - Use the same vmin/vmax/center across subplots to keep colors comparable.
    """
    import numpy as np, matplotlib.pyplot as plt

    # ---- Validate & coerce inputs ----
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    labels = np.asarray(labels, int).ravel()
    cv = np.asarray(cluster_values, float).ravel()

    if not (x.size == y.size == labels.size):
        raise ValueError("x, y, labels must have the same length")
    if cv.ndim != 1:
        raise ValueError("cluster_values must be a 1D array of length K")
    K = cv.size
    if labels.min() < 0 or labels.max() >= K:
        raise ValueError("labels contain indices outside [0, K-1] for given cluster_values")

    # ---- Build normalization from cluster_values (global, consistent scale) ----
    finite_cv = cv[np.isfinite(cv)]
    if vmin is None: vmin = float(np.nanmin(finite_cv)) if finite_cv.size else 0.0
    if vmax is None: vmax = float(np.nanmax(finite_cv)) if finite_cv.size else 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=float(center), vmax=vmax) if center is not None else Normalize(vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    # ---- Convex hull (Andrew's monotone chain; avoids SciPy dependency) ----
    def _hull(pts):
        """Return CCW convex hull vertices; if <3 unique points, return input."""
        pts = np.asarray(pts, float)
        if pts.shape[0] <= 2:
            return pts
        pts = np.unique(pts, axis=0)         # deduplicate
        if pts.shape[0] <= 2:
            return pts

        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

        pts = pts[np.lexsort((pts[:,1], pts[:,0]))]  # sort by x then y
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))
        upper = []
        for p in pts[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))
        return np.array(lower[:-1] + upper[:-1], float)

    # ---- Draw filled hull per cluster, colored by its own value ----
    patches = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size < 3:
            continue  # need >= 3 points to form a polygon
        pts = np.c_[x[idx], y[idx]]
        hull = _hull(pts)
        if hull.shape[0] >= 3:
            # Map this cluster's value to RGBA; NaNs become transparent via alpha=0
            val = float(cv[k])
            face = mapper.to_rgba(val)
            poly = ax.fill(hull[:, 0], hull[:, 1],
                           facecolor=face, edgecolor=edgecolors,
                           alpha=alpha, zorder=0)
            patches.extend(poly)

    # ---- Colorbar (global scale for all clusters) ----
    kw = dict(fraction=0.046, pad=0.04)
    if cbar_kwargs: kw.update(cbar_kwargs)
    cb = plt.colorbar(mapper, ax=ax, **kw)
    if cbar_label:
        cb.set_label(cbar_label)

    return patches, cb, norm

# %% code cell 92
def plot_abc_mu_panels(galaxy_halo_MA, sat_ind, Ksub=30, Kgal=30, vmin=-1, vmax=1, cmap='bwr'):
    import numpy as np, matplotlib.pyplot as plt
    ba_sub = galaxy_halo_MA['abc_dm'][sat_ind,1]/galaxy_halo_MA['abc_dm'][sat_ind,0]
    ca_sub = galaxy_halo_MA['abc_dm'][sat_ind,2]/galaxy_halo_MA['abc_dm'][sat_ind,0]
    ba_gal = galaxy_halo_MA['abc_star'][sat_ind,1]/galaxy_halo_MA['abc_star'][sat_ind,0]
    ca_gal = galaxy_halo_MA['abc_star'][sat_ind,2]/galaxy_halo_MA['abc_star'][sat_ind,0]
    valid = (0<ba_sub)&(ba_sub<1)&(0<ca_sub)&(ca_sub<1)&(0<ba_gal)&(ba_gal<1)&(0<ca_gal)&(ca_gal<1)

    km_sub = KMeans2DClusterer(n_clusters=Ksub, random_state=42).fit(ba_sub[valid], ca_sub[valid])
    km_gal = KMeans2DClusterer(n_clusters=Kgal, random_state=42).fit(ba_gal[valid], ca_gal[valid])
    labs_sub, masks_sub = km_sub.get_numeric_labels(), km_sub.get_boolean_index()
    labs_gal, masks_gal = km_gal.get_numeric_labels(), km_gal.get_boolean_index()

    def muses(side, K, masks):
        Fs=[f'R_major_{side}', f'R_medium_{side}', f'R_minor_{side}']
        M=np.zeros((K,3))
        for k in range(K):
            m=masks[k]
            for j,f in enumerate(Fs):
                a=galaxy_halo_MA[f][sat_ind][valid][m]
                M[k,j]=dw.fit(np.r_[a,-a])['mu']
        return M

    Msub, Mgal = muses('subhalo', Ksub, masks_sub), muses('galaxy', Kgal, masks_gal)

    fig, ax = plt.subplots(3,2, figsize=(18,25), constrained_layout=True)
    data = [
        ('$\\vec{r}$ - Subhalo major',  km_sub, ba_sub, ca_sub, labs_sub, Msub[:,0], 0,0),
        ('$\\vec{r}$ - Subhalo medium', km_sub, ba_sub, ca_sub, labs_sub, Msub[:,1], 1,0),
        ('$\\vec{r}$ - Subhalo minor',  km_sub, ba_sub, ca_sub, labs_sub, Msub[:,2], 2,0),
        ('$\\vec{r}$ - Galaxy major',   km_gal, ba_gal, ca_gal, labs_gal, Mgal[:,0], 0,1),
        ('$\\vec{r}$ - Galaxy medium',  km_gal, ba_gal, ca_gal, labs_gal, Mgal[:,1], 1,1),
        ('$\\vec{r}$ - Galaxy minor',   km_gal, ba_gal, ca_gal, labs_gal, Mgal[:,2], 2,1),
    ]

    for title, km, ba, ca, labs, clv, r, c in data:
        a=ax[r,c]
        km.plot_boundaries(a, mode="hull", linestyle="-", linewidth=1., color="k", alpha=0.7)
        a.set_aspect('equal', adjustable='box')
        a.grid(True)
        a.set(xlim=(0,1), ylim=(0.0001,1))
        a.set_xlabel(r'$b/a$', fontsize=18)
        a.set_ylabel(r'$c/a$', fontsize=18)
        a.set_title(title, fontsize=24)


        # draw_by_cluster_value(a, ba[valid], ca[valid], labs, cluster_values=clv,
        #                       cmap=cmap, vmin=vmin, vmax=vmax, center=None,
        #                       s=6, alpha=0.6, edgecolors='none', cbar_label='', cbar_kwargs=None)
        scatter_by_cluster_value(a, ba[valid], ca[valid], labs, cluster_values=clv,
                              cmap=cmap, vmin=vmin, vmax=vmax, center=None,
                              s=6, alpha=0.6, edgecolors='none', cbar_label='', cbar_kwargs=None)

        T_list=(1.0, 2/3, 1/3, 0.0)
        x = np.linspace(0, 1, 600)
        a.fill_between(x, x, 1, color='grey',alpha=0.3, zorder=0)
        a.plot([0, 1], [0, 0], "k", lw=3); a.plot([1, 1], [0, 1], "k", lw=3)
        a.plot([0, 0], [0, 1], "k", lw=3); a.plot([0, 1], [0, 1], "k", lw=3)
        a.plot(0, 0, "ko", ms=16); a.plot(1, 0, "ko", ms=16); a.plot(1, 1, "ko", ms=16)
        a.text(0.02, 0.08, "Needle", color=clist[2], fontsize=20, rotation=90, va="bottom", weight='bold')
        a.text(0.5, 0.04, "Elliptic Disk", color=clist[2], fontsize=20, ha="center", weight='bold')
        a.text(1.04, 0.5, "Oblate Spheroid", color=clist[2], fontsize=20, rotation=90, va="center", weight='bold')
        a.text(0.48, 0.48, "Prolate Spheroid", color=clist[2], fontsize=20, rotation=45,ha='center',
               va="bottom", weight='bold')
        a.text(0.85, 1.02, "Sphere", color=clist[2], fontsize=20, ha="left", va="bottom", weight='bold')
        a.text(0.80, 0.02, "Circular Disk", color=clist[2], fontsize=20, ha="left", va="bottom", weight='bold')

        s = np.linspace(0, 1, 600)                     
        jlab = int(0.30*(len(s)-1))                    
        for T in T_list:
            q = np.sqrt(np.clip(1.0 - T + T * s**2, 0.0, 1.0))  # x = a2/a1
            a.plot(q, s, ls="--", color='grey', lw=2.5)

            if np.isclose(T, 2/3):
                label = r"$T=2/3$"
            elif np.isclose(T, 1/3):
                label = r"$T=1/3$"
            elif np.isclose(T, 1.0):
                label = r"$T=1$"
            elif np.isclose(T, 0.0):
                label = r"$T=0$"
            else:
                label = rf"$T={T:g}$"


            xl, yl = float(q[jlab]), float(s[jlab])

            xl = min(max(xl, 0.03), 0.97)
            yl = min(max(yl, 0.06), 0.40)       
            rotation=45
            a.text(xl-0.1, yl-0.2, label,rotation=rotation, color='k', fontsize=20)
    return fig, ax

# %% code cell 93
fig, ax=plot_abc_mu_panels(galaxy_halo_MA, sat_ind, Ksub=50, Kgal=30, vmin=-0.8, vmax=0.8, cmap='seismic')
fig.savefig('babc.png')

# %% code cell 94
for i in range(len(submass_inds)):
    plt.figure(figsize=(10,8))
    plt.scatter(galaxy_halo_MA['flatness_dm'][sat_ind&submass_inds[i]],galaxy_halo_MA['flatness_star'][sat_ind&submass_inds[i]],
                s=abs(galaxy_halo_MA['major'][sat_ind&submass_inds[i]])*10,color=clist[i+3],label=submass_labels[i],alpha=0.1)
    plt.xlabel('subhalo flatness')
    plt.ylabel('galaxy flatness')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.title('satellite')
    # plt.loglog()
    plt.legend()

# %% code cell 95

for i in range(len(submass_inds)):
    plt.figure(figsize=(10,8))
    plt.scatter(galaxy_halo_MA['flatness_dm'][cen_ind&submass_inds[i]],galaxy_halo_MA['flatness_star'][cen_ind&submass_inds[i]],
                s=abs(galaxy_halo_MA['major'][cen_ind&submass_inds[i]])*10,color=clist[i+3],label=submass_labels[i],alpha=0.1)
    plt.xlabel('subhalo flatness')
    plt.ylabel('galaxy flatness')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.title('central')
    # plt.loglog()
    plt.legend()

# %% code cell 96
plt.scatter(galaxy_halo_MA['SubhaloMassType'][sat_ind,4],galaxy_halo_MA['flatness_star'][sat_ind],
            s=1,color=clist[1],label='galaxy flatness',alpha=0.4)
plt.scatter(galaxy_halo_MA['SubhaloMassType'][sat_ind,4],galaxy_halo_MA['flatness_dm'][sat_ind],
            s=1,color=clist[4],label='subhalo flatness',alpha=0.4)
plt.xlim(1,10)
plt.ylim(-1,1)
# plt.semilogx()
plt.legend()

# %% code cell 97
galaxy_halo_MA.keys()

# %% code cell 98
mu_set_label='R_major_star'
plot_mu(mu_set=mu_set_label,x_set=galaxy_halo_MA['SubhaloMassType'][:,4],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$10^{10}M_\odot$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=100,Halo_info_dlim=1,
            make_plot=True,return_eb=False,logx=True)

# %% code cell 99
mu_set_label='R_major_dm'
plot_mu(mu_set=mu_set_label,x_set=galaxy_halo_MA['M_vir_Halo'][:],muind=sat_ind&inhalo_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$M_\odot$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=1e15,Halo_info_dlim=1e12,
            make_plot=True,return_eb=False,logx=True,)

# %% code cell 100
mu_set_label='R_major_star'
plot_mu(mu_set=mu_set_label,x_set='flatness_star',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel='flatness_star',title=mu_set_label,color=clist[0],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=True,return_eb=False,logx=False)

# %% code cell 101
mu_set_label='R_minor_star'
plot_mu(mu_set=mu_set_label,x_set='flatness_star',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel='flatness_star',title=mu_set_label,color=clist[0],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=True,return_eb=False,logx=False)

# %% code cell 102
mu_set_label='R_major_star'
plot_mu(mu_set=mu_set_label,x_set='flatness_dm',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness_dm',title=mu_set_label,color=clist[0],
        Halo_info_ulim=0.8,Halo_info_dlim=0.5,
            make_plot=True,return_eb=False,logx=False)

# %% code cell 103
axv=visualize_star_system(ax=None, components=['position_vector','central','satellite','subhalo','subhalo_axis','satellite_axis'], 
                           misalignment_angle=30, size_factor=1, sat_sub_vec=r'$\vec{V}$', 
                           galaxy_color=clist[6], show_dashed_axis=True, 
                           title="Galaxy System Misalignment")
fig=axv.figure
fig.savefig('gsm.png',dpi=300)

# %% code cell 104
# visualize_star_system(ax=None, components=['position_vector','satellite','subvec'], sat_sub_vec=r'$\vec{L}$',
#                            misalignment_angle=30, size_factor=1, 
#                            galaxy_color=clist[6], show_dashed_axis=True, 
#                            title="Galaxy System Misalignment")

# %% code cell 105
# visualize_star_system(ax=None, components=['position_vector','satellite','satellite_axis'], 
#                            misalignment_angle=30, size_factor=1, 
#                            galaxy_color=clist[6], show_dashed_axis=True, 
#                            title="Galaxy System Misalignment")

# %% code cell 106
# visualize_star_system(ax=None, components=['central'], 
#                            misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
#                            title="Galaxy System Misalignment")

# %% code cell 107

# %% code cell 108
# num_particles = 10000
# positions = np.random.rand(num_particles, 3) * 1000
# masses = np.ones(num_particles) * 1e10
# data =compute_gravitational_potential(positions, masses, grid_size=32)

# %% code cell 109
mu_set_label='major'
halomass,cenmu1,cenmuerr1=plot_mu(mu_set=mu_set_label,x_set=galaxy_halo_MA['SubhaloMass']/0.6774,muind=cen_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$10^{10}M_\odot$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=5e4,Halo_info_dlim=10,
            make_plot=False,return_eb=True,logx=True,ax=None)
sshmassratio,cenmu2,cenmuerr2=plot_mu(mu_set=mu_set_label,
        x_set=((galaxy_halo_MA['SubhaloMass'][:]-galaxy_halo_MA['SubhaloMassType'][:,1])/galaxy_halo_MA['SubhaloMassType'][:,1]),
                                      muind=cen_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=20,xlabel=r'$10^{10}M_\odot$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=0.25,Halo_info_dlim=0.01,
            make_plot=False,return_eb=True,logx=False,ax=None)


labels = [
    r'$r/r_{\rm vir}<0.1$',r'$0.1<r/r_{\rm vir}<0.3$',
    r'$0.3<r/r_{\rm vir}<0.5$',
    r'$0.5<r/r_{\rm vir}<1.2$',
]

submass_labels=[
    r'$11<\lg M_{\rm sub}/M_\odot<11.5$',
    r'$11.5<\lg M_{\rm sub}/M_\odot<12$',
    r'$12.0<\lg M_{\rm sub}/M_\odot<13.0$',
    r'$\lg M_{\rm sub}/M_\odot>13.0$',
]
rrvir_ind=[
rrvir0001,
rrvir0103, rrvir0305,
(rrvir0508 | rrvir0812)]

submass_inds=[sub_mass_11_115,sub_mass_115_12,sub_mass_12_13,sub_mass_13_plus]



Halo_info_ulims=[10,5,4,2]
fig = plt.figure(figsize = (18,15))
gs = fig.add_gridspec(2,2, hspace=0.1, wspace=0.1)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax1 = plt.subplot(gs[0,0])

visualize_star_system(ax=ax1, components=['central'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Central galaxy-Halo")
ax2 = plt.subplot(gs[0,1])
ax2.errorbar(halomass*1e10,cenmu1,yerr=cenmuerr1,fmt='-o',color=clist[0])
ax2.set_xlabel(r'$M_{\odot}$',fontsize=15)
ax2.set_ylabel(r'$\mu$',fontsize=15)
ax2.semilogx()
ax2.set_ylim(-0.,1)
ax3 = plt.subplot(gs[1,0])
# ax3.errorbar(sshmassratio,cenmu2,yerr=cenmuerr2,fmt='o',color=clist[0])
plot_mu(mu_set='major',
        x_set=((galaxy_halo_MA['SubhaloMass'][:]-galaxy_halo_MA['SubhaloMassType'][:,1])/galaxy_halo_MA['SubhaloMassType'][:,1]),muind=cen_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[0],
            Halo_info_ulim=0.2,Halo_info_dlim=0.01,fmt='-o',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax3,label="Total"
           )
for ii in range(len(labels)):
    plot_mu(mu_set='major',
        x_set=((galaxy_halo_MA['SubhaloMass'][:]-galaxy_halo_MA['SubhaloMassType'][:,1])/galaxy_halo_MA['SubhaloMassType'][:,1]),muind=cen_ind&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[ii+1],
            Halo_info_ulim=0.2,Halo_info_dlim=0.01,fmt='-o',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax3,label=submass_labels[ii]
           )
ax3.set_ylim(-0.3,1)
ax3.set_xlabel(r'$M_b/M_{\rm DM}$',fontsize=15)
ax3.set_ylabel(r'$\mu$',fontsize=15)
ax3.legend()
# ax3.semilogx()

ax4 = plt.subplot(gs[1,1])
# ax4.errorbar(fg,cenmu3,yerr=cenmuerr3,fmt='o',color=clist[0])

plot_mu(mu_set='minor',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=cen_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[0],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy (minor)',
            make_plot=1,return_eb=True,logx=False,ax=ax4)
plot_mu(mu_set='minor',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=cen_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[2],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo (minor)',
            make_plot=1,return_eb=True,logx=False,ax=ax4)

plot_mu(mu_set='medium',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=cen_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[5],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy (medium)',
            make_plot=1,return_eb=True,logx=False,ax=ax4)

plot_mu(mu_set='medium',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=cen_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[11],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo (medium)',
            make_plot=1,return_eb=True,logx=False,ax=ax4)

plot_mu(mu_set='major',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=cen_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[3],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy (major)',
            make_plot=1,return_eb=True,logx=False,ax=ax4)

plot_mu(mu_set='major',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=cen_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[8],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo (major)',
            make_plot=1,return_eb=True,logx=False,ax=ax4)
ax4.set_ylim(-0.2,1)
ax4.legend()
ax4.set_title('')
ax4.set_xlabel('flatness(galaxy/subhalo)',fontsize=15)
ax4.set_ylabel(r'$\mu$',fontsize=15)
fig.savefig('./plots/central.png')

# %% code cell 110
fig = plt.figure(figsize = (18,25))
gs = fig.add_gridspec(3,2, hspace=0.1, wspace=0.2)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax1 = plt.subplot(gs[0,0])

visualize_star_system(ax=ax1, components=['satellite','subhalo','satellite_axis','subhalo_axis'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Subhalo")



ax2 = plt.subplot(gs[0,1])

plot_mu(mu_set='minor',
        x_set='S_over_R_vir',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[0],
            Halo_info_ulim=2,Halo_info_dlim=0.01,fmt='-o',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax2,label="Total"
           )

for ii in range(len(labels)):
    plot_mu(mu_set='minor',
        x_set='S_over_R_vir',muind=sat_ind&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[ii+1],
            Halo_info_ulim=2,Halo_info_dlim=0.01,fmt='-o',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax2,label=submass_labels[ii]
           )
ax2.legend()
ax2.set_ylim(0.5,1)
ax2.set_xlabel(r'$r/r_{\rm vir}$',fontsize=15)
ax2.set_ylabel(r'$\mu$',fontsize=15)
ax3 = plt.subplot(gs[1,0])
plot_mu(mu_set='major',
            x_set=1/fall2virrot_full,muind=sat_ind&(fall2virrot_full>0.001)&(fall2virrot_full<1000),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title='',color=clist[0],
            Halo_info_ulim=100,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax3,label='Total'
           )
for ii in range(len(submass_labels)-1):
    plot_mu(mu_set='major',
            x_set=1/fall2virrot_full,muind=sat_ind&(fall2virrot_full>0.001)&(fall2virrot_full<1000)&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title='',color=clist[ii+1],
            Halo_info_ulim=100,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax3,label=submass_labels[ii]
           )
ax3.set_xlabel(r'$\tau_{\rm vir}/\hat{\tau}_{\rm orbit}$',fontsize=15)
ax3.set_ylabel(r'$\mu$',fontsize=15)
# ax3.set_title('subhalo major'+'galaxy major',fontsize=15)
ax3.set_ylim(0,1)
ax3.legend()

ax4 = plt.subplot(gs[1,1])
plot_mu(mu_set='major',
            x_set=1/fall2axisrot_full,muind=sat_ind&(fall2axisrot_full>0.001)&(fall2axisrot_full<1000),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title='',color=clist[0],
            Halo_info_ulim=100,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax4,label='Total'
           )
for ii in range(len(submass_labels)-1):
    plot_mu(mu_set='major',
            x_set=1/fall2axisrot_full,muind=sat_ind&(fall2axisrot_full>0.001)&(fall2axisrot_full<1000)&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title='',color=clist[ii+1],
            Halo_info_ulim=100,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax4,label=submass_labels[ii]
           )
ax4.set_xlabel(r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$',fontsize=15)
ax4.set_ylabel(r'$\mu$',fontsize=15)
# ax4.set_title('subhalo major'+'galaxy major',fontsize=15)
ax4.legend()
ax4.semilogx()
ax4.set_ylim(-0.,1)


ax5 = plt.subplot(gs[2,0])
# ax3.errorbar(sshmassratio,cenmu2,yerr=cenmuerr2,fmt='o',color=clist[0])
plot_mu(mu_set='major',
        x_set=((galaxy_halo_MA['SubhaloMass'][:]-galaxy_halo_MA['SubhaloMassType'][:,1])/galaxy_halo_MA['SubhaloMassType'][:,1]),
        muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[0],
            Halo_info_ulim=0.2,Halo_info_dlim=0.01,fmt='o-',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax5,label="Total"
           )
for ii in range(len(labels)):
    plot_mu(mu_set='major',
        x_set=((galaxy_halo_MA['SubhaloMass'][:]-galaxy_halo_MA['SubhaloMassType'][:,1])/galaxy_halo_MA['SubhaloMassType'][:,1]),
            muind=sat_ind&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[ii+1],
            Halo_info_ulim=0.2,Halo_info_dlim=0.01,fmt='o-',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax5,label=submass_labels[ii]
           )
ax5.set_ylim(-0.3,1)
ax5.set_xlabel(r'$M_b/M_{\rm DM}$',fontsize=15)
ax5.set_ylabel(r'$\mu$',fontsize=15)
ax5.legend()
# ax3.semilogx()

ax6 = plt.subplot(gs[2,1])
# ax4.errorbar(fg,cenmu3,yerr=cenmuerr3,fmt='o',color=clist[0])

plot_mu(mu_set='minor',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[0],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy(minor)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)
plot_mu(mu_set='minor',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[2],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo(minor)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)


plot_mu(mu_set='medium',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[5],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy(medium)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)
plot_mu(mu_set='medium',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[11],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo(medium)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)


plot_mu(mu_set='major',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[3],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy(major)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)
plot_mu(mu_set='major',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[8],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo(major)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)

ax6.set_ylim(-0.2,1)
ax6.legend()
ax6.set_title('')
ax6.set_xlabel('flatness(galaxy/subhalo)',fontsize=15)
ax6.set_ylabel(r'$\mu$',fontsize=15)


fig.savefig('./plots/sat_sub.png')

# %% code cell 111


    
    


#===============================================================================================






fig = plt.figure(figsize = (18,25))
gs = fig.add_gridspec(3,2, hspace=0.2, wspace=0.1)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax1 = plt.subplot(gs[0,0])

visualize_star_system(ax=ax1, components=['position_vector','satellite','subvec','satellite_axis'], sat_sub_vec=r'$\vec{V}$',
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Postion")
ax2 = plt.subplot(gs[0,1])
for ii in range(len(submass_labels)):
    plot_mu(mu_set='R_V',x_set='S_over_R_vir',muind=sat_ind&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[ii+1],
            Halo_info_ulim=2,Halo_info_dlim=0.001,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax2,label=submass_labels[ii]
           )
plot_mu(mu_set='R_V',x_set='S_over_R_vir',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=2,Halo_info_dlim=0.001,
            make_plot=1,return_eb=0,logx=False,label='Total',ax=ax2)
ax2.set_xlabel(r'$r/r_{\rm vir}$',fontsize=15)
ax2.set_ylabel(r'$\mu$',fontsize=15)
ax2.legend(frameon=False)
ax2.set_title(r'$\vec{r}-\vec{V}$',fontsize=15)
ax2.set_ylim(0.3,1)
# ax2.semilogx()

ax3 = plt.subplot(gs[1,0])
plot_mu(mu_set='R_major_star',
            x_set=1/fall2virrot_full,muind=sat_ind&(fall2virrot_full>0.001)&(fall2virrot_full<1000),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[0],
            Halo_info_ulim=100,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax3,label='Total'
           )
for ii in range(len(submass_labels)-1):
    plot_mu(mu_set='R_major_star',
            x_set=1/fall2virrot_full,muind=sat_ind&(fall2virrot_full>0.001)&(fall2virrot_full<1000)&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[ii+1],
            Halo_info_ulim=100,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax3,label=submass_labels[ii]
           )
ax3.set_xlabel(r'$\tau_{\rm vir}/\hat{\tau}_{\rm orbit}$',fontsize=15)
ax3.set_ylabel(r'$\mu$',fontsize=15)
ax3.set_title(r'$\vec{r}-$'+'galaxy major',fontsize=15)
ax3.set_ylim(-0.2,1)
ax3.legend()

ax4 = plt.subplot(gs[1,1])
plot_mu(mu_set='R_major_star',
            x_set=1/fall2axisrot_full,muind=sat_ind&(fall2axisrot_full>0.001)&(fall2axisrot_full<1000),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[0],
            Halo_info_ulim=500,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax4,label='Total'
           )
for ii in range(len(submass_labels)-1):
    plot_mu(mu_set='R_major_star',
            x_set=1/fall2axisrot_full,muind=sat_ind&(fall2axisrot_full>0.001)&(fall2axisrot_full<1000)&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[ii+1],
            Halo_info_ulim=500,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax4,label=submass_labels[ii]
           )
ax4.set_xlabel(r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$',fontsize=15)
ax4.set_ylabel(r'$\mu$',fontsize=15)
ax4.set_title(r'$\vec{r}-$'+'galaxy major',fontsize=15)
ax4.legend()
ax4.semilogx()
ax4.set_ylim(-0.5,1)

ax5 = plt.subplot(gs[2,0])
# ax3.errorbar(sshmassratio,cenmu2,yerr=cenmuerr2,fmt='o',color=clist[0])
plot_mu(mu_set='R_major_star',
        x_set='S_over_R_vir',
        muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[0],
            Halo_info_ulim=1,Halo_info_dlim=0.01,fmt='o-',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax5,label="Total"
           )
for ii in range(len(labels)-1):
    plot_mu(mu_set='R_major_star',
        x_set='S_over_R_vir',
            muind=sat_ind&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[ii+1],
            Halo_info_ulim=1,Halo_info_dlim=0.01,fmt='o-',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax5,label=submass_labels[ii]
           )
ax5.set_ylim(-0.3,1)
ax5.set_xlabel(r'$r/r_{\rm vir}$',fontsize=15)
ax5.set_ylabel(r'$\mu$',fontsize=15)
ax5.set_title(r'$\vec{r}-$'+'galaxy major',fontsize=15)
ax5.legend()
# ax3.semilogx()

ax6 = plt.subplot(gs[2,1])
# ax4.errorbar(fg,cenmu3,yerr=cenmuerr3,fmt='o',color=clist[0])

plot_mu(mu_set='R_major_star',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[3],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy(major)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)
plot_mu(mu_set='R_major_star',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[8],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo(major)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)


plot_mu(mu_set='R_minor_star',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[0],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy(minor)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)
plot_mu(mu_set='R_minor_star',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[2],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo(minor)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)



plot_mu(mu_set='R_medium_star',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[5],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy(medium)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)
plot_mu(mu_set='R_medium_star',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[11],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo(medium)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)


ax6.set_ylim(-1,1)
ax6.legend()
ax6.set_title('')
ax6.set_xlabel('flatness(galaxy/subhalo)',fontsize=15)
ax6.set_title(r'$\vec{r}-$'+'galaxy major',fontsize=15)
ax6.set_ylabel(r'$\mu$',fontsize=15)


fig.savefig('./plots/radial_star.png')

# %% code cell 112


    
    


#===============================================================================================






fig = plt.figure(figsize = (18,25))
gs = fig.add_gridspec(3,2, hspace=0.2, wspace=0.1)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax1 = plt.subplot(gs[0,0])

visualize_star_system(ax=ax1, components=['position_vector','subhalo','subhalo_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Subhalo-Postion")
ax2 = plt.subplot(gs[0,1])
for ii in range(len(submass_labels)):
    plot_mu(mu_set='R_major_dm',x_set='S_over_R_vir',muind=sat_ind&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[ii+1],
            Halo_info_ulim=2,Halo_info_dlim=0.001,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax2,label=submass_labels[ii]
           )
plot_mu(mu_set='R_major_dm',x_set='S_over_R_vir',muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=mu_set_label,color=clist[0],
        Halo_info_ulim=2,Halo_info_dlim=0.001,
            make_plot=1,return_eb=0,logx=False,label='Total',ax=ax2)
ax2.set_xlabel(r'$r/r_{\rm vir}$',fontsize=15)
ax2.set_ylabel(r'$\mu$',fontsize=15)
ax2.legend(frameon=False)
ax2.set_title('R_major_dm',fontsize=15)
ax2.set_ylim(0.3,1)
# ax2.semilogx()

ax3 = plt.subplot(gs[1,0])
plot_mu(mu_set='R_major_star',
            x_set=1/fall2virrot_full,muind=sat_ind&(fall2virrot_full>0.001)&(fall2virrot_full<1000),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[0],
            Halo_info_ulim=100,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax3,label='Total'
           )
for ii in range(len(submass_labels)-1):
    plot_mu(mu_set='R_major_star',
            x_set=1/fall2virrot_full,muind=sat_ind&(fall2virrot_full>0.001)&(fall2virrot_full<1000)&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[ii+1],
            Halo_info_ulim=100,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax3,label=submass_labels[ii]
           )
ax3.set_xlabel(r'$\tau_{\rm vir}/\hat{\tau}_{\rm orbit}$',fontsize=15)
ax3.set_ylabel(r'$\mu$',fontsize=15)
ax3.set_title(r'$\vec{r}-$'+'galaxy major',fontsize=15)
ax3.set_ylim(-0.2,1)
ax3.legend()

ax4 = plt.subplot(gs[1,1])
plot_mu(mu_set='R_major_star',
            x_set=1/fall2axisrot_full,muind=sat_ind&(fall2axisrot_full>0.001)&(fall2axisrot_full<1000),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[0],
            Halo_info_ulim=500,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax4,label='Total major'
           )



for ii in range(len(submass_labels)-1):
    plot_mu(mu_set='R_major_star',
            x_set=1/fall2axisrot_full,muind=sat_ind&(fall2axisrot_full>0.001)&(fall2axisrot_full<1000)&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[ii+1],
            Halo_info_ulim=500,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax4,label=submass_labels[ii]
           )
ax4.set_xlabel(r'$\tau_{\rm axis}/\hat{\tau}_{\rm orbit}$',fontsize=15)
ax4.set_ylabel(r'$\mu$',fontsize=15)
ax4.set_title(r'$\vec{r}-$'+'galaxy major',fontsize=15)
ax4.legend()
ax4.semilogx()
ax4.set_ylim(-0.5,1)

ax5 = plt.subplot(gs[2,0])
# ax3.errorbar(sshmassratio,cenmu2,yerr=cenmuerr2,fmt='o',color=clist[0])
plot_mu(mu_set='R_major_dm',
        x_set='S_over_R_vir',
        muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[0],
            Halo_info_ulim=3,Halo_info_dlim=0.01,fmt='o-',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax5,label="Total major"
           )

plot_mu(mu_set='R_medium_dm',
        x_set='S_over_R_vir',
        muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[2],
            Halo_info_ulim=3,Halo_info_dlim=0.01,fmt='o-',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax5,label="Total medium"
           )

plot_mu(mu_set='R_minor_dm',
        x_set='S_over_R_vir',
        muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='',title='',color=clist[4],
            Halo_info_ulim=3,Halo_info_dlim=0.01,fmt='o-',
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax5,label="Total minor"
           )

subflatedges=np.linspace(-0.5,0.5,10)

for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="{:.2f}<Flatness<{:.2f}".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='R_minor_dm',
        x_set='S_over_R_vir',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{r}-\vec{V}$',color=clist[ii+1],
            Halo_info_ulim=3,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax5,label=subflatlabels
           )

ax5.set_ylim(-1,1)
ax5.set_xlabel(r'$r/r_{\rm vir}$',fontsize=15)
ax5.set_ylabel(r'$\mu$',fontsize=15)
ax5.set_title(r'$\vec{r}-$'+'subhalo major',fontsize=15)
ax5.legend()
# ax3.semilogx()

ax6 = plt.subplot(gs[2,1])
# ax4.errorbar(fg,cenmu3,yerr=cenmuerr3,fmt='o',color=clist[0])

plot_mu(mu_set='R_major_star',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[3],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy(major)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)
plot_mu(mu_set='R_major_dm',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[8],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo(major)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)


plot_mu(mu_set='R_minor_star',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[0],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy(minor)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)
plot_mu(mu_set='R_minor_dm',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[2],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo(minor)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)



plot_mu(mu_set='R_medium_star',
        x_set=galaxy_halo_MA['flatness_star'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[5],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='galaxy(medium)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)
plot_mu(mu_set='R_medium_dm',
        x_set=galaxy_halo_MA['flatness_dm'][:],muind=sat_ind,
        galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='flatness',title=mu_set_label,color=clist[11],
        Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='subhalo(medium)',
            make_plot=1,return_eb=True,logx=False,ax=ax6)


ax6.set_ylim(-1,1)
ax6.legend()
ax6.set_title('')
ax6.set_xlabel('flatness(galaxy/subhalo)',fontsize=15)
ax6.set_title(r'$\vec{r}-$'+'galaxy major',fontsize=15)
ax6.set_ylabel(r'$\mu$',fontsize=15)


fig.savefig('./plots/radial_dm.png')

# %% code cell 113
fig = plt.figure(figsize = (20,30))
gs = fig.add_gridspec(4,3, hspace=0.2, wspace=0.2)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax00 = plt.subplot(gs[0,0])

visualize_star_system(ax=ax00, components=['central'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Central galaxy-Halo")

ax10 = plt.subplot(gs[1,0])

visualize_star_system(ax=ax10, components=['satellite','subhalo','satellite_axis','subhalo_axis'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Subhalo")

ax20 = plt.subplot(gs[2,0])
visualize_star_system(ax=ax20, components=['position_vector','satellite','satellite_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Postion")



ax30 = plt.subplot(gs[3,0])
visualize_star_system(ax=ax30, components=['position_vector','subhalo','subhalo_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Subhalo-Postion")

fmt_ls    = {'galaxy':'-o','subhalo':'--o'}
axiscls   = {'major' :clist[0],'medium' :clist[2],'minor' :clist[4]}

ax01 = plt.subplot(gs[0,1])

# for obj in fmt_ls.keys():
#     for axes_ in axiscls.keys():
#         plot_mu(mu_set=axes_,
#             x_set=galaxy_halo_MA['flatness_{}'.format(obj)][:],muind=cen_ind,
#             galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
#             Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='{} ({})'.format(obj,axes_),fmt=fmt_ls[obj],
#                 make_plot=1,return_eb=True,logx=False,ax=ax01)


subflatedges=np.linspace(-0.5,0.5,5)

for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="{:.2f}<f_sub<{:.2f}".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='minor',
        x_set='flatness_star',muind=cen_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax01,label=subflatlabels+'(minor)'
           )
    plot_mu(mu_set='major',
        x_set='flatness_star',muind=cen_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii+4],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax01,label=subflatlabels+'(major)'
           )

ax01.set_title('')
ax01.legend(ncol=4,fontsize=8)
ax01.set_xlabel('Flatness')
ax01.set_xlim(-0.5,0.5)
ax01.set_ylim(0,1)

ax11 = plt.subplot(gs[1,1])


for obj in fmt_ls.keys():
    for axes_ in axiscls.keys():
        plot_mu(mu_set=axes_,
            x_set=galaxy_halo_MA['flatness_{}'.format(obj)][:],muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='{} ({})'.format(obj,axes_),fmt=fmt_ls[obj],
                make_plot=1,return_eb=True,logx=False,ax=ax11)




ax11.set_title('')
ax11.legend(ncol=3)
ax11.set_xlabel('Flatness')
ax11.set_xlim(-0.5,0.5)
ax11.set_ylim(0,1)

ax21 = plt.subplot(gs[2,1])

for obj in ['galaxy']:
    for axes_ in axiscls.keys():
        plot_mu(mu_set='R_{}_{}'.format(axes_,obj),
            x_set=galaxy_halo_MA['flatness_{}'.format(obj)][:],muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='{} ({})'.format(obj,axes_),fmt=fmt_ls[obj],
                make_plot=1,return_eb=True,logx=False,ax=ax21)

ax21.set_title('')
ax21.legend()
ax21.set_xlabel('Flatness')
ax21.set_xlim(-0.5,0.5)
ax21.set_ylim(-0.5,0.5)

ax31 = plt.subplot(gs[3,1])

for obj in ['subhalo']:
    for axes_ in axiscls.keys():
        plot_mu(mu_set='R_{}_{}'.format(axes_,obj),
            x_set=galaxy_halo_MA['flatness_{}'.format(obj)][:],muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label='{} ({})'.format(obj,axes_),fmt=fmt_ls[obj],
                make_plot=1,return_eb=True,logx=False,ax=ax31)

ax31.set_title('')
ax31.legend()
ax31.set_xlabel('Flatness')
ax31.set_xlim(-0.5,0.5)
ax31.set_ylim(-1,1)


ax02 = plt.subplot(gs[0,2])


for obj in ['galaxy']:
    for axes_ in axiscls.keys():
        plot_mu(mu_set='T_grp_{}_{}'.format(axes_,obj),
            x_set=galaxy_halo_MA['flatness_{}'.format(obj)][:],muind=np.ones_like(cen_ind),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label=r'$\vec{T}_{\rm grp}$'+'-{} {}'.format(obj,axes_),fmt='-^',
                make_plot=1,return_eb=True,logx=False,ax=ax02)
        plot_mu(mu_set='T_tot_{}_{}'.format(axes_,obj),
            x_set=galaxy_halo_MA['flatness_{}'.format(obj)][:],muind=np.ones_like(cen_ind),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label=r'$\vec{T}_{\rm tot}$'+'-{} {}'.format(obj,axes_),fmt='--^',
                make_plot=1,return_eb=True,logx=False,ax=ax02)
ax02.set_title('')
ax02.legend(ncol=3)
ax02.set_xlabel('Flatness')
ax02.set_xlim(-0.5,0.5)
ax02.set_ylim(-1,1)

ax12 = plt.subplot(gs[1,2])

for obj in ['subhalo']:
    for axes_ in axiscls.keys():
        plot_mu(mu_set='T_grp_{}_{}'.format(axes_,obj),
            x_set=galaxy_halo_MA['flatness_{}'.format(obj)][:],muind=np.ones_like(cen_ind),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label=r'$\vec{T}_{\rm grp}$'+'-{} {}'.format(obj,axes_),fmt='-^',
                make_plot=1,return_eb=True,logx=False,ax=ax12)
        plot_mu(mu_set='T_tot_{}_{}'.format(axes_,obj),
            x_set=galaxy_halo_MA['flatness_{}'.format(obj)][:],muind=np.ones_like(cen_ind),
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,label=r'$\vec{T}_{\rm tot}$'+'-{} {}'.format(obj,axes_),fmt='--^',
                make_plot=1,return_eb=True,logx=False,ax=ax12)
ax12.set_title('')
ax12.legend(ncol=2)
ax12.set_xlabel('Flatness')
ax12.set_xlim(-0.5,0.5)
ax12.set_ylim(-1,1)




ax22 = plt.subplot(gs[2,2])



for obj in ['galaxy']:
    for axes_ in axiscls.keys():
        plot_mu(mu_set='R_{}_{}'.format(axes_,obj),
            x_set='S_over_R_vir',muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
            Halo_info_ulim=2,Halo_info_dlim=0,label='{} ({})'.format(obj,axes_),fmt=fmt_ls[obj],
                make_plot=1,return_eb=True,logx=False,ax=ax22)

ax22.set_title('')
ax22.legend()
ax22.set_xlabel(r'$r/r_{\rm vir}$',fontsize=15)
ax22.set_xlim(0,2)
ax22.set_ylim(-0.4,0.4)


ax32 = plt.subplot(gs[3,2])


for obj in ['subhalo']:
    for axes_ in axiscls.keys():
        plot_mu(mu_set='R_{}_{}'.format(axes_,obj),
            x_set='S_over_R_vir',muind=sat_ind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=axiscls[axes_],
            Halo_info_ulim=5,Halo_info_dlim=0,label='{} ({})'.format(obj,axes_),fmt=fmt_ls[obj],
                make_plot=1,return_eb=True,logx=False,ax=ax32)

ax32.set_title('')
ax32.legend()
ax32.set_xlabel(r'$r/r_{\rm vir}$',fontsize=15)
ax32.set_xlim(0,5)
ax32.set_ylim(-1,1)



fig.savefig('./plots/all.png')

# %% code cell 114

# %% code cell 115
# FoF group prolate shape bins diff schemes, central

# %% code cell 116
plt.clf()
fig = plt.figure(figsize = (18,6))
gs = fig.add_gridspec(1,3, hspace=0.2, wspace=0.2)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax00 = plt.subplot(gs[0,0])
visualize_star_system(ax=ax00, components=['position_vector','satellite','satellite_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Postion")
ax01 = plt.subplot(gs[0,1])

axes_list=['major','medium','minor']
for axax in range(3):
    plot_mu(mu_set=f'R_T_grp_{axes_list[axax]}',
                x_set='S_over_R_vir',muind=sat_ind,
                galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=clist[axax],
                Halo_info_ulim=5,Halo_info_dlim=0,label=f'Group {axes_list[axax]} renorm.',
                    make_plot=1,return_eb=True,logx=False,ax=ax01)
    plot_mu(mu_set=f'R_T_grp_{axes_list[axax]}',
                x_set='R_over_R_vir',muind=sat_ind,
                galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=clist[axax],
                Halo_info_ulim=5,Halo_info_dlim=0,label=f'Group {axes_list[axax]}',
                    make_plot=1,return_eb=True,logx=False,ax=ax01,fmt='--o')
    # plot_mu(mu_set=f'R_T_tot_{axes_list[axax]}',
    #             x_set='S_over_R_vir',muind=sat_ind,
    #             galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=clist[axax],
    #             Halo_info_ulim=5,Halo_info_dlim=0,label=f'Field {axes_list[axax]}',
    #                 make_plot=1,return_eb=True,logx=False,ax=ax01,fmt='--o')
ax01.legend(ncol=3)
ax01.set_title('')
ax01.legend()
ax01.set_xlabel(r'$r/r_{\rm vir}$',fontsize=15)
ax01.set_xlim(0,5)
ax01.set_ylim(-1,1)
ax02 = plt.subplot(gs[0,2])
# for axax in range(axes_list.size):
#     plot_mu(mu_set='T_grp_major_star',
#                 x_set='S_over_R_vir',muind=sat_ind,
#                 galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=clist[0],
#                 Halo_info_ulim=5,Halo_info_dlim=0,label='Group',
#                     make_plot=1,return_eb=True,logx=False,ax=ax02)
#     plot_mu(mu_set='T_tot_major_star',
#                 x_set='S_over_R_vir',muind=sat_ind,
#                 galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title=mu_set_label,color=clist[2],
#                 Halo_info_ulim=5,Halo_info_dlim=0,label='Field',
#                     make_plot=1,return_eb=True,logx=False,ax=ax02)

# %% code cell 117
galaxy_halo_MA.keys()

# %% code cell 118
plt.clf()
fig = plt.figure(figsize = (18,24))
gs = fig.add_gridspec(4,2, hspace=0.2, wspace=0.2)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax00 = plt.subplot(gs[0,0])
visualize_star_system(ax=ax00, components=['central'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Central galaxy-Halo")
ax01 = plt.subplot(gs[0,1])


subflatedges=np.linspace(-0.4,0.4,5)

for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='minor',
        x_set='flatness_star',muind=cen_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax01,label=subflatlabels+'(minor)'
           )
for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='major',
        x_set='flatness_star',muind=cen_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii+4],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax01,label=subflatlabels+'(major)'
           )
for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='medium',
        x_set='flatness_star',muind=cen_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii+12],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax01,label=subflatlabels+'(medium)'
           )
ax01.set_title('Central axes - Halo axes', weight='bold')
ax01.legend(ncol=3,fontsize=8)
ax01.set_xlabel('Flatness of galaxies')
ax01.set_xlim(-0.5,0.5)
ax01.set_ylim(-0.2,1)


ax10 = plt.subplot(gs[1,0])

visualize_star_system(ax=ax10, components=['satellite','subhalo','satellite_axis','subhalo_axis'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Subhalo")
ax11 = plt.subplot(gs[1,1])
for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='minor',
        x_set='flatness_star',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax11,label=subflatlabels+'(minor)'
           )
for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='major',
        x_set='flatness_star',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii+4],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax11,label=subflatlabels+'(major)'
           )

for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='medium',
        x_set='flatness_star',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii+12],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax11,label=subflatlabels+'(medium)'
           )

ax11.set_title('Sat.e axes - Sub. axes', weight='bold')
ax11.legend(ncol=3,fontsize=8)
ax11.set_xlabel('Flatness of galaxies')
ax11.set_xlim(-0.5,0.5)
ax11.set_ylim(0,1)

ax20 = plt.subplot(gs[2,0])
visualize_star_system(ax=ax20, components=['position_vector','satellite','satellite_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Postion")


ax21 = plt.subplot(gs[2,1])
for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='R_minor_star',
        x_set='flatness_star',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax21,label=subflatlabels+'(minor)'
           )
for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='R_major_star',
        x_set='flatness_star',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii+4],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax21,label=subflatlabels+'(major)'
           )
for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='R_medium_star',
        x_set='flatness_star',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii+12],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax21,label=subflatlabels+'(medium)'
           )

ax21.set_title('Pos. - Sat. axes', weight='bold')
ax21.legend(ncol=3,fontsize=8)
ax21.set_xlabel('Flatness of galaxies')
ax21.set_xlim(-0.5,0.5)
ax21.set_ylim(-1,1)


ax30 = plt.subplot(gs[3,0])
visualize_star_system(ax=ax30, components=['position_vector','subhalo','subhalo_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Subhalo-Postion")


ax31 = plt.subplot(gs[3,1])
for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='R_minor_dm',
        x_set='flatness_star',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax31,label=subflatlabels+'(minor)'
           )
for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='R_major_dm',
        x_set='flatness_star',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii+4],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax31,label=subflatlabels+'(major)'
           )

for ii in range(len(subflatedges)-1):
    subflatind=(galaxy_halo_MA['flatness_dm'][:]>subflatedges[ii])&(galaxy_halo_MA['flatness_dm'][:]<subflatedges[ii+1])
    subflatlabels="[{:.1f},{:.1f}]".format(subflatedges[ii],subflatedges[ii+1])
    plot_mu(mu_set='R_medium_dm',
        x_set='flatness_star',muind=sat_ind&subflatind,
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel='Flatness',title='',color=DH[ii+12],
            Halo_info_ulim=0.5,Halo_info_dlim=-0.5,
            make_plot=1,return_eb=0,logx=False,in_plot='mu',ax=ax31,label=subflatlabels+'(medium)'
           )

ax31.set_title('Pos. - Sub. axes', weight='bold')
ax31.legend(ncol=3,fontsize=8)
ax31.set_xlabel('Flatness of galaxies')
ax31.set_xlim(-0.5,0.5)
ax31.set_ylim(-1,1)
fig.savefig('./plots/sat_sub_flatness.png')

# %% code cell 119
plt.clf()
fig = plt.figure(figsize = (18,12))
gs = fig.add_gridspec(2,2, hspace=0.2, wspace=0.2)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

ax00 = plt.subplot(gs[0,0])
visualize_star_system(ax=ax00, components=['central'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Central galaxy-Halo")

ax01 = plt.subplot(gs[0,1])

visualize_star_system(ax=ax01, components=['satellite','subhalo','satellite_axis','subhalo_axis'], 
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Subhalo")


ax10 = plt.subplot(gs[1,0])
visualize_star_system(ax=ax10, components=['position_vector','satellite','satellite_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Satellite-Postion")



ax11 = plt.subplot(gs[1,1])
visualize_star_system(ax=ax11, components=['position_vector','subhalo','subhalo_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Subhalo-Postion")


fig.savefig('./plots/IA_sketch.pdf')

# %% code cell 120
# IPython-only: %load_ext autoreload
# IPython-only: %autoreload 2

# %% code cell 121
import importlib
import theo_mu

importlib.reload(theo_mu)

# %% code cell 122
from theo_mu import compute_mu_curve

# %% code cell 123
rrvs_ls=np.logspace(-1,0,30)
mu1125, kappa, sigma = compute_mu_curve(
    x_array=rrvs_ls,
    M_host_msun_h=10**(11.25), z=0.0,
    beta=0.05, epsilon=0.4, 
    N_relax=(3.5-rrvs_ls)**1.5,
    # non-circular orbit controls:
    use_noncircular=True,
    mu_align_min=0.4,
    mu_align_max=0.8
)
mu1175, kappa, sigma = compute_mu_curve(
    x_array=rrvs_ls,
    M_host_msun_h=10**(11.75), z=0.0,
        beta=0.05, epsilon=0.4,
    N_relax=(3.5-rrvs_ls)**1.5,
    # non-circular orbit controls:
    use_noncircular=True,
    mu_align_min=0.4,
    mu_align_max=0.8
)
mu125, kappa, sigma = compute_mu_curve(
    x_array=rrvs_ls,
    M_host_msun_h=10**(12.5), z=0.0,
        beta=0.05, epsilon=0.4, 
    N_relax=(3.5-rrvs_ls)**1.5,
    # non-circular orbit controls:
    use_noncircular=True,
    mu_align_min=0.4,
    mu_align_max=0.8
)

# %% code cell 124
mu125

# %% code cell 125
plt.clf()
fig = plt.figure(figsize = (18,8))
gs = fig.add_gridspec(1,2, hspace=0.2, wspace=0.2)

sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'


ax00 = plt.subplot(gs[0,0])
visualize_star_system(ax=ax00, components=['position_vector','subhalo','subhalo_axis'],
                           misalignment_angle=30, size_factor=1, show_dashed_axis=True, 
                           title="Subhalo-Postion")


ax2 = plt.subplot(gs[0,1])
for ii in range(len(submass_labels)-1):
    plot_mu(mu_set='R_major_dm',x_set='S_over_R_vir',muind=sat_ind&submass_inds[ii],
            galaxy_halo_MA=galaxy_halo_MA,bins=10,xlabel=r'$r/r_{\rm vir}$',title=r'$\vec{T}-\vec{V}$',color=clist[ii+1],
            Halo_info_ulim=1,Halo_info_dlim=0.1,
            make_plot=1,return_eb=0,logx=True,in_plot='mu',ax=ax2,label=submass_labels[ii]
           )

ax2.set_xlabel(r'$r/r_{\rm vir}$',fontsize=15)
ax2.set_ylabel(r'$\mu$',fontsize=15)
ax2.legend(frameon=False)
ax2.set_title(r'$\vec{r}$'+'-subhalo major',fontsize=15)
ax2.set_ylim(0.2,1)
ax2.plot(rrvs_ls,-mu1125,color=clist[1])
ax2.plot(rrvs_ls,-mu1175,color=clist[2])
ax2.plot(rrvs_ls,-mu125,color=clist[3])

# fig.savefig('./plots/sat_sub_flatness.png')

# %% code cell 126
import Gyro_halo

importlib.reload(Gyro_halo)
from Gyro_halo import DMHaloTidalSimulator

# %% code cell 127
sim = DMHaloTidalSimulator(
    t0=0.0, t1=40.0, dt=0.005,
    equal_volume=True,
    v_long_world=(1, 0.2, 0.93),  
    omega0=(0.0, 0., 0.),          
    lambda0=(1.20, 1., 0.85),       
    T_diag=(-0.00040, -0.0001, 0.0009),    
    T_rot_rate_x=1,              
    out_dir="outputs",
    show_progress_sim=True,
    show_progress_render=True
)

# %% code cell 128
# (1) Compute arrays only
arrays = sim.simulate_arrays()

# You now have:
# arrays['t']                -> (N,)
# arrays['lambda_hist']      -> (N,3)
# arrays['dlambda_hist']     -> (N,3)
# arrays['R_hist']           -> (N,3,3)  (columns are e_a, e_b, e_c in WORLD)
# arrays['omega_hist']       -> (N,3)
# arrays['Tbody_hist']       -> (N,3,3)
# arrays['major_axis_hist']  -> (N,3)

# (2) One-line rendering (3 MP4s + lambda time-series PNG)
L_hist = arrays['lambda_hist']; t = arrays['t']
abscosMA=get_cosMA(arrays['R_hist'][:][:,0],arrays['major_axis_hist'] )
# plt.figure(figsize=(7,4))
# plt.plot(t, L_hist[:,0], label=r'$\lambda_a$')
# plt.plot(t, L_hist[:,1], label=r'$\lambda_b$')
# plt.plot(t, L_hist[:,2], label=r'$\lambda_c$')
# plt.xlabel('t'); plt.ylabel('lambda'); plt.grid(True, linestyle=':', linewidth=0.5); plt.legend()

# %% code cell 129
plt.plot(t,np.degrees(np.arccos(abscosMA)))
plt.xlabel('t'); plt.ylabel(r'$\theta_{\rm MA}$')
plt.ylim(0,180)

# %% code cell 130
sim.plot_lambda(arrays,lambda_plot='lambda_timeseries.png')

# %% code cell 131
# sim.render_all_videos_from_arrays(
#     arrays,
#     fps=60, trail_seconds=5.0, xyz_lim=1.2, dpi=100,
#     out_names=('ea_tip.mp4','eb_tip.mp4','ec_tip.mp4')
# )

# %% code cell 132
import plots_volume_and_shapes

importlib.reload(plots_volume_and_shapes)

# %% code cell 133
h5file = "galaxy_properties_099_m20_with_bins.hdf5"
from plots_volume_and_shapes import plot_mu_vs_volume_fraction

fig, axs = plot_mu_vs_volume_fraction(
    "galaxy_properties_099_m20_with_bins.hdf5",
    nbins= 8,vfrac_min= 1e-2,
    target='dm',
    chi_range=(-0.6, 0.6),
    n_chi_bins=7,
    cmap="coolwarm"
)

# %% code cell 134

# %% code cell 135

# %% code cell 136

# %% code cell 137

# %% code cell 138

# %% code cell 139

# %% code cell 140

# %% code cell 141

# %% code cell 142

# %% code cell 143

# %% code cell 144
