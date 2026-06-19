"""Exported code from notebooks/raw_20260618/orbit.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% code cell 1
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from orbit_nfw import OrbitSimulator, default_tng_cosmology
import pyccl as ccl
import seaborn as sns

# %% code cell 2
from IPython.display import Video

# %% code cell 3
from arts import *

# %% code cell 4
clist=['#c02c38','#c2c116','#3c9566','#1177b0','#ff7c38','#be8936','#e03e36','#b80d57','#700961','#11659a','#abcdef','#fedcba']

DH=['#A73D30','#C16355','#D77E73','#F0D0C6',
    '#0C52B5','#387CBC','#5F81C2','#79B9DC',
    '#81521D','#C1823E','#DAB25B','#E9D077',
    '#305937','#718A70','#68A270','#8FC198',]
get_colors(clist)

# %% code cell 5
# --- host setup (NFW + cosmology) ---
cosmo = default_tng_cosmology()
sim = OrbitSimulator(
    M200c=1e4,   # = 1e14 Msun/h
    c=10.0,
    z=0.0,
    cosmo=cosmo
)

# %% code cell 6

def make_nfw_profiles(host):
    """
    Build callables from the NFWHost used in the orbit integrator.

    Parameters
    ----------
    host : NFWHost
        The host stored in OrbitSimulator.host

    Returns
    -------
    Menc, rho, Phi : callables
        Each takes r in ckpc/h (scalar or array).
    """
    a = host.a
    h = host.h

    def _to_kpc_phys(r):
        r = np.asarray(r, dtype=float)
        return (r / h) * a  # ckpc/h -> physical kpc

    def Menc(r):
        """Enclosed mass M(<r) in 1e10 Msun/h."""
        rk = _to_kpc_phys(r)
        M_msun = host.menc(rk)               # Msun
        return (M_msun * h) / 1e10           # 1e10 Msun/h

    def rho(r):
        """Density rho(r) in (1e10 Msun/h) / (ckpc/h)^3."""
        rk = _to_kpc_phys(r)
        rho_msun_kpc3 = host.rho(rk)         # Msun/kpc^3 (physical)
        return rho_msun_kpc3 * (a**3) / (1e10 * h**2)

    def Phi(r):
        """Potential Phi(r) in (km/s)^2 (Phi(∞)=0)."""
        rk = _to_kpc_phys(r)
        return host.phi(rk)

    return Menc, rho, Phi

# %% code cell 7
Menc, rho, Phi = make_nfw_profiles(sim.host)

# %% code cell 8

# %% code cell 9
def R200c(cosmo, M200c, z=0.0):
    a = 1/(1+z); h = cosmo["h"]
    rho_c = ccl.rho_x(cosmo, a, "critical")/1e9          # Msun/kpc^3
    R_kpc = (3*(M200c*1e10/h)/(4*np.pi*200*rho_c))**(1/3) # kpc (physical)
    return h*R_kpc/a                                     # ckpc/h

# %% code cell 10
r200c= R200c(cosmo, M200c=1e4, z=0.0)

# %% code cell 11
r200c

# %% code cell 12
#E=v_r^2/2+Phi+L^2/r^2

# %% code cell 13
# --- run orbit ---
r0=r200c
vt=100
vr=10
print(f'{vr**2/2+Phi(r0)+vt**2/2:.0f}')
init={
    'E':vr**2/2+Phi(r0)+vt**2/2,           # (km/s)^2, bound orbit usually E<0
    'L':r0*vt,      # (ckpc/h)*(km/s)
    'r0':r0,        # ckpc/h
    'phi0':0.0,            # rad
    'vr_sign':1,              # +1 outward, -1 inward
    't_end':11.0,           # Gyr
    'dt':0.01,            # Gyr
    'm_sub':0.1,
    'soften':1.0,     
    'r_merge':30.0,      
    'v_merge':None,     
    'with_strip':True,   
    'alpha_sub':2.5,     
    'r_trunc0' :1, 
}

    

res = sim.run(
    **init,
    with_df=True,           # switch OFF
    show_progress=True,
    progress_desc="Orbit (no DF)",
    progress_update_every=10,
)

resdf = sim.run(
    **init,
    with_df=True,           # switch OFF
    lnLambda=1.0,
    show_progress=True,
    progress_desc="Orbit (with DF)",
    progress_update_every=10,
)

# %% code cell 14
sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'
theta=np.linspace(0,2*np.pi,500)
fig, ax = plt.subplots(figsize=(8, 8))
# ax.plot(res.pos[:,0],res.pos[:,1],color=clist[5],label='no DF')
ax.plot(resdf.pos[:,0],resdf.pos[:,1],color=clist[9],label='orbit')
ax.scatter(res.pos[0,0],res.pos[0,1],s=20,marker='o',color=clist[4],label='start')
# ax.scatter(res.pos[-1,0],res.pos[-1,1],s=20,marker='o',color=clist[5],label='end no DF')
ax.scatter(resdf.pos[-1,0],resdf.pos[-1,1],s=20,marker='o',color=clist[0],label='end')
ax.scatter(0,0,s=20,marker='x',color='k',label='centre')
ax.plot(r200c*np.cos(theta),r200c*np.sin(theta),ls='--',color=clist[8],label=r'$R_{200c}$')
ax.set_xlim(-3.3*r200c,3.3*r200c)
ax.set_ylim(-3.3*r200c,3.3*r200c)
ax.set_xlabel(r'$x\;[ckpc/h]$')
ax.set_ylabel(r'$y\;[ckpc/h]$')
ax.legend(frameon=False,ncol=5,fontsize=12)
ax.set_aspect('equal')
# fig.savefig('bf.png',dpi=300)

# %% code cell 15
from orbit_viz import save_orbit_movie6,save_orbit_movie3

# %% code cell 16
out= save_orbit_movie6(
    res,
    r200c=r200c,
    fps=30,
    duration=12,
    outfile="orbit6_tri.mp4",
    figsize=(16, 9),
    dpi=200,
    trail_gyr=2.0,
    codec="mp4v", 
    show_progress=True,
)
print(out)

# %% code cell 17
out= save_orbit_movie3(
    res,
    r200c=r200c,
    fps=30,
    duration=12,
    outfile="orbit3_tri.mp4",
    figsize=(16, 9),
    dpi=200,
    trail_gyr=2.0,
    codec="mp4v", 
    show_progress=True
)
print(out)

# %% code cell 18
from IPython.display import HTML

# %% code cell 19
HTML("""
<video width="960"  height="540" controls autoplay loop muted>
    <source src="/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/orbit3_tri.mp4" type="video/mp4">
</video>
""")

# %% code cell 20

Video("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/orbit3_tri.mp4", width=960, height=540, embed=False)

# %% code cell 21
sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(res.t-11,res.r,color=clist[5],label='w/o DF')
ax.plot(resdf.t-11,resdf.r,color=clist[3],label='w/ DF')
ax.set_xlabel(r'$t\;[Gyr]$')
ax.set_ylabel(r'$r\;[ckpc/h]$')
ax.legend()

# %% code cell 22

# %% code cell 23

# %% code cell 24
sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(res.t-11,-res.Trr,color=clist[5],label='w/o DF')
ax.plot(resdf.t-11,-resdf.Trr,color=clist[3],label='w/ DF')
ax.set_xlabel(r'$t\;[Gyr]$')
ax.set_ylabel(r'$-T_{rr}\;[km \cdot s^{-2} \cdot h^2 \cdot ckpc^{-2}]$')
ax.legend()

# %% code cell 25
sns.set(style='ticks')
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(res.t-11,res.Tphiphi,color=clist[5],label='w/o DF')
ax.plot(resdf.t-11,resdf.Tphiphi,color=clist[3],label='w/ DF')
ax.set_xlabel(r'$t\;[Gyr]$')
ax.set_ylabel(r'$T_{\phi\phi}\;[km \cdot s^{-2} \cdot h^2 \cdot ckpc^{-2}]$')
ax.legend()

# %% code cell 26
def multimass():
    M200c=1e4
    sim = OrbitSimulator(
    M200c=M200c,   # = 1e14 Msun/h
    c=10.0,
    z=0.0,
    cosmo=cosmo
    )
    r200c= R200c(cosmo, M200c=M200c, z=0.0)
    Menc, rho, Phi = make_nfw_profiles(sim.host)
    # --- run orbit ---
    r0=r200c
    vt=100
    vr=100
    print(f'{vr**2/2+Phi(r0)+vt**2/2:.0f}')
    init={
        'E':vr**2/2+Phi(r0)+vt**2/2,           # (km/s)^2, bound orbit usually E<0
        'L':r0*vt,      # (ckpc/h)*(km/s)
        'r0':r0,        # ckpc/h
        'phi0':0.0,            # rad
        'vr_sign':-1,              # +1 outward, -1 inward
        't_end':11.0,           # Gyr
        'dt':0.01,            # Gyr
        'soften':1.0,     
        'r_merge':25.0,      
        'v_merge':None,     
        'with_strip':True,   
        'alpha_sub':2.5,     
        'r_trunc0' :None, 
        
    }
    sns.set(style='ticks')
    plt.rcParams['xtick.direction']  = 'in'
    plt.rcParams['ytick.direction']  = 'in'
    plt.rcParams['mathtext.fontset'] = 'cm'
    theta=np.linspace(0,2*np.pi,500)
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.scatter(r0,0,color=clist[5],zorder=10)

    ax.scatter(0,0,s=20,marker='x',color='k')
    ax.plot(r200c*np.cos(theta),r200c*np.sin(theta),ls='--',color=clist[8])
  
    for inds in [0,1,2,3]:
        res = sim.run(
            **init,
            m_sub=10**inds,
            with_df=True,           # switch OFF
            show_progress=True,
            progress_desc="Orbit (DF)",
            progress_update_every=10,
        )
        dex=10+inds
        ax.plot(res.pos[:,0],res.pos[:,1],color=clist[inds],label=r'$m_{\rm sub}=$'+rf'$10^{{{dex}}}$'+r'$M_\odot$')
    ax.set_xlim(-1.3*r200c,1.3*r200c)
    ax.set_ylim(-1.3*r200c,1.3*r200c)
    ax.set_xlabel(r'$x\;[ckpc/h]$')
    ax.set_ylabel(r'$y\;[ckpc/h]$')
    ax.legend(frameon=False,ncol=4,fontsize=12)
    ax.set_aspect('equal')
    fig.savefig('DF.png',dpi=300)

# %% code cell 27
multimass()

# %% code cell 28
def get_cosMA(v1, v2):

    dot_products = np.sum(v1 * v2, axis=1)

    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    
    cos_sim = dot_products / (norm_v1 * norm_v2 )
    
    
    return cos_sim

# %% code cell 29
r0=r200c
vt=100
vr=100
inits={
        'E':vr**2/2+Phi(r0)+vt**2/2,           # (km/s)^2, bound orbit usually E<0
        'L':r0*vt,      # (ckpc/h)*(km/s)
        'r0':r0,        # ckpc/h
        'phi0':0.0,            # rad
        'vr_sign':-1,              # +1 outward, -1 inward
        't_end':11.0,           # Gyr
        'dt':0.01,            # Gyr
        'soften':1.0,     
        'r_merge':25.0,      
        'v_merge':None,     
        'with_strip':True,   
        'alpha_sub':2.5,     
        'r_trunc0' :None, 
        
    }
for inds in [0,1,2,3]:
    res = sim.run(
            **inits,
            m_sub=10**inds,
            with_df=True,           # switch OFF
            show_progress=True,
            progress_desc="Orbit (no DF)",
            progress_update_every=10,
        )
    dex=10+inds
    plt.plot(res.t,res.r_t/res.r_t[0],color=clist[inds],label=r'$m_{\rm sub}=$'+rf'$10^{{{dex}}}$'+r'$M_\odot$')
plt.xlabel(r'$t\;[Gyr]$')
plt.ylabel(r'$M$')
plt.semilogy()
plt.legend(ncol=2)

# %% code cell 30
def multiangle():
    M200c=1e4
    sim = OrbitSimulator(
    M200c=M200c,   # = 1e14 Msun/h
    c=10.0,
    z=0.0,
    cosmo=cosmo
    )
    r200c= R200c(cosmo, M200c=M200c, z=0.0)
    Menc, rho, Phi = make_nfw_profiles(sim.host)
    # --- run orbit ---
    
    sns.set(style='ticks')
    plt.rcParams['xtick.direction']  = 'in'
    plt.rcParams['ytick.direction']  = 'in'
    plt.rcParams['mathtext.fontset'] = 'cm'
    theta=np.linspace(0,2*np.pi,500)
    fig, ax = plt.subplots(figsize=(12, 12))
    r0=r200c
    ax.scatter(r0,0,color=clist[5],zorder=10)

    ax.scatter(0,0,s=20,marker='x',color='k')
    ax.plot(r200c*np.cos(theta),r200c*np.sin(theta),ls='--',color=clist[8])
    v = 300
    print(v)
    for i,  ang in enumerate(np.linspace(0,np.pi/2,10)):
        try:
            vt=v*np.sin(ang)
            vr=v*np.cos(ang)
            print(f'{vr**2/2+Phi(r0)+vt**2/2:.0f}')
            init={
                'E':vr**2/2+Phi(r0)+vt**2/2,           # (km/s)^2, bound orbit usually E<0
                'L':r0*vt,      # (ckpc/h)*(km/s)
                'r0':r0,        # ckpc/h
                'phi0':0.0,            # rad
                'vr_sign':-1,              # +1 outward, -1 inward
                't_end':11.0,           # Gyr
                'dt':0.01,            # Gyr
                'soften':1.0,     
                'r_merge':50.0,      
                'v_merge':None,     
                'with_strip':True,   
                'alpha_sub':2.5,     
                'r_trunc0' :None, 
                
            }
            res = sim.run(
                **init,
                m_sub=1e2,
                with_df=True,           # switch OFF
                show_progress=True,
                progress_desc="Orbit",
                progress_update_every=10,
            )
            ax.plot(res.pos[:,0],res.pos[:,1],color=clist[i],label=r'$\theta={:.0f}^\circ$'.format(np.degrees(ang)))
        except:
            pass
    ax.set_xlim(-1.3*r200c,1.3*r200c)
    ax.set_ylim(-1.3*r200c,1.3*r200c)
    ax.set_xlabel(r'$x\;[ckpc/h]$')
    ax.set_ylabel(r'$y\;[ckpc/h]$')
    ax.legend(frameon=False,ncol=4,fontsize=12)
    ax.set_aspect('equal')
    fig.savefig('ang.png',dpi=300)

# %% code cell 31
multiangle()

# %% code cell 32
