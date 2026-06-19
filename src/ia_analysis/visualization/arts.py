# arts.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc
from tqdm import tqdm
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.colors import LogNorm
import math
try:
    from ia_analysis.visualization.DWE import DimrothWatson
    dw = DimrothWatson()
except Exception:  # optional dependency used only by alignment-distribution helpers
    DimrothWatson = None
    dw = None
clist=['#c02c38','#c2c116','#3c9566','#1177b0','#ff7c38','#bec936','#e03e36','#b80d57','#700961','#11659a','#abcdef','#fedcba']

DH=['#A73D30','#C16355','#D77E73','#F0D0C6',
    '#0C52B5','#387CBC','#5F81C2','#79B9DC',
    '#81521D','#C1823E','#DAB25B','#E9D077',
    '#305937','#718A70','#68A270','#8FC198',]




def extract_dominant_colors(image_path, num_colors=12):
    from PIL import Image  # For image processing
    from collections import Counter  # For counting occurrences
    from sklearn.cluster import KMeans  # For color clustering algorithm
    """
    Extract dominant colors from an image and return their hexadecimal codes.
    
    Parameters:
    image_path (str): Path to the input image file
    num_colors (int): Number of dominant colors to extract (default: 12)
    
    Returns:
    list: Hexadecimal color codes sorted by prominence in the image
    """
    
    # Open and process the image
    image = Image.open(image_path)  # Load image from file
    image = image.convert('RGB')  # Convert to RGB color space
    image = image.resize((150, 150))  # Resize for faster processing while maintaining enough detail
    
    # Convert image to numerical array
    img_array = np.array(image)  # Create NumPy array representation
    pixels = img_array.reshape((-1, 3))  # Reshape to 2D array (pixel list)
    
    # Apply K-Means clustering to identify dominant colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)  # Initialize with set random state
    kmeans.fit(pixels)  # Process pixel data to find color clusters
    
    # Analyze clustering results
    cluster_labels = kmeans.labels_  # Get pixel-to-cluster assignments
    cluster_centers = kmeans.cluster_centers_  # Get RGB values of cluster centers
    
    # Count cluster sizes to determine prominence
    cluster_counts = Counter(cluster_labels)  # Tally pixels per cluster
    
    # Sort colors by prominence (most frequent first)
    sorted_colors = sorted(
        [(count, color) for color, count in zip(cluster_centers, cluster_counts.values())],
        key=lambda x: x[0],
        reverse=True
    )
    
    # Convert RGB vectors to hexadecimal strings
    hex_colors = []
    for count, color in sorted_colors:
        # Round RGB values to integers (0-255 range)
        r, g, b = int(round(color[0])), int(round(color[1])), int(round(color[2]))
        
        # Format as 6-digit hexadecimal code
        hex_code = f"#{r:02X}{g:02X}{b:02X}"  # :02X pads to 2 uppercase hex digits
        hex_colors.append(hex_code)
    
    return hex_colors



def get_colors(clist,filename=None,ind=True):
    import matplotlib.patches as patches
    ncols = int(np.sqrt(len(clist))+0.6)
    size = 0.8  
    gap = 0.2  
    padding = 0.4 
    nrows = (len(clist) + ncols - 1) // ncols
    fig, ax = plt.subplots(figsize=(ncols*2, nrows*2))
    ax.set_aspect('equal')
    ax.axis('off')
    
    
    
    for i, color in enumerate(clist):
        row = i // ncols
        col = i % ncols
        

        x = col * (size + gap)
        y = (nrows - row - 1) * (size + gap) 
        
        rect = patches.Rectangle(
            (x, y), size, size, 
            linewidth=0.5, 
            edgecolor='white', 
            facecolor=color
        )
        ax.add_patch(rect)
        
        ax.text(
            x + size/2, y + size/2, 
            color, 
            ha='center', va='center', 
            color='white', #if i not in [1, 5, 10] else 'black', 
            fontsize=9, weight='bold'
        )
        if ind:
            ax.text(
                x + size/2, y - padding/7, 
                str(i), 
                ha='center', va='top', 
                fontsize=12, 
                color='black', weight='bold'
            )
    
    plt.xlim(-gap, ncols * (size + gap) )
    plt.ylim(-padding, nrows * (size + gap))
    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename,dpi=300)
    plt.show()
def plot_3d_scatter(xyz,xyz2=None, title="3D Scatter Plot", color='b',color2='g',alpha=0.3,alpha2=0.3,
                    size=20, size2=20,axis_length = 5, arrow=None,arrow2=None,arrow_color='b',arrow_color2='g',
                    arrow_label='DM major',arrow2_label='Stellar major',
                    particle_label='DM particles',particle_label2='Stellar particles',zoom_scale=1):
    """
    Plot a 3D scatter plot of the input coordinates.

    Parameters
    ----------
    x : array-like
        The x coordinates of the points.
    y : array-like
        The y coordinates of the points.
    z : array-like
        The z coordinates of the points.
    title : str, optional
        Title of the plot (default is "3D Scatter Plot").
    color : str or array-like, optional
        Color of the points (default is 'blue').
    size : float, optional
        Size of the points (default is 20).
    
    Returns
    -------
    None
    """
    # Create a new figure for 3D plotting
    x,y,z=xyz[:,0],xyz[:,1],xyz[:,2]
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    center = np.sum(xyz,axis=0)/len(xyz)
    # Plot the scatter points
    ax.scatter(x, y, z, c=color, s=size, alpha=alpha,label=particle_label)
    if xyz2 is not None:
        x2, y2, z2 = xyz2[:, 0], xyz2[:, 1], xyz2[:, 2]
        ax.scatter(x2, y2, z2, c=color2, s=size2, alpha=alpha2,label=particle_label2)
    # ax.scatter(center[0], center[1], center[2], c='k', s=10, alpha=1)
    if arrow is not None:
        arrow/=np.linalg.norm(arrow**2)
        ax.quiver(center[0], center[1], center[2],
                  arrow[0]*axis_length, 
                  arrow[1]*axis_length, 
                  arrow[2]*axis_length, 
                  color=clist[6], linewidth=3, arrow_length_ratio=0.1, label=arrow_label)
    if arrow2 is not None:
        arrow2/=np.linalg.norm(arrow2**2)
        ax.quiver(center[0], center[1], center[2],
                  arrow2[0]*axis_length, 
                  arrow2[1]*axis_length, 
                  arrow2[2]*axis_length, 
                  color=clist[8], linewidth=3, arrow_length_ratio=0.1, label=arrow2_label)
    ax.set_aspect('equal')
    # Set labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(title)
    ax.set_xlim(center[0]-zoom_scale*axis_length,center[0]+zoom_scale*axis_length)
    ax.set_ylim(center[1]-zoom_scale*axis_length,center[1]+zoom_scale*axis_length)
    ax.set_zlim(center[2]-zoom_scale*axis_length,center[2]+zoom_scale*axis_length)
    ax.legend()
    # Show the plot
    plt.tight_layout()
    plt.show()
    return fig,ax

def plot_3d_scatter_animated(fig,ax,n_frames=180, interval=50, elev=30, azim_start=0):
    """
    Create an animated 3D scatter plot with rotating view around the origin.
    
    Parameters
    ----------
    n_frames : int, optional
        Number of animation frames (default is 180).
    interval : int, optional
        Delay between frames in milliseconds (default is 50).
    elev : float, optional
        Elevation angle in degrees (default is 30).
    azim_start : float, optional
        Starting azimuth angle in degrees (default is 0).
    
    Returns
    -------
    animation.FuncAnimation
        The animation object for display in Jupyter Notebook.
    """

    ax.view_init(elev=elev, azim=azim_start)  # Initial view angle

    def update(frame):
        # Calculate new azimuth angle (full 360° rotation)
        azim = (azim_start + frame * 360 / n_frames) % 360
        ax.view_init(elev=elev, azim=azim)
        return fig,
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=n_frames,
        interval=interval,
        blit=False  # Disable blitting for 3D animations
    )
    
    plt.tight_layout()
    plt.close(fig)  # Prevent duplicate static plot display
    return anim


def visualize_galaxy_system(
    ax=None,
    components=['central', 'satellite'],
    sat_sub_vec='',
    cen_vec='',
    misalignment_angle=60,
    size_factor=0.8,
    galaxy_color=clist[6],
    show_dashed_axis=True,
    show_central_black_axis=False,
    title="Galaxy System Misalignment"
):

    def draw_axis(
        ax,
        pos,
        angle,
        length=1.5,
        color='k',
        alpha=1.0,
        zorder=5,
        show_dashed_axis=show_dashed_axis
    ):
        """Draw an orientation axis with arrow at given position and angle."""
        dx = length * np.cos(np.radians(angle))
        dy = length * np.sin(np.radians(angle))

        ax.arrow(
            pos[0], pos[1],
            dx, dy,
            width=0.03 * size_factor,
            color=color,
            head_length=0.2 * size_factor,
            alpha=alpha,
            zorder=zorder
        )

        if show_dashed_axis:
            ax.plot(
                [pos[0] - dx, pos[0]],
                [pos[1] - dy, pos[1]],
                color='k',
                ls='--',
                alpha=0.3 * alpha,
                lw=1.5,
                zorder=zorder
            )

    # Create axes if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Configure plot area
    ax.set_aspect('equal')
    ax.set_xlim(-7, 7)
    ax.set_ylim(-5, 5)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ------------- MAIN DARK MATTER HALO -------------
    halo_alpha = 0.7
    halo = Ellipse(
        xy=(0, 0),
        width=14 * size_factor,
        height=10 * size_factor,
        angle=0,
        fc='#DDDDDD',
        ec='black',
        alpha=halo_alpha,
        lw=1.5,
        zorder=1
    )
    ax.add_patch(halo)

    # Halo orientation axis
    ax.plot(
        [-8, 8],
        [0, 0],
        'k',
        ls='-',
        lw=1.5,
        alpha=0.2,
        zorder=2
    )

    ax.text(
        -6.5,
        -0.5,
        'Halo axis',
        fontsize=16,
        weight='bold',
        alpha=1.0 if 'halo' in components else 0.5
    )

    # ------------- CENTRAL SYSTEM -------------
    central_pos = (0, 0)
    central_alpha = 0.9 if 'central' in components else 0.2

    # Central host halo:
    # same center as central galaxy, gray like subhalo,
    # but with a slightly different orientation.
    central_host_halo_pos = central_pos
    central_host_halo_alpha = 1.0 if 'central' in components else 0.2
    central_host_halo_angle = 5

    central_host_halo = Ellipse(
        xy=central_host_halo_pos,
        width=2.6 * size_factor,
        height=1.6 * size_factor,
        angle=central_host_halo_angle,
        fc='lightgray',
        ec='gray',
        alpha=central_host_halo_alpha,
        zorder=3,
        hatch='////'
    )
    ax.add_patch(central_host_halo)

    # Central host halo axis:
    # follows the central host halo orientation.
    if 'central' in components:
        draw_axis(
            ax,
            central_host_halo_pos,
            central_host_halo_angle,
            1.4 * size_factor,
            color='dimgray',
            alpha=central_host_halo_alpha,
            zorder=10
        )

    # Central galaxy
    central_galaxy_angle = 20

    central_galaxy = Ellipse(
        xy=central_pos,
        width=2.2 * size_factor,
        height=1.3 * size_factor,
        angle=central_galaxy_angle,
        fc=clist[4],
        ec=galaxy_color,
        alpha=central_alpha,
        zorder=4,
        hatch='////'
    )
    ax.add_patch(central_galaxy)

    if 'central' in components:
        # Central galaxy colored axis
        draw_axis(
            ax,
            central_pos,
            central_galaxy_angle,
            1.5 * size_factor,
            color=clist[6],
            alpha=0.7
        )

        # Optional central black axis.
        # Default: not drawn.
        if show_central_black_axis:
            draw_axis(
                ax,
                central_pos,
                0,
                1.5 * size_factor,
                color='k',
                alpha=central_alpha
            )

    # Central galaxy label
    ax.text(
        0,
        -1.5 * size_factor,
        'Central galaxy',
        ha='center',
        color=clist[4],
        fontsize=14,
        weight='bold',
        alpha=central_alpha
    )

    # ------------- SATELLITE SYSTEM -------------
    satellite_pos = (3.5 * size_factor, 2.5 * size_factor)
    satellite_alpha = 0.9 if 'satellite' in components else 0.2
    subhalo_alpha = 1.0 if 'subhalo' in components else 0.2

    # Dark matter subhalo
    subhalo = Ellipse(
        xy=satellite_pos,
        width=2.2 * size_factor,
        height=1.4 * size_factor,
        angle=50,
        fc='lightgray',
        ec='gray',
        alpha=subhalo_alpha,
        zorder=4,
        hatch='////'
    )
    ax.add_patch(subhalo)

    # Subhalo label
    ax.text(
        satellite_pos[0] - 0.65 * size_factor,
        satellite_pos[1] + 1.0 * size_factor,
        'Subhalo',
        ha='center',
        color='dimgray',
        fontsize=13,
        weight='bold',
        alpha=subhalo_alpha,
        zorder=10
    )

    # Satellite galaxy
    satellite_angle = 50 + misalignment_angle

    satellite = Ellipse(
        xy=satellite_pos,
        width=1.4 * size_factor,
        height=0.6 * size_factor,
        angle=satellite_angle,
        fc=galaxy_color,
        ec=galaxy_color,
        alpha=satellite_alpha,
        zorder=4
    )
    ax.add_patch(satellite)

    if 'satellite_axis' in components:
        draw_axis(
            ax,
            satellite_pos,
            satellite_angle,
            1.2 * size_factor,
            color=clist[6],
            alpha=satellite_alpha
        )

    # Satellite galaxy label
    ax.text(
        satellite_pos[0],
        satellite_pos[1] - 1.5 * size_factor,
        'Satellite galaxy',
        ha='center',
        color=galaxy_color,
        fontsize=13,
        weight='bold',
        alpha=satellite_alpha
    )

    if 'subhalo_axis' in components:
        draw_axis(
            ax,
            satellite_pos,
            50,
            1.2 * size_factor,
            color='dimgray',
            alpha=subhalo_alpha,
            zorder=10
        )

    if 'subvec' in components:
        draw_axis(
            ax,
            satellite_pos,
            170,
            1.2 * size_factor,
            color=clist[4],
            alpha=0.9,
            zorder=10,
            show_dashed_axis=False
        )

        end_ponit = (
            satellite_pos[0] - 1.8 * size_factor,
            satellite_pos[1]
        )

        ax.text(
            *end_ponit,
            sat_sub_vec,
            fontsize=24,
            color=clist[4],
            alpha=0.9,
            weight='bold',
            zorder=20
        )

    if 'cenvec' in components:
        draw_axis(
            ax,
            central_host_halo_pos,
            170,
            1.2 * size_factor,
            color=clist[4],
            alpha=central_host_halo_alpha,
            zorder=10,
            show_dashed_axis=False
        )

        end_ponit = (
            central_host_halo_pos[0] - 1.8 * size_factor,
            central_host_halo_pos[1]
        )

        ax.text(
            *end_ponit,
            cen_vec,
            fontsize=24,
            color=clist[4],
            alpha=central_host_halo_alpha,
            weight='bold',
            zorder=20
        )

    # ------------- POSITION VECTOR -------------
    if 'position_vector' in components:
        vector_alpha = 1.0

        vector_midpoint = (
            central_pos[0] + satellite_pos[0] / 2 - 0.3 * size_factor,
            central_pos[1] + satellite_pos[1] / 2 + 0.1 * size_factor
        )

        ax.text(
            *vector_midpoint,
            r'$\vec{r}$',
            fontsize=24,
            color='dodgerblue',
            alpha=vector_alpha,
            weight='bold'
        )

        ax.arrow(
            central_pos[0],
            central_pos[1],
            satellite_pos[0],
            satellite_pos[1],
            width=0.03 * size_factor,
            color=DH[5],
            length_includes_head=True,
            zorder=5,
            head_width=0.2 * size_factor,
            alpha=vector_alpha
        )
    else:
        vector_alpha = 0.3

    # Final formatting
    ax.set_title(title, fontsize=16, pad=15, weight='bold')
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-5.1, 5.1)

    fig.tight_layout()

    return ax

def plot_mu(mu_set,x_set,muind,galaxy_halo_MA,bins=20,xlabel='',title='',color=None,Halo_info_ulim=None,Halo_info_dlim=None,
            make_plot=True,ncol=1,return_eb=False,logx=False,logy=False,in_plot='mu',ax=None,ylim=None,label=None,fmt='-o',show_progress=False):
    if type(mu_set)==str:
        mu_0=galaxy_halo_MA[mu_set][muind]
    else:
        mu_0=mu_set[muind]
    if type(x_set)==str:
        Halo_info0= galaxy_halo_MA[x_set][muind] 
    else:
        Halo_info0= x_set[muind]
    
    
    mu_sym=np.concatenate([mu_0, -mu_0])
    Halo_info=np.concatenate([Halo_info0, Halo_info0])
    if logx:
        logbin=np.logspace(start=np.log10(Halo_info_dlim), stop=np.log10(Halo_info_ulim), num=bins+1)
        Halo_info_bins=np.histogram(Halo_info[(Halo_info<Halo_info_ulim)&(Halo_info>Halo_info_dlim)],bins=logbin)
    else:
        Halo_info_bins=np.histogram(Halo_info[(Halo_info<Halo_info_ulim)&(Halo_info>Halo_info_dlim)],bins=bins)
    
    Halo_info_bin_edges=Halo_info_bins[1]
    
    Halo_info_bin_centres=np.zeros(bins)
    muses=np.zeros(bins)
    muserr=np.zeros(bins)
    ks=np.zeros(bins)
    kerr=np.zeros(bins)
    not_nan = np.ones(bins,dtype='bool')
    if show_progress:
        for ii in tqdm(range(bins)):
            binind=(Halo_info>Halo_info_bin_edges[ii])&(Halo_info<Halo_info_bin_edges[ii+1])
            Halo_info_bin_centres[ii] = np.mean(Halo_info[binind]) 
            try:
                results = dw.fit(mu_sym[binind])
                muses[ii]  =  results['mu']
                muserr[ii] =  results['mu_error']
                ks[ii]  =  results['kappa']
                kerr[ii] =  results['kappa_error']
            except:
                not_nan[ii]=False
    else:
        for ii in range(bins):
            binind=(Halo_info>Halo_info_bin_edges[ii])&(Halo_info<Halo_info_bin_edges[ii+1])
            Halo_info_bin_centres[ii] = np.mean(Halo_info[binind]) 
            try:
                results = dw.fit(mu_sym[binind])
                muses[ii]  =  results['mu']
                muserr[ii] =  results['mu_error']
                ks[ii]  =  results['kappa']
                kerr[ii] =  results['kappa_error']
            except:
                not_nan[ii]=False
        
    
    # plt.close()
    if make_plot:
        if ax is None:
            fig = plt.figure(figsize = (8,5))
            gs = fig.add_gridspec(1,1, hspace=0, wspace=0)
            ax = plt.subplot(gs[0,0])
        else:
            fig = ax.figure
        sns.set(style='ticks')
        plt.rcParams['xtick.direction']  = 'in'
        plt.rcParams['ytick.direction']  = 'in'
        plt.rcParams['mathtext.fontset'] = 'cm'

        # fmt = {'0': ('o', '-'), '2': ('s', '--')}
        if in_plot=='mu':
            ax.errorbar(Halo_info_bin_centres[not_nan],muses[not_nan],yerr=muserr[not_nan],fmt=fmt,color=color,label=label)
            ax.set_ylabel(r'$\mu$')
        elif in_plot=='kappa':
            ax.errorbar(Halo_info_bin_centres[not_nan],ks[not_nan],yerr=kerr[not_nan],fmt=fmt,color=color,label=label)
            ax.set_ylabel(r'$\kappa$')
        else:
            raise('????')
        if logx and logy:
            ax.loglog()
        elif logx:
            ax.semilogx()
        elif logy:
            ax.semilogy()
        else:
            pass
        if label:    
            ax.legend(frameon=False,ncol=1)
        fig.tight_layout()
        # ax.set_xlim(0,1.51)
        # ax.set_ylim(0.7,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        
        ax.set(ylim=ylim)
        
    if return_eb:
        return Halo_info_bin_centres,muses,muserr
    else:
        return ax

def plot_fit_with_errors(cos_theta, bins=30, 
                         hist_color='blue', hist_alpha=0.7, hist_lw=1.5,
                         fit_color='red', fit_alpha=1.0, fit_lw=2,
                         ci_color='red', ci_alpha=0.3,
                         return_ax=True):
    """
    Visualize cosθ data histogram with fitted Dimroth-Watson distribution and confidence intervals.
    
    This function takes an array of cosθ values, fits a Dimroth-Watson distribution using maximum likelihood estimation,
    and creates a comprehensive visualization showing:
    - Data histogram (step plot)
    - Fitted probability density function
    - Confidence interval band around the fit
    
    Parameters
    ----------
    cos_theta : array-like
        Input array of cosθ values (angular misalignment data)
    bins : int, optional
        Number of bins for the histogram (default: 30)
    hist_color : str, optional
        Color for the histogram step plot (default: 'blue')
    hist_alpha : float, optional
        Transparency level for the histogram (range: 0-1, default: 0.7)
    hist_lw : float, optional
        Line width for the histogram step plot (default: 1.5)
    fit_color : str, optional
        Color for the fitted distribution curve (default: 'red')
    fit_alpha : float, optional
        Transparency level for the fitted curve (range: 0-1, default: 1.0)
    fit_lw : float, optional
        Line width for the fitted curve (default: 2)
    ci_color : str, optional
        Color for the confidence interval band (default: 'red')
    ci_alpha : float, optional
        Transparency level for the confidence interval band (range: 0-1, default: 0.3)
    return_ax : bool, optional
        Whether to return the matplotlib Axes object for further customization (default: True)
        If False, displays the plot immediately and returns only the fit results.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object containing the visualization (only if return_ax=True)
    fit_result : dict
        Dictionary containing fitting results with keys:
        - 'mu': Best-fit alignment strength parameter
        - 'mu_error': Standard error of μ
        - 'kappa': Best-fit shape parameter κ
        - 'kappa_error': Standard error of κ
        - 'success': Optimization success flag
    
    Notes
    -----
    The visualization includes:
    1. Step histogram of input data
    2. Fitted Dimroth-Watson probability density curve
    3. 95% confidence interval band (μ ± 1.96*SE)
    4. Statistical annotation box with sample size and fit parameters
    5. Legend explaining plot elements
    
    Example Usage
    -------------
    >>> import numpy as np
    >>> from dimroth_watson import DimrothWatson
    >>> 
    >>> # Generate sample data
    >>> dw = DimrothWatson()
    >>> samples = dw.rvs(dw.mu_to_k(0.6), size=1000)
    >>> 
    >>> # Create plot with default parameters
    >>> ax, fit_result = plot_fit_with_errors(samples)
    >>> 
    >>> # Customize plot appearance
    >>> ax, fit_result = plot_fit_with_errors(
    ...     samples,
    ...     bins=40,
    ...     hist_color='darkgreen',
    ...     hist_alpha=0.8,
    ...     fit_color='purple',
    ...     ci_color='purple',
    ...     ci_alpha=0.2
    ... )
    >>> 
    >>> # Further customize the axes
    >>> ax.set_title("Customized Alignment Distribution")
    >>> ax.set_xlim(-1.1, 1.1)
    >>> plt.savefig("custom_fit.png")
    >>> plt.show()
    """
    # Initialize distribution and fit to data
    dw = DimrothWatson()
    fit_result = dw.fit(cos_theta,error_method="bootstrap")
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram as step plot (no fill)
    hist, bin_edges, _ = ax.hist(
        cos_theta, 
        bins=bins, 
        density=True, 
        histtype='step', 
        color=hist_color, 
        alpha=hist_alpha,
        linewidth=hist_lw,
        fill=False,
        label='Data Histogram'
    )
    
    # Generate points for fitted curve
    x = np.linspace(-1, 1, 500)
    mu = fit_result['mu']
    mu_error = fit_result['mu_error']
    
    # Calculate fitted PDF
    pdf_fit = dw.pdf_mu(x, mu)
    
    # Calculate 95% confidence interval bounds (μ ± 1.96*SE)
    pdf_upper = dw.pdf_mu(x, mu + 1.96 * mu_error)
    pdf_lower = dw.pdf_mu(x, mu - 1.96 * mu_error)
    
    # Plot confidence interval as shaded band
    ax.fill_between(
        x, pdf_lower, pdf_upper, 
        color=ci_color, 
        alpha=ci_alpha,
        label='95% Confidence Interval'
    )
    
    # Plot fitted PDF curve
    ax.plot(
        x, pdf_fit, 
        color=fit_color, 
        alpha=fit_alpha,
        linewidth=fit_lw,
        label=f'Fit: μ = {mu:.3f} ± {mu_error:.3f}'
    )
    
    # Add plot decorations
    ax.set_title('Dimroth-Watson Distribution Fit')
    ax.set_xlabel(r'$\cos\theta$')
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistical annotation
    stats_text = f'N = {len(cos_theta)}\nμ = {mu:.3f} ± {mu_error:.3f}'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Return based on user preference
    if return_ax:
        return ax, fit_result
    else:
        plt.show()
        return fit_result


from typing import Optional, Tuple, List

def _convex_hull_monotone_chain(pts: np.ndarray) -> np.ndarray:
    """Pure-NumPy 2D convex hull (monotone chain). Returns vertices CCW."""
    P = np.unique(pts, axis=0)
    if P.shape[0] <= 2:
        return P
    P = P[np.lexsort((P[:,1], P[:,0]))]
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in P:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(tuple(p))
    upper=[]
    for p in reversed(P):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1]+upper[:-1], dtype=float)

class KMeans2DClusterer:
    """
    2D k-means on (x,y). Uses scikit-learn if available, else a NumPy Lloyd fallback.
    After fit():
      - centers_ : (K,2)
      - labels_  : (N,)
      - masks_   : List[(N,) bool] per-cluster
    plot_boundaries(...): draw Voronoi (needs scipy) or per-cluster convex hulls.
    """

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4,
                 random_state: Optional[int] = None, n_init: int = 10):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = int(n_clusters)
        self.max_iter   = int(max_iter)
        self.tol        = float(tol)
        self.random_state = random_state
        self.n_init     = int(n_init)

        # fitted
        self.centers_: Optional[np.ndarray] = None
        self.labels_:  Optional[np.ndarray] = None
        self.masks_:   Optional[List[np.ndarray]] = None
        self._X_:      Optional[np.ndarray] = None  # cached data for hulls

    # ---------- public API ----------

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KMeans2DClusterer":
        """Run k-means and populate centers_, labels_, masks_."""
        X = self._stack_xy(x, y)
        self._X_ = X
        K = self.n_clusters
        # Try sklearn
        try:
            from sklearn.cluster import KMeans  # type: ignore
            km = KMeans(n_clusters=K, n_init=self.n_init, max_iter=self.max_iter,
                        tol=self.tol, random_state=self.random_state)
            labels = km.fit_predict(X)
            centers = km.cluster_centers_.astype(float)
        except Exception:
            labels, centers = self._kmeans_numpy(X, K, self.max_iter, self.tol,
                                                 self.random_state, self.n_init)
        self.centers_ = centers
        self.labels_  = labels.astype(int)
        self.masks_   = [(self.labels_ == k) for k in range(K)]
        return self

    def get_centers(self) -> np.ndarray:
        self._check_fitted()
        return self.centers_.copy()

    def get_numeric_labels(self) -> np.ndarray:
        self._check_fitted()
        return self.labels_.copy()

    def get_boolean_index(self) -> List[np.ndarray]:
        self._check_fitted()
        return [m.copy() for m in self.masks_]

    def plot_boundaries(self, ax, *,
                        mode: str = "voronoi",
                        linestyle: str = "--",
                        linewidth: float = 1.5,
                        color: str = "k",
                        alpha: float = 0.8) -> None:
        """Draw cluster boundaries on given Axes: mode='voronoi' or 'hull'."""
        self._check_fitted()
        if self.centers_.shape[0] < 2:
            return
        mode = mode.lower()
        if mode == "voronoi":
            self._plot_voronoi(ax, linestyle, linewidth, color, alpha)
        elif mode == "hull":
            self._plot_hulls(ax, linestyle, linewidth, color, alpha)
        else:
            raise ValueError("mode must be 'voronoi' or 'hull'")

    # ---------- internals ----------

    @staticmethod
    def _stack_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        if x.size != y.size:
            raise ValueError("x and y must have the same length")
        if x.size == 0:
            raise ValueError("empty input")
        return np.column_stack([x, y])

    def _check_fitted(self) -> None:
        if self.centers_ is None or self.labels_ is None:
            raise RuntimeError("Model is not fitted. Call .fit(x, y) first.")

    def _kmeans_numpy(self, X: np.ndarray, K: int, max_iter: int, tol: float,
                      random_state: Optional[int], n_init: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simple Lloyd's algorithm with multiple random restarts."""
        rng = np.random.default_rng(random_state)
        best_inertia = np.inf
        best_labels = None
        best_centers = None
        N = X.shape[0]

        for _ in range(max(1, n_init)):
            centers = X[rng.choice(N, size=K, replace=False)].copy()
            for _it in range(max_iter):
                d2 = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
                labels = np.argmin(d2, axis=1)
                new_centers = centers.copy()
                for k in range(K):
                    mk = (labels == k)
                    if mk.any():
                        new_centers[k] = X[mk].mean(axis=0)
                    else:
                        new_centers[k] = X[rng.integers(0, N)]
                shift = np.linalg.norm(new_centers - centers)
                centers = new_centers
                if shift <= tol:
                    break
            inertia = np.sum(((X - centers[labels])**2))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()
        return best_labels.astype(int), best_centers.astype(float)

    def _plot_voronoi(self, ax, linestyle, linewidth, color, alpha) -> None:
        """Finite Voronoi ridges of centers (requires scipy)."""
        try:
            from scipy.spatial import Voronoi  # type: ignore
        except Exception:
            # Fallback: draw hulls if scipy missing
            self._plot_hulls(ax, linestyle, linewidth, color, alpha)
            return
        vor = Voronoi(self.centers_)
        for v0, v1 in vor.ridge_vertices:
            if v0 >= 0 and v1 >= 0:
                p0, p1 = vor.vertices[v0], vor.vertices[v1]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha)

    def _plot_hulls(self, ax, linestyle, linewidth, color, alpha) -> None:
        """Convex hull of points in each cluster. Prefers scipy, else NumPy hull."""
        X = self._X_
        if X is None:  # should not happen after fit
            return
        try:
            from scipy.spatial import ConvexHull  # type: ignore
            use_scipy = True
        except Exception:
            use_scipy = False

        for k in range(self.n_clusters):
            pts = X[self.labels_ == k]
            if pts.shape[0] < 3:
                continue
            if use_scipy:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
            else:
                poly = _convex_hull_monotone_chain(pts)
                if poly.shape[0] < 3:
                    continue
            xx = np.r_[poly[:,0], poly[0,0]]
            yy = np.r_[poly[:,1], poly[0,1]]
            ax.plot(xx, yy, linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha)


# -----------------------------------------------------------------------------
# Shell-wise halo dynamics plotting utilities
# -----------------------------------------------------------------------------

def principal_plane_basis_from_points(positions, masses=None):
    """
    Return the principal-axis basis of a point cloud using the unnormalised
    shape tensor I_ij = sum m x_i x_j.

    Parameters
    ----------
    positions : array-like, shape (N, 3)
        Relative Cartesian coordinates.
    masses : array-like or None
        Particle masses. If None, unit weights are used.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Columns are major, intermediate, and minor axes.
    evals : ndarray, shape (3,)
        Eigenvalues sorted in descending order.
    """
    X = np.asarray(positions, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if masses is None:
        m = np.ones(X.shape[0], dtype=float)
    else:
        m = np.asarray(masses, dtype=float)
        if m.shape != (X.shape[0],):
            raise ValueError("masses must have shape (N,)")
    I = np.einsum("n,ni,nj->ij", m, X, X)
    I = 0.5 * (I + I.T)
    vals, vecs = np.linalg.eigh(I)
    order = np.argsort(vals)[::-1]
    return vecs[:, order], vals[order]


def project_to_principal_plane(positions, basis, length_factor=1.0):
    """
    Project 3D relative coordinates onto the major--intermediate-axis plane.

    Parameters
    ----------
    positions : array-like, shape (N, 3)
        Relative coordinates in simulation units.
    basis : ndarray, shape (3, 3)
        Principal-axis basis. Column 0 is major, column 1 intermediate.
    length_factor : float
        Multiplicative factor converting input length units to the desired
        plotting unit, e.g. physical kpc.
    """
    X = np.asarray(positions, dtype=float)
    R = np.asarray(basis, dtype=float)
    return np.column_stack([X @ R[:, 0], X @ R[:, 1]]) * float(length_factor)


def ellipse_from_projected_points(xy, weights=None, nsigma=2.0):
    """Mass-weighted covariance ellipse from projected 2D points."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 5:
        return None
    if weights is None:
        w = np.ones(xy.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (xy.shape[0],):
            raise ValueError("weights must have shape (N,)")
    good = np.isfinite(xy).all(axis=1) & np.isfinite(w) & (w > 0)
    if np.count_nonzero(good) < 5:
        return None
    xy = xy[good]
    w = w[good]
    cen = np.sum(xy * w[:, None], axis=0) / np.sum(w)
    Y = xy - cen[None, :]
    C = np.einsum("n,ni,nj->ij", w, Y, Y) / np.sum(w)
    vals, vecs = np.linalg.eigh(0.5 * (C + C.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
        return None
    width = 2.0 * float(nsigma) * np.sqrt(vals[0])
    height = 2.0 * float(nsigma) * np.sqrt(vals[1])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return cen, width, height, angle


def same_xy_limits(xy, pad_fraction=0.08, percentile=99.5):
    """Symmetric x/y limits for a projected point cloud."""
    xy = np.asarray(xy, dtype=float)
    lim = np.nanpercentile(np.abs(xy), percentile)
    if not np.isfinite(lim) or lim <= 0:
        lim = np.nanmax(np.abs(xy))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    lim *= 1.0 + float(pad_fraction)
    return (-lim, lim), (-lim, lim)


def add_kpc_scalebar(ax, xlim=None, ylim=None, length_kpc=None, label=None, loc=(0.08, 0.08)):
    """Add a simple black scale bar to a 2D kpc plot."""
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
    x0, x1 = map(float, xlim)
    y0, y1 = map(float, ylim)
    xrange = abs(x1 - x0)
    yrange = abs(y1 - y0)
    if length_kpc is None:
        raw = 0.18 * xrange
        pow10 = 10 ** np.floor(np.log10(max(raw, 1e-6)))
        candidates = np.array([1, 2, 5, 10], dtype=float) * pow10
        length_kpc = float(candidates[np.argmin(np.abs(candidates - raw))])
    if label is None:
        label = f"{length_kpc:g} kpc"
    xb = x0 + float(loc[0]) * xrange
    yb = y0 + float(loc[1]) * yrange
    ax.plot([xb, xb + length_kpc], [yb, yb], lw=3, color="black", solid_capstyle="butt")
    ax.text(xb + 0.5 * length_kpc, yb + 0.035 * yrange, label,
            ha="center", va="bottom", fontsize=9, color="black")
    return ax


def _contour_levels_from_hist(H, n_levels=6):
    """
    Build log-spaced contour levels from a 2D histogram.
    """
    vals = np.asarray(H, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 10:
        return None

    vmin = np.nanpercentile(vals, 60)
    vmax = np.nanpercentile(vals, 99.7)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None

    return np.geomspace(vmin, vmax, n_levels)


def plot_radial_shells_pretty(
    X, masses, shell_masks, basis,
    title=None,
    bins=180,
    ellipse_nsigma=2.0,
    output_path=None,
    contour_color="#a8f0ff",
    contour_lw=1.0,
):
    """
    Single square panel:
    projected density heatmap + radial-shell ellipses + density contours.
    """
    x, y = _project_to_basis(X, basis)
    xlim, ylim = _nice_limits(x, y, pad_frac=0.10)

    H, xe, ye = _mass_hist2d(x, y, masses, bins=bins, xlim=xlim, ylim=ylim)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("black")

    im = _imshow_heatmap(ax, H, xlim, ylim, cmap="magma")

    # ---- density contours ----
    levels = _contour_levels_from_hist(H, n_levels=6)
    if levels is not None:
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        XX, YY = np.meshgrid(xc, yc)
        Hc = np.array(H, copy=True)
        Hc[Hc <= 0] = np.nan

        ax.contour(
            XX, YY, Hc,
            levels=levels,
            colors=contour_color,
            linewidths=contour_lw,
            alpha=0.8,
        )

    # shell colors: cool -> warm
    shell_colors = plt.cm.viridis(np.linspace(0.12, 0.95, len(shell_masks)))

    for i, (mask, color) in enumerate(zip(shell_masks, shell_colors)):
        if mask is None or np.sum(mask) < 10:
            continue
        xs = x[mask]
        ys = y[mask]
        ws = masses[mask]
        ell = _ellipse_from_points_2d(xs, ys, w=ws, nsigma=ellipse_nsigma)
        if ell is None:
            continue

        patch = Ellipse(
            xy=ell["center"],
            width=ell["width"],
            height=ell["height"],
            angle=ell["angle"],
            fill=False,
            lw=2.1,
            edgecolor=color,
            alpha=0.97,
        )
        ax.add_patch(patch)

        # annotate shell index near ellipse edge
        tx = ell["center"][0] + 0.52 * ell["width"] * np.cos(np.radians(ell["angle"]))
        ty = ell["center"][1] + 0.52 * ell["width"] * np.sin(np.radians(ell["angle"]))
        ax.text(
            tx, ty, f"{i+1}",
            color=color,
            fontsize=10.5,
            weight="bold",
            ha="center", va="center",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.35, pad=1.5),
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("major-axis plane [kpc]")
    ax.set_ylabel("intermediate-axis plane [kpc]")
    ax.grid(False)

    # scale bar
    span = xlim[1] - xlim[0]
    barlen = _choose_scalebar_length(span)
    _add_scalebar(ax, xlim, ylim, barlen, color="white", lw=2.8, fontsize=10.5)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("projected particle mass density [arb. unit]")

    if title is not None:
        ax.set_title(title, pad=10)

    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    return fig, ax


def _binding_shell_label(i, n_shell, most_bound_first=True):
    """
    Human-readable shell label for binding-energy ordered shells.
    """
    rank = i + 1

    if most_bound_first:
        if i == 0:
            desc = "most bound"
        elif i == n_shell - 1:
            desc = "least bound"
        elif i < max(1, n_shell // 3):
            desc = "tightly bound"
        elif i >= max(1, 2 * n_shell // 3):
            desc = "loosely bound"
        else:
            desc = "intermediate"
    else:
        # If shell order is reversed
        if i == 0:
            desc = "least bound"
        elif i == n_shell - 1:
            desc = "most bound"
        elif i < max(1, n_shell // 3):
            desc = "loosely bound"
        elif i >= max(1, 2 * n_shell // 3):
            desc = "tightly bound"
        else:
            desc = "intermediate"

    return f"shell {rank} ({desc})"
def _project_to_basis(X, basis):
    """
    Project 3D positions onto the first two axes of basis.
    basis: shape (3, 3), columns or rows spanning principal basis
    """
    B = np.asarray(basis)
    if B.shape != (3, 3):
        raise ValueError("basis must have shape (3,3)")
    # use basis vectors as columns
    xp = X @ B[:, 0]
    yp = X @ B[:, 1]
    return xp, yp


def _weighted_cov_2d(x, y, w=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[good]
    y = y[good]
    w = w[good]

    if x.size < 3:
        return None

    wsum = np.sum(w)
    xm = np.sum(w * x) / wsum
    ym = np.sum(w * y) / wsum

    dx = x - xm
    dy = y - ym

    cxx = np.sum(w * dx * dx) / wsum
    cyy = np.sum(w * dy * dy) / wsum
    cxy = np.sum(w * dx * dy) / wsum

    cov = np.array([[cxx, cxy],
                    [cxy, cyy]], dtype=float)
    center = np.array([xm, ym], dtype=float)
    return center, cov


def _ellipse_from_points_2d(x, y, w=None, nsigma=2.0):
    out = _weighted_cov_2d(x, y, w=w)
    if out is None:
        return None

    center, cov = out
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # full width / height
    width = 2.0 * nsigma * np.sqrt(max(evals[0], 0.0))
    height = 2.0 * nsigma * np.sqrt(max(evals[1], 0.0))
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))

    return {
        "center": center,
        "width": width,
        "height": height,
        "angle": angle,
    }


def _nice_limits(x, y, pad_frac=0.08):
    vals = np.concatenate([np.asarray(x), np.asarray(y)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-10, 10), (-10, 10)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin)

    if not np.isfinite(span) or span <= 0:
        span = 1.0

    span *= (1.0 + 2.0 * pad_frac)

    half = 0.5 * span
    return (cx - half, cx + half), (cy - half, cy + half)


def _choose_scalebar_length(span_kpc):
    """
    Choose a nice scale-bar length in kpc.
    """
    candidates = np.array([2, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000], dtype=float)
    target = 0.22 * span_kpc
    idx = np.argmin(np.abs(candidates - target))
    return float(candidates[idx])


def _add_scalebar(ax, xlim, ylim, length_kpc, color="white", lw=2.5, fontsize=10):
    x0, x1 = xlim
    y0, y1 = ylim

    dx = x1 - x0
    dy = y1 - y0

    xb = x0 + 0.08 * dx
    yb = y0 + 0.08 * dy

    ax.plot([xb, xb + length_kpc], [yb, yb], color=color, lw=lw, solid_capstyle="butt")
    ax.text(
        xb + 0.5 * length_kpc,
        yb + 0.035 * dy,
        f"{int(length_kpc) if float(length_kpc).is_integer() else length_kpc:g} kpc",
        color=color,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        weight="bold",
    )


def _mass_hist2d(x, y, w, bins=180, xlim=None, ylim=None):
    if xlim is None:
        xlim = (np.nanmin(x), np.nanmax(x))
    if ylim is None:
        ylim = (np.nanmin(y), np.nanmax(y))

    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[xlim, ylim], weights=w)
    # transpose for imshow
    return H.T, xe, ye


def _imshow_heatmap(ax, H, xlim, ylim, cmap="magma", vmin=None, vmax=None):
    Hplot = np.array(H, copy=True)
    Hplot[Hplot <= 0] = np.nan

    if np.isfinite(np.nanmax(Hplot)):
        if vmin is None:
            positive = Hplot[np.isfinite(Hplot)]
            vmin = np.nanpercentile(positive, 5) if positive.size else 1.0
            vmax = np.nanpercentile(positive, 99.5) if positive.size else 1.0
            vmin = max(vmin, np.nanmin(positive)) if positive.size else 1.0
            if vmax <= vmin:
                vmax = vmin * 1.5
        im = ax.imshow(
            Hplot,
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=cmap,
            norm=LogNorm(vmin=max(vmin, 1e-12), vmax=vmax),
            interpolation="nearest",
            aspect="equal",
        )
    else:
        im = ax.imshow(
            np.zeros_like(Hplot),
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=cmap,
            aspect="equal",
        )
    return im

def plot_binding_shell_panels_pretty(
    X, masses, shell_masks, basis,
    title=None,
    bins=120,
    ellipse_nsigma=2.0,
    output_path=None,
    most_bound_first=True,
    contour_color="#a8f0ff",
    contour_lw=0.9,
):
    """
    Multi-panel square subplots:
    each panel shows the shell particle mass density heatmap.
    All panels share the same x/y range and scale.

    By default, shell_masks are assumed to be ordered from
    most bound -> least bound.
    """
    x, y = _project_to_basis(X, basis)
    xlim, ylim = _nice_limits(x, y, pad_frac=0.10)

    n_shell = len(shell_masks)
    ncol = min(3, n_shell)
    nrow = int(math.ceil(n_shell / ncol))

    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(4.8 * ncol, 4.8 * nrow),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    # Precompute all heatmaps to get a common color scale
    H_list = []
    positive_vals = []

    for mask in shell_masks:
        if mask is None or np.sum(mask) < 5:
            H = np.zeros((bins, bins))
        else:
            H, _, _ = _mass_hist2d(
                x[mask], y[mask], masses[mask],
                bins=bins, xlim=xlim, ylim=ylim
            )
        H_list.append(H)
        vals = H[np.isfinite(H) & (H > 0)]
        if vals.size:
            positive_vals.append(vals)

    if positive_vals:
        all_pos = np.concatenate(positive_vals)
        vmin = max(np.nanpercentile(all_pos, 10), np.nanmin(all_pos))
        vmax = np.nanpercentile(all_pos, 99.5)
        if vmax <= vmin:
            vmax = vmin * 1.5
    else:
        vmin, vmax = 1.0, 2.0

    shell_colors = plt.cm.viridis(np.linspace(0.12, 0.95, n_shell))
    last_im = None

    # need edges for contours
    _, xe, ye = _mass_hist2d(x, y, masses, bins=bins, xlim=xlim, ylim=ylim)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    XX, YY = np.meshgrid(xc, yc)

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("black")

        if i >= n_shell:
            ax.set_axis_off()
            continue

        mask = shell_masks[i]
        H = H_list[i]
        last_im = _imshow_heatmap(ax, H, xlim, ylim, cmap="magma", vmin=vmin, vmax=vmax)

        # contours
        levels = _contour_levels_from_hist(H, n_levels=5)
        if levels is not None:
            Hc = np.array(H, copy=True)
            Hc[Hc <= 0] = np.nan
            ax.contour(
                XX, YY, Hc,
                levels=levels,
                colors=contour_color,
                linewidths=contour_lw,
                alpha=0.8,
            )

        # ellipse
        if mask is not None and np.sum(mask) >= 10:
            ell = _ellipse_from_points_2d(x[mask], y[mask], w=masses[mask], nsigma=ellipse_nsigma)
            if ell is not None:
                patch = Ellipse(
                    xy=ell["center"],
                    width=ell["width"],
                    height=ell["height"],
                    angle=ell["angle"],
                    fill=False,
                    lw=2.0,
                    edgecolor=shell_colors[i],
                    alpha=0.98,
                )
                ax.add_patch(patch)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")

        label = _binding_shell_label(i, n_shell, most_bound_first=most_bound_first)
        ax.set_title(label, color="black", pad=6)

        # in-panel label
        ax.text(
            0.03, 0.97,
            label,
            transform=ax.transAxes,
            ha="left", va="top",
            color="white",
            fontsize=9.8,
            weight="bold",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.35, pad=2.0),
        )

        ax.set_xlabel("major-axis plane [kpc]")
        ax.set_ylabel("intermediate-axis plane [kpc]")

        span = xlim[1] - xlim[0]
        barlen = _choose_scalebar_length(span)
        _add_scalebar(ax, xlim, ylim, barlen, color="white", lw=2.5, fontsize=9.5)

    if title is not None:
        fig.suptitle(title, y=1.02)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
        cbar.set_label("projected particle mass density [arb. unit]")

    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    return fig, axes
