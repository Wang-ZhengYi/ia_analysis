


import numpy as np

def gen_nfw(rs=1.0, size=1000, r_min=0.01, r_max=10.0):
    """
    Generate 3D points distributed according to an NFW radial density profile.

    Parameters
    ----------
    rs : float
        Scale radius of the NFW profile (default 1.0).
    size : int
        Number of points to generate (default 1000).
    r_min : float
        Minimum radius from origin (default 0.01).
    r_max : float
        Maximum radius to sample from (default 10.0).

    Returns
    -------
    xyz : ndarray
        Array of shape (size, 3) containing Cartesian coordinates.
    """
    # Sample sky positions (RA, Dec) uniformly on a sphere
    ra = np.random.uniform(0, 2 * np.pi, size)
    dec_uniform = np.random.uniform(-1, 1, size)  # uniform in sin(dec)
    dec = np.arcsin(dec_uniform)  # actual Dec in radians

    
    # Sample radial distances according to the power-law profile
    r_grid = np.linspace(r_min, r_max, 1000)
    
    # Calculate NFW number density (proportional to r²ρ(r))
    # NFW density: ρ(r) ∝ 1/(r(rs + r)^2)
    # Radial PDF for points: dN/dr ∝ r² ρ(r) ∝ r/(rs + r)^2
    radial_pdf = r_grid/ (rs + r_grid)**2#np.exp(-r_grid**2/rs**2/2) #
    radial_pdf /= np.trapz(radial_pdf, r_grid)  # normalize
    cdf = np.cumsum(radial_pdf) * (r_grid[1] - r_grid[0])
    u = np.random.rand(size)
    r = np.interp(u, cdf, r_grid)

    # Convert to Cartesian coordinates
    x = r * np.cos(dec) * np.cos(ra)
    y = r * np.cos(dec) * np.sin(ra)
    z = r * np.sin(dec)
    # plt.plot(r_grid,radial_pdf)

    return np.column_stack((x, y, z))






def transform_points_to_ellipsoid(points, a, b, c, principal_axis):
    """
    Transform points on a unit sphere to an ellipsoid with specified dimensions and orientation
    
    Parameters:
        points: Array of points (N,3) - expected to be on a unit sphere
        a, b, c: Semi-axes lengths of the ellipsoid
        principal_axis: Main axis direction vector (will be normalized)
        
    Returns:
        transformed_points: Transformed points (N,3)
    """

    # norms = np.linalg.norm(points, axis=1, keepdims=True)
    # points_normalized = points / np.where(norms == 0, 1, norms)
    
    # Normalize principal axis
    axis_norm = np.linalg.norm(principal_axis)
    if axis_norm < 1e-12:
        raise ValueError("Principal axis vector cannot be zero")
    u = principal_axis / axis_norm
    
    # Create a rotation matrix to align the x-axis to the principal axis
    R = create_rotation_matrix(u)
    
    # Transform points
    unrotat_points = points*np.array([a, b, c])
    rotated_points = unrotat_points@ R.T

    
    return unrotat_points,rotated_points

@njit
def create_rotation_matrix(principal_axis):
    """Create a rotation matrix that maps the x-axis onto `principal_axis`.

    Notes
    -----
    This is written to be Numba-friendly (no np.allclose).
    Uses Rodrigues' rotation formula with careful handling of near-parallel cases.
    """
    # Normalize principal_axis
    norms = np.linalg.norm(principal_axis)
    if norms == 0.0:
        # Numba-friendly: return identity for pathological input
        return np.eye(3)
    principal_axis = principal_axis / norms

    # Target axis to rotate from
    x_axis = np.array([1.0, 0.0, 0.0])

    # cos(theta) between x_axis and principal_axis
    c = principal_axis[0]  # dot(x_axis, principal_axis)

    # If nearly parallel to +x, return identity
    if 1.0 - c < 1e-12:
        return np.eye(3)

    # If nearly parallel to -x, return 180-degree rotation about y (or z)
    if 1.0 + c < 1e-12:
        return np.array([[-1.0,  0.0,  0.0],
                         [ 0.0,  1.0,  0.0],
                         [ 0.0,  0.0, -1.0]])

    # Rotation axis v = x_axis × principal_axis
    v = np.cross(x_axis, principal_axis)
    s = np.linalg.norm(v)
    # Build cross-product matrix [v]_x
    vx = np.array([[0.0,   -v[2],  v[1]],
                   [v[2],   0.0,  -v[0]],
                   [-v[1],  v[0],  0.0]])

    # Rodrigues: R = I + [v]_x + [v]_x^2 * (1-c)/s^2
    R = np.eye(3) + vx + (vx @ vx) * (1.0 - c) / (s * s)
    return R
