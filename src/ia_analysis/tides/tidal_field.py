"""Potential-field gridding and tidal tensor construction.

Purpose
-------
This module converts particle or mass samples into gridded potential and tidal
fields, then interpolates tidal tensors back to halo or galaxy positions.

Provides
--------
- Numba-accelerated Cloud-In-Cell mass assignment.
- Fourier-space potential and tidal tensor solvers.
- Acceleration-derived tidal tensor support with consistent sign conventions.
- Regular-grid interpolation utilities for downstream catalog pipelines.

Notes
-----
The public builders use a shared ``legacy_tidal_sign`` option so mass-derived,
potential-derived, and acceleration-derived tensors can be compared with the
same convention.
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.interpolate import RegularGridInterpolator
from numba import njit, set_num_threads

# Sign convention note
# --------------------
# All public tidal builders below use the same ``legacy_tidal_sign`` flag:
#   True  -> T_ij(k) = + k_i k_j Phi(k)
#   False -> T_ij(k) = - k_i k_j Phi(k)
# For acceleration samples, with a = -grad Phi, the equivalent Fourier-space
# expression is T_ij(k) = + i k_j A_i(k) for the legacy=True convention.
# This keeps mass-derived, potential-derived, and acceleration-derived tidal
# tensors on the same sign convention and in the same units.

# =========================
# Low-level: CIC assignment
# =========================
@njit(fastmath=True)
def _assign_mass_CIC(positions, masses, Nx, Ny, Nz, bbox_min, dx, dy, dz):
    """
    Numba-accelerated Cloud-In-Cell (CIC) mass assignment.
    Accumulates particle masses to grid cell corners with trilinear weights.
    Returns a mass grid (not yet divided by cell volume).

    Parameters
    ----------
    positions : (N,3) float64
        Particle positions in ckpc/h
    masses : (N,) float64
        Particle masses in Msun
    Nx, Ny, Nz : int
        Grid dimensions
    bbox_min : (3,) float64
        Minimum corner of bounding box in ckpc/h
    dx, dy, dz : float
        Grid spacings in ckpc/h

    Returns
    -------
    density_field : (Nx, Ny, Nz) ndarray
        Mass density grid in Msun/(ckpc/h)³ (after division by cell volume externally)
    """
    density_field = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    for n in range(positions.shape[0]):
        x, y, z = positions[n, :]
        m = masses[n]

        # Base cell indices
        i = int(np.floor((x - bbox_min[0]) / dx))
        j = int(np.floor((y - bbox_min[1]) / dy))
        k = int(np.floor((z - bbox_min[2]) / dz))

        # Clamp indices to valid range for trilinear stencil
        i = max(0, min(i, Nx - 2))
        j = max(0, min(j, Ny - 2))
        k = max(0, min(k, Nz - 2))

        # Fractional position within cell
        x0 = bbox_min[0] + i * dx
        y0 = bbox_min[1] + j * dy
        z0 = bbox_min[2] + k * dz
        tx = max(0.0, min((x - x0) / dx, 1.0))
        ty = max(0.0, min((y - y0) / dy, 1.0))
        tz = max(0.0, min((z - z0) / dz, 1.0))

        # Trilinear weights
        w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
        w100 = tx * (1.0 - ty) * (1.0 - tz)
        w010 = (1.0 - tx) * ty * (1.0 - tz)
        w110 = tx * ty * (1.0 - tz)
        w001 = (1.0 - tx) * (1.0 - ty) * tz
        w101 = tx * (1.0 - ty) * tz
        w011 = (1.0 - tx) * ty * tz
        w111 = tx * ty * tz

        # Accumulate mass to the 8 surrounding nodes
        density_field[i,   j,   k  ] += m * w000
        density_field[i+1, j,   k  ] += m * w100
        density_field[i,   j+1, k  ] += m * w010
        density_field[i+1, j+1, k  ] += m * w110
        density_field[i,   j,   k+1] += m * w001
        density_field[i+1, j,   k+1] += m * w101
        density_field[i,   j+1, k+1] += m * w011
        density_field[i+1, j+1, k+1] += m * w111

    return density_field


@njit(fastmath=True)
def _assign_scalar_CIC(positions, values, Nx, Ny, Nz, bbox_min, dx, dy, dz):
    """
    Numba-accelerated CIC assignment for an arbitrary scalar field.
    Accumulates weighted sums and weights for later normalization.

    Parameters
    ----------
    positions : (N,3) float64
        Sample positions in ckpc/h
    values : (N,) float64
        Scalar values at sample points
    Nx, Ny, Nz : int
        Grid dimensions
    bbox_min : (3,) float64
        Minimum corner of bounding box in ckpc/h
    dx, dy, dz : float
        Grid spacings in ckpc/h

    Returns
    -------
    sum_grid : (Nx, Ny, Nz) ndarray
        Weighted sum of values
    weight_grid : (Nx, Ny, Nz) ndarray
        Accumulated weights for normalization
    """
    sum_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    weight_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    for n in range(positions.shape[0]):
        x, y, z = positions[n, :]
        v = values[n]

        # Base cell indices
        i = int(np.floor((x - bbox_min[0]) / dx))
        j = int(np.floor((y - bbox_min[1]) / dy))
        k = int(np.floor((z - bbox_min[2]) / dz))

        # Clamp indices
        i = max(0, min(i, Nx - 2))
        j = max(0, min(j, Ny - 2))
        k = max(0, min(k, Nz - 2))

        # Fractional position within cell
        x0 = bbox_min[0] + i * dx
        y0 = bbox_min[1] + j * dy
        z0 = bbox_min[2] + k * dz
        tx = max(0.0, min((x - x0) / dx, 1.0))
        ty = max(0.0, min((y - y0) / dy, 1.0))
        tz = max(0.0, min((z - z0) / dz, 1.0))

        # Trilinear weights
        w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
        w100 = tx * (1.0 - ty) * (1.0 - tz)
        w010 = (1.0 - tx) * ty * (1.0 - tz)
        w110 = tx * ty * (1.0 - tz)
        w001 = (1.0 - tx) * (1.0 - ty) * tz
        w101 = tx * (1.0 - ty) * tz
        w011 = (1.0 - tx) * ty * tz
        w111 = tx * ty * tz

        # Weighted sum and weights
        sum_grid[i,   j,   k  ] += v * w000
        sum_grid[i+1, j,   k  ] += v * w100
        sum_grid[i,   j+1, k  ] += v * w010
        sum_grid[i+1, j+1, k  ] += v * w110
        sum_grid[i,   j,   k+1] += v * w001
        sum_grid[i+1, j,   k+1] += v * w101
        sum_grid[i,   j+1, k+1] += v * w011
        sum_grid[i+1, j+1, k+1] += v * w111

        weight_grid[i,   j,   k  ] += w000
        weight_grid[i+1, j,   k  ] += w100
        weight_grid[i,   j+1, k  ] += w010
        weight_grid[i+1, j+1, k  ] += w110
        weight_grid[i,   j,   k+1] += w001
        weight_grid[i+1, j,   k+1] += w101
        weight_grid[i,   j+1, k+1] += w011
        weight_grid[i+1, j+1, k+1] += w111

    return sum_grid, weight_grid


# =========================
# Grid helpers (coordinates)
# =========================
def _make_grid_and_spacing(positions, grid_size=256, boundary_padding=0.1):
    """
    Create a padded bounding box and regular grid enclosing particle positions.

    Parameters
    ----------
    positions : (N,3) ndarray
        Particle positions in ckpc/h
    grid_size : int or tuple
        Grid resolution (single value or per-axis)
    boundary_padding : float
        Fractional padding relative to data extent

    Returns
    -------
    grid_dims : tuple
        (Nx, Ny, Nz) grid dimensions
    grid_coords : tuple
        (xcoords, ycoords, zcoords) grid coordinates
    spacings : tuple
        (dx, dy, dz) grid spacings
    bbox : tuple
        (bbox_min, bbox_max) bounding box corners
    """
    positions = np.asarray(positions, dtype=np.float64)

    if isinstance(grid_size, int):
        grid_dims = (grid_size, grid_size, grid_size)
    else:
        grid_dims = tuple(int(g) for g in grid_size)

    Nx, Ny, Nz = grid_dims

    pos_min = np.min(positions, axis=0)
    pos_max = np.max(positions, axis=0)
    extent = pos_max - pos_min
    pad = boundary_padding * extent

    bbox_min = pos_min - pad
    bbox_max = pos_max + pad

    xcoords = np.linspace(bbox_min[0], bbox_max[0], Nx)
    ycoords = np.linspace(bbox_min[1], bbox_max[1], Ny)
    zcoords = np.linspace(bbox_min[2], bbox_max[2], Nz)

    dx = xcoords[1] - xcoords[0]
    dy = ycoords[1] - ycoords[0]
    dz = zcoords[1] - zcoords[0]

    return grid_dims, (xcoords, ycoords, zcoords), (dx, dy, dz), (bbox_min, bbox_max)


def _spacing_from_grid_coords(grid_coords):
    """
    Extract grid dimensions and spacings from coordinate arrays.

    Parameters
    ----------
    grid_coords : tuple
        (xcoords, ycoords, zcoords) coordinate arrays

    Returns
    -------
    grid_dims : tuple
        (Nx, Ny, Nz) grid dimensions
    spacings : tuple
        (dx, dy, dz) grid spacings
    """
    xcoords, ycoords, zcoords = grid_coords
    Nx, Ny, Nz = len(xcoords), len(ycoords), len(zcoords)
    if Nx < 2 or Ny < 2 or Nz < 2:
        raise ValueError("Grid must have at least 2 points along each axis")
    dx = float(xcoords[1] - xcoords[0])
    dy = float(ycoords[1] - ycoords[0])
    dz = float(zcoords[1] - zcoords[0])
    return (Nx, Ny, Nz), (dx, dy, dz)


def _k_grids(Nx, Ny, Nz, dx, dy, dz):
    """
    Construct wave-number grids for spectral operations.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid dimensions
    dx, dy, dz : float
        Grid spacings

    Returns
    -------
    Kx, Ky, Kz : ndarray
        Wave-number components in Fourier space
    k_sq : ndarray
        Squared magnitude of wave-number vectors
    """
    kx = 2.0 * np.pi * fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * fftfreq(Nz, d=dz)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_sq = Kx**2 + Ky**2 + Kz**2
    return Kx, Ky, Kz, k_sq


# ============================================
# High-level: potential + tidal tensor pipeline
# ============================================
def compute_gravitational_potential(positions, masses, grid_size=256,
                                     boundary_padding=0.1, softening=0.01,
                                     G=4.302e-3, h=0.7, nthreads=None,
                                     legacy_tidal_sign=True):
    """
    Compute gravitational potential and tidal tensor from particle positions
    by solving Poisson's equation for the CIC-gridded density.

    Returns a dict with flattened arrays:
        {
          'coordinates': (M,3),
          'potential': (M,),
          'Txx','Txy','Txz','Tyy','Tyz','Tzz': (M,)
        }

    Units
    -----
    positions: ckpc/h
    masses: Msun
    potential: km²/s²
    tidal tensor: km²/s²/(ckpc/h)²
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_particles = positions.shape[0]

    if np.isscalar(masses):
        masses = np.full(n_particles, float(masses), dtype=np.float64)
    else:
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape[0] != n_particles:
            raise ValueError("Masses must match number of positions")

    if nthreads is not None:
        try:
            set_num_threads(int(nthreads))
        except Exception:
            pass

    # Effective G for comoving coords (simple h factor; adapt as needed)
    G_effective = G * h

    grid_dims, grid_coords, spacings, bbox = _make_grid_and_spacing(
        positions, grid_size, boundary_padding
    )
    Nx, Ny, Nz = grid_dims
    dx, dy, dz = spacings
    bbox_min, bbox_max = bbox
    cell_volume = dx * dy * dz

    # Density field via CIC
    density_field = _assign_mass_CIC(
        positions, masses, Nx, Ny, Nz,
        np.asarray(bbox_min, dtype=np.float64), dx, dy, dz
    )
    density_field /= cell_volume  # Msun/(ckpc/h)^3

    # Poisson solve in Fourier space
    rho_k = fftn(density_field)
    Kx, Ky, Kz, k_sq = _k_grids(Nx, Ny, Nz, dx, dy, dz)

    # Simple Fourier softening
    box_size = bbox_max[0] - bbox_min[0]
    softening_eff = softening * 2 * np.pi / box_size
    denom = k_sq + softening_eff**2
    denom[0, 0, 0] = np.inf  # avoid div-by-zero on the zero mode

    phi_k = -4.0 * np.pi * G_effective * rho_k / denom
    potential_field = np.real(ifftn(phi_k))

    # Tidal tensor from Φ(k):  T_ij(k) = ± k_i k_j Φ(k)
    sign = 1.0 if legacy_tidal_sign else -1.0
    tidal_components = [
        np.real(ifftn(sign * (Kx*Kx) * phi_k)),  # Txx
        np.real(ifftn(sign * (Kx*Ky) * phi_k)),  # Txy
        np.real(ifftn(sign * (Kx*Kz) * phi_k)),  # Txz
        np.real(ifftn(sign * (Ky*Ky) * phi_k)),  # Tyy
        np.real(ifftn(sign * (Ky*Kz) * phi_k)),  # Tyz
        np.real(ifftn(sign * (Kz*Kz) * phi_k)),  # Tzz
    ]
    Txx, Txy, Txz, Tyy, Tyz, Tzz = tidal_components

    xcoords, ycoords, zcoords = grid_coords
    X, Y, Z = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')

    coords_flat = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    res = {
        'coordinates': coords_flat,
        'potential': potential_field.ravel(),
        'Txx': Txx.ravel(),
        'Txy': Txy.ravel(),
        'Txz': Txz.ravel(),
        'Tyy': Tyy.ravel(),
        'Tyz': Tyz.ravel(),
        'Tzz': Tzz.ravel()
    }
    return res


# ======================================================
# Grid-based potential/acceleration -> tidal computation
# ======================================================
def grid_potential_and_tidal(positions, potentials, grid_size=256,
                             boundary_padding=0.1, grid_coords=None,
                             fill_value=0.0, legacy_tidal_sign=True,
                             input_type='potential',
                             reconstruct_potential_from_acc=True):
    """
    Grid a scalar *potential* field OR a vector *acceleration* field, then compute the tidal tensor.

    Default behavior (backward compatible):
      - Assumes `potentials` is scalar Φ(x) sampled at `positions`,
        grids it with CIC, FFTs it, and returns T_ij = ∂_i∂_j Φ.

    New option:
      - If `input_type='acceleration'`, then `potentials` must be the *acceleration* a(x) with shape (N,3).
        Using a = -∇Φ, the physical tensor is T_ij = -∂_j a_i.
        Here the returned sign follows the same ``legacy_tidal_sign`` convention
        used by the mass/potential branches.  For legacy_tidal_sign=True this
        means T_ij(k) = + i k_j A_i(k), equivalent to +k_i k_j Φ(k).
        Optionally reconstruct Φ via Φ(k) = i (k·A(k)) / (k^2 + eps) with zero mode set to 0.

    Units
    -----
    positions: ckpc/h
    potentials: km²/s² (if input_type='potential')
    accelerations: km/s² (if input_type='acceleration')
    tidal tensor: km²/s²/(ckpc/h)²

    Parameters
    ----------
    positions : (N,3) ndarray
        Sample positions in ckpc/h
    potentials : (N,) or (N,3) ndarray
        - If input_type='potential' (default): scalar Φ at positions
        - If input_type='acceleration': vector a at positions
    grid_size : int or tuple
        Grid resolution (default 256)
    boundary_padding : float
        Fractional padding added around the sample extent (default 0.1)
    grid_coords : tuple or None
        Predefined grid coordinates (xcoords, ycoords, zcoords); build if None
    fill_value : float
        Value for grid nodes without contributing samples (default 0.0)
    legacy_tidal_sign : bool
        If False, flips the overall sign of the tidal tensor
    input_type : {'potential','acceleration'}
        Select interpretation of the second argument; default 'potential' (backward compatible)
    reconstruct_potential_from_acc : bool
        Only used when input_type='acceleration'; if True, also reconstruct Φ from a

    Returns
    -------
    res : dict
        {
          'coordinates': (M,3),  # grid points (ckpc/h)
          'potential': (M,),      # gridded Φ or reconstructed Φ (zeros if disabled)
          'Txx','Txy','Txz','Tyy','Tyz','Tzz': (M,)
        }
    """
    positions = np.asarray(positions, dtype=np.float64)

    # Grid construction or use provided coordinates
    if grid_coords is None:
        grid_dims, grid_coords, spacings, bbox = _make_grid_and_spacing(
            positions, grid_size, boundary_padding
        )
        Nx, Ny, Nz = grid_dims
        dx, dy, dz = spacings
        bbox_min = bbox[0]
    else:
        xcoords, ycoords, zcoords = grid_coords
        grid_dims, spacings = _spacing_from_grid_coords(grid_coords)
        Nx, Ny, Nz = grid_dims
        dx, dy, dz = spacings
        bbox_min = np.array([xcoords[0], ycoords[0], zcoords[0]], dtype=np.float64)

    Kx, Ky, Kz, k_sq = _k_grids(Nx, Ny, Nz, dx, dy, dz)
    eps = 1e-30
    sign = 1.0 if legacy_tidal_sign else -1.0

    mode = str(input_type).lower().strip()
    if mode == 'potential':
        # ----- Old behavior (unchanged by default) -----
        potentials = np.asarray(potentials, dtype=np.float64)
        if potentials.ndim != 1 or potentials.shape[0] != positions.shape[0]:
            raise ValueError("For input_type='potential', `potentials` must be a 1D array (N,)")

        sum_grid, weight_grid = _assign_scalar_CIC(
            positions, potentials, Nx, Ny, Nz,
            np.asarray(bbox_min, dtype=np.float64), dx, dy, dz
        )
        potential_grid = np.divide(
            sum_grid,
            weight_grid + eps,
            where=weight_grid > eps,
            out=np.full_like(sum_grid, fill_value)
        )

        Phi_k = fftn(potential_grid)
        tidal_components = [
            np.real(ifftn(sign * (Kx*Kx) * Phi_k)),  # Txx
            np.real(ifftn(sign * (Kx*Ky) * Phi_k)),  # Txy
            np.real(ifftn(sign * (Kx*Kz) * Phi_k)),  # Txz
            np.real(ifftn(sign * (Ky*Ky) * Phi_k)),  # Tyy
            np.real(ifftn(sign * (Ky*Kz) * Phi_k)),  # Tyz
            np.real(ifftn(sign * (Kz*Kz) * Phi_k)),  # Tzz
        ]
        Txx, Txy, Txz, Tyy, Tyz, Tzz = tidal_components

    elif mode == 'acceleration':
        # ----- New behavior: use acceleration samples -----
        acc = np.asarray(potentials, dtype=np.float64)  # reuse arg name to keep signature unchanged
        if acc.ndim != 2 or acc.shape[1] != 3 or acc.shape[0] != positions.shape[0]:
            raise ValueError("For input_type='acceleration', `potentials` must be an array (N,3) of accelerations")

        # Grid each acceleration component with CIC and normalize
        Ax_sum, Ax_w = _assign_scalar_CIC(
            positions, acc[:, 0], Nx, Ny, Nz,
            np.asarray(bbox_min, dtype=np.float64), dx, dy, dz
        )
        Ay_sum, Ay_w = _assign_scalar_CIC(
            positions, acc[:, 1], Nx, Ny, Nz,
            np.asarray(bbox_min, dtype=np.float64), dx, dy, dz
        )
        Az_sum, Az_w = _assign_scalar_CIC(
            positions, acc[:, 2], Nx, Ny, Nz,
            np.asarray(bbox_min, dtype=np.float64), dx, dy, dz
        )
        Ax = np.divide(Ax_sum, Ax_w + eps, where=Ax_w > eps, out=np.full_like(Ax_sum, fill_value))
        Ay = np.divide(Ay_sum, Ay_w + eps, where=Ay_w > eps, out=np.full_like(Ay_sum, fill_value))
        Az = np.divide(Az_sum, Az_w + eps, where=Az_w > eps, out=np.full_like(Az_sum, fill_value))

        # Keep the acceleration-derived tensor on the same sign convention as
        # the mass/potential branches.  With a = -grad Phi,
        #     + i k_j A_i(k) = + k_i k_j Phi(k),
        # which is the legacy_tidal_sign=True convention used above.
        ik = 1j
        acc_sign = 1.0 if legacy_tidal_sign else -1.0
        Ax_k = fftn(Ax); Ay_k = fftn(Ay); Az_k = fftn(Az)

        Txx = np.real(ifftn(acc_sign * ik * Kx * Ax_k))
        Txy = np.real(ifftn(acc_sign * ik * Ky * Ax_k))
        Txz = np.real(ifftn(acc_sign * ik * Kz * Ax_k))

        Tyx = np.real(ifftn(acc_sign * ik * Kx * Ay_k))
        Tyy = np.real(ifftn(acc_sign * ik * Ky * Ay_k))
        Tyz = np.real(ifftn(acc_sign * ik * Kz * Ay_k))

        Tzx = np.real(ifftn(acc_sign * ik * Kx * Az_k))
        Tzy = np.real(ifftn(acc_sign * ik * Ky * Az_k))
        Tzz = np.real(ifftn(acc_sign * ik * Kz * Az_k))

        # Symmetrize the tensor (should be symmetric for conservative fields)
        Txy = 0.5 * (Txy + Tyx)
        Txz = 0.5 * (Txz + Tzx)
        Tyz = 0.5 * (Tyz + Tzy)

        # Optionally reconstruct Φ from a:  Φ(k) = i (k·A)/k^2
        if reconstruct_potential_from_acc:
            divA_k = (Kx * Ax_k + Ky * Ay_k + Kz * Az_k)
            Phi_k = 1j * divA_k / (k_sq + eps)
            Phi_k[0, 0, 0] = 0.0  # remove zero mode (Φ is defined up to an additive constant)
            potential_grid = np.real(ifftn(Phi_k))
        else:
            potential_grid = np.zeros_like(Txx)

    else:
        raise ValueError("`input_type` must be 'potential' (default) or 'acceleration'.")

    # Pack flattened grid to point-cloud-like dict (same schema as before)
    xcoords, ycoords, zcoords = grid_coords
    X, Y, Z = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')
    coords_flat = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    res = {
        'coordinates': coords_flat,
        'potential': potential_grid.ravel(),
        'Txx': Txx.ravel(),
        'Txy': Txy.ravel(),
        'Txz': Txz.ravel(),
        'Tyy': Tyy.ravel(),
        'Tyz': Tyz.ravel(),
        'Tzz': Tzz.ravel()
    }
    return res


# ======================
# Interpolation utilities
# ======================
class PotentialInterpolator:
    """
    Interpolator for potential and tidal tensor fields on a regular grid.

    Usage:
        interp = PotentialInterpolator(grid_coords, potential_field, tidal_tensor_field)
        potential, tidal_tensor = interp(position)
    """
    def __init__(self, grid_coords, potential_field, tidal_tensor_field,
                 bounds_error=False, fill_value=0.0):
        """
        Parameters
        ----------
        grid_coords : tuple
            (xcoords, ycoords, zcoords) grid coordinates
        potential_field : (Nx, Ny, Nz) ndarray
            Potential values on grid
        tidal_tensor_field : (6, Nx, Ny, Nz) ndarray
            Tidal tensor components on grid in order [xx, xy, xz, yy, yz, zz]
        bounds_error : bool
            Whether to raise error for out-of-bounds queries
        fill_value : float
            Value for out-of-bounds queries
        """
        self.grid_coords = grid_coords
        self.phi_interp = RegularGridInterpolator(
            grid_coords, potential_field,
            method='linear',
            bounds_error=bounds_error,
            fill_value=fill_value
        )
        self.tidal_interps = [
            RegularGridInterpolator(
                grid_coords, comp,
                method='linear',
                bounds_error=bounds_error,
                fill_value=fill_value
            )
            for comp in tidal_tensor_field
        ]

    def __call__(self, position):
        """
        Interpolate potential and tidal tensor at a given position.

        Parameters
        ----------
        position : (3,) array_like
            Query position in ckpc/h

        Returns
        -------
        potential : float
            Interpolated potential in km²/s²
        tidal_tensor : (3,3) ndarray
            Interpolated tidal tensor in km²/s²/(ckpc/h)²
        """
        pos = np.atleast_2d(position)
        potential = self.phi_interp(pos)[0]
        components = [interp(pos)[0] for interp in self.tidal_interps]
        Txx, Txy, Txz, Tyy, Tyz, Tzz = components
        tidal_tensor = np.array([
            [Txx, Txy, Txz],
            [Txy, Tyy, Tyz],
            [Txz, Tyz, Tzz]
        ])
        return potential, tidal_tensor


def interpolate_potential_and_tidal(position, grid_coords, potential_field, tidal_tensor_field):
    """
    Single-point interpolation of potential and tidal tensor.

    Parameters
    ----------
    position : (3,) array_like
        Query position in ckpc/h
    grid_coords : tuple
        (xcoords, ycoords, zcoords) grid coordinates
    potential_field : (Nx, Ny, Nz) ndarray
        Potential values on grid
    tidal_tensor_field : (6, Nx, Ny, Nz) ndarray
        Tidal tensor components on grid in order [xx, xy, xz, yy, yz, zz]

    Returns
    -------
    potential : float
        Interpolated potential in km²/s²
    tidal_tensor : (3,3) ndarray
        Interpolated tidal tensor in km²/s²/(ckpc/h)²
    """
    phi_interp = RegularGridInterpolator(
        grid_coords, potential_field,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    potential = phi_interp(np.atleast_2d(position))[0]

    Txx, Txy, Txz, Tyy, Tyz, Tzz = tidal_tensor_field
    tidal_interps = [
        RegularGridInterpolator(
            grid_coords, comp,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        for comp in (Txx, Txy, Txz, Tyy, Tyz, Tzz)
    ]
    vals = [interp(np.atleast_2d(position))[0] for interp in tidal_interps]
    Txx, Txy, Txz, Tyy, Tyz, Tzz = vals

    tidal_tensor = np.array([
        [Txx, Txy, Txz],
        [Txy, Tyy, Tyz],
        [Txz, Tyz, Tzz]
    ])
    return potential, tidal_tensor



