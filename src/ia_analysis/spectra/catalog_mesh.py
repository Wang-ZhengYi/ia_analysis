# CatMesh.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CatMesh.py

Catalog mesh utilities for periodic simulation boxes using Pylians3 MAS_library.

This module builds, from a *galaxy catalog* (positions, optional velocities, optional shapes/spins):

1) Tracer overdensity mesh:
   - g_mesh = n/<n> - 1

2) Velocity meshes from galaxy velocities (two modes):
   - vmesh="theta": theta_mesh = -(∇·v)/(a H(z))   (dimensionless)
   - vmesh="vector": vx_mesh, vy_mesh, vz_mesh     (km/s in the rotated LOS frame)

   The velocity meshes are number-weighted by default; you may pass `v_weights`
   (e.g., stellar mass weights) to construct a weighted velocity field:
        v(x) = (Σ w v)/(Σ w)

3) Intrinsic-shape meshes from either:
   - I_matrices: per-galaxy 3x3 shape tensors -> gamma(+/x) -> E/B
   - spin_vectors: per-galaxy spins -> epsilon(+/x) -> gamma(+/x) -> E/B
   - e1,e2: direct (+/x) components (only supported when LOS=(0,0,1))

Common features:
- Periodic box; optional self-folding (L -> L/fold, positions wrapped into [0,L))
- Arbitrary LOS: rotate so LOS -> +z, then compute (+/x) and E/B in that frame.

Returned dict keys (always present unless stated):
- g_mesh
- (optional) theta_mesh  if vmesh="theta"
- (optional) vx_mesh, vy_mesh, vz_mesh if vmesh="vector"
- E_mesh, B_mesh, gamma_plus_mesh, gamma_cross_mesh (zeros if no shapes)
- gamma_plus, gamma_cross, finite_shape_mask
- meta: bookkeeping (shot noise, shape noise, normalization, etc.)

Dependencies:
- numpy
- MAS_library (Pylians3)
- pyccl (needed only for vmesh="theta" normalization or for pos_unit physical units)

Optional:
- Iana.py with epsilon_from_shape_matrix, epsilon_from_spin
"""

from dataclasses import dataclass

import numpy as np
try:
    import MAS_library as MASL
except Exception:  # pragma: no cover - optional HPC dependency
    MASL = None
try:
    import pyccl as ccl
except Exception:  # pragma: no cover - optional cosmology dependency
    ccl = None


def _shape_projectors():
    try:
        from ia_analysis.shapes.Iana import epsilon_from_shape_matrix, epsilon_from_spin
    except Exception as exc:
        try:
            from Iana import epsilon_from_shape_matrix, epsilon_from_spin
        except Exception as legacy_exc:
            raise ImportError(
                "Catalog IA mesh construction requires ia_analysis.shapes.Iana "
                "for epsilon_from_shape_matrix and epsilon_from_spin."
            ) from legacy_exc
        if epsilon_from_shape_matrix is None:  # pragma: no cover - defensive
            raise ImportError("epsilon_from_shape_matrix is unavailable.") from exc
    return epsilon_from_shape_matrix, epsilon_from_spin


# ----------------------------
# small math helpers
# ----------------------------
class MeshMath:
    @staticmethod
    def normalize_los(los):
        v = np.asarray(los, dtype=float)
        if v.shape != (3,):
            raise ValueError("los must be length-3.")
        n = np.linalg.norm(v)
        if n == 0.0:
            raise ValueError("los must be non-zero.")
        return v / n

    @staticmethod
    def wrap_positions(pos, boxsize):
        return np.mod(pos, float(boxsize))

    @staticmethod
    def rotation_matrix_to_z(los_unit):
        z = np.array([0.0, 0.0, 1.0], dtype=float)
        a = np.asarray(los_unit, dtype=float)

        if np.allclose(a, z, atol=1e-12):
            return np.eye(3)

        if np.allclose(a, -z, atol=1e-12):
            R = np.eye(3)
            R[1, 1] = -1.0
            R[2, 2] = -1.0
            return R

        v = np.cross(a, z)
        s = np.linalg.norm(v)
        c = float(np.dot(a, z))

        vx = np.array(
            [[0.0, -v[2], v[1]],
             [v[2], 0.0, -v[0]],
             [-v[1], v[0], 0.0]],
            dtype=float,
        )
        R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
        return R

    @staticmethod
    def rotate_vectors(x, R):
        x = np.asarray(x, dtype=float)
        return (R @ x.T).T

    @staticmethod
    def rotate_tensors(I, R):
        I = np.asarray(I, dtype=float)
        return np.einsum("ik,nkl,jl->nij", R, I, R, optimize=True)

    @staticmethod
    def convert_length_to_mpc_h(x, unit, cosmo):
        unit = str(unit).strip().lower()
        if unit in ("mpc/h", "mpch", "mpcph", "mpc_per_h"):
            return x
        if unit in ("kpc/h", "kpch", "kpcph", "kpc_per_h"):
            return x / 1000.0
        if unit == "mpc":
            if cosmo is None:
                raise ValueError("unit='Mpc' requires cosmo (to get h).")
            return x * float(cosmo["h"])
        if unit == "kpc":
            if cosmo is None:
                raise ValueError("unit='kpc' requires cosmo (to get h).")
            return x * float(cosmo["h"]) / 1000.0
        raise ValueError("unit must be one of {'Mpc/h','kpc/h','Mpc','kpc'}.")


@dataclass
class CatalogMeshConfig:
    boxsize: float
    nmesh: int
    mas_gal: str = "CIC"
    mas_shape: str = "CIC"
    los: tuple = (0.0, 0.0, 1.0)
    pos_unit: str = "Mpc/h"


class CatalogMeshBuilder:
    def __init__(self, config, cosmo=None):
        self.cfg = config
        self.cosmo = cosmo
        self._los_unit = MeshMath.normalize_los(self.cfg.los)
        self._R = MeshMath.rotation_matrix_to_z(self._los_unit)

    @property
    def rotation_matrix(self):
        return self._R.copy()

    # -------------- painting primitives --------------
    @staticmethod
    def _require_mas_library():
        if MASL is None:
            raise ImportError("CatalogMeshBuilder requires MAS_library from Pylians3.")

    def _paint_counts(self, pos_mpc_h, mas, L):
        self._require_mas_library()
        mesh = np.zeros((self.cfg.nmesh, self.cfg.nmesh, self.cfg.nmesh), dtype=np.float32)
        MASL.MA(pos_mpc_h.astype(np.float32), mesh, float(L), str(mas))
        return mesh

    def _paint_weighted(self, pos_mpc_h, weights, mas, L):
        self._require_mas_library()
        mesh = np.zeros((self.cfg.nmesh, self.cfg.nmesh, self.cfg.nmesh), dtype=np.float32)
        MASL.MA(pos_mpc_h.astype(np.float32), mesh, float(L), str(mas), W=np.asarray(weights, dtype=np.float32))
        return mesh

    @staticmethod
    def _counts_to_overdensity(counts):
        mean = float(np.mean(counts, dtype=np.float64))
        if mean <= 0.0:
            return np.zeros_like(counts, dtype=np.float32)
        out = counts.astype(np.float32) / np.float32(mean)
        out -= np.float32(1.0)
        return out

    @staticmethod
    def _safe_divide(num, den, eps=1e-30):
        out = np.zeros_like(num, dtype=np.float32)
        m = den > eps
        out[m] = (num[m] / (den[m] + eps)).astype(np.float32)
        return out

    # -------------- cosmology helper --------------
    @staticmethod
    def _Hz_kms_per_Mpc(cosmo, z):
        if ccl is None:
            raise ImportError("vmesh='theta' requires pyccl.")
        a = 1.0 / (1.0 + float(z))
        Ez = float(ccl.h_over_h0(cosmo, a))
        H0 = 100.0 * float(cosmo["h"])
        return H0 * Ez

    # -------------- E/B from gamma meshes --------------
    def _eb_from_gamma_mesh(self, gamma_plus, gamma_cross, L):
        gp = np.asarray(gamma_plus, dtype=np.float32)
        gx = np.asarray(gamma_cross, dtype=np.float32)
        if gp.shape != gx.shape or gp.ndim != 3:
            raise ValueError("gamma meshes must be 3D and same shape.")

        N = gp.shape[0]
        dx = float(L) / float(N)

        k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        kperp2 = (KX * KX + KY * KY).astype(np.float64)
        mask = kperp2 > 0.0

        cos2phi = np.zeros((N, N), dtype=np.float64)
        sin2phi = np.zeros((N, N), dtype=np.float64)
        cos2phi[mask] = (KX[mask] * KX[mask] - KY[mask] * KY[mask]) / kperp2[mask]
        sin2phi[mask] = (2.0 * KX[mask] * KY[mask]) / kperp2[mask]
        cos2phi = cos2phi[:, :, None]
        sin2phi = sin2phi[:, :, None]

        gp_k = np.fft.fftn(gp)
        gx_k = np.fft.fftn(gx)

        E_k = gp_k * cos2phi + gx_k * sin2phi
        B_k = -gp_k * sin2phi + gx_k * cos2phi

        E_k[~mask, :] = 0.0
        B_k[~mask, :] = 0.0

        E = np.fft.ifftn(E_k).real.astype(np.float32)
        B = np.fft.ifftn(B_k).real.astype(np.float32)
        return E, B

    # -------------- gamma projections --------------
    @staticmethod
    def _responsivity_from_epsilon(e1, e2, clip_R_min=0.05):
        e1 = np.asarray(e1, dtype=np.float64)
        e2 = np.asarray(e2, dtype=np.float64)
        good = np.isfinite(e1) & np.isfinite(e2)
        if not np.any(good):
            return float(clip_R_min)
        e2mean = float(np.mean(e1[good] ** 2 + e2[good] ** 2))
        R = 1.0 - 0.5 * e2mean
        if not np.isfinite(R):
            R = clip_R_min
        return float(max(R, clip_R_min))

    def _project_shapes_to_gamma(self, I_rot):
        N = I_rot.shape[0]
        los_z = np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=float), (N, 3))
        epsilon_from_shape_matrix, _ = _shape_projectors()
        gp, gx = epsilon_from_shape_matrix(I_rot, los_z)  # type: ignore
        return np.asarray(gp, dtype=float), np.asarray(gx, dtype=float)

    def _project_spin_to_gamma(self, L_rot, apply_responsivity=True, responsivity=None, clip_R_min=0.05):
        L_rot = np.asarray(L_rot, dtype=float)
        if L_rot.ndim != 2 or L_rot.shape[1] != 3:
            raise ValueError("spin_vectors must have shape (N,3).")
        N = L_rot.shape[0]
        los_z = np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=float), (N, 3))
        _, epsilon_from_spin = _shape_projectors()
        eplus, ecross = epsilon_from_spin(L_rot, los_z)  # type: ignore
        eplus = np.asarray(eplus, dtype=float)
        ecross = np.asarray(ecross, dtype=float)
        if not apply_responsivity:
            return eplus, ecross
        R = float(responsivity) if responsivity is not None else self._responsivity_from_epsilon(eplus, ecross, clip_R_min=clip_R_min)
        return eplus / (2.0 * R), ecross / (2.0 * R)

    def _project_e1e2_to_gamma(self, e1, e2, e_are_gamma=True, apply_responsivity=True, responsivity=None, clip_R_min=0.05):
        e1 = np.asarray(e1, dtype=float)
        e2 = np.asarray(e2, dtype=float)
        if e1.shape != e2.shape:
            raise ValueError("e1 and e2 must have same shape.")
        if e_are_gamma:
            return e1, e2
        if not apply_responsivity:
            return e1, e2
        R = float(responsivity) if responsivity is not None else self._responsivity_from_epsilon(e1, e2, clip_R_min=clip_R_min)
        return e1 / (2.0 * R), e2 / (2.0 * R)

    def _gamma_meshes_and_noise_from_components(self, pos_use, g_plus, g_cross, L, mask_finite_shapes=True, verbose=False):
        g_plus = np.asarray(g_plus, dtype=float)
        g_cross = np.asarray(g_cross, dtype=float)

        finite = np.isfinite(g_plus) & np.isfinite(g_cross)
        if verbose:
            print(f"[CatMesh] finite gamma fraction = {np.mean(finite):.6f}")

        if mask_finite_shapes:
            pos_p = pos_use[finite]
            gp = g_plus[finite]
            gx = g_cross[finite]
        else:
            pos_p = pos_use
            gp = np.where(finite, g_plus, 0.0)
            gx = np.where(finite, g_cross, 0.0)

        if gp.size > 0:
            sigma2_gp = float(np.mean(np.asarray(gp, dtype=np.float64) ** 2))
            sigma2_gx = float(np.mean(np.asarray(gx, dtype=np.float64) ** 2))
            shape_sigma2 = 0.5 * (sigma2_gp + sigma2_gx)
        else:
            shape_sigma2 = 0.0

        sum_gp = self._paint_weighted(pos_p, gp, self.cfg.mas_shape, L)
        sum_gx = self._paint_weighted(pos_p, gx, self.cfg.mas_shape, L)
        counts = self._paint_counts(pos_p, self.cfg.mas_shape, L)
        mean_counts = float(np.mean(counts, dtype=np.float64))

        if mean_counts <= 0.0:
            zmesh = np.zeros((self.cfg.nmesh, self.cfg.nmesh, self.cfg.nmesh), dtype=np.float32)
            return dict(
                gamma_plus_mesh=zmesh,
                gamma_cross_mesh=zmesh,
                gamma_plus=g_plus,
                gamma_cross=g_cross,
                finite_mask=finite,
                N_shape=int(pos_p.shape[0]),
                shape_sigma2=float(shape_sigma2),
            )

        gp_mesh = (sum_gp / np.float32(mean_counts)).astype(np.float32, copy=False)
        gx_mesh = (sum_gx / np.float32(mean_counts)).astype(np.float32, copy=False)
        return dict(
            gamma_plus_mesh=gp_mesh,
            gamma_cross_mesh=gx_mesh,
            gamma_plus=g_plus,
            gamma_cross=g_cross,
            finite_mask=finite,
            N_shape=int(pos_p.shape[0]),
            shape_sigma2=float(shape_sigma2),
        )

    # -------------- velocity meshes --------------
    def _build_velocity_meshes(self, pos_use, vel_use, L, vmesh, z, v_weights=None):
        """
        Build either theta_mesh or vector meshes from catalog velocities.

        pos_use : (N,3) positions in Mpc/h (rotated frame, wrapped)
        vel_use : (N,3) velocities in km/s (rotated frame)
        v_weights : (N,) optional weights used to define v(x) = Σw v / Σw
        """
        N = int(self.cfg.nmesh)
        vmesh = str(vmesh).strip().lower()

        if v_weights is None:
            w = np.ones(pos_use.shape[0], dtype=np.float64)
        else:
            w = np.asarray(v_weights, dtype=np.float64)
            if w.shape != (pos_use.shape[0],):
                raise ValueError("v_weights must have shape (N,) matching pos/vel length.")

        rho = self._paint_weighted(pos_use, w, self.cfg.mas_gal, L)
        px = self._paint_weighted(pos_use, w * vel_use[:, 0], self.cfg.mas_gal, L)
        py = self._paint_weighted(pos_use, w * vel_use[:, 1], self.cfg.mas_gal, L)
        pz = self._paint_weighted(pos_use, w * vel_use[:, 2], self.cfg.mas_gal, L)

        vx = self._safe_divide(px, rho)
        vy = self._safe_divide(py, rho)
        vz = self._safe_divide(pz, rho)

        if vmesh == "vector":
            return dict(vx_mesh=vx, vy_mesh=vy, vz_mesh=vz)

        if vmesh != "theta":
            raise ValueError("vmesh must be 'theta' or 'vector'.")

        if self.cosmo is None:
            raise ValueError("vmesh='theta' requires cosmo (pyccl.Cosmology).")
        if z is None:
            raise ValueError("vmesh='theta' requires z.")

        vx_k = np.fft.fftn(vx)
        vy_k = np.fft.fftn(vy)
        vz_k = np.fft.fftn(vz)

        dx = float(L) / float(N)
        k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")

        div_k = 1j * (KX * vx_k + KY * vy_k + KZ * vz_k)  # (km/s)/(Mpc/h)
        div_x = np.fft.ifftn(div_k).real.astype(np.float32)

        a = 1.0 / (1.0 + float(z))
        Hz = self._Hz_kms_per_Mpc(self.cosmo, float(z))  # km/s/Mpc
        h = float(self.cosmo["h"])
        div_per_mpc = div_x * h  # (km/s)/Mpc
        theta = -div_per_mpc / (a * Hz)

        return dict(theta_mesh=theta.astype(np.float32, copy=False))

    # -------------- main build --------------
    def build(
        self,
        pos,
        vel=None,
        vmesh=None,          # None | "theta" | "vector"
        v_weights=None,      # weights for velocity field definition
        I_matrices=None,
        spin_vectors=None,
        e1=None,
        e2=None,
        los=None,
        space="real",
        z=None,              # used for vmesh="theta" and for RSD if space="rsd"
        e_are_gamma=True,
        apply_responsivity=True,
        responsivity=None,
        mask_finite_shapes=True,
        verbose=False,
        folding_factor=1,
        boxsize_override=None,
    ):
        # LOS override
        if los is not None:
            los_unit = MeshMath.normalize_los(los)
            R = MeshMath.rotation_matrix_to_z(los_unit)
        else:
            R = self._R

        # box + folding
        base_L = float(self.cfg.boxsize) if boxsize_override is None else float(boxsize_override)
        fold = int(folding_factor)
        if fold < 1:
            raise ValueError("folding_factor must be >= 1.")
        L = base_L / float(fold)
        V = L ** 3

        # positions -> Mpc/h -> wrap -> rotate -> wrap
        pos = np.asarray(pos, dtype=float)
        pos_mpc_h = MeshMath.convert_length_to_mpc_h(pos, self.cfg.pos_unit, self.cosmo)
        pos_mpc_h = MeshMath.wrap_positions(pos_mpc_h, L)
        pos_rot = MeshMath.wrap_positions(MeshMath.rotate_vectors(pos_mpc_h, R), L)

        # optional velocities: rotate to LOS frame (km/s)
        vel_rot = None
        if vel is not None:
            vel = np.asarray(vel, dtype=float)
            if vel.shape != pos.shape:
                raise ValueError("vel must have the same shape as pos (N,3).")
            vel_rot = MeshMath.rotate_vectors(vel, R)

        space = str(space).strip().lower()
        if space not in ("real", "rsd"):
            raise ValueError("space must be 'real' or 'rsd'.")

        # For catalog velocity meshes, we require real-space positions (otherwise ambiguous)
        if vmesh is not None and space != "real":
            raise ValueError("Velocity meshes (vmesh) require space='real'.")

        pos_use = pos_rot

        # tracer overdensity g
        Ng = int(pos_use.shape[0])
        counts_g = self._paint_counts(pos_use, self.cfg.mas_gal, L)
        g_mesh = self._counts_to_overdensity(counts_g)
        nbar_gal = Ng / V if Ng > 0 else 0.0
        shotnoise_gg = (1.0 / nbar_gal) if nbar_gal > 0 else None

        out = dict(g_mesh=g_mesh)

        # velocity meshes
        if vmesh is not None:
            if vel_rot is None:
                raise ValueError("vmesh requested but vel is None.")
            out.update(self._build_velocity_meshes(pos_use, vel_rot, L, vmesh, z, v_weights=v_weights))

        # shapes input selection
        have_I = I_matrices is not None
        have_spin = spin_vectors is not None
        have_e = (e1 is not None) or (e2 is not None)

        if have_e and not ((e1 is not None) and (e2 is not None)):
            raise ValueError("If providing (e1,e2), both must be provided.")
        if (int(have_I) + int(have_spin) + int(have_e)) > 1:
            raise ValueError("Provide only one of: I_matrices OR spin_vectors OR (e1,e2).")

        shape_source = "none"

        if have_I:
            shape_source = "shape"
            I = np.asarray(I_matrices, dtype=float)
            if I.ndim != 3 or I.shape[1:] != (3, 3) or I.shape[0] != pos.shape[0]:
                raise ValueError("I_matrices must have shape (N,3,3) and match pos length.")
            I_rot = MeshMath.rotate_tensors(I, R)
            gp, gx = self._project_shapes_to_gamma(I_rot)
            gam = self._gamma_meshes_and_noise_from_components(pos_use, gp, gx, L, mask_finite_shapes=mask_finite_shapes, verbose=verbose)

        elif have_spin:
            shape_source = "spin"
            Lv = np.asarray(spin_vectors, dtype=float)
            if Lv.ndim != 2 or Lv.shape[1] != 3 or Lv.shape[0] != pos.shape[0]:
                raise ValueError("spin_vectors must have shape (N,3) and match pos length.")
            L_rot = MeshMath.rotate_vectors(Lv, R)
            gp, gx = self._project_spin_to_gamma(L_rot, apply_responsivity=bool(apply_responsivity), responsivity=responsivity)
            gam = self._gamma_meshes_and_noise_from_components(pos_use, gp, gx, L, mask_finite_shapes=mask_finite_shapes, verbose=verbose)

        elif have_e:
            shape_source = "e1e2"
            if not np.allclose(R, np.eye(3), atol=1e-8):
                raise NotImplementedError("e1/e2 input only supported for LOS=(0,0,1). Use I_matrices or spin_vectors.")
            gp, gx = self._project_e1e2_to_gamma(
                np.asarray(e1, dtype=float),
                np.asarray(e2, dtype=float),
                e_are_gamma=bool(e_are_gamma),
                apply_responsivity=bool(apply_responsivity),
                responsivity=responsivity,
            )
            gam = self._gamma_meshes_and_noise_from_components(pos_use, gp, gx, L, mask_finite_shapes=mask_finite_shapes, verbose=verbose)

        else:
            zmesh = np.zeros((self.cfg.nmesh, self.cfg.nmesh, self.cfg.nmesh), dtype=np.float32)
            gam = dict(
                gamma_plus_mesh=zmesh,
                gamma_cross_mesh=zmesh,
                gamma_plus=np.zeros(pos_use.shape[0], dtype=float),
                gamma_cross=np.zeros(pos_use.shape[0], dtype=float),
                finite_mask=np.zeros(pos_use.shape[0], dtype=bool),
                N_shape=0,
                shape_sigma2=0.0,
            )

        E_mesh, B_mesh = self._eb_from_gamma_mesh(gam["gamma_plus_mesh"], gam["gamma_cross_mesh"], L)

        Nshape = int(gam["N_shape"])
        nbar_shape = Nshape / V if Nshape > 0 else 0.0
        shape_sigma2 = float(gam["shape_sigma2"])
        shape_noise = (shape_sigma2 / nbar_shape) if nbar_shape > 0 else None

        meta = dict(
            boxsize=float(L),
            volume=float(V),
            nmesh=int(self.cfg.nmesh),
            mas_gal=str(self.cfg.mas_gal),
            mas_shape=str(self.cfg.mas_shape),
            folding_factor=int(fold),
            space="real",
            shape_source=str(shape_source),
            los_rotated=(0.0, 0.0, 1.0),
            rotation_matrix=np.asarray(R, dtype=float),

            N_gal=int(Ng),
            nbar_gal=float(nbar_gal),
            shotnoise_gg=None if shotnoise_gg is None else float(shotnoise_gg),

            N_shape=int(Nshape),
            nbar_shape=float(nbar_shape),
            shape_sigma2=float(shape_sigma2),
            shape_noise_EE=None if shape_noise is None else float(shape_noise),
            shape_noise_BB=None if shape_noise is None else float(shape_noise),
        )

        out.update(
            dict(
                E_mesh=E_mesh,
                B_mesh=B_mesh,
                gamma_plus_mesh=gam["gamma_plus_mesh"],
                gamma_cross_mesh=gam["gamma_cross_mesh"],
                gamma_plus=gam["gamma_plus"],
                gamma_cross=gam["gamma_cross"],
                finite_shape_mask=gam["finite_mask"],
                meta=meta,
            )
        )
        if vmesh is not None:
            meta["vmesh"] = str(vmesh).strip().lower()
            meta["v_weighting"] = "weighted" if v_weights is not None else "uniform"
            if meta["vmesh"] == "theta":
                meta["z"] = float(z)
                a = 1.0 / (1.0 + float(z))
                Hz = self._Hz_kms_per_Mpc(self.cosmo, float(z))
                meta["a"] = float(a)
                meta["Hz_kms_per_Mpc"] = float(Hz)

        return out
