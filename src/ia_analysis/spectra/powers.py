# powers.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
powers.py
=========

Compute power spectrum multipoles (ell = 0,2,4) from mesh fields using Pylians3.

This module provides a class-based estimator that:
- Accepts a dict of 3D meshes (cubic, same Nmesh).
- Computes auto- and cross-power spectra multipoles using Pylians ``Pk``/``XPk``.
- Optionally applies a survey window mesh and returns pseudo-P(k) (mode coupling not deconvolved).
- Appends shot noise and shape noise estimates into the output dict if provided via `meta`.

Conventions
-----------
- ``boxsize`` is the physical side length of the mesh being Fourier transformed, in Mpc/h.
  For self-folded fields this must be the folded length, e.g. ``Lbox / fold``.
- ``power_norm_boxsize`` optionally sets the box-size convention for returned P(k).
  If ``power_norm_boxsize=Lbox`` and ``boxsize=Lbox/fold``, all returned spectra
  are multiplied by ``fold**3``. This restores the original simulation-volume
  P(k) normalization while keeping the folded k-grid.
- Pylians multipoles require axis-aligned LOS (x/y/z).

Noise output
------------
If `meta` contains the following keys (as produced by Mesh.py):
- shotnoise_gg
- shape_noise_EE, shape_noise_BB
- (optional) shotnoise_dd (if you built delta from particles and kept its meta)

then the estimator writes:

out["noise"] = {
  "shotnoise_gg": ...,
  "shotnoise_dd": ... (if available),
  "shape_noise_EE": ...,
  "shape_noise_BB": ...
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import Pk_library as PKL


@dataclass
class PowerConfig:
    """
    Configuration for power spectrum estimation.

    Parameters
    ----------
    boxsize : float
        Physical side length of the mesh passed to Pylians, in Mpc/h.
        For self-folded meshes this should be the folded length Lbox/fold.
    power_norm_boxsize : float, optional
        Box-size convention for returned power spectra. If None, no volume
        correction is applied and P(k) is normalized to ``boxsize``.
        If set to the original simulation box size while ``boxsize`` is the
        folded box size, spectra are multiplied by (power_norm_boxsize/boxsize)^3.
    los : sequence of float
        LOS direction. Must be axis-aligned for Pylians (e.g., (0,0,1)).
    ells : sequence of int
        Multipoles to return. Must be subset of {0,2,4}.
    threads : int
        Threads for Pylians.
    """

    boxsize: float
    power_norm_boxsize: Optional[float] = None
    los: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    ells: Tuple[int, int, int] = (0, 2, 4)
    threads: int = 8


class PowerSpectrumEstimator:
    """Compute auto- and cross-power spectrum multipoles from mesh fields."""

    def __init__(self, config: PowerConfig) -> None:
        self.cfg = config
        self._axis = self._los_axis(self.cfg.los)
        self._ells = tuple(int(e) for e in self.cfg.ells)
        if any(e not in (0, 2, 4) for e in self._ells):
            raise ValueError("Supported multipoles: ell in {0,2,4}.")

        if float(self.cfg.boxsize) <= 0.0:
            raise ValueError("boxsize must be positive.")
        if self.cfg.power_norm_boxsize is not None and float(self.cfg.power_norm_boxsize) <= 0.0:
            raise ValueError("power_norm_boxsize must be positive when provided.")

    @staticmethod
    def _as_field(f: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(f)
        if arr.ndim != 3 or arr.shape[0] != arr.shape[1] or arr.shape[0] != arr.shape[2]:
            raise ValueError(f"Field '{name}' must be a cubic 3D mesh (N,N,N).")
        if not np.isfinite(arr).all():
            raise ValueError(f"Field '{name}' contains non-finite values.")
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _los_axis(los: Sequence[float]) -> int:
        v = np.asarray(los, dtype=float)
        if v.shape != (3,):
            raise ValueError("los must be length-3.")
        ax = int(np.argmax(np.abs(v)))
        if not np.isclose(abs(v[ax]), 1.0, atol=1e-8) or (np.sum(np.abs(v) > 1e-8) != 1):
            raise ValueError("Pylians multipoles require axis-aligned LOS (e.g., (0,0,1)).")
        return ax

    def _power_norm_boxsize(self) -> float:
        """Return the box-size convention used for the output P(k)."""
        if self.cfg.power_norm_boxsize is None:
            return float(self.cfg.boxsize)
        return float(self.cfg.power_norm_boxsize)

    def _volume_correction_factor(self) -> float:
        """
        Multiplicative factor applied to Pylians spectra.

        Pylians returns spectra normalized to the volume of the mesh actually
        Fourier transformed, i.e. ``boxsize**3``. For a self-folded mesh, the
        mesh side length is Lbox/fold. If the desired convention is the original
        simulation volume Lbox**3, this factor is (Lbox / (Lbox/fold))**3 = fold**3.
        """
        L_mesh = float(self.cfg.boxsize)
        L_norm = self._power_norm_boxsize()
        return float((L_norm / L_mesh) ** 3)

    @staticmethod
    def apply_window(
        field: np.ndarray,
        window: np.ndarray,
        *,
        subtract_mean: bool = True,
        name: str = "field",
    ) -> np.ndarray:
        """
        Apply a window mesh W(x) to a field and return the masked field.

        Parameters
        ----------
        field : ndarray
            Input mesh.
        window : ndarray
            Window mesh (same shape as field).
        subtract_mean : bool
            If True, subtract window-weighted mean before masking.
        name : str
            Field name for error messages.

        Returns
        -------
        masked : ndarray
            Masked mesh W * (field - <field>_W) if subtract_mean else W*field.
        """
        f = PowerSpectrumEstimator._as_field(field, name)
        W = PowerSpectrumEstimator._as_field(window, "window")
        if f.shape != W.shape:
            raise ValueError("window must have same shape as field.")

        if subtract_mean:
            Wsum = float(np.sum(W, dtype=np.float64))
            meanW = float(np.sum(W * f, dtype=np.float64) / Wsum) if Wsum > 0 else 0.0
            f = f - np.float32(meanW)

        return (W * f).astype(np.float32, copy=False)

    @staticmethod
    def _rebin_pk(
        k: np.ndarray,
        P0: np.ndarray,
        P2: np.ndarray,
        P4: np.ndarray,
        Nm: np.ndarray,
        k_edges: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rebin native (k,Pell) to given k_edges using Nmodes weights."""
        k = np.asarray(k, dtype=float)
        P0 = np.asarray(P0, dtype=float)
        P2 = np.asarray(P2, dtype=float)
        P4 = np.asarray(P4, dtype=float)
        Nm = np.asarray(Nm, dtype=float)
        edges = np.asarray(k_edges, dtype=float)

        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("k_edges must be 1D, length>=2, strictly increasing.")

        nb = edges.size - 1
        kc = 0.5 * (edges[:-1] + edges[1:])
        out = np.full((3, nb), np.nan)

        ind = np.digitize(k, edges) - 1
        for b in range(nb):
            m = ind == b
            if not np.any(m):
                continue
            w = Nm[m]
            wsum = np.sum(w)
            if wsum <= 0:
                continue
            out[0, b] = float(np.sum(P0[m] * w) / wsum)
            out[1, b] = float(np.sum(P2[m] * w) / wsum)
            out[2, b] = float(np.sum(P4[m] * w) / wsum)

        return kc, out

    @staticmethod
    def _rebin_nmodes(k: np.ndarray, Nm: np.ndarray, k_edges: np.ndarray) -> np.ndarray:
        """Rebin native mode counts to output k bins by summing Nmodes."""
        k = np.asarray(k, dtype=float)
        Nm = np.asarray(Nm, dtype=float)
        edges = np.asarray(k_edges, dtype=float)
        nb = edges.size - 1
        out = np.zeros(nb, dtype=float)
        ind = np.digitize(k, edges) - 1
        for b in range(nb):
            m = ind == b
            if np.any(m):
                out[b] = float(np.sum(Nm[m], dtype=np.float64))
        return out

    @staticmethod
    def _pack_noise(meta: Optional[Dict[str, Any]], volume_factor: float = 1.0) -> Dict[str, Any]:
        """
        Pack shot/shape noise terms into a standardized dict.

        Parameters
        ----------
        meta : dict, optional
            Metadata dict potentially containing noise keys.

        Returns
        -------
        noise : dict
            Noise terms (keys included only if present).
        """
        noise: Dict[str, Any] = {}
        if not meta:
            return noise

        for k in ("shotnoise_gg", "shotnoise_dd", "shape_noise_EE", "shape_noise_BB"):
            if k not in meta:
                continue
            val = meta[k]
            if val is None:
                noise[k] = None
                continue
            try:
                noise[k] = float(val) * float(volume_factor)
            except (TypeError, ValueError):
                noise[k] = val
        return noise

    def compute(
        self,
        *,
        meshes: Dict[str, np.ndarray],
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        mas: Optional[Dict[str, str]] = None,
        k_edges: Optional[np.ndarray] = None,
        window_mode: str = "box",
        window: Optional[np.ndarray] = None,
        window_subtract_mean: bool = True,
        meta: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute P_ell(k) for requested auto/cross pairs.

        Parameters
        ----------
        meshes : dict
            Mapping name -> mesh array, each (N,N,N).
        pairs : sequence of (str, str), optional
            Spectra to compute. If None, computes all autos and all crosses.
        mas : dict, optional
            MAS deconvolution option per field name passed to Pylians.
            If None, uses "None" for all.
        k_edges : ndarray, optional
            If provided, rebin the native Pylians k-bins to these edges.
        window_mode : {'box','mesh'}
            'box' assumes periodic box (no window).
            'mesh' applies window to each field and returns pseudo-P(k).
        window : ndarray, optional
            Window mesh used if window_mode='mesh'.
        window_subtract_mean : bool
            If True, subtract window-weighted mean before masking.
        meta : dict, optional
            Metadata dict (e.g., from Mesh.py) used to append noise terms.
        verbose : bool
            If True, print diagnostics.

        Returns
        -------
        out : dict
            Power spectra multipoles plus ``out["noise"]`` when available.
        """
        L = float(self.cfg.boxsize)
        L_norm = self._power_norm_boxsize()
        volume_factor = self._volume_correction_factor()
        ells = self._ells
        axis = int(self._axis)

        # Validate fields
        field_names = sorted(meshes.keys())
        fields = {k: self._as_field(meshes[k], k) for k in field_names}

        # Window handling
        wmode = str(window_mode).strip().lower()
        if wmode == "mesh":
            if window is None:
                raise ValueError("window_mode='mesh' requires `window` mesh.")
            for k in field_names:
                fields[k] = self.apply_window(fields[k], window, subtract_mean=window_subtract_mean, name=k)
            if verbose:
                print("[powers] window_mode='mesh': returning pseudo-P(k) of masked fields (mode coupling not deconvolved).")
        elif wmode in ("box", "none", "periodic"):
            # periodic simulation box (no survey window)
            pass
        else:
            raise ValueError("window_mode must be 'box'/'none' (periodic) or 'mesh'.")

        # MAS per field
        if mas is None:
            mas_map = {k: "None" for k in field_names}
        else:
            mas_map = {k: str(mas.get(k, "None")) for k in field_names}

        # Determine pairs
        if pairs is None:
            pairs_list: List[Tuple[str, str]] = []
            for i, a in enumerate(field_names):
                pairs_list.append((a, a))
                for b in field_names[i + 1 :]:
                    pairs_list.append((a, b))
        else:
            pairs_list = [(str(a), str(b)) for (a, b) in pairs]
            for a, b in pairs_list:
                if a not in fields or b not in fields:
                    raise ValueError(f"Requested pair ({a},{b}) but missing mesh.")

        out: Dict[str, Any] = dict(
            boxsize=L,
            power_norm_boxsize=L_norm,
            volume_factor=volume_factor,
            los=np.array(self.cfg.los, dtype=float),
            axis=axis,
            ells=np.array(ells, dtype=int),
            k_edges=None if k_edges is None else np.asarray(k_edges, dtype=float),
            pairs=pairs_list,
            fields=field_names,
            mas=mas_map,
            window_mode=wmode,
        )

        # Compute noise terms (if provided)
        out["noise"] = self._pack_noise(meta, volume_factor=volume_factor)

        # Build ordered list of unique fields
        needed = sorted(set([a for a, _ in pairs_list] + [b for _, b in pairs_list]))
        f_list = [fields[k] for k in needed]
        mas_list = [mas_map[k] for k in needed]

        # Pylians call
        if len(needed) == 1:
            name = needed[0]
            if verbose:
                print(f"[powers] PKL.Pk for field '{name}'")
            P = PKL.Pk(f_list[0], L, axis, mas_list[0], int(self.cfg.threads), bool(verbose))
            k_native = np.asarray(P.k3D, dtype=float)
            Nm = np.asarray(P.Nmodes3D, dtype=float)
            auto_tri = {name: (np.asarray(P.Pk[:, 0]), np.asarray(P.Pk[:, 1]), np.asarray(P.Pk[:, 2]))}
            cross_tri: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        else:
            if verbose:
                print(f"[powers] PKL.XPk for fields: {needed}")
            P = PKL.XPk(f_list, L, axis, MAS=mas_list, threads=int(self.cfg.threads))
            k_native = np.asarray(P.k3D, dtype=float)
            Nm = np.asarray(P.Nmodes3D, dtype=float)

            auto_tri = {}
            for i, nm in enumerate(needed):
                auto_tri[nm] = (
                    np.asarray(P.Pk[:, 0, i], dtype=float),
                    np.asarray(P.Pk[:, 1, i], dtype=float),
                    np.asarray(P.Pk[:, 2, i], dtype=float),
                )

            def pair_index(i: int, j: int, n: int) -> int:
                idx = 0
                for a in range(n):
                    for b in range(a + 1, n):
                        if a == i and b == j:
                            return idx
                        idx += 1
                raise RuntimeError("pair not found in XPk output")

            cross_tri = {}
            nF = len(needed)
            for i in range(nF):
                for j in range(i + 1, nF):
                    pidx = pair_index(i, j, nF)
                    cross_tri[(needed[i], needed[j])] = (
                        np.asarray(P.XPk[:, 0, pidx], dtype=float),
                        np.asarray(P.XPk[:, 1, pidx], dtype=float),
                        np.asarray(P.XPk[:, 2, pidx], dtype=float),
                    )

        # Optional rebinning
        if k_edges is None:
            k_out = k_native

            def tri_to_Pell(tri):
                return np.vstack(tri)  # (3, nbins)
        else:
            edges = np.asarray(k_edges, dtype=float)
            any_name = needed[0]
            kc, _ = self._rebin_pk(
                k_native,
                auto_tri[any_name][0],
                auto_tri[any_name][1],
                auto_tri[any_name][2],
                Nm,
                edges,
            )
            k_out = kc

            def tri_to_Pell(tri):
                _, outP = self._rebin_pk(k_native, tri[0], tri[1], tri[2], Nm, edges)
                return outP

        out["k"] = k_out
        out["Nmodes_native"] = Nm
        if k_edges is None:
            out["Nmodes"] = Nm.astype(float, copy=False)
        else:
            out["Nmodes"] = self._rebin_nmodes(k_native, Nm, edges)

        ell_all = np.array([0, 2, 4], dtype=int)
        take = np.array([np.where(ell_all == e)[0][0] for e in ells], dtype=int)

        def get_tri(a: str, b: str):
            if a == b:
                return auto_tri[a]
            if (a, b) in cross_tri:
                return cross_tri[(a, b)]
            if (b, a) in cross_tri:
                return cross_tri[(b, a)]
            return None

        for a, b in pairs_list:
            tri = get_tri(a, b)
            key = f"P_{a}{b}"
            if tri is None:
                out[key + "_ell"] = None
                if 0 in ells:
                    out[key] = None
                continue

            Pell_all = tri_to_Pell(tri) * volume_factor  # (3, nbins)
            Pell = Pell_all[take, :]                    # (Nells, Nk)
            out[key + "_ell"] = Pell.T  # (Nk, Nells)
            if 0 in ells:
                i0 = int(np.where(np.array(ells) == 0)[0][0])
                out[key] = Pell[i0, :]

        return out
