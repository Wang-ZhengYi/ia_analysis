# -*- coding: utf-8 -*-
"""
Shape and inertia utilities for halo / galaxy analysis.

Main features
-------------
1) Iterative self-consistent ellipsoidal tensor measurement:
       I_iters(particles, masses=None, velocities=None, accelerations=None, ...)

   * If only positions are given:
         I = I_iters(...)
   * If positions + velocities are given:
         I, dI = I_iters(...)
   * If positions + velocities + accelerations are given:
         I, dI, ddI = I_iters(...)

   The public tensor modes are preserved:

       tensor_mode = "full"
           I_ij   = Σ m x_i x_j
           dI_ij  = Σ m (v_i x_j + x_i v_j)
           ddI_ij = Σ m (a_i x_j + x_i a_j + 2 v_i v_j)

       tensor_mode = "mass_normalized"
           I_ij   = Σ (m/M) x_i x_j
           dI_ij  = Σ (m/M) (v_i x_j + x_i v_j)
           ddI_ij = Σ (m/M) (a_i x_j + x_i a_j + 2 v_i v_j)

       tensor_mode = "reduced"
           I_ij   = Σ m x_i x_j / r_ell^2

   where r_ell^2 is built from the converged ellipsoidal geometry rather than
   from the spherical radius. This follows the Appendix-A spirit of iterating
   the ellipsoidal metric until the eigensystem is stable.

   For `full` and `mass_normalized`, the same converged ellipsoidal geometry is
   used to define the iterative subset, and the final tensor is evaluated on the
   final ellipsoidal subset. For `reduced`, the same converged geometry is used
   both in the subset definition and in the tensor weight.

   IMPORTANT:
   - dI and ddI for `reduced` are evaluated with the converged ellipsoidal
     metric held fixed. This makes the derivatives consistent with the final
     tensor definition used for the shape measurement.
   - The higher-order response of the converged metric itself is not modeled
     explicitly; doing so would require solving a fully implicit perturbation
     problem for the evolving eigensystem.

2) Axis extraction:
       axes, vecs = compute_axis(I)

3) Sampling variance estimates (non-resampling):
   - Provided as standalone eig_var(...)
   - AND integrated into ShapeKin with caching to avoid recomputation.

4) Figure-rotation estimator:
       out = beta_fig(I, dI, ddI)

   Returns the instantaneous principal-axis-frame angular velocity and
   angular acceleration inferred from the evolution of the eigensystem.

Robustness rules (pipeline-friendly)
------------------------------------
- Iteration returns the best available final tensor even without strict
  convergence, rather than aborting the pipeline.
- Bad / non-finite particles are filtered before the tensor iteration.
- Variance estimation (eig_var / iters_var / ShapeKin cached var) never aborts:
    * any error -> return NaNs of correct shapes
    * emit logger.warning(...)
    * cache failure (do NOT recompute) unless user explicitly force=True

Compatibility
-------------
- Existing calling style is preserved.
- Old pipelines that call ShapeKin.run_shape() then ShapeKin.var_eig() still work.
"""

import logging
import numpy as np

logger = logging.getLogger("shape")


# ============================================================
# Basic tensor helpers
# ============================================================


def _symmetrize(M):
    """Return 0.5 * (M + M.T)."""
    M = np.asarray(M, dtype=np.float64)
    return 0.5 * (M + M.T)


def _proper_rotation_frame(R, eps=1e-12):
    """
    Return a proper right-handed orthonormal frame in SO(3).

    Eigenvectors of a symmetric tensor are sign-degenerate.  This fixes the
    handedness convention while leaving the physical unoriented axes unchanged.
    """
    Q = np.asarray(R, dtype=np.float64)
    if Q.shape != (3, 3) or not np.all(np.isfinite(Q)):
        raise ValueError("R must be a finite 3x3 matrix")

    Q = Q.copy()
    for j in range(3):
        n = float(np.linalg.norm(Q[:, j]))
        if (not np.isfinite(n)) or n < eps:
            raise ValueError("Degenerate eigenvector frame")
        Q[:, j] /= n

    if np.linalg.det(Q) < 0.0:
        Q[:, 2] *= -1.0

    U, _, Vt = np.linalg.svd(Q)
    Q = U @ Vt
    if np.linalg.det(Q) < 0.0:
        U[:, -1] *= -1.0
        Q = U @ Vt

    return Q.astype(np.float64)


def _sorted_eigh(M):
    """
    Eigen-decomposition of a symmetric 3x3 matrix, sorted by descending eigenvalue.

    The returned eigenvector matrix is a right-handed SO(3) frame with columns
    ordered as major, intermediate, minor axes.
    """
    evals, evecs = np.linalg.eigh(_symmetrize(M))
    idx = np.argsort(evals)[::-1]
    evals = evals[idx].astype(np.float64)
    evecs = _proper_rotation_frame(evecs[:, idx].astype(np.float64))
    return evals, evecs


def _major_axis_metric_from_tensor(I, eps=1e-30):
    """
    Build a scale-free ellipsoidal metric from a 3x3 symmetric tensor.

    If the tensor eigenvalues are λ_a >= λ_b >= λ_c, define the axis ratios
        q = b / a = sqrt(|λ_b| / |λ_a|)
        s = c / a = sqrt(|λ_c| / |λ_a|)
    and construct the body-frame metric
        G_body = diag(1, 1/q^2, 1/s^2).

    The corresponding lab-frame metric is
        G = E G_body E^T,
    where the columns of E are the major-, intermediate-, and minor-axis
    eigenvectors.

    Then the self-consistent ellipsoidal radius is
        r_ell^2 = x^T G x.

    This fixes the otherwise arbitrary overall scale and keeps only the shape
    information (axis ratios + orientation), which is what the iteration needs.
    """
    evals, evecs = _sorted_eigh(I)
    lam = np.maximum(np.abs(evals), eps)

    a = np.sqrt(lam[0])
    b = np.sqrt(lam[1])
    c = np.sqrt(lam[2])
    if (not np.isfinite(a + b + c)) or a <= 0.0 or b <= 0.0 or c <= 0.0:
        raise ValueError("Invalid eigenvalues while building ellipsoidal metric")

    q = max(b / a, np.sqrt(eps))
    s = max(c / a, np.sqrt(eps))

    G_body = np.diag([1.0, 1.0 / (q * q), 1.0 / (s * s)])
    G = evecs @ G_body @ evecs.T
    G = _symmetrize(G)

    return {
        "metric": G,
        "evals": evals,
        "evecs": evecs,
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "q": float(q),
        "s": float(s),
    }


def _ellipsoidal_radius2(P, metric, eps=1e-30):
    """Return r_ell^2 = x^T G x for all particles."""
    P = np.asarray(P, dtype=np.float64)
    G = _symmetrize(np.asarray(metric, dtype=np.float64))
    r2 = np.einsum("ni,ij,nj->n", P, G, P)
    return np.maximum(r2, eps)


def _base_particle_arrays(particles, masses=None, velocities=None, accelerations=None, Pos=None):
    """
    Convert inputs to arrays, shift to the chosen center, and build a finite mask.

    Bad particles are not allowed to poison the tensor iteration. Any row with
    non-finite position / mass / velocity / acceleration is excluded here.
    Non-positive masses are also excluded when masses are provided.
    """
    R = np.asarray(particles, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != 3:
        raise ValueError("`particles` must have shape (N,3)")
    N = R.shape[0]

    if masses is None:
        m = np.ones(N, dtype=np.float64)
    else:
        m = np.asarray(masses, dtype=np.float64)
        if m.shape != (N,):
            raise ValueError("`masses` must have the same length as `particles`")

    V = None
    if velocities is not None:
        V = np.asarray(velocities, dtype=np.float64)
        if V.shape != (N, 3):
            raise ValueError("`velocities` must have shape (N,3) matching `particles`")

    A = None
    if accelerations is not None:
        A = np.asarray(accelerations, dtype=np.float64)
        if A.shape != (N, 3):
            raise ValueError("`accelerations` must have shape (N,3) matching `particles`")

    if Pos is None:
        center = None
        P = R.copy()
    else:
        center = np.asarray(Pos, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("`Pos` must be a finite 3-vector")
        P = R - center

    good = np.all(np.isfinite(P), axis=1) & np.isfinite(m) & (m > 0.0)
    if V is not None:
        good &= np.all(np.isfinite(V), axis=1)
    if A is not None:
        good &= np.all(np.isfinite(A), axis=1)

    return P, m, V, A, center, good


def _constant_weight_prefactor(masses, tensor_mode="full"):
    """
    Return the time-independent per-particle prefactor α_i.

    The actual weight is:
      full            : w_i = α_i
      mass_normalized : w_i = α_i
      reduced         : w_i = α_i / r_ell,i^2
    """
    m = np.asarray(masses, dtype=np.float64)

    if tensor_mode == "full":
        return m.copy()

    if tensor_mode == "mass_normalized":
        M = np.sum(m)
        if (not np.isfinite(M)) or (M <= 0.0):
            raise ValueError("Non-positive total mass for tensor_mode='mass_normalized'")
        return m / M

    if tensor_mode == "reduced":
        return m.copy()

    raise ValueError("`tensor_mode` must be one of: 'full', 'mass_normalized', 'reduced'")


def _compute_tensor_from_metric(P, masses, velocities=None, accelerations=None,
                                tensor_mode="reduced", metric=None, eps=1e-30):
    """
    Compute I, dI, ddI using a fixed ellipsoidal metric.

    Parameters
    ----------
    P : (N,3)
        Coordinates relative to the chosen center.
    masses : (N,)
    velocities : (N,3) or None
    accelerations : (N,3) or None
    tensor_mode : {"full", "mass_normalized", "reduced"}
    metric : (3,3) or None
        Ellipsoidal metric G used in the reduced tensor denominator
            r_ell^2 = x^T G x.
        If None, the identity matrix is used.
    eps : float
        Small positive floor for r_ell^2.

    Notes
    -----
    - For `full` and `mass_normalized`, the metric does not enter the weight.
    - For `reduced`, the converged ellipsoidal metric enters both I and its
      kinematic derivatives. The metric is treated as fixed during the local
      derivative evaluation.
    """
    P = np.asarray(P, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("`P` must have shape (N,3)")
    if m.shape != (P.shape[0],):
        raise ValueError("`masses` must have length N")

    if metric is None:
        G = np.eye(3, dtype=np.float64)
    else:
        G = _symmetrize(np.asarray(metric, dtype=np.float64))
        if G.shape != (3, 3) or not np.all(np.isfinite(G)):
            raise ValueError("`metric` must be a finite (3,3) matrix")

    alpha = _constant_weight_prefactor(m, tensor_mode=tensor_mode)

    need_dI = velocities is not None
    need_ddI = (velocities is not None) and (accelerations is not None)

    if tensor_mode in ("full", "mass_normalized"):
        w = alpha
        wd = np.zeros(P.shape[0], dtype=np.float64) if need_dI else None
        wdd = np.zeros(P.shape[0], dtype=np.float64) if need_ddI else None

    else:
        r2 = _ellipsoidal_radius2(P, G, eps=eps)
        w = alpha / r2

        wd = None
        wdd = None

        if need_dI:
            V = np.asarray(velocities, dtype=np.float64)
            if V.shape != P.shape:
                raise ValueError("`velocities` must have shape (N,3)")
            GP = P @ G
            sdot = 2.0 * np.sum(GP * V, axis=1)
            wd = -alpha * sdot / (r2 * r2)

        if need_ddI:
            V = np.asarray(velocities, dtype=np.float64)
            A = np.asarray(accelerations, dtype=np.float64)
            if V.shape != P.shape:
                raise ValueError("`velocities` must have shape (N,3)")
            if A.shape != P.shape:
                raise ValueError("`accelerations` must have shape (N,3)")

            GP = P @ G
            GV = V @ G
            sdot = 2.0 * np.sum(GP * V, axis=1)
            sddot = 2.0 * (np.sum(GV * V, axis=1) + np.sum(GP * A, axis=1))
            wdd = 2.0 * alpha * (sdot * sdot) / (r2 * r2 * r2) - alpha * sddot / (r2 * r2)

    I = np.einsum("i,ij,ik->jk", w, P, P)
    I = _symmetrize(I)

    if velocities is None:
        return I, None, None

    V = np.asarray(velocities, dtype=np.float64)
    if V.shape != P.shape:
        raise ValueError("`velocities` must have shape (N,3)")

    dI = (
        np.einsum("i,ij,ik->jk", wd, P, P) +
        np.einsum("i,ij,ik->jk", w, V, P) +
        np.einsum("i,ij,ik->jk", w, P, V)
    )
    dI = _symmetrize(dI)

    if accelerations is None:
        return I, dI, None

    A = np.asarray(accelerations, dtype=np.float64)
    if A.shape != P.shape:
        raise ValueError("`accelerations` must have shape (N,3)")

    ddI = (
        np.einsum("i,ij,ik->jk", wdd, P, P) +
        2.0 * np.einsum("i,ij,ik->jk", wd, V, P) +
        2.0 * np.einsum("i,ij,ik->jk", wd, P, V) +
        np.einsum("i,ij,ik->jk", w, A, P) +
        np.einsum("i,ij,ik->jk", w, P, A) +
        2.0 * np.einsum("i,ij,ik->jk", w, V, V)
    )
    ddI = _symmetrize(ddI)

    return I, dI, ddI


def compute_inertia_tensor(particles, masses=None, velocities=None, accelerations=None,
                           Pos=None, tensor_mode="reduced", metric=None, eps=1e-30):
    """
    Compute I and optionally dI, ddI for a single, non-iterated tensor evaluation.

    Parameters
    ----------
    metric : (3,3) or None
        Optional ellipsoidal metric. If None, the identity metric is used, which
        reduces the `reduced` tensor to the usual spherical m/r^2 form.
    """
    P, m, V, A, _, good = _base_particle_arrays(
        particles, masses=masses, velocities=velocities, accelerations=accelerations, Pos=Pos
    )

    if np.count_nonzero(good) == 0:
        return np.full((3, 3), np.nan), None, None

    return _compute_tensor_from_metric(
        P[good],
        m[good],
        velocities=None if V is None else V[good],
        accelerations=None if A is None else A[good],
        tensor_mode=tensor_mode,
        metric=metric,
        eps=eps,
    )


def _compute_raw_I(P, masses, tensor_mode="full", metric=None, eps=1e-30):
    """Compute the instantaneous tensor I for a fixed metric."""
    I, _, _ = _compute_tensor_from_metric(P, masses, tensor_mode=tensor_mode, metric=metric, eps=eps)
    return I


def _compute_dI(P, V, masses, tensor_mode="full", metric=None, eps=1e-30):
    """Compute dI for a fixed metric."""
    _, dI, _ = _compute_tensor_from_metric(
        P, masses, velocities=V, accelerations=None, tensor_mode=tensor_mode, metric=metric, eps=eps
    )
    return dI


def _compute_ddI(P, V, A, masses, tensor_mode="full", metric=None, eps=1e-30):
    """Compute ddI for a fixed metric."""
    _, _, ddI = _compute_tensor_from_metric(
        P, masses, velocities=V, accelerations=A, tensor_mode=tensor_mode, metric=metric, eps=eps
    )
    return ddI


# ============================================================
# Iterative ellipsoid selection / self-consistent tensor
# ============================================================


def _select_with_metric(P, metric, percentile=100.0, eps=1e-30):
    """
    Select particles using the current ellipsoidal metric.

    If percentile < 100, keep the inner percentile of ellipsoidal radius.
    If percentile >= 100, keep all particles.
    """
    r2 = _ellipsoidal_radius2(P, metric, eps=eps)
    if percentile is None or float(percentile) >= 100.0:
        mask = np.isfinite(r2)
    else:
        rr = np.sqrt(r2)
        cut = np.percentile(rr, float(percentile))
        mask = rr <= cut
    return mask, r2


def _rel_eval_change(evals_new, evals_old, eps=1e-30):
    """Maximum relative change between two eigenvalue triplets."""
    evals_new = np.asarray(evals_new, dtype=np.float64)
    evals_old = np.asarray(evals_old, dtype=np.float64)
    den = np.maximum(np.maximum(np.abs(evals_new), np.abs(evals_old)), eps)
    return float(np.max(np.abs(evals_new - evals_old) / den))


def I_iters(
    particles,
    masses=None,
    velocities=None,
    accelerations=None,
    Pos=None,
    r_ell=None,
    percentile=100,
    max_iter=100,
    tol=0.01,
    return_dI=False,
    return_ddI=False,
    return_mask=False,
    return_converged=False,
    tensor_mode="reduced",
    eps=1e-30,
):
    """
    Iterative self-consistent ellipsoidal tensor computation with optional dI, ddI.

    MINIMAL-CHANGE UPDATE (requested)
    ---------------------------------
    The ellipsoidal *geometry* (metric / axis ratios / orientation) used for the
    iterative subset selection is determined **only** from a mass-normalized
    second-moment tensor:

        I_geom = Σ (m/M_subset) x_i x_j

    This ensures the eigenvalues have units of L^2, so sqrt(eigenvalues) are
    RMS-length scales, and the iterative geometry is a purely geometric criterion
    (independent of total mass scaling).

    After the geometry and final ellipsoidal subset are determined, the final
    output tensor(s) are evaluated on that final subset using the *requested*
    tensor_mode:

        tensor_mode = "full"           : I = Σ m x_i x_j
        tensor_mode = "mass_normalized": I = Σ (m/M) x_i x_j
        tensor_mode = "reduced"        : I = Σ m x_i x_j / r_ell^2, with r_ell^2 = x^T G x

    Notes
    -----
    - The optional `r_ell` input remains a **spherical** pre-aperture cut applied
      before ellipsoidal refinement. This preserves existing pipeline behavior.
    - For `reduced`, dI and ddI are evaluated with the converged ellipsoidal
      metric held fixed, consistent with the final tensor definition.
    - The algorithm returns the best available tensor even if strict convergence
      is not reached; it does not abort the pipeline.
    - Bad / non-finite particles are filtered before iteration.

    Returns
    -------
    I or (I, dI?, ddI?, mask?, converged?)
    """
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)

    # -------------------------
    # Empty input handling
    # -------------------------
    if particles is None or len(particles) == 0:
        out = [nan33.copy()]
        if return_dI:
            out.append(nan33.copy())
        if return_ddI:
            out.append(nan33.copy())
        if return_mask:
            out.append(np.zeros(0, dtype=bool))
        if return_converged:
            out.append(False)
        return tuple(out) if len(out) > 1 else nan33.copy()

    # -------------------------
    # Build base arrays and finite mask
    # -------------------------
    P, m, V, A, _, good = _base_particle_arrays(
        particles,
        masses=masses,
        velocities=velocities,
        accelerations=accelerations,
        Pos=Pos,
    )
    N = P.shape[0]

    # Optional spherical pre-aperture cut (in native length unit).
    if r_ell is not None:
        rval = float(r_ell)
        if np.isfinite(rval) and rval > 0.0:
            rad2 = np.sum(P * P, axis=1)
            good &= rad2 <= (rval * rval)
        else:
            good &= False

    if np.count_nonzero(good) < 3:
        out = [nan33.copy()]
        if return_dI:
            out.append(nan33.copy())
        if return_ddI:
            out.append(nan33.copy())
        if return_mask:
            out.append(good.copy())
        if return_converged:
            out.append(False)
        return tuple(out) if len(out) > 1 else nan33.copy()

    # Work arrays restricted to finite + pre-aperture particles
    work_idx = np.where(good)[0]
    P_work = P[work_idx]
    m_work = m[work_idx]
    V_work = None if V is None else V[work_idx]
    A_work = None if A is None else A[work_idx]

    # -------------------------
    # Iterate geometry (metric) using mass-normalized tensor ONLY
    # -------------------------
    G = np.eye(3, dtype=np.float64)  # current ellipsoidal metric
    mask_local = np.ones(P_work.shape[0], dtype=bool)

    I_last = nan33.copy()         # last successfully computed *final-mode* tensor (best effort)
    evals_last = None
    converged = False

    for _ in range(int(max_iter)):
        # Select subset by current ellipsoidal radius (metric G)
        mask_try, _ = _select_with_metric(P_work, G, percentile=percentile, eps=eps)
        if np.count_nonzero(mask_try) < 3:
            break

        try:
            # Geometry tensor: ALWAYS mass-normalized (units L^2), independent of requested tensor_mode
            I_geom = _compute_raw_I(
                P_work[mask_try],
                m_work[mask_try],
                tensor_mode="mass_normalized",
                metric=None,
                eps=eps,
            )
            if not np.all(np.isfinite(I_geom)):
                break

            geom = _major_axis_metric_from_tensor(I_geom, eps=eps)
            evals_try = geom["evals"]
            G_try = geom["metric"]

        except Exception:
            break

        # Keep the mask for best-effort fallback
        mask_local = mask_try

        # Optionally keep a best-effort tensor in the requested mode for fallback
        try:
            I_last = _compute_raw_I(
                P_work[mask_try],
                m_work[mask_try],
                tensor_mode=tensor_mode,
                metric=G,   # reduced needs a metric; full/mass_normalized ignore it
                eps=eps,
            )
        except Exception:
            pass

        # Convergence check on eigenvalues of the *geometry tensor*
        if evals_last is not None:
            if _rel_eval_change(evals_try, evals_last, eps=eps) < float(tol):
                G = G_try
                evals_last = evals_try
                converged = True
                break

        G = G_try
        evals_last = evals_try

    # -------------------------
    # Final subset and final tensor evaluation on the final subset
    # -------------------------
    try:
        mask_final_local, _ = _select_with_metric(P_work, G, percentile=percentile, eps=eps)
        if np.count_nonzero(mask_final_local) < 3:
            mask_final_local = mask_local.copy()

        P_fin = P_work[mask_final_local]
        m_fin = m_work[mask_final_local]
        V_fin = None if V_work is None else V_work[mask_final_local]
        A_fin = None if A_work is None else A_work[mask_final_local]

        I_fin, dI_fin, ddI_fin = _compute_tensor_from_metric(
            P_fin,
            m_fin,
            velocities=V_fin if (return_dI or return_ddI) else None,
            accelerations=A_fin if return_ddI else None,
            tensor_mode=tensor_mode,
            metric=G,  # reduced uses final G; others ignore
            eps=eps,
        )

        if not np.all(np.isfinite(I_fin)):
            raise FloatingPointError("Final tensor contains non-finite values")

    except Exception:
        # Best-effort fallback: keep last computed tensor if finite; otherwise NaNs
        I_fin = I_last if np.all(np.isfinite(I_last)) else nan33.copy()
        dI_fin = nan33.copy() if return_dI else None
        ddI_fin = nan33.copy() if return_ddI else None
        mask_final_local = mask_local.copy()
        converged = False

    # Build full-length boolean mask aligned with the original particle array
    mask = np.zeros(N, dtype=bool)
    mask[work_idx[mask_final_local]] = True

    # -------------------------
    # Pack outputs (compat)
    # -------------------------
    results = [I_fin]

    if return_dI:
        if velocities is None:
            results.append(nan33.copy())
        else:
            results.append(np.asarray(dI_fin, dtype=np.float64) if dI_fin is not None else nan33.copy())

    if return_ddI:
        if velocities is None or accelerations is None:
            results.append(nan33.copy())
        else:
            results.append(np.asarray(ddI_fin, dtype=np.float64) if ddI_fin is not None else nan33.copy())

    if return_mask:
        results.append(mask)

    if return_converged:
        results.append(bool(converged))

    return tuple(results) if len(results) > 1 else I_fin


# ============================================================
# Axis extraction
# ============================================================


def compute_axis(I):
    """
    Compute principal axes and eigenvectors from a 3x3 symmetric tensor.

    Returns
    -------
    axes : dict {'a','b','c'} with a>=b>=c are sqrt(|evals|)
    vecs : dict {'e1','e2','e3'} eigenvectors matched to a,b,c
    """
    I = np.asarray(I, dtype=np.float64)
    if I.shape != (3, 3):
        raise ValueError("I must be (3,3)")
    evals, evecs = _sorted_eigh(I)
    a, b, c = (np.sqrt(np.abs(evals[0])), np.sqrt(np.abs(evals[1])), np.sqrt(np.abs(evals[2])))
    axes = {"a": float(a), "b": float(b), "c": float(c)}
    vecs = {
        "e1": evecs[:, 0].astype(float),
        "e2": evecs[:, 1].astype(float),
        "e3": evecs[:, 2].astype(float),
    }
    return axes, vecs


# ============================================================
# Variance estimator
# ============================================================


def eig_var(particles, masses=None, Pos=None, mask=None, normalize=True, eps=1e-30, chunk=50000):
    """
    Standalone estimator: sampling variances of eigenvalues and pointing cosines.

    NOTE (compat):
    - This function keeps the strict behavior (may raise) for legacy usage.
    - Pipeline code should prefer eig_var_safe(...) which never raises.
    """
    R = np.asarray(particles, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != 3:
        raise ValueError("`particles` must have shape (N,3)")
    N = R.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 particles")

    if masses is None:
        m = np.ones(N, dtype=np.float64)
    else:
        m = np.asarray(masses, dtype=np.float64)
        if m.shape[0] != N:
            raise ValueError("`masses` must have length N")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != N:
            raise ValueError("`mask` must have length N")
    else:
        mask = np.ones(N, dtype=bool)

    X = R if Pos is None else (R - np.asarray(Pos, dtype=np.float64))
    good = mask & np.all(np.isfinite(X), axis=1) & np.isfinite(m) & (m > 0.0)

    X = X[good]
    m = m[good]

    if X.shape[0] < 3:
        raise ValueError("Need at least 3 finite particles after masking")

    if normalize:
        M = np.sum(m)
        if not np.isfinite(M) or M <= 0.0:
            raise ValueError("Non-positive total mass in selected set")
        w = m / M
    else:
        w = m

    Neff = 1.0 / np.sum(w ** 2)
    I = np.einsum("i,ij,ik->jk", w, X, X)

    evals, evecs = _sorted_eigh(I)
    Xb = X @ evecs
    Ib = np.einsum("i,ij,ik->jk", w, Xb, Xb)

    comps = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    idx_map = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}

    ww = w ** 2
    C6 = np.zeros((6, 6), dtype=np.float64)

    Ns = Xb.shape[0]
    for s0 in range(0, Ns, int(chunk)):
        s1 = min(Ns, s0 + int(chunk))
        Xc = Xb[s0:s1]
        wwc = ww[s0:s1]

        d = np.empty((s1 - s0, 6), dtype=np.float64)
        for t, (i, j) in enumerate(comps):
            d[:, t] = Xc[:, i] * Xc[:, j] - Ib[i, j]

        C6 += (d.T * wwc) @ d

    def _var_Ib(i, j):
        if i > j:
            i, j = j, i
        return float(C6[idx_map[(i, j)], idx_map[(i, j)]])

    var_evals = np.array([_var_Ib(0, 0), _var_Ib(1, 1), _var_Ib(2, 2)], dtype=np.float64)

    mean_theta2 = np.zeros(3, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if j == i:
                continue
            gap = evals[i] - evals[j]
            mean_theta2[i] += _var_Ib(min(i, j), max(i, j)) / (gap * gap + eps)

    var_cos = 0.25 * mean_theta2 ** 2

    return {
        "I": I,
        "evals": evals,
        "evecs": evecs,
        "Neff": float(Neff),
        "var_evals": var_evals,
        "mean_theta2": mean_theta2,
        "var_cos": var_cos,
    }


def eig_var_safe(particles, masses=None, Pos=None, mask=None, normalize=True, eps=1e-30, chunk=50000,
                 *, warn=True, context="eig_var_safe"):
    """
    Safe wrapper around eig_var(...):
    - Never raises.
    - On failure: return NaNs with correct shapes and emit logger.warning(...) if warn=True.
    """
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)
    nan3 = np.full(3, np.nan, dtype=np.float64)
    try:
        return eig_var(particles, masses=masses, Pos=Pos, mask=mask,
                       normalize=normalize, eps=eps, chunk=chunk)
    except Exception as e:
        if warn:
            logger.warning("%s failed; returning NaNs. Reason: %s", context, str(e))
        return {
            "I": nan33,
            "evals": nan3.copy(),
            "evecs": nan33.copy(),
            "Neff": float("nan"),
            "var_evals": nan3.copy(),
            "mean_theta2": nan3.copy(),
            "var_cos": nan3.copy(),
        }


def iters_var(
    particles,
    masses=None,
    Pos=None,
    velocities=None,
    accelerations=None,
    percentile=98.0,
    max_iter=100,
    tol=0.01,
    return_dI=False,
    return_ddI=False,
    normalize=True,
    eps=1e-30,
    chunk=50000,
    tensor_mode="reduced",
    r_ell=None,
):
    """
    Convenience wrapper:
    1) run I_iters(..., return_mask=True) to obtain the final ellipsoidal subset,
    2) compute eigenvalue / pointing-cosine variances on the SAME subset.

    Robustness:
    - Never raises due to variance computation.
    - If I_iters itself raises (e.g. bad input shapes), that is a hard error.
    """
    out = {}

    res = I_iters(
        particles=particles,
        masses=masses,
        velocities=velocities,
        accelerations=accelerations,
        Pos=Pos,
        r_ell=r_ell,
        percentile=percentile,
        max_iter=max_iter,
        tol=tol,
        return_dI=return_dI,
        return_ddI=return_ddI,
        return_mask=True,
        return_converged=False,
        tensor_mode=tensor_mode,
        eps=eps,
    )

    if return_dI and return_ddI:
        I, dI, ddI, mask = res
        out["dI"] = dI
        out["ddI"] = ddI
    elif return_dI:
        I, dI, mask = res
        out["dI"] = dI
    elif return_ddI:
        I, ddI, mask = res
        out["ddI"] = ddI
    else:
        I, mask = res

    out["I"] = I
    out["mask"] = mask

    stats = eig_var_safe(
        particles,
        masses=masses,
        Pos=Pos,
        mask=mask,
        normalize=normalize,
        eps=eps,
        chunk=chunk,
        warn=True,
        context="iters_var/eig_var",
    )
    out.update(stats)
    return out


# ============================================================
# Kinematics helpers
# ============================================================


def _unit_vector(v):
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return (v / n).astype(np.float64)


def ang_mom(particles, velocities, masses=None, Pos=None, v_ref=None, mask=None):
    """
    Total angular momentum vector L and its unit direction Lhat for the selected set.
    Velocities are evaluated relative to v_ref. If v_ref is None, the mass-weighted
    mean velocity of the selected set is used.
    """
    R = np.asarray(particles, dtype=np.float64)
    V = np.asarray(velocities, dtype=np.float64)
    if R.shape != V.shape:
        raise ValueError("`particles` and `velocities` must have the same shape (N,3)")
    N = R.shape[0]

    m = None if masses is None else np.asarray(masses, dtype=np.float64)
    if m is not None and m.shape[0] != N:
        raise ValueError("`masses` must have length N")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != N:
            raise ValueError("`mask` must have length N")
        R = R[mask]
        V = V[mask]
        if m is not None:
            m = m[mask]

    good = np.all(np.isfinite(R), axis=1) & np.all(np.isfinite(V), axis=1)
    if m is not None:
        good &= np.isfinite(m)

    R = R[good]
    V = V[good]
    if m is not None:
        m = m[good]

    if R.shape[0] < 3:
        L = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        return L, L.copy()

    r = R if Pos is None else (R - np.asarray(Pos, dtype=np.float64))

    if v_ref is None:
        if m is None:
            v0 = np.mean(V, axis=0)
        else:
            M = np.sum(m)
            v0 = np.sum(V * m[:, None], axis=0) / M
    else:
        v0 = np.asarray(v_ref, dtype=np.float64)

    v = V - v0

    if m is None:
        L = np.sum(np.cross(r, v), axis=0)
    else:
        L = np.sum(np.cross(r, v) * m[:, None], axis=0)

    Lhat = _unit_vector(L)
    return L.astype(np.float64), Lhat


def kin_energy(velocities, masses=None, v_ref=None, mask=None):
    """
    Total kinetic energy K = 0.5 Σ m |v - v_ref|^2 for the selected set.
    If v_ref is None, uses the (mass-weighted) mean velocity of the selected set.
    """
    V = np.asarray(velocities, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("`velocities` must have shape (N,3)")
    N = V.shape[0]

    m = None if masses is None else np.asarray(masses, dtype=np.float64)
    if m is not None and m.shape[0] != N:
        raise ValueError("`masses` must have length N")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != N:
            raise ValueError("`mask` must have length N")
        V = V[mask]
        if m is not None:
            m = m[mask]

    good = np.all(np.isfinite(V), axis=1)
    if m is not None:
        good &= np.isfinite(m)

    V = V[good]
    if m is not None:
        m = m[good]

    if V.shape[0] == 0:
        return float("nan")

    if v_ref is None:
        if m is None:
            v0 = np.mean(V, axis=0)
        else:
            M = np.sum(m)
            v0 = np.sum(V * m[:, None], axis=0) / M
    else:
        v0 = np.asarray(v_ref, dtype=np.float64)

    dv = V - v0
    vv = np.sum(dv * dv, axis=1)

    if m is None:
        return 0.5 * float(np.sum(vv))
    return 0.5 * float(np.sum(m * vv))


def kappa_rot(particles, velocities, masses, Pos=None, v_ref=None, mask=None, Lhat=None):
    """
    Compute kappa_rot = K_rot / K_tot for the selected set using the angular-momentum axis.

    Returns dict with:
        kappa_rot, L, Lhat, K_tot, K_rot, N_used
    """
    R = np.asarray(particles, dtype=np.float64)
    V = np.asarray(velocities, dtype=np.float64)
    if R.shape != V.shape:
        raise ValueError("`particles` and `velocities` must have the same shape (N,3)")
    N = R.shape[0]

    m = None if masses is None else np.asarray(masses, dtype=np.float64)
    if m is not None and m.shape[0] != N:
        raise ValueError("`masses` must have length N")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != N:
            raise ValueError("`mask` must have length N")
        R = R[mask]
        V = V[mask]
        if m is not None:
            m = m[mask]

    good = np.all(np.isfinite(R), axis=1) & np.all(np.isfinite(V), axis=1)
    if m is not None:
        good &= np.isfinite(m)

    R = R[good]
    V = V[good]
    if m is not None:
        m = m[good]

    n_used = int(R.shape[0])
    if n_used < 3:
        return {
            "kappa_rot": float("nan"),
            "L": np.array([np.nan, np.nan, np.nan], dtype=np.float64),
            "Lhat": np.array([np.nan, np.nan, np.nan], dtype=np.float64),
            "K_tot": float("nan"),
            "K_rot": float("nan"),
            "N_used": n_used,
        }

    r = R if Pos is None else (R - np.asarray(Pos, dtype=np.float64))

    if v_ref is None:
        if m is None:
            v0 = np.mean(V, axis=0)
        else:
            M = np.sum(m)
            v0 = np.sum(V * m[:, None], axis=0) / M
    else:
        v0 = np.asarray(v_ref, dtype=np.float64)

    v = V - v0

    if Lhat is None:
        L, Lhat0 = ang_mom(R, V, masses=m, Pos=Pos, v_ref=v0, mask=None)
        Lhat = Lhat0
    else:
        Lhat = _unit_vector(np.asarray(Lhat, dtype=np.float64))
        if m is None:
            L = np.sum(np.cross(r, v), axis=0)
        else:
            L = np.sum(np.cross(r, v) * m[:, None], axis=0)

    if not np.all(np.isfinite(Lhat)):
        return {
            "kappa_rot": float("nan"),
            "L": L.astype(np.float64),
            "Lhat": Lhat.astype(np.float64),
            "K_tot": float("nan"),
            "K_rot": float("nan"),
            "N_used": n_used,
        }

    r_par = (r @ Lhat)[:, None] * Lhat[None, :]
    r_perp = r - r_par
    Rperp = np.linalg.norm(r_perp, axis=1)

    rv = np.cross(r, v)
    num = rv @ Lhat

    v_phi = np.zeros(n_used, dtype=np.float64)
    good = Rperp > 0.0
    v_phi[good] = num[good] / Rperp[good]

    vv = np.sum(v * v, axis=1)
    if m is None:
        K_tot = 0.5 * float(np.sum(vv))
        K_rot = 0.5 * float(np.sum(v_phi * v_phi))
    else:
        K_tot = 0.5 * float(np.sum(m * vv))
        K_rot = 0.5 * float(np.sum(m * (v_phi * v_phi)))

    kap = float(K_rot / K_tot) if np.isfinite(K_tot) and K_tot > 0.0 else float("nan")

    return {
        "kappa_rot": kap,
        "L": L.astype(np.float64),
        "Lhat": Lhat.astype(np.float64),
        "K_tot": float(K_tot),
        "K_rot": float(K_rot),
        "N_used": n_used,
    }


# ============================================================
# Figure rotation from I, dI, ddI
# ============================================================


def beta_fig(I, dI, ddI, eps=1e-30, rel_gap_min=1e-12):
    """
    Estimate instantaneous figure-rotation angular velocity and angular acceleration
    from I, dI, ddI.

    Parameters
    ----------
    I, dI, ddI : (3,3)
        Symmetric tensor and its first/second time derivatives.
    eps : float
        Small numerical floor.
    rel_gap_min : float
        Relative eigenvalue-gap threshold below which the estimate is considered
        ill-conditioned and NaNs are returned.

    Returns
    -------
    out : dict
        {
          "evals": (3,),
          "evecs": (3,3),
          "lambda_dot": (3,),
          "omega_fig": (3,),   # body-frame instantaneous angular velocity
          "beta_fig": (3,),    # body-frame instantaneous angular acceleration
          "W": (3,3),          # antisymmetric body-frame angular-velocity matrix
          "Wdot": (3,3),       # antisymmetric body-frame angular-acceleration matrix
          "valid": bool,
        }
    """
    nan3 = np.full(3, np.nan, dtype=np.float64)
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)

    try:
        I = _symmetrize(np.asarray(I, dtype=np.float64))
        dI = _symmetrize(np.asarray(dI, dtype=np.float64))
        ddI = _symmetrize(np.asarray(ddI, dtype=np.float64))

        if I.shape != (3, 3) or dI.shape != (3, 3) or ddI.shape != (3, 3):
            raise ValueError("I, dI, ddI must all have shape (3,3)")

        evals, evecs = _sorted_eigh(I)

        scale = max(np.max(np.abs(evals)), eps)
        gaps = np.array([
            abs(evals[0] - evals[1]),
            abs(evals[0] - evals[2]),
            abs(evals[1] - evals[2]),
        ], dtype=np.float64)

        if np.min(gaps) / scale < rel_gap_min:
            logger.warning("beta_fig ill-conditioned: nearly degenerate eigenvalues.")
            return {
                "evals": evals,
                "evecs": evecs,
                "lambda_dot": nan3.copy(),
                "omega_fig": nan3.copy(),
                "beta_fig": nan3.copy(),
                "W": nan33.copy(),
                "Wdot": nan33.copy(),
                "valid": False,
            }

        S = evecs.T @ dI @ evecs
        T = evecs.T @ ddI @ evecs

        lambda_dot = np.diag(S).astype(np.float64)

        W = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                gap = evals[j] - evals[i]
                if abs(gap) < eps:
                    gap = np.sign(gap) * eps if gap != 0.0 else eps
                W[i, j] = S[i, j] / gap

        W = 0.5 * (W - W.T)

        C = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                s0 = 0.0
                for k in range(3):
                    s0 += W[i, k] * W[k, j] * (2.0 * evals[k] - evals[i] - evals[j])
                C[i, j] = s0

        Wdot = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                gap = evals[j] - evals[i]
                if abs(gap) < eps:
                    gap = np.sign(gap) * eps if gap != 0.0 else eps
                rhs = T[i, j] - 2.0 * (lambda_dot[j] - lambda_dot[i]) * W[i, j] - C[i, j]
                Wdot[i, j] = rhs / gap

        Wdot = 0.5 * (Wdot - Wdot.T)

        omega_fig = np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)
        beta_vec = np.array([Wdot[2, 1], Wdot[0, 2], Wdot[1, 0]], dtype=np.float64)

        return {
            "evals": evals,
            "evecs": evecs,
            "lambda_dot": lambda_dot,
            "omega_fig": omega_fig,
            "beta_fig": beta_vec,
            "W": W,
            "Wdot": Wdot,
            "valid": True,
        }

    except Exception as e:
        logger.warning("beta_fig failed; returning NaNs. Reason: %s", str(e))
        return {
            "evals": nan3.copy(),
            "evecs": nan33.copy(),
            "lambda_dot": nan3.copy(),
            "omega_fig": nan3.copy(),
            "beta_fig": nan3.copy(),
            "W": nan33.copy(),
            "Wdot": nan33.copy(),
            "valid": False,
        }


# ============================================================
# ShapeKin with integrated + cached variance (robust)
# ============================================================


class ShapeKin:
    """
    Stateful wrapper for:
      - iterative ellipsoidal subset (mask),
      - I (+ optional dI/ddI) on that subset,
      - eigen decomposition,
      - sampling variances (Neff, var_evals, mean_theta2, var_cos) on the same subset.

    Robustness + caching rules
    --------------------------
    - run_shape(...) computes and caches variance by default.
    - Variance computation never raises:
        * any error -> cached NaNs, warning logged, and pipeline continues.
    - A cached failure is not recomputed unless force=True.
    """
    __slots__ = (
        "pos", "mass", "vel", "acc", "center",
        "mask", "I", "dI", "ddI", "evals", "evecs", "converged",
        "Neff", "var_evals", "mean_theta2", "var_cos",
        "_var_ready", "_var_normalize",
    )

    def __init__(self, particles, masses=None, velocities=None, accelerations=None, Pos=None):
        self.pos = np.asarray(particles, dtype=np.float64)
        self.mass = None if masses is None else np.asarray(masses, dtype=np.float64)
        self.vel = None if velocities is None else np.asarray(velocities, dtype=np.float64)
        self.acc = None if accelerations is None else np.asarray(accelerations, dtype=np.float64)
        self.center = None if Pos is None else np.asarray(Pos, dtype=np.float64)

        self.mask = None
        self.I = None
        self.dI = None
        self.ddI = None
        self.evals = None
        self.evecs = None
        self.converged = False

        self.Neff = float("nan")
        self.var_evals = np.full(3, np.nan, dtype=np.float64)
        self.mean_theta2 = np.full(3, np.nan, dtype=np.float64)
        self.var_cos = np.full(3, np.nan, dtype=np.float64)
        self._var_ready = False
        self._var_normalize = None

        n = self.pos.shape[0]
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("`particles` must have shape (N,3)")
        if self.mass is not None and self.mass.shape[0] != n:
            raise ValueError("`masses` must have length N")
        if self.vel is not None and self.vel.shape != self.pos.shape:
            raise ValueError("`velocities` must have shape (N,3)")
        if self.acc is not None and self.acc.shape != self.pos.shape:
            raise ValueError("`accelerations` must have shape (N,3)")

    def _compute_var_on_mask(self, normalize=True, eps=1e-30, chunk=50000):
        """
        Compute Neff, Var(eigenvalues), and principal-axis pointing uncertainties
        on the FINAL ellipsoidal subset.
    
        Consistency
        -----------
        We perform the uncertainty propagation in the eigenbasis of the *geometry*
        tensor (mass-normalized second moment):
            I_geom = Σ (m/M) x x^T
        evaluated on the final subset. This makes eigen-gaps scale in L^2 and
        avoids mixing scales from the requested output tensor_mode.
    
        Robustness
        ----------
        - Never raises.
        - Any failure -> return NaNs and log a warning.
        - Near-degenerate eigensystems: return finite Neff/var_evals, but set
          direction-related outputs (mean_theta2, var_cos) to NaN.
        """
        nan_Neff = float("nan")
        nan3 = np.full(3, np.nan, dtype=np.float64)
    
        try:
            if self.mask is None:
                raise RuntimeError("run_shape(...) not executed: mask is None")
    
            mask = np.asarray(self.mask, dtype=bool)
            if mask.size == 0 or np.count_nonzero(mask) < 3:
                raise ValueError("Need at least 3 particles in final subset")
    
            # Relative positions on the final subset
            X = self.pos[mask] if self.center is None else (self.pos[mask] - self.center)
            good = np.all(np.isfinite(X), axis=1)
    
            # Masses on the final subset
            if self.mass is None:
                m = np.ones(X.shape[0], dtype=np.float64)
            else:
                m = self.mass[mask].astype(np.float64, copy=False)
                good &= np.isfinite(m) & (m > 0.0)
    
            X = X[good]
            m = m[good]
            if X.shape[0] < 3:
                raise ValueError("Need at least 3 finite particles in final subset")
    
            # Geometry weights: always mass-normalized on the final subset
            M = np.sum(m)
            if (not np.isfinite(M)) or (M <= 0.0):
                raise ValueError("Non-positive total mass in final subset")
            w = m / M
    
            # Effective sample size
            Neff = 1.0 / np.sum(w * w)
            if (not np.isfinite(Neff)) or (Neff <= 0.0):
                raise ValueError("Invalid Neff")
    
            # --- Define evals_use/evecs_use HERE (fixes your NameError) ---
            I_geom = np.einsum("i,ij,ik->jk", w, X, X)
            I_geom = _symmetrize(I_geom)
            evals_use, evecs_use = _sorted_eigh(I_geom)
    
            # Rotate into geometry principal frame
            Xb = X @ evecs_use
            Ib = np.einsum("i,ij,ik->jk", w, Xb, Xb)
    
            # Accumulate covariance of the 6 independent tensor components in chunks
            comps = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
            idx_map = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}
    
            ww = w * w
            C6 = np.zeros((6, 6), dtype=np.float64)
    
            Ns = Xb.shape[0]
            for s0 in range(0, Ns, int(chunk)):
                s1 = min(Ns, s0 + int(chunk))
                Xc = Xb[s0:s1]
                wwc = ww[s0:s1]
    
                d = np.empty((s1 - s0, 6), dtype=np.float64)
                for t, (i, j) in enumerate(comps):
                    d[:, t] = Xc[:, i] * Xc[:, j] - Ib[i, j]
    
                C6 += (d.T * wwc) @ d
    
            def _var_Ib(i, j):
                if i > j:
                    i, j = j, i
                return float(C6[idx_map[(i, j)], idx_map[(i, j)]])
    
            # Eigenvalue variances (diagonal body-frame components)
            var_evals = np.array([_var_Ib(0, 0), _var_Ib(1, 1), _var_Ib(2, 2)], dtype=np.float64)
    
            # Direction uncertainties: guard against near-degenerate eigenvalues
            scale = max(np.max(np.abs(evals_use)), eps)
            gaps = np.array(
                [abs(evals_use[0] - evals_use[1]), abs(evals_use[0] - evals_use[2]), abs(evals_use[1] - evals_use[2])],
                dtype=np.float64,
            )
            rel_gap_min = 1e-3  # expose via cfg if needed
            if np.min(gaps) / scale < rel_gap_min:
                # Axis directions are ill-defined; keep var_evals but drop direction errors.
                return float(Neff), var_evals, nan3.copy(), nan3.copy()
    
            mean_theta2 = np.zeros(3, dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    if j == i:
                        continue
                    gap = evals_use[i] - evals_use[j]
                    mean_theta2[i] += _var_Ib(min(i, j), max(i, j)) / (gap * gap + eps)
    
            # Small-angle convention kept consistent with your current code:
            var_cos = 0.25 * mean_theta2 * mean_theta2
    
            return float(Neff), var_evals, mean_theta2, var_cos
    
        except Exception as e:
            logger.warning("ShapeKin variance failed -> cached NaNs (no recompute). Reason: %s", str(e))
            return nan_Neff, nan3.copy(), nan3.copy(), nan3.copy()

    def run_shape(
        self,
        percentile=98.0,
        max_iter=100,
        tol=0.01,
        return_dI=False,
        return_ddI=False,
        r_ell=None,
        tensor_mode="reduced",
        compute_var=True,
        normalize=True,
        eps=1e-30,
        chunk=50000,
        force_recompute_var=False,
    ):
        """
        Run iterative selection + compute tensors, and optionally compute variances once.

        Pipeline-safe:
        - Any failure in shape measurement (including eigen decomposition) does NOT raise.
        - On failure: caches NaNs of correct shapes and returns self.
        - Variance follows the existing rule: failure caches NaNs and will not
          recompute unless forced.
        """
        nan33 = np.full((3, 3), np.nan, dtype=np.float64)
        nan3 = np.full(3, np.nan, dtype=np.float64)

        try:
            res = I_iters(
                particles=self.pos,
                masses=self.mass,
                velocities=self.vel,
                accelerations=self.acc,
                Pos=self.center,
                r_ell=r_ell,
                percentile=percentile,
                max_iter=max_iter,
                tol=tol,
                return_dI=return_dI,
                return_ddI=return_ddI,
                return_mask=True,
                return_converged=True,
                tensor_mode=tensor_mode,
                eps=eps,
            )

            if return_dI and return_ddI:
                self.I, self.dI, self.ddI, self.mask, self.converged = res
            elif return_dI:
                self.I, self.dI, self.mask, self.converged = res
                self.ddI = None
            elif return_ddI:
                self.I, self.ddI, self.mask, self.converged = res
                self.dI = None
            else:
                self.I, self.mask, self.converged = res
                self.dI = None
                self.ddI = None

        except Exception as e:
            logger.warning("ShapeKin.run_shape failed in I_iters; returning NaNs. Reason: %s", str(e))
            self.I = nan33.copy()
            self.dI = nan33.copy() if return_dI else None
            self.ddI = nan33.copy() if return_ddI else None
            self.mask = np.zeros(self.pos.shape[0], dtype=bool)
            self.converged = False
            self.evals = nan3.copy()
            self.evecs = nan33.copy()
            if compute_var:
                self.Neff = float("nan")
                self.var_evals = nan3.copy()
                self.mean_theta2 = nan3.copy()
                self.var_cos = nan3.copy()
                self._var_ready = True
                self._var_normalize = bool(normalize)
            return self

        try:
            evals, evecs = _sorted_eigh(np.asarray(self.I, dtype=np.float64))
            self.evals = evals
            self.evecs = evecs
        except Exception as e:
            logger.warning("ShapeKin.run_shape eigen decomposition failed; returning NaNs. Reason: %s", str(e))
            self.evals = nan3.copy()
            self.evecs = nan33.copy()
            if compute_var:
                self.Neff = float("nan")
                self.var_evals = nan3.copy()
                self.mean_theta2 = nan3.copy()
                self.var_cos = nan3.copy()
                self._var_ready = True
                self._var_normalize = bool(normalize)
            return self

        if compute_var:
            if (not self._var_ready) or force_recompute_var or (self._var_normalize != bool(normalize)):
                self.Neff, self.var_evals, self.mean_theta2, self.var_cos = self._compute_var_on_mask(
                    normalize=normalize, eps=eps, chunk=chunk
                )
                self._var_ready = True
                self._var_normalize = bool(normalize)

        return self

    def var_eig(self, normalize=True, force=False, eps=1e-30, chunk=50000):
        """
        Return variance products.

        Pipeline-safe:
        - Never raises.
        - If run_shape(...) not executed or variance stage failed -> returns NaNs.
        - Will not recompute after a failure (cached NaNs) unless force=True.
        """
        nan33 = np.full((3, 3), np.nan, dtype=np.float64)
        nan3 = np.full(3, np.nan, dtype=np.float64)

        try:
            if self.mask is None:
                raise RuntimeError("Call run_shape(...) first to define final subset.")

            if (not self._var_ready) or force or (self._var_normalize != bool(normalize)):
                self.Neff, self.var_evals, self.mean_theta2, self.var_cos = self._compute_var_on_mask(
                    normalize=normalize, eps=eps, chunk=chunk
                )
                self._var_ready = True
                self._var_normalize = bool(normalize)

            return {
                "I": np.asarray(self.I, dtype=np.float64) if self.I is not None else nan33.copy(),
                "evals": np.asarray(self.evals, dtype=np.float64) if self.evals is not None else nan3.copy(),
                "evecs": np.asarray(self.evecs, dtype=np.float64) if self.evecs is not None else nan33.copy(),
                "Neff": float(self.Neff),
                "var_evals": np.asarray(self.var_evals, dtype=np.float64),
                "mean_theta2": np.asarray(self.mean_theta2, dtype=np.float64),
                "var_cos": np.asarray(self.var_cos, dtype=np.float64),
            }

        except Exception as e:
            logger.warning("ShapeKin.var_eig failed; returning NaNs. Reason: %s", str(e))
            return {
                "I": nan33.copy(),
                "evals": nan3.copy(),
                "evecs": nan33.copy(),
                "Neff": float("nan"),
                "var_evals": nan3.copy(),
                "mean_theta2": nan3.copy(),
                "var_cos": nan3.copy(),
            }

    def L(self, v_ref=None):
        """Return (L, Lhat). Pipeline-safe: never raises; returns NaNs on failure."""
        nan3 = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        try:
            if self.vel is None or self.mask is None:
                raise RuntimeError("Need velocities and run_shape() mask.")
            return ang_mom(self.pos, self.vel, masses=self.mass, Pos=self.center, v_ref=v_ref, mask=self.mask)
        except Exception as e:
            logger.warning("ShapeKin.L failed; returning NaNs. Reason: %s", str(e))
            return nan3.copy(), nan3.copy()

    def K(self, v_ref=None):
        """Return K_tot. Pipeline-safe: never raises; returns NaN on failure."""
        try:
            if self.vel is None or self.mask is None:
                raise RuntimeError("Need velocities and run_shape() mask.")
            return kin_energy(self.vel, masses=self.mass, v_ref=v_ref, mask=self.mask)
        except Exception as e:
            logger.warning("ShapeKin.K failed; returning NaN. Reason: %s", str(e))
            return float("nan")

    def kappa(self, v_ref=None, Lhat=None):
        """Return kappa_rot dict. Pipeline-safe: never raises; returns NaNs on failure."""
        nan3 = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        try:
            if self.vel is None or self.mask is None:
                raise RuntimeError("Need velocities and run_shape() mask.")
            return kappa_rot(self.pos, self.vel, masses=self.mass, Pos=self.center, v_ref=v_ref,
                             mask=self.mask, Lhat=Lhat)
        except Exception as e:
            logger.warning("ShapeKin.kappa failed; returning NaNs. Reason: %s", str(e))
            return {
                "kappa_rot": float("nan"),
                "L": nan3.copy(),
                "Lhat": nan3.copy(),
                "K_tot": float("nan"),
                "K_rot": float("nan"),
                "N_used": int(np.count_nonzero(self.mask)) if self.mask is not None else 0,
            }

    def beta_fig(self, eps=1e-30, rel_gap_min=1e-12):
        """
        Convenience wrapper around the module-level beta_fig(...) using self.I, self.dI, self.ddI.

        Returns
        -------
        dict
            Same output as the module-level beta_fig(...).
        """
        nan3 = np.full(3, np.nan, dtype=np.float64)
        nan33 = np.full((3, 3), np.nan, dtype=np.float64)
        try:
            if self.I is None or self.dI is None or self.ddI is None:
                raise RuntimeError("Need I, dI, ddI. Run run_shape(return_dI=True, return_ddI=True) first.")
            return beta_fig(self.I, self.dI, self.ddI, eps=eps, rel_gap_min=rel_gap_min)
        except Exception as e:
            logger.warning("ShapeKin.beta_fig failed; returning NaNs. Reason: %s", str(e))
            return {
                "evals": nan3.copy(),
                "evecs": nan33.copy(),
                "lambda_dot": nan3.copy(),
                "omega_fig": nan3.copy(),
                "beta_fig": nan3.copy(),
                "W": nan33.copy(),
                "Wdot": nan33.copy(),
                "valid": False,
            }
