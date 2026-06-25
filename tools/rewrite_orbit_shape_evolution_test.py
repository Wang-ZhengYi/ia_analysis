"""Expand the orbit/shape validation notebook with an initial-condition suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


NOTEBOOK = Path(
    "/cosma/home/dp203/dc-wang17/iaia/pipeline/05_orbit_stripping/"
    "orbit_shape_evolution_test.ipynb"
)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


SHAPE_MARKDOWN = r"""
## 7. Shape definition and calculation

This notebook describes the shape of a collisionless component by a real,
symmetric, positive-definite tensor

\[
S(t)=R(t)\,
\mathrm{diag}\!\left(a^2(t),b^2(t),c^2(t)\right)R^{\mathsf T}(t),
\qquad a\ge b\ge c>0.
\]

The columns of \(R\) are the instantaneous major, intermediate, and minor
axes. Diagonalizing \(S\) gives eigenvalues
\(\lambda_1\ge\lambda_2\ge\lambda_3>0\), from which the semi-axis lengths and
axis ratios are

\[
a=\sqrt{\lambda_1},\qquad
b=\sqrt{\lambda_2},\qquad
c=\sqrt{\lambda_3},\qquad
q=b/a,\qquad s=c/a.
\]

### Relation to a particle shape measurement

For an actual particle distribution, the tensor can be measured with the full
second moment

\[
I_{ij}=\frac{\sum_n m_n x_{n,i}x_{n,j}}{\sum_n m_n},
\]

or with a reduced tensor,

\[
I^{\rm red}_{ij}=
\frac{\sum_n m_n x_{n,i}x_{n,j}/r_n^2}{\sum_n m_n},
\]

usually inside an iteratively updated ellipsoidal aperture. The semi-analytic
tensor \(S\) used here has the same eigenvector and axis-ratio interpretation,
but is evolved from the orbit tidal history rather than remeasured from
particles at every time.

### Tidal equilibrium tensor

At each orbit sample the inertial tidal tensor is symmetrized and diagonalized,

\[
T(t)e_\alpha(t)=\tau_\alpha(t)e_\alpha(t),
\qquad \tau_1\ge\tau_2\ge\tau_3.
\]

The equilibrium shape aligns its longest axis with the most extensive tidal
eigenvector. Mass loss changes the equilibrium axis lengths anisotropically:

\[
a_\alpha^{\rm eq}(t)
=a_{\alpha,0}\,f_m(t)^{\gamma_\alpha},
\qquad
f_m(t)=M(t)/M(0),
\]

where the default exponents are
\((\gamma_a,\gamma_b,\gamma_c)=(0.10,0.18,0.30)\). The minor axis therefore
shrinks fastest as the bound mass decreases. The equilibrium tensor is

\[
S_{\rm eq}(t)=
R_T(t)\,
\mathrm{diag}\!\left[
(a^{\rm eq})^2,(b^{\rm eq})^2,(c^{\rm eq})^2
\right]R_T^{\mathsf T}(t).
\]

### Time response

The model assumes first-order relaxation,

\[
\frac{dS}{dt}=-\frac{S-S_{\rm eq}}{\tau_{\rm shape}},
\qquad
\tau_{\rm shape}
=N_{\rm shape}\frac{2\pi}{|\Omega|},
\qquad
|\Omega|=\frac{|\boldsymbol r\times\boldsymbol v|}{r^2}.
\]

Over one numerical interval \(\Delta t\), the exact constant-target update is

\[
S_{i+1}=S_i+
\left[1-\exp(-\Delta t/\tau_{\rm shape})\right]
(S_{{\rm eq},i}-S_i).
\]

After every update, \(S\) is projected back onto the positive-definite cone by
clipping non-positive eigenvalues. Eigenvector signs are chosen continuously
between adjacent samples because an axis and its negative represent the same
physical direction.

The reported shape--tide misalignment is the acute major-axis angle

\[
\theta_{\rm shape,tide}
=\cos^{-1}\!\left(
\left|\hat e^{\,S}_{\rm major}\cdot
\hat e^{\,T}_{\rm major}\right|
\right),
\qquad 0^\circ\le\theta\le90^\circ.
\]

This closure is a controlled response model, not a replacement for iterative
particle shape measurement. Its purpose is to compare how different orbital
histories alter alignment and axis ratios under identical response parameters.

For a spherical host, the two tangential tidal eigenvalues can be degenerate.
Their eigenvectors are then not individually unique, so an angle involving one
specific tangential eigenvector may jump or remain at \(90^\circ\) without
indicating a numerical error. The batch suite therefore also reports the
shape-major--radial angle

\[
\theta_{\rm shape,radial}
=\cos^{-1}\!\left(
\left|\hat e^{\,S}_{\rm major}\cdot\hat r\right|
\right),
\]

which remains geometrically well defined in a spherical potential, and a tidal
anisotropy measure based only on eigenvalues.
"""


SUITE_MARKDOWN = r"""
## 9. Multi-orbit initial-condition suite

The single fiducial orbit above is insufficient for validating a coupled
orbit--shape model. We therefore vary the initial phase-space point while
holding the host, subhalo mass, stripping law, initial shape tensor, and shape
response parameters fixed.

The suite contains:

- **fiducial infall**: the original reference orbit;
- **near circular**: small radial speed and larger tangential speed;
- **radial plunge**: low angular momentum and rapid inward motion;
- **outer high angular momentum**: a large-radius orbit with a long period;
- **inner fast orbit**: starts deeper in the host and samples a stronger tide;
- **outgoing phase**: starts after pericentre with positive radial velocity.

Each initial condition is run twice:

1. `conservative`: no dynamical friction, but the same stripping closure is
   retained so shape changes can still respond to mass loss;
2. `with_df`: Chandrasekhar dynamical friction and stripping are both enabled.

For every run, the notebook records pericentre, apocentre, final mass fraction,
maximum tidal eigenvalue magnitude, final \(b/a\), final \(c/a\), final
shape--tide angle, and the minimum eigenvalue of \(S\). This makes failures and
initial-condition trends visible numerically rather than only by inspection.
"""


SUITE_CODE = r"""
# ================================================================
# 9. Multi-orbit initial-condition experiment
# ================================================================
from pathlib import Path
import pandas as pd

PIPELINE_DIR = Path.cwd()
OUTPUT_DIR = PIPELINE_DIR / "outputs" / "orbit_shape_evolution_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORBIT_CASES = [
    dict(case="fiducial_infall", r0=450.0, phi0=0.25, vr0=-80.0,  vt0=140.0),
    dict(case="near_circular",   r0=450.0, phi0=0.25, vr0=-10.0,  vt0=210.0),
    dict(case="radial_plunge",   r0=450.0, phi0=0.25, vr0=-180.0, vt0=70.0),
    dict(case="outer_high_L",    r0=650.0, phi0=0.60, vr0=-35.0,  vt0=220.0),
    dict(case="inner_fast",      r0=260.0, phi0=1.10, vr0=-60.0,  vt0=185.0),
    dict(case="outgoing_phase",  r0=320.0, phi0=2.00, vr0=110.0,  vt0=145.0),
]

SUITE_T_END = 3.0
SUITE_DT = 0.01
SUITE_SHAPE_ORBITS = 0.50


def run_orbit_shape_case(case, *, with_df):
    # Run one orbit, evolve its shape tensor, and return diagnostics.
    r0 = float(case["r0"])
    vr0 = float(case["vr0"])
    vt0 = float(case["vt0"])
    energy = 0.5 * (vr0**2 + vt0**2) + sim.host.phi(
        r0 * len_fac, soften_kpc=SOFTEN * len_fac
    )
    angular_momentum = r0 * vt0
    result = sim.run(
        E=energy,
        L=angular_momentum,
        r0=r0,
        phi0=float(case["phi0"]),
        vr_sign=1 if vr0 >= 0 else -1,
        t_end=SUITE_T_END,
        dt=SUITE_DT,
        soften=SOFTEN,
        with_df=bool(with_df),
        with_strip=True,
        m_sub=M_SUB0,
        lnLambda=3.0,
        alpha_sub=ALPHA_SUB,
        gamma_p=ALPHA_SUB,
        show_progress=False,
    )

    mass_fraction = np.clip(
        result.M / max(float(result.M[0]), 1e-30), 1e-4, 1.0
    )
    history = evolve_shape_tensor(
        time=result.t,
        T_series=result.T_in,
        mass_fraction=mass_fraction,
        pos=result.pos,
        vel=result.v,
        S0=S0,
        n_shape_orbits=SUITE_SHAPE_ORBITS,
    )

    tidal_strength = np.array(
        [np.max(np.abs(np.linalg.eigvalsh(0.5 * (T + T.T)))) for T in result.T_in]
    )
    tidal_anisotropy = np.array([
        (
            np.ptp(np.linalg.eigvalsh(0.5 * (T + T.T)))
            / max(np.max(np.abs(np.linalg.eigvalsh(0.5 * (T + T.T)))), 1e-30)
        )
        for T in result.T_in
    ])
    shape_radial_angle = np.empty(len(result.t))
    for index, (shape_tensor, position) in enumerate(zip(history["S"], result.pos)):
        _, frame = sorted_eigh_symmetric(shape_tensor, descending=True)
        shape_radial_angle[index] = angle_deg(frame[:, 0], position)
    min_shape_eigenvalue = min(
        float(np.linalg.eigvalsh(S).min()) for S in history["S"]
    )
    mode = "with_df" if with_df else "conservative"
    run_name = f"{case['case']}__{mode}"

    time_table = pd.DataFrame({
        "time_gyr": result.t,
        "radius_ckpc_h": result.r,
        "bound_mass_1e10_msun_h": result.M,
        "mass_fraction": mass_fraction,
        "b_over_a": history["q"][:, 0],
        "c_over_a": history["q"][:, 1],
        "shape_tide_angle_deg": history["angle_major_tide_major_deg"],
        "shape_radial_angle_deg": shape_radial_angle,
        "tau_shape_gyr": history["tau_shape_gyr"],
        "tidal_strength": tidal_strength,
        "tidal_anisotropy": tidal_anisotropy,
    })
    time_table.to_csv(OUTPUT_DIR / f"{run_name}.csv", index=False)

    summary = {
        "case": case["case"],
        "mode": mode,
        "r0_ckpc_h": r0,
        "vr0_kms": vr0,
        "vt0_kms": vt0,
        "specific_energy_kms2": energy,
        "specific_angular_momentum": angular_momentum,
        "samples": len(result.t),
        "merged": bool(result.merged),
        "pericentre_ckpc_h": float(np.nanmin(result.r)),
        "apocentre_ckpc_h": float(np.nanmax(result.r)),
        "final_mass_fraction": float(mass_fraction[-1]),
        "maximum_tidal_strength": float(np.nanmax(tidal_strength)),
        "final_b_over_a": float(history["q"][-1, 0]),
        "final_c_over_a": float(history["q"][-1, 1]),
        "final_shape_tide_angle_deg": float(
            history["angle_major_tide_major_deg"][-1]
        ),
        "final_shape_radial_angle_deg": float(shape_radial_angle[-1]),
        "median_shape_tide_angle_deg": float(
            np.nanmedian(history["angle_major_tide_major_deg"])
        ),
        "median_shape_radial_angle_deg": float(np.nanmedian(shape_radial_angle)),
        "maximum_tidal_anisotropy": float(np.nanmax(tidal_anisotropy)),
        "minimum_shape_eigenvalue": min_shape_eigenvalue,
    }
    return result, history, time_table, summary


SUITE_RUNS = {}
summary_rows = []
for case in ORBIT_CASES:
    for with_df in (False, True):
        result, history, table, summary = run_orbit_shape_case(
            case, with_df=with_df
        )
        key = (case["case"], summary["mode"])
        SUITE_RUNS[key] = {
            "orbit": result,
            "shape": history,
            "table": table,
        }
        summary_rows.append(summary)

ORBIT_SHAPE_SUMMARY = pd.DataFrame(summary_rows)
ORBIT_SHAPE_SUMMARY.to_csv(
    OUTPUT_DIR / "orbit_shape_initial_condition_summary.csv", index=False
)
ORBIT_SHAPE_SUMMARY
"""


PLOT_CODE = r"""
# ================================================================
# 10. Comparative plots for all initial conditions
# ================================================================
case_colors = dict(zip(
    [case["case"] for case in ORBIT_CASES],
    plt.cm.tab10(np.linspace(0.0, 0.85, len(ORBIT_CASES))),
))
mode_styles = {"conservative": "-", "with_df": "--"}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for (case_name, mode), products in SUITE_RUNS.items():
    orbit = products["orbit"]
    shape_history = products["shape"]
    label = f"{case_name} / {mode}"
    style = mode_styles[mode]
    color = case_colors[case_name]

    axes[0, 0].plot(orbit.x, orbit.y, style, color=color, lw=1.2, label=label)
    axes[0, 1].plot(orbit.t, orbit.r, style, color=color, lw=1.2)
    axes[0, 2].plot(
        orbit.t, orbit.M / orbit.M[0], style, color=color, lw=1.2
    )
    axes[1, 0].plot(
        orbit.t, shape_history["q"][:, 0], style, color=color, lw=1.2
    )
    axes[1, 1].plot(
        orbit.t, shape_history["q"][:, 1], style, color=color, lw=1.2
    )
    axes[1, 2].plot(
        orbit.t,
        shape_history["angle_major_tide_major_deg"],
        style,
        color=color,
        lw=1.2,
    )

axes[0, 0].set(xlabel="x [ckpc/h]", ylabel="y [ckpc/h]", title="Orbit plane")
axes[0, 0].set_aspect("equal", adjustable="datalim")
axes[0, 1].set(xlabel="Time [Gyr]", ylabel="r [ckpc/h]", title="Orbital radius")
axes[0, 2].set(
    xlabel="Time [Gyr]", ylabel="M/M0", title="Irreversible stripping"
)
axes[1, 0].set(xlabel="Time [Gyr]", ylabel="b/a", title="Intermediate-axis ratio")
axes[1, 1].set(xlabel="Time [Gyr]", ylabel="c/a", title="Minor-axis ratio")
axes[1, 2].set(
    xlabel="Time [Gyr]",
    ylabel="Angle [deg]",
    title="Major shape--tide misalignment",
    ylim=(0, 90),
)
for ax in axes.flat:
    ax.grid(alpha=0.2)
axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "multi_initial_condition_orbit_shape_evolution.png",
    dpi=220,
    bbox_inches="tight",
)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for (case_name, mode), products in SUITE_RUNS.items():
    table = products["table"]
    axes[0].plot(
        table["time_gyr"], table["shape_radial_angle_deg"],
        mode_styles[mode], color=case_colors[case_name], lw=1.2,
    )
    axes[1].plot(
        table["time_gyr"], table["tidal_anisotropy"],
        mode_styles[mode], color=case_colors[case_name], lw=1.2,
    )
axes[0].set(
    xlabel="Time [Gyr]", ylabel="Angle [deg]",
    title="Shape major axis--radial direction", ylim=(0, 90),
)
axes[1].set(
    xlabel="Time [Gyr]", ylabel="Eigenvalue anisotropy",
    title="Tidal anisotropy",
)
for ax in axes:
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "shape_radial_alignment_and_tidal_anisotropy.png",
    dpi=220,
    bbox_inches="tight",
)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for mode, marker in (("conservative", "o"), ("with_df", "s")):
    panel = ORBIT_SHAPE_SUMMARY.query("mode == @mode")
    axes[0].scatter(
        panel["pericentre_ckpc_h"],
        panel["final_mass_fraction"],
        marker=marker,
        s=65,
        label=mode,
    )
    axes[1].scatter(
        panel["pericentre_ckpc_h"],
        panel["final_c_over_a"],
        marker=marker,
        s=65,
        label=mode,
    )
    axes[2].scatter(
        panel["maximum_tidal_strength"],
        panel["final_shape_tide_angle_deg"],
        marker=marker,
        s=65,
        label=mode,
    )
axes[0].set(xlabel="Pericentre [ckpc/h]", ylabel="Final M/M0")
axes[1].set(xlabel="Pericentre [ckpc/h]", ylabel="Final c/a")
axes[2].set(xlabel="Maximum tidal strength", ylabel="Final angle [deg]")
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
fig.suptitle("Initial-condition trends")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "orbit_shape_initial_condition_trends.png",
    dpi=220,
    bbox_inches="tight",
)
plt.show()


metric_columns = [
    "pericentre_ckpc_h",
    "apocentre_ckpc_h",
    "final_mass_fraction",
    "maximum_tidal_strength",
    "final_b_over_a",
    "final_c_over_a",
    "final_shape_tide_angle_deg",
    "final_shape_radial_angle_deg",
]
corr = ORBIT_SHAPE_SUMMARY[metric_columns].corr()
fig, ax = plt.subplots(figsize=(8.2, 7.0))
image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(metric_columns)), metric_columns, rotation=60, ha="right")
ax.set_yticks(range(len(metric_columns)), metric_columns)
for i in range(len(metric_columns)):
    for j in range(len(metric_columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(image, ax=ax, label="Pearson correlation")
ax.set_title("Orbit--shape summary correlation matrix")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "orbit_shape_summary_correlation.png",
    dpi=220,
    bbox_inches="tight",
)
plt.show()
"""


CHECK_MARKDOWN = r"""
## 11. Batch validation criteria

The suite applies the following numerical requirements to every orbit:

- all radii, masses, axis ratios, and angles are finite;
- radius remains positive;
- the stripping history is irreversible;
- the shape tensor remains positive definite;
- \(0<b/a\le1\) and \(0<c/a\le b/a\);
- the acute shape--tide angle remains in \([0^\circ,90^\circ]\).

These checks are numerical consistency tests. They do not imply that the
semi-analytic response parameters are calibrated to a particular simulation.
"""


CHECK_CODE = r"""
# ================================================================
# 11. Batch pass/fail checks
# ================================================================
batch_checks = []
for (case_name, mode), products in SUITE_RUNS.items():
    orbit = products["orbit"]
    history = products["shape"]
    q = history["q"]
    angle = history["angle_major_tide_major_deg"]
    min_eig = min(np.linalg.eigvalsh(S).min() for S in history["S"])
    conditions = {
        "finite_radius": np.all(np.isfinite(orbit.r)),
        "positive_radius": np.all(orbit.r > 0),
        "finite_mass": np.all(np.isfinite(orbit.M)),
        "irreversible_mass": np.all(np.diff(orbit.M) <= 1e-10),
        "positive_definite_shape": min_eig > 0,
        "valid_axis_ratios": (
            np.all(np.isfinite(q))
            and np.all((q[:, 0] > 0) & (q[:, 0] <= 1))
            and np.all((q[:, 1] > 0) & (q[:, 1] <= q[:, 0] + 1e-12))
        ),
        "valid_alignment_angle": (
            np.all(np.isfinite(angle))
            and np.min(angle) >= -1e-12
            and np.max(angle) <= 90 + 1e-12
        ),
    }
    batch_checks.append({
        "case": case_name,
        "mode": mode,
        **conditions,
        "all_pass": all(conditions.values()),
    })

ORBIT_SHAPE_CHECKS = pd.DataFrame(batch_checks)
ORBIT_SHAPE_CHECKS.to_csv(
    OUTPUT_DIR / "orbit_shape_batch_checks.csv", index=False
)
print(ORBIT_SHAPE_CHECKS.to_string(index=False))
if not ORBIT_SHAPE_CHECKS["all_pass"].all():
    failed = ORBIT_SHAPE_CHECKS.loc[
        ~ORBIT_SHAPE_CHECKS["all_pass"], ["case", "mode"]
    ]
    raise AssertionError(f"Orbit-shape batch checks failed:\n{failed}")
"""


FINAL_MARKDOWN = r"""
## 12. Interpretation and extension

The parameter suite should be interpreted comparatively:

- smaller pericentre generally produces a stronger tidal history;
- stronger tides and greater mass loss tend to reduce \(c/a\);
- dynamical friction can move an orbit to smaller radii and alter the time
  available for shape relaxation;
- a short relaxation time follows the tidal eigenframe quickly, whereas a
  large `SUITE_SHAPE_ORBITS` preserves orbital memory and larger
  misalignments.

Useful next tests are:

1. vary the initial intrinsic axis ratios and initial orientation;
2. vary `mass_exponents` and `n_shape_orbits`;
3. split stellar, dark-matter, and radial-shell response tensors;
4. replace the equilibrium closure with fits measured from TNG or
   ClusterSims;
5. sample initial conditions with a Latin-hypercube or cosmological infall
   distribution and regress final shape observables against orbit features.

All tables and figures from this notebook are written to
`outputs/orbit_shape_evolution_test/`.
"""


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    removal_markers = (
        "## 9. Multi-orbit initial-condition suite",
        "# 9. Multi-orbit initial-condition experiment",
        "# 10. Comparative plots for all initial conditions",
        "## 11. Batch validation criteria",
        "# 11. Batch pass/fail checks",
        "## 12. Interpretation and extension",
    )
    cells = [
        cell for cell in notebook["cells"]
        if not any(
            marker in (
                "".join(cell.get("source", []))
                if isinstance(cell.get("source"), list)
                else cell.get("source", "")
            )
            for marker in removal_markers
        )
    ]

    # Replace the existing shape-method markdown with a complete derivation.
    cells[17] = markdown(SHAPE_MARKDOWN)

    import_cell = next(
        cell for cell in cells
        if cell["cell_type"] == "code"
        and "install_pyccl_fallback" in (
            "".join(cell.get("source", []))
            if isinstance(cell.get("source"), list)
            else cell.get("source", "")
        )
    )
    import_source = (
        "".join(import_cell["source"])
        if isinstance(import_cell["source"], list)
        else import_cell["source"]
    )
    bootstrap_marker = "# Repository bootstrap for direct py312 execution"
    if bootstrap_marker not in import_source:
        import_source = import_source.replace(
            "from __future__ import annotations\n",
            """from __future__ import annotations

# Repository bootstrap for direct py312 execution
from pathlib import Path
import os
import sys

REPO_ROOT = Path("/cosma/home/dp203/dc-wang17/iaia/ia_analysis")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
NOTEBOOK_OUTPUT_DIR = (
    Path("/cosma/home/dp203/dc-wang17/iaia/pipeline/05_orbit_stripping")
    / "outputs" / "orbit_shape_evolution_test"
)
NOTEBOOK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(NOTEBOOK_OUTPUT_DIR / ".matplotlib"))

""",
        )
    import_cell["source"] = import_source

    # Insert the multi-orbit experiment after the existing single-orbit plots.
    insert_at = next(
        index for index, cell in enumerate(cells)
        if "## 8. Orbit-template feature extraction" in (
            "".join(cell.get("source", []))
            if isinstance(cell.get("source"), list)
            else cell.get("source", "")
        )
    )
    additions = [
        markdown(SUITE_MARKDOWN),
        code(SUITE_CODE),
        code(PLOT_CODE),
        markdown(CHECK_MARKDOWN),
        code(CHECK_CODE),
        markdown(FINAL_MARKDOWN),
    ]
    cells[insert_at:insert_at] = additions
    notebook["cells"] = cells

    # Update the top-level description.
    cells[0] = markdown(
        """
# Orbit integration and shape-evolution validation notebook

This notebook validates the installed `ia_analysis.orbits` implementation
using one detailed fiducial calculation and a twelve-run initial-condition
suite. It tests conservative integration, tidal tensors, dynamical friction,
stripping, shape-tensor response, orbit-template features, numerical
invariants, and correlations between orbital history and final shape.

The shape calculation is derived explicitly in Markdown before the
implementation. All batch tables and figures are written to
`outputs/orbit_shape_evolution_test/`.
"""
    )

    for index, cell in enumerate(cells):
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
        cell["id"] = hashlib.sha1(f"orbit-shape:{index}".encode()).hexdigest()[:12]
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Rewrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
