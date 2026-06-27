#!/usr/bin/env python
"""Create the complete HOD and component IA-HOD workflow notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_NOTEBOOK = REPO_ROOT / "notebooks" / "pipelines" / "09_full_hod_ia_workflow.ipynb"
EXTERNAL_NOTEBOOK = REPO_ROOT.parent / "pipeline" / "02_hod_lrg_elg" / "full_hod_ia_workflow.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def build_notebook(default_workflow_expression: str) -> nbf.NotebookNode:
    """Build one notebook with a location-specific default output directory."""
    cells = [
        markdown(
            """
            # Complete HOD and Component-Based IA-HOD Workflow

            This notebook runs the complete lightweight HOD analysis layer provided by
            `ia_analysis.hod`. It is organized as a reproducible scientific workflow:

            ```text
            halo and galaxy catalogues
                -> catalog standardization and validation
                -> ordinary HOD and LRG/ELG sample HOD
                -> fixed-mass assembly-biased HOD
                -> satellite phase-space and binding-layer statistics
                -> component-level intrinsic-alignment measurements
                -> conditional IA-strength models and fitting
                -> optional orientation forward model
                -> pairwise omega/eta validation
                -> covariance, plots, and saved products
            ```

            The default configuration uses a synthetic catalog so every cell can run
            without simulation files. Set `INPUT_MODE = "files"` and configure the
            input paths to analyze real standardized or user-mapped catalogues.

            All comments and documentation are in English. Outputs are written below
            the workflow directory configured in the next section.
            """
        ),
        markdown(
            """
            ## 1. Imports and workflow configuration

            This cell locates the source-first repository, imports the public HOD API,
            sets the input mode, and creates output directories. Plotting parameters
            are intentionally exposed here so they can be changed without modifying
            package code.
            """
        ),
        code(
            f"""
            from __future__ import annotations

            import json
            import os
            import sys
            from pathlib import Path

            import h5py
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            PYTHON = Path(sys.executable)

            def find_project_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "ia_analysis").exists():
                        return candidate
                    nested = candidate / "ia_analysis"
                    if (nested / "pyproject.toml").exists() and (nested / "src" / "ia_analysis").exists():
                        return nested
                raise FileNotFoundError("Could not locate the ia_analysis source-first project root.")

            PROJECT_ROOT = find_project_root(Path.cwd().resolve())
            DEFAULT_WORKFLOW_DIR = {default_workflow_expression}
            WORKFLOW_DIR = Path(os.environ.get("IA_HOD_WORKFLOW_DIR", DEFAULT_WORKFLOW_DIR)).expanduser().resolve()
            if str(PROJECT_ROOT / "src") not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT / "src"))

            from ia_analysis.hod import api as hod
            from ia_analysis.hod import plotting as hod_plot

            INPUT_MODE = "synthetic"  # Change to "files" for real catalogues.
            HALO_TABLE_PATH = WORKFLOW_DIR / "inputs" / "halos.csv"
            GALAXY_TABLE_PATH = WORKFLOW_DIR / "inputs" / "galaxies.csv"
            HALO_COLUMN_MAP = {{}}      # Canonical name -> input column name.
            GALAXY_COLUMN_MAP = {{}}    # Canonical name -> input column name.

            RANDOM_SEED = 17
            BOX_SIZE = 300.0
            VOLUME = BOX_SIZE ** 3
            SAMPLE_LABELS = ("all", "LRG", "ELG")
            MASS_EDGES = np.logspace(11.5, 14.8, 9)
            RADIUS_EDGES = np.linspace(0.0, 1.5, 9)
            SECONDARY_EDGES = np.array([-np.inf, 0.0, np.inf])
            FIGURE_DPI = 180
            FIGURE_FORMAT = "png"

            OUTPUT_DIR = WORKFLOW_DIR / "outputs" / "full_hod_ia_workflow"
            FIGURE_DIR = OUTPUT_DIR / "figures"
            TABLE_DIR = OUTPUT_DIR / "tables"
            DATA_DIR = OUTPUT_DIR / "data"
            for directory in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, DATA_DIR):
                directory.mkdir(parents=True, exist_ok=True)

            def save_figure(fig, name: str):
                path = FIGURE_DIR / f"{{name}}.{{FIGURE_FORMAT}}"
                fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(fig)
                return path

            print("Project root:", PROJECT_ROOT)
            print("Workflow directory:", WORKFLOW_DIR)
            print("Output directory:", OUTPUT_DIR)
            print("Input mode:", INPUT_MODE)
            """
        ),
        markdown(
            """
            ## 2. Input catalogues

            The synthetic generator includes central and satellite galaxies, LRG/ELG
            labels, host and tidal axes, radial vectors, velocities, binding-energy
            layers, spin, angular momentum, and figure-rotation axes. These fields
            exercise the full workflow.

            For real data, the file loader accepts CSV, TSV, Parquet, or HDF5 tables.
            Use the column maps above when input names differ from the canonical HOD
            schema documented in `docs/hod.md`.
            """
        ),
        code(
            """
            def random_unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
                vectors = rng.normal(size=(count, 3))
                return vectors / np.linalg.norm(vectors, axis=1)[:, None]

            def orthonormal_frames(rng: np.random.Generator, count: int):
                major = random_unit_vectors(rng, count)
                trial = random_unit_vectors(rng, count)
                intermediate = trial - np.einsum("ij,ij->i", trial, major)[:, None] * major
                intermediate /= np.linalg.norm(intermediate, axis=1)[:, None]
                minor = np.cross(major, intermediate)
                return major, intermediate, minor

            def biased_axial_vectors(rng, references, strength=0.75, noise=0.35):
                references = np.asarray(references, dtype=float)
                signs = rng.choice([-1.0, 1.0], size=len(references))
                vectors = strength * signs[:, None] * references + noise * rng.normal(size=references.shape)
                return vectors / np.linalg.norm(vectors, axis=1)[:, None]

            def make_synthetic_catalog(seed=RANDOM_SEED, n_halo=240):
                rng = np.random.default_rng(seed)
                halo_id = np.arange(n_halo)
                log_mass = rng.uniform(11.7, 14.6, n_halo)
                mass = 10.0 ** log_mass
                halo_position = rng.uniform(0.0, BOX_SIZE, size=(n_halo, 3))
                halo_velocity = rng.normal(0.0, 250.0, size=(n_halo, 3))
                host_major, host_intermediate, host_minor = orthonormal_frames(rng, n_halo)
                tidal_major, tidal_intermediate, tidal_minor = orthonormal_frames(rng, n_halo)

                concentration_trend = 8.5 - 0.7 * (log_mass - 12.5)
                concentration = concentration_trend + rng.normal(0.0, 1.1, n_halo)
                environment = 0.35 * (log_mass - 13.0) + rng.normal(0.0, 1.0, n_halo)
                tidal_anisotropy = 0.25 * environment + rng.normal(0.0, 0.8, n_halo)
                formation_time = 0.6 * concentration - 0.25 * log_mass + rng.normal(0.0, 0.5, n_halo)
                rvir = 0.25 * (mass / 1.0e12) ** (1.0 / 3.0)

                halos = pd.DataFrame({
                    "halo_id": halo_id,
                    "host_id": halo_id,
                    "mass": mass,
                    "rvir": rvir,
                    "position": list(halo_position),
                    "velocity": list(halo_velocity),
                    "concentration": concentration,
                    "environment": environment,
                    "tidal_anisotropy": tidal_anisotropy,
                    "formation_time": formation_time,
                    "spin": list(random_unit_vectors(rng, n_halo)),
                    "axis_ratio_ba": rng.uniform(0.65, 0.95, n_halo),
                    "axis_ratio_ca": rng.uniform(0.40, 0.80, n_halo),
                    "triaxiality": rng.uniform(0.0, 1.0, n_halo),
                    "host_major_axis": list(host_major),
                    "host_intermediate_axis": list(host_intermediate),
                    "host_minor_axis": list(host_minor),
                    "tidal_major_axis": list(tidal_major),
                    "tidal_intermediate_axis": list(tidal_intermediate),
                    "tidal_minor_axis": list(tidal_minor),
                })

                rows = []
                galaxy_id = 0
                concentration_residual = (concentration - concentration_trend) / np.std(concentration - concentration_trend)
                for index in range(n_halo):
                    central_probability = 1.0 / (1.0 + np.exp(-(log_mass[index] - 12.0) / 0.28))
                    sample_probability = 1.0 / (1.0 + np.exp(-(log_mass[index] - 13.0)))
                    sample_label = "LRG" if rng.random() < sample_probability else "ELG"
                    if rng.random() < central_probability:
                        central_axis = biased_axial_vectors(rng, host_major[index:index + 1], strength=0.85)[0]
                        rows.append({
                            "galaxy_id": galaxy_id, "halo_id": halo_id[index], "host_id": halo_id[index],
                            "is_central": True, "is_satellite": False, "sample_label": sample_label,
                            "stellar_mass": 10.0 ** (9.8 + 0.55 * (log_mass[index] - 12.0) + rng.normal(0, 0.15)),
                            "sfr": 0.2 if sample_label == "LRG" else 3.0,
                            "color": "red" if sample_label == "LRG" else "blue",
                            "position": halo_position[index], "velocity": halo_velocity[index],
                            "orientation": central_axis, "shape_major_axis": central_axis,
                            "shape_intermediate_axis": host_intermediate[index],
                            "shape_minor_axis": host_minor[index],
                            "subhalo_major_axis": central_axis, "subhalo_minor_axis": host_minor[index],
                            "spin": random_unit_vectors(rng, 1)[0],
                            "angular_momentum": random_unit_vectors(rng, 1)[0],
                            "radial_vector": host_major[index], "r_over_rvir": 0.0,
                            "binding_energy": -3.0, "binding_energy_layer": "inner",
                            "binding_energy_layer_axis": host_major[index],
                            "figure_rotation_axis": host_minor[index],
                        })
                        galaxy_id += 1

                    mean_sat = max((mass[index] / 4.0e12) ** 0.9 * np.exp(0.18 * concentration_residual[index]), 0.0)
                    n_satellite = rng.poisson(mean_sat)
                    for satellite_index in range(n_satellite):
                        radius_fraction = np.clip(rng.beta(1.7, 3.5) * 1.5, 0.03, 1.45)
                        radial = random_unit_vectors(rng, 1)[0]
                        radius = radius_fraction * rvir[index]
                        position = halo_position[index] + radius * radial
                        radial_speed = rng.normal(-30.0, 120.0)
                        tangential = random_unit_vectors(rng, 1)[0]
                        tangential -= np.dot(tangential, radial) * radial
                        tangential /= max(np.linalg.norm(tangential), 1e-12)
                        velocity = halo_velocity[index] + radial_speed * radial + rng.normal(160.0, 35.0) * tangential
                        layer = "inner" if radius_fraction < 0.4 else ("middle" if radius_fraction < 0.8 else "outer")
                        shape_axis = biased_axial_vectors(
                            rng,
                            radial[None, :],
                            strength=0.80 - 0.12 * radius_fraction + 0.08 * concentration_residual[index],
                            noise=0.42,
                        )[0]
                        rows.append({
                            "galaxy_id": galaxy_id, "halo_id": halo_id[index], "host_id": halo_id[index],
                            "is_central": False, "is_satellite": True, "sample_label": sample_label,
                            "stellar_mass": 10.0 ** (9.3 + 0.35 * (log_mass[index] - 12.0) + rng.normal(0, 0.22)),
                            "sfr": 0.1 if sample_label == "LRG" else 4.0,
                            "color": "red" if sample_label == "LRG" else "blue",
                            "position": position, "velocity": velocity, "orientation": shape_axis,
                            "shape_major_axis": shape_axis,
                            "shape_intermediate_axis": host_intermediate[index],
                            "shape_minor_axis": np.cross(shape_axis, host_intermediate[index]),
                            "subhalo_major_axis": biased_axial_vectors(rng, shape_axis[None, :], strength=0.9)[0],
                            "subhalo_minor_axis": host_minor[index],
                            "spin": random_unit_vectors(rng, 1)[0],
                            "angular_momentum": np.cross(radial, velocity - halo_velocity[index]),
                            "radial_vector": radial, "r_over_rvir": radius_fraction,
                            "binding_energy": -1.0 / max(radius_fraction, 0.05),
                            "binding_energy_layer": layer,
                            "binding_energy_layer_axis": shape_axis,
                            "figure_rotation_axis": np.cross(radial, shape_axis),
                        })
                        galaxy_id += 1
                return halos, pd.DataFrame(rows)

            def load_table(path: Path) -> pd.DataFrame:
                suffix = path.suffix.lower()
                if suffix == ".csv":
                    return pd.read_csv(path)
                if suffix in {".tsv", ".txt", ".dat"}:
                    return pd.read_csv(path, sep=None, engine="python")
                if suffix == ".parquet":
                    return pd.read_parquet(path)
                if suffix in {".h5", ".hdf5"}:
                    return pd.read_hdf(path)
                raise ValueError(f"Unsupported table format: {path}")

            if INPUT_MODE == "synthetic":
                halo_input, galaxy_input = make_synthetic_catalog()
            elif INPUT_MODE == "files":
                halo_input = load_table(HALO_TABLE_PATH)
                galaxy_input = load_table(GALAXY_TABLE_PATH)
            else:
                raise ValueError("INPUT_MODE must be 'synthetic' or 'files'.")

            print("Raw halo rows:", len(halo_input))
            print("Raw galaxy rows:", len(galaxy_input))
            """
        ),
        markdown(
            """
            ## 3. Standardize and validate the catalog

            The adapter converts input mappings, structured arrays, or DataFrames into
            a common `HODCatalog`. It validates halo IDs and masses, infers missing
            central/satellite flags, and preserves all additional science columns.
            """
        ),
        code(
            """
            catalog = hod.standardize_hod_catalog(
                halo_input,
                galaxy_input,
                halo_columns=HALO_COLUMN_MAP,
                galaxy_columns=GALAXY_COLUMN_MAP,
            )
            hod.validate_hod_catalog(catalog)

            catalog_summary = pd.DataFrame({
                "quantity": ["haloes", "galaxies", "centrals", "satellites", "LRG", "ELG"],
                "count": [
                    len(catalog.halos),
                    len(catalog.galaxies),
                    int(catalog.galaxies["is_central"].sum()),
                    int(catalog.galaxies["is_satellite"].sum()),
                    int((catalog.galaxies["sample_label"] == "LRG").sum()),
                    int((catalog.galaxies["sample_label"] == "ELG").sum()),
                ],
            })
            catalog_summary.to_csv(TABLE_DIR / "catalog_summary.csv", index=False)
            display(catalog_summary)
            display(catalog.halos.head(3))
            display(catalog.galaxies.head(3))
            """
        ),
        markdown(
            """
            ## 4. Ordinary HOD: all galaxies, LRG, and ELG

            This stage measures `N_cen(M)`, `N_sat(M)`, `N_tot(M)`, occupation
            variance, `P(N|M)`, satellite fraction, and number density. The same mass
            bins are used for all samples so LRG/ELG curves can be compared directly.
            """
        ),
        code(
            """
            hod_measurements = {}
            for sample in SAMPLE_LABELS:
                label = None if sample == "all" else sample
                measurement = hod.measure_hod(catalog, mass_bins=MASS_EDGES, sample_label=label)
                hod_measurements[sample] = measurement
                hod.measurement_to_dataframe(measurement).to_csv(TABLE_DIR / f"hod_{sample}.csv", index=False)
                hod.save_hod_measurement_hdf5(measurement, DATA_DIR / f"hod_{sample}.hdf5")

            satellite_number_summary = pd.DataFrame([
                {
                    "sample": sample,
                    "satellite_fraction": hod.measure_satellite_fraction(catalog, sample_label=None if sample == "all" else sample),
                    "number_density": hod.measure_number_density(
                        catalog, volume=VOLUME, sample_label=None if sample == "all" else sample
                    ),
                }
                for sample in SAMPLE_LABELS
            ])
            satellite_number_summary.to_csv(TABLE_DIR / "sample_number_density.csv", index=False)

            occupation_distribution = hod.measure_occupation_distribution(catalog, mass_bins=MASS_EDGES)
            occupation_distribution.to_csv(TABLE_DIR / "occupation_distribution.csv", index=False)

            fig, ax = hod_plot.plot_hod(hod_measurements["all"], label="all")
            save_figure(fig, "ordinary_hod_all")
            fig, ax = hod_plot.plot_hod_by_sample(hod_measurements, component="mean_tot")
            ax.set_title("Total HOD by galaxy sample")
            save_figure(fig, "hod_lrg_elg_comparison")

            display(satellite_number_summary)
            display(hod.measurement_to_dataframe(hod_measurements["all"]))
            """
        ),
        markdown(
            """
            ## 5. Zheng-style HOD prediction and fit

            The package provides the standard softened central threshold and
            power-law satellite occupation. This cell fits the five Zheng parameters
            to the measured all-galaxy HOD using SciPy least squares and overlays the
            fitted prediction.
            """
        ),
        code(
            """
            ordinary = hod_measurements["all"]
            valid = (ordinary.n_halo > 0) & np.isfinite(ordinary.mean_cen) & np.isfinite(ordinary.mean_sat)
            zheng_fit = hod.fit_zheng_hod(
                ordinary.mass_centers[valid],
                ordinary.mean_cen[valid],
                ordinary.mean_sat[valid],
            )
            zheng_prediction = hod.predict_zheng_hod(ordinary.mass_centers, **zheng_fit["parameters"])
            zheng_table = pd.DataFrame({
                "mass": ordinary.mass_centers,
                "measured_central": ordinary.mean_cen,
                "measured_satellite": ordinary.mean_sat,
                "measured_total": ordinary.mean_tot,
                "model_central": zheng_prediction["central"],
                "model_satellite": zheng_prediction["satellite"],
                "model_total": zheng_prediction["total"],
            })
            zheng_table.to_csv(TABLE_DIR / "zheng_hod_fit.csv", index=False)
            (OUTPUT_DIR / "zheng_hod_parameters.json").write_text(
                json.dumps(zheng_fit["parameters"], indent=2), encoding="utf-8"
            )

            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            ax.plot(ordinary.mass_centers, ordinary.mean_cen, "o", label="Measured central")
            ax.plot(ordinary.mass_centers, ordinary.mean_sat, "s", label="Measured satellite")
            ax.plot(ordinary.mass_centers, ordinary.mean_tot, "^", label="Measured total")
            ax.plot(ordinary.mass_centers, zheng_prediction["central"], "-", label="Zheng central")
            ax.plot(ordinary.mass_centers, zheng_prediction["satellite"], "--", label="Zheng satellite")
            ax.plot(ordinary.mass_centers, zheng_prediction["total"], ":", label="Zheng total")
            ax.set(xscale="log", yscale="log", xlabel="Halo mass", ylabel="Mean occupation")
            ax.legend(frameon=False, ncol=2)
            ax.grid(alpha=0.2)
            save_figure(fig, "zheng_hod_fit")
            display(pd.Series(zheng_fit["parameters"], name="best_fit"))
            """
        ),
        markdown(
            """
            ## 6. Fixed-mass assembly-biased HOD

            Assembly bias must be measured at fixed halo mass. The package first
            removes the mean mass trend of each secondary property and standardizes
            the residuals. It then splits haloes into quantiles and compares high and
            low occupation.

            This notebook measures concentration, environment, and tidal-anisotropy
            HOD splits. The output explicitly includes high/low ratios and
            differences.
            """
        ),
        code(
            """
            assembly_results = {}
            for secondary in ("concentration", "environment", "tidal_anisotropy"):
                result = hod.measure_assembly_hod(
                    catalog,
                    secondary_property=secondary,
                    quantiles=2,
                    mass_bins=MASS_EDGES,
                )
                assembly_results[secondary] = result
                rows = []
                for quantile, measurement in result["measurements"].items():
                    table = hod.measurement_to_dataframe(measurement)
                    table["quantile"] = quantile
                    rows.append(table)
                pd.concat(rows, ignore_index=True).to_csv(TABLE_DIR / f"assembly_hod_{secondary}.csv", index=False)
                fig, ax = hod_plot.plot_assembly_hod(result)
                ax.set_title(f"Fixed-mass {secondary} HOD")
                save_figure(fig, f"assembly_hod_{secondary}")

            concentration_labels, concentration_standardized = hod.split_by_concentration_quantiles(
                catalog.halos["mass"],
                catalog.halos["concentration"],
                quantiles=2,
                mass_bins=MASS_EDGES,
            )
            environment_labels, environment_standardized = hod.split_by_environment_quantiles(
                catalog.halos["mass"],
                catalog.halos["environment"],
                quantiles=2,
                mass_bins=MASS_EDGES,
            )
            catalog.halos["concentration_quantile"] = concentration_labels
            catalog.halos["environment_quantile"] = environment_labels
            catalog.halos["concentration_standardized"] = concentration_standardized
            catalog.halos["environment_standardized"] = environment_standardized

            decorated = hod.decorated_hod_prediction(
                catalog.halos["mass"],
                concentration_standardized,
                central_amplitude=0.45,
                satellite_amplitude=0.25,
                parameters=zheng_fit["parameters"],
            )
            decorated_table = pd.DataFrame({
                "halo_id": catalog.halos["halo_id"],
                "mass": catalog.halos["mass"],
                "concentration_standardized": concentration_standardized,
                "decorated_central": decorated["central"],
                "decorated_satellite": decorated["satellite"],
                "decorated_total": decorated["total"],
            })
            decorated_table.to_csv(TABLE_DIR / "decorated_hod_prediction.csv", index=False)
            display(decorated_table.head())
            """
        ),
        markdown(
            """
            ## 7. Satellite phase-space and binding-energy layers

            Occupation assembly bias does not determine where satellites live or how
            they move. This stage measures:

            - radial occupation in `r/Rvir`;
            - velocity anisotropy by halo mass;
            - occupation of inner, middle, and outer binding-energy layers;
            - phase-space summaries split by concentration and environment quantile;
            - satellite-position alignment with the host major axis.
            """
        ),
        code(
            """
            radial_profile = hod.measure_radial_profile_hod(
                catalog, radius_edges=RADIUS_EDGES, mass_edges=MASS_EDGES
            )
            velocity_anisotropy = hod.measure_velocity_anisotropy_hod(
                catalog, mass_edges=MASS_EDGES
            )
            binding_layers = hod.measure_binding_energy_layer_occupation(catalog)
            phase_space = {
                "radial_profile": radial_profile,
                "velocity_anisotropy": velocity_anisotropy,
                "binding_layers": binding_layers,
            }
            for name, table in phase_space.items():
                table.to_csv(TABLE_DIR / f"phase_space_{name}.csv", index=False)

            concentration_phase_space = hod.measure_phase_space_assembly_bias(
                catalog, quantile_column="concentration_quantile"
            )
            environment_phase_space = hod.measure_phase_space_assembly_bias(
                catalog, quantile_column="environment_quantile"
            )

            joined_satellites = hod.join_halo_galaxy_properties(
                catalog.halos,
                catalog.galaxies.loc[catalog.galaxies["is_satellite"]],
            )
            host_axis_alignment = hod.measure_host_axis_phase_space_alignment(
                np.vstack(joined_satellites["radial_vector"]),
                np.vstack(joined_satellites["host_major_axis"]),
            )

            fig, ax = hod_plot.plot_phase_space_hod(radial_profile)
            ax.set_title("Satellite radial occupation")
            save_figure(fig, "satellite_radial_profile")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            axes[0].plot(
                np.sqrt(velocity_anisotropy["mass_low"] * velocity_anisotropy["mass_high"]),
                velocity_anisotropy["beta"],
                "o-",
            )
            axes[0].set(xscale="log", xlabel="Halo mass", ylabel="Velocity anisotropy beta")
            layer_counts = binding_layers.groupby("binding_energy_layer")["occupation"].sum()
            axes[1].bar(layer_counts.index.astype(str), layer_counts.values)
            axes[1].set(xlabel="Binding-energy layer", ylabel="Satellite occupation")
            for ax in axes:
                ax.grid(alpha=0.2)
            save_figure(fig, "velocity_anisotropy_and_binding_layers")
            print("Satellite-position / host-major-axis alignment:", host_axis_alignment)
            """
        ),
        markdown(
            """
            ## 8. IA reference-vector bank

            The reference bank resolves every physical axis available in the joined
            catalog. The scalar axial statistic is

            `A = <|e dot q_ref|^2> - 1/3`,

            which is invariant under `e -> -e` and `q_ref -> -q_ref`.
            """
        ),
        code(
            """
            joined_all = hod.join_halo_galaxy_properties(catalog.halos, catalog.galaxies)
            reference_bank = hod.build_reference_bank(joined_all)
            reference_summary = pd.DataFrame({
                "reference": list(reference_bank),
                "count": [len(values) for values in reference_bank.values()],
                "finite_fraction": [np.isfinite(values).all(axis=1).mean() for values in reference_bank.values()],
            })
            reference_summary.to_csv(TABLE_DIR / "ia_reference_bank.csv", index=False)

            test_orientation = np.vstack(joined_all["shape_major_axis"])
            test_reference = np.vstack(joined_all["radial_vector"])
            sign_test = np.allclose(
                hod.alignment_cos2_minus_one_third(test_orientation, test_reference),
                hod.alignment_cos2_minus_one_third(-test_orientation, -test_reference),
            )
            print("Resolved references:", sorted(reference_bank))
            print("Axial sign-invariance test:", sign_test)
            display(reference_summary)
            """
        ),
        markdown(
            """
            ## 9. Component-level IA-HOD measurements

            Each component uses a distinct physical reference axis and population.
            This stage measures the main data vector:

            - central-host and central-tidal alignment;
            - satellite-radial, host, subhalo, tidal, velocity, and spin alignment;
            - binding-energy-layer and figure-rotation alignment.

            The results are saved independently so downstream fitting can select only
            the components relevant to a particular physical model.
            """
        ),
        code(
            """
            component_calls = {
                "central_host": hod.measure_central_host_alignment,
                "central_tidal": hod.measure_central_tidal_alignment,
                "satellite_radial": hod.measure_satellite_radial_alignment,
                "satellite_host": hod.measure_satellite_host_alignment,
                "satellite_subhalo": hod.measure_satellite_subhalo_alignment,
                "satellite_tidal": hod.measure_satellite_tidal_alignment,
                "satellite_velocity": hod.measure_satellite_velocity_alignment,
                "satellite_spin": hod.measure_satellite_spin_alignment,
                "binding_layer": hod.measure_binding_layer_alignment,
                "figure_rotation": hod.measure_figure_rotation_alignment,
            }
            ia_components = {}
            component_rows = []
            for name, function in component_calls.items():
                measurement = function(catalog)
                ia_components[name] = measurement
                hod.save_ia_component_hdf5(measurement, DATA_DIR / f"ia_component_{name}.hdf5")
                component_rows.append({
                    "component": name,
                    "reference": measurement.reference,
                    "population": measurement.population,
                    "value": float(np.asarray(measurement.values)),
                    "error": float(np.asarray(measurement.errors)),
                    "count": int(np.asarray(measurement.counts)),
                })
            component_summary = pd.DataFrame(component_rows)
            component_summary.to_csv(TABLE_DIR / "ia_component_summary.csv", index=False)

            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.errorbar(
                np.arange(len(component_summary)),
                component_summary["value"],
                yerr=component_summary["error"],
                fmt="o",
            )
            ax.set_xticks(np.arange(len(component_summary)), component_summary["component"], rotation=45, ha="right")
            ax.axhline(0.0, color="0.7", lw=0.8)
            ax.set_ylabel("Alignment A")
            ax.grid(alpha=0.2)
            save_figure(fig, "ia_component_summary")
            display(component_summary)
            """
        ),
        markdown(
            """
            ## 10. IA dependence on mass, radius, sample, assembly, and layer

            A global IA amplitude is not sufficient for this project. This stage
            computes:

            - a mass-radius grid for satellite radial alignment;
            - LRG and ELG component measurements;
            - IA split by fixed-mass concentration and environment residuals;
            - alignment in binding-energy layers.
            """
        ),
        code(
            """
            mass_radius_alignment = hod.measure_mass_radius_alignment_grid(
                catalog,
                component="satellite_radial",
                reference="radial_vector",
                population="satellite",
                mass_edges=MASS_EDGES,
                radius_edges=RADIUS_EDGES,
            )
            hod.save_ia_component_hdf5(mass_radius_alignment, DATA_DIR / "ia_mass_radius_satellite_radial.hdf5")

            sample_alignment = hod.measure_sample_dependent_ia_hod(
                catalog,
                sample_labels=("LRG", "ELG"),
                component="satellite_radial",
                reference="radial_vector",
                population="satellite",
                mass_edges=MASS_EDGES,
            )
            concentration_alignment = hod.measure_assembly_dependent_ia_hod(
                catalog,
                component="satellite_radial_concentration",
                reference="radial_vector",
                population="satellite",
                mass_edges=MASS_EDGES,
                secondary_column="concentration_standardized",
                secondary_edges=SECONDARY_EDGES,
            )
            environment_alignment = hod.measure_assembly_dependent_ia_hod(
                catalog,
                component="satellite_radial_environment",
                reference="radial_vector",
                population="satellite",
                mass_edges=MASS_EDGES,
                secondary_column="environment_standardized",
                secondary_edges=SECONDARY_EDGES,
            )
            layer_alignment = hod.measure_alignment_hod_components(
                catalog,
                component="binding_layer_alignment",
                reference="binding_energy_layer_axis",
                population="satellite",
                layer_column="binding_energy_layer",
            )

            for name, measurement in {
                **{f"sample_{key}": value for key, value in sample_alignment.items()},
                "concentration": concentration_alignment,
                "environment": environment_alignment,
                "binding_layer": layer_alignment,
            }.items():
                hod.save_ia_component_hdf5(measurement, DATA_DIR / f"ia_conditional_{name}.hdf5")

            fig, ax = hod_plot.plot_mass_radius_alignment_grid(mass_radius_alignment)
            ax.set_title("Satellite radial alignment: mass-radius grid")
            save_figure(fig, "ia_mass_radius_grid")
            fig, ax = hod_plot.plot_ia_component_comparison(sample_alignment)
            ax.set_title("LRG and ELG satellite radial alignment")
            save_figure(fig, "ia_lrg_elg_comparison")
            display(pd.DataFrame({
                "layer": layer_alignment.layer_labels,
                "alignment": np.asarray(layer_alignment.values),
                "count": np.asarray(layer_alignment.counts),
            }))
            """
        ),
        markdown(
            """
            ## 11. Conditional IA-strength models and fitting

            Each component has a bounded conditional field

            `mu_k = tanh(a0 + aM log10(M/M0) + ar log10(x/x0) + aS S_tilde + sample_term + layer_term)`.

            This cell constructs a multi-component model and fits a simple mass trend
            to the measured satellite-radial alignment. The model is deliberately
            lightweight and uses SciPy least squares rather than MCMC.
            """
        ),
        code(
            """
            satellite_mass_alignment = hod.measure_alignment_hod_components(
                catalog,
                component="satellite_radial",
                reference="radial_vector",
                population="satellite",
                mass_edges=MASS_EDGES,
            )
            mass_centers = np.sqrt(MASS_EDGES[:-1] * MASS_EDGES[1:])
            valid = (
                np.asarray(satellite_mass_alignment.counts) > 0
            ) & np.isfinite(satellite_mass_alignment.values)
            ia_fit = hod.fit_ia_component_model(
                mass_centers[valid],
                np.asarray(satellite_mass_alignment.values)[valid],
                errors=np.maximum(np.asarray(satellite_mass_alignment.errors)[valid], 0.02),
                name="satellite_radial",
                reference="radial_vector",
                population="satellite",
            )
            fitted_component = ia_fit["model"]

            central_host_model = hod.IAComponentModel(
                "central_host",
                "host_major_axis",
                "central",
                hod.IAStrengthParameters(mu0=0.25, beta_mass=0.20),
                sample_terms={"LRG": 0.12, "ELG": -0.08},
            )
            satellite_radial_model = fitted_component
            satellite_tidal_model = hod.IAComponentModel(
                "satellite_tidal",
                "tidal_major_axis",
                "satellite",
                hod.IAStrengthParameters(mu0=-0.05, beta_mass=0.10, beta_radius=-0.18, beta_secondary=0.20),
                layer_terms={"inner": 0.15, "middle": 0.0, "outer": -0.12},
            )
            component_model = hod.ComponentIAHODModel(
                [central_host_model, satellite_radial_model, satellite_tidal_model]
            )

            model_curve = hod.predict_ia_component(fitted_component, mass=mass_centers, radius=0.5, secondary=0.0)
            ia_fit_table = pd.DataFrame({
                "mass": mass_centers,
                "measured": satellite_mass_alignment.values,
                "error": satellite_mass_alignment.errors,
                "model": model_curve,
                "count": satellite_mass_alignment.counts,
            })
            ia_fit_table.to_csv(TABLE_DIR / "ia_satellite_radial_fit.csv", index=False)

            fig, ax = hod_plot.plot_alignment_component(satellite_mass_alignment, model=model_curve)
            ax.set_title("Conditional IA-strength fit")
            save_figure(fig, "ia_conditional_model_fit")
            print("IA fit parameters:", fitted_component.parameters)
            """
        ),
        markdown(
            """
            ## 12. Optional orientation forward model

            The long-term target is a multi-reference axial distribution

            `p(e) proportional to exp[sum_k kappa_k (e dot q_k)^2]`.

            The first implementation provides deterministic component moments and a
            robust single-reference sampler. This cell assigns mock orientations
            around satellite radial vectors and verifies their mean axial alignment.
            """
        ),
        code(
            """
            satellite_frame = joined_satellites.reset_index(drop=True)
            mock_orientations = hod.sample_orientations_from_reference(
                np.vstack(satellite_frame["radial_vector"]),
                kappa=3.0,
                random_state=RANDOM_SEED,
            )
            mock_catalog = hod.assign_mock_orientations(
                satellite_frame,
                np.vstack(satellite_frame["radial_vector"]),
                kappa=3.0,
                random_state=RANDOM_SEED,
            )
            mock_alignment = hod.measure_reference_alignment(
                mock_orientations,
                np.vstack(satellite_frame["radial_vector"]),
            )
            forward_moments = hod.predict_orientation_moments(
                component_model.predict(
                    mass=satellite_frame["mass"].to_numpy(),
                    radius=satellite_frame["r_over_rvir"].to_numpy(),
                    secondary=satellite_frame["concentration_standardized"].to_numpy(),
                )
            )
            pd.DataFrame(mock_orientations, columns=["orientation_x", "orientation_y", "orientation_z"]).to_csv(
                TABLE_DIR / "mock_orientations.csv", index=False
            )
            print("Mock orientation radial alignment:", mock_alignment)
            print("Forward-moment components:", tuple(forward_moments))
            """
        ),
        markdown(
            """
            ## 13. Pairwise xi, omega, and eta validation

            The first-PR estimators use simple NumPy pair loops and support
            central-central, central-satellite, satellite-satellite, all-all,
            one-halo, and two-halo categories. They are intended for small validation
            samples; optimized correlation wrappers can be added later.
            """
        ),
        code(
            """
            pair_sample = catalog.galaxies.sample(min(len(catalog.galaxies), 450), random_state=RANDOM_SEED)
            positions = np.vstack(pair_sample["position"])
            orientations = np.vstack(pair_sample["shape_major_axis"])
            pair_edges = np.logspace(-2, np.log10(80.0), 16)

            pairwise_results = {}
            for category in ("all-all", "central-central", "central-satellite", "satellite-satellite", "1-halo", "2-halo"):
                pairwise_results[category] = hod.measure_pairwise_ia(
                    positions,
                    orientations,
                    pair_edges,
                    is_central=pair_sample["is_central"].to_numpy(),
                    host_id=pair_sample["halo_id"].to_numpy(),
                    category=category,
                )
                pd.DataFrame({
                    "r": pairwise_results[category]["rmid"],
                    "omega": pairwise_results[category]["omega"],
                    "eta": pairwise_results[category]["eta"],
                    "count": pairwise_results[category]["counts"],
                }).to_csv(TABLE_DIR / f"pairwise_ia_{category}.csv", index=False)

            xi_summary = hod.measure_xi_gg(positions, pair_edges)
            fig, ax = hod_plot.plot_pairwise_omega_eta(pairwise_results["all-all"])
            ax.set_title("All-pair omega and eta")
            save_figure(fig, "pairwise_omega_eta")
            display(pd.DataFrame({
                "r": pairwise_results["all-all"]["rmid"],
                "omega": pairwise_results["all-all"]["omega"],
                "eta": pairwise_results["all-all"]["eta"],
                "pairs": pairwise_results["all-all"]["counts"],
            }))
            """
        ),
        markdown(
            """
            ## 14. Covariance and uncertainty utilities

            This stage demonstrates diagonal covariance, covariance regularization,
            bootstrap resampling, and jackknife resampling on the object-level radial
            alignment statistic.
            """
        ),
        code(
            """
            satellite_alignment_values = hod.alignment_cos2_minus_one_third(
                np.vstack(satellite_frame["shape_major_axis"]),
                np.vstack(satellite_frame["radial_vector"]),
            )
            bootstrap = hod.bootstrap_measurement(
                satellite_alignment_values,
                lambda values: np.array([np.mean(values)]),
                n_resamples=256,
                random_state=RANDOM_SEED,
            )
            jackknife = hod.jackknife_measurement(
                satellite_alignment_values,
                lambda values: np.array([np.mean(values)]),
            )
            diagonal = hod.diagonal_covariance([max(float(np.std(satellite_alignment_values)), 1e-3)])
            regularized = hod.regularize_covariance(diagonal)
            covariance_summary = pd.DataFrame({
                "method": ["bootstrap", "jackknife", "diagonal_regularized"],
                "mean": [
                    float(bootstrap["mean"][0]),
                    float(jackknife["mean"][0]),
                    float(np.mean(satellite_alignment_values)),
                ],
                "variance": [
                    float(np.atleast_2d(bootstrap["covariance"])[0, 0]),
                    float(np.atleast_2d(jackknife["covariance"])[0, 0]),
                    float(regularized[0, 0]),
                ],
            })
            covariance_summary.to_csv(TABLE_DIR / "ia_covariance_summary.csv", index=False)
            display(covariance_summary)
            """
        ),
        markdown(
            """
            ## 15. Final product inventory and reproducibility record

            The final cell records configuration, package coverage, saved tables,
            HDF5 measurements, and figures. This makes it straightforward to compare
            runs with different samples, mass bins, secondary properties, or plotting
            settings.
            """
        ),
        code(
            """
            run_configuration = {
                "input_mode": INPUT_MODE,
                "project_root": str(PROJECT_ROOT),
                "workflow_dir": str(WORKFLOW_DIR),
                "output_dir": str(OUTPUT_DIR),
                "random_seed": RANDOM_SEED,
                "box_size": BOX_SIZE,
                "sample_labels": list(SAMPLE_LABELS),
                "mass_edges": MASS_EDGES.tolist(),
                "radius_edges": RADIUS_EDGES.tolist(),
                "secondary_properties": ["concentration", "environment", "tidal_anisotropy"],
                "ia_components": list(ia_components),
            }
            (OUTPUT_DIR / "run_configuration.json").write_text(
                json.dumps(run_configuration, indent=2), encoding="utf-8"
            )

            product_inventory = pd.DataFrame([
                {
                    "path": str(path.relative_to(OUTPUT_DIR)),
                    "size_bytes": path.stat().st_size,
                    "suffix": path.suffix,
                }
                for path in sorted(OUTPUT_DIR.rglob("*"))
                if path.is_file()
            ])
            product_inventory.to_csv(OUTPUT_DIR / "product_inventory.csv", index=False)
            display(product_inventory)
            print(f"Workflow completed. Wrote {len(product_inventory)} products to {OUTPUT_DIR}")
            """
        ),
        markdown(
            """
            ## Interpretation and next steps

            This notebook separates three physically distinct channels:

            1. **Occupation assembly bias** — secondary properties change the number
               of centrals or satellites at fixed halo mass.
            2. **Phase-space assembly bias** — secondary properties change satellite
               radius, velocity anisotropy, or binding-energy-layer occupation.
            3. **IA assembly bias** — secondary properties change alignment strength
               at fixed mass and radius.

            For production analysis, replace the synthetic input with project
            catalogues, verify units and central/satellite definitions, increase pair
            sample size only when computationally appropriate, and fit each IA
            component with a covariance matched to the simulation or survey volume.
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3.12 (py312)",
        "language": "python",
        "name": "py312",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return notebook


def write_notebook(path: Path, default_workflow_expression: str) -> None:
    """Write a deterministic notebook with cleared outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(default_workflow_expression)
    nbf.write(notebook, path)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-only", action="store_true")
    parser.add_argument("--external-only", action="store_true")
    args = parser.parse_args()
    if not args.external_only:
        write_notebook(INTERNAL_NOTEBOOK, 'PROJECT_ROOT / "notebooks" / "pipelines"')
    if not args.internal_only:
        write_notebook(EXTERNAL_NOTEBOOK, 'PROJECT_ROOT.parent / "pipeline" / "02_hod_lrg_elg"')


if __name__ == "__main__":
    main()
