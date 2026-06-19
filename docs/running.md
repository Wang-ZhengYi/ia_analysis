# Running

## Legacy Entrypoints

Run these from the repository root:

```bash
python run_cs.py --help
python run_tng.py --help
python ia_pk_cs.py --help
python Cov.py --help
```

## Package Entrypoints

After installing the package, or when `PYTHONPATH=src` is set:

```bash
python -m ia_analysis.pipelines.run_cs --help
python -m ia_analysis.pipelines.run_tng --help
python -m ia_analysis.spectra.ia_pk_cs --help
python -m ia_analysis.covariance.Cov --help
```

## Configuration

Current CLIs keep their historical arguments. `configs/example_paths.json`
documents the path conventions used on COSMA and the environment variables used
by the TNG loader:

- `TNG_API_KEY`
- `TNG_SIM_NAME`
- `TNG_CACHE_DIR`

Prefer passing data paths through CLI flags or environment variables rather than
editing source files.

