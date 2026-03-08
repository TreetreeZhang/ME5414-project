# ME5414 LP Solver Comparison (Even Matric No. -> Simplex)

This project compares:
- Non-interior-point method: **Simplex** (`highs-ds`)
- Interior-point method: **IPM** (`highs-ipm`)

Both solvers handle LP in the form:
- maximize `c^T x`
- subject to `A x <= b`, `x >= 0`

## Environment

```bash
conda env create -f environment.yml
conda activate me5414-lp
```

## Run Experiments

```bash
PYTHONPATH=src python scripts/run_experiments.py --repeats 5
```

## Outputs

- Detailed results: `outputs/results_*.csv`
- Aggregated summary: `outputs/summary_*.csv`
- Full solver logs: `logs/solver_run_*.log`

## Logging system

The logging system is in `src/me5414/logging_utils.py` and includes:
- Console + file logging
- Rotating file handler
- Timestamped log filenames
- Local persistence in `logs/`
