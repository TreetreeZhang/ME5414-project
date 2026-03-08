# ME5414 LP Solver Comparison (Even Matric Number)

For even matric number, the non-interior-point method must be **Simplex**.
This project compares:
- **Simplex** (`highs-ds`)
- **Interior-point** (`highs-ipm`)

## Project Structure

- `configs/experiment_default.json`: experiment configuration
- `scripts/setup_env.sh`: conda environment setup
- `scripts/check_env.py`: dependency check
- `scripts/run_experiments.py`: CLI entry point
- `src/me5414/core/`: config and shared dataclasses
- `src/me5414/generation/`: LP instance generation
- `src/me5414/solvers/`: solver wrappers
- `src/me5414/pipeline/`: experiment orchestration and reporting
- `src/me5414/cli/`: argument parsing and app bootstrap
- `src/me5414/io/`: persistent local logging
- `tests/test_smoke.py`: single smoke test suite

## Problem Scenarios

All generated LPs are bounded and have known optimum for error checking.

- `baseline`: regular dense instances
- `degenerate`: near-duplicate constraints + tiny slacks
- `ill_conditioned`: wide coefficient scaling to stress numerics

## Setup

```bash
bash scripts/setup_env.sh
conda activate me5414-lp
python scripts/check_env.py
```

## Run

```bash
python scripts/run_experiments.py --repeats 5
```

## Test

```bash
pytest
```

## Outputs

Each run writes to `outputs/runs/run_<timestamp>/`:
- `results.csv`
- `summary.csv`
- `metadata.txt`

Solver logs are saved locally under `logs/solver_run_*.log`.
