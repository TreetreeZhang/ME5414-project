from __future__ import annotations

import argparse
from pathlib import Path
import logging

from me5414.logging_utils import setup_logging
from me5414.experiment import ExperimentConfig, run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ME5414 LP solver comparison experiments")
    parser.add_argument("--seed", type=int, default=5414)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-fixed", type=int, default=80)
    parser.add_argument("--m-fixed", type=int, default=80)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(log_dir=args.log_dir, logger_name="solver_run", level=logging.INFO)

    cfg = ExperimentConfig(
        seed=args.seed,
        repeats=args.repeats,
        n_fixed=args.n_fixed,
        m_fixed=args.m_fixed,
    )
    run_experiments(cfg=cfg, out_dir=args.out_dir, logger=logger)


if __name__ == "__main__":
    main()
