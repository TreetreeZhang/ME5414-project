from __future__ import annotations

import argparse
import logging
from pathlib import Path

from me5414.core.config import ExperimentConfig
from me5414.io.logging_utils import setup_logging
from me5414.pipeline.runner import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ME5414 LP solver comparison experiments")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_default.json"))
    parser.add_argument("--seed", type=int, default=5414)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-fixed", type=int, default=80)
    parser.add_argument("--m-fixed", type=int, default=80)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def run_cli() -> None:
    args = parse_args()
    logger = setup_logging(log_dir=args.log_dir, logger_name="solver_run", level=logging.INFO)

    cfg = ExperimentConfig.from_json(args.config) if args.config.exists() else ExperimentConfig()
    cfg.seed = args.seed
    cfg.repeats = args.repeats
    cfg.n_fixed = args.n_fixed
    cfg.m_fixed = args.m_fixed

    run_experiments(cfg=cfg, out_dir=args.out_dir, logger=logger)
