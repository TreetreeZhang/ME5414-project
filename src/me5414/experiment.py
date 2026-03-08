from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
import pandas as pd
import numpy as np

from me5414.problem_generator import generate_lp_problem
from me5414.solvers import solve_with_simplex, solve_with_ipm


@dataclass
class ExperimentConfig:
    seed: int = 5414
    repeats: int = 5
    n_fixed: int = 80
    m_fixed: int = 80
    n_values: tuple[int, ...] = (20, 40, 80, 120, 160)
    m_values: tuple[int, ...] = (20, 40, 80, 120, 160)
    tolerances: tuple[float, ...] = (1e-6, 1e-7, 1e-8, 1e-9)


def _run_case(section: str, n: int, m: int, tol: float, repeat_id: int, rng: np.random.Generator, logger: logging.Logger):
    p = generate_lp_problem(n=n, m=m, rng=rng)
    rs = solve_with_simplex(p.c, p.A, p.b, tol=tol)
    ri = solve_with_ipm(p.c, p.A, p.b, tol=tol)

    logger.info(
        "section=%s repeat=%d n=%d m=%d tol=%.1e | simplex: ok=%s time=%.6fs nit=%d obj=%.8f | ipm: ok=%s time=%.6fs nit=%d obj=%.8f",
        section,
        repeat_id,
        n,
        m,
        tol,
        rs.success,
        rs.runtime_sec,
        rs.iterations,
        rs.objective,
        ri.success,
        ri.runtime_sec,
        ri.iterations,
        ri.objective,
    )

    return [
        {
            "section": section,
            "repeat": repeat_id,
            "n": n,
            "m": m,
            "tol": tol,
            "known_opt_obj": p.known_opt_obj,
            "abs_err": abs(rs.objective - p.known_opt_obj),
            **asdict(rs),
        },
        {
            "section": section,
            "repeat": repeat_id,
            "n": n,
            "m": m,
            "tol": tol,
            "known_opt_obj": p.known_opt_obj,
            "abs_err": abs(ri.objective - p.known_opt_obj),
            **asdict(ri),
        },
    ]


def run_experiments(cfg: ExperimentConfig, out_dir: Path, logger: logging.Logger) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    rows: list[dict] = []
    t0 = time.perf_counter()

    logger.info("Starting experiments with config=%s", cfg)

    for n in cfg.n_values:
        for k in range(cfg.repeats):
            rows.extend(_run_case("vary_n", n, cfg.m_fixed, 1e-8, k, rng, logger))

    for m in cfg.m_values:
        for k in range(cfg.repeats):
            rows.extend(_run_case("vary_m", cfg.n_fixed, m, 1e-8, k, rng, logger))

    for tol in cfg.tolerances:
        for k in range(cfg.repeats):
            rows.extend(_run_case("vary_tol", cfg.n_fixed, cfg.m_fixed, tol, k, rng, logger))

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["section", "method"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
            mean_iterations=("iterations", "mean"),
            mean_abs_err=("abs_err", "mean"),
        )
        .sort_values(["section", "mean_runtime_sec"])
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"results_{stamp}.csv"
    summary_path = out_dir / f"summary_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)

    logger.info("Saved detailed results: %s", csv_path)
    logger.info("Saved summary results: %s", summary_path)
    logger.info("Finished experiments in %.2fs", time.perf_counter() - t0)
    return csv_path, summary_path
