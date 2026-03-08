from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import logging
import time
import numpy as np
import pandas as pd

from me5414.core.config import ExperimentConfig
from me5414.generation.problem_generator import generate_lp_problem
from me5414.solvers.linear_solvers import solve_with_simplex, solve_with_ipm
from me5414.pipeline.reporting import build_summary, create_run_dir, write_outputs


def _solve_one_repeat(
    section: str,
    scenario: str,
    problem_n: int,
    problem_m: int,
    tol: float,
    repeat_id: int,
    known_opt_obj: float,
    density: float,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    logger: logging.Logger,
) -> list[dict]:
    simplex = solve_with_simplex(c, A, b, tol=tol)
    ipm = solve_with_ipm(c, A, b, tol=tol)

    logger.info(
        "section=%s scenario=%s repeat=%d n=%d m=%d tol=%.1e dens=%.3f | simplex: ok=%s time=%.6fs nit=%d obj=%.8f | ipm: ok=%s time=%.6fs nit=%d obj=%.8f",
        section,
        scenario,
        repeat_id,
        problem_n,
        problem_m,
        tol,
        density,
        simplex.success,
        simplex.runtime_sec,
        simplex.iterations,
        simplex.objective,
        ipm.success,
        ipm.runtime_sec,
        ipm.iterations,
        ipm.objective,
    )

    rows = []
    for result in (simplex, ipm):
        rows.append(
            {
                "section": section,
                "scenario": scenario,
                "repeat": repeat_id,
                "n": problem_n,
                "m": problem_m,
                "tol": tol,
                "density": density,
                "known_opt_obj": known_opt_obj,
                "abs_err": abs(result.objective - known_opt_obj),
                **asdict(result),
            }
        )
    return rows


def run_experiments(cfg: ExperimentConfig, out_dir: Path, logger: logging.Logger) -> tuple[Path, Path]:
    run_dir = create_run_dir(out_dir)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict] = []

    t0 = time.perf_counter()
    logger.info("Starting experiments with config=%s", cfg)

    for scenario in cfg.scenarios:
        for n in cfg.n_values:
            p = generate_lp_problem(
                n=n,
                m=cfg.m_fixed,
                scenario=scenario,
                active_ratio=cfg.active_ratio,
                zero_ratio=cfg.zero_ratio,
                rng=rng,
            )
            for rep in range(cfg.repeats):
                rows.extend(
                    _solve_one_repeat(
                        section="vary_n",
                        scenario=scenario,
                        problem_n=n,
                        problem_m=cfg.m_fixed,
                        tol=1e-8,
                        repeat_id=rep,
                        known_opt_obj=p.known_opt_obj,
                        density=p.density,
                        c=p.c,
                        A=p.A,
                        b=p.b,
                        logger=logger,
                    )
                )

        for m in cfg.m_values:
            p = generate_lp_problem(
                n=cfg.n_fixed,
                m=m,
                scenario=scenario,
                active_ratio=cfg.active_ratio,
                zero_ratio=cfg.zero_ratio,
                rng=rng,
            )
            for rep in range(cfg.repeats):
                rows.extend(
                    _solve_one_repeat(
                        section="vary_m",
                        scenario=scenario,
                        problem_n=cfg.n_fixed,
                        problem_m=m,
                        tol=1e-8,
                        repeat_id=rep,
                        known_opt_obj=p.known_opt_obj,
                        density=p.density,
                        c=p.c,
                        A=p.A,
                        b=p.b,
                        logger=logger,
                    )
                )

        for tol in cfg.tolerances:
            p = generate_lp_problem(
                n=cfg.n_fixed,
                m=cfg.m_fixed,
                scenario=scenario,
                active_ratio=cfg.active_ratio,
                zero_ratio=cfg.zero_ratio,
                rng=rng,
            )
            for rep in range(cfg.repeats):
                rows.extend(
                    _solve_one_repeat(
                        section="vary_tol",
                        scenario=scenario,
                        problem_n=cfg.n_fixed,
                        problem_m=cfg.m_fixed,
                        tol=tol,
                        repeat_id=rep,
                        known_opt_obj=p.known_opt_obj,
                        density=p.density,
                        c=p.c,
                        A=p.A,
                        b=p.b,
                        logger=logger,
                    )
                )

    df = pd.DataFrame(rows)
    summary = build_summary(df)

    csv_path, summary_path = write_outputs(
        df=df,
        summary=summary,
        run_dir=run_dir,
        metadata={
            "seed": str(cfg.seed),
            "repeats": str(cfg.repeats),
            "scenarios": ",".join(cfg.scenarios),
        },
    )

    logger.info("Saved detailed results: %s", csv_path)
    logger.info("Saved summary results: %s", summary_path)
    logger.info("Run directory: %s", run_dir)
    logger.info("Finished experiments in %.2fs", time.perf_counter() - t0)
    return csv_path, summary_path
