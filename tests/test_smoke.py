from __future__ import annotations

import numpy as np

from me5414.generation.problem_generator import generate_lp_problem
from me5414.solvers.linear_solvers import solve_with_simplex, solve_with_ipm


def test_problem_and_solvers_smoke() -> None:
    rng = np.random.default_rng(5414)
    p = generate_lp_problem(
        n=24,
        m=36,
        scenario="degenerate",
        active_ratio=0.6,
        zero_ratio=0.35,
        rng=rng,
    )
    simplex = solve_with_simplex(p.c, p.A, p.b, tol=1e-8)
    ipm = solve_with_ipm(p.c, p.A, p.b, tol=1e-8)

    assert simplex.success
    assert ipm.success
