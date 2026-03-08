from __future__ import annotations

import time
import numpy as np
from scipy.optimize import linprog

from me5414.core.models import SolveResult


def solve_with_simplex(c: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float) -> SolveResult:
    t0 = time.perf_counter()
    res = linprog(
        c=-c,
        A_ub=A,
        b_ub=b,
        bounds=[(0, None)] * len(c),
        method="highs-ds",  # dual simplex
        options={"dual_feasibility_tolerance": tol, "primal_feasibility_tolerance": tol},
    )
    dt = time.perf_counter() - t0
    obj = float(c @ res.x) if res.success and res.x is not None else float("nan")
    return SolveResult(
        method="simplex(highs-ds)",
        success=bool(res.success),
        status=int(res.status),
        message=str(res.message),
        objective=obj,
        iterations=int(getattr(res, "nit", -1)),
        runtime_sec=dt,
    )


def solve_with_ipm(c: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float) -> SolveResult:
    t0 = time.perf_counter()
    res = linprog(
        c=-c,
        A_ub=A,
        b_ub=b,
        bounds=[(0, None)] * len(c),
        method="highs-ipm",
        options={"dual_feasibility_tolerance": tol, "primal_feasibility_tolerance": tol},
    )
    dt = time.perf_counter() - t0
    obj = float(c @ res.x) if res.success and res.x is not None else float("nan")
    return SolveResult(
        method="interior-point(highs-ipm)",
        success=bool(res.success),
        status=int(res.status),
        message=str(res.message),
        objective=obj,
        iterations=int(getattr(res, "nit", -1)),
        runtime_sec=dt,
    )
