from __future__ import annotations

import numpy as np

from me5414.core.models import LPProblem


def _build_matrix(n: int, m: int, scenario: str, rng: np.random.Generator) -> np.ndarray:
    if scenario == "baseline":
        A = rng.uniform(0.2, 2.0, size=(m, n))
        return A

    if scenario == "degenerate":
        A = rng.uniform(0.05, 1.8, size=(m, n))
        clones = max(1, m // 6)
        for row in range(clones):
            src = int(rng.integers(0, m))
            noise = rng.normal(0.0, 2e-4, size=n)
            A[row] = np.clip(A[src] * (1.0 + rng.normal(0.0, 8e-4)) + noise, 1e-8, None)
        return A

    if scenario == "ill_conditioned":
        rank = max(5, min(m, n) // 6)
        U = rng.uniform(0.05, 1.2, size=(m, rank))
        V = rng.uniform(0.05, 1.2, size=(rank, n))
        A = U @ V + 1e-4 * rng.uniform(0.0, 1.0, size=(m, n))
        row_scale = 10.0 ** rng.uniform(-3.0, 3.0, size=m)
        col_scale = 10.0 ** rng.uniform(-2.0, 2.0, size=n)
        A = (row_scale[:, None] * A) * col_scale[None, :]
        return np.clip(A, 1e-10, None)

    raise ValueError(f"Unsupported scenario: {scenario}")


def generate_lp_problem(
    n: int,
    m: int,
    scenario: str,
    active_ratio: float,
    zero_ratio: float,
    rng: np.random.Generator,
) -> LPProblem:
    """
    Generate a bounded LP max c^T x s.t. Ax<=b, x>=0 with known optimum x*.
    The construction satisfies KKT conditions by design.
    """
    A = _build_matrix(n=n, m=m, scenario=scenario, rng=rng)

    x_star = rng.uniform(0.0, 3.5, size=n)
    zero_mask = rng.random(size=n) < zero_ratio
    x_star[zero_mask] = 0.0

    num_active = int(max(1, min(m - 1, round(active_ratio * m))))
    active_idx = rng.choice(m, size=num_active, replace=False)
    active_mask = np.zeros(m, dtype=bool)
    active_mask[active_idx] = True

    y = np.zeros(m, dtype=float)
    y[active_mask] = rng.uniform(0.25, 2.2, size=num_active)

    s = np.zeros(m, dtype=float)
    if scenario == "degenerate":
        s[~active_mask] = rng.uniform(1e-8, 2e-4, size=np.count_nonzero(~active_mask))
    elif scenario == "ill_conditioned":
        s[~active_mask] = 10.0 ** rng.uniform(-7.0, -2.0, size=np.count_nonzero(~active_mask))
    else:
        s[~active_mask] = rng.uniform(1e-2, 1.0, size=np.count_nonzero(~active_mask))

    r = np.zeros(n, dtype=float)
    r[zero_mask] = rng.uniform(0.05, 1.2, size=np.count_nonzero(zero_mask))

    c = A.T @ y - r
    b = A @ x_star + s
    obj_star = float(c @ x_star)

    density = float(np.count_nonzero(A) / A.size)
    return LPProblem(
        c=c,
        A=A,
        b=b,
        known_opt_x=x_star,
        known_opt_obj=obj_star,
        scenario=scenario,
        density=density,
    )
