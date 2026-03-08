from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LPProblem:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    known_opt_x: np.ndarray
    known_opt_obj: float


def generate_lp_problem(n: int, m: int, rng: np.random.Generator) -> LPProblem:
    """Generate a bounded LP max c^T x s.t. Ax<=b, x>=0 with known optimum x*."""
    A = rng.uniform(0.2, 2.0, size=(m, n))

    # Create a sparse nonnegative optimal point x*.
    x_star = rng.uniform(0.0, 3.0, size=n)
    zero_mask = rng.random(size=n) < 0.35
    x_star[zero_mask] = 0.0

    y = rng.uniform(0.5, 2.0, size=m)
    r = rng.uniform(0.0, 0.8, size=n)
    r[~zero_mask] = 0.0  # complementary slackness: x_i*>0 => r_i=0

    c = A.T @ y - r
    b = A @ x_star  # all constraints active at optimum

    obj_star = float(c @ x_star)
    return LPProblem(c=c, A=A, b=b, known_opt_x=x_star, known_opt_obj=obj_star)
