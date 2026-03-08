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
    scenario: str
    density: float


@dataclass
class SolveResult:
    method: str
    success: bool
    status: int
    message: str
    objective: float
    iterations: int
    runtime_sec: float
