from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    seed: int = 5414
    repeats: int = 5
    n_fixed: int = 80
    m_fixed: int = 80
    n_values: tuple[int, ...] = (20, 40, 80, 120, 160)
    m_values: tuple[int, ...] = (20, 40, 80, 120, 160)
    tolerances: tuple[float, ...] = (1e-6, 1e-7, 1e-8, 1e-9)
    scenarios: tuple[str, ...] = ("baseline", "degenerate", "ill_conditioned")
    active_ratio: float = 0.6
    zero_ratio: float = 0.35


    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            seed=int(data.get("seed", cls.seed)),
            repeats=int(data.get("repeats", cls.repeats)),
            n_fixed=int(data.get("n_fixed", cls.n_fixed)),
            m_fixed=int(data.get("m_fixed", cls.m_fixed)),
            n_values=tuple(int(x) for x in data.get("n_values", cls.n_values)),
            m_values=tuple(int(x) for x in data.get("m_values", cls.m_values)),
            tolerances=tuple(float(x) for x in data.get("tolerances", cls.tolerances)),
            scenarios=tuple(str(x) for x in data.get("scenarios", cls.scenarios)),
            active_ratio=float(data.get("active_ratio", cls.active_ratio)),
            zero_ratio=float(data.get("zero_ratio", cls.zero_ratio)),
        )
