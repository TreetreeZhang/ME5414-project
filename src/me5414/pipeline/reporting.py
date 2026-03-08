from __future__ import annotations

from pathlib import Path
import time
import pandas as pd


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["section", "scenario", "method"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
            mean_iterations=("iterations", "mean"),
            mean_abs_err=("abs_err", "mean"),
        )
        .sort_values(["section", "scenario", "mean_runtime_sec"])
    )


def create_run_dir(out_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, run_dir: Path, metadata: dict[str, str]) -> tuple[Path, Path]:
    csv_path = run_dir / "results.csv"
    summary_path = run_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)

    metadata_lines = [f"{k}={v}" for k, v in metadata.items()]
    (run_dir / "metadata.txt").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    return csv_path, summary_path
