from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure local package import works without requiring PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from me5414.logging_utils import setup_logging


def main() -> None:
    logger = setup_logging(log_dir="logs", logger_name="env_check", level=logging.INFO)
    logger.info("Environment check started.")

    import numpy as np
    import scipy
    import pandas as pd
    import matplotlib

    logger.info("numpy version: %s", np.__version__)
    logger.info("scipy version: %s", scipy.__version__)
    logger.info("pandas version: %s", pd.__version__)
    logger.info("matplotlib version: %s", matplotlib.__version__)
    logger.info("Environment check completed successfully.")


if __name__ == "__main__":
    main()
