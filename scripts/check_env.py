from __future__ import annotations

import logging

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
