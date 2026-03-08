#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="me5414-lp"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] Environment '$ENV_NAME' already exists."
else
  echo "[INFO] Creating environment '$ENV_NAME' from environment.yml ..."
  conda env create -f environment.yml
fi

echo "[INFO] Verifying packages inside '$ENV_NAME' ..."
conda run -n "$ENV_NAME" python -c "import numpy, scipy, pandas, matplotlib; print('OK:', numpy.__version__, scipy.__version__, pandas.__version__, matplotlib.__version__)"

echo "[DONE] Environment is ready."
echo "Activate it with: conda activate $ENV_NAME"
echo "Run project with: python scripts/run_experiments.py --repeats 5"
