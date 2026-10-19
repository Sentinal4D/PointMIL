#!/usr/bin/env bash
# Create the conda environment, install the package and run the unit tests.
# No CUDA extensions are compiled: PointMIL is pure PyTorch.
# Uses mamba or micromamba when available (much faster solver), otherwise conda.
# Override with CONDA_TOOL=conda|mamba|micromamba and ENV_NAME=<name>.
set -euo pipefail
cd "$(dirname "$0")"
ENV_NAME=${ENV_NAME:-pointmil}
if [ -z "${CONDA_TOOL:-}" ]; then
  for t in mamba micromamba conda; do
    if command -v "$t" >/dev/null 2>&1; then CONDA_TOOL=$t; break; fi
  done
fi
if [ -z "${CONDA_TOOL:-}" ]; then
  echo "conda, mamba or micromamba not found. Install Miniforge (https://github.com/conda-forge/miniforge) or use: pip install -e ." >&2
  exit 1
fi
echo "using $CONDA_TOOL"
if "$CONDA_TOOL" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "environment '$ENV_NAME' already exists; installing the package into it (remove it first for a clean build)"
else
  "$CONDA_TOOL" create -y -n "$ENV_NAME" -f environment.yml 2>/dev/null || "$CONDA_TOOL" env create -n "$ENV_NAME" -f environment.yml
fi
run() { "$CONDA_TOOL" run -n "$ENV_NAME" "$@"; }
run pip install -e ".[dev]"
run python -c "import torch, pointmil; print('pointmil', pointmil.__version__, '| torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
run python -m pytest tests -q --ignore=tests/equivalence || { echo "unit tests failed; see output above" >&2; exit 1; }
echo "Installed. Activate with '$CONDA_TOOL activate $ENV_NAME' and set POINTMIL_DATA to your dataset root (see README.md, Data preparation)."
