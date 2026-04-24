#!/usr/bin/env bash
# Build the SAM 3 env on OSC (Ascend or Cardinal).
# Run this ONCE on a login node: bash scripts/osc_setup.sh
#
# Creates:
#   ~/miniconda3/envs/sam3        Python 3.12 env with torch+CUDA + SAM 3
#   ~/sam3                        Meta's SAM 3 repo (editable install)

set -euo pipefail

ENV_NAME="sam3"
SAM3_DIR="$HOME/sam3"

echo "[setup] loading miniconda3 module"
module load miniconda3/24.1.2-py310

# Create env if missing
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] creating conda env '$ENV_NAME' (python 3.12)"
    conda create -y -n "$ENV_NAME" python=3.12
else
    echo "[setup] conda env '$ENV_NAME' already exists, skipping create"
fi

# Activate
# shellcheck disable=SC1091
source activate "$ENV_NAME"

echo "[setup] installing torch + CUDA 12.8 wheels"
pip install --upgrade pip
pip install "torch==2.10.0" torchvision --index-url https://download.pytorch.org/whl/cu128

# Clone + install SAM 3 (editable)
if [ ! -d "$SAM3_DIR" ]; then
    echo "[setup] cloning Meta SAM 3 into $SAM3_DIR"
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
fi
echo "[setup] pip install -e sam3"
(cd "$SAM3_DIR" && pip install -e .)

echo "[setup] extra deps used by analyze_days.py"
# SAM 3 still imports pkg_resources, which was dropped from setuptools 81.
# Pin to <81 until upstream migrates to importlib.metadata.
pip install "setuptools<81" numpy pillow matplotlib

echo "[setup] smoke test"
python - <<'PY'
import torch, sam3
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
print("sam3 module:", sam3.__file__)
PY

echo "[setup] DONE. Activate later with:"
echo "  module load miniconda3 && source activate $ENV_NAME"
