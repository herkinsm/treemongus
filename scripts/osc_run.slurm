#!/usr/bin/env bash
#SBATCH --account=PAS0228
#SBATCH --job-name=sam3-apples
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=logs/sam3-apples-%j.out
#SBATCH --error=logs/sam3-apples-%j.err

# Submit from the login node of the cluster you want:
#   ssh cardinal.osc.edu      then   sbatch scripts/osc_run.slurm     (H100)
#   ssh ascend.osc.edu        then   sbatch scripts/osc_run.slurm     (A100)
#
# Override paths by exporting before sbatch, e.g.
#   DATA_ROOT=/fs/scratch/PAS0228/mherkins/All2023 sbatch scripts/osc_run.slurm

set -euo pipefail

module load miniconda3
source activate sam3

# --- paths ---------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$HOME/All2023}"
OUT_DIR="${OUT_DIR:-$HOME/sam3_out}"
CODE_DIR="${CODE_DIR:-$HOME/sam3-apple-analysis}"

# --- sanity --------------------------------------------------------------
mkdir -p "$OUT_DIR" logs
echo "[run] host=$(hostname) gpu:"
nvidia-smi || { echo "no GPU on this node"; exit 1; }
echo "[run] DATA_ROOT=$DATA_ROOT"
echo "[run] OUT_DIR=$OUT_DIR"
echo "[run] CODE_DIR=$CODE_DIR"

# --- run -----------------------------------------------------------------
cd "$CODE_DIR"
python analyze_days.py --root "$DATA_ROOT" --out "$OUT_DIR"
