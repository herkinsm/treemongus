#!/bin/bash
#SBATCH --job-name=sam3_flowers_all
#SBATCH --account=PAS0228
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.log

# Full-dataset CPU run. The CPU shim in analyze_days.py handles the
# CUDA / bfloat16 patches when no GPU is allocated, so this script
# does not request --gpus-per-node. CPU is also chosen because the
# GPU partition queue routinely sits at 4-8h+ priority pending,
# longer than the actual job.
#
# To submit:   sbatch run_flowers_all.sh
# To monitor:  squeue -u $USER
# To check:    tail -f sam3_flowers_all_<jobid>.log

cd ~/sam3-apple-analysis && git pull

# Make PyTorch use all the CPUs we asked SLURM for. Without this,
# OMP / MKL default to 1 thread and SAM 3 inference is single-
# threaded -- which is what makes bulk CPU runs feel painfully
# slow even on multi-core nodes.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[batch] threading: OMP/MKL/OPENBLAS=$OMP_NUM_THREADS"

# Activate the sam3 conda env. SLURM batch jobs start with a fresh
# shell, so even if the user has `conda activate sam3` in their
# interactive session, the job won't inherit it. Source conda's
# init script then explicitly activate. Falls back to the env's
# python binary directly if conda activation fails (e.g. conda
# init script moved).
if [ -f "$HOME/.conda/envs/sam3/bin/python" ]; then
    PY="$HOME/.conda/envs/sam3/bin/python"
elif [ -f "/users/PAS0228/mherkins/.conda/envs/sam3/bin/python" ]; then
    PY="/users/PAS0228/mherkins/.conda/envs/sam3/bin/python"
else
    PY="python"
fi

# Best-effort conda activation; the absolute-path PY above is the
# real safety net.
for conda_init in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/.conda/etc/profile.d/conda.sh" \
    "/apps/anaconda3/etc/profile.d/conda.sh" \
    "/users/PAS0228/mherkins/.conda/etc/profile.d/conda.sh"
do
    if [ -f "$conda_init" ]; then
        # shellcheck disable=SC1090
        . "$conda_init"
        conda activate sam3 2>/dev/null && break
    fi
done

echo "[batch] Using python: $PY"
"$PY" -c "import sys; print('[batch] python:', sys.executable); import sam3; print('[batch] sam3 OK')" \
    || { echo "[batch] sam3 import failed -- aborting"; exit 1; }

"$PY" analyze_days.py \
  --root "/fs/scratch/PAS0228" \
  --out "$HOME/sam3_all_v29" \
  --save-overlays --save-empty-overlays \
  --depth --tree-mask --prgb \
  --require-all-modalities --require-info-modality --skip-no-roi \
  --sample-per-session 100 --sample-mode sequential \
  --threshold 0.01 \
  --prompts apple branch trunk flower leaf fruitlet \
  --flower-multi-prompts flower blossom "apple blossom" \
  --flower-require-blossom-color --flower-min-blossom-color-frac 0.10 \
  --flower-min-area-px 4 --no-flower-reject-yellow \
  --flower-min-local-depth-std-mm 200 \
  --mask-min-depth-spread-mm 0 --mask-max-depth-row-corr 1.0 \
  --prgb-min-overlap 0.10 --prgb-dilate-px 40 --prgb-extend-vertical --prgb-skip-centroid-check \
  --tile-grid 2 2 --tile-overlap 0.2 --tile-nms-iou 0.15 \
  --tree-mask-min-overlap 0.10 \
  --flower-y-min 0 --flower-y-max 380 \
  --flower-max-area-px 12000 --flower-max-bbox-area-px 60000 \
  --flower-min-circularity 0.45 --flower-min-mask-density 0.30 \
  --flower-min-valid-depth-frac 0 \
  --flower-depth-min-mm 500 --flower-depth-max-mm 5000 \
  --flower-refine-min-area-px 15 --flower-refine-max-aspect 3.5 \
  --flower-min-soft-score 0.25 --flower-soft-w-sam 2.5 \
  --flower-context-ring-px 20 --flower-context-min-canopy-frac 0.20 \
  --flower-context-depth-tol-mm 2000 \
  --flower-petal-ndvi-mean 0.05 --flower-petal-ndvi-std 0.30 \
  --flower-canopy-ndvi-min 0.20 \
  --track --show-roi
