#!/bin/bash
#SBATCH --job-name=sam3_yolo_labels
#SBATCH --account=PAS0228
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.log

# Generate flower bounding-box labels for YOLO training.
#
# Differs from run_flowers_all.sh:
#   1) NO --prgb anywhere -- SAM looks at the whole frame and labels
#      every flower it can find, regardless of which tree they're on.
#      A YOLO detector at inference time has no ROI; training labels
#      should match that.
#   2) Looser flower gates than the previous strict-precision config,
#      because the strict version was UNDER-predicting. The
#      track-min-frames 3 stability filter still removes most false
#      positives at the dataset assembly step (make_yolo_dataset.py).
#
# After this job completes, run:
#   python make_yolo_dataset.py --in $HOME/sam3_yolo_labels \
#       --out $HOME/yolo_dataset --val-frac 0.2 --min-track-frames 3
# to assemble images + labels into a YOLO-compatible folder layout.

cd ~/sam3-apple-analysis && git pull

# CPU threading
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[batch] threading: OMP/MKL/OPENBLAS=$OMP_NUM_THREADS"

# Conda env
if [ -f "$HOME/.conda/envs/sam3/bin/python" ]; then
    PY="$HOME/.conda/envs/sam3/bin/python"
elif [ -f "/users/PAS0228/mherkins/.conda/envs/sam3/bin/python" ]; then
    PY="/users/PAS0228/mherkins/.conda/envs/sam3/bin/python"
else
    PY="python"
fi
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
"$PY" -c "import sam3; print('[batch] sam3 OK')" \
    || { echo "[batch] sam3 import failed -- aborting"; exit 1; }

"$PY" analyze_days.py \
  --root "/fs/scratch/PAS0228" \
  --out "$HOME/sam3_yolo_labels" \
  --save-overlays --save-empty-overlays --save-masks \
  --depth --tree-mask \
  --require-all-modalities --require-info-modality \
  --sample-per-session 100 --sample-mode sequential \
  --threshold 0.005 \
  --prompts flower \
  --flower-multi-prompts flower blossom "apple blossom" \
  --flower-require-blossom-color --flower-min-blossom-color-frac 0.10 \
  --flower-min-area-px 4 --no-flower-reject-yellow \
  --flower-min-local-depth-std-mm 200 \
  --mask-min-depth-spread-mm 0 --mask-max-depth-row-corr 1.0 \
  --tile-grid 2 2 --tile-overlap 0.2 --tile-nms-iou 0.15 \
  --tree-mask-min-overlap 0.10 \
  --flower-y-min 0 --flower-y-max 380 \
  --flower-max-area-px 12000 --flower-max-bbox-area-px 60000 \
  --flower-min-circularity 0.40 --flower-min-mask-density 0.25 \
  --flower-refine-min-area-px 12 --flower-refine-max-aspect 4.0 \
  --flower-min-soft-score 0.20 --flower-soft-w-sam 1.5 \
  --flower-context-ring-px 20 --flower-context-min-canopy-frac 0.15 \
  --flower-context-depth-tol-mm 2500 \
  --flower-petal-ndvi-mean 0.05 --flower-petal-ndvi-std 0.35 \
  --flower-canopy-ndvi-min 0.15 \
  --debug-rejection-log \
  --track --track-min-frames 3
