#!/bin/bash
#SBATCH --job-name=sam3_yolo_test
#SBATCH --account=PAS0228
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.log

# 30-min smoke test of run_yolo_labels_balanced.sh's full feature set
# scoped to ONE day with sparse stride sampling. Catches feature-
# interaction bugs before committing to a 16h batch.
#
# Differences from run_yolo_labels_balanced.sh:
#   --root             "/fs/scratch/PAS0228/2023 day 4"  (was: full root)
#   --out              "$HOME/sam3_yolo_test"            (was: sam3_yolo_labels)
#   --sample-stride    50                                (was: 20)
#   walltime           1h                                (was: 16h)
#
# Submit:    sbatch run_yolo_test.sh
# Or run interactive (no sbatch needed at 30 min):
#            bash run_yolo_test.sh    # only if you have an active sam3 conda env

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
  --root "/fs/scratch/PAS0228/2023 day 4" \
  --out "$HOME/sam3_yolo_test" \
  --save-overlays --save-empty-overlays --save-masks \
  --save-canopy-masks --save-canopy-overlay \
  --depth --tree-mask \
  --require-all-modalities --require-info-modality \
  --sample-per-session 0 --sample-mode stride --sample-stride 50 \
  --threshold 0.005 \
  --prompts flower \
  --flower-multi-prompts flower blossom "apple blossom" \
  --flower-require-blossom-color --flower-min-blossom-color-frac 0.10 \
  --flower-min-area-px 4 --no-flower-reject-yellow \
  --flower-max-mask-green-frac 0.55 --flower-green-blossom-override-frac 0.20 \
  --flower-min-peaks-per-1000px 3.0 \
  --flower-peak-min-distance-px 5 --flower-peak-threshold-abs 80 \
  --flower-peak-min-area-px 60 \
  --flower-peak-min-distance-px2 9 --flower-peak-prominence-min 5 \
  --flower-min-anther-holes-per-1000px 1.5 \
  --flower-anther-petal-v-min 100 \
  --flower-anther-hole-min-area-px 2 --flower-anther-hole-max-area-px 60 \
  --flower-anther-min-area-px 100 \
  --depth-min-mm 600 --depth-max-mm 3000 --depth-near-frac 0.40 \
  --mask-min-depth-spread-mm 0 --mask-max-depth-row-corr 1.0 \
  --tile-grid 2 2 --tile-overlap 0.2 --tile-nms-iou 0.15 \
  --use-build-tree-mask --tree-mask-min-overlap 0.10 --tree-mask-dilate-px 8 \
  --canopy-sam-prompt "apple tree" --canopy-sam-min-score 0.15 \
  --canopy-sam-min-pixels 500 --canopy-sam-min-lower-frac 0.20 \
  --canopy-sam-min-valid-depth-frac 0.30 \
  --canopy-sam-supplement-with-edge-aug \
  --canopy-include-edge-trees --canopy-edge-tree-min-area-px 400 \
  --canopy-edge-tree-min-height-px 80 --canopy-edge-tree-max-top-row 350 \
  --canopy-edge-tree-min-aspect-ratio 0.8 \
  --canopy-edge-tree-max-depth-std-mm 3000 \
  --canopy-edge-tree-min-green-frac 0.0 \
  --canopy-edge-tree-max-depth-mm 3000 \
  --flower-edge-margin-sides-px 0 \
  --flower-require-tree-in-frame --flower-foreground-canopy-max-depth-mm 2500 \
  --flower-max-behind-foreground-mm 1000 \
  --track-canopy --canopy-track-iou 0.3 --canopy-track-max-age 5 --canopy-track-min-cc-area 500 \
  --canopy-track-method trunk --canopy-trunk-min-score 0.20 --canopy-trunk-prompt "apple tree trunk" \
  --canopy-trunk-reject-green-stakes --canopy-trunk-max-green-pct 0.35 \
  --canopy-trunk-green-threshold 15 --canopy-trunk-min-brown-pct 0.20 \
  --canopy-trunk-max-depth-mm 3000 --canopy-trunk-depth-min-pixels 5 \
  --flower-y-min 0 --flower-y-max 380 \
  --flower-max-area-px 12000 --flower-max-bbox-area-px 60000 \
  --flower-white-s-max 50 --flower-white-v-min 120 --flower-pink-v-min 90 \
  --flower-b-minus-r-max 5 --flower-pink-b-minus-r-max 0 \
  --flower-g-minus-r-max 12 --flower-top-frame-penalty-row 50 \
  --flower-phenology off --flower-bloom-peak-doy 125 \
  --flower-fill-anther-holes \
  --split-clusters --split-min-blossom-area-px 30 \
  --split-min-marker-distance-px 5 --split-seed-dilate-px 5 \
  --split-area-cap --flower-area-per-flower-px 200 \
  --flower-max-ground-row 400 --flower-min-confirmed-pct-ground 10.0 \
  --flower-use-texture --flower-texture-threshold 2.5 --flower-edge-threshold 6.0 \
  --flower-ir-positive-min 80 --flower-ir-petal-min 100 --flower-ir-sky-ceil 60 \
  --flower-confirmed-real --flower-near-tree-radius-px 25 \
  --flower-depth-coverage-threshold 0.40 \
  --flower-max-depth-cap-mm 3500 --flower-max-depth-cap-min-pixels 20 \
  --flower-exclude-sky-smooth --flower-exclude-sky-warm \
  --flower-exclude-sky-upper --flower-exclude-sky-overcast \
  --flower-exclude-sky-grey \
  --flower-two-tier-mask \
  --flower-exclude-bark --flower-exclude-dark --flower-exclude-ground-grass \
  --flower-compute-density-score --flower-confidence-scale \
  --flower-min-circularity 0.35 --flower-min-mask-density 0.20 \
  --flower-refine-min-area-px 10 --flower-refine-max-aspect 4.5 \
  --flower-min-soft-score 0.15 --flower-soft-w-sam 1.0 \
  --flower-context-ring-px 20 --flower-context-min-canopy-frac 0.0 \
  --flower-context-depth-tol-mm 3000 \
  --flower-petal-ndvi-mean 0.05 --flower-petal-ndvi-std 0.50 \
  --flower-canopy-ndvi-min 0.0 --flower-canopy-ndvi-softness 0.0 \
  --flower-min-local-depth-std-mm 80 \
  --debug-rejection-log \
  --track --track-min-frames 3
