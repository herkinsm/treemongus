#!/bin/bash
#SBATCH --job-name=sam3_flowers_all
#SBATCH --account=PAS0228
#SBATCH --time=8:00:00
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

python analyze_days.py \
  --root "/fs/scratch/PAS0228" \
  --out "$HOME/sam3_all_v29" \
  --save-overlays --save-masks \
  --depth --tree-mask --prgb \
  --require-all-modalities --skip-no-roi \
  --sample-per-session 0 \
  --threshold 0.01 \
  --prompts apple branch trunk flower leaf fruitlet \
  --flower-multi-prompts flower blossom "apple blossom" \
  --flower-require-blossom-color --flower-min-blossom-color-frac 0.10 \
  --flower-min-area-px 4 --no-flower-reject-yellow \
  --flower-min-local-depth-std-mm 200 \
  --mask-min-depth-spread-mm 0 --mask-max-depth-row-corr 1.0 \
  --prgb-min-overlap 0.30 --prgb-dilate-px 0 --prgb-extend-vertical \
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
  --debug-rejection-log --debug-overlay \
  --track --show-roi
