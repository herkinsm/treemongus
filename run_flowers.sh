#!/bin/bash
#SBATCH --job-name=sam3_flowers
#SBATCH --account=PAS0228
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.log

cd ~/sam3-apple-analysis && git pull

python analyze_days.py --root "/fs/scratch/PAS0228" --out "$HOME/sam3_1to100_v29" --save-overlays --save-masks --depth --tree-mask --prgb --require-all-modalities --skip-no-roi --frame-range 1 100 --threshold 0.01 --prompts apple branch trunk flower leaf fruitlet --flower-multi-prompts flower blossom "apple blossom" --flower-require-blossom-color --flower-min-blossom-color-frac 0.10 --flower-min-area-px 4 --no-flower-reject-yellow --flower-min-local-depth-std-mm 200 --mask-min-depth-spread-mm 0 --mask-max-depth-row-corr 1.0 --prgb-min-overlap 0.50 --prgb-dilate-px 0 --prgb-extend-vertical --tile-grid 2 2 --tile-overlap 0.2 --tile-nms-iou 0.15 --tree-mask-min-overlap 0.10 --flower-y-min 0 --flower-y-max 380 --flower-max-area-px 12000 --flower-max-bbox-area-px 60000 --flower-min-circularity 0.55 --flower-min-mask-density 0.40 --flower-min-valid-depth-frac 0 --flower-depth-min-mm 500 --flower-depth-max-mm 5000 --flower-min-soft-score 0.35 --flower-context-ring-px 20 --flower-context-min-canopy-frac 0.30 --flower-context-depth-tol-mm 1500 --flower-petal-ndvi-mean 0.10 --flower-petal-ndvi-std 0.20 --flower-canopy-ndvi-min 0.30 --track --show-roi
