#!/bin/bash
#SBATCH --job-name=sam2_trees
#SBATCH --account=PAS0228
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.log

cd ~/sam3-apple-analysis && git pull

python sam2_orchard_segmenter.py \
    --root "/fs/scratch/PAS0228" \
    --out "$HOME/sam3_1to100_v29" \
    --flower-csv "$HOME/sam3_1to100_v29/results.csv" \
    --flower-prompt flower \
    --flower-count-col est_flowers \
    --frame-range 1 100 \
    --device cuda \
    -v
