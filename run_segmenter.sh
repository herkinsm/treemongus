#!/bin/bash
#SBATCH --job-name=sam2_trees
#SBATCH --account=PAS0228
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.log

# ── Required: set these before submitting ────────────────────────────────────
# SESSION_PATH: full path to the session folder (contains RGB/, depth/, PRGB/, Info/)
# SESSION_NAME: must exactly match the 'session' column in results.csv
#               (run: awk -F',' 'NR>1 {print $3}' ~/sam3_50to80_v29/results.csv | sort -u)
SESSION_PATH="/fs/scratch/PAS0228/2023 day 1/2023 day 1/Row1/Session1"
SESSION_NAME="Session1"

# ── Optional: adjust if needed ───────────────────────────────────────────────
FLOWER_CSV="$HOME/sam3_50to80_v29/results.csv"
OUT_DIR="$HOME/sam3_50to80_v29/trees/${SESSION_NAME}"
TREE_SPACING=3.0   # metres between trees in this orchard row
FRAME_START=50
FRAME_STOP=80
# ─────────────────────────────────────────────────────────────────────────────

cd ~/sam3-apple-analysis && git pull

python sam2_orchard_segmenter.py \
    "$SESSION_PATH" \
    --loader all2023 \
    --frame-range "$FRAME_START" "$FRAME_STOP" \
    --output-dir "$OUT_DIR" \
    --flower-csv "$FLOWER_CSV" \
    --flower-session "$SESSION_NAME" \
    --flower-prompt flower \
    --flower-count-col est_flowers \
    --tree-spacing-m "$TREE_SPACING" \
    --device cuda \
    -v
