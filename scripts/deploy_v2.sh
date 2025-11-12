#!/bin/bash
set -euo pipefail

# ==== Conda ====
eval "$(conda shell.bash hook)"
conda activate openvla-oft

# ==== Config ====
DATASET_NAME="${DATASET_NAME:-behavior_turn_on_radio}"
# 👇 point to the merged folder you created
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/openvla-7b--100000_merged}"

NUM_IMAGES="${NUM_IMAGES:-1}"
USE_FILM="${USE_FILM:-False}"
USE_L1="${USE_L1:-True}"
USE_PROPRIO="${USE_PROPRIO:-True}"
CENTER_CROP="${CENTER_CROP:-True}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "[deploy] CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "[deploy] DATASET_NAME=$DATASET_NAME"
echo "[deploy] FLAGS: L1=$USE_L1, FiLM=$USE_FILM, num_images=$NUM_IMAGES, proprio=$USE_PROPRIO, center_crop=$CENTER_CROP"
echo "[deploy] Serving on $HOST:$PORT"

python vla-scripts/deploy.py \
  --pretrained_checkpoint "$CHECKPOINT_DIR" \
  --use_l1_regression "$USE_L1" \
  --use_film "$USE_FILM" \
  --num_images_in_input "$NUM_IMAGES" \
  --use_proprio "$USE_PROPRIO" \
  --center_crop "$CENTER_CROP" \
  --unnorm_key "$DATASET_NAME" \
  --host "$HOST" \
  --port "$PORT"
