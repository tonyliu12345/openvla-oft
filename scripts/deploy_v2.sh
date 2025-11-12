#!/bin/bash
set -euo pipefail

# ==== Conda ====
eval "$(conda shell.bash hook)"
conda activate openvla-oft

# ==== Config (override via env if you want) ====
DATASET_NAME="${DATASET_NAME:-behavior_turn_on_radio}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/openvla-7b--100000_chkpt}"

# If your 100k run actually used 3 images or FiLM, override like:
#   NUM_IMAGES=3 USE_FILM=True ./deploy.sh
NUM_IMAGES="${NUM_IMAGES:-1}"         # 1 by default
USE_FILM="${USE_FILM:-False}"         # False by default
USE_L1="${USE_L1:-True}"              # True for your runs
USE_PROPRIO="${USE_PROPRIO:-True}"    # True for your runs
CENTER_CROP="${CENTER_CROP:-True}"    # keep True
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "[deploy] CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "[deploy] DATASET_NAME=$DATASET_NAME"
echo "[deploy] FLAGS: L1=$USE_L1, FiLM=$USE_FILM, num_images=$NUM_IMAGES, proprio=$USE_PROPRIO, center_crop=$CENTER_CROP"
echo "[deploy] Serving on $HOST:$PORT"

# ==== Launch server ====
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
