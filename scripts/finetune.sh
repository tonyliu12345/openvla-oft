#!/bin/bash
#SBATCH --job-name="train_behavior"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=350G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/sc/train_behavior_%j.out
#SBATCH --error=outputs/sc/train_behavior_%j.err
# notifications for job done & fail
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=wsai@stanford.edu

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate openvla-oft

# ---- Paths & run info ----
DATASET_ROOT_PATH=/vision/u/yinhang/data/openvla
DATASET_NAME=behavior_turn_on_radio
CHECKPOINT_PATH=/vision/u/yinhang/forked_openvla/b1k-baselines/baselines/openvla-oft/checkpoints
RUN_ID=10_acts_chunk--continuous_acts--L1_regression--3img--proprio_state--film--bs1--lora16

mkdir -p "$CHECKPOINT_PATH"

# ---- Logging ----

export WANDB_ENTITY=tonyliu12345
export WANDB_PROJECT=B1K

# ---- HF cache ----
export HF_HOME=/vision/u/yinhang/cache/huggingface

# ---- Help CUDA memory fragmentation ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TF_CPP_MIN_LOG_LEVEL=2

# Force fp16 in finetune.py (your prior workaround)
sed -i 's/torch\.bfloat16/torch.float16/g' vla-scripts/finetune.py

# ---- Main launch ----
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "$DATASET_ROOT_PATH" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$CHECKPOINT_PATH" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --lora_rank 16 \
  --run_id_note "$RUN_ID" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_project "$WANDB_PROJECT" \
  --image_aug True
  # Optional eval:
  # --use_val_set True \
  # --val_freq 10000 \
