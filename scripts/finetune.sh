#!/bin/bash
#SBATCH --job-name="train_behavior"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=490G
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

DATASET_ROOT_PATH=/vision/u/yinhang/data/openvla
DATASET_NAME=behavior_turn_on_radio
CHECKPOINT_PATH=/vision/u/yinhang/forked_openvla/b1k-baselines/baselines/openvla-oft/checkpoints
RUN_ID=10_acts_chunk--LoRA_only--3img--proprio--film--bs1--lora8

mkdir -p "$CHECKPOINT_PATH"

export WANDB_ENTITY=tonyliu12345
export WANDB_PROJECT=B1K
export HF_HOME=/vision/u/yinhang/cache/huggingface

# CUDA visibility for 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TF_CPP_MIN_LOG_LEVEL=2

# Force fp16 on Titan RTX (no BF16 support)
sed -i 's/torch\.bfloat16/torch.float16/g' vla-scripts/finetune.py

# Start without image augs (flip later if you want)
AUG_FLAG="--image_aug False"

# Avoid port collisions
export MASTER_PORT=$((12000 + RANDOM % 20000))

# Force fp16 on Titan RTX (no BF16 support)
sed -i 's/torch\.bfloat16/torch.float16/g' vla-scripts/finetune.py

# Replace PEFT target_modules='all-linear' with explicit list
sed -i "s/target_modules *= *'all-linear'/target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']/g" vla-scripts/finetune.py
sed -i 's/target_modules *= *"all-linear"/target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]/g' vla-scripts/finetune.py


torchrun --standalone --nproc-per-node 8 --master-port $MASTER_PORT vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "$DATASET_ROOT_PATH" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$CHECKPOINT_PATH" \
  --use_l1_regression False \
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
  --lora_rank 8 \
  --run_id_note "$RUN_ID" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_project "$WANDB_PROJECT" \
  $AUG_FLAG
