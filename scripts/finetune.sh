#!/bin/bash
#SBATCH --job-name="train_behavior"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=490G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/sc/train_behavior_%j.out
#SBATCH --error=outputs/sc/train_behavior_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=wsai@stanford.edu

set -euo pipefail
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "working directory=${SLURM_SUBMIT_DIR:-$PWD}"

# ===== Conda =====
eval "$(conda shell.bash hook)"
conda activate openvla-oft
python -V

# ===== Paths & run ids =====
DATASET_ROOT_PATH=/vision/u/yinhang/data/openvla
DATASET_NAME=behavior_turn_on_radio
CHECKPOINT_PATH=/vision/u/yinhang/forked_openvla/b1k-baselines/baselines/openvla-oft/checkpoints
mkdir -p "$CHECKPOINT_PATH"

RUN_ID_BASE=10_acts_chunk--LoRA_only--1img--proprio--film--bs1--lora4
RUN_TAG=$(date +%y%m%d_%H%M%S)
RUN_ID="${RUN_ID_BASE}--${RUN_TAG}"

# ===== Caches & WANDB =====
export WANDB_ENTITY=tonyliu12345
export WANDB_PROJECT=B1K
export HF_HOME=/vision/u/yinhang/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false

# ===== GPU / Torch / NCCL =====
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TF_CPP_MIN_LOG_LEVEL=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING || true
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=1
# Mitigate fragmentation on older CUDA (PyTorch allocator)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Avoid port collisions
export MASTER_PORT=$((12000 + RANDOM % 20000))

# ===== One-time source edits for Titan RTX + PEFT target modules =====
# Force fp16 (Titan RTX lacks bf16)
if grep -q "torch.bfloat16" vla-scripts/finetune.py; then
  sed -i 's/torch\.bfloat16/torch.float16/g' vla-scripts/finetune.py
fi

# Replace PEFT target_modules='all-linear' with explicit list (both quote styles)
if grep -q "target_modules *= *'all-linear'" vla-scripts/finetune.py; then
  sed -i "s/target_modules *= *'all-linear'/target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']/g" vla-scripts/finetune.py
fi
if grep -q 'target_modules *= *"all-linear"' vla-scripts/finetune.py; then
  sed -i 's/target_modules *= *"all-linear"/target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]/g' vla-scripts/finetune.py
fi

# DDP: remove extra autograd traversal cost
if grep -q "find_unused_parameters=True" vla-scripts/finetune.py; then
  sed -i 's/find_unused_parameters=True/find_unused_parameters=False/g' vla-scripts/finetune.py
fi

# HARD FREEZE any non-LoRA heads if defaults are True in your tree
python - <<'PY'
import re, sys
p="vla-scripts/finetune.py"
s=open(p,"r",encoding="utf-8").read()
s=re.sub(r"(train_action_head\s*=\s*)True", r"\1False", s)
s=re.sub(r"(train_proprio_projector\s*=\s*)True", r"\1False", s)
open(p,"w",encoding="utf-8").write(s)
print("[freeze-patch] train_action_head/train_proprio_projector set to False", file=sys.stderr)
PY

# ===== Image augmentation toggle (start OFF; flip later if desired) =====
AUG_FLAG="--image_aug False"

# ===== Memory knobs =====
NUM_IMAGES=1     # drop from 3 -> 1 for big activation savings
LORA_RANK=4      # was 8; a bit lighter on memory

# ===== Launch =====
torchrun --standalone --nproc-per-node 8 --master-port "$MASTER_PORT" vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "$DATASET_ROOT_PATH" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$CHECKPOINT_PATH" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input $NUM_IMAGES \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --lora_rank $LORA_RANK \
  --run_id_note "$RUN_ID" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_project "$WANDB_PROJECT" \
  $AUG_FLAG

echo "Done: $RUN_ID"
