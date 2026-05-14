#!/bin/bash
#SBATCH --job-name=autovla_sft_full_gpu4v100
#SBATCH --partition=gpu4v100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/10_sft_full_gpu4v100_%j.out
#SBATCH --error=logs/10_sft_full_gpu4v100_%j.err

set -e

# Print job info
echo "================================================"
echo "AutoVLA SFT Training - FULL Dataset (19k samples)"
echo "GPU Partition: gpu4v100 (4x V100)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================"

cd /work/amd456/autovla/autovla-nuscenes-reproduction

# Load environment
source /home/amd456/.bashrc
conda activate autovla

# Find best checkpoint from test run
TEST_RUN_DIR="runs/sft/2026-04-16_15-15-17"
if [ -d "$TEST_RUN_DIR" ]; then
    # Get the best checkpoint (lowest loss)
    BEST_CKPT=$(ls -1 "$TEST_RUN_DIR"/epoch=*-loss=*.ckpt | sort -V | tail -1)
    echo "Found best checkpoint from test run: $BEST_CKPT"
else
    BEST_CKPT=""
fi

# Check if we're resuming from a previous full training job
FULL_RUN_DIR="runs/sft/full_training_gpu4v100"
RESUME_CKPT=""

if [ -d "$FULL_RUN_DIR" ]; then
    # Look for the latest checkpoint from previous run
    LATEST_CKPT=$(find "$FULL_RUN_DIR" -name "epoch=*-loss=*.ckpt" -type f | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_CKPT="$LATEST_CKPT"
        echo "Resume checkpoint found: $RESUME_CKPT"
    fi
fi

# Use resume checkpoint if available, else use test run checkpoint
if [ -n "$RESUME_CKPT" ]; then
    CKPT_ARG="--ckpt $RESUME_CKPT"
    echo "Resuming from previous training run..."
elif [ -n "$BEST_CKPT" ]; then
    CKPT_ARG="--ckpt $BEST_CKPT"
    echo "Starting from test run checkpoint..."
else
    CKPT_ARG=""
    echo "Starting fresh training..."
fi

# Run training
echo "Starting SFT training with config: config/training/qwen2.5-vl-3B-mix-sft.yaml"
python tools/run_sft.py \
    --config config/training/qwen2.5-vl-3B-mix-sft.yaml \
    $CKPT_ARG

echo "================================================"
echo "SFT Training complete!"
echo "End time: $(date)"
echo "================================================"
echo ""
echo "Next step: Reinforcement Fine-Tuning (RFT)"
echo "  sbatch slurm_scripts/11_rft_training.slurm"
