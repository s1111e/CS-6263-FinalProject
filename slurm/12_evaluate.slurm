#!/bin/bash
#SBATCH --job-name=autovla_eval_nuscenes
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=/work/amd456/autovla/logs/12_evaluate.out
#SBATCH --error=/work/amd456/autovla/logs/12_evaluate.err

set -e

cd /work/amd456/autovla/autovla-nuscenes-reproduction

echo "================================================"
echo "Step 12: AutoVLA Evaluation on nuScenes"
echo "================================================"
echo "Start time: $(date)"
echo ""

# Load anaconda module
module load anaconda3
eval "$(conda shell.bash hook)"

# Activate main autovla environment
conda activate autovla

echo "Python version: $(python --version)"
echo "Conda env: $(conda info | grep 'active environment')"
echo ""

# Verify GPU
echo "GPU Check:"
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Select checkpoint/config (override with EVAL_CHECKPOINT and optionally EVAL_CONFIG)
CHECKPOINT=""
CONFIG_PATH=""

if [ ! -z "$EVAL_CHECKPOINT" ]; then
    CHECKPOINT="$EVAL_CHECKPOINT"
    if [ -z "$EVAL_CONFIG" ]; then
        CONFIG_PATH="config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"
    else
        CONFIG_PATH="$EVAL_CONFIG"
    fi
    echo "Using user-provided checkpoint override"
else
    echo "Looking for GRPO checkpoint..."
    CHECKPOINT=$(find /work/amd456/autovla/runs/grpo -name "rft-step*-reward*.ckpt" -type f \
        | sed -E 's#^(.*rft-step)([0-9]+)(-reward.*)$#\2 \1\2\3#' \
        | sort -n \
        | tail -1 \
        | awk '{print $2}')

    if [ -z "$CHECKPOINT" ]; then
        echo "No GRPO checkpoint found. Falling back to best SFT checkpoint (lowest loss)..."
        CHECKPOINT=$(find /work/amd456/autovla/runs/sft -name "epoch=*-loss=*.ckpt" -type f \
            | sed -E 's#^(.*-loss=)([0-9]+\.[0-9]+)(\.ckpt)$#\2 \1\2\3#' \
            | sort -g \
            | head -1 \
            | awk '{print $2}')
        CONFIG_PATH="config/training/qwen2.5-vl-3B-mix-sft.yaml"
    else
        CONFIG_PATH="config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"
    fi
fi

if [ -z "$CHECKPOINT" ]; then
    echo "❌ ERROR: No checkpoint found!"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint does not exist: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ ERROR: Config does not exist: $CONFIG_PATH"
    exit 1
fi

echo "✓ Using checkpoint: $CHECKPOINT"
echo "✓ Using config: $CONFIG_PATH"
echo ""

# Resolve segmentation data directory for collision evaluation
echo "Checking segmentation data for collision evaluation..."
SEG_DATA_PATH="/work/amd456/autovla/data/nusc_eval_seg"
if [ -d "/work/amd456/autovla/data/nusc_eval_seg/nusc_eval_seg_6s" ]; then
    SEG_DATA_PATH="/work/amd456/autovla/data/nusc_eval_seg/nusc_eval_seg_6s"
fi

SEG_COUNT=$(find "$SEG_DATA_PATH" -maxdepth 1 -name "*.pt" -type f 2>/dev/null | wc -l)
if [ "$SEG_COUNT" -eq 0 ]; then
    echo "❌ ERROR: Segmentation data not found at: $SEG_DATA_PATH"
    echo "Download from: https://drive.google.com/file/d/1YMiKqUi_rhdxdrNVXKFuwEg3xAGm-dvU/view?usp=sharing"
    exit 1
fi
echo "✓ Segmentation data found: $SEG_DATA_PATH ($SEG_COUNT files)"
echo ""

# Evaluation Configuration
echo "================================================"
echo "Evaluation Configuration"
echo "================================================"
echo "Dataset: nuScenes validation (5.6k samples)"
echo "Metrics:"
echo "  - Average Displacement Error (ADE)"
echo "  - Collision Rate"
echo "  - Path Deviation Mean Score (PDMS)"
echo "  - Realistic Feasibility Score (RFS)"
echo ""

# Run Evaluation
# Use NUM_EVAL_SAMPLES env var if set, otherwise evaluate all samples
NUM_SAMPLES_ARG=""
if [ ! -z "$NUM_EVAL_SAMPLES" ]; then
    NUM_SAMPLES_ARG="--num_samples $NUM_EVAL_SAMPLES"
    echo "Limiting evaluation to $NUM_EVAL_SAMPLES samples"
fi
echo "Starting nuScenes evaluation..."
python tools/eval/nusc_eval.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT" \
    --seg_data_path "$SEG_DATA_PATH" \
    $NUM_SAMPLES_ARG

echo ""
echo "================================================"
echo "Evaluation complete!"
echo "End time: $(date)"
echo "================================================"
echo ""
echo "✅ AutoVLA evaluation job completed!"
echo ""
echo "Training pipeline finished! 🎉"
