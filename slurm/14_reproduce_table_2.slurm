#!/bin/bash
#SBATCH --job-name=reproduce_table_2
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/14_reproduce_table_2.out
#SBATCH --error=logs/14_reproduce_table_2.err

set -e

cd /work/amd456/autovla/autovla-nuscenes-reproduction

echo "================================================"
echo "Step 14: Reproduce Table 2 - Runtime Analysis"
echo "================================================"
echo "Start time: $(date)"
echo ""

# Load and activate conda environment
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate autovla

echo "Python version: $(python --version)"
echo "Conda env:      $(conda info --envs | grep '\*')"
echo ""

# Check GPU
echo "GPU Check:"
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Verify checkpoint exists
CHECKPOINT="runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi
echo "✓ Checkpoint found: $CHECKPOINT"
echo ""

# Create output directory
mkdir -p autovla-nuscenes-reproduction/evaluation_results

echo "================================================"
echo "Running Table 2 Runtime Analysis"
echo "================================================"
echo "Config: config/training/qwen2.5-vl-3B-mix-sft.yaml"
echo "Checkpoint: $CHECKPOINT"
echo "Num Samples: 500"
echo ""

# Run runtime profiling
python autovla-nuscenes-reproduction/reproduce_table_2_runtime.py \
    --config config/training/qwen2.5-vl-3B-mix-sft.yaml \
    --checkpoint "$CHECKPOINT" \
    --output autovla-nuscenes-reproduction/evaluation_results/table_2_runtime.json \
    --num_samples 500 \
    --device cuda:0 \
    --max-new-tokens 128 \
    --min-new-tokens 1 \
    --greedy

echo ""
echo "================================================"
echo "✅ Table 2 runtime analysis job completed!"
echo "================================================"
echo "Results saved to: autovla-nuscenes-reproduction/evaluation_results/table_2_runtime.json"
echo "End time: $(date)"
