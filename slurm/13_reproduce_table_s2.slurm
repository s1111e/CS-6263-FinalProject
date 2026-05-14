#!/bin/bash
#SBATCH --job-name=reproduce_table_s2
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/13_reproduce_table_s2.out
#SBATCH --error=logs/13_reproduce_table_s2.err

set -e

echo "================================================"
echo "Step 13: Reproduce Table S2 - NuScenes Benchmark"
echo "================================================"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
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

# Verify segmentation data exists
SEG_PATH="data/nusc_eval_seg/nusc_eval_seg_6s"
if [ ! -d "$SEG_PATH" ]; then
    echo "❌ Error: Segmentation data not found: $SEG_PATH"
    exit 1
fi
echo "✓ Segmentation data found: $SEG_PATH"
echo ""

# Create output directory
mkdir -p autovla-nuscenes-reproduction/evaluation_results

echo "================================================"
echo "Running Table S2 Evaluation"
echo "================================================"
echo "Config: config/training/qwen2.5-vl-3B-mix-sft.yaml"
echo "Checkpoint: $CHECKPOINT"
echo "Segmentation: $SEG_PATH"
echo ""

# Run evaluation on full validation set
python autovla-nuscenes-reproduction/reproduce_table_s2_nuscenes.py \
    --config config/training/qwen2.5-vl-3B-mix-sft.yaml \
    --checkpoint "$CHECKPOINT" \
    --seg_data_path "$SEG_PATH" \
    --output autovla-nuscenes-reproduction/evaluation_results/table_s2.json \
    --device cuda:0

echo ""
echo "================================================"
echo "✅ Table S2 reproduction job completed!"
echo "================================================"
echo "Results saved to: autovla-nuscenes-reproduction/evaluation_results/table_s2.json"
echo "End time: $(date)"
