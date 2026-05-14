#!/bin/bash
#SBATCH --job-name=autovla_repro_quicktest
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/15_reproduce_quicktest.out
#SBATCH --error=logs/15_reproduce_quicktest.err

set -e

echo "================================================"
echo "Step 15: AutoVLA Reproduction - Quick Test"
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
echo "Running Quick Test (50 samples)"
echo "================================================"
echo ""

# Test Table S2 with 50 samples
echo "📊 Testing Table S2 Reproduction (50 samples)..."
python autovla-nuscenes-reproduction/reproduce_table_s2_nuscenes.py \
    --config config/training/qwen2.5-vl-3B-mix-sft.yaml \
    --checkpoint "$CHECKPOINT" \
    --seg_data_path "$SEG_PATH" \
    --num_samples 50 \
    --output autovla-nuscenes-reproduction/evaluation_results/table_s2_quicktest.json \
    --device cuda:0

echo ""
echo "✅ Results saved to: autovla-nuscenes-reproduction/evaluation_results/table_s2_quicktest.json"
echo ""

# Test Table 2 with 20 samples
echo "⏱️  Testing Table 2 Reproduction (20 samples)..."
python autovla-nuscenes-reproduction/reproduce_table_2_runtime.py \
    --config config/training/qwen2.5-vl-3B-mix-sft.yaml \
    --checkpoint "$CHECKPOINT" \
    --output autovla-nuscenes-reproduction/evaluation_results/table_2_quicktest.json \
    --num_samples 20 \
    --device cuda:0

echo ""
echo "✅ Results saved to: autovla-nuscenes-reproduction/evaluation_results/table_2_quicktest.json"
echo ""

echo "================================================"
echo "✅ Quick Test Complete!"
echo "================================================"
echo "Next steps:"
echo "1. Check results in evaluation_results/"
echo "2. If OK, submit full jobs:"
echo "   sbatch slurm_scripts/13_reproduce_table_s2.slurm"
echo "   sbatch slurm_scripts/14_reproduce_table_2.slurm"
echo "End time: $(date)"
