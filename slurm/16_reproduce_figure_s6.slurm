#!/bin/bash
#SBATCH --job-name=reproduce_figure_s6
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/16_reproduce_figure_s6.out
#SBATCH --error=logs/16_reproduce_figure_s6.err

set -e

cd /work/amd456/autovla/autovla-nuscenes-reproduction

# Activate conda environment
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

echo "======================================"
echo "Figure S6: Qualitative Results"
echo "======================================"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "Conda env: $(conda info --envs | grep '\*')"
echo ""

# Check checkpoint
if [ -f "runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt" ]; then
    echo "✓ Checkpoint found"
else
    echo "✗ Checkpoint NOT FOUND"
    exit 1
fi

# Check data
if [ -d "data/nuscenes_processed/val" ]; then
    echo "✓ NuScenes processed scene JSONs found"
else
    echo "✗ NuScenes processed scene JSONs NOT FOUND"
    exit 1
fi

echo ""

# Run qualitative figure generation
python autovla-nuscenes-reproduction/reproduce_figure_s6_qualitative.py \
    --config config/training/qwen2.5-vl-3B-mix-sft.yaml \
    --checkpoint runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt \
    --data-path data/nuscenes_processed/ \
    --split val \
    --num-samples 2 \
    --output-dir autovla-nuscenes-reproduction/evaluation_results/ \
    --device cuda:0

echo ""
echo "End: $(date)"
echo "======================================"
echo "Figure S6 generation complete!"
echo "Check: autovla-nuscenes-reproduction/evaluation_results/figure_s6_qualitative_results.png"
echo "======================================"

# Show output
if [ -f "autovla-nuscenes-reproduction/evaluation_results/figure_s6_qualitative_results.png" ]; then
    echo "✓ Figure created:"
    ls -lh autovla-nuscenes-reproduction/evaluation_results/figure_s6_*.png
else
    echo "✗ Figure NOT created - check errors above"
    exit 1
fi
