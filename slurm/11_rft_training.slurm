#!/bin/bash
#SBATCH --job-name=autovla_rft_training
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=71:00:00
#SBATCH --output=/work/amd456/autovla/logs/11_rft_training_2500.out
#SBATCH --error=/work/amd456/autovla/logs/11_rft_training_2500.err

set -e

cd /work/amd456/autovla/autovla-nuscenes-reproduction

echo "================================================"
echo "Step 11: AutoVLA RFT (Reinforcement Fine-Tuning)"
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

# Check SFT checkpoint
echo "Checking SFT checkpoint..."
SFT_CKPT=$(find /work/amd456/autovla/runs/sft -name "epoch=4*.ckpt" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$SFT_CKPT" ]; then
    echo "✓ SFT checkpoint found: $SFT_CKPT"
else
    echo "❌ No SFT checkpoint found in runs/sft"
    exit 1
fi
echo ""

# Update RFT config to use the latest SFT checkpoint
RFT_CONFIG="/work/amd456/autovla/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"
echo "Updating RFT config checkpoint path..."
sed -i "s|^  sft_model_path:.*|  sft_model_path: $SFT_CKPT|" "$RFT_CONFIG"
grep "^  sft_model_path:" "$RFT_CONFIG"
echo ""

# RFT Training Configuration
echo "================================================"
echo "RFT Training Configuration"
echo "================================================"
echo "Algorithm: GRPO (Group Relative Policy Optimization)"
echo "Steps: 6000"
echo "Learning rate: 3e-5"
echo "LoRA rank: 8"
echo "Dataset: nuScenes (2500 train with DriveLM CoT)"
echo ""

# Run RFT Training
echo "Starting RFT training..."
bash scripts/run_rft.sh

echo ""
echo "================================================"
echo "RFT Training complete!"
echo "End time: $(date)"
echo "================================================"
echo ""
echo "✅ RFT training job completed!"
echo ""
echo "Next step: Evaluation on nuScenes"
echo "  sbatch slurm_scripts/12_evaluate.slurm"
