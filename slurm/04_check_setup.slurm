#!/bin/bash
#SBATCH --job-name=autovla_04_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/04_check.out
#SBATCH --error=logs/04_check.err
#SBATCH --partition=compute1

echo "========== STEP 4: Check Setup =========="
echo "Job: $SLURM_JOB_ID"
echo "Time: $(date)"

cd /work/amd456/autovla/autovla-nuscenes-reproduction

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate autovla

echo "✓ Checking Python..."
python --version

echo "✓ Checking PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo "✓ Checking models..."
ls -lh ~/.cache/huggingface/models/ 2>/dev/null || echo "Models path ready"

echo "========== ALL SETUP COMPLETE =========="
echo "Ready to start training!"