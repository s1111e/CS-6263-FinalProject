#!/bin/bash
#SBATCH --job-name=autovla_03_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/03_download.out
#SBATCH --error=logs/03_download.err
#SBATCH --partition=compute1

echo "========== STEP 3: Download Qwen Models =========="
echo "Job: $SLURM_JOB_ID"
echo "Time: $(date)"

cd /work/amd456/autovla/autovla-nuscenes-reproduction

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate autovla

echo "Downloading Qwen models..."
bash scripts/download_qwen.sh

echo "========== DONE =========="
echo "Next: sbatch slurm_scripts/04_check_setup.slurm"