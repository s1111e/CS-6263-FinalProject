#!/bin/bash
#SBATCH --job-name=autovla_02_install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/02_install.out
#SBATCH --error=logs/02_install.err
#SBATCH --partition=compute1

echo "========== STEP 2: Install Packages =========="
echo "Job: $SLURM_JOB_ID"
echo "Time: $(date)"

cd /work/amd456/autovla/autovla-nuscenes-reproduction

module load anaconda3
source activate autovla

echo "Installing AutoVLA..."
pip install -e . --no-warn-conflicts

echo "Running install script..."
bash install.sh

echo "Installing navsim..."
cd navsim
pip install -e . --no-warn-conflicts
cd ..

echo "========== DONE =========="
echo "Next: sbatch slurm_scripts/03_download_models.slurm"
