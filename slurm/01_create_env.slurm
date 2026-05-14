#!/bin/bash
#SBATCH --job-name=autovla_01_create_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/01_create_env.out
#SBATCH --error=logs/01_create_env.err
#SBATCH --partition=compute1

echo "========== STEP 1: Create Conda Environment =========="
echo "Job: $SLURM_JOB_ID"
echo "Time: $(date)"

cd /work/amd456/autovla/autovla-nuscenes-reproduction
mkdir -p logs

# Load modules
module load anaconda3

echo "Creating environment..."
conda env create -f environment.yml -y

echo "========== DONE =========="
echo "Next: sbatch slurm_scripts/02_install_packages.slurm"
