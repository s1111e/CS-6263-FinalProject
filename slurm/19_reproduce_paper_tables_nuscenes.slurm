#!/bin/bash
#SBATCH --job-name=reproduce_paper_tables_nusc
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/19_reproduce_paper_tables_nuscenes.out
#SBATCH --error=logs/19_reproduce_paper_tables_nuscenes.err

set -e

cd /work/amd456/autovla/autovla-nuscenes-reproduction

echo "================================================"
echo "Step 19: Reproduce Paper Tables - nuScenes Scope"
echo "================================================"
echo "Start time: $(date)"
echo ""

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate autovla

echo "Python version: $(python --version)"
echo "Conda env:      $(conda info --envs | grep '*')"
echo ""

echo "Checking required artifacts..."

if [ ! -d "data/nuscenes_processed/train" ]; then
    echo "ERROR: Missing data/nuscenes_processed/train"
    exit 1
fi

if [ ! -d "data/nuscenes_processed/val" ]; then
    echo "ERROR: Missing data/nuscenes_processed/val"
    exit 1
fi

if [ ! -f "logs/12_evaluate.out" ]; then
    echo "WARNING: logs/12_evaluate.out not found; Table S2 local values may be empty"
fi

if [ ! -f "evaluation_results_data_scaling/table4_kdisk_metrics.json" ]; then
    echo "WARNING: table4_kdisk_metrics.json not found; Table 4 local values may be empty"
fi

mkdir -p autovla-nuscenes-reproduction/evaluation_results

echo ""
echo "Running paper table collector..."
python autovla-nuscenes-reproduction/reproduce_paper_tables_nuscenes.py

echo ""
echo "Generated files:"
ls -lh autovla-nuscenes-reproduction/evaluation_results/paper_tables_nuscenes.*

echo ""
echo "End time: $(date)"
echo "================================================"
echo "Paper table collection complete."
echo "================================================"
