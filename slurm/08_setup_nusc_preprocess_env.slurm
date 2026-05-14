#!/bin/bash
#SBATCH --job-name=setup_nusc_env
#SBATCH --partition=compute1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/work/amd456/autovla/logs/08_setup_nusc_env.out
#SBATCH --error=/work/amd456/autovla/logs/08_setup_nusc_env.err

set -e

cd /work/amd456/autovla/autovla-nuscenes-reproduction

echo "================================================"
echo "Step 8: Setup nuScenes Preprocessing Environment"
echo "================================================"
echo "Start time: $(date)"
echo ""

# Load anaconda module
module load anaconda3
eval "$(conda shell.bash hook)"

# Create separate environment if it doesn't exist
echo "Creating/updating nuScenes preprocessing environment..."
if ! conda info --envs | grep -q autovla_nusc_preprocess; then
    echo "Creating new conda environment from environment_nusc_preprocess.yml..."
    conda env create -f environment_nusc_preprocess.yml -q
else
    echo "Environment already exists, updating..."
    conda env update -f environment_nusc_preprocess.yml --prune -q
fi

# Activate environment
conda activate autovla_nusc_preprocess

echo "✓ Python version: $(python --version)"
echo "✓ Conda env: $(conda info | grep 'active environment')"
echo ""

# Install Arc PyTorch
echo "Installing PyTorch from Arc HPC..."
pip install torch torchvision -q
echo "✓ PyTorch installed"

# Fix NumPy compatibility with nuscenes-devkit
echo "Fixing NumPy compatibility..."
pip install "numpy<2" -q
echo "✓ NumPy <2 installed"

# Verify all dependencies
echo ""
echo "Verifying dependencies:"
python -c "import numpy; print('✓ numpy:', numpy.__version__)"
python -c "import torch; print('✓ torch:', torch.__version__)"
python -c "import tqdm; print('✓ tqdm:', tqdm.__version__)"
python -c "import nuscenes; print('✓ nuscenes-devkit installed')"
python -c "import pyquaternion; print('✓ pyquaternion installed')"

echo ""
echo "================================================"
echo "Environment setup complete!"
echo "End time: $(date)"
echo "================================================"
echo ""
echo "Ready for preprocessing. Run: sbatch 09_preprocess_nuscenes.slurm"
