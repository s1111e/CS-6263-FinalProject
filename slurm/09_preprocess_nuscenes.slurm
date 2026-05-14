#!/bin/bash
#SBATCH --job-name=preprocess_nusc
#SBATCH --partition=compute1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --output=/work/amd456/autovla/logs/09_preprocess_nuscenes.out
#SBATCH --error=/work/amd456/autovla/logs/09_preprocess_nuscenes.err

set -e

cd /work/amd456/autovla/autovla-nuscenes-reproduction

echo "================================================"
echo "Step 9: nuScenes Preprocessing with DriveLM"
echo "================================================"
echo "Start time: $(date)"
echo ""

# Create output directory
mkdir -p /work/amd456/autovla/data/nuscenes_processed

# Load anaconda module
module load anaconda3
eval "$(conda shell.bash hook)"

# Activate prepared environment
echo "Activating prepared nuScenes environment..."
conda activate autovla_nusc_preprocess

echo "Python version: $(python --version)"
echo "Conda env: $(conda info | grep 'active environment')"
echo ""

# Check DriveLM file
DRIVELM_PATH="/work/amd456/autovla/dataset/drivelm/v1_1_train_nus.json"
if [ -f "$DRIVELM_PATH" ]; then
    echo "✅ DriveLM file found: $DRIVELM_PATH"
    DRIVELM_ARG="--drivelm_path $DRIVELM_PATH"
else
    echo "⚠️  DriveLM file not found at $DRIVELM_PATH"
    echo "⚠️  Processing training data WITHOUT DriveLM annotations"
    DRIVELM_ARG=""
fi

# Process training data (with DriveLM if available)
echo ""
echo "Processing nuScenes TRAINING data..."
python tools/preprocessing/nusc_sample_generation.py \
    --nuscenes_path /work/amd456/autovla/dataset/nuscenes/ \
    --output_dir /work/amd456/autovla/data/nuscenes_processed/train \
    --split train \
    --version v1.0-trainval \
    $DRIVELM_ARG

# Process validation data (no DriveLM - validation only)
echo ""
echo "Processing nuScenes VALIDATION data..."
python tools/preprocessing/nusc_sample_generation.py \
    --nuscenes_path /work/amd456/autovla/dataset/nuscenes/ \
    --output_dir /work/amd456/autovla/data/nuscenes_processed/val \
    --split val \
    --version v1.0-trainval

echo ""
echo "================================================"
echo "Preprocessing complete!"
echo "End time: $(date)"
echo "================================================"

# Verify outputs
echo ""
echo "Train samples: $(find /work/amd456/autovla/data/nuscenes_processed/train -name "*.json" 2>/dev/null | wc -l)"
echo "Val samples: $(find /work/amd456/autovla/data/nuscenes_processed/val -name "*.json" 2>/dev/null | wc -l)"
echo ""
echo "Output directory size:"
du -sh /work/amd456/autovla/data/nuscenes_processed/

echo ""
echo "✅ Preprocessing job completed!"
