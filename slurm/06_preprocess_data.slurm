#!/bin/bash
#SBATCH --job-name=autovla_preprocess_06
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/06_preprocess.out
#SBATCH --error=logs/06_preprocess.err

# ============================================================
# STEP 6: Preprocess nuScenes Dataset
# ============================================================

echo "========== STEP 6: Preprocess nuScenes Dataset ==========="
echo "Job: $SLURM_JOB_ID"
echo "Time: $(date)"

# Activate conda environment
echo "Activating conda environment..."
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate autovla

# Install required packages (nuscenes-devkit from GitHub)
echo "Installing nuscenes-devkit from GitHub..."
pip install git+https://github.com/nutonomy/nuscenes-devkit.git

# Set paths
NUSCENES_PATH="/work/amd456/autovla/dataset/nuscenes/v1.0-trainval"
OUTPUT_DIR="/work/amd456/autovla/data/nuscenes_processed"
PROJECT_ROOT="/work/amd456/autovla"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Verify nuScenes directory exists
if [ ! -d "${NUSCENES_PATH}" ]; then
    echo "❌ Error: nuScenes path not found: ${NUSCENES_PATH}"
    exit 1
fi

echo "✅ nuScenes path verified: ${NUSCENES_PATH}"

# Check for metadata
if [ ! -f "${NUSCENES_PATH}/sample.json" ]; then
    echo "❌ Error: sample.json not found in ${NUSCENES_PATH}"
    exit 1
fi

echo "✅ Metadata verified"

# Change to project directory
cd "${PROJECT_ROOT}" || exit

# Start timer
START_TIME=$(date +%s)

# ============================================================
# PREPROCESSING
# ============================================================

echo ""
echo "============================================================"
echo "Processing training split..."
echo "============================================================"

# Process training data
python tools/preprocessing/nusc_sample_generation.py \
    --nuscenes_path "${NUSCENES_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --split train \
    --version v1.0-trainval

if [ $? -ne 0 ]; then
    echo "❌ Training preprocessing failed"
    exit 1
fi

echo "✅ Training split processed"

echo ""
echo "============================================================"
echo "Processing validation split..."
echo "============================================================"

# Process validation data
python tools/preprocessing/nusc_sample_generation.py \
    --nuscenes_path "${NUSCENES_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --split val \
    --version v1.0-trainval

if [ $? -ne 0 ]; then
    echo "❌ Validation preprocessing failed"
    exit 1
fi

echo "✅ Validation split processed"

# ============================================================
# VERIFICATION
# ============================================================

echo ""
echo "============================================================"
echo "Verifying preprocessed data..."
echo "============================================================"

# Check output structure
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "❌ Error: Output directory not created"
    exit 1
fi

echo "Output directory contents:"
ls -lh "${OUTPUT_DIR}/"

# Count files
TRAIN_FILES=$(find "${OUTPUT_DIR}" -name "*train*.json" 2>/dev/null | wc -l)
VAL_FILES=$(find "${OUTPUT_DIR}" -name "*val*.json" 2>/dev/null | wc -l)

echo "Training JSON files: ${TRAIN_FILES}"
echo "Validation JSON files: ${VAL_FILES}"

# Check total size
TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
echo "Total preprocessed data size: ${TOTAL_SIZE}"

# End timer
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========== DONE =========="
echo "Processing completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Next: Configure training and run SFT training (Step 7)"
echo "Check: ls -lh ${OUTPUT_DIR}/"
