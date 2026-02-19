#!/bin/bash
#SBATCH --job-name=train3
#SBATCH --partition=main
#SBATCH --qos=default
#SBATCH --cpus-per-task=32
#SBATCH --mem=45gb
#SBATCH --gpus=A6000:1
#SBATCH --time=72:00:00
#SBATCH --output=dit_only_train_%j.out
#SBATCH --error=dit_only_train_%j.err

# ----------------------------
# User-configurable paths
# Change these to your own folder paths before running.
# ----------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/ATLAS}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/trajectory-generation/scripts/training/run_cbg_conditioned_training.py}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/trajectory-generation/configs/phase2-config/ATLAS/demogroups/poi-tv.yaml}"

srun python "${TRAIN_SCRIPT}" --config "${CONFIG_PATH}"