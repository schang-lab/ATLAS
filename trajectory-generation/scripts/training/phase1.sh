#!/bin/bash
#SBATCH --job-name=unified
#SBATCH --partition=main
#SBATCH --qos=default
#SBATCH --cpus-per-task=8
#SBATCH --gpus=A6000:1
#SBATCH --time=72:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# ----------------------------
# User-configurable paths
# Change these to your own folder paths before running.
# ----------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/ATLAS}"
VENV_PATH="${VENV_PATH:-/path/to/venv/bin/activate}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/trajectory-generation/scripts/training/train_dit_only.py}"
PHASE1_CONFIG="${PHASE1_CONFIG:-${PROJECT_ROOT}/trajectory-generation/configs/phase1-config/config_phase1_128.yml}"
AUTOENCODER_PATH="${AUTOENCODER_PATH:-/path/to/phase1_autoencoder/checkpoint}"
DATA_DIR="${DATA_DIR:-/path/to/YOUR_DATA_FOLDER}"
LATENT_PCA_PATH="${LATENT_PCA_PATH:-/path/to/latent_pca/train.pt}"

if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}"
else
  echo "Warning: VENV_PATH not found: ${VENV_PATH}"
fi

WANDB_MODE="${WANDB_MODE:-online}"

srun python "${TRAIN_SCRIPT}" \
    -b 512 \
    --gradient_accumulation_steps 1 \
    -e 100 \
    -t 1000 \
    -config "${PHASE1_CONFIG}" \
    --autoencoder_path "${AUTOENCODER_PATH}" \
    --OPTIM_LR 1e-4 \
    --training_phase phase1 \
    --use_wandb \
    --wandb_project dit-embee \
    --wandb_run_name baseline-23 \
    --seed 23 \
    --enable_validation \
    --eval_steps 500 \
    --eval_samples 1000 \
    --save_steps 5000 \
    --log_steps 500 \
    --data_dir "${DATA_DIR}" \
    --data_type controlled \
    --beta_schedule cosine \
    --prediction_type x0 \
    --ddim_probe_every 500 \
    --ddim_probe_start 500 \
    --ddim_probe_batch 64 \
    --latent_pca_path "${LATENT_PCA_PATH}"