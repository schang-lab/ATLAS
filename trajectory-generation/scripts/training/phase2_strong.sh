#!/bin/bash
#SBATCH --job-name=strong-0.1
#SBATCH --partition=main
#SBATCH --qos=default
#SBATCH --cpus-per-task=32
#SBATCH --mem=45gb
#SBATCH --gpus=A6000:1
#SBATCH --time=72:00:00
#SBATCH --output=strong-0.1_%j.out
#SBATCH --error=strong-0.1_%j.err

# ----------------------------
# User-configurable paths
# Change these to your own folder paths before running.
# ----------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/ATLAS}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/trajectory-generation/scripts/training/train_dit_only.py}"
PHASE2_STRONG_CONFIG="${PHASE2_STRONG_CONFIG:-${PROJECT_ROOT}/trajectory-generation/configs/phase2-config/strong/dit_strong_config.yaml}"
AUTOENCODER_PATH="${AUTOENCODER_PATH:-/path/to/phase1_autoencoder/checkpoint}"
SUPERVISED_SPLIT_DIR="${SUPERVISED_SPLIT_DIR:-/path/to/atlas_world/supervised_splits/demogroups_aligned}"
LATENT_PCA_PATH="${LATENT_PCA_PATH:-/path/to/latent_pca/train.pt}"
DIT_CHECKPOINT_PATH="${DIT_CHECKPOINT_PATH:-/path/to/dit_checkpoint/dit_step_xxx.pt}"

srun python "${TRAIN_SCRIPT}" \
  --CONFIG "${PHASE2_STRONG_CONFIG}" \
  --autoencoder_path "${AUTOENCODER_PATH}" \
  --data_dir "${SUPERVISED_SPLIT_DIR}" \
  --data_type controlled \
  --training_phase phase1 \
  --TIMESTEPS 1000 \
  --prediction_type x0 \
  --beta_schedule cosine \
  --timestep_sampling uniform \
  --BATCH_SIZE 1024 \
  --gradient_accumulation_steps 1\
  -e 1000 \
  --OPTIM_LR 1e-5 \
  --log_steps 100 \
  --save_steps 10000 \
  --warmup_steps 0 \
  --use_wandb \
  --wandb_project dit-embee \
  --enable_validation \
  --eval_steps 500 \
  --eval_samples 1000 \
  --latent_pca_path "${LATENT_PCA_PATH}" \
  --dit_checkpoint_path "${DIT_CHECKPOINT_PATH}" \
  --wandb_run_name strong-supervise-0.1 \
  --coord_dropout 0.1