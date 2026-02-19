#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --gpus=A6000:1
#SBATCH --time=72:00:00

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
nvidia-smi
echo "=============================="

# ----------------------------
# User-configurable paths
# ----------------------------
# Set these before submitting, or export them in your shell.
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/ATLAS}"
VENV_PATH="${VENV_PATH:-/path/to/venv/bin/activate}"
DATA_FOLDER="${DATA_FOLDER:-/path/to/split_data/controlled}"
OUTPUT_BASE="${OUTPUT_BASE:-/path/to/outputs/BART}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-example_train}"
WANDB_PROJECT="${WANDB_PROJECT:-BART}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}"
else
  echo "Warning: VENV_PATH not found: ${VENV_PATH}"
fi

SCRIPT_DIR="${PROJECT_ROOT}/autoencoder/train_autoencoder"
cd "${SCRIPT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME_PREFIX}_${TS}"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/epoch_evaluations"

srun python train_phase1_pretrain.py \
    --max_length 64 \
    --data_folder "${DATA_FOLDER}" \
    --output_dir "${OUTPUT_DIR}" \
    --encoder_layers 4 \
    --decoder_layers 2 \
    --d_model 256 \
    --encoder_ffn_dim 1024 \
    --decoder_ffn_dim 1024 \
    --encoder_attention_heads 4 \
    --decoder_attention_heads 4 \
    --batch_size 512 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --epochs 60 \
    --logging_steps 100 \
    --save_steps 5000 \
    --eval_steps 100 \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_name "${RUN_NAME_PREFIX}_${TS}" \
    --enable_epoch_evaluation \
    --eval_sample_size 4000 \
    --eval_frequency 1 \
    --eval_batch_size 64 \
    --eval_output_dir "${EVAL_OUTPUT_DIR}"

echo "Run completed: ${RUN_NAME_PREFIX}_${TS}"