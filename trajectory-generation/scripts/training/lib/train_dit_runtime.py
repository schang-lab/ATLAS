from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim
import yaml

from src.dit import DiT
from src.training import get_default_args
from src.checkpoint_utils import load_training_checkpoint


def apply_seed(args) -> None:
    if args.seed is None:
        return
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Using random seed: {seed}")


def print_training_configuration(args) -> None:
    print("=== DiT Training Configuration ===")
    print(f"Training phase: {args.training_phase}")
    print(f"Autoencoder path: {args.autoencoder_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Data type: {args.data_type}")
    if args.data_type == "controlled":
        print("Training mode: Conditional (with work/home coordinates)")
    elif args.data_type == "uncontrolled":
        print("Training mode: Unconditional (no individual attributes)")
    else:
        print("Training mode: Unified (both conditional and unconditional)")
        print(f"Conditional dropout rate: {args.conditional_dropout}")
    print(f"Coord dropout (coords only): {getattr(args, 'coord_dropout', 0.0)}")

    if args.training_phase == "phase2":
        print(f"Ablation mode: {args.ablation_mode}")
    elif args.ablation_mode != "both":
        print(f"Warning: Ablation mode '{args.ablation_mode}' is ignored for phase1 training")

    print(f"Anchor loss: {'enabled' if args.use_anchor_loss else 'disabled'}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Validation: {'enabled' if args.enable_validation else 'disabled'}")
    if args.enable_validation:
        if args.eval_samples:
            print(f"Validation samples: {args.eval_samples}")
        print(f"Validation guidance scale: {getattr(args, 'validation_guidance_scale', 1.0)}")
    else:
        print(f"Validation guidance scale: {getattr(args, 'validation_guidance_scale', 1.0)} (unused)")
    print("Step-based training:")
    print(f"  Log every {args.log_steps} steps")
    print(f"  Save every {args.save_steps} steps")
    print(f"  Eval every {args.eval_steps} steps")
    print(f"  Max steps: {args.max_steps if args.max_steps else 'epochs * batches'}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Beta schedule: {args.beta_schedule}")
    if args.beta_schedule.lower() == "cosine":
        print(f"    cosine_s: {args.cosine_s}")
    elif args.beta_schedule.lower() in {"logsnr", "logsnr_linear", "log-snr"}:
        print(f"    logsnr_max: {args.logsnr_max}, logsnr_min: {args.logsnr_min}")
    print(f"  Latent scale: {args.latent_scale}")
    print(f"Wandb logging: {'enabled' if args.use_wandb else 'disabled'}")
    if args.use_wandb:
        print(f"  Project: {args.wandb_project}")
        print(f"  Run name: {args.wandb_run_name}")

    if hasattr(args, "dit_checkpoint_path") and args.dit_checkpoint_path is not None:
        print("DiT checkpoint loading: enabled")
        print(f"  Checkpoint path: {args.dit_checkpoint_path}")
        print("  Mode: Fine-tuning from existing model")

    if getattr(args, "diag_decode_every", 0):
        print(f"Diagnostic decoding every {args.diag_decode_every} steps")
        print(f"Diagnostic guidance scale: {getattr(args, 'diag_guidance_scale', 1.0)}")
    else:
        print("DiT checkpoint loading: disabled (training from scratch)")
    print("=== End Configuration ===\n")


def setup_accelerator_and_device(args) -> Tuple[Accelerator, torch.device, str]:
    if args.force_cpu:
        print("Forcing CPU usage as requested.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.is_available = lambda: False
    elif torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. Using CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    args.is_main_process = accelerator.is_main_process

    timestamp = args.timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return accelerator, accelerator.device, timestamp


def resolve_resume_checkpoint_path(args) -> Optional[str]:
    if args.resume_from_checkpoint is None:
        return None
    print(f"Using explicit checkpoint path: {args.resume_from_checkpoint}")
    return args.resume_from_checkpoint


def resolve_resume_wandb_id(resume_checkpoint_path: Optional[str]) -> Optional[str]:
    if not resume_checkpoint_path or not os.path.exists(resume_checkpoint_path):
        return None
    try:
        checkpoint_preview = torch.load(resume_checkpoint_path, map_location="cpu")
        resume_wandb_id = checkpoint_preview.get("wandb_run_id", None)
        if resume_wandb_id:
            print(f"Found wandb_id in checkpoint: {resume_wandb_id}")
        else:
            print("No wandb_id found in checkpoint - will create new run")
        return resume_wandb_id
    except Exception as e:
        print(f"Warning: Could not preview checkpoint for wandb_id: {e}")
        return None


def load_resume_state_if_available(args, device: torch.device, accelerator: Accelerator, resume_checkpoint_path: Optional[str]):
    if resume_checkpoint_path is None:
        return None

    dit_params = get_default_args(DiT)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if "DiT" in config:
        dit_params.update(config["DiT"])

    dummy_dit_model = DiT(**dit_params).to(device)
    dummy_optimizer = optim.Adam(dummy_dit_model.parameters(), lr=args.OPTIM_LR)
    global_step_resume, epoch_resume, best_val_loss_resume, wandb_id_resume = load_training_checkpoint(
        resume_checkpoint_path, dummy_dit_model, dummy_optimizer, accelerator, args
    )
    resume_state = {
        "global_step": global_step_resume,
        "epoch": epoch_resume,
        "best_val_loss": best_val_loss_resume,
        "wandb_id": wandb_id_resume,
    }
    print("Checkpoint loaded successfully - training will resume from saved state")
    print(f"Restored training_phase: {args.training_phase}")
    return resume_state


def infer_sequence_length_from_config(args) -> None:
    try:
        with open(args.config, "r") as f:
            pre_cfg = yaml.safe_load(f) or {}
        dit_cfg = pre_cfg.get("DiT", {}) if isinstance(pre_cfg, dict) else {}
        if isinstance(dit_cfg, dict) and "image_size" in dit_cfg:
            args.sequence_length = int(dit_cfg["image_size"])
            print(f"Inferred dataset sequence_length={args.sequence_length} from DiT.image_size in config")
    except Exception as e:
        print(f"Warning: could not infer sequence length from config ({e}); using training defaults.")


def init_wandb_if_enabled(args, accelerator: Accelerator, timestamp: str, resume_wandb_id: Optional[str]) -> None:
    if not (args.use_wandb and accelerator.is_main_process):
        return

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        print("Wandb API key set from command line argument")
    elif os.getenv("WANDB_API_KEY"):
        print("Wandb API key loaded from environment variable")
    else:
        print("Warning: WANDB_API_KEY is not set; wandb.init may fail if authentication is required")

    wandb_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name or f"dit_{args.training_phase}_{args.data_type}_{timestamp}",
        "config": {
            "training_phase": args.training_phase,
            "autoencoder_path": args.autoencoder_path,
            "data_type": args.data_type,
            "conditional_dropout": args.conditional_dropout,
            "ablation_mode": args.ablation_mode,
            "batch_size": args.BATCH_SIZE,
            "learning_rate": args.OPTIM_LR,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "timesteps": args.TIMESTEPS,
            "use_anchor_loss": args.use_anchor_loss,
            "anchor_loss_weight": args.anchor_loss_weight,
            "enable_validation": args.enable_validation,
            "log_steps": args.log_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "enable_length_condition": getattr(args, "enable_length_condition", False),
        },
    }

    effective_wandb_id = resume_wandb_id or args.wandb_id
    if effective_wandb_id:
        wandb_kwargs["id"] = effective_wandb_id
        wandb_kwargs["resume"] = "must"
        if resume_wandb_id:
            print(f"Resuming wandb run from checkpoint with ID: {effective_wandb_id}")
        else:
            print(f"Resuming wandb run with manually provided ID: {effective_wandb_id}")
    else:
        print("Creating new wandb run")

    wandb.init(**wandb_kwargs)
    print(f"Wandb initialized: {wandb.run.name}")
