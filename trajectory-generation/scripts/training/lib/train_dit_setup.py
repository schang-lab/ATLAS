from __future__ import annotations

import inspect
import os
import argparse
import json
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

from src.dit import DiT
from src.diffusion_model import GaussianDiffusion
from src.helpers import normalize_prediction_type
from src.latent_pca import LatentPCA
from src.training import get_default_args


def load_config_and_prepare_dit(args, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[LatentPCA]]:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dit_params = get_default_args(DiT)
    if "DiT" in config:
        dit_params.update(config["DiT"])

    if "prediction_type" in config:
        args.prediction_type = normalize_prediction_type(config["prediction_type"])
    if "timestep_sampling" in config:
        args.timestep_sampling = config["timestep_sampling"]

    args.prediction_type = normalize_prediction_type(getattr(args, "prediction_type", "epsilon"))
    args.timestep_sampling = getattr(args, "timestep_sampling", "logsnr")
    args.timestep_sampling = args.timestep_sampling.lower().replace("-", "_")

    print(f"Loaded config: {config}")
    print(f"DiT config: {config.get('DiT', 'Not found')}")
    print(f"Anchor loss enabled: {args.use_anchor_loss}")
    if args.use_anchor_loss:
        print(f"Anchor loss weight: {args.anchor_loss_weight}")
    print(f"Diffusion prediction type: {args.prediction_type}")
    print(f"Timestep sampling strategy: {args.timestep_sampling}")

    if getattr(args, "enable_length_condition", False):
        length_vocab_size = int(getattr(args, "length_vocab_size", 513))
        if length_vocab_size <= 0:
            length_vocab_size = 513
        dit_params["use_length_condition"] = True
        dit_params["length_vocab_size"] = length_vocab_size
        print(f"Trajectory length conditioning enabled with vocab size: {length_vocab_size}")
    else:
        dit_params["use_length_condition"] = False
        print("Trajectory length conditioning disabled")

    latent_pca = None
    if args.latent_pca_path:
        if args.training_phase != "phase1":
            raise ValueError("--latent_pca_path currently requires training_phase='phase1'")
        print(f"Loading latent PCA artifacts from {args.latent_pca_path}")
        latent_pca = LatentPCA(args.latent_pca_path, device)
        dit_params["in_channels"] = latent_pca.component_dim
        print(
            f"Overriding DiT in_channels to PCA components: {latent_pca.component_dim} "
            f"(original latent dim {latent_pca.latent_dim})"
        )

    print("\n=== Training Configuration Summary ===")
    print(f"Training phase: {args.training_phase}")
    if args.training_phase == "phase2":
        print(f"Ablation mode: {args.ablation_mode}")
        print("Expected latent dimension: Will be loaded from autoencoder args.json")
    print(f"DiT input channels: {dit_params.get('in_channels', 'default')}")
    print(f"DiT use_length_condition: {dit_params.get('use_length_condition', False)}")
    print(f"DiT length_vocab_size: {dit_params.get('length_vocab_size', 'N/A')}")
    print(f"Autoencoder path: {args.autoencoder_path}")
    print("=== End Configuration Summary ===\n")
    return config, dit_params, latent_pca


def build_dit_model(args, device: torch.device, dit_params: Dict[str, Any]) -> Tuple[DiT, str]:
    print(f"Creating DiT with parameters: {dit_params}")
    dit_model = DiT(**dit_params).to(device)
    rank = os.environ.get("RANK", "NA")
    dit_param_count = sum(p.numel() for p in dit_model.parameters())
    print(f"[Rank {rank}] DiT module={inspect.getfile(DiT)} params={dit_param_count}", flush=True)

    if hasattr(args, "dit_checkpoint_path") and args.dit_checkpoint_path is not None:
        print("\n=== Loading DiT Checkpoint for Fine-tuning ===")
        print(f"Loading DiT model from: {args.dit_checkpoint_path}")

        if not os.path.exists(args.dit_checkpoint_path):
            raise FileNotFoundError(f"DiT checkpoint not found: {args.dit_checkpoint_path}")

        checkpoint = torch.load(args.dit_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            if "step" in checkpoint:
                print(f"Checkpoint was saved at step: {checkpoint['step']}")
            if "epoch" in checkpoint:
                print(f"Checkpoint was saved at epoch: {checkpoint['epoch']}")
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = dit_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        print("Successfully loaded DiT checkpoint for fine-tuning!")
        print("Note: You can now add new loss components (like length loss) for fine-tuning")
    else:
        print("Training DiT from scratch")
    return dit_model, rank


def build_noise_scheduler(args, device: torch.device, rank: str) -> GaussianDiffusion:
    schedule_kwargs = {}
    schedule_key = args.beta_schedule.lower()
    if schedule_key == "cosine":
        schedule_kwargs = {"s": args.cosine_s}
    elif schedule_key in {"logsnr", "logsnr_linear", "log-snr"}:
        schedule_kwargs = {"logsnr_max": args.logsnr_max, "logsnr_min": args.logsnr_min}

    noise_scheduler = GaussianDiffusion(
        timesteps=args.TIMESTEPS,
        schedule=args.beta_schedule,
        schedule_kwargs=schedule_kwargs,
    ).to(device)
    print(f"[Rank {rank}] Beta schedule: {args.beta_schedule} (kwargs: {schedule_kwargs})", flush=True)
    ns_param_count = sum(p.numel() for p in noise_scheduler.parameters())
    print(f"[Rank {rank}] Noise scheduler params={ns_param_count}", flush=True)
    return noise_scheduler


def validate_autoencoder_files_and_load_args(args, latent_model_path: str, dit_params: Dict[str, Any]):
    if args.training_phase == "phase1":
        required_files = ["config.json"]
        model_files = ["model.safetensors", "pytorch_model.bin"]
        has_model_file = any(os.path.exists(os.path.join(latent_model_path, f)) for f in model_files)
        if not has_model_file:
            raise FileNotFoundError(f"No model file found in Phase 1 directory: {latent_model_path}")
    else:
        required_files = ["config.json"]

    for file in required_files:
        file_path = os.path.join(latent_model_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    if args.training_phase != "phase2":
        return None

    args_json_path = os.path.join(latent_model_path, "args.json")
    if not os.path.exists(args_json_path):
        parent_dir = os.path.dirname(latent_model_path)
        args_json_path = os.path.join(parent_dir, "args.json")
        if not os.path.exists(args_json_path):
            raise FileNotFoundError(f"args.json not found in {latent_model_path} or {parent_dir}")
        print(f"Found args.json in parent directory: {args_json_path}")
    else:
        print(f"Found args.json in current directory: {args_json_path}")

    with open(args_json_path, "rt") as f:
        latent_model_args = json.load(f)
    latent_argparse = argparse.Namespace(**latent_model_args)

    expected_latent_dim = latent_argparse.dim_ae
    dit_in_channels = dit_params.get("in_channels", expected_latent_dim)
    print(f"Autoencoder latent dimension: {expected_latent_dim}")
    print(f"DiT in_channels: {dit_in_channels}")

    if dit_in_channels != expected_latent_dim:
        print(f"Warning: DiT in_channels ({dit_in_channels}) != autoencoder latent dim ({expected_latent_dim})")
        print("Please update the config file to match the autoencoder latent dimension")
    else:
        print(" DiT configuration matches autoencoder latent dimension")
    return latent_argparse
