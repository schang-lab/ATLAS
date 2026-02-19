from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import BartForConditionalGeneration

from auto_encoder.traj_compressed_ae import BARTLatentCompression


def _load_state_dict_for_inspection(latent_model_path: str):
    model_files = ["model.safetensors", "pytorch_model.bin"]
    state_dict = None
    for file in model_files:
        file_path = os.path.join(latent_model_path, file)
        if os.path.exists(file_path):
            if file.endswith(".safetensors"):
                from safetensors import safe_open

                with safe_open(file_path, framework="pt", device="cpu") as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
            else:
                state_dict = torch.load(file_path, map_location="cpu")
            print(f"Loaded state dict from {file} with {len(state_dict)} keys")
            break
    if state_dict is None:
        raise FileNotFoundError(f"No model weights found in {latent_model_path}")
    return state_dict


def _fallback_manual_load(autoencoder, latent_model_path: str, device: torch.device) -> None:
    model_files = ["model.pt", "model.safetensors", "pytorch_model.bin"]
    model_file = None
    for file in model_files:
        file_path = os.path.join(latent_model_path, file)
        if os.path.exists(file_path):
            model_file = file_path
            break
    if model_file is None:
        raise FileNotFoundError(f"No model file found. Tried: {model_files}")

    print(f"Loading autoencoder from: {model_file}")
    if model_file.endswith(".pt"):
        ae_model = torch.load(model_file, map_location=device)
        autoencoder.load_state_dict(ae_model["model"])
        return

    if model_file.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(model_file, framework="pt", device="cpu") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
    else:
        state_dict = torch.load(model_file, map_location=device)

    try:
        autoencoder.load_state_dict(state_dict, strict=False)
        print(" Autoencoder loaded successfully (non-strict mode)")
    except Exception as e:
        print(f"Warning: Could not load state dict: {e}")
        print("Continuing with randomly initialized autoencoder...")


def build_autoencoder(args, latent_model_path: str, ae_config, latent_argparse: Optional[object], device: torch.device):
    if args.training_phase == "phase1":
        print("Loading Phase 1 BART autoencoder...")
        if not os.path.isdir(latent_model_path):
            raise ValueError(f"Phase 1 autoencoder path must be a directory: {latent_model_path}")
        checkpoint_files = os.listdir(latent_model_path)
        if "model.safetensors" not in checkpoint_files and "pytorch_model.bin" not in checkpoint_files:
            raise ValueError(f"Invalid Phase 1 checkpoint directory: {latent_model_path}")
        autoencoder = BartForConditionalGeneration.from_pretrained(latent_model_path)
        print("Phase 1 autoencoder: Loaded from checkpoint directory")
        print("Phase 1 autoencoder: Using simple BART without compression")
        return autoencoder

    print("Loading Phase 2 compressed autoencoder...")
    use_coords = args.ablation_mode in ["coords_only", "both"]
    use_subcategories = args.ablation_mode in ["subcat_only", "both"]
    print("Phase 2 autoencoder configuration:")
    print(f"  - Ablation mode: {args.ablation_mode}")
    print(f"  - Use coordinates: {use_coords}")
    print(f"  - Use subcategories: {use_subcategories}")
    print(f"  - No compression: {getattr(latent_argparse, 'no_compression', False)}")

    autoencoder = BARTLatentCompression(
        config=ae_config,
        num_encoder_latents=latent_argparse.num_encoder_latents,
        num_decoder_latents=latent_argparse.num_decoder_latents,
        dim_ae=latent_argparse.dim_ae,
        num_layers=getattr(latent_argparse, "num_layers", 2),
        l2_normalize_latents=latent_argparse.l2_normalize_latents,
        use_coords=use_coords,
        num_sub_categories=getattr(latent_argparse, "num_sub_categories", None) if use_subcategories else None,
        use_position_embedding=getattr(latent_argparse, "use_position_embedding", True),
        transformer_decoder=getattr(latent_argparse, "transformer_decoder", False),
        no_compression=getattr(latent_argparse, "no_compression", False),
    )

    print(f"Loading full autoencoder from: {latent_model_path}")
    print("Examining saved model structure...")
    state_dict = _load_state_dict_for_inspection(latent_model_path)

    print("Analyzing saved model structure:")
    perceiver_keys = [k for k in state_dict.keys() if "perceiver" in k]
    bart_keys = [k for k in state_dict.keys() if any(x in k for x in ["encoder", "decoder", "lm_head"])]
    coord_keys = [k for k in state_dict.keys() if "coord" in k]
    subcat_keys = [k for k in state_dict.keys() if "sub_category" in k]
    print(f"  Perceiver keys: {len(perceiver_keys)}")
    print(f"  BART keys: {len(bart_keys)}")
    print(f"  Coordinate keys: {len(coord_keys)}")
    print(f"  Sub-category keys: {len(subcat_keys)}")
    if coord_keys:
        print(f"  Sample coord key: {coord_keys[0]} -> {state_dict[coord_keys[0]].shape}")
    if subcat_keys:
        print(f"  Sample subcat key: {subcat_keys[0]} -> {state_dict[subcat_keys[0]].shape}")

    coord_output_dim = None
    if coord_keys:
        for key in coord_keys:
            if "weight" in key and "0" in key:
                coord_input_dim = state_dict[key].shape[1]
                coord_output_dim = state_dict[key].shape[0]
                print(f"  Detected coordinate input dimension: {coord_input_dim}")
                print(f"  Detected coordinate output dimension (dim_ae): {coord_output_dim}")
                break

    subcat_vocab_size = None
    if subcat_keys:
        for key in subcat_keys:
            if "embedding" in key and "weight" in key:
                subcat_vocab_size = state_dict[key].shape[0]
                subcat_embed_dim = state_dict[key].shape[1]
                print(f"  Detected sub-category vocabulary size: {subcat_vocab_size}")
                print(f"  Detected sub-category embedding dimension: {subcat_embed_dim}")
                break

    print("Creating autoencoder with parameters matching saved model...")
    actual_dim_ae = coord_output_dim if coord_output_dim is not None else latent_argparse.dim_ae
    actual_num_sub_categories = (
        subcat_vocab_size if subcat_vocab_size is not None else getattr(latent_argparse, "num_sub_categories", None)
    )
    print("Using detected parameters:")
    print(f"  dim_ae: {actual_dim_ae}")
    print(f"  num_sub_categories: {actual_num_sub_categories}")
    print(f"  use_coords: {use_coords}")
    print(f"  use_subcategories: {use_subcategories}")
    print(f"  no_compression: {getattr(latent_argparse, 'no_compression', False)}")

    try:
        autoencoder = BARTLatentCompression.from_pretrained(
            latent_model_path,
            config=ae_config,
            num_encoder_latents=latent_argparse.num_encoder_latents,
            num_decoder_latents=latent_argparse.num_decoder_latents,
            dim_ae=actual_dim_ae,
            num_layers=getattr(latent_argparse, "num_layers", 2),
            l2_normalize_latents=latent_argparse.l2_normalize_latents,
            use_coords=use_coords,
            num_sub_categories=actual_num_sub_categories if use_subcategories else None,
            use_position_embedding=getattr(latent_argparse, "use_position_embedding", True),
            transformer_decoder=getattr(latent_argparse, "transformer_decoder", False),
            no_compression=getattr(latent_argparse, "no_compression", False),
        )
        print(" Full autoencoder loaded successfully using from_pretrained")
    except Exception as e:
        print(f"Warning: Could not load using from_pretrained: {e}")
        print("Falling back to manual loading...")
        _fallback_manual_load(autoencoder, latent_model_path, device)

    return autoencoder
