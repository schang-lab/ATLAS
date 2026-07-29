#!/usr/bin/env python3
"""
Train a VAE backbone for the ATLAS trajectory generation framework.

This script mirrors train_dit_only.py but replaces the DiT + diffusion backbone
with a VAE. The ATLAS aggregate demographic loss and anchor loss are compatible
with any backbone that produces (B, T, D) latent sequences — the VAE satisfies
this interface.

Usage:
    python train_vae.py \
        --config configs/config_vae_phase1.yml \
        --autoencoder_path /path/to/pretrained_autoencoder \
        --data_dir /path/to/split_data \
        --data_type unified \
        --max_steps 50000 \
        --BATCH_SIZE 64
"""

import os
import sys
import math
import random
from datetime import datetime
from typing import Optional
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import optim
import yaml
from tqdm import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs

import wandb

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vae import TrajectoryVAE
from src.training import (
    trajectory_dataset,
    create_directory,
    get_model_size,
    save_training_info,
    get_default_args,
    load_restart_training_parameters,
)
from src.losses import compute_anchor_loss
from src.latent_pca import LatentPCA

from auto_encoder.traj_compressed_ae import BARTLatentCompression

from transformers import (
    BertTokenizerFast,
    BartForConditionalGeneration,
    BartConfig,
)
from transformers.modeling_outputs import BaseModelOutput


DEMO_PARAM_PREFIXES = (
    "age_embedding",
    "gender_embedding",
    "demo_scale",
    "demo_shift",
)


def _is_demo_param(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in DEMO_PARAM_PREFIXES)


def _load_vae_init_checkpoint(
    vae_model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Warm-start a VAE, allowing newly added demo-branch parameters."""
    if not checkpoint_path:
        return

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"VAE init checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for model_key in ("model", "state_dict", "model_state_dict"):
            if model_key in checkpoint:
                state = checkpoint[model_key]
                break
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported VAE checkpoint format: {checkpoint_path}")

    # Saved checkpoints in this script are unwrapped, but tolerate DDP/DataParallel prefixes.
    if state and all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    model_state = vae_model.state_dict()
    compatible_state = {}
    unexpected_keys = []
    skipped_shape_keys = []
    for key, value in state.items():
        if not torch.is_tensor(value):
            unexpected_keys.append(key)
            continue
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape_keys.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state[key] = value

    if not compatible_state:
        raise ValueError(f"No compatible tensors found in VAE init checkpoint: {checkpoint_path}")

    missing_keys, load_unexpected = vae_model.load_state_dict(compatible_state, strict=False)

    print(f"Loaded VAE init checkpoint: {checkpoint_path}")
    print(f"  Compatible tensors loaded: {len(compatible_state)} / {len(model_state)}")
    if missing_keys:
        print(f"  Missing keys initialized from current model: {missing_keys}")
    if unexpected_keys or load_unexpected:
        print(f"  Unexpected checkpoint keys ignored: {unexpected_keys + list(load_unexpected)}")
    if skipped_shape_keys:
        print(f"  Shape-mismatched checkpoint keys ignored: {skipped_shape_keys}")


def _build_vae_optimizer(vae_model: torch.nn.Module, args) -> optim.Optimizer:
    """Build optimizer, optionally holding backbone updates at lr=0 for demo warmup."""
    demo_only_steps = int(getattr(args, "demo_only_steps", 0) or 0)
    freeze_backbone = bool(getattr(args, "freeze_backbone_for_demo", False))
    requested_demo_tuning = freeze_backbone or demo_only_steps > 0
    if requested_demo_tuning and not bool(getattr(vae_model, "use_demo_condition", False)):
        raise ValueError("Demo-only VAE tuning was requested, but use_demo_condition is false.")
    use_demo_groups = bool(getattr(vae_model, "use_demo_condition", False)) and requested_demo_tuning

    if not use_demo_groups:
        return optim.AdamW(vae_model.parameters(), lr=args.OPTIM_LR, weight_decay=1e-4)

    named_params = [(name, param) for name, param in vae_model.named_parameters()]
    demo_params = [param for name, param in named_params if _is_demo_param(name)]
    backbone_params = [param for name, param in named_params if not _is_demo_param(name)]
    if not demo_params:
        raise ValueError("Demo-only VAE tuning requested, but no demo branch parameters were found.")

    optimizer = optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": 0.0,
                "weight_decay": 1e-4,
                "name": "backbone",
            },
            {
                "params": demo_params,
                "lr": args.OPTIM_LR,
                "weight_decay": 1e-4,
                "name": "demo",
            },
        ]
    )
    mode = "entire run" if freeze_backbone else f"first {demo_only_steps} optimizer steps"
    print(
        f"Demo tuning enabled: backbone lr=0 for {mode}; "
        f"demo lr={args.OPTIM_LR}, backbone_lr_scale={args.backbone_lr_scale}"
    )
    return optimizer


def _set_backbone_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "backbone":
            group["lr"] = lr


def _shift_demo_ids_in_attrs(
    attrs: Optional[torch.Tensor],
    vae_model: torch.nn.Module,
) -> Optional[torch.Tensor]:
    """
    Shift raw 0-based demo ids to 1-based for embedding lookup.
    Same logic as train_dit_only.py but works with VAE model.
    """
    if attrs is None:
        return None
    if attrs.dim() != 2 or attrs.size(1) < 6:
        return attrs

    base_model = getattr(vae_model, "module", vae_model)
    if not getattr(base_model, "use_demo_condition", False):
        return attrs

    age_idx = -2
    gender_idx = -1
    age_raw = attrs[:, age_idx].long()
    gender_raw = attrs[:, gender_idx].long()

    missing = (age_raw < 0) | (gender_raw < 0)

    age_shifted = (age_raw.clamp_min(0) + 1).to(dtype=attrs.dtype, device=attrs.device)
    gender_shifted = (gender_raw.clamp_min(0) + 1).to(dtype=attrs.dtype, device=attrs.device)

    if missing.any():
        age_shifted = age_shifted.clone()
        gender_shifted = gender_shifted.clone()
        age_shifted[missing] = 0
        gender_shifted[missing] = 0

    attrs_out = attrs.clone()
    attrs_out[:, age_idx] = age_shifted
    attrs_out[:, gender_idx] = gender_shifted
    return attrs_out


def _extract_target_latents(batch, autoencoder, args, latent_pca):
    """Extract BART encoder latents from a batch (shared by train and val)."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    with torch.no_grad():
        if args.training_phase == "phase1":
            encoder_outputs = autoencoder.get_encoder()(
                input_ids=input_ids, attention_mask=attention_mask
            )
            target_latents = encoder_outputs.last_hidden_state
        else:
            encoder_outputs = autoencoder.get_encoder()(
                input_ids=input_ids, attention_mask=attention_mask
            )
            segment_coords = None
            sub_categories = None
            if args.ablation_mode in ["coords_only", "both"]:
                if "lat" in batch and "lon" in batch:
                    segment_coords = torch.stack([batch["lat"], batch["lon"]], dim=-1)
            if args.ablation_mode in ["subcat_only", "both"]:
                sub_categories = batch.get("sub_categories", None)

            if hasattr(autoencoder, "no_compression") and autoencoder.no_compression:
                enhanced_outputs = autoencoder._add_features_no_compression(
                    encoder_outputs, attention_mask, segment_coords, sub_categories
                )
                target_latents = enhanced_outputs["last_hidden_state"]
            else:
                target_latents = autoencoder.get_diffusion_latent(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    segment_coords=segment_coords,
                    sub_categories=sub_categories,
                )

    if latent_pca is not None:
        target_latents = latent_pca.project(target_latents)

    latent_scale = getattr(args, "latent_scale", 1.0)
    if latent_scale != 1.0:
        target_latents = target_latents / latent_scale

    return target_latents


def _prepare_attrs(batch, args, vae_model):
    """Prepare attribute tensor from a batch (shared by train and val)."""
    attrs = batch["attrs"]
    length_id = batch.get("length_id", None)

    if length_id is not None and getattr(args, "enable_length_condition", False):
        length_tensor = length_id.float().unsqueeze(-1).to(attrs.device)
        attrs = torch.cat([attrs, length_tensor], dim=1)

    attrs = _shift_demo_ids_in_attrs(attrs, vae_model)
    return attrs


def validate_vae(
    vae_model,
    autoencoder,
    valid_dataloader,
    args,
    accelerator,
    beta_kl: float,
    latent_pca: Optional[LatentPCA] = None,
    max_batches: Optional[int] = None,
):
    """Run validation and return (avg_total, avg_recon, avg_kl, avg_anchor)."""
    vae_model.eval()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    anchor_loss_sum = 0.0
    num_batches = 0

    # For unified data: track conditional vs unconditional separately
    cond_loss_sum = 0.0
    uncond_loss_sum = 0.0
    num_cond = 0
    num_uncond = 0

    latent_scale = getattr(args, "latent_scale", 1.0)

    with torch.no_grad():
        for batch in valid_dataloader:
            if max_batches is not None and num_batches >= max_batches:
                break

            attrs = _prepare_attrs(batch, args, vae_model)
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            target_latents = _extract_target_latents(batch, autoencoder, args, latent_pca)

            # Conditional dropout for unified data (same logic as training)
            attr_embeds_for_model = None
            if args.data_type == "unified":
                is_conditional = batch.get("is_conditional", None)
                if is_conditional is not None:
                    # No dropout during validation — use actual conditional flags
                    if is_conditional.any():
                        attr_embeds_for_model = attrs.clone()
                        attr_embeds_for_model[~is_conditional] = 0
                    else:
                        attr_embeds_for_model = None
                else:
                    attr_embeds_for_model = attrs
            else:
                if attrs is not None and attrs.abs().sum() > 0:
                    attr_embeds_for_model = attrs

            base_model = getattr(vae_model, "module", vae_model)
            loss_dict = base_model.compute_loss(
                x=target_latents,
                attr_embeds=attr_embeds_for_model,
                attention_mask=attention_mask,
                beta_kl=beta_kl,
            )

            vae_loss = loss_dict["total_loss"]
            recon_loss = loss_dict["recon_loss"]
            kl_loss = loss_dict["kl_loss"]

            anchor_loss = torch.tensor(0.0, device=target_latents.device)
            if args.use_anchor_loss:
                predicted_x0 = loss_dict["recon"]
                if latent_scale != 1.0:
                    predicted_x0 = predicted_x0 * latent_scale
                anchor_loss, _, _ = compute_anchor_loss(
                    predicted_x0=predicted_x0,
                    labels=labels,
                    attention_mask=attention_mask,
                    autoencoder=autoencoder,
                    latent_scale=1.0,
                    latent_pca=latent_pca,
                    training_phase=args.training_phase,
                    return_per_sample=False,
                )

            batch_total = vae_loss + args.anchor_loss_weight * anchor_loss

            total_loss_sum += batch_total.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            anchor_loss_sum += anchor_loss.item()
            num_batches += 1

            # Track conditional vs unconditional for unified data
            if args.data_type == "unified":
                is_conditional = batch.get("is_conditional", None)
                if is_conditional is not None:
                    n_c = is_conditional.sum().item()
                    n_u = (~is_conditional).sum().item()
                    num_cond += n_c
                    num_uncond += n_u
                    # Approximate per-split loss using batch loss weighted by fraction
                    bs = is_conditional.numel()
                    if n_c > 0:
                        cond_loss_sum += batch_total.item() * (n_c / bs)
                    if n_u > 0:
                        uncond_loss_sum += batch_total.item() * (n_u / bs)

    vae_model.train()

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_total = total_loss_sum / num_batches
    avg_recon = recon_loss_sum / num_batches
    avg_kl = kl_loss_sum / num_batches
    avg_anchor = anchor_loss_sum / num_batches

    if args.data_type == "unified" and (num_cond + num_uncond) > 0:
        avg_cond = cond_loss_sum / max(num_batches, 1)
        avg_uncond = uncond_loss_sum / max(num_batches, 1)
        return avg_total, avg_recon, avg_kl, avg_anchor, num_cond, num_uncond, avg_cond, avg_uncond

    return avg_total, avg_recon, avg_kl, avg_anchor


def train_vae(
    timestamp,
    args,
    vae_model,
    autoencoder,
    train_dataloader,
    valid_dataloader,
    training_dir,
    optimizer,
    accelerator,
    tokenizer_vocab=None,
    latent_pca: Optional[LatentPCA] = None,
):
    best_val_loss = torch.tensor(float("inf"))
    global_step = 0
    epoch = 0

    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        total_steps = args.EPOCHS * len(train_dataloader)

    print(f"Training for {total_steps} steps")
    running_total_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    running_anchor_loss = 0.0
    num_batches_since_log = 0
    accumulation_steps = 0

    # KL annealing
    kl_anneal_steps = getattr(args, "kl_anneal_steps", 5000)
    kl_beta_max = getattr(args, "kl_beta_max", 0.001)
    demo_only_steps = int(getattr(args, "demo_only_steps", 0) or 0)
    freeze_backbone_for_demo = bool(getattr(args, "freeze_backbone_for_demo", False))
    demo_warmup_active = (not freeze_backbone_for_demo) and demo_only_steps > 0
    backbone_unfrozen = not demo_warmup_active

    train_iter = iter(train_dataloader)
    progress_bar = tqdm(
        total=total_steps, desc="VAE Training", disable=not accelerator.is_main_process
    )

    while global_step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            epoch += 1
            if accelerator.is_main_process:
                print(f"\n--- EPOCH {epoch} ---")

        if global_step == 0 and accelerator.is_main_process:
            print(f"Batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if hasattr(value, "shape"):
                    print(f"  {key}: {value.shape}")

        vae_model.train(True)

        # Extract data from batch
        attrs = _prepare_attrs(batch, args, vae_model)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        target_latents = _extract_target_latents(batch, autoencoder, args, latent_pca)

        latent_scale = getattr(args, "latent_scale", 1.0)

        # --- VAE forward + loss ---
        # KL annealing: linearly increase beta from 0 to kl_beta_max
        if kl_anneal_steps > 0:
            beta_kl = min(1.0, global_step / kl_anneal_steps) * kl_beta_max
        else:
            beta_kl = kl_beta_max

        # Handle conditional dropout for unified training.
        # Unlike DiT (which uses zeroed attrs + force_unconditional flag),
        # we pass None for unconditional samples so the VAE receives no
        # conditioning signal at all (zeroed attrs would still produce a
        # learned bias from wide_fc and embedding lookups).
        attr_embeds_for_model = None
        if args.data_type == "unified":
            is_conditional = batch.get("is_conditional", None)
            if is_conditional is not None:
                dropout_mask = (
                    torch.rand(len(is_conditional), device=is_conditional.device)
                    < args.conditional_dropout
                )
                should_be_conditional = is_conditional & ~dropout_mask
                if should_be_conditional.any():
                    # Keep attrs only for conditional rows; unconditional
                    # rows are truly zeroed and will be masked out in
                    # _embed_attrs (which returns None when all-zero, and
                    # masks per-row when mixed).
                    attr_embeds_for_model = attrs.clone()
                    attr_embeds_for_model[~should_be_conditional] = 0
                else:
                    attr_embeds_for_model = None
            else:
                attr_embeds_for_model = attrs
        else:
            if attrs is not None and attrs.abs().sum() > 0:
                attr_embeds_for_model = attrs

        loss_dict = vae_model.module.compute_loss(
            x=target_latents,
            attr_embeds=attr_embeds_for_model,
            attention_mask=attention_mask,
            beta_kl=beta_kl,
        ) if hasattr(vae_model, 'module') else vae_model.compute_loss(
            x=target_latents,
            attr_embeds=attr_embeds_for_model,
            attention_mask=attention_mask,
            beta_kl=beta_kl,
        )

        vae_loss = loss_dict["total_loss"]
        recon_loss = loss_dict["recon_loss"]
        kl_loss = loss_dict["kl_loss"]

        # Optional anchor loss (decode predicted latents through BART)
        anchor_loss = torch.tensor(0.0, device=target_latents.device)
        if args.use_anchor_loss:
            predicted_x0 = loss_dict["recon"]
            if latent_scale != 1.0:
                predicted_x0 = predicted_x0 * latent_scale
            anchor_loss, anchor_stats, _ = compute_anchor_loss(
                predicted_x0=predicted_x0,
                labels=labels,
                attention_mask=attention_mask,
                autoencoder=autoencoder,
                latent_scale=1.0,  # already scaled above
                latent_pca=latent_pca,
                training_phase=args.training_phase,
                return_per_sample=False,
            )

        total_loss = vae_loss + args.anchor_loss_weight * anchor_loss
        total_loss = total_loss / args.gradient_accumulation_steps

        running_total_loss += total_loss.detach() * args.gradient_accumulation_steps
        running_recon_loss += recon_loss.detach()
        running_kl_loss += kl_loss.detach()
        running_anchor_loss += anchor_loss.detach()
        num_batches_since_log += 1
        accumulation_steps += 1

        accelerator.backward(total_loss)

        if accumulation_steps % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            accumulation_steps = 0
            global_step += 1
            progress_bar.update(1)

            if demo_warmup_active and not backbone_unfrozen and global_step >= demo_only_steps:
                backbone_lr = args.OPTIM_LR * float(getattr(args, "backbone_lr_scale", 0.1))
                _set_backbone_lr(optimizer, backbone_lr)
                backbone_unfrozen = True
                if accelerator.is_main_process:
                    print(
                        f"Demo-only warmup complete at step {global_step}; "
                        f"backbone lr set to {backbone_lr:.6g}"
                    )

            # Logging
            if global_step % args.log_steps == 0 and accelerator.is_main_process:
                n = num_batches_since_log
                avg_total = running_total_loss / n
                avg_recon = running_recon_loss / n
                avg_kl = running_kl_loss / n
                avg_anchor = running_anchor_loss / n

                print(
                    f"Step {global_step} - Total: {avg_total.item():.4f}, "
                    f"Recon: {avg_recon.item():.4f}, "
                    f"KL: {avg_kl.item():.4f} (beta={beta_kl:.6f}), "
                    f"Anchor: {avg_anchor.item():.4f}"
                )

                if args.use_wandb and wandb.run is not None:
                    log_payload = {
                        "train/total_loss": avg_total.item(),
                        "train/recon_loss": avg_recon.item(),
                        "train/kl_loss": avg_kl.item(),
                        "train/kl_beta": beta_kl,
                        "train/anchor_loss": avg_anchor.item(),
                        "train/learning_rate": optimizer.param_groups[-1]["lr"],
                        "train/epoch": epoch,
                        "train/step": global_step,
                    }
                    for i, group in enumerate(optimizer.param_groups):
                        group_name = group.get("name", f"group_{i}")
                        log_payload[f"train/lr_{group_name}"] = group["lr"]
                    wandb.log(log_payload)

                running_total_loss = 0.0
                running_recon_loss = 0.0
                running_kl_loss = 0.0
                running_anchor_loss = 0.0
                num_batches_since_log = 0

            # Validation
            if (
                args.enable_validation
                and valid_dataloader is not None
                and global_step % args.eval_steps == 0
                and accelerator.is_main_process
            ):
                eval_samples = getattr(args, "eval_samples", None)
                max_val_batches = None
                if eval_samples is not None and eval_samples > 0:
                    val_bs = getattr(args, "BATCH_SIZE", 64)
                    max_val_batches = max(1, eval_samples // val_bs)

                val_results = validate_vae(
                    vae_model=vae_model,
                    autoencoder=autoencoder,
                    valid_dataloader=valid_dataloader,
                    args=args,
                    accelerator=accelerator,
                    beta_kl=beta_kl,
                    latent_pca=latent_pca,
                    max_batches=max_val_batches,
                )

                if args.data_type == "unified" and len(val_results) == 8:
                    avg_val_total, avg_val_recon, avg_val_kl, avg_val_anchor, n_cond, n_uncond, avg_cond, avg_uncond = val_results
                    print(
                        f"Validation Step {global_step} - Total: {avg_val_total:.4f}, "
                        f"Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}, "
                        f"Anchor: {avg_val_anchor:.4f}"
                    )
                    print(
                        f"  Unified - Conditional: {n_cond} samples (loss: {avg_cond:.4f}), "
                        f"Unconditional: {n_uncond} samples (loss: {avg_uncond:.4f})"
                    )
                    if args.use_wandb and wandb.run is not None:
                        wandb.log({
                            "val/total_loss": avg_val_total,
                            "val/recon_loss": avg_val_recon,
                            "val/kl_loss": avg_val_kl,
                            "val/anchor_loss": avg_val_anchor,
                            "val/conditional_loss": avg_cond,
                            "val/unconditional_loss": avg_uncond,
                            "val/conditional_samples": n_cond,
                            "val/unconditional_samples": n_uncond,
                            "val/step": global_step,
                        })
                else:
                    avg_val_total, avg_val_recon, avg_val_kl, avg_val_anchor = val_results[:4]
                    print(
                        f"Validation Step {global_step} - Total: {avg_val_total:.4f}, "
                        f"Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}, "
                        f"Anchor: {avg_val_anchor:.4f}"
                    )
                    if args.use_wandb and wandb.run is not None:
                        wandb.log({
                            "val/total_loss": avg_val_total,
                            "val/recon_loss": avg_val_recon,
                            "val/kl_loss": avg_val_kl,
                            "val/anchor_loss": avg_val_anchor,
                            "val/step": global_step,
                        })

                # Save best model
                if avg_val_total < best_val_loss:
                    best_val_loss = torch.tensor(avg_val_total)
                    with training_dir():
                        state = vae_model.module.state_dict() if hasattr(vae_model, 'module') else vae_model.state_dict()
                        torch.save(state, "vae_best_val.pt")
                        print(f"Saved best validation model with val loss: {best_val_loss.item():.4f}")

            # Save checkpoint
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                with training_dir("state_dicts"):
                    model_path = f"vae_step_{global_step}.pt"
                    state = vae_model.module.state_dict() if hasattr(vae_model, 'module') else vae_model.state_dict()
                    torch.save(state, model_path)
                    print(f"Saved VAE checkpoint: {model_path}")

                # Also save full checkpoint for resume
                with training_dir():
                    torch.save(
                        {
                            "model": state,
                            "optimizer": optimizer.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch,
                            "best_val_loss": best_val_loss.item(),
                        },
                        "vae_checkpoint_latest.pt",
                    )

            # Check if done
            if global_step >= total_steps:
                break

    progress_bar.close()

    # Final validation
    if args.enable_validation and valid_dataloader is not None and accelerator.is_main_process:
        print("\n--- Final Validation ---")
        final_beta_kl = kl_beta_max  # use final beta for last validation
        val_results = validate_vae(
            vae_model=vae_model,
            autoencoder=autoencoder,
            valid_dataloader=valid_dataloader,
            args=args,
            accelerator=accelerator,
            beta_kl=final_beta_kl,
            latent_pca=latent_pca,
        )
        avg_val_total = val_results[0]
        avg_val_recon = val_results[1]
        avg_val_kl = val_results[2]
        avg_val_anchor = val_results[3]
        print(
            f"Final Validation - Total: {avg_val_total:.4f}, "
            f"Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}, "
            f"Anchor: {avg_val_anchor:.4f}"
        )
        if avg_val_total < best_val_loss:
            best_val_loss = torch.tensor(avg_val_total)

    # Save final model
    if accelerator.is_main_process:
        with training_dir():
            state = vae_model.module.state_dict() if hasattr(vae_model, 'module') else vae_model.state_dict()
            torch.save(state, "vae_final.pt")
            torch.save(
                {
                    "model": state,
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss.item(),
                },
                "vae_checkpoint_final.pt",
            )
            print(f"Saved final VAE model (best val loss: {best_val_loss.item():.4f})")

    if args.use_wandb and accelerator.is_main_process and wandb.run is not None:
        wandb.finish()


def main(args):
    if args.seed is not None:
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

    print(f"=== VAE Training Configuration ===")
    print(f"Training phase: {args.training_phase}")
    print(f"Autoencoder path: {args.autoencoder_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Data type: {args.data_type}")
    print(f"Anchor loss: {'enabled' if args.use_anchor_loss else 'disabled'}")
    print(f"KL beta max: {args.kl_beta_max}")
    print(f"KL anneal steps: {args.kl_anneal_steps}")
    if args.vae_init_checkpoint:
        print(f"VAE init checkpoint: {args.vae_init_checkpoint}")
    if args.freeze_backbone_for_demo or args.demo_only_steps > 0:
        mode = "entire run" if args.freeze_backbone_for_demo else f"{args.demo_only_steps} steps"
        print(
            f"Demo branch warm-start tuning: backbone lr=0 for {mode}, "
            f"backbone_lr_scale={args.backbone_lr_scale}"
        )
    print(f"Validation: {'enabled' if args.enable_validation else 'disabled'}")
    if args.enable_validation:
        print(f"  Eval every {args.eval_steps} steps, samples: {args.eval_samples}")
    print(f"=== End Configuration ===\n")

    latent_pca = None

    # Device setup
    if args.force_cpu:
        print("Forcing CPU usage.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.is_available = lambda: False
    elif torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    args.is_main_process = accelerator.is_main_process
    device = accelerator.device
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Wandb
    if args.use_wandb and accelerator.is_main_process:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"vae_{args.training_phase}_{args.data_type}_{timestamp}",
            config=vars(args),
        )
        print(f"Wandb initialized: {wandb.run.name}")

    # Training directory
    dir_path = f"./training_vae_{timestamp}"
    training_dir = create_directory(dir_path)

    # Load autoencoder config first — needed to validate sequence length
    latent_model_path = args.autoencoder_path
    if not os.path.exists(latent_model_path):
        raise FileNotFoundError(f"Autoencoder path does not exist: {latent_model_path}")
    ae_config = BartConfig.from_json_file(os.path.join(latent_model_path, "config.json"))

    # Infer sequence length from VAE config, but clamp to BART's max_position_embeddings
    # to prevent position embedding out-of-bounds errors.
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        vae_cfg = config.get("VAE", {})
        if "image_size" in vae_cfg:
            vae_seq_len = int(vae_cfg["image_size"])
            # Hugging Face BART adds its positional offset internally; the
            # public max_position_embeddings value is the usable sequence
            # length. Do not subtract 2 here, or the dataset target length will
            # no longer match the VAE decoder/checkpoint image_size.
            bart_max_len = getattr(ae_config, "max_position_embeddings", 1024)
            if vae_seq_len > bart_max_len:
                print(
                    f"WARNING: VAE image_size={vae_seq_len} exceeds BART "
                    f"max_position_embeddings={bart_max_len}. "
                    f"Clamping sequence_length to {bart_max_len}."
                )
                args.sequence_length = bart_max_len
            else:
                args.sequence_length = vae_seq_len
            print(f"Using sequence_length={args.sequence_length} "
                  f"(VAE image_size={vae_seq_len}, BART max_pos={getattr(ae_config, 'max_position_embeddings', '?')})")
    except Exception as e:
        print(f"Warning: could not infer sequence length from config ({e})")

    # Load dataset
    print(f"Loading dataset for {args.training_phase} training...")
    train_dataloader, valid_dataloader, _, tokenizer_vocab = trajectory_dataset(
        args, data_dir=args.data_dir, data_type=args.data_type
    )

    # Parse VAE config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    vae_params = {}
    if "VAE" in config:
        vae_params.update(config["VAE"])

    # Length conditioning
    if getattr(args, "enable_length_condition", False):
        length_vocab_size = int(getattr(args, "length_vocab_size", 513))
        vae_params["use_length_condition"] = True
        vae_params["length_vocab_size"] = length_vocab_size
        print(f"Length conditioning enabled (vocab size: {length_vocab_size})")
    else:
        vae_params["use_length_condition"] = False

    # Demo conditioning from config
    if "use_demo_condition" not in vae_params:
        vae_params["use_demo_condition"] = getattr(args, "use_demo_condition", False)
    if "num_age_bins" not in vae_params:
        vae_params["num_age_bins"] = getattr(args, "num_age_bins", 0)
    if "num_genders" not in vae_params:
        vae_params["num_genders"] = getattr(args, "num_genders", 0)

    # Numerical stability options for from-scratch VAE runs. Defaults are off so
    # existing checkpoints/experiments keep the previous behavior unless opted in.
    if getattr(args, "clamp_logvar", False):
        vae_params["clamp_logvar"] = True
        vae_params["logvar_min"] = args.logvar_min
        vae_params["logvar_max"] = args.logvar_max
        print(
            f"VAE log_var clamping enabled: "
            f"[{args.logvar_min}, {args.logvar_max}]"
        )
    if getattr(args, "init_logvar_bias", None) is not None:
        vae_params["init_logvar_bias"] = args.init_logvar_bias
        print(f"VAE fc_log_var bias initialized to {args.init_logvar_bias}")

    # PCA
    if args.latent_pca_path:
        if args.training_phase != "phase1":
            raise ValueError("--latent_pca_path currently requires training_phase='phase1'")
        print(f"Loading latent PCA from {args.latent_pca_path}")
        latent_pca = LatentPCA(args.latent_pca_path, device)
        vae_params["in_channels"] = latent_pca.component_dim
        print(f"Overriding VAE in_channels to PCA dim: {latent_pca.component_dim}")

    # Create VAE model
    print(f"Creating VAE with parameters: {vae_params}")
    vae_model = TrajectoryVAE(**vae_params).to(device)
    _load_vae_init_checkpoint(vae_model, args.vae_init_checkpoint, device)

    model_size_MB = get_model_size(vae_model)
    print(f"VAE model size: {model_size_MB:.1f} MB")

    # Load autoencoder (BART) — frozen, used only for target latent extraction + anchor loss
    if args.training_phase == "phase1":
        autoencoder = BartForConditionalGeneration.from_pretrained(latent_model_path).to(device)
    else:
        autoencoder = BARTLatentCompression.from_pretrained(latent_model_path).to(device)
        if getattr(args, "no_compression", False):
            autoencoder.no_compression = True
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = _build_vae_optimizer(vae_model, args)

    # Accelerate
    vae_model, optimizer, train_dataloader = accelerator.prepare(
        vae_model, optimizer, train_dataloader
    )
    if valid_dataloader is not None:
        valid_dataloader = accelerator.prepare(valid_dataloader)

    # Save config
    save_training_info(args, timestamp, [vae_params], {}, model_size_MB, training_dir)

    # Train
    train_vae(
        timestamp=timestamp,
        args=args,
        vae_model=vae_model,
        autoencoder=autoencoder,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        training_dir=training_dir,
        optimizer=optimizer,
        accelerator=accelerator,
        tokenizer_vocab=tokenizer_vocab,
        latent_pca=latent_pca,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Train VAE backbone for ATLAS")

    # Core
    parser.add_argument("--config", "-config", type=str, required=True, help="VAE config YAML")
    parser.add_argument("--autoencoder_path", type=str, required=True, help="Path to pretrained BART autoencoder")
    parser.add_argument("--data_dir", type=str, default="split_data_new", help="Data directory")
    parser.add_argument("--data_type", type=str, choices=["controlled", "uncontrolled", "unified"], default="unified")
    parser.add_argument("--training_phase", type=str, choices=["phase1", "phase2"], default="phase1")
    parser.add_argument("--ablation_mode", type=str, choices=["coords_only", "subcat_only", "both", "neither", "pure"], default="both")
    parser.add_argument("--no_compression", action="store_true", default=False)
    parser.add_argument("--force_full_attention_mask", action="store_true", default=False)

    # Training
    parser.add_argument("-b", "--BATCH_SIZE", type=int, default=64, help="Batch size")
    parser.add_argument("-e", "--EPOCHS", type=int, default=100)
    parser.add_argument("-lr", "--OPTIM_LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--conditional_dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)

    # VAE-specific
    parser.add_argument("--kl_beta_max", type=float, default=0.001, help="Max KL weight (beta-VAE)")
    parser.add_argument("--kl_anneal_steps", type=int, default=5000, help="Steps to linearly anneal KL beta from 0 to max")
    parser.add_argument(
        "--clamp_logvar",
        action="store_true",
        default=False,
        help="Clamp VAE log_var before sampling and KL computation for numerical stability.",
    )
    parser.add_argument(
        "--logvar_min",
        type=float,
        default=-20.0,
        help="Minimum log_var value when --clamp_logvar is enabled.",
    )
    parser.add_argument(
        "--logvar_max",
        type=float,
        default=10.0,
        help="Maximum log_var value when --clamp_logvar is enabled.",
    )
    parser.add_argument(
        "--init_logvar_bias",
        type=float,
        default=None,
        help="Optional initial bias for the VAE log_var head, e.g. -5.0 for a conservative initial posterior std.",
    )

    # Anchor loss
    parser.add_argument("--use_anchor_loss", action="store_true", default=False)
    parser.add_argument("--anchor_loss_weight", type=float, default=1.0)

    # Latent
    parser.add_argument("--latent_scale", type=float, default=1.0)
    parser.add_argument("--latent_pca_path", type=str, default=None)

    # Length conditioning
    parser.add_argument("--enable_length_condition", action="store_true", default=False)
    parser.add_argument("--length_vocab_size", type=int, default=513)

    # Demo conditioning
    parser.add_argument("--use_demo_condition", action="store_true", default=False)
    parser.add_argument("--num_age_bins", type=int, default=0)
    parser.add_argument("--num_genders", type=int, default=0)
    parser.add_argument(
        "--vae_init_checkpoint",
        type=str,
        default=None,
        help="Optional VAE checkpoint to warm-start from, e.g. a home/work-only baseline.",
    )
    parser.add_argument(
        "--freeze_backbone_for_demo",
        action="store_true",
        default=False,
        help="With demo conditioning, keep non-demo VAE parameter-group lr at 0 for the full run.",
    )
    parser.add_argument(
        "--demo_only_steps",
        type=int,
        default=0,
        help="With demo conditioning, train demo branch only for this many optimizer steps before unfreezing backbone.",
    )
    parser.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.1,
        help="Backbone LR multiplier after --demo_only_steps warmup.",
    )

    # Logging & checkpointing
    parser.add_argument("--log_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vae-trajectory-generation")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)

    # Misc
    parser.add_argument("--timestamp", "-ts", type=str, default=None)
    parser.add_argument("--force_cpu", "-force_cpu", action="store_true", default=False)
    parser.add_argument("-n", "--NUM_WORKERS", type=int, default=8)
    parser.add_argument("-p", "--PARAMETERS", type=str, default=None)
    parser.add_argument("-rd", "--RESTART_DIRECTORY", type=str, default=None)

    # Validation (minimal — can be extended)
    parser.add_argument("--enable_validation", action="store_true", default=False)
    parser.add_argument("--eval_samples", type=int, default=256)

    args = parser.parse_args()
    main(args)
