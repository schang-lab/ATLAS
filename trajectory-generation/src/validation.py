import os
from datetime import datetime
import torch.utils.data
from torch import optim
from argparse import ArgumentParser
import yaml

import numpy as np

import argparse
import json
import math

from tqdm import tqdm
from tqdm import trange
import numpy as np
from typing import Optional

from accelerate import Accelerator, DistributedDataParallelKwargs

# Add wandb import
import wandb

from src.dit import DiT
from src.diffusion_model import GaussianDiffusion, ModelPredictions
from src.training import (trajectory_dataset,
                          create_directory,
                          get_model_params,
                          get_model_size,
                          save_training_info,
                          get_default_args,
                          load_restart_training_parameters)
from src.losses import (_masked_latent_similarity, 
                        _token_accuracies_from_generate,
                        _sequence_lengths_excluding_specials,
                        _collect_special_token_ids,
                        compute_anchor_loss)

from src.checkpoint_utils import (save_training_checkpoint,
                                 load_training_checkpoint)



from auto_encoder.traj_compressed_ae import BARTLatentCompression
from src.latent_pca import LatentPCA
from src.helpers import normalize_prediction_type

from transformers import (
    BertTokenizerFast,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
import pandas as pd

def _predict_noise_with_guidance(
    model: torch.nn.Module,
    noise_scheduler: GaussianDiffusion,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    attrs: Optional[torch.Tensor],
    prediction_type: str,
    guidance_scale: Optional[float] = None,
    conditional_mask: Optional[torch.Tensor] = None
) -> ModelPredictions:
    """Predict denoiser outputs with optional classifier-free guidance and convert to both views."""

    pred_type = normalize_prediction_type(prediction_type)

    if attrs is None or guidance_scale is None or abs(guidance_scale - 1.0) < 1e-6:
        model_output = model(x=x_t, t=timesteps, attr_embeds=attrs)
    else:
        attrs = attrs.to(device=x_t.device, dtype=x_t.dtype)

        if conditional_mask is not None:
            conditional_mask = conditional_mask.to(device=attrs.device, dtype=torch.bool)
        else:
            conditional_mask = attrs.abs().sum(dim=1) > 0

        unconditional_attrs = attrs.clone()
        unconditional_attrs[conditional_mask] = 0

        model_output = GaussianDiffusion.classifier_free_guidance(
            denoiser=model,
            x_t=x_t,
            t=timesteps,
            conditional_attrs=attrs,
            guidance_scale=guidance_scale,
            unconditional_attrs=unconditional_attrs,
            conditional_mask=conditional_mask
        )

    return noise_scheduler.model_predictions(
        model_output=model_output,
        x_t=x_t,
        t=timesteps,
        prediction_type=pred_type
    )

def validate_model(dit_model, noise_scheduler, autoencoder, valid_dataloader, args, accelerator=None, latent_pca: Optional[LatentPCA] = None):
    """
    Evaluate the model on validation set
    """
    dit_model.eval()

    use_anchor_loss = getattr(args, 'use_anchor_loss', False)
    anchor_loss_weight = getattr(args, 'anchor_loss_weight', 1.0)

    total_val_loss = 0.0
    total_diffusion_loss = 0.0
    total_anchor_loss = 0.0
    num_val_batches = 0
    
    # Track separate losses for conditional vs unconditional samples in unified training
    conditional_loss = 0.0
    unconditional_loss = 0.0
    num_conditional = 0
    num_unconditional = 0
    
    # Validation diagnostics accumulators
    val_diag_enabled = getattr(args, 'diag_decode_val', False)
    eval_limit = int(getattr(args, 'eval_samples', 0) or 0)
    eval_limit = max(eval_limit, 0)
    processed_samples = 0

    target_total = int(getattr(args, 'diag_decode_total', 0) or 0)
    if eval_limit > 0 and target_total > 0:
        target_total = min(target_total, eval_limit)
    covered_total = 0
    batches_used = 0
    sum_cos, sum_l2, count_pos = 0.0, 0.0, 0
    sum_accH_len, sum_accH_str = 0.0, 0.0
    sum_accX_len, sum_accX_str = 0.0, 0.0
    sum_unkH_hits, sum_unkX_hits = 0, 0
    sum_unk_token_den = 0
    sum_t_mean_weighted = 0.0
    t_min_global = float('inf')
    t_max_global = float('-inf')
    
    val_diag_batches = 0
    val_diag_total = target_total if target_total > 0 else len(valid_dataloader)

    cfg_auto = getattr(autoencoder, 'config', None)
    unk_token_id = getattr(cfg_auto, 'unk_token_id', None) if cfg_auto is not None else None

    prediction_type = normalize_prediction_type(getattr(args, 'prediction_type', 'epsilon'))
    timestep_sampling = getattr(args, 'timestep_sampling', 'logsnr')
    timestep_sampling = timestep_sampling.lower().replace('-', '_')

    loader_batch_size = getattr(valid_dataloader, 'batch_size', None)
    total_batches_estimate = None
    if eval_limit > 0 and loader_batch_size:
        total_batches_estimate = math.ceil(eval_limit / loader_batch_size)
    elif hasattr(valid_dataloader, '__len__'):
        try:
            total_batches_estimate = len(valid_dataloader)
        except TypeError:
            total_batches_estimate = None

    main_pbar = None
    if getattr(args, 'verbose', False):
        try:
            main_pbar = tqdm(total=total_batches_estimate, desc="Validation", leave=False)
        except Exception:
            main_pbar = None
    should_show_val_pbar = val_diag_enabled and getattr(args, 'is_main_process', True)
    val_diag_pbar = tqdm(total=val_diag_total, desc="ValDiag batches", leave=False) if should_show_val_pbar else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dataloader):
            if eval_limit > 0 and processed_samples >= eval_limit:
                break
            attrs = batch['attrs']
            length_id = batch.get('length_id', None)
            is_conditional = batch.get('is_conditional', None)
            attention_mask = batch['attention_mask']
            input_ids = batch['input_ids']
            labels = batch['labels']

            original_batch_size = input_ids.size(0)
            if eval_limit > 0:
                remaining = eval_limit - processed_samples
                if remaining <= 0:
                    break
                if remaining < original_batch_size:
                    slice_obj = slice(0, remaining)
                    attrs = attrs[slice_obj]
                    if length_id is not None:
                        length_id = length_id[slice_obj]
                    attention_mask = attention_mask[slice_obj]
                    input_ids = input_ids[slice_obj]
                    labels = labels[slice_obj]
                    if is_conditional is not None:
                        is_conditional = is_conditional[slice_obj]
                    for key in ('lat', 'lon', 'sub_categories', 'top_categories', 'origin_dest'):
                        value = batch.get(key, None)
                        if isinstance(value, torch.Tensor):
                            batch[key] = value[slice_obj]

            
            # Get target latents from autoencoder
            latent_scale = getattr(args, "latent_scale", 1.0)

            if args.training_phase == "phase1":
                encoder_outputs = autoencoder.get_encoder()(input_ids=input_ids,
                                                            attention_mask=attention_mask)
                ae_latents = encoder_outputs.last_hidden_state
            else:
                encoder_outputs = autoencoder.get_encoder()(input_ids=input_ids,
                                                            attention_mask=attention_mask)

                segment_coords = None
                sub_categories = None

                if args.ablation_mode in ["coords_only", "both"]:
                    if 'lat' in batch and 'lon' in batch:
                        segment_coords = torch.stack([batch['lat'], batch['lon']], dim=-1)

                if args.ablation_mode in ["subcat_only", "both"]:
                    sub_categories = batch.get('sub_categories', None)

                if hasattr(autoencoder, 'no_compression') and autoencoder.no_compression:
                    enhanced_outputs = autoencoder._add_features_no_compression(
                        encoder_outputs, attention_mask, segment_coords, sub_categories
                    )
                    ae_latents = enhanced_outputs['last_hidden_state']
                else:
                    ae_latents = autoencoder.get_diffusion_latent(encoder_outputs=encoder_outputs,
                                                                  attention_mask=attention_mask,
                                                                  segment_coords=segment_coords,
                                                                  sub_categories=sub_categories)

            target_latents = ae_latents
            if latent_pca is not None:
                target_latents = latent_pca.project(target_latents)

            if latent_scale != 1.0:
                target_latents = target_latents / latent_scale

            # Diffusion validation step
            batch_size = target_latents.shape[0]
            
            # Sample random timesteps
            t = noise_scheduler.sample_timesteps(batch_size, device=target_latents.device, method=timestep_sampling)
            
            # Add noise to target latents
            noise = torch.randn_like(target_latents)
            noisy_latents = noise_scheduler.q_sample(target_latents, t, noise)
            
            conditional_mask_val = None

            # Handle conditional vs unconditional validation for unified training
            if length_id is not None and getattr(args, 'enable_length_condition', False):
                length_tensor = length_id.float().unsqueeze(-1).to(attrs.device)
                attrs = torch.cat([attrs, length_tensor], dim=1)

            if args.data_type == "unified" and is_conditional is not None:
                if is_conditional.any():
                    attr_embeds_for_model = attrs.clone()
                    attr_embeds_for_model[~is_conditional] = 0
                    conditional_mask_val = is_conditional
                else:
                    attr_embeds_for_model = None
            else:
                if attrs.abs().sum() == 0:
                    attr_embeds_for_model = None
                else:
                    attr_embeds_for_model = attrs
                    conditional_mask_val = attr_embeds_for_model.abs().sum(dim=1) > 0

            # Predict noise with optional CFG
            validation_guidance_scale = getattr(args, "validation_guidance_scale", 1.0)
            predictions = _predict_noise_with_guidance(
                model=dit_model,
                noise_scheduler=noise_scheduler,
                x_t=noisy_latents,
                timesteps=t,
                attrs=attr_embeds_for_model,
                prediction_type=prediction_type,
                guidance_scale=validation_guidance_scale,
                conditional_mask=conditional_mask_val
            )
            predicted_noise = predictions.noise
            predicted_x0 = predictions.x_start
            predicted_v = predictions.v
            
            
            # Compute diffusion loss (per-sample for unified training tracking)
            if prediction_type == "epsilon":
                diffusion_pred = predicted_noise
                diffusion_target = noise
            elif prediction_type == "v":
                diffusion_pred = predicted_v
                diffusion_target = noise_scheduler.predict_v_from_start(
                    x_t=noisy_latents,
                    t=t,
                    x_start=target_latents
                ).detach()
            else:
                diffusion_pred = predicted_x0
                diffusion_target = target_latents

            mse_per_token = F.mse_loss(
                diffusion_pred,
                diffusion_target,
                reduction='none'
            )
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(mse_per_token.dtype)
                masked_mse = mse_per_token * mask
                if args.data_type == "unified" and is_conditional is not None:
                    denom_per_sample = mask.sum(dim=[1, 2]).clamp_min(1.0)
                    diffusion_loss_per_sample = masked_mse.sum(dim=[1, 2]) / denom_per_sample
                    diffusion_loss = diffusion_loss_per_sample.mean()
                else:
                    denom = mask.sum().clamp_min(1.0)
                    diffusion_loss = masked_mse.sum() / denom
            else:
                if args.data_type == "unified" and is_conditional is not None:
                    diffusion_loss_per_sample = mse_per_token.mean(dim=[1, 2])
                    diffusion_loss = diffusion_loss_per_sample.mean()
                else:
                    diffusion_loss = mse_per_token.mean()
            
            
            anchor_loss = torch.tensor(0.0, device=predicted_x0.device)
            anchor_per_sample = None
            if use_anchor_loss:
                anchor_loss, _, anchor_per_sample = compute_anchor_loss(
                    predicted_x0=predicted_x0,
                    labels=labels,
                    attention_mask=attention_mask,
                    autoencoder=autoencoder,
                    latent_scale=latent_scale,
                    latent_pca=latent_pca,
                    training_phase=args.training_phase,
                    return_per_sample=(args.data_type == "unified" and is_conditional is not None)
                )
            
            # Validation diagnostics over multiple batches if requested
            if val_diag_enabled:
                # Decide if we should process this batch for diagnostics
                process_this = False
                if target_total == 0:
                    process_this = (batches_used == 0)
                else:
                    process_this = (covered_total < target_total)
                if process_this:
                    if val_diag_pbar is not None:
                        val_diag_pbar.update(1)
                        val_diag_batches += 1
                    try:
                        # Progress bar for validation
                        val_pbar = None
                        if target_total > 0 and batch_idx == 0:
                            try:
                                val_pbar = tqdm(total=target_total, desc="Diag decode (val)", leave=False)
                            except Exception:
                                val_pbar = None
                        K_cap = getattr(args, 'diag_decode_batch', 8)
                        remaining = max(0, target_total - covered_total) if target_total > 0 else K_cap
                        K = min(K_cap, batch_size, remaining)
                        if K > 0:
                            pred_x0 = predicted_x0
                            sim = _masked_latent_similarity(target_latents[:K], pred_x0[:K], attention_mask[:K])
                            sum_cos += sim['cosine'] * sim['count']
                            sum_l2 += sim['l2'] * sim['count']
                            count_pos += sim['count']
                            re_enc_H = BaseModelOutput(last_hidden_state=ae_latents[:K].clone())
                            gen_H = autoencoder.generate(
                                encoder_outputs=re_enc_H,
                                attention_mask=attention_mask[:K],
                                max_length=attention_mask.size(1),
                                min_length=4,
                                num_beams=getattr(args, 'diag_num_beams', 4),
                                length_penalty=1.2,
                                repetition_penalty=1.1,
                                no_repeat_ngram_size=0,
                                do_sample=False,
                            )
                            accs_H = _token_accuracies_from_generate(gen_H, labels[:K], attention_mask[:K], autoencoder)
                            sum_accH_len += accs_H['lenient'] * K
                            sum_accH_str += accs_H['strict'] * K
                            px0_scaled = pred_x0[:K] * latent_scale if latent_scale != 1.0 else pred_x0[:K]
                            if latent_pca is not None:
                                px0_decode = latent_pca.unproject(px0_scaled)
                            else:
                                px0_decode = px0_scaled.clone()
                            re_enc_X0 = BaseModelOutput(last_hidden_state=px0_decode)
                            gen_X0 = autoencoder.generate(
                                encoder_outputs=re_enc_X0,
                                attention_mask=attention_mask[:K],
                                max_length=attention_mask.size(1),
                                min_length=4,
                                num_beams=getattr(args, 'diag_num_beams', 4),
                                length_penalty=1.2,
                                repetition_penalty=1.1,
                                no_repeat_ngram_size=0,
                                do_sample=False,
                            )
                            accs_X0 = _token_accuracies_from_generate(gen_X0, labels[:K], attention_mask[:K], autoencoder)
                            sum_accX_len += accs_X0['lenient'] * K
                            sum_accX_str += accs_X0['strict'] * K
                            if unk_token_id is not None:
                                mask_flat = attention_mask[:K].bool().view(-1)
                                valid_tokens = int(mask_flat.sum().item())
                                if valid_tokens > 0:
                                    sum_unk_token_den += valid_tokens
                                    gen_H_flat = gen_H[:, :attention_mask.size(1)].reshape(-1)
                                    gen_X_flat = gen_X0[:, :attention_mask.size(1)].reshape(-1)
                                    hits_H = (gen_H_flat[mask_flat] == int(unk_token_id)).sum().item()
                                    hits_X = (gen_X_flat[mask_flat] == int(unk_token_id)).sum().item()
                                    sum_unkH_hits += int(hits_H)
                                    sum_unkX_hits += int(hits_X)
                            t_subset = t[:K].float()
                            sum_t_mean_weighted += t_subset.mean().item() * K
                            t_min_global = min(t_min_global, t_subset.min().item())
                            t_max_global = max(t_max_global, t_subset.max().item())
                            covered_total += K
                            batches_used += 1

                            # Optional: short DDIM reconstruction diagnostic (more faithful to inference)
                            try:
                                ddim_steps = int(getattr(args, 'diag_ddim_steps', 0) or 0)
                            except Exception:
                                ddim_steps = 0
                            if ddim_steps > 0:
                                # Build per-sample step schedule from current t down to 0 in `ddim_steps` segments
                                # Start from the already computed noisy_latents for fairness
                                x_ddim = noisy_latents[:K].clone() #B × L × D
                                t_orig = t[:K]
                                # Prepare conditional inputs for CFG
                                if args.data_type == "unified" and is_conditional is not None:
                                    # reuse attr_embeds_for_model built earlier in this batch
                                    cond_attrs = attrs.clone()
                                    cond_attrs[~is_conditional] = 0
                                    cond_mask = is_conditional[:K]
                                else:
                                    cond_attrs = attrs if attrs.abs().sum() != 0 else None
                                    cond_mask = (cond_attrs.abs().sum(dim=1) > 0) if cond_attrs is not None else None
                                guidance = getattr(args, 'validation_guidance_scale', 1.0)

                                for s in range(ddim_steps):
                                    # current and next timesteps per sample (vectorized, monotonically decreasing)
                                    t_curr = torch.floor(t_orig.float() * (ddim_steps - s) / max(1, ddim_steps)).long().to(x_ddim.device)
                                    t_curr = torch.clamp(t_curr, min=0)
                                    t_next = torch.floor(t_orig.float() * (ddim_steps - s - 1) / max(1, ddim_steps)).long().to(x_ddim.device)
                                    t_next = torch.clamp(t_next, min=0)

                                    attrs_slice = cond_attrs[:K] if cond_attrs is not None else None
                                    cfg_scale = guidance if attrs_slice is not None else None

                                    ddim_preds = _predict_noise_with_guidance(
                                        model=dit_model,
                                        noise_scheduler=noise_scheduler,
                                        x_t=x_ddim,
                                        timesteps=t_curr,
                                        attrs=attrs_slice,
                                        prediction_type=prediction_type,
                                        guidance_scale=cfg_scale,
                                        conditional_mask=cond_mask
                                    )

                                    eps_pred = ddim_preds.noise
                                    x0_pred = ddim_preds.x_start

                                    if s < ddim_steps - 1:
                                        x_ddim = noise_scheduler.ddim_sample(x_ddim, t_curr, t_next, eps_pred, eta=0.0)
                                    else:
                                        x_ddim = x0_pred

                                # Compare latents and decode for accuracies
                                max_available = labels.size(1)
                                latent_len = x_ddim.size(1)
                                mask_len = attention_mask.size(1)
                                seq_len = min(latent_len, mask_len, max_available)
                                if batch_idx == 0:
                                    print(
                                        f"[ValDiag-DDIM] seq_len={seq_len} | latent_len={latent_len} | mask_len={mask_len} | label_len={max_available}"
                                    )
                                no_compression_mode = getattr(autoencoder, 'no_compression', False)
                                target_mask = attention_mask[:K, :seq_len].bool()

                                target_lat_trim = target_latents[:K, :seq_len]
                                x_ddim_trim = x_ddim[:K, :seq_len]

                                x_ddim_decode = x_ddim_trim * latent_scale if latent_scale != 1.0 else x_ddim_trim
                                if latent_pca is not None:
                                    x_ddim_decode = latent_pca.unproject(x_ddim_decode)

                                sim_ddim = _masked_latent_similarity(target_lat_trim, x_ddim_trim, target_mask)
                                # Accumulate under separate keys by piggybacking on existing accumulators via wandb logging below
                                # We store them temporarily on the args object to aggregate once at the end
                                if not hasattr(args, '_val_ddim_metrics'):
                                    args._val_ddim_metrics = {
                                        'sum_cos': 0.0, 'sum_l2': 0.0, 'count_pos': 0,
                                        'sum_acc_len': 0.0, 'sum_acc_str': 0.0, 'covered': 0,
                                        'sum_len_target': 0.0, 'sum_len_generated': 0.0,
                                        'sum_len_abs_diff': 0.0, 'len_counts': 0
                                    }
                                    if getattr(args, 'is_main_process', True):
                                        args._val_ddim_metrics['cos_vals'] = []
                                if not hasattr(args, '_val_ddim_special_ids'):
                                    args._val_ddim_special_ids = _collect_special_token_ids(autoencoder)
                                args._val_ddim_metrics['sum_cos'] += sim_ddim['cosine'] * sim_ddim['count']
                                args._val_ddim_metrics['sum_l2'] += sim_ddim['l2'] * sim_ddim['count']
                                args._val_ddim_metrics['count_pos'] += sim_ddim['count']

                                # Collect per-position cosine distribution for diagnostics
                                if getattr(args, 'is_main_process', True):
                                    valid_mask_flat = target_mask.view(-1)
                                    if valid_mask_flat.any():
                                        h_norm = F.normalize(target_lat_trim, dim=-1)
                                        x_norm = F.normalize(x_ddim_trim, dim=-1)
                                        cos_pos = (h_norm * x_norm).sum(dim=-1).view(-1)[valid_mask_flat].detach().cpu()
                                        cap = 200000
                                        cos_store = args._val_ddim_metrics.setdefault('cos_vals', [])
                                        remaining = cap - len(cos_store)
                                        if remaining > 0 and cos_pos.numel() > 0:
                                            take = min(int(remaining), int(cos_pos.numel()))
                                            cos_store.extend(cos_pos[:take].tolist())

                                re_enc_DDIM = BaseModelOutput(last_hidden_state=x_ddim_decode.clone())

                                # Decode DDIM latents with progress feedback
                                decode_pbar = getattr(args, '_val_ddim_decode_pbar', None)
                                if getattr(args, 'diag_decode_val', False) and decode_pbar is None and getattr(args, 'is_main_process', True):
                                    try:
                                        total = target_total if target_total > 0 else None
                                        decode_pbar = tqdm(total=total, desc="Diag DDIM decode", leave=False)
                                    except Exception:
                                        decode_pbar = None
                                    else:
                                        args._val_ddim_decode_pbar = decode_pbar

                                gen_DDIM = autoencoder.generate(
                                    encoder_outputs=re_enc_DDIM,
                                    attention_mask=target_mask,
                                    max_length=seq_len,
                                    min_length=4,
                                    num_beams=getattr(args, 'diag_num_beams', 4),
                                    length_penalty=1.2,
                                    repetition_penalty=1.1,
                                    no_repeat_ngram_size=0,
                                    do_sample=False,
                                )
                                if decode_pbar is not None and getattr(args, 'is_main_process', True):
                                    decode_pbar.update(K)

                                labels_trim = labels[:K, :seq_len]
                                special_ids = getattr(args, '_val_ddim_special_ids', set())
                                target_lengths = _sequence_lengths_excluding_specials(
                                    labels_trim,
                                    special_ids=special_ids,
                                    attention_mask=target_mask
                                )
                                generated_lengths = _sequence_lengths_excluding_specials(
                                    gen_DDIM[:, :seq_len],
                                    special_ids=special_ids
                                )
                                t_len_cpu = target_lengths.detach().to('cpu')
                                g_len_cpu = generated_lengths.detach().to('cpu')
                                args._val_ddim_metrics['sum_len_target'] += float(t_len_cpu.sum().item())
                                args._val_ddim_metrics['sum_len_generated'] += float(g_len_cpu.sum().item())
                                len_diff = (g_len_cpu - t_len_cpu).abs()
                                args._val_ddim_metrics['sum_len_abs_diff'] += float(len_diff.sum().item())
                                args._val_ddim_metrics['len_counts'] += int(t_len_cpu.numel())
                                accs_DDIM = _token_accuracies_from_generate(gen_DDIM, labels_trim, target_mask, autoencoder)
                                args._val_ddim_metrics['sum_acc_len'] += accs_DDIM['lenient'] * K
                                args._val_ddim_metrics['sum_acc_str'] += accs_DDIM['strict'] * K
                                args._val_ddim_metrics['covered'] += K

                                # Track UNK usage from generated sequences
                                cfg = getattr(autoencoder, 'config', None)
                                unk_id = getattr(cfg, 'unk_token_id', None) if cfg is not None else None
                                if unk_id is not None:
                                    gen_trim = gen_DDIM[:, :seq_len]
                                    valid_flat = target_mask.view(-1)
                                    if valid_flat.numel() > 0 and valid_flat.any():
                                        gen_flat = gen_trim.view(-1)[valid_flat]
                                        total_valid = int(valid_flat.sum().item())
                                        unk_hits = (gen_flat == int(unk_id)).sum().item()
                                        args._val_ddim_metrics.setdefault('unk_gen_hits', 0)
                                        args._val_ddim_metrics.setdefault('unk_gen_total', 0)
                                        args._val_ddim_metrics['unk_gen_hits'] += int(unk_hits)
                                        args._val_ddim_metrics['unk_gen_total'] += total_valid

                                # Additional DDIM diagnostics: UNK, rank/top-k, margin/confidence, per-position cosine percentiles
                                if not no_compression_mode:
                                    try:
                                        # 1) Get logits once (no sampling) for DDIM latents
                                        dd_out = autoencoder(
                                            encoder_outputs=re_enc_DDIM,
                                            attention_mask=target_mask,
                                            labels=labels_trim,
                                            return_dict=True
                                        )
                                        logits_ddim = dd_out.logits  # (K, L, V)
                                        valid_mask = target_mask.bool()

                                        # 2) Token ranks, P@k, margin, confidence
                                        true_logits = logits_ddim.gather(-1, labels_trim.unsqueeze(-1)).squeeze(-1)  # (K, L)
                                        rank = (logits_ddim > true_logits.unsqueeze(-1)).sum(dim=-1) + 1  # (K, L)
                                        # Flatten to avoid any shape/broadcast mismatches
                                        valid_flat = valid_mask.view(-1)
                                        rank_flat = rank.view(-1)
                                        if valid_flat.any():
                                            rank_sel = rank_flat[valid_flat]
                                            args._val_ddim_metrics.setdefault('rank_sum', 0.0)
                                            args._val_ddim_metrics.setdefault('rank_cnt', 0)
                                            args._val_ddim_metrics['rank_sum'] += rank_sel.float().sum().item()
                                            args._val_ddim_metrics['rank_cnt'] += int(rank_sel.numel())
                                            # P@k hits
                                            for k_ in (1, 5, 10):
                                                hits = (rank_sel <= k_).sum().item()
                                                args._val_ddim_metrics.setdefault(f'p_at_{k_}_hits', 0)
                                                args._val_ddim_metrics[f'p_at_{k_}_hits'] += int(hits)
                                            args._val_ddim_metrics.setdefault('p_denom', 0)
                                            args._val_ddim_metrics['p_denom'] += int(rank_sel.numel())

                                        top1_vals, top1_ids = logits_ddim.max(dim=-1)  # (K, L)
                                        import torch as _torch
                                        probs_ddim = _torch.softmax(logits_ddim, dim=-1)
                                        # margin
                                        margin_vals = (true_logits - top1_vals).view(-1)[valid_flat]
                                        if margin_vals.numel() > 0:
                                            args._val_ddim_metrics.setdefault('margin_sum', 0.0)
                                            args._val_ddim_metrics.setdefault('margin_cnt', 0)
                                            args._val_ddim_metrics['margin_sum'] += margin_vals.float().sum().item()
                                            args._val_ddim_metrics['margin_cnt'] += int(margin_vals.numel())
                                        # confidence
                                        conf_vals = probs_ddim.max(dim=-1).values.view(-1)[valid_flat]
                                        if conf_vals.numel() > 0:
                                            args._val_ddim_metrics.setdefault('conf_sum', 0.0)
                                            args._val_ddim_metrics.setdefault('conf_cnt', 0)
                                            args._val_ddim_metrics['conf_sum'] += conf_vals.float().sum().item()
                                            args._val_ddim_metrics['conf_cnt'] += int(conf_vals.numel())

                                        # 3) UNK diagnostics
                                        cfg = getattr(autoencoder, 'config', None)
                                        unk_id = getattr(cfg, 'unk_token_id', None) if cfg is not None else None
                                        if unk_id is not None and 0 <= int(unk_id) < logits_ddim.size(-1):
                                            unk_top1 = (top1_ids.view(-1)[valid_flat] == int(unk_id)).sum().item()
                                            args._val_ddim_metrics.setdefault('unk_top1_cnt', 0)
                                            args._val_ddim_metrics.setdefault('unk_valid_cnt', 0)
                                            args._val_ddim_metrics['unk_top1_cnt'] += int(unk_top1)
                                            args._val_ddim_metrics['unk_valid_cnt'] += int(valid_flat.sum().item())
                                            unk_prob_vals = probs_ddim[..., int(unk_id)].view(-1)[valid_flat]
                                            args._val_ddim_metrics.setdefault('unk_prob_sum', 0.0)
                                            args._val_ddim_metrics['unk_prob_sum'] += float(unk_prob_vals.float().sum().item())

                                    except Exception as _ddim_e:
                                        print(f"[ValDiag-DDIM] extra metrics failed: {_ddim_e}")
                            if val_pbar is not None:
                                try:
                                    val_pbar.update(K)
                                except Exception:
                                    pass
                    except Exception as _e:
                        print(f"[ValDiag] Warning: validation diagnostic failed: {_e}")
            
            # Combine losses
            val_loss = diffusion_loss + anchor_loss_weight * anchor_loss
            
            total_val_loss += val_loss.item()
            total_diffusion_loss += diffusion_loss.item()
            total_anchor_loss += anchor_loss.item()
            num_val_batches += 1
            
            # Track conditional vs unconditional losses for unified training
            if args.data_type == "unified" and is_conditional is not None:
                conditional_mask = is_conditional
                unconditional_mask = ~is_conditional
                
                # Use per-sample losses for proper masking
                anchor_sample = anchor_per_sample if anchor_per_sample is not None else diffusion_loss_per_sample.new_zeros(diffusion_loss_per_sample.shape)
                per_sample_loss = diffusion_loss_per_sample + anchor_loss_weight * anchor_sample
                
                if conditional_mask.any():
                    conditional_loss += per_sample_loss[conditional_mask].sum().item()
                    num_conditional += conditional_mask.sum().item()
                
                if unconditional_mask.any():
                    unconditional_loss += per_sample_loss[unconditional_mask].sum().item()
                    num_unconditional += unconditional_mask.sum().item()

            processed_samples += batch_size
            if main_pbar is not None:
                try:
                    main_pbar.update(1)
                except Exception:
                    pass

    if val_diag_pbar is not None:
        try:
            val_diag_pbar.close()
        except Exception:
            pass

    if main_pbar is not None:
        try:
            main_pbar.close()
        except Exception:
            pass

    # Reduce DDIM diagnostics across processes, if applicable
    if val_diag_enabled and hasattr(args, '_val_ddim_metrics') and accelerator is not None and accelerator.num_processes > 1:
        reduce_keys = [
            'sum_cos', 'sum_l2', 'count_pos', 'sum_acc_len', 'sum_acc_str', 'covered',
            'sum_len_target', 'sum_len_generated', 'sum_len_abs_diff', 'len_counts',
            'unk_gen_hits', 'unk_gen_total', 'rank_sum', 'rank_cnt', 'p_at_1_hits',
            'p_at_5_hits', 'p_at_10_hits', 'p_denom', 'margin_sum', 'margin_cnt',
            'conf_sum', 'conf_cnt', 'unk_top1_cnt', 'unk_valid_cnt', 'unk_prob_sum'
        ]
        for key in reduce_keys:
            value = float(args._val_ddim_metrics.get(key, 0.0))
            tensor = torch.tensor(value, device=accelerator.device, dtype=torch.float64)
            reduced = accelerator.reduce(tensor, reduction='sum')
            args._val_ddim_metrics[key] = reduced.item()

    # Calculate averages
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    avg_val_diffusion = total_diffusion_loss / num_val_batches if num_val_batches > 0 else 0.0
    avg_val_anchor = total_anchor_loss / num_val_batches if num_val_batches > 0 else 0.0
    
    # Calculate conditional vs unconditional averages for unified training
    avg_conditional_loss = conditional_loss / num_conditional if num_conditional > 0 else 0.0
    avg_unconditional_loss = unconditional_loss / num_unconditional if num_unconditional > 0 else 0.0
    
    # Emit aggregated validation diagnostics if any
    if val_diag_enabled and batches_used > 0:
        mean_cos = (sum_cos / count_pos) if count_pos > 0 else 0.0
        mean_l2 = (sum_l2 / count_pos) if count_pos > 0 else 0.0
        accH_len = (sum_accH_len / covered_total) if covered_total > 0 else 0.0
        accH_str = (sum_accH_str / covered_total) if covered_total > 0 else 0.0
        accX_len = (sum_accX_len / covered_total) if covered_total > 0 else 0.0
        accX_str = (sum_accX_str / covered_total) if covered_total > 0 else 0.0
        t_mean = (sum_t_mean_weighted / covered_total) if covered_total > 0 else 0.0
        unk_rate_H = (sum_unkH_hits / sum_unk_token_den) if sum_unk_token_den > 0 else 0.0
        unk_rate_X = (sum_unkX_hits / sum_unk_token_den) if sum_unk_token_den > 0 else 0.0
        den_display = sum_unk_token_den if sum_unk_token_den > 0 else 0
        print(
            f"[ValDiag] cos={mean_cos:.4f} l2={mean_l2:.4f} | acc(H') lenient={accH_len:.4f} strict={accH_str:.4f} "
            f"| acc(x0) lenient={accX_len:.4f} strict={accX_str:.4f} "
            f"| unk(H')={unk_rate_H:.4f} ({sum_unkH_hits}/{den_display}) "
            f"unk(x0)={unk_rate_X:.4f} ({sum_unkX_hits}/{den_display}) "
            f"| covered={covered_total} batches={batches_used} | t(mean/min/max)={t_mean:.1f}/{t_min_global:.0f}/{t_max_global:.0f}"
        )
        if args.use_wandb and wandb.run is not None:
            wandb.log({
                "val_diag/latent_cosine": mean_cos,
                "val_diag/latent_l2": mean_l2,
                "val_diag/acc_decode_Hprime_lenient": accH_len,
                "val_diag/acc_decode_Hprime_strict": accH_str,
                "val_diag/acc_decode_x0_lenient": accX_len,
                "val_diag/acc_decode_x0_strict": accX_str,
                "val_diag/unk_decode_Hprime_rate": unk_rate_H,
                "val_diag/unk_decode_x0_rate": unk_rate_X,
                "val_diag/unk_decode_token_total": sum_unk_token_den,
                "val_diag/covered": covered_total,
                "val_diag/batches": batches_used,
                "val_diag/t_mean": t_mean,
                "val_diag/t_min": t_min_global if t_min_global != float('inf') else 0.0,
                "val_diag/t_max": t_max_global if t_max_global != float('-inf') else 0.0,
            })

        # Log optional DDIM diagnostics if computed
        if hasattr(args, '_val_ddim_metrics') and args._val_ddim_metrics['covered'] > 0:
            dd = args._val_ddim_metrics
            mean_cos_ddim = (dd['sum_cos'] / dd['count_pos']) if dd['count_pos'] > 0 else 0.0
            mean_l2_ddim = (dd['sum_l2'] / dd['count_pos']) if dd['count_pos'] > 0 else 0.0
            acc_ddim_len = dd['sum_acc_len'] / dd['covered']
            acc_ddim_str = dd['sum_acc_str'] / dd['covered']
            len_den = max(1, dd.get('len_counts', 0))
            mean_len_target = dd.get('sum_len_target', 0.0) / len_den
            mean_len_generated = dd.get('sum_len_generated', 0.0) / len_den
            mean_len_abs_diff = dd.get('sum_len_abs_diff', 0.0) / len_den
            # Aggregate extra metrics if present
            mean_rank = (dd.get('rank_sum', 0.0) / dd.get('rank_cnt', 1)) if dd.get('rank_cnt', 0) > 0 else 0.0
            pden = max(1, dd.get('p_denom', 0))
            p_at_1 = dd.get('p_at_1_hits', 0) / pden
            p_at_5 = dd.get('p_at_5_hits', 0) / pden
            p_at_10 = dd.get('p_at_10_hits', 0) / pden
            mean_margin = (dd.get('margin_sum', 0.0) / dd.get('margin_cnt', 1)) if dd.get('margin_cnt', 0) > 0 else 0.0
            mean_conf = (dd.get('conf_sum', 0.0) / dd.get('conf_cnt', 1)) if dd.get('conf_cnt', 0) > 0 else 0.0
            unk_rate = (dd.get('unk_top1_cnt', 0) / max(1, dd.get('unk_valid_cnt', 0))) if dd.get('unk_valid_cnt', 0) > 0 else 0.0
            unk_prob = (dd.get('unk_prob_sum', 0.0) / max(1, dd.get('unk_valid_cnt', 0))) if dd.get('unk_valid_cnt', 0) > 0 else 0.0
            unk_gen_rate = (dd.get('unk_gen_hits', 0) / max(1, dd.get('unk_gen_total', 0))) if dd.get('unk_gen_total', 0) > 0 else 0.0
            try:
                import torch as _torch
                if dd.get('cos_vals', None):
                    cos_t = _torch.tensor(dd['cos_vals'])
                    cosine_p10 = _torch.quantile(cos_t, 0.10).item()
                    cosine_p50 = _torch.quantile(cos_t, 0.50).item()
                    cosine_p90 = _torch.quantile(cos_t, 0.90).item()
                else:
                    cosine_p10 = cosine_p50 = cosine_p90 = 0.0
            except Exception:
                cosine_p10 = cosine_p50 = cosine_p90 = 0.0
            print(
                f"[ValDiag-DDIM] steps={getattr(args,'diag_ddim_steps',0)} | cos={mean_cos_ddim:.4f} l2={mean_l2_ddim:.4f} "
                f"| acc(ddim) lenient={acc_ddim_len:.4f} strict={acc_ddim_str:.4f} "
                f"| len(target)={mean_len_target:.2f} len(gen)={mean_len_generated:.2f} diff={mean_len_abs_diff:.2f} "
                f"| covered={dd['covered']} | unk(gen)={unk_gen_rate:.4f}"
            )
            if args.use_wandb and wandb.run is not None:
                wandb.log({
                    "val_diag_ddim/steps": int(getattr(args,'diag_ddim_steps',0) or 0),
                    "val_diag_ddim/latent_cosine": mean_cos_ddim,
                    "val_diag_ddim/latent_l2": mean_l2_ddim,
                    "val_diag_ddim/len_target": mean_len_target,
                    "val_diag_ddim/len_generated": mean_len_generated,
                    "val_diag_ddim/len_abs_diff": mean_len_abs_diff,
                    "val_diag_ddim/acc_decode_lenient": acc_ddim_len,
                    "val_diag_ddim/acc_decode_strict": acc_ddim_str,
                    "val_diag_ddim/covered": dd['covered'],
                    "val_diag_ddim/mean_rank": mean_rank,
                    "val_diag_ddim/p_at_1": p_at_1,
                    "val_diag_ddim/p_at_5": p_at_5,
                    "val_diag_ddim/p_at_10": p_at_10,
                    "val_diag_ddim/margin": mean_margin,
                    "val_diag_ddim/top1_conf": mean_conf,
                    "val_diag_ddim/unk_rate": unk_rate,
                    "val_diag_ddim/unk_prob": unk_prob,
                    "val_diag_ddim/unk_rate_generated": unk_gen_rate,
                    "val_diag_ddim/cosine_p10": cosine_p10,
                    "val_diag_ddim/cosine_p50": cosine_p50,
                    "val_diag_ddim/cosine_p90": cosine_p90,
                })
        # Close DDIM progress bar if it was created
        ddim_pbar = getattr(args, '_val_ddim_decode_pbar', None)
        if ddim_pbar is not None:
            try:
                ddim_pbar.close()
            except Exception:
                pass
            delattr(args, '_val_ddim_decode_pbar')
    if val_diag_pbar is not None:
        try:
            val_diag_pbar.close()
        except Exception:
            pass
    
    # Return validation results with unified training statistics
    if args.data_type == "unified":
        return avg_val_loss, avg_val_diffusion, avg_val_anchor, num_conditional, num_unconditional, avg_conditional_loss, avg_unconditional_loss
    else:
        return avg_val_loss, avg_val_diffusion, avg_val_anchor
