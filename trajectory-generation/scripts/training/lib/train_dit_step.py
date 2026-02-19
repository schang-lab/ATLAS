from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses import compute_anchor_loss
from src.validation import _predict_noise_with_guidance
from lib.train_dit_helpers import shift_demo_ids_in_attrs


def execute_train_batch_step(
    *,
    args,
    batch,
    dit_model,
    noise_scheduler,
    autoencoder,
    accelerator,
    latent_pca,
    prediction_type,
    timestep_sampling,
    running_train_loss,
    running_diffusion_loss,
    running_anchor_loss,
    num_batches_since_log,
    accumulation_steps,
):
    # Extract data from batch
    attrs = batch["attrs"]  # Base attributes (coords + optional length/demo)
    length_id = batch.get("length_id", None)  # Optional discrete trajectory length condition
    is_conditional = batch.get("is_conditional", None)  # Track which samples are conditional
    attention_mask = batch["attention_mask"]
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    _dwell_times = batch.get("dwell_times", None)  # dwell times if available

    # Append discrete trajectory length identifier when enabled
    if length_id is not None and getattr(args, "enable_length_condition", False):
        length_tensor = length_id.float().unsqueeze(-1).to(attrs.device)
        attrs = torch.cat([attrs, length_tensor], dim=1)

    # Shift raw demo ids to 1-based range expected by AttrBlock.
    attrs = shift_demo_ids_in_attrs(attrs, dit_model)

    # Get target latents from autoencoder
    with torch.no_grad():
        if args.training_phase == "phase1":
            encoder_outputs = autoencoder.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            target_latents = encoder_outputs.last_hidden_state
        else:
            encoder_outputs = autoencoder.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            segment_coords = None
            sub_categories = None
            if args.ablation_mode in ["coords_only", "both"] and "lat" in batch and "lon" in batch:
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

    # Diffusion training step
    batch_size = target_latents.shape[0]
    t = noise_scheduler.sample_timesteps(batch_size, device=target_latents.device, method=timestep_sampling)
    noise = torch.randn_like(target_latents)
    noisy_latents = noise_scheduler.q_sample(target_latents, t, noise)

    conditional_mask_for_forward = None
    if args.data_type == "unified":
        if is_conditional is not None:
            dropout_mask = torch.rand(len(is_conditional), device=is_conditional.device) < args.conditional_dropout
            should_be_conditional = is_conditional & ~dropout_mask
            if should_be_conditional.any():
                attr_embeds_for_model = attrs.clone()
                attr_embeds_for_model[~should_be_conditional] = 0
                conditional_mask_for_forward = should_be_conditional
            else:
                attr_embeds_for_model = None
                conditional_mask_for_forward = None
        else:
            attr_embeds_for_model = attrs
            if attr_embeds_for_model is not None:
                conditional_mask_for_forward = attr_embeds_for_model.abs().sum(dim=1) > 0
    else:
        if attrs is None or attrs.abs().sum() == 0:
            attr_embeds_for_model = None
        else:
            attr_embeds_for_model = attrs
            conditional_mask_for_forward = attr_embeds_for_model.abs().sum(dim=1) > 0

    coord_dropout_p = float(getattr(args, "coord_dropout", 0.0) or 0.0)
    if coord_dropout_p > 0.0 and attr_embeds_for_model is not None:
        if not (0.0 <= coord_dropout_p <= 1.0):
            raise ValueError("--coord_dropout must be in [0, 1]")
        if attr_embeds_for_model.dim() == 2 and attr_embeds_for_model.size(1) >= 4:
            if conditional_mask_for_forward is None:
                conditional_rows = attr_embeds_for_model.abs().sum(dim=1) > 0
            else:
                conditional_rows = conditional_mask_for_forward.bool()
            drop_mask = torch.rand(attr_embeds_for_model.size(0), device=attr_embeds_for_model.device) < coord_dropout_p
            drop_mask = drop_mask & conditional_rows
            if drop_mask.any():
                attr_embeds_for_model = attr_embeds_for_model.clone()
                attr_embeds_for_model[drop_mask, :4] = 0
                conditional_mask_for_forward = attr_embeds_for_model.abs().sum(dim=1) > 0

    train_guidance_scale = getattr(args, "train_guidance_scale", 1.0)
    predictions = _predict_noise_with_guidance(
        model=dit_model,
        noise_scheduler=noise_scheduler,
        x_t=noisy_latents,
        timesteps=t,
        attrs=attr_embeds_for_model,
        prediction_type=prediction_type,
        guidance_scale=train_guidance_scale,
        conditional_mask=conditional_mask_for_forward,
    )
    predicted_noise = predictions.noise
    predicted_x0 = predictions.x_start
    predicted_v = predictions.v

    if prediction_type == "epsilon":
        diffusion_pred = predicted_noise
        diffusion_target = noise
    elif prediction_type == "v":
        diffusion_pred = predicted_v
        diffusion_target = noise_scheduler.predict_v_from_start(
            x_t=noisy_latents,
            t=t,
            x_start=target_latents,
        ).detach()
    else:
        diffusion_pred = predicted_x0
        diffusion_target = target_latents

    mse_per_token = F.mse_loss(diffusion_pred, diffusion_target, reduction="none")
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(diffusion_pred.dtype)
        masked_mse = mse_per_token * mask
        denom = mask.sum().clamp_min(1.0)
        diffusion_loss = masked_mse.sum() / denom
    else:
        diffusion_loss = mse_per_token.mean()

    anchor_loss = torch.tensor(0.0, device=predicted_x0.device)
    if args.use_anchor_loss:
        anchor_loss, _anchor_stats, _ = compute_anchor_loss(
            predicted_x0=predicted_x0,
            labels=labels,
            attention_mask=attention_mask,
            autoencoder=autoencoder,
            latent_scale=latent_scale,
            latent_pca=latent_pca,
            training_phase=args.training_phase,
            return_per_sample=False,
        )

    total_loss = diffusion_loss + args.anchor_loss_weight * anchor_loss
    total_loss = total_loss / args.gradient_accumulation_steps

    running_train_loss += total_loss.detach() * args.gradient_accumulation_steps
    running_diffusion_loss += diffusion_loss.detach()
    running_anchor_loss += anchor_loss.detach()
    num_batches_since_log += 1
    accumulation_steps += 1

    accelerator.backward(total_loss)

    step_ctx = {
        "target_latents": target_latents,
        "noisy_latents": noisy_latents,
        "predictions": predictions,
        "t": t,
        "attention_mask": attention_mask,
        "labels": labels,
        "latent_scale": latent_scale,
    }
    counters = {
        "running_train_loss": running_train_loss,
        "running_diffusion_loss": running_diffusion_loss,
        "running_anchor_loss": running_anchor_loss,
        "num_batches_since_log": num_batches_since_log,
        "accumulation_steps": accumulation_steps,
    }
    return step_ctx, counters
