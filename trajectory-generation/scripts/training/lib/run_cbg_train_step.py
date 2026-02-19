from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from src.diffusion_model import GaussianDiffusion
from lib.run_cbg_bootstrap import _posterior_sample, sample_latents


def training_step(trainer) -> Tuple[torch.Tensor, Dict[str, float]]:
    # For supervised-only baselines (diffusion_mse), it can be useful to skip the expensive
    # aggregate sampling/decoding path when lambda_agg == 0. In that case we also avoid
    # sampling a region/cbg unless the MSE branch needs pi_cbg (demo_source='pi').
    skip_aggregate = (
        trainer.skip_agg_when_lambda_zero
        and trainer.lambda_agg <= 0.0
        and not trainer.use_entropy_reg
        and not trainer.use_unique_reg
        and not trainer.llp_enabled
    )
    mse_needs_cbg = bool(
        trainer.use_diffusion_mse
        and getattr(trainer, "mse_demo_source", "null") == "pi"
    )

    cbg = trainer.sample_cbg() if ((not skip_aggregate) or mse_needs_cbg) else "N/A"
    stats: Dict[str, float] = {"cbg": cbg}

    ent_loss = poi_probs = None  # type: ignore[assignment]
    uniq_loss = None

    # ------------------------------------------------------------------
    # Two training modes:
    # 1) Default: full reverse sampling (grad only on t=0)
    # 2) Simple x0 (optional): fast-forward to small t and compute loss on
    #    predicted x0 for the last K steps (grad over those K steps only)
    # ------------------------------------------------------------------
    if skip_aggregate:
        agg_loss = torch.zeros((), device=trainer.device, dtype=torch.float32)
    else:
        batch = trainer.cache.sample(cbg, trainer.batch_size, device=trainer.device)

        # Decide which demo ids to use for conditioning + LLP grouping.
        # Raw ids are 0-based; embeddings expect 1-based (0 reserved for null/pad).
        if trainer.llp_enabled and trainer.llp_demo_source == "pi" and trainer.num_age_bins > 0 and trainer.num_genders > 0:
            age_raw, gender_raw, llp_demo_stats = trainer._sample_demo_ids_from_pi(cbg, trainer.batch_size)
            stats.update(llp_demo_stats)
        else:
            # Default behavior: use cache-provided per-trajectory labels.
            age_raw = batch.age_bin.to(device=trainer.device)
            gender_raw = batch.gender_id.to(device=trainer.device)
            stats["llp_demo_source_pi"] = 0.0

        # Pass raw attributes to DiT:
        # Always include continuous: [work_lat, work_lon, home_lat, home_lon]
        attrs_parts = [batch.work, batch.home]
        # Optionally append demo ids when enabled: [age_id, gender_id]
        base_dit = getattr(trainer.dit, "module", trainer.dit)  # unwrap DDP / Accelerate wrapper if present
        if getattr(base_dit.attr_embed, "use_demo_condition", False):
            # Shift demographic IDs by +1 so that 0 is reserved as a padding/null id.
            # Make sure DiT.num_age_bins / num_genders match the *unshifted* max values.
            age_f = (age_raw.long().clamp_min(0) + 1).to(dtype=batch.work.dtype).unsqueeze(-1)
            gender_f = (gender_raw.long().clamp_min(0) + 1).to(dtype=batch.work.dtype).unsqueeze(-1)
            attrs_parts.extend([age_f, gender_f])
        attrs = torch.cat(attrs_parts, dim=-1)
        # Optional: random coord dropout for the aggregate/LLP branch (demo remains, if present).
        if trainer.aggregate_coord_dropout > 0.0:
            mask = torch.rand(attrs.size(0), device=attrs.device) < float(trainer.aggregate_coord_dropout)
            if mask.any():
                attrs = attrs.clone()
                attrs[mask, :4] = 0.0
                stats["coord_dropout_p"] = float(trainer.aggregate_coord_dropout)
                stats["coord_dropout_n"] = float(mask.sum().item())

        if not trainer.simple_x0:
            latents = sample_latents(
                diffusion=trainer.scheduler,
                dit=trainer.dit,
                cond=attrs,
                seq_len=trainer.seq_len,
                latent_dim=trainer.latent_dim,
                prediction_type=trainer.prediction_type,
                guidance_scale=trainer.guidance_scale,
                sampler=trainer.aggregate_sampler,
                ddim_steps=trainer.aggregate_ddim_steps,
                ddim_eta=trainer.aggregate_ddim_eta,
                device=trainer.device,
            )
            # If PCA is configured, unproject to AE latent dimension before decoding
            decode_latents = trainer.latent_pca.unproject(latents) if trainer.latent_pca is not None else latents
            encoder_outputs = BaseModelOutput(last_hidden_state=decode_latents)
            # Use sampled variable lengths for the aggregate decoding/POI marginal loss.
            agg_attn_mask = trainer._sample_length_mask(cbg, latents.size(0))
            # BART requires decoder input: construct full BOS decoder_input_ids as starting token sequence.
            bos_id = trainer.autoencoder.config.decoder_start_token_id or trainer.autoencoder.config.bos_token_id
            if bos_id is None:
                raise ValueError("Autoencoder config must define bos_token_id or decoder_start_token_id.")
            decoder_input_ids = torch.full(
                (latents.size(0), trainer.seq_len),
                bos_id,
                device=trainer.device,
                dtype=torch.long,
            )
            decoder_out = trainer.autoencoder(
                encoder_outputs=encoder_outputs,
                attention_mask=agg_attn_mask,      # encoder attention mask
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_dict=True,
            )
            # Ban specials for aggregate/LLP view by shifting logits of specials far negative.
            logits = decoder_out.logits
            if trainer.num_special_tokens > 0:
                # In-place subtraction is fine here; this is a local view used only for aggregate loss.
                logits[..., :trainer.num_special_tokens] = logits[..., :trainer.num_special_tokens] - 1e4
            poi_probs = torch.softmax(logits, dim=-1)
            # Drop special tokens at the start of the vocab so that the last dimension
            # matches the POI-only vocabulary used in p_poi.csv.
            if trainer.num_special_tokens > 0:
                poi_probs = poi_probs[..., trainer.num_special_tokens:]
            poi_probs = trainer._apply_poi_token_mask_to_probs(poi_probs, stats=stats, epsilon=trainer.aggregate_loss_eps)
            agg_loss, feat_stats = trainer._aggregate_feature_losses(
                cbg=cbg,
                poi_probs=poi_probs,
                attention_mask=agg_attn_mask,
                age_raw=age_raw,
                gender_raw=gender_raw,
            )
            stats.update(feat_stats)
            # Optional entropy regularizer to encourage trajectory-level POI diversity.
            ent_loss = trainer._entropy_regularizer(poi_probs, agg_attn_mask)
            # Optional expected-unique-POI regularizer.
            uniq_loss = trainer._unique_regularizer(poi_probs, agg_attn_mask)
        else:
            # Simple x0 training over last K steps near t=0 (K>=1)
            timesteps = trainer.scheduler.num_timesteps
            K = min(max(1, trainer.simple_x0_last_k), timesteps)
            # Initialize at pure noise
            latents = torch.randn(attrs.size(0), trainer.seq_len, trainer.latent_dim, device=trainer.device)
            # Fast-forward (no grad) from T-1 down to K to obtain x_t at t=K
            for idx in range(timesteps - 1, K - 1, -1):
                t = torch.full((latents.size(0),), idx, device=trainer.device, dtype=torch.long)
                with torch.no_grad():
                    model_out = GaussianDiffusion.classifier_free_guidance(
                        denoiser=trainer.dit,
                        x_t=latents,
                        t=t,
                        conditional_attrs=attrs,
                        guidance_scale=trainer.guidance_scale,
                    )
                    preds = trainer.scheduler.model_predictions(
                        model_out, latents, t, prediction_type=trainer.prediction_type
                    )
                    latents = _posterior_sample(trainer.scheduler, preds.x_start, latents, t)

            # Now compute losses on predicted x0 for the final K steps with gradient
            predicted_x0_list: List[torch.Tensor] = []
            x_t = latents
            for idx in range(K - 1, -1, -1):
                t = torch.full((x_t.size(0),), idx, device=trainer.device, dtype=torch.long)
                model_out = GaussianDiffusion.classifier_free_guidance(
                    denoiser=trainer.dit,
                    x_t=x_t,
                    t=t,
                    conditional_attrs=attrs,
                    guidance_scale=trainer.guidance_scale,
                )
                preds = trainer.scheduler.model_predictions(
                    model_out, x_t, t, prediction_type=trainer.prediction_type
                )
                predicted_x0_list.append(preds.x_start)
                if idx > 0:
                    # One posterior step with gradient enabled
                    posterior_mean, _, posterior_log_var = trainer.scheduler.q_posterior(
                        x_start=preds.x_start, x_t=x_t, t=t
                    )
                    noise = torch.randn_like(x_t)
                    x_t = posterior_mean + torch.exp(0.5 * posterior_log_var) * noise

            # Shared decoding inputs
            agg_attn_mask = trainer._sample_length_mask(cbg, attrs.size(0))
            bos_id = trainer.autoencoder.config.decoder_start_token_id or trainer.autoencoder.config.bos_token_id
            if bos_id is None:
                raise ValueError("Autoencoder config must define bos_token_id or decoder_start_token_id.")
            decoder_input_ids = torch.full(
                (attrs.size(0), trainer.seq_len), bos_id, device=trainer.device, dtype=torch.long
            )

            agg_losses: List[torch.Tensor] = []
            ent_losses: List[torch.Tensor] = []
            uniq_losses: List[torch.Tensor] = []
            for x0 in predicted_x0_list:
                decode_latents = trainer.latent_pca.unproject(x0) if trainer.latent_pca is not None else x0
                encoder_outputs = BaseModelOutput(last_hidden_state=decode_latents)
                decoder_out = trainer.autoencoder(
                    encoder_outputs=encoder_outputs,
                    attention_mask=agg_attn_mask,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                    return_dict=True,
                )
                # Ban specials for aggregate/LLP view before softmax.
                logits = decoder_out.logits
                if trainer.num_special_tokens > 0:
                    logits[..., :trainer.num_special_tokens] = logits[..., :trainer.num_special_tokens] - 1e4
                poi_probs = torch.softmax(logits, dim=-1)
                if trainer.num_special_tokens > 0:
                    poi_probs = poi_probs[..., trainer.num_special_tokens:]
                poi_probs = trainer._apply_poi_token_mask_to_probs(poi_probs, stats=stats, epsilon=trainer.aggregate_loss_eps)
                loss_k, _ = trainer._aggregate_feature_losses(
                    cbg=cbg,
                    poi_probs=poi_probs,
                    attention_mask=agg_attn_mask,
                    age_raw=age_raw,
                    gender_raw=gender_raw,
                )
                agg_losses.append(loss_k)
                # Entropy regularizer per step; average over K at the end.
                ent_losses.append(trainer._entropy_regularizer(poi_probs, agg_attn_mask))
                uniq_losses.append(trainer._unique_regularizer(poi_probs, agg_attn_mask))
            agg_loss = torch.stack(agg_losses).mean()
            ent_loss = torch.stack(ent_losses).mean() if ent_losses else poi_probs.new_zeros((), requires_grad=False)
            uniq_loss = torch.stack(uniq_losses).mean() if uniq_losses else poi_probs.new_zeros((), requires_grad=False)

    # Base loss: aggregate KL scaled by lambda_agg.
    total_loss = trainer.lambda_agg * agg_loss

    # Add entropy regularizer if enabled (separately scaled).
    if trainer.use_entropy_reg and trainer.lambda_entropy > 0.0 and ent_loss is not None and ent_loss.requires_grad:
        total_loss = total_loss + trainer.lambda_entropy * ent_loss
        stats["entropy_loss"] = float(ent_loss.detach().item())

    # Add expected-unique-POI regularizer if enabled.
    if trainer.use_unique_reg and trainer.lambda_unique > 0.0 and uniq_loss is not None and uniq_loss.requires_grad:
        total_loss = total_loss + trainer.lambda_unique * uniq_loss
        stats["unique_loss"] = float(uniq_loss.detach().item())

    # Optional diffusion MSE branch (supervised with AE targets)
    mse_loss_val = None
    if trainer.use_diffusion_mse and trainer.mse_loader is not None:
        try:
            ab = next(trainer.mse_iter)
        except StopIteration:
            trainer.mse_iter = iter(trainer.mse_loader)
            ab = next(trainer.mse_iter)

        input_ids = ab["input_ids"].to(trainer.device)
        attn_mask = ab["attention_mask"].to(trainer.device)
        # Defensive checks: these are common sources of CUDA device-side asserts.
        # 1) token ids must be within BART vocab range
        # 2) sequence length must not exceed BART position embeddings
        try:
            emb = trainer.autoencoder.get_input_embeddings()
            vocab_size = int(getattr(emb, "num_embeddings", 0) or emb.weight.size(0))
            max_id = int(input_ids.max().item()) if input_ids.numel() > 0 else -1
            min_id = int(input_ids.min().item()) if input_ids.numel() > 0 else 0
            if min_id < 0 or max_id >= vocab_size:
                raise ValueError(
                    f"MSE branch input_ids out of range for BART vocab: min={min_id}, max={max_id}, "
                    f"vocab_size={vocab_size}. This usually means diffusion_mse.data_dir uses a different "
                    "tokenizer/vocab than the phase-1 BART autoencoder was trained with."
                )
            max_pos = int(getattr(getattr(trainer.autoencoder, "config", None), "max_position_embeddings", 0) or 0)
            if max_pos > 0 and input_ids.size(1) > max_pos:
                raise ValueError(
                    f"MSE branch sequence length {int(input_ids.size(1))} exceeds BART max_position_embeddings "
                    f"{max_pos}. Ensure diffusion.seq_len matches your autoencoder and that the MSE dataset "
                    "is built with the same sequence length."
                )
        except Exception as e:
            # Raise as ValueError so the error message is visible without CUDA DSA noise.
            raise ValueError(str(e))
        # Build AE target latents
        with torch.no_grad():
            enc_out = trainer.autoencoder.get_encoder()(input_ids=input_ids, attention_mask=attn_mask)
            target_latents = enc_out.last_hidden_state
            if trainer.latent_pca is not None:
                target_latents = trainer.latent_pca.project(target_latents)

        # Diffusion supervision on targets
        bs = target_latents.size(0)
        # Match timestep sampling strategy from the main DiT trainer.
        t_mse = trainer.scheduler.sample_timesteps(
            bs,
            device=target_latents.device,
            method=trainer.timestep_sampling,
        )
        noise = torch.randn_like(target_latents)
        noisy = trainer.scheduler.q_sample(target_latents, t_mse, noise)

        cond_attrs = ab.get("attrs", None)
        conditional_mask_for_forward = None

        # For unified training, mirror the conditional dropout behavior from train_dit_only.py
        is_conditional = ab.get("is_conditional", None)
        if cond_attrs is not None:
            cond_attrs = cond_attrs.to(trainer.device)
            # MSE branch demographic conditioning behavior is controlled by diffusion_mse.demo_source:
            #   - null: ignore demo dims (non-cheating; uses only work/home coords)
            #   - pi: sample demo ids from pi_cbg (age x gender) for this batch
            #   - data: use demo ids from dataset's attrs (requires all_attr_results_with_demo.npy)
            base_dit_mse = getattr(trainer.dit, "module", trainer.dit)
            use_demo_mse = getattr(getattr(base_dit_mse, "attr_embed", None), "use_demo_condition", False)
            if use_demo_mse:
                use_length_mse = bool(getattr(getattr(base_dit_mse, "attr_embed", None), "use_length_condition", False))
                demo_start = 4 + (1 if use_length_mse else 0)
                needed_dim = demo_start + 2
                # If attrs are shorter, pad to the expected minimum.
                if cond_attrs.dim() >= 2:
                    feat_dim = cond_attrs.size(-1)
                    if feat_dim < needed_dim:
                        if getattr(trainer, "mse_demo_source", "null") == "data":
                            raise ValueError(
                                "diffusion_mse.demo_source='data' requires supervised attrs with demo ids, "
                                f"but got attrs dim={feat_dim}. Ensure your MSE data_dir contains "
                                "all_attr_results_with_demo.npy (6-D attrs)."
                            )
                        pad = needed_dim - feat_dim
                        zeros = torch.zeros(cond_attrs.size(0), pad, device=cond_attrs.device, dtype=cond_attrs.dtype)
                        cond_attrs = torch.cat([cond_attrs, zeros], dim=-1)
                    # At this point, cond_attrs has at least `needed_dim` dims.
                    cond_attrs = cond_attrs.clone()
                    demo_source = getattr(trainer, "mse_demo_source", "null")
                    if demo_source == "pi" and (trainer.num_age_bins > 0) and (trainer.num_genders > 0):
                        # Sample demo ids from the current CBG mixture pi_cbg (age x gender)
                        D_age = int(trainer.num_age_bins)
                        D_gen = int(trainer.num_genders)
                        D = D_age * D_gen
                        # pi_cbg over D groups
                        pi = trainer.cache.as_torch_distribution(
                            cbg,
                            num_age_bins=D_age,
                            num_genders=D_gen,
                            device=cond_attrs.device,
                        )
                        # Normalize defensively
                        eps = 1e-8
                        pi = (pi + eps) / (pi.sum() + eps * pi.numel())
                        B = cond_attrs.size(0)
                        # Sample group indices with replacement
                        group_idx = torch.multinomial(pi, num_samples=B, replacement=True)
                        age_raw = torch.div(group_idx, D_gen, rounding_mode="floor")
                        gender_raw = group_idx % D_gen
                        # Shift to 1-based ids for embeddings (0 is padding/null)
                        age_shifted = (age_raw + 1).to(dtype=cond_attrs.dtype)
                        gender_shifted = (gender_raw + 1).to(dtype=cond_attrs.dtype)
                        cond_attrs[:, demo_start] = age_shifted
                        cond_attrs[:, demo_start + 1] = gender_shifted
                    elif demo_source == "data":
                        # Dataset demo ids are expected to be 0-based (age_bin in [0..A-1], gender_id in [0..G-1]).
                        # Shift to 1-based ids for embeddings (0 reserved for null/pad).
                        age_raw_ids = cond_attrs[:, demo_start].to(torch.long)
                        gender_raw_ids = cond_attrs[:, demo_start + 1].to(torch.long)
                        if (age_raw_ids < 0).any() or (gender_raw_ids < 0).any():
                            raise ValueError(
                                "diffusion_mse.demo_source='data' found missing demo ids (negative values) in the "
                                "supervised dataset attrs. Filter out missing age/gender trajectories (e.g., build "
                                "a selected-only controlled split aligned to your LLP-world)."
                            )
                        age_ids = age_raw_ids.clamp_min(0)
                        gender_ids = gender_raw_ids.clamp_min(0)
                        if trainer.num_age_bins > 0:
                            age_ids = age_ids.clamp_max(trainer.num_age_bins - 1)
                        if trainer.num_genders > 0:
                            gender_ids = gender_ids.clamp_max(trainer.num_genders - 1)
                        cond_attrs[:, demo_start] = (age_ids + 1).to(dtype=cond_attrs.dtype)
                        cond_attrs[:, demo_start + 1] = (gender_ids + 1).to(dtype=cond_attrs.dtype)
                    else:
                        # Default: null out demo dims so MSE uses only work/home coords.
                        cond_attrs[:, demo_start:] = 0
            if is_conditional is not None:
                is_conditional = is_conditional.to(trainer.device).bool()
                if trainer.mse_conditional_dropout > 0.0:
                    dropout_mask = torch.rand(
                        len(is_conditional),
                        device=is_conditional.device,
                    ) < trainer.mse_conditional_dropout
                    should_be_conditional = is_conditional & ~dropout_mask
                else:
                    should_be_conditional = is_conditional

                if should_be_conditional.any():
                    attr_embeds_for_model = cond_attrs.clone()
                    attr_embeds_for_model[~should_be_conditional] = 0
                    conditional_mask_for_forward = should_be_conditional
                else:
                    attr_embeds_for_model = None
                    conditional_mask_for_forward = None
            else:
                # Non-unified setting: use attributes as-is.
                attr_embeds_for_model = cond_attrs
        else:
            attr_embeds_for_model = None

        model_out = GaussianDiffusion.classifier_free_guidance(
            denoiser=trainer.dit,
            x_t=noisy,
            t=t_mse,
            conditional_attrs=attr_embeds_for_model,
            guidance_scale=trainer.guidance_scale,
            conditional_mask=conditional_mask_for_forward,
        )
        preds = trainer.scheduler.model_predictions(model_out, noisy, t_mse, prediction_type=trainer.prediction_type)

        if trainer.prediction_type == "epsilon":
            mse_per_token = F.mse_loss(preds.noise, noise, reduction="none")
        elif trainer.prediction_type == "v":
            target_v = trainer.scheduler.predict_v_from_start(x_t=noisy, t=t_mse, x_start=target_latents).detach()
            mse_per_token = F.mse_loss(preds.v, target_v, reduction="none")
        else:  # x0
            mse_per_token = F.mse_loss(preds.x_start, target_latents, reduction="none")

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1).to(dtype=mse_per_token.dtype)
            masked = mse_per_token * mask
            denom = mask.sum().clamp_min(1.0)
            mse_loss_val = masked.sum() / denom
        else:
            mse_loss_val = mse_per_token.mean()

        total_loss = total_loss + trainer.lambda_mse * mse_loss_val
        stats["mse_loss"] = float(mse_loss_val.detach().item())

    stats["agg_loss"] = float(agg_loss.detach().item())
    return total_loss, stats
