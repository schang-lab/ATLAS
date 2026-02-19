from __future__ import annotations

import torch
import wandb
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput

from src.losses import _masked_latent_similarity, _token_accuracies_from_generate
from src.validation import _predict_noise_with_guidance


def run_train_diag_decode(
    *,
    args,
    global_step,
    accelerator,
    autoencoder,
    train_iter,
    train_dataloader,
    target_latents,
    noisy_latents,
    predictions,
    t,
    attention_mask,
    labels,
    latent_pca,
    latent_scale,
    noise_scheduler,
    timestep_sampling,
    dit_model,
    prediction_type,
):
    try:
        if hasattr(autoencoder, "no_compression") and autoencoder.no_compression:
            target_total = int(getattr(args, "diag_decode_total", 0) or 0)
            to_cover = max(0, target_total)
            covered = 0
            batch_size = target_latents.shape[0]
            # accumulators
            sum_cos, sum_l2, count_pos = 0.0, 0.0, 0
            sum_accH_len, sum_accH_str = 0.0, 0.0
            sum_accX_len, sum_accX_str = 0.0, 0.0
            batches_used = 0
            # Always include current batch first
            diag_batches = [(target_latents, noisy_latents, predictions, t, attention_mask, labels)]
            # Optionally pull extra batches from the iterator
            if to_cover > 0:
                extra_needed = max(0, to_cover - batch_size)
            else:
                extra_needed = 0
            # Optional progress bar
            diag_pbar = None
            if to_cover > 0 and accelerator.is_main_process:
                try:
                    diag_pbar = tqdm(total=to_cover, desc="Diag decode (train)", leave=False)
                except Exception:
                    diag_pbar = None
            while extra_needed > 0:
                try:
                    nbatch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    nbatch = next(train_iter)
                # Prepare targets for that extra batch
                with torch.no_grad():
                    n_input_ids = nbatch["input_ids"]
                    n_attention_mask = nbatch["attention_mask"]
                    n_labels = nbatch["labels"]
                    n_encoder_outputs = autoencoder.get_encoder()(input_ids=n_input_ids, attention_mask=n_attention_mask)
                    if args.training_phase == "phase2":
                        n_segment_coords = None
                        n_sub_categories = None
                        if args.ablation_mode in ["coords_only", "both"] and "lat" in nbatch and "lon" in nbatch:
                            n_segment_coords = torch.stack([nbatch["lat"], nbatch["lon"]], dim=-1)
                        if args.ablation_mode in ["subcat_only", "both"] and "sub_categories" in nbatch:
                            n_sub_categories = nbatch.get("sub_categories", None)
                        if hasattr(autoencoder, "no_compression") and autoencoder.no_compression:
                            n_enhanced = autoencoder._add_features_no_compression(
                                n_encoder_outputs, n_attention_mask, n_segment_coords, n_sub_categories
                            )
                            n_target_latents = n_enhanced["last_hidden_state"]
                        else:
                            n_target_latents = autoencoder.get_diffusion_latent(
                                n_encoder_outputs, n_attention_mask, n_segment_coords, n_sub_categories
                            )
                    else:
                        n_target_latents = n_encoder_outputs.last_hidden_state
                    if latent_pca is not None:
                        n_target_latents = latent_pca.project(n_target_latents)
                        if latent_scale != 1.0:
                            n_target_latents = n_target_latents / latent_scale

                    n_bs = n_target_latents.shape[0]
                    n_t = noise_scheduler.sample_timesteps(
                        n_bs,
                        device=n_target_latents.device,
                        method=timestep_sampling,
                    )
                    n_noise = torch.randn_like(n_target_latents)
                    n_noisy = noise_scheduler.q_sample(n_target_latents, n_t, n_noise)

                    n_attrs = nbatch.get("attrs", None)
                    if n_attrs is not None and getattr(args, "enable_length_condition", False):
                        n_length_id = nbatch.get("length_id", None)
                        if n_length_id is not None:
                            n_length_tensor = n_length_id.float().unsqueeze(-1).to(n_attrs.device)
                            n_attrs = torch.cat([n_attrs, n_length_tensor], dim=1)
                    n_is_conditional = nbatch.get("is_conditional", None)
                    diag_guidance_scale = getattr(args, "diag_guidance_scale", 1.0)
                    n_predictions = _predict_noise_with_guidance(
                        model=dit_model,
                        noise_scheduler=noise_scheduler,
                        x_t=n_noisy,
                        timesteps=n_t,
                        attrs=n_attrs,
                        prediction_type=prediction_type,
                        guidance_scale=diag_guidance_scale,
                        conditional_mask=n_is_conditional,
                    )
                diag_batches.append((n_target_latents, n_noisy, n_predictions, n_t, n_attention_mask, n_labels))
                extra_needed -= n_bs
            # Iterate diag batches until reaching diag_decode_total
            for (tl, nz, preds, tt, am, lb) in diag_batches:
                if to_cover == 0:
                    K = min(args.diag_decode_batch, tl.shape[0])
                else:
                    remaining = max(0, to_cover - covered)
                    if remaining == 0:
                        break
                    K = min(tl.shape[0], remaining)
                with torch.no_grad():
                    px0 = preds.x_start
                    if latent_scale != 1.0:
                        tl_scaled = tl[:K] * latent_scale
                        px0_scaled = px0[:K] * latent_scale
                    else:
                        tl_scaled = tl[:K]
                        px0_scaled = px0[:K]
                    if latent_pca is not None:
                        tl_decode = latent_pca.unproject(tl_scaled)
                        px0_decode = latent_pca.unproject(px0_scaled)
                    else:
                        tl_decode = tl_scaled
                        px0_decode = px0_scaled
                    s = _masked_latent_similarity(tl[:K], px0[:K], am[:K])
                    sum_cos += s["cosine"] * s["count"]
                    sum_l2 += s["l2"] * s["count"]
                    count_pos += s["count"]
                    reH = BaseModelOutput(last_hidden_state=tl_decode.clone())
                    gH = autoencoder.generate(
                        encoder_outputs=reH,
                        attention_mask=am[:K],
                        max_length=am.size(1),
                        min_length=4,
                        num_beams=getattr(args, "diag_num_beams", 4),
                        length_penalty=1.2,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=0,
                        do_sample=False,
                    )
                    aH = _token_accuracies_from_generate(gH, lb[:K], am[:K], autoencoder)
                    sum_accH_len += aH["lenient"] * K
                    sum_accH_str += aH["strict"] * K
                    reX = BaseModelOutput(last_hidden_state=px0_decode.clone())
                    gX = autoencoder.generate(
                        encoder_outputs=reX,
                        attention_mask=am[:K],
                        max_length=am.size(1),
                        min_length=4,
                        num_beams=getattr(args, "diag_num_beams", 4),
                        length_penalty=1.2,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=0,
                        do_sample=False,
                    )
                    aX = _token_accuracies_from_generate(gX, lb[:K], am[:K], autoencoder)
                    sum_accX_len += aX["lenient"] * K
                    sum_accX_str += aX["strict"] * K
                covered += K
                batches_used += 1
                if diag_pbar is not None:
                    try:
                        diag_pbar.update(K)
                    except Exception:
                        pass
                if to_cover and covered >= to_cover:
                    break
            if diag_pbar is not None:
                try:
                    diag_pbar.close()
                except Exception:
                    pass
            mean_cos = (sum_cos / count_pos) if count_pos > 0 else 0.0
            mean_l2 = (sum_l2 / count_pos) if count_pos > 0 else 0.0
            accH_len = (sum_accH_len / covered) if covered > 0 else 0.0
            accH_str = (sum_accH_str / covered) if covered > 0 else 0.0
            accX_len = (sum_accX_len / covered) if covered > 0 else 0.0
            accX_str = (sum_accX_str / covered) if covered > 0 else 0.0
            print(
                f"[Diag] step {global_step} | cos={mean_cos:.4f} l2={mean_l2:.4f} | "
                f"acc(H') lenient={accH_len:.4f} strict={accH_str:.4f} | "
                f"acc(x0) lenient={accX_len:.4f} strict={accX_str:.4f} | "
                f"covered={covered} batches={batches_used}"
            )
            if args.use_wandb and accelerator.is_main_process and wandb.run is not None:
                wandb.log(
                    {
                        "diag/latent_cosine": mean_cos,
                        "diag/latent_l2": mean_l2,
                        "diag/acc_decode_Hprime_lenient": accH_len,
                        "diag/acc_decode_Hprime_strict": accH_str,
                        "diag/acc_decode_x0_lenient": accX_len,
                        "diag/acc_decode_x0_strict": accX_str,
                        "diag/covered": covered,
                        "diag/step": global_step,
                    }
                )
    except Exception as e:
        print(f"[Diag] Warning: diagnostic decoding failed: {e}")
    return train_iter
