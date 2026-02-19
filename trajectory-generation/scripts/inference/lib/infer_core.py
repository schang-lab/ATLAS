from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

# Ensure trajectory-generation root is on sys.path when imported standalone.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)

from src.helpers import normalize_prediction_type
from src.latent_pca import LatentPCA
from src.validation import _predict_noise_with_guidance


class ForbiddenTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, forbidden_token_ids: List[int]):
        super().__init__()
        self.forbidden_token_ids = sorted({int(t) for t in forbidden_token_ids})

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores is None or scores.numel() == 0 or not self.forbidden_token_ids:
            return scores
        vocab_size = scores.size(-1)
        for tid in self.forbidden_token_ids:
            if 0 <= tid < vocab_size:
                scores[:, tid] = float("-inf")
        return scores


class UnkTokenPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, unk_token_ids: List[int], penalty: float = 5.0):
        super().__init__()
        self.unk_token_ids = sorted(set(int(t) for t in unk_token_ids))
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty <= 0 or scores is None or scores.numel() == 0:
            return scores
        vocab_size = scores.size(-1)
        for tid in self.unk_token_ids:
            if 0 <= tid < vocab_size:
                scores[:, tid] = scores[:, tid] - self.penalty
        return scores


class TokenSpecificPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, token_penalties: Dict[int, float]):
        super().__init__()
        self.token_penalties = {int(k): float(v) for k, v in token_penalties.items() if float(v) > 0}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.token_penalties or scores is None or scores.numel() == 0:
            return scores
        vocab_size = scores.size(-1)
        for tid, penalty in self.token_penalties.items():
            if 0 <= tid < vocab_size:
                scores[:, tid] = scores[:, tid] - penalty
        return scores


def convert_poi_sequences_to_coordinates_infer(
    poi_sequences: torch.Tensor,
    vocab: Dict[str, int],
    poi_coords: Dict[str, Tuple[float, float]],
    attrs_4d: np.ndarray,
    max_length: int,
    poi_home_token: str = "POI_HOME",
    poi_work_token: str = "POI_WORK",
    poi_other_token: str = "POI_OTHER",
) -> np.ndarray:
    """
    Convert decoded POI token sequences to coordinate trajectories.

    - Skips all BART special tokens for coordinate conversion.
    - Skips POI_OTHER entirely (no coordinates).
    - Injects coordinates for POI_HOME/POI_WORK using attrs_4d.
    """
    if attrs_4d.ndim != 2 or attrs_4d.shape[1] < 4:
        raise ValueError(f"attrs_4d must be (B, >=4) [work_lat, work_lon, home_lat, home_lon], got {attrs_4d.shape}")

    batch_size, seq_len = poi_sequences.shape
    if attrs_4d.shape[0] != batch_size:
        raise ValueError(f"attrs rows must match batch size: attrs={attrs_4d.shape[0]} vs seq_batch={batch_size}")

    bart_special_ids = {0, 1, 2, 3, 4}
    id_to_token = {v: k for k, v in vocab.items()}

    out = np.zeros((batch_size, 2, max_length), dtype=np.float32)
    kept = 0
    skipped_other = 0
    skipped_special = 0
    missing_map = 0
    injected = 0

    for b in range(batch_size):
        work_lat, work_lon, home_lat, home_lon = attrs_4d[b, 0:4].tolist()
        out_idx = 0
        for t in range(seq_len):
            if out_idx >= max_length:
                break
            tid = int(poi_sequences[b, t].item())
            if tid in bart_special_ids:
                skipped_special += 1
                continue
            tok = id_to_token.get(tid, "[UNK]")

            if tok == poi_other_token:
                skipped_other += 1
                continue
            if tok == poi_home_token:
                out[b, 0, out_idx] = float(home_lat)
                out[b, 1, out_idx] = float(home_lon)
                out_idx += 1
                injected += 1
                continue
            if tok == poi_work_token:
                out[b, 0, out_idx] = float(work_lat)
                out[b, 1, out_idx] = float(work_lon)
                out_idx += 1
                injected += 1
                continue

            if tok in poi_coords:
                lat, lon = poi_coords[tok]
                out[b, 0, out_idx] = float(lat)
                out[b, 1, out_idx] = float(lon)
                out_idx += 1
                kept += 1
            else:
                missing_map += 1

    total_seen = kept + missing_map + injected
    if total_seen > 0:
        miss_rate = missing_map / total_seen
        print(
            f"[Infer] coord conversion: kept={kept}, injected_home/work={injected}, "
            f"generated_other={skipped_other}, skipped_special={skipped_special}, "
            f"missing_map={missing_map}, miss_rate={miss_rate:.3f}"
        )
    return out


def convert_poi_sequences_to_poi_ids_infer(
    poi_sequences: torch.Tensor,
    vocab: Dict[str, int],
    max_length: int,
    poi_other_token: str,
) -> List[List[str]]:
    batch_size, seq_len = poi_sequences.shape
    filter_special_ids = {0, 1, 2, 4}
    id_to_token = {v: k for k, v in vocab.items()}
    sequences: List[List[str]] = []
    dropped_other = 0

    for b in range(batch_size):
        out: List[str] = []
        for t in range(seq_len):
            if len(out) >= max_length:
                break
            tid = int(poi_sequences[b, t].item())
            if tid in filter_special_ids:
                continue
            tok = id_to_token.get(tid, "[UNK]")
            if tok == poi_other_token:
                dropped_other += 1
                continue
            out.append(tok)
        sequences.append(out)

    if dropped_other:
        print(f"[Infer] dropped {dropped_other} occurrences of {poi_other_token} in POI-id sequences")
    return sequences


def convert_poi_sequences_to_coordinates_carlos(*args, **kwargs):
    """Backward-compatible alias for convert_poi_sequences_to_coordinates_infer."""
    return convert_poi_sequences_to_coordinates_infer(*args, **kwargs)


def convert_poi_sequences_to_poi_ids_carlos(*args, **kwargs):
    """Backward-compatible alias for convert_poi_sequences_to_poi_ids_infer."""
    return convert_poi_sequences_to_poi_ids_infer(*args, **kwargs)


@torch.no_grad()
def sample_dit_with_autoencoder(
    dit_model,
    noise_scheduler,
    autoencoder,
    attr_embeds: Optional[torch.Tensor],
    timesteps: int,
    num_inference_steps: int,
    guidance_scale: float,
    max_traj_length: int,
    min_traj_length: int,
    num_beams: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    training_phase: str,
    generation_config: dict,
    actual_batch_size: Optional[int] = None,
    ddim_eta: float = 0.0,
    latent_scale: float = 1.0,
    latent_pca: Optional[LatentPCA] = None,
    latent_mapper: Optional[torch.nn.Module] = None,
    length_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
    unk_token_ids: Optional[List[int]] = None,
    prediction_type: str = "epsilon",
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(dit_model.parameters()).device
    model_dtype = getattr(dit_model, "dtype", torch.float32)
    if attr_embeds is not None:
        attr_embeds = attr_embeds.to(device=device, dtype=model_dtype)

    batch_size = actual_batch_size if actual_batch_size is not None else (attr_embeds.shape[0] if attr_embeds is not None else 1)
    prediction_type = normalize_prediction_type(prediction_type)

    if hasattr(dit_model.input_proj, "__getitem__"):
        input_dim = dit_model.input_proj[0].in_features
    else:
        input_dim = dit_model.input_proj.in_features
    latent_shape = (batch_size, dit_model.image_size, input_dim)

    latents = torch.randn(latent_shape, device=device, dtype=dit_model.dtype)
    timesteps_tensor = torch.linspace(timesteps - 1, 0, num_inference_steps, device=device).long()

    conditional_mask = None
    if attr_embeds is not None:
        conditional_mask = attr_embeds.abs().sum(dim=1) > 0

    dit_model.eval()
    for i, t in enumerate(tqdm(timesteps_tensor, desc="Diffusion steps", leave=False)):
        t_batch = t.expand(batch_size)
        preds = _predict_noise_with_guidance(
            model=dit_model,
            noise_scheduler=noise_scheduler,
            x_t=latents,
            timesteps=t_batch,
            attrs=attr_embeds,
            prediction_type=prediction_type,
            guidance_scale=guidance_scale,
            conditional_mask=conditional_mask,
        )
        noise_pred = preds.noise
        x0_pred = preds.x_start

        if i < len(timesteps_tensor) - 1:
            next_t = timesteps_tensor[i + 1]
            latents = noise_scheduler.ddim_sample(
                latents,
                t_batch,
                next_t.expand(batch_size),
                noise_pred,
                eta=ddim_eta,
            )
        else:
            latents = x0_pred

    mapped_latents = latents
    if latent_mapper is not None:
        mapped_latents = latent_mapper(latents)
    ae_latents = latent_pca.unproject(mapped_latents) if latent_pca is not None else mapped_latents
    decoder_inputs = ae_latents * latent_scale if latent_scale != 1.0 else ae_latents

    bs = decoder_inputs.shape[0]
    seq_len = decoder_inputs.shape[1]
    attention_mask = torch.ones(bs, seq_len, device=latents.device, dtype=torch.bool)

    length_tensor = None
    if length_ids is not None:
        if isinstance(length_ids, torch.Tensor):
            length_tensor = length_ids.to(device=decoder_inputs.device, dtype=torch.long).view(-1)
        else:
            length_tensor = torch.as_tensor(length_ids, device=decoder_inputs.device, dtype=torch.long).view(-1)
        if length_tensor.numel() != bs:
            raise ValueError(f"Length id count mismatch: expected {bs}, received {length_tensor.numel()}")

    forbidden_token_ids = generation_config.get("forbidden_token_ids", [])
    logits_processors = []
    if forbidden_token_ids:
        logits_processors.append(ForbiddenTokensLogitsProcessor(forbidden_token_ids=forbidden_token_ids))

    unk_penalty = generation_config.get("unk_logit_penalty", 0.0)
    if unk_penalty and unk_penalty > 0:
        penalty_ids = [] if unk_token_ids is None else [int(t) for t in unk_token_ids if t is not None]
        if penalty_ids:
            logits_processors.append(UnkTokenPenaltyLogitsProcessor(unk_token_ids=penalty_ids, penalty=unk_penalty))

    poi_special_token_ids = generation_config.get("poi_special_token_ids", [])
    poi_special_penalty = generation_config.get("poi_special_token_penalty", 0.0)
    if poi_special_penalty > 0.0 and poi_special_token_ids:
        logits_processors.append(UnkTokenPenaltyLogitsProcessor(unk_token_ids=poi_special_token_ids, penalty=poi_special_penalty))

    poi_home_work_token_ids = generation_config.get("poi_home_work_token_ids", [])
    poi_home_work_penalty = generation_config.get("poi_home_work_penalty", 0.0)
    if poi_home_work_penalty > 0.0 and poi_home_work_token_ids:
        logits_processors.append(UnkTokenPenaltyLogitsProcessor(unk_token_ids=poi_home_work_token_ids, penalty=poi_home_work_penalty))

    token_specific_penalties = {}
    poi_home_token_id = generation_config.get("poi_home_token_id")
    poi_home_penalty = generation_config.get("poi_home_penalty", 0.0)
    if poi_home_token_id is not None and poi_home_penalty > 0.0:
        token_specific_penalties[poi_home_token_id] = poi_home_penalty
    poi_work_token_id = generation_config.get("poi_work_token_id")
    poi_work_penalty = generation_config.get("poi_work_penalty", 0.0)
    if poi_work_token_id is not None and poi_work_penalty > 0.0:
        token_specific_penalties[poi_work_token_id] = poi_work_penalty
    poi_other_token_id = generation_config.get("poi_other_token_id")
    poi_other_penalty = generation_config.get("poi_other_penalty", 0.0)
    if poi_other_token_id is not None and poi_other_penalty > 0.0:
        token_specific_penalties[poi_other_token_id] = poi_other_penalty
    if token_specific_penalties:
        logits_processors.append(TokenSpecificPenaltyLogitsProcessor(token_penalties=token_specific_penalties))

    decoder_start_id = getattr(autoencoder.config, "decoder_start_token_id", None)
    decoder_offset = 1 if decoder_start_id is not None else 0
    eos_token_id = getattr(autoencoder.config, "eos_token_id", None)
    eos_offset = 1 if eos_token_id is not None else 0
    extra_specials = decoder_offset + eos_offset
    pad_token_id = getattr(autoencoder.config, "pad_token_id", 0)
    decoder_start_token_id = int(decoder_start_id) if decoder_start_id is not None else 1
    eos_token_id_for_generate = int(eos_token_id) if eos_token_id is not None else 2
    pad_token_id_for_generate = int(pad_token_id) if pad_token_id is not None else 0
    bart_max_pos = getattr(autoencoder.config, "max_position_embeddings", 64)

    def _run_generation(hidden_states: torch.Tensor, attn_mask: torch.Tensor, target_tokens: Optional[int] = None):
        max_allowed = min(max_traj_length + extra_specials, int(bart_max_pos))
        if max_allowed <= extra_specials:
            raise ValueError(
                f"max_length ({max_allowed}) is too small after applying constraints. "
                f"max_traj_length={max_traj_length}, bart_max_pos={bart_max_pos}, extra_specials={extra_specials}"
            )
        content_cap = max(1, int(max_allowed - extra_specials))
        if target_tokens is not None:
            desired = max(1, int(target_tokens))
            if desired > content_cap:
                desired = content_cap
            max_length_local = int(desired + extra_specials)
            min_length_local = int(max_length_local)
        else:
            max_length_local = int(max_allowed)
            min_length_local = int(max(min_traj_length + extra_specials, extra_specials + 1))
            if min_length_local > max_length_local:
                min_length_local = max_length_local

        return autoencoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden_states, hidden_states=None, attentions=None),
            attention_mask=attn_mask,
            max_length=max_length_local,
            min_length=min_length_local,
            num_beams=num_beams if not do_sample else 1,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=pad_token_id_for_generate,
            eos_token_id=eos_token_id_for_generate,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=True,
            repetition_penalty=generation_config.get("repetition_penalty", 1.1),
            length_penalty=generation_config.get("length_penalty", 1.2),
            no_repeat_ngram_size=generation_config.get("no_repeat_ngram_size", 0),
            logits_processor=LogitsProcessorList(logits_processors),
            output_scores=True,
            return_dict_in_generate=True,
        )

    if length_tensor is None:
        generate_output = _run_generation(decoder_inputs, attention_mask, target_tokens=None)
        generated_ids = generate_output.sequences if hasattr(generate_output, "sequences") else generate_output
    else:
        length_values = length_tensor.detach().cpu().tolist()
        index_groups: Dict[int, List[int]] = {}
        for idx, length_value in enumerate(length_values):
            index_groups.setdefault(int(length_value), []).append(int(idx))
        per_sample_sequences: List[Optional[torch.Tensor]] = [None] * bs
        print("  [Infer] Enforcing empirical length targets per sample:")
        for raw_length, sample_indices in index_groups.items():
            target_tokens = int(raw_length)
            adjusted_tokens = target_tokens
            if adjusted_tokens <= 0:
                print(f"    Warning: requested length {target_tokens} <= 0; forcing to 1 token")
                adjusted_tokens = 1
            if adjusted_tokens > max_traj_length:
                adjusted_tokens = int(max_traj_length)
            subset_hidden = decoder_inputs[sample_indices]
            subset_mask = attention_mask[sample_indices]
            subset_output = _run_generation(subset_hidden, subset_mask, target_tokens=adjusted_tokens)
            subset_sequences = subset_output.sequences if hasattr(subset_output, "sequences") else subset_output
            for local_idx, sample_idx in enumerate(sample_indices):
                per_sample_sequences[sample_idx] = subset_sequences[local_idx : local_idx + 1]
        if any(seq is None for seq in per_sample_sequences):
            missing = [i for i, seq in enumerate(per_sample_sequences) if seq is None]
            raise RuntimeError(f"Missing sequences for indices: {missing}")
        max_seq_len = max(seq.shape[1] for seq in per_sample_sequences if seq is not None)
        padded_sequences = []
        for seq in per_sample_sequences:
            assert seq is not None
            if seq.shape[1] < max_seq_len:
                pad = torch.zeros(seq.shape[0], max_seq_len - seq.shape[1], dtype=seq.dtype, device=seq.device)
                padded_sequences.append(torch.cat([seq, pad], dim=1))
            else:
                padded_sequences.append(seq)
        generated_ids = torch.cat(padded_sequences, dim=0)

    return generated_ids, ae_latents
