from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.diffusion_model import GaussianDiffusion


def shift_demo_ids_in_attrs(
    attrs: Optional[torch.Tensor],
    dit_model: torch.nn.Module,
) -> Optional[torch.Tensor]:
    """
    Shift raw demo ids (age, gender) from 0-based to 1-based, keeping 0 as null/padding.
    """
    if attrs is None:
        return None
    if attrs.dim() != 2 or attrs.size(1) < 6:
        return attrs

    base_dit = getattr(dit_model, "module", dit_model)
    attr_block = getattr(base_dit, "attr_embed", None)
    if attr_block is None or not getattr(attr_block, "use_demo_condition", False):
        return attrs

    age_idx = -2
    gender_idx = -1
    age_raw = attrs[:, age_idx].long()
    gender_raw = attrs[:, gender_idx].long()

    missing = (age_raw < 0) | (gender_raw < 0)
    age_clamped = age_raw.clamp_min(0)
    gender_clamped = gender_raw.clamp_min(0)
    age_shifted = (age_clamped + 1).to(dtype=attrs.dtype, device=attrs.device)
    gender_shifted = (gender_clamped + 1).to(dtype=attrs.dtype, device=attrs.device)

    if missing.any():
        age_shifted = age_shifted.clone()
        gender_shifted = gender_shifted.clone()
        age_shifted[missing] = 0
        gender_shifted[missing] = 0

    attrs_out = attrs.clone()
    attrs_out[:, age_idx] = age_shifted
    attrs_out[:, gender_idx] = gender_shifted
    return attrs_out


@torch.no_grad()
def run_ddim_probe(
    model: torch.nn.Module,
    noise_scheduler: GaussianDiffusion,
    steps: int,
    batch_size: int,
    seq_len: int,
    latent_dim: int,
    device: torch.device,
    prediction_type: str = "epsilon",
) -> dict:
    """Deterministic DDIM sampling in compressed space; returns probe logs per step."""

    z = torch.randn(batch_size, seq_len, latent_dim, device=device)
    stride = max(1, noise_scheduler.num_timesteps // steps)
    timestep_schedule = noise_scheduler._get_ddim_sampling_timesteps(
        batch=batch_size,
        device=device,
        step=stride,
    )
    if not torch.all(timestep_schedule[-1] == 0):
        timestep_schedule.append(torch.zeros_like(timestep_schedule[-1]))

    logs = {
        "z_norm": [],
        "z0hat_token_norm": [],
        "eps_norm": [],
        "z_mean_abs_mu": [],
        "z_mean_abs_sigma_dev": [],
        "eps_cos_proxy": [],
    }

    for idx, t in enumerate(timestep_schedule):
        raw_out = model(x=z, t=t, attr_embeds=None)
        preds = noise_scheduler.model_predictions(raw_out, z, t, prediction_type)
        eps_pred = preds.noise
        z0_hat = preds.x_start

        logs["z_norm"].append(z.norm(dim=-1).mean().item())
        logs["z0hat_token_norm"].append(z0_hat.norm(dim=-1).mean().item())
        logs["eps_norm"].append(eps_pred.norm(dim=-1).mean().item())
        flat = z0_hat.reshape(-1, latent_dim)
        mu = flat.mean(dim=0)
        sd = flat.std(dim=0).clamp_min(1e-6)
        logs["z_mean_abs_mu"].append(mu.abs().mean().item())
        logs["z_mean_abs_sigma_dev"].append((sd - 1.0).abs().mean().item())
        cos = F.cosine_similarity(z.reshape(batch_size, -1), eps_pred.reshape(batch_size, -1), dim=-1).mean().item()
        logs["eps_cos_proxy"].append(cos)

        if idx == len(timestep_schedule) - 1:
            break

        t_prev = timestep_schedule[idx + 1]
        z = noise_scheduler.ddim_sample(
            z,
            t,
            t_prev,
            eps_pred,
            eta=0.0,
        )

    return logs
