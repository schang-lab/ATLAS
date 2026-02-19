from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch import nn
from transformers import BartForConditionalGeneration

from src.data import CBGConditionCache, POIMarginalStore
from src.diffusion_model import GaussianDiffusion
from src.dit import DiT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DiT with CBG aggregate supervision.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--device", type=str, default=None, help="Override device in config (cpu|cuda).")
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dit(diff_cfg: Dict[str, Any], device: torch.device) -> DiT:
    config_path = diff_cfg["config"]
    checkpoint_path = diff_cfg["checkpoint"]
    with open(config_path, "r", encoding="utf-8") as f:
        dit_params = yaml.safe_load(f)
    if "DiT" in dit_params:
        dit_kwargs = dit_params["DiT"]
    else:
        dit_kwargs = dit_params
    dit = DiT(**dit_kwargs).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = dit.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading DiT: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading DiT: {unexpected}")
    return dit


def build_autoencoder(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """Always load Phase-1 BART autoencoder (no compression)."""
    ae_dir = cfg["path"]
    autoencoder = BartForConditionalGeneration.from_pretrained(ae_dir)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    return autoencoder


def build_condition_encoder(cfg: Dict[str, Any], device: torch.device):
    # Deprecated: condition encoder not used in this finetuning path
    return None


def build_noise_scheduler(diff_cfg: Dict[str, Any]) -> GaussianDiffusion:
    schedule_kwargs = diff_cfg.get("schedule_kwargs", {})
    scheduler = GaussianDiffusion(
        timesteps=int(diff_cfg["timesteps"]),
        schedule=diff_cfg.get("beta_schedule", "linear"),
        schedule_kwargs=schedule_kwargs,
    )
    return scheduler


def select_cbgs(cache: CBGConditionCache, poi_store: POIMarginalStore, allowed: Optional[List[str]]) -> List[str]:
    cache_cbgs = [str(c) for c in cache.available_cbgs()]
    poi_cbgs = [str(c) for c in poi_store.available_cbgs()]
    available = set(cache_cbgs) & set(poi_cbgs)
    if allowed:
        allowed_set = set(allowed)
        available &= allowed_set
    if not available:
        print("[DEBUG] cache_cbgs (first 10):", cache_cbgs[:10])
        print("[DEBUG] poi_cbgs   (first 10):", poi_cbgs[:10])
        print("[DEBUG] allowed_cbgs:", allowed)
        print("[DEBUG] overlap:", sorted(available))
        raise ValueError("No overlapping CBGs between cache and POI marginals.")
    return sorted(available)


@torch.no_grad()
def _posterior_sample(
    diffusion: GaussianDiffusion,
    preds: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    posterior_mean, _, posterior_log_var = diffusion.q_posterior(
        x_start=preds,
        x_t=x_t,
        t=t,
    )
    noise = torch.randn_like(x_t)
    return posterior_mean + torch.exp(0.5 * posterior_log_var) * noise


def sample_latents(
    diffusion: GaussianDiffusion,
    dit: DiT,
    cond: torch.Tensor,
    *,
    seq_len: int,
    latent_dim: int,
    prediction_type: str,
    guidance_scale: float,
    sampler: str = "ddpm",
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
    device: torch.device,
) -> torch.Tensor:
    diffusion = diffusion.to(device)
    latents = torch.randn(cond.size(0), seq_len, latent_dim, device=device)
    timesteps = diffusion.num_timesteps

    sampler = str(sampler or "ddpm").lower().strip()
    if sampler not in {"ddpm", "ddim"}:
        raise ValueError(f"sampler must be one of ['ddpm','ddim'] (got {sampler!r})")

    if sampler == "ddpm":
        for idx in reversed(range(timesteps)):
            t = torch.full((latents.size(0),), idx, device=device, dtype=torch.long)
            if idx == 0:
                model_out = GaussianDiffusion.classifier_free_guidance(
                    denoiser=dit,
                    x_t=latents,
                    t=t,
                    conditional_attrs=cond,
                    guidance_scale=guidance_scale,
                )
                preds = diffusion.model_predictions(model_out, latents, t, prediction_type=prediction_type)
                latents = preds.x_start
            else:
                with torch.no_grad():
                    model_out = GaussianDiffusion.classifier_free_guidance(
                        denoiser=dit,
                        x_t=latents,
                        t=t,
                        conditional_attrs=cond,
                        guidance_scale=guidance_scale,
                    )
                    preds = diffusion.model_predictions(model_out, latents, t, prediction_type=prediction_type)
                    latents = _posterior_sample(diffusion, preds.x_start, latents, t)
        return latents

    steps = int(ddim_steps)
    if steps < 2:
        steps = 2
    steps = min(steps, int(timesteps))
    times = np.linspace(0, timesteps - 1, num=steps, dtype=np.int64)
    times = np.unique(times)
    if times[-1] != (timesteps - 1):
        times = np.unique(np.concatenate([times, np.array([timesteps - 1], dtype=np.int64)]))
    if times[0] != 0:
        times = np.unique(np.concatenate([np.array([0], dtype=np.int64), times]))
    times = times[::-1]

    eta = float(ddim_eta or 0.0)
    for i in range(len(times) - 1):
        t_curr_i = int(times[i])
        t_next_i = int(times[i + 1])
        if t_curr_i <= 0:
            break
        t_curr = torch.full((latents.size(0),), t_curr_i, device=device, dtype=torch.long)
        t_next = torch.full((latents.size(0),), t_next_i, device=device, dtype=torch.long)
        with torch.no_grad():
            model_out = GaussianDiffusion.classifier_free_guidance(
                denoiser=dit,
                x_t=latents,
                t=t_curr,
                conditional_attrs=cond,
                guidance_scale=guidance_scale,
            )
            preds = diffusion.model_predictions(model_out, latents, t_curr, prediction_type=prediction_type)
            latents = diffusion.ddim_sample(latents, t_curr, t_next, preds.noise, eta=eta)

    t0 = torch.zeros((latents.size(0),), device=device, dtype=torch.long)
    model_out = GaussianDiffusion.classifier_free_guidance(
        denoiser=dit,
        x_t=latents,
        t=t0,
        conditional_attrs=cond,
        guidance_scale=guidance_scale,
    )
    preds0 = diffusion.model_predictions(model_out, latents, t0, prediction_type=prediction_type)
    return preds0.x_start
