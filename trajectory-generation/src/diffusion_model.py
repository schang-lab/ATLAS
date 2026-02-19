import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn

from .helpers import default, log, extract, normalize_prediction_type


@dataclass
class ModelPredictions:
    """Container for harmonized diffusion predictions."""

    noise: torch.Tensor
    x_start: torch.Tensor
    v: torch.Tensor
    model_output: torch.Tensor


def _betas_for_linear_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    scale = 1000.0 / timesteps
    beta_start_scaled = beta_start * scale
    beta_end_scaled = beta_end * scale
    return torch.linspace(beta_start_scaled, beta_end_scaled, timesteps, dtype=torch.float64)


def _betas_for_cosine_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)

    def _alpha_bar_fn(u: torch.Tensor) -> torch.Tensor:
        return torch.cos(((u / timesteps + s) / (1 + s)) * math.pi / 2) ** 2

    alphas_bar = _alpha_bar_fn(t)
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(min=1e-8, max=0.999)


def _betas_for_logsnr_linear_schedule(
    timesteps: int,
    logsnr_max: float = 30.0,
    logsnr_min: float = -2.0,
) -> torch.Tensor:
    t = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
    log_snr = logsnr_max + (logsnr_min - logsnr_max) * t
    alphas_bar = torch.sigmoid(log_snr)
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(min=1e-8, max=0.999)


class GaussianDiffusion(nn.Module):

    def __init__(
            self,
            *,
            timesteps: int,
            schedule: str = "linear",
            schedule_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        :param timesteps: Number of timesteps in the Diffusion Process.
        """
        super().__init__()

        # Timesteps < 20 => scale > 50 => beta_end > 1 => alphas[-1] < 0 => sqrt_alphas_cumprod[-1] is NaN
        assert not timesteps < 20,  f'timsteps must be at least 20'
        self.num_timesteps = timesteps

        schedule = schedule.lower()
        schedule_kwargs = schedule_kwargs or {}

        if schedule == "linear":
            betas = _betas_for_linear_schedule(timesteps, **schedule_kwargs)
        elif schedule == "cosine":
            betas = _betas_for_cosine_schedule(timesteps, **schedule_kwargs)
        elif schedule in {"logsnr", "logsnr_linear", "log-snr"}:
            betas = _betas_for_logsnr_linear_schedule(timesteps, **schedule_kwargs)
        else:
            raise ValueError(f"Unsupported beta schedule '{schedule}'")

        self.beta_schedule = schedule

        # Diffusion model constants/buffers. See https://arxiv.org/pdf/2006.11239.pdf
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0) # $\prod_{i=1}^t \alpha_i$
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # register buffer helper function
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)

        # Register variance schedule related buffers
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Buffer for diffusion calculations q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        log_snr = torch.log(alphas_cumprod.clamp(min=1e-20)) - torch.log((1. - alphas_cumprod).clamp(min=1e-20))
        register_buffer('log_snr', log_snr)
        self._log_snr_min = float(log_snr.min().item())
        self._log_snr_max = float(log_snr.max().item())

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        # Posterior variance:
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # Clipped because posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', log(posterior_variance, eps=1e-20))

        # Buffers for calculating the q_posterior mean $\~{\mu}$.
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def _get_times(self, batch_size: int, noise_level: float, *, device: torch.device) -> torch.tensor:
        return torch.full((batch_size,), int(self.num_timesteps * noise_level), device=device, dtype=torch.long)

    def _sample_random_times(self, batch_size: int, *, device: torch.device) -> torch.tensor:
        """Sample discrete timesteps uniformly in index space."""

        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    def _sample_random_times_log_snr(self, batch_size: int, *, device: torch.device) -> torch.tensor:
        """Sample timesteps uniformly in log-SNR space."""

        log_snr = self.log_snr.to(device=device)
        log_snr_range = self._log_snr_max - self._log_snr_min
        if abs(log_snr_range) < 1e-8:
            return self._sample_random_times(batch_size, device=device)
        target = torch.rand(batch_size, device=device) * log_snr_range + self._log_snr_min
        diff = torch.abs(target.unsqueeze(1) - log_snr.unsqueeze(0))
        indices = torch.argmin(diff, dim=1)
        return indices.to(dtype=torch.long)

    def sample_timesteps(self, batch_size: int, *, device: torch.device, method: str = "uniform") -> torch.tensor:
        """Sample timesteps according to the requested distribution."""

        method_normalized = method.lower().replace('-', '_')
        if method_normalized in {"uniform", "uniform_time", "time", "index"}:
            return self._sample_random_times(batch_size, device=device)
        if method_normalized in {"logsnr", "log_snr", "uniform_log_snr", "logsnr_uniform"}:
            return self._sample_random_times_log_snr(batch_size, device=device)
        raise ValueError(f"Unsupported timestep sampling method '{method}'")

    def _get_sampling_timesteps(self, batch: int, *, device: torch.device) -> list[torch.tensor]:
        time_transitions = []

        for i in reversed(range(self.num_timesteps)):
            time_transitions.append((torch.full((batch,), i, device=device, dtype=torch.long)))

        return time_transitions

    def q_posterior(self, x_start: torch.tensor, x_t: torch.tensor, t: torch.tensor) -> tuple[torch.tensor,
                                                                                              torch.tensor,
                                                                                              torch.tensor]:
        """
        Calculates q_posterior parameters given a starting image
        :code:`x_start` (x_0) and a noised image :code:`x_t`.
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # Extract the value corresponding to the current time from the buffers, and then reshape to (b, 1, 1, 1)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start: torch.tensor, t: torch.tensor, noise: torch.tensor = None) -> torch.tensor:

        noise = default(noise, lambda: torch.randn_like(x_start))

        noised = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return noised

    def predict_start_from_noise(self, x_t: torch.tensor, t: torch.tensor, noise: torch.tensor) -> torch.tensor:
        """
        Given a noised image and its noise component, calculated the unnoised image :code:`x_0`.
        """
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        """Compute the noise component given a clean sample prediction."""

        sqrt_alpha_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        denom = sqrt_one_minus_alpha_cumprod.clamp_min(1e-8)
        return (x_t - sqrt_alpha_cumprod * x_start) / denom

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Recover x0 given the model's v prediction."""

        sqrt_alpha_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha_cumprod * x_t - sqrt_one_minus_alpha_cumprod * v

    def predict_noise_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Recover noise given the model's v prediction."""

        sqrt_alpha_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha_cumprod * v + sqrt_one_minus_alpha_cumprod * x_t

    def predict_v_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        """Compute v given a clean sample prediction."""

        noise = self.predict_noise_from_start(x_t, t, x_start)
        sqrt_alpha_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * x_start

    def model_predictions(
            self,
            model_output: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor,
            prediction_type: str = "epsilon"
    ) -> ModelPredictions:
        """Convert model output into both noise and x_start predictions."""

        pred_type = normalize_prediction_type(prediction_type)

        if pred_type == "epsilon":
            noise = model_output
            x_start = self.predict_start_from_noise(x_t, t, noise)
            v = self.predict_v_from_start(x_t, t, x_start)
        elif pred_type == "x_start":
            x_start = model_output
            noise = self.predict_noise_from_start(x_t, t, x_start)
            v = self.predict_v_from_start(x_t, t, x_start)
        elif pred_type == "v":
            v = model_output
            x_start = self.predict_start_from_v(x_t, t, v)
            noise = self.predict_noise_from_v(x_t, t, v)
        else:
            raise ValueError(f"Unsupported prediction type '{prediction_type}'")

        return ModelPredictions(noise=noise, x_start=x_start, v=v, model_output=model_output)

    # for ddim
    def _get_ddim_sampling_timesteps(self, batch: int, *, device: torch.device, step: int) -> list[torch.tensor]:
        time_transitions = []

        skipped_time_steps = range(0, self.num_timesteps, step)

        for i in reversed(skipped_time_steps):
            time_transitions.append((torch.full((batch,), i, device=device, dtype=torch.long)))

        return time_transitions

    # refer to https://github.com/ermongroup/ddim/blob/main/functions/denoising.py
    def compute_alpha(self, t):
        beta = torch.cat([torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
        return a

    def ddim_sample(self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor,
                    eps: torch.Tensor, eta: float = 0.0) -> torch.Tensor:

        # alpha_bar_t
        # alpha_bar_t = extract(self.alphas_cumprod, t+1, x.shape)
        # # t-1
        # alpha_bar_prev = extract(self.alphas_cumprod, t_prev+1, x.shape)

        alpha_bar_t = self.compute_alpha(t)
        alpha_bar_prev = self.compute_alpha(t_prev)

        pred_x0 = torch.sqrt(1. / alpha_bar_t) * x - torch.sqrt(1. / alpha_bar_t - 1)*eps

        # sigma_t
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(
            1 - alpha_bar_t / alpha_bar_prev)
        # sample noise
        noise = torch.randn_like(x) if eta > 0.0 else 0.
        # update x_{t-1}
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + \
                 torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * eps + \
                 sigma_t * noise

        return x_prev

    def ddim_sample_from_xstart(self, x, t, t_prev, x_start, eta=0.0):
        alpha_bar_t = self.compute_alpha(t)
        alpha_bar_prev = self.compute_alpha(t_prev)

        eps = (x - torch.sqrt(alpha_bar_t) * x_start) / torch.sqrt(1 - alpha_bar_t)

        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        noise = torch.randn_like(x) if eta > 0.0 else 0.
        x_prev = torch.sqrt(alpha_bar_prev) * x_start + \
                 torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * eps + \
                 sigma_t * noise
        return x_prev

    @staticmethod
    def classifier_free_guidance(
            denoiser: nn.Module,
            x_t: torch.Tensor,
            t: torch.Tensor,
            *,
            conditional_attrs: Optional[torch.Tensor],
            guidance_scale: Optional[float],
            unconditional_attrs: Optional[torch.Tensor] = None,
            conditional_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict denoiser output using classifier-free guidance.

        Args:
            denoiser: Model (e.g. DiT) that predicts noise.
            x_t: Current latents at timestep ``t`` with shape (B, ...).
            t: Timesteps tensor with shape (B,).
            conditional_attrs: Attribute tensor for the conditional branch. ``None`` disables CFG.
            guidance_scale: Guidance strength. 1.0 recovers the conditional prediction, 0.0 the unconditional.
            unconditional_attrs: Optional tensor for the unconditional branch. When ``None`` it is derived
                by zeroing conditional rows in ``conditional_attrs``.
            conditional_mask: Boolean tensor (B,) selecting rows that should be treated as conditional.

        Returns:
            Guided noise prediction tensor matching the shape of ``x_t``.
        """

        if conditional_attrs is None or guidance_scale is None:
            return denoiser(x=x_t, t=t, attr_embeds=conditional_attrs)

        if isinstance(guidance_scale, torch.Tensor):
            guidance_scale = float(guidance_scale.item())

        # Skip extra work when guidance is effectively disabled.
        if abs(guidance_scale - 1.0) < 1e-6:
            return denoiser(x=x_t, t=t, attr_embeds=conditional_attrs)

        if conditional_mask is None:
            conditional_mask = conditional_attrs.abs().sum(dim=1) > 0
        conditional_mask = conditional_mask.bool()

        if unconditional_attrs is None:
            unconditional_attrs = conditional_attrs.clone()
            unconditional_attrs[conditional_mask] = 0

        if abs(guidance_scale) < 1e-6:
            return denoiser(x=x_t, t=t, attr_embeds=unconditional_attrs)

        x_cat = torch.cat([x_t, x_t], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        attrs_cat = torch.cat([unconditional_attrs, conditional_attrs], dim=0)

        model_out = denoiser(x=x_cat, t=t_cat, attr_embeds=attrs_cat)
        out_uncond, out_cond = model_out.chunk(2, dim=0)

        expand_dims = (-1,) + (1,) * (out_uncond.ndim - 1)
        mask = conditional_mask.view(*expand_dims).to(out_uncond.dtype)
        guidance = guidance_scale * (out_cond - out_uncond)
        guidance = guidance * mask

        return out_uncond + guidance
