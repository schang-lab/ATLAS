"""Lightweight mapper modules that align DiT latent predictions with the decoder Z-space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


@dataclass
class LatentMapperConfig:
    """Configuration for :class:`LatentMapper` construction."""

    latent_dim: int
    hidden_dim: int | None = None
    num_hidden_layers: int = 2
    dropout: float = 0.0
    activation: str = "gelu"
    use_layernorm: bool = True


class LatentMapper(nn.Module):
    """Small MLP with residual connection to align DiT latents with BART decoder expectations."""

    def __init__(self, config: LatentMapperConfig):
        super().__init__()
        hidden_dim = config.hidden_dim or config.latent_dim
        if config.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        act = _get_activation(config.activation)

        layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for layer_idx in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act)
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            if layer_idx != config.num_hidden_layers - 1:
                in_dim = hidden_dim

        # Final projection back to latent dimension
        layers.append(nn.Linear(hidden_dim, config.latent_dim))

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(config.latent_dim) if config.use_layernorm else nn.Identity()

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply mapper with residual connection.

        Args:
            latents: Tensor of shape ``(batch, seq_len, latent_dim)`` in Z-space.

        Returns:
            Tensor of same shape after alignment transformation.
        """
        residual = latents
        aligned = self.mlp(latents)
        aligned = aligned + residual
        return self.norm(aligned)


def build_latent_mapper(latent_dim: int, **kwargs) -> LatentMapper:
    """Convenience constructor that mirrors :class:`LatentMapperConfig` arguments."""
    config = LatentMapperConfig(latent_dim=latent_dim, **kwargs)
    return LatentMapper(config)


def collect_decoder_cross_attention_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return decoder cross-attention parameters (k/v/out projections) for fine-tuning."""
    params: List[nn.Parameter] = []
    decoder = getattr(getattr(model, "model", model), "decoder", None)
    if decoder is None:
        return params

    for layer in decoder.layers:
        encoder_attn = getattr(layer, "encoder_attn", None)
        if encoder_attn is None:
            continue
        for proj_name in ("k_proj", "v_proj", "out_proj"):
            proj = getattr(encoder_attn, proj_name, None)
            if proj is None:
                continue
            for parameter in proj.parameters():
                params.append(parameter)
    return params


def freeze_model_parameters(model: nn.Module, exceptions: Sequence[nn.Parameter] | None = None) -> None:
    """Freeze all model parameters except those explicitly provided."""
    exceptions = set(exceptions or [])
    for param in model.parameters():
        param.requires_grad = param in exceptions


def mark_parameters_trainable(parameters: Iterable[nn.Parameter]) -> None:
    """Helper to set ``requires_grad=True`` for provided parameters."""
    for param in parameters:
        param.requires_grad = True


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")
