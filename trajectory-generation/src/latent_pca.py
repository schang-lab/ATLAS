from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


@dataclass
class PCAArtifact:
    mean: torch.Tensor
    std: torch.Tensor
    components: torch.Tensor
    explained_variance: torch.Tensor
    whitening: bool


class LatentPCA:
    """Utility wrapper to project/unproject latents using precomputed PCA."""

    def __init__(self, artifact_path: Union[str, Path], device: Union[torch.device, str] = "cpu") -> None:
        ckpt = torch.load(artifact_path, map_location="cpu")

        self.mean = ckpt["mean"].float()
        self.std = ckpt["std"].float().clamp_min(1e-6)
        self.components = ckpt["components"].float()
        self.explained_variance = ckpt.get("explained_variance")
        if self.explained_variance is not None:
            self.explained_variance = self.explained_variance.float().clamp_min(1e-6)

        self.whitening = bool(ckpt.get("whitening", False))

        self.latent_dim = self.mean.numel()
        self.component_dim = self.components.size(0)

        if self.whitening and self.explained_variance is None:
            raise ValueError("PCA artifact marked as whitened but missing explained_variance")

        if self.whitening:
            self._whiten_scale = torch.sqrt(self.explained_variance).view(1, 1, -1)
        else:
            self._whiten_scale = None

        self.mean_expand = self.mean.view(1, 1, -1)
        self.std_expand = self.std.view(1, 1, -1)

        self.device = torch.device(device)
        self.to(self.device)

    def to(self, device: Union[torch.device, str]) -> "LatentPCA":
        device = torch.device(device)
        self.device = device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.components = self.components.to(device)
        if self.explained_variance is not None:
            self.explained_variance = self.explained_variance.to(device)
        if getattr(self, "_whiten_scale", None) is not None:
            self._whiten_scale = self._whiten_scale.to(device)
        self.mean_expand = self.mean.view(1, 1, -1)
        self.std_expand = self.std.view(1, 1, -1)
        return self

    def project(self, latents: torch.Tensor) -> torch.Tensor:
        """Project AE latents into PCA coordinate space."""
        z_hat = (latents - self.mean_expand) / self.std_expand
        coords = torch.einsum("btd,rd->btr", z_hat, self.components)
        if self._whiten_scale is not None:
            coords = coords / self._whiten_scale
        return coords

    def unproject(self, coords: torch.Tensor) -> torch.Tensor:
        """Reconstruct AE latents from PCA coordinates."""
        z = coords
        if self._whiten_scale is not None:
            z = z * self._whiten_scale
        latents = torch.einsum("btr,rd->btd", z, self.components)
        latents = latents * self.std_expand + self.mean_expand
        return latents
