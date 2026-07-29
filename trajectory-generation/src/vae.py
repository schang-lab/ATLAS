"""
Trajectory VAE: a Variational Autoencoder backbone for the ATLAS framework.

Operates in the same latent space as DiT — input/output are (B, T, D) tensors
produced by the BART encoder (optionally PCA-projected).  The VAE learns to
reconstruct these latents through a bottleneck, conditioned on attributes
(work/home coords, optional length, optional demographics).

The model reuses AttrBlock from dit.py for attribute conditioning so that
the same attribute format and demo-id conventions apply.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class TrajectoryVAE(nn.Module):
    """
    Sequence-level VAE that operates on BART latent sequences.

    Encoder:  (B, T, D_in) -> mu, log_var  each (B, Z)
    Decoder:  (B, Z) + cond -> (B, T, D_in)

    Where D_in = in_channels (BART latent dim, possibly after PCA),
          T    = image_size  (sequence length),
          Z    = latent_code_dim (VAE bottleneck).
    """

    def __init__(
        self,
        image_size: int = 512,
        in_channels: int = 512,
        hidden_size: int = 256,
        latent_code_dim: int = 64,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        # Attribute conditioning (same interface as DiT)
        use_length_condition: bool = False,
        length_vocab_size: int = 513,
        use_demo_condition: bool = False,
        num_age_bins: int = 0,
        num_genders: int = 0,
        demo_hidden_dim: int = 256,
        # KL weight (beta-VAE)
        beta_kl: float = 1.0,
        clamp_logvar: bool = False,
        logvar_min: float = -20.0,
        logvar_max: float = 10.0,
        init_logvar_bias: Optional[float] = None,
        **unused,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.latent_code_dim = latent_code_dim
        self.beta_kl = beta_kl
        self.clamp_logvar = bool(clamp_logvar)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.init_logvar_bias = init_logvar_bias

        # ---- Input / output projections ----
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_channels),
        )

        # ---- Positional embedding (fixed sin-cos) ----
        self.pos_embed = nn.Parameter(
            torch.zeros(1, image_size, hidden_size), requires_grad=False
        )
        pos_embed_np = _get_1d_sincos_pos_embed(hidden_size, image_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float().unsqueeze(0))

        # ---- Encoder (Transformer) ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        # Encoder -> VAE latent (masked average pooling is done in encode())
        self.fc_mu = nn.Linear(hidden_size, latent_code_dim)
        self.fc_log_var = nn.Linear(hidden_size, latent_code_dim)

        # ---- Decoder (Transformer) ----
        # Project z + condition -> initial decoder sequence
        # We need to expand the latent code back to (T, hidden_size)
        self.latent_to_seq = nn.Sequential(
            nn.Linear(latent_code_dim + hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, image_size * hidden_size),
        )

        dec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_layers)

        # ---- Attribute conditioning (reuse AttrBlock logic inline) ----
        self.use_length_condition = use_length_condition
        self.use_demo_condition = use_demo_condition
        self.num_age_bins = int(num_age_bins)
        self.num_genders = int(num_genders)
        self.length_vocab_size = length_vocab_size

        # Wide part (coordinates)
        self.wide_fc = nn.Linear(4, hidden_size)

        if self.use_length_condition:
            self.length_embedding = nn.Embedding(length_vocab_size, demo_hidden_dim)
            self.length_proj = nn.Linear(demo_hidden_dim, hidden_size)

        if self.use_demo_condition:
            self.age_embedding = nn.Embedding(self.num_age_bins + 1, demo_hidden_dim, padding_idx=0)
            self.gender_embedding = nn.Embedding(self.num_genders + 1, demo_hidden_dim, padding_idx=0)
            # FiLM conditioning: demographics modulate spatial signal via scale & shift
            self.demo_scale = nn.Sequential(
                nn.Linear(demo_hidden_dim * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.demo_shift = nn.Sequential(
                nn.Linear(demo_hidden_dim * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Attribute embedding (mirrors AttrBlock from dit.py)
    # ------------------------------------------------------------------ #
    @property
    def attr_embed(self):
        """Expose self so _shift_demo_ids_in_attrs can introspect."""
        return self

    @property
    def embedding_dim(self):
        return self.hidden_size

    def _embed_attrs(self, attr: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute attribute embedding vector (B, hidden_size).
        Returns None if attr is None or if the entire batch is unconditional (all zeros)."""
        if attr is None:
            return None

        # Detect fully-unconditional rows: if all elements are zero the row
        # carries no signal and we should return None so the decoder gets no
        # conditioning at all (avoids leaking a learned bias through wide_fc(0)).
        is_active = attr.abs().sum(dim=-1) > 0  # (B,)
        if not is_active.any():
            return None

        continuous_attrs = attr[:, 0:4]
        param_dtype = next(self.wide_fc.parameters()).dtype
        wide_out = self.wide_fc(continuous_attrs.to(dtype=param_dtype))

        idx = 4
        length_out = None
        if self.use_length_condition and attr.shape[1] > idx:
            length_id = attr[:, idx].long().clamp(0, self.length_vocab_size - 1)
            length_out = self.length_proj(self.length_embedding(length_id))
            idx += 1

        demo_scale = None
        demo_shift = None
        if self.use_demo_condition and attr.shape[1] >= idx + 2:
            age_id = attr[:, idx].long()
            gender_id = attr[:, idx + 1].long()
            if self.training:
                # Strict check during training to catch data quality issues early.
                if (age_id < 0).any() or (age_id > self.num_age_bins).any():
                    raise ValueError(
                        f"age_id out of range [0, {self.num_age_bins}]: "
                        f"min={age_id.min().item()}, max={age_id.max().item()}"
                    )
                if (gender_id < 0).any() or (gender_id > self.num_genders).any():
                    raise ValueError(
                        f"gender_id out of range [0, {self.num_genders}]: "
                        f"min={gender_id.min().item()}, max={gender_id.max().item()}"
                    )
            else:
                age_id = age_id.clamp(0, self.num_age_bins)
                gender_id = gender_id.clamp(0, self.num_genders)
            age_emb = self.age_embedding(age_id)
            gender_emb = self.gender_embedding(gender_id)
            cat = torch.cat([age_emb, gender_emb], dim=1)
            demo_scale = self.demo_scale(cat)
            demo_shift = self.demo_shift(cat)

        out = wide_out
        if length_out is not None:
            out = out + length_out
        # FiLM: demographics modulate the spatial signal instead of adding to it
        if demo_scale is not None:
            out = out * (1 + demo_scale) + demo_shift

        # Zero out unconditional rows so they receive no conditioning signal.
        out = out * is_active.unsqueeze(-1).to(out.dtype)
        return out

    # ------------------------------------------------------------------ #
    #  Core VAE methods
    # ------------------------------------------------------------------ #
    def encode(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Encode input latent sequence to VAE bottleneck.
        x: (B, T, D_in)
        attention_mask: (B, T) with 1 for valid, 0 for padding (optional)
        cond: (B, H) attribute conditioning (optional)
        Returns: mu (B, Z), log_var (B, Z)
        """
        h = self.input_proj(x)  # (B, T, H)
        h = h + self.pos_embed[:, :h.size(1), :]

        # Inject attribute conditioning into every position so the encoder
        # can learn demographic-aware latent representations.
        if cond is not None:
            h = h + cond.unsqueeze(1)  # (B, 1, H) broadcast across T

        # Build key-padding mask for the Transformer encoder.
        # nn.TransformerEncoder expects src_key_padding_mask with True for
        # positions to *ignore*, so we invert the attention_mask.
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()  # (B, T)

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B, T, H)

        # Masked average pooling over time (only valid positions contribute).
        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).to(h.dtype)  # (B, T, 1)
            h_masked = h * mask_f
            denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)
            h_pooled = h_masked.sum(dim=1) / denom  # (B, H)
        else:
            h_pooled = h.mean(dim=1)  # (B, H)

        mu = self.fc_mu(h_pooled)
        log_var = self.fc_log_var(h_pooled)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z from q(z|x) using reparameterization trick."""
        if self.clamp_logvar:
            log_var = log_var.clamp(min=self.logvar_min, max=self.logvar_max)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode from VAE latent code back to sequence.
        z: (B, Z)
        cond: (B, H) attribute embedding
        Returns: (B, T, D_in)
        """
        B = z.size(0)

        if cond is not None:
            z_cond = torch.cat([z, cond], dim=-1)  # (B, Z + H)
        else:
            z_cond = torch.cat([z, torch.zeros(B, self.hidden_size, device=z.device, dtype=z.dtype)], dim=-1)

        # Expand to sequence
        h = self.latent_to_seq(z_cond)  # (B, T * H)
        h = h.view(B, self.image_size, self.hidden_size)  # (B, T, H)
        h = h + self.pos_embed[:, :h.size(1), :]

        h = self.decoder(h)  # (B, T, H)
        out = self.output_proj(h)  # (B, T, D_in)
        return out

    def forward(
        self,
        x: torch.Tensor,
        attr_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **unused,
    ):
        """
        Full VAE forward pass.
        x: (B, T, D_in) target latent sequences
        attr_embeds: (B, F) raw attributes (same format as DiT)
        attention_mask: (B, T) 1=valid, 0=pad (optional)

        Returns dict with: recon, mu, log_var, z
        """
        # Compute attribute conditioning
        cond = self._embed_attrs(attr_embeds)

        # Encode (with masking + conditioning)
        mu, log_var = self.encode(x, attention_mask=attention_mask, cond=cond)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        recon = self.decode(z, cond)

        return {
            "recon": recon,
            "mu": mu,
            "log_var": log_var,
            "z": z,
        }

    def generate(
        self,
        batch_size: int,
        attr_embeds: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate new latent sequences by sampling from the prior p(z) = N(0, I).
        Returns: (B, T, D_in)
        """
        if device is None:
            device = next(self.parameters()).device

        param_dtype = next(self.parameters()).dtype
        z = torch.randn(batch_size, self.latent_code_dim, device=device, dtype=param_dtype)
        if attr_embeds is not None:
            attr_embeds = attr_embeds.to(device=device)
        cond = self._embed_attrs(attr_embeds) if attr_embeds is not None else None
        return self.decode(z, cond)

    def compute_loss(
        self,
        x: torch.Tensor,
        attr_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        beta_kl: Optional[float] = None,
    ) -> dict:
        """
        Compute VAE loss = reconstruction + beta * KL.
        x: (B, T, D_in) target latents
        Returns dict with: total_loss, recon_loss, kl_loss, and vae outputs
        """
        out = self.forward(x, attr_embeds=attr_embeds, attention_mask=attention_mask)
        recon = out["recon"]
        mu = out["mu"]
        log_var = out["log_var"]
        if self.clamp_logvar:
            log_var = log_var.clamp(min=self.logvar_min, max=self.logvar_max)

        # Reconstruction loss (MSE, optionally masked).
        # mask broadcasts to (B, T, D) so we divide by the total number of
        # valid *elements* (not just valid timesteps) to keep the per-element
        # scale consistent with the unmasked path.
        mse = F.mse_loss(recon, x, reduction="none")  # (B, T, D)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(mse.dtype)  # (B, T, 1)
            mse = mse * mask  # broadcast to (B, T, D)
            D = mse.size(-1)
            denom = (mask.sum() * D).clamp_min(1.0)  # total valid elements across (B, T, D)
            recon_loss = mse.sum() / denom
        else:
            recon_loss = mse.mean()

        # KL divergence: KL(q(z|x) || N(0,I))
        # Sum over latent dim, mean over batch (standard VAE form).
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()

        beta = beta_kl if beta_kl is not None else self.beta_kl
        total_loss = recon_loss + beta * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            **out,
        }

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()

        # Zero-init the last layer of FiLM scale/shift so at init:
        # out * (1 + 0) + 0 = out  (demographics have no effect initially)
        if self.use_demo_condition:
            nn.init.zeros_(self.demo_scale[-1].weight)
            nn.init.zeros_(self.demo_scale[-1].bias)
            nn.init.zeros_(self.demo_shift[-1].weight)
            nn.init.zeros_(self.demo_shift[-1].bias)

        if self.init_logvar_bias is not None:
            nn.init.constant_(self.fc_log_var.bias, float(self.init_logvar_bias))

    def get_attribute_info(self):
        """Same interface as DiT.get_attribute_info() for compatibility."""
        return {
            'expected_attr_dim': 4
                + (1 if self.use_length_condition else 0)
                + (2 if self.use_demo_condition else 0),
            'model_config': {
                'use_length_condition': self.use_length_condition,
                'length_vocab_size': self.length_vocab_size,
                'embedding_dim': self.hidden_size,
                'hidden_dim': self.hidden_size,
                'use_demo_condition': self.use_demo_condition,
                'num_age_bins': self.num_age_bins,
                'num_genders': self.num_genders,
            }
        }


# ------------------------------------------------------------------ #
#  Positional embedding helper (from dit.py)
# ------------------------------------------------------------------ #

def _get_1d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"hidden_size must be even for sincos pos embed, got {embed_dim}")
    grid = np.expand_dims(np.arange(grid_size, dtype=float), axis=0)
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = grid.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb
