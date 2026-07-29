"""VOLUNTEER VAE adapted for ATLAS framework.

Key changes from original VOLUNTEER:
- User embedding replaced with home/work conditioning, with optional demographics
- Location vocabulary is POI token IDs (from BART tokenizer, vocab_size ~9506)
- Time embedding unchanged (dwell-time differences)
- Outputs POI softmax probabilities directly (no autoencoder needed)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Embedding modules
# ---------------------------------------------------------------------------

class AbsoluteTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for absolute timestamps."""

    def __init__(self, d_model: int, base_period: float = 10080.0):
        super().__init__()
        self.d_model = d_model
        self.base_period = base_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) float — absolute time in minutes."""
        device = x.device
        half = self.d_model // 2
        freq = torch.arange(half, device=device, dtype=torch.float32)
        freq = (2.0 * np.pi / self.base_period) * torch.pow(
            1e-4, 2.0 * freq / self.d_model
        )  # (half,)
        # (batch, seq, 1) * (1, 1, half) -> (batch, seq, half)
        angles = x.unsqueeze(-1) * freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (batch, seq, d_model)
        return emb


class DemoConditionBlock(nn.Module):
    """Replaces VOLUNTEER's USR_EMB with spatial conditioning plus optional demos.

    Inputs per time-step (broadcast along sequence):
        age_bin   : (batch,) int   — age group index (0..num_age-1, or -1/padding)
        gender_id : (batch,) int   — gender index   (0..num_gender-1, or -1/padding)
        home      : (batch, 2) float — (lat, lon)
        work      : (batch, 2) float — (lat, lon)

    Output: (batch, seq_len, demo_emb_size)
    """

    def __init__(
        self,
        num_age_bins: int = 4,
        num_genders: int = 2,
        demo_emb_dim: int = 64,
        spatial_emb_dim: int = 64,
        output_dim: int = 128,
        use_demo_condition: bool = True,
        conditioning_type: str = "legacy",
    ):
        super().__init__()
        self.use_demo_condition = use_demo_condition
        self.conditioning_type = str(conditioning_type or "legacy").lower().strip()
        if self.conditioning_type not in {"legacy", "film_zero_init"}:
            raise ValueError("conditioning_type must be one of: legacy, film_zero_init")
        # +1 for padding/unknown index (we shift ids by +1 so 0 = padding)
        self.age_emb = nn.Embedding(num_age_bins + 1, demo_emb_dim, padding_idx=0)
        self.gender_emb = nn.Embedding(num_genders + 1, demo_emb_dim, padding_idx=0)
        self.spatial_fc = nn.Linear(4, spatial_emb_dim)
        if self.conditioning_type == "legacy":
            self.proj = nn.Linear(2 * demo_emb_dim + spatial_emb_dim, output_dim)
            self.spatial_proj = None
            self.demo_scale = None
            self.demo_shift = None
        else:
            self.proj = None
            self.spatial_proj = nn.Linear(spatial_emb_dim, output_dim)
            self.demo_scale = nn.Sequential(
                nn.Linear(demo_emb_dim * 2, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )
            self.demo_shift = nn.Sequential(
                nn.Linear(demo_emb_dim * 2, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )
        self._init_weights()

    def _init_weights(self) -> None:
        if self.age_emb.padding_idx is not None:
            with torch.no_grad():
                self.age_emb.weight[self.age_emb.padding_idx].zero_()
        if self.gender_emb.padding_idx is not None:
            with torch.no_grad():
                self.gender_emb.weight[self.gender_emb.padding_idx].zero_()
        if self.conditioning_type == "film_zero_init":
            nn.init.zeros_(self.demo_scale[-1].weight)
            nn.init.zeros_(self.demo_scale[-1].bias)
            nn.init.zeros_(self.demo_shift[-1].weight)
            nn.init.zeros_(self.demo_shift[-1].bias)

    def forward(
        self,
        age_bin: torch.Tensor,
        gender_id: torch.Tensor,
        home: torch.Tensor,
        work: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        # Shift ids by +1 so that original -1 (unknown) maps to 0 (padding_idx)
        if self.use_demo_condition:
            age_idx = (age_bin + 1).clamp(min=0).long()       # (batch,)
            gender_idx = (gender_id + 1).clamp(min=0).long()  # (batch,)
        else:
            age_idx = torch.zeros_like(age_bin, dtype=torch.long)
            gender_idx = torch.zeros_like(gender_id, dtype=torch.long)

        a_emb = self.age_emb(age_idx)        # (batch, demo_emb_dim)
        g_emb = self.gender_emb(gender_idx)  # (batch, demo_emb_dim)
        hw = torch.cat([home, work], dim=-1)  # (batch, 4)
        s_emb = F.relu(self.spatial_fc(hw))   # (batch, spatial_emb_dim)

        if self.conditioning_type == "legacy":
            combined = torch.cat([a_emb, g_emb, s_emb], dim=-1)  # (batch, 2*demo+spatial)
            out = self.proj(combined)  # (batch, output_dim)
        else:
            spatial_out = self.spatial_proj(s_emb)
            cat = torch.cat([a_emb, g_emb], dim=-1)
            scale = self.demo_scale(cat)
            shift = self.demo_shift(cat)
            out = spatial_out * (1 + scale) + shift

        # Broadcast along time axis
        return out.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, output_dim)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):

    def __init__(
        self,
        loc_emb_size: int,
        tim_emb_size: int,
        demo_emb_size: int,
        pos_emb_size: int,
        hidden_size: int,
        latent_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        input_size = loc_emb_size + tim_emb_size + demo_emb_size + pos_emb_size
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.mean_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, latent_size),
        )
        self.logvar_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, latent_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, x_emb: torch.Tensor) -> tuple:
        """x_emb: (batch, seq, input_size) -> mean, logvar each (batch, seq, latent)."""
        B = x_emb.size(0)
        h0 = torch.zeros(1, B, self.hidden_size, device=x_emb.device)
        c0 = torch.zeros(1, B, self.hidden_size, device=x_emb.device)
        hidden, _ = self.rnn(self.dropout(x_emb), (h0, c0))
        mean = self.mean_fc(hidden)
        logvar = self.logvar_fc(hidden)
        return mean, logvar


class Decoder(nn.Module):

    def __init__(
        self,
        latent_size: int,
        demo_emb_size: int,
        pos_emb_size: int,
        hidden_size: int,
        loc_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        input_size = latent_size + demo_emb_size + pos_emb_size
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        # Location head
        self.loc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.SELU(),
            nn.Linear(hidden_size // 4, loc_size),
        )
        # Dwell-time head (exponential rate parameter)
        self.tim_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, z: torch.Tensor) -> tuple:
        """z: (batch, seq, latent+demo+pos) -> loc_probs (batch,seq,vocab), dwell (batch,seq)."""
        B = z.size(0)
        h0 = torch.zeros(1, B, self.hidden_size, device=z.device)
        c0 = torch.zeros(1, B, self.hidden_size, device=z.device)
        hidden, _ = self.rnn(self.dropout(z), (h0, c0))
        loc_logits = self.loc_head(hidden)             # (B, T, vocab)
        loc_probs = F.softmax(loc_logits, dim=-1)      # (B, T, vocab)
        dwell = torch.exp(self.tim_head(hidden)).squeeze(-1)  # (B, T)
        return loc_probs, dwell, loc_logits


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class VolunteerVAE(nn.Module):
    """VOLUNTEER VAE with demographic conditioning for ATLAS."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.vocab_size = cfg["vocab_size"]          # 9506
        self.loc_emb_size = cfg.get("loc_emb_size", 256)
        self.tim_emb_size = cfg.get("tim_emb_size", 64)
        self.demo_emb_size = cfg.get("demo_emb_size", 128)
        self.pos_emb_size = cfg.get("pos_emb_size", 64)
        self.hidden_size = cfg.get("hidden_size", 512)
        self.latent_size = cfg.get("latent_size", 256)
        self.dropout = cfg.get("dropout", 0.2)
        self.num_age_bins = cfg.get("num_age_bins", 4)
        self.num_genders = cfg.get("num_genders", 2)
        self.max_seq_len = cfg.get("max_seq_len", 64)
        self.use_demo_condition = bool(cfg.get("use_demo_condition", True))
        self.demo_conditioning_type = str(cfg.get("demo_conditioning_type", "legacy"))

        # Embeddings
        self.loc_emb = nn.Embedding(self.vocab_size, self.loc_emb_size, padding_idx=0)
        self.tim_emb = nn.Embedding(
            cfg.get("tim_buckets", 1440), self.tim_emb_size
        )
        self.pos_emb = AbsoluteTimeEmbedding(self.pos_emb_size)
        self.demo_block = DemoConditionBlock(
            num_age_bins=self.num_age_bins,
            num_genders=self.num_genders,
            demo_emb_dim=64,
            spatial_emb_dim=64,
            output_dim=self.demo_emb_size,
            use_demo_condition=self.use_demo_condition,
            conditioning_type=self.demo_conditioning_type,
        )

        # Encoder & decoder
        self.encoder = Encoder(
            loc_emb_size=self.loc_emb_size,
            tim_emb_size=self.tim_emb_size,
            demo_emb_size=self.demo_emb_size,
            pos_emb_size=self.pos_emb_size,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            dropout=self.dropout,
        )
        self.decoder = Decoder(
            latent_size=self.latent_size,
            demo_emb_size=self.demo_emb_size,
            pos_emb_size=self.pos_emb_size,
            hidden_size=self.hidden_size,
            loc_size=self.vocab_size,
            dropout=self.dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_encoder_input(
        self,
        loc: torch.Tensor,
        tim: torch.Tensor,
        pos: torch.Tensor,
        demo_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate all embeddings for encoder input."""
        l = self.loc_emb(loc)      # (B, T, loc_emb)
        t = self.tim_emb(tim)      # (B, T, tim_emb)
        p = self.pos_emb(pos)      # (B, T, pos_emb)
        return torch.cat([l, t, demo_emb, p], dim=-1)

    def _build_decoder_input(
        self,
        z: torch.Tensor,
        demo_emb: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        p = self.pos_emb(pos)
        return torch.cat([z, demo_emb, p], dim=-1)

    def forward(self, batch: dict) -> dict:
        """Forward pass for training.

        batch keys:
            loc      : (B, T) long   — POI token ids
            tim      : (B, T) long   — discretized dwell times
            pos      : (B, T) float  — absolute timestamps (minutes)
            mask     : (B, T) float  — attention mask (1=valid, 0=pad)
            age_bin  : (B,)   long
            gender_id: (B,)   long
            home     : (B, 2) float
            work     : (B, 2) float
        """
        loc = batch["loc"]
        tim = batch["tim"]
        pos = batch["pos"]
        mask = batch["mask"]
        seq_len = loc.size(1)

        # Demographic conditioning
        demo_emb = self.demo_block(
            batch["age_bin"], batch["gender_id"],
            batch["home"], batch["work"],
            seq_len,
        )

        # Encode
        enc_input = self._build_encoder_input(loc, tim, pos, demo_emb)
        mean, logvar = self.encoder(enc_input)

        # Reparameterize
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)

        # Decode
        dec_input = self._build_decoder_input(z, demo_emb, pos)
        loc_probs, dwell, loc_logits = self.decoder(dec_input)

        return {
            "loc_probs": loc_probs,     # (B, T, vocab) — softmax probs
            "loc_logits": loc_logits,   # (B, T, vocab) — raw logits
            "dwell": dwell,             # (B, T)
            "mean": mean,               # (B, T, latent)
            "logvar": logvar,           # (B, T, latent)
            "mask": mask,
        }

    @staticmethod
    def vae_loss(output: dict, batch: dict) -> dict:
        """Compute standard VAE ELBO loss.

        Returns dict with individual terms + total loss (all as tensors for backprop).
        """
        mean = output["mean"]
        logvar = output["logvar"]
        loc_probs = output["loc_probs"]
        dwell = output["dwell"]
        mask = output["mask"]

        # --- KL divergence ---
        kl_per_token = 0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar)  # (B,T,latent)
        kl_per_token = kl_per_token.sum(dim=-1)  # (B, T)
        kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

        # --- Location reconstruction (NLL) ---
        loc_target = batch["loc"]  # (B, T)
        log_probs = torch.log(loc_probs + 1e-8)
        nll_loc = F.nll_loss(
            log_probs.view(-1, loc_probs.size(-1)),
            loc_target.view(-1),
            reduction="none",
        ).view_as(loc_target)
        nll_loc = (nll_loc * mask).sum() / mask.sum().clamp(min=1)

        # --- Dwell time (exponential NLL) ---
        tim_target = batch["tim"].float()  # (B, T)
        # Exponential distribution: -log(lambda * exp(-lambda * t))
        # = -log(lambda) + lambda * t;  dwell = 1/lambda (rate param from exp())
        rate = dwell.clamp(min=1e-3)
        nll_tim = -torch.log(1 - torch.exp(-rate / 1000) + 1e-8) + (rate / 1000) * tim_target
        nll_tim = (nll_tim * mask).sum() / mask.sum().clamp(min=1)

        loss = kl + nll_loc + nll_tim

        return {
            "loss": loss,
            "kl": kl.detach(),
            "nll_loc": nll_loc.detach(),
            "nll_tim": nll_tim.detach(),
        }

    @torch.no_grad()
    def generate(
        self,
        age_bin: torch.Tensor,
        gender_id: torch.Tensor,
        home: torch.Tensor,
        work: torch.Tensor,
        max_len: int = 64,
        temperature: float = 1.0,
    ) -> dict:
        """Autoregressive generation conditioned on demographics.

        Args:
            age_bin:   (B,) long
            gender_id: (B,) long
            home:      (B, 2) float
            work:      (B, 2) float

        Returns dict with:
            loc_ids:   (B, max_len) long — generated POI token ids
            dwell:     (B, max_len) float — generated dwell times
        """
        self.eval()
        B = age_bin.size(0)
        device = next(self.parameters()).device

        # Sample latent
        z = torch.randn(B, max_len, self.latent_size, device=device)

        # Dummy timestamps (incremental, will be refined by generated dwell)
        pos = torch.zeros(B, max_len, device=device)

        demo_emb = self.demo_block(age_bin, gender_id, home, work, max_len)
        dec_input = self._build_decoder_input(z, demo_emb, pos)
        loc_probs, dwell_pred, _ = self.decoder(dec_input)

        # Sample locations from probabilities
        if temperature != 1.0:
            # Re-apply temperature to logits
            loc_logits = torch.log(loc_probs + 1e-8) / temperature
            loc_probs = F.softmax(loc_logits, dim=-1)

        loc_ids = torch.multinomial(
            loc_probs.view(-1, self.vocab_size), 1
        ).view(B, max_len)

        return {
            "loc_ids": loc_ids,
            "dwell": dwell_pred,
            "loc_probs": loc_probs,
        }
