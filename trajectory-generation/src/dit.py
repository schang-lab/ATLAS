# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x.float()).to(x.dtype), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x.float()).to(x.dtype), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
        # # --- stabilization: bound FiLM outputs ---
        # # tanh limits |shift|, |scale| ≤ 1, 
        # # and the small gain (≈ 0.1) means FiLM can change activations by at most ±10%.
        # # Multiplying the final projection by 0.1 
        # # keeps the initial latent norms around the AE scale 
        # # (~10²–10³ instead of 10⁶).
        # gain = 0.1
        # shift = gain * torch.tanh(shift)
        # scale = gain * torch.tanh(scale)

        # # normalization before modulation
        # x = self.norm_final(x)
        # x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # # small output scale to keep latents near AE magnitude
        # out = self.linear(x) * 0.1  # or learnable self.log_scale.exp()

        # return out

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            image_size=32,
            in_channels=128,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            hidden_size=256,
            cond_feature=False,
            learn_sigma=False,
            use_fp16=False,
            use_length_condition=False,
            length_vocab_size=513,
            use_demo_condition: bool = False,
            num_age_bins: int = 0,
            num_genders: int = 0,
            demo_hidden_dim: int = 256,
            # Optional: attribute branch scaling/normalization (to balance coords vs demo conditioning).
            # Defaults preserve existing behavior.
            attr_wide_gain_init: float = 1.0,
            attr_demo_gain_init: float = 1.0,
            attr_gain_learnable: bool = True,
            attr_branch_norm: str = "none",  # one of: "none", "layernorm"
            **unused
    ):
        super().__init__()
        self.image_size = image_size
        self.dtype = torch.float16 if use_fp16 else torch.float32
        # hidden_size = in_channels
        self.cond_feature = cond_feature

        self.learn_sigma = learn_sigma
        self.num_heads = num_heads

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # self.input_proj = nn.Identity() if in_channels == hidden_size else nn.Linear(in_channels, hidden_size)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.attr_embed = AttrBlock(
            embedding_dim=hidden_size,
            use_length_condition=use_length_condition,
            length_vocab_size=length_vocab_size,
            use_demo_condition=use_demo_condition,
            num_age_bins=num_age_bins,
            num_genders=num_genders,
            hidden_dim=demo_hidden_dim,
            wide_gain_init=attr_wide_gain_init,
            demo_gain_init=attr_demo_gain_init,
            gain_learnable=attr_gain_learnable,
            branch_norm=attr_branch_norm,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, image_size, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, in_channels)
        self.initialize_weights()

    def get_attribute_info(self):
        """Get information about the expected attribute format and model configuration"""
        info = {
            'expected_attr_dim': 4
                                 + (1 if getattr(self.attr_embed, 'use_length_condition', False) else 0)
                                 + (2 if getattr(self.attr_embed, 'use_demo_condition', False) else 0),
            'attr_format': {
                'work_lat': 'continuous (float)',
                'work_lon': 'continuous (float)', 
                'home_lat': 'continuous (float)',
                'home_lon': 'continuous (float)',
                'length_id': 'categorical (0 to length_vocab_size-1)' if getattr(self.attr_embed, 'use_length_condition', False) else 'not used',
                'age_id': 'categorical (0..num_age_bins), 0 means null' if getattr(self.attr_embed, 'use_demo_condition', False) else 'not used',
                'gender_id': 'categorical (0..num_genders), 0 means null' if getattr(self.attr_embed, 'use_demo_condition', False) else 'not used',
            },
            'model_config': {
                'use_length_condition': self.attr_embed.use_length_condition,
                'length_vocab_size': self.attr_embed.length_vocab_size,
                'embedding_dim': self.attr_embed.embedding_dim,
                'hidden_dim': self.attr_embed.hidden_dim,
                'use_demo_condition': getattr(self.attr_embed, 'use_demo_condition', False),
                'num_age_bins': getattr(self.attr_embed, 'num_age_bins', 0),
                'num_genders': getattr(self.attr_embed, 'num_genders', 0)
            }
        }
        return info

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[1])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self,
                x,
                t,
                use_cross_attention=False,
                attr_embeds=None,
                cond_drop_prob=None,
                cond_drop_mask=None,
                force_unconditional: bool = False,
                **unused):
        """
        Forward pass of DiT.
        
        Args:
            x: (N, T, D) tensor of spatial inputs (latent representations)
            t: (N,) tensor of diffusion timesteps
            attr_embeds: (N, 5) tensor of attributes when length conditioning is enabled:
                - Index 0-1: work_lat, work_lon (continuous)
                - Index 2-3: home_lat, home_lon (continuous)
                - Index 4: trajectory length id (categorical)
        """
        # x = x.squeeze(1)  # (N, T, D)
        x = self.input_proj(x)
        x = x + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)  # (N, D)
        c = t
        attr_embedding = None

        if attr_embeds is not None and not force_unconditional:
            drop_mask = None
            if cond_drop_mask is not None:
                drop_mask = cond_drop_mask.to(device=attr_embeds.device, dtype=torch.bool)
            elif cond_drop_prob is not None:
                if isinstance(cond_drop_prob, torch.Tensor):
                    drop_mask = cond_drop_prob.to(device=attr_embeds.device, dtype=torch.bool)
                else:
                    if cond_drop_prob > 0:
                        drop_mask = torch.rand(attr_embeds.shape[0], device=attr_embeds.device) < float(cond_drop_prob)
            if drop_mask is not None and drop_mask.any():
                attr_embeds = attr_embeds.clone()
                attr_embeds[drop_mask] = 0
            attr_embedding = self.attr_embed(attr_embeds)  # (N, D)
        elif force_unconditional and attr_embeds is not None:
            zeros = torch.zeros_like(attr_embeds)
            attr_embedding = self.attr_embed(zeros)

        if attr_embedding is not None:
            c = c + attr_embedding

        # x_type = x.dtype
        x, c = x.to(self.dtype), c.to(self.dtype)

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, D)

        # x = x.unsqueeze(1)  # (N, 1, T, D)
        return x



class AttrBlock(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        hidden_dim=256,
        length_vocab_size=513,
        use_length_condition=False,
        use_demo_condition: bool = False,
        num_age_bins: int = 0,
        num_genders: int = 0,
        # Optional: balance coordinate ("wide") and demographic ("demo") branches.
        wide_gain_init: float = 1.0,
        demo_gain_init: float = 1.0,
        gain_learnable: bool = True,
        branch_norm: str = "none",  # "none" | "layernorm"
    ):
        super(AttrBlock, self).__init__()
        
        self.use_length_condition = use_length_condition
        self.use_demo_condition = use_demo_condition
        self.num_age_bins = int(num_age_bins)
        self.num_genders = int(num_genders)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.length_vocab_size = length_vocab_size
        self.branch_norm = str(branch_norm or "none").lower().strip()
        if self.branch_norm not in {"none", "layernorm"}:
            raise ValueError(f"AttrBlock.branch_norm must be 'none' or 'layernorm' (got {self.branch_norm!r})")

        # Learnable (or fixed) scalar gains to balance branches.
        # - Keep defaults at 1.0 to preserve backward behavior.
        # - These can be tuned (or learned) to amplify demo relative to coords.
        wg = torch.tensor(float(wide_gain_init), dtype=torch.float32)
        dg = torch.tensor(float(demo_gain_init), dtype=torch.float32)
        if bool(gain_learnable):
            self.wide_gain = nn.Parameter(wg)
            self.demo_gain = nn.Parameter(dg)
        else:
            self.register_buffer("wide_gain", wg)
            self.register_buffer("demo_gain", dg)

        # Optional per-branch normalization to reduce scale dominance.
        self.wide_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6) if self.branch_norm == "layernorm" else None
        self.demo_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6) if self.branch_norm == "layernorm" else None

        # Wide part (linear model for continuous attributes)
        # Handles 4 continuous attributes: [work_lat, work_lon, home_lat, home_lon]
        self.wide_fc = nn.Linear(4, embedding_dim)

        if self.use_length_condition:
            self.length_embedding = nn.Embedding(length_vocab_size, hidden_dim)
            self.length_proj = nn.Linear(hidden_dim, embedding_dim)
            self.combine_fc = nn.Linear(embedding_dim * 2, embedding_dim)
            self._init_embeddings()
        else:
            self.combine_fc = None

        # Demo (age, gender) deep branch
        if self.use_demo_condition:
            # padding_idx=0 keeps row-0 "null" (zero and excluded from training updates)
            demo_hidden = hidden_dim
            self.age_embedding = nn.Embedding(self.num_age_bins + 1, demo_hidden, padding_idx=0)
            self.gender_embedding = nn.Embedding(self.num_genders + 1, demo_hidden, padding_idx=0)
            self.demo_fc1 = nn.Linear(demo_hidden * 2, embedding_dim)
            self.demo_fc2 = nn.Linear(embedding_dim, embedding_dim)
            if self.age_embedding.padding_idx is not None:
                with torch.no_grad():
                    self.age_embedding.weight[self.age_embedding.padding_idx].zero_()
            if self.gender_embedding.padding_idx is not None:
                with torch.no_grad():
                    self.gender_embedding.weight[self.gender_embedding.padding_idx].zero_()
        else:
            self.age_embedding = None
            self.gender_embedding = None
            self.demo_fc1 = None
            self.demo_fc2 = None

    def _init_embeddings(self):
        """Initialize embedding weights with proper scaling"""
        if self.use_length_condition:
            nn.init.normal_(self.length_embedding.weight, std=0.02)
            nn.init.normal_(self.length_proj.weight, std=0.02)
            nn.init.constant_(self.length_proj.bias, 0)
            nn.init.normal_(self.combine_fc.weight, std=0.02)
            nn.init.constant_(self.combine_fc.bias, 0)

    def get_vocabulary_info(self):
        """Get information about the vocabulary sizes"""
        info = {
            'length_vocab_size': self.length_vocab_size,
            'continuous_features': 4,
            'demo_age_bins': self.num_age_bins if self.use_demo_condition else 0,
            'demo_genders': self.num_genders if self.use_demo_condition else 0,
            'total_features': 4
                              + (1 if self.use_length_condition else 0)
                              + (2 if self.use_demo_condition else 0)
        }
        return info

    def forward(self, attr):
        # attr format: [work_lat, work_lon, home_lat, home_lon, (optional) length_id, (optional) age_id, (optional) gender_id]
        # Extract continuous coordinates (indices 0-3)
        continuous_attrs = attr[:, 0:4]  # [work_lat, work_lon, home_lat, home_lon]
        
        # Wide part - handles continuous coordinate attributes
        wide_out = self.wide_fc(continuous_attrs.float())
        if self.wide_norm is not None:
            wide_out = self.wide_norm(wide_out)
        wide_out = wide_out * self.wide_gain.to(dtype=wide_out.dtype)

        idx = 4
        length_out = None
        if self.use_length_condition and attr.shape[1] > idx:
            length_id = attr[:, idx].long()
            length_id = torch.clamp(length_id, 0, self.length_vocab_size - 1)
            length_embed = self.length_embedding(length_id)
            length_out = self.length_proj(length_embed)
            idx += 1

        demo_out = None
        if self.use_demo_condition and attr.shape[1] >= idx + 2:
            age_id = attr[:, idx].long().clamp_min(0)
            gender_id = attr[:, idx + 1].long().clamp_min(0)
            if (self.age_embedding is not None) and (self.gender_embedding is not None):
                age_emb = self.age_embedding(age_id)
                gender_emb = self.gender_embedding(gender_id)
                cat = torch.cat([age_emb, gender_emb], dim=1)
                deep = F.relu(self.demo_fc1(cat))
                demo_out = self.demo_fc2(deep)
                if self.demo_norm is not None:
                    demo_out = self.demo_norm(demo_out)
                demo_out = demo_out * self.demo_gain.to(dtype=demo_out.dtype)

        # Preserve legacy combine path when only length is used
        if self.use_length_condition and (demo_out is None) and (length_out is not None) and (self.combine_fc is not None):
            return self.combine_fc(torch.cat([wide_out, length_out], dim=1))

        out = wide_out
        if length_out is not None:
            out = out + length_out
        if demo_out is not None:
            out = out + demo_out
        return out
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height
    return:
    pos_embed: [grid_size, embed_dim]
    """
    grid = np.expand_dims(np.arange(grid_size, dtype=float), axis=0)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
