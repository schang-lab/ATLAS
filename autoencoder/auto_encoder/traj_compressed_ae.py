import torch
# import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F

from dataclasses import dataclass

from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
)

import math
import numpy as np
from einops import rearrange, reduce, repeat


class BARTLatentCompression(BartForConditionalGeneration):
    def __init__(self,
                 config,
                 num_encoder_latents,
                 num_decoder_latents,
                 dim_ae,
                 num_layers=2,
                 l2_normalize_latents=True,
                 use_coords=False,
                 num_top_categories=None,
                 use_position_embedding=True,
                 transformer_decoder=False,
                 no_compression=False):
        super().__init__(config)
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents
        self.use_coords = use_coords
        self.num_top_categories = num_top_categories
        self.no_compression = no_compression

        # Normalisation helpers to keep decoder inputs in a stable range
        self.output_layer_norm = nn.LayerNorm(config.d_model)
        self.feature_norm_eps = 1e-06
        self.max_feature_scale = 10.0

        if not no_compression:
            self.perceiver_ae = PerceiverAutoEncoder(dim_lm=config.d_model,
                                                     num_encoder_latents=num_encoder_latents,
                                                     num_decoder_latents=num_decoder_latents,
                                                     dim_ae=dim_ae,
                                                     depth=num_layers,
                                                     transformer_decoder=transformer_decoder,
                                                     l2_normalize_latents=l2_normalize_latents,
                                                     use_coords=use_coords,
                                                     num_top_categories=num_top_categories,
                                                     use_position_embedding=use_position_embedding)
        else:
            # For no-compression mode, we only need feature embedders
            self._init_feature_embedders(config.d_model)
    
    def _init_feature_embedders(self, dim_lm):
        """Initialize feature embedders for no-compression mode"""
        # Use SAME approach as PerceiverResampler compression mode:
        # Split full dim_lm evenly between feature types
        
        # Count number of feature types  
        num_feature_types = 0
        if self.use_coords:
            num_feature_types += 1
        if self.num_top_categories is not None and self.num_top_categories > 0:
            num_feature_types += 1
        
        if num_feature_types > 0:
            # Distribute dimensions evenly between feature types (same as compression mode)
            base_dim = dim_lm // num_feature_types
            remainder = dim_lm % num_feature_types
            
            # Calculate dimensions for each feature type (same logic as PerceiverResampler)
            coord_dim = base_dim + (1 if remainder > 0 and self.use_coords else 0)
            remainder = max(0, remainder - 1) if self.use_coords else remainder
            
            top_cat_dim = base_dim + (1 if remainder > 0 and self.num_top_categories is not None else 0)
            
            # Initialize coordinate embedding
            if self.use_coords:
                self.coord_mlp = nn.Sequential(
                    nn.Linear(2, coord_dim),
                    nn.ReLU(),
                    nn.Linear(coord_dim, coord_dim)
                )
            
            # Initialize top category embedding
            if self.num_top_categories is not None and self.num_top_categories > 0:
                self.top_category_embedding = nn.Embedding(self.num_top_categories, top_cat_dim)
            
            # Calculate total feature dimension
            total_feature_dim = 0
            if self.use_coords:
                total_feature_dim += coord_dim
            if self.num_top_categories is not None and self.num_top_categories > 0:
                total_feature_dim += top_cat_dim
            
            # Feature fusion layer
            if total_feature_dim > 0:
                self.feature_fusion = nn.Linear(total_feature_dim, dim_lm)
            
            # Store dimensions for debugging
            self.coord_dim = coord_dim if self.use_coords else 0
            self.top_cat_dim = top_cat_dim if (self.num_top_categories is not None and self.num_top_categories > 0) else 0

    def get_diffusion_latent(self, encoder_outputs, attention_mask, segment_coords=None, 
                           top_categories=None):
        """
        :param encoder_outputs: raw latent H of segment-level seq. from encoder
        :param attention_mask:
        :param segment_coords: segment location features (N,2) latitude, longtitude
        :param top_categories: top category IDs for each POI in the sequence
        :return: compressed latent Z
        """
        hidden_state = encoder_outputs[0]
        latent = self.perceiver_ae.encode(hidden_state, attention_mask.bool(), 
                                        segment_coords=segment_coords,
                                        top_categories=top_categories)
        return latent

    def get_decoder_input(self, diffusion_latent):
        r"""
        :param diffusion_latent: Z
        :return: decompressed latent \hat{H}
        """
        return self.perceiver_ae.decode(diffusion_latent)

    # Map encoder outputs to decoder inputs
    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask, segment_coords=None,
                                      top_categories=None):
        r"""
        :param encoder_outputs:
        :param attention_mask:
        :param segment_coords:
        :param top_categories: top category IDs for each POI in the sequence
        :return: H -> Z -> \hat{H} for compression training OR H + features for no-compression training
        """
        if self.no_compression:
            # No compression mode: add coordinate/top category features directly to encoder outputs
            return self._add_features_no_compression(encoder_outputs, attention_mask, 
                                                   segment_coords, top_categories)
        else:
            # Compression mode: H -> Z -> \hat{H}
            diffusion_latent = self.get_diffusion_latent(encoder_outputs,
                                                         attention_mask,
                                                         segment_coords=segment_coords,
                                                         top_categories=top_categories)

            # Z -> H and normalise before handing back to BART
            decoder_hidden = self.get_decoder_input(diffusion_latent)
            encoder_outputs['last_hidden_state'] = self.output_layer_norm(decoder_hidden)

            return encoder_outputs
    
    def _add_features_no_compression(self, encoder_outputs, attention_mask, segment_coords=None,
                                   top_categories=None):
        """Add coordinate and top category features directly to encoder outputs without compression"""
        hidden_state = encoder_outputs[0]  # (batch_size, seq_len, dim_lm)
        batch_size, seq_len, dim_lm = hidden_state.shape
        
        feature_list = []
        
        # Add coordinate features
        if self.use_coords and segment_coords is not None:
            coord_features = self.coord_mlp(segment_coords)  # (batch_size, seq_len, coord_dim)
            feature_list.append(coord_features)
        
        # Add top category features
        if (self.num_top_categories is not None and self.num_top_categories > 0 
            and top_categories is not None):
            topcat_features = self.top_category_embedding(top_categories)  # (batch_size, seq_len, top_cat_dim)
            feature_list.append(topcat_features)
        
        # Combine and fuse features
        if feature_list:
            combined_features = torch.cat(feature_list, dim=-1)  # (batch_size, seq_len, feature_dim)
            fused_features = self.feature_fusion(combined_features)  # (batch_size, seq_len, dim_lm)

            # Match fused feature scale to the encoder hidden state for stability
            hidden_std = hidden_state.detach().std(dim=-1, keepdim=True) + self.feature_norm_eps
            fused_std = fused_features.detach().std(dim=-1, keepdim=True) + self.feature_norm_eps
            scale = torch.clamp(hidden_std / fused_std, max=self.max_feature_scale)
            fused_features = fused_features * scale

            # Add features to original hidden state and normalise
            enhanced_hidden_state = hidden_state + fused_features
            encoder_outputs['last_hidden_state'] = self.output_layer_norm(enhanced_hidden_state)
        else:
            # Even without extra features, keep activations in a stable range
            encoder_outputs['last_hidden_state'] = self.output_layer_norm(hidden_state)

        return encoder_outputs



def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

def exists(x):
    return x is not None


def divisible_by(numer, denom):
    return (numer % denom) == 0


# NN components
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma


def FeedForward(dim, mult=4, dropout=0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim)
    )


# Standard attention
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            qk_norm=True,
    ):
        super().__init__()
        hidden_dim = dim
        heads = dim // dim_head
        assert divisible_by(dim, heads), 'dimension must be divisible by number of heads'

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = nn.LayerNorm(dim)

        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
            self,
            x,
    ):
        h = self.heads

        x = self.norm(x)

        qkv = (self.to_q(x), self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        sim = einsum('b h i d, b h j d -> b h i j', self.query_norm(q) * self.scale, self.key_norm(k))
        
        # Clamp attention scores to prevent overflow/underflow
        sim = torch.clamp(sim, min=-50.0, max=50.0)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)
        
        # Additional numerical stability check
        if torch.isnan(attn).any():
            # Fallback to uniform attention if NaN detected
            attn = torch.ones_like(attn) / attn.size(-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PerceiverAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_latent,
            dim_head=64,
            qk_norm=True,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        inner_dim = max(dim_latent, dim)
        self.heads = inner_dim // dim_head

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim_latent)

        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()

        self.to_q = nn.Linear(dim_latent, inner_dim, bias=False)
        if dim_latent != dim:
            self.latent_to_kv = nn.Linear(dim_latent, inner_dim * 2, bias=False)
        else:
            self.latent_to_kv = None
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_latent),
        )

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)
        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        if exists(self.latent_to_kv):
            kv_input = torch.cat([self.to_kv(x), self.latent_to_kv(latents)], dim=1)
        else:
            kv_input = torch.cat([self.to_kv(x), self.to_kv(latents)], dim=1)
        k, v = rearrange(kv_input, 'b n (split d) -> split b n d', split=2)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # similarities and masking

        q_norm = self.query_norm(q) * self.scale
        k_norm = self.key_norm(k)

        q_fp32 = q_norm.to(torch.float32)
        k_fp32 = k_norm.to(torch.float32)
        sim = torch.matmul(q_fp32, k_fp32.transpose(-1, -2))

        # Clamp attention scores to prevent overflow/underflow
        sim = torch.clamp(sim, min=-50.0, max=50.0)

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention with numerical stability

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # Additional numerical stability check
        if torch.isnan(attn).any():
            # Fallback to uniform attention if NaN detected
            attn = torch.ones_like(attn) / attn.size(-1)

        v_fp32 = v.to(torch.float32)
        out = torch.matmul(attn, v_fp32)
        out = out.to(q.dtype)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_latent,
            depth,
            dim_head=64,
            num_latents=16,
            max_seq_len=512,
            ff_mult=4,
            legacy=False,
            l2_normalize_latents=True,
            use_coords=False,
            num_top_categories=None,
            use_position_embedding=True
    ):
        super().__init__()
        self.use_position_embedding = use_position_embedding
        if self.use_position_embedding:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        else:
            self.pos_emb = None

        if legacy:
            dim_out = dim_latent
            dim_latent = dim

        self.latents = nn.Parameter(torch.empty(num_latents, dim_latent))
        nn.init.normal_(self.latents, mean=0.0, std=0.01)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(
                    dim=dim, dim_latent=dim_latent, dim_head=dim_head),
                FeedForward(dim=dim_latent, mult=ff_mult)
            ]))

        self.l2_normalize_latents = l2_normalize_latents

        self.final_norm = nn.LayerNorm(dim_latent)
        self.output_proj = nn.Linear(dim_latent, dim_out) if legacy else nn.Identity()

        self.use_coords = use_coords
        self.num_top_categories = num_top_categories
        
        if self.use_coords:
            # Calculate dimension allocation for coordinates and top categories only
            # Count how many feature types we have
            num_feature_types = 1  # coordinates always present
            if num_top_categories is not None and num_top_categories > 0:
                num_feature_types += 1
            
            # Distribute dimensions as evenly as possible
            base_dim = dim // num_feature_types
            remainder = dim % num_feature_types
            
            # Assign dimensions (give extra dimensions to coordinates first)
            coord_dim = base_dim + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)
            
            # Top category dimension
            if num_top_categories is not None and num_top_categories > 0:
                top_cat_dim = base_dim + (1 if remainder > 0 else 0)
            else:
                top_cat_dim = 0
            
            # Coordinate embedding
            self.coord_mlp = nn.Sequential(
                nn.Linear(2, coord_dim),  # (lat, lon) → coordinate embedding
                nn.ReLU(),
                nn.Linear(coord_dim, coord_dim)
            )
            
            # Top category embedding only
            if num_top_categories is not None and num_top_categories > 0:
                self.top_category_embedding = nn.Embedding(num_top_categories, top_cat_dim)
                # Use safer initialization for embedding
                with torch.no_grad():
                    self.top_category_embedding.weight.normal_(0.0, 0.01)
            else:
                self.top_category_embedding = None
            
            # Feature fusion layer to combine coordinates and top categories
            total_feature_dim = coord_dim + top_cat_dim
            
            # Verify that total_feature_dim equals dim
            assert total_feature_dim == dim, f"Feature dimensions don't add up: {total_feature_dim} != {dim}"
            
            # Store dimensions
            self.coord_dim = coord_dim
            self.top_cat_dim = top_cat_dim
            self.total_feature_dim = total_feature_dim
                
            self.feature_fusion = nn.Sequential(
                nn.Linear(total_feature_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        

    def forward(self, x, mask=None, segment_coords=None, top_categories=None):
        if self.use_position_embedding and self.pos_emb is not None:
            pos_emb = self.pos_emb(x)
            x_with_pos = x + pos_emb
        else:
            # Skip position embeddings for ablation study
            x_with_pos = x

        if self.use_coords:
            feature_list = []
            
            # 1. Coordinate features - only if not None and not all zeros
            if segment_coords is not None:
                # Check if coordinates are meaningful (not all zeros for ablation)
                coords_meaningful = segment_coords.abs().sum() > 1e-6
                if coords_meaningful:
                    coord_embed = self.coord_mlp(segment_coords)  # (B, L, coord_dim)
                    feature_list.append(coord_embed)
            
            # 2. Top category features - only if not None and not all zeros  
            if (self.top_category_embedding is not None and 
                top_categories is not None):
                # Check if top categories are meaningful (not all zeros for ablation)
                topcats_meaningful = top_categories.sum() > 0
                if topcats_meaningful:
                    top_cat_embed = self.top_category_embedding(top_categories)  # (B, L, top_cat_dim)
                    feature_list.append(top_cat_embed)
            
            # 3. Fuse available features - only if we have any meaningful features
            if feature_list:
                combined_features = torch.cat(feature_list, dim=-1)  # (B, L, variable_feature_dim)
                
                # Adaptive feature fusion based on actual concatenated dimension
                current_feature_dim = combined_features.shape[-1]
                
                # Use the appropriate fusion layer or create one on-the-fly
                if current_feature_dim == self.feature_fusion[0].in_features:
                    # Standard case: both features enabled
                    fused_features = self.feature_fusion(combined_features)
                else:
                    # Adaptive case: only some features enabled
                    # Create a simple linear projection to match expected output dimension
                    if not hasattr(self, 'adaptive_fusion_cache'):
                        self.adaptive_fusion_cache = {}
                    
                    if current_feature_dim not in self.adaptive_fusion_cache:
                        adaptive_fusion = nn.Sequential(
                            nn.Linear(current_feature_dim, self.feature_fusion[0].out_features),
                            nn.ReLU(),
                            nn.Linear(self.feature_fusion[0].out_features, self.feature_fusion[0].out_features)
                        ).to(combined_features.device)
                        self.adaptive_fusion_cache[current_feature_dim] = adaptive_fusion
                    
                    fused_features = self.adaptive_fusion_cache[current_feature_dim](combined_features)
                
                x_with_pos = x_with_pos + fused_features

        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])
        
        # Safety check and fix for extreme latent values
        if not self.latents.is_meta and torch.abs(self.latents).max() > 10.0:
            with torch.no_grad():
                self.latents.data.normal_(0.0, 0.01)
                latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        latents = self.output_proj(self.final_norm(latents))
        
        # Normalize latents to norm sqrt(d_latent)
        if self.l2_normalize_latents:
            latents = F.normalize(latents, dim=-1) * math.sqrt(latents.shape[-1])
                
        return latents


class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim_input,
            dim_tx,
            depth,
            dim_head=64,
            max_seq_len=512,
            ff_mult=4,
    ):
        super().__init__()
        self.pos_emb = AbsolutePositionalEmbedding(dim_tx, max_seq_len)

        self.input_proj = nn.Linear(dim_input, dim_tx)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim=dim_tx, dim_head=dim_head),
                FeedForward(dim=dim_tx, mult=ff_mult)
            ]))

        self.final_norm = nn.LayerNorm(dim_tx)
        self.output_proj = nn.Identity()
        

    def forward(self, x, mask=None):

        assert not exists(mask)
        x = self.input_proj(x)
        pos_emb = self.pos_emb(x)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.output_proj(self.final_norm(x))


# compression network based on Perceiver

class PerceiverAutoEncoder(nn.Module):
    def __init__(
            self,
            *,
            dim_lm,
            dim_ae,
            depth,
            dim_head=64,
            num_encoder_latents=8,
            num_decoder_latents=32,
            max_seq_len=512,
            ff_mult=4,
            encoder_only=False,
            transformer_decoder=False,
            l2_normalize_latents=True,
            use_coords=False,
            num_top_categories=None,
            use_position_embedding=True
    ):
        super().__init__()
        self.encoder_only = encoder_only
        self.use_coords = use_coords

        if self.encoder_only:
            assert dim_ae == dim_lm

        # Create perceiver encoder with top_category support only
        self.perceiver_encoder = PerceiverResampler(dim=dim_lm,
                                                    dim_latent=dim_ae,
                                                    depth=depth,
                                                    dim_head=dim_head,
                                                    num_latents=num_encoder_latents, # number of latents
                                                    max_seq_len=max_seq_len, # number of input
                                                    ff_mult=ff_mult,
                                                    l2_normalize_latents=l2_normalize_latents,
                                                    use_coords=use_coords,
                                                    num_top_categories=num_top_categories,
                                                    use_position_embedding=use_position_embedding)

        if transformer_decoder:
            self.perceiver_decoder = Transformer(dim_input=dim_ae,
                                                 dim_tx=dim_lm,
                                                 depth=depth,
                                                 dim_head=dim_head,
                                                 max_seq_len=num_encoder_latents,
                                                 ff_mult=ff_mult)
        else:
            self.perceiver_decoder = PerceiverResampler(dim=dim_ae,
                                                        dim_latent=dim_lm,
                                                        depth=depth,
                                                        dim_head=dim_head,
                                                        num_latents=num_decoder_latents, # number of decoder output
                                                        max_seq_len=num_encoder_latents, # number of latents
                                                        ff_mult=ff_mult)

    def decode(self, ae_latent):
        return self.perceiver_decoder(ae_latent)

    def encode(self, encoder_outputs, attention_mask, segment_coords=None, 
              top_categories=None):
            return self.perceiver_encoder(encoder_outputs,
                                          mask=attention_mask.bool(),
                                          segment_coords=segment_coords,
                                          top_categories=top_categories)

    def forward(self, encoder_outputs, attention_mask, segment_coords=None,
               top_categories=None):
        encoder_latents = self.perceiver_encoder(
            encoder_outputs, mask=attention_mask.bool(), segment_coords=segment_coords,
            top_categories=top_categories)

        decoder_latents = self.perceiver_decoder(encoder_latents)
        return decoder_latents
