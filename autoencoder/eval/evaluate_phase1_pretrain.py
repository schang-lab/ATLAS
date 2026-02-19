"""
Evaluation script for Phase 1 pretrained BART autoencoder.

This script evaluates the reconstruction quality of a pretrained BART model
using four key metrics:
1. Token-level Accuracy: Exact match percentage for each position
2. Sequence-level Accuracy: Percentage of perfectly reconstructed trajectories  
3. BLEU Score: N-gram overlap quality (BLEU-1 through BLEU-4)
4. Edit Distance: Normalized Levenshtein distance between sequences (token-level)

"""

import os
import logging
import random
import math
from pathlib import Path
from collections import defaultdict
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BartForConditionalGeneration
from torch.utils.data import DataLoader

# Import BLEU and edit distance utilities
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from Levenshtein import distance as levenshtein_distance
from scipy.stats import spearmanr, pearsonr

_THIS_DIR = Path(__file__).resolve().parent
_EVAL_SRC = _THIS_DIR / "src"
if str(_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(_EVAL_SRC))

from eval_phase1_cli import parse_eval_phase1_args
from eval_phase1_dataset import EvaluationDataset, collate_fn
from eval_phase1_io import (
    aggregate_metrics,
    load_poi_metadata,
    load_test_data,
    load_tokenizer,
    save_results,
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
LATENT_SKIP_TOKENS = {
    '[PAD]', '<pad>', '[CLS]', '<cls>', '[SEP]', '</s>', '<s>',
    '[MASK]', '<mask>'
}
DEFAULT_ANISOTROPY_DIRS = 32
MAX_KNN_ANCHORS = 2048


class LatentGeometryProbe:
    """Collect encoder latents and compute geometry/anisotropy metrics."""

    def __init__(
        self,
        poi_coord_map,
        special_tokens,
        *,
        max_buffer=6000,
        pair_samples=20000,
        triplet_samples=5000,
        knn_k=10,
        geo_threshold_km=0.5,
        num_random_dirs=DEFAULT_ANISOTROPY_DIRS,
        random_seed=42,
    ):
        self.poi_coord_map = poi_coord_map or {}
        self.special_tokens = set(special_tokens or []) | LATENT_SKIP_TOKENS
        self.max_buffer = max(1, int(max_buffer or 1))
        self.pair_samples = max(0, int(pair_samples or 0))
        self.triplet_samples = max(0, int(triplet_samples or 0))
        self.knn_k = max(1, int(knn_k or 1))
        self.geo_threshold_km = max(0.0, float(geo_threshold_km or 0.0))
        self.num_random_dirs = max(1, int(num_random_dirs or DEFAULT_ANISOTROPY_DIRS))
        self.rng = random.Random(random_seed)
        self.knn_anchor_cap = min(MAX_KNN_ANCHORS, self.max_buffer)
        self._token_length_warned = False

        self.latents = []
        self.tokens = []
        self.categories = []
        self.coords = []

    def add_batch(self, hidden_states, token_sequences, attention_mask):
        if hidden_states is None:
            return
        if token_sequences is None:
            return
        if isinstance(attention_mask, torch.Tensor):
            mask_tensor = attention_mask.cpu()
        else:
            mask_tensor = torch.tensor(attention_mask)

        seq_list = list(token_sequences)
        if not seq_list:
            return

        batch_size, seq_len, _ = hidden_states.shape
        effective_batch = min(batch_size, len(seq_list), mask_tensor.size(0))
        if effective_batch < batch_size and not self._token_length_warned:
            logger.warning(
                "Latent probe received %d token sequences for %d hidden states; truncating",
                len(seq_list), batch_size
            )
            self._token_length_warned = True

        for batch_idx in range(effective_batch):
            tokens = seq_list[batch_idx]
            mask_row = mask_tensor[batch_idx].tolist()
            for pos_idx, mask_value in enumerate(mask_row):
                if mask_value == 0 or pos_idx >= seq_len:
                    continue
                token = tokens[pos_idx] if pos_idx < len(tokens) else None
                if token is None or token in self.special_tokens:
                    continue
                vector = hidden_states[batch_idx, pos_idx].detach().clone().to(torch.float32)
                metadata = self.poi_coord_map.get(token)
                lat = metadata.get('lat') if metadata else None
                lon = metadata.get('lon') if metadata else None
                coord_tuple = (float(lat), float(lon)) if lat is not None and lon is not None else None
                category = str(metadata.get('top_category', 'unknown')) if metadata else 'unknown'

                self.latents.append(vector)
                self.tokens.append(token)
                self.categories.append(category)
                self.coords.append(coord_tuple)

        self._truncate_buffers()

    def _truncate_buffers(self):
        excess = len(self.latents) - self.max_buffer
        if excess <= 0:
            return
        self.latents = self.latents[excess:]
        self.tokens = self.tokens[excess:]
        self.categories = self.categories[excess:]
        self.coords = self.coords[excess:]

    def finalize(self):
        vector_count = len(self.latents)
        metrics = {
            'latent_probe_num_vectors': vector_count,
            'latent_pair_count': 0,
            'latent_geo_pair_count': 0,
            'latent_cat_pair_count': 0,
            'geo_triplet_count': 0,
            'cat_triplet_count': 0,
            'latent_knn_anchor_count': 0,
        }

        if vector_count < 2:
            metrics['latent_probe_warning'] = 'Not enough latent vectors collected for probe metrics'
            return metrics

        latents_tensor = torch.stack(self.latents, dim=0)
        pair_indices = self._sample_unique_pairs(vector_count)
        metrics['latent_pair_count'] = len(pair_indices)

        geo_latent_dists = []
        geo_geo_dists = []
        geo_latent_cos = []
        cat_cosines = []
        cat_matches = []
        cat_l2 = []

        for i, j in pair_indices:
            vec_i = latents_tensor[i]
            vec_j = latents_tensor[j]
            l2 = torch.norm(vec_i - vec_j, p=2).item()
            cos = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0)).item()
            cat_matches.append(1 if self.categories[i] == self.categories[j] else 0)
            cat_cosines.append(cos)
            cat_l2.append(l2)
            geo_dist = self._geo_distance_between_indices(i, j)
            if geo_dist is not None:
                geo_latent_dists.append(l2)
                geo_geo_dists.append(geo_dist)
                geo_latent_cos.append(cos)

        metrics['latent_geo_pair_count'] = len(geo_latent_dists)
        metrics['latent_cat_pair_count'] = len(cat_matches)
        metrics['latent_geo_spearman'] = self._safe_correlation(spearmanr, geo_latent_dists, geo_geo_dists)
        metrics['latent_geo_pearson'] = self._safe_correlation(pearsonr, geo_latent_dists, geo_geo_dists)
        metrics['latent_geo_spearman_cos'] = self._safe_correlation(spearmanr, geo_latent_cos, geo_geo_dists)
        metrics['latent_geo_pearson_cos'] = self._safe_correlation(pearsonr, geo_latent_cos, geo_geo_dists)
        metrics['latent_cat_pearson'] = self._safe_correlation(pearsonr, cat_l2, cat_matches)
        metrics['latent_cat_pearson_cos'] = self._safe_correlation(pearsonr, cat_cosines, cat_matches)

        geo_success_l2, geo_success_cos, geo_total = self._geo_triplet_accuracy(latents_tensor)
        metrics['geo_triplet_count'] = geo_total
        metrics['geo_triplet_accuracy'] = (geo_success_l2 / geo_total) if geo_total else None
        metrics['geo_triplet_accuracy_cos'] = (geo_success_cos / geo_total) if geo_total else None

        cat_success_l2, cat_success_cos, cat_total = self._category_triplet_accuracy(latents_tensor)
        metrics['cat_triplet_count'] = cat_total
        metrics['cat_triplet_accuracy'] = (cat_success_l2 / cat_total) if cat_total else None
        metrics['cat_triplet_accuracy_cos'] = (cat_success_cos / cat_total) if cat_total else None

        cat_knn_cos, geo_knn_cos, cat_knn_l2, geo_knn_l2, knn_total = self._knn_recall(latents_tensor)
        metrics['latent_knn_anchor_count'] = knn_total
        metrics[f'cat_knn_recall@{self.knn_k}'] = cat_knn_l2
        metrics[f'geo_knn_recall@{self.knn_k}'] = geo_knn_l2
        metrics[f'cat_knn_recall@{self.knn_k}_cos'] = cat_knn_cos
        metrics[f'geo_knn_recall@{self.knn_k}_cos'] = geo_knn_cos

        anisotropy_stats = self._compute_anisotropy(latents_tensor)
        metrics.update(anisotropy_stats)
        return metrics

    def _sample_unique_pairs(self, total):
        max_pairs = total * (total - 1) // 2
        target = min(self.pair_samples, max_pairs)
        if target <= 0:
            return []
        pairs = set()
        pairs_list = []
        while len(pairs_list) < target:
            i, j = self.rng.sample(range(total), 2)
            if i > j:
                i, j = j, i
            key = (i, j)
            if key in pairs:
                continue
            pairs.add(key)
            pairs_list.append(key)
        return pairs_list

    def _geo_distance_between_indices(self, i, j):
        coord_i = self.coords[i]
        coord_j = self.coords[j]
        if coord_i is None or coord_j is None:
            return None
        return haversine_distance_km(coord_i[0], coord_i[1], coord_j[0], coord_j[1])

    def _safe_correlation(self, fn, x_vals, y_vals):
        if len(x_vals) < 2 or len(y_vals) < 2:
            return None
        try:
            corr, _ = fn(x_vals, y_vals)
        except Exception:
            return None
        if corr is None or math.isnan(corr):
            return None
        return float(corr)

    def _geo_triplet_accuracy(self, latents_tensor):
        candidates = [idx for idx, coord in enumerate(self.coords) if coord is not None]
        if not candidates:
            return 0, 0, 0
        anchors = self._sample_subset(candidates, self.triplet_samples)
        successes_l2 = 0
        successes_cos = 0
        total = 0
        for anchor in anchors:
            pos_idx, pos_dist = self._closest_geo_neighbor(anchor)
            if pos_idx is None or pos_dist is None or pos_dist > self.geo_threshold_km:
                continue
            neg_idx, _ = self._farthest_geo_neighbor(anchor)
            if neg_idx is None or neg_idx == pos_idx:
                continue
            d_ap = torch.norm(latents_tensor[anchor] - latents_tensor[pos_idx], p=2).item()
            d_an = torch.norm(latents_tensor[anchor] - latents_tensor[neg_idx], p=2).item()
            cos_ap = F.cosine_similarity(
                latents_tensor[anchor].unsqueeze(0), latents_tensor[pos_idx].unsqueeze(0)
            ).item()
            cos_an = F.cosine_similarity(
                latents_tensor[anchor].unsqueeze(0), latents_tensor[neg_idx].unsqueeze(0)
            ).item()
            total += 1
            if d_ap < d_an:
                successes_l2 += 1
            if cos_ap > cos_an:
                successes_cos += 1
        return successes_l2, successes_cos, total

    def _category_triplet_accuracy(self, latents_tensor):
        category_to_indices = defaultdict(list)
        for idx, cat in enumerate(self.categories):
            category_to_indices[cat].append(idx)

        anchors = [idx for idx, cat in enumerate(self.categories) if len(category_to_indices[cat]) > 1]
        if not anchors:
            return 0, 0, 0
        anchors = self._sample_subset(anchors, self.triplet_samples)

        successes_l2 = 0
        successes_cos = 0
        total = 0
        all_indices = list(range(len(self.categories)))
        for anchor in anchors:
            cat = self.categories[anchor]
            positives = [idx for idx in category_to_indices[cat] if idx != anchor]
            negatives = [idx for idx in all_indices if self.categories[idx] != cat]
            if not positives or not negatives:
                continue
            pos_idx = self.rng.choice(positives)
            neg_idx = self.rng.choice(negatives)
            d_ap = torch.norm(latents_tensor[anchor] - latents_tensor[pos_idx], p=2).item()
            d_an = torch.norm(latents_tensor[anchor] - latents_tensor[neg_idx], p=2).item()
            cos_ap = F.cosine_similarity(
                latents_tensor[anchor].unsqueeze(0), latents_tensor[pos_idx].unsqueeze(0)
            ).item()
            cos_an = F.cosine_similarity(
                latents_tensor[anchor].unsqueeze(0), latents_tensor[neg_idx].unsqueeze(0)
            ).item()
            total += 1
            if d_ap < d_an:
                successes_l2 += 1
            if cos_ap > cos_an:
                successes_cos += 1
        return successes_l2, successes_cos, total

    def _closest_geo_neighbor(self, anchor):
        anchor_coord = self.coords[anchor]
        if anchor_coord is None:
            return None, None
        best_idx = None
        best_dist = None
        for idx, coord in enumerate(self.coords):
            if idx == anchor or coord is None:
                continue
            dist = haversine_distance_km(anchor_coord[0], anchor_coord[1], coord[0], coord[1])
            if dist is None:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx, best_dist

    def _farthest_geo_neighbor(self, anchor):
        anchor_coord = self.coords[anchor]
        if anchor_coord is None:
            return None, None
        best_idx = None
        best_dist = None
        for idx, coord in enumerate(self.coords):
            if idx == anchor or coord is None:
                continue
            dist = haversine_distance_km(anchor_coord[0], anchor_coord[1], coord[0], coord[1])
            if dist is None:
                continue
            if best_dist is None or dist > best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx, best_dist

    def _sample_subset(self, data, limit):
        if len(data) <= limit:
            return list(data)
        return self.rng.sample(list(data), limit)

    def _knn_recall(self, latents_tensor):
        total = latents_tensor.size(0)
        if total <= 1:
            return None, None, None, None, 0
        k = min(self.knn_k, total - 1)
        normalized = F.normalize(latents_tensor, dim=-1)
        anchors = list(range(total))
        if len(anchors) > self.knn_anchor_cap:
            anchors = self.rng.sample(anchors, self.knn_anchor_cap)

        cat_hits_cos = 0
        geo_hits_cos = 0
        cat_hits_l2 = 0
        geo_hits_l2 = 0
        for anchor in anchors:
            sims = torch.mv(normalized, normalized[anchor])
            sims[anchor] = -float('inf')
            top_cos = torch.topk(sims, k=k).indices.tolist()

            l2_dists = torch.cdist(
                latents_tensor[anchor].unsqueeze(0), latents_tensor, p=2
            ).squeeze(0)
            l2_dists[anchor] = float('inf')
            top_l2 = torch.topk(-l2_dists, k=k).indices.tolist()

            cat_hit_cos = any(
                self.categories[idx] == self.categories[anchor] and self.categories[idx] != 'unknown'
                for idx in top_cos
            )
            geo_hit_cos = any(self._geo_within_threshold(anchor, idx) for idx in top_cos)
            cat_hit_l2 = any(
                self.categories[idx] == self.categories[anchor] and self.categories[idx] != 'unknown'
                for idx in top_l2
            )
            geo_hit_l2 = any(self._geo_within_threshold(anchor, idx) for idx in top_l2)

            if cat_hit_cos:
                cat_hits_cos += 1
            if geo_hit_cos:
                geo_hits_cos += 1
            if cat_hit_l2:
                cat_hits_l2 += 1
            if geo_hit_l2:
                geo_hits_l2 += 1
        total_anchors = len(anchors)
        cat_recall_cos = (cat_hits_cos / total_anchors) if total_anchors else None
        geo_recall_cos = (geo_hits_cos / total_anchors) if total_anchors else None
        cat_recall_l2 = (cat_hits_l2 / total_anchors) if total_anchors else None
        geo_recall_l2 = (geo_hits_l2 / total_anchors) if total_anchors else None
        return cat_recall_cos, geo_recall_cos, cat_recall_l2, geo_recall_l2, total_anchors

    def _geo_within_threshold(self, anchor, idx):
        dist = self._geo_distance_between_indices(anchor, idx)
        return dist is not None and dist <= self.geo_threshold_km

    def _compute_anisotropy(self, latents_tensor):
        if latents_tensor.size(0) < 2:
            return {
                'latent_anisotropy_score': None,
                'latent_anisotropy_pairwise_mean': None,
                'latent_anisotropy_spectral_ratio': None,
            }
        vecs = F.normalize(latents_tensor, dim=-1)
        rand_dirs = torch.randn(self.num_random_dirs, vecs.size(1))
        rand_dirs = F.normalize(rand_dirs, dim=-1)
        sims = torch.matmul(rand_dirs, vecs.T)
        dir_means = sims.mean(dim=1)
        anisotropy = dir_means.max().item()

        sim_matrix = torch.matmul(vecs, vecs.T)
        mask = torch.ones(sim_matrix.shape[0], sim_matrix.shape[1], dtype=torch.bool)
        mask.fill_diagonal_(False)
        if mask.any():
            pairwise_mean = sim_matrix[mask].mean().item()
        else:
            pairwise_mean = None

        centered = vecs - vecs.mean(dim=0, keepdim=True)
        cov = torch.matmul(centered.T, centered) / max(vecs.size(0) - 1, 1)
        try:
            eigvals = torch.linalg.eigvalsh(cov)
            total = eigvals.sum().item()
            spectral_ratio = eigvals[-1].item() / total if total > 0 else None
        except RuntimeError:
            spectral_ratio = None

        return {
            'latent_anisotropy_score': anisotropy,
            'latent_anisotropy_pairwise_mean': pairwise_mean,
            'latent_anisotropy_spectral_ratio': spectral_ratio,
        }


class TrajectoryEvaluator:
    """Main evaluator class with reconstruction + latent metrics"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device,
        generation_config=None,
        latent_probe_config=None,
        poi_coord_map=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.smoothing_function = SmoothingFunction().method1
        
        # Default generation configuration with repetition penalty
        self.generation_config = generation_config or {
            'max_length': 510,  # Maximum generation length
            'min_length': 4,    # Minimum generation length  
            'num_beams': 1,     # Greedy decoding for speed (can be increased for quality)
            'do_sample': False, # Deterministic generation
            'repetition_penalty': 1.2,  # Penalize repetitions
            'no_repeat_ngram_size': 3,   # Prevent 3-gram repetitions
            'length_penalty': 1.0        # Neutral length penalty
        }
        self.latent_probe = None
        if latent_probe_config and poi_coord_map:
            special_tokens = set(self.tokenizer.all_special_tokens or []) | LATENT_SKIP_TOKENS
            self.latent_probe = LatentGeometryProbe(
                poi_coord_map=poi_coord_map,
                special_tokens=special_tokens,
                max_buffer=latent_probe_config.get('max_buffer', 6000),
                pair_samples=latent_probe_config.get('pair_samples', 20000),
                triplet_samples=latent_probe_config.get('triplet_samples', 5000),
                knn_k=latent_probe_config.get('knn_k', 10),
                geo_threshold_km=latent_probe_config.get('geo_threshold_km', 0.5),
                num_random_dirs=latent_probe_config.get('num_random_dirs', DEFAULT_ANISOTROPY_DIRS),
                random_seed=latent_probe_config.get('random_seed', 42),
            )

    def normalize_token(self, token):
        """Normalize tokens to handle equivalent representations"""
        # Map any legacy specials to BERT-style specials
        if token in ['<unk>', '[UNK]']:
            return '[UNK]'
        if token in ['<mask>', '[MASK]']:
            return '[MASK]'
        if token in ['<pad>', '[PAD]']:
            return '[PAD]'
        
        # Keep other tokens as-is
        return token
    
    def compute_token_accuracy_lenient(self, original_tokens, reconstructed_tokens):
        """Compute token-level accuracy with token normalization"""
        if len(original_tokens) != len(reconstructed_tokens):
            min_len = min(len(original_tokens), len(reconstructed_tokens))
            original_tokens = original_tokens[:min_len]
            reconstructed_tokens = reconstructed_tokens[:min_len]
        
        if len(original_tokens) == 0:
            return 0.0
        
        # Normalize tokens before comparison
        normalized_orig = [self.normalize_token(token) for token in original_tokens]
        normalized_recon = [self.normalize_token(token) for token in reconstructed_tokens]
        
        matches = sum(1 for orig, recon in zip(normalized_orig, normalized_recon) if orig == recon)
        return matches / len(normalized_orig)

    def compute_token_accuracy(self, original_tokens, reconstructed_tokens):
        """Compute token-level accuracy with token normalization"""
        original_length = len(original_tokens)  # Store BEFORE truncation

        if len(original_tokens) != len(reconstructed_tokens):
            min_len = min(len(original_tokens), len(reconstructed_tokens))
            original_tokens = original_tokens[:min_len]
            reconstructed_tokens = reconstructed_tokens[:min_len]
        
        if len(original_tokens) == 0:
            return 0.0
        
        # Normalize tokens before comparison
        normalized_orig = [self.normalize_token(token) for token in original_tokens]
        normalized_recon = [self.normalize_token(token) for token in reconstructed_tokens]
        
        matches = sum(1 for orig, recon in zip(normalized_orig, normalized_recon) if orig == recon)
        return matches / original_length
    
    def compute_sequence_accuracy(self, original_tokens, reconstructed_tokens):
        """Compute sequence-level accuracy with normalization"""
        normalized_orig = [self.normalize_token(token) for token in original_tokens]
        normalized_recon = [self.normalize_token(token) for token in reconstructed_tokens]
        return 1.0 if normalized_orig == normalized_recon else 0.0
    
    def compute_bleu_score(self, original_tokens, reconstructed_tokens):
        """Compute BLEU scores with token normalization"""
        if len(original_tokens) == 0 or len(reconstructed_tokens) == 0:
            return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        # Normalize tokens
        normalized_orig = [self.normalize_token(token) for token in original_tokens]
        normalized_recon = [self.normalize_token(token) for token in reconstructed_tokens]
        
        reference = [normalized_orig]
        candidate = normalized_recon
        
        try:
            bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=self.smoothing_function)
            bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing_function)
            bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing_function)
            bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing_function)
        except:
            # Fallback for edge cases
            bleu_1 = bleu_2 = bleu_3 = bleu_4 = 0.0
            
        return {
            'bleu_1': bleu_1,
            'bleu_2': bleu_2, 
            'bleu_3': bleu_3,
            'bleu_4': bleu_4
        }
    
    def compute_edit_distance(self, original_tokens, reconstructed_tokens):
        """Compute normalized edit distance with token normalization"""
        normalized_orig = [self.normalize_token(token) for token in original_tokens]
        normalized_recon = [self.normalize_token(token) for token in reconstructed_tokens]
        
        if len(normalized_orig) == 0 and len(normalized_recon) == 0:
            return 0.0
        
        max_len = max(len(normalized_orig), len(normalized_recon))
        if max_len == 0:
            return 0.0
            
        # Use token-level edit distance directly (more standard for token sequences)
        edit_dist = levenshtein_distance(normalized_orig, normalized_recon)
        # Normalize by the maximum number of tokens
        normalized_dist = edit_dist / max_len if max_len > 0 else 0.0
        
        return normalized_dist
    
    def compute_jaccard_similarity(self, original_tokens, reconstructed_tokens):
        """Compute Jaccard similarity with token normalization"""
        # Normalize tokens first
        normalized_orig = [self.normalize_token(token) for token in original_tokens]
        normalized_recon = [self.normalize_token(token) for token in reconstructed_tokens]
        
        # Filter out BERT-style special tokens (already normalized)
        special_tokens = {'[PAD]', '[MASK]', '[CLS]', '[SEP]'}
        
        filtered_orig = [token for token in normalized_orig if token not in special_tokens]
        filtered_recon = [token for token in normalized_recon if token not in special_tokens]
        
        # Convert to sets for Jaccard computation
        set_original = set(filtered_orig)
        set_reconstructed = set(filtered_recon)
        
        # Handle edge cases
        if len(set_original) == 0 and len(set_reconstructed) == 0:
            return 1.0  # Both empty, perfect match
        elif len(set_original) == 0 or len(set_reconstructed) == 0:
            return 0.0  # One empty, no overlap
        
        # Compute Jaccard similarity
        intersection = len(set_original & set_reconstructed)
        union = len(set_original | set_reconstructed)
        jaccard = intersection / union if union > 0 else 0.0
        
        return jaccard
    
    def reconstruct_batch(self, batch):
        """Reconstruct a batch of sequences"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Guardrail: prevent out-of-range / negative token IDs from crashing CUDA kernels.
        # This can happen if the dataset already contains token IDs from a different vocab.
        vocab_size = int(getattr(self.model.config, "vocab_size", 0) or 0)
        unk_id = self.tokenizer.unk_token_id
        if vocab_size > 0 and unk_id is not None:
            with torch.no_grad():
                min_id = int(input_ids.min().item()) if input_ids.numel() else 0
                max_id = int(input_ids.max().item()) if input_ids.numel() else 0
                if min_id < 0 or max_id >= vocab_size:
                    if not hasattr(self, "_warned_bad_input_ids"):
                        self._warned_bad_input_ids = True
                        logger.warning(
                            "Detected out-of-range input_ids (min=%d, max=%d) for model vocab_size=%d; "
                            "replacing invalid IDs with unk_token_id=%d to avoid CUDA assert.",
                            min_id, max_id, vocab_size, int(unk_id)
                        )
                    invalid = (input_ids < 0) | (input_ids >= vocab_size)
                    input_ids = input_ids.masked_fill(invalid, int(unk_id))
        
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            outputs = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                return_dict_in_generate=True,
                **self.generation_config
            )
        
        return outputs.sequences, encoder_outputs.last_hidden_state
    
    def clean_reconstructed_sequence(self, recon_ids):
        """Clean reconstructed sequence by removing unwanted special tokens"""
        cleaned_ids = []
        
        # Tokens to filter out (but NOT eos/sep - we use that to stop)
        tokens_to_filter = {
            self.tokenizer.pad_token_id,    # [PAD]
            self.tokenizer.cls_token_id,    # [CLS]
            self.tokenizer.mask_token_id,   # [MASK]
        }
        
        # filter out BERT style tokens explicitly present in vocab (except [SEP] which is our EOS)
        bert_style_tokens = ['[PAD]', '[CLS]', '[MASK]']  # Don't include [SEP] - it's our stopping signal
        for token_str in bert_style_tokens:
            if token_str in self.tokenizer.get_vocab():
                token_id = self.tokenizer.get_vocab()[token_str]
                tokens_to_filter.add(token_id)
        
        # Remove None values
        tokens_to_filter = {tid for tid in tokens_to_filter if tid is not None}
        
        for token_id in recon_ids:
            # Stop at EOS token
            if token_id == self.tokenizer.sep_token_id:
                break
            # Skip unwanted tokens (but keep UNK)
            if token_id not in tokens_to_filter:
                cleaned_ids.append(token_id)
        
        # Convert to tokens
        if cleaned_ids:
            recon_tokens = self.tokenizer.convert_ids_to_tokens(cleaned_ids)
        else:
            recon_tokens = []
            
        return recon_tokens
    
    def evaluate_batch(self, batch):
        """Evaluate a single batch"""
        # Get reconstructions
        reconstructed_ids, encoder_hidden = self.reconstruct_batch(batch)
        
        batch_metrics = []
        batch_detailed_results = []
        
        for i, (original_sequence, original_length, individual_id, sample_idx) in enumerate(
            zip(batch['original_sequences'], batch['original_lengths'], 
                batch['individual_ids'], batch['sample_indices'])):
            
            # Clean reconstructed sequence properly
            recon_ids = reconstructed_ids[i].cpu().tolist()
            recon_tokens = self.clean_reconstructed_sequence(recon_ids)
            
            # Use original unpadded sequence (already in token format)
            original_tokens = original_sequence
            
            # Compute all metrics
            token_acc_lenient = self.compute_token_accuracy_lenient(original_tokens, recon_tokens)
            token_acc = self.compute_token_accuracy(original_tokens, recon_tokens)
            seq_acc = self.compute_sequence_accuracy(original_tokens, recon_tokens)
            bleu_scores = self.compute_bleu_score(original_tokens, recon_tokens)
            edit_dist = self.compute_edit_distance(original_tokens, recon_tokens)
            jaccard_sim = self.compute_jaccard_similarity(original_tokens, recon_tokens)
            
            # Metrics for aggregation
            sample_metrics = {
                'token_accuracy_lenient': token_acc_lenient,
                'token_accuracy': token_acc,
                'sequence_accuracy': seq_acc,
                'edit_distance': edit_dist,
                'jaccard_similarity': jaccard_sim,
                **bleu_scores,
                'original_length': len(original_tokens),
                'reconstructed_length': len(recon_tokens)
            }
            
            # Detailed results for individual analysis
            detailed_result = {
                'individual_id': individual_id,
                'sample_idx': sample_idx,
                'original_sequence': original_tokens,
                'reconstructed_sequence': recon_tokens,
                'metrics': sample_metrics
            }
            
            batch_metrics.append(sample_metrics)
            batch_detailed_results.append(detailed_result)

        if self.latent_probe is not None:
            self.latent_probe.add_batch(
                encoder_hidden.detach().cpu(),
                batch['token_sequences'],
                batch['attention_mask']
            )
        
        return batch_metrics, batch_detailed_results


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in kilometers."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    d_lat = lat2_rad - lat1_rad
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return EARTH_RADIUS_KM * c


def main():
    args = parse_eval_phase1_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model = BartForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = load_tokenizer(args.model_path)
    model.to(device)
    model.eval()

    # Cap evaluation sequence length to the model's positional embedding limit.
    # If args.max_length exceeds max_position_embeddings, BART will index past position embeddings
    # and can trigger CUDA device-side asserts / indexing errors.
    model_max_pos = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    effective_max_length = args.max_length
    if model_max_pos > 0 and args.max_length > model_max_pos:
        logger.warning(
            "Requested --max_length=%d exceeds model.config.max_position_embeddings=%d; "
            "capping evaluation max_length to %d to avoid position-embedding indexing errors.",
            args.max_length, model_max_pos, model_max_pos,
        )
        effective_max_length = model_max_pos
    elif model_max_pos > 0 and args.max_length <= 0:
        effective_max_length = model_max_pos

    # Also cap generation length for consistency
    if model_max_pos > 0 and args.generation_max_length > model_max_pos:
        logger.warning(
            "Requested --generation_max_length=%d exceeds model.config.max_position_embeddings=%d; "
            "capping generation_max_length to %d.",
            args.generation_max_length, model_max_pos, model_max_pos,
        )
        args.generation_max_length = model_max_pos
    if args.generation_max_length > effective_max_length:
        logger.warning(
            "Requested --generation_max_length=%d exceeds effective evaluation max_length=%d; "
            "capping generation_max_length to %d.",
            args.generation_max_length, effective_max_length, effective_max_length,
        )
        args.generation_max_length = effective_max_length
    
    # Load test data
    test_df = load_test_data(
        controlled_folder=args.controlled_folder,
        uncontrolled_folder=args.uncontrolled_folder,
        data_folder=args.data_folder,
        split='test'
    )
    
    # Sample subset if requested
    if args.sample_size is not None and args.sample_size < len(test_df):
        logger.info(f"Sampling {args.sample_size} samples from {len(test_df)} total samples (seed={args.random_seed})")
        test_df = test_df.sample(n=args.sample_size, random_state=args.random_seed).reset_index(drop=True)
        logger.info(f"Using sampled dataset: {len(test_df)} samples")

    poi_coord_map = None
    if args.enable_latent_probe:
        poi_coord_map = load_poi_metadata(
            controlled_folder=args.controlled_folder,
            data_folder=args.data_folder,
            poi_metadata_path=args.poi_metadata_path
        )
        if not poi_coord_map:
            raise ValueError(
                "Latent probe enabled but POI metadata could not be loaded. "
                "Pass --poi_metadata_path or disable the probe."
            )
    else:
        logger.info("Latent probe disabled; skipping POI metadata load")

    # Create dataset and dataloader
    test_dataset = EvaluationDataset(test_df, tokenizer, effective_max_length)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Setup generation configuration
    generation_config = {
        'max_length': args.generation_max_length,
        'min_length': args.generation_min_length,
        'num_beams': args.num_beams,
        'do_sample': args.do_sample,
        'repetition_penalty': args.repetition_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'length_penalty': args.length_penalty,
        'top_k': args.top_k,                    
        'temperature': args.temperature   
    }
    
    # Add top_p for sampling mode
    if args.do_sample:
        generation_config['top_p'] = args.top_p
        generation_config['temperature'] = args.temperature

    latent_probe_config = None
    if args.enable_latent_probe:
        latent_probe_config = {
            'max_buffer': args.max_latent_buffer,
            'pair_samples': args.latent_pair_samples,
            'triplet_samples': args.latent_triplet_samples,
            'knn_k': args.latent_knn_k,
            'geo_threshold_km': args.geo_neighbor_km,
            'num_random_dirs': DEFAULT_ANISOTROPY_DIRS,
            'random_seed': args.random_seed,
        }

    # Initialize evaluator
    evaluator = TrajectoryEvaluator(
        model,
        tokenizer,
        device,
        generation_config,
        latent_probe_config=latent_probe_config,
        poi_coord_map=poi_coord_map
    )
    logger.info(f"Generation config: {generation_config}")
    
    # Run evaluation
    logger.info("Starting evaluation...")
    all_metrics = []
    all_detailed_results = []
    
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch_metrics, batch_detailed_results = evaluator.evaluate_batch(batch)
        all_metrics.extend(batch_metrics)
        all_detailed_results.extend(batch_detailed_results)
    
    # Aggregate results
    logger.info("Aggregating results...")
    aggregated_results = aggregate_metrics(all_metrics)
    
    # Add summary statistics
    aggregated_results['total_samples'] = len(all_metrics)
    aggregated_results['perfect_reconstructions'] = sum(1 for m in all_metrics if m['sequence_accuracy'] == 1.0)
    aggregated_results['perfect_reconstruction_rate'] = aggregated_results['perfect_reconstructions'] / len(all_metrics)

    latent_metrics = {}
    if evaluator.latent_probe is not None:
        latent_metrics = evaluator.latent_probe.finalize()
        aggregated_results.update(latent_metrics)
    
    # Print results
    logger.info("=== EVALUATION RESULTS ===")
    logger.info(f"Total samples evaluated: {aggregated_results['total_samples']}")
    logger.info(f"Token-level accuracy: {aggregated_results['token_accuracy_mean']:.4f} ± {aggregated_results['token_accuracy_std']:.4f}")
    logger.info(f"Token-level accuracy (lenient): {aggregated_results['token_accuracy_lenient_mean']:.4f} ± {aggregated_results['token_accuracy_lenient_std']:.4f}")
    logger.info(f"Sequence-level accuracy: {aggregated_results['sequence_accuracy_mean']:.4f} ({aggregated_results['perfect_reconstructions']}/{aggregated_results['total_samples']} perfect)")
    logger.info(f"Jaccard similarity: {aggregated_results['jaccard_similarity_mean']:.4f} ± {aggregated_results['jaccard_similarity_std']:.4f}")
    logger.info(f"BLEU-1: {aggregated_results['bleu_1_mean']:.4f} ± {aggregated_results['bleu_1_std']:.4f}")
    logger.info(f"BLEU-2: {aggregated_results['bleu_2_mean']:.4f} ± {aggregated_results['bleu_2_std']:.4f}")
    logger.info(f"BLEU-3: {aggregated_results['bleu_3_mean']:.4f} ± {aggregated_results['bleu_3_std']:.4f}")
    logger.info(f"BLEU-4: {aggregated_results['bleu_4_mean']:.4f} ± {aggregated_results['bleu_4_std']:.4f}")
    logger.info(f"Edit distance (normalized): {aggregated_results['edit_distance_mean']:.4f} ± {aggregated_results['edit_distance_std']:.4f}")
    logger.info(f"Average sequence length - Original: {aggregated_results['original_length_mean']:.1f} (range: {aggregated_results['original_length_min']}-{aggregated_results['original_length_max']})")
    logger.info(f"Average sequence length - Reconstructed: {aggregated_results['reconstructed_length_mean']:.1f} (range: {aggregated_results['reconstructed_length_min']}-{aggregated_results['reconstructed_length_max']})")
    if latent_metrics:
        logger.info("=== LATENT PROBE METRICS ===")
        for key in sorted(latent_metrics.keys()):
            logger.info(f"{key}: {latent_metrics[key]}")

    # Save results
    save_results(aggregated_results, all_detailed_results, args.output_dir, args)
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
