"""
Execution pipeline for DiT-only inference with demographic conditioning (infer-demo mode).
Moved out of inference_demo.py main() for separation of CLI from logic.
"""
from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Ensure trajectory-generation root is on sys.path when imported standalone.
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)

from src.dit import DiT
from src.diffusion_model import GaussianDiffusion
from src.latent_pca import LatentPCA
from src.models.z_mapper import build_latent_mapper

from .infer_core import (
    convert_poi_sequences_to_coordinates_infer,
    convert_poi_sequences_to_poi_ids_infer,
    sample_dit_with_autoencoder,
)
from .infer_shared import load_or_compute_length_ids, load_poi_mapping, select_length_subset


def _load_autoencoder(path: str, device: torch.device) -> torch.nn.Module:
    """Load the phase-1 BART autoencoder from a local HF checkpoint directory."""
    from transformers import BartConfig, BartForConditionalGeneration

    config_path = os.path.join(path, "config.json")
    try:
        if os.path.exists(config_path):
            ae_config = BartConfig.from_json_file(config_path)
            model = BartForConditionalGeneration.from_pretrained(path, config=ae_config)
        else:
            model = BartForConditionalGeneration.from_pretrained(path)
    except Exception:
        model = BartForConditionalGeneration.from_pretrained(path)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _load_controlled_attrs_with_demo(controlled_split_dir: Path) -> np.ndarray:
    p = controlled_split_dir / "all_attr_results_with_demo.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (this script requires per-trajectory demo attrs).")
    arr = np.load(p, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Expected attrs shape [N,6], got {arr.shape} at {p}")
    return arr.astype(np.float32, copy=False)


def _shift_demo_ids_for_dit(
    attrs: torch.Tensor,
    dit_model: torch.nn.Module,
    *,
    keep_missing_demo_as_null: bool,
) -> torch.Tensor:
    """
    Shift raw demo ids (age_bin, gender_id) by +1 for DiT demo embeddings.
    """
    if attrs is None or attrs.dim() != 2:
        return attrs

    base_dit = getattr(dit_model, "module", dit_model)
    attr_block = getattr(base_dit, "attr_embed", None)
    if attr_block is None or not getattr(attr_block, "use_demo_condition", False):
        return attrs

    use_length = bool(getattr(attr_block, "use_length_condition", False))
    age_idx = 5 if use_length else 4
    gender_idx = age_idx + 1
    if attrs.size(1) <= gender_idx:
        raise ValueError(
            f"DiT expects demo conditioning but attrs dim={attrs.size(1)} is too small "
            f"(need at least {gender_idx+1} dims)."
        )

    conditional_rows = attrs.abs().sum(dim=1) > 0

    age_raw = attrs[:, age_idx].to(torch.long)
    gender_raw = attrs[:, gender_idx].to(torch.long)

    missing = (age_raw < 0) | (gender_raw < 0)
    if missing.any() and not keep_missing_demo_as_null:
        raise ValueError(
            "Found missing demo ids (<0) in attrs but --keep_missing_demo is not set. "
            "Filter them out (recommended) or pass --keep_missing_demo to map them to null."
        )

    age_clamped = age_raw.clamp_min(0)
    gender_clamped = gender_raw.clamp_min(0)
    age_shifted = age_clamped + 1
    gender_shifted = gender_clamped + 1

    if missing.any():
        age_shifted = age_shifted.clone()
        gender_shifted = gender_shifted.clone()
        age_shifted[missing] = 0
        gender_shifted[missing] = 0

    if (~conditional_rows).any():
        age_shifted = age_shifted.clone()
        gender_shifted = gender_shifted.clone()
        age_shifted[~conditional_rows] = 0
        gender_shifted[~conditional_rows] = 0

    out = attrs.clone()
    out[:, age_idx] = age_shifted.to(dtype=out.dtype)
    out[:, gender_idx] = gender_shifted.to(dtype=out.dtype)
    return out


def _insert_length_before_demo(attrs: np.ndarray, length_ids: np.ndarray) -> np.ndarray:
    """Return attrs with length id inserted after coords and before (age,gender)."""
    if attrs.ndim != 2:
        raise ValueError(f"attrs must be 2D, got {attrs.shape}")
    if length_ids.ndim != 1 or length_ids.shape[0] != attrs.shape[0]:
        raise ValueError(f"length_ids mismatch: attrs={attrs.shape} length_ids={length_ids.shape}")
    if attrs.shape[1] < 6:
        return np.column_stack([attrs, length_ids.reshape(-1, 1)])
    coords = attrs[:, :4]
    demo = attrs[:, 4:6]
    return np.concatenate([coords, length_ids.reshape(-1, 1).astype(np.float32), demo], axis=1)


def run_inference_demo(args) -> None:
    """Execute the DiT demo-conditioned inference pipeline with the given parsed arguments."""
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")

    output_dir = args.output_dir or f"{args.output_prefix}_{args.data_type}_{args.data_split}"
    os.makedirs(output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    if "prediction_type" in config:
        from src.helpers import normalize_prediction_type
        args.prediction_type = normalize_prediction_type(config["prediction_type"])

    dit_params = config.get("DiT", {}) or {}

    if "image_size" in dit_params and args.max_traj_length == 512:
        inferred_max = int(dit_params["image_size"])
        if inferred_max != args.max_traj_length:
            print(f"Auto-setting max_traj_length={inferred_max} from DiT.image_size (was {args.max_traj_length})")
            args.max_traj_length = inferred_max

    latent_pca = None
    if args.latent_pca_path:
        latent_pca = LatentPCA(args.latent_pca_path, device)
        print(f"Loaded latent PCA artifact: components={latent_pca.component_dim}")
        dit_params["in_channels"] = latent_pca.component_dim

    dit_params["use_demo_condition"] = True
    if "num_age_bins" not in dit_params or "num_genders" not in dit_params:
        raise ValueError("DiT config must set num_age_bins and num_genders for demo conditioning.")

    if args.enable_length_condition:
        dit_params["use_length_condition"] = True
        dit_params["length_vocab_size"] = int(getattr(args, "length_vocab_size", 513))
    else:
        dit_params["use_length_condition"] = False

    dit_model = DiT(**dit_params).to(device)

    model_path = None
    if args.model_file:
        model_path = args.model_file if os.path.isabs(args.model_file) else os.path.join(args.model_dir, args.model_file)
    else:
        candidates = [
            os.path.join(args.model_dir, "dit_final.pt"),
            os.path.join(args.model_dir, "dit_model.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            pt_files = sorted(Path(args.model_dir).glob("*.pt"))
            if pt_files:
                model_path = str(pt_files[-1])
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not locate a model checkpoint under {args.model_dir}")

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        if "dit" in state and isinstance(state["dit"], dict):
            state = state["dit"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
    dit_model.load_state_dict(state, strict=False)
    dit_model.eval()
    print(f"Loaded DiT checkpoint: {model_path}")

    autoencoder = _load_autoencoder(args.autoencoder_path, device)
    autoencoder.eval()

    timesteps = int(config.get("timesteps", 1000))
    schedule_name = str(config.get("beta_schedule", config.get("schedule", args.beta_schedule)))
    schedule_kwargs: Dict[str, float] = {}
    schedule_key = schedule_name.lower()
    if schedule_key == "cosine":
        schedule_kwargs["s"] = float(args.cosine_s)
    elif schedule_key in {"logsnr", "logsnr_linear", "log-snr"}:
        schedule_kwargs["logsnr_max"] = float(args.logsnr_max)
        schedule_kwargs["logsnr_min"] = float(args.logsnr_min)

    noise_scheduler = GaussianDiffusion(
        timesteps=timesteps,
        schedule=schedule_name,
        schedule_kwargs=schedule_kwargs,
    ).to(device)

    latent_mapper = None
    if args.latent_mapper_path:
        mapper_config_path = args.mapper_config
        if mapper_config_path is None:
            candidate_paths = [
                Path(args.latent_mapper_path).parent / "training_config.json",
                Path(args.autoencoder_path).parent / "training_config.json",
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    mapper_config_path = str(candidate)
                    break
        if mapper_config_path is None:
            raise FileNotFoundError("Could not locate mapper configuration; specify --mapper_config")
        with open(mapper_config_path, "r", encoding="utf-8") as f:
            mapper_config = json.load(f)
        latent_dim = mapper_config.get("latent_dim")
        if latent_pca is not None:
            latent_dim = latent_pca.component_dim
        if latent_dim is None:
            raise ValueError("Unable to determine latent dimension for mapper.")
        latent_mapper = build_latent_mapper(
            latent_dim=latent_dim,
            hidden_dim=mapper_config.get("mapper_hidden_dim"),
            num_hidden_layers=mapper_config.get("mapper_layers", 2),
            dropout=mapper_config.get("mapper_dropout", 0.0),
            activation=mapper_config.get("mapper_activation", "gelu"),
        ).to(device)
        mapper_state = torch.load(args.latent_mapper_path, map_location=device)
        if isinstance(mapper_state, dict) and "state_dict" in mapper_state:
            mapper_state = mapper_state["state_dict"]
        latent_mapper.load_state_dict(mapper_state, strict=True)
        latent_mapper.eval()
        print(f"Loaded latent mapper from {args.latent_mapper_path}")

    need_length_ids = args.enable_length_condition or args.force_empirical_length
    sampled_indices = None
    sampled_lengths = None

    if args.data_type == "uncontrolled":
        base_dim = 6
        sampled_attrs = np.zeros((args.num_samples, base_dim), dtype=np.float32)
    elif args.data_type == "controlled":
        controlled_split_dir = Path(args.test_data_dir) / "controlled" / args.data_split
        all_attrs = _load_controlled_attrs_with_demo(controlled_split_dir)
        age = all_attrs[:, 4].astype(np.int64, copy=False)
        gender = all_attrs[:, 5].astype(np.int64, copy=False)
        if args.keep_missing_demo:
            valid_idx = np.arange(len(all_attrs), dtype=np.int64)
        else:
            valid_idx = np.where((age >= 0) & (gender >= 0))[0].astype(np.int64)
        if valid_idx.size == 0:
            raise RuntimeError("No controlled trajectories remain after filtering missing demo.")

        num_samples = min(args.num_samples, int(valid_idx.size))
        if num_samples < int(valid_idx.size):
            sampled_indices = np.random.choice(valid_idx, num_samples, replace=False).astype(np.int64)
        else:
            sampled_indices = valid_idx
        sampled_attrs = all_attrs[sampled_indices]

        if args.manual_demo is not None:
            age_id, gender_id = int(args.manual_demo[0]), int(args.manual_demo[1])
            sampled_attrs = sampled_attrs.copy()
            sampled_attrs[:, 4] = float(age_id)
            sampled_attrs[:, 5] = float(gender_id)

        if need_length_ids and sampled_attrs.shape[0] > 0:
            controlled_cache = load_or_compute_length_ids(controlled_split_dir, args.max_traj_length)
            sampled_lengths = select_length_subset(controlled_cache, sampled_attrs.shape[0], sampled_indices)
            if args.enable_length_condition and sampled_lengths is not None:
                sampled_attrs = _insert_length_before_demo(sampled_attrs, sampled_lengths)
    else:
        controlled_split_dir = Path(args.test_data_dir) / "controlled" / args.data_split
        all_controlled = _load_controlled_attrs_with_demo(controlled_split_dir)
        age = all_controlled[:, 4].astype(np.int64, copy=False)
        gender = all_controlled[:, 5].astype(np.int64, copy=False)
        if args.keep_missing_demo:
            valid_idx = np.arange(len(all_controlled), dtype=np.int64)
        else:
            valid_idx = np.where((age >= 0) & (gender >= 0))[0].astype(np.int64)
        if valid_idx.size == 0:
            raise RuntimeError("No controlled trajectories remain after filtering missing demo.")

        num_controlled = int(args.num_samples * args.controlled_ratio)
        num_uncontrolled = int(args.num_samples - num_controlled)
        if num_controlled > 0:
            actual_controlled = min(num_controlled, int(valid_idx.size))
            sampled_indices = (
                np.random.choice(valid_idx, actual_controlled, replace=False).astype(np.int64)
                if actual_controlled < int(valid_idx.size)
                else valid_idx[:actual_controlled]
            )
            controlled_attrs = all_controlled[sampled_indices]
        else:
            sampled_indices = np.empty((0,), dtype=np.int64)
            controlled_attrs = np.empty((0, 6), dtype=np.float32)

        uncontrolled_attrs = np.zeros((num_uncontrolled, 6), dtype=np.float32)
        sampled_attrs = np.concatenate([controlled_attrs, uncontrolled_attrs], axis=0)

        if args.manual_demo is not None and controlled_attrs.shape[0] > 0:
            age_id, gender_id = int(args.manual_demo[0]), int(args.manual_demo[1])
            sampled_attrs[:controlled_attrs.shape[0], 4] = float(age_id)
            sampled_attrs[:controlled_attrs.shape[0], 5] = float(gender_id)

        if need_length_ids:
            length_segments = []
            if controlled_attrs.shape[0] > 0:
                controlled_cache = load_or_compute_length_ids(controlled_split_dir, args.max_traj_length)
                length_segments.append(select_length_subset(controlled_cache, controlled_attrs.shape[0], sampled_indices))
            if uncontrolled_attrs.shape[0] > 0:
                uncontrolled_split_dir = Path(args.test_data_dir) / "uncontrolled" / args.data_split
                uncontrolled_cache = load_or_compute_length_ids(uncontrolled_split_dir, args.max_traj_length)
                length_segments.append(select_length_subset(uncontrolled_cache, uncontrolled_attrs.shape[0]))
            sampled_lengths = np.concatenate(length_segments) if length_segments else None
            if args.enable_length_condition and sampled_lengths is not None:
                sampled_attrs = _insert_length_before_demo(sampled_attrs, sampled_lengths)

    if args.manual_home_work is not None:
        home_lat, home_lon, work_lat, work_lon = args.manual_home_work
        base = np.array([work_lat, work_lon, home_lat, home_lon], dtype=np.float32)
        if args.manual_demo is None:
            raise ValueError("--manual_home_work requires --manual_demo in this demo-conditioned inference script.")
        age_id, gender_id = int(args.manual_demo[0]), int(args.manual_demo[1])
        demo = np.array([float(age_id), float(gender_id)], dtype=np.float32)
        core = np.concatenate([base, demo], axis=0).reshape(1, 6)
        sampled_attrs = np.repeat(core, sampled_attrs.shape[0], axis=0)
        if need_length_ids:
            manual_length = args.manual_length_id if args.manual_length_id is not None else args.max_traj_length
            sampled_lengths = np.full(sampled_attrs.shape[0], int(manual_length), dtype=np.int64)
            if args.enable_length_condition:
                sampled_attrs = _insert_length_before_demo(sampled_attrs, sampled_lengths)

    if args.zero_coords:
        if sampled_attrs.shape[1] < 4:
            raise ValueError("--zero_coords requires at least 4 attribute dimensions (work_lat, work_lon, home_lat, home_lon)")
        sampled_attrs = sampled_attrs.copy()
        sampled_attrs[:, :4] = 0.0
        print(f"[INFO] Zeroed out coordinates (indices 0-3) for demo-only conditioning. Shape: {sampled_attrs.shape}")

    attr_tensor_raw = torch.from_numpy(sampled_attrs).float().to(device)

    if args.data_type == "unified":
        num_controlled = int(args.num_samples * args.controlled_ratio)
        flags = np.concatenate([np.ones(num_controlled, dtype=bool), np.zeros(args.num_samples - num_controlled, dtype=bool)])
        flag_tensor = torch.from_numpy(flags[:attr_tensor_raw.shape[0]]).bool().to(device)
        if need_length_ids and sampled_lengths is not None:
            length_tensor = torch.from_numpy(sampled_lengths[:attr_tensor_raw.shape[0]]).long().to(device)
            attr_dataloader = DataLoader(TensorDataset(attr_tensor_raw, length_tensor, flag_tensor), batch_size=args.batch_size, shuffle=False)
        else:
            attr_dataloader = DataLoader(TensorDataset(attr_tensor_raw, flag_tensor), batch_size=args.batch_size, shuffle=False)
    else:
        if need_length_ids and sampled_lengths is not None:
            length_tensor = torch.from_numpy(sampled_lengths[:attr_tensor_raw.shape[0]]).long().to(device)
            attr_dataloader = DataLoader(TensorDataset(attr_tensor_raw, length_tensor), batch_size=args.batch_size, shuffle=False)
        else:
            attr_dataloader = DataLoader(TensorDataset(attr_tensor_raw), batch_size=args.batch_size, shuffle=False)

    split_dir = (Path(args.test_data_dir) / "controlled" / args.data_split) if args.data_type == "unified" else (Path(args.test_data_dir) / args.data_type / args.data_split)
    vocab, poi_coords = load_poi_mapping(str(split_dir / "tokenizer"), str(split_dir / "poi_map_feature.csv"))
    print(f"Loaded vocab size={len(vocab)} | poi coords entries={len(poi_coords)}")

    unk_token_ids = []
    for unk in ("[UNK]", "<unk>"):
        if unk in vocab:
            unk_token_ids.append(int(vocab[unk]))

    forbidden_token_ids: List[int] = []
    if args.forbid_token_ids:
        for part in str(args.forbid_token_ids).replace(",", " ").split():
            try:
                forbidden_token_ids.append(int(part))
            except Exception:
                pass

    poi_home_token_id = vocab.get(args.poi_home_token, None)
    poi_work_token_id = vocab.get(args.poi_work_token, None)
    poi_other_token_id = vocab.get(args.poi_other_token, None)

    if args.forbid_poi_special_tokens:
        for tid in (poi_home_token_id, poi_work_token_id, poi_other_token_id):
            if tid is not None:
                forbidden_token_ids.append(int(tid))
    elif args.forbid_poi_other_only and poi_other_token_id is not None:
        forbidden_token_ids.append(int(poi_other_token_id))

    forbidden_token_ids = sorted(set(forbidden_token_ids))

    generation_config: Dict[str, object] = {
        "num_beams": int(args.num_beams),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "repetition_penalty": float(args.repetition_penalty),
        "length_penalty": float(args.length_penalty),
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
        "unk_logit_penalty": float(args.unk_logit_penalty),
        "forbidden_token_ids": forbidden_token_ids,
        "poi_home_token_id": poi_home_token_id,
        "poi_work_token_id": poi_work_token_id,
        "poi_other_token_id": poi_other_token_id,
        "poi_special_penalty": float(args.penalize_poi_special_tokens),
        "poi_home_penalty": float(args.penalize_poi_home),
        "poi_work_penalty": float(args.penalize_poi_work),
        "poi_other_penalty": float(args.penalize_poi_other),
        "poi_home_work_penalty": float(args.penalize_poi_home_work),
    }

    all_sequences: List[torch.Tensor] = []
    all_latents: List[torch.Tensor] = []
    all_coords_batches: List[np.ndarray] = []
    all_poi_sequences: List[List[str]] = []
    all_attrs_batches: List[np.ndarray] = []
    all_length_targets: List[np.ndarray] = []

    global_token_counter = Counter()
    global_total_tokens = 0
    global_poi_home_count = 0
    global_poi_work_count = 0
    global_poi_other_count = 0

    for _, batch_data in enumerate(tqdm(attr_dataloader, desc="Generating trajectories")):
        if args.data_type == "unified":
            if need_length_ids:
                attrs_raw, length_ids, conditional_flags = batch_data
            else:
                attrs_raw, conditional_flags = batch_data
                length_ids = None
            attrs_for_model = attrs_raw.clone()
            if conditional_flags is not None and not conditional_flags.all():
                attrs_for_model[~conditional_flags] = 0
        else:
            if need_length_ids:
                attrs_raw, length_ids = batch_data
            else:
                attrs_raw = batch_data[0]
                length_ids = None
            attrs_for_model = None if args.data_type == "uncontrolled" else attrs_raw

        attrs_for_model_shifted = None
        if attrs_for_model is not None:
            attrs_for_model_shifted = _shift_demo_ids_for_dit(
                attrs_for_model,
                dit_model,
                keep_missing_demo_as_null=bool(args.keep_missing_demo),
            )

        enforced_lengths = length_ids if (args.force_empirical_length and length_ids is not None) else None

        seq_ids, latents = sample_dit_with_autoencoder(
            dit_model=dit_model,
            noise_scheduler=noise_scheduler,
            autoencoder=autoencoder,
            attr_embeds=attrs_for_model_shifted,
            timesteps=timesteps,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            max_traj_length=args.max_traj_length,
            min_traj_length=args.min_traj_length,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            training_phase="phase1",
            generation_config=generation_config,
            actual_batch_size=attrs_raw.shape[0],
            ddim_eta=args.ddim_eta,
            latent_scale=args.latent_scale,
            latent_pca=latent_pca,
            latent_mapper=latent_mapper,
            length_ids=enforced_lengths,
            unk_token_ids=unk_token_ids,
            prediction_type=args.prediction_type,
        )

        attrs_np = attrs_raw.detach().cpu().numpy()
        attrs_np_4d = attrs_np[:, :4] if attrs_np.shape[1] >= 4 else np.zeros((attrs_np.shape[0], 4), dtype=np.float32)

        seq_ids_cpu = seq_ids.detach().cpu()
        coords = convert_poi_sequences_to_coordinates_infer(
            poi_sequences=seq_ids_cpu,
            vocab=vocab,
            poi_coords=poi_coords,
            attrs_4d=attrs_np_4d,
            max_length=args.max_traj_length,
            poi_home_token=args.poi_home_token,
            poi_work_token=args.poi_work_token,
            poi_other_token=args.poi_other_token,
        )
        poi_seqs = convert_poi_sequences_to_poi_ids_infer(
            poi_sequences=seq_ids_cpu,
            vocab=vocab,
            max_length=args.max_traj_length,
            poi_other_token=args.poi_other_token,
        )

        all_sequences.append(seq_ids_cpu)
        all_latents.append((latents.detach().cpu() * args.latent_scale) if args.latent_scale != 1.0 else latents.detach().cpu())
        all_coords_batches.append(coords)
        all_poi_sequences.extend(poi_seqs)
        all_attrs_batches.append(attrs_np)
        if enforced_lengths is not None:
            all_length_targets.append(enforced_lengths.detach().cpu().numpy().astype(np.int64))

        try:
            id_to_token = {v: k for k, v in vocab.items()}
            flat_ids = seq_ids_cpu.reshape(-1).tolist()
            flat_tokens = [id_to_token.get(tid, "<unk>") for tid in flat_ids]
            special_tokens = {"<pad>", "<s>", "</s>", "<unk>", "<mask>", "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"}
            non_special = [t for t in flat_tokens if t not in special_tokens]
            batch_cnt = Counter(non_special)
            global_token_counter.update(batch_cnt)
            global_total_tokens += len(non_special)
            global_poi_home_count += batch_cnt.get(args.poi_home_token, 0)
            global_poi_work_count += batch_cnt.get(args.poi_work_token, 0)
            global_poi_other_count += batch_cnt.get(args.poi_other_token, 0)
        except Exception:
            pass

    max_seq_len = max(seq.shape[1] for seq in all_sequences) if all_sequences else 0
    padded_sequences = []
    for seq in all_sequences:
        if seq.shape[1] < max_seq_len:
            pad = torch.zeros(seq.shape[0], max_seq_len - seq.shape[1], dtype=seq.dtype)
            padded_sequences.append(torch.cat([seq, pad], dim=1))
        else:
            padded_sequences.append(seq)
    seq_out = torch.cat(padded_sequences, dim=0) if padded_sequences else torch.empty(0)
    lat_out = torch.cat(all_latents, dim=0) if all_latents else torch.empty(0)
    coord_out = np.concatenate(all_coords_batches, axis=0) if all_coords_batches else np.zeros((0, 2, args.max_traj_length), dtype=np.float32)
    attrs_out = np.concatenate(all_attrs_batches, axis=0) if all_attrs_batches else np.zeros((0, 6), dtype=np.float32)
    length_out = np.concatenate(all_length_targets, axis=0) if all_length_targets else None

    np.save(os.path.join(output_dir, "generated_sequences.npy"), seq_out.numpy())
    np.save(os.path.join(output_dir, "generated_latents.npy"), lat_out.numpy())
    np.save(os.path.join(output_dir, "generated_coordinates.npy"), coord_out)
    np.save(os.path.join(output_dir, "sampled_attributes.npy"), attrs_out)
    np.save(os.path.join(output_dir, "sampled_attributes_4d.npy"), attrs_out[:, :4] if attrs_out.shape[1] >= 4 else attrs_out)
    if length_out is not None:
        np.save(os.path.join(output_dir, "target_length_ids.npy"), length_out)

    with open(os.path.join(output_dir, "generated_poi_sequences.pkl"), "wb") as f:
        pickle.dump(all_poi_sequences, f)

    generation_params = {
        "num_samples": int(args.num_samples),
        "data_type": args.data_type,
        "data_split": args.data_split,
        "guidance_scale": float(args.guidance_scale),
        "prediction_type": args.prediction_type,
        "beta_schedule": args.beta_schedule,
        "timesteps": timesteps,
        "num_steps": int(args.num_steps),
        "max_traj_length": int(args.max_traj_length),
        "min_traj_length": int(args.min_traj_length),
        "poi_home_token": args.poi_home_token,
        "poi_work_token": args.poi_work_token,
        "poi_other_token": args.poi_other_token,
        "autoencoder_path": os.path.abspath(args.autoencoder_path),
        "latent_pca_path": args.latent_pca_path,
        "latent_mapper_path": args.latent_mapper_path,
        "latent_scale": float(args.latent_scale),
        "keep_missing_demo": bool(args.keep_missing_demo),
        "manual_demo": args.manual_demo,
        "zero_coords": bool(args.zero_coords),
        "enable_length_condition": bool(args.enable_length_condition),
        "force_empirical_length": bool(args.force_empirical_length),
        "random_seed": int(args.random_seed),
        "generation_config": generation_config,
    }
    with open(os.path.join(output_dir, "generation_params.json"), "w") as f:
        json.dump(generation_params, f, indent=2)

    if global_total_tokens > 0:
        poi_keys = set(poi_coords.keys())
        top_tokens = [t for t, _ in global_token_counter.most_common(100)]
        top100_hit = sum(1 for t in top_tokens if t in poi_keys) / max(1, len(top_tokens))
        unique_non_special = len(global_token_counter)
        print(f"\n[Infer-demo] Global generation statistics:")
        print(f"  Total non-special tokens: {global_total_tokens}")
        print(f"  Unique POI tokens: {unique_non_special}")
        print(f"  Top 100 hit rate: {top100_hit:.3f}")
        print(f"  POI special tokens: HOME={global_poi_home_count}, WORK={global_poi_work_count}, OTHER={global_poi_other_count}")

    print(f"\nSaved demo inference outputs to: {output_dir}")
