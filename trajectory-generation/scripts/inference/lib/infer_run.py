"""
Execution pipeline for DiT-only inference (infer mode).
Moved out of inference.py main() for separation of CLI from logic.
"""
from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BartConfig, BartForConditionalGeneration

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
from .infer_shared import (
    append_length_condition,
    load_demo_pairs_from_attrs_with_demo,
    load_or_compute_length_ids,
    load_poi_mapping,
    sample_demo_pairs_from_real_demo,
    select_length_subset,
)


def run_inference(args) -> None:
    """Execute the DiT inference pipeline with the given parsed arguments."""
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")

    # Output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = f"{args.output_prefix}_{args.data_type}_{args.data_split}"
    os.makedirs(output_dir, exist_ok=True)

    # Load config + build DiT
    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    if "prediction_type" in config:
        from src.helpers import normalize_prediction_type
        args.prediction_type = normalize_prediction_type(config["prediction_type"])

    dit_params = config.get("DiT", {}) or {}

    # Auto-infer max_traj_length from DiT image_size if not explicitly set
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

    # length conditioning compatibility
    if args.enable_length_condition:
        dit_params["use_length_condition"] = True
        dit_params["length_vocab_size"] = int(getattr(args, "length_vocab_size", 513))
    else:
        dit_params["use_length_condition"] = False

    dit_model = DiT(**dit_params).to(device)

    # Resolve model file
    if args.model_file:
        model_path = args.model_file if os.path.isabs(args.model_file) else os.path.join(args.model_dir, args.model_file)
    else:
        for cand in ["dit_best.pt", "dit_final.pt", "dit_model.pt"]:
            p = os.path.join(args.model_dir, cand)
            if os.path.exists(p):
                model_path = p
                break
        else:
            raise FileNotFoundError(f"No .pt model found in {args.model_dir}. Use --model_file.")

    print(f"Loading DiT weights from: {model_path}")
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

    # Scheduler
    timesteps = int(config.get("timesteps", 1000))
    schedule_kwargs = {}
    schedule_key = args.beta_schedule.lower()
    if schedule_key == "cosine":
        schedule_kwargs = {"s": args.cosine_s}
    elif schedule_key in {"logsnr", "logsnr_linear", "log-snr"}:
        schedule_kwargs = {"logsnr_max": args.logsnr_max, "logsnr_min": args.logsnr_min}

    noise_scheduler = GaussianDiffusion(
        timesteps=timesteps,
        schedule=args.beta_schedule,
        schedule_kwargs=schedule_kwargs,
    ).to(device)

    # Load autoencoder (phase1 only here)
    ae_config = BartConfig.from_json_file(os.path.join(args.autoencoder_path, "config.json"))
    autoencoder = BartForConditionalGeneration.from_pretrained(args.autoencoder_path, config=ae_config).to(device)
    autoencoder.eval()

    # Optional mapper
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

    # Attributes sampling (controlled only by default for infer mode)
    need_length_ids = args.enable_length_condition or args.force_empirical_length
    unconditional_mask = None

    if args.data_type == "uncontrolled":
        sampled_attrs = np.zeros((args.num_samples, 4), dtype=np.float32)
        sampled_indices = None
        sampled_lengths = None
        unconditional_mask = np.ones((sampled_attrs.shape[0],), dtype=bool)
    elif args.data_type == "controlled":
        controlled_attr_path_6d = f"{args.test_data_dir}/controlled/{args.data_split}/all_attr_results_with_demo.npy"
        controlled_attr_path_4d = f"{args.test_data_dir}/controlled/{args.data_split}/all_attr_results.npy"
        if os.path.exists(controlled_attr_path_6d):
            all_attrs = np.load(controlled_attr_path_6d, allow_pickle=True)
            print(f"[Infer] Loaded 6D attrs (with demo) from {controlled_attr_path_6d}")
        else:
            all_attrs = np.load(controlled_attr_path_4d, allow_pickle=True)
            print(f"[Infer] Loaded 4D attrs (no demo) from {controlled_attr_path_4d}")
        num_samples = min(args.num_samples, len(all_attrs))
        if num_samples < len(all_attrs):
            sampled_indices = np.random.choice(len(all_attrs), num_samples, replace=False)
            sampled_attrs = all_attrs[sampled_indices]
        else:
            sampled_attrs = all_attrs
            sampled_indices = np.arange(len(sampled_attrs)) if len(sampled_attrs) > 0 else None

        sampled_lengths = None
        if need_length_ids and len(sampled_attrs) > 0:
            controlled_split_dir = Path(args.test_data_dir) / "controlled" / args.data_split
            controlled_cache = load_or_compute_length_ids(controlled_split_dir, args.max_traj_length)
            sampled_lengths = select_length_subset(controlled_cache, len(sampled_attrs), sampled_indices)
            if args.enable_length_condition:
                sampled_attrs = append_length_condition(sampled_attrs, sampled_lengths)
        unconditional_mask = np.zeros((sampled_attrs.shape[0],), dtype=bool)
    else:
        controlled_attr_path_6d = f"{args.test_data_dir}/controlled/{args.data_split}/all_attr_results_with_demo.npy"
        controlled_attr_path_4d = f"{args.test_data_dir}/controlled/{args.data_split}/all_attr_results.npy"
        if os.path.exists(controlled_attr_path_6d):
            all_controlled_attrs = np.load(controlled_attr_path_6d, allow_pickle=True)
            print(f"[Infer] Loaded 6D attrs (with demo) from {controlled_attr_path_6d}")
        else:
            all_controlled_attrs = np.load(controlled_attr_path_4d, allow_pickle=True)
            print(f"[Infer] Loaded 4D attrs (no demo) from {controlled_attr_path_4d}")
        num_controlled = int(args.num_samples * args.controlled_ratio)
        num_uncontrolled = args.num_samples - num_controlled
        if num_controlled > 0:
            actual_controlled = min(num_controlled, len(all_controlled_attrs))
            sampled_indices = np.random.choice(len(all_controlled_attrs), actual_controlled, replace=False) if actual_controlled < len(all_controlled_attrs) else None
            controlled_attrs = all_controlled_attrs[sampled_indices] if sampled_indices is not None else all_controlled_attrs[:actual_controlled]
            controlled_flags = np.ones(len(controlled_attrs), dtype=bool)
        else:
            controlled_attrs = np.empty((0, 4), dtype=np.float32)
            controlled_flags = np.empty((0,), dtype=bool)

        attr_dim = controlled_attrs.shape[1] if len(controlled_attrs) > 0 else 4
        uncontrolled_attrs = np.zeros((num_uncontrolled, attr_dim), dtype=np.float32)
        uncontrolled_flags = np.zeros((num_uncontrolled,), dtype=bool)
        sampled_attrs = np.concatenate([controlled_attrs, uncontrolled_attrs], axis=0)
        unconditional_mask = np.concatenate([np.zeros((controlled_attrs.shape[0],), dtype=bool),
                                             np.ones((uncontrolled_attrs.shape[0],), dtype=bool)], axis=0)
        sampled_lengths = None
        if need_length_ids:
            length_segments = []
            controlled_split_dir = Path(args.test_data_dir) / "controlled" / args.data_split
            controlled_cache = load_or_compute_length_ids(controlled_split_dir, args.max_traj_length)
            if len(controlled_attrs) > 0:
                length_segments.append(select_length_subset(controlled_cache, len(controlled_attrs), sampled_indices))
            if len(uncontrolled_attrs) > 0:
                uncontrolled_split_dir = Path(args.test_data_dir) / "uncontrolled" / args.data_split
                uncontrolled_cache = load_or_compute_length_ids(uncontrolled_split_dir, args.max_traj_length)
                if uncontrolled_cache is None and controlled_cache is not None:
                    print(
                        f"[Infer] Warning: missing empirical length ids under {uncontrolled_split_dir}; "
                        f"falling back to controlled lengths from {controlled_split_dir}."
                    )
                    uncontrolled_cache = controlled_cache
                length_segments.append(select_length_subset(uncontrolled_cache, len(uncontrolled_attrs)))
            sampled_lengths = np.concatenate(length_segments) if length_segments else None
            if args.enable_length_condition and sampled_lengths is not None:
                sampled_attrs = append_length_condition(sampled_attrs, sampled_lengths)

    if args.manual_home_work is not None:
        home_lat, home_lon, work_lat, work_lon = args.manual_home_work
        base = np.array([work_lat, work_lon, home_lat, home_lon], dtype=np.float32)
        sampled_attrs = np.tile(base, (sampled_attrs.shape[0], 1))
        if need_length_ids:
            manual_length = args.manual_length_id if args.manual_length_id is not None else args.max_traj_length
            sampled_lengths = np.full(sampled_attrs.shape[0], manual_length, dtype=np.int64)
            if args.enable_length_condition:
                sampled_attrs = append_length_condition(sampled_attrs, sampled_lengths)

    if args.assign_random_demo:
        demo_source_path = args.demo_source_attr_with_demo_npy
        if demo_source_path is None:
            demo_source_path = f"{args.test_data_dir}/controlled/{args.data_split}/all_attr_results_with_demo.npy"
        if unconditional_mask is None:
            unconditional_mask = np.ones((sampled_attrs.shape[0],), dtype=bool)
        assign_mask = unconditional_mask if bool(args.demo_only_for_unconditional) else np.ones_like(unconditional_mask, dtype=bool)
        n_assign = int(assign_mask.sum())
        if n_assign > 0:
            real_demo = load_demo_pairs_from_attrs_with_demo(demo_source_path)
            demo_pairs = sample_demo_pairs_from_real_demo(
                n=n_assign,
                real_demo_pairs=real_demo,
                seed=int(args.random_seed),
            ).astype(np.float32)

            if sampled_attrs.ndim != 2:
                raise ValueError(f"sampled_attrs must be 2D, got shape {sampled_attrs.shape}")
            if sampled_attrs.shape[1] < 6:
                pad = np.zeros((sampled_attrs.shape[0], 6 - sampled_attrs.shape[1]), dtype=sampled_attrs.dtype)
                sampled_attrs = np.concatenate([sampled_attrs, pad], axis=1)
            sampled_attrs[assign_mask, -2:] = demo_pairs
        else:
            print("[Infer] Warning: demo assignment requested but assign_mask is empty (no rows to assign).")

    attr_tensor = torch.from_numpy(sampled_attrs).float().to(device)
    if args.data_type == "unified":
        if unconditional_mask is not None and unconditional_mask.shape[0] == sampled_attrs.shape[0]:
            flags = (~unconditional_mask).astype(bool, copy=False)
        else:
            flags = np.concatenate([np.ones(int(args.num_samples * args.controlled_ratio), dtype=bool),
                                    np.zeros(args.num_samples - int(args.num_samples * args.controlled_ratio), dtype=bool)])
        flag_tensor = torch.from_numpy(flags[:attr_tensor.shape[0]]).bool().to(device)
        if need_length_ids and sampled_lengths is not None:
            length_tensor = torch.from_numpy(sampled_lengths[:attr_tensor.shape[0]]).long().to(device)
            attr_dataloader = DataLoader(TensorDataset(attr_tensor, length_tensor, flag_tensor), batch_size=args.batch_size, shuffle=False)
        else:
            attr_dataloader = DataLoader(TensorDataset(attr_tensor, flag_tensor), batch_size=args.batch_size, shuffle=False)
    else:
        if need_length_ids and sampled_lengths is not None:
            length_tensor = torch.from_numpy(sampled_lengths[:attr_tensor.shape[0]]).long().to(device)
            attr_dataloader = DataLoader(TensorDataset(attr_tensor, length_tensor), batch_size=args.batch_size, shuffle=False)
        else:
            attr_dataloader = DataLoader(TensorDataset(attr_tensor), batch_size=args.batch_size, shuffle=False)

    if args.data_type == "unified":
        split_dir = f"{args.test_data_dir}/controlled/{args.data_split}"
    else:
        split_dir = f"{args.test_data_dir}/{args.data_type}/{args.data_split}"
    tokenizer_path = f"{split_dir}/tokenizer"
    poi_coords_path = f"{split_dir}/poi_map_feature.csv"
    vocab, poi_coords = load_poi_mapping(tokenizer_path, poi_coords_path)
    print(f"Loaded vocab size={len(vocab)} | poi coords entries={len(poi_coords)}")

    try:
        special_tokens = {"[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"}
        vocab_tokens = {tok for tok in vocab.keys() if tok not in special_tokens}
        poi_keys = set(poi_coords.keys())
        inter = vocab_tokens & poi_keys
        match_rate = (len(inter) / max(1, len(poi_keys))) if poi_keys else 0.0
        print(f"[Infer] vocab–coords match_rate: {match_rate:.3f} ({len(inter)}/{len(poi_keys)})")
    except Exception as e:
        print(f"[Infer] vocab–coords match_rate failed: {e}")

    unk_token_ids = []
    for unk_token in ("[UNK]"):
        token_id = vocab.get(unk_token)
        if token_id is not None:
            unk_token_ids.append(int(token_id))
    if not unk_token_ids:
        unk_token_ids = None

    forbidden_token_ids: List[int] = []
    raw_forbid = (args.forbid_token_ids or "").strip()
    if raw_forbid and raw_forbid.lower() not in {"none", "off", "disable"}:
        forbidden_token_ids = [int(tok.strip()) for tok in raw_forbid.split(",") if tok.strip()]

    if args.forbid_poi_special_tokens:
        poi_special_tokens = [args.poi_home_token, args.poi_work_token, args.poi_other_token]
        for token_str in poi_special_tokens:
            token_id = vocab.get(token_str)
            if token_id is not None:
                if token_id not in forbidden_token_ids:
                    forbidden_token_ids.append(int(token_id))
                    print(f"[Infer] Auto-forbidding token: {token_str} (ID: {token_id})")
            else:
                print(f"[Infer] Warning: Token '{token_str}' not found in vocab, cannot forbid")

    if args.forbid_poi_other_only:
        token_id = vocab.get(args.poi_other_token)
        if token_id is not None:
            if token_id not in forbidden_token_ids:
                forbidden_token_ids.append(int(token_id))
                print(f"[Infer] Auto-forbidding POI_OTHER only: {args.poi_other_token} (ID: {token_id})")
        else:
            print(f"[Infer] Warning: Token '{args.poi_other_token}' not found in vocab, cannot forbid")

    poi_special_token_ids: List[int] = []
    if args.penalize_poi_special_tokens > 0.0:
        poi_special_tokens = [args.poi_home_token, args.poi_work_token, args.poi_other_token]
        for token_str in poi_special_tokens:
            token_id = vocab.get(token_str)
            if token_id is not None and token_id not in forbidden_token_ids:
                poi_special_token_ids.append(int(token_id))
                print(f"[Infer] Will penalize token: {token_str} (ID: {token_id}) with penalty={args.penalize_poi_special_tokens}")
            elif token_id is not None:
                print(f"[Infer] Token {token_str} (ID: {token_id}) is already forbidden, skipping penalty")

    poi_home_work_token_ids: List[int] = []
    if args.penalize_poi_home_work > 0.0:
        for token_str in [args.poi_home_token, args.poi_work_token]:
            token_id = vocab.get(token_str)
            if token_id is not None and token_id not in forbidden_token_ids:
                poi_home_work_token_ids.append(int(token_id))
                print(f"[Infer] Will penalize POI_HOME/WORK token: {token_str} (ID: {token_id}) with penalty={args.penalize_poi_home_work}")
            elif token_id is not None:
                print(f"[Infer] Token {token_str} (ID: {token_id}) is already forbidden, skipping penalty")

    poi_home_token_id = None
    poi_work_token_id = None
    poi_other_token_id = None
    if args.penalize_poi_home > 0.0:
        token_id = vocab.get(args.poi_home_token)
        if token_id is not None and token_id not in forbidden_token_ids:
            poi_home_token_id = int(token_id)
            print(f"[Infer] Will penalize POI_HOME only: {args.poi_home_token} (ID: {poi_home_token_id}) with penalty={args.penalize_poi_home}")
        elif token_id is not None:
            print(f"[Infer] POI_HOME (ID: {token_id}) is already forbidden, skipping penalty")

    if args.penalize_poi_work > 0.0:
        token_id = vocab.get(args.poi_work_token)
        if token_id is not None and token_id not in forbidden_token_ids:
            poi_work_token_id = int(token_id)
            print(f"[Infer] Will penalize POI_WORK only: {args.poi_work_token} (ID: {poi_work_token_id}) with penalty={args.penalize_poi_work}")
        elif token_id is not None:
            print(f"[Infer] POI_WORK (ID: {token_id}) is already forbidden, skipping penalty")

    if args.penalize_poi_other > 0.0:
        token_id = vocab.get(args.poi_other_token)
        if token_id is not None and token_id not in forbidden_token_ids:
            poi_other_token_id = int(token_id)
            print(f"[Infer] Will penalize POI_OTHER only: {args.poi_other_token} (ID: {poi_other_token_id}) with penalty={args.penalize_poi_other}")
        elif token_id is not None:
            print(f"[Infer] POI_OTHER (ID: {token_id}) is already forbidden, skipping penalty")

    generation_config = {
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "unk_logit_penalty": args.unk_logit_penalty,
        "forbidden_token_ids": forbidden_token_ids,
        "poi_special_token_ids": poi_special_token_ids if args.penalize_poi_special_tokens > 0.0 else [],
        "poi_special_token_penalty": args.penalize_poi_special_tokens,
        "poi_home_work_token_ids": poi_home_work_token_ids if args.penalize_poi_home_work > 0.0 else [],
        "poi_home_work_penalty": args.penalize_poi_home_work,
        "poi_home_token_id": poi_home_token_id,
        "poi_home_penalty": args.penalize_poi_home,
        "poi_work_token_id": poi_work_token_id,
        "poi_work_penalty": args.penalize_poi_work,
        "poi_other_token_id": poi_other_token_id,
        "poi_other_penalty": args.penalize_poi_other,
    }

    print(f"\n[Infer] Generation settings:")
    print(f"  Forbidden tokens: {len(forbidden_token_ids)} token IDs")
    if args.forbid_poi_special_tokens:
        print(f"  POI special tokens: ALL FORBIDDEN (POI_HOME, POI_WORK, POI_OTHER)")
    elif args.forbid_poi_other_only:
        print(f"  POI special tokens: POI_OTHER FORBIDDEN only")
    if args.penalize_poi_special_tokens > 0.0:
        print(f"  POI special tokens penalty: {args.penalize_poi_special_tokens} (applied to HOME, WORK, OTHER)")
    if args.penalize_poi_home_work > 0.0:
        print(f"  POI_HOME/WORK penalty: {args.penalize_poi_home_work} (applied to HOME, WORK only)")
    if args.penalize_poi_home > 0.0:
        print(f"  POI_HOME penalty: {args.penalize_poi_home}")
    if args.penalize_poi_work > 0.0:
        print(f"  POI_WORK penalty: {args.penalize_poi_work}")
    if args.penalize_poi_other > 0.0:
        print(f"  POI_OTHER penalty: {args.penalize_poi_other}")
    print()

    all_sequences = []
    all_latents = []
    all_coords_batches = []
    all_poi_sequences: List[List[str]] = []
    all_attrs_batches = []
    all_length_targets = []

    global_token_counter = Counter()
    global_total_tokens = 0
    global_poi_home_count = 0
    global_poi_work_count = 0
    global_poi_other_count = 0

    for batch_num, batch_data in enumerate(tqdm(attr_dataloader, desc="Generating trajectories")):
        if args.data_type == "unified":
            if need_length_ids:
                attrs, length_ids, conditional_flags = batch_data
            else:
                attrs, conditional_flags = batch_data
                length_ids = None
            if conditional_flags is not None and not conditional_flags.all():
                attrs_for_model = attrs.clone()
                attrs_for_model[~conditional_flags] = 0
            else:
                attrs_for_model = attrs
        else:
            if need_length_ids:
                attrs, length_ids = batch_data
            else:
                attrs = batch_data[0]
                length_ids = None
            conditional_flags = None
            attrs_for_model = None if args.data_type == "uncontrolled" else attrs

        enforced_lengths = length_ids if (args.force_empirical_length and length_ids is not None) else None

        seq_ids, latents = sample_dit_with_autoencoder(
            dit_model=dit_model,
            noise_scheduler=noise_scheduler,
            autoencoder=autoencoder,
            attr_embeds=attrs_for_model,
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
            actual_batch_size=attrs.shape[0],
            ddim_eta=args.ddim_eta,
            latent_scale=args.latent_scale,
            latent_pca=latent_pca,
            latent_mapper=latent_mapper,
            length_ids=enforced_lengths,
            unk_token_ids=unk_token_ids,
            prediction_type=args.prediction_type,
        )

        attrs_np = attrs.detach().cpu().numpy()
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
        except Exception as e:
            print(f"[Infer] Failed to accumulate batch {batch_num} stats: {e}")

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
    if all_attrs_batches:
        max_attr_dim = max(attrs.shape[1] for attrs in all_attrs_batches)
        padded_attrs = []
        for attrs in all_attrs_batches:
            if attrs.shape[1] < max_attr_dim:
                pad = np.zeros((attrs.shape[0], max_attr_dim - attrs.shape[1]), dtype=attrs.dtype)
                padded_attrs.append(np.concatenate([attrs, pad], axis=1))
            else:
                padded_attrs.append(attrs)
        attrs_out = np.concatenate(padded_attrs, axis=0)
    else:
        attrs_out = np.zeros((0, 4), dtype=np.float32)
    length_out = np.concatenate(all_length_targets, axis=0) if all_length_targets else None

    np.save(os.path.join(output_dir, "generated_sequences.npy"), seq_out.numpy())
    np.save(os.path.join(output_dir, "generated_latents.npy"), lat_out.numpy())
    np.save(os.path.join(output_dir, "generated_coordinates.npy"), coord_out)
    np.save(os.path.join(output_dir, "sampled_attributes.npy"), attrs_out)
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
        "generation_config": generation_config,
        "random_seed": int(args.random_seed),
    }
    with open(os.path.join(output_dir, "generation_params.json"), "w") as f:
        json.dump(generation_params, f, indent=2)

    if global_total_tokens > 0:
        poi_keys = set(poi_coords.keys())
        top_tokens = [t for t, _ in global_token_counter.most_common(100)]
        top100_hit = sum(1 for t in top_tokens if t in poi_keys) / max(1, len(top_tokens))
        unique_non_special = len(global_token_counter)

        print(f"\n[Infer] Global generation statistics (all batches):")
        print(f"  Total non-special tokens: {global_total_tokens}")
        print(f"  Unique POI tokens: {unique_non_special}")
        print(f"  Top 100 hit rate: {top100_hit:.3f} (tokens with coordinate mapping)")
        print(f"  POI special tokens: HOME={global_poi_home_count}, WORK={global_poi_work_count}, OTHER={global_poi_other_count}")
        if unique_non_special > 0:
            print(f"  Top 10 most frequent tokens:")
            for token, count in global_token_counter.most_common(10):
                percentage = (count / global_total_tokens) * 100
                has_coords = "✓" if token in poi_keys else "✗"
                print(f"    {token}: {count} ({percentage:.1f}%) {has_coords}")

    print(f"\nSaved inference outputs to: {output_dir}")
