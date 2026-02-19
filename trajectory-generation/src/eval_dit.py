
#!/usr/bin/env python3
"""Standalone evaluation script for DiT-only checkpoints."""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Tuple, Optional

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import BartConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.training import trajectory_dataset, get_default_args  # noqa: E402
from src.validation import validate_model  # noqa: E402
from src.dit import DiT  # noqa: E402
from src.diffusion_model import GaussianDiffusion  # noqa: E402
from src.helpers import normalize_prediction_type  # noqa: E402
from auto_encoder.traj_compressed_ae import BARTLatentCompression  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DiT-only checkpoint")
    parser.add_argument('--config', required=True, help='Path to DiT config YAML')
    parser.add_argument('--checkpoint', required=True, help='Path to DiT checkpoint (.pt/.safetensors)')
    parser.add_argument('--autoencoder-path', required=True, help='Path to autoencoder directory')
    parser.add_argument('--data-dir', required=True, help='Root directory of pre-split data used for training')
    parser.add_argument('--data-type', choices=['controlled', 'uncontrolled', 'unified'], default='unified')
    parser.add_argument('--training-phase', choices=['phase1', 'phase2'], default='phase2')
    parser.add_argument('--ablation-mode', choices=['coords_only', 'subcat_only', 'both', 'neither', 'pure'], default='both')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--diag-decode-val', action='store_true', help='Enable diagnostic decoding during validation')
    parser.add_argument('--diag-decode-total', type=int, default=0, help='Target number of samples for diag decode (per process)')
    parser.add_argument('--diag-decode-batch', type=int, default=16, help='Batch size for diag decode')
    parser.add_argument('--diag-num-beams', type=int, default=1)
    parser.add_argument('--diag-ddim-steps', type=int, default=0)
    parser.add_argument('--diag-guidance-scale', type=float, default=1.0)
    parser.add_argument('--prediction-type', type=str, default='epsilon',
                        help="Diffusion prediction target: 'epsilon', 'x0', or 'v'.")
    parser.add_argument('--timestep-sampling', type=str, default='logsnr', choices=['uniform', 'logsnr'],
                        help="Distribution to sample diffusion timesteps when computing validation losses.")
    parser.add_argument('--use-anchor-loss', action='store_true',
                        help='Enable anchor loss component during evaluation to match training settings')
    parser.add_argument('--anchor-loss-weight', type=float, default=1.0,
                        help='Weight applied to the anchor loss term during evaluation (default: 1.0)')
    parser.add_argument('--eval-samples', type=int, default=0, help='Limit number of validation samples (0 = all)')
    parser.add_argument('--output', type=str, default=None, help='Optional JSON file to dump metrics')
    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument(
        '--force-no-compression',
        action='store_true',
        help='Override autoencoder config and disable latent compression during evaluation'
    )
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_dit(config: dict) -> DiT:
    dit_params = {**get_default_args(DiT), **config.get('DiT', {})}
    return DiT(**dit_params)


def load_dit_checkpoint(model: DiT, checkpoint_path: str, verbose: bool = False) -> None:
    state = torch.load(checkpoint_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose and (missing or unexpected):
        print(f"Warning: missing keys {missing}, unexpected keys {unexpected}")


def load_autoencoder(args, config: dict) -> Tuple[BARTLatentCompression, Optional[int], bool]:
    config_path = os.path.join(args.autoencoder_path, 'config.json')
    ae_config = BartConfig.from_json_file(config_path)
    no_compression = getattr(ae_config, 'no_compression', False)

    if args.force_no_compression:
        no_compression = True
        ae_config.no_compression = True
    elif args.training_phase == 'phase2':
        cardiff_cfg = config.get('cardiff', {}) if isinstance(config, dict) else {}
        seq_len = cardiff_cfg.get('second_stage_seq_len')
        decoder_latents = getattr(ae_config, 'num_decoder_latents', None)
        if seq_len and decoder_latents and seq_len != decoder_latents:
            if getattr(args, 'is_main_process', True) or getattr(args, 'verbose', False):
                print(
                    f"[eval_dit] second_stage_seq_len ({seq_len}) != num_decoder_latents ({decoder_latents}); "
                    "enabling no_compression so DiT receives the expected sequence length."
                )
            no_compression = True
            ae_config.no_compression = True

    def _instantiate(disable_compression: bool, allow_mismatch: bool) -> BARTLatentCompression:
        ae_config.no_compression = disable_compression
        return BARTLatentCompression.from_pretrained(
            args.autoencoder_path,
            config=ae_config,
            num_encoder_latents=getattr(ae_config, 'num_encoder_latents', 16),
            num_decoder_latents=getattr(ae_config, 'num_decoder_latents', 32),
            dim_ae=getattr(ae_config, 'dim_ae', ae_config.d_model),
            use_coords=True,
            num_sub_categories=getattr(ae_config, 'num_sub_categories', None),
            no_compression=disable_compression,
            ignore_mismatched_sizes=allow_mismatch
        )

    actual_no_compression = no_compression
    try:
        autoencoder = _instantiate(no_compression, allow_mismatch=no_compression and args.force_no_compression)
    except RuntimeError as err:
        if no_compression and not args.force_no_compression:
            if getattr(args, 'is_main_process', True) or getattr(args, 'verbose', False):
                print('[eval_dit] Falling back to compressed latents because the checkpoint '
                      'weights do not match the no-compression architecture.\n'
                      f'  Original load error: {err}')
            actual_no_compression = False
            autoencoder = _instantiate(False, allow_mismatch=False)
        else:
            raise

    target_seq_len = config.get('cardiff', {}).get('second_stage_seq_len', None)
    if not actual_no_compression:
        target_seq_len = getattr(ae_config, 'num_decoder_latents', target_seq_len)

    return autoencoder, target_seq_len, actual_no_compression


def main() -> None:
    args = parse_args()
    args.prediction_type = normalize_prediction_type(getattr(args, 'prediction_type', 'epsilon'))
    args.timestep_sampling = getattr(args, 'timestep_sampling', 'logsnr')
    args.timestep_sampling = args.timestep_sampling.lower().replace('-', '_')

    if args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.cuda.is_available = lambda: False  # type: ignore

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    args.is_main_process = accelerator.is_main_process

    if accelerator.is_main_process or args.verbose:
        print("=== DiT Evaluation ===")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print("======================")

    config = load_config(args.config)

    if 'prediction_type' in config:
        args.prediction_type = normalize_prediction_type(config['prediction_type'])
    if 'timestep_sampling' in config:
        args.timestep_sampling = str(config['timestep_sampling']).lower().replace('-', '_')

    args.prediction_type = normalize_prediction_type(args.prediction_type)
    args.timestep_sampling = args.timestep_sampling.lower().replace('-', '_')

    seq_len_override = None
    autoencoder = None
    autoencoder_no_compression = False

    if args.training_phase == 'phase2':
        autoencoder, seq_len_override, autoencoder_no_compression = load_autoencoder(args, config)
        base_seq_len = config.get('cardiff', {}).get('second_stage_seq_len', 512)
        seq_len = seq_len_override or base_seq_len
        config.setdefault('DiT', {})['in_channels'] = seq_len
        config['DiT']['image_size'] = seq_len
    else:
        cardiff_cfg = config.get('cardiff', {})
        seq_len = cardiff_cfg.get('second_stage_seq_len', cardiff_cfg.get('first_stage_len', 512))
        config.setdefault('DiT', {})['in_channels'] = seq_len
        config['DiT']['image_size'] = seq_len

    dit_model = build_dit(config)
    load_dit_checkpoint(dit_model, args.checkpoint, verbose=args.verbose)

    if args.training_phase != 'phase2':
        from transformers import BartForConditionalGeneration
        autoencoder = BartForConditionalGeneration.from_pretrained(args.autoencoder_path)

    if config.get('cardiff'):
        timesteps = config['cardiff'].get('second_stage_seq_len', 512)
    else:
        timesteps = config.get('TIMESTEPS', 1000)
    noise_scheduler = GaussianDiffusion(timesteps=timesteps)

    class EvalArgs:
        pass

    eval_args = EvalArgs()
    eval_args.data_dir = args.data_dir
    eval_args.data_type = args.data_type
    eval_args.BATCH_SIZE = args.batch_size
    eval_args.NUM_WORKERS = args.num_workers
    eval_args.training_phase = args.training_phase
    eval_args.autoencoder_path = args.autoencoder_path
    eval_args.ablation_mode = 'both' if args.ablation_mode == 'both' else args.ablation_mode
    eval_args.config = args.config
    eval_args.TIMESTEPS = timesteps
    eval_args.use_anchor_loss = args.use_anchor_loss
    eval_args.anchor_loss_weight = args.anchor_loss_weight
    eval_args.enable_validation = True
    eval_args.eval_samples = args.eval_samples
    eval_args.diag_decode_val = args.diag_decode_val
    eval_args.diag_decode_total = args.diag_decode_total
    eval_args.diag_decode_batch = args.diag_decode_batch
    eval_args.diag_num_beams = args.diag_num_beams
    eval_args.diag_ddim_steps = args.diag_ddim_steps
    eval_args.validation_guidance_scale = args.diag_guidance_scale
    eval_args.prediction_type = args.prediction_type
    eval_args.timestep_sampling = args.timestep_sampling
    eval_args.use_wandb = False
    eval_args.is_main_process = accelerator.is_main_process

    _, val_loader, _, _ = trajectory_dataset(
        eval_args,
        testset=False,
        data_dir=args.data_dir,
        data_type=args.data_type
    )

    dit_model, autoencoder, val_loader = accelerator.prepare(dit_model, autoencoder, val_loader)
    noise_scheduler = noise_scheduler.to(accelerator.device)

    metrics = validate_model(
        dit_model,
        noise_scheduler,
        autoencoder,
        val_loader,
        eval_args,
        accelerator=accelerator
    )

    if accelerator.is_main_process:
        print("=== Evaluation Results ===")
        if args.data_type == 'unified':
            (avg_loss, avg_diff, avg_anchor,
             num_cond, num_uncond,
             avg_cond_loss, avg_uncond_loss) = metrics
            print(f"Total loss: {avg_loss:.4f}")
            print(f"Diffusion loss: {avg_diff:.4f}")
            print(f"Anchor loss: {avg_anchor:.4f}")
            print(f"Conditional samples: {num_cond} (loss {avg_cond_loss:.4f})")
            print(f"Unconditional samples: {num_uncond} (loss {avg_uncond_loss:.4f})")
        else:
            avg_loss, avg_diff, avg_anchor = metrics
            print(f"Total loss: {avg_loss:.4f}")
            print(f"Diffusion loss: {avg_diff:.4f}")
            print(f"Anchor loss: {avg_anchor:.4f}")

    if args.output and accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(metrics if isinstance(metrics, (tuple, list)) else metrics.__dict__, f, indent=2)
        print(f"Metrics written to {args.output}")


if __name__ == '__main__':
    main()
