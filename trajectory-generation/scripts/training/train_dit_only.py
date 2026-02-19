import os
import os
import sys
from argparse import ArgumentParser

# Allow running this script from any working directory by adding `trajectory-generation/` to sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)

from src.training import (
                          create_directory,
                          get_model_size,
    load_restart_training_parameters,
                          save_training_info,
    trajectory_dataset,
)
from lib.train_dit_loop import DiTTrain
from lib.train_dit_runtime import (
    apply_seed,
    infer_sequence_length_from_config,
    init_wandb_if_enabled,
    load_resume_state_if_available,
    print_training_configuration,
    resolve_resume_checkpoint_path,
    resolve_resume_wandb_id,
    setup_accelerator_and_device,
)
from lib.train_dit_setup import (
    build_dit_model,
    build_noise_scheduler,
    load_config_and_prepare_dit,
    validate_autoencoder_files_and_load_args,
)
from lib.train_dit_autoencoder import build_autoencoder
from lib.train_dit_finalize import finalize_trainable_components
from transformers import BartConfig
from src.helpers import normalize_prediction_type


def main(args):
    apply_seed(args)
    print_training_configuration(args)
    latent_pca = None

    accelerator, device, timestamp = setup_accelerator_and_device(args)
    
    # Determine checkpoint path for resuming
    resume_checkpoint_path = resolve_resume_checkpoint_path(args)
    
    # Check if resuming from checkpoint to get wandb_id
    resume_wandb_id = resolve_resume_wandb_id(resume_checkpoint_path)
    
    # Initialize wandb if enabled (after accelerator is created)
    init_wandb_if_enabled(args, accelerator, timestamp, resume_wandb_id)

    # Create training directory
    dir_path = f"./training_dit_only_{timestamp}"
    training_dir = create_directory(dir_path)

    # If loading from a parameters/training directory
    if args.RESTART_DIRECTORY is not None:
        args = load_restart_training_parameters(args)
    elif args.PARAMETERS is not None:
        args = load_restart_training_parameters(args, justparams=True)

    # Load checkpoint for resuming training if specified (BEFORE autoencoder creation)
    resume_state = load_resume_state_if_available(args, device, accelerator, resume_checkpoint_path)

    # Dataset - pass training phase information
    # Pre-load config to infer the desired sequence length for the token dataset.
    # Important: trajectory_dataset() is called before the main config parsing below,
    # and it currently decides max_length (padding/truncation) from args.
    # For phase1 DiT training, we want this to match DiT.image_size (e.g., 64),
    # otherwise BART will output latents with a different T than DiT expects.
    infer_sequence_length_from_config(args)

    print(f"Loading dataset for {args.training_phase} training...")
    train_dataloader, valid_dataloader, _, tokenizer_vocab = trajectory_dataset(
        args,
        data_dir=args.data_dir,
        data_type=args.data_type
    )
    
    print(f"\n=== Length Conditioning ===")
    if getattr(args, 'enable_length_condition', False):
        inferred_vocab = int(getattr(args, 'length_vocab_size', 0))
        if inferred_vocab <= 0:
            inferred_vocab = 513
            args.length_vocab_size = inferred_vocab
        print(f"Trajectory length conditioning enabled (vocab size: {args.length_vocab_size})")
    else:
        print("Trajectory length conditioning disabled")

    # Load autoencoder path for config loading
    latent_model_path = args.autoencoder_path
    
    # Validate autoencoder path exists
    if not os.path.exists(latent_model_path):
        raise FileNotFoundError(f"Autoencoder path does not exist: {latent_model_path}")

    # Config for ori-AE (needed for vocabulary size)
    ae_config = BartConfig.from_json_file(os.path.join(latent_model_path, "config.json"))
    
    config, dit_params, latent_pca = load_config_and_prepare_dit(args, device)
    dit_model, rank = build_dit_model(args, device, dit_params)
    noise_scheduler = build_noise_scheduler(args, device, rank)

    # Get model size
    model_size_MB = get_model_size(dit_model)
    print(f'DiT Model size MB: {model_size_MB}')

    # Save training info
    save_training_info(args, timestamp, [dit_params], {}, model_size_MB, training_dir)

    latent_argparse = validate_autoencoder_files_and_load_args(args, latent_model_path, dit_params)

    autoencoder = build_autoencoder(args, latent_model_path, ae_config, latent_argparse, device)

    autoencoder, optimizer, dit_model, train_dataloader, valid_dataloader = finalize_trainable_components(
        args=args,
        rank=rank,
        device=device,
        accelerator=accelerator,
        autoencoder=autoencoder,
        latent_pca=latent_pca,
        dit_model=dit_model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        resume_checkpoint_path=resume_checkpoint_path,
    )

    # Frozen modules already moved to device above; ensure scheduler on device
    # noise_scheduler = noise_scheduler.to(accelerator.device)

    # Checkpoint loading already done earlier - just use the resume_state

    # Start training
    DiTTrain(timestamp, args, dit_model, noise_scheduler, autoencoder, train_dataloader, valid_dataloader, training_dir, optimizer, accelerator, tokenizer_vocab, resume_state, latent_pca)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--PARAMETERS", dest="PARAMETERS", help="Parameters directory to load from",
                        default=None, type=str)
    parser.add_argument("-n", "--NUM_WORKERS", dest="NUM_WORKERS", help="Number of workers for DataLoader", default=8,
                        type=int)
    parser.add_argument("-b", "--BATCH_SIZE", dest="BATCH_SIZE", help="Batch size", default=64, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("-e", "--EPOCHS", dest="EPOCHS", help="Number of training epochs", default=100, type=int)
    # Note: TRAIN_VALID_FRAC removed - using pre-split train/val/test directories
    parser.add_argument("-t", "--TIMESTEPS", dest="TIMESTEPS", help="Number of timesteps in Diffusion process",
                        default=1000, type=int)
    parser.add_argument("--prediction_type", type=str, default="epsilon",
                        help="Training target for the diffusion model: 'epsilon', 'x0', or 'v'.")
    parser.add_argument("--timestep_sampling", type=str, default="logsnr",
                        choices=["uniform", "logsnr"],
                        help="Strategy for sampling diffusion timesteps during training ('uniform' index space or 'logsnr').")
    parser.add_argument("--beta_schedule", type=str, default="linear",
                        choices=["linear", "cosine", "logsnr", "logsnr_linear", "log-snr"],
                        help="Variance schedule type for GaussianDiffusion")
    parser.add_argument("--cosine_s", type=float, default=0.008,
                        help="Cosine schedule offset parameter (only used when beta_schedule=cosine)")
    parser.add_argument("--logsnr_max", type=float, default=30.0,
                        help="Maximum log-SNR for logsnr schedule")
    parser.add_argument("--logsnr_min", type=float, default=-2.0,
                        help="Minimum log-SNR for logsnr schedule")
    parser.add_argument("--latent_scale", type=float, default=1.0,
                        help="Divide autoencoder latents by this factor during training (set >1.0 to shrink latent magnitudes)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (sets Python/NumPy/PyTorch + cuDNN deterministic mode)")
    parser.add_argument("-lr", "--OPTIM_LR", dest="OPTIM_LR", help="Learning rate for Adam optimizer", default=1e-4,
                        type=float)
    parser.add_argument("-rd", "--RESTART_DIRECTORY", dest="RESTART_DIRECTORY",
                        help="Training directory to resume training from if restarting.", default=None, type=str)
    parser.add_argument("--dit_checkpoint_path", type=str, default=None,
                        help="Path to DiT model checkpoint for fine-tuning (e.g., dit_step_10000.pt or dit_best_val.pt). "
                             "Recommended to use lower learning rate (e.g., 1e-5) when fine-tuning.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to complete training checkpoint for resuming training (e.g., training_checkpoint_step_10000.pt). "
                             "This will resume training from the exact step, including optimizer state and wandb run.")
    parser.add_argument("--auto_resume_from_dir", type=str, default=None,
                        help="Automatically find and resume from the latest checkpoint in the specified training directory "
                             "(e.g., ./training_dit_only_20231201_120000). Will look for training_checkpoint_latest.pt first.")
    parser.add_argument("-ts", "--TIMESTAMP", dest="timestamp", help="Timestamp for training directory", type=str,
                        default=None)
    parser.add_argument("-config", "--CONFIG", dest="config", help="model config", type=str,
                        default="config.yml")
    parser.add_argument("-force_cpu", "--force_cpu",
                        dest="force_cpu", help="Force CPU usage even if CUDA is available", 
                        action="store_true", default=False)
    
    # Autoencoder arguments
    parser.add_argument("--training_phase", type=str, choices=["phase1", "phase2"], default="phase1",
                        help="Training phase: phase1 (simple BART) or phase2 (compressed autoencoder)")
    parser.add_argument("--autoencoder_path", type=str, required=True,
                        help="Path to pretrained autoencoder model directory")
    parser.add_argument("--ablation_mode", type=str, choices=["coords_only", "subcat_only", "both", "neither", "pure"],
                        default="both", help="Ablation mode for phase2 training (ignored for phase1)")
    parser.add_argument("--no_compression", action="store_true", default=False,
                        help="Skip compression and work with full 512-length sequences while adding coordinate/subcategory features")
    parser.add_argument("--force_full_attention_mask", action="store_true", default=False,
                        help="Override attention masks to all ones so diffusion also learns padded positions")
    
    # Anchor loss arguments
    parser.add_argument("--use_anchor_loss", action="store_true", default=False,
                        help="Enable anchor loss regularization between decoded tokens and ground truth")
    parser.add_argument("--anchor_loss_weight", type=float, default=1.0,
                        help="Weight applied to the anchor loss term (default: 1.0)")
    
    # Validation arguments
    parser.add_argument("--enable_validation", action="store_true", default=False,
                        help="Enable validation evaluation during training")
    parser.add_argument("--eval_samples", type=int, default=256,
                        help="Number of validation samples to use per evaluation (default: use all)")
    parser.add_argument("--validation_guidance_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale to apply during validation (1.0 disables guidance)")
    
    # Data path arguments
    parser.add_argument("--data_dir", type=str, default="split_data_new",
                        help="Base directory containing train/val/test splits")
    parser.add_argument("--data_type", type=str, choices=["controlled", "uncontrolled", "unified"], default="unified",
                        help="Type of data to use (controlled, uncontrolled, or unified for mixed training)")
    parser.add_argument("--conditional_dropout", type=float, default=0.2,
                        help="Probability of dropping attributes during unified training (for robust conditional/unconditional learning)")
    parser.add_argument("--coord_dropout", type=float, default=0.0,
                        help="Probability of dropping ONLY the first 4 coord dims (work/home lat/lon) for conditional samples. "
                             "Keeps other attrs (e.g., demographic ids) intact.")
    
    # Trajectory length conditioning arguments
    parser.add_argument("--enable_length_condition", action="store_true", default=False,
                        help="Enable discrete trajectory length conditioning (appends length id to attributes)")
    parser.add_argument("--length_vocab_size", type=int, default=513,
                        help="Vocabulary size for trajectory length ids (default: 513 covering lengths 0-512)")
    
    # Step-based logging arguments
    parser.add_argument("--log_steps", type=int, default=500,
                        help="Log training loss every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Run validation every N steps (if validation is enabled)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of training steps (overrides epochs)")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps for learning rate scheduler")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="dit-trajectory-generation",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="Wandb run ID to resume (use 'wandb run list' to find existing run ID)")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="Wandb API key (optional, can also use environment variable)")

    # Latent PCA projection
    parser.add_argument("--latent_pca_path", type=str, default=None,
                        help="Path to PCA artifact for projecting autoencoder latents before diffusion")

    # Diagnostics: decode H' and predicted x0 during training (no_compression)
    parser.add_argument("--diag_decode_every", type=int, default=0,
                        help="Run diagnostic decoding every N optimizer steps (0=disabled)")
    parser.add_argument("--diag_decode_batch", type=int, default=8,
                        help="How many samples to decode in diagnostics")
    parser.add_argument("--diag_num_beams", type=int, default=4,
                        help="Number of beams for diagnostic generation")
    parser.add_argument("--diag_decode_val", action="store_true", default=False,
                        help="Run diagnostic decoding on the first validation batch as well")
    parser.add_argument("--diag_decode_total", type=int, default=0,
                        help="Target total trajectories per diagnostic (0=use only current batch)")
    parser.add_argument("--diag_guidance_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale for diagnostic decoding (1.0 disables guidance)")
    parser.add_argument("--shuffle_val", action="store_true", default=False,
                        help="Enable shuffling of validation dataloader for varied diagnostic subsets")
    parser.add_argument("--diag_ddim_steps", type=int, default=0,
                        help="Run a short DDIM chain in validation diagnostics (0=disabled)."
                             " If >0, from q(x_t|x_0) we run k steps t->0 to reconstruct before decoding.")
    parser.add_argument("--ddim_probe_every", type=int, default=0,
                        help="If >0, run a compressed-space DDIM probe every N steps (0 disables)")
    parser.add_argument("--ddim_probe_steps", type=int, default=25,
                        help="Number of DDIM steps for the probe")
    parser.add_argument("--ddim_probe_batch", type=int, default=8,
                        help="Probe batch size in latent space")
    parser.add_argument("--ddim_probe_start", type=int, default=0,
                        help="Start running DDIM probes after this global step")

    args = parser.parse_args()
    args.prediction_type = normalize_prediction_type(getattr(args, 'prediction_type', 'epsilon'))
    args.timestep_sampling = getattr(args, 'timestep_sampling', 'logsnr')
    args.timestep_sampling = args.timestep_sampling.lower().replace('-', '_')
    
    # Validate validation arguments
    if not args.enable_validation and (args.eval_samples is not None):
        print("Warning: Validation arguments provided but --enable_validation not set. Validation will be skipped.")
    
    main(args)
