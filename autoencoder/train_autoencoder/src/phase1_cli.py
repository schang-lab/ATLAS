import argparse
import os
from pathlib import Path


def parse_phase1_args(logger):
    parser = argparse.ArgumentParser(description="Phase 1: BART Pretraining with Masked Language Modeling")

    # Data arguments
    parser.add_argument("--data_folder", type=str, help="Single data folder path (legacy mode)")
    parser.add_argument("--controlled_folder", type=str, help="Path to controlled data folder")
    parser.add_argument("--uncontrolled_folder", type=str, help="Path to uncontrolled data folder")
    parser.add_argument("--dual_folders", nargs=2, metavar=("CONTROLLED", "UNCONTROLLED"), help="Controlled and uncontrolled folder paths")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--use_predefined_splits", action="store_true", default=True, help="Use predefined train/val splits instead of random splitting")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./phase1_pretrain_output", help="Output directory for model and logs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")

    # Masking arguments
    parser.add_argument("--mask_prob", type=float, default=0.3, help="Probability of masking tokens")
    parser.add_argument("--span_mask", action="store_true", default=True, help="Use span masking")
    parser.add_argument("--max_span", type=int, default=3, help="Maximum span length for masking")

    # Logging arguments
    parser.add_argument("--eval_steps", type=int, default=1500, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save frequency")
    parser.add_argument("--logging_steps", type=int, default=500, help="Logging frequency")

    # Model loading arguments
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to pretrained BART model to resume training from")
    parser.add_argument("--reset_optimizer", action="store_true", help="Reset optimizer state when resuming (start fresh optimization)")

    # Model architecture arguments
    parser.add_argument("--encoder_layers", type=int, default=6, help="Number of encoder layers (default: 6, was 4 in original)")
    parser.add_argument("--decoder_layers", type=int, default=4, help="Number of decoder layers (default: 4, was 2 in original)")
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden dimension (default: 512, was 256 in original)")
    parser.add_argument("--encoder_ffn_dim", type=int, default=2048, help="Encoder feed-forward dimension (default: 2048, was 512 in original)")
    parser.add_argument("--decoder_ffn_dim", type=int, default=2048, help="Decoder feed-forward dimension (default: 2048, was 512 in original)")
    parser.add_argument("--encoder_attention_heads", type=int, default=4, help="Number of encoder attention heads (default: 4, was 4 in original)")
    parser.add_argument("--decoder_attention_heads", type=int, default=4, help="Number of decoder attention heads (default: 4, was 4 in original)")

    # Wandb arguments
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="trajectory-autoencoder-phase1", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, help="Wandb run name")
    parser.add_argument("--resume_wandb_id", type=str, help="Wandb run ID to resume from")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="Optional wandb API key. Prefer setting WANDB_API_KEY env variable.")

    # Epoch Evaluation arguments
    parser.add_argument("--enable_epoch_evaluation", action="store_true", help="Enable evaluation at the end of each epoch")
    parser.add_argument("--eval_sample_size", type=int, default=1000, help="Number of test samples to evaluate per epoch (smaller = faster)")
    parser.add_argument("--eval_frequency", type=int, default=1, help="Evaluate every N epochs (1 = every epoch, 5 = every 5th epoch)")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for epoch evaluation (smaller = less GPU memory)")
    parser.add_argument("--eval_output_dir", type=str, default=None, help="Directory for epoch evaluation results (default: {output_dir}/epoch_evaluations)")

    args = parser.parse_args()

    # Handle dual_folders argument
    if args.dual_folders:
        args.controlled_folder, args.uncontrolled_folder = args.dual_folders

    # Validate required data source args
    if not any([args.data_folder, (args.controlled_folder and args.uncontrolled_folder)]):
        raise ValueError("Must provide either --data_folder or both --controlled_folder and --uncontrolled_folder")

    # Validate resume arguments
    if args.resume_from_checkpoint and not os.path.exists(args.resume_from_checkpoint):
        raise ValueError(f"Checkpoint path not found: {args.resume_from_checkpoint}")

    # Smart checkpoint recommendation
    if args.resume_from_checkpoint:
        checkpoint_dir = Path(args.resume_from_checkpoint).parent
        checkpoint_name = Path(args.resume_from_checkpoint).name
        if checkpoint_dir.exists():
            checkpoints = sorted([p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()])
            if checkpoint_name == "final_model" and checkpoints:
                latest_checkpoint = checkpoints[-1]
                logger.info(" Consider using %s instead of final_model", latest_checkpoint.name)
                logger.info("   Reason: Checkpoints preserve optimizer state and learning rate schedule")
                logger.info("   Available checkpoints: %s", [p.name for p in checkpoints[-3:]])

    # Set up evaluation output directory
    if args.enable_epoch_evaluation and args.eval_output_dir is None:
        args.eval_output_dir = os.path.join(args.output_dir, "epoch_evaluations")

    os.makedirs(args.output_dir, exist_ok=True)
    return args
