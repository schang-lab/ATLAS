import os
import logging
import sys
from pathlib import Path

from transformers import (
    BartConfig,
)

_THIS_DIR = Path(__file__).resolve().parent
_PHASE1_SRC = _THIS_DIR / "src"
if str(_PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(_PHASE1_SRC))

from phase1_data import (
    MaskedTrajectoryCollator,
    PretrainTrajectoryDataset,
)
from phase1_cli import parse_phase1_args
from phase1_common import is_global_process_zero
from phase1_training import (
    build_callbacks,
    build_or_load_model,
    build_trainer,
    build_training_args,
    finalize_and_log_metrics,
    initialize_wandb,
    load_phase1_data,
)


# ========== Logging Setup ==========
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_bart_config_phase1(
    vocab_size,
    end_token_id=2,
    max_position_embeddings=64,
    encoder_layers=6,
    decoder_layers=4,
    d_model=512,
    encoder_ffn_dim=1024,
    decoder_ffn_dim=1024,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
):
    """
    Create BART configuration for Phase 1 pretraining with configurable architecture.
    
    Default configuration is larger than original:
    - Encoder layers: 6 (was 4)
    - Decoder layers: 4 (was 2) 
    - d_model: 512 (was 256)
    - FFN dim: 1024 (was 512)
    - Attention heads: 8 (was 4)
    """
    config = BartConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        encoder_layers=encoder_layers,
        encoder_ffn_dim=encoder_ffn_dim,
        encoder_attention_heads=encoder_attention_heads,
        decoder_layers=decoder_layers,
        decoder_ffn_dim=decoder_ffn_dim,
        decoder_attention_heads=decoder_attention_heads,
        d_model=d_model,
        scale_embedding=False,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=end_token_id,
        forced_eos_token_id=end_token_id,
        decoder_start_token_id=1,
    )
    return config


def train_phase1_pretrain(args):
    """Main training function for Phase 1 pretraining"""
    initialize_wandb(args, logger)

    bundle = load_phase1_data(args, logger)
    tokenizer = bundle.tokenizer

    # Create datasets
    train_dataset = PretrainTrajectoryDataset(
        poi_sequences_df=bundle.train_df,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    eval_dataset = PretrainTrajectoryDataset(
        poi_sequences_df=bundle.val_df,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Test data loaded above if epoch evaluation is enabled
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create masked trajectory collator
    data_collator = MaskedTrajectoryCollator(
        tokenizer=tokenizer,
        mask_prob=args.mask_prob,
        span_mask=args.span_mask,
        max_span=args.max_span
    )
    
    model = build_or_load_model(args, tokenizer, logger, create_bart_config_phase1)
    
    logger.info("Using random initialization for all token embeddings")
    
    training_args = build_training_args(args, logger)
    callbacks = build_callbacks(args, tokenizer, bundle.test_df, logger)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("*** Starting Phase 1 Masked BART Pretraining ***")
    trainer.train()
    
    # Save final model (main process only)
    final_model_path = os.path.join(args.output_dir, "final_model")
    if is_global_process_zero():
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Saved final model to: {final_model_path}")
    
    finalize_and_log_metrics(args, logger, trainer, eval_dataset)
    
    return final_model_path


def main():
    args = parse_phase1_args(logger)
    final_model_path = train_phase1_pretrain(args)
    logger.info(f"Phase 1 pretraining completed. Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()