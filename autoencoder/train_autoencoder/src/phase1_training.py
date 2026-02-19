import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from transformers import BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

from phase1_common import is_global_process_zero
from phase1_data import load_sequences_only, load_tokenizer
from phase1_eval import EpochEvaluationCallback


@dataclass
class Phase1DataBundle:
    tokenizer: object
    train_df: object
    val_df: object
    test_df: object | None


def initialize_wandb(args, logger):
    if args.disable_wandb or not is_global_process_zero():
        return

    if getattr(args, "wandb_api_key", None):
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY is not set. Wandb init may fail in non-interactive environments.")

    if args.wandb_name is None:
        args.wandb_name = f"phase1-pretrain-{args.batch_size}bs-{args.epochs}ep"

    tags = ["phase1", "pretrain", "masked-lm"]
    notes = "Phase 1: BART pretraining with masked language modeling"
    if args.resume_from_checkpoint:
        notes += f" (resumed from {os.path.basename(args.resume_from_checkpoint)})"

    if args.resume_wandb_id:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=tags,
            notes=notes,
            resume="must",
            id=args.resume_wandb_id,
        )
    else:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=tags,
            notes=notes,
        )
    logger.info("Wandb initialized: %s/%s", args.wandb_project, args.wandb_name)


def load_phase1_data(args, logger):
    use_dual = bool(args.controlled_folder and args.uncontrolled_folder)
    if use_dual:
        tokenizer = load_tokenizer(
            controlled_folder=args.controlled_folder,
            uncontrolled_folder=args.uncontrolled_folder,
        )
        train_df = load_sequences_only(
            controlled_folder=args.controlled_folder,
            uncontrolled_folder=args.uncontrolled_folder,
            split="train",
        )
        val_df = load_sequences_only(
            controlled_folder=args.controlled_folder,
            uncontrolled_folder=args.uncontrolled_folder,
            split="val",
        )
        test_df = (
            load_sequences_only(
                controlled_folder=args.controlled_folder,
                uncontrolled_folder=args.uncontrolled_folder,
                split="test",
            )
            if args.enable_epoch_evaluation
            else None
        )
    else:
        tokenizer = load_tokenizer(data_folder=args.data_folder)
        train_df = load_sequences_only(data_folder=args.data_folder, split="train")
        val_df = load_sequences_only(data_folder=args.data_folder, split="val")
        test_df = (
            load_sequences_only(data_folder=args.data_folder, split="test")
            if args.enable_epoch_evaluation
            else None
        )

    logger.info("Loaded dataframes: train=%s, val=%s, test=%s", len(train_df), len(val_df), len(test_df) if test_df is not None else 0)
    return Phase1DataBundle(tokenizer=tokenizer, train_df=train_df, val_df=val_df, test_df=test_df)


def build_or_load_model(args, tokenizer, logger, create_config_fn):
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        logger.info("Loading pretrained BART model from: %s", args.resume_from_checkpoint)
        try:
            model = BartForConditionalGeneration.from_pretrained(args.resume_from_checkpoint)
            logger.info("Successfully loaded pretrained model with %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")

            if checkpoint_path.name.startswith("checkpoint-"):
                logger.info("Loaded from training checkpoint - optimizer state will be preserved")
            elif checkpoint_path.name == "final_model":
                logger.info("Loaded from final model - optimizer state not available")

            current_vocab_size = len(tokenizer)
            model_vocab_size = model.config.vocab_size
            if current_vocab_size != model_vocab_size:
                logger.warning("Vocabulary size mismatch: current=%s, model=%s", current_vocab_size, model_vocab_size)
                logger.info("Resizing model embeddings to match current tokenizer...")
                model.resize_token_embeddings(current_vocab_size)
            return model
        except Exception as e:
            logger.error("Failed to load pretrained model: %s", e)
            logger.info("Falling back to creating new model...")
            args.resume_from_checkpoint = None

    logger.info("Creating new BART model...")
    end_token_id = tokenizer.sep_token_id if getattr(tokenizer, "sep_token_id", None) is not None else tokenizer.get_vocab().get("[SEP]", 2)
    bart_config = create_config_fn(
        vocab_size=len(tokenizer),
        end_token_id=end_token_id,
        max_position_embeddings=args.max_length,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        d_model=args.d_model,
        encoder_ffn_dim=args.encoder_ffn_dim,
        decoder_ffn_dim=args.decoder_ffn_dim,
        encoder_attention_heads=args.encoder_attention_heads,
        decoder_attention_heads=args.decoder_attention_heads,
    )
    model = BartForConditionalGeneration(bart_config)
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Created new BART model with %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    return model


def build_training_args(args, logger):
    training_args_dict = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "dataloader_num_workers": 16,
        "dataloader_pin_memory": True,
        "save_total_limit": 3,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "wandb" if not args.disable_wandb else None,
        "run_name": args.wandb_name if not args.disable_wandb else None,
        "fp16": getattr(args, "fp16", False),
        "bf16": getattr(args, "bf16", True),
        "gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
    }
    training_args_dict["ddp_find_unused_parameters"] = False

    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        is_checkpoint = checkpoint_path.name.startswith("checkpoint-")
        is_final_model = checkpoint_path.name == "final_model"
        if is_checkpoint:
            logger.info("Resuming from checkpoint: %s", checkpoint_path.name)
            logger.info("Will preserve optimizer state and learning rate schedule")
            training_args_dict["resume_from_checkpoint"] = args.resume_from_checkpoint
        elif is_final_model:
            logger.info("Resuming from final model: %s", checkpoint_path)
            logger.warning("final_model typically doesn't contain optimizer state")
            logger.warning("Training will restart with fresh optimizer and warmup")
        else:
            logger.info("Resuming from model: %s", checkpoint_path)
            logger.info("Optimizer state preservation depends on model format")

        if args.reset_optimizer:
            logger.info("Optimizer state will be reset (fresh optimization)")
            training_args_dict.pop("resume_from_checkpoint", None)

    try:
        training_args_dict["eval_strategy"] = "steps"
        return Seq2SeqTrainingArguments(**training_args_dict)
    except TypeError:
        training_args_dict["evaluation_strategy"] = "steps"
        try:
            return Seq2SeqTrainingArguments(**training_args_dict)
        except TypeError:
            for key in ["bf16", "fp16", "gradient_checkpointing"]:
                training_args_dict.pop(key, None)
            return Seq2SeqTrainingArguments(**training_args_dict)


def build_callbacks(args, tokenizer, test_df, logger):
    callbacks = []
    if args.enable_epoch_evaluation and test_df is not None and is_global_process_zero():
        logger.info("Setting up epoch evaluation callback...")
        eval_args = type(
            "EvalArgs",
            (),
            {
                "eval_sample_size": args.eval_sample_size,
                "eval_frequency": args.eval_frequency,
                "eval_batch_size": args.eval_batch_size,
                "eval_output_dir": args.eval_output_dir,
                "max_length": args.max_length,
            },
        )()
        callbacks.append(
            EpochEvaluationCallback(
                test_poi_sequences_df=test_df,
                tokenizer=tokenizer,
                eval_args=eval_args,
                wandb_enabled=not args.disable_wandb,
            )
        )
        logger.info("Epoch evaluation enabled: every %s epochs with %s samples", args.eval_frequency, args.eval_sample_size)
    return callbacks


def build_trainer(model, training_args, train_dataset, eval_dataset, data_collator, callbacks, tokenizer):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "callbacks": callbacks,
    }
    try:
        trainer_kwargs["processing_class"] = tokenizer
        return Seq2SeqTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs["tokenizer"] = tokenizer
        return Seq2SeqTrainer(**trainer_kwargs)


def finalize_and_log_metrics(args, logger, trainer, eval_dataset):
    if is_global_process_zero():
        logger.info("*** Final Evaluation ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    try:
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
    except OverflowError:
        metrics["perplexity"] = float("inf")

    if is_global_process_zero():
        logger.info("Final evaluation metrics: %s", metrics)
        if not args.disable_wandb:
            wandb.log({"final_metrics": metrics})
            wandb.finish()
    return metrics
