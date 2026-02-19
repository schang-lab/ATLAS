"""DiT training loop: step-by-step training with diffusion + optional anchor loss."""
from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm

import wandb

from src.helpers import normalize_prediction_type
from src.latent_pca import LatentPCA

from lib.train_dit_helpers import run_ddim_probe
from lib.train_dit_diag import run_train_diag_decode
from lib.train_dit_post import finalize_training, run_periodic_validation, save_periodic_checkpoint
from lib.train_dit_step import execute_train_batch_step


def DiTTrain(
    timestamp,
    args,
    dit_model,
    noise_scheduler,
    autoencoder,
    train_dataloader,
    valid_dataloader,
    training_dir,
    optimizer,
    accelerator,
    tokenizer_vocab=None,
    resume_state=None,
    latent_pca: Optional[LatentPCA] = None,
):
    best_loss = torch.tensor(9999999)
    best_val_loss = torch.tensor(9999999)

    # Initialize training state (can be overridden by resume_state)
    global_step = 0
    epoch = 0

    prediction_type = normalize_prediction_type(getattr(args, 'prediction_type', 'epsilon'))
    timestep_sampling = getattr(args, 'timestep_sampling', 'logsnr')
    timestep_sampling = timestep_sampling.lower().replace('-', '_')

    # Apply resume state if provided
    if resume_state is not None:
        global_step = resume_state['global_step']
        epoch = resume_state['epoch']
        best_val_loss = resume_state['best_val_loss']
        print(f"Resuming training from step {global_step}, epoch {epoch}")

    # Calculate total steps
    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        total_steps = args.EPOCHS * len(train_dataloader)

    print(f"Training for {total_steps} steps (max_steps: {args.max_steps}, epochs: {args.EPOCHS})")
    print(f"Starting from step {global_step}, epoch {epoch}")
    running_train_loss = 0.0
    running_diffusion_loss = 0.0
    running_anchor_loss = 0.0
    num_batches_since_log = 0
    accumulation_steps = 0

    # Create infinite iterator for training data
    train_iter = iter(train_dataloader)

    # Progress bar for steps
    progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not accelerator.is_main_process)

    while global_step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            # Restart iterator for next epoch
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            epoch += 1
            if accelerator.is_main_process:
                print(f"\n--- EPOCH {epoch} ---")

        # Debug: print batch keys only once
        if global_step == 0 and accelerator.is_main_process:
            print(f"Batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")

        dit_model.train(True)

        step_ctx, counters = execute_train_batch_step(
            args=args,
            batch=batch,
            dit_model=dit_model,
            noise_scheduler=noise_scheduler,
            autoencoder=autoencoder,
            accelerator=accelerator,
            latent_pca=latent_pca,
            prediction_type=prediction_type,
            timestep_sampling=timestep_sampling,
            running_train_loss=running_train_loss,
            running_diffusion_loss=running_diffusion_loss,
            running_anchor_loss=running_anchor_loss,
            num_batches_since_log=num_batches_since_log,
            accumulation_steps=accumulation_steps,
        )
        target_latents = step_ctx["target_latents"]
        noisy_latents = step_ctx["noisy_latents"]
        predictions = step_ctx["predictions"]
        t = step_ctx["t"]
        attention_mask = step_ctx["attention_mask"]
        labels = step_ctx["labels"]
        latent_scale = step_ctx["latent_scale"]
        running_train_loss = counters["running_train_loss"]
        running_diffusion_loss = counters["running_diffusion_loss"]
        running_anchor_loss = counters["running_anchor_loss"]
        num_batches_since_log = counters["num_batches_since_log"]
        accumulation_steps = counters["accumulation_steps"]

        # Update weights every gradient_accumulation_steps
        if accumulation_steps % args.gradient_accumulation_steps == 0:
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()
            accumulation_steps = 0
            global_step += 1
            progress_bar.update(1)

            # Log training loss every log_steps
            if global_step % args.log_steps == 0 and accelerator.is_main_process:
                avg_total_loss = running_train_loss / num_batches_since_log
                avg_diffusion_loss = running_diffusion_loss / num_batches_since_log
                avg_anchor_loss = running_anchor_loss / num_batches_since_log if args.use_anchor_loss else None

                if args.use_anchor_loss and avg_anchor_loss is not None:
                    print(f'Step {global_step} - Total: {avg_total_loss.item():.4f}, '
                          f'Diffusion: {avg_diffusion_loss.item():.4f}, '
                          f'Anchor: {avg_anchor_loss.item():.4f} '
                          f'(weight: {args.anchor_loss_weight:.4f})')
                else:
                    print(f'Step {global_step} - Total: {avg_total_loss.item():.4f}, '
                          f'Diffusion: {avg_diffusion_loss.item():.4f}')

                # Log to wandb
                if args.use_wandb and accelerator.is_main_process and wandb.run is not None:
                    log_dict = {
                        "train/total_loss": avg_total_loss.item(),
                        "train/diffusion_loss": avg_diffusion_loss.item(),
                    }
                    if args.use_anchor_loss and avg_anchor_loss is not None:
                        log_dict.update({
                            "train/anchor_loss": avg_anchor_loss.item(),
                            "train/anchor_weight": args.anchor_loss_weight,
                        })
                    wandb.log(log_dict | {
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "train/epoch": epoch,
                        "train/step": global_step,
                    })

                # Reset running losses
                running_train_loss = 0.0
                running_diffusion_loss = 0.0
                running_anchor_loss = 0.0
                num_batches_since_log = 0

            run_ddim_now = (
                getattr(args, "ddim_probe_every", 0) > 0
                and global_step >= getattr(args, "ddim_probe_start", 0)
                and global_step % args.ddim_probe_every == 0
                and accelerator.is_main_process
            )
            if run_ddim_now:
                training_state = dit_model.training
                dit_model.eval()
                try:
                    probe_logs = run_ddim_probe(
                        model=dit_model,
                        noise_scheduler=noise_scheduler,
                        steps=args.ddim_probe_steps,
                        batch_size=args.ddim_probe_batch,
                        seq_len=target_latents.size(1),
                        latent_dim=target_latents.size(2),
                        device=noisy_latents.device,
                        prediction_type=prediction_type,
                    )
                finally:
                    if training_state:
                        dit_model.train(True)

                ddim_summary = {f"ddim/{k}": v[-1] for k, v in probe_logs.items() if v}
                summary_str = " | ".join(f"{k}={v:.4f}" for k, v in ddim_summary.items())
                print(f"DDIM probe @ step {global_step}: {summary_str}")

                if args.use_wandb and wandb.run is not None:
                    wandb.log(ddim_summary | {"train/step": global_step})

            # Diagnostic decoding over multiple mini-batches (no_compression only)
            if getattr(args, 'diag_decode_every', 0) and (global_step % args.diag_decode_every == 0):
                train_iter = run_train_diag_decode(
                    args=args,
                    global_step=global_step,
                    accelerator=accelerator,
                    autoencoder=autoencoder,
                    train_iter=train_iter,
                    train_dataloader=train_dataloader,
                    target_latents=target_latents,
                    noisy_latents=noisy_latents,
                    predictions=predictions,
                    t=t,
                    attention_mask=attention_mask,
                    labels=labels,
                    latent_pca=latent_pca,
                    latent_scale=latent_scale,
                    noise_scheduler=noise_scheduler,
                    timestep_sampling=timestep_sampling,
                    dit_model=dit_model,
                    prediction_type=prediction_type,
                )

            # Run validation every eval_steps
            if args.enable_validation and global_step % args.eval_steps == 0:
                best_val_loss = run_periodic_validation(
                    global_step=global_step,
                    args=args,
                    dit_model=dit_model,
                    noise_scheduler=noise_scheduler,
                    autoencoder=autoencoder,
                    valid_dataloader=valid_dataloader,
                    accelerator=accelerator,
                    latent_pca=latent_pca,
                    best_val_loss=best_val_loss,
                    training_dir=training_dir,
                )

            # Save checkpoint every save_steps
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_periodic_checkpoint(
                    global_step=global_step,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    args=args,
                    dit_model=dit_model,
                    optimizer=optimizer,
                    training_dir=training_dir,
                    accelerator=accelerator,
                )

    progress_bar.close()

    finalize_training(
        args=args,
        dit_model=dit_model,
        noise_scheduler=noise_scheduler,
        autoencoder=autoencoder,
        valid_dataloader=valid_dataloader,
        accelerator=accelerator,
        latent_pca=latent_pca,
        training_dir=training_dir,
    )
