from __future__ import annotations

import torch
import wandb

from src.checkpoint_utils import save_training_checkpoint
from src.validation import validate_model


def run_periodic_validation(
    *,
    global_step,
    args,
    dit_model,
    noise_scheduler,
    autoencoder,
    valid_dataloader,
    accelerator,
    latent_pca,
    best_val_loss,
    training_dir,
):
    validation_results = validate_model(
        dit_model,
        noise_scheduler,
        autoencoder,
        valid_dataloader,
        args,
        accelerator=accelerator,
        latent_pca=latent_pca,
    )

    if args.data_type == "unified":
        avg_val_loss, avg_val_diffusion, avg_val_anchor, num_conditional, num_unconditional, avg_conditional_loss, avg_unconditional_loss = validation_results
        print(
            f"Validation Step {global_step} - Total: {avg_val_loss:.4f}, "
            f"Diffusion: {avg_val_diffusion:.4f}, "
            f"Anchor: {avg_val_anchor:.4f}"
        )
        print(
            f"  Unified Training - Conditional: {num_conditional} samples (loss: {avg_conditional_loss:.4f}), "
            f"Unconditional: {num_unconditional} samples (loss: {avg_unconditional_loss:.4f})"
        )
        if args.use_wandb and accelerator.is_main_process and wandb.run is not None:
            wandb.log(
                {
                    "val/total_loss": avg_val_loss,
                    "val/diffusion_loss": avg_val_diffusion,
                    "val/anchor_loss": avg_val_anchor,
                    "val/conditional_loss": avg_conditional_loss,
                    "val/unconditional_loss": avg_unconditional_loss,
                    "val/conditional_samples": num_conditional,
                    "val/unconditional_samples": num_unconditional,
                    "val/step": global_step,
                }
            )
    else:
        avg_val_loss, avg_val_diffusion, avg_val_anchor = validation_results
        print(
            f"Validation Step {global_step} - Total: {avg_val_loss:.4f}, "
            f"Diffusion: {avg_val_diffusion:.4f}, "
            f"Anchor: {avg_val_anchor:.4f}"
        )
        if args.use_wandb and accelerator.is_main_process and wandb.run is not None:
            wandb.log(
                {
                    "val/total_loss": avg_val_loss,
                    "val/diffusion_loss": avg_val_diffusion,
                    "val/anchor_loss": avg_val_anchor,
                    "val/step": global_step,
                    "val/enable_length_condition": getattr(args, "enable_length_condition", False),
                }
            )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        with training_dir():
            if accelerator.num_processes > 1:
                torch.save(dit_model.module.state_dict(), "dit_best_val.pt")
            else:
                torch.save(dit_model.state_dict(), "dit_best_val.pt")
            print(f"Saved best validation model with val loss: {best_val_loss:.4f}")
    return best_val_loss


def save_periodic_checkpoint(
    *,
    global_step,
    epoch,
    best_val_loss,
    args,
    dit_model,
    optimizer,
    training_dir,
    accelerator,
):
    wandb_run_id = wandb.run.id if args.use_wandb else None
    save_training_checkpoint(
        dit_model,
        optimizer,
        global_step,
        epoch,
        best_val_loss,
        training_dir,
        accelerator,
        args,
        wandb_run_id,
    )

    with training_dir("state_dicts"):
        model_path = f"dit_step_{global_step}.pt"
        if accelerator.num_processes > 1:
            torch.save(dit_model.module.state_dict(), model_path)
        else:
            torch.save(dit_model.state_dict(), model_path)
        print(f"Saved model-only checkpoint: {model_path}")


def finalize_training(
    *,
    args,
    dit_model,
    noise_scheduler,
    autoencoder,
    valid_dataloader,
    accelerator,
    latent_pca,
    training_dir,
):
    if args.enable_validation and accelerator.is_main_process:
        print("\n--- Final Validation ---")
        validation_results = validate_model(
            dit_model,
            noise_scheduler,
            autoencoder,
            valid_dataloader,
            args,
            accelerator=accelerator,
            latent_pca=latent_pca,
        )

        if args.data_type == "unified":
            avg_val_loss, avg_val_diffusion, avg_val_anchor, num_conditional, num_unconditional, avg_conditional_loss, avg_unconditional_loss = validation_results
            print(
                f"Final Validation - Total: {avg_val_loss:.4f}, "
                f"Diffusion: {avg_val_diffusion:.4f}, "
                f"Anchor: {avg_val_anchor:.4f}"
            )
        else:
            avg_val_loss, avg_val_diffusion, avg_val_anchor = validation_results
            print(
                f"Final Validation - Total: {avg_val_loss:.4f}, "
                f"Diffusion: {avg_val_diffusion:.4f}, "
                f"Anchor: {avg_val_anchor:.4f}"
            )

    if accelerator.is_main_process:
        with training_dir():
            if accelerator.num_processes > 1:
                torch.save(dit_model.module.state_dict(), "dit_final.pt")
            else:
                torch.save(dit_model.state_dict(), "dit_final.pt")
            print("Saved final model")

    if args.use_wandb and accelerator.is_main_process and wandb.run is not None:
        wandb.finish()
