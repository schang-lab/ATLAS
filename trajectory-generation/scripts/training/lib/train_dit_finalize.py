from __future__ import annotations

import torch
from torch import optim

from src.checkpoint_utils import load_training_checkpoint


def finalize_trainable_components(
    args,
    rank: str,
    device: torch.device,
    accelerator,
    autoencoder,
    latent_pca,
    dit_model,
    train_dataloader,
    valid_dataloader,
    resume_checkpoint_path,
):
    autoencoder = autoencoder.to(device)
    if latent_pca is not None:
        latent_pca.to(device)

    autoencoder_params = list(autoencoder.parameters())
    ae_param_count = sum(p.numel() for p in autoencoder_params)
    print(f"[Rank {rank}] Autoencoder params={ae_param_count}", flush=True)
    for param in autoencoder_params:
        param.requires_grad = False

    dit_parameters = list(dit_model.parameters())
    trainable_param_count = sum(p.numel() for p in dit_parameters if p.requires_grad)
    print(f"[Rank {rank}] DiT trainable param tensors={len(dit_parameters)} | elements={trainable_param_count}", flush=True)
    optimizer = optim.Adam(dit_parameters, lr=args.OPTIM_LR)

    if resume_checkpoint_path is not None:
        try:
            print("Restoring model and optimizer from checkpoint into training objects")
            load_training_checkpoint(resume_checkpoint_path, dit_model, optimizer, accelerator, args)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint into training objects: {e}")

    if accelerator.num_processes > 1:
        if torch.distributed.is_available() and torch.distributed.is_initialized() and accelerator.device.type == "cuda":
            torch.distributed.barrier(device_ids=[accelerator.device.index])
        else:
            accelerator.wait_for_everyone()

    dit_model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        dit_model, optimizer, train_dataloader, valid_dataloader
    )
    return autoencoder, optimizer, dit_model, train_dataloader, valid_dataloader
