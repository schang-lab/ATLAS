from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs


def setup_accelerator(config: Dict[str, Any], device_override: Optional[str]) -> Tuple[Accelerator, str]:
    device_str = device_override or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(cpu=(device_str == "cpu"), kwargs_handlers=[ddp_kwargs])
    return accelerator, device_str


def init_wandb_if_enabled(config: Dict[str, Any], accelerator: Accelerator) -> bool:
    train_cfg = config.get("training", {}) or {}
    enable_wandb = bool(train_cfg.get("wandb", False))
    wandb_project = str(train_cfg.get("wandb_project") or config.get("wandb_project") or "trajectory-cbg-finetune")
    wandb_entity = train_cfg.get("wandb_entity") or config.get("wandb_entity") or None

    if accelerator.is_main_process and enable_wandb:
        init_kwargs = {"project": wandb_project, "config": config}
        if wandb_entity:
            init_kwargs["entity"] = wandb_entity
        wandb.init(**init_kwargs)
    return enable_wandb


def resume_trainer_if_needed(config: Dict[str, Any], trainer, accelerator: Accelerator) -> None:
    resume_path = str(config.get("training", {}).get("resume_from") or "") or None
    if not resume_path:
        return
    try:
        payload = torch.load(resume_path, map_location=accelerator.device)
        if not isinstance(payload, dict):
            print(f"[WARN] Resume payload at {resume_path} is not a dict; skipping")
            return

        try:
            unwrapped = accelerator.unwrap_model(trainer.dit)
        except Exception:
            unwrapped = trainer.dit
        if "dit" in payload:
            missing, unexpected = unwrapped.load_state_dict(payload["dit"], strict=False)
            if missing:
                print(f"[WARN] Resume missing keys: {missing}")
            if unexpected:
                print(f"[WARN] Resume unexpected keys: {unexpected}")

        if "optimizer" in payload:
            try:
                trainer.optimizer.load_state_dict(payload["optimizer"])
            except Exception as opt_e:
                print(f"[WARN] Failed to restore optimizer state ({opt_e}); continuing with fresh optimizer")

        trainer.start_step = int(payload.get("step", 0) or 0)
        print(f"[INFO] Resumed from {resume_path} (start_step={trainer.start_step})")
    except Exception as e:
        print(f"[WARN] Failed to resume from {resume_path}: {e}")
