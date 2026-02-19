from __future__ import annotations

import torch
import wandb

from lib.run_cbg_bootstrap import load_yaml, parse_args
from lib.run_cbg_runtime import (
    init_wandb_if_enabled,
    resume_trainer_if_needed,
    setup_accelerator,
)


def run_entrypoint(trainer_cls) -> None:
    args = parse_args()
    config = load_yaml(args.config)
    accelerator, _ = setup_accelerator(config, args.device)
    device = accelerator.device

    enable_wandb = init_wandb_if_enabled(config, accelerator)
    trainer = trainer_cls(config, device)
    trainer.dit, trainer.optimizer = accelerator.prepare(trainer.dit, trainer.optimizer)
    resume_trainer_if_needed(config, trainer, accelerator)
    trainer.run(accelerator)

    if accelerator.is_main_process and enable_wandb and wandb.run is not None:
        wandb.finish()
