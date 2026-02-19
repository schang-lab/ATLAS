from __future__ import annotations

import random
from typing import Dict

import torch
from accelerate import Accelerator


def validate(trainer, accelerator: Accelerator) -> Dict[str, float]:
    if not trainer.val_enabled or trainer.val_cache is None or trainer.val_poi_store is None or not trainer.val_cbgs:
        accelerator.wait_for_everyone()
        return {}

    trainer.dit.eval()
    total_loss = 0.0
    total_agg = 0.0
    total_entropy = 0.0
    total_unique = 0.0
    n = 0
    num_cbgs = len(trainer.val_cbgs)
    sum_loss_by_idx = torch.zeros(num_cbgs, device=accelerator.device, dtype=torch.float32)
    sum_agg_by_idx = torch.zeros(num_cbgs, device=accelerator.device, dtype=torch.float32)
    count_by_idx = torch.zeros(num_cbgs, device=accelerator.device, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(trainer.val_num_batches):
            cbg_idx = random.randrange(num_cbgs)
            cbg = trainer.val_cbgs[cbg_idx]
            loss, stats = trainer._aggregate_loss_for(
                cbg=cbg,
                cache=trainer.val_cache,
                poi_store=trainer.val_poi_store,
                batch_size=trainer.val_batch_size,
                num_special_tokens=trainer.val_num_special_tokens,
                llp_demo_source=trainer.val_llp_demo_source,
                coord_dropout_p=trainer.val_aggregate_coord_dropout,
            )
            total_loss += float(loss.detach().item())
            total_agg += float(stats.get("agg_loss", 0.0))
            total_entropy += float(stats.get("entropy_loss", 0.0))
            total_unique += float(stats.get("unique_loss", 0.0))
            n += 1
            if trainer.val_log_by_cbg:
                sum_loss_by_idx[cbg_idx] += loss.detach().to(dtype=torch.float32)
                sum_agg_by_idx[cbg_idx] += torch.tensor(float(stats.get("agg_loss", 0.0)), device=accelerator.device)
                count_by_idx[cbg_idx] += 1.0

    pack = torch.tensor(
        [total_loss, total_agg, total_entropy, total_unique, float(n)],
        device=accelerator.device,
        dtype=torch.float32,
    )
    gathered = accelerator.gather(pack)
    out: Dict[str, float] = {}
    if accelerator.is_main_process:
        if gathered is None or gathered.numel() == 0:
            accelerator.wait_for_everyone()
            trainer.dit.train()
            return {}

        if gathered.dim() == 1:
            summed = gathered
        else:
            summed = gathered.sum(dim=0)

        denom = float(summed[4].item()) if summed.numel() >= 5 and float(summed[4].item()) > 0 else 1.0
        out["val_loss"] = float(summed[0].item() / denom)
        out["val_agg_loss"] = float(summed[1].item() / denom)
        if trainer.use_entropy_reg and trainer.lambda_entropy > 0.0:
            out["val_entropy_loss"] = float(summed[2].item() / denom)
        if trainer.use_unique_reg and trainer.lambda_unique > 0.0:
            out["val_unique_loss"] = float(summed[3].item() / denom)

    if trainer.val_log_by_cbg and num_cbgs > 0:
        by_cbg_pack = torch.stack([sum_loss_by_idx, sum_agg_by_idx, count_by_idx], dim=0)
        by_cbg_gathered = accelerator.gather(by_cbg_pack)

        if accelerator.is_main_process and by_cbg_gathered is not None and by_cbg_gathered.numel() > 0:
            if by_cbg_gathered.dim() == 2:
                by_cbg_sum = by_cbg_gathered
            else:
                by_cbg_sum = by_cbg_gathered.sum(dim=0)

            sum_loss = by_cbg_sum[0]
            sum_agg = by_cbg_sum[1]
            cnt = by_cbg_sum[2].clamp_min(0.0)
            if trainer.val_max_cbgs_to_log > 0:
                max_to_log = min(trainer.val_max_cbgs_to_log, num_cbgs)
            else:
                max_to_log = num_cbgs if num_cbgs <= 8 else 0

            logged = 0
            for i, cbg in enumerate(trainer.val_cbgs):
                if logged >= max_to_log:
                    break
                if float(cnt[i].item()) <= 0:
                    continue
                denom_i = float(cnt[i].item())
                out[f"val_loss_by_cbg/{cbg}"] = float(sum_loss[i].item() / denom_i)
                out[f"val_agg_loss_by_cbg/{cbg}"] = float(sum_agg[i].item() / denom_i)
                out[f"val_batches_by_cbg/{cbg}"] = float(denom_i)
                logged += 1

    accelerator.wait_for_everyone()
    trainer.dit.train()
    return out
