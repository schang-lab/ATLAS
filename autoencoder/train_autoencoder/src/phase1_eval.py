import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from phase1_common import is_global_process_zero

logger = logging.getLogger(__name__)


class EpochEvaluationCallback(TrainerCallback):
    """
    Uses evaluate_phase1_pretrain.py functions for consistent evaluation during training.
    """

    def __init__(self, test_poi_sequences_df, tokenizer, eval_args, wandb_enabled=True):
        self.test_poi_sequences_df = test_poi_sequences_df
        self.tokenizer = tokenizer
        self.eval_args = eval_args
        self.wandb_enabled = wandb_enabled
        self.eval_output_dir = Path(eval_args.eval_output_dir)
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.eval_output_dir / "epoch_metrics.csv"
        self.init_metrics_csv()

        logger.info("EpochEvaluationCallback initialized:")
        logger.info("Test dataset size: %s", len(test_poi_sequences_df))
        logger.info("Sample size per epoch: %s", eval_args.eval_sample_size)
        logger.info("Evaluation frequency: every %s epochs", eval_args.eval_frequency)
        logger.info("Output directory: %s", self.eval_output_dir)

    def init_metrics_csv(self):
        if not self.metrics_file.exists():
            headers = [
                "epoch",
                "step",
                "token_accuracy_mean",
                "token_accuracy_std",
                "token_accuracy_lenient_mean",
                "sequence_accuracy_mean",
                "bleu_1_mean",
                "bleu_2_mean",
                "bleu_3_mean",
                "bleu_4_mean",
                "edit_distance_mean",
                "jaccard_similarity_mean",
                "original_length_mean",
                "reconstructed_length_mean",
                "perfect_reconstructions",
                "total_samples",
                "perfect_rate",
                "evaluation_time_seconds",
            ]
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not is_global_process_zero():
            return

        current_epoch = int(state.epoch)
        if current_epoch % self.eval_args.eval_frequency != 0:
            return

        logger.info("Starting epoch %s evaluation...", current_epoch)
        start_time = datetime.now()
        try:
            metrics = self.run_evaluation(model, current_epoch, state.global_step)
            self.save_metrics_to_csv(metrics, current_epoch, state.global_step, start_time)
            if self.wandb_enabled and wandb.run is not None:
                wandb_metrics = {f"epoch_eval/{k}": v for k, v in metrics.items() if k != "evaluation_time_seconds"}
                wandb_metrics["epoch_eval/epoch"] = current_epoch
                wandb.log(wandb_metrics, step=state.global_step)
            eval_time = (datetime.now() - start_time).total_seconds()
            logger.info("Epoch %s evaluation completed in %.1fs", current_epoch, eval_time)
            logger.info("Token accuracy: %.4f", metrics["token_accuracy_mean"])
            logger.info("Sequence accuracy: %.4f", metrics["sequence_accuracy_mean"])
            logger.info("BLEU-1: %.4f", metrics["bleu_1_mean"])
        except Exception as e:
            logger.error("Epoch evaluation failed: %s", e)
            import traceback

            logger.error("Traceback: %s", traceback.format_exc())

    def run_evaluation(self, model, epoch, step):
        sys.path.append(str(Path(__file__).parent.parent / "eval"))
        from evaluate_phase1_pretrain import EvaluationDataset, TrajectoryEvaluator, collate_fn, aggregate_metrics

        test_df = self.test_poi_sequences_df
        if self.eval_args.eval_sample_size < len(test_df):
            test_df = test_df.sample(n=self.eval_args.eval_sample_size, random_state=42 + epoch).reset_index(drop=True)

        eval_dataset = EvaluationDataset(test_df, self.tokenizer, max_length=self.eval_args.max_length)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.eval_args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=8,
        )

        generation_config = {
            "max_length": self.eval_args.max_length,
            "min_length": 14,
            "num_beams": 4,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 0,
            "length_penalty": 1.0,
        }

        device = next(model.parameters()).device
        evaluator = TrajectoryEvaluator(model, self.tokenizer, device, generation_config)

        all_metrics = []
        all_detailed_results = []
        model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                batch_metrics, batch_detailed_results = evaluator.evaluate_batch(batch)
                all_metrics.extend(batch_metrics)
                all_detailed_results.extend(batch_detailed_results)
        model.train()

        aggregated_results = aggregate_metrics(all_metrics)
        aggregated_results["total_samples"] = len(all_metrics)
        aggregated_results["perfect_reconstructions"] = sum(1 for m in all_metrics if m["sequence_accuracy"] == 1.0)
        aggregated_results["perfect_rate"] = aggregated_results["perfect_reconstructions"] / len(all_metrics)
        return aggregated_results

    def save_metrics_to_csv(self, metrics, epoch, step, start_time):
        eval_time = (datetime.now() - start_time).total_seconds()
        row = [
            epoch,
            step,
            metrics.get("token_accuracy_mean", 0.0),
            metrics.get("token_accuracy_std", 0.0),
            metrics.get("token_accuracy_lenient_mean", 0.0),
            metrics.get("sequence_accuracy_mean", 0.0),
            metrics.get("bleu_1_mean", 0.0),
            metrics.get("bleu_2_mean", 0.0),
            metrics.get("bleu_3_mean", 0.0),
            metrics.get("bleu_4_mean", 0.0),
            metrics.get("edit_distance_mean", 0.0),
            metrics.get("jaccard_similarity_mean", 0.0),
            metrics.get("original_length_mean", 0.0),
            metrics.get("reconstructed_length_mean", 0.0),
            metrics.get("perfect_reconstructions", 0),
            metrics.get("total_samples", 0),
            metrics.get("perfect_rate", 0.0),
            eval_time,
        ]
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(",".join(map(str, row)) + "\n")
