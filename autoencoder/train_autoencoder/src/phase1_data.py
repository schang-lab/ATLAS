import logging
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)


class MaskedTrajectoryCollator:
    """
    Custom data collator for masked trajectory training.
    Implements span masking similar to SpanBERT for trajectory sequences.
    """

    def __init__(self, tokenizer, mask_prob=0.3, span_mask=True, max_span=3):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_prob = mask_prob
        self.span_mask = span_mask
        self.max_span = max_span

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        # Ignore padding positions in the loss.
        pad_positions = attention_mask == 0
        labels = labels.clone()
        labels[pad_positions] = -100

        corrupted_input = input_ids.clone()
        for i in range(input_ids.size(0)):
            corrupted_input[i] = self.corrupt_sequence(input_ids[i])

        return {
            "input_ids": corrupted_input,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def corrupt_sequence(self, seq):
        seq = seq.tolist()
        valid_len = len([x for x in seq if x != self.pad_token_id])
        if valid_len == 0:
            return torch.tensor(seq)

        indices = list(range(valid_len))
        num_mask = int(self.mask_prob * valid_len)
        random.shuffle(indices)

        corrupted = seq[:]
        i = 0
        while num_mask > 0 and i < len(indices):
            idx = indices[i]
            span_len = random.randint(1, self.max_span) if self.span_mask else 1
            for j in range(span_len):
                if idx + j < valid_len and corrupted[idx + j] != self.pad_token_id:
                    corrupted[idx + j] = self.mask_token_id
                    num_mask -= 1
                    if num_mask <= 0:
                        break
            i += 1
        return torch.tensor(corrupted)


class PretrainTrajectoryDataset(Dataset):
    """
    Dataset for Phase 1 pretraining with basic trajectory sequences.
    Focuses on reconstruction without compression features.
    """

    def __init__(self, poi_sequences_df, tokenizer, max_length=512):
        self.poi_sequences = poi_sequences_df["unique_id_seq"].tolist()
        self.individual_ids = poi_sequences_df["individual_id"].tolist()
        self.cities = poi_sequences_df["city"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.poi_sequences)

    def __getitem__(self, idx):
        placekey_sequence = self.poi_sequences[idx]
        input_ids = self.tokenizer.convert_tokens_to_ids(placekey_sequence)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
        else:
            pad_length = self.max_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": input_ids_tensor.clone(),
        }


def load_dual_sequences_only(controlled_folder, uncontrolled_folder, split="train"):
    """Load and combine controlled + uncontrolled sequence data only."""
    controlled_folder = Path(controlled_folder)
    uncontrolled_folder = Path(uncontrolled_folder)

    logger.info("Loading dual dataset sequences for Phase 1 pretraining (split: %s)...", split)

    controlled_data_path = controlled_folder / split / "final_segments_all_train_data.pkl"
    controlled_sequences_df = pd.read_pickle(controlled_data_path)
    logger.info("Loaded %s controlled sequences from %s", len(controlled_sequences_df), controlled_data_path)

    uncontrolled_data_path = uncontrolled_folder / split / "final_segments_all_train_data.pkl"
    uncontrolled_sequences_df = pd.read_pickle(uncontrolled_data_path)
    logger.info(
        "Loaded %s uncontrolled sequences from %s",
        len(uncontrolled_sequences_df),
        uncontrolled_data_path,
    )

    combined_sequences_df = pd.concat([controlled_sequences_df, uncontrolled_sequences_df], ignore_index=True)
    logger.info("Combined dataset: %s total sequences", len(combined_sequences_df))
    return combined_sequences_df


def load_sequences_only(controlled_folder=None, uncontrolled_folder=None, data_folder=None, split="train"):
    """Load only sequence data for Phase 1 pretraining."""
    if controlled_folder is not None and uncontrolled_folder is not None:
        return load_dual_sequences_only(controlled_folder, uncontrolled_folder, split)

    data_folder = Path(data_folder)
    logger.info("Loading processed POI sequences for Phase 1 (split: %s)...", split)
    data_path = data_folder / split / "final_segments_all_train_data.pkl"
    poi_sequences_df = pd.read_pickle(data_path)
    logger.info("Loaded %s POI sequences from %s", len(poi_sequences_df), data_path)
    return poi_sequences_df


def load_tokenizer(controlled_folder=None, uncontrolled_folder=None, data_folder=None):
    """Load tokenizer once and reuse for all data loading operations."""
    base_folder = Path(controlled_folder) if controlled_folder is not None else Path(data_folder)
    tokenizer_paths = [base_folder / "train" / "tokenizer"]

    tokenizer_path = None
    for path in tokenizer_paths:
        if path.exists():
            tokenizer_path = path
            break
    if tokenizer_path is None:
        raise FileNotFoundError(f"Could not find tokenizer in any of the expected locations: {tokenizer_paths}")

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    logger.info("Loaded tokenizer from %s with vocab size: %s", tokenizer_path, len(tokenizer))
    return tokenizer
