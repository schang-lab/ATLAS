import numpy as np
import torch


class EvaluationDataset:
    """Simple dataset for evaluation without masking"""

    def __init__(self, poi_sequences_df, tokenizer, max_length=512):
        self.poi_sequences = poi_sequences_df["unique_id_seq"].tolist()
        self.df = poi_sequences_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.poi_sequences)

    def __getitem__(self, idx):
        placekey_sequence = self.poi_sequences[idx]
        row = self.df.iloc[idx]

        individual_id = None
        for id_col in ["individual_id", "caid"]:
            if id_col in row:
                individual_id = row[id_col]
                break
        if individual_id is None:
            individual_id = f"sample_{idx}"

        seq_list = list(placekey_sequence) if placekey_sequence is not None else []
        is_id_sequence = any(isinstance(tok, (int, np.integer)) for tok in seq_list)

        special_token_strs = {"[PAD]", "[CLS]", "[SEP]", "[MASK]"}
        special_token_ids = {
            tid
            for tid in [
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.mask_token_id,
            ]
            if tid is not None
        }

        if is_id_sequence:
            input_ids = [int(t) for t in seq_list]
            token_sequence_raw = (
                self.tokenizer.convert_ids_to_tokens([t if t >= 0 else (self.tokenizer.unk_token_id or 0) for t in input_ids])
                if input_ids
                else []
            )
        else:
            token_sequence_raw = [str(t) for t in seq_list]
            input_ids = self.tokenizer.convert_tokens_to_ids(token_sequence_raw)

        if is_id_sequence:
            true_original_ids = [tid for tid in input_ids if tid not in special_token_ids]
            true_original_tokens = self.tokenizer.convert_ids_to_tokens(true_original_ids) if true_original_ids else []
        else:
            true_original_tokens = [tok for tok in token_sequence_raw if tok not in special_token_strs]
        true_original_length = len(true_original_tokens)

        placekey_list = token_sequence_raw
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            token_sequence = placekey_list[: self.max_length]
        else:
            pad_length = self.max_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
            token_sequence = placekey_list + ["[PAD]"] * pad_length
        if len(token_sequence) < self.max_length:
            token_sequence = token_sequence + ["[PAD]"] * (self.max_length - len(token_sequence))

        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "original_sequence": true_original_tokens,
            "original_length": true_original_length,
            "individual_id": individual_id,
            "sample_idx": idx,
            "token_sequence": [str(tok) for tok in token_sequence],
        }


def collate_fn(batch):
    """Custom collate function for evaluation"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    original_sequences = [item["original_sequence"] for item in batch]
    original_lengths = [item["original_length"] for item in batch]
    individual_ids = [item["individual_id"] for item in batch]
    sample_indices = [item["sample_idx"] for item in batch]
    token_sequences = [item["token_sequence"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "original_sequences": original_sequences,
        "original_lengths": original_lengths,
        "individual_ids": individual_ids,
        "sample_indices": sample_indices,
        "token_sequences": token_sequences,
    }
