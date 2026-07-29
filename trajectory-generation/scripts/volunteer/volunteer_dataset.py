"""Dataset loader for VOLUNTEER-ATLAS training.

Reads from EMBEE_SPLIT_DATA/controlled/{train,val,test}/
and produces batches compatible with VolunteerVAE.forward().
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class VolunteerTrajectoryDataset(Dataset):
    """Loads pre-tokenized trajectory data for VOLUNTEER-ATLAS training.

    Each sample returns:
        loc      : (T,) long   — POI token IDs
        tim      : (T,) long   — discretized dwell times (minutes, clamped)
        pos      : (T,) float  — absolute timestamps in minutes from epoch
        mask     : (T,) float  — attention mask (1=valid, 0=pad)
        age_bin  : ()   long
        gender_id: ()   long
        home     : (2,) float  — (lat, lon)
        work     : (2,) float  — (lat, lon)
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 64,
        tim_buckets: int = 1440,
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.tim_buckets = tim_buckets

        # Load direct token -> id mapping. We only need convert_tokens_to_ids()
        # for already-tokenized POI strings, so avoid HuggingFace tokenizer-class
        # warnings from mixed BertTokenizer / PreTrainedTokenizerFast metadata.
        vocab_path = self.data_dir / "tokenizer" / "vocab.txt"
        with open(vocab_path, "r") as f:
            vocab = [line.rstrip("\n") for line in f]
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.pad_id = self.token_to_id.get("[PAD]", self.token_to_id.get("<pad>", 0))
        self.unk_id = self.token_to_id.get("[UNK]", self.token_to_id.get("<unk>", self.pad_id))

        # Load trajectory segments
        with open(self.data_dir / "final_segments_all_train_data.pkl", "rb") as f:
            segments_df = pickle.load(f)

        # Convert token strings to integer IDs
        self.loc_ids = []
        self.attention_masks = []
        for _, row in segments_df.iterrows():
            token_strs = row["unique_id_seq"]
            ids = [self.token_to_id.get(str(token), self.unk_id) for token in token_strs]
            self.loc_ids.append(ids)
            self.attention_masks.append(row["attention_mask"])

        # Embee attrs:
        #   all_attr_results.npy           -> [work_lat, work_lon, home_lat, home_lon]
        #   all_attr_results_with_demo.npy -> [work_lat, work_lon, home_lat, home_lon, ..., age_bin, gender_id]
        attrs_with_demo_path = self.data_dir / "all_attr_results_with_demo.npy"
        attrs_path = self.data_dir / "all_attr_results.npy"
        self.has_demo_attrs = attrs_with_demo_path.exists()
        if self.has_demo_attrs:
            attrs = np.load(attrs_with_demo_path, allow_pickle=True).astype(np.float32)
            if attrs.ndim != 2 or attrs.shape[1] < 6:
                raise ValueError(
                    f"Expected all_attr_results_with_demo.npy to have shape [N,6+], got {attrs.shape}"
                )
        elif attrs_path.exists():
            attrs = np.load(attrs_path, allow_pickle=True).astype(np.float32)
            if attrs.ndim != 2 or attrs.shape[1] < 4:
                raise ValueError(f"Expected all_attr_results.npy to have shape [N,4+], got {attrs.shape}")
        else:
            raise FileNotFoundError(
                f"Neither all_attr_results_with_demo.npy nor all_attr_results.npy exists in {self.data_dir}"
            )
        self.work = attrs[:, 0:2]      # (N, 2)
        self.home = attrs[:, 2:4]      # (N, 2)
        if self.has_demo_attrs:
            self.age_bin = attrs[:, -2]    # (N,)
            self.gender_id = attrs[:, -1]  # (N,)
        else:
            self.age_bin = np.full(attrs.shape[0], -1, dtype=np.float32)
            self.gender_id = np.full(attrs.shape[0], -1, dtype=np.float32)

        # Load dwell times (in seconds) -> convert to minutes, discretize
        dwell_raw = np.load(
            self.data_dir / "all_dwell.npy", allow_pickle=True
        ).astype(np.float32)
        # Convert seconds -> minutes, clamp to [0, tim_buckets-1]
        self.dwell = np.clip(dwell_raw / 60.0, 0, tim_buckets - 1).astype(np.int64)

        # Load absolute timestamps
        ts_raw = np.load(
            self.data_dir / "all_timestamp.npy", allow_pickle=True
        )
        # Convert to minutes from first timestamp
        self.timestamps = self._timestamps_to_minutes(ts_raw)

        self.n_samples = len(self.loc_ids)
        assert self.n_samples == attrs.shape[0], (
            f"Mismatch: {self.n_samples} segments vs {attrs.shape[0]} attrs"
        )

    def _timestamps_to_minutes(self, ts_raw: np.ndarray) -> np.ndarray:
        """Convert numpy datetime64 timestamps to float minutes from trajectory start."""
        result = np.zeros_like(ts_raw, dtype=np.float64)
        for i in range(ts_raw.shape[0]):
            row = ts_raw[i]
            # Find first valid (non-NaT) timestamp
            valid = ~np.isnat(row) if hasattr(row[0], 'astype') else np.ones(len(row), dtype=bool)
            try:
                ts_dt = row.astype("datetime64[s]").astype(np.float64)
                start = ts_dt[0]
                result[i] = (ts_dt - start) / 60.0  # minutes from start
            except (ValueError, TypeError):
                result[i] = np.arange(len(row), dtype=np.float64) * 30.0  # fallback
        return result.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        loc = torch.tensor(self.loc_ids[idx][: self.max_seq_len], dtype=torch.long)
        mask = torch.tensor(
            self.attention_masks[idx][: self.max_seq_len], dtype=torch.float32
        )
        tim = torch.tensor(self.dwell[idx, : self.max_seq_len], dtype=torch.long)
        pos = torch.tensor(
            self.timestamps[idx, : self.max_seq_len], dtype=torch.float32
        )
        home = torch.tensor(self.home[idx], dtype=torch.float32)
        work = torch.tensor(self.work[idx], dtype=torch.float32)
        age_bin = torch.tensor(int(self.age_bin[idx]), dtype=torch.long)
        gender_id = torch.tensor(int(self.gender_id[idx]), dtype=torch.long)

        return {
            "loc": loc,
            "tim": tim,
            "pos": pos,
            "mask": mask,
            "age_bin": age_bin,
            "gender_id": gender_id,
            "home": home,
            "work": work,
        }


def collate_fn(batch: list) -> dict:
    """Pad variable-length sequences in a batch."""
    pad_values = {"loc": 0, "tim": 0, "pos": 0.0, "mask": 0.0}
    result = {}

    for key in ["loc", "tim", "pos", "mask"]:
        seqs = [b[key] for b in batch]
        padded = pad_sequence(seqs, batch_first=True, padding_value=pad_values[key])
        result[key] = padded

    for key in ["age_bin", "gender_id"]:
        result[key] = torch.stack([b[key] for b in batch])

    for key in ["home", "work"]:
        result[key] = torch.stack([b[key] for b in batch])

    return result


def build_dataloaders(
    data_root: str,
    batch_size: int = 64,
    max_seq_len: int = 64,
    num_workers: int = 4,
) -> dict:
    """Build train/val/test DataLoaders.

    Args:
        data_root: path to split_data_embee_2026/controlled/
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        split_dir = Path(data_root) / split
        if not split_dir.exists():
            continue
        ds = VolunteerTrajectoryDataset(str(split_dir), max_seq_len=max_seq_len)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
