import inspect
import json
import os
import pickle
import torch
import numpy as np

from contextlib import contextmanager

import torch.utils.data

try:
    from src.dit import DiT
except (ImportError, ModuleNotFoundError):
    DiT = None
try:
    from src import sd_unet
    from src.sd_unet import Unet
except (ImportError, ModuleNotFoundError):
    sd_unet = None
    Unet = None
from src.helpers import exists
try:
    from src.cardiff import Cardiff
except (ImportError, ModuleNotFoundError):
    Cardiff = None
from src.utils import batch_count_lengths




from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader

import pandas as pd
import numpy as np

from transformers import (
    BertTokenizerFast,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def parse_sequence(seq):
    if isinstance(seq, str):
        # Handle space-separated string sequences
        return seq.strip().split()
    elif isinstance(seq, list):
        # Handle list sequences (POI IDs are already strings)
        return [str(x) for x in seq]  # Ensure all elements are strings
    else:
        raise ValueError("unknown format")


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        sequences,
        tokenizer,
        segment_coord_map,
        max_length=128,
        attention_masks=None,
        segment_category_map=None,
        training_phase="phase2",
        ablation_mode="both",
        trajectory_length_ids=None,
        rebuild_attention_masks=True,
        force_full_attention_mask=False,
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.segment_coord_map = segment_coord_map
        self.segment_category_map = segment_category_map
        self.attention_masks = attention_masks
        self.training_phase = training_phase
        self.ablation_mode = ablation_mode
        # Store trajectory length identifiers for direct indexing
        self.trajectory_length_ids = trajectory_length_ids
        # Decide whether to regenerate attention masks from padding pattern
        self.rebuild_attention_masks = rebuild_attention_masks
        # Force attention masks to include padded positions when requested
        self.force_full_attention_mask = force_full_attention_mask

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Handle sequence length
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
        
        # Get coordinates and categories for each POI in sequence
        lat = []
        lon = []
        top_categories = []
        sub_categories = []
        
        for poi in seq:
            coords = self.segment_coord_map.get(str(poi), (0.0, 0.0))
            lat.append(coords[0])
            lon.append(coords[1])
            
            if self.segment_category_map:
                categories = self.segment_category_map.get(str(poi), (0, 0))
                top_categories.append(categories[0])
                sub_categories.append(categories[1])
            else:
                top_categories.append(0)
                sub_categories.append(0)
        
        # Pad coordinates and categories to max_length
        while len(lat) < self.max_length:
            lat.append(0.0)
            lon.append(0.0)
            top_categories.append(0)
            sub_categories.append(0)
        
        # Tokenize the sequence
        # Convert POI sequence to token IDs directly
        token_ids = []
        for poi in seq:
            token_id = self.tokenizer.convert_tokens_to_ids(str(poi))
            if token_id is None:
                token_id = self.tokenizer.unk_token_id
            token_ids.append(token_id)
        
        # Pad token IDs to max_length
        while len(token_ids) < self.max_length:
            token_ids.append(self.tokenizer.pad_token_id)
        
        # Use pre-computed attention mask if available
        if self.force_full_attention_mask:
            attention_mask = [1] * self.max_length
        elif self.rebuild_attention_masks or self.attention_masks is None:
            attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in token_ids]
        else:
            attention_mask = self.attention_masks[idx]
            # Ensure attention mask is the right length
            if len(attention_mask) > self.max_length:
                attention_mask = attention_mask[:self.max_length]
            elif len(attention_mask) < self.max_length:
                attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))
        
        # Base item with essential data
        item = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        # Add coordinates if needed for the current phase/ablation mode
        if self.training_phase == "phase2" and self.ablation_mode in ["coords_only", "both"]:
            item["lat"] = torch.tensor(lat, dtype=torch.float)
            item["lon"] = torch.tensor(lon, dtype=torch.float)
        
        # Add categories if needed for the current phase/ablation mode
        if self.training_phase == "phase2" and self.ablation_mode in ["subcat_only", "both"]:
            item["sub_categories"] = torch.tensor(sub_categories, dtype=torch.long)
        
        # Note: top_categories is removed as it's not used in phase2 training

        # Labels are the same as input_ids for language modeling
        item["labels"] = item["input_ids"].clone()

        # Include discrete length identifier when available
        if self.trajectory_length_ids is not None:
            length_id = int(self.trajectory_length_ids[idx])
            item["length_id"] = torch.tensor(length_id, dtype=torch.long)

        return item

class CombinedDataset(Dataset):
    def __init__(self, tensor_dataset, token_dataset):
        assert len(tensor_dataset) == len(token_dataset)
        self.tensor_dataset = tensor_dataset
        self.token_dataset = token_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, idx):
        attrs, dwell_times = self.tensor_dataset[idx]
        token_data = self.token_dataset[idx]
        combined_sample = {
            "attrs": attrs,
            "dwell_times": dwell_times,
            "input_ids": token_data["input_ids"],
            "labels": token_data["labels"],
            "attention_mask": token_data["attention_mask"],
        }
        
        # Include trajectory length id when provided
        if "length_id" in token_data:
            combined_sample["length_id"] = token_data["length_id"]

        # Only include coordinates and categories for Phase 2 with appropriate ablation modes
        if hasattr(self.token_dataset, 'training_phase') and self.token_dataset.training_phase == "phase2":
            if hasattr(self.token_dataset, 'ablation_mode'):
                ablation_mode = self.token_dataset.ablation_mode
                if ablation_mode in ["coords_only", "both"]:
                    if "lat" in token_data and "lon" in token_data:
                        combined_sample["lat"] = token_data["lat"]
                        combined_sample["lon"] = token_data["lon"]
                if ablation_mode in ["subcat_only", "both"]:
                    if "sub_categories" in token_data:
                        combined_sample["sub_categories"] = token_data["sub_categories"]
        
        return combined_sample


class LabeledDataset(Dataset):
    """Wrapper dataset that adds conditional/unconditional labels to samples."""
    def __init__(self, base_dataset, is_conditional):
        self.base_dataset = base_dataset
        self.is_conditional = is_conditional

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample['is_conditional'] = self.is_conditional
        return sample


def load_unified_dataset(args, testset=False, data_dir="split_data_new", training_phase="phase2", ablation_mode="both"):
    """
    Load and combine controlled and uncontrolled datasets for unified training.
    """
    from torch.utils.data import ConcatDataset

    use_length_condition = getattr(args, 'enable_length_condition', False)

    # Determine split
    split = "test" if testset else "train"

    rebuild_masks = getattr(args, 'rebuild_attention_masks', True)
    force_full_attention = getattr(args, 'force_full_attention_mask', False)
    
    # Load controlled data
    controlled_dir = f"{data_dir}/controlled/{split}"
    print(f"Loading controlled data from {controlled_dir}...")
    
    # Load uncontrolled data  
    uncontrolled_dir = f"{data_dir}/uncontrolled/{split}"
    print(f"Loading uncontrolled data from {uncontrolled_dir}...")
    
    # Check if both directories exist
    import os
    if not os.path.exists(controlled_dir):
        raise FileNotFoundError(f"Controlled data directory not found: {controlled_dir}")
    if not os.path.exists(uncontrolled_dir):
        raise FileNotFoundError(f"Uncontrolled data directory not found: {uncontrolled_dir}")
    
    # Load POI metadata from controlled data (assume same POI vocabulary)
    segment_df = pd.read_csv(f"{controlled_dir}/poi_map_feature.csv")
    lat_min, lat_max = segment_df["lat"].min(), segment_df["lat"].max()
    lon_min, lon_max = segment_df["lon"].min(), segment_df["lon"].max()
    segment_df["norm_lat"] = 2 * (segment_df["lat"] - lat_min) / (lat_max - lat_min) - 1
    segment_df["norm_lon"] = 2 * (segment_df["lon"] - lon_min) / (lon_max - lon_min) - 1
    
    # Create mappings
    segment_coord_map = {}
    segment_category_map = {}
    
    if training_phase == "phase2" and ablation_mode in ["subcat_only", "both"]:
        unique_top_categories = segment_df['top_category'].dropna().unique()
        unique_sub_categories = segment_df['sub_category'].dropna().unique()
        top_category_to_id = {cat: i for i, cat in enumerate(unique_top_categories)}
        sub_category_to_id = {cat: i for i, cat in enumerate(unique_sub_categories)}
    else:
        top_category_to_id = {}
        sub_category_to_id = {}
    
    for _, row in segment_df.iterrows():
        poi_id = row.poi_id
        segment_coord_map[poi_id] = (row.norm_lat, row.norm_lon)
        top_cat_id = top_category_to_id.get(row.top_category, 0) if pd.notna(row.top_category) else 0
        sub_cat_id = sub_category_to_id.get(row.sub_category, 0) if pd.notna(row.sub_category) else 0
        segment_category_map[poi_id] = (top_cat_id, sub_cat_id)
    
    # Load controlled dataset components
    print("Loading controlled dataset components...")
    controlled_trajs = pd.read_pickle(f"{controlled_dir}/final_segments_all_train_data.pkl")
    # Prefer extended attributes with demographics when available (e.g., [work, work, home, home, age, gender])
    controlled_attrs_with_demo_path = f"{controlled_dir}/all_attr_results_with_demo.npy"
    controlled_attrs_path = f"{controlled_dir}/all_attr_results.npy"
    if os.path.exists(controlled_attrs_with_demo_path):
        print(f"Loading extended attributes with demo from {controlled_attrs_with_demo_path}...")
        controlled_attrs_np = np.load(controlled_attrs_with_demo_path, allow_pickle=True)
    else:
        controlled_attrs_np = np.load(controlled_attrs_path, allow_pickle=True)
    controlled_attrs = torch.from_numpy(controlled_attrs_np).float()
    controlled_dwell = np.load(f"{controlled_dir}/all_dwell.npy", allow_pickle=True)
    controlled_dwell = torch.from_numpy(controlled_dwell).float()
    
    controlled_length_ids = None
    controlled_length_path = f'{controlled_dir}/trajectory_length_ids.npy'
    if use_length_condition and os.path.exists(controlled_length_path):
        try:
            controlled_length_ids = np.load(controlled_length_path, allow_pickle=True).astype(np.int64)
            print(f"Loaded controlled trajectory length ids: {controlled_length_ids.shape}")
        except Exception as exc:
            print(f"Warning: failed to load controlled trajectory length ids ({exc}); recomputing later.")
            controlled_length_ids = None
    elif use_length_condition:
        print("No controlled trajectory length ids found; will compute on the fly.")
    
    # Load uncontrolled dataset components
    print("Loading uncontrolled dataset components...")
    uncontrolled_trajs = pd.read_pickle(f"{uncontrolled_dir}/final_segments_all_train_data.pkl")
    uncontrolled_dwell = np.load(f"{uncontrolled_dir}/all_dwell.npy", allow_pickle=True)  
    uncontrolled_dwell = torch.from_numpy(uncontrolled_dwell).float()
    # Create dummy attributes for uncontrolled data
    uncontrolled_attrs = torch.zeros(len(uncontrolled_trajs), 4)  # 4 dummy attributes
    
    uncontrolled_length_ids = None
    uncontrolled_length_path = f'{uncontrolled_dir}/trajectory_length_ids.npy'
    if use_length_condition and os.path.exists(uncontrolled_length_path):
        try:
            uncontrolled_length_ids = np.load(uncontrolled_length_path, allow_pickle=True).astype(np.int64)
            print(f"Loaded uncontrolled trajectory length ids: {uncontrolled_length_ids.shape}")
        except Exception as exc:
            print(f"Warning: failed to load uncontrolled trajectory length ids ({exc}); recomputing later.")
            uncontrolled_length_ids = None
    elif use_length_condition:
        print("No uncontrolled trajectory length ids found; will compute on the fly.")
    
    # Create tokenizer (from controlled data)
    print("Loading tokenizer vocabulary for unified dataset...")
    # Try to load from pickle file first (old format)
    tokenizer_vocab_path = f'{controlled_dir}/tokenizer_vocab.pkl'
    if os.path.exists(tokenizer_vocab_path):
        with open(tokenizer_vocab_path, 'rb') as f:
            tokenizer_vocab = pickle.load(f)
        print(f"Loaded tokenizer vocabulary from pickle: {len(tokenizer_vocab)} tokens")
    else:
        # Try to load from vocab.txt (new format)
        vocab_txt_path = f'{controlled_dir}/tokenizer/vocab.txt'
        if os.path.exists(vocab_txt_path):
            tokenizer_vocab = []
            with open(vocab_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    token = line.strip()
                    if token:  # Skip empty lines
                        tokenizer_vocab.append(token)
            print(f"Loaded tokenizer vocabulary from vocab.txt: {len(tokenizer_vocab)} tokens")
        else:
            raise FileNotFoundError(f"Neither tokenizer_vocab.pkl nor tokenizer/vocab.txt found in {controlled_dir}")
    
    from transformers import BertTokenizerFast
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for token in tokenizer_vocab:
            f.write(f"{token}\n")
        temp_vocab_file = f.name
    
    try:
        tokenizer = BertTokenizerFast(vocab_file=temp_vocab_file, do_lower_case=False)
        tokenizer.add_special_tokens({
            "bos_token": "[CLS]", "eos_token": "[SEP]", "pad_token": "[PAD]",
            "mask_token": "[MASK]", "unk_token": "[UNK]"
        })
    finally:
        os.unlink(temp_vocab_file)
    
    # Process controlled sequences
    controlled_sequences = []
    controlled_attention_masks = []
    for idx, row in controlled_trajs.iterrows():
        sequence = row['unique_id_seq']
        attention_mask = row['attention_mask']
        if isinstance(sequence, str):
            sequence = parse_sequence(sequence)
        controlled_sequences.append(sequence)
        controlled_attention_masks.append(attention_mask)
    
    # Process uncontrolled sequences  
    uncontrolled_sequences = []
    uncontrolled_attention_masks = []
    for idx, row in uncontrolled_trajs.iterrows():
        sequence = row['unique_id_seq']
        attention_mask = row['attention_mask']
        if isinstance(sequence, str):
            sequence = parse_sequence(sequence)
        uncontrolled_sequences.append(sequence)
        uncontrolled_attention_masks.append(attention_mask)
    
    # Create datasets
    # Respect an explicit requested sequence length (e.g., from DiT.image_size in train_dit_only.py).
    requested_len = getattr(args, "sequence_length", None)
    if requested_len is not None:
        max_seq_length = int(requested_len)
    else:
        max_seq_length = max(
            len(controlled_sequences[0]) if controlled_sequences else 0,
            len(uncontrolled_sequences[0]) if uncontrolled_sequences else 0,
            512  # fallback
        )

    if use_length_condition:
        current_vocab = int(getattr(args, 'length_vocab_size', 0))
        args.length_vocab_size = max(current_vocab, max_seq_length + 1)
        if controlled_length_ids is None or len(controlled_length_ids) != len(controlled_sequences):
            print("Computing controlled trajectory length ids...")
            controlled_length_ids = np.asarray(
                batch_count_lengths(controlled_sequences, max_length=max_seq_length),
                dtype=np.int64,
            )
        if uncontrolled_length_ids is None or len(uncontrolled_length_ids) != len(uncontrolled_sequences):
            print("Computing uncontrolled trajectory length ids...")
            uncontrolled_length_ids = np.asarray(
                batch_count_lengths(uncontrolled_sequences, max_length=max_seq_length),
                dtype=np.int64,
            )
    
    # Controlled dataset with labels
    controlled_token_dataset = TrajectoryDataset(
        controlled_sequences,
        tokenizer,
        segment_coord_map,
        max_seq_length,
        controlled_attention_masks,
        segment_category_map,
        training_phase,
        ablation_mode,
        trajectory_length_ids=controlled_length_ids if use_length_condition else None,
        rebuild_attention_masks=rebuild_masks,
        force_full_attention_mask=force_full_attention,
    )
    controlled_tensor_dataset = TensorDataset(controlled_attrs, controlled_dwell)
    controlled_combined = CombinedDataset(controlled_tensor_dataset, controlled_token_dataset)
    
    # Add conditional labels to controlled data  
    controlled_labeled = LabeledDataset(controlled_combined, is_conditional=True)
    
    # Uncontrolled dataset with labels
    uncontrolled_token_dataset = TrajectoryDataset(
        uncontrolled_sequences,
        tokenizer,
        segment_coord_map,
        max_seq_length,
        uncontrolled_attention_masks,
        segment_category_map,
        training_phase,
        ablation_mode,
        trajectory_length_ids=uncontrolled_length_ids if use_length_condition else None,
        rebuild_attention_masks=rebuild_masks,
        force_full_attention_mask=force_full_attention,
    )
    uncontrolled_tensor_dataset = TensorDataset(uncontrolled_attrs, uncontrolled_dwell)
    uncontrolled_combined = CombinedDataset(uncontrolled_tensor_dataset, uncontrolled_token_dataset)
    
    # Add conditional labels to uncontrolled data
    uncontrolled_labeled = LabeledDataset(uncontrolled_combined, is_conditional=False)
    
    # Combine datasets
    unified_dataset = ConcatDataset([controlled_labeled, uncontrolled_labeled])
    
    
    print(f"Unified dataset created:")
    print(f"  Controlled samples: {len(controlled_labeled)}")
    print(f"  Uncontrolled samples: {len(uncontrolled_labeled)}")
    print(f"  Total samples: {len(unified_dataset)}")
    
    # Create custom collate function for unified data
    def unified_collate_fn(batch):
        # Separate batch components
        attrs_list = []
        dwell_list = []
        token_features = []
        is_conditional_list = []
        
        for sample in batch:
            attrs_list.append(sample['attrs'])
            dwell_list.append(sample['dwell_times'])
            is_conditional_list.append(sample['is_conditional'])
            
            # Extract token features and ensure they're tensors
            token_feat = {}
            token_keys = {"input_ids", "labels", "attention_mask", "lat", "lon", "sub_categories", "length_id"}
            for k in token_keys:
                if k in sample:
                    # Ensure it's a tensor, not numpy array
                    if isinstance(sample[k], np.ndarray):
                        token_feat[k] = torch.from_numpy(sample[k])
                    elif isinstance(sample[k], list):
                        # Convert list to tensor directly
                        token_feat[k] = torch.tensor(sample[k])
                    else:
                        token_feat[k] = sample[k]
            token_features.append(token_feat)
        
        # Pre-stack tensors to avoid slow conversion
        attrs_tensor = torch.stack(attrs_list)
        dwell_tensor = torch.stack(dwell_list)
        is_conditional_tensor = torch.tensor(is_conditional_list)
        
        # Ensure all token features are properly formatted for the collator
        # The collator expects lists of tensors, not numpy arrays
        processed_token_features = []
        for token_feat in token_features:
            processed_feat = {}
            for k, v in token_feat.items():
                if isinstance(v, torch.Tensor):
                    processed_feat[k] = v
                elif isinstance(v, (list, np.ndarray)):
                    processed_feat[k] = torch.tensor(v)
                else:
                    processed_feat[k] = v
            processed_token_features.append(processed_feat)
        
        # Create a custom collator that handles our specific data format
        # This avoids the slow numpy array conversion in DataCollatorForSeq2Seq
        batch_size = len(processed_token_features)
        
        # Manually collate the token features to avoid the slow conversion
        collated_tokens = {}
        
        # Get all keys from the first sample
        if processed_token_features:
            keys = processed_token_features[0].keys()
            
            for key in keys:
                if key in ["input_ids", "labels", "attention_mask"]:
                    # These are the main fields that DataCollatorForSeq2Seq handles
                    # Stack them manually to avoid the slow conversion
                    tensors = [sample[key] for sample in processed_token_features]
                    # Pad to the maximum length in this batch
                    max_len = max(tensor.size(0) for tensor in tensors)
                    padded_tensors = []
                    
                    for tensor in tensors:
                        if tensor.size(0) < max_len:
                            # Pad with appropriate token
                            if key == "labels":
                                pad_token = -100  # Ignore index for labels
                            else:
                                pad_token = tokenizer.pad_token_id
                            padding = torch.full((max_len - tensor.size(0),), pad_token, dtype=tensor.dtype)
                            padded_tensor = torch.cat([tensor, padding])
                        else:
                            padded_tensor = tensor
                        padded_tensors.append(padded_tensor)
                    
                    collated_tokens[key] = torch.stack(padded_tensors)
                else:
                    # For other fields (lat, lon, sub_categories), just stack them
                    tensors = [sample[key] for sample in processed_token_features]
                    collated_tokens[key] = torch.stack(tensors)
        
        # Add other features
        result = {
            'attrs': attrs_tensor,
            'dwell_times': dwell_tensor,
            'is_conditional': is_conditional_tensor,
        }
        result.update(collated_tokens)
        return result
    
    # Create dataloaders
    if testset:
        dataloader = torch.utils.data.DataLoader(
            unified_dataset, batch_size=args.BATCH_SIZE, shuffle=False,
            num_workers=args.NUM_WORKERS, collate_fn=unified_collate_fn
        )
        return dataloader
    else:
        # For training, also load validation data
        train_dataloader = torch.utils.data.DataLoader(
            unified_dataset, batch_size=args.BATCH_SIZE, shuffle=True,
            num_workers=args.NUM_WORKERS, collate_fn=unified_collate_fn
        )
        
        # Load unified validation data
        print("Loading unified validation data...")
        val_split = "val"  # Use val split for validation
        
        # Load controlled validation data
        controlled_val_dir = f"{data_dir}/controlled/{val_split}"
        uncontrolled_val_dir = f"{data_dir}/uncontrolled/{val_split}"
        
        if os.path.exists(controlled_val_dir) and os.path.exists(uncontrolled_val_dir):
            # Load validation data from val directories
            print(f"Loading validation data from {controlled_val_dir} and {uncontrolled_val_dir}...")
            
            # Load controlled validation data
            controlled_val_trajs = pd.read_pickle(f"{controlled_val_dir}/final_segments_all_train_data.pkl")
            controlled_val_attrs_with_demo_path = f"{controlled_val_dir}/all_attr_results_with_demo.npy"
            controlled_val_attrs_path = f"{controlled_val_dir}/all_attr_results.npy"
            if os.path.exists(controlled_val_attrs_with_demo_path):
                print(f"Loading extended validation attributes with demo from {controlled_val_attrs_with_demo_path}...")
                controlled_val_attrs_np = np.load(controlled_val_attrs_with_demo_path, allow_pickle=True)
            else:
                controlled_val_attrs_np = np.load(controlled_val_attrs_path, allow_pickle=True)
            controlled_val_attrs = torch.from_numpy(controlled_val_attrs_np).float()
            controlled_val_dwell = np.load(f"{controlled_val_dir}/all_dwell.npy", allow_pickle=True)
            controlled_val_dwell = torch.from_numpy(controlled_val_dwell).float()
            
            # Load controlled validation origin/destination features
            controlled_val_length_ids = None
            controlled_val_length_path = f'{controlled_val_dir}/trajectory_length_ids.npy'
            if use_length_condition and os.path.exists(controlled_val_length_path):
                try:
                    controlled_val_length_ids = np.load(controlled_val_length_path, allow_pickle=True).astype(np.int64)
                    print(f"Loaded controlled validation trajectory length ids: {controlled_val_length_ids.shape}")
                except Exception as exc:
                    print(f"Warning: failed to load controlled validation trajectory length ids ({exc}); recomputing later.")
                    controlled_val_length_ids = None
            elif use_length_condition:
                print("No controlled validation trajectory length ids found; will compute on the fly.")
            
            # Load uncontrolled validation data
            uncontrolled_val_trajs = pd.read_pickle(f"{uncontrolled_val_dir}/final_segments_all_train_data.pkl")
            uncontrolled_val_dwell = np.load(f"{uncontrolled_val_dir}/all_dwell.npy", allow_pickle=True)  
            uncontrolled_val_dwell = torch.from_numpy(uncontrolled_val_dwell).float()
            
            uncontrolled_val_attrs = torch.zeros(len(uncontrolled_val_trajs), 4)  # 4 dummy attributes
            
            # Load uncontrolled validation origin/destination features
            uncontrolled_val_length_ids = None
            uncontrolled_val_length_path = f'{uncontrolled_val_dir}/trajectory_length_ids.npy'
            if use_length_condition and os.path.exists(uncontrolled_val_length_path):
                try:
                    uncontrolled_val_length_ids = np.load(uncontrolled_val_length_path, allow_pickle=True).astype(np.int64)
                    print(f"Loaded uncontrolled validation trajectory length ids: {uncontrolled_val_length_ids.shape}")
                except Exception as exc:
                    print(f"Warning: failed to load uncontrolled validation trajectory length ids ({exc}); recomputing later.")
                    uncontrolled_val_length_ids = None
            elif use_length_condition:
                print("No uncontrolled validation trajectory length ids found; will compute on the fly.")
            
            # Process controlled validation sequences
            controlled_val_sequences = []
            controlled_val_attention_masks = []
            for idx, row in controlled_val_trajs.iterrows():
                sequence = row['unique_id_seq']
                attention_mask = row['attention_mask']
                if isinstance(sequence, str):
                    sequence = parse_sequence(sequence)
                controlled_val_sequences.append(sequence)
                controlled_val_attention_masks.append(attention_mask)
            
            # Process uncontrolled validation sequences  
            uncontrolled_val_sequences = []
            uncontrolled_val_attention_masks = []
            for idx, row in uncontrolled_val_trajs.iterrows():
                sequence = row['unique_id_seq']
                attention_mask = row['attention_mask']
                if isinstance(sequence, str):
                    sequence = parse_sequence(sequence)
                uncontrolled_val_sequences.append(sequence)
                uncontrolled_val_attention_masks.append(attention_mask)
            
            # Create validation datasets
            # IMPORTANT: keep validation sequence length consistent with training/DiT.image_size.
            requested_len = getattr(args, "sequence_length", None)
            if requested_len is not None:
                max_val_seq_length = int(requested_len)
            else:
                max_val_seq_length = max(
                    len(controlled_val_sequences[0]) if controlled_val_sequences else 0,
                    len(uncontrolled_val_sequences[0]) if uncontrolled_val_sequences else 0,
                    512  # fallback
                )

            if use_length_condition:
                current_vocab = int(getattr(args, 'length_vocab_size', 0))
                args.length_vocab_size = max(current_vocab, max_val_seq_length + 1)
                if controlled_val_length_ids is None or len(controlled_val_length_ids) != len(controlled_val_sequences):
                    print("Computing controlled validation trajectory length ids...")
                    controlled_val_length_ids = np.asarray(
                        batch_count_lengths(controlled_val_sequences, max_length=max_val_seq_length),
                        dtype=np.int64,
                    )
                if uncontrolled_val_length_ids is None or len(uncontrolled_val_length_ids) != len(uncontrolled_val_sequences):
                    print("Computing uncontrolled validation trajectory length ids...")
                    uncontrolled_val_length_ids = np.asarray(
                        batch_count_lengths(uncontrolled_val_sequences, max_length=max_val_seq_length),
                        dtype=np.int64,
                    )
            
            # Controlled validation dataset
            controlled_val_token_dataset = TrajectoryDataset(
                controlled_val_sequences,
                tokenizer,
                segment_coord_map,
                max_val_seq_length,
                controlled_val_attention_masks,
                segment_category_map,
                training_phase,
                ablation_mode,
                trajectory_length_ids=controlled_val_length_ids if use_length_condition else None,
                rebuild_attention_masks=rebuild_masks,
                force_full_attention_mask=force_full_attention,
            )
            controlled_val_tensor_dataset = TensorDataset(controlled_val_attrs, controlled_val_dwell)
            controlled_val_combined = CombinedDataset(controlled_val_tensor_dataset, controlled_val_token_dataset)
            controlled_val_labeled = LabeledDataset(controlled_val_combined, is_conditional=True)
            
            # Uncontrolled validation dataset
            uncontrolled_val_token_dataset = TrajectoryDataset(
                uncontrolled_val_sequences,
                tokenizer,
                segment_coord_map,
                max_val_seq_length,
                uncontrolled_val_attention_masks,
                segment_category_map,
                training_phase,
                ablation_mode,
                trajectory_length_ids=uncontrolled_val_length_ids if use_length_condition else None,
                rebuild_attention_masks=rebuild_masks,
                force_full_attention_mask=force_full_attention,
            )
            uncontrolled_val_tensor_dataset = TensorDataset(uncontrolled_val_attrs, uncontrolled_val_dwell)
            uncontrolled_val_combined = CombinedDataset(uncontrolled_val_tensor_dataset, uncontrolled_val_token_dataset)
            uncontrolled_val_labeled = LabeledDataset(uncontrolled_val_combined, is_conditional=False)
            
            # Combine validation datasets
            val_unified_dataset = ConcatDataset([controlled_val_labeled, uncontrolled_val_labeled])
            
            print(f"Validation dataset created:")
            print(f"  Controlled validation samples: {len(controlled_val_labeled)}")
            print(f"  Uncontrolled validation samples: {len(uncontrolled_val_labeled)}")
            print(f"  Total validation samples: {len(val_unified_dataset)}")
            
            val_dataloader = torch.utils.data.DataLoader(
                val_unified_dataset, batch_size=args.BATCH_SIZE, shuffle=getattr(args, 'shuffle_val', False),
                num_workers=args.NUM_WORKERS, collate_fn=unified_collate_fn
            )
        else:
            print("Warning: Validation data not found, using training data for validation")
            val_dataloader = torch.utils.data.DataLoader(
                unified_dataset, batch_size=args.BATCH_SIZE, shuffle=getattr(args, 'shuffle_val', False),
                num_workers=args.NUM_WORKERS, collate_fn=unified_collate_fn
            )
        
        adjacency_matrix = None
        return train_dataloader, val_dataloader, adjacency_matrix, tokenizer_vocab


def trajectory_dataset(args, testset=False, data_dir="split_data_new", data_type="controlled"):
    """
    Load dataset using pre-split train/val/test directories.
    Supports unified training by combining controlled and uncontrolled data.
    """
    training_phase = getattr(args, 'training_phase', 'phase2')  # Default to phase2 for backward compatibility
    ablation_mode = getattr(args, 'ablation_mode', 'both')      # Default to both features
    rebuild_masks = getattr(args, 'rebuild_attention_masks', True)
    force_full_attention = getattr(args, 'force_full_attention_mask', False)
    # Whether to append a discrete trajectory length id to attrs
    use_length_condition = getattr(args, 'enable_length_condition', False)
    
    print(f"Loading dataset using pre-split directories for {training_phase}...")
    
    if data_type == "unified":
        # Unified training: combine controlled and uncontrolled data
        print("Loading unified dataset (controlled + uncontrolled)...")
        return load_unified_dataset(args, testset, data_dir, training_phase, ablation_mode)

    def _resolve_split_dir(root: str, requested_type: str, split: str):
        """
        Resolve the on-disk split directory for a requested data_type.

        Historical note: some experiments want "unconditional" training (no attrs)
        while still consuming the same tokenized POI sequences stored under
        `controlled/{split}`. In that case users set `data_type=uncontrolled`
        but the `uncontrolled/{split}` directory may not exist. We fall back to
        `controlled/{split}` for file reads while still treating attrs as absent.
        """
        preferred = os.path.join(root, requested_type, split)
        if os.path.exists(preferred):
            return preferred, False
        if requested_type == "uncontrolled":
            fallback = os.path.join(root, "controlled", split)
            if os.path.exists(fallback):
                print(
                    f"Warning: requested data_type='uncontrolled' but '{preferred}' does not exist; "
                    f"falling back to '{fallback}' for sequences (attrs will be ignored)."
                )
                return fallback, True
        return preferred, False

    # Single data type loading (original logic)
    # Determine data directory based on testset flag
    requested_data_type = data_type
    split_name = "test" if testset else "train"
    current_data_dir, _used_fallback = _resolve_split_dir(data_dir, requested_data_type, split_name)
    if testset:
        print(f"Loading test data from {current_data_dir}...")
    else:
        print(f"Loading training data from {current_data_dir}...")
    
    # Load POI metadata for coordinate mapping and categories (needed for both phases)
    segment_df = pd.read_csv(f"{current_data_dir}/poi_map_feature.csv")
    lat_min, lat_max = segment_df["lat"].min(), segment_df["lat"].max()
    lon_min, lon_max = segment_df["lon"].min(), segment_df["lon"].max()
    segment_df["norm_lat"] = 2 * (segment_df["lat"] - lat_min) / (lat_max - lat_min) - 1
    segment_df["norm_lon"] = 2 * (segment_df["lon"] - lon_min) / (lon_max - lon_min) - 1
    
    # Only create category mappings if needed for phase2
    if training_phase == "phase2" and ablation_mode in ["subcat_only", "both"]:
        unique_top_categories = segment_df['top_category'].dropna().unique()
        unique_sub_categories = segment_df['sub_category'].dropna().unique()
        
        top_category_to_id = {cat: i for i, cat in enumerate(unique_top_categories)}
        sub_category_to_id = {cat: i for i, cat in enumerate(unique_sub_categories)}
        
        print(f"Found {len(unique_top_categories)} top categories, {len(unique_sub_categories)} sub categories")
    else:
        print("Skipping category mappings (not needed for current phase/ablation mode)")
        unique_top_categories = []
        unique_sub_categories = []
        top_category_to_id = {}
        sub_category_to_id = {}
    
    # Create comprehensive mapping including coordinates and categories
    segment_coord_map = {}
    segment_category_map = {}
    
    for _, row in segment_df.iterrows():
        poi_id = row.poi_id
        segment_coord_map[poi_id] = (row.norm_lat, row.norm_lon)
        
        # Handle missing categories
        top_cat_id = top_category_to_id.get(row.top_category, 0) if pd.notna(row.top_category) else 0
        sub_cat_id = sub_category_to_id.get(row.sub_category, 0) if pd.notna(row.sub_category) else 0
        
        segment_category_map[poi_id] = (top_cat_id, sub_cat_id)
    
    # Load pre-processed data from split directory
    print("Loading POI sequences...")
    trajs_df = pd.read_pickle(f"{current_data_dir}/final_segments_all_train_data.pkl")
    
    # Load attributes (only available for controlled data)
    attrs = None
    if requested_data_type == "uncontrolled":
        print("Data type is uncontrolled: ignoring attribute files (unconditional training)")
    else:
        attrs_with_demo_path = f'{current_data_dir}/all_attr_results_with_demo.npy'
        attrs_path = f'{current_data_dir}/all_attr_results.npy'
        if os.path.exists(attrs_with_demo_path):
            print(f"Loading extended attributes with demo from {attrs_with_demo_path}...")
            attrs_np = np.load(attrs_with_demo_path, allow_pickle=True)
            attrs = torch.from_numpy(attrs_np).float()
            print(f"Loaded extended attributes: {attrs.shape}")
        elif os.path.exists(attrs_path):
            print("Loading attributes...")
            attrs_np = np.load(attrs_path, allow_pickle=True)
            attrs = torch.from_numpy(attrs_np).float()
            print(f"Loaded attributes: {attrs.shape}")
        else:
            print("No attributes found - using unconditional training")
            # Create dummy attributes for consistent batch processing
            # We'll handle this in the training loop by setting attr_embeds=None
    
    print("Loading timestamps...")
    timestamps = np.load(f'{current_data_dir}/all_timestamp.npy', allow_pickle=True)
    
    print("Loading dwell times...")
    dwell_times = np.load(f'{current_data_dir}/all_dwell.npy', allow_pickle=True)
    dwell_times = torch.from_numpy(dwell_times).float()
    
    
    print("Loading tokenizer vocabulary...")
    # Try to load from pickle file first (old format)
    tokenizer_vocab_path = f'{current_data_dir}/tokenizer_vocab.pkl'
    if os.path.exists(tokenizer_vocab_path):
        with open(tokenizer_vocab_path, 'rb') as f:
            tokenizer_vocab = pickle.load(f)
        print(f"Loaded tokenizer vocabulary from pickle: {len(tokenizer_vocab)} tokens")
    else:
        # Try to load from vocab.txt (new format)
        vocab_txt_path = f'{current_data_dir}/tokenizer/vocab.txt'
        if os.path.exists(vocab_txt_path):
            tokenizer_vocab = []
            with open(vocab_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    token = line.strip()
                    if token:  # Skip empty lines
                        tokenizer_vocab.append(token)
            print(f"Loaded tokenizer vocabulary from vocab.txt: {len(tokenizer_vocab)} tokens")
        else:
            raise FileNotFoundError(f"Neither tokenizer_vocab.pkl nor tokenizer/vocab.txt found in {current_data_dir}")
    
    # Create tokenizer from vocabulary
    from transformers import BertTokenizerFast
    
    # Create a temporary vocab file for the tokenizer
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for token in tokenizer_vocab:
            f.write(f"{token}\n")
        temp_vocab_file = f.name
    
    try:
        tokenizer = BertTokenizerFast(vocab_file=temp_vocab_file, do_lower_case=False)
        tokenizer.add_special_tokens({
            "bos_token": "[CLS]",
            "eos_token": "[SEP]", 
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "unk_token": "[UNK]"
        })
    finally:
        os.unlink(temp_vocab_file)  # Clean up temp file
    
    print(f"✓ Created tokenizer with vocab size: {len(tokenizer_vocab)}")
    
    # Extract sequences and attention masks from DataFrame
    all_sequences = []
    attention_masks = []
    
    for idx, row in trajs_df.iterrows():
        sequence = row['unique_id_seq']
        attention_mask = row['attention_mask']
        
        # Convert sequence to parsed format if needed
        if isinstance(sequence, str):
            sequence = parse_sequence(sequence)
        
        all_sequences.append(sequence)
        attention_masks.append(attention_mask)
    
    print(f"Loaded {len(all_sequences)} sequences")
    if attrs is not None:
        print(f"Attributes shape: {attrs.shape}")
    else:
        print("Attributes: None (unconditional training)")
    print(f"Dwell times shape: {dwell_times.shape}")
    
    # Create dataset
    # NOTE: historically this code forced max_seq_length >= 512, because the original
    # pipeline used 512-length token sequences. For phase1 diffusion pretraining, we
    # often train BART with shorter max_length (e.g., 64). In that case DiT.image_size
    # must match the token sequence length, otherwise DiT positional embeddings will
    # have a different T than the BART encoder latents (runtime error).
    requested_len = getattr(args, "sequence_length", None)
    if requested_len is not None:
        max_seq_length = int(requested_len)
    else:
        max_seq_length = max(len(all_sequences[0]) if all_sequences else 0, 512)

    length_ids = None
    length_path = f'{current_data_dir}/trajectory_length_ids.npy'
    if use_length_condition and os.path.exists(length_path):
        try:
            length_ids = np.load(length_path, allow_pickle=True).astype(np.int64)
            print(f"Loaded trajectory length ids: {length_ids.shape}")
        except Exception as exc:
            print(f"Warning: failed to load trajectory length ids ({exc}); recomputing.")
            length_ids = None
    elif use_length_condition:
        print("No trajectory length ids found; will compute on the fly.")

    if use_length_condition:
        current_vocab = int(getattr(args, 'length_vocab_size', 0))
        args.length_vocab_size = max(current_vocab, max_seq_length + 1)

    if use_length_condition and (length_ids is None or len(length_ids) != len(all_sequences)):
        print("Computing trajectory length ids for current split...")
        length_ids = np.asarray(
            batch_count_lengths(all_sequences, max_length=max_seq_length),
            dtype=np.int64,
        )

    token_dataset = TrajectoryDataset(
        all_sequences, 
        tokenizer,
        segment_coord_map=segment_coord_map,
        segment_category_map=segment_category_map,
        max_length=max_seq_length,
        attention_masks=attention_masks,  # Pass pre-computed attention masks
        training_phase=training_phase,
        ablation_mode=ablation_mode,
        trajectory_length_ids=length_ids if use_length_condition else None,
        rebuild_attention_masks=rebuild_masks,
        force_full_attention_mask=force_full_attention,
    )
    
    # Create tensor dataset for attributes and other features
    if attrs is not None:
        tensor_dataset = TensorDataset(attrs, dwell_times)
    else:
        # Create dummy attributes for consistent batch processing
        dummy_attrs = torch.zeros(len(dwell_times), 4)  # 4 attributes: work_lat, work_lon, home_lat, home_lon
        tensor_dataset = TensorDataset(dummy_attrs, dwell_times)
    
    # Combine datasets
    dataset = CombinedDataset(tensor_dataset, token_dataset)
    
    def custom_collate(batch):
        """Custom collate function that handles both tensor and token data"""
        token_keys = {"input_ids", "labels", "attention_mask", "lat", "lon", "top_categories", "sub_categories", "length_id"}
        token_features = []
        other_features = {}

        for sample in batch:
            token_feat = {}
            for k in token_keys:
                if k in sample:
                    # Convert to tensor immediately to avoid numpy array lists
                    if isinstance(sample[k], np.ndarray):
                        token_feat[k] = torch.from_numpy(sample[k])
                    else:
                        token_feat[k] = sample[k]
            token_features.append(token_feat)
            
            for k, v in sample.items():
                if k not in token_keys:
                    if k not in other_features:
                        other_features[k] = []
                    other_features[k].append(v)

        for k in other_features:
            other_features[k] = torch.stack(other_features[k], dim=0)

        # Manual collation to avoid slow numpy array conversion
        collated_tokens = {}
        if token_features:
            keys = token_features[0].keys()
            
            for key in keys:
                if key in ["input_ids", "labels", "attention_mask"]:
                    # Stack and pad these fields manually
                    tensors = [sample[key] for sample in token_features]
                    max_len = max(tensor.size(0) for tensor in tensors)
                    padded_tensors = []
                    
                    for tensor in tensors:
                        if tensor.size(0) < max_len:
                            # Pad with appropriate token
                            if key == "labels":
                                pad_token = -100  # Ignore index for labels
                            else:
                                pad_token = tokenizer.pad_token_id
                            padding = torch.full((max_len - tensor.size(0),), pad_token, dtype=tensor.dtype)
                            padded_tensor = torch.cat([tensor, padding])
                        else:
                            padded_tensor = tensor
                        padded_tensors.append(padded_tensor)
                    
                    collated_tokens[key] = torch.stack(padded_tensors)
                else:
                    # For other fields, just stack them
                    tensors = [sample[key] for sample in token_features]
                    collated_tokens[key] = torch.stack(tensors)
        
        other_features.update(collated_tokens)
        return other_features
    
    # Create dataloaders for pre-split data
    if testset:
        # Load test data
        test_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.BATCH_SIZE,
            shuffle=False,  # No shuffling for test
            num_workers=args.NUM_WORKERS,
            collate_fn=custom_collate
        )
        return test_dataloader
    else:
        # Load training data
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.BATCH_SIZE,
            shuffle=True,
            num_workers=args.NUM_WORKERS,
            collate_fn=custom_collate
        )
        
        # Load validation data from separate directory
        print("Loading validation data...")
        val_data_dir, _val_used_fallback = _resolve_split_dir(data_dir, requested_data_type, "val")

        # Fallback if validation split is missing: use training dataset for validation
        if (not os.path.exists(val_data_dir)) or (not os.path.exists(f"{val_data_dir}/final_segments_all_train_data.pkl")):
            print("Warning: Validation data not found, using training data for validation")
            valid_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.BATCH_SIZE,
                shuffle=False,  # No shuffling for validation
                num_workers=args.NUM_WORKERS,
                collate_fn=custom_collate
            )
            return train_dataloader, valid_dataloader, None, tokenizer_vocab
        
        # Load validation data
        val_trajs_df = pd.read_pickle(f"{val_data_dir}/final_segments_all_train_data.pkl")
        
        # Load validation attributes (only available for controlled data)
        val_attrs = None
        if requested_data_type == "uncontrolled":
            # Create dummy attributes for consistent batch processing
            val_attrs = torch.zeros(len(val_trajs_df), 4)  # 4 attributes
        else:
            val_attrs_with_demo_path = f"{val_data_dir}/all_attr_results_with_demo.npy"
            val_attrs_path = f"{val_data_dir}/all_attr_results.npy"
            if os.path.exists(val_attrs_with_demo_path):
                val_attrs_np = np.load(val_attrs_with_demo_path, allow_pickle=True)
                val_attrs = torch.from_numpy(val_attrs_np).float()
            elif os.path.exists(val_attrs_path):
                val_attrs_np = np.load(val_attrs_path, allow_pickle=True)
                val_attrs = torch.from_numpy(val_attrs_np).float()
            else:
                # Create dummy attributes for consistent batch processing
                val_attrs = torch.zeros(len(val_trajs_df), 4)  # 4 attributes
        
        val_timestamps = np.load(f'{val_data_dir}/all_timestamp.npy', allow_pickle=True)
        val_dwell_times = np.load(f'{val_data_dir}/all_dwell.npy', allow_pickle=True)
        val_dwell_times = torch.from_numpy(val_dwell_times).float()
        
        # Create validation dataset
        val_all_sequences = []
        val_attention_masks = []
        
        for idx, row in val_trajs_df.iterrows():
            sequence = row['unique_id_seq']
            attention_mask = row['attention_mask']
            
            if isinstance(sequence, str):
                sequence = parse_sequence(sequence)
            
            val_all_sequences.append(sequence)
            val_attention_masks.append(attention_mask)
        
        # Keep validation sequence length consistent with training/DiT.image_size when provided.
        requested_len = getattr(args, "sequence_length", None)
        if requested_len is not None:
            val_max_seq_length = int(requested_len)
        else:
            val_max_seq_length = max(len(val_all_sequences[0]) if val_all_sequences else 0, 512)

        val_length_ids = None
        val_length_path = f'{val_data_dir}/trajectory_length_ids.npy'
        if use_length_condition and os.path.exists(val_length_path):
            try:
                val_length_ids = np.load(val_length_path, allow_pickle=True).astype(np.int64)
                print(f"Loaded validation trajectory length ids: {val_length_ids.shape}")
            except Exception as exc:
                print(f"Warning: failed to load validation trajectory length ids ({exc}); recomputing.")
                val_length_ids = None
        elif use_length_condition:
            print("No validation trajectory length ids found; will compute on the fly.")

        if use_length_condition:
            current_vocab = int(getattr(args, 'length_vocab_size', 0))
            args.length_vocab_size = max(current_vocab, val_max_seq_length + 1)

        if use_length_condition and (val_length_ids is None or len(val_length_ids) != len(val_all_sequences)):
            print("Computing validation trajectory length ids...")
            val_length_ids = np.asarray(
                batch_count_lengths(val_all_sequences, max_length=val_max_seq_length),
                dtype=np.int64,
            )
        
        val_token_dataset = TrajectoryDataset(
            val_all_sequences, 
            tokenizer,
            segment_coord_map=segment_coord_map,
            segment_category_map=segment_category_map,
            max_length=val_max_seq_length,
            attention_masks=val_attention_masks,
            training_phase=training_phase,
            ablation_mode=ablation_mode,
            trajectory_length_ids=val_length_ids if use_length_condition else None,
            rebuild_attention_masks=rebuild_masks,
            force_full_attention_mask=force_full_attention,
        )
        
        val_tensor_dataset = TensorDataset(val_attrs, val_dwell_times)
        val_dataset = CombinedDataset(val_tensor_dataset, val_token_dataset)
        
        valid_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.BATCH_SIZE,
            shuffle=False,  # No shuffling for validation
            num_workers=args.NUM_WORKERS,
            collate_fn=custom_collate
        )
        
        return train_dataloader, valid_dataloader, None, tokenizer_vocab


def load_restart_training_parameters(args, justparams=False):
    if justparams:
        params = args.PARAMETERS
    else:
        directory = args.RESTART_DIRECTORY
        # Get file to parse
        params = os.path.join(directory, "parameters")

    file = list(filter(lambda x: x.startswith("training_"), os.listdir(params)))[0]
    with open(os.path.join(params, file), 'r') as f:
        lines = f.readlines()

    # Parse relevant args into dict
    to_keep = ["MAX_NUM_WORDS", "IMG_SIDE_LEN", "T5_NAME", "TIMESTEPS", "prediction_type", "timestep_sampling"]
    lines = list(filter(lambda x: True if True in [x.startswith(f"--{i}") for i in to_keep] else False, lines))
    d = {}
    for line in lines:
        s = line.split("=")
        try:
            d[s[0][2:]] = int(s[1][:-1])
        except:
            d[s[0][2:]] = s[1][:-1]

    # Replace relevant values in arg dict
    args.__dict__ = {**args.__dict__, **d}
    return args




def create_directory(dir_path):
    """
    creates
    subdirectories "parameters", "state_dicts", and "tmp" under the parent directory which can be similarly
    :param dir_path: Path of directory to create
    """
    original_dir = os.getcwd()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        for i in ["parameters", "state_dicts", "tmp", "results"]:
            os.makedirs(os.path.join(dir_path, i), exist_ok=True)

    @contextmanager
    def cm(subpath=""):
        target_path = os.path.join(dir_path, subpath)
        os.makedirs(target_path, exist_ok=True)
        os.chdir(target_path)
        yield
        os.chdir(original_dir)

    return cm


def get_model_size(gen_model):
    """Returns model size in MB"""
    param_size = 0
    for param in gen_model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in gen_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024 ** 2


def save_training_info(args, timestamp, unets_params, cardiff_params, model_size, training_dir):
    # Save the training parameters
    with training_dir("parameters"):
        with open(f"training_parameters_{timestamp}.txt", "w") as f:
            for i in args.__dict__.keys():
                f.write(f'--{i}={getattr(args, i)}\n')

    with training_dir():
        with open('training_progress.txt', 'a') as f:
            restart_directory = getattr(args, "RESTART_DIRECTORY", None)
            if restart_directory is not None:
                f.write(f"STARTED FROM CHECKPOINT {restart_directory}\n")
            f.write(f'model size: {model_size:.3f}MB\n\n')

    # Save parameters
    with training_dir("parameters"):
        for idx, param in enumerate(unets_params):
            with open(f'denoiser_{idx}_params_{timestamp}.json', 'w') as f:
                json.dump(param, f, indent=4)
        with open(f'cardiff_params_{timestamp}.json', 'w') as f:
            json.dump(cardiff_params, f, indent=4)


def get_model_params(parameters_dir):
    im_params = None
    unets_params = []

    # Find appropriate files
    for file in os.listdir(parameters_dir):
        if file.startswith('cardiff'):
            im_params = file
        elif file.startswith('denoiser_'):
            unets_params.append(file)

    # Make sure UNets params are sorted properly
    unets_params = sorted(unets_params, key=lambda x: int(x.split('_')[1]))

    for idx, filepath in enumerate(unets_params):
        print(filepath)
        with open(os.path.join(parameters_dir, f'{filepath}'), 'r') as f:
            unets_params[idx] = json.loads(f.read())

    with open(os.path.join(parameters_dir, f'{im_params}'), 'r') as f:
        im_params = json.loads(f.read())

    return unets_params, im_params


def get_default_args(object):
    """Returns a dictionary of the default arguments of a function or class"""
    # For any subclass of Unet but not Unet itself
    if issubclass(object, sd_unet.Unet) and not object is sd_unet.Unet:
        return {**get_default_args(sd_unet.Unet), **object.defaults}

    signature = inspect.signature(object)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def _read_params(directory, filename):
    """Returns dictionary from JSON config file in the parameters folder of a training directory"""
    with open(os.path.join(directory, "parameters", filename), 'r') as _file:
        return json.loads(_file.read())


def load_params(directory):
    """
    Loads parameters from a training directory
    :param directory: Path of training directory generated by training
    :return: (unets_params, cardiff_params)
    """
    # Files in parameters directory
    files = os.listdir(os.path.join(directory, "parameters"))
    # Filter only param files for U-Nets
    unets_params_files = sorted(list(filter(lambda x: x.startswith("denoiser_", ), files)),
                                key=lambda x: int(x.split("_")[1]))

    unets_params = [_read_params(directory, f) for f in unets_params_files]
    cardiff_params_files = _read_params(directory, list(filter(lambda x: x.startswith("cardiff_"), files))[0])
    return unets_params, cardiff_params_files


def _instatiate_cardiff(directory):
    denoisers_params, cardiff_params_files = load_params(directory)

    return Cardiff(denoisers=[DiT(**denoisers_params[0]), Unet(**denoisers_params[1])], **cardiff_params_files)
