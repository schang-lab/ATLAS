import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)


def load_test_data(controlled_folder=None, uncontrolled_folder=None, data_folder=None, split="test"):
    """Load test data for evaluation."""
    if controlled_folder is not None and uncontrolled_folder is not None:
        controlled_folder = Path(controlled_folder)
        uncontrolled_folder = Path(uncontrolled_folder)
        logger.info("Loading test data from dual folders...")

        controlled_test_path = controlled_folder / split / "final_segments_all_train_data.pkl"
        controlled_test_df = pd.read_pickle(controlled_test_path)
        logger.info("Loaded %s controlled test sequences", len(controlled_test_df))

        uncontrolled_test_path = uncontrolled_folder / split / "final_segments_all_train_data.pkl"
        uncontrolled_test_df = pd.read_pickle(uncontrolled_test_path)
        logger.info("Loaded %s uncontrolled test sequences", len(uncontrolled_test_df))

        test_df = pd.concat([controlled_test_df, uncontrolled_test_df], ignore_index=True)
        logger.info("Combined test dataset: %s total sequences", len(test_df))
        return test_df

    data_folder = Path(data_folder)
    test_path = data_folder / split / "final_segments_all_train_data.pkl"
    test_df = pd.read_pickle(test_path)
    logger.info("Loaded %s test sequences from %s", len(test_df), test_path)
    return test_df


def find_poi_metadata_path(controlled_folder=None, data_folder=None, poi_metadata_path=None):
    candidates = []
    if poi_metadata_path:
        candidates.append(Path(poi_metadata_path))

    def _extend(folder):
        if not folder:
            return
        folder_path = Path(folder)
        candidates.append(folder_path / "train" / "poi_map_feature.csv")
        candidates.append(folder_path / "poi_map_feature.csv")

    _extend(controlled_folder)
    _extend(data_folder)
    for path in candidates:
        if path and path.exists():
            return path
    return None


def load_poi_metadata(controlled_folder=None, data_folder=None, poi_metadata_path=None):
    metadata_path = find_poi_metadata_path(
        controlled_folder=controlled_folder,
        data_folder=data_folder,
        poi_metadata_path=poi_metadata_path,
    )
    if metadata_path is None:
        logger.warning("Latent probe requested but poi_map_feature.csv could not be located.")
        return {}

    logger.info("Loading POI metadata from %s", metadata_path)
    poi_metadata_df = pd.read_csv(metadata_path)
    if not {"poi_id"}.issubset(poi_metadata_df.columns):
        raise ValueError(f"POI metadata file {metadata_path} missing required columns: {{'poi_id'}}")

    coord_map = {}
    for _, row in poi_metadata_df.iterrows():
        poi_id = str(row["poi_id"])
        lat = row["lat"] if "lat" in row and not pd.isna(row["lat"]) else None
        lon = row["lon"] if "lon" in row and not pd.isna(row["lon"]) else None
        top_category = row.get("top_category")
        if pd.isna(top_category) or not top_category:
            top_category = row.get("category")
        if pd.isna(top_category) or not top_category:
            top_category = "unknown"
        coord_map[poi_id] = {
            "lat": float(lat) if lat is not None else None,
            "lon": float(lon) if lon is not None else None,
            "top_category": str(top_category),
        }
    logger.info("Loaded %s POI metadata rows for latent probe", len(coord_map))
    return coord_map


def load_tokenizer(model_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    logger.info("Loaded tokenizer with vocab size: %s", len(tokenizer))
    return tokenizer


def aggregate_metrics(all_metrics):
    if not all_metrics:
        return {}
    aggregated = defaultdict(list)
    for sample_metrics in all_metrics:
        for key, value in sample_metrics.items():
            aggregated[key].append(value)
    results = {}
    for key, values in aggregated.items():
        if key in ["original_length", "reconstructed_length"]:
            results[f"{key}_mean"] = np.mean(values)
            results[f"{key}_std"] = np.std(values)
            results[f"{key}_min"] = np.min(values)
            results[f"{key}_max"] = np.max(values)
        else:
            results[f"{key}_mean"] = np.mean(values)
            results[f"{key}_std"] = np.std(values)
    return results


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def save_results(results, detailed_results, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    results_to_save = convert_numpy_types(results)
    detailed_results_to_save = convert_numpy_types(detailed_results)

    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2)

    detailed_file = os.path.join(output_dir, "detailed_reconstructions.json")
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results_to_save, f, indent=2)

    csv_file = os.path.join(output_dir, "detailed_reconstructions.csv")
    detailed_df = pd.DataFrame(
        [
            {
                "individual_id": result["individual_id"],
                "sample_idx": result["sample_idx"],
                "original_sequence": " ".join(result["original_sequence"]),
                "reconstructed_sequence": " ".join(result["reconstructed_sequence"]),
                "original_length": result["metrics"]["original_length"],
                "reconstructed_length": result["metrics"]["reconstructed_length"],
                "token_accuracy": result["metrics"]["token_accuracy"],
                "token_accuracy_lenient": result["metrics"]["token_accuracy_lenient"],
                "sequence_accuracy": result["metrics"]["sequence_accuracy"],
                "edit_distance": result["metrics"]["edit_distance"],
                "jaccard_similarity": result["metrics"]["jaccard_similarity"],
                "bleu_1": result["metrics"]["bleu_1"],
                "bleu_2": result["metrics"]["bleu_2"],
                "bleu_3": result["metrics"]["bleu_3"],
                "bleu_4": result["metrics"]["bleu_4"],
            }
            for result in detailed_results_to_save
        ]
    )
    detailed_df.to_csv(csv_file, index=False)

    config_file = os.path.join(output_dir, "evaluation_config.json")
    config = {
        "model_path": args.model_path,
        "controlled_folder": getattr(args, "controlled_folder", None),
        "uncontrolled_folder": getattr(args, "uncontrolled_folder", None),
        "data_folder": getattr(args, "data_folder", None),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "sample_size": args.sample_size,
        "random_seed": args.random_seed,
        "generation_config": {
            "max_length": args.generation_max_length,
            "min_length": args.generation_min_length,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "length_penalty": args.length_penalty,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample,
            "top_p": args.top_p if args.do_sample else None,
        },
        "enable_latent_probe": args.enable_latent_probe,
        "poi_metadata_path": getattr(args, "poi_metadata_path", None),
        "latent_pair_samples": getattr(args, "latent_pair_samples", None),
        "latent_triplet_samples": getattr(args, "latent_triplet_samples", None),
        "latent_knn_k": getattr(args, "latent_knn_k", None),
        "geo_neighbor_km": getattr(args, "geo_neighbor_km", None),
        "max_latent_buffer": getattr(args, "max_latent_buffer", None),
        "evaluation_date": datetime.now().isoformat(),
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Results saved to %s", output_dir)
    logger.info("- Aggregated results: %s", results_file)
    logger.info("- Detailed reconstructions (JSON): %s", detailed_file)
    logger.info("- Detailed reconstructions (CSV): %s", csv_file)
