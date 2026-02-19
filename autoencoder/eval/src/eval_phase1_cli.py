import argparse


def parse_eval_phase1_args():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 pretrained BART autoencoder")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model directory")

    # Data arguments
    parser.add_argument("--data_folder", type=str, help="Single data folder path (legacy mode)")
    parser.add_argument("--controlled_folder", type=str, help="Path to controlled data folder")
    parser.add_argument("--uncontrolled_folder", type=str, help="Path to uncontrolled data folder")
    parser.add_argument("--dual_folders", nargs=2, metavar=("CONTROLLED", "UNCONTROLLED"), help="Controlled and uncontrolled folder paths")

    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length (will be capped to model.config.max_position_embeddings if smaller)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory for results")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to evaluate (None = full dataset)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling")

    # Generation arguments
    parser.add_argument("--generation_max_length", type=int, default=512, help="Maximum length for generation (should be < max_length to leave room for special tokens)")
    parser.add_argument("--generation_min_length", type=int, default=20, help="Minimum length for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty for generation (1.0 = no penalty, >1.0 = penalize repetitions)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Prevent n-gram repetitions (0 = no constraint, 2-3 recommended)")
    parser.add_argument("--length_penalty", type=float, default=1.2, help="Length penalty for generation (1.0 = neutral, >1.0 = prefer longer sequences)")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search (1 = greedy decoding, 2-4 = beam search)")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy/beam search")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter (only used when do_sample=True)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (lower = more confident predictions)")

    # Latent probe arguments
    parser.add_argument("--enable_latent_probe", action="store_true", help="Enable latent geometry/anisotropy evaluation (requires POI metadata)")
    parser.add_argument("--poi_metadata_path", type=str, default=None, help="Explicit path to poi_map_feature.csv (overrides auto-detection)")
    parser.add_argument("--latent_pair_samples", type=int, default=20000, help="Maximum number of latent pairs to sample for correlation metrics")
    parser.add_argument("--latent_triplet_samples", type=int, default=5000, help="Maximum number of triplets for margin accuracy metrics")
    parser.add_argument("--latent_knn_k", type=int, default=10, help="k for latent-space kNN recall metrics")
    parser.add_argument("--geo_neighbor_km", type=float, default=0.5, help="Geographic distance threshold (km) for positives/recall")
    parser.add_argument("--max_latent_buffer", type=int, default=6000, help="Cap on the number of latent vectors stored for probe metrics")
    args = parser.parse_args()

    if args.dual_folders:
        args.controlled_folder, args.uncontrolled_folder = args.dual_folders

    if not any([args.data_folder, (args.controlled_folder and args.uncontrolled_folder)]):
        raise ValueError("Must provide either --data_folder or both --controlled_folder and --uncontrolled_folder")

    return args
