"""Utility helpers for trajectory length computation used across training/inference."""

from __future__ import annotations

from typing import Iterable, List, Sequence

# Tokens considered non-POI and excluded from discrete length counts.
SPECIAL_TOKENS = {
    "[SEP]", "[PAD]", "[CLS]", "[MASK]",
}


def count_poi_tokens(sequence: Sequence[str], max_length: int | None = None) -> int:
    """Count POI tokens in a trajectory, excluding special tokens.

    Args:
        sequence: Trajectory represented as a sequence of token strings.
        max_length: Optional truncation length. When provided, the count is
            restricted to the first ``max_length`` entries.

    Returns:
        Number of POI tokens after filtering specials.
    """
    if sequence is None:
        return 0

    filtered = 0
    upper_bound = len(sequence) if max_length is None else min(len(sequence), max_length)
    for idx in range(upper_bound):
        token = sequence[idx]
        if token in SPECIAL_TOKENS:
            continue
        filtered += 1
    return filtered


def batch_count_lengths(
    sequences: Iterable[Sequence[str]], max_length: int | None = None
) -> List[int]:
    """Compute length ids for a batch of trajectories."""
    return [count_poi_tokens(seq, max_length=max_length) for seq in sequences]
