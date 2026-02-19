import os


def is_global_process_zero():
    """Return True if this is the main (global rank 0) process."""
    rank = os.environ.get("RANK") or os.environ.get("ACCELERATE_PROCESS_INDEX")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return True
