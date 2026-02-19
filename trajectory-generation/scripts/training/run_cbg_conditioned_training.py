#!/usr/bin/env python3
"""
Fine-tune a pretrained diffusion model with CBG-conditioned aggregate supervision.

Example:
    python trajectory-generation/scripts/training/run_cbg_conditioned_training.py \
        --config /abs/path/to/trajectory-generation/configs/cbg_conditioned_training.yaml
"""

from __future__ import annotations

import os
import sys

# Allow running this script from any working directory by adding `trajectory-generation/` to sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)

from lib.run_cbg_entry import run_entrypoint
from lib.run_cbg_trainer import AggregateTrainer


def main() -> None:
    run_entrypoint(AggregateTrainer)


if __name__ == "__main__":
    main()
