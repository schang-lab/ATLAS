from __future__ import annotations

import os
import sys

# Robust sys.path bootstrap for stable src/scripts imports when imported standalone.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_EVAL_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from eval import (  # noqa: F401
        MappingConfig,
        SPECIAL_TOKENS_DEFAULT,
        UNK_TOKENS_DEFAULT,
        _as_list,
        _grid_counts,
        _hist_1d,
        _hist_2d,
        _jsd,
        _wasserstein_2d,
        flatten_coords,
        load_poi_coords,
        origins_destinations,
    )
