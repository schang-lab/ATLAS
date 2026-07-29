"""ATLAS trajectory-generation package.

Importing this package puts the sibling ``autoencoder/`` directory on ``sys.path``
so that ``from auto_encoder.traj_compressed_ae import BARTLatentCompression``
resolves regardless of the working directory the entry script was launched from.
"""

import sys
from pathlib import Path

_TRAJGEN_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TRAJGEN_ROOT.parent

for _candidate in (_TRAJGEN_ROOT, _REPO_ROOT / "autoencoder"):
    _path = str(_candidate)
    if _candidate.exists() and _path not in sys.path:
        sys.path.insert(0, _path)
