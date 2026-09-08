"""Custom add-on workflows built on top of native `boltz predict`.

These are UI/helpers only — Boltz itself has no add-on modes.
Each mode in the UI calls the matching module (e.g. Binder remodel →
`binder_remodel`) to expand a JSON config into Boltz YAML documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

ADD_ONS_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = ADD_ONS_DIR / "configs"
BOLTZ_ROOT = ADD_ONS_DIR.parent


def list_config_files() -> List[Path]:
    """List JSON configs under add_ons/configs/."""
    if not CONFIGS_DIR.is_dir():
        return []
    return sorted(CONFIGS_DIR.glob("*.json"))
