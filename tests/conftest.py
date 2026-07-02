"""Test setup: make scripts/ and the Kit extension package importable."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
COASTER_DIR = ROOT / "source" / "extensions" / "omni.marble.coaster" / "omni" / "marble" / "coaster"

for p in (str(SCRIPTS_DIR), str(COASTER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
