"""Put the pack directory on sys.path, exactly as the worker host does
when it resolves manifest entries."""

from __future__ import annotations

import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))
