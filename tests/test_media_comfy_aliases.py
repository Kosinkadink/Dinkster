"""Media alias fixtures shared by video template tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).parent.parent
ALIAS_PATH = ROOT / "packages" / "dinkster-nodes-media-io" / "comfy-aliases.json"


def _registry() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(ALIAS_PATH.read_text(encoding="utf-8")))
