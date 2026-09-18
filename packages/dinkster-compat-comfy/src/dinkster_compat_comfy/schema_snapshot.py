"""Pinned upstream schemas for import metadata, never executable classes."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def core_schema_snapshot() -> dict[str, Any]:
    return json.loads(files("dinkster_compat_comfy").joinpath("core_schemas.json").read_text())
