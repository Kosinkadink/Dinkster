from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_model_sampling_flux_golden() -> dict[str, Any]:
    resource = files(__package__).joinpath("data/model_sampling_flux_non_square.json")
    document: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return document


__all__ = ["load_model_sampling_flux_golden"]
