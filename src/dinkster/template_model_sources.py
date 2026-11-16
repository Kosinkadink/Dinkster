"""Advisory filename leads from declared official template model sources."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import cast

from .model_settings import MODEL_FOLDERS


class TemplateModelSources:
    def __init__(self) -> None:
        data = cast(
            "dict[str, object]",
            json.loads(files("dinkster").joinpath("template-model-sources.json").read_text()),
        )
        sources: dict[tuple[str, str], set[str]] = {}
        for row in cast("list[dict[str, str]]", data["models"]):
            sources.setdefault((row["directory"], row["name"]), set()).add(row["url"])
        self._sources = {key: tuple(sorted(urls)) for key, urls in sources.items()}

    def for_name(self, name: str, kind: str) -> tuple[str, ...]:
        """Keep ambiguous declarations as leads, never choose or authorize a transfer."""
        folder = MODEL_FOLDERS.get(kind)
        if folder is None or not name or "/" in name or "\\" in name or name in {".", ".."}:
            return ()
        sources = self._sources.get((folder, name), ())
        if kind == "model/text-encoder":
            sources = tuple(sorted(set(sources) | set(self._sources.get(("clip", name), ()))))
        return sources
