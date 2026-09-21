"""Validated, atomic persistence for pack-declared settings."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path

from dinkster_protocol import PackSettingsSchema


class PackSettingsStore:
    """Persist complete pack setting objects without exposing partial writes."""

    def __init__(self, root: Path | None) -> None:
        self._root = root
        self._memory: dict[str, dict[str, object]] = {}
        self._lock = threading.Lock()

    def read(self, pack_id: str, schema: PackSettingsSchema) -> dict[str, object]:
        with self._lock:
            if self._root is None:
                return dict(self._memory.get(pack_id, schema.defaults))
            path = self._root / f"{pack_id}.json"
            try:
                raw = json.loads(path.read_text("utf-8"))
            except FileNotFoundError:
                return schema.defaults
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"stored pack settings are unreadable: {exc}") from exc
            return schema.validate(raw)

    def write(
        self, pack_id: str, schema: PackSettingsSchema, values: Mapping[str, object]
    ) -> dict[str, object]:
        validated = schema.validate(values)
        with self._lock:
            if self._root is None:
                self._memory[pack_id] = validated
                return dict(validated)
            self._root.mkdir(parents=True, exist_ok=True)
            path = self._root / f"{pack_id}.json"
            descriptor, temporary = tempfile.mkstemp(prefix=f".{pack_id}.", dir=self._root)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
                    json.dump(validated, file, sort_keys=True, separators=(",", ":"))
                    file.write("\n")
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
            return dict(validated)
