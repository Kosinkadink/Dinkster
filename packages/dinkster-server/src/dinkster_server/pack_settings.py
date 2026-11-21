"""Validated, atomic persistence for pack-declared settings."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from dinkster_library import LibraryClient
from dinkster_protocol import PackSettingsSchema


class PackSettingsStore:
    """Persist complete pack setting objects without exposing partial writes."""

    def __init__(self, root: Path | LibraryClient | None) -> None:
        self._helper = root if isinstance(root, LibraryClient) else None
        self._root = root if isinstance(root, Path) else None
        if self._helper is not None and "settings" not in cast(
            "list[str]", self._helper.hello["capabilities"]
        ):
            raise ValueError("library helper does not support pack settings")
        self._memory: dict[str, dict[str, object]] = {}
        self._revisions: dict[str, int] = {}
        self._lock = threading.Lock()

    def _read_shared(self, pack_id: str, schema: PackSettingsSchema) -> dict[str, object]:
        assert self._helper is not None
        record = cast(
            "dict[str, object]", self._helper.call("settings.get", pack=pack_id)["settings"]
        )
        values = cast("dict[str, object]", record["values"])
        projected = schema.validate(
            {name: values.get(name, field.default) for name, field in schema.fields.items()}
        )
        self._memory[pack_id] = values
        self._revisions[pack_id] = cast(int, record["revision"])
        return dict(projected)

    def read(self, pack_id: str, schema: PackSettingsSchema) -> dict[str, object]:
        with self._lock:
            if self._helper is not None:
                return self._read_shared(pack_id, schema)
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
            if self._helper is not None:
                if pack_id not in self._revisions:
                    self._read_shared(pack_id, schema)
                previous = self._memory[pack_id]
                patch = {
                    name: value
                    for name, value in validated.items()
                    if name not in previous or previous[name] != value
                }
                record = cast(
                    "dict[str, object]",
                    self._helper.call(
                        "settings.patch",
                        pack=pack_id,
                        revision=self._revisions[pack_id],
                        patch=patch,
                    )["settings"],
                )
                self._memory[pack_id] = cast("dict[str, object]", record["values"])
                self._revisions[pack_id] = cast(int, record["revision"])
                return dict(validated)
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
