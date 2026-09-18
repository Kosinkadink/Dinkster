"""Provenance: where content can be obtained (DESIGN 3.12).

A provenance record is metadata *about an identity*: source URLs and
mirrors, license, free-form notes. It is what finally decouples "which
model does this workflow need" (a digest in the graph) from "where do I
get it" (records anyone can maintain, merge, and ship separately from
hand-made templates). A machine that lacks the bytes consults provenance,
fetches from a mirror it chooses, and verifies the digest it already knew
- the record is a *lead*, never an authority: verification is always
against the digest, so a wrong or malicious URL can waste bandwidth but
can never plant wrong bytes (fetch.py enforces this via AssetVault).

The store is a JSON file keyed by digest, written atomically for the same
reason the library index is: it can be shared between instances. Merging
ordinary records is additive because provenance accretes. Named source layers
are replaceable and removable so a subscription can withdraw only the leads
it contributed.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from .identity import require_digest


@dataclass(frozen=True)
class ProvenanceRecord:
    """Everything known about where one identity's bytes can be obtained."""

    digest: str
    sources: tuple[str, ...] = ()
    license: str = ""
    note: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        require_digest(self.digest)

    def merged(self, other: ProvenanceRecord) -> ProvenanceRecord:
        """Union of two records for one digest: sources accrete in order
        (self's first), later non-empty scalars win."""
        seen = dict.fromkeys(self.sources)
        seen.update(dict.fromkeys(other.sources))
        return ProvenanceRecord(
            digest=self.digest,
            sources=tuple(seen),
            license=other.license or self.license,
            note=other.note or self.note,
            metadata={**self.metadata, **other.metadata},
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "sources": list(self.sources),
            "license": self.license,
            "note": self.note,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> ProvenanceRecord | None:
        digest = wire.get("digest")
        if not isinstance(digest, str):
            return None
        sources_raw = wire.get("sources")
        sources = (
            tuple(str(url) for url in cast("list[object]", sources_raw))
            if isinstance(sources_raw, list)
            else ()
        )
        metadata_raw = wire.get("metadata")
        metadata = (
            {str(k): v for k, v in cast("Mapping[object, object]", metadata_raw).items()}
            if isinstance(metadata_raw, Mapping)
            else {}
        )
        try:
            return cls(
                digest=digest,
                sources=sources,
                license=str(wire.get("license", "")),
                note=str(wire.get("note", "")),
                metadata=metadata,
            )
        except Exception:  # noqa: BLE001 - malformed rows are skipped, not fatal
            return None


class ProvenanceStore:
    """Digest-keyed provenance records persisted as one JSON file.

    Loads eagerly and conservatively (malformed rows are dropped, a
    malformed file is an empty store); saves atomically after every
    mutation. Named source layers can be replaced or removed as a unit,
    while ordinary records remain additive. Small by nature - records are
    URLs and notes, not bytes."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._records: dict[str, ProvenanceRecord] = {}
        self._source_records: dict[str, dict[str, ProvenanceRecord]] = {}
        try:
            loaded: object = json.loads(self._path.read_text("utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(loaded, list):
            self._records = self._load_records(cast("list[object]", loaded))
            return
        if not isinstance(loaded, Mapping):
            return
        loaded_wire = cast("Mapping[str, object]", loaded)
        if loaded_wire.get("dinksterProvenance") != 1:
            return
        records = loaded_wire.get("records")
        if isinstance(records, list):
            self._records = self._load_records(cast("list[object]", records))
        source_records = loaded_wire.get("sourceRecords")
        if not isinstance(source_records, Mapping):
            return
        for source, rows in cast("Mapping[object, object]", source_records).items():
            if isinstance(source, str) and source and isinstance(rows, list):
                self._source_records[source] = self._load_records(cast("list[object]", rows))

    @staticmethod
    def _load_records(rows: list[object]) -> dict[str, ProvenanceRecord]:
        records: dict[str, ProvenanceRecord] = {}
        for row in rows:
            if isinstance(row, Mapping):
                record = ProvenanceRecord.from_wire(cast("Mapping[str, object]", row))
                if record is not None:
                    records[record.digest] = record
        return records

    def add(self, record: ProvenanceRecord) -> ProvenanceRecord:
        """Insert or additively merge; returns the stored record."""
        with self._lock:
            existing = self._records.get(record.digest)
            merged = existing.merged(record) if existing is not None else record
            self._records[record.digest] = merged
            try:
                self._save()
            except OSError:
                if existing is None:
                    del self._records[record.digest]
                else:
                    self._records[record.digest] = existing
                raise
            return self._merged_record(record.digest)

    def replace_source(self, source: str, records: tuple[ProvenanceRecord, ...]) -> None:
        """Replace every record contributed by one removable source."""
        if not source:
            raise ValueError("provenance source must be non-empty")
        replacement: dict[str, ProvenanceRecord] = {}
        for record in records:
            existing = replacement.get(record.digest)
            replacement[record.digest] = existing.merged(record) if existing is not None else record
        with self._lock:
            if self._source_records.get(source) == replacement or (
                not replacement and source not in self._source_records
            ):
                return
            existing = self._source_records.get(source)
            if replacement:
                self._source_records[source] = replacement
            else:
                self._source_records.pop(source, None)
            try:
                self._save()
            except OSError:
                if existing is None:
                    self._source_records.pop(source, None)
                else:
                    self._source_records[source] = existing
                raise

    def remove_source(self, source: str) -> bool:
        """Remove one named source without disturbing any other records."""
        with self._lock:
            if source not in self._source_records:
                return False
            existing = self._source_records.pop(source)
            try:
                self._save()
            except OSError:
                self._source_records[source] = existing
                raise
            return True

    def source_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._source_records)

    def _merged_record(self, digest: str) -> ProvenanceRecord:
        record = self._records.get(digest)
        for records in self._source_records.values():
            contributed = records.get(digest)
            if contributed is not None:
                record = contributed if record is None else record.merged(contributed)
        assert record is not None
        return record

    def get(self, digest: str) -> ProvenanceRecord | None:
        digest = require_digest(digest)
        with self._lock:
            if digest not in self._records and not any(
                digest in records for records in self._source_records.values()
            ):
                return None
            return self._merged_record(digest)

    def sources(self, digest: str) -> tuple[str, ...]:
        """Source URLs for a digest, best-first - the fetcher's input."""
        record = self.get(digest)
        return record.sources if record is not None else ()

    def records(self) -> tuple[ProvenanceRecord, ...]:
        with self._lock:
            digests = set(self._records)
            for records in self._source_records.values():
                digests.update(records)
            return tuple(self._merged_record(digest) for digest in sorted(digests))

    def _save(self) -> None:
        serialized = json.dumps(
            {
                "dinksterProvenance": 1,
                "records": [self._records[digest].to_wire() for digest in sorted(self._records)],
                "sourceRecords": {
                    source: [records[digest].to_wire() for digest in sorted(records)]
                    for source, records in self._source_records.items()
                },
            },
            indent=1,
        )
        tmp = self._path.with_name(self._path.name + f".tmp-{os.getpid()}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_text(serialized, "utf-8")
            os.replace(tmp, self._path)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
