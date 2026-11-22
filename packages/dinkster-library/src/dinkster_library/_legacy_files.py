"""Convert frozen preferences and sources without touching live stores."""

from __future__ import annotations

import json
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._provenance import provenance_document, replace_layer
from ._records import object_value
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def convert_sources(backup: Path, records: Records) -> None:
    """Keep raw source documents and their independent provenance layers."""
    library = backup / "library"
    layers: dict[str, object] = {}
    path = library / "provenance.json"
    if path.is_file():
        document: object = json.loads(path.read_text("utf-8"))
        if isinstance(document, list):
            ordinary = cast("list[object]", document)
        else:
            value = object_value(document, "provenance store")
            if type(value.get("dinksterProvenance")) is not int or value["dinksterProvenance"] != 1:
                raise LibraryError("migration", "Unsupported provenance store format")
            ordinary = value.get("records", [])
            layers = object_value(value.get("sourceRecords", {}), "provenance layers")
        if not isinstance(ordinary, list):
            raise LibraryError("migration", "Provenance records must be a list")
        for row in cast("list[object]", ordinary):
            records.dispatch("sources.provenance.add", {"record": row})
        for source, rows in layers.items():
            if not source or not isinstance(rows, list):
                raise LibraryError("migration", "Provenance layers require named record lists")
            incoming: dict[str, dict[str, object]] = {}
            for row in cast("list[object]", rows):
                record = provenance_document(row)
                incoming[cast(str, record["digest"])] = record
            with records.db:
                replace_layer(records, source, incoming)
    path = library / "resolver-indexes.json"
    if path.is_file():
        value = object_value(json.loads(path.read_text("utf-8")), "resolver store")
        if (
            type(value.get("dinksterResolverSubscriptions")) is not int
            or value["dinksterResolverSubscriptions"] != 1
        ):
            raise LibraryError("migration", "Unsupported resolver store format")
        rows = value.get("subscriptions")
        if not isinstance(rows, list):
            raise LibraryError("migration", "Resolver subscriptions must be a list")
        bootstrap = value.get("officialBootstrap")
        for raw in cast("list[object]", rows):
            subscription = object_value(raw, "resolver subscription")
            identifier = subscription.get("id")
            source = "resolver-index:" + str(identifier)
            saved_leads = layers.get(source)
            if saved_leads is not None and not isinstance(saved_leads, list):
                raise LibraryError("migration", "Resolver provenance must be a list")
            leads: list[object] = (
                cast("list[object]", saved_leads) if saved_leads is not None else []
            )
            if saved_leads is None:
                index = object_value(subscription.get("index"), "resolver index")
                entries = index.get("entries")
                if not isinstance(entries, list):
                    raise LibraryError("migration", "Resolver entries must be a list")
                for entry in cast("list[object]", entries):
                    row = object_value(entry, "resolver entry")
                    metadata: dict[str, object] = {
                        "name": row.get("name", ""),
                        "resolverIndex": subscription.get("source"),
                    }
                    for name in ("kind", "size", "family", "variant", "components", "gated"):
                        if name in row:
                            metadata[name] = row[name]
                    leads.append(
                        {
                            "digest": row.get("digest"),
                            "sources": row.get("urls", []),
                            "license": row.get("license", ""),
                            "note": row.get("notes", ""),
                            "metadata": metadata,
                        }
                    )
            arguments: dict[str, object] = {
                "updates": [
                    {
                        "id": identifier,
                        "revision": 0,
                        "snapshot": {"subscription": subscription, "records": leads},
                    }
                ]
            }
            if (
                bootstrap is not None
                and object_value(bootstrap, "official bootstrap").get("subscriptionId")
                == identifier
            ):
                arguments["officialBootstrap"] = bootstrap
            records.dispatch("sources.resolvers.patch", arguments)
        if bootstrap is not None:
            records.dispatch("sources.resolvers.bootstrap.put", {"configuration": bootstrap})
    path = library / "public-acquisition-receipts.json"
    if path.is_file():
        value = object_value(json.loads(path.read_text("utf-8")), "receipt store")
        if (
            type(value.get("publicAcquisitionReceipts")) is not int
            or value["publicAcquisitionReceipts"] != 1
        ):
            raise LibraryError("migration", "Unsupported acquisition receipt store format")
        rows = value.get("receipts")
        if not isinstance(rows, list):
            raise LibraryError("migration", "Acquisition receipts must be a list")
        for row in cast("list[object]", rows):
            records.dispatch("sources.receipts.put", {"receipt": row})


def convert_preferences(backup: Path, records: Records, environment: str | None) -> None:
    """Populate only unpublished preferences from the verified source snapshot."""
    library = backup / "library"
    for filename, operation in (
        ("project-folders.json", "folders.projects.patch"),
        ("mounts.toml", "folders.grants.patch"),
    ):
        path = library / filename
        if path.is_file():
            document = (
                tomllib.loads(path.read_text("utf-8"))
                if path.suffix == ".toml"
                else object_value(json.loads(path.read_text("utf-8")), filename)
            )
            records.dispatch(operation, {"revision": 0, "patch": document})
    settings = library / "settings.json"
    if settings.is_file():
        if not environment:
            raise LibraryError(
                "migration", "Environment settings require an explicit environment ID"
            )
        records.dispatch(
            "settings.patch",
            {
                "environment": environment,
                "revision": 0,
                "patch": object_value(json.loads(settings.read_text("utf-8")), "settings"),
            },
        )
    packs = library / "pack-settings"
    for path in sorted(packs.rglob("*.json")):
        records.dispatch(
            "settings.patch",
            {
                "pack": path.relative_to(packs).with_suffix("").as_posix(),
                "revision": 0,
                "patch": object_value(json.loads(path.read_text("utf-8")), "pack settings"),
            },
        )
    account = backup / "account"
    previous = account / "model-settings.sqlite"
    settings = account / "model-settings.json"
    selections = account / "model-file-selections.json"
    if not any(path.is_file() for path in (previous, settings, selections)):
        return
    target = records.root / "account"
    target.mkdir()
    with closing(sqlite3.connect(target / "model-settings.sqlite")) as db:
        if previous.is_file():
            if settings.is_file() or selections.is_file():
                raise LibraryError("migration", "Account preferences have competing source stores")
            with closing(
                sqlite3.connect(previous.as_uri() + "?mode=ro&immutable=1", uri=True)
            ) as source:
                if source.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise LibraryError("migration", "Unsupported account preference format")
                source.backup(db)
            return
        values: dict[str, object] = {}
        if settings.is_file():
            document = object_value(json.loads(settings.read_text("utf-8")), "model settings")
            if type(document.get("version")) is not int or document["version"] != 1:
                raise LibraryError("migration", "Unsupported model preference format")
            raw_locations = document.get("locations")
            if not isinstance(raw_locations, list):
                raise LibraryError("migration", "Model locations must be a list")
            locations: dict[str, object] = {}
            for raw in cast("list[object]", raw_locations):
                row = dict(object_value(raw, "model location"))
                identifier = row.pop("id", None)
                if not isinstance(identifier, str) or not identifier or identifier in locations:
                    raise LibraryError("migration", "Model location IDs must be distinct strings")
                locations[identifier] = row
            values = {key: value for key, value in document.items() if key != "version"}
            values.update(locations=locations, order=list(locations))
        if selections.is_file():
            rows: object = json.loads(selections.read_text("utf-8"))
            if not isinstance(rows, list):
                raise LibraryError("migration", "Model file selections must be a list")
            files: dict[str, object] = {}
            for raw in cast("list[object]", rows):
                row = object_value(raw, "model file selection")
                asset = object_value(row.get("asset"), "selected model asset")
                identifier = asset.get("virtualPath")
                if not isinstance(identifier, str) or not identifier or identifier in files:
                    raise LibraryError("migration", "Selected model paths must be distinct strings")
                files[identifier] = row
            values["files"] = files
        with db:
            db.execute(
                "CREATE TABLE preferences (id INTEGER PRIMARY KEY CHECK(id=1), "
                "revision INTEGER NOT NULL, document TEXT NOT NULL)"
            )
            db.execute("PRAGMA user_version=1")
            db.execute(
                "INSERT INTO preferences VALUES (1,1,?)",
                (json.dumps(values, allow_nan=False),),
            )
