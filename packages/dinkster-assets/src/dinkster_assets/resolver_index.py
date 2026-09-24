"""Forward-compatible declarative resolver-index format."""

from __future__ import annotations
from dinkster_values import MEBIBYTE

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import cast
from urllib.parse import urlsplit

from .component_manifest import AssetComponentManifest
from .identity import AssetError, require_digest
from .json_metadata import freeze_json, thaw_json
from .kind import require_asset_kind
from .p2p_descriptor import P2PDescriptorError, P2PDescriptorV1, validate_p2p_descriptor

RESOLVER_INDEX_VERSION = 1
RESOLVER_INDEX_MAX_BYTES = 16 * MEBIBYTE
RESOLVER_INDEX_MAX_ENTRIES = 100_000
RESOLVER_INDEX_MAX_URLS = 64
RESOLVER_INDEX_MAX_STRING = 4096

_REGION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")


class ResolverIndexError(AssetError):
    """A resolver index is malformed, unsupported, or over its limits."""


def _string(
    wire: Mapping[str, object],
    field_name: str,
    *,
    required: bool = False,
) -> str:
    if field_name not in wire and not required:
        return ""
    value = wire.get(field_name)
    if not isinstance(value, str):
        raise ResolverIndexError(f"{field_name!r} must be a string")
    if required and not value.strip():
        raise ResolverIndexError(f"{field_name!r} must be non-empty")
    if len(value) > RESOLVER_INDEX_MAX_STRING:
        raise ResolverIndexError(f"{field_name!r} exceeds {RESOLVER_INDEX_MAX_STRING} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ResolverIndexError(f"{field_name!r} must not contain control characters")
    return value


def require_https_url(value: str, field_name: str = "URL") -> str:
    if any(character.isspace() for character in value):
        raise ResolverIndexError(f"{field_name} must not contain raw whitespace")
    if len(value) > RESOLVER_INDEX_MAX_STRING:
        raise ResolverIndexError(f"{field_name} exceeds {RESOLVER_INDEX_MAX_STRING} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ResolverIndexError(f"{field_name} must not contain control characters")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ResolverIndexError(f"{field_name} is invalid: {value!r}") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.netloc.rsplit("@", 1)[-1].endswith(":")
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ResolverIndexError(
            f"{field_name} must be an HTTPS URL without credentials or a fragment: {value!r}"
        )
    return value


def require_region(value: str) -> str:
    if _REGION_RE.fullmatch(value) is None:
        raise ResolverIndexError("region must be 1-32 lowercase alphanumeric/hyphen characters")
    return value


def _urls(value: object, field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ResolverIndexError(f"{field_name} must be a non-empty list of HTTPS URLs")
    raw = cast("Sequence[object]", value)
    if not raw and not allow_empty:
        raise ResolverIndexError(f"{field_name} must be a non-empty list of HTTPS URLs")
    if len(raw) > RESOLVER_INDEX_MAX_URLS:
        raise ResolverIndexError(f"{field_name} accepts at most {RESOLVER_INDEX_MAX_URLS} URLs")
    urls: list[str] = []
    for position, url in enumerate(raw):
        if not isinstance(url, str):
            raise ResolverIndexError(f"{field_name}[{position}] must be a string")
        urls.append(require_https_url(url, f"{field_name}[{position}]"))
    if len(set(urls)) != len(urls):
        raise ResolverIndexError(f"{field_name} must not contain duplicate URLs")
    return tuple(urls)


@dataclass(frozen=True)
class ResolverIndexEntry:
    digest: str
    name: str
    urls: tuple[str, ...]
    kind: str = ""
    size: int = -1
    license: str = ""
    notes: str = ""
    regions: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    family: str = ""
    variant: Mapping[str, object] = field(default_factory=dict[str, object])
    component_manifest: AssetComponentManifest | None = None
    p2p: P2PDescriptorV1 | None = None
    gated: bool = False

    def __post_init__(self) -> None:
        if type(self.gated) is not bool:
            raise ResolverIndexError("gated must be a boolean")

    def urls_for(self, region: str = "") -> tuple[str, ...]:
        regional = self.regions.get(region, ()) if region else ()
        return tuple(dict.fromkeys((*regional, *self.urls)))

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "digest": self.digest,
            "name": self.name,
            "urls": list(self.urls),
        }
        if self.kind:
            wire["kind"] = self.kind
        if self.size >= 0:
            wire["size"] = self.size
        if self.license:
            wire["license"] = self.license
        if self.gated:
            wire["gated"] = self.gated
        if self.notes:
            wire["notes"] = self.notes
        if self.regions:
            wire["regions"] = {region: list(urls) for region, urls in self.regions.items()}
        if self.family:
            wire["family"] = self.family
        if self.variant:
            wire["variant"] = thaw_json(self.variant)
        if self.component_manifest is not None:
            wire["components"] = self.component_manifest.to_wire()
        if self.p2p is not None:
            wire["p2p"] = self.p2p.to_wire()
        return wire


@dataclass(frozen=True)
class ResolverIndex:
    entries: tuple[ResolverIndexEntry, ...]
    name: str = ""
    description: str = ""
    homepage: str = ""
    updated: str = ""

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "dinksterResolver": RESOLVER_INDEX_VERSION,
            "entries": [entry.to_wire() for entry in self.entries],
        }
        if self.name:
            wire["name"] = self.name
        if self.description:
            wire["description"] = self.description
        if self.homepage:
            wire["homepage"] = self.homepage
        if self.updated:
            wire["updated"] = self.updated
        return wire


def _entry_from_wire(wire: Mapping[str, object], position: int) -> ResolverIndexEntry:
    prefix = f"entries[{position}]"
    digest = _string(wire, "digest", required=True)
    try:
        require_digest(digest)
    except AssetError as exc:
        raise ResolverIndexError(f"{prefix}.digest: {exc}") from exc
    name = _string(wire, "name", required=True)
    kind = _string(wire, "kind")
    if kind:
        try:
            require_asset_kind(kind)
        except AssetError as exc:
            raise ResolverIndexError(f"{prefix}.kind: {exc}") from exc
    size = wire.get("size", -1)
    if size != -1 and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
        raise ResolverIndexError(f"{prefix}.size must be a non-negative integer")
    license_value = _string(wire, "license")
    notes = _string(wire, "notes")
    family = _string(wire, "family")
    regions_raw = wire.get("regions", {})
    if not isinstance(regions_raw, Mapping):
        raise ResolverIndexError(f"{prefix}.regions must be an object")
    regions: dict[str, tuple[str, ...]] = {}
    for region, regional_urls in cast("Mapping[object, object]", regions_raw).items():
        if not isinstance(region, str):
            raise ResolverIndexError(f"{prefix}.regions keys must be strings")
        try:
            require_region(region)
        except ResolverIndexError as exc:
            raise ResolverIndexError(f"{prefix}.regions[{region!r}]: {exc}") from exc
        regions[region] = _urls(regional_urls, f"{prefix}.regions[{region!r}]")
    variant_raw = wire.get("variant", {})
    if not isinstance(variant_raw, Mapping):
        raise ResolverIndexError(f"{prefix}.variant must be an object")
    components_raw = wire.get("components")
    if isinstance(components_raw, Sequence) and not isinstance(components_raw, (str, bytes)):
        for component_position, component in enumerate(cast("Sequence[object]", components_raw)):
            if not isinstance(component, Mapping):
                continue
            component_wire = cast("Mapping[object, object]", component)
            if any(not isinstance(key, str) for key in component_wire):
                raise ResolverIndexError(
                    f"{prefix}.components[{component_position}] keys must be strings"
                )
    try:
        variant = cast(
            "Mapping[str, object]",
            freeze_json(
                cast("Mapping[object, object]", variant_raw),
                f"{prefix}.variant",
            ),
        )
        component_manifest = (
            AssetComponentManifest.from_wire(cast("object", components_raw))
            if "components" in wire
            else None
        )
    except (AssetError, RecursionError) as exc:
        raise ResolverIndexError(f"{prefix}: {exc}") from exc
    p2p: P2PDescriptorV1 | None = None
    if "p2p" in wire:
        p2p_raw = wire["p2p"]
        try:
            if cast("int", size) <= 0:
                raise P2PDescriptorError("descriptor requires a positive entry size")
            if not isinstance(p2p_raw, Mapping) or any(
                not isinstance(key, str) for key in cast("Mapping[object, object]", p2p_raw)
            ):
                raise P2PDescriptorError("descriptor must be an object")
            p2p = validate_p2p_descriptor(
                cast("Mapping[str, object]", p2p_raw),
                asset_digest=digest,
                size=cast("int", size),
            )
        except P2PDescriptorError:
            p2p = None
    urls = _urls(wire.get("urls"), f"{prefix}.urls", allow_empty=p2p is not None)
    return ResolverIndexEntry(
        digest=digest,
        name=name,
        urls=urls,
        kind=kind,
        size=cast("int", size),
        license=license_value,
        notes=notes,
        regions=MappingProxyType(regions),
        family=family,
        variant=variant,
        component_manifest=component_manifest,
        p2p=p2p,
        gated=cast("bool", wire.get("gated", False)),
    )


def resolver_index_from_wire(wire: Mapping[str, object]) -> ResolverIndex:
    if wire.get("nextCursor") is not None:
        raise ResolverIndexError("partial resolver index requires HTTP cursor pagination")
    version = wire.get("dinksterResolver")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ResolverIndexError("'dinksterResolver' must be an integer format version")
    if version != RESOLVER_INDEX_VERSION:
        raise ResolverIndexError(
            f"unsupported resolver index version {version}; supported version is "
            f"{RESOLVER_INDEX_VERSION}"
        )
    entries_raw = wire.get("entries")
    if not isinstance(entries_raw, Sequence) or isinstance(entries_raw, (str, bytes)):
        raise ResolverIndexError("'entries' must be a list")
    entries_wire = cast("Sequence[object]", entries_raw)
    if len(entries_wire) > RESOLVER_INDEX_MAX_ENTRIES:
        raise ResolverIndexError(
            f"resolver index accepts at most {RESOLVER_INDEX_MAX_ENTRIES} entries"
        )
    entries: list[ResolverIndexEntry] = []
    digests: set[str] = set()
    for position, raw_entry in enumerate(entries_wire):
        if not isinstance(raw_entry, Mapping):
            raise ResolverIndexError(f"entries[{position}] must be an object")
        entry = _entry_from_wire(cast("Mapping[str, object]", raw_entry), position)
        if entry.digest in digests:
            raise ResolverIndexError(f"entries[{position}] duplicates digest {entry.digest}")
        digests.add(entry.digest)
        entries.append(entry)
    name = _string(wire, "name")
    description = _string(wire, "description")
    homepage = _string(wire, "homepage")
    if homepage:
        require_https_url(homepage, "homepage")
    updated = _string(wire, "updated")
    if updated:
        if _RFC3339_RE.fullmatch(updated) is None:
            raise ResolverIndexError("'updated' must be an RFC 3339 timestamp")
        try:
            timestamp = updated[:-1] + "+00:00" if updated.endswith("Z") else updated
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise ResolverIndexError("'updated' must be an RFC 3339 timestamp") from exc
        if parsed.tzinfo is None:
            raise ResolverIndexError("'updated' must include a timezone")
    return ResolverIndex(
        entries=tuple(entries),
        name=name,
        description=description,
        homepage=homepage,
        updated=updated,
    )


def _reject_json_constant(value: str) -> object:
    raise ResolverIndexError(f"invalid JSON number {value}")


def parse_resolver_index(data: bytes | str) -> ResolverIndex:
    return resolver_index_from_wire(decode_resolver_index_document(data))


def decode_resolver_index_document(data: bytes | str) -> Mapping[str, object]:
    """Decode bounded JSON without discarding pagination or future fields."""
    if isinstance(data, str):
        try:
            size = len(data.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ResolverIndexError("resolver index must be valid UTF-8") from exc
        text = data
    else:
        size = len(data)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResolverIndexError("resolver index must be valid UTF-8") from exc
    if size > RESOLVER_INDEX_MAX_BYTES:
        raise ResolverIndexError(f"resolver index exceeds {RESOLVER_INDEX_MAX_BYTES} bytes")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ResolverIndexError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        decoded: object = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ResolverIndexError(
            f"invalid resolver index JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise ResolverIndexError(f"invalid resolver index JSON: {exc}") from exc
    except RecursionError as exc:
        raise ResolverIndexError("resolver index JSON is nested too deeply") from exc
    if not isinstance(decoded, Mapping):
        raise ResolverIndexError("resolver index document must be an object")
    return cast("Mapping[str, object]", decoded)
