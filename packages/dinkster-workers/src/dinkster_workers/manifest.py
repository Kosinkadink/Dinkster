"""Pack manifests: what a pack is, declared as data (DESIGN 3.6).

A pack ships a ``dinkster-pack.toml`` next to its code:

    [pack]
    name = "my-pack"
    namespaces = ["my-pack"]
    requires = ["numpy>=1.26"]

    [pack.entry]
    nodes = "my_pack:NODES"
    types = "my_pack:register_types"

    [pack.extension]
    schema = "my_pack.extensions:schema_contributions"
    privileges = ["schema"]
    capabilities = []

``entry.nodes`` names a sequence of Node classes; ``entry.types`` (optional)
names a callable taking a TypeRegistry; ``entry.reservations`` (optional)
names a ReservationPlanner - pack policy for what its invocations
materialize, consulted by worker wrappers for memory admission (DESIGN
3.10); ``entry.consumers`` (optional) names a callable returning a
``Mapping[str, Shedder]`` - the pack's governed memory consumers (a
resident pool), announced through the hello handshake and served to the
parent's governor via the memory relay when the pack runs isolated.
``entry.telemetry`` (optional) names a zero-arg callable returning a
``Mapping[str, MeasuredMemory]`` - the pack's measured device memory
(free/total per residency class, in the worker's own device namespace),
reported across the boundary in hello/memoryReport frames for the
parent's observability. Informational only, never admission.
Optional ``[pack.presentation]`` declares badge data,
``[[pack.blueprints]]`` entries declare starter workflow documents, and
``[[pack.assets]]`` entries declare distributable assets (models the
pack's nodes need, digest-pinned, packaged and/or remote) - all
advisory, warn-and-drop validated, never identity. Model packs use strict
``[[pack.vision-providers]]`` entries to associate those assets and runtime
requirements with stable schemas they implement through ``[pack] executes``.
Loading a manifest reads TOML only - no pack code is imported until
a worker host resolves the entries inside its own process (hazard H5:
loading has no host side effects).

The additive ``[pack.extension]`` table deliberately mirrors the existing
``[pack.entry]`` key-to-``module:attr`` shape while separating schema, server,
inference, frontend, and training imports. Its ``privileges`` and closed
``capabilities`` lists are authorization/audit declarations only in S0-A.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import tomllib
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar, cast

from dinkster_assets import (
    AssetComponentManifest,
    AssetError,
    AssetNeed,
    AssetSource,
    DeclaredAsset,
    PackagedSource,
    RemoteSource,
)
from dinkster_protocol import (
    EXTENSION_CAPABILITIES,
    EXTENSION_SCOPES,
    ExtensionDeclaration,
    ExtensionEntryPoints,
)
from dinkster_protocol.frontend_modules import FrontendModule
from dinkster_protocol.pack_surfaces import pack_surfaces_from_wire
from dinkster_schema import (
    ComfyAliasRegistry,
    ComfyGroupRegistry,
    canonical_name,
    claim_covers,
    claims_conflict,
    comfy_alias_registry_from_wire,
    comfy_group_registry_from_wire,
    core_logger,
    validate_name,
)
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


class ManifestError(Exception):
    """A pack manifest is missing, unreadable, or malformed."""


PACK_HOST_CONTRACT = "dinkster-pack-host/1"
PACK_AUTHOR_API_CONTRACT = "dinkster-api/v1"
PACK_INFERENCE_CONTRACT = "dinkster-inference/1"
_PACK_REGISTRY_CONTRIBUTION_SURFACES: Mapping[str, str] = MappingProxyType(
    {
        "dinkster-model-families": "inference.families",
        "dinkster-samplers": "inference.samplers",
        "dinkster-schedulers": "inference.schedulers",
    }
)


_ABBR_MAX = 8
_MARK_MAX_CODEPOINTS = 8
_COLOR_CHARS = frozenset("0123456789abcdefABCDEF")

# The pack-icon contract agreed with the frontend: exactly 64x64 so the
# renderer has zero aspect/fit logic (and a future atlas has uniform
# cells), static PNG/WebP only (createImageBitmap decodes both natively;
# animation would silently render frame 0), bounded bytes.
ICON_SIZE = 64
ICON_MAX_BYTES = 64 * 1024

# The pack-blueprint contract agreed with the frontend: blueprints are
# plain-data workflow documents (graphs, never assets), so 1 MiB per file
# is generous. Bytes are held in memory at validation to guarantee
# digest/byte coherence - hence the additional per-pack total budget, or
# one pack could hold unbounded memory through many just-under-cap files.
BLUEPRINT_MAX_BYTES = 1024 * 1024
BLUEPRINT_PACK_MAX_BYTES = 16 * 1024 * 1024

# The pack-template contract (roadmap "templates surface"): a template is
# a complete starter workflow document - same plain-data format and the
# same caps as blueprints, because both are graphs, never assets. Asset
# requirements ride as REFERENCES to the pack's own [[pack.assets]] ids;
# a template index over descriptors therefore stays kilobytes while the
# models it needs stay digest-pinned needs behind consent.
TEMPLATE_MAX_BYTES = BLUEPRINT_MAX_BYTES
TEMPLATE_PACK_MAX_BYTES = BLUEPRINT_PACK_MAX_BYTES

DOC_PAGE_MAX_BYTES = 256 * 1024
DOC_IMAGE_MAX_BYTES = 2 * 1024 * 1024
DOC_VIDEO_MAX_BYTES = 16 * 1024 * 1024
DOC_PACK_MAX_BYTES = 32 * 1024 * 1024
DOC_LOCALES = frozenset({"en", "zh"})
LOCALE_CATALOG_MAX_BYTES = 1024 * 1024
LOCALE_CATALOG_PACK_MAX_BYTES = 16 * 1024 * 1024
DOC_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".webm": "video/webm",
    ".webp": "image/webp",
}
_DOC_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
_DOC_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
_DOC_FENCE_RE = re.compile(r"^(`{3,})([^`]*)$")
_DOC_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_DOC_FRONT_MATTER_RE = re.compile(
    rb"\A\+\+\+\r?\n(.*?)^\+\+\+(?:\r?\n|\Z)", re.MULTILINE | re.DOTALL
)
_LOCALE_TAG_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{1,8})*$")

COMFY_ALIASES_FILENAME = "comfy-aliases.json"
COMFY_GROUPS_FILENAME = "comfy-groups.json"
COMFY_ALIASES_MAX_BYTES = 4 * 1024 * 1024
COMFY_ALIASES_MAX_DEPTH = 64
COMFY_ALIASES_MAX_ITEMS = 100_000


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number: {value}")
    return result


def _validate_json_budget(value: object) -> None:
    count = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > COMFY_ALIASES_MAX_ITEMS:
            raise ValueError(f"JSON item count exceeds {COMFY_ALIASES_MAX_ITEMS}")
        if depth > COMFY_ALIASES_MAX_DEPTH:
            raise ValueError(f"JSON nesting depth exceeds {COMFY_ALIASES_MAX_DEPTH}")
        if isinstance(item, dict):
            obj = cast("dict[object, object]", item)
            stack.extend((child, depth + 1) for child in obj.values())
        elif isinstance(item, list):
            array = cast("list[object]", item)
            stack.extend((child, depth + 1) for child in array)


_RegistryT = TypeVar("_RegistryT")


def _load_comfy_registry(
    manifest_path: Path,
    filename: str,
    kind: str,
    decode: Callable[[object], _RegistryT],
) -> _RegistryT | None:
    path = manifest_path.with_name(filename)
    if not path.exists() and not path.is_symlink():
        return None
    try:
        if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
            raise ManifestError(f"{path}: must not escape the pack directory (symlink?)")
        with path.open("rb") as file:
            data = file.read(COMFY_ALIASES_MAX_BYTES + 1)
    except ManifestError:
        raise
    except OSError as exc:
        raise ManifestError(f"{path}: unreadable: {exc}") from exc
    if len(data) > COMFY_ALIASES_MAX_BYTES:
        raise ManifestError(f"{path}: exceeds the {COMFY_ALIASES_MAX_BYTES}-byte cap")
    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
        _validate_json_budget(document)
        return decode(document)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise ManifestError(f"{path}: invalid {kind} registry: {exc}") from None


def load_comfy_aliases(manifest_path: Path) -> ComfyAliasRegistry | None:
    """Load the strict adjacent alias registry without importing pack code."""
    return _load_comfy_registry(
        manifest_path,
        COMFY_ALIASES_FILENAME,
        "comfy alias",
        comfy_alias_registry_from_wire,
    )


def load_comfy_groups(manifest_path: Path) -> ComfyGroupRegistry | None:
    """Load the strict adjacent group registry without importing pack code."""
    return _load_comfy_registry(
        manifest_path,
        COMFY_GROUPS_FILENAME,
        "comfy group",
        comfy_group_registry_from_wire,
    )


@dataclass(frozen=True)
class PackIcon:
    """A validated pack icon, from ``[pack.presentation] icon``.

    ``digest`` (``sha256:<hex>`` over the file bytes) is the immutability
    contract: the bytes served for a digest never change - changing the
    icon means a new digest on the wire, which is what licenses clients
    to cache decoded bitmaps forever. ``media_type`` is sniffed from the
    bytes, never trusted from the file extension. ``data`` carries the
    validated bytes themselves (the cap is 64 KiB): whoever serves the
    icon serves exactly what was digested, so the file changing on disk
    after load can never make the wire lie."""

    path: Path
    media_type: str
    digest: str
    data: bytes = dataclass_field(repr=False, default=b"")


def _png_meta(data: bytes) -> tuple[int, int, bool] | None:
    """(width, height, animated) for PNG bytes, else None. A bounded chunk
    walk: dimensions from IHDR, animation from an acTL chunk appearing
    before the first IDAT (the APNG marker) - never a raw byte search,
    which could false-positive inside compressed data."""
    if len(data) < 33 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    if data[12:16] != b"IHDR":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    animated = False
    offset = 8
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        if kind == b"acTL":
            animated = True
            break
        if kind == b"IDAT":
            break
        offset += 12 + length  # length + type + data + crc
    return width, height, animated


def _webp_meta(data: bytes) -> tuple[int, int, bool] | None:
    """(width, height, animated) for WebP bytes (VP8X/VP8L/VP8 forms),
    else None."""
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    fourcc = data[12:16]
    if fourcc == b"VP8X":
        animated = bool(data[20] & 0x02)
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height, animated
    if fourcc == b"VP8L":
        if data[20] != 0x2F:
            return None
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height, False
    if fourcc == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height, False
    return None


def validate_pack_icon(manifest_path: Path, declared: str) -> tuple[PackIcon | None, str | None]:
    """Validate one declared icon against the contract; ``(icon, None)``
    or ``(None, reason)``. Shared by the warn-and-drop manifest path and
    by doctor, so the loader and the linter can never disagree."""
    rel = Path(declared)
    if rel.is_absolute() or ".." in rel.parts:
        return None, "must be a relative path inside the pack directory"
    path = manifest_path.parent / rel
    try:
        # resolve() follows symlinks, so a link inside the pack pointing
        # outside it fails containment just like a literal `..` would.
        if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
            return None, "must not escape the pack directory (symlink?)"
        source_size = path.stat().st_size
        if source_size > ICON_MAX_BYTES:
            return None, f"is {source_size} bytes; the cap is {ICON_MAX_BYTES}"
        data = path.read_bytes()
    except OSError as exc:
        return None, f"file unreadable: {exc}"
    if len(data) > ICON_MAX_BYTES:
        return None, f"is {len(data)} bytes; the cap is {ICON_MAX_BYTES}"
    if (meta := _png_meta(data)) is not None:
        media_type = "image/png"
    elif (meta := _webp_meta(data)) is not None:
        media_type = "image/webp"
    else:
        return None, "must be PNG or WebP (sniffed from the bytes, not the extension)"
    width, height, animated = meta
    if animated:
        return None, "must be static (animated icons would render as frame 0)"
    if (width, height) != (ICON_SIZE, ICON_SIZE):
        return None, f"must be exactly {ICON_SIZE}x{ICON_SIZE}, got {width}x{height}"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    return PackIcon(path=path, media_type=media_type, digest=digest, data=data), None


def _read_document_file(
    manifest_path: Path,
    declared: object,
    *,
    max_bytes: int,
    cap_reason: str,
) -> tuple[bytes | None, Path | None, str | None]:
    """Read and validate a pack-shipped workflow document file: the
    backend-ownable checks only (path containment incl. symlink escape,
    byte cap, UTF-8, well-formed JSON with an object at the top level) -
    never document-format semantics, which are frontend-owned. Shared by
    blueprints and templates so the two surfaces can never disagree.
    ``(data, path, None)`` on success, ``(None, None, reason)`` on any
    failure."""
    if not isinstance(declared, str) or not declared:
        return None, None, "must declare a non-empty string 'file'"
    rel = Path(declared)
    if rel.is_absolute() or ".." in rel.parts:
        return None, None, f"file {declared!r} must be a relative path inside the pack directory"
    path = manifest_path.parent / rel
    try:
        # resolve() follows symlinks, so a link inside the pack pointing
        # outside it fails containment just like a literal `..` would.
        if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
            return None, None, f"file {declared!r} must not escape the pack directory (symlink?)"
        source_size = path.stat().st_size
        if source_size > max_bytes:
            return (
                None,
                None,
                (
                    f"file {declared!r} is {source_size} bytes; "
                    f"the cap is {max_bytes} ({cap_reason})"
                ),
            )
        data = path.read_bytes()
    except OSError as exc:
        return None, None, f"file {declared!r} unreadable: {exc}"
    if len(data) > max_bytes:
        return (
            None,
            None,
            (f"file {declared!r} is {len(data)} bytes; the cap is {max_bytes} ({cap_reason})"),
        )
    try:
        document = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError:
        return None, None, f"file {declared!r} is not valid UTF-8"
    except json.JSONDecodeError as exc:
        return None, None, f"file {declared!r} is not valid JSON: {exc}"
    if not isinstance(document, dict):
        return None, None, f"file {declared!r} must contain a JSON object at the top level"
    return data, path, None


@dataclass(frozen=True)
class PackBlueprint:
    """A validated pack blueprint, from one ``[[pack.blueprints]]`` entry.

    A blueprint is plain DATA - a starter workflow document (one subgraph
    with a content-level boundary) shipped verbatim by the pack, in exactly
    the format the frontend authors and saves. The backend never interprets
    document semantics (the frontend owns the loader); it validates only
    what it can own: well-formed JSON, size caps, path containment, and
    the digest. ``digest`` (``sha256:<hex>`` over the file bytes) is the
    immutability contract shared with icons: the bytes served for a digest
    never change - a changed blueprint is a new digest on the wire - which
    licenses clients to cache decoded documents forever. ``data`` carries
    the validated bytes so what is served is exactly what was digested,
    no matter what happens to the file on disk after load."""

    id: str
    name: str
    digest: str
    path: Path
    description: str = ""
    tags: tuple[str, ...] = ()
    boundary_inputs: tuple[str, ...] = ()
    """Author-declared boundary input type ids, passed through VERBATIM to
    the wire descriptor. Search-affordance hints for frontends (port
    filters over blueprint boundaries without fetching bodies); the
    backend never derives them from the document or checks them against
    it - declaration-vs-document cross-checks belong to a frontend-owned
    rule set, not here."""
    boundary_outputs: tuple[str, ...] = ()
    """Author-declared boundary output type ids; same verbatim contract."""
    data: bytes = dataclass_field(repr=False, default=b"")


def validate_pack_blueprint(
    manifest_path: Path, entry: object
) -> tuple[PackBlueprint | None, str | None]:
    """Validate one ``[[pack.blueprints]]`` entry against the contract;
    ``(blueprint, None)`` or ``(None, reason)``. Shared by the
    warn-and-drop manifest path and by doctor, so the loader and the
    linter can never disagree. Backend-ownable checks only: id/name/tags
    shape, path containment, byte cap, UTF-8, well-formed JSON with an
    object at the top level, digest capture - never document-format
    semantics, which are frontend-owned."""
    if not isinstance(entry, dict):
        return None, "must be a table (a [[pack.blueprints]] entry)"
    table = cast("dict[str, object]", entry)

    blueprint_id = table.get("id")
    if not isinstance(blueprint_id, str) or not blueprint_id:
        return None, "must declare a non-empty string 'id'"
    id_problem = validate_name(blueprint_id)
    if id_problem is not None:
        return None, f"id {blueprint_id!r} {id_problem}"

    name = table.get("name")
    if not isinstance(name, str) or not name:
        return None, "must declare a non-empty string 'name'"

    description = table.get("description", "")
    if not isinstance(description, str):
        return None, "'description' must be a string"

    tags_raw = table.get("tags", [])
    if not isinstance(tags_raw, list) or not all(
        isinstance(tag, str) and tag for tag in cast("list[object]", tags_raw)
    ):
        return None, "'tags' must be a list of non-empty strings"
    tags = tuple(cast("list[str]", tags_raw))

    def string_list(key: str) -> tuple[str, ...] | None:
        raw = table.get(key, [])
        if not isinstance(raw, list) or not all(
            isinstance(item, str) and item for item in cast("list[object]", raw)
        ):
            return None
        return tuple(cast("list[str]", raw))

    boundary_inputs = string_list("boundary_inputs")
    if boundary_inputs is None:
        return None, "'boundary_inputs' must be a list of non-empty strings"
    boundary_outputs = string_list("boundary_outputs")
    if boundary_outputs is None:
        return None, "'boundary_outputs' must be a list of non-empty strings"

    data, path, problem = _read_document_file(
        manifest_path,
        table.get("file"),
        max_bytes=BLUEPRINT_MAX_BYTES,
        cap_reason="blueprints are graphs, not assets",
    )
    if data is None or path is None:
        return None, problem

    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    return (
        PackBlueprint(
            id=blueprint_id,
            name=name,
            digest=digest,
            path=path,
            description=description,
            tags=tags,
            boundary_inputs=boundary_inputs,
            boundary_outputs=boundary_outputs,
            data=data,
        ),
        None,
    )


def _parse_blueprints(raw: object, manifest_path: Path) -> tuple[PackBlueprint, ...]:
    """Parse ``[[pack.blueprints]]`` with warn-and-drop PER BLUEPRINT: a
    malformed entry is a logged diagnostic that drops that blueprint only -
    the pack and its sibling blueprints always survive (same philosophy as
    presentation). Duplicate ids keep the first declaration; entries past
    the per-pack total byte budget drop with a warning."""
    if raw is None:
        return ()
    log = core_logger("workers")
    if not isinstance(raw, list):
        log.warning(
            "%s: [[pack.blueprints]] must be an array of tables; ignoring it",
            manifest_path,
        )
        return ()
    blueprints: list[PackBlueprint] = []
    seen: set[str] = set()
    total = 0
    for index, entry in enumerate(cast("list[object]", raw)):
        blueprint, problem = validate_pack_blueprint(manifest_path, entry)
        if blueprint is None:
            log.warning(
                "%s: [[pack.blueprints]] entry %d %s; dropping it",
                manifest_path,
                index,
                problem,
            )
            continue
        if blueprint.id in seen:
            log.warning(
                "%s: [[pack.blueprints]] entry %d duplicates id %r; dropping it "
                "(the first declaration wins)",
                manifest_path,
                index,
                blueprint.id,
            )
            continue
        if total + len(blueprint.data) > BLUEPRINT_PACK_MAX_BYTES:
            log.warning(
                "%s: [[pack.blueprints]] entry %d (%r) would exceed the per-pack "
                "blueprint budget of %d bytes; dropping it",
                manifest_path,
                index,
                blueprint.id,
                BLUEPRINT_PACK_MAX_BYTES,
            )
            continue
        seen.add(blueprint.id)
        total += len(blueprint.data)
        blueprints.append(blueprint)
    return tuple(blueprints)


def load_pack_blueprints(path: Path | str) -> tuple[PackBlueprint, ...]:
    """Read ONLY ``[[pack.blueprints]]`` from a ``dinkster-pack.toml``.

    The blueprint on-ramp for packs that are not (yet) Dinkster packs,
    riding the same presentation-only manifest as icons: a legacy ComfyUI
    pack can ship starter workflows without being a loadable pack
    manifest. Everything about it is advisory: a missing file is (),
    and malformed TOML or entries warn-and-drop exactly like manifest
    loading - blueprints can never break composition."""
    manifest_path = Path(path)
    try:
        with open(manifest_path, "rb") as fh:
            document = tomllib.load(fh)
    except FileNotFoundError:
        return ()
    except (OSError, tomllib.TOMLDecodeError) as exc:
        core_logger("workers").warning(
            "%s: unreadable pack manifest for blueprints (%s); ignoring it",
            manifest_path,
            exc,
        )
        return ()
    pack = document.get("pack")
    if not isinstance(pack, dict):
        return ()
    raw = cast("dict[str, object]", pack).get("blueprints")
    return _parse_blueprints(raw, manifest_path)


@dataclass(frozen=True)
class PackTemplate:
    """A validated pack template, from one ``[[pack.templates]]`` entry.

    A template is a COMPLETE starter workflow document shipped verbatim
    by the pack - same plain-data format, caps, and non-interpretation
    boundary as blueprints (the backend validates bytes, never document
    semantics). The difference is scope and requirements: a blueprint is
    a reusable subgraph for the palette; a template is a whole document a
    user opens to start from, and it may REQUIRE assets (models) to run.
    Those requirements ride as ``assets`` - references to the pack's own
    ``[[pack.assets]]`` ids - so the template index stays kilobytes while
    the models stay digest-pinned declarations behind acquisition
    consent. Never embedded bytes: that is the ComfyUI 10GB-pip-package
    mistake this contract exists to avoid. ``digest`` (``sha256:<hex>``
    over the file bytes) is the same immutability contract as icons and
    blueprints: bytes served for a digest never change, licensing clients
    to cache decoded documents forever."""

    id: str
    name: str
    digest: str
    path: Path
    description: str = ""
    tags: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()
    """Pack-local ``[[pack.assets]]`` ids this template requires.
    Validated to exist among the pack's surviving declarations at parse
    time (a dangling reference drops the template - its acquisition plan
    could never be constructed); passed through as ids on the wire, where
    clients join them against the pack's asset descriptors."""
    data: bytes = dataclass_field(repr=False, default=b"")


@dataclass(frozen=True)
class PackDocAsset:
    """One validated pack-relative documentation asset."""

    source: str
    digest: str
    media_type: str
    data: bytes = dataclass_field(repr=False, default=b"")


@dataclass(frozen=True)
class PackDocPage:
    """One validated localized node or guide Markdown page."""

    kind: str
    id: str
    locale: str
    title: str
    summary: str
    digest: str
    assets: tuple[PackDocAsset, ...] = ()
    schema_version: int | None = None
    order: int | None = None
    tags: tuple[str, ...] = ()
    guide_kind: str | None = None
    node_references: tuple[str, ...] = ()
    data: bytes = dataclass_field(repr=False, default=b"")


@dataclass(frozen=True)
class PackDocs:
    """Validated content discovered beneath an explicit ``[pack.docs]``."""

    default_locale: str
    pages: tuple[PackDocPage, ...] = ()
    assets: tuple[PackDocAsset, ...] = ()
    validation_problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackLocaleCatalog:
    """One validated pack translation catalog, preserved byte-for-byte."""

    locale: str
    digest: str
    node_references: tuple[str, ...] = ()
    data: bytes = dataclass_field(repr=False, default=b"")


def _catalog_text(value: object, where: str) -> str | None:
    if not isinstance(value, str) or not value:
        return f"{where} must be a non-empty string"
    return None


def _catalog_node_port_map(value: object, where: str) -> str | None:
    if not isinstance(value, dict):
        return f"{where} must be an object"
    for port_id, raw in cast("dict[str, object]", value).items():
        if validate_name(port_id) is not None:
            return f"{where} keys must be valid port ids"
        if not isinstance(raw, dict):
            return f"{where}.{port_id} must be an object"
        record = cast("dict[str, object]", raw)
        unknown = set(record) - {"displayName", "doc"}
        if unknown:
            return f"{where}.{port_id} has unknown fields: {', '.join(sorted(unknown))}"
        if not record:
            return f"{where}.{port_id} must translate displayName or doc"
        for field, text in record.items():
            if problem := _catalog_text(text, f"{where}.{port_id}.{field}"):
                return problem
    return None


def _catalog_combo_map(value: object, where: str) -> str | None:
    if not isinstance(value, dict):
        return f"{where} must be an object"
    for input_id, raw in cast("dict[str, object]", value).items():
        if validate_name(input_id) is not None:
            return f"{where} keys must be valid input ids"
        if not isinstance(raw, dict) or not raw:
            return f"{where}.{input_id} must be a non-empty object"
        for option, text in cast("dict[str, object]", raw).items():
            if not option:
                return f"{where}.{input_id} keys must be non-empty strings"
            if problem := _catalog_text(text, f"{where}.{input_id}.{option}"):
                return problem
    return None


def _validate_locale_catalog(
    document: object,
    path: Path,
    namespaces: tuple[str, ...],
    blueprint_ids: set[str],
    guide_ids: set[str],
) -> tuple[tuple[str, ...] | None, str | None]:
    if not isinstance(document, dict):
        return None, f"{path} root must be an object"
    table = cast("dict[str, object]", document)
    unknown = set(table) - {"nodes", "blueprints", "guides", "searchTerms"}
    if unknown:
        return None, f"{path} has unknown fields: {', '.join(sorted(unknown))}"
    node_references: set[str] = set()
    nodes = table.get("nodes", {})
    if not isinstance(nodes, dict):
        return None, f"{path} nodes must be an object"
    for node_type, raw in cast("dict[str, object]", nodes).items():
        if validate_name(node_type) is not None:
            return None, f"{path} nodes keys must be valid node types"
        if not any(claim_covers(claim, node_type) for claim in namespaces):
            return None, f"{path} node {node_type!r} is not owned by this pack"
        if not isinstance(raw, dict) or not raw:
            return None, f"{path} nodes.{node_type} must be a non-empty object"
        record = cast("dict[str, object]", raw)
        unknown = set(record) - {"displayName", "description", "inputs", "outputs", "combos"}
        if unknown:
            return (
                None,
                f"{path} nodes.{node_type} has unknown fields: {', '.join(sorted(unknown))}",
            )
        for field in ("displayName", "description"):
            if field in record and (
                problem := _catalog_text(record[field], f"{path} nodes.{node_type}.{field}")
            ):
                return None, problem
        for field in ("inputs", "outputs"):
            if field in record and (
                problem := _catalog_node_port_map(
                    record[field], f"{path} nodes.{node_type}.{field}"
                )
            ):
                return None, problem
        if "combos" in record and (
            problem := _catalog_combo_map(record["combos"], f"{path} nodes.{node_type}.combos")
        ):
            return None, problem
        node_references.add(node_type)

    blueprints = table.get("blueprints", {})
    if not isinstance(blueprints, dict):
        return None, f"{path} blueprints must be an object"
    for blueprint_id, raw in cast("dict[str, object]", blueprints).items():
        if blueprint_id not in blueprint_ids:
            return None, f"{path} references unknown blueprint {blueprint_id!r}"
        if not isinstance(raw, dict) or not raw:
            return None, f"{path} blueprints.{blueprint_id} must be a non-empty object"
        record = cast("dict[str, object]", raw)
        unknown = set(record) - {"name", "description"}
        if unknown:
            return (
                None,
                f"{path} blueprints.{blueprint_id} has unknown fields: "
                f"{', '.join(sorted(unknown))}",
            )
        for field, text in record.items():
            if problem := _catalog_text(text, f"{path} blueprints.{blueprint_id}.{field}"):
                return None, problem

    guides = table.get("guides", {})
    if not isinstance(guides, dict):
        return None, f"{path} guides must be an object"
    for guide_id, raw in cast("dict[str, object]", guides).items():
        if guide_id not in guide_ids:
            return None, f"{path} references unknown guide {guide_id!r}"
        if not isinstance(raw, dict) or set(cast("dict[str, object]", raw)) != {"title"}:
            return None, f"{path} guides.{guide_id} must contain only title"
        if problem := _catalog_text(
            cast("dict[str, object]", raw)["title"], f"{path} guides.{guide_id}.title"
        ):
            return None, problem

    search_terms = table.get("searchTerms", {})
    if not isinstance(search_terms, dict):
        return None, f"{path} searchTerms must be an object"
    for node_type, raw in cast("dict[str, object]", search_terms).items():
        if validate_name(node_type) is not None:
            return None, f"{path} searchTerms keys must be valid node types"
        if not any(claim_covers(claim, node_type) for claim in namespaces):
            return None, f"{path} searchTerms node {node_type!r} is not owned by this pack"
        if not isinstance(raw, list) or not raw:
            return None, f"{path} searchTerms.{node_type} must be a non-empty array"
        for index, text in enumerate(cast("list[object]", raw)):
            if problem := _catalog_text(text, f"{path} searchTerms.{node_type}[{index}]"):
                return None, problem
        node_references.add(node_type)
    return tuple(sorted(node_references)), None


def _parse_locale_catalogs(
    manifest_path: Path,
    namespaces: tuple[str, ...],
    blueprint_ids: set[str],
    guide_ids: set[str],
) -> tuple[tuple[PackLocaleCatalog, ...], tuple[str, ...]]:
    root = manifest_path.parent / "locales"
    if not root.exists() and not root.is_symlink():
        return (), ()
    log = core_logger("workers")
    if root.is_symlink():
        problem = f"{root} must not be a symlink"
        log.warning("%s: %s; dropping locale catalogs", manifest_path, problem)
        return (), (problem,)
    if not root.is_dir():
        problem = f"{root} must be a directory"
        log.warning("%s: %s; dropping locale catalogs", manifest_path, problem)
        return (), (problem,)
    problems: list[str] = []
    catalogs: list[PackLocaleCatalog] = []
    total = 0
    try:
        entries = sorted(root.iterdir())
    except OSError as exc:
        problem = f"{root} is unreadable: {exc}"
        log.warning("%s: %s; dropping locale catalogs", manifest_path, problem)
        return (), (problem,)
    locale_counts: dict[str, int] = {}
    for path in entries:
        if path.suffix.lower() == ".json":
            locale = path.stem.lower()
            locale_counts[locale] = locale_counts.get(locale, 0) + 1
    collisions = {locale for locale, count in locale_counts.items() if count > 1}
    for path in entries:
        if path.is_symlink():
            problems.append(f"{path} must not be a symlink")
            continue
        if not path.is_file() or path.suffix != ".json":
            problems.append(f"{path} must be a locale JSON file directly beneath {root}")
            continue
        locale = path.stem.lower()
        if locale in collisions:
            problems.append(f"{path} collides with another catalog for locale {locale!r}")
            continue
        if path.stem != locale or _LOCALE_TAG_RE.fullmatch(locale) is None:
            problems.append(f"{path} filename must be a canonical lowercase locale tag")
            continue
        try:
            source_size = path.stat().st_size
        except OSError as exc:
            problems.append(f"{path} is unreadable: {exc}")
            continue
        total += source_size
        if source_size > LOCALE_CATALOG_MAX_BYTES:
            problems.append(f"{path} exceeds the {LOCALE_CATALOG_MAX_BYTES}-byte cap")
            continue
        if total > LOCALE_CATALOG_PACK_MAX_BYTES:
            problems.append(f"{path} exceeds the {LOCALE_CATALOG_PACK_MAX_BYTES}-byte pack budget")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            problems.append(f"{path} is unreadable: {exc}")
            continue
        total += len(data) - source_size
        if len(data) > LOCALE_CATALOG_MAX_BYTES:
            problems.append(f"{path} exceeds the {LOCALE_CATALOG_MAX_BYTES}-byte cap")
            continue
        if total > LOCALE_CATALOG_PACK_MAX_BYTES:
            problems.append(f"{path} exceeds the {LOCALE_CATALOG_PACK_MAX_BYTES}-byte pack budget")
            continue
        try:
            document = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
                parse_float=_finite_json_float,
            )
            _validate_json_budget(document)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            problems.append(f"{path} is invalid JSON: {exc}")
            continue
        node_references, problem = _validate_locale_catalog(
            document, path, namespaces, blueprint_ids, guide_ids
        )
        if node_references is None:
            assert problem is not None
            problems.append(problem)
            continue
        catalogs.append(
            PackLocaleCatalog(
                locale=locale,
                digest="sha256:" + hashlib.sha256(data).hexdigest(),
                node_references=node_references,
                data=data,
            )
        )
    for problem in problems:
        log.warning("%s: %s; dropping locale catalog", manifest_path, problem)
    return tuple(catalogs), tuple(problems)


def _docs_relative_root(manifest_path: Path, declared: object) -> tuple[Path | None, str | None]:
    if not isinstance(declared, str) or not declared:
        return None, "'dir' must be a non-empty string"
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        return None, f"dir {declared!r} must be relative to the pack directory"
    root = manifest_path.parent / relative
    if root.is_symlink():
        return None, f"dir {declared!r} must not be a symlink"
    try:
        if not root.resolve().is_relative_to(manifest_path.parent.resolve()):
            return None, f"dir {declared!r} must not escape the pack directory"
    except OSError as exc:
        return None, f"dir {declared!r} cannot be resolved: {exc}"
    if not root.is_dir():
        return None, f"dir {declared!r} is not a directory"
    symlink = next((path for path in root.rglob("*") if path.is_symlink()), None)
    if symlink is not None:
        return None, f"dir {declared!r} contains symlink {symlink.relative_to(root)}"
    return root, None


def _read_doc_file(path: Path, root: Path, max_bytes: int) -> tuple[bytes | None, str | None]:
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            return None, f"{path} escapes the documentation directory through a symlink"
        size = path.stat().st_size
        if size > max_bytes:
            return None, f"{path} is {size} bytes; the cap is {max_bytes}"
        data = path.read_bytes()
    except OSError as exc:
        return None, f"{path} is unreadable: {exc}"
    if len(data) > max_bytes:
        return None, f"{path} is {len(data)} bytes; the cap is {max_bytes}"
    return data, None


def _doc_front_matter(
    data: bytes, path: Path
) -> tuple[dict[str, object] | None, bytes | None, str | None]:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return None, None, f"{path} is not valid UTF-8"
    if not data.startswith((b"+++\n", b"+++\r\n")):
        return None, None, f"{path} must begin with +++ front matter"
    match = _DOC_FRONT_MATTER_RE.match(data)
    if match is None:
        return None, None, f"{path} has unterminated +++ front matter"
    try:
        values = tomllib.loads(match.group(1).decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return None, None, f"{path} has invalid front matter: {exc}"
    unknown = set(values) - {"title", "summary", "schema_version"}
    if unknown:
        return None, None, f"{path} has unknown front matter fields: {', '.join(sorted(unknown))}"
    title = values.get("title")
    summary = values.get("summary")
    schema_version = values.get("schema_version")
    if not isinstance(title, str) or not title:
        return None, None, f"{path} front matter title must be a non-empty string"
    if not isinstance(summary, str) or not summary:
        return None, None, f"{path} front matter summary must be a non-empty string"
    if schema_version is not None and (type(schema_version) is not int or schema_version < 1):
        return None, None, f"{path} front matter schema_version must be a positive integer"
    return values, data[match.end() :], None


def _doc_body_references(
    body: str,
    path: Path,
    assets: Mapping[str, PackDocAsset],
    blueprint_ids: set[str],
    template_ids: set[str],
) -> tuple[tuple[str, ...] | None, tuple[str, ...], str | None]:
    prose: list[str] = []
    dinkster_blocks: dict[str, list[str]] = {
        "dinkster-example": [],
        "dinkster-media": [],
        "dinkster-node": [],
    }
    fence_length = 0
    fence_kind = ""
    fenced_lines: list[str] = []
    for line in body.splitlines():
        fence = _DOC_FENCE_RE.fullmatch(line)
        if fence_length == 0:
            if fence is None:
                prose.append(line)
                continue
            fence_length = len(fence.group(1))
            fence_kind = fence.group(2).strip()
        elif (
            fence is not None and len(fence.group(1)) >= fence_length and not fence.group(2).strip()
        ):
            if fence_kind in dinkster_blocks:
                dinkster_blocks[fence_kind].append("\n".join(fenced_lines))
            fence_length = 0
            fence_kind = ""
            fenced_lines = []
        elif fence_kind in dinkster_blocks:
            fenced_lines.append(line)
    if fence_length and fence_kind in dinkster_blocks:
        return None, (), f"{path} has an unterminated {fence_kind} block"
    visible = _DOC_INLINE_CODE_RE.sub("", "\n".join(prose))
    for target in _DOC_LINK_RE.findall(visible):
        if not target.startswith(("https:", "mailto:", "dinkster:")):
            return None, (), f"{path} link target {target!r} uses a disallowed scheme"
    sources = list(_DOC_IMAGE_RE.findall(visible))
    typed_sources = [(source, "image") for source in sources]
    for media_block in dinkster_blocks["dinkster-media"]:
        try:
            block = tomllib.loads(media_block)
        except tomllib.TOMLDecodeError as exc:
            return None, (), f"{path} has an invalid dinkster-media block: {exc}"
        unknown = set(block) - {"asset", "caption", "poster"}
        if unknown:
            return (
                None,
                (),
                f"{path} dinkster-media has unknown fields: {', '.join(sorted(unknown))}",
            )
        asset = block.get("asset")
        if not isinstance(asset, str) or not asset:
            return None, (), f"{path} dinkster-media asset must be a non-empty string"
        sources.append(asset)
        typed_sources.append((asset, "video"))
        poster = block.get("poster")
        if poster is not None:
            if not isinstance(poster, str) or not poster:
                return None, (), f"{path} dinkster-media poster must be a non-empty string"
            sources.append(poster)
            typed_sources.append((poster, "image"))
        caption = block.get("caption")
        if caption is not None and not isinstance(caption, str):
            return None, (), f"{path} dinkster-media caption must be a string"
    for example_block in dinkster_blocks["dinkster-example"]:
        try:
            block = tomllib.loads(example_block)
        except tomllib.TOMLDecodeError as exc:
            return None, (), f"{path} has an invalid dinkster-example block: {exc}"
        unknown = set(block) - {"blueprint", "template", "caption"}
        if unknown:
            return (
                None,
                (),
                f"{path} dinkster-example has unknown fields: {', '.join(sorted(unknown))}",
            )
        blueprint = block.get("blueprint")
        template = block.get("template")
        if (blueprint is None) == (template is None):
            return (
                None,
                (),
                f"{path} dinkster-example must declare exactly one blueprint or template",
            )
        target_kind = "blueprint" if blueprint is not None else "template"
        target = blueprint if blueprint is not None else template
        if not isinstance(target, str) or not target or validate_name(target) is not None:
            return None, (), f"{path} dinkster-example {target_kind} must be a valid non-empty id"
        known_targets = blueprint_ids if target_kind == "blueprint" else template_ids
        if target not in known_targets:
            return None, (), f"{path} dinkster-example references unknown {target_kind} {target!r}"
        caption = block.get("caption")
        if caption is not None and not isinstance(caption, str):
            return None, (), f"{path} dinkster-example caption must be a string"
    node_references: list[str] = []
    for node_block in dinkster_blocks["dinkster-node"]:
        try:
            block = tomllib.loads(node_block)
        except tomllib.TOMLDecodeError as exc:
            return None, (), f"{path} has an invalid dinkster-node block: {exc}"
        unknown = set(block) - {"node"}
        if unknown:
            return (
                None,
                (),
                f"{path} dinkster-node has unknown fields: {', '.join(sorted(unknown))}",
            )
        node = block.get("node")
        if not isinstance(node, str) or not node or validate_name(node) is not None:
            return None, (), f"{path} dinkster-node node must be a valid non-empty node type"
        node_references.append(node)
    for source in sources:
        relative = Path(source)
        if not source.startswith("assets/") or relative.is_absolute() or ".." in relative.parts:
            return None, (), f"{path} asset source {source!r} must be beneath assets/"
        if source not in assets:
            return None, (), f"{path} references missing or invalid asset {source!r}"
    for source, required_type in typed_sources:
        if not assets[source].media_type.startswith(f"{required_type}/"):
            return None, (), f"{path} requires {source!r} to be a {required_type} asset"
    return tuple(dict.fromkeys(sources)), tuple(dict.fromkeys(node_references)), None


def _guide_metadata(path: Path) -> tuple[int | None, tuple[str, ...], str | None, str | None]:
    config = path / "guide.toml"
    if not config.is_file():
        return None, (), None, None
    try:
        values = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return None, (), None, f"{config} is invalid: {exc}"
    unknown = set(values) - {"order", "tags", "guide_kind"}
    if unknown:
        return None, (), None, f"{config} has unknown fields: {', '.join(sorted(unknown))}"
    order = values.get("order")
    if order is not None and type(order) is not int:
        return None, (), None, f"{config} order must be an integer"
    raw_tags = values.get("tags", [])
    if not isinstance(raw_tags, list):
        return None, (), None, f"{config} tags must be non-empty strings"
    tags: list[str] = []
    for tag in cast("list[object]", raw_tags):
        if not isinstance(tag, str) or not tag:
            return None, (), None, f"{config} tags must be non-empty strings"
        tags.append(tag)
    guide_kind = values.get("guide_kind")
    if guide_kind is not None and guide_kind != "tour":
        return None, (), None, f"{config} guide_kind must be 'tour' when present"
    return (
        order,
        tuple(tags),
        cast("str | None", guide_kind),
        None,
    )


def _parse_docs(
    raw: object,
    manifest_path: Path,
    blueprint_ids: set[str],
    template_ids: set[str],
) -> tuple[PackDocs | None, str | None]:
    if raw is None:
        return None, None
    log = core_logger("workers")
    if not isinstance(raw, dict):
        problem = "must be a table"
        log.warning("%s: [pack.docs] %s; ignoring it", manifest_path, problem)
        return None, problem
    table = cast("dict[str, object]", raw)
    unknown = set(table) - {"dir", "default_locale"}
    if unknown:
        problem = f"has unknown fields {', '.join(sorted(unknown))}"
        log.warning("%s: [pack.docs] %s; ignoring it", manifest_path, problem)
        return None, problem
    default_locale = table.get("default_locale", "en")
    if not isinstance(default_locale, str) or default_locale not in DOC_LOCALES:
        problem = f"default_locale must be one of {', '.join(sorted(DOC_LOCALES))}"
        log.warning("%s: [pack.docs] %s; ignoring it", manifest_path, problem)
        return None, problem
    root, problem = _docs_relative_root(manifest_path, table.get("dir", "docs"))
    if root is None:
        log.warning("%s: [pack.docs] %s; ignoring it", manifest_path, problem)
        return None, problem

    assets: dict[str, PackDocAsset] = {}
    total = 0
    asset_root = root / "assets"
    if asset_root.is_dir():
        for path in sorted(item for item in asset_root.rglob("*") if item.is_file()):
            source = path.relative_to(root).as_posix()
            media_type = DOC_MEDIA_TYPES.get(path.suffix.lower())
            if media_type is None:
                log.warning(
                    "%s: documentation asset %r has a disallowed extension; dropping it",
                    manifest_path,
                    source,
                )
                continue
            cap = DOC_VIDEO_MAX_BYTES if media_type.startswith("video/") else DOC_IMAGE_MAX_BYTES
            data, problem = _read_doc_file(path, root, cap)
            if data is None:
                log.warning(
                    "%s: %s; dropping documentation asset %r", manifest_path, problem, source
                )
                continue
            if total + len(data) > DOC_PACK_MAX_BYTES:
                log.warning(
                    "%s: documentation asset %r exceeds the %d-byte pack budget; dropping it",
                    manifest_path,
                    source,
                    DOC_PACK_MAX_BYTES,
                )
                continue
            total += len(data)
            assets[source] = PackDocAsset(
                source=source,
                digest="sha256:" + hashlib.sha256(data).hexdigest(),
                media_type=media_type,
                data=data,
            )

    pages: list[PackDocPage] = []
    validation_problems: list[str] = []
    for kind, directory in (("node", "nodes"), ("guide", "guides")):
        content_root = root / directory
        if not content_root.is_dir():
            continue
        for content_dir in sorted(path for path in content_root.iterdir() if path.is_dir()):
            content_id = content_dir.name
            if validate_name(content_id) is not None:
                log.warning(
                    "%s: documentation %s id %r is invalid; dropping its pages",
                    manifest_path,
                    kind,
                    content_id,
                )
                continue
            order: int | None = None
            tags: tuple[str, ...] = ()
            guide_kind: str | None = None
            if kind == "guide":
                order, tags, guide_kind, problem = _guide_metadata(content_dir)
                if problem is not None:
                    log.warning("%s: %s; dropping guide %r", manifest_path, problem, content_id)
                    continue
            for path in sorted(content_dir.glob("*.md")):
                locale = path.stem
                if locale not in DOC_LOCALES:
                    log.warning(
                        "%s: documentation page %s has unsupported locale %r; dropping it",
                        manifest_path,
                        path,
                        locale,
                    )
                    continue
                data, problem = _read_doc_file(path, root, DOC_PAGE_MAX_BYTES)
                if data is None:
                    log.warning("%s: %s; dropping documentation page", manifest_path, problem)
                    continue
                front, body, problem = _doc_front_matter(data, path)
                if front is None or body is None:
                    log.warning("%s: %s; dropping documentation page", manifest_path, problem)
                    continue
                sources, node_references, problem = _doc_body_references(
                    body.decode("utf-8"), path, assets, blueprint_ids, template_ids
                )
                if sources is None:
                    assert problem is not None
                    validation_problems.append(problem)
                    log.warning("%s: %s; dropping documentation page", manifest_path, problem)
                    continue
                if total + len(data) > DOC_PACK_MAX_BYTES:
                    log.warning(
                        "%s: documentation page %s exceeds the %d-byte pack budget; dropping it",
                        manifest_path,
                        path,
                        DOC_PACK_MAX_BYTES,
                    )
                    continue
                total += len(data)
                pages.append(
                    PackDocPage(
                        kind=kind,
                        id=content_id,
                        locale=locale,
                        title=cast("str", front["title"]),
                        summary=cast("str", front["summary"]),
                        schema_version=cast("int | None", front.get("schema_version")),
                        digest="sha256:" + hashlib.sha256(body).hexdigest(),
                        assets=tuple(assets[source] for source in sources),
                        order=order,
                        tags=tags,
                        guide_kind=guide_kind,
                        node_references=node_references,
                        data=body,
                    )
                )

    valid_keys = {(page.kind, page.id) for page in pages if page.locale == default_locale}
    for key in sorted({(page.kind, page.id) for page in pages} - valid_keys):
        log.warning(
            "%s: documentation %s %r has no %r page; dropping all of its pages",
            manifest_path,
            key[0],
            key[1],
            default_locale,
        )
    pages = [page for page in pages if (page.kind, page.id) in valid_keys]
    used_sources = {asset.source for page in pages for asset in page.assets}
    return (
        PackDocs(
            default_locale=default_locale,
            pages=tuple(pages),
            assets=tuple(asset for source, asset in assets.items() if source in used_sources),
            validation_problems=tuple(validation_problems),
        ),
        None,
    )


def validate_pack_template(
    manifest_path: Path, entry: object
) -> tuple[PackTemplate | None, str | None]:
    """Validate one ``[[pack.templates]]`` entry against the contract;
    ``(template, None)`` or ``(None, reason)``. Shared by the
    warn-and-drop manifest path and by doctor, so the loader and the
    linter can never disagree. Backend-ownable checks only - the same
    boundary as blueprints. Asset REFERENCES are shape-checked here
    (non-empty strings); existence against the pack's declarations is
    the parser's job, because only the parser knows which declarations
    survived."""
    if not isinstance(entry, dict):
        return None, "must be a table (a [[pack.templates]] entry)"
    table = cast("dict[str, object]", entry)

    template_id = table.get("id")
    if not isinstance(template_id, str) or not template_id:
        return None, "must declare a non-empty string 'id'"
    id_problem = validate_name(template_id)
    if id_problem is not None:
        return None, f"id {template_id!r} {id_problem}"

    name = table.get("name")
    if not isinstance(name, str) or not name:
        return None, "must declare a non-empty string 'name'"

    description = table.get("description", "")
    if not isinstance(description, str):
        return None, "'description' must be a string"

    tags_raw = table.get("tags", [])
    if not isinstance(tags_raw, list) or not all(
        isinstance(tag, str) and tag for tag in cast("list[object]", tags_raw)
    ):
        return None, "'tags' must be a list of non-empty strings"
    tags = tuple(cast("list[str]", tags_raw))

    assets_raw = table.get("assets", [])
    if not isinstance(assets_raw, list) or not all(
        isinstance(ref, str) and ref for ref in cast("list[object]", assets_raw)
    ):
        return None, "'assets' must be a list of non-empty [[pack.assets]] id strings"
    assets = tuple(cast("list[str]", assets_raw))

    data, path, problem = _read_document_file(
        manifest_path,
        table.get("file"),
        max_bytes=TEMPLATE_MAX_BYTES,
        cap_reason="templates are graphs, not assets - reference models via [[pack.assets]] ids",
    )
    if data is None or path is None:
        return None, problem

    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    return (
        PackTemplate(
            id=template_id,
            name=name,
            digest=digest,
            path=path,
            description=description,
            tags=tags,
            assets=assets,
            data=data,
        ),
        None,
    )


def _parse_templates(
    raw: object, manifest_path: Path, declared_assets: tuple[str, ...]
) -> tuple[PackTemplate, ...]:
    """Parse ``[[pack.templates]]`` with warn-and-drop PER TEMPLATE, the
    same philosophy as blueprints: malformed entries, duplicate ids, and
    entries past the per-pack byte budget drop individually and the pack
    always survives. One additional rule: a template referencing an asset
    id absent from the pack's SURVIVING ``[[pack.assets]]`` declarations
    drops - shipping it would advertise a starter workflow whose
    requirements can never be resolved to a digest-pinned need, and a
    template that silently cannot acquire its models is worse than no
    template."""
    if raw is None:
        return ()
    log = core_logger("workers")
    if not isinstance(raw, list):
        log.warning(
            "%s: [[pack.templates]] must be an array of tables; ignoring it",
            manifest_path,
        )
        return ()
    known_assets = set(declared_assets)
    templates: list[PackTemplate] = []
    seen: set[str] = set()
    total = 0
    for index, entry in enumerate(cast("list[object]", raw)):
        template, problem = validate_pack_template(manifest_path, entry)
        if template is None:
            log.warning(
                "%s: [[pack.templates]] entry %d %s; dropping it",
                manifest_path,
                index,
                problem,
            )
            continue
        if template.id in seen:
            log.warning(
                "%s: [[pack.templates]] entry %d duplicates id %r; dropping it "
                "(the first declaration wins)",
                manifest_path,
                index,
                template.id,
            )
            continue
        dangling = [ref for ref in template.assets if ref not in known_assets]
        if dangling:
            log.warning(
                "%s: [[pack.templates]] entry %d (%r) references undeclared "
                "asset ids %s; dropping it (declare them as [[pack.assets]] "
                "entries so the requirements stay digest-pinned)",
                manifest_path,
                index,
                template.id,
                ", ".join(repr(ref) for ref in dangling),
            )
            continue
        if total + len(template.data) > TEMPLATE_PACK_MAX_BYTES:
            log.warning(
                "%s: [[pack.templates]] entry %d (%r) would exceed the per-pack "
                "template budget of %d bytes; dropping it",
                manifest_path,
                index,
                template.id,
                TEMPLATE_PACK_MAX_BYTES,
            )
            continue
        seen.add(template.id)
        total += len(template.data)
        templates.append(template)
    return tuple(templates)


def load_pack_templates(path: Path | str, *, pack: str) -> tuple[PackTemplate, ...]:
    """Read ONLY ``[[pack.templates]]`` from a ``dinkster-pack.toml``.

    The template on-ramp for packs that are not (yet) Dinkster packs, riding
    the same presentation-only manifest as icons, blueprints, and assets:
    a legacy ComfyUI pack can ship starter workflows without being a
    loadable pack manifest. Asset references validate against the SAME
    file's ``[[pack.assets]]`` declarations (``pack`` names the pack-table
    id those packaged sources resolve under). Everything about it is
    advisory: a missing file is (), and malformed TOML or entries
    warn-and-drop exactly like manifest loading - templates can never
    break composition."""
    manifest_path = Path(path)
    try:
        with open(manifest_path, "rb") as fh:
            document = tomllib.load(fh)
    except FileNotFoundError:
        return ()
    except (OSError, tomllib.TOMLDecodeError) as exc:
        core_logger("workers").warning(
            "%s: unreadable pack manifest for templates (%s); ignoring it",
            manifest_path,
            exc,
        )
        return ()
    pack_table = document.get("pack")
    if not isinstance(pack_table, dict):
        return ()
    pack_table = cast("dict[str, object]", pack_table)
    assets = _parse_assets(pack_table.get("assets"), manifest_path, pack)
    return _parse_templates(
        pack_table.get("templates"),
        manifest_path,
        tuple(asset.id for asset in assets),
    )


def validate_pack_asset(
    manifest_path: Path, entry: object, pack: str
) -> tuple[DeclaredAsset | None, str | None]:
    """Validate one ``[[pack.assets]]`` entry against the contract;
    ``(asset, None)`` or ``(None, reason)``. Shared by the warn-and-drop
    manifest path and by doctor, so the loader and the linter can never
    disagree. ``pack`` is the pack-table id packaged sources resolve
    under - the host maps it to the installed artifact root.

    Declarations are data about bytes, never the bytes: a packaged
    ``file`` must exist inside the pack directory (checked, not read -
    models can be gigabytes; the digest verifies at acquisition), and
    remote ``urls`` are leads the digest disciplines. ``nodes`` names
    node types whose instantiation requires the asset, passed through
    VERBATIM (legacy node types are not grammar-valid names); the
    backend never derives the association from node code."""
    if not isinstance(entry, dict):
        return None, "must be a table (a [[pack.assets]] entry)"
    table = cast("dict[str, object]", entry)

    asset_id = table.get("id")
    if not isinstance(asset_id, str) or not asset_id:
        return None, "must declare a non-empty string 'id'"
    id_problem = validate_name(asset_id)
    if id_problem is not None:
        return None, f"id {asset_id!r} {id_problem}"

    name = table.get("name")
    if not isinstance(name, str) or not name:
        return None, "must declare a non-empty string 'name'"

    digest = table.get("digest")
    if not isinstance(digest, str) or not digest:
        return None, (
            "must declare a 'digest' ('blake3:<64 hex>') - distribution "
            "without a pin would let any source plant any bytes"
        )

    kind = table.get("kind", "")
    if not isinstance(kind, str):
        return None, "'kind' must be a string"

    size = table.get("size", -1)
    if not isinstance(size, int) or isinstance(size, bool) or size < -1:
        return None, "'size' must be a non-negative integer"

    media_type = table.get("media_type", "")
    if not isinstance(media_type, str):
        return None, "'media_type' must be a string"

    components_raw = table.get("components")
    try:
        component_manifest = (
            AssetComponentManifest.from_wire(components_raw) if components_raw is not None else None
        )
    except AssetError as exc:
        return None, f"'components' invalid: {exc}"

    nodes_raw = table.get("nodes", [])
    if not isinstance(nodes_raw, list) or not all(
        isinstance(node, str) and node for node in cast("list[object]", nodes_raw)
    ):
        return None, "'nodes' must be a list of non-empty node-type strings"
    nodes = tuple(cast("list[str]", nodes_raw))

    sources: list[AssetSource] = []
    declared_file = table.get("file")
    if declared_file is not None:
        if not isinstance(declared_file, str) or not declared_file:
            return None, "'file' must be a non-empty relative path string"
        rel = Path(declared_file)
        if rel.is_absolute() or ".." in rel.parts:
            return None, (
                f"file {declared_file!r} must be a relative path inside the pack directory"
            )
        path = manifest_path.parent / rel
        # resolve() follows symlinks, so a link inside the pack pointing
        # outside it fails containment just like a literal `..` would.
        if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
            return None, (f"file {declared_file!r} must not escape the pack directory (symlink?)")
        if not path.is_file():
            return None, f"file {declared_file!r} does not exist in the pack"
        try:
            sources.append(PackagedSource(pack=pack, path=rel.as_posix()))
        except AssetError as exc:
            return None, f"file {declared_file!r} invalid: {exc}"

    urls_raw = table.get("urls", [])
    if not isinstance(urls_raw, list) or not all(
        isinstance(url, str) for url in cast("list[object]", urls_raw)
    ):
        return None, "'urls' must be a list of http(s) URL strings"
    for url in cast("list[str]", urls_raw):
        try:
            sources.append(RemoteSource(url=url))
        except AssetError as exc:
            return None, str(exc)

    if not sources:
        return None, ("must declare at least one source (a packaged 'file' or remote 'urls')")

    try:
        need = AssetNeed(
            name=name,
            digest=digest,
            kind=kind,
            size=size,
            media_type=media_type,
            component_manifest=component_manifest,
            sources=tuple(sources),
        )
        asset = DeclaredAsset(id=asset_id, need=need, nodes=nodes)
    except AssetError as exc:
        return None, str(exc)
    return asset, None


def _parse_assets(raw: object, manifest_path: Path, pack: str) -> tuple[DeclaredAsset, ...]:
    """Parse ``[[pack.assets]]`` with warn-and-drop PER ENTRY: a malformed
    declaration is a logged diagnostic that drops that asset only - the
    pack and its sibling declarations always survive (same philosophy as
    presentation and blueprints). Duplicate ids keep the first."""
    if raw is None:
        return ()
    log = core_logger("workers")
    if not isinstance(raw, list):
        log.warning(
            "%s: [[pack.assets]] must be an array of tables; ignoring it",
            manifest_path,
        )
        return ()
    assets: list[DeclaredAsset] = []
    seen: set[str] = set()
    for index, entry in enumerate(cast("list[object]", raw)):
        asset, problem = validate_pack_asset(manifest_path, entry, pack)
        if asset is None:
            log.warning(
                "%s: [[pack.assets]] entry %d %s; dropping it",
                manifest_path,
                index,
                problem,
            )
            continue
        if asset.id in seen:
            log.warning(
                "%s: [[pack.assets]] entry %d duplicates id %r; dropping it "
                "(the first declaration wins)",
                manifest_path,
                index,
                asset.id,
            )
            continue
        seen.add(asset.id)
        assets.append(asset)
    return tuple(assets)


def load_pack_assets(path: Path | str, *, pack: str) -> tuple[DeclaredAsset, ...]:
    """Read ONLY ``[[pack.assets]]`` from a ``dinkster-pack.toml``.

    The asset on-ramp for packs that are not (yet) Dinkster packs, riding
    the same presentation-only manifest as icons and blueprints: a legacy
    ComfyUI pack (or its user) can declare the models its nodes need
    without the file being a loadable pack manifest. ``pack`` is the
    pack-table id packaged sources resolve under. Everything about it is
    advisory: a missing file is (), and malformed TOML or entries
    warn-and-drop exactly like manifest loading - declarations can never
    break composition."""
    manifest_path = Path(path)
    try:
        with open(manifest_path, "rb") as fh:
            document = tomllib.load(fh)
    except FileNotFoundError:
        return ()
    except (OSError, tomllib.TOMLDecodeError) as exc:
        core_logger("workers").warning(
            "%s: unreadable pack manifest for assets (%s); ignoring it",
            manifest_path,
            exc,
        )
        return ()
    pack_table = document.get("pack")
    if not isinstance(pack_table, dict):
        return ()
    raw = cast("dict[str, object]", pack_table).get("assets")
    return _parse_assets(raw, manifest_path, pack)


@dataclass(frozen=True)
class PackPresentation:
    """Author-declared compact identity, from ``[pack.presentation]``.

    Presentation only, never identity: the pack name is the sole lookup
    key, marks/abbrs may collide across packs, and the frontend owns
    fallbacks for packs that declare nothing. Malformed fields are
    warn-and-drop at load - a bad emoji must not stop a pack from loading.
    """

    display_name: str = ""
    abbr: str = ""
    """Short ASCII label (<= 8 chars) for search palettes and list rows."""
    mark: str = ""
    """Single compact glyph (an emoji or 1-2 characters) for node header
    chips. Validated as <= 8 code points with no whitespace/control
    characters - a pragmatic bound for one grapheme cluster, since exact
    segmentation is presentation policy that belongs to the renderer."""
    color: str = ""
    """Chip fill as ``#rrggbb``; contrast handling is the frontend's."""
    icon: PackIcon | None = None
    """Validated raster badge (exactly 64x64 static PNG/WebP shipped in
    the pack directory), declared as a relative path. Renders over the
    chip fill; precedence (icon > mark > derived initials) is frontend
    policy."""


@dataclass(frozen=True)
class PackContracts:
    """Versioned host surfaces consumed by one pack."""

    host: str
    api: str
    inference: str | None = None


@dataclass(frozen=True)
class PackDependency:
    """One pack ordering dependency and its compatible release range."""

    pack: str
    version: str

    def accepts(self, version: str) -> bool:
        return SpecifierSet(self.version).contains(version, prereleases=True)


@dataclass(frozen=True)
class PackRegistryRequirement:
    """One exact descriptor required from a named host registry."""

    registry: str
    id: str


@dataclass(frozen=True)
class PackRegistryProvider:
    """One exact registry descriptor provided by this pack."""

    registry: str
    id: str


@dataclass(frozen=True)
class PackCapabilityRequirement:
    """One provider-agnostic capability and its compatible release range."""

    id: str
    version: str

    def accepts(self, version: str) -> bool:
        return SpecifierSet(self.version).contains(version, prereleases=True)


@dataclass(frozen=True)
class PackRequirements:
    """Registry descriptors and capabilities consumed by one pack."""

    registry: tuple[PackRegistryRequirement, ...] = ()
    capabilities: tuple[PackCapabilityRequirement, ...] = ()


@dataclass(frozen=True)
class PackProvides:
    """Registry descriptors supplied by one pack contribution."""

    registry: tuple[PackRegistryProvider, ...] = ()


def unmatched_registry_providers(
    provides: PackProvides,
    contributions: Iterable[tuple[str, str]],
) -> tuple[PackRegistryProvider, ...]:
    """Return provider claims absent from the pack's inference declarations."""
    registered = {
        (canonical_name(surface_id), canonical_name(descriptor_id))
        for surface_id, descriptor_id in contributions
    }
    return tuple(
        provider
        for provider in provides.registry
        if (
            canonical_name(
                _PACK_REGISTRY_CONTRIBUTION_SURFACES.get(canonical_name(provider.registry), "")
            ),
            canonical_name(provider.id),
        )
        not in registered
    )


@dataclass(frozen=True)
class PackSandboxNeeds:
    """OS resources an isolated pack asks the host to grant."""

    gpu: bool = False
    network: bool = False
    writable_mounts: bool = False


@dataclass(frozen=True)
class PackCapability:
    """One versioned capability provided by a pack."""

    id: str
    version: str


@dataclass(frozen=True)
class VisionProvider:
    """One model-backed vision implementation supplied by a pack.

    The pack name is the compatibility provider id retained by pinned legacy
    workflows. ``choice`` names the owner-controlled provider vocabulary,
    while ``node`` names the stable schema this pack implements through
    ``[pack] executes``. The remaining fields make model and runtime
    requirements inspectable without importing provider code.
    """

    choice: str
    node: str
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    batching: str
    artifacts: tuple[str, ...]
    model: str | None = None


@dataclass(frozen=True)
class GenerationProvider:
    """One external generation implementation supplied by a pack."""

    choice: str
    node: str
    label: str | None = None


def vision_provider_to_wire(provider: VisionProvider) -> dict[str, object]:
    wire: dict[str, object] = {
        "choice": provider.choice,
        "node": provider.node,
        "devices": list(provider.devices),
        "dtypes": list(provider.dtypes),
        "batching": provider.batching,
        "artifacts": list(provider.artifacts),
    }
    if provider.model is not None:
        wire["model"] = provider.model
    return wire


def generation_provider_to_wire(provider: GenerationProvider) -> dict[str, object]:
    wire: dict[str, object] = {"choice": provider.choice, "node": provider.node}
    if provider.label is not None:
        wire["label"] = provider.label
    return wire


def _provider_wire_name(value: object, *, field: str, namespaced: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    problem = validate_name(value)
    if problem is not None:
        raise ValueError(f"{field} {value!r} {problem}")
    if namespaced and "." not in value:
        raise ValueError(f"{field} {value!r} must be namespaced")
    return value


def _provider_wire_strings(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a list of strings")
    items = tuple(cast("Sequence[object]", value))
    if (not allow_empty and not items) or not all(isinstance(item, str) for item in items):
        qualifier = "a" if allow_empty else "a non-empty"
        raise ValueError(f"{field} must be {qualifier} list of strings")
    strings = cast("tuple[str, ...]", items)
    if len(strings) != len(set(strings)):
        raise ValueError(f"{field} contains duplicates")
    return strings


def vision_providers_from_wire(raw: object) -> tuple[VisionProvider, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("visionProviders must be a list")
    providers: list[VisionProvider] = []
    seen_nodes: set[str] = set()
    expected = {"choice", "node", "devices", "dtypes", "batching", "artifacts", "model"}
    required = expected - {"model"}
    for index, item in enumerate(cast("Sequence[object]", raw)):
        subject = f"visionProviders entry {index}"
        if not isinstance(item, Mapping):
            raise ValueError(f"{subject} must be an object")
        fields = cast("Mapping[object, object]", item)
        keys = set(fields)
        if keys - expected or required - keys:
            raise ValueError(f"{subject} has malformed fields")
        choice = _provider_wire_name(fields["choice"], field=f"{subject} choice", namespaced=True)
        node = _provider_wire_name(fields["node"], field=f"{subject} node")
        if node in seen_nodes:
            raise ValueError(f"visionProviders repeats node {node!r}")
        seen_nodes.add(node)
        devices = _provider_wire_strings(fields["devices"], field=f"{subject} devices")
        if invalid := sorted(set(devices) - set(KNOWN_ACCELERATORS)):
            raise ValueError(f"{subject} devices contains unknown values: {', '.join(invalid)}")
        dtypes = _provider_wire_strings(fields["dtypes"], field=f"{subject} dtypes")
        if invalid := sorted(set(dtypes) - set(VISION_PROVIDER_DTYPES)):
            raise ValueError(f"{subject} dtypes contains unknown values: {', '.join(invalid)}")
        batching = fields["batching"]
        if batching not in VISION_PROVIDER_BATCHING:
            raise ValueError(f"{subject} batching is unsupported")
        artifacts = _provider_wire_strings(
            fields["artifacts"], field=f"{subject} artifacts", allow_empty=True
        )
        model = fields.get("model")
        if model is not None and (not isinstance(model, str) or not model):
            raise ValueError(f"{subject} model must be a non-empty string")
        providers.append(
            VisionProvider(
                choice=choice,
                node=node,
                devices=devices,
                dtypes=dtypes,
                batching=cast("str", batching),
                artifacts=artifacts,
                model=model,
            )
        )
    return tuple(providers)


def generation_providers_from_wire(raw: object) -> tuple[GenerationProvider, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("generationProviders must be a list")
    providers: list[GenerationProvider] = []
    seen_nodes: set[str] = set()
    expected = {"choice", "node", "label"}
    required = expected - {"label"}
    for index, item in enumerate(cast("Sequence[object]", raw)):
        subject = f"generationProviders entry {index}"
        if not isinstance(item, Mapping):
            raise ValueError(f"{subject} must be an object")
        fields = cast("Mapping[object, object]", item)
        keys = set(fields)
        if keys - expected or required - keys:
            raise ValueError(f"{subject} has malformed fields")
        choice = _provider_wire_name(fields["choice"], field=f"{subject} choice", namespaced=True)
        node = _provider_wire_name(fields["node"], field=f"{subject} node")
        if node in seen_nodes:
            raise ValueError(f"generationProviders repeats node {node!r}")
        seen_nodes.add(node)
        label = fields.get("label")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError(f"{subject} label must be a non-empty string")
        providers.append(GenerationProvider(choice=choice, node=node, label=label))
    return tuple(providers)


@dataclass(frozen=True)
class PackManifest:
    name: str
    namespaces: tuple[str, ...]
    """Node-type namespace claims, from ``[pack] namespaces`` (default:
    the pack's own name). Grammar-validated at load; every node type the
    pack announces must fall under one of these (doctor validates, local
    composition enforces, the future registry grants). Claims, not
    grants: a manifest is publisher-controlled input, so nothing here is
    authoritative beyond this machine."""
    nodes_entry: str
    types_entry: str | None
    reservations_entry: str | None
    consumers_entry: str | None
    choices_entry: str | None
    """Optional ``[pack.entry] choices``: a callable returning the pack's
    combo choice lists (choice-list id -> sequence of value strings).
    Choice ids are namespaced names under the pack's claims, exactly like
    node types (``comfy.samplers``); the lists feed remote ComboWidget
    routes (``/api/choices/{id}``). Enumerated once at worker startup,
    inside the child - UI vocabulary, never identity."""
    skips_entry: str | None
    """Optional ``[pack.entry] skips``: a callable returning classified
    compat translation skips (source node name -> reason). Enumerated once
    at worker startup and carried to instance diagnostics; advisory only."""
    telemetry_entry: str | None
    """Optional ``[pack.entry] telemetry``: a zero-arg callable returning
    the pack's measured device memory (``Mapping[str, MeasuredMemory]``,
    keys in the worker's own device namespace). Reported across the
    boundary for the parent governor's observability - informational
    only, never admission."""
    schema_reload_entry: str | None
    """Optional ``[pack.entry] schema_reload``: an async callable that
    returns when this worker's schema source has a valid replacement.
    Launched workers notify their parent, which owns atomic replacement."""
    arm_nodes_entry: str | None
    """Optional ``[pack.entry] arm_nodes`` mapping same-session arm names
    to alternate Node body classes. It must be paired with ``[pack.arms]``."""
    requires: tuple[str, ...]
    path: Path
    """The manifest file itself; the pack root is its parent directory."""
    contracts: PackContracts | None = None
    """Explicit host/API/inference contracts. None marks a legacy manifest."""
    dependencies: tuple[PackDependency, ...] = ()
    """Pack ordering dependencies, sorted by canonical pack id."""
    requirements: PackRequirements = PackRequirements()
    """Exact registry descriptors and versioned capabilities this pack consumes."""
    provides: PackProvides = PackProvides()
    """Exact registry descriptors supplied by this pack's contribution."""
    sandbox: PackSandboxNeeds = PackSandboxNeeds()
    """Requested OS resources. The host remains authoritative over every grant."""
    sandbox_declared: bool = False
    """True only when the manifest contains a [pack.sandbox] table."""
    capabilities: tuple[PackCapability, ...] = ()
    """Versioned capabilities this pack provides to the composed generation."""
    vision_providers: tuple[VisionProvider, ...] = ()
    """Model-backed vision schemas this pack implements, with explicit
    artifact, device, dtype, and batching requirements."""
    generation_providers: tuple[GenerationProvider, ...] = ()
    """External generation schemas this pack implements."""
    source_staging_entry: str | None = None
    """Optional per-invocation source staging provider factory."""
    presentation: PackPresentation | None = None
    """Optional [pack.presentation]; None when the pack declares nothing."""
    blueprints: tuple[PackBlueprint, ...] = ()
    """Validated [[pack.blueprints]] entries: starter workflow documents
    the pack ships as plain data. Warn-and-drop per entry at load; never
    identity (excluded from schema signatures and execution identity)."""
    assets: tuple[DeclaredAsset, ...] = ()
    """Validated [[pack.assets]] entries: digest-pinned assets the pack
    distributes (packaged in the artifact and/or via remote URLs), with
    optional node-type associations for preflight. Declarations only -
    nothing downloads at load, composition, or install; acquisition
    stays behind digest-exact consent. Warn-and-drop per entry; never
    identity."""
    templates: tuple[PackTemplate, ...] = ()
    """Validated [[pack.templates]] entries: complete starter workflow
    documents the pack ships as plain data, with asset requirements as
    references into this manifest's own [[pack.assets]] ids. Warn-and-drop
    per entry at load; never identity (excluded from schema signatures
    and execution identity)."""
    docs: PackDocs | None = None
    """Validated pack documentation, present only with ``[pack.docs]``."""
    docs_problem: str | None = None
    """Invalid explicit ``[pack.docs]`` declaration retained for doctor."""
    locale_catalogs: tuple[PackLocaleCatalog, ...] = ()
    """Validated pack-local translation catalogs, preserved as source bytes."""
    locale_catalog_problems: tuple[str, ...] = ()
    """Invalid locale catalogs retained for structured doctor findings."""
    platforms: tuple[str, ...] = ()
    """Optional ``[pack] platforms``: the OSes the pack declares it works
    on, as ``sys.platform`` values (``linux``/``win32``/``darwin`` - the
    same vocabulary PEP 508 ``sys_platform`` markers use). Empty means
    unrestricted. ADVISORY: install plans warn loudly on a mismatched
    host, they never refuse - the author's claim informs the user, the
    user decides."""
    extra_requires: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Optional ``[pack.extra-requires]``: accelerator-conditional
    dependencies, (accelerator, requirements) pairs sorted by key.
    Accelerator is the ONE dimension PEP 508 markers cannot express -
    OS/python conditionals belong in markers inside any requires list;
    hardware vendor conditionals belong here. Provisioning appends the
    host accelerator's list to ``requires``."""
    executes: tuple[str, ...] = ()
    """Optional ``[pack] executes``: node types this pack implements as an
    alternate executor without owning them. Schemas listed here must match
    the owning pack's schema signature exactly and never enter the serving
    surface, routing, or attribution -
    they become invocable only when the host explicitly enrolls this pack
    as a dispatch arm for those types. Claims, not grants: like namespace
    claims, a manifest is publisher-controlled input, so listing a type
    here grants nothing by itself."""
    schema_only: tuple[str, ...] = ()
    """Optional ``[pack] schema-only`` node types this pack owns but does
    not execute. A separate pack must claim each type through ``executes``."""
    arms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Validated ``[pack.arms]`` declarations for schemas this pack owns
    or lists in ``executes``, in manifest order."""
    extension: ExtensionDeclaration = ExtensionDeclaration()
    """Validated ``[pack.extension]`` scope entries and authority requests.

    Entry references remain strings here. Manifest discovery never imports or
    resolves pack code; activation owns that later, in the appropriate scope.
    """
    extension_declared: bool = False
    """True only when the manifest contains a [pack.extension] table.

    The empty declaration is a useful value default, but absence is behavior:
    packs without the additive table must not appear in ExtensionSnapshot.
    """
    workgroup_handler_entry: str | None = None
    """Optional zero-argument factory for a launched-child workgroup handler."""
    comfy_aliases: ComfyAliasRegistry | None = None
    """Strict import-only translation data from adjacent ``comfy-aliases.json``."""
    comfy_groups: ComfyGroupRegistry | None = None
    """Strict group-pattern data from adjacent ``comfy-groups.json``."""

    def requires_for(self, accelerator: str) -> tuple[str, ...]:
        """The full requirement list for a host: base ``requires`` plus
        the matching ``[pack.extra-requires]`` entry, if any."""
        for key, extras in self.extra_requires:
            if key == accelerator:
                return self.requires + extras
        return self.requires

    @property
    def root(self) -> Path:
        return self.path.parent


_CONTRACT_ID = re.compile(r"^[a-z][a-z0-9-]*/v?[1-9][0-9]*$")
_RELEASE_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def _canonical_namespaced_id(value: object, *, field: str, manifest_path: Path) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{manifest_path}: {field} must be a string")
    problem = validate_name(value)
    if problem is not None:
        raise ManifestError(f"{manifest_path}: {field} {value!r} {problem}")
    if "." not in value:
        raise ManifestError(f"{manifest_path}: {field} {value!r} must be namespaced")
    return value


def _version_range(value: object, *, field: str, manifest_path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{manifest_path}: {field} must be a non-empty version range string")
    try:
        version = str(SpecifierSet(value))
    except InvalidSpecifier as exc:
        raise ManifestError(
            f"{manifest_path}: {field} has invalid version range {value!r}: {exc}"
        ) from exc
    if not version:
        raise ManifestError(f"{manifest_path}: {field} must constrain the provider version")
    return version


def _release_version(value: object, *, field: str, manifest_path: Path) -> str:
    if not isinstance(value, str) or _RELEASE_VERSION.fullmatch(value) is None:
        raise ManifestError(
            f"{manifest_path}: {field} must be a 'major.minor.patch' version string"
        )
    try:
        return str(Version(value))
    except InvalidVersion as exc:  # pragma: no cover - regex is stricter
        raise ManifestError(f"{manifest_path}: {field} has invalid version {value!r}") from exc


def _parse_contracts(raw: object, manifest_path: Path) -> PackContracts | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ManifestError(f"{manifest_path}: [pack.contracts] must be a table")
    table = cast("dict[str, object]", raw)
    unknown = sorted(set(table) - {"host", "api", "inference"})
    if unknown:
        raise ManifestError(
            f"{manifest_path}: [pack.contracts] unknown fields: {', '.join(unknown)}"
        )
    values: dict[str, str | None] = {}
    for field in ("host", "api"):
        value = table.get(field)
        if not isinstance(value, str) or _CONTRACT_ID.fullmatch(value) is None:
            raise ManifestError(
                f"{manifest_path}: [pack.contracts] {field} must be a "
                "'lowercase-contract/positive-version' string"
            )
        values[field] = value
    inference = table.get("inference")
    if inference is not None and (
        not isinstance(inference, str) or _CONTRACT_ID.fullmatch(inference) is None
    ):
        raise ManifestError(
            f"{manifest_path}: [pack.contracts] inference must be a "
            "'lowercase-contract/positive-version' string"
        )
    return PackContracts(
        host=cast("str", values["host"]),
        api=cast("str", values["api"]),
        inference=inference,
    )


def _parse_pack_dependencies(raw: object, manifest_path: Path) -> tuple[PackDependency, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ManifestError(
            f"{manifest_path}: [pack.dependencies] must be a table of pack id -> version range"
        )
    dependencies: dict[str, PackDependency] = {}
    for declared, version_raw in cast("dict[str, object]", raw).items():
        problem = validate_name(declared)
        canonical = canonical_name(declared)
        if problem is not None:
            raise ManifestError(
                f"{manifest_path}: [pack.dependencies] pack id {declared!r} {problem}"
            )
        if declared != canonical:
            raise ManifestError(
                f"{manifest_path}: [pack.dependencies] pack id {declared!r} is not canonical "
                f"(expected {canonical!r})"
            )
        if canonical in dependencies:
            previous = dependencies[canonical].pack
            raise ManifestError(
                f"{manifest_path}: [pack.dependencies] pack ids {previous!r} and "
                f"{declared!r} are one canonical identity"
            )
        version = _version_range(
            version_raw,
            field=f"[pack.dependencies] {declared!r}",
            manifest_path=manifest_path,
        )
        dependencies[canonical] = PackDependency(canonical, version)
    return tuple(dependencies[name] for name in sorted(dependencies))


def _parse_registry_descriptor_map(
    raw: object,
    manifest_path: Path,
    *,
    subject: str,
    declaration: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, dict):
        raise ManifestError(
            f"{manifest_path}: {subject} registry must be a table of registry ids "
            "to descriptor-id lists"
        )
    registry: dict[tuple[str, str], tuple[str, str]] = {}
    for registry_declared, ids_raw in cast("dict[str, object]", raw).items():
        registry_id = _canonical_namespaced_id(
            registry_declared,
            field=f"{subject} registry id",
            manifest_path=manifest_path,
        )
        if not isinstance(ids_raw, list) or not all(
            isinstance(item, str) for item in cast("list[object]", ids_raw)
        ):
            raise ManifestError(
                f"{manifest_path}: {subject} registry {registry_declared!r} "
                "must be a list of descriptor ids"
            )
        for item in cast("list[str]", ids_raw):
            descriptor_id = _canonical_namespaced_id(
                item,
                field=f"{subject} registry {registry_declared!r} descriptor",
                manifest_path=manifest_path,
            )
            key = (canonical_name(registry_id), canonical_name(descriptor_id))
            if key in registry:
                raise ManifestError(
                    f"{manifest_path}: {subject} repeats registry {declaration} "
                    f"{registry_id!r} {descriptor_id!r}"
                )
            registry[key] = (registry_id, descriptor_id)
    return tuple(registry[key] for key in sorted(registry))


def _parse_requirements(raw: object, manifest_path: Path) -> PackRequirements:
    if raw is None:
        return PackRequirements()
    if not isinstance(raw, dict):
        raise ManifestError(f"{manifest_path}: [pack.requirements] must be a table")
    table = cast("dict[str, object]", raw)
    unknown = sorted(set(table) - {"registry", "capabilities"})
    if unknown:
        raise ManifestError(
            f"{manifest_path}: [pack.requirements] unknown fields: {', '.join(unknown)}"
        )

    registry = tuple(
        PackRegistryRequirement(registry_id, descriptor_id)
        for registry_id, descriptor_id in _parse_registry_descriptor_map(
            table.get("registry", {}),
            manifest_path,
            subject="[pack.requirements]",
            declaration="requirement",
        )
    )

    capabilities_raw = table.get("capabilities", {})
    if not isinstance(capabilities_raw, dict):
        raise ManifestError(
            f"{manifest_path}: [pack.requirements] capabilities must be a table of "
            "capability id to version range"
        )
    capabilities: dict[str, PackCapabilityRequirement] = {}
    for declared, version_raw in cast("dict[str, object]", capabilities_raw).items():
        capability_id = _canonical_namespaced_id(
            declared,
            field="[pack.requirements] capability id",
            manifest_path=manifest_path,
        )
        canonical = canonical_name(capability_id)
        if canonical in capabilities:
            raise ManifestError(
                f"{manifest_path}: [pack.requirements] repeats capability {declared!r}"
            )
        capabilities[canonical] = PackCapabilityRequirement(
            capability_id,
            _version_range(
                version_raw,
                field=f"[pack.requirements] capability {declared!r}",
                manifest_path=manifest_path,
            ),
        )
    return PackRequirements(
        registry=registry,
        capabilities=tuple(capabilities[key] for key in sorted(capabilities)),
    )


def _parse_provides(raw: object, manifest_path: Path) -> PackProvides:
    if raw is None:
        return PackProvides()
    if not isinstance(raw, dict):
        raise ManifestError(f"{manifest_path}: [pack.provides] must be a table")
    table = cast("dict[str, object]", raw)
    unknown = sorted(set(table) - {"registry"})
    if unknown:
        raise ManifestError(
            f"{manifest_path}: [pack.provides] unknown fields: {', '.join(unknown)}"
        )
    return PackProvides(
        registry=tuple(
            PackRegistryProvider(registry_id, descriptor_id)
            for registry_id, descriptor_id in _parse_registry_descriptor_map(
                table.get("registry", {}),
                manifest_path,
                subject="[pack.provides]",
                declaration="provider",
            )
        )
    )


def _parse_capabilities(raw: object, manifest_path: Path) -> tuple[PackCapability, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ManifestError(
            f"{manifest_path}: [pack.capabilities] must be a table of capability id to version"
        )
    capabilities: dict[str, PackCapability] = {}
    for declared, version_raw in cast("dict[str, object]", raw).items():
        capability_id = _canonical_namespaced_id(
            declared,
            field="[pack.capabilities] id",
            manifest_path=manifest_path,
        )
        canonical = canonical_name(capability_id)
        if canonical in capabilities:
            raise ManifestError(
                f"{manifest_path}: [pack.capabilities] repeats capability {declared!r}"
            )
        capabilities[canonical] = PackCapability(
            capability_id,
            _release_version(
                version_raw,
                field=f"[pack.capabilities] {declared!r}",
                manifest_path=manifest_path,
            ),
        )
    return tuple(capabilities[key] for key in sorted(capabilities))


VISION_PROVIDER_DTYPES = ("float16", "bfloat16", "float32")
VISION_PROVIDER_BATCHING = ("batch", "per-image")


def _vision_provider_vocabulary(
    value: object,
    *,
    field: str,
    allowed: tuple[str, ...],
    subject: str,
    manifest_path: Path,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in cast("list[object]", value))
    ):
        raise ManifestError(
            f"{manifest_path}: {subject} {field} must be a non-empty list of strings"
        )
    items = cast("list[str]", value)
    if len(set(items)) != len(items):
        raise ManifestError(f"{manifest_path}: {subject} {field} contains duplicates")
    invalid = sorted(set(items) - set(allowed))
    if invalid:
        raise ManifestError(
            f"{manifest_path}: {subject} {field} contains unknown values: "
            f"{', '.join(invalid)}; expected {', '.join(allowed)}"
        )
    return tuple(item for item in allowed if item in items)


def _parse_vision_providers(
    raw: object,
    *,
    manifest_path: Path,
    executes: tuple[str, ...],
    assets: tuple[DeclaredAsset, ...],
) -> tuple[VisionProvider, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ManifestError(
            f"{manifest_path}: [[pack.vision-providers]] must be a non-empty array of tables"
        )
    declared_assets = {asset.id: asset for asset in assets}
    providers: list[VisionProvider] = []
    seen_nodes: set[str] = set()
    required_fields = {"choice", "node", "devices", "dtypes", "batching", "artifacts"}
    expected_fields = required_fields | {"model"}
    for index, entry in enumerate(cast("list[object]", raw)):
        subject = f"[[pack.vision-providers]] entry {index}"
        if not isinstance(entry, dict):
            raise ManifestError(f"{manifest_path}: {subject} must be a table")
        table = cast("dict[str, object]", entry)
        unknown = sorted(set(table) - expected_fields)
        missing = sorted(required_fields - set(table))
        if unknown:
            raise ManifestError(f"{manifest_path}: {subject} unknown fields: {', '.join(unknown)}")
        if missing:
            raise ManifestError(f"{manifest_path}: {subject} missing fields: {', '.join(missing)}")
        choice = _canonical_namespaced_id(
            table["choice"], field=f"{subject} choice", manifest_path=manifest_path
        )
        node = table["node"]
        if not isinstance(node, str) or not node:
            raise ManifestError(f"{manifest_path}: {subject} node must be a non-empty string")
        if node not in executes:
            raise ManifestError(
                f"{manifest_path}: {subject} node {node!r} must appear in [pack] executes"
            )
        if node in seen_nodes:
            raise ManifestError(f"{manifest_path}: [[pack.vision-providers]] repeats node {node!r}")
        seen_nodes.add(node)
        devices = _vision_provider_vocabulary(
            table["devices"],
            field="devices",
            allowed=KNOWN_ACCELERATORS,
            subject=subject,
            manifest_path=manifest_path,
        )
        dtypes = _vision_provider_vocabulary(
            table["dtypes"],
            field="dtypes",
            allowed=VISION_PROVIDER_DTYPES,
            subject=subject,
            manifest_path=manifest_path,
        )
        batching = table["batching"]
        if batching not in VISION_PROVIDER_BATCHING:
            raise ManifestError(
                f"{manifest_path}: {subject} batching must be one of "
                f"{', '.join(VISION_PROVIDER_BATCHING)}"
            )
        artifacts_raw = table["artifacts"]
        if not isinstance(artifacts_raw, list) or not all(
            isinstance(item, str) and item for item in cast("list[object]", artifacts_raw)
        ):
            raise ManifestError(f"{manifest_path}: {subject} artifacts must be a list of asset ids")
        artifacts = tuple(cast("list[str]", artifacts_raw))
        if len(set(artifacts)) != len(artifacts):
            raise ManifestError(f"{manifest_path}: {subject} artifacts contains duplicates")
        for asset_id in artifacts:
            asset = declared_assets.get(asset_id)
            if asset is None:
                raise ManifestError(
                    f"{manifest_path}: {subject} references unknown [[pack.assets]] id {asset_id!r}"
                )
        model = table.get("model")
        if model is not None and (not isinstance(model, str) or not model):
            raise ManifestError(f"{manifest_path}: {subject} model must be a non-empty string")
        providers.append(
            VisionProvider(
                choice=choice,
                node=node,
                devices=devices,
                dtypes=dtypes,
                batching=cast("str", batching),
                artifacts=artifacts,
                model=model,
            )
        )
    return tuple(providers)


def _parse_generation_providers(
    raw: object,
    *,
    manifest_path: Path,
    executes: tuple[str, ...],
) -> tuple[GenerationProvider, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ManifestError(
            f"{manifest_path}: [[pack.generation-providers]] must be a non-empty array of tables"
        )
    providers: list[GenerationProvider] = []
    seen_nodes: set[str] = set()
    required_fields = {"choice", "node"}
    expected_fields = required_fields | {"label"}
    for index, entry in enumerate(cast("list[object]", raw)):
        subject = f"[[pack.generation-providers]] entry {index}"
        if not isinstance(entry, dict):
            raise ManifestError(f"{manifest_path}: {subject} must be a table")
        table = cast("dict[str, object]", entry)
        unknown = sorted(set(table) - expected_fields)
        missing = sorted(required_fields - set(table))
        if unknown:
            raise ManifestError(f"{manifest_path}: {subject} unknown fields: {', '.join(unknown)}")
        if missing:
            raise ManifestError(f"{manifest_path}: {subject} missing fields: {', '.join(missing)}")
        choice = _canonical_namespaced_id(
            table["choice"], field=f"{subject} choice", manifest_path=manifest_path
        )
        node = table["node"]
        if not isinstance(node, str) or not node:
            raise ManifestError(f"{manifest_path}: {subject} node must be a non-empty string")
        if node not in executes:
            raise ManifestError(
                f"{manifest_path}: {subject} node {node!r} must appear in [pack] executes"
            )
        if node in seen_nodes:
            raise ManifestError(
                f"{manifest_path}: [[pack.generation-providers]] repeats node {node!r}"
            )
        seen_nodes.add(node)
        label = table.get("label")
        if label is not None and (not isinstance(label, str) or not label):
            raise ManifestError(f"{manifest_path}: {subject} label must be a non-empty string")
        providers.append(GenerationProvider(choice=choice, node=node, label=label))
    return tuple(providers)


def _parse_extension(raw: object, manifest_path: Path) -> ExtensionDeclaration:
    """Parse the additive [pack.extension] declaration without importing it."""
    if raw is None:
        return ExtensionDeclaration()
    if not isinstance(raw, dict):
        raise ManifestError(f"{manifest_path}: [pack.extension] must be a table")
    table = cast("dict[str, object]", raw)
    allowed = set(EXTENSION_SCOPES) | {
        "privileges",
        "capabilities",
        "routes",
        "events",
        "frontend-modules",
    }
    unknown_fields = sorted(set(table) - allowed)
    if unknown_fields:
        raise ManifestError(
            f"{manifest_path}: [pack.extension] unknown fields: {', '.join(unknown_fields)}"
        )

    entry_values: dict[str, str | None] = {}
    for scope in EXTENSION_SCOPES:
        value = table.get(scope)
        if value is not None and (
            not isinstance(value, str)
            or value.count(":") != 1
            or not all(value.split(":"))
            or any(char.isspace() for char in value)
        ):
            raise ManifestError(
                f"{manifest_path}: [pack.extension] {scope} must be a 'module:attr' string"
            )
        entry_values[scope] = value

    privileges = _parse_extension_vocabulary(
        table.get("privileges"),
        field="privileges",
        known=EXTENSION_SCOPES,
        manifest_path=manifest_path,
    )
    capabilities = _parse_extension_vocabulary(
        table.get("capabilities"),
        field="capabilities",
        known=EXTENSION_CAPABILITIES,
        manifest_path=manifest_path,
    )
    missing_privileges = sorted(
        scope
        for scope, value in entry_values.items()
        if value is not None and scope not in privileges
    )
    if missing_privileges:
        raise ManifestError(
            f"{manifest_path}: [pack.extension] entries require matching privileges: "
            f"{', '.join(missing_privileges)}"
        )
    try:
        routes, events = pack_surfaces_from_wire(table)
        modules = table.get("frontend-modules", [])
        if not isinstance(modules, list):
            raise ValueError("frontend-modules must be an array")
        return ExtensionDeclaration(
            entries=ExtensionEntryPoints(**entry_values),
            privileges=privileges,
            capabilities=capabilities,
            routes=routes,
            events=events,
            frontend_modules=tuple(
                FrontendModule.from_manifest(item) for item in cast(list[object], modules)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{manifest_path}: [pack.extension] {exc}") from exc


def _parse_extension_vocabulary(
    raw: object,
    *,
    field: str,
    known: tuple[str, ...],
    manifest_path: Path,
) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(
        isinstance(item, str) for item in cast("list[object]", raw)
    ):
        raise ManifestError(f"{manifest_path}: [pack.extension] {field} must be a list of strings")
    items = cast("list[str]", raw)
    declared = set(items)
    if len(items) != len(declared):
        duplicates = tuple(sorted(value for value in declared if items.count(value) > 1))
        raise ManifestError(
            f"{manifest_path}: [pack.extension] duplicate {field}: {', '.join(duplicates)}"
        )
    unknown = tuple(sorted(declared - set(known)))
    if unknown:
        raise ManifestError(
            f"{manifest_path}: [pack.extension] unknown {field}: {', '.join(unknown)}; "
            f"known values: {', '.join(known)}"
        )
    return tuple(value for value in known if value in declared)


def _parse_sandbox(raw: object, manifest_path: Path) -> PackSandboxNeeds:
    if raw is None:
        return PackSandboxNeeds()
    if not isinstance(raw, dict):
        raise ManifestError(f"{manifest_path}: [pack.sandbox] must be a table")
    table = cast("dict[object, object]", raw)
    known = {"gpu", "network", "writable-mounts"}
    unknown = sorted(str(key) for key in set(table) - known)
    if unknown:
        raise ManifestError(f"{manifest_path}: [pack.sandbox] unknown fields: {', '.join(unknown)}")
    values: dict[str, bool] = {}
    for field in known:
        value = table.get(field, False)
        if type(value) is not bool:
            raise ManifestError(f"{manifest_path}: [pack.sandbox] {field} must be a boolean")
        values[field] = value
    return PackSandboxNeeds(
        gpu=values["gpu"],
        network=values["network"],
        writable_mounts=values["writable-mounts"],
    )


def load_manifest(path: Path | str) -> PackManifest:
    manifest_path = Path(path)
    try:
        with open(manifest_path, "rb") as fh:
            document = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ManifestError(f"pack manifest not found: {manifest_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"invalid TOML in {manifest_path}: {exc}") from exc

    pack = document.get("pack")
    if not isinstance(pack, dict):
        raise ManifestError(f"{manifest_path}: missing [pack] table")
    pack = cast("dict[str, object]", pack)

    name = pack.get("name")
    if not isinstance(name, str) or not name:
        raise ManifestError(f"{manifest_path}: [pack] name must be a non-empty string")
    name_problem = validate_name(name)
    if name_problem is not None:
        raise ManifestError(f"{manifest_path}: [pack] name {name!r} {name_problem}")

    namespaces = _parse_namespaces(pack.get("namespaces"), name, manifest_path)

    entry = pack.get("entry")
    if not isinstance(entry, dict):
        raise ManifestError(f"{manifest_path}: missing [pack.entry] table")
    entry = cast("dict[str, object]", entry)

    nodes_entry = entry.get("nodes")
    if not isinstance(nodes_entry, str) or ":" not in nodes_entry:
        raise ManifestError(f"{manifest_path}: [pack.entry] nodes must be a 'module:attr' string")
    types_entry = entry.get("types")
    if types_entry is not None and (not isinstance(types_entry, str) or ":" not in types_entry):
        raise ManifestError(f"{manifest_path}: [pack.entry] types must be a 'module:attr' string")
    reservations_entry = entry.get("reservations")
    if reservations_entry is not None and (
        not isinstance(reservations_entry, str) or ":" not in reservations_entry
    ):
        raise ManifestError(
            f"{manifest_path}: [pack.entry] reservations must be a 'module:attr' string"
        )
    consumers_entry = entry.get("consumers")
    if consumers_entry is not None and (
        not isinstance(consumers_entry, str) or ":" not in consumers_entry
    ):
        raise ManifestError(
            f"{manifest_path}: [pack.entry] consumers must be a 'module:attr' string"
        )
    choices_entry = entry.get("choices")
    if choices_entry is not None and (
        not isinstance(choices_entry, str) or ":" not in choices_entry
    ):
        raise ManifestError(f"{manifest_path}: [pack.entry] choices must be a 'module:attr' string")
    skips_entry = entry.get("skips")
    if skips_entry is not None and (not isinstance(skips_entry, str) or ":" not in skips_entry):
        raise ManifestError(f"{manifest_path}: [pack.entry] skips must be a 'module:attr' string")
    telemetry_entry = entry.get("telemetry")
    if telemetry_entry is not None and (
        not isinstance(telemetry_entry, str) or ":" not in telemetry_entry
    ):
        raise ManifestError(
            f"{manifest_path}: [pack.entry] telemetry must be a 'module:attr' string"
        )
    schema_reload_entry = entry.get("schema_reload")
    if schema_reload_entry is not None and (
        not isinstance(schema_reload_entry, str) or ":" not in schema_reload_entry
    ):
        raise ManifestError(
            f"{manifest_path}: [pack.entry] schema_reload must be a 'module:attr' string"
        )
    source_staging_entry = entry.get("source_staging")
    if source_staging_entry is not None and (
        not isinstance(source_staging_entry, str) or ":" not in source_staging_entry
    ):
        raise ManifestError(
            f"{manifest_path}: [pack.entry] source_staging must be a 'module:attr' string"
        )
    arm_nodes_entry = entry.get("arm_nodes")
    if arm_nodes_entry is not None and (
        not isinstance(arm_nodes_entry, str) or ":" not in arm_nodes_entry
    ):
        raise ManifestError(
            f"{manifest_path}: [pack.entry] arm_nodes must be a 'module:attr' string"
        )
    workgroup_handler_entry = entry.get("workgroup_handler")
    if workgroup_handler_entry is not None and (
        not isinstance(workgroup_handler_entry, str)
        or workgroup_handler_entry.count(":") != 1
        or not all(workgroup_handler_entry.split(":"))
        or any(char.isspace() for char in workgroup_handler_entry)
    ):
        raise ManifestError(
            f"{manifest_path}: [pack.entry] workgroup_handler must be a 'module:attr' string"
        )
    executes = _parse_executes(pack.get("executes"), manifest_path)
    if not namespaces and not executes:
        raise ManifestError(
            f"{manifest_path}: empty [pack] namespaces requires [pack] executes - "
            f"a pack that claims nothing can only execute node types another "
            f"pack owns"
        )
    schema_only = _parse_schema_only(pack.get("schema-only"), namespaces, manifest_path)
    overlap = set(executes) & set(schema_only)
    if overlap:
        raise ManifestError(
            f"{manifest_path}: node types cannot appear in both [pack] executes "
            f"and schema-only: {', '.join(sorted(overlap))}"
        )
    arms = _parse_arms(pack.get("arms"), namespaces, executes, manifest_path)
    armed = {node_type for _arm, node_types in arms for node_type in node_types}
    overlap = armed & set(schema_only)
    if overlap:
        raise ManifestError(
            f"{manifest_path}: schema-only node types cannot declare [pack.arms] bodies: "
            f"{', '.join(sorted(overlap))}"
        )
    if bool(arms) != (arm_nodes_entry is not None):
        raise ManifestError(
            f"{manifest_path}: [pack.arms] and [pack.entry] arm_nodes must "
            "either both be present or both be omitted"
        )

    requires_raw = pack.get("requires", [])
    if not isinstance(requires_raw, list) or not all(
        isinstance(item, str) for item in cast("list[object]", requires_raw)
    ):
        raise ManifestError(f"{manifest_path}: [pack] requires must be a list of strings")

    assets = _parse_assets(pack.get("assets"), manifest_path, name)
    blueprints = _parse_blueprints(pack.get("blueprints"), manifest_path)
    templates = _parse_templates(
        pack.get("templates"), manifest_path, tuple(asset.id for asset in assets)
    )
    docs, docs_problem = _parse_docs(
        pack.get("docs"),
        manifest_path,
        {blueprint.id for blueprint in blueprints},
        {template.id for template in templates},
    )
    locale_catalogs, locale_catalog_problems = _parse_locale_catalogs(
        manifest_path,
        namespaces,
        {blueprint.id for blueprint in blueprints},
        {page.id for page in (() if docs is None else docs.pages) if page.kind == "guide"},
    )
    return PackManifest(
        name=name,
        namespaces=namespaces,
        nodes_entry=nodes_entry,
        types_entry=types_entry,
        reservations_entry=reservations_entry,
        consumers_entry=consumers_entry,
        choices_entry=choices_entry,
        skips_entry=skips_entry,
        telemetry_entry=telemetry_entry,
        schema_reload_entry=schema_reload_entry,
        arm_nodes_entry=arm_nodes_entry,
        source_staging_entry=source_staging_entry,
        requires=tuple(cast("list[str]", requires_raw)),
        path=manifest_path,
        contracts=_parse_contracts(pack.get("contracts"), manifest_path),
        dependencies=_parse_pack_dependencies(pack.get("dependencies"), manifest_path),
        requirements=_parse_requirements(pack.get("requirements"), manifest_path),
        provides=_parse_provides(pack.get("provides"), manifest_path),
        sandbox=_parse_sandbox(pack.get("sandbox"), manifest_path),
        sandbox_declared="sandbox" in pack,
        capabilities=_parse_capabilities(pack.get("capabilities"), manifest_path),
        vision_providers=_parse_vision_providers(
            pack.get("vision-providers"),
            manifest_path=manifest_path,
            executes=executes,
            assets=assets,
        ),
        generation_providers=_parse_generation_providers(
            pack.get("generation-providers"),
            manifest_path=manifest_path,
            executes=executes,
        ),
        workgroup_handler_entry=workgroup_handler_entry,
        presentation=_parse_presentation(pack.get("presentation"), name, manifest_path),
        blueprints=blueprints,
        assets=assets,
        templates=templates,
        docs=docs,
        docs_problem=docs_problem,
        locale_catalogs=locale_catalogs,
        locale_catalog_problems=locale_catalog_problems,
        platforms=_parse_platforms(pack.get("platforms"), manifest_path),
        extra_requires=_parse_extra_requires(pack.get("extra-requires"), manifest_path),
        executes=executes,
        schema_only=schema_only,
        arms=arms,
        extension=_parse_extension(pack.get("extension"), manifest_path),
        extension_declared="extension" in pack,
        comfy_aliases=load_comfy_aliases(manifest_path),
        comfy_groups=load_comfy_groups(manifest_path),
    )


KNOWN_PLATFORMS = ("darwin", "linux", "win32")
"""``sys.platform`` values ``[pack] platforms`` may declare - the same
vocabulary PEP 508 ``sys_platform`` markers use, so authors never juggle
two OS namings."""

KNOWN_ACCELERATORS = ("cpu", "cuda", "mps", "rocm", "xpu")
"""Accelerator keys ``[pack.extra-requires]`` may declare. A closed set
on purpose: an unknown key would be silently dead on every host, which is
exactly the kind of typo that must fail at load, not at user runtime."""


def _parse_platforms(raw: object, manifest_path: Path) -> tuple[str, ...]:
    """``[pack] platforms``: advisory OS-compatibility claims. Fatal when
    malformed - a typo'd platform list would warn on every host or none,
    both silently wrong."""
    if raw is None:
        return ()
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) for item in cast("list[object]", raw))
    ):
        raise ManifestError(
            f"{manifest_path}: [pack] platforms must be a non-empty list of "
            f"strings (omit it for no OS restriction)"
        )
    platforms = tuple(sorted(set(cast("list[str]", raw))))
    for entry in platforms:
        if entry not in KNOWN_PLATFORMS:
            raise ManifestError(
                f"{manifest_path}: [pack] platforms entry {entry!r} is not a "
                f"sys.platform value; expected one of {', '.join(KNOWN_PLATFORMS)}"
            )
    return platforms


def _parse_extra_requires(
    raw: object, manifest_path: Path
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``[pack.extra-requires]``: accelerator -> extra requirements.
    Fatal when malformed, and keys are validated against the known
    accelerator set - an unknown key would never match any host."""
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ManifestError(
            f"{manifest_path}: [pack.extra-requires] must be a table of "
            f"accelerator -> list of requirement strings"
        )
    entries: list[tuple[str, tuple[str, ...]]] = []
    for key, value in cast("dict[str, object]", raw).items():
        if key not in KNOWN_ACCELERATORS:
            raise ManifestError(
                f"{manifest_path}: [pack.extra-requires] key {key!r} is not a "
                f"known accelerator; expected one of {', '.join(KNOWN_ACCELERATORS)}"
            )
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in cast("list[object]", value)
        ):
            raise ManifestError(
                f"{manifest_path}: [pack.extra-requires] {key} must be a list "
                f"of requirement strings"
            )
        entries.append((key, tuple(cast("list[str]", value))))
    return tuple(sorted(entries))


def _parse_namespaces(raw: object, pack_name: str, manifest_path: Path) -> tuple[str, ...]:
    """Parse ``[pack] namespaces``: grammar-valid, mutually non-overlapping
    claims. Absent means the pack claims its own name (the default).
    Malformed claims are fatal, not warn-and-drop - a claim the loader
    silently dropped would leave the pack's node types uncovered and fail
    composition anyway, with a worse message. Reserved roots are NOT
    rejected here: whether a reserved claim may compose is trust policy
    (the host's, later the registry's), not manifest shape.
    An explicit empty list claims nothing. load_manifest allows that
    shape only alongside a non-empty ``executes`` (doctor, composition,
    and the registry check the actual node-type coverage), so a pure
    executor never collides with the pack that owns the node types it
    implements."""
    if raw is None:
        return (pack_name,)
    if not isinstance(raw, list) or not all(
        isinstance(item, str) for item in cast("list[object]", raw)
    ):
        raise ManifestError(
            f"{manifest_path}: [pack] namespaces must be a list of strings "
            f"(omit it to claim the pack name; an empty list claims nothing)"
        )
    claims = cast("list[str]", raw)
    for claim in claims:
        problem = validate_name(claim)
        if problem is not None:
            raise ManifestError(f"{manifest_path}: [pack] namespaces entry {claim!r} {problem}")
    for index, claim in enumerate(claims):
        for earlier in claims[:index]:
            if claims_conflict(earlier, claim):
                detail = (
                    "are the same claim (separators '-', '_' and '.' are one equivalence class)"
                    if canonical_name(earlier) == canonical_name(claim)
                    else "overlap (one nests inside the other); keep only the broader claim"
                )
                raise ManifestError(
                    f"{manifest_path}: [pack] namespaces entries {earlier!r} and {claim!r} {detail}"
                )
    return tuple(claims)


def _parse_executes(raw: object, manifest_path: Path) -> tuple[str, ...]:
    """Parse ``[pack] executes``: node types the pack claims to implement
    as an alternative executor. Absent means none.
    Entries pass through VERBATIM - node types are not restricted to the
    pack-name grammar (legacy types carry uppercase tails), and whether a
    claim matches the owning pack's schema is enrollment's job, not the
    loader's. Malformed lists are fatal like namespaces, not
    warn-and-drop: a silently dropped claim would surface later as an
    enrollment failure with a worse message."""
    if raw is None:
        return ()
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) and item for item in cast("list[object]", raw))
    ):
        raise ManifestError(
            f"{manifest_path}: [pack] executes must be a non-empty list of "
            f"non-empty node-type strings (omit it to claim none)"
        )
    claims = cast("list[str]", raw)
    seen: set[str] = set()
    for claim in claims:
        if claim in seen:
            raise ManifestError(f"{manifest_path}: [pack] executes lists {claim!r} more than once")
        seen.add(claim)
    return tuple(claims)


def _parse_schema_only(
    raw: object, namespaces: tuple[str, ...], manifest_path: Path
) -> tuple[str, ...]:
    """Parse owned schemas that deliberately publish no execution body."""
    if raw is None:
        return ()
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) and item for item in cast("list[object]", raw))
    ):
        raise ManifestError(
            f"{manifest_path}: [pack] schema-only must be a non-empty list of "
            "non-empty node-type strings (omit it to claim none)"
        )
    node_types = cast("list[str]", raw)
    if len(set(node_types)) != len(node_types):
        raise ManifestError(f"{manifest_path}: [pack] schema-only lists a node type more than once")
    for node_type in node_types:
        if not any(claim_covers(claim, node_type) for claim in namespaces):
            raise ManifestError(
                f"{manifest_path}: [pack] schema-only node type {node_type!r} "
                "is not owned by this pack"
            )
    return tuple(node_types)


_ARM_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _parse_arms(
    raw: object,
    namespaces: tuple[str, ...],
    executes: tuple[str, ...],
    manifest_path: Path,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Parse same-session bodies for schemas this pack owns or executes."""
    if raw is None:
        return ()
    if not isinstance(raw, dict) or not raw:
        raise ManifestError(
            f"{manifest_path}: [pack.arms] must be a non-empty table of "
            "arm name -> non-empty node-type list"
        )
    parsed: list[tuple[str, tuple[str, ...]]] = []
    seen_names: set[str] = set()
    for arm, node_types_raw in cast("dict[object, object]", raw).items():
        if not isinstance(arm, str) or arm == "__proto__" or _ARM_NAME.fullmatch(arm) is None:
            raise ManifestError(
                f"{manifest_path}: [pack.arms] arm name {arm!r} must match "
                "[A-Za-z0-9_-]+ and must not be '__proto__'"
            )
        canonical = arm.replace("_", "-").lower()
        if canonical in seen_names:
            raise ManifestError(f"{manifest_path}: [pack.arms] declares duplicate arm name {arm!r}")
        seen_names.add(canonical)
        if (
            not isinstance(node_types_raw, list)
            or not node_types_raw
            or not all(
                isinstance(item, str) and item for item in cast("list[object]", node_types_raw)
            )
        ):
            raise ManifestError(
                f"{manifest_path}: [pack.arms] {arm!r} must be a non-empty "
                "list of non-empty node-type strings"
            )
        node_types = cast("list[str]", node_types_raw)
        if len(set(node_types)) != len(node_types):
            raise ManifestError(
                f"{manifest_path}: [pack.arms] {arm!r} lists a node type more than once"
            )
        for node_type in node_types:
            if node_type not in executes and not any(
                claim_covers(claim, node_type) for claim in namespaces
            ):
                raise ManifestError(
                    f"{manifest_path}: [pack.arms] {arm!r} node type "
                    f"{node_type!r} is neither owned nor listed in [pack] executes"
                )
        parsed.append((arm, tuple(node_types)))
    return tuple(parsed)


def load_pack_presentation(path: Path | str, *, pack_name: str = "") -> PackPresentation | None:
    """Read ONLY ``[pack.presentation]`` from a ``dinkster-pack.toml``.

    The presentation on-ramp for packs that are not (yet) Dinkster packs: a
    legacy ComfyUI pack can ship (or its user can drop in) a manifest that
    declares nothing but its badge, and a host composing it through a
    compat layer picks the declaration up without the file having to be a
    loadable pack manifest. Everything about it is advisory: a missing
    file is None, and malformed TOML or fields warn-and-drop exactly like
    manifest loading - presentation can never break composition.
    """
    manifest_path = Path(path)
    try:
        with open(manifest_path, "rb") as fh:
            document = tomllib.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, tomllib.TOMLDecodeError) as exc:
        core_logger("workers").warning(
            "%s: unreadable pack presentation file (%s); ignoring it",
            manifest_path,
            exc,
        )
        return None
    pack = document.get("pack")
    if not isinstance(pack, dict):
        return None
    raw = cast("dict[str, object]", pack).get("presentation")
    return _parse_presentation(raw, pack_name, manifest_path)


def _parse_presentation(
    raw: object, pack_name: str, manifest_path: Path
) -> PackPresentation | None:
    """Parse [pack.presentation] with warn-and-drop per field: presentation
    can never stop a pack from loading, and a dropped field is a logged
    diagnostic, never a silent normalization."""
    if raw is None:
        return None
    log = core_logger("workers")
    if not isinstance(raw, dict):
        log.warning("%s: [pack.presentation] must be a table; ignoring it", manifest_path)
        return None
    table = cast("dict[str, object]", raw)

    def field(key: str) -> str | None:
        value = table.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            log.warning(
                "%s: [pack.presentation] %s must be a non-empty string; dropping it",
                manifest_path,
                key,
            )
            return None
        return value

    display_name = field("display_name") or ""

    abbr = field("abbr") or ""
    if abbr and (len(abbr) > _ABBR_MAX or not abbr.isascii() or not abbr.isprintable()):
        log.warning(
            "%s: [pack.presentation] abbr %r must be printable ASCII of at most "
            "%d characters; dropping it",
            manifest_path,
            abbr,
            _ABBR_MAX,
        )
        abbr = ""

    mark = field("mark") or ""
    if mark and (
        len(mark) > _MARK_MAX_CODEPOINTS
        or any(ch.isspace() or unicodedata.category(ch).startswith("C") for ch in mark)
    ):
        log.warning(
            "%s: [pack.presentation] mark %r must be one compact glyph "
            "(<= %d code points, no whitespace/control); dropping it",
            manifest_path,
            mark,
            _MARK_MAX_CODEPOINTS,
        )
        mark = ""

    color = field("color") or ""
    if color and not (
        len(color) == 7 and color[0] == "#" and all(ch in _COLOR_CHARS for ch in color[1:])
    ):
        log.warning(
            "%s: [pack.presentation] color %r must be '#rrggbb'; dropping it",
            manifest_path,
            color,
        )
        color = ""

    icon: PackIcon | None = None
    declared_icon = field("icon") or ""
    if declared_icon:
        icon, problem = validate_pack_icon(manifest_path, declared_icon)
        if problem is not None:
            log.warning(
                "%s: [pack.presentation] icon %r %s; dropping it",
                manifest_path,
                declared_icon,
                problem,
            )

    if not (display_name or abbr or mark or color or icon):
        return None
    return PackPresentation(
        display_name=display_name or pack_name,
        abbr=abbr,
        mark=mark,
        color=color,
        icon=icon,
    )


def resolve_entry(entry: str) -> object:
    """Import a 'module:attr' entry point. Runs pack code - worker hosts
    call this inside their own process, never the engine's (hazard H5)."""
    module_name, _, attr = entry.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ManifestError(f"entry '{entry}': module has no attribute '{attr}'") from exc
