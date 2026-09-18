"""Private read-only HTTP adapter for Federated Asset DTO V1."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from aiohttp import web
from dinkster_assets.acquisition_plan import AcquisitionSource
from dinkster_assets.identity import AssetError
from dinkster_assets.resolution import (
    MountMaterialization,
    ProviderMirror,
    ResolutionSnapshot,
    ResolutionStore,
)

from . import federated_assets_v1 as dto
from .auth import Principal, principal_for, resolve_scope

_CURSOR_TTL_SECONDS = 300
_STATE_KEY: web.AppKey[_CatalogState] = web.AppKey("federated_asset_catalog")


@dataclass(frozen=True, slots=True)
class _CatalogSnapshot:
    generation: str
    items: tuple[dto.CandidateV1, ...]
    aliases: Mapping[tuple[str, str], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _CatalogState:
    store: ResolutionStore
    sources: tuple[AcquisitionSource, ...]
    provider_policy: Mapping[str, frozenset[str]]
    cursor_key: bytes
    clock: Callable[[], float]


class _CursorError(ValueError):
    def __init__(self, reason: Literal["malformed", "query-mismatch", "stale-snapshot", "expired"]):
        self.reason = reason
        super().__init__(reason)


class _AuthorityError(RuntimeError):
    pass


def _error(
    status: int, error: dto.ErrorV1, *, headers: Mapping[str, str] | None = None
) -> web.Response:
    body = dto.encode_error_response(dto.ErrorResponseV1(error), status=status)
    return web.json_response(body, status=status, headers=headers)


async def _request_model(
    request: web.Request,
    decoder: Callable[[object], dto.CatalogRequestV1 | dto.CandidatesRequestV1],
) -> dto.CatalogRequestV1 | dto.CandidatesRequestV1 | web.Response:
    try:
        body = await request.json(loads=json.loads)
        return decoder(body)
    except dto.FederatedAssetCodecError as exc:
        return _error(400, dto.ErrorV1("invalid-request", reason=exc.reason, field=exc.field))
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        return _error(400, dto.ErrorV1("invalid-request", reason="body must be valid JSON"))


def _authorized_scope(
    request: web.Request, explicit_scope: str | None
) -> tuple[Principal, str] | web.Response:
    principal = principal_for(request)
    try:
        scope = resolve_scope(principal, "assets:read", explicit_scope)
    except web.HTTPBadRequest as exc:
        return _error(
            400,
            dto.ErrorV1("invalid-request", reason=exc.reason or "scope required", field="scope"),
        )
    except web.HTTPForbidden:
        return _error(403, dto.ErrorV1("forbidden", reason="scope requires assets:read"))
    return principal, scope


def _provider_requirements(source: AcquisitionSource | None) -> dto.ProviderRequirementsV1:
    if source is None:
        return dto.ProviderRequirementsV1()
    return dto.ProviderRequirementsV1(
        credential=source.credential_requirement_id,
        license=source.license_requirement_id,
        cost=source.cost_requirement_id,
        policy_override=source.policy_override_requirement_id,
    )


def _mime_matches(media_type: str, accepts: tuple[str, ...]) -> bool:
    return not accepts or any(
        accept == media_type or (accept.endswith("/*") and media_type.startswith(accept[:-1]))
        for accept in accepts
    )


def _compatibility(
    context: dto.ResolveContextV1 | None,
    requirements: dto.CandidateRequirementsV1,
    media_type: str | None,
) -> tuple[Literal["compatible", "incompatible", "unknown"], str]:
    if context is None:
        return "unknown", "compatibility context required"
    if media_type is not None and not _mime_matches(media_type, context.accept):
        return "incompatible", "media type not accepted"
    if requirements.loaders or requirements.runtimes or requirements.hardware:
        return "unknown", "runtime compatibility facts unavailable"
    if context.accept and media_type is None:
        return "unknown", "media type unavailable"
    return "compatible", ""


def _mount_kind(row: MountMaterialization) -> str:
    if row.asset_kind:
        return row.asset_kind
    major = row.ref.media_type.partition("/")[0]
    if major in {"audio", "image", "video"}:
        return f"media/{major}"
    return "asset/file"


def _synthetic_logical_id(asset_kind: str, digest: str) -> str:
    algorithm, separator, value = digest.partition(":")
    if not separator:
        raise _AuthorityError("mount digest has no algorithm separator")
    return f"mounted/{asset_kind}/{algorithm}/{value}"


def _visible_snapshot(
    state: _CatalogState,
    scope: str,
    providers: frozenset[str],
    context: dto.ResolveContextV1 | None,
) -> _CatalogSnapshot:
    facts = state.store.snapshot(scope)
    return _adapt_snapshot(facts, state.sources, providers, scope, context)


def _adapt_snapshot(
    facts: ResolutionSnapshot,
    sources: tuple[AcquisitionSource, ...],
    providers: frozenset[str],
    scope: str,
    context: dto.ResolveContextV1 | None,
) -> _CatalogSnapshot:
    logical = {row.logical_id: row for row in facts.logical_models}
    variants = {(row.logical_id, row.variant_id): row for row in facts.variants}
    locals_by_digest = {row.ref.digest: row.ref for row in facts.locals if row.scope == scope}
    mounts_by_kind_digest: dict[tuple[str, str], MountMaterialization] = {}
    for row in sorted(
        facts.mounts,
        key=lambda item: (item.priority, item.mount_id, item.ref.virtual_path),
    ):
        mounts_by_kind_digest.setdefault((_mount_kind(row), row.ref.digest), row)
    source_facts: dict[tuple[str, str, str], AcquisitionSource] = {}
    identities: dict[tuple[str, str], AcquisitionSource] = {}
    for source in sorted(
        sources,
        key=lambda row: (row.source.provider_id, row.source.source_id, row.digest),
    ):
        if source.source.provider_id not in providers:
            continue
        key = (source.source.provider_id, source.source.source_id, source.digest)
        previous = source_facts.get(key)
        if previous is not None and previous != source:
            raise _AuthorityError("conflicting authorized provider source facts")
        identity = (source.source.provider_id, source.source.source_id)
        previous_identity = identities.get(identity)
        if previous_identity is not None and previous_identity != source:
            raise _AuthorityError("authorized provider source identity is ambiguous")
        identities[identity] = source
        source_facts[key] = source

    mirrors_by_source: dict[tuple[str, str, str], ProviderMirror] = {}
    for mirror in facts.mirrors:
        if mirror.source.provider_id in providers:
            mirrors_by_source[
                (mirror.source.provider_id, mirror.source.source_id, mirror.digest)
            ] = mirror

    items: dict[tuple[str, str, str], dto.CandidateV1] = {}
    represented_mounts: set[tuple[str, str]] = set()

    def add_item(item: dto.CandidateV1) -> None:
        key = (item.logical_id, item.variant_id, item.digest)
        previous = items.get(key)
        if previous is not None and previous != item:
            raise _AuthorityError("candidate identity has conflicting facts")
        items[key] = item

    for link in facts.links:
        model = logical.get(link.logical_id)
        variant = variants.get((link.logical_id, link.variant_id))
        if model is None or variant is None:
            continue
        represented_mounts.add((model.asset_kind, link.digest))
        ref = locals_by_digest.get(link.digest)
        if ref is None:
            mount = mounts_by_kind_digest.get((model.asset_kind, link.digest))
            ref = mount.ref if mount is not None else None
        authorities = sorted(
            (
                source
                for source in source_facts.values()
                if source.digest == link.digest and source.asset_kind == model.asset_kind
            ),
            key=lambda row: (row.source.provider_id, row.source.source_id),
        )
        if len(authorities) > 64:
            raise _AuthorityError("candidate has more than 64 authorized provider sources")
        provider_sources: list[dto.ProviderSourceV1] = []
        media_type = ref.media_type if ref is not None else None
        size = ref.size if ref is not None else None
        for source in authorities:
            mirror = mirrors_by_source.get(
                (source.source.provider_id, source.source.source_id, source.digest)
            )
            if size is None:
                size = source.expected_size
            elif size != source.expected_size:
                raise _AuthorityError(
                    "authorized provider size conflicts with local materialization"
                )
            if media_type is None:
                media_type = source.media_type
            elif media_type != source.media_type:
                raise _AuthorityError(
                    "authorized provider media type conflicts with local materialization"
                )
            available = (
                source.state == "available" and mirror is not None and mirror.state == "available"
            )
            provider_sources.append(
                dto.ProviderSourceV1(
                    source=dto.SourceIdentityV1(source.source.provider_id, source.source.source_id),
                    status="available" if available else "unavailable",
                    reason=(
                        ""
                        if available
                        else source.unavailable_reason
                        or (mirror.reason if mirror is not None else "")
                        or "provider mirror unavailable"
                    ),
                    requires=_provider_requirements(source),
                )
            )
        if ref is not None:
            availability: Literal["local", "downloadable", "unavailable"] = "local"
            availability_reason = "local materialization"
        elif any(source.status == "available" for source in provider_sources):
            availability = "downloadable"
            availability_reason = "authorized provider mirror available"
        else:
            availability = "unavailable"
            reasons = sorted({source.reason for source in provider_sources if source.reason})
            availability_reason = (
                reasons[0] if reasons else "no local or authorized provider materialization"
            )
        requirements = dto.CandidateRequirementsV1(
            tuple(sorted(variant.requirements.loaders)),
            tuple(sorted(variant.requirements.runtimes)),
            tuple(sorted(variant.requirements.hardware)),
        )
        compatibility, compatibility_reason = _compatibility(context, requirements, media_type)
        add_item(
            dto.CandidateV1(
                logical_id=model.logical_id,
                family=model.family,
                asset_kind=model.asset_kind,
                variant_id=variant.variant_id,
                dtype=variant.dtype,
                quantization=variant.quantization,
                format=variant.format,
                role=variant.role,
                requirements=requirements,
                digest=link.digest,
                size=size,
                media_type=media_type,
                availability_status=availability,
                availability_reason=availability_reason,
                compatibility_status=compatibility,
                compatibility_reason=compatibility_reason,
                asset_ref=ref,
                provider_sources=tuple(provider_sources),
            )
        )
    empty_requirements = dto.CandidateRequirementsV1()
    for (asset_kind, digest), mount in mounts_by_kind_digest.items():
        if (asset_kind, digest) in represented_mounts:
            continue
        ref = mount.ref
        add_item(
            dto.CandidateV1(
                logical_id=_synthetic_logical_id(asset_kind, digest),
                family="unclassified",
                asset_kind=asset_kind,
                variant_id="unclassified",
                dtype="unclassified",
                quantization="none",
                format="unclassified",
                role="asset",
                requirements=empty_requirements,
                digest=digest,
                size=ref.size,
                media_type=ref.media_type,
                availability_status="local",
                availability_reason="ready local mount",
                compatibility_status="compatible",
                compatibility_reason="",
                asset_ref=ref,
                provider_sources=(),
            )
        )
    ordered = tuple(items[key] for key in sorted(items))
    aliases: dict[tuple[str, str], list[str]] = {}
    for alias in facts.aliases:
        key = (alias.logical_id, alias.variant_id or "")
        aliases.setdefault(key, []).append(alias.alias)
    frozen_aliases = MappingProxyType(
        {key: tuple(sorted(values)) for key, values in sorted(aliases.items())}
    )
    generation_input = {
        "items": [dto.encode_candidate(item) for item in ordered],
        "aliases": [[*key, *values] for key, values in frozen_aliases.items()],
    }
    generation = hashlib.sha256(
        json.dumps(generation_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _CatalogSnapshot(generation, ordered, frozen_aliases)


def _query_binding(value: dto.CatalogRequestV1 | dto.CandidatesRequestV1) -> str:
    if isinstance(value, dto.CatalogRequestV1):
        wire = dto.encode_catalog_request(value)
    else:
        wire = dto.encode_candidates_request(value)
        wire.pop("hints", None)
    wire.pop("scope", None)
    wire.pop("cursor", None)
    wire.pop("limit", None)
    if isinstance(value, dto.CatalogRequestV1):
        if "query" in wire:
            wire["query"] = cast(str, wire["query"]).casefold()
            if wire["query"] == "":
                wire.pop("query")
        complete_filters = {
            "availability": {"local", "downloadable", "unavailable"},
            "compatibility": {"compatible", "incompatible", "unknown"},
        }
        for key, complete in complete_filters.items():
            if key in wire:
                values = set(cast(list[str], wire[key]))
                if values == complete:
                    wire.pop(key)
                else:
                    wire[key] = sorted(values)
    context = wire.get("context")
    if type(context) is dict:
        context_object = cast(dict[str, object], context)
        accepts = context_object.get("accept")
        if isinstance(accepts, list):
            context_object["accept"] = sorted(set(cast(list[str], accepts)))
    return hashlib.sha256(
        json.dumps(wire, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _encode_cursor(
    state: _CatalogState,
    *,
    principal: Principal,
    scope: str,
    query: str,
    generation: str,
    offset: int,
) -> str:
    body = json.dumps(
        {
            "v": 1,
            "p": principal.principal_id,
            "s": scope,
            "q": query,
            "g": generation,
            "o": offset,
            "e": int(state.clock()) + _CURSOR_TTL_SECONDS,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    signature = hmac.new(state.cursor_key, body, hashlib.sha256).digest()
    return ".".join(
        base64.urlsafe_b64encode(part).rstrip(b"=").decode() for part in (body, signature)
    )


def _decode_cursor(
    state: _CatalogState,
    cursor: str | None,
    *,
    principal: Principal,
    scope: str,
    query: str,
    generation: str,
) -> int:
    if cursor is None:
        return 0
    try:
        body_token, signature_token = cursor.split(".")
        body = _decode_cursor_part(body_token)
        signature = _decode_cursor_part(signature_token)
        if not hmac.compare_digest(
            signature, hmac.new(state.cursor_key, body, hashlib.sha256).digest()
        ):
            raise _CursorError("malformed")
        raw_payload: object = json.loads(body)
        if type(raw_payload) is not dict:
            raise _CursorError("malformed")
        payload = cast(dict[str, object], raw_payload)
        if set(payload) != {"v", "p", "s", "q", "g", "o", "e"}:
            raise _CursorError("malformed")
        if payload["v"] != 1 or type(payload["o"]) is not int or payload["o"] < 0:
            raise _CursorError("malformed")
        if type(payload["e"]) is not int:
            raise _CursorError("malformed")
    except _CursorError:
        raise
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _CursorError("malformed") from exc
    if payload["e"] <= state.clock():
        raise _CursorError("expired")
    if (payload["p"], payload["s"], payload["q"]) != (principal.principal_id, scope, query):
        raise _CursorError("query-mismatch")
    if payload["g"] != generation:
        raise _CursorError("stale-snapshot")
    offset = payload["o"]
    assert type(offset) is int
    return offset


def _decode_cursor_part(token: str) -> bytes:
    if not token or "=" in token:
        raise _CursorError("malformed")
    try:
        decoded = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise _CursorError("malformed") from exc
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode()
    if canonical != token:
        raise _CursorError("malformed")
    return decoded


def _page(
    state: _CatalogState,
    items: tuple[dto.CandidateV1, ...],
    *,
    cursor: str | None,
    limit: int,
    principal: Principal,
    scope: str,
    query: str,
    generation: str,
) -> tuple[tuple[dto.CandidateV1, ...], str | None]:
    offset = _decode_cursor(
        state,
        cursor,
        principal=principal,
        scope=scope,
        query=query,
        generation=generation,
    )
    if offset > len(items):
        raise _CursorError("query-mismatch")
    page = items[offset : offset + limit]
    next_offset = offset + len(page)
    next_cursor = (
        _encode_cursor(
            state,
            principal=principal,
            scope=scope,
            query=query,
            generation=generation,
            offset=next_offset,
        )
        if next_offset < len(items)
        else None
    )
    return page, next_cursor


def _catalog_filter(
    request: dto.CatalogRequestV1, snapshot: _CatalogSnapshot
) -> tuple[dto.CandidateV1, ...]:
    query = request.query.casefold() if request.query is not None else None
    result: list[dto.CandidateV1] = []
    for item in snapshot.items:
        if request.asset_kind is not None and item.asset_kind != request.asset_kind:
            continue
        if request.context is not None and item.asset_kind != request.context.asset_kind:
            continue
        if (
            request.availability is not None
            and item.availability_status not in request.availability
        ):
            continue
        if (
            request.compatibility is not None
            and item.compatibility_status not in request.compatibility
        ):
            continue
        aliases = snapshot.aliases.get(
            (item.logical_id, item.variant_id), ()
        ) + snapshot.aliases.get((item.logical_id, ""), ())
        if query is not None and not any(
            query in value.casefold() for value in (item.logical_id, item.family, *aliases)
        ):
            continue
        result.append(item)
    return tuple(result)


def _selected(
    request: dto.CandidatesRequestV1, items: tuple[dto.CandidateV1, ...]
) -> tuple[
    Literal["resolved", "missing", "ambiguous", "incompatible"],
    dto.SelectedCandidateV1 | None,
]:
    source: dto.SourceIdentityV1 | None = None
    if request.selection is not None:
        pool = tuple(
            item
            for item in items
            if (item.logical_id, item.variant_id, item.digest)
            == (
                request.selection.logical_id,
                request.selection.variant_id,
                request.selection.digest,
            )
        )
        source = request.selection.source
        reason = "explicit-selection"
    elif request.expected is not None and request.expected.digest is not None:
        pool = tuple(item for item in items if item.digest == request.expected.digest)
        reason = "expected-digest"
    else:
        pool = items
        reason = "compatible-only"

    compatible = tuple(item for item in pool if item.compatibility_status == "compatible")
    if len(compatible) > 1:
        return "ambiguous", None
    if compatible:
        chosen = compatible[0]
    elif len(pool) == 1:
        chosen = pool[0]
    elif pool:
        return "ambiguous", None
    else:
        return "missing", None
    selected = dto.SelectedCandidateV1(
        chosen.logical_id, chosen.variant_id, chosen.digest, reason, source
    )
    if chosen.compatibility_status == "incompatible":
        return "incompatible", selected
    if chosen.availability_status == "local" and chosen.compatibility_status == "compatible":
        return "resolved", selected
    return "missing", selected


def _selection_error(
    request: dto.CandidatesRequestV1,
    snapshot: _CatalogSnapshot,
    providers: frozenset[str],
) -> web.Response | None:
    if request.selection is not None:
        selected_source = request.selection.source
        if selected_source is not None and selected_source.provider_id not in providers:
            return _error(
                403,
                dto.ErrorV1("forbidden", reason="selected provider unauthorized"),
            )
        exact = next(
            (
                item
                for item in snapshot.items
                if (item.logical_id, item.variant_id, item.digest)
                == (
                    request.selection.logical_id,
                    request.selection.variant_id,
                    request.selection.digest,
                )
            ),
            None,
        )
        if exact is not None and exact.asset_kind != request.context.asset_kind:
            return _error(
                422,
                dto.ErrorV1(
                    "wrong-kind",
                    expected_kind=request.context.asset_kind,
                    actual_kind=exact.asset_kind,
                ),
            )
        if exact is not None and selected_source is not None:
            source = next(
                (row for row in exact.provider_sources if row.source == selected_source),
                None,
            )
            if source is None or source.status != "available":
                return _error(
                    409,
                    dto.ErrorV1(
                        "source-unavailable",
                        reason="selected source unavailable",
                        selection=request.selection,
                    ),
                )
    if request.expected is not None and request.expected.digest is not None:
        expected_digest = request.expected.digest
        requested_kind = any(
            item.digest == expected_digest and item.asset_kind == request.context.asset_kind
            for item in snapshot.items
        )
        other_kind = next(
            (
                item
                for item in snapshot.items
                if item.digest == expected_digest and item.asset_kind != request.context.asset_kind
            ),
            None,
        )
        if not requested_kind and other_kind is not None:
            return _error(
                422,
                dto.ErrorV1(
                    "wrong-kind",
                    expected_kind=request.context.asset_kind,
                    actual_kind=other_kind.asset_kind,
                ),
            )
    return None


async def _catalog(request: web.Request) -> web.Response:
    decoded = await _request_model(request, dto.decode_catalog_request)
    if isinstance(decoded, web.Response):
        return decoded
    assert isinstance(decoded, dto.CatalogRequestV1)
    authorized = _authorized_scope(request, decoded.scope)
    if isinstance(authorized, web.Response):
        return authorized
    principal, scope = authorized
    state = request.app[_STATE_KEY]
    providers = state.provider_policy.get(scope)
    if providers is None:
        return _error(
            403,
            dto.ErrorV1("forbidden", reason="scope not authorized by federated asset policy"),
        )
    try:
        snapshot = _visible_snapshot(state, scope, providers, decoded.context)
        items = _catalog_filter(decoded, snapshot)
        page, next_cursor = _page(
            state,
            items,
            cursor=decoded.cursor,
            limit=decoded.limit,
            principal=principal,
            scope=scope,
            query=_query_binding(decoded),
            generation=snapshot.generation,
        )
        response = dto.CatalogResponseV1(page, next_cursor)
        return web.json_response(dto.encode_catalog_response(response))
    except _CursorError as exc:
        return _error(400, dto.ErrorV1("cursor-invalid", reason=exc.reason))
    except (_AuthorityError, AssetError, sqlite3.Error, dto.FederatedAssetCodecError):
        return _error(503, dto.ErrorV1("service-unavailable", reason="catalog-authority-conflict"))


async def _candidates(request: web.Request) -> web.Response:
    decoded = await _request_model(request, dto.decode_candidates_request)
    if isinstance(decoded, web.Response):
        return decoded
    assert isinstance(decoded, dto.CandidatesRequestV1)
    authorized = _authorized_scope(request, decoded.scope)
    if isinstance(authorized, web.Response):
        return authorized
    principal, scope = authorized
    state = request.app[_STATE_KEY]
    providers = state.provider_policy.get(scope)
    if providers is None:
        return _error(
            403,
            dto.ErrorV1("forbidden", reason="scope not authorized by federated asset policy"),
        )
    try:
        snapshot = _visible_snapshot(state, scope, providers, decoded.context)
        selection_error = _selection_error(decoded, snapshot, providers)
        if selection_error is not None:
            return selection_error
        items = tuple(
            item for item in snapshot.items if item.asset_kind == decoded.context.asset_kind
        )
        status, selected = _selected(decoded, items)
        if (
            decoded.expected is not None
            and decoded.expected.size is not None
            and selected is not None
        ):
            selected_item = next(
                item
                for item in items
                if (item.logical_id, item.variant_id, item.digest)
                == (selected.logical_id, selected.variant_id, selected.digest)
            )
            if selected_item.size is not None and selected_item.size != decoded.expected.size:
                return _error(
                    422,
                    dto.ErrorV1(
                        "integrity-mismatch",
                        expected_size=decoded.expected.size,
                        observed_size=selected_item.size,
                    ),
                )
        page, next_cursor = _page(
            state,
            items,
            cursor=decoded.cursor,
            limit=decoded.limit,
            principal=principal,
            scope=scope,
            query=_query_binding(decoded),
            generation=snapshot.generation,
        )
        response = dto.CandidatesResponseV1(status, page, selected, next_cursor)
        return web.json_response(dto.encode_candidates_response(response))
    except _CursorError as exc:
        return _error(400, dto.ErrorV1("cursor-invalid", reason=exc.reason))
    except (_AuthorityError, AssetError, sqlite3.Error, dto.FederatedAssetCodecError):
        return _error(503, dto.ErrorV1("service-unavailable", reason="catalog-authority-conflict"))


def add_federated_asset_routes(
    app: web.Application,
    *,
    paths: Mapping[str, str],
    store: ResolutionStore,
    sources: Sequence[AcquisitionSource],
    provider_policy: Mapping[str, frozenset[str]],
    cursor_key: bytes | None = None,
    clock: Callable[[], float] | None = None,
) -> Mapping[tuple[str, str], str]:
    if set(paths) != {"catalog", "candidates"}:
        raise ValueError("federated asset paths must contain exactly catalog and candidates")
    catalog_path = paths["catalog"]
    candidates_path = paths["candidates"]
    if any(not path.startswith("/") or "{" in path or "}" in path for path in paths.values()):
        raise ValueError("federated asset paths must be fixed absolute paths")
    if catalog_path == candidates_path:
        raise ValueError("federated asset catalog and candidates paths must differ")
    key = secrets.token_bytes(32) if cursor_key is None else bytes(cursor_key)
    if len(key) < 32:
        raise ValueError("federated asset cursor key must contain at least 32 bytes")
    frozen_policy = MappingProxyType(
        {scope: frozenset(providers) for scope, providers in provider_policy.items()}
    )
    app[_STATE_KEY] = _CatalogState(
        store,
        tuple(sources),
        frozen_policy,
        key,
        clock or time.time,
    )
    app.router.add_post(catalog_path, _catalog)
    app.router.add_post(candidates_path, _candidates)
    return {
        ("POST", catalog_path): "assets:read",
        ("POST", candidates_path): "assets:read",
    }
