"""Pure models and JSON codecs for the private Federated Asset DTO V1 seam.

This module deliberately contains no routes, authentication, persistence,
resolution, acquisition, or cursor signing. It validates the frozen wire
grammar at construction and JSON trust boundaries so later route code cannot
represent a malformed DTO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal, Never, TypeAlias, TypeVar, cast

from dinkster_assets.model import AssetRef

JsonObject: TypeAlias = dict[str, object]
_T = TypeVar("_T")

_DIGEST = re.compile(r"blake3:[0-9a-f]{64}")
_MIME = re.compile(r"[A-Za-z0-9!#$&^_.+-]+/(?:[A-Za-z0-9!#$&^_.+-]+|\*)")
_POINTER = re.compile(r"(?:/(?:[^~/]|~[01])*)*")
_SAFE_INTEGER: Final = 2**53 - 1
_SELECTED_REASONS = frozenset(
    {
        "reference-mapping",
        "explicit-selection",
        "trusted-source",
        "expected-digest",
        "compatible-only",
    }
)


class FederatedAssetCodecError(ValueError):
    """One closed DTO value is invalid at ``field``."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


def _fail(field: str, reason: str) -> Never:
    raise FederatedAssetCodecError(field, reason)


def _object(
    value: object,
    field: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> JsonObject:
    if type(value) is not dict:
        _fail(field, "must be a plain JSON object")
    obj = cast(JsonObject, value)
    if any(type(key) is not str for key in obj):
        _fail(field, "keys must be strings")
    keys = set(obj)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        name = sorted(missing)[0]
        missing_field = name if field in {"request", "response"} else f"{field}.{name}"
        _fail(missing_field, "is required")
    if unknown:
        _fail(field, f"unknown field {sorted(unknown)[0]!r}")
    return obj


def _string(
    value: object,
    field: str,
    *,
    nonempty: bool = False,
    maximum: int | None = None,
    control_free: bool = False,
    printable: bool = False,
) -> str:
    if type(value) is not str:
        _fail(field, "must be a string")
    result = value
    if nonempty and not result:
        _fail(field, "must be nonempty")
    if maximum is not None and len(result) > maximum:
        _fail(field, f"must contain at most {maximum} characters")
    if control_free and any(ord(char) < 32 or ord(char) == 127 for char in result):
        _fail(field, "must not contain control characters")
    if printable and any(not char.isprintable() for char in result):
        _fail(field, "must be printable")
    return result


def _identifier(value: object, field: str, *, maximum: int | None = None) -> str:
    return _string(value, field, nonempty=True, maximum=maximum, control_free=True)


def _integer(value: object, field: str, *, minimum: int = 0, maximum: int = _SAFE_INTEGER) -> int:
    if type(value) is not int:
        _fail(field, "must be an integer")
    result = value
    if not minimum <= result <= maximum:
        _fail(field, f"must be between {minimum} and {maximum}")
    return result


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        _fail(field, "must be a boolean")
    return value


def _literal(value: object, field: str, allowed: frozenset[str]) -> str:
    result = _string(value, field)
    if result not in allowed:
        _fail(field, f"must be one of {sorted(allowed)!r}")
    return result


def _digest(value: object, field: str) -> str:
    result = _string(value, field)
    if _DIGEST.fullmatch(result) is None:
        _fail(field, "must be a canonical BLAKE3 digest")
    return result


def _scope(value: object, field: str) -> str:
    result = _string(value, field, nonempty=True)
    if result != result.strip() or any(char.isspace() for char in result):
        _fail(field, "must be a nonempty whitespace-free string")
    return result


def _reason(value: object, field: str, *, nonempty: bool | None = None) -> str:
    result = _string(value, field, maximum=256, control_free=True)
    if nonempty is True and not result:
        _fail(field, "must be nonempty")
    if nonempty is False and result:
        _fail(field, "must be empty")
    return result


def _tuple(
    value: object,
    field: str,
    *,
    maximum: int | None = None,
    json: bool = False,
) -> tuple[object, ...]:
    allowed = (list,) if json else (list, tuple)
    if type(value) not in allowed:
        _fail(field, "must be an array")
    result = tuple(cast(list[object] | tuple[object, ...], value))
    if maximum is not None and len(result) > maximum:
        _fail(field, f"must contain at most {maximum} items")
    return result


def _copy_collection(
    value: list[_T] | tuple[_T, ...], field: str, *, maximum: int | None = None
) -> tuple[_T, ...]:
    """Copy one constructor collection while rejecting arbitrary iterables."""
    return cast(tuple[_T, ...], _tuple(value, field, maximum=maximum))


def _string_tuple(
    value: object,
    field: str,
    *,
    maximum: int | None = None,
    item_maximum: int | None = None,
    printable: bool = False,
    json: bool = False,
) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{field}[{index}]", maximum=item_maximum, printable=printable)
        for index, item in enumerate(_tuple(value, field, maximum=maximum, json=json))
    )


def _contract_version(value: object, field: str) -> None:
    if type(value) is not int or value != 1:
        _fail(field, "must be the integer 1")


def _asset_ref(value: object, field: str) -> AssetRef:
    obj = _object(
        value,
        field,
        required=frozenset({"digest", "name", "size", "mediaType", "virtualPath"}),
    )
    return AssetRef(
        digest=_digest(obj["digest"], f"{field}.digest"),
        name=_string(obj["name"], f"{field}.name"),
        size=_integer(obj["size"], f"{field}.size"),
        media_type=_string(obj["mediaType"], f"{field}.mediaType"),
        virtual_path=_string(obj["virtualPath"], f"{field}.virtualPath"),
    )


def _validate_asset_ref(value: AssetRef, field: str) -> AssetRef:
    if type(value) is not AssetRef:
        _fail(field, "must be an AssetRef")
    _digest(value.digest, f"{field}.digest")
    _string(value.name, f"{field}.name")
    _integer(value.size, f"{field}.size")
    _string(value.media_type, f"{field}.mediaType")
    _string(value.virtual_path, f"{field}.virtualPath")
    if value.resolver is not None:
        _fail(field, "must not carry a resolver")
    return value


def encode_asset_ref(value: AssetRef) -> JsonObject:
    ref = _validate_asset_ref(value, "assetRef")
    return {
        "digest": ref.digest,
        "name": ref.name,
        "size": ref.size,
        "mediaType": ref.media_type,
        "virtualPath": ref.virtual_path,
    }


@dataclass(frozen=True, slots=True)
class SourceIdentityV1:
    provider_id: str
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _identifier(self.provider_id, "source.providerId"))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source.sourceId"))


def decode_source_identity(value: object, field: str = "source") -> SourceIdentityV1:
    obj = _object(value, field, required=frozenset({"providerId", "sourceId"}))
    return SourceIdentityV1(
        _identifier(obj["providerId"], f"{field}.providerId"),
        _identifier(obj["sourceId"], f"{field}.sourceId"),
    )


def encode_source_identity(value: SourceIdentityV1) -> JsonObject:
    checked = SourceIdentityV1(value.provider_id, value.source_id)
    return {"providerId": checked.provider_id, "sourceId": checked.source_id}


@dataclass(frozen=True, slots=True)
class CandidateSelectionV1:
    logical_id: str
    variant_id: str
    digest: str
    source: SourceIdentityV1 | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_id", _identifier(self.logical_id, "selection.logicalId"))
        object.__setattr__(self, "variant_id", _identifier(self.variant_id, "selection.variantId"))
        object.__setattr__(self, "digest", _digest(self.digest, "selection.digest"))
        if self.source is not None and type(self.source) is not SourceIdentityV1:
            _fail("selection.source", "must be a SourceIdentityV1")


def decode_candidate_selection(value: object, field: str = "selection") -> CandidateSelectionV1:
    obj = _object(
        value,
        field,
        required=frozenset({"logicalId", "variantId", "digest"}),
        optional=frozenset({"source"}),
    )
    return CandidateSelectionV1(
        logical_id=_identifier(obj["logicalId"], f"{field}.logicalId"),
        variant_id=_identifier(obj["variantId"], f"{field}.variantId"),
        digest=_digest(obj["digest"], f"{field}.digest"),
        source=decode_source_identity(obj["source"], f"{field}.source")
        if "source" in obj
        else None,
    )


def encode_candidate_selection(value: CandidateSelectionV1) -> JsonObject:
    checked = CandidateSelectionV1(value.logical_id, value.variant_id, value.digest, value.source)
    result: JsonObject = {
        "logicalId": checked.logical_id,
        "variantId": checked.variant_id,
        "digest": checked.digest,
    }
    if checked.source is not None:
        result["source"] = encode_source_identity(checked.source)
    return result


@dataclass(frozen=True, slots=True)
class ResolveContextV1:
    asset_kind: str
    node_type: str
    input_id: str
    accept: tuple[str, ...] = ()
    type_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "asset_kind", _identifier(self.asset_kind, "context.assetKind", maximum=512)
        )
        object.__setattr__(
            self, "node_type", _identifier(self.node_type, "context.schema.nodeType", maximum=512)
        )
        object.__setattr__(
            self, "input_id", _identifier(self.input_id, "context.schema.inputId", maximum=512)
        )
        if self.type_id is not None:
            object.__setattr__(
                self, "type_id", _identifier(self.type_id, "context.schema.typeId", maximum=512)
            )
        accepts = _string_tuple(self.accept, "context.accept", maximum=32)
        for index, item in enumerate(accepts):
            if item == "*/*" or _MIME.fullmatch(item) is None:
                _fail(f"context.accept[{index}]", "must be an exact MIME type or type/*")
        object.__setattr__(self, "accept", accepts)


def decode_resolve_context(value: object, field: str = "context") -> ResolveContextV1:
    obj = _object(value, field, required=frozenset({"assetKind", "schema", "accept"}))
    schema = _object(
        obj["schema"],
        f"{field}.schema",
        required=frozenset({"nodeType", "inputId"}),
        optional=frozenset({"typeId"}),
    )
    return ResolveContextV1(
        asset_kind=_identifier(obj["assetKind"], f"{field}.assetKind", maximum=512),
        node_type=_identifier(schema["nodeType"], f"{field}.schema.nodeType", maximum=512),
        input_id=_identifier(schema["inputId"], f"{field}.schema.inputId", maximum=512),
        type_id=_identifier(schema["typeId"], f"{field}.schema.typeId", maximum=512)
        if "typeId" in schema
        else None,
        accept=_string_tuple(obj["accept"], f"{field}.accept", maximum=32, json=True),
    )


def encode_resolve_context(value: ResolveContextV1) -> JsonObject:
    checked = ResolveContextV1(
        value.asset_kind, value.node_type, value.input_id, value.accept, value.type_id
    )
    schema: JsonObject = {"nodeType": checked.node_type, "inputId": checked.input_id}
    if checked.type_id is not None:
        schema["typeId"] = checked.type_id
    return {"assetKind": checked.asset_kind, "schema": schema, "accept": list(checked.accept)}


@dataclass(frozen=True, slots=True)
class ExpectedV1:
    digest: str | None = None
    size: int | None = None

    def __post_init__(self) -> None:
        if self.digest is None and self.size is None:
            _fail("expected", "must contain digest or size")
        if self.digest is not None:
            object.__setattr__(self, "digest", _digest(self.digest, "expected.digest"))
        if self.size is not None:
            object.__setattr__(self, "size", _integer(self.size, "expected.size"))


def decode_expected(value: object, field: str = "expected") -> ExpectedV1:
    obj = _object(
        value,
        field,
        required=frozenset(),
        optional=frozenset({"digest", "size"}),
    )
    return ExpectedV1(
        digest=_digest(obj["digest"], f"{field}.digest") if "digest" in obj else None,
        size=_integer(obj["size"], f"{field}.size") if "size" in obj else None,
    )


def encode_expected(value: ExpectedV1) -> JsonObject:
    checked = ExpectedV1(value.digest, value.size)
    result: JsonObject = {}
    if checked.digest is not None:
        result["digest"] = checked.digest
    if checked.size is not None:
        result["size"] = checked.size
    return result


_HINT_FIELDS = ("source", "reference", "loaderPath", "modelType", "displayName")


@dataclass(frozen=True, slots=True)
class HintsV1:
    source: str | None = None
    reference: str | None = None
    loader_path: str | None = None
    model_type: str | None = None
    display_name: str | None = None

    def __post_init__(self) -> None:
        for attribute in ("source", "reference", "loader_path", "model_type", "display_name"):
            value = getattr(self, attribute)
            if value is not None:
                object.__setattr__(
                    self, attribute, _string(value, f"hints.{attribute}", maximum=1024)
                )


def decode_hints(value: object, field: str = "hints") -> HintsV1:
    obj = _object(value, field, required=frozenset(), optional=frozenset(_HINT_FIELDS))
    values = {key: _string(obj[key], f"{field}.{key}", maximum=1024) for key in obj}
    return HintsV1(
        source=values.get("source"),
        reference=values.get("reference"),
        loader_path=values.get("loaderPath"),
        model_type=values.get("modelType"),
        display_name=values.get("displayName"),
    )


def encode_hints(value: HintsV1) -> JsonObject:
    checked = HintsV1(
        value.source, value.reference, value.loader_path, value.model_type, value.display_name
    )
    result: JsonObject = {}
    for key, item in zip(
        _HINT_FIELDS,
        (
            checked.source,
            checked.reference,
            checked.loader_path,
            checked.model_type,
            checked.display_name,
        ),
        strict=True,
    ):
        if item is not None:
            result[key] = item
    return result


@dataclass(frozen=True, slots=True)
class CandidateRequirementsV1:
    loaders: tuple[str, ...] = ()
    runtimes: tuple[str, ...] = ()
    hardware: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "loaders", _string_tuple(self.loaders, "requirements.loaders"))
        object.__setattr__(self, "runtimes", _string_tuple(self.runtimes, "requirements.runtimes"))
        object.__setattr__(self, "hardware", _string_tuple(self.hardware, "requirements.hardware"))


@dataclass(frozen=True, slots=True)
class ProviderRequirementsV1:
    credential: str | None = None
    license: str | None = None
    cost: str | None = None
    policy_override: str | None = None

    def __post_init__(self) -> None:
        for attribute in ("credential", "license", "cost", "policy_override"):
            value = getattr(self, attribute)
            if value is not None:
                object.__setattr__(
                    self,
                    attribute,
                    _string(
                        value, f"requires.{attribute}", nonempty=True, maximum=256, printable=True
                    ),
                )


@dataclass(frozen=True, slots=True)
class ProviderSourceV1:
    source: SourceIdentityV1
    status: Literal["available", "unavailable"]
    reason: str
    requires: ProviderRequirementsV1 = ProviderRequirementsV1()

    def __post_init__(self) -> None:
        if type(self.source) is not SourceIdentityV1:
            _fail("providerSource.source", "must be a SourceIdentityV1")
        object.__setattr__(
            self,
            "status",
            _literal(self.status, "providerSource.status", frozenset({"available", "unavailable"})),
        )
        object.__setattr__(self, "reason", _reason(self.reason, "providerSource.reason"))
        if type(self.requires) is not ProviderRequirementsV1:
            _fail("providerSource.requires", "must be ProviderRequirementsV1")


@dataclass(frozen=True, slots=True)
class CandidateV1:
    logical_id: str
    family: str
    asset_kind: str
    variant_id: str
    dtype: str
    quantization: str
    format: str
    role: str
    requirements: CandidateRequirementsV1
    digest: str
    availability_status: Literal["local", "downloadable", "unavailable"]
    availability_reason: str
    compatibility_status: Literal["compatible", "incompatible", "unknown"]
    compatibility_reason: str
    provider_sources: tuple[ProviderSourceV1, ...] = ()
    size: int | None = None
    media_type: str | None = None
    asset_ref: AssetRef | None = None

    def __post_init__(self) -> None:
        for attribute in (
            "logical_id",
            "family",
            "asset_kind",
            "variant_id",
            "dtype",
            "quantization",
            "format",
            "role",
        ):
            object.__setattr__(
                self, attribute, _identifier(getattr(self, attribute), f"candidate.{attribute}")
            )
        if type(self.requirements) is not CandidateRequirementsV1:
            _fail("candidate.requirements", "must be CandidateRequirementsV1")
        object.__setattr__(self, "digest", _digest(self.digest, "candidate.digest"))
        if self.size is not None:
            object.__setattr__(self, "size", _integer(self.size, "candidate.size"))
        if self.media_type is not None:
            object.__setattr__(self, "media_type", _string(self.media_type, "candidate.mediaType"))
        object.__setattr__(
            self,
            "availability_status",
            _literal(
                self.availability_status,
                "candidate.availability.status",
                frozenset({"local", "downloadable", "unavailable"}),
            ),
        )
        object.__setattr__(
            self,
            "availability_reason",
            _reason(self.availability_reason, "candidate.availability.reason"),
        )
        object.__setattr__(
            self,
            "compatibility_status",
            _literal(
                self.compatibility_status,
                "candidate.compatibility.status",
                frozenset({"compatible", "incompatible", "unknown"}),
            ),
        )
        object.__setattr__(
            self,
            "compatibility_reason",
            _reason(
                self.compatibility_reason,
                "candidate.compatibility.reason",
                nonempty=self.compatibility_status != "compatible",
            ),
        )
        sources = _copy_collection(self.provider_sources, "candidate.providerSources", maximum=64)
        if any(type(source) is not ProviderSourceV1 for source in sources):
            _fail("candidate.providerSources", "must contain ProviderSourceV1 values")
        keys = tuple((source.source.provider_id, source.source.source_id) for source in sources)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            _fail(
                "candidate.providerSources",
                "must be strictly ordered with unique source identities",
            )
        object.__setattr__(self, "provider_sources", sources)
        if self.availability_status == "local":
            if self.asset_ref is None:
                _fail("candidate.assetRef", "is required for local availability")
            ref = _validate_asset_ref(self.asset_ref, "candidate.assetRef")
            if ref.digest != self.digest:
                _fail("candidate.assetRef.digest", "must equal candidate.digest")
        elif self.asset_ref is not None:
            _fail("candidate.assetRef", "is allowed only for local availability")


def _decode_requirements(value: object, field: str) -> CandidateRequirementsV1:
    obj = _object(value, field, required=frozenset({"loaders", "runtimes", "hardware"}))
    return CandidateRequirementsV1(
        _string_tuple(obj["loaders"], f"{field}.loaders", json=True),
        _string_tuple(obj["runtimes"], f"{field}.runtimes", json=True),
        _string_tuple(obj["hardware"], f"{field}.hardware", json=True),
    )


def _decode_provider_requires(value: object, field: str) -> ProviderRequirementsV1:
    keys = frozenset({"credential", "license", "cost", "policyOverride"})
    obj = _object(value, field, required=frozenset(), optional=keys)
    values = {
        key: _string(obj[key], f"{field}.{key}", nonempty=True, maximum=256, printable=True)
        for key in obj
    }
    return ProviderRequirementsV1(
        credential=values.get("credential"),
        license=values.get("license"),
        cost=values.get("cost"),
        policy_override=values.get("policyOverride"),
    )


def _decode_provider_source(value: object, field: str) -> ProviderSourceV1:
    obj = _object(value, field, required=frozenset({"source", "status", "reason", "requires"}))
    return ProviderSourceV1(
        source=decode_source_identity(obj["source"], f"{field}.source"),
        status=cast(
            Literal["available", "unavailable"],
            _literal(obj["status"], f"{field}.status", frozenset({"available", "unavailable"})),
        ),
        reason=_reason(obj["reason"], f"{field}.reason"),
        requires=_decode_provider_requires(obj["requires"], f"{field}.requires"),
    )


def decode_candidate(value: object, field: str = "candidate") -> CandidateV1:
    required = frozenset(
        {
            "logicalId",
            "family",
            "assetKind",
            "variantId",
            "dtype",
            "quantization",
            "format",
            "role",
            "requirements",
            "digest",
            "availability",
            "compatibility",
            "providerSources",
        }
    )
    obj = _object(
        value, field, required=required, optional=frozenset({"size", "mediaType", "assetRef"})
    )
    availability = _object(
        obj["availability"], f"{field}.availability", required=frozenset({"status", "reason"})
    )
    compatibility = _object(
        obj["compatibility"], f"{field}.compatibility", required=frozenset({"status", "reason"})
    )
    sources = tuple(
        _decode_provider_source(item, f"{field}.providerSources[{index}]")
        for index, item in enumerate(
            _tuple(
                obj["providerSources"],
                f"{field}.providerSources",
                maximum=64,
                json=True,
            )
        )
    )
    return CandidateV1(
        logical_id=_identifier(obj["logicalId"], f"{field}.logicalId"),
        family=_identifier(obj["family"], f"{field}.family"),
        asset_kind=_identifier(obj["assetKind"], f"{field}.assetKind"),
        variant_id=_identifier(obj["variantId"], f"{field}.variantId"),
        dtype=_identifier(obj["dtype"], f"{field}.dtype"),
        quantization=_identifier(obj["quantization"], f"{field}.quantization"),
        format=_identifier(obj["format"], f"{field}.format"),
        role=_identifier(obj["role"], f"{field}.role"),
        requirements=_decode_requirements(obj["requirements"], f"{field}.requirements"),
        digest=_digest(obj["digest"], f"{field}.digest"),
        size=_integer(obj["size"], f"{field}.size") if "size" in obj else None,
        media_type=_string(obj["mediaType"], f"{field}.mediaType") if "mediaType" in obj else None,
        availability_status=cast(
            Literal["local", "downloadable", "unavailable"],
            _literal(
                availability["status"],
                f"{field}.availability.status",
                frozenset({"local", "downloadable", "unavailable"}),
            ),
        ),
        availability_reason=_reason(availability["reason"], f"{field}.availability.reason"),
        compatibility_status=cast(
            Literal["compatible", "incompatible", "unknown"],
            _literal(
                compatibility["status"],
                f"{field}.compatibility.status",
                frozenset({"compatible", "incompatible", "unknown"}),
            ),
        ),
        compatibility_reason=_reason(compatibility["reason"], f"{field}.compatibility.reason"),
        asset_ref=_asset_ref(obj["assetRef"], f"{field}.assetRef") if "assetRef" in obj else None,
        provider_sources=sources,
    )


def _encode_provider_requires(value: ProviderRequirementsV1) -> JsonObject:
    checked = ProviderRequirementsV1(
        value.credential, value.license, value.cost, value.policy_override
    )
    result: JsonObject = {}
    for key, item in (
        ("credential", checked.credential),
        ("license", checked.license),
        ("cost", checked.cost),
        ("policyOverride", checked.policy_override),
    ):
        if item is not None:
            result[key] = item
    return result


def encode_candidate(value: CandidateV1) -> JsonObject:
    checked = CandidateV1(**{field: getattr(value, field) for field in value.__dataclass_fields__})
    result: JsonObject = {
        "logicalId": checked.logical_id,
        "family": checked.family,
        "assetKind": checked.asset_kind,
        "variantId": checked.variant_id,
        "dtype": checked.dtype,
        "quantization": checked.quantization,
        "format": checked.format,
        "role": checked.role,
        "requirements": {
            "loaders": list(checked.requirements.loaders),
            "runtimes": list(checked.requirements.runtimes),
            "hardware": list(checked.requirements.hardware),
        },
        "digest": checked.digest,
        "availability": {
            "status": checked.availability_status,
            "reason": checked.availability_reason,
        },
        "compatibility": {
            "status": checked.compatibility_status,
            "reason": checked.compatibility_reason,
        },
        "providerSources": [
            {
                "source": encode_source_identity(source.source),
                "status": source.status,
                "reason": source.reason,
                "requires": _encode_provider_requires(source.requires),
            }
            for source in checked.provider_sources
        ],
    }
    if checked.size is not None:
        result["size"] = checked.size
    if checked.media_type is not None:
        result["mediaType"] = checked.media_type
    if checked.asset_ref is not None:
        result["assetRef"] = encode_asset_ref(checked.asset_ref)
    return result


def _candidates(
    value: object, field: str, *, maximum: int | None = None
) -> tuple[CandidateV1, ...]:
    items = tuple(
        decode_candidate(item, f"{field}[{index}]")
        for index, item in enumerate(_tuple(value, field, maximum=maximum, json=True))
    )
    _validate_candidate_order(items, field)
    return items


def _validate_candidate_order(items: tuple[CandidateV1, ...], field: str) -> None:
    keys = tuple((item.logical_id, item.variant_id, item.digest) for item in items)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        _fail(field, "must be strictly ordered with unique candidate identities")


def _validate_expected_selection(
    expected: ExpectedV1 | None, selection: CandidateSelectionV1 | None, field: str
) -> None:
    if expected is not None and selection is not None and expected.digest is not None:
        if expected.digest != selection.digest:
            _fail(field, "expected.digest must equal selection.digest")


def _optional_scope(obj: JsonObject, field: str = "scope") -> str | None:
    return _scope(obj[field], field) if field in obj else None


def _optional_cursor(obj: JsonObject, field: str = "cursor") -> str | None:
    return _string(obj[field], field, maximum=4096) if field in obj else None


def _limit(obj: JsonObject, field: str = "limit") -> int:
    return _integer(obj[field], field, minimum=1, maximum=100) if field in obj else 50


@dataclass(frozen=True, slots=True)
class CatalogRequestV1:
    scope: str | None = None
    query: str | None = None
    asset_kind: str | None = None
    context: ResolveContextV1 | None = None
    availability: tuple[str, ...] | None = None
    compatibility: tuple[str, ...] | None = None
    cursor: str | None = None
    limit: int = 50

    def __post_init__(self) -> None:
        if self.scope is not None:
            object.__setattr__(self, "scope", _scope(self.scope, "scope"))
        if self.query is not None:
            object.__setattr__(self, "query", _string(self.query, "query", maximum=512))
        if self.asset_kind is not None:
            object.__setattr__(
                self, "asset_kind", _identifier(self.asset_kind, "assetKind", maximum=512)
            )
        if self.context is not None and type(self.context) is not ResolveContextV1:
            _fail("context", "must be ResolveContextV1")
        if (
            self.asset_kind is not None
            and self.context is not None
            and self.asset_kind != self.context.asset_kind
        ):
            _fail("assetKind", "must agree with context.assetKind")
        for attribute, allowed in (
            ("availability", frozenset({"local", "downloadable", "unavailable"})),
            ("compatibility", frozenset({"compatible", "incompatible", "unknown"})),
        ):
            values = getattr(self, attribute)
            if values is not None:
                checked = tuple(
                    _literal(item, f"{attribute}[{index}]", allowed)
                    for index, item in enumerate(_tuple(values, attribute))
                )
                if len(set(checked)) != len(checked):
                    _fail(attribute, "must not contain duplicates")
                object.__setattr__(self, attribute, checked)
        if self.cursor is not None:
            object.__setattr__(self, "cursor", _string(self.cursor, "cursor", maximum=4096))
        object.__setattr__(self, "limit", _integer(self.limit, "limit", minimum=1, maximum=100))


_CATALOG_REQUEST_KEYS = frozenset(
    {"scope", "query", "assetKind", "context", "availability", "compatibility", "cursor", "limit"}
)


def decode_catalog_request(value: object) -> CatalogRequestV1:
    obj = _object(
        value, "request", required=frozenset({"contractVersion"}), optional=_CATALOG_REQUEST_KEYS
    )
    _contract_version(obj["contractVersion"], "contractVersion")
    return CatalogRequestV1(
        scope=_optional_scope(obj),
        query=_string(obj["query"], "query", maximum=512) if "query" in obj else None,
        asset_kind=_identifier(obj["assetKind"], "assetKind", maximum=512)
        if "assetKind" in obj
        else None,
        context=decode_resolve_context(obj["context"]) if "context" in obj else None,
        availability=_string_tuple(obj["availability"], "availability", json=True)
        if "availability" in obj
        else None,
        compatibility=_string_tuple(obj["compatibility"], "compatibility", json=True)
        if "compatibility" in obj
        else None,
        cursor=_optional_cursor(obj),
        limit=_limit(obj),
    )


def encode_catalog_request(value: CatalogRequestV1) -> JsonObject:
    checked = CatalogRequestV1(
        **{field: getattr(value, field) for field in value.__dataclass_fields__}
    )
    result: JsonObject = {"contractVersion": 1}
    for key, item in (
        ("scope", checked.scope),
        ("query", checked.query),
        ("assetKind", checked.asset_kind),
        ("cursor", checked.cursor),
    ):
        if item is not None:
            result[key] = item
    if checked.context is not None:
        result["context"] = encode_resolve_context(checked.context)
    if checked.availability is not None:
        result["availability"] = list(checked.availability)
    if checked.compatibility is not None:
        result["compatibility"] = list(checked.compatibility)
    result["limit"] = checked.limit
    return result


@dataclass(frozen=True, slots=True)
class CandidatesRequestV1:
    context: ResolveContextV1
    scope: str | None = None
    expected: ExpectedV1 | None = None
    hints: HintsV1 | None = None
    selection: CandidateSelectionV1 | None = None
    cursor: str | None = None
    limit: int = 50

    def __post_init__(self) -> None:
        if type(self.context) is not ResolveContextV1:
            _fail("context", "must be ResolveContextV1")
        if self.scope is not None:
            object.__setattr__(self, "scope", _scope(self.scope, "scope"))
        if self.expected is not None and type(self.expected) is not ExpectedV1:
            _fail("expected", "must be ExpectedV1")
        if self.hints is not None and type(self.hints) is not HintsV1:
            _fail("hints", "must be HintsV1")
        if self.selection is not None and type(self.selection) is not CandidateSelectionV1:
            _fail("selection", "must be CandidateSelectionV1")
        _validate_expected_selection(self.expected, self.selection, "selection")
        if self.cursor is not None:
            object.__setattr__(self, "cursor", _string(self.cursor, "cursor", maximum=4096))
        object.__setattr__(self, "limit", _integer(self.limit, "limit", minimum=1, maximum=100))


def decode_candidates_request(value: object) -> CandidatesRequestV1:
    optional = frozenset({"scope", "expected", "hints", "selection", "cursor", "limit"})
    obj = _object(
        value, "request", required=frozenset({"contractVersion", "context"}), optional=optional
    )
    _contract_version(obj["contractVersion"], "contractVersion")
    return CandidatesRequestV1(
        context=decode_resolve_context(obj["context"]),
        scope=_optional_scope(obj),
        expected=decode_expected(obj["expected"]) if "expected" in obj else None,
        hints=decode_hints(obj["hints"]) if "hints" in obj else None,
        selection=decode_candidate_selection(obj["selection"]) if "selection" in obj else None,
        cursor=_optional_cursor(obj),
        limit=_limit(obj),
    )


def encode_candidates_request(value: CandidatesRequestV1) -> JsonObject:
    checked = CandidatesRequestV1(
        **{field: getattr(value, field) for field in value.__dataclass_fields__}
    )
    result: JsonObject = {
        "contractVersion": 1,
        "context": encode_resolve_context(checked.context),
        "limit": checked.limit,
    }
    for key, item in (("scope", checked.scope), ("cursor", checked.cursor)):
        if item is not None:
            result[key] = item
    if checked.expected is not None:
        result["expected"] = encode_expected(checked.expected)
    if checked.hints is not None:
        result["hints"] = encode_hints(checked.hints)
    if checked.selection is not None:
        result["selection"] = encode_candidate_selection(checked.selection)
    return result


@dataclass(frozen=True, slots=True)
class ConsentsV1:
    license: tuple[str, ...] | None = None
    cost: tuple[str, ...] | None = None
    policy_override: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        total = 0
        for attribute in ("license", "cost", "policy_override"):
            raw = getattr(self, attribute)
            if raw is None:
                continue
            values = _string_tuple(
                raw,
                f"consents.{attribute}",
                maximum=16,
                item_maximum=256,
                printable=True,
            )
            if any(not item for item in values):
                _fail(f"consents.{attribute}", "ids must be nonempty")
            object.__setattr__(self, attribute, values)
            total += len(values)
        if total > 32:
            _fail("consents", "must contain at most 32 ids total")


@dataclass(frozen=True, slots=True)
class DocumentOccurrenceV1:
    pointer: str
    current: AssetRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "pointer", _pointer(self.pointer, "document.occurrence.pointer"))
        _validate_asset_ref(self.current, "document.occurrence.current")


@dataclass(frozen=True, slots=True)
class ResolveDocumentV1:
    digest: str
    occurrences: tuple[DocumentOccurrenceV1, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", _document_token(self.digest, "document.digest"))
        rows = _copy_collection(self.occurrences, "document.occurrences", maximum=256)
        if any(type(row) is not DocumentOccurrenceV1 for row in rows):
            _fail("document.occurrences", "must contain DocumentOccurrenceV1 values")
        object.__setattr__(self, "occurrences", rows)


def _document_token(value: object, field: str) -> str:
    return _string(value, field, maximum=4096, control_free=True)


def _pointer(value: object, field: str) -> str:
    result = _string(value, field)
    if _POINTER.fullmatch(result) is None:
        _fail(field, "must be an RFC 6901 pointer")
    return result


def _decode_consents(value: object, field: str) -> ConsentsV1:
    obj = _object(
        value,
        field,
        required=frozenset(),
        optional=frozenset({"license", "cost", "policyOverride"}),
    )
    return ConsentsV1(
        license=_string_tuple(obj["license"], f"{field}.license", maximum=16, json=True)
        if "license" in obj
        else None,
        cost=_string_tuple(obj["cost"], f"{field}.cost", maximum=16, json=True)
        if "cost" in obj
        else None,
        policy_override=_string_tuple(
            obj["policyOverride"],
            f"{field}.policyOverride",
            maximum=16,
            json=True,
        )
        if "policyOverride" in obj
        else None,
    )


def _encode_consents(value: ConsentsV1) -> JsonObject:
    checked = ConsentsV1(value.license, value.cost, value.policy_override)
    result: JsonObject = {}
    if checked.license is not None:
        result["license"] = list(checked.license)
    if checked.cost is not None:
        result["cost"] = list(checked.cost)
    if checked.policy_override is not None:
        result["policyOverride"] = list(checked.policy_override)
    return result


def _decode_document(value: object, field: str) -> ResolveDocumentV1:
    obj = _object(value, field, required=frozenset({"digest", "occurrences"}))
    occurrences = tuple(
        DocumentOccurrenceV1(
            pointer=_pointer(row["pointer"], f"{field}.occurrences[{index}].pointer"),
            current=_asset_ref(row["current"], f"{field}.occurrences[{index}].current"),
        )
        for index, item in enumerate(
            _tuple(obj["occurrences"], f"{field}.occurrences", maximum=256, json=True)
        )
        for row in [
            _object(
                item, f"{field}.occurrences[{index}]", required=frozenset({"pointer", "current"})
            )
        ]
    )
    return ResolveDocumentV1(_document_token(obj["digest"], f"{field}.digest"), occurrences)


def _encode_document(value: ResolveDocumentV1) -> JsonObject:
    checked = ResolveDocumentV1(value.digest, value.occurrences)
    return {
        "digest": checked.digest,
        "occurrences": [
            {"pointer": row.pointer, "current": encode_asset_ref(row.current)}
            for row in checked.occurrences
        ],
    }


@dataclass(frozen=True, slots=True)
class ResolveRequestV1:
    mode: Literal["existing", "acquire-managed"]
    context: ResolveContextV1
    scope: str | None = None
    expected: ExpectedV1 | None = None
    hints: HintsV1 | None = None
    selection: CandidateSelectionV1 | None = None
    consents: ConsentsV1 | None = None
    document: ResolveDocumentV1 | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _literal(self.mode, "mode", frozenset({"existing", "acquire-managed"}))
        )
        if type(self.context) is not ResolveContextV1:
            _fail("context", "must be ResolveContextV1")
        if self.scope is not None:
            object.__setattr__(self, "scope", _scope(self.scope, "scope"))
        for attribute, expected_type in (
            ("expected", ExpectedV1),
            ("hints", HintsV1),
            ("selection", CandidateSelectionV1),
            ("consents", ConsentsV1),
            ("document", ResolveDocumentV1),
        ):
            item = getattr(self, attribute)
            if item is not None and type(item) is not expected_type:
                _fail(attribute, f"must be {expected_type.__name__}")
        _validate_expected_selection(self.expected, self.selection, "selection")


def decode_resolve_request(value: object) -> ResolveRequestV1:
    optional = frozenset({"scope", "expected", "hints", "selection", "consents", "document"})
    obj = _object(
        value,
        "request",
        required=frozenset({"contractVersion", "mode", "context"}),
        optional=optional,
    )
    _contract_version(obj["contractVersion"], "contractVersion")
    return ResolveRequestV1(
        mode=cast(
            Literal["existing", "acquire-managed"],
            _literal(obj["mode"], "mode", frozenset({"existing", "acquire-managed"})),
        ),
        context=decode_resolve_context(obj["context"]),
        scope=_optional_scope(obj),
        expected=decode_expected(obj["expected"]) if "expected" in obj else None,
        hints=decode_hints(obj["hints"]) if "hints" in obj else None,
        selection=decode_candidate_selection(obj["selection"]) if "selection" in obj else None,
        consents=_decode_consents(obj["consents"], "consents") if "consents" in obj else None,
        document=_decode_document(obj["document"], "document") if "document" in obj else None,
    )


def encode_resolve_request(value: ResolveRequestV1) -> JsonObject:
    checked = ResolveRequestV1(
        **{field: getattr(value, field) for field in value.__dataclass_fields__}
    )
    result: JsonObject = {
        "contractVersion": 1,
        "mode": checked.mode,
        "context": encode_resolve_context(checked.context),
    }
    if checked.scope is not None:
        result["scope"] = checked.scope
    if checked.expected is not None:
        result["expected"] = encode_expected(checked.expected)
    if checked.hints is not None:
        result["hints"] = encode_hints(checked.hints)
    if checked.selection is not None:
        result["selection"] = encode_candidate_selection(checked.selection)
    if checked.consents is not None:
        result["consents"] = _encode_consents(checked.consents)
    if checked.document is not None:
        result["document"] = _encode_document(checked.document)
    return result


@dataclass(frozen=True, slots=True)
class SelectedCandidateV1:
    logical_id: str
    variant_id: str
    digest: str
    reason: str
    source: SourceIdentityV1 | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_id", _identifier(self.logical_id, "selectedCandidate.logicalId")
        )
        object.__setattr__(
            self, "variant_id", _identifier(self.variant_id, "selectedCandidate.variantId")
        )
        object.__setattr__(self, "digest", _digest(self.digest, "selectedCandidate.digest"))
        object.__setattr__(
            self, "reason", _literal(self.reason, "selectedCandidate.reason", _SELECTED_REASONS)
        )
        if self.source is not None and type(self.source) is not SourceIdentityV1:
            _fail("selectedCandidate.source", "must be SourceIdentityV1")


def _decode_selected(value: object, field: str = "selectedCandidate") -> SelectedCandidateV1:
    obj = _object(
        value,
        field,
        required=frozenset({"logicalId", "variantId", "digest", "reason"}),
        optional=frozenset({"source"}),
    )
    return SelectedCandidateV1(
        _identifier(obj["logicalId"], f"{field}.logicalId"),
        _identifier(obj["variantId"], f"{field}.variantId"),
        _digest(obj["digest"], f"{field}.digest"),
        _literal(obj["reason"], f"{field}.reason", _SELECTED_REASONS),
        decode_source_identity(obj["source"], f"{field}.source") if "source" in obj else None,
    )


def _encode_selected(value: SelectedCandidateV1) -> JsonObject:
    checked = SelectedCandidateV1(
        value.logical_id, value.variant_id, value.digest, value.reason, value.source
    )
    result: JsonObject = {
        "logicalId": checked.logical_id,
        "variantId": checked.variant_id,
        "digest": checked.digest,
        "reason": checked.reason,
    }
    if checked.source is not None:
        result["source"] = encode_source_identity(checked.source)
    return result


@dataclass(frozen=True, slots=True)
class CatalogResponseV1:
    items: tuple[CandidateV1, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        items = _copy_collection(self.items, "items", maximum=100)
        if any(type(item) is not CandidateV1 for item in items):
            _fail("items", "must contain CandidateV1 values")
        _validate_candidate_order(items, "items")
        object.__setattr__(self, "items", items)
        if self.next_cursor is not None:
            object.__setattr__(
                self, "next_cursor", _string(self.next_cursor, "nextCursor", maximum=4096)
            )


@dataclass(frozen=True, slots=True)
class CandidatesResponseV1:
    status: Literal["resolved", "missing", "ambiguous", "incompatible"]
    items: tuple[CandidateV1, ...]
    selected_candidate: SelectedCandidateV1 | None = None
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _literal(
                self.status,
                "status",
                frozenset({"resolved", "missing", "ambiguous", "incompatible"}),
            ),
        )
        items = _copy_collection(self.items, "items", maximum=100)
        if any(type(item) is not CandidateV1 for item in items):
            _fail("items", "must contain CandidateV1 values")
        _validate_candidate_order(items, "items")
        object.__setattr__(self, "items", items)
        if self.selected_candidate is not None:
            if type(self.selected_candidate) is not SelectedCandidateV1:
                _fail("selectedCandidate", "must be SelectedCandidateV1")
            match = next(
                (
                    item
                    for item in items
                    if _candidate_key(item) == _selected_key(self.selected_candidate)
                ),
                None,
            )
            if (
                match is not None
                and self.status == "resolved"
                and not (
                    match.availability_status == "local"
                    and match.compatibility_status == "compatible"
                )
            ):
                _fail("status", "resolved selection must be local and compatible")
        if self.status == "resolved" and self.selected_candidate is None:
            _fail("selectedCandidate", "is required for resolved status")
        if self.status == "ambiguous" and self.selected_candidate is not None:
            _fail("selectedCandidate", "is not allowed for ambiguous status")
        if self.next_cursor is not None:
            object.__setattr__(
                self, "next_cursor", _string(self.next_cursor, "nextCursor", maximum=4096)
            )


def _candidate_key(value: CandidateV1) -> tuple[str, str, str]:
    return value.logical_id, value.variant_id, value.digest


def _selected_key(value: SelectedCandidateV1) -> tuple[str, str, str]:
    return value.logical_id, value.variant_id, value.digest


def decode_catalog_response(value: object) -> CatalogResponseV1:
    obj = _object(
        value,
        "response",
        required=frozenset({"contractVersion", "items"}),
        optional=frozenset({"nextCursor"}),
    )
    _contract_version(obj["contractVersion"], "contractVersion")
    return CatalogResponseV1(
        _candidates(obj["items"], "items", maximum=100),
        _string(obj["nextCursor"], "nextCursor", maximum=4096) if "nextCursor" in obj else None,
    )


def encode_catalog_response(value: CatalogResponseV1) -> JsonObject:
    checked = CatalogResponseV1(value.items, value.next_cursor)
    result: JsonObject = {
        "contractVersion": 1,
        "items": [encode_candidate(item) for item in checked.items],
    }
    if checked.next_cursor is not None:
        result["nextCursor"] = checked.next_cursor
    return result


def decode_candidates_response(value: object) -> CandidatesResponseV1:
    obj = _object(
        value,
        "response",
        required=frozenset({"contractVersion", "status", "items"}),
        optional=frozenset({"selectedCandidate", "nextCursor"}),
    )
    _contract_version(obj["contractVersion"], "contractVersion")
    return CandidatesResponseV1(
        status=cast(
            Literal["resolved", "missing", "ambiguous", "incompatible"],
            _literal(
                obj["status"],
                "status",
                frozenset({"resolved", "missing", "ambiguous", "incompatible"}),
            ),
        ),
        items=_candidates(obj["items"], "items", maximum=100),
        selected_candidate=_decode_selected(obj["selectedCandidate"])
        if "selectedCandidate" in obj
        else None,
        next_cursor=_string(obj["nextCursor"], "nextCursor", maximum=4096)
        if "nextCursor" in obj
        else None,
    )


def encode_candidates_response(value: CandidatesResponseV1) -> JsonObject:
    checked = CandidatesResponseV1(
        value.status, value.items, value.selected_candidate, value.next_cursor
    )
    result: JsonObject = {
        "contractVersion": 1,
        "status": checked.status,
        "items": [encode_candidate(item) for item in checked.items],
    }
    if checked.selected_candidate is not None:
        result["selectedCandidate"] = _encode_selected(checked.selected_candidate)
    if checked.next_cursor is not None:
        result["nextCursor"] = checked.next_cursor
    return result


@dataclass(frozen=True, slots=True)
class RepairPreconditionV1:
    pointer: str
    equals: AssetRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "pointer", _pointer(self.pointer, "precondition.pointer"))
        _validate_asset_ref(self.equals, "precondition.equals")


@dataclass(frozen=True, slots=True)
class RepairReplacementV1:
    pointer: str
    value: AssetRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "pointer", _pointer(self.pointer, "replacement.pointer"))
        _validate_asset_ref(self.value, "replacement.value")


@dataclass(frozen=True, slots=True)
class AssetRefRepairSuggestionV1:
    document_digest: str
    preconditions: tuple[RepairPreconditionV1, ...]
    replacements: tuple[RepairReplacementV1, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "document_digest", _document_token(self.document_digest, "repair.documentDigest")
        )
        preconditions = _copy_collection(self.preconditions, "repair.preconditions")
        replacements = _copy_collection(self.replacements, "repair.replacements")
        if not preconditions or not replacements:
            _fail("repair", "preconditions and replacements must be nonempty")
        if any(type(item) is not RepairPreconditionV1 for item in preconditions):
            _fail("repair.preconditions", "must contain RepairPreconditionV1 values")
        if any(type(item) is not RepairReplacementV1 for item in replacements):
            _fail("repair.replacements", "must contain RepairReplacementV1 values")
        precondition_pointers = tuple(item.pointer for item in preconditions)
        replacement_pointers = tuple(item.pointer for item in replacements)
        if len(set(precondition_pointers)) != len(precondition_pointers) or len(
            set(replacement_pointers)
        ) != len(replacement_pointers):
            _fail("repair", "precondition and replacement targets must be unique")
        if set(precondition_pointers) != set(replacement_pointers):
            _fail("repair", "precondition and replacement target sets must be equal and unique")
        for index, pointer in enumerate(precondition_pointers):
            for other in precondition_pointers[index + 1 :]:
                if _pointer_contains(pointer, other) or _pointer_contains(other, pointer):
                    _fail("repair", "targets must not overlap as ancestor and descendant")
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "replacements", replacements)


def _pointer_contains(parent: str, child: str) -> bool:
    return parent == "" or child.startswith(parent + "/")


def decode_repair(value: object, field: str = "repair") -> AssetRefRepairSuggestionV1:
    obj = _object(
        value,
        field,
        required=frozenset(
            {"type", "version", "atomic", "documentDigest", "preconditions", "replacements"}
        ),
    )
    if obj["type"] != "asset-ref-repair" or type(obj["type"]) is not str:
        _fail(f"{field}.type", "must be 'asset-ref-repair'")
    _contract_version(obj["version"], f"{field}.version")
    if _boolean(obj["atomic"], f"{field}.atomic") is not True:
        _fail(f"{field}.atomic", "must be true")
    preconditions = tuple(
        RepairPreconditionV1(
            _pointer(row["pointer"], f"{field}.preconditions[{index}].pointer"),
            _asset_ref(row["equals"], f"{field}.preconditions[{index}].equals"),
        )
        for index, item in enumerate(
            _tuple(obj["preconditions"], f"{field}.preconditions", json=True)
        )
        for row in [
            _object(
                item, f"{field}.preconditions[{index}]", required=frozenset({"pointer", "equals"})
            )
        ]
    )
    replacements = tuple(
        RepairReplacementV1(
            _pointer(row["pointer"], f"{field}.replacements[{index}].pointer"),
            _asset_ref(row["value"], f"{field}.replacements[{index}].value"),
        )
        for index, item in enumerate(
            _tuple(obj["replacements"], f"{field}.replacements", json=True)
        )
        for row in [
            _object(
                item, f"{field}.replacements[{index}]", required=frozenset({"pointer", "value"})
            )
        ]
    )
    return AssetRefRepairSuggestionV1(
        _document_token(obj["documentDigest"], f"{field}.documentDigest"),
        preconditions,
        replacements,
    )


def encode_repair(value: AssetRefRepairSuggestionV1) -> JsonObject:
    checked = AssetRefRepairSuggestionV1(
        value.document_digest, value.preconditions, value.replacements
    )
    return {
        "type": "asset-ref-repair",
        "version": 1,
        "atomic": True,
        "documentDigest": checked.document_digest,
        "preconditions": [
            {"pointer": item.pointer, "equals": encode_asset_ref(item.equals)}
            for item in checked.preconditions
        ],
        "replacements": [
            {"pointer": item.pointer, "value": encode_asset_ref(item.value)}
            for item in checked.replacements
        ],
    }


@dataclass(frozen=True, slots=True)
class ResolveResponseV1:
    status: Literal["resolved-existing", "resolved-remapped", "acquired"]
    selected: AssetRef
    selected_candidate: SelectedCandidateV1
    repair: AssetRefRepairSuggestionV1 | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _literal(
                self.status,
                "status",
                frozenset({"resolved-existing", "resolved-remapped", "acquired"}),
            ),
        )
        selected = _validate_asset_ref(self.selected, "selected")
        if type(self.selected_candidate) is not SelectedCandidateV1:
            _fail("selectedCandidate", "must be SelectedCandidateV1")
        if selected.digest != self.selected_candidate.digest:
            _fail("selected.digest", "must equal selectedCandidate.digest")
        if self.repair is not None:
            if type(self.repair) is not AssetRefRepairSuggestionV1:
                _fail("repair", "must be AssetRefRepairSuggestionV1")


def decode_resolve_response(value: object) -> ResolveResponseV1:
    obj = _object(
        value,
        "response",
        required=frozenset({"contractVersion", "status", "selected", "selectedCandidate"}),
        optional=frozenset({"repair"}),
    )
    _contract_version(obj["contractVersion"], "contractVersion")
    return ResolveResponseV1(
        status=cast(
            Literal["resolved-existing", "resolved-remapped", "acquired"],
            _literal(
                obj["status"],
                "status",
                frozenset({"resolved-existing", "resolved-remapped", "acquired"}),
            ),
        ),
        selected=_asset_ref(obj["selected"], "selected"),
        selected_candidate=_decode_selected(obj["selectedCandidate"]),
        repair=decode_repair(obj["repair"]) if "repair" in obj else None,
    )


def encode_resolve_response(value: ResolveResponseV1) -> JsonObject:
    checked = ResolveResponseV1(
        value.status, value.selected, value.selected_candidate, value.repair
    )
    result: JsonObject = {
        "contractVersion": 1,
        "status": checked.status,
        "selected": encode_asset_ref(checked.selected),
        "selectedCandidate": _encode_selected(checked.selected_candidate),
    }
    if checked.repair is not None:
        result["repair"] = encode_repair(checked.repair)
    return result


_ERROR_STATUS: Final[dict[str, int]] = {
    "invalid-request": 400,
    "cursor-invalid": 400,
    "authentication-required": 401,
    "forbidden": 403,
    "not-found": 404,
    "selection-required": 409,
    "not-available": 409,
    "incompatible": 409,
    "credential-required": 409,
    "license-required": 409,
    "cost-required": 409,
    "policy-override-required": 409,
    "source-unavailable": 409,
    "no-compatible-destination": 409,
    "mapping-conflict": 409,
    "wrong-kind": 422,
    "integrity-mismatch": 422,
    "acquisition-failed": 502,
    "service-unavailable": 503,
}


@dataclass(frozen=True, slots=True)
class ErrorV1:
    code: str
    reason: str | None = None
    field: str | None = None
    candidates: tuple[CandidateV1, ...] | None = None
    truncated: bool | None = None
    requirement_id: str | None = None
    selection: CandidateSelectionV1 | None = None
    expected_kind: str | None = None
    actual_kind: str | None = None
    expected_digest: str | None = None
    observed_digest: str | None = None
    expected_size: int | None = None
    observed_size: int | None = None

    def __post_init__(self) -> None:
        code = _literal(self.code, "error.code", frozenset(_ERROR_STATUS))
        object.__setattr__(self, "code", code)
        present = {
            name
            for name in (
                "reason",
                "field",
                "candidates",
                "truncated",
                "requirement_id",
                "selection",
                "expected_kind",
                "actual_kind",
                "expected_digest",
                "observed_digest",
                "expected_size",
                "observed_size",
            )
            if getattr(self, name) is not None
        }
        required, optional = _error_fields(code)
        missing = required - present
        unknown = present - required - optional
        if missing:
            missing_field = sorted(missing)[0]
            field = "error.candidates" if missing_field == "candidates" else "error"
            _fail(field, f"{code} requires {missing_field}")
        if unknown:
            _fail("error", f"{code} does not allow {sorted(unknown)[0]}")
        if self.reason is not None:
            if code == "cursor-invalid":
                object.__setattr__(
                    self,
                    "reason",
                    _literal(
                        self.reason,
                        "error.reason",
                        frozenset({"malformed", "query-mismatch", "stale-snapshot", "expired"}),
                    ),
                )
            elif code == "selection-required":
                object.__setattr__(
                    self,
                    "reason",
                    _literal(
                        self.reason,
                        "error.reason",
                        frozenset(
                            {
                                "digestless",
                                "different-digest",
                                "multiple-variants",
                                "ambiguous-alias",
                            }
                        ),
                    ),
                )
            else:
                object.__setattr__(self, "reason", _reason(self.reason, "error.reason"))
        if self.field is not None:
            object.__setattr__(self, "field", _string(self.field, "error.field"))
        if self.candidates is not None:
            candidates = _copy_collection(self.candidates, "error.candidates", maximum=100)
            if any(type(item) is not CandidateV1 for item in candidates):
                _fail("error.candidates", "must contain at most 100 CandidateV1 values")
            _validate_candidate_order(candidates, "error.candidates")
            object.__setattr__(self, "candidates", candidates)
        if self.truncated is not None:
            object.__setattr__(self, "truncated", _boolean(self.truncated, "error.truncated"))
        if self.requirement_id is not None:
            object.__setattr__(
                self,
                "requirement_id",
                _string(
                    self.requirement_id,
                    "error.requirementId",
                    nonempty=True,
                    maximum=256,
                    printable=True,
                ),
            )
        if self.selection is not None and type(self.selection) is not CandidateSelectionV1:
            _fail("error.selection", "must be CandidateSelectionV1")
        for attribute in ("expected_kind", "actual_kind"):
            item = getattr(self, attribute)
            if item is not None:
                object.__setattr__(self, attribute, _identifier(item, f"error.{attribute}"))
        for attribute in ("expected_digest", "observed_digest"):
            item = getattr(self, attribute)
            if item is not None:
                object.__setattr__(self, attribute, _digest(item, f"error.{attribute}"))
        for attribute in ("expected_size", "observed_size"):
            item = getattr(self, attribute)
            if item is not None:
                object.__setattr__(self, attribute, _integer(item, f"error.{attribute}"))
        if code == "integrity-mismatch":
            digest_pair = self.expected_digest is not None and self.observed_digest is not None
            size_pair = self.expected_size is not None and self.observed_size is not None
            if not digest_pair and not size_pair:
                _fail("error", "integrity-mismatch requires a complete digest or size pair")
            if digest_pair and self.expected_digest == self.observed_digest:
                _fail("error", "digest pair must mismatch")
            if size_pair and self.expected_size == self.observed_size:
                _fail("error", "size pair must mismatch")


def _error_fields(code: str) -> tuple[set[str], set[str]]:
    if code == "invalid-request":
        return {"reason"}, {"field"}
    if code == "cursor-invalid":
        return {"reason"}, set()
    if code in {"authentication-required", "forbidden", "not-found"}:
        return {"reason"}, set()
    if code in {"selection-required", "not-available", "incompatible"}:
        return {"reason", "candidates", "truncated"}, set()
    if code in {
        "credential-required",
        "license-required",
        "cost-required",
        "policy-override-required",
    }:
        return {"requirement_id", "selection"}, set()
    if code in {
        "source-unavailable",
        "no-compatible-destination",
        "mapping-conflict",
        "acquisition-failed",
        "service-unavailable",
    }:
        return {"reason"}, {"selection"}
    if code == "wrong-kind":
        return {"expected_kind"}, {"actual_kind"}
    if code == "integrity-mismatch":
        return set(), {"expected_digest", "observed_digest", "expected_size", "observed_size"}
    raise AssertionError(code)


@dataclass(frozen=True, slots=True)
class ErrorResponseV1:
    error: ErrorV1

    def __post_init__(self) -> None:
        if type(self.error) is not ErrorV1:
            _fail("error", "must be ErrorV1")


_ERROR_WIRE_TO_ATTR = {
    "field": "field",
    "candidates": "candidates",
    "truncated": "truncated",
    "requirementId": "requirement_id",
    "selection": "selection",
    "expectedKind": "expected_kind",
    "actualKind": "actual_kind",
    "expectedDigest": "expected_digest",
    "observedDigest": "observed_digest",
    "expectedSize": "expected_size",
    "observedSize": "observed_size",
}
_ERROR_ATTR_TO_WIRE = {value: key for key, value in _ERROR_WIRE_TO_ATTR.items()}


def decode_error_response(value: object, *, status: int) -> ErrorResponseV1:
    envelope = _object(value, "response", required=frozenset({"contractVersion", "error"}))
    _contract_version(envelope["contractVersion"], "contractVersion")
    raw = _object(
        envelope["error"],
        "error",
        required=frozenset({"code"}),
        optional=frozenset({"reason", *_ERROR_WIRE_TO_ATTR}),
    )
    code = _literal(raw["code"], "error.code", frozenset(_ERROR_STATUS))
    if _ERROR_STATUS[code] != status:
        _fail("error.code", f"requires HTTP status {_ERROR_STATUS[code]}")
    kwargs: dict[str, object] = {"code": code}
    if "reason" in raw:
        kwargs["reason"] = _string(raw["reason"], "error.reason")
    for wire, attribute in _ERROR_WIRE_TO_ATTR.items():
        if wire not in raw:
            continue
        item = raw[wire]
        if item is None:
            _fail(f"error.{wire}", "must be absent rather than null")
        if wire == "candidates":
            item = _candidates(item, "error.candidates", maximum=100)
        elif wire == "selection":
            item = decode_candidate_selection(item, "error.selection")
        kwargs[attribute] = item
    return ErrorResponseV1(ErrorV1(**kwargs))  # type: ignore[arg-type]


def encode_error_response(value: ErrorResponseV1, *, status: int) -> JsonObject:
    checked = ErrorV1(
        **{field: getattr(value.error, field) for field in value.error.__dataclass_fields__}
    )
    expected_status = _ERROR_STATUS[checked.code]
    if status != expected_status:
        _fail("error.code", f"requires HTTP status {expected_status}")
    error: JsonObject = {"code": checked.code}
    if checked.reason is not None:
        error["reason"] = checked.reason
    for attribute, wire in _ERROR_ATTR_TO_WIRE.items():
        item = getattr(checked, attribute)
        if item is None:
            continue
        if attribute == "candidates":
            item = [
                encode_candidate(candidate) for candidate in cast(tuple[CandidateV1, ...], item)
            ]
        elif attribute == "selection":
            item = encode_candidate_selection(cast(CandidateSelectionV1, item))
        error[wire] = item
    return {"contractVersion": 1, "error": error}
