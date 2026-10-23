from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest
from dinkster_server import federated_assets_v1 as dto

DIGEST_A = "blake3:" + "a" * 64
DIGEST_B = "blake3:" + "b" * 64


def asset_ref_wire(digest: str = DIGEST_A, *, name: str = "model.safetensors") -> dict[str, object]:
    return {
        "digest": digest,
        "name": name,
        "size": 123,
        "mediaType": "application/octet-stream",
        "virtualPath": "models/model.safetensors",
    }


def candidate_wire(
    digest: str = DIGEST_A,
    *,
    logical_id: str = "flux/dev",
    availability: str = "local",
    compatibility: str = "compatible",
) -> dict[str, object]:
    wire: dict[str, object] = {
        "logicalId": logical_id,
        "family": "flux",
        "assetKind": "model/diffusion",
        "variantId": "fp8",
        "dtype": "float8-e4m3fn",
        "quantization": "fp8",
        "format": "safetensors",
        "role": "diffusion",
        "requirements": {
            "loaders": ["flux"],
            "runtimes": [],
            "hardware": ["cuda"],
        },
        "digest": digest,
        "size": 123,
        "mediaType": "application/octet-stream",
        "availability": {"status": availability, "reason": ""},
        "compatibility": {
            "status": compatibility,
            "reason": "" if compatibility == "compatible" else "unsupported",
        },
        "providerSources": [
            {
                "source": {"providerId": "provider-a", "sourceId": "source-a"},
                "status": "available",
                "reason": "",
                "requires": {},
            }
        ],
    }
    if availability == "local":
        wire["assetRef"] = asset_ref_wire(digest)
    return wire


def context_wire() -> dict[str, object]:
    return {
        "assetKind": "model/diffusion",
        "schema": {"nodeType": "dinkster.load", "inputId": "model"},
        "accept": [],
    }


def selected_wire(digest: str = DIGEST_A) -> dict[str, object]:
    return {
        "logicalId": "flux/dev",
        "variantId": "fp8",
        "digest": digest,
        "reason": "expected-digest",
    }


def round_trip(decoder: object, encoder: object, wire: dict[str, object]) -> None:
    decode = decoder  # keep assertions readable without a generic helper dependency
    encode = encoder
    model = decode(wire)  # type: ignore[operator]
    encoded = encode(model)  # type: ignore[operator]
    assert encoded == wire
    assert json.loads(json.dumps(encoded)) == wire


def test_federated_asset_dto_v1_minimal_requests_and_defaults_round_trip() -> None:
    catalog = dto.decode_catalog_request({"contractVersion": 1})
    candidates = dto.decode_candidates_request({"contractVersion": 1, "context": context_wire()})

    assert catalog.limit == candidates.limit == 50
    assert dto.encode_catalog_request(catalog) == {"contractVersion": 1, "limit": 50}
    assert dto.encode_candidates_request(candidates)["limit"] == 50


def test_federated_asset_dto_v1_all_optional_request_fields_round_trip() -> None:
    selection = {
        "logicalId": "flux/dev",
        "variantId": "fp8",
        "digest": DIGEST_A,
        "source": {"providerId": "provider-a", "sourceId": "source-a"},
    }
    catalog = {
        "contractVersion": 1,
        "scope": "workspace-a",
        "query": "flux",
        "assetKind": "model/diffusion",
        "context": {
            **context_wire(),
            "schema": {
                "nodeType": "dinkster.load",
                "inputId": "model",
                "typeId": "dinkster.asset",
            },
            "accept": ["application/safetensors", "model/*"],
        },
        "availability": ["local", "downloadable"],
        "compatibility": ["compatible", "unknown"],
        "cursor": "opaque cursor",
        "limit": 100,
    }
    candidates = {
        "contractVersion": 1,
        "scope": "workspace-a",
        "context": context_wire(),
        "expected": {"digest": DIGEST_A, "size": 123},
        "hints": {
            "source": "provider",
            "reference": "workflow:model",
            "loaderPath": "loader",
            "modelType": "flux",
            "displayName": "Flux",
        },
        "selection": selection,
        "cursor": "opaque cursor",
        "limit": 1,
    }
    round_trip(dto.decode_catalog_request, dto.encode_catalog_request, catalog)
    round_trip(dto.decode_candidates_request, dto.encode_candidates_request, candidates)


@pytest.mark.parametrize(
    ("wire", "field"),
    [
        ({}, "contractVersion"),
        ({"contractVersion": True}, "contractVersion"),
        ({"contractVersion": 1, "extra": 1}, "request"),
        ({"contractVersion": 1, "scope": None}, "scope"),
        ({"contractVersion": 1, "scope": "has space"}, "scope"),
        ({"contractVersion": 1, "query": "x" * 513}, "query"),
        ({"contractVersion": 1, "cursor": "x" * 4097}, "cursor"),
        ({"contractVersion": 1, "limit": True}, "limit"),
        ({"contractVersion": 1, "limit": 0}, "limit"),
        ({"contractVersion": 1, "availability": ("local",)}, "availability"),
        ({"contractVersion": 1, "availability": ["local", "local"]}, "availability"),
        ({"contractVersion": 1, "compatibility": ["other"]}, "compatibility[0]"),
    ],
)
def test_federated_asset_dto_v1_catalog_rejects_closed_shape_and_type_errors(
    wire: dict[str, object], field: str
) -> None:
    with pytest.raises(dto.FederatedAssetCodecError) as caught:
        dto.decode_catalog_request(wire)
    assert caught.value.field == field


@pytest.mark.parametrize("accept", ["*/*", "image", "image/", " image/png", "image/png "])
def test_federated_asset_dto_v1_context_mime_and_identifier_bounds(accept: str) -> None:
    wire = context_wire()
    wire["accept"] = [accept]
    with pytest.raises(dto.FederatedAssetCodecError, match="context.accept"):
        dto.decode_resolve_context(wire)

    over = context_wire()
    over["schema"] = {"nodeType": "x" * 513, "inputId": "model"}
    with pytest.raises(dto.FederatedAssetCodecError, match="nodeType"):
        dto.decode_resolve_context(over)


def test_federated_asset_dto_v1_asset_ref_is_exact_and_safe() -> None:
    for change in (
        {"digest": "A" * 64},
        {"size": True},
        {"size": 2**53},
        {"mediaType": None},
        {"extra": "no"},
    ):
        wire = asset_ref_wire()
        wire.update(change)
        candidate = candidate_wire()
        candidate["assetRef"] = wire
        with pytest.raises(dto.FederatedAssetCodecError):
            dto.decode_candidate(candidate)

    empty_strings = asset_ref_wire(name="")
    empty_strings["mediaType"] = ""
    empty_strings["virtualPath"] = ""
    candidate = candidate_wire()
    candidate["assetRef"] = empty_strings
    assert dto.decode_candidate(candidate).asset_ref is not None


def test_federated_asset_dto_v1_candidate_round_trip_and_invariants() -> None:
    wire = candidate_wire()
    candidate = dto.decode_candidate(wire)
    assert dto.encode_candidate(candidate) == wire

    bad = candidate_wire()
    bad["assetRef"] = asset_ref_wire(DIGEST_B)
    with pytest.raises(dto.FederatedAssetCodecError, match="must equal"):
        dto.decode_candidate(bad)

    bad = candidate_wire(availability="downloadable")
    bad["assetRef"] = asset_ref_wire()
    with pytest.raises(dto.FederatedAssetCodecError, match="allowed only"):
        dto.decode_candidate(bad)

    bad = candidate_wire(compatibility="unknown")
    compatibility = bad["compatibility"]
    assert isinstance(compatibility, dict)
    compatibility["reason"] = ""
    with pytest.raises(dto.FederatedAssetCodecError, match="must be nonempty"):
        dto.decode_candidate(bad)

    bad = candidate_wire()
    bad["providerSources"] = [
        {
            "source": {"providerId": "z", "sourceId": "z"},
            "status": "available",
            "reason": "",
            "requires": {},
        },
        {
            "source": {"providerId": "a", "sourceId": "a"},
            "status": "available",
            "reason": "",
            "requires": {},
        },
    ]
    with pytest.raises(dto.FederatedAssetCodecError, match="strictly ordered"):
        dto.decode_candidate(bad)


def test_federated_asset_dto_v1_candidate_and_provider_output_order_is_verified() -> None:
    first = candidate_wire(logical_id="a")
    second = candidate_wire(DIGEST_B, logical_id="b")
    response = {"contractVersion": 1, "items": [first, second]}
    assert dto.encode_catalog_response(dto.decode_catalog_response(response)) == response

    response["items"] = [second, first]
    with pytest.raises(dto.FederatedAssetCodecError, match="strictly ordered"):
        dto.decode_catalog_response(response)

    response["items"] = [first, first]
    with pytest.raises(dto.FederatedAssetCodecError, match="strictly ordered"):
        dto.decode_catalog_response(response)


def test_federated_asset_dto_v1_expected_selection_and_selected_states() -> None:
    request = {
        "contractVersion": 1,
        "context": context_wire(),
        "expected": {"digest": DIGEST_A},
        "selection": {
            "logicalId": "flux/dev",
            "variantId": "fp8",
            "digest": DIGEST_B,
        },
    }
    with pytest.raises(dto.FederatedAssetCodecError, match="must equal"):
        dto.decode_candidates_request(request)

    resolved = {
        "contractVersion": 1,
        "status": "resolved",
        "items": [candidate_wire()],
        "selectedCandidate": selected_wire(),
    }
    assert dto.encode_candidates_response(dto.decode_candidates_response(resolved)) == resolved

    missing = {**resolved, "status": "missing"}
    assert dto.decode_candidates_response(missing).selected_candidate is not None
    incompatible = {
        **resolved,
        "status": "incompatible",
        "items": [candidate_wire(compatibility="incompatible")],
    }
    assert dto.decode_candidates_response(incompatible).selected_candidate is not None

    missing_selection = dict(resolved)
    del missing_selection["selectedCandidate"]
    with pytest.raises(dto.FederatedAssetCodecError):
        dto.decode_candidates_response(missing_selection)

    for status in ("resolved", "missing", "incompatible"):
        paginated = {
            "contractVersion": 1,
            "status": status,
            "items": [],
            "selectedCandidate": selected_wire(DIGEST_B),
        }
        assert (
            dto.encode_candidates_response(dto.decode_candidates_response(paginated)) == paginated
        )

    for item in (
        candidate_wire(availability="downloadable"),
        candidate_wire(compatibility="incompatible"),
    ):
        with pytest.raises(dto.FederatedAssetCodecError, match="local and compatible"):
            dto.decode_candidates_response({**resolved, "items": [item]})

    ambiguous = {**resolved, "status": "ambiguous"}
    with pytest.raises(dto.FederatedAssetCodecError, match="not allowed"):
        dto.decode_candidates_response(ambiguous)


ERROR_CASES = [
    (400, {"code": "invalid-request", "reason": "bad", "field": "limit"}),
    (400, {"code": "cursor-invalid", "reason": "expired"}),
    (401, {"code": "authentication-required", "reason": "required"}),
    (403, {"code": "forbidden", "reason": "denied"}),
    (404, {"code": "not-found", "reason": "missing"}),
    (
        409,
        {
            "code": "selection-required",
            "reason": "different-digest",
            "candidates": [candidate_wire()],
            "truncated": False,
        },
    ),
    (
        409,
        {
            "code": "not-available",
            "reason": "missing",
            "candidates": [],
            "truncated": True,
        },
    ),
    (
        409,
        {
            "code": "incompatible",
            "reason": "unsupported",
            "candidates": [],
            "truncated": False,
        },
    ),
    *[
        (
            409,
            {
                "code": code,
                "requirementId": "requirement-a",
                "selection": {
                    "logicalId": "flux/dev",
                    "variantId": "fp8",
                    "digest": DIGEST_A,
                },
            },
        )
        for code in (
            "credential-required",
            "license-required",
            "cost-required",
            "policy-override-required",
        )
    ],
    *[
        (409, {"code": code, "reason": "unavailable"})
        for code in ("source-unavailable", "no-compatible-destination", "mapping-conflict")
    ],
    (422, {"code": "wrong-kind", "expectedKind": "model/diffusion"}),
    (
        422,
        {
            "code": "integrity-mismatch",
            "expectedDigest": DIGEST_A,
            "observedDigest": DIGEST_B,
            "expectedSize": 1,
            "observedSize": 2,
        },
    ),
    (502, {"code": "acquisition-failed", "reason": "failed"}),
    (503, {"code": "service-unavailable", "reason": "acquisition-not-enabled"}),
]


@pytest.mark.parametrize(("status", "error"), ERROR_CASES)
def test_federated_asset_dto_v1_all_error_variants_round_trip(
    status: int, error: dict[str, object]
) -> None:
    wire = {"contractVersion": 1, "error": error}
    model = dto.decode_error_response(wire, status=status)
    assert dto.encode_error_response(model, status=status) == wire


def test_federated_asset_dto_v1_error_variants_are_closed_and_status_bound() -> None:
    with pytest.raises(dto.FederatedAssetCodecError, match="HTTP status"):
        dto.decode_error_response(
            {"contractVersion": 1, "error": {"code": "not-found", "reason": "x"}},
            status=400,
        )
    with pytest.raises(dto.FederatedAssetCodecError, match="does not allow"):
        dto.decode_error_response(
            {
                "contractVersion": 1,
                "error": {"code": "not-found", "reason": "x", "field": "x"},
            },
            status=404,
        )
    for error in (
        {"code": "integrity-mismatch", "expectedDigest": DIGEST_A},
        {
            "code": "integrity-mismatch",
            "expectedSize": 1,
            "observedSize": 1,
        },
    ):
        with pytest.raises(dto.FederatedAssetCodecError, match="mismatch|complete"):
            dto.decode_error_response({"contractVersion": 1, "error": error}, status=422)


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, {"code": "invalid-request", "reason": "bad", "field": None}),
        (422, {"code": "wrong-kind", "expectedKind": "model/diffusion", "actualKind": None}),
        (
            422,
            {
                "code": "integrity-mismatch",
                "expectedDigest": DIGEST_A,
                "observedDigest": DIGEST_B,
                "expectedSize": None,
            },
        ),
    ],
)
def test_federated_asset_dto_v1_error_optional_fields_reject_null(
    status: int, error: dict[str, object]
) -> None:
    with pytest.raises(dto.FederatedAssetCodecError, match="absent rather than null"):
        dto.decode_error_response({"contractVersion": 1, "error": error}, status=status)


@pytest.mark.parametrize("field", ["", "line\nfield", "x" * 4097])
def test_federated_asset_dto_v1_error_field_is_an_unbounded_exact_string(field: str) -> None:
    wire = {
        "contractVersion": 1,
        "error": {"code": "invalid-request", "reason": "bad", "field": field},
    }
    assert (
        dto.encode_error_response(dto.decode_error_response(wire, status=400), status=400) == wire
    )


def test_federated_asset_dto_v1_response_pages_refuse_more_than_100_items() -> None:
    wires = [candidate_wire(logical_id=f"candidate-{index:03}") for index in range(101)]
    items = tuple(dto.decode_candidate(wire) for wire in wires)
    with pytest.raises(dto.FederatedAssetCodecError, match="at most 100"):
        dto.CatalogResponseV1(items)
    with pytest.raises(dto.FederatedAssetCodecError, match="at most 100"):
        dto.CandidatesResponseV1("missing", items)
    with pytest.raises(dto.FederatedAssetCodecError, match="at most 100"):
        dto.decode_catalog_response({"contractVersion": 1, "items": wires})
    with pytest.raises(dto.FederatedAssetCodecError, match="at most 100"):
        dto.decode_candidates_response({"contractVersion": 1, "status": "missing", "items": wires})


def test_federated_asset_dto_v1_constructor_collections_reject_arbitrary_iterables() -> None:
    candidate = dto.decode_candidate(candidate_wire())
    error = dto.ErrorV1(
        "selection-required",
        reason="digestless",
        candidates=(candidate,),
        truncated=False,
    )
    constructors = (
        ("candidate.providerSources", lambda value: replace(candidate, provider_sources=value)),
        ("items", lambda value: dto.CatalogResponseV1(value)),
        ("items", lambda value: dto.CandidatesResponseV1("missing", value)),
        ("error.candidates", lambda value: replace(error, candidates=value)),
    )
    for expected_field, constructor in constructors:
        for value in (1, None, (item for item in ()), set()):
            with pytest.raises(dto.FederatedAssetCodecError) as caught:
                constructor(value)  # type: ignore[arg-type]
            assert caught.value.field == expected_field


def test_federated_asset_dto_v1_records_copy_freeze_hash_and_encode_fresh_json() -> None:
    accepts = ["image/png"]
    context = dto.ResolveContextV1("media/image", "dinkster.load", "image", accepts)  # type: ignore[arg-type]
    accepts.append("image/jpeg")
    assert context.accept == ("image/png",)
    assert hash(context)
    with pytest.raises(FrozenInstanceError):
        context.asset_kind = "changed"  # type: ignore[misc]

    encoded = dto.encode_resolve_context(context)
    encoded_accept = encoded["accept"]
    assert isinstance(encoded_accept, list)
    encoded_accept.append("mutated")
    assert context.accept == ("image/png",)

    wire = candidate_wire()
    candidate = dto.decode_candidate(wire)
    providers = wire["providerSources"]
    assert isinstance(providers, list)
    providers.clear()
    assert len(candidate.provider_sources) == 1
    assert hash(candidate)
