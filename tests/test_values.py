import gc
import threading
from typing import cast

import pytest
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    CURVE_TYPE,
    INLINE_STRING_CAP,
    Curve,
    EncodedPayload,
    TypeRegistry,
    default_decode,
    default_encode,
    register_core_types,
)


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def test_curve_boundary_preserves_declared_dependencies_and_source_forms() -> None:
    from dinkster_assets import AssetRef, digest_bytes
    from dinkster_assets.value import value_asset_references
    from dinkster_workers.boundary import ValueCodec

    registry = make_registry()
    codec = ValueCodec(registry)
    curve = Curve(((2, -3), (-1, 4), (2, 7)), "linear")
    wrapped = registry.wrap(CURVE_TYPE, curve)
    assert value_asset_references({"curve": wrapped}, complete_registry=registry) == {}
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    ref = AssetRef(digest_bytes(b"curve-source"), "source.json", 12)
    blobs[cast(int, wire["metaBlob"])] = default_encode(
        {**wrapped.meta.entries, "asset_refs": [ref.to_wire()]}
    )
    received, _ = codec.decode(wire, blobs, [])
    assert received.resolve() == curve
    assert received.fingerprint == wrapped.fingerprint
    assert value_asset_references({"curve": received}, complete_registry=registry) == {
        ref.digest: ref
    }
    payload = cast("dict[str, object]", wire["payload"])
    oversized = Curve(tuple((index, -index) for index in range(4097)))
    for encoded, expected in (
        (b'{"points":[{"position":0,"value":1}]}', Curve(((0, 1),))),
        (b"[]", Curve(())),
        (registry.spec(CURVE_TYPE).encode(oversized), oversized),
    ):
        blobs[cast(int, payload["blob"])] = encoded
        received, _ = codec.decode(wire, blobs, [])
        assert received.resolve() == expected


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"points":[],"asset":{"digest":"hidden"}}',
        b'{"points":[{"position":0,"value":1,"asset":"hidden"}]}',
        b'{"points":[[0,{"asset":"hidden"}]]}',
        b'{"points":[[false,1]]}',
        b'{"points":[[0,NaN]]}',
        b"not json",
    ],
)
def test_curve_boundary_refuses_invalid_payload_before_exposure(encoded) -> None:
    from dinkster_workers.boundary import EncodedPayloadValidationError, ValueCodec

    registry = make_registry()
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(registry.wrap(CURVE_TYPE, Curve(())), blobs, segments)
    payload = cast("dict[str, object]", wire["payload"])
    blobs[cast(int, payload["blob"])] = encoded
    with pytest.raises(EncodedPayloadValidationError):
        codec.decode(wire, blobs, [])


def test_type_claims_refuse_overlapping_owners_and_preserve_reference_access() -> None:
    registry = make_registry()
    with registry.registering_pack("mypack", ("mypack",)):
        registry.register("mypack.thing")
    assert registry.owner_of("mypack.thing") == "mypack"
    assert registry.owner_of(CORE_INT) == "core"
    for claim in ("mypack", "mypack.child", "mypack-child"):
        with pytest.raises(ValueError, match="overlaps"):
            registry.claim_types("other", (claim,))
    with registry.registering_pack("consumer", ("consumer",)):
        assert registry.wrap("mypack.thing", {"count": 3}).payload.load() == {"count": 3}
        with pytest.raises(ValueError, match="must be prefixed"):
            registry.register("foreign.thing")
    assert registry.type_owners()["mypack.thing"] == "mypack"


def test_type_claims_survive_staging_without_leaking_to_the_live_registry() -> None:
    live = make_registry()
    staged = live.copy()
    with staged.registering_pack("mypack", ("mypack.thing",)):
        staged.register("mypack.thing")
    assert live.owner_of("mypack.thing") is None
    live.replace_from(staged)
    assert live.owner_of("mypack.thing") == "mypack"
    with pytest.raises(ValueError, match="overlaps"):
        live.claim_types("other", ("mypack",))


def test_type_registration_enforces_exact_claim_and_pack_prefix() -> None:
    registry = make_registry()
    with registry.registering_pack("mypack", ("mypack.thing",)):
        with pytest.raises(ValueError, match="outside its declared type claims"):
            registry.register("mypack.other")
        with pytest.raises(ValueError, match="must be prefixed"):
            registry.register("THING")
        registry.register("mypack.thing")
    with registry.registering_pack("official", ("THING",), allow_bare=True):
        registry.register("THING")
    with pytest.raises(ValueError, match="overlaps"):
        registry.claim_types("another-official", ("THING",))


def test_declared_type_can_activate_and_release_its_codec_and_providers() -> None:
    registry = make_registry()
    registry.declare_types("mypack", ("mypack",), ("mypack.thing",))
    assert registry.owner_of("mypack.thing") == "mypack"
    assert "mypack.thing" not in registry
    with registry.registering_pack("mypack", ("mypack",)):
        registry.register("mypack.thing")
    registry.register_asset_decoder(
        "list<mypack.thing>", provider_id="mypack.decoder", decode=lambda obj: obj
    )
    registry.register_batch_merge("mypack.thing", provider_id="mypack.merge", merge=list)
    registry.remove_pack_types("mypack")
    assert registry.owner_of("mypack.thing") is None
    assert "mypack.thing" not in registry
    assert registry.asset_decoder_for("list<mypack.thing>") is None
    assert registry.batch_merge_for("mypack.thing") is None
    assert registry.owner_of(CORE_INT) == "core"
    assert registry.wrap(CORE_INT, 17).payload.load() == 17
    with registry.registering_pack("mypack", ("mypack",)):
        registry.register("mypack.thing")
    assert registry.wrap("mypack.thing", 23).payload.load() == 23


def test_builtin_curve_owner_survives_pack_removal_in_core_registry() -> None:
    registry = TypeRegistry()
    assert registry.owner_of(CURVE_TYPE) == "core"
    with registry.registering_pack("official-provider", (), allow_bare=True):
        register_core_types(registry)
    registry.declare_types("official-provider", (), (CURVE_TYPE,), allow_bare=True)
    assert registry.owner_of(CURVE_TYPE) == "core"
    assert CURVE_TYPE in registry
    for pack in ("another-official", "dinkster-nodes-foundation"):
        with pytest.raises(ValueError, match="overlaps"):
            registry.claim_types(pack, (CURVE_TYPE,))
    registry.remove_pack_types("official-provider")
    registry.remove_pack_types("dinkster-nodes-foundation")
    assert registry.type_owners()[CURVE_TYPE] == "core"
    assert registry.wrap(CURVE_TYPE, {"points": [[2, 7]]}).resolve() == Curve(((2, 7),))
    assert registry.wrap(CORE_INT, 17).resolve() == 17


def test_encoded_buffer_limits_access_to_its_logical_size() -> None:
    payload = EncodedPayload.from_buffer(
        "test.value",
        memoryview(b"payload-padding"),
        None,
        "shm",
        lambda: None,
        size=7,
    )

    assert payload.size == 7
    with payload.borrow_data() as view:
        assert bytes(view) == b"payload"
    assert payload.data == b"payload"


def test_encoded_buffer_borrow_pins_owner_past_payload_lifetime() -> None:
    released: list[bool] = []
    payload = EncodedPayload.from_buffer(
        "test.value",
        memoryview(b"payload"),
        None,
        "shm",
        lambda: released.append(True),
    )
    borrowed = payload.borrow_data()
    view = borrowed.__enter__()

    del payload
    gc.collect()
    assert bytes(view) == b"payload"
    assert released == []

    borrowed.__exit__(None, None, None)
    del borrowed, view
    gc.collect()
    assert released == [True]


def test_encoded_buffer_concurrent_borrow_and_materialize() -> None:
    released: list[bool] = []
    payload = EncodedPayload.from_buffer(
        "test.value",
        memoryview(b"payload"),
        None,
        "shm",
        lambda: released.append(True),
    )
    barrier = threading.Barrier(8)
    results: list[bytes] = []

    def borrow() -> None:
        barrier.wait()
        with payload.borrow_data() as view:
            results.append(bytes(view))

    def materialize() -> None:
        barrier.wait()
        results.append(payload.data)

    threads = [threading.Thread(target=borrow) for _ in range(4)] + [
        threading.Thread(target=materialize) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [b"payload"] * 8
    assert released == [True]


def test_same_payload_different_type_differs() -> None:
    registry = make_registry()
    registry.register("custom.answer")
    assert registry.wrap(CORE_INT, 42).fingerprint != registry.wrap("custom.answer", 42).fingerprint


@pytest.mark.parametrize(
    ("type_id", "item", "encoded"),
    [
        (CORE_INT, -(2**63), b"json:-9223372036854775808"),
        (CORE_INT, 2**64 - 1, b"json:18446744073709551615"),
        (CORE_FLOAT, 3, b"json:3"),
        (CORE_FLOAT, -2.75, b"json:-2.75"),
        (CORE_BOOLEAN, False, b"json:false"),
        (CORE_COMBO, True, b"json:true"),
        (CORE_COMBO, 2**53 - 1, b"json:9007199254740991"),
        (CORE_STRING, "outside-menu", b'json:"outside-menu"'),
        ("HISTOGRAM", [2, 0, 7], b"json:[2,0,7]"),
        ("core.absent", None, b"json:null"),
    ],
)
def test_core_primitive_dependencies_follow_declared_metadata(type_id, item, encoded) -> None:
    from dinkster_assets import AssetRef, digest_bytes
    from dinkster_assets.value import value_asset_references
    from dinkster_workers.boundary import ValueCodec

    registry = make_registry()
    spec = registry.spec(type_id)
    wrapped = registry.wrap(type_id, item)
    assert spec.encode(wrapped.resolve()) == encoded
    assert value_asset_references({"value": wrapped}, complete_registry=registry) == {}
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    received, _ = codec.decode(wire, blobs, [])
    assert received.resolve() == item
    assert type(received.resolve()) is type(item)
    assert received.fingerprint == wrapped.fingerprint
    assert value_asset_references({"value": received}, complete_registry=registry) == {}
    validator = spec.validate_encoded
    assert validator is not None
    validator(encoded, {"asset_refs": []})
    ref = AssetRef(digest_bytes(b"external"), "external.bin", 8)
    blobs[cast(int, wire["metaBlob"])] = default_encode(
        {**wrapped.meta.entries, "asset_refs": [ref.to_wire()]}
    )
    received, _ = codec.decode(wire, blobs, [])
    assert received.resolve() == item
    assert value_asset_references({"value": received}, complete_registry=registry) == {
        ref.digest: ref
    }


@pytest.mark.parametrize(
    ("type_id", "invalid"),
    [
        (CORE_INT, True),
        (CORE_INT, 3.0),
        (CORE_INT, {"asset": "hidden"}),
        (CORE_FLOAT, True),
        (CORE_FLOAT, float("inf")),
        (CORE_FLOAT, float("nan")),
        (CORE_FLOAT, [1.5]),
        (CORE_BOOLEAN, 1),
        (CORE_BOOLEAN, "true"),
        ("core.absent", 0),
        ("core.absent", {"asset": "hidden"}),
    ],
)
def test_core_primitives_refuse_non_scalar_runtime_and_encoded_values(type_id, invalid) -> None:
    registry = make_registry()
    with pytest.raises((TypeError, ValueError)):
        registry.wrap(type_id, invalid)
    validator = registry.spec(type_id).validate_encoded
    assert validator is not None
    for encoded in (default_encode(invalid), b"pickle:unused", b"json: 3"):
        with pytest.raises((TypeError, ValueError)):
            validator(encoded, {})


class ComboStringSubclass(str):
    pass


@pytest.mark.parametrize(
    "selected", [False, True, 0, 1, 8, 10, 1.5, -2.75, "8", -(2**53 - 1), 2**53 - 1]
)
def test_core_combo_preserves_scalar_type_and_identity(selected: object) -> None:
    registry = make_registry()
    value = registry.wrap(CORE_COMBO, selected)
    spec = registry.spec(CORE_COMBO)
    decoded = spec.decode(spec.encode(value.resolve()))
    assert type(decoded) is type(selected)
    assert decoded == selected
    assert type(registry.inline_of(value)) is type(selected)
    assert registry.inline_of(value) == selected
    assert spec.validate_encoded is not None
    spec.validate_encoded(spec.encode(selected), {})
    assert (
        value.fingerprint != registry.wrap(CORE_COMBO, str(selected)).fingerprint
        or type(selected) is str
    )
    if type(selected) is bool:
        assert value.fingerprint != registry.wrap(CORE_COMBO, int(selected)).fingerprint


class Odd:
    """Module-level so pickle can serialize it."""

    def __init__(self) -> None:
        self.x = 7

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Odd) and other.x == self.x

    def __hash__(self) -> int:
        return hash(self.x)


def test_default_codec_pickle_fallback() -> None:
    data = default_encode(Odd())
    assert data.startswith(b"pickle:")
    assert default_decode(data) == Odd()


def test_duplicate_registration_rejected() -> None:
    registry = make_registry()
    try:
        registry.register(CORE_INT)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# -- inline scalars (DESIGN 3.5) ----------------------------------------------


def test_core_scalars_inline() -> None:
    """int/float/bool/short string carry their value inline so peeking them
    never needs a payload round trip."""
    registry = make_registry()
    assert registry.inline_of(registry.wrap(CORE_INT, 42)) == 42
    assert registry.inline_of(registry.wrap(CORE_FLOAT, 2.5)) == 2.5
    assert registry.inline_of(registry.wrap(CORE_BOOLEAN, True)) is True
    assert registry.inline_of(registry.wrap(CORE_STRING, "hi")) == "hi"


def test_long_strings_do_not_inline() -> None:
    """INLINE_STRING_CAP is a central policy, not a per-type opt-out: the
    inline channel carries peeks, never documents."""
    registry = make_registry()
    at_cap = registry.wrap(CORE_STRING, "x" * INLINE_STRING_CAP)
    text = "y" * (INLINE_STRING_CAP + 1) + "\u03a9\n"
    over_cap = registry.wrap(CORE_STRING, text)
    assert registry.inline_of(at_cap) == "x" * INLINE_STRING_CAP
    assert registry.inline_of(over_cap) is None
    rendered = registry.render(over_cap, "text")
    assert rendered.mime == "text/plain"
    assert rendered.data == text.encode("utf-8")
    assert registry.render(at_cap).data == b"x" * INLINE_STRING_CAP


def test_buggy_inline_fn_degrades_to_not_inline() -> None:
    """Inline rides event/descriptor serialization paths: a pack bug must
    degrade to omission, never fail the run."""
    registry = make_registry()

    def boom(obj: object) -> object:
        raise RuntimeError("pack bug")

    registry.register("mypack.buggy", inline=boom)
    assert registry.inline_of(registry.wrap("mypack.buggy", 1)) is None


# -- renditions (DESIGN 3.5) ---------------------------------------------------


def test_rendition_default_negotiation() -> None:
    registry = make_registry()
    registry.register("mypack.doc")
    registry.register_rendition("mypack.doc", "txt", mime="text/plain", render=lambda obj: b"t")
    registry.register_rendition("mypack.doc", "html", mime="text/html", render=lambda obj: b"h")
    kinds = {(s.kind, s.default) for s in registry.renditions_of("mypack.doc")}
    assert kinds == {("txt", True), ("html", False)}
    assert registry.render(registry.wrap("mypack.doc", "x")).kind == "txt"
    assert registry.render(registry.wrap("mypack.doc", "x"), "html").data == b"h"
