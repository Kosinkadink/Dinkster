import gc
from typing import cast

import pytest
from dinkster_values import (
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    CURVE_TYPE,
    INLINE_STRING_CAP,
    Curve,
    EncodedPayload,
    TypeRegistry,
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


# -- inline scalars (DESIGN 3.5) ----------------------------------------------


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
