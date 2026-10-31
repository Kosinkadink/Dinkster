import copy
import gc
import pickle
import threading
from collections.abc import Mapping
from dataclasses import asdict, fields, replace
from typing import cast

import pytest
from dinkster_values import (
    ASSET_BASE_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    CURVE_TYPE,
    INLINE_STRING_CAP,
    BufferEncoding,
    Curve,
    EncodedPayload,
    Rendition,
    TypeRegistry,
    Value,
    default_decode,
    default_encode,
    make_absent_value,
    register_core_types,
)


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def test_core_registration_registers_curve_and_preserves_first_authority() -> None:
    registry = make_registry()
    curve = Curve(((3.5, -2.0), (-1.0, 4.25), (0.0, 0.5)), "linear")
    spec = registry.spec(CURVE_TYPE)

    assert spec.decode(spec.encode(curve)) == curve
    assert registry.wrap(CURVE_TYPE, curve).resolve() == curve

    first = TypeRegistry()
    first_spec = first.register(
        CURVE_TYPE, encode=lambda _value: b"first", decode=lambda _data: "first"
    )
    register_core_types(first)

    assert first.spec(CURVE_TYPE) is first_spec
    assert first.spec(CURVE_TYPE).decode(b"ignored") == "first"


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
    assert registry.owner_of(CORE_INT) == "dinkster-nodes-foundation"
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


def test_host_codec_does_not_claim_a_pack_type_before_its_owner_announces() -> None:
    registry = make_registry()
    registry.register("mypack.thing")
    assert registry.owner_of("mypack.thing") is None
    registry.declare_types("mypack", ("mypack",), ("mypack.thing",))
    assert registry.type_owners()["mypack.thing"] == "mypack"
    assert registry.wrap("mypack.thing", 19).payload.load() == 19
    with pytest.raises(ValueError, match="overlaps"):
        registry.claim_types("other", ("mypack",))


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
    assert registry.owner_of(CORE_INT) == "dinkster-nodes-foundation"
    assert registry.wrap(CORE_INT, 17).payload.load() == 17
    with registry.registering_pack("mypack", ("mypack",)):
        registry.register("mypack.thing")
    assert registry.wrap("mypack.thing", 23).payload.load() == 23


def test_declared_asset_targets_survive_staging_without_granting_local_codecs() -> None:
    registry = make_registry()
    registry.register(ASSET_BASE_TYPE)
    staged = registry.copy()
    staged.declare_types("mypack", ("mypack",), ("mypack.thing",))
    assert "asset<mypack.thing>" not in registry
    registry.replace_from(staged)
    assert "asset<list<mypack.thing>>" in registry
    assert registry.spec("asset<mypack.thing>") is registry.spec(ASSET_BASE_TYPE)
    assert "mypack.thing" not in registry
    with pytest.raises(KeyError, match="unregistered value type"):
        registry.spec("mypack.thing")
    with pytest.raises(KeyError, match="decode-target atom"):
        registry.spec("asset<mypack.unknown>")
    registry.remove_pack_types("mypack")
    assert "asset<mypack.thing>" not in registry
    assert "asset<mypack.thing>" in staged
    with pytest.raises(KeyError, match="decode-target atom"):
        registry.spec("asset<mypack.thing>")


def test_builtin_curve_owner_survives_pack_removal_in_core_registry() -> None:
    registry = make_registry()
    assert registry.owner_of(CURVE_TYPE) == "dinkster-nodes-foundation"
    assert CURVE_TYPE in registry
    with pytest.raises(ValueError, match="overlaps"):
        registry.claim_types("another-official", (CURVE_TYPE,))
    registry.remove_pack_types("official-provider")
    registry.remove_pack_types("dinkster-nodes-foundation")
    assert registry.type_owners()[CURVE_TYPE] == "dinkster-nodes-foundation"
    assert registry.wrap(CURVE_TYPE, {"points": [[2, 7]]}).resolve() == Curve(((2, 7),))
    assert registry.wrap(CORE_INT, 17).resolve() == 17


def test_encoded_buffer_is_read_only_and_materializes_on_demand() -> None:
    backing = bytearray(b"encoded payload")
    released: list[bool] = []
    payload = EncodedPayload.from_buffer(
        "test.value",
        memoryview(backing),
        lambda data: data.decode(),
        "shm",
        lambda: released.append(True),
    )

    with payload.borrow_data() as view:
        assert view.readonly
        assert bytes(view) == backing
        assert released == []

    assert payload.data == bytes(backing)
    assert released == [True]
    assert payload.load() == "encoded payload"


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


def test_encoded_buffer_preserves_encoded_payload_dataclass_contract() -> None:
    payload = EncodedPayload.from_buffer(
        "test.value",
        memoryview(b"payload"),
        None,
        "shm",
        lambda: None,
    )
    owned = EncodedPayload("test.value", b"payload", None, "shm")

    assert payload == owned
    assert repr(payload) == repr(owned)
    assert [item.name for item in fields(payload)][:4] == [
        "type_id",
        "data",
        "decoder",
        "transport",
    ]
    assert fields(payload)[3].default == "inline"
    assert EncodedPayload.transport == "inline"
    dataclass_params = EncodedPayload.__dataclass_params__  # pyright: ignore[reportAttributeAccessIssue]
    assert dataclass_params.init is True
    assert asdict(payload)["data"] == b"payload"
    assert replace(payload, transport="inline") == EncodedPayload(
        "test.value", b"payload", None, "inline"
    )
    assert copy.deepcopy(payload) == owned
    assert pickle.loads(pickle.dumps(payload)) == owned


def test_equal_values_share_fingerprints() -> None:
    registry = make_registry()
    assert registry.wrap(CORE_INT, 42).fingerprint == registry.wrap(CORE_INT, 42).fingerprint
    assert registry.wrap(CORE_INT, 42).fingerprint != registry.wrap(CORE_INT, 43).fingerprint


def test_same_payload_different_type_differs() -> None:
    registry = make_registry()
    registry.register("custom.answer")
    assert registry.wrap(CORE_INT, 42).fingerprint != registry.wrap("custom.answer", 42).fingerprint


def test_core_types_use_comfy_socket_identities_without_legacy_aliases() -> None:
    registry = make_registry()

    assert {CORE_INT, CORE_FLOAT, CORE_STRING, CORE_COMBO, CORE_BOOLEAN, "DICT", "ARRAY"} <= set(
        registry.type_ids()
    )
    legacy_type_ids = {f"core.{name}" for name in ("int", "float", "string", "combo", "boolean")}
    assert not legacy_type_ids & set(registry.type_ids())


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


def test_core_string_rejects_embedded_payloads_and_noncanonical_wire() -> None:
    registry = make_registry()
    spec = registry.spec(CORE_STRING)
    assert spec.asset_references_complete
    assert spec.validate_encoded is not None
    text = 'a "quoted" path'
    assert registry.wrap(CORE_STRING, text).resolve() == text
    assert spec.encode(text) == b'json:"a \\"quoted\\" path"'
    spec.validate_encoded(spec.encode(text), {})
    for invalid in (None, True, 7, {"digest": "not-a-scalar"}, ["text"]):
        with pytest.raises(TypeError, match="STRING expects a string"):
            registry.wrap(CORE_STRING, invalid)
    for wire in (b"pickle:unused", b"json:null", b'json:{"asset":1}', b'json: "spaced"'):
        with pytest.raises((TypeError, ValueError)):
            spec.validate_encoded(wire, {})


def test_core_combo_is_string_shaped_but_type_distinct() -> None:
    registry = make_registry()
    assert CORE_COMBO in registry

    combo = registry.wrap(CORE_COMBO, "euler")
    string = registry.wrap(CORE_STRING, "euler")
    assert combo.type_id == CORE_COMBO
    assert combo.resolve() == "euler"
    assert isinstance(combo.resolve(), str)
    assert registry.inline_of(combo) == "euler"

    spec = registry.spec(CORE_COMBO)
    assert spec.decode(spec.encode("euler")) == "euler"
    assert combo.fingerprint != string.fingerprint


def test_core_combo_keeps_canonical_codec_and_fingerprint() -> None:
    registry = make_registry()
    combo = registry.wrap(CORE_COMBO, "euler")
    spec = registry.spec(CORE_COMBO)

    assert spec.encode("euler") == b'json:"euler"'
    assert combo.fingerprint == "5cc02dd05421d92f092e0c685a9ae179ce868b7c"
    assert registry.wrap(CORE_COMBO, combo.resolve()).fingerprint == combo.fingerprint


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


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        [],
        ["euler"],
        ComboStringSubclass("euler"),
        float("nan"),
        float("inf"),
        -(2**53),
        2**53,
    ],
)
def test_core_combo_rejects_non_scalar_values(invalid: object) -> None:
    registry = make_registry()
    with pytest.raises((TypeError, ValueError), match="COMBO expects"):
        registry.wrap(CORE_COMBO, invalid)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"json:",
        b"json:null",
        b"json:NaN",
        b"json:Infinity",
        b"json:-Infinity",
        b"json:{}",
        b"json:[]",
        b'json:["euler"]',
        b'json: "euler"',
        b'pickle:"euler"',
        b"json:9007199254740992",
        b"json:-9007199254740992",
        b"json:9007199254740993",
        b"json:-9007199254740993",
    ],
)
def test_core_combo_rejects_noncanonical_or_non_scalar_encoded_values(
    data: bytes,
) -> None:
    validator = make_registry().spec(CORE_COMBO).validate_encoded
    assert validator is not None
    with pytest.raises((TypeError, ValueError)):
        validator(data, {})


def test_default_codec_json_roundtrip() -> None:
    data = default_encode({"b": 1, "a": [1, 2.5, "x", None, True]})
    assert data.startswith(b"json:")
    assert default_decode(data) == {"b": 1, "a": [1, 2.5, "x", None, True]}


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


def test_bare_name_registration_works() -> None:
    """Sane defaults: registering with nothing but a name is fully functional."""
    registry = TypeRegistry()
    registry.register("mypack.mystery")
    value = registry.wrap("mypack.mystery", {"weird": [1, 2, 3]})
    assert value.resolve() == {"weird": [1, 2, 3]}
    assert value.fingerprint
    assert not registry.spec("mypack.mystery").declared_codec


def test_custom_fingerprint_and_meta_used() -> None:
    registry = TypeRegistry()
    registry.register(
        "mypack.tagged",
        fingerprint=lambda obj: f"tagged:{obj}",
        meta=lambda obj: {"length": len(str(obj))},
    )
    value = registry.wrap("mypack.tagged", "hello")
    assert value.fingerprint == "tagged:hello"
    assert value.meta.get("length") == 5


def test_encoded_validator_registration_is_optional_and_preserved() -> None:
    registry = TypeRegistry()
    plain = registry.register("mypack.plain")
    calls: list[tuple[bytes, Mapping[str, object]]] = []

    def validate(data: bytes, metadata: Mapping[str, object]) -> None:
        calls.append((data, metadata))

    validated = registry.register("mypack.validated", validate_encoded=validate)
    assert plain.validate_encoded is None
    assert validated.validate_encoded is validate
    assert calls == []


def test_buffer_encoding_requires_and_preserves_a_declared_codec() -> None:
    registry = TypeRegistry()

    def prepare(_obj: object) -> BufferEncoding:
        return BufferEncoding(0, lambda _buffer: 0)

    with pytest.raises(ValueError, match="requires a declared codec"):
        registry.register("mypack.invalid", prepare_buffer_encoding=prepare)
    spec = registry.register(
        "mypack.buffered",
        encode=lambda obj: str(obj).encode(),
        decode=lambda data: data.decode(),
        prepare_buffer_encoding=prepare,
    )
    assert spec.prepare_buffer_encoding is prepare


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


def test_absence_does_not_inline() -> None:
    """Absence has its own channel (typed core.absent envelopes and
    node_skipped); inlining None would collapse "not inline" and "absent"
    into one ambiguous signal."""
    registry = make_registry()
    absent = make_absent_value(origin="n1")
    assert registry.inline_of(absent) is None


def test_undeclared_types_never_inline() -> None:
    registry = make_registry()
    registry.register("mypack.opaque")
    assert registry.inline_of(registry.wrap("mypack.opaque", 42)) is None


def test_inline_policy_rejects_non_scalars() -> None:
    """A pack inline fn cannot smuggle rich objects: only JSON-native
    scalars survive the central policy."""
    registry = make_registry()
    registry.register("mypack.leaky", inline=lambda obj: obj)
    assert registry.inline_of(registry.wrap("mypack.leaky", {"a": 1})) is None
    assert registry.inline_of(registry.wrap("mypack.leaky", [1, 2])) is None
    assert registry.inline_of(registry.wrap("mypack.leaky", None)) is None


def test_buggy_inline_fn_degrades_to_not_inline() -> None:
    """Inline rides event/descriptor serialization paths: a pack bug must
    degrade to omission, never fail the run."""
    registry = make_registry()

    def boom(obj: object) -> object:
        raise RuntimeError("pack bug")

    registry.register("mypack.buggy", inline=boom)
    assert registry.inline_of(registry.wrap("mypack.buggy", 1)) is None


def test_list_values_do_not_inline() -> None:
    """List type ids are constructor syntax, never registered atoms, so the
    top-level list never inlines - elements do, via descriptor recursion."""
    registry = make_registry()
    value = registry.wrap("list<INT>", [1, 2, 3])
    assert registry.inline_of(value) is None


# -- renditions (DESIGN 3.5) ---------------------------------------------------


def test_rendition_registration_and_render() -> None:
    registry = make_registry()
    registry.register("mypack.doc")
    spec = registry.register_rendition(
        "mypack.doc", "txt", mime="text/plain", render=lambda obj: str(obj).encode()
    )
    assert spec.default  # first registration becomes the default
    assert spec.cache_key("fingerprint") == "fingerprint/txt"
    rendition = registry.render(registry.wrap("mypack.doc", "hello"))
    assert (rendition.kind, rendition.mime, rendition.data) == (
        "txt",
        "text/plain",
        b"hello",
    )
    # Explicit kind resolves the same spec.
    assert registry.render(registry.wrap("mypack.doc", "x"), "txt").data == b"x"


def test_rendition_mime_can_resolve_from_value_metadata() -> None:
    registry = make_registry()
    registry.register("mypack.media", meta=lambda obj: {"format": str(obj)})
    spec = registry.register_rendition(
        "mypack.media",
        "original",
        mime=lambda metadata: f"media/{metadata['format']}",
        render=lambda obj: str(obj).encode(),
    )
    value = registry.wrap("mypack.media", "example")
    assert spec.mime_for(value.meta.entries) == "media/example"
    rendition = registry.render(value)
    assert (rendition.mime, rendition.data) == ("media/example", b"example")


def test_rendition_version_rotates_cache_identity() -> None:
    registry = make_registry()
    registry.register("mypack.doc")
    spec = registry.register_rendition(
        "mypack.doc",
        "txt",
        mime="text/plain",
        render=lambda obj: str(obj).encode(),
        version="container-v2",
    )
    assert spec.selector == "txt/container-v2"
    assert spec.cache_key("fingerprint") == "fingerprint/txt/container-v2"


def test_parameterized_rendition_normalizes_render_and_cache_identity() -> None:
    registry = make_registry()
    registry.register("mypack.audio", meta=lambda _obj: {"batch": 2})

    def normalize(
        parameters: Mapping[str, str], metadata: Mapping[str, object]
    ) -> Mapping[str, str]:
        assert metadata["batch"] == 2
        return {"batch": str(int(parameters["batch"])), "window": parameters["window"]}

    def render(obj: object, parameters: Mapping[str, str]) -> bytes:
        return f"{obj}:{parameters['batch']}:{parameters['window']}".encode()

    spec = registry.register_rendition(
        "mypack.audio",
        "window",
        mime="audio/wav",
        render=render,
        version="pcm-v1",
        parameters=("batch", "window"),
        defaults={"batch": "0"},
        limits={"durationSeconds": 30},
        normalize=normalize,
    )
    value = registry.wrap("mypack.audio", "audio")
    normalized = spec.normalize_parameters({"batch": "01", "window": "2,3"}, value.meta.entries)
    assert normalized == {"batch": "1", "window": "2,3"}
    assert spec.cache_key("fingerprint", normalized) == (
        "fingerprint/window/pcm-v1?batch=1&window=2%2C3"
    )
    assert registry.render(value, "window", normalized).data == b"audio:1:2,3"


@pytest.mark.parametrize(
    ("first_kind", "first_version", "second_kind", "second_version"),
    [
        ("png", "stored-v1", "png/stored-v1", None),
        ("png/stored-v1", None, "png", "stored-v1"),
        ("png", "stored-v1", "png/stored-v1", "nested-v2"),
        ("png/stored-v1", "nested-v2", "png", "stored-v1"),
    ],
)
def test_rendition_registration_rejects_kind_selector_collisions(
    first_kind: str,
    first_version: str | None,
    second_kind: str,
    second_version: str | None,
) -> None:
    registry = make_registry()
    registry.register("mypack.doc")
    registry.register_rendition(
        "mypack.doc",
        first_kind,
        mime="image/png",
        render=lambda obj: b"png",
        version=first_version,
    )
    with pytest.raises(ValueError, match=r"selector already registered: png/stored-v1$"):
        registry.register_rendition(
            "mypack.doc",
            second_kind,
            mime="image/png",
            render=lambda obj: b"png",
            version=second_version,
        )


def test_rendition_default_negotiation() -> None:
    registry = make_registry()
    registry.register("mypack.doc")
    registry.register_rendition("mypack.doc", "txt", mime="text/plain", render=lambda obj: b"t")
    registry.register_rendition("mypack.doc", "html", mime="text/html", render=lambda obj: b"h")
    kinds = {(s.kind, s.default) for s in registry.renditions_of("mypack.doc")}
    assert kinds == {("txt", True), ("html", False)}
    assert registry.render(registry.wrap("mypack.doc", "x")).kind == "txt"
    assert registry.render(registry.wrap("mypack.doc", "x"), "html").data == b"h"


def test_local_rendition_replaces_matching_worker_relay() -> None:
    registry = make_registry()
    registry.register("mypack.doc")

    async def unused_mime(_metadata: Mapping[str, object]) -> str:
        raise AssertionError("relay callback must be replaced")

    async def unused_resolve(
        _metadata: Mapping[str, object], _parameters: Mapping[str, str]
    ) -> tuple[str, Mapping[str, str]]:
        raise AssertionError("relay callback must be replaced")

    async def unused_render(_value: Value, _parameters: Mapping[str, str]) -> Rendition:
        raise AssertionError("relay callback must be replaced")

    registry.register_relayed_rendition(
        "mypack.doc",
        "txt",
        mime="text/plain",
        default=True,
        version=None,
        parameters=(),
        defaults=None,
        limits=None,
        owner="worker",
        resolve_mime=unused_mime,
        resolve=unused_resolve,
        render=unused_render,
    )

    local = registry.register_rendition(
        "mypack.doc", "txt", mime="text/plain", render=lambda _obj: b"local"
    )
    assert local.default
    assert local.relay_owner is None
    assert registry.render(registry.wrap("mypack.doc", "x")).data == b"local"


def test_rendition_registration_rejections() -> None:
    registry = make_registry()
    with pytest.raises(KeyError, match="unregistered"):
        registry.register_rendition("mypack.nope", "txt", mime="text/plain", render=lambda obj: b"")
    registry.register("mypack.doc")
    registry.register_rendition("mypack.doc", "txt", mime="text/plain", render=lambda obj: b"")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_rendition(
            "mypack.doc", "txt", mime="text/x-other", render=lambda obj: b""
        )
    with pytest.raises(ValueError, match="default rendition already exists"):
        registry.register_rendition(
            "mypack.doc", "html", mime="text/html", render=lambda obj: b"", default=True
        )
    with pytest.raises(ValueError, match="non-empty"):
        registry.register_rendition("mypack.doc", "", mime="text/plain", render=lambda obj: b"")
    with pytest.raises(ValueError, match="version"):
        registry.register_rendition(
            "mypack.doc", "html", mime="text/html", render=lambda obj: b"", version=""
        )


def test_render_unknown_kind_raises() -> None:
    registry = make_registry()
    registry.register("mypack.doc")
    with pytest.raises(KeyError, match="no rendition"):
        registry.render(registry.wrap("mypack.doc", "x"))
    registry.register_rendition("mypack.doc", "txt", mime="text/plain", render=lambda obj: b"")
    with pytest.raises(KeyError, match="no rendition 'pdf'"):
        registry.render(registry.wrap("mypack.doc", "x"), "pdf")
