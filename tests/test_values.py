import copy
import gc
import pickle
import threading
from collections.abc import Mapping
from dataclasses import asdict, fields, replace

import pytest
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    INLINE_STRING_CAP,
    BufferEncoding,
    EncodedPayload,
    TypeRegistry,
    default_decode,
    default_encode,
    make_absent_value,
    register_core_types,
)


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


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
    assert combo.fingerprint == "adfaa12b6da4ac3f28ffc6a3ec6712f845573052"
    assert registry.wrap(CORE_COMBO, combo.resolve()).fingerprint == combo.fingerprint


class ComboStringSubclass(str):
    pass


@pytest.mark.parametrize(
    "invalid", [True, 1, 1.5, None, {}, [], ["euler"], ComboStringSubclass("euler")]
)
def test_core_combo_rejects_non_string_values(invalid: object) -> None:
    registry = make_registry()
    with pytest.raises(TypeError, match="core.combo expects a string"):
        registry.wrap(CORE_COMBO, invalid)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"json:",
        b"json:null",
        b"json:true",
        b"json:1",
        b"json:1.5",
        b"json:{}",
        b"json:[]",
        b'json:["euler"]',
        b'json: "euler"',
        b'pickle:"euler"',
    ],
)
def test_core_combo_rejects_noncanonical_or_non_string_encoded_values(
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
    over_cap = registry.wrap(CORE_STRING, "x" * (INLINE_STRING_CAP + 1))
    assert registry.inline_of(at_cap) == "x" * INLINE_STRING_CAP
    assert registry.inline_of(over_cap) is None


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
    value = registry.wrap("list<core.int>", [1, 2, 3])
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
