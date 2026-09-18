"""Stage-2 proving tests: strict safetensors header inspection and
detection-as-data family signatures (docs/native-inference-plan.md
stage 2). Signature key layouts and constants mirror the ComfyUI
reference @ 947c2749 (comfy/model_detection.py, comfy/supported_models.py)."""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    UINT8,
    DimField,
    KeySignature,
    MalformedSafetensors,
    RankIs,
    ShapeIn,
    ShapeIs,
    TensorGeometry,
    WeightEntry,
    WeightSource,
    builtin_families,
    builtin_family_registry,
    load_safetensors_header,
)

# --- helpers -----------------------------------------------------------


def write_safetensors(
    path: Path,
    tensors: Mapping[str, tuple[str, tuple[int, ...]]],
    metadata: Mapping[str, str] | None = None,
    *,
    mangle: str | None = None,
) -> Path:
    """Write a syntactically real safetensors file with zeroed payload."""
    bits = {"F32": 32, "F16": 16, "BF16": 16, "U8": 8, "I64": 64}
    header: dict[str, object] = {}
    cursor = 0
    for key, (dtype, shape) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * bits[dtype] // 8
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    raw = json.dumps(header).encode()
    if mangle == "truncate_header":
        raw = raw[:-4]
    body = struct.pack("<Q", len(raw)) + raw + b"\0" * cursor
    if mangle == "truncate_payload":
        body = body[:-1]
    path.write_bytes(body)
    return path


class DictSource:
    """In-memory WeightSource over {key: shape}; dtype is immaterial to
    detection."""

    def __init__(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self._shapes = dict(shapes)

    def keys(self) -> Sequence[str]:
        return tuple(self._shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self._shapes[key], FLOAT16)
        return WeightEntry(key=key, geometry=geometry, offset=0, nbytes=geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


_: type[WeightSource] = DictSource  # protocol conformance, checked statically


def prefixed(prefix: str, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    return {prefix + key: shape for key, shape in shapes.items()}


# Minimal key layouts carrying exactly the evidence each signature reads.
SD15_SHAPES: dict[str, tuple[int, ...]] = {
    "input_blocks.0.0.weight": (320, 4, 3, 3),
    "input_blocks.1.1.proj_in.weight": (320, 320, 1, 1),
    "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, 768),
}
SDXL_SHAPES: dict[str, tuple[int, ...]] = {
    "input_blocks.0.0.weight": (320, 4, 3, 3),
    "label_emb.0.0.weight": (1280, 2816),
    "input_blocks.4.1.proj_in.weight": (640, 640),
    "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": (640, 2048),
    # depth profile [0, 0, 2, 2, 10, 10]: block 4 reaches index 1,
    # block 7 reaches index 9
    "input_blocks.4.1.transformer_blocks.1.attn2.to_k.weight": (640, 2048),
    "input_blocks.7.1.transformer_blocks.9.attn2.to_k.weight": (1280, 2048),
}
REFINER_SHAPES: dict[str, tuple[int, ...]] = {
    "input_blocks.0.0.weight": (384, 4, 3, 3),
    "label_emb.0.0.weight": (1536, 2560),
    "input_blocks.4.1.proj_in.weight": (768, 768),
    "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": (768, 1280),
    # depth profile [0, 0, 4, 4, 4, 4, 0, 0]: block 4 reaches index 3
    "input_blocks.4.1.transformer_blocks.3.attn2.to_k.weight": (768, 1280),
}
FLUX_SHAPES: dict[str, tuple[int, ...]] = {
    "img_in.weight": (3072, 64),
    "txt_in.weight": (3072, 4096),
    "vector_in.in_layer.weight": (3072, 768),
    "double_blocks.0.img_attn.norm.key_norm.scale": (128,),
    "guidance_in.in_layer.weight": (3072, 256),
}
SCHNELL_SHAPES = {k: v for k, v in FLUX_SHAPES.items() if not k.startswith("guidance_in.")}


# --- safetensors: the happy path ---------------------------------------


def test_safetensors_header_parses(tmp_path: Path) -> None:
    path = write_safetensors(
        tmp_path / "ok.safetensors",
        {"a.weight": ("F32", (2, 3)), "b.bias": ("F16", (4,))},
        {"format": "pt"},
    )
    source = load_safetensors_header(path)
    assert source.keys() == ("a.weight", "b.bias")
    a = source.entry("a.weight")
    assert a.geometry == TensorGeometry((2, 3), FLOAT32)
    assert a.nbytes == 24
    # offsets are absolute: 8-byte prefix + header + payload-relative begin
    header_len = struct.unpack("<Q", path.read_bytes()[:8])[0]
    assert a.offset == 8 + header_len
    assert source.entry("b.bias").offset == a.offset + 24
    assert source.metadata() == {"format": "pt"}


@pytest.mark.parametrize(
    ("dtype", "packed", "expected"),
    (
        ("F16", struct.pack("<e", 0.125), 0.125),
        ("BF16", struct.pack("<H", 0x3E00), 0.125),
        ("F32", struct.pack("<f", 42.5), 42.5),
    ),
)
def test_safetensors_reads_explicit_float_configuration_scalar(
    tmp_path: Path, dtype: str, packed: bytes, expected: float
) -> None:
    path = write_safetensors(tmp_path / "scalar.safetensors", {"value": (dtype, ())})
    source = load_safetensors_header(path)
    with path.open("r+b") as handle:
        handle.seek(source.entry("value").offset)
        handle.write(packed)
    assert source.read_float_scalar("value") == expected


def test_safetensors_float_configuration_scalar_refuses_nonfinite(
    tmp_path: Path,
) -> None:
    path = write_safetensors(tmp_path / "scalar.safetensors", {"value": ("F32", ())})
    source = load_safetensors_header(path)
    with path.open("r+b") as handle:
        handle.seek(source.entry("value").offset)
        handle.write(struct.pack("<f", float("nan")))
    with pytest.raises(ValueError, match="finite"):
        source.read_float_scalar("value")


def test_safetensors_reads_uint8_configuration_exact_bytes(tmp_path: Path) -> None:
    payload = b'{"format":"nvfp4","full_precision_matrix_mult":false}'
    path = write_safetensors(
        tmp_path / "config.safetensors", {"x.comfy_quant": ("U8", (len(payload),))}
    )
    source = load_safetensors_header(path)
    with path.open("r+b") as handle:
        handle.seek(source.entry("x.comfy_quant").offset)
        handle.write(payload)
    assert source.read_uint8_configuration("x.comfy_quant") == payload


def test_bound_configuration_handle_never_reopens_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_safetensors(
        tmp_path / "configuration.safetensors", {"scalar": ("F32", ()), "config": ("U8", (3,))}
    )
    source = load_safetensors_header(path)
    with path.open("r+b") as handle:
        handle.seek(source.entry("scalar").offset)
        handle.write(struct.pack("<f", 42.5) + b"abc")
    with path.open("rb") as handle:
        bound = replace(source, configuration_file=handle)

        def refuse_open(*args: object, **kwargs: object) -> None:
            raise AssertionError("configuration must use the bound descriptor")

        monkeypatch.setattr(Path, "open", refuse_open)
        assert bound.read_float_scalar("scalar") == 42.5
        assert bound.read_uint8_configuration("config") == b"abc"
        with pytest.raises(ValueError, match="2-byte"):
            bound.read_uint8_configuration("config", limit=2)
        assert not handle.closed


def test_safetensors_configuration_reads_use_the_pinned_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors = {"scale": ("F32", ()), "config": ("U8", (4,))}
    path = write_safetensors(tmp_path / "source.safetensors", tensors)
    source = load_safetensors_header(path)
    with path.open("r+b") as handle:
        handle.seek(source.entry("scale").offset)
        handle.write(struct.pack("<f", 0.125))
        handle.seek(source.entry("config").offset)
        handle.write(b"true")
    with path.open("rb") as verified:
        pinned = replace(source, configuration_file=verified)
        replacement = write_safetensors(tmp_path / "replacement.safetensors", tensors)
        original_open = Path.open
        monkeypatch.setattr(Path, "open", lambda target, *args: original_open(replacement, *args))
        assert source.read_float_scalar("scale") == 0.0
        assert source.read_uint8_configuration("config") == b"\0" * 4
        assert pinned.read_float_scalar("scale") == 0.125
        assert pinned.read_uint8_configuration("config") == b"true"
        assert not verified.closed


@pytest.mark.parametrize(
    ("dtype", "shape", "message"),
    (("F16", (4,), "rank-1 uint8"), ("U8", (), "rank-1 uint8"), ("U8", (65_537,), "65536-byte")),
)
def test_safetensors_uint8_configuration_refuses_dtype_rank_and_cap(
    tmp_path: Path, dtype: str, shape: tuple[int, ...], message: str
) -> None:
    source = load_safetensors_header(
        write_safetensors(tmp_path / "bad.safetensors", {"config": (dtype, shape)})
    )
    with pytest.raises(ValueError, match=message):
        source.read_uint8_configuration("config")


def test_safetensors_uint8_configuration_honors_explicit_limit(tmp_path: Path) -> None:
    """A caller-declared byte cap admits payloads above the default cap
    (the Flux2 dev text encoder embeds a ~19 MB tekken_model) while
    still refusing anything above the declared limit."""
    payload = bytes(range(256)) * 512
    assert len(payload) > 65_536
    path = write_safetensors(
        tmp_path / "big.safetensors", {"tekken_model": ("U8", (len(payload),))}
    )
    source = load_safetensors_header(path)
    with path.open("r+b") as handle:
        handle.seek(source.entry("tekken_model").offset)
        handle.write(payload)
    assert source.read_uint8_configuration("tekken_model", limit=len(payload)) == payload
    with pytest.raises(ValueError, match=f"{len(payload) - 1}-byte"):
        source.read_uint8_configuration("tekken_model", limit=len(payload) - 1)


def test_safetensors_dtype_mapping(tmp_path: Path) -> None:
    path = write_safetensors(tmp_path / "u8.safetensors", {"q": ("U8", (5,))})
    assert load_safetensors_header(path).entry("q").geometry.dtype == UINT8


def test_safetensors_full_reference_dtype_table(tmp_path: Path) -> None:
    """Every code in the reference Dtype enum (huggingface/safetensors)
    decodes; shapes chosen so sub-byte totals are byte-aligned."""
    codes_bits = {
        "BOOL": 8, "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, "U8": 8, "I8": 8,
        "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8, "F8_E4M3FNUZ": 8,
        "F8_E5M2FNUZ": 8, "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
        "I32": 32, "U32": 32, "F32": 32, "C64": 64, "F64": 64,
        "I64": 64, "U64": 64,
    }  # fmt: skip
    header: dict[str, object] = {}
    cursor = 0
    for code, bits in codes_bits.items():
        nbytes = 8 * bits // 8  # 8 elements is byte-aligned for 4/6-bit
        header[f"t.{code}"] = {
            "dtype": code,
            "shape": [8],
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
    raw = json.dumps(header).encode()
    path = tmp_path / "all-dtypes.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * cursor)
    source = load_safetensors_header(path)
    for code, bits in codes_bits.items():
        dtype = source.entry(f"t.{code}").geometry.dtype
        assert dtype.bits == bits, code


def test_safetensors_subbyte_alignment(tmp_path: Path) -> None:
    """Sub-byte totals must fill whole bytes (the reference's
    MisalignedSlice): two F4 elements pack into one byte; three do not."""
    ok = {"w": {"dtype": "F4", "shape": [2], "data_offsets": [0, 1]}}
    raw = json.dumps(ok).encode()
    path = tmp_path / "f4-ok.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0")
    assert load_safetensors_header(path).entry("w").nbytes == 1

    bad = {"w": {"dtype": "F4", "shape": [3], "data_offsets": [0, 2]}}
    raw = json.dumps(bad).encode()
    path = tmp_path / "f4-bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0")
    _expect_malformed(path, "do not fill whole bytes")


def test_safetensors_ignores_unknown_entry_fields(tmp_path: Path) -> None:
    """Unknown descriptor fields are ignored, matching the reference
    deserializer (serde default, no deny_unknown_fields)."""
    header = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], "x": 1}}
    raw = json.dumps(header).encode()
    path = tmp_path / "extra.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    assert load_safetensors_header(path).entry("w").nbytes == 4


def test_safetensors_never_reads_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inspection touches only the 8-byte prefix and the header."""
    path = write_safetensors(
        tmp_path / "spy.safetensors", {"w": ("F32", (1024,))}, {"format": "pt"}
    )
    header_len = struct.unpack("<Q", path.read_bytes()[:8])[0]
    limit = 8 + header_len
    real_open = Path.open

    class SpyHandle:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def __enter__(self) -> SpyHandle:
            return self

        def __exit__(self, *exc: object) -> None:
            self._handle.__exit__(*exc)  # type: ignore[attr-defined]

        def read(self, n: int = -1) -> bytes:
            data: bytes = self._handle.read(n)  # type: ignore[attr-defined]
            position: int = self._handle.tell()  # type: ignore[attr-defined]
            assert position <= limit, "read crossed into the payload"
            return data

        def fileno(self) -> int:
            return self._handle.fileno()  # type: ignore[attr-defined,no-any-return]

        def seek(self, offset: int) -> int:
            return self._handle.seek(offset)  # type: ignore[attr-defined,no-any-return]

    def spying_open(self: Path, *args: object, **kwargs: object) -> SpyHandle:
        return SpyHandle(real_open(self, *args, **kwargs))  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", spying_open)
    source = load_safetensors_header(path)
    assert source.entry("w").nbytes == 4096


def test_safetensors_zero_tensors_legal(tmp_path: Path) -> None:
    path = write_safetensors(tmp_path / "meta.safetensors", {}, {"note": "header-only"})
    source = load_safetensors_header(path)
    assert source.keys() == ()
    assert source.metadata() == {"note": "header-only"}


def test_safetensors_source_mappings_immutable(tmp_path: Path) -> None:
    path = write_safetensors(tmp_path / "ro.safetensors", {"w": ("F32", (1,))})
    source = load_safetensors_header(path)
    with pytest.raises(TypeError):
        source.entries["x"] = source.entries["w"]  # type: ignore[index]


# --- safetensors: strict malformation diagnostics ----------------------


def _expect_malformed(path: Path, fragment: str) -> None:
    with pytest.raises(MalformedSafetensors, match=fragment):
        load_safetensors_header(path)


def test_safetensors_rejects_tiny_file(tmp_path: Path) -> None:
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(b"\x01\x02")
    _expect_malformed(path, "8-byte header length is missing")


def test_safetensors_rejects_zero_and_overrun_header(tmp_path: Path) -> None:
    zero = tmp_path / "zero.safetensors"
    zero.write_bytes(struct.pack("<Q", 0) + b"{}")
    _expect_malformed(zero, "header length is 0")
    overrun = tmp_path / "overrun.safetensors"
    overrun.write_bytes(struct.pack("<Q", 999) + b"{}")
    _expect_malformed(overrun, "overruns")
    huge = tmp_path / "huge.safetensors"
    huge.write_bytes(struct.pack("<Q", 200_000_000) + b"{}")
    _expect_malformed(huge, "cap")


def test_safetensors_rejects_bad_json(tmp_path: Path) -> None:
    path = write_safetensors(
        tmp_path / "cut.safetensors", {"w": ("F32", (2,))}, mangle="truncate_header"
    )
    _expect_malformed(path, "not valid JSON")


def test_safetensors_rejects_duplicate_keys(tmp_path: Path) -> None:
    entry = '{"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}'
    raw = f'{{"w": {entry}, "w": {entry}}}'.encode()
    path = tmp_path / "dup.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    _expect_malformed(path, "duplicate key")


def test_safetensors_rejects_non_object_header(tmp_path: Path) -> None:
    raw = b"[1, 2]"
    path = tmp_path / "arr.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    _expect_malformed(path, "JSON object")


def test_safetensors_rejects_bad_entries(tmp_path: Path) -> None:
    cases: dict[str, tuple[object, str]] = {
        "dtype": ({"dtype": 7, "shape": [1], "data_offsets": [0, 4]}, "dtype must be"),
        "unknown_dtype": (
            {"dtype": "F99", "shape": [1], "data_offsets": [0, 4]},
            "unknown dtype code",
        ),
        "shape": ({"dtype": "F32", "shape": [1, -2], "data_offsets": [0, 4]}, "non-negative"),
        "bool_dim": ({"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}, "non-negative"),
        "offsets": ({"dtype": "F32", "shape": [1], "data_offsets": [0]}, "pair of integers"),
        "not_object": (3, "must be a JSON object"),
    }
    for name, (entry, fragment) in cases.items():
        raw = json.dumps({"w": entry}).encode()
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
        _expect_malformed(path, fragment)


def test_safetensors_rejects_range_violations(tmp_path: Path) -> None:
    def build(offsets_by_key: dict[str, list[int]], payload: int) -> Path:
        header = {
            key: {"dtype": "F32", "shape": [(o[1] - o[0]) // 4], "data_offsets": o}
            for key, o in offsets_by_key.items()
        }
        raw = json.dumps(header).encode()
        path = tmp_path / f"r{len(list(tmp_path.iterdir()))}.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * payload)
        return path

    _expect_malformed(build({"w": [0, 8]}, 4), "outside")  # end past payload
    _expect_malformed(build({"a": [0, 4], "b": [0, 4]}, 8), "overlaps")
    _expect_malformed(build({"a": [0, 4], "b": [8, 12]}, 12), "gap")
    _expect_malformed(build({"a": [0, 4]}, 8), "cover")  # trailing slack


def test_safetensors_rejects_wrong_nbytes(tmp_path: Path) -> None:
    header = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    path = tmp_path / "nb.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    _expect_malformed(path, "geometry needs 8")


def test_safetensors_rejects_non_string_metadata(tmp_path: Path) -> None:
    header = {"__metadata__": {"steps": 30}}
    raw = json.dumps(header).encode()
    path = tmp_path / "meta.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    _expect_malformed(path, "must be a string")


def test_safetensors_missing_file_is_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_safetensors_header(tmp_path / "absent.safetensors")


# --- signatures ---------------------------------------------------------


def test_signature_validation() -> None:
    with pytest.raises(ValueError, match="positive requirement"):
        KeySignature(family_id="x.y")
    with pytest.raises(ValueError, match="prefix"):
        KeySignature(family_id="x.y", required=("k",), prefixes=())
    with pytest.raises(ValueError, match="groups"):
        KeySignature(family_id="x.y", required_any=((),))
    with pytest.raises(ValueError, match="axis"):
        ShapeIs("k", -1, 3)
    with pytest.raises(ValueError, match="non-empty"):
        ShapeIn("k", 0, ())
    with pytest.raises(ValueError, match="rank"):
        RankIs("k", -1)
    with pytest.raises(ValueError, match="empty"):
        DimField("", "k", 0)


def test_signature_prefix_resolution_and_evidence() -> None:
    sig = KeySignature(
        family_id="x.y",
        prefixes=("model.diffusion_model.", ""),
        required=("conv.weight",),
        constraints=(ShapeIs("conv.weight", 0, 8),),
        fields=(DimField("channels", "conv.weight", 0),),
    )
    bare = DictSource({"conv.weight": (8, 3)})
    combined = DictSource(prefixed("model.diffusion_model.", {"conv.weight": (8, 3)}))
    for source, prefix in ((bare, ""), (combined, "model.diffusion_model.")):
        evidence = sig.detect(source)
        assert evidence is not None
        assert evidence.fields["key_prefix"] == prefix
        assert evidence.fields["channels"] == 8
        assert evidence.matched_keys == (prefix + "conv.weight",)


def test_shape_in_accepts_only_declared_dimensions() -> None:
    signature = KeySignature(
        family_id="x.y",
        required=("conv.weight",),
        constraints=(ShapeIn("conv.weight", 1, (4, 9)),),
    )
    assert signature.detect(DictSource({"conv.weight": (8, 4)})) is not None
    assert signature.detect(DictSource({"conv.weight": (8, 9)})) is not None
    assert signature.detect(DictSource({"conv.weight": (8, 8)})) is None


def test_signature_guards_every_dereference() -> None:
    sig = KeySignature(
        family_id="x.y",
        required=("a",),
        constraints=(ShapeIs("b", 1, 4), RankIs("c", 2)),
        fields=(DimField("d0", "d", 5),),
    )
    assert sig.detect(DictSource({"b": (1, 4), "c": (1, 1), "d": (9,) * 6})) is None  # a missing
    assert sig.detect(DictSource({"a": (1,), "c": (1, 1), "d": (9,) * 6})) is None  # b missing
    assert (
        sig.detect(DictSource({"a": (1,), "b": (4,), "c": (1, 1), "d": (9,) * 6})) is None
    )  # b axis out of range
    assert (
        sig.detect(DictSource({"a": (1,), "b": (1, 4), "c": (1,), "d": (9,) * 6})) is None
    )  # c rank mismatch
    assert (
        sig.detect(DictSource({"a": (1,), "b": (1, 4), "c": (1, 1), "d": (9,)})) is None
    )  # d axis out of range
    hit = sig.detect(DictSource({"a": (1,), "b": (1, 4), "c": (1, 1), "d": (9,) * 6}))
    assert hit is not None and hit.fields["d0"] == 9


def test_signature_absent_and_required_any() -> None:
    sig = KeySignature(
        family_id="x.y",
        required=("a",),
        required_any=(("norm.scale", "norm.weight"),),
        absent=("forbidden",),
    )
    assert sig.detect(DictSource({"a": (1,), "norm.weight": (1,)})) is not None
    assert sig.detect(DictSource({"a": (1,), "norm.scale": (1,)})) is not None
    assert sig.detect(DictSource({"a": (1,)})) is None
    assert sig.detect(DictSource({"a": (1,), "norm.scale": (1,), "forbidden": (1,)})) is None


def test_signature_absent_toplevel_ignores_prefix() -> None:
    """Top-level markers (SDXL's v_pred/edm_*) exclude a match even
    when the signature resolves under a prefix."""
    sig = KeySignature(
        family_id="x.y",
        prefixes=("model.diffusion_model.", ""),
        required=("conv.weight",),
        absent_toplevel=("v_pred",),
    )
    combined = prefixed("model.diffusion_model.", {"conv.weight": (8, 3)})
    assert sig.detect(DictSource(combined)) is not None
    assert sig.detect(DictSource({**combined, "v_pred": (1,)})) is None
    assert sig.detect(DictSource({"conv.weight": (8, 3), "v_pred": (1,)})) is None


def test_signature_snapshots_caller_sequences() -> None:
    """A caller-held list cannot mutate a validated signature into a
    match-anything detector."""
    required = ["a"]
    groups = [["norm.scale", "norm.weight"]]
    sig = KeySignature(
        family_id="x.y",
        required=required,  # type: ignore[arg-type]  # runtime hardening
        required_any=groups,  # type: ignore[arg-type]
    )
    required.clear()
    groups[0].clear()
    assert sig.required == ("a",)
    assert sig.required_any == (("norm.scale", "norm.weight"),)
    assert sig.detect(DictSource({"unrelated": (1,)})) is None


# --- the grounded catalog ----------------------------------------------


@pytest.mark.parametrize(
    ("family_id", "shapes"),
    [
        ("dinkster.sd15", SD15_SHAPES),
        ("dinkster.sdxl", SDXL_SHAPES),
        ("dinkster.sdxl_refiner", REFINER_SHAPES),
        ("dinkster.flux_dev", FLUX_SHAPES),
        ("dinkster.flux_schnell", SCHNELL_SHAPES),
    ],
)
def test_catalog_detects_each_family(family_id: str, shapes: dict[str, tuple[int, ...]]) -> None:
    registry = builtin_family_registry()
    for prefix in ("", "model.diffusion_model."):
        result = registry.detect(DictSource(prefixed(prefix, shapes)))
        assert result.ambiguous == ()
        assert result.best is not None, f"{family_id} not detected at prefix {prefix!r}"
        assert result.best.family_id == family_id
        assert [e.family_id for e in result.candidates] == [family_id]


def test_catalog_families_are_mutually_exclusive() -> None:
    registry = builtin_family_registry()
    unknown = registry.detect(DictSource({"some.random.weight": (3, 3)}))
    assert unknown.best is None and unknown.candidates == ()
    # SD2-style context dim (1024) must not be claimed by SD15
    sd2 = dict(SD15_SHAPES)
    sd2["input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"] = (320, 1024)
    assert registry.detect(DictSource(sd2)).best is None
    # chroma-style flux (distilled guidance, no vector_in/guidance_in)
    chroma = {
        "img_in.weight": (3072, 64),
        "txt_in.weight": (3072, 4096),
        "double_blocks.0.img_attn.norm.key_norm.scale": (128,),
        "distilled_guidance_layer.norms.0.scale": (64,),
    }
    assert registry.detect(DictSource(chroma)).best is None


@pytest.mark.parametrize(
    ("name", "shapes"),
    [
        # instruct-pix2pix widens the first conv to 8 in-channels
        # (comfy/supported_models.py SD15_instructpix2pix @ 947c2749)
        ("sd15-pix2pix", {**SD15_SHAPES, "input_blocks.0.0.weight": (320, 8, 3, 3)}),
        ("sdxl-pix2pix", {**SDXL_SHAPES, "input_blocks.0.0.weight": (320, 8, 3, 3)}),
        # SSD1B: transformer_depth [0, 0, 2, 2, 4, 4] - block 7 stops
        # at index 3, so the depth-10 key is missing
        (
            "ssd1b",
            {
                k: v
                for k, v in SDXL_SHAPES.items()
                if k != "input_blocks.7.1.transformer_blocks.9.attn2.to_k.weight"
            },
        ),
        # Segmind Vega: [0, 0, 1, 1, 2, 2] - block 4 stops at index 0
        (
            "vega",
            {
                k: v
                for k, v in SDXL_SHAPES.items()
                if not k.startswith(
                    (
                        "input_blocks.4.1.transformer_blocks.1.",
                        "input_blocks.7.1.transformer_blocks.9.",
                    )
                )
            },
        ),
        # FluxInpaint: img_in takes 384 (in_channels 96)
        ("flux-inpaint", {**FLUX_SHAPES, "img_in.weight": (3072, 384)}),
        # LongCat: independent 3584-wide vector-free family.
        (
            "longcat",
            {
                key: shape
                for key, shape in SCHNELL_SHAPES.items()
                if not key.startswith("vector_in.")
            }
            | {"txt_in.weight": (3072, 3584)},
        ),
    ],
)
def test_catalog_rejects_variant_checkpoints(name: str, shapes: dict[str, tuple[int, ...]]) -> None:
    """Deferred variants stay unrecognized - never misdetected as the
    base family (catalog docstring; ROADMAP 'Native inference')."""
    registry = builtin_family_registry()
    for prefix in ("", "model.diffusion_model."):
        result = registry.detect(DictSource(prefixed(prefix, shapes)))
        assert result.best is None, f"{name} misdetected at prefix {prefix!r}"


def test_catalog_detects_sdxl_vpred_top_level_at_both_unet_prefixes() -> None:
    registry = builtin_family_registry()
    for prefix in ("", "model.diffusion_model."):
        shapes = {**prefixed(prefix, SDXL_SHAPES), "v_pred": (1,), "ztsnr": (1,)}
        result = registry.detect(DictSource(shapes))
        assert result.best is not None
        assert result.best.family_id == "dinkster.sdxl"
        assert result.best.fields["parameterization"] == "v_prediction"
        assert result.best.fields["zsnr"] is True
        assert result.best.matched_keys[-2:] == ("v_pred", "ztsnr")


def test_catalog_detects_sd15_inpaint_combined_and_split() -> None:
    registry = builtin_family_registry()
    inpaint = {**SD15_SHAPES, "input_blocks.0.0.weight": (320, 9, 3, 3)}
    for prefix in ("", "model.diffusion_model."):
        result = registry.detect(DictSource(prefixed(prefix, inpaint)))
        assert result.best is not None
        assert result.best.family_id == "dinkster.sd15"
        assert result.best.fields["in_channels"] == 9


def test_catalog_detects_sdxl_inpaint_combined_and_split() -> None:
    registry = builtin_family_registry()
    inpaint = {**SDXL_SHAPES, "input_blocks.0.0.weight": (320, 9, 3, 3)}
    for prefix in ("", "model.diffusion_model."):
        result = registry.detect(DictSource(prefixed(prefix, inpaint)))
        assert result.best is not None
        assert result.best.family_id == "dinkster.sdxl"
        assert result.best.fields["in_channels"] == 9
        assert result.best.fields["adm_in_channels"] == 2816


def test_catalog_rejects_unsupported_sdxl_sampling_variants() -> None:
    """Playground remains a separate slice."""
    registry = builtin_family_registry()
    combined = prefixed("model.diffusion_model.", SDXL_SHAPES)
    for marker_set in (("edm_mean", "edm_std"), ("edm_mean",)):
        shapes = dict(combined)
        for marker in marker_set:
            shapes[marker] = (1,)
        assert registry.detect(DictSource(shapes)).best is None, marker_set


def test_catalog_detects_sdxl_edm_vpred_marker_shapes() -> None:
    registry = builtin_family_registry()
    for prefix in ("", "model.diffusion_model."):
        shapes = {
            **prefixed(prefix, SDXL_SHAPES),
            "edm_vpred.sigma_max": (),
            "edm_vpred.sigma_min": (1,),
        }
        result = registry.detect(DictSource(shapes))
        assert result.best is not None
        assert result.best.family_id == "dinkster.sdxl"
        assert result.best.fields["parameterization"] == "v_prediction"
        assert result.best.fields["sampling_space"] == "continuous_edm"

    missing_max = {**SDXL_SHAPES, "edm_vpred.sigma_min": ()}
    bad_shape = {**SDXL_SHAPES, "edm_vpred.sigma_max": (2,)}
    assert registry.detect(DictSource(missing_max)).best is None
    assert registry.detect(DictSource(bad_shape)).best is None


def test_catalog_evidence_fields() -> None:
    registry = builtin_family_registry()
    result = registry.detect(DictSource(SDXL_SHAPES))
    assert result.best is not None
    assert result.best.fields["context_dim"] == 2048
    assert result.best.fields["adm_in_channels"] == 2816
    flux = registry.detect(DictSource(FLUX_SHAPES))
    assert flux.best is not None
    assert flux.best.fields["hidden_size"] == 3072
    assert flux.best.fields["context_in_dim"] == 4096


def test_catalog_constants_are_grounded() -> None:
    by_id = {family.id: family for family in builtin_families()}
    assert by_id["dinkster.sd15"].single_stream_latent().scale_factor == 0.18215
    assert by_id["dinkster.sdxl"].single_stream_latent().scale_factor == 0.13025
    assert by_id["dinkster.sdxl"].memory_factor == 0.8
    flux = by_id["dinkster.flux_dev"]
    flux_latent = flux.single_stream_latent()
    assert flux_latent.channels == 16
    assert flux_latent.scale_factor == 0.3611
    assert flux_latent.shift_factor == 0.1159
    assert flux.sampling.shift == 1.15
    assert flux.memory_factor == 3.1
    # float64 evaluation of flux_time_shift(1.15, 1.0, 1/10000)
    # = exp(1.15) / (exp(1.15) + 9999)
    assert flux.sampling.sigma_min == pytest.approx(0.0003157511457805717, rel=1e-12)
    schnell = by_id["dinkster.flux_schnell"]
    assert schnell.sampling.shift == 1.0
    # ModelSamplingDiscreteFlow @ shift=1, multiplier=1: first step
    assert schnell.sampling.sigma_min == 0.001
    assert by_id["dinkster.sd15"].sampling.sigma_max == pytest.approx(14.614641229333646)
    assert by_id["dinkster.sd15"].wiring.text_encoders == ("dinkster.clip_l",)
    assert flux.wiring.vae_prefix == "vae."
    assert flux.wiring.text_encoders == ("dinkster.clip_l", "dinkster.t5xxl")


def test_detection_end_to_end_from_safetensors(tmp_path: Path) -> None:
    """The full stage-2 pipeline: file -> strict header -> detection."""
    tensors = {"model.diffusion_model." + key: ("F16", shape) for key, shape in SD15_SHAPES.items()}
    path = write_safetensors(tmp_path / "sd15.safetensors", tensors, {"format": "pt"})
    source = load_safetensors_header(path)
    result = builtin_family_registry().detect(source)
    assert result.best is not None
    assert result.best.family_id == "dinkster.sd15"
    assert result.best.fields["key_prefix"] == "model.diffusion_model."
    assert result.best.fields["context_dim"] == 768
