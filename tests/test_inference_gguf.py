"""GGUF container, diffusion admission, and upstream decode conformance."""

from __future__ import annotations

import hashlib
import math
import os
import struct
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from itertools import count
from pathlib import Path
from typing import BinaryIO

import pytest
from dinkster_inference import (
    ADMITTED_GGML_TYPES,
    FLOAT32,
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
    T5_XXL_CONFIG,
    UMT5_XXL_CONFIG,
    GGMLType,
    GGUFArtifactIdentity,
    GGUFDecodeError,
    GGUFEncodedLayout,
    GGUFExecutionKind,
    GGUFExecutionRoute,
    GGUFIdentityError,
    GGUFMappingError,
    GGUFMetadataValue,
    GGUFResidencyRefusal,
    GGUFSource,
    GGUFStorageRefusal,
    GGUFStorageRefusalCode,
    GGUFTensor,
    GGUFValueType,
    GGUFWeightSource,
    MalformedGGUF,
    Registry,
    T5Config,
    TensorGeometry,
    admit_gguf_encoded_storage,
    build_runtime_identity_from_facts,
    builtin_gguf_storage_registry,
    decode_ggml_blocks,
    decode_gguf_encoded_storage,
    gguf_runtime_identity_facts,
    identify_gguf_artifact,
    load_gguf,
    load_gguf_weight_source,
    map_gguf_component,
    map_gguf_diffusion_component,
    map_gguf_text_component,
    t5_layout,
)
from dinkster_inference.gguf_identity import gguf_manifest_sha256

_UPSTREAM_COMMIT = "d83f72d463287ab9c50b4bc18ee332104a963889"
_F32 = ADMITTED_GGML_TYPES[0]
_F16 = ADMITTED_GGML_TYPES[1]
_ENCODED_STORAGE_INDEX = count()


def _string(value: str | bytes) -> bytes:
    encoded = value.encode() if isinstance(value, str) else value
    return struct.pack("<Q", len(encoded)) + encoded


def _scalar(value_type: GGUFValueType, value: object) -> bytes:
    formats = {
        GGUFValueType.UINT8: "B",
        GGUFValueType.INT8: "b",
        GGUFValueType.UINT16: "H",
        GGUFValueType.INT16: "h",
        GGUFValueType.UINT32: "I",
        GGUFValueType.INT32: "i",
        GGUFValueType.FLOAT32: "f",
        GGUFValueType.BOOL: "b",
        GGUFValueType.UINT64: "Q",
        GGUFValueType.INT64: "q",
        GGUFValueType.FLOAT64: "d",
    }
    if value_type is GGUFValueType.STRING:
        return _string(str(value))
    return struct.pack("<" + formats[value_type], value)


def _metadata(key: str | bytes, value_type: GGUFValueType | int, value: object) -> bytes:
    encoded = _string(key) + struct.pack("<I", value_type)
    if value_type == GGUFValueType.ARRAY:
        element_type, items = value  # type: ignore[misc]
        encoded += struct.pack("<IQ", element_type, len(items))
        return encoded + b"".join(_scalar(element_type, item) for item in items)
    return encoded + _scalar(GGUFValueType(value_type), value)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _tensor_bytes(type_code: int, wire_shape: tuple[int, ...]) -> int:
    ggml_type = ADMITTED_GGML_TYPES.get(type_code)
    if ggml_type is None:
        return 0
    elements = math.prod(wire_shape)
    return elements // ggml_type.block_elements * ggml_type.block_bytes


def _write_gguf(
    path: Path,
    *,
    metadata: tuple[bytes, ...] = (),
    tensors: tuple[tuple[str | bytes, tuple[int, ...], int, int | None], ...] = (),
    alignment: int = 32,
    version: int = 3,
    add_quantization_version: bool = True,
) -> Path:
    fields = list(metadata)
    if add_quantization_version and any(
        ADMITTED_GGML_TYPES.get(type_code, _F32).quantized for _, _, type_code, _ in tensors
    ):
        fields.append(_metadata("general.quantization_version", GGUFValueType.UINT32, 2))
    header = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(fields)))
    header.extend(b"".join(fields))
    expected_offset = 0
    for name, shape, type_code, override_offset in tensors:
        offset = expected_offset if override_offset is None else override_offset
        header.extend(_string(name))
        header.extend(struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "q" * len(shape), *shape))
        header.extend(struct.pack("<IQ", type_code, offset))
        expected_offset = _align(offset + _tensor_bytes(type_code, shape), alignment)
    payload_start = _align(len(header), alignment)
    payload_size = expected_offset if expected_offset <= 1_000_000 else 0
    path.write_bytes(bytes(header) + bytes(payload_start - len(header) + payload_size))
    return path


def _expect_malformed(path: Path, match: str) -> None:
    with pytest.raises(MalformedGGUF, match=match):
        load_gguf(path)


def test_parser_accepts_v3_without_claiming_component_admission(tmp_path: Path) -> None:
    metadata = tuple(
        _metadata(f"value.{value_type.name}", value_type, value)
        for value_type, value in (
            (GGUFValueType.UINT8, 255),
            (GGUFValueType.INT8, -2),
            (GGUFValueType.UINT16, 65535),
            (GGUFValueType.INT16, -3),
            (GGUFValueType.UINT32, 7),
            (GGUFValueType.INT32, -8),
            (GGUFValueType.FLOAT32, 1.25),
            (GGUFValueType.BOOL, True),
            (GGUFValueType.STRING, "metadata only"),
            (GGUFValueType.UINT64, 1 << 40),
            (GGUFValueType.INT64, -(1 << 40)),
            (GGUFValueType.FLOAT64, -2.5),
        )
    ) + (_metadata("value.ARRAY", GGUFValueType.ARRAY, (GGUFValueType.INT32, (1, 2, 3))),)
    source = load_gguf(_write_gguf(tmp_path / "metadata.gguf", metadata=metadata))

    assert source.version == 3
    assert source.keys() == ()
    assert source.metadata_values["value.ARRAY"] == GGUFMetadataValue(
        GGUFValueType.ARRAY, (1, 2, 3), GGUFValueType.INT32
    )
    with pytest.raises(GGUFMappingError, match="general.architecture"):
        map_gguf_diffusion_component(source)


def test_parser_records_admitted_physical_types_shapes_and_ranges(tmp_path: Path) -> None:
    tensors = tuple(
        (f"tensor.{item.name}", (item.block_elements, 2), item.code, None)
        for item in ADMITTED_GGML_TYPES.values()
    )
    source = load_gguf(_write_gguf(tmp_path / "types.gguf", tensors=tensors))

    assert tuple(tensor.ggml_type for tensor in source.tensors.values()) == tuple(
        ADMITTED_GGML_TYPES.values()
    )
    for tensor in source.tensors.values():
        assert tensor.shape == (2, tensor.ggml_type.block_elements)
        assert tensor.offset == source.data_offset + tensor.relative_offset
        assert tensor.nbytes == 2 * tensor.ggml_type.block_bytes
    with pytest.raises(TypeError):
        source.tensors["new"] = next(iter(source.tensors.values()))  # type: ignore[index]


@pytest.mark.parametrize("cut", (0, 3, 7, 12, 23, 31))
def test_malformed_corpus_refuses_truncation(tmp_path: Path, cut: int) -> None:
    path = _write_gguf(tmp_path / f"truncated-{cut}.gguf", tensors=(("w", (32,), Q4_0.code, None),))
    path.write_bytes(path.read_bytes()[:cut])
    _expect_malformed(path, "truncated|missing|counts")


def test_malformed_corpus_refuses_magic_versions_and_overflowing_counts(
    tmp_path: Path,
) -> None:
    base = _write_gguf(tmp_path / "base.gguf").read_bytes()
    cases = {
        "magic": (b"NOPE" + base[4:], "magic"),
        "v2": (base[:4] + struct.pack("<I", 2) + base[8:], "version must be 3"),
        "v4": (base[:4] + struct.pack("<I", 4) + base[8:], "version must be 3"),
        "endian": (base[:4] + struct.pack(">I", 3) + base[8:], "big-endian"),
        "tensor-count": (
            base[:8] + struct.pack("<Q", (1 << 64) - 1) + base[16:],
            "counts",
        ),
        "metadata-count": (
            base[:16] + struct.pack("<Q", (1 << 64) - 1) + base[24:],
            "counts",
        ),
    }
    for name, (data, match) in cases.items():
        path = tmp_path / f"bad-{name}.gguf"
        path.write_bytes(data)
        _expect_malformed(path, match)


def test_malformed_corpus_refuses_metadata_types_values_and_keys(tmp_path: Path) -> None:
    cases = (
        ((_string("x") + struct.pack("<I", 99),), "invalid metadata type"),
        (
            (_string("x") + struct.pack("<IIQ", GGUFValueType.ARRAY, GGUFValueType.ARRAY, 0),),
            "element type must be scalar",
        ),
        ((_metadata("x", GGUFValueType.BOOL, 2),), "boolean"),
        ((_metadata("", GGUFValueType.UINT8, 1),), "must not be empty"),
        ((_metadata(b"\xff", GGUFValueType.UINT8, 1),), "UTF-8"),
        ((_metadata("bad\x00key", GGUFValueType.UINT8, 1),), "contains NUL"),
        (
            (
                _metadata("same", GGUFValueType.UINT8, 1),
                _metadata("same", GGUFValueType.UINT8, 2),
            ),
            "duplicate metadata",
        ),
    )
    for index, (metadata, match) in enumerate(cases):
        _expect_malformed(
            _write_gguf(tmp_path / f"metadata-{index}.gguf", metadata=metadata), match
        )

    count = _string("x") + struct.pack(
        "<IIQ", GGUFValueType.ARRAY, GGUFValueType.UINT8, (1 << 64) - 1
    )
    _expect_malformed(_write_gguf(tmp_path / "array-count.gguf", metadata=(count,)), "array count")


def test_malformed_corpus_refuses_alignment_and_tensor_table_errors(
    tmp_path: Path,
) -> None:
    cases = (
        (
            (_metadata("general.alignment", GGUFValueType.UINT32, 24),),
            (),
            "power of two",
        ),
        ((), (("", (1,), _F32.code, None),), "name must not be empty"),
        ((), (("bad\x00name", (1,), _F32.code, None),), "contains NUL"),
        ((), (("n" * 128, (1,), _F32.code, None),), "shorter than 128"),
        ((), (("w", (1, 1, 1, 1, 1), _F32.code, None),), "rank 5"),
        ((), (("w", (-1,), _F32.code, None),), "non-negative"),
        ((), (("w", ((1 << 62), 4), _F32.code, None),), "overflows int64"),
        ((), (("w", (1,), 3, None),), "unsupported ggml type Q4_1"),
        ((), (("w", (1,), 99, None),), "invalid ggml type"),
        ((), (("w", (31,), Q4_0.code, None),), "not divisible"),
        (
            (),
            (("same", (1,), _F32.code, None), ("same", (1,), _F32.code, None)),
            "duplicate tensor",
        ),
        (
            (),
            (("a", (1,), _F32.code, None), ("b", (1,), _F32.code, 48)),
            "misaligned",
        ),
        (
            (),
            (("a", (1,), _F32.code, None), ("b", (1,), _F32.code, 0)),
            "overlaps",
        ),
    )
    for index, (metadata, tensors, match) in enumerate(cases):
        _expect_malformed(
            _write_gguf(tmp_path / f"tensor-{index}.gguf", metadata=metadata, tensors=tensors),
            match,
        )


def test_malformed_corpus_refuses_quantization_and_data_span_errors(
    tmp_path: Path,
) -> None:
    no_version = _write_gguf(
        tmp_path / "no-version.gguf",
        tensors=(("w", (32,), Q4_0.code, None),),
        add_quantization_version=False,
    )
    _expect_malformed(no_version, "quantization_version")

    wrong_version = _write_gguf(
        tmp_path / "wrong-version.gguf",
        metadata=(_metadata("general.quantization_version", GGUFValueType.STRING, "2"),),
        tensors=(("w", (32,), Q4_0.code, None),),
        add_quantization_version=False,
    )
    _expect_malformed(wrong_version, "must be UINT32")

    valid = _write_gguf(tmp_path / "data.gguf", tensors=(("w", (32,), Q4_0.code, None),))
    raw = valid.read_bytes()
    valid.write_bytes(raw[:-1])
    _expect_malformed(valid, "truncated")
    valid.write_bytes(raw + b"x")
    _expect_malformed(valid, "trailing bytes")

    padded = _write_gguf(tmp_path / "padding.gguf", tensors=(("w", (32,), Q4_0.code, None),))
    raw = bytearray(padded.read_bytes())
    source = load_gguf(padded)
    raw[source.data_offset - 1] = 1
    padded.write_bytes(raw)
    _expect_malformed(padded, "padding")


def _source(
    metadata: Mapping[str, GGUFMetadataValue], shapes: Mapping[str, tuple[int, ...]]
) -> GGUFSource:
    tensors = {
        name: GGUFTensor(name, shape, tuple(reversed(shape)), _F32, 0, 0, 0)
        for name, shape in shapes.items()
    }
    return GGUFSource(Path("mapping.gguf"), 3, 32, 0, tensors, metadata)


def _architecture(value: str) -> dict[str, GGUFMetadataValue]:
    return {"general.architecture": GGUFMetadataValue(GGUFValueType.STRING, value)}


_SD15_SHAPES = {
    "input_blocks.0.0.weight": (320, 4, 3, 3),
    "input_blocks.1.1.proj_in.weight": (320, 320, 1, 1),
    "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, 768),
}
_FLUX_SHAPES = {
    "img_in.weight": (3072, 64),
    "txt_in.weight": (3072, 4096),
    "vector_in.in_layer.weight": (3072, 768),
    "double_blocks.0.img_attn.norm.key_norm.scale": (128,),
    "guidance_in.in_layer.weight": (3072, 256),
}
_SDXL_SHAPES = {
    "input_blocks.0.0.weight": (320, 4, 3, 3),
    "label_emb.0.0.weight": (1280, 2816),
    "input_blocks.4.1.proj_in.weight": (640, 640),
    "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": (640, 2048),
    "input_blocks.4.1.transformer_blocks.1.attn2.to_k.weight": (640, 2048),
    "input_blocks.7.1.transformer_blocks.9.attn2.to_k.weight": (1280, 2048),
}
_SDXL_REFINER_SHAPES = {
    "input_blocks.0.0.weight": (384, 4, 3, 3),
    "label_emb.0.0.weight": (1536, 2560),
    "input_blocks.4.1.proj_in.weight": (768, 768),
    "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": (768, 1280),
    "input_blocks.4.1.transformer_blocks.3.attn2.to_k.weight": (768, 1280),
}


@pytest.mark.parametrize(
    ("architecture", "shapes", "family_id"),
    (
        ("sd1", _SD15_SHAPES, "dinkster.sd15"),
        ("sdxl", _SDXL_SHAPES, "dinkster.sdxl"),
        ("sdxl", _SDXL_REFINER_SHAPES, "dinkster.sdxl_refiner"),
        ("flux", _FLUX_SHAPES, "dinkster.flux_dev"),
        (
            "flux",
            {
                key: shape
                for key, shape in _FLUX_SHAPES.items()
                if key != "guidance_in.in_layer.weight"
            },
            "dinkster.flux_schnell",
        ),
    ),
)
def test_mapper_admits_every_declared_architecture_family(
    architecture: str, shapes: Mapping[str, tuple[int, ...]], family_id: str
) -> None:
    mapped = map_gguf_diffusion_component(_source(_architecture(architecture), shapes))
    assert mapped.family_id == family_id


def test_mapper_admits_explicit_bare_and_prefixed_diffusion_profiles(tmp_path: Path) -> None:
    sd15 = map_gguf_diffusion_component(_source(_architecture("sd1"), _SD15_SHAPES))
    assert (sd15.mapper_id, sd15.family_id, sd15.component, sd15.tensor_prefix) == (
        "dinkster.gguf.diffusion.v1",
        "dinkster.sd15",
        "diffusion",
        "",
    )
    assert tuple(sd15.tensors) == tuple(_SD15_SHAPES)

    prefix = "model.diffusion_model."
    prefixed = {prefix + key: shape for key, shape in _FLUX_SHAPES.items()}
    prefixed["first_stage_model.decoder.weight"] = (4, 4)
    flux = map_gguf_diffusion_component(_source(_architecture("flux"), prefixed))
    assert flux.family_id == "dinkster.flux_dev"
    assert flux.tensor_prefix == prefix
    assert "first_stage_model.decoder.weight" not in flux.tensors
    assert flux.tensors["img_in.weight"].source_name == prefix + "img_in.weight"

    metadata = [_metadata("general.architecture", GGUFValueType.STRING, "sd1")]
    tensors = []
    for key, shape in _SD15_SHAPES.items():
        name = prefix + key
        metadata.append(
            _metadata(
                "comfy.gguf.orig_shape." + name,
                GGUFValueType.ARRAY,
                (GGUFValueType.INT32, shape),
            )
        )
        tensors.append((name, (256, math.prod(shape) // 256), Q4_0.code, None))
    parsed = load_gguf(
        _write_gguf(
            tmp_path / "prefixed-sd1.gguf",
            metadata=tuple(metadata),
            tensors=tuple(tensors),
        )
    )
    assert map_gguf_diffusion_component(parsed).family_id == "dinkster.sd15"


def test_mapper_applies_only_typed_exact_original_shape_metadata() -> None:
    metadata = _architecture("sd1")
    metadata["comfy.gguf.orig_shape.input_blocks.0.0.weight"] = GGUFMetadataValue(
        GGUFValueType.ARRAY,
        _SD15_SHAPES["input_blocks.0.0.weight"],
        GGUFValueType.INT32,
    )
    physical = dict(_SD15_SHAPES)
    physical["input_blocks.0.0.weight"] = (320, 4, 9)
    mapped = map_gguf_diffusion_component(_source(metadata, physical))
    assert mapped.tensors["input_blocks.0.0.weight"].logical_shape == (
        320,
        4,
        3,
        3,
    )

    bad = dict(metadata)
    bad["comfy.gguf.orig_shape.unknown"] = GGUFMetadataValue(
        GGUFValueType.ARRAY, (1,), GGUFValueType.INT32
    )
    with pytest.raises(GGUFMappingError, match="unknown tensor"):
        map_gguf_diffusion_component(_source(bad, physical))


def test_mapper_names_unknown_conflicting_and_ambiguous_refusals() -> None:
    with pytest.raises(GGUFMappingError, match="unsupported.*qwen_image"):
        map_gguf_diffusion_component(_source(_architecture("qwen_image"), {}))
    with pytest.raises(GGUFMappingError, match="unknown 'sd1'"):
        map_gguf_diffusion_component(_source(_architecture("sd1"), {"weight": (1,)}))
    with pytest.raises(GGUFMappingError, match="conflicts.*dinkster.sd15"):
        map_gguf_diffusion_component(_source(_architecture("flux"), _SD15_SHAPES))
    with pytest.raises(GGUFMappingError, match="ambiguous.*dinkster.flux_dev.*dinkster.sd15"):
        map_gguf_diffusion_component(
            _source(_architecture("sd1"), {**_SD15_SHAPES, **_FLUX_SHAPES})
        )

    prefix = "model.diffusion_model."
    prefixed_sd15 = {prefix + key: shape for key, shape in _SD15_SHAPES.items()}
    with pytest.raises(GGUFMappingError, match="ambiguous.*dinkster.flux_dev.*dinkster.sd15"):
        map_gguf_diffusion_component(
            _source(_architecture("sd1"), {**prefixed_sd15, **_FLUX_SHAPES})
        )

    prefixed_sdxl = {prefix + key: shape for key, shape in _SDXL_SHAPES.items()}
    prefixed_sdxl["edm_mean"] = (1,)
    prefixed_sdxl["edm_std"] = (1,)
    with pytest.raises(GGUFMappingError, match="unknown 'sdxl'"):
        map_gguf_diffusion_component(_source(_architecture("sdxl"), prefixed_sdxl))


# llama.cpp t5encoder export names (city96 t5/umt5 encoder GGUFs)
# for the HF-style model keys t5_layout produces.
_T5_GGUF_EXACT = {
    "shared.weight": "token_embd.weight",
    "encoder.final_layer_norm.weight": "enc.output_norm.weight",
}
_T5_GGUF_SUFFIXES = {
    "layer.0.SelfAttention.q.weight": "attn_q.weight",
    "layer.0.SelfAttention.k.weight": "attn_k.weight",
    "layer.0.SelfAttention.v.weight": "attn_v.weight",
    "layer.0.SelfAttention.o.weight": "attn_o.weight",
    "layer.0.SelfAttention.relative_attention_bias.weight": "attn_rel_b.weight",
    "layer.0.layer_norm.weight": "attn_norm.weight",
    "layer.1.DenseReluDense.wi_0.weight": "ffn_gate.weight",
    "layer.1.DenseReluDense.wi_1.weight": "ffn_up.weight",
    "layer.1.DenseReluDense.wo.weight": "ffn_down.weight",
    "layer.1.layer_norm.weight": "ffn_norm.weight",
}


def _llama_t5_name(model_key: str) -> str:
    exact = _T5_GGUF_EXACT.get(model_key)
    if exact is not None:
        return exact
    block, _, suffix = model_key.removeprefix("encoder.block.").partition(".")
    return f"enc.blk.{block}.{_T5_GGUF_SUFFIXES[suffix]}"


def _llama_t5_shapes(config: T5Config) -> dict[str, tuple[int, ...]]:
    return {_llama_t5_name(key): shape for key, shape in t5_layout(config).items()}


@pytest.mark.parametrize("architecture", ("t5encoder", "t5"))
@pytest.mark.parametrize(
    ("config", "component"),
    ((T5_XXL_CONFIG, "t5xxl"), (UMT5_XXL_CONFIG, "umt5xxl")),
)
def test_text_mapper_admits_exact_xxl_layouts_under_both_architectures(
    architecture: str, config: T5Config, component: str
) -> None:
    mapped = map_gguf_text_component(_source(_architecture(architecture), _llama_t5_shapes(config)))
    assert (
        mapped.mapper_id,
        mapped.architecture,
        mapped.family_id,
        mapped.component,
        mapped.tensor_prefix,
    ) == ("dinkster.gguf.text.v1", architecture, f"dinkster.text.{component}", component, "")
    layout = t5_layout(config)
    assert set(mapped.tensors) == set(layout)
    for model_key, tensor in mapped.tensors.items():
        assert tensor.model_key == model_key
        assert tensor.source_name == _llama_t5_name(model_key)
        assert tensor.logical_shape == layout[model_key]
    public = mapped.public_tensors()
    assert set(public) == set(layout)
    assert public["shared.weight"].source_name == "token_embd.weight"


def test_text_mapper_names_architecture_and_tensor_name_refusals() -> None:
    shapes = _llama_t5_shapes(T5_XXL_CONFIG)
    with pytest.raises(GGUFMappingError, match="unsupported text GGUF architecture 'qwen3'"):
        map_gguf_text_component(_source(_architecture("qwen3"), shapes))
    with pytest.raises(GGUFMappingError, match=r"unknown 't5encoder' text tensor name.*dec\.blk"):
        map_gguf_text_component(
            _source(
                _architecture("t5encoder"),
                {**shapes, "dec.blk.0.attn_q.weight": (4096, 4096)},
            )
        )
    for name in ("enc.blk.01.attn_q.weight", "enc.blk.0.attn_qq.weight", "spiece_model"):
        with pytest.raises(GGUFMappingError, match="unknown 't5' text tensor name"):
            map_gguf_text_component(_source(_architecture("t5"), {**shapes, name: (1,)}))


def test_text_mapper_chains_exact_geometry_refusals() -> None:
    shapes = _llama_t5_shapes(UMT5_XXL_CONFIG)
    with pytest.raises(GGUFMappingError, match="unsupported text GGUF layout: unknown T5 geometry"):
        map_gguf_text_component(
            _source(_architecture("t5encoder"), {**shapes, "token_embd.weight": (32000, 4096)})
        )
    with pytest.raises(GGUFMappingError, match="not a T5 text-encoder state dict"):
        map_gguf_text_component(
            _source(
                _architecture("t5encoder"),
                {key: value for key, value in shapes.items() if key != "token_embd.weight"},
            )
        )
    with pytest.raises(GGUFMappingError, match="does not match the UMT5-XXL layout"):
        map_gguf_text_component(
            _source(_architecture("t5encoder"), {**shapes, "enc.blk.3.ffn_down.weight": (16, 16)})
        )


def test_component_dispatcher_routes_by_declared_architecture() -> None:
    diffusion = map_gguf_component(_source(_architecture("sd1"), _SD15_SHAPES))
    assert (diffusion.component, diffusion.family_id) == ("diffusion", "dinkster.sd15")
    text = map_gguf_component(_source(_architecture("t5encoder"), _llama_t5_shapes(T5_XXL_CONFIG)))
    assert (text.component, text.family_id) == ("t5xxl", "dinkster.text.t5xxl")
    with pytest.raises(
        GGUFMappingError,
        match=(
            r"unsupported GGUF architecture 'qwen3' "
            r"\(supported: flux, sd1, sdxl, t5, t5encoder\)"
        ),
    ):
        map_gguf_component(_source(_architecture("qwen3"), {}))
    with pytest.raises(GGUFMappingError, match="missing required general.architecture"):
        map_gguf_component(_source({}, {}))


def _decode_vectors() -> tuple[tuple[object, bytes, str], ...]:
    q40 = struct.pack("<H", 0x3400) + bytes((index * 29 + 7) & 0xFF for index in range(16))
    q80 = struct.pack("<H", 0x3800) + bytes((index * 9 - 127) & 0xFF for index in range(32))
    scales = bytes((index * 37 + 11) & 0xFF for index in range(12))
    low = bytes((index * 53 + 19) & 0xFF for index in range(128))
    q4k = struct.pack("<HH", 0x3400, 0x2C00) + scales + low
    high = bytes((index * 71 + 5) & 0xFF for index in range(32))
    q5k = struct.pack("<HH", 0x3400, 0x2C00) + scales + high + low
    upper = bytes((index * 71 + 5) & 0xFF for index in range(64))
    subgroup_scales = struct.pack("<16b", *(index * 15 - 113 for index in range(16)))
    q6k = low + upper + subgroup_scales + struct.pack("<H", 0x3400)
    return (
        (Q4_0, q40, "2fc5ff46c0513030258e9c0854469950e50e08f75d87cb9555038ca674c9862f"),
        (Q8_0, q80, "53f15ea0f50d0a163beeee97effa4af53c0734bdb9941daa300765e32e23c845"),
        (Q4_K, q4k, "8a16be5acf43c202d01459b00360cce0cf13b96e4d3297c8208687414592e5a9"),
        (Q5_K, q5k, "a72d917bcfbf046730719706ddf1e1edb63f187dec8dfd91ac675fe01a21d7b2"),
        (Q6_K, q6k, "fa0b39ef876c55d997e2b4c4da37c5f7f3bd5f8d0c3a6c756421562e88bae06f"),
    )


@pytest.mark.parametrize(("ggml_type", "encoded", "expected_sha256"), _decode_vectors())
def test_reference_decoders_match_pinned_upstream_fp32_vectors(
    ggml_type: object, encoded: bytes, expected_sha256: str
) -> None:
    assert _UPSTREAM_COMMIT == "d83f72d463287ab9c50b4bc18ee332104a963889"
    values = decode_ggml_blocks(ggml_type, encoded)  # type: ignore[arg-type]
    output = struct.pack(f"<{len(values)}f", *values)
    assert len(values) == ggml_type.block_elements  # type: ignore[attr-defined]
    assert hashlib.sha256(output).hexdigest() == expected_sha256
    assert all(math.isfinite(value) for value in values)


def test_reference_decoder_validates_blocks_and_concatenates() -> None:
    for ggml_type, encoded, _ in _decode_vectors():
        one = decode_ggml_blocks(ggml_type, encoded)  # type: ignore[arg-type]
        assert decode_ggml_blocks(ggml_type, encoded + encoded) == one + one  # type: ignore[arg-type]
        with pytest.raises(GGUFDecodeError, match="not a multiple"):
            decode_ggml_blocks(ggml_type, encoded[:-1])  # type: ignore[arg-type]
    with pytest.raises(GGUFDecodeError, match="no admitted reference decoder"):
        decode_ggml_blocks(_F32, b"\0\0\0\0")


def _encoded_storage(
    tmp_path: Path, ggml_type: GGMLType, encoded: bytes
) -> tuple[GGUFWeightSource, str, bytes]:
    target_key = "input_blocks.0.0.weight"
    metadata = [_metadata("general.architecture", GGUFValueType.STRING, "sd1")]
    tensors = []
    for key, logical_shape in _SD15_SHAPES.items():
        metadata.append(
            _metadata(
                "comfy.gguf.orig_shape." + key,
                GGUFValueType.ARRAY,
                (GGUFValueType.INT32, logical_shape),
            )
        )
        element_count = math.prod(logical_shape)
        tensors.append(
            (
                key,
                (ggml_type.block_elements, element_count // ggml_type.block_elements),
                ggml_type.code,
                None,
            )
        )
    path = _write_gguf(
        tmp_path / f"storage-{ggml_type.name}-{next(_ENCODED_STORAGE_INDEX)}.gguf",
        metadata=tuple(metadata),
        tensors=tuple(tensors),
    )
    source = load_gguf(path)
    parsed = source.tensor(target_key)
    payload = encoded * (parsed.numel // ggml_type.block_elements)
    raw = bytearray(path.read_bytes())
    raw[parsed.offset : parsed.offset + parsed.nbytes] = payload
    path.write_bytes(raw)
    return load_gguf_weight_source(path, residency_mode="speed"), target_key, payload


def _rewrite_held_artifact(path: Path, data: bytes) -> bool:
    try:
        path.write_bytes(data)
    except PermissionError:
        if os.name != "nt":
            raise
        return False
    return True


def _replace_held_artifact(path: Path, replacement: Path) -> Path:
    retained = path.with_name(path.name + ".retained")
    path.replace(retained)
    replacement.replace(path)
    return retained


def test_encoded_storage_registry_declares_exact_supported_layouts() -> None:
    registry = builtin_gguf_storage_registry()
    assert registry.ids() == (
        "dinkster.gguf.q4_0",
        "dinkster.gguf.q8_0",
        "dinkster.gguf.q4_k",
        "dinkster.gguf.q5_k",
        "dinkster.gguf.q6_k",
    )
    assert tuple(layout.ggml_type for layout in registry) == (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K)


@pytest.mark.parametrize(("ggml_type", "encoded", "_expected_sha256"), _decode_vectors())
def test_admission_keeps_file_slice_and_binds_parser_layout_facts(
    tmp_path: Path, ggml_type: GGMLType, encoded: bytes, _expected_sha256: str
) -> None:
    source, model_key, payload = _encoded_storage(tmp_path, ggml_type, encoded)
    storage = admit_gguf_encoded_storage(source, model_key)

    assert storage.file_slice.read() == payload
    assert storage.source_path == source.path
    assert (storage.mapper_id, storage.architecture, storage.family_id, storage.component) == (
        "dinkster.gguf.diffusion.v1",
        "sd1",
        "dinkster.sd15",
        "diffusion",
    )
    assert storage.source_name == model_key
    assert storage.model_key == model_key
    assert storage.logical_shape == _SD15_SHAPES[model_key]
    assert storage.offset == source.source.tensor(model_key).offset
    assert storage.descriptor.dtype_tag == ggml_type.name
    assert storage.descriptor.ggml_type is source.source.tensor(model_key).ggml_type
    assert storage.descriptor.block_elements == ggml_type.block_elements
    assert storage.descriptor.block_bytes == ggml_type.block_bytes
    assert storage.descriptor.element_count == math.prod(_SD15_SHAPES[model_key])
    assert storage.descriptor.encoded_bytes == len(payload)
    with pytest.raises(FrozenInstanceError):
        storage.descriptor.element_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        storage.file_slice = storage.file_slice  # type: ignore[misc]
    with pytest.raises(ValueError, match="dimensions"):
        replace(storage.descriptor, block_bytes=storage.descriptor.block_bytes + 1)
    with pytest.raises(ValueError, match="logical shape"):
        replace(storage, logical_shape=(1,))


@pytest.mark.parametrize(("ggml_type", "encoded", "expected_sha256"), _decode_vectors())
def test_encoded_storage_decode_is_byte_identical_to_reference_decoder(
    tmp_path: Path, ggml_type: GGMLType, encoded: bytes, expected_sha256: str
) -> None:
    source, model_key, payload = _encoded_storage(tmp_path, ggml_type, encoded)
    storage = admit_gguf_encoded_storage(source, model_key)

    expected = decode_ggml_blocks(ggml_type, payload)
    actual = decode_gguf_encoded_storage(storage)
    expected_bytes = struct.pack(f"<{len(expected)}f", *expected)
    actual_bytes = struct.pack(f"<{len(actual)}f", *actual)
    assert actual_bytes == expected_bytes
    block_bytes = ggml_type.block_elements * 4
    assert hashlib.sha256(actual_bytes[:block_bytes]).hexdigest() == expected_sha256


def test_weight_source_binds_mapped_geometry_ranges_and_runtime_facts(
    tmp_path: Path,
) -> None:
    source, model_key, payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    weight_source = source

    assert weight_source.source_format == "gguf"
    assert weight_source.component_map.family_id == "dinkster.sd15"
    assert weight_source.keys() == tuple(weight_source.component_map.tensors)
    entry = weight_source.entry(model_key)
    parsed = weight_source.source.tensor(model_key)
    assert entry.geometry == TensorGeometry(_SD15_SHAPES[model_key], FLOAT32)
    assert (entry.offset, entry.nbytes) == (parsed.offset, len(payload))
    assert any(
        fact.startswith("gguf.artifact.file_sha256=") for fact in weight_source.runtime_facts
    )
    assert "gguf.route.kind=reference-decode" in weight_source.runtime_facts


_SPEED_ROUTE_FACTS = (
    "gguf.route.provider_key=dinkster-gguf-cpu-reference",
    "gguf.route.implementation_version=v1",
    "gguf.route.kind=reference-decode",
    "gguf.route.device_kind=cpu",
    "gguf.route.device_capability=generic",
    "gguf.route.compute_dtype=float32",
    "gguf.route.accumulation_dtype=float32",
)
_MEMORY_ROUTE_FACTS = (
    "gguf.route.provider_key=dinkster-gguf-torch-onuse",
    "gguf.route.implementation_version=v1",
    "gguf.route.kind=bounded-decode",
    "gguf.route.device_kind=any",
    "gguf.route.device_capability=generic",
    "gguf.route.compute_dtype=float32",
    "gguf.route.accumulation_dtype=float32",
)
_BALANCED_ROUTE_FACTS = (
    "gguf.route.provider_key=dinkster-gguf-torch-onuse",
    "gguf.route.implementation_version=v1",
    "gguf.route.kind=cached-decode",
    "gguf.route.device_kind=any",
    "gguf.route.device_capability=generic",
    "gguf.route.compute_dtype=float32",
    "gguf.route.accumulation_dtype=float32",
    "gguf.route.decoded_cache=auto",
)


def test_residency_modes_pin_route_facts_and_share_planned_geometry(tmp_path: Path) -> None:
    speed, model_key, _payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    assert speed.residency_mode == "speed"
    assert speed.runtime_facts[-len(_SPEED_ROUTE_FACTS) :] == _SPEED_ROUTE_FACTS

    memory = load_gguf_weight_source(speed.path, residency_mode="memory")
    assert memory.residency_mode == "memory"
    assert memory.runtime_facts[-len(_MEMORY_ROUTE_FACTS) :] == _MEMORY_ROUTE_FACTS
    artifact_facts = speed.runtime_facts[: -len(_SPEED_ROUTE_FACTS)]
    assert memory.runtime_facts[: -len(_MEMORY_ROUTE_FACTS)] == artifact_facts
    assert memory.runtime_facts != speed.runtime_facts
    assert _identity_with_gguf_facts(memory.runtime_facts) != _identity_with_gguf_facts(
        speed.runtime_facts
    )

    balanced = load_gguf_weight_source(speed.path, residency_mode="balanced")
    assert balanced.residency_mode == "balanced"
    assert balanced.decoded_cache_budget is None
    assert balanced.runtime_facts[-len(_BALANCED_ROUTE_FACTS) :] == _BALANCED_ROUTE_FACTS
    assert balanced.runtime_facts[: -len(_BALANCED_ROUTE_FACTS)] == artifact_facts
    identities = {
        _identity_with_gguf_facts(source.runtime_facts) for source in (speed, memory, balanced)
    }
    assert len(identities) == 3

    # The mode changes execution residency only; planning geometry is shared.
    assert memory.keys() == speed.keys()
    assert memory.entry(model_key) == speed.entry(model_key)
    assert balanced.keys() == speed.keys()
    assert balanced.entry(model_key) == speed.entry(model_key)

    with pytest.raises(GGUFResidencyRefusal, match="unknown GGUF residency mode 'turbo'"):
        load_gguf_weight_source(
            speed.path,
            residency_mode="turbo",  # type: ignore[arg-type]
        )


def test_balanced_decoded_cache_budget_is_normalized_and_gated(tmp_path: Path) -> None:
    speed, _model_key, _payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])

    explicit = load_gguf_weight_source(
        speed.path, residency_mode="balanced", decoded_cache_budget=2_147_483_648
    )
    assert explicit.decoded_cache_budget == 2_147_483_648
    assert explicit.runtime_facts[-1] == "gguf.route.decoded_cache=2147483648"

    auto = load_gguf_weight_source(speed.path, residency_mode="balanced")
    assert auto.runtime_facts[-1] == "gguf.route.decoded_cache=auto"
    # The budget is part of the deterministic route identity.
    assert _identity_with_gguf_facts(explicit.runtime_facts) != _identity_with_gguf_facts(
        auto.runtime_facts
    )
    zero = load_gguf_weight_source(speed.path, residency_mode="balanced", decoded_cache_budget=0)
    assert zero.runtime_facts[-1] == "gguf.route.decoded_cache=0"

    for mode in ("speed", "memory"):
        with pytest.raises(GGUFResidencyRefusal, match="balanced residency mode only"):
            load_gguf_weight_source(
                speed.path,
                residency_mode=mode,  # type: ignore[arg-type]
                decoded_cache_budget=1,
            )
    with pytest.raises(GGUFResidencyRefusal, match="non-negative byte count"):
        load_gguf_weight_source(speed.path, residency_mode="balanced", decoded_cache_budget=-1)


def test_auto_residency_resolves_balanced_for_every_encoded_layout(tmp_path: Path) -> None:
    q8_path = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])[0].path
    default = load_gguf_weight_source(q8_path)
    assert default.residency_mode == "balanced"
    assert default.decoded_cache_budget is None
    assert default.runtime_facts[-len(_BALANCED_ROUTE_FACTS) :] == _BALANCED_ROUTE_FACTS
    # Route facts carry only the resolved mode: an auto-resolved source is
    # byte-identical to an explicit balanced one.
    explicit = load_gguf_weight_source(q8_path, residency_mode="balanced")
    assert default.runtime_facts == explicit.runtime_facts
    spelled = load_gguf_weight_source(q8_path, residency_mode="auto")
    assert spelled.runtime_facts == default.runtime_facts

    # Every encoded-resident layout, not just Q8_0, resolves balanced.
    for ggml_type, encoded, _digest in _decode_vectors():
        assert isinstance(ggml_type, GGMLType)
        path = _encoded_storage(tmp_path, ggml_type, encoded)[0].path
        resolved = load_gguf_weight_source(path)
        assert resolved.residency_mode == "balanced"
        assert resolved.runtime_facts[-len(_BALANCED_ROUTE_FACTS) :] == _BALANCED_ROUTE_FACTS

    # Tuning the cache requires naming the mode that owns it.
    with pytest.raises(GGUFResidencyRefusal, match="balanced residency mode only"):
        load_gguf_weight_source(q8_path, decoded_cache_budget=1)


def test_memory_residency_admits_text_maps_and_refuses_unregistered_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every builtin encoded layout is admitted; the refusal guards a
    # future admitted quant type that lacks an encoded-resident
    # layout, exercised here by hiding Q4_K from the layout table.
    q4k, _model_key, _payload = _encoded_storage(tmp_path, Q4_K, _decode_vectors()[2][1])
    assert q4k.residency_mode == "speed"
    admitted_q4k = load_gguf_weight_source(q4k.path, residency_mode="memory")
    assert admitted_q4k.residency_mode == "memory"

    import dinkster_inference.gguf as gguf_module

    hidden = {
        code: layout
        for code, layout in gguf_module._ENCODED_LAYOUT_BY_CODE.items()
        if layout.ggml_type is not Q4_K
    }
    monkeypatch.setattr(gguf_module, "_ENCODED_LAYOUT_BY_CODE", hidden)
    with pytest.raises(
        GGUFResidencyRefusal, match="requires every quantized tensor to use an encoded-resident"
    ):
        load_gguf_weight_source(q4k.path, residency_mode="memory")
    fallback = load_gguf_weight_source(q4k.path)
    assert fallback.residency_mode == "speed"
    assert fallback.runtime_facts[-len(_SPEED_ROUTE_FACTS) :] == _SPEED_ROUTE_FACTS
    monkeypatch.undo()

    q8_path = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])[0].path
    text_map = map_gguf_text_component(
        _source(_architecture("t5encoder"), _llama_t5_shapes(T5_XXL_CONFIG))
    )
    monkeypatch.setattr("dinkster_inference.gguf.map_gguf_component", lambda source: text_map)
    admitted = load_gguf_weight_source(q8_path, residency_mode="memory")
    assert admitted.residency_mode == "memory"
    assert admitted.component_map.mapper_id == "dinkster.gguf.text.v1"
    assert admitted.runtime_facts[6:] == (
        "gguf.route.provider_key=dinkster-gguf-torch-onuse",
        "gguf.route.implementation_version=v1",
        "gguf.route.kind=bounded-decode",
        "gguf.route.device_kind=any",
        "gguf.route.device_capability=generic",
        "gguf.route.compute_dtype=float32",
        "gguf.route.accumulation_dtype=float32",
    )


def test_encoded_storage_admission_names_unsupported_type_and_identity_refusals(
    tmp_path: Path,
) -> None:
    source, model_key, _payload = _encoded_storage(tmp_path, _F16, bytes(2))
    with pytest.raises(GGUFStorageRefusal) as unsupported:
        admit_gguf_encoded_storage(source, model_key)
    assert unsupported.value.code is GGUFStorageRefusalCode.UNSUPPORTED_QUANT_TYPE

    with pytest.raises(GGUFStorageRefusal) as absent:
        admit_gguf_encoded_storage(source, "different.model.key")
    assert absent.value.code is GGUFStorageRefusalCode.SOURCE_IDENTITY_MISMATCH

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        replace(source, runtime_facts=tuple(reversed(source.runtime_facts)))

    tensors = dict(source.source.tensors)
    parsed = tensors[model_key]
    tensors[model_key] = replace(
        parsed,
        offset=parsed.offset - 1,
        relative_offset=parsed.relative_offset - 1,
    )
    forged_source = replace(source.source, tensors=tensors)
    forged_map = map_gguf_diffusion_component(forged_source)
    forged_facts = list(source.runtime_facts)
    forged_facts[1] = "gguf.artifact.manifest_sha256=" + gguf_manifest_sha256(
        forged_source, forged_map
    )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        replace(
            source,
            source=forged_source,
            component_map=forged_map,
            runtime_facts=tuple(forged_facts),
        )


def test_encoded_storage_admission_names_layout_and_range_mismatches(tmp_path: Path) -> None:
    encoded = _decode_vectors()[0][1]
    source, model_key, payload = _encoded_storage(tmp_path, Q4_0, encoded)

    forged_registry: Registry[GGUFEncodedLayout] = Registry()
    forged_registry.register(
        GGUFEncodedLayout("dinkster.gguf.forged", replace(Q4_0, block_bytes=Q4_0.block_bytes + 1))
    )
    with pytest.raises(GGUFStorageRefusal) as layout:
        admit_gguf_encoded_storage(source, model_key, registry=forged_registry)
    assert layout.value.code is GGUFStorageRefusalCode.LAYOUT_MISMATCH

    duplicate_registry = builtin_gguf_storage_registry()
    duplicate_registry.register(GGUFEncodedLayout("dinkster.gguf.duplicate-q4", Q4_0))
    with pytest.raises(GGUFStorageRefusal) as duplicate:
        admit_gguf_encoded_storage(source, model_key, registry=duplicate_registry)
    assert duplicate.value.code is GGUFStorageRefusalCode.LAYOUT_MISMATCH

    storage = admit_gguf_encoded_storage(source, model_key)
    rewritten = _rewrite_held_artifact(
        source.path, b"replacement".ljust(source.path.stat().st_size, b"\0")
    )
    if not rewritten:
        assert storage.file_slice.read() == payload
        return
    with pytest.raises(OSError, match="changed after identity verification"):
        storage.file_slice.read()


def test_verified_authority_rehashes_unlinked_long_unicode_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep = tmp_path
    for index in range(4):
        deep /= f"unicode-{index}-\u6a21\u578b-" + "x" * 64
    deep.mkdir(parents=True)
    authority, model_key, payload = _encoded_storage(deep, Q8_0, _decode_vectors()[1][1])
    assert len(str(authority.path)) > 260
    original_rehash = authority._artifact._detached_artifact_is_unchanged
    rehashed: list[bool] = []

    def record_rehash(fingerprint: tuple[int, int, int, int, int]) -> bool:
        rehashed.append(True)
        return original_rehash(fingerprint)

    monkeypatch.setattr(authority._artifact, "_detached_artifact_is_unchanged", record_rehash)

    renamed = authority.path.with_name("renamed-\u91cf\u5316.gguf")
    authority.path.replace(renamed)
    renamed.unlink()

    storage = admit_gguf_encoded_storage(authority, model_key)
    assert storage.file_slice.read() == payload
    assert rehashed == [True]


def test_verified_authority_streams_current_artifact_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _model_key, _payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    snapshot_b = bytearray(authority.path.read_bytes())
    key = b"general.quantization_version"
    value_offset = snapshot_b.index(key) + len(key) + 4
    snapshot_b[value_offset : value_offset + 4] = struct.pack("<I", 3)
    replacement = authority.path.with_suffix(".replacement.gguf")
    replacement.write_bytes(snapshot_b)
    retained = _replace_held_artifact(authority.path, replacement)

    def reject_read_bytes(_path: Path) -> bytes:
        raise AssertionError("GGUF authority must not call Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    try:
        loaded = load_gguf_weight_source(authority.path)
    finally:
        retained.unlink()

    assert loaded.source.metadata_values["general.quantization_version"].value == 3
    assert authority.source.metadata_values["general.quantization_version"].value == 2


def test_verified_authority_fails_closed_after_artifact_truncation(tmp_path: Path) -> None:
    authority, model_key, payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    storage = admit_gguf_encoded_storage(authority, model_key)
    if not _rewrite_held_artifact(authority.path, b"GGUF"):
        assert storage.file_slice.read() == payload
        return

    with pytest.raises(OSError, match="changed after identity verification"):
        storage.file_slice.read()


def test_verified_authority_rejects_rewrite_with_restored_mtime(tmp_path: Path) -> None:
    authority, model_key, payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    storage = admit_gguf_encoded_storage(authority, model_key)
    before = authority.path.stat()
    changed = bytearray(authority.path.read_bytes())
    changed[storage.offset + 2] ^= 1
    if not _rewrite_held_artifact(authority.path, bytes(changed)):
        assert storage.file_slice.read() == payload
        return
    os.utime(authority.path, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(OSError, match="changed after identity verification"):
        storage.file_slice.read()


def test_verified_authority_rejects_out_of_range_rewrite_with_restored_mtime(
    tmp_path: Path,
) -> None:
    authority, model_key, payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    storage = admit_gguf_encoded_storage(authority, model_key)
    before = authority.path.stat()
    changed = bytearray(authority.path.read_bytes())
    changed[0] ^= 1
    if not _rewrite_held_artifact(authority.path, bytes(changed)):
        assert storage.file_slice.read() == payload
        return
    os.utime(authority.path, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(OSError, match="changed after identity verification"):
        storage.file_slice.read()


@pytest.mark.skipif(os.name == "nt", reason="Windows authority denies concurrent writers")
def test_verified_authority_rejects_mutation_between_parse_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])[0].path
    parsed = load_gguf(path)
    target = parsed.tensor("input_blocks.0.0.weight")
    before = path.stat()

    import dinkster_inference.gguf as gguf_module

    original_parse = gguf_module._GGUFArtifactHandle.parse

    def parse_then_mutate(self: gguf_module._GGUFArtifactHandle) -> GGUFSource:
        source = original_parse(self)
        with path.open("r+b") as file:
            file.seek(target.offset + 2)
            value = file.read(1)
            file.seek(target.offset + 2)
            file.write(bytes((value[0] ^ 1,)))
            file.flush()
            os.fsync(file.fileno())
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return source

    monkeypatch.setattr(gguf_module._GGUFArtifactHandle, "parse", parse_then_mutate)
    with pytest.raises(OSError, match="changed during identity verification"):
        load_gguf_weight_source(path)


def test_gguf_artifact_identity_is_path_independent_and_payload_complete(tmp_path: Path) -> None:
    encoded = _decode_vectors()[0][1]
    source, model_key, _payload = _encoded_storage(tmp_path, Q4_0, encoded)
    first = identify_gguf_artifact(source.source)

    moved_path = tmp_path / "moved.gguf"
    moved_path.write_bytes(source.path.read_bytes())
    moved = identify_gguf_artifact(load_gguf(moved_path))
    assert moved == first

    raw = bytearray(moved_path.read_bytes())
    raw[source.source.tensor(model_key).offset + 2] ^= 1
    moved_path.write_bytes(raw)
    payload_mutation = identify_gguf_artifact(load_gguf(moved_path))
    assert payload_mutation.file_sha256 != first.file_sha256
    assert payload_mutation.manifest_sha256 == first.manifest_sha256


@pytest.mark.skipif(os.name != "nt", reason="Windows share-mode contract")
def test_gguf_artifact_identity_denies_windows_writer_during_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _model_key, _payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    path = tmp_path / "identity-writer-denial.gguf"
    path.write_bytes(authority.path.read_bytes())
    source = load_gguf(path)

    import dinkster_inference.gguf_identity as identity_module

    original_load = load_gguf

    def load_then_attempt_write(
        artifact_path: Path, *, _artifact_file: BinaryIO | None = None
    ) -> GGUFSource:
        current = original_load(artifact_path, _artifact_file=_artifact_file)
        with pytest.raises(PermissionError):
            with artifact_path.open("r+b"):
                pass
        return current

    monkeypatch.setattr(identity_module, "load_gguf", load_then_attempt_write)
    identify_gguf_artifact(source)


def test_gguf_manifest_identity_is_canonical_and_interpretation_sensitive(tmp_path: Path) -> None:
    source, _model_key, _payload = _encoded_storage(tmp_path, Q4_0, _decode_vectors()[0][1])
    first = identify_gguf_artifact(source.source)
    reordered = replace(
        source.source,
        tensors=dict(reversed(tuple(source.source.tensors.items()))),
        metadata_values=dict(reversed(tuple(source.source.metadata_values.items()))),
    )
    assert identify_gguf_artifact(reordered) == first

    raw = source.path.read_bytes()
    version_2 = _metadata("general.quantization_version", GGUFValueType.UINT32, 2)
    version_3 = _metadata("general.quantization_version", GGUFValueType.UINT32, 3)
    assert raw.count(version_2) == 1
    replacement = source.path.with_suffix(".replacement.gguf")
    replacement.write_bytes(raw.replace(version_2, version_3, 1))
    retained = _replace_held_artifact(source.path, replacement)
    try:
        changed = identify_gguf_artifact(load_gguf(source.path))
        assert changed.file_sha256 != first.file_sha256
        assert changed.manifest_sha256 != first.manifest_sha256

        with pytest.raises(GGUFIdentityError, match="facts changed"):
            identify_gguf_artifact(source.source)
    finally:
        retained.unlink()


def _gguf_route(**changes: object) -> GGUFExecutionRoute:
    return replace(
        GGUFExecutionRoute(
            provider_id="dinkster.gguf.reference",
            implementation_version="v1",
            kind=GGUFExecutionKind.REFERENCE_DECODE,
            device_kind="cpu",
            device_capability="generic",
            compute_dtype="float32",
            accumulation_dtype="float32",
        ),
        **changes,
    )


def _identity_with_gguf_facts(facts: tuple[str, ...]) -> str:
    return build_runtime_identity_from_facts(
        "dinkster.sd15",
        ("family=dinkster.sd15",),
        diffusion_dtype="float32",
        text_dtype="float32",
        vae_dtype="float32",
        fp8_matmul=False,
        runtime_facts=facts,
    )


def test_gguf_route_rotates_runtime_identity() -> None:
    artifact = GGUFArtifactIdentity(
        file_sha256="1" * 64,
        manifest_sha256="2" * 64,
        mapper_id="dinkster.gguf.diffusion.v1",
        architecture="sd1",
        family_id="dinkster.sd15",
        component="diffusion",
    )
    route = _gguf_route()
    base_facts = gguf_runtime_identity_facts(artifact, route)
    base = _identity_with_gguf_facts(base_facts)
    assert base == _identity_with_gguf_facts(base_facts)
    route_changes = (
        {"provider_id": "dinkster.gguf.bounded"},
        {"implementation_version": "v2"},
        {"kind": GGUFExecutionKind.BOUNDED_DECODE},
        {"device_kind": "cuda"},
        {"device_capability": "sm89"},
        {"compute_dtype": "float16"},
        {"accumulation_dtype": "bfloat16"},
    )
    for change in route_changes:
        changed_facts = gguf_runtime_identity_facts(artifact, _gguf_route(**change))
        assert base != _identity_with_gguf_facts(changed_facts)
    assert gguf_runtime_identity_facts(
        artifact,
        _gguf_route(provider_id="vendor.gguf-provider"),
    ) == gguf_runtime_identity_facts(
        artifact,
        _gguf_route(provider_id="vendor.gguf_provider"),
    )


def test_gguf_identity_values_refuse_malformed_facts() -> None:
    artifact = GGUFArtifactIdentity(
        file_sha256="1" * 64,
        manifest_sha256="2" * 64,
        mapper_id="dinkster.gguf.diffusion.v1",
        architecture="sd1",
        family_id="dinkster.sd15",
        component="diffusion",
    )
    with pytest.raises(FrozenInstanceError):
        artifact.file_sha256 = "3" * 64  # type: ignore[misc]
    with pytest.raises(ValueError, match="canonical lowercase token"):
        _gguf_route(device_capability="SM 89")
    with pytest.raises(ValueError, match="invalid id"):
        _gguf_route(provider_id="reference")
    with pytest.raises(TypeError, match="GGUFExecutionKind"):
        _gguf_route(kind="reference-decode")
    with pytest.raises(ValueError, match="newline-free"):
        replace(artifact, mapper_id="dinkster.gguf.diffusion.v1\nforged")
