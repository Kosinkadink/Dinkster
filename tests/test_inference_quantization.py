"""Proving tests for torch-free checkpoint quantization classification.

Synthetic headers cover all three quantization spellings and every
documented refusal. Header-only tests against the installed Flux
checkpoints pin the artifact counts and prove that classification
never needs tensor payloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT8_E5M2,
    FLOAT8_E8M0,
    FLOAT16,
    FLOAT32,
    INT8,
    KNOWN_QUANT_FORMATS,
    SUPPORTED_QUANT_FORMATS,
    UINT8,
    UNSUPPORTED_QUANT_FORMATS,
    LayerQuant,
    QuantizationError,
    QuantSplit,
    TensorGeometry,
    load_safetensors_header,
    split_quantization,
)

FLUX_NEW_FP8 = Path("/home/kosin/ComfyUI/models/diffusion_models/flux1-dev-fp8-new.safetensors")
FLUX_LEGACY_FP8 = Path(
    "/home/kosin/ComfyUI/models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors"
)
FLUX_PLAIN_FP8 = Path("/home/kosin/ComfyUI/models/diffusion_models/flux1-dev-fp8.safetensors")


def geometry(shape: tuple[int, ...], dtype=FLOAT32) -> TensorGeometry:
    return TensorGeometry(shape, dtype)


def metadata(layers: object) -> dict[str, str]:
    return {"_quantization_metadata": json.dumps({"layers": layers})}


def new_header(*, input_scale: bool = True, weight_dtype=FLOAT8_E4M3) -> dict[str, TensorGeometry]:
    result = {
        "block.weight": geometry((64, 32), weight_dtype),
        "block.weight_scale": geometry((), FLOAT32),
    }
    if input_scale:
        result["block.input_scale"] = geometry((), FLOAT32)
    return result


def test_supported_formats_include_exact_metadata_nvfp4() -> None:
    assert SUPPORTED_QUANT_FORMATS == {
        "float8_e4m3fn",
        "float8_e5m2",
        "nvfp4",
        "int8_tensorwise",
    }


def test_known_formats_separate_typed_contracts_from_runtime_support() -> None:
    assert KNOWN_QUANT_FORMATS == {
        "float8_e4m3fn",
        "float8_e5m2",
        "nvfp4",
        "mxfp8",
        "int8_tensorwise",
        "convrot_w4a4",
        "asym_w4a8_int8",
    }
    assert UNSUPPORTED_QUANT_FORMATS == {
        "mxfp8",
        "convrot_w4a4",
        "asym_w4a8_int8",
    }


def test_layer_quant_preserves_legacy_positional_constructor() -> None:
    quant = LayerQuant(
        "block",
        None,
        "block.weight",
        "block.weight_scale",
        "block.input_scale",
        "block.comfy_quant",
        True,
    )
    assert quant.config == "block.comfy_quant"
    assert quant.full_precision_matmul is True
    assert quant.weight_scale_2 is None
    assert quant.pre_quant_scale is None


def test_no_artifacts_passes_through() -> None:
    geometries = {"block.weight": geometry((64, 32), FLOAT16)}
    split = split_quantization(geometries)
    assert isinstance(split, QuantSplit)
    assert split.layers == {}
    assert split.architecture == geometries


def test_metadata_classifies_layer_and_strips_scales() -> None:
    split = split_quantization(new_header(), metadata({"block": {"format": "float8_e4m3fn"}}))
    assert split.layers == {
        "block": LayerQuant(
            layer="block",
            format="float8_e4m3fn",
            weight="block.weight",
            weight_scale="block.weight_scale",
            input_scale="block.input_scale",
        )
    }
    assert split.architecture == {"block.weight": geometry((64, 32), FLOAT8_E4M3)}


def test_metadata_accepts_absent_input_scale() -> None:
    split = split_quantization(
        new_header(input_scale=False),
        metadata({"block": {"format": "float8_e4m3fn"}}),
    )
    assert split.layers["block"].input_scale is None


def test_metadata_sets_full_precision_matmul() -> None:
    split = split_quantization(
        new_header(),
        metadata(
            {
                "block": {
                    "format": "float8_e4m3fn",
                    "full_precision_matrix_mult": True,
                }
            }
        ),
    )
    assert split.layers["block"].full_precision_matmul


def test_fp8_metadata_keeps_existing_tolerant_config_semantics() -> None:
    split = split_quantization(
        new_header(),
        metadata(
            {
                "block": {
                    "format": "float8_e4m3fn",
                    "full_precision_matrix_mult": "enabled",
                    "future_parameter": 1,
                }
            }
        ),
    )
    assert split.layers["block"].full_precision_matmul


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not json", "malformed"),
        (json.dumps({"layers": []}), "layers must be an object"),
        (json.dumps({"layers": {"block": []}}), "config must be an object"),
    ],
)
def test_metadata_rejects_malformed_payload(payload: str, match: str) -> None:
    with pytest.raises(QuantizationError, match=match):
        split_quantization(new_header(), {"_quantization_metadata": payload})


def test_metadata_rejects_unknown_format() -> None:
    with pytest.raises(QuantizationError, match="unknown quantization format 'future_quant'"):
        split_quantization(new_header(), metadata({"block": {"format": "future_quant"}}))


def test_metadata_rejects_missing_weight() -> None:
    with pytest.raises(QuantizationError, match="block.weight"):
        split_quantization(
            {"block.weight_scale": geometry((), FLOAT32)},
            metadata({"block": {"format": "float8_e4m3fn"}}),
        )


def test_metadata_rejects_missing_weight_scale() -> None:
    with pytest.raises(QuantizationError, match="missing.*weight_scale"):
        split_quantization(
            {"block.weight": geometry((64, 32), FLOAT8_E4M3)},
            metadata({"block": {"format": "float8_e4m3fn"}}),
        )


@pytest.mark.parametrize(
    "bad_scale",
    [geometry((2,), FLOAT32), geometry((), FLOAT16)],
)
def test_metadata_rejects_invalid_weight_scale(bad_scale: TensorGeometry) -> None:
    geometries = new_header()
    geometries["block.weight_scale"] = bad_scale
    with pytest.raises(QuantizationError, match="single float32 scale"):
        split_quantization(geometries, metadata({"block": {"format": "float8_e4m3fn"}}))


def test_metadata_rejects_weight_dtype_mismatch() -> None:
    with pytest.raises(QuantizationError, match="stored dtype bfloat16.*does not match"):
        split_quantization(
            new_header(weight_dtype=BFLOAT16),
            metadata({"block": {"format": "float8_e4m3fn"}}),
        )


def test_metadata_rejects_orphan_scale() -> None:
    geometries = new_header()
    geometries["other.weight_scale"] = geometry((), FLOAT32)
    with pytest.raises(QuantizationError, match="other.weight_scale"):
        split_quantization(geometries, metadata({"block": {"format": "float8_e4m3fn"}}))


def test_comfy_quant_classifies_layer_and_optional_input_scale() -> None:
    geometries = new_header()
    geometries["block.comfy_quant"] = geometry((30,), UINT8)
    split = split_quantization(geometries)
    assert split.layers["block"] == LayerQuant(
        layer="block",
        format=None,
        weight="block.weight",
        weight_scale="block.weight_scale",
        input_scale="block.input_scale",
        config="block.comfy_quant",
    )
    assert set(split.architecture) == {"block.weight"}


def test_comfy_quant_rejects_missing_weight_scale() -> None:
    geometries = {
        "block.weight": geometry((64, 32), FLOAT8_E4M3),
        "block.comfy_quant": geometry((30,), UINT8),
    }
    with pytest.raises(QuantizationError, match="missing.*weight_scale"):
        split_quantization(geometries)


def test_comfy_quant_rejects_orphan_scale() -> None:
    geometries = new_header()
    geometries["block.comfy_quant"] = geometry((30,), UINT8)
    geometries["other.input_scale"] = geometry((), FLOAT32)
    with pytest.raises(QuantizationError, match="other.input_scale"):
        split_quantization(geometries)


def legacy_header(
    marker_dtype=FLOAT32,
    marker_shape: tuple[int, ...] = (0,),
    weight_dtype=FLOAT8_E4M3,
    *,
    input_scale: bool = True,
) -> dict[str, TensorGeometry]:
    result = {
        "scaled_fp8": geometry(marker_shape, marker_dtype),
        "block.weight": geometry((64, 32), weight_dtype),
        "block.scale_weight": geometry((), FLOAT32),
    }
    if input_scale:
        result["block.scale_input"] = geometry((), FLOAT32)
    return result


def test_legacy_float32_marker_classifies_e4m3_layer() -> None:
    split = split_quantization(legacy_header())
    assert split.layers["block"] == LayerQuant(
        layer="block",
        format="float8_e4m3fn",
        weight="block.weight",
        weight_scale="block.scale_weight",
        input_scale="block.scale_input",
    )
    assert set(split.architecture) == {"block.weight"}


def test_legacy_e5m2_marker_honors_marker_dtype() -> None:
    split = split_quantization(legacy_header(marker_dtype=FLOAT8_E5M2, weight_dtype=FLOAT8_E5M2))
    assert split.layers["block"].format == "float8_e5m2"


def test_legacy_two_element_marker_sets_full_precision_for_every_layer() -> None:
    geometries = legacy_header(marker_shape=(2,))
    geometries.update(
        {
            "other.weight": geometry((32, 32), FLOAT8_E4M3),
            "other.scale_weight": geometry((), FLOAT32),
        }
    )
    split = split_quantization(geometries)
    assert all(layer.full_precision_matmul for layer in split.layers.values())


def test_legacy_rejects_unported_marker_dtype() -> None:
    with pytest.raises(QuantizationError, match="marker dtype 'float16'"):
        split_quantization(legacy_header(marker_dtype=FLOAT16))


def test_legacy_rejects_orphan_input_scale() -> None:
    geometries = {
        "scaled_fp8": geometry((0,), FLOAT32),
        "block.weight": geometry((64, 32), FLOAT8_E4M3),
        "block.scale_input": geometry((), FLOAT32),
    }
    with pytest.raises(QuantizationError, match="scale_input"):
        split_quantization(geometries)


def test_legacy_rejects_weight_dtype_mismatch() -> None:
    with pytest.raises(QuantizationError, match="float8_e5m2.*does not match"):
        split_quantization(legacy_header(weight_dtype=FLOAT8_E5M2))


def test_metadata_precedence_ignores_lower_precedence_comfy_quant() -> None:
    geometries = new_header()
    geometries["block.comfy_quant"] = geometry((30,), UINT8)
    split = split_quantization(geometries, metadata({"block": {"format": "float8_e4m3fn"}}))
    assert split.layers["block"].format == "float8_e4m3fn"
    assert "block.comfy_quant" not in split.architecture


def nvfp4_header(*, pre_quant_scale: bool = False) -> dict[str, TensorGeometry]:
    result = {
        "block.weight": geometry((128, 32), UINT8),
        "block.weight_scale": geometry((128, 4), FLOAT8_E4M3),
        "block.weight_scale_2": geometry((), FLOAT32),
        "block.input_scale": geometry((), FLOAT32),
    }
    if pre_quant_scale:
        result["block.pre_quant_scale"] = geometry((64,), FLOAT16)
    return result


def test_nvfp4_metadata_recovers_logical_geometry_and_artifact_keys() -> None:
    split = split_quantization(
        nvfp4_header(pre_quant_scale=True),
        metadata({"block": {"format": "nvfp4"}}),
    )
    assert split.architecture == {"block.weight": geometry((128, 64), UINT8)}
    assert split.layers["block"] == LayerQuant(
        layer="block",
        format="nvfp4",
        weight="block.weight",
        weight_scale="block.weight_scale",
        input_scale="block.input_scale",
        weight_scale_2="block.weight_scale_2",
        pre_quant_scale="block.pre_quant_scale",
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"layers":{"block":{"format":"nvfp4"},"block":{"format":"nvfp4"}}}',
        '{"layers":{"block":{"format":"nvfp4","format":"float8_e4m3fn"}}}',
    ],
)
def test_nvfp4_metadata_rejects_duplicate_json_members(payload: str) -> None:
    with pytest.raises(QuantizationError, match="duplicate JSON object member"):
        split_quantization(
            nvfp4_header(),
            {"_quantization_metadata": payload},
        )


@pytest.mark.parametrize("value", ["true", 1, [], {}])
def test_nvfp4_metadata_requires_boolean_full_precision(value: object) -> None:
    with pytest.raises(
        QuantizationError,
        match="NVFP4 full_precision_matrix_mult must be a bool",
    ):
        split_quantization(
            nvfp4_header(),
            metadata(
                {
                    "block": {
                        "format": "nvfp4",
                        "full_precision_matrix_mult": value,
                    }
                }
            ),
        )


@pytest.mark.parametrize(
    ("key", "replacement", "match"),
    [
        ("block.weight", geometry((128, 32), FLOAT8_E4M3), "rank-2 uint8"),
        ("block.weight", geometry((128,), UINT8), "rank-2 uint8"),
        ("block.weight", geometry((127, 32), UINT8), "16-aligned"),
        ("block.weight_scale", geometry((128, 3), FLOAT8_E4M3), "shape"),
        ("block.weight_scale", geometry((128, 4), FLOAT16), "float8_e4m3fn"),
        ("block.weight_scale_2", geometry((2,), FLOAT32), "scalar float32"),
        ("block.weight_scale_2", geometry((1,), FLOAT32), "scalar float32"),
        ("block.input_scale", geometry((), FLOAT16), "scalar float32"),
    ],
)
def test_nvfp4_metadata_rejects_wrong_dtype_or_shape(
    key: str, replacement: TensorGeometry, match: str
) -> None:
    header = nvfp4_header()
    header[key] = replacement
    with pytest.raises(QuantizationError, match=match):
        split_quantization(header, metadata({"block": {"format": "nvfp4"}}))


@pytest.mark.parametrize(
    "missing",
    ["block.weight_scale", "block.weight_scale_2"],
)
def test_nvfp4_metadata_rejects_missing_artifact(missing: str) -> None:
    header = nvfp4_header()
    del header[missing]
    with pytest.raises(QuantizationError, match="missing"):
        split_quantization(header, metadata({"block": {"format": "nvfp4"}}))


def test_nvfp4_metadata_accepts_missing_optional_input_scale() -> None:
    header = nvfp4_header()
    del header["block.input_scale"]
    split = split_quantization(header, metadata({"block": {"format": "nvfp4"}}))
    assert split.layers["block"].input_scale is None


@pytest.mark.parametrize(
    "pre",
    [geometry((63,), FLOAT16), geometry((64,), UINT8), geometry((1, 64), FLOAT32)],
)
def test_nvfp4_metadata_rejects_bad_pre_quant_scale(pre: TensorGeometry) -> None:
    header = nvfp4_header()
    header["block.pre_quant_scale"] = pre
    with pytest.raises(QuantizationError, match="pre-quant scale"):
        split_quantization(header, metadata({"block": {"format": "nvfp4"}}))


def test_nvfp4_metadata_rejects_extra_artifact_and_parameter() -> None:
    header = nvfp4_header()
    header["block.scale_input"] = geometry((), FLOAT32)
    with pytest.raises(QuantizationError, match="scale_input"):
        split_quantization(header, metadata({"block": {"format": "nvfp4"}}))
    with pytest.raises(QuantizationError, match="unexpected NVFP4 metadata"):
        split_quantization(
            nvfp4_header(),
            metadata({"block": {"format": "nvfp4", "group_size": 16}}),
        )


def test_nvfp4_payload_only_comfy_quant_requires_payload_seam() -> None:
    header = nvfp4_header()
    header["block.comfy_quant"] = geometry((19,), UINT8)
    with pytest.raises(QuantizationError, match="payload is unavailable"):
        split_quantization(header)


def test_weight_scale_2_is_an_orphan_without_nvfp4_metadata() -> None:
    with pytest.raises(QuantizationError, match="weight_scale_2"):
        split_quantization({"block.weight_scale_2": geometry((), FLOAT32)})


def unsupported_header(quant_format: str) -> dict[str, TensorGeometry]:
    if quant_format == "mxfp8":
        return {
            "block.weight": geometry((128, 256), FLOAT8_E4M3),
            "block.weight_scale": geometry((128, 8), FLOAT8_E8M0),
        }
    if quant_format == "int8_tensorwise":
        return {
            "block.weight": geometry((128, 256), INT8),
            "block.weight_scale": geometry((), FLOAT32),
        }
    if quant_format == "convrot_w4a4":
        return {
            "block.weight": geometry((128, 128), INT8),
            "block.weight_scale": geometry((128,), FLOAT32),
        }
    if quant_format == "asym_w4a8_int8":
        return {
            "block.weight": geometry((128, 128), INT8),
            "block.weight_s_rel": geometry((128, 16), FLOAT8_E4M3),
            "block.weight_s_channel": geometry((128,), FLOAT32),
            "block.weight_correction": geometry((16, 128), BFLOAT16),
            "block.weight_codebook": geometry((16,), FLOAT32),
        }
    raise AssertionError(quant_format)


@pytest.mark.parametrize(
    ("quant_format", "parameters", "logical_shape", "payloads", "executable"),
    [
        ("mxfp8", {}, (128, 256), {"weight_scale"}, False),
        (
            "int8_tensorwise",
            {"convrot": False, "convrot_groupsize": 256},
            (128, 256),
            {"weight_scale"},
            True,
        ),
        (
            "convrot_w4a4",
            {"convrot_groupsize": 256, "linear_dtype": "int4"},
            (128, 256),
            {"weight_scale"},
            False,
        ),
        (
            "asym_w4a8_int8",
            {"group_size": 16, "convrot_groupsize": 256},
            (128, 256),
            {
                "weight_s_rel",
                "weight_s_channel",
                "weight_correction",
                "weight_codebook",
            },
            False,
        ),
    ],
)
def test_metadata_types_exact_unsupported_payload_contracts(
    quant_format: str,
    parameters: dict[str, object],
    logical_shape: tuple[int, int],
    payloads: set[str],
    executable: bool,
) -> None:
    split = split_quantization(
        unsupported_header(quant_format),
        metadata({"block": {"format": quant_format}}),
    )
    quant = split.layers["block"]
    assert quant.format == quant_format
    assert quant.parameters == parameters
    assert quant.logical_shape == logical_shape
    assert set(quant.payloads) == payloads
    assert quant.executable is executable
    assert split.architecture["block.weight"].shape == logical_shape
    if quant_format == "mxfp8":
        assert quant.input_scale is None


@pytest.mark.parametrize("quant_format", sorted(UNSUPPORTED_QUANT_FORMATS))
def test_payload_config_types_exact_unsupported_contracts(quant_format: str) -> None:
    header = unsupported_header(quant_format)
    payload = json.dumps({"format": quant_format}).encode()
    header["block.comfy_quant"] = geometry((len(payload),), UINT8)
    split = split_quantization(header, payload_reader=lambda _key: payload)
    assert split.layers["block"].format == quant_format
    assert split.layers["block"].config == "block.comfy_quant"
    assert split.layers["block"].executable is False


@pytest.mark.parametrize("source", ["metadata", "payload"])
def test_mxfp8_optional_input_scale_is_typed_and_consumed(source: str) -> None:
    header = unsupported_header("mxfp8")
    header["block.input_scale"] = geometry((), FLOAT32)
    if source == "metadata":
        split = split_quantization(
            header,
            metadata({"block": {"format": "mxfp8"}}),
        )
    else:
        payload = json.dumps({"format": "mxfp8"}).encode()
        header["block.comfy_quant"] = geometry((len(payload),), UINT8)
        split = split_quantization(header, payload_reader=lambda _key: payload)

    quant = split.layers["block"]
    assert quant.input_scale == "block.input_scale"
    assert quant.payloads["input_scale"] == "block.input_scale"
    assert "block.input_scale" not in split.architecture
    assert quant.executable is False


@pytest.mark.parametrize(
    "replacement",
    [
        geometry((), FLOAT16),
        geometry((1,), FLOAT32),
        geometry((2, 1), FLOAT32),
    ],
    ids=("wrong-dtype", "wrong-rank", "wrong-shape"),
)
def test_mxfp8_optional_input_scale_rejects_wrong_geometry(
    replacement: TensorGeometry,
) -> None:
    header = unsupported_header("mxfp8")
    header["block.input_scale"] = replacement
    with pytest.raises(
        QuantizationError,
        match=r"MXFP8 input scale must be float32 with shape \(\)",
    ):
        split_quantization(header, metadata({"block": {"format": "mxfp8"}}))


def test_unsupported_int8_config_without_payload_reader_refuses_provider_free() -> None:
    header = unsupported_header("int8_tensorwise")
    header["block.comfy_quant"] = geometry((32,), UINT8)
    with pytest.raises(QuantizationError, match="payload is unavailable.*not an unresolved FP8"):
        split_quantization(header)


def test_int8_tensorwise_convrot_variant_is_explicit_and_strict() -> None:
    header = unsupported_header("int8_tensorwise")
    header["block.weight_scale"] = geometry((128, 1), FLOAT32)
    split = split_quantization(
        header,
        metadata(
            {
                "block": {
                    "format": "int8_tensorwise",
                    "params": {"convrot": True, "convrot_groupsize": 256},
                }
            }
        ),
    )
    assert split.layers["block"].parameters == {
        "convrot": True,
        "convrot_groupsize": 256,
    }


def test_convrot_w4a4_accepts_proven_top_level_and_nested_spellings() -> None:
    top = split_quantization(
        unsupported_header("convrot_w4a4"),
        metadata(
            {
                "block": {
                    "format": "convrot_w4a4",
                    "convrot_groupsize": 256,
                    "linear_dtype": "int8",
                }
            }
        ),
    )
    nested = split_quantization(
        unsupported_header("convrot_w4a4"),
        metadata(
            {
                "block": {
                    "format": "convrot_w4a4",
                    "params": {"convrot_groupsize": 256, "linear_dtype": "int8"},
                }
            }
        ),
    )
    assert top.layers["block"].parameters == nested.layers["block"].parameters
    assert top.layers["block"].logical_shape == (128, 256)


def test_w4a8_uses_kitchen_payload_names_and_never_weight_scale() -> None:
    split = split_quantization(
        unsupported_header("asym_w4a8_int8"),
        metadata({"block": {"format": "asym_w4a8_int8"}}),
    )
    assert "weight_scale" not in split.layers["block"].payloads
    for required in ("block.weight_s_rel", "block.weight_s_channel"):
        bad = unsupported_header("asym_w4a8_int8")
        del bad[required]
        with pytest.raises(QuantizationError, match=f"missing '{required}'"):
            split_quantization(bad, metadata({"block": {"format": "asym_w4a8_int8"}}))
    wrong_name = unsupported_header("asym_w4a8_int8")
    wrong_name["block.weight_scale"] = geometry((128, 16), FLOAT8_E4M3)
    with pytest.raises(QuantizationError, match="weight_scale"):
        split_quantization(wrong_name, metadata({"block": {"format": "asym_w4a8_int8"}}))


def test_w4a8_accepts_both_optional_payloads_absent() -> None:
    header = unsupported_header("asym_w4a8_int8")
    del header["block.weight_correction"]
    del header["block.weight_codebook"]
    split = split_quantization(header, metadata({"block": {"format": "asym_w4a8_int8"}}))
    assert set(split.layers["block"].payloads) == {
        "weight_s_rel",
        "weight_s_channel",
    }


@pytest.mark.parametrize(
    ("quant_format", "key", "replacement", "match"),
    [
        ("mxfp8", "block.weight", geometry((127, 256), FLOAT8_E4M3), "32-padded"),
        ("mxfp8", "block.weight_scale", geometry((128, 7), FLOAT8_E8M0), "shape"),
        ("int8_tensorwise", "block.weight", geometry((128, 256), UINT8), "rank-2 int8"),
        ("convrot_w4a4", "block.weight_scale", geometry((128, 1), FLOAT32), "shape"),
        ("asym_w4a8_int8", "block.weight_s_rel", geometry((128, 15), FLOAT32), "shape"),
        ("asym_w4a8_int8", "block.weight_s_channel", geometry((1,), FLOAT32), "shape"),
        ("asym_w4a8_int8", "block.weight_correction", geometry((16, 128), UINT8), "floating"),
        ("asym_w4a8_int8", "block.weight_codebook", geometry((15,), FLOAT32), "shape"),
    ],
)
def test_unsupported_contracts_reject_wrong_dtype_shape_and_padding(
    quant_format: str,
    key: str,
    replacement: TensorGeometry,
    match: str,
) -> None:
    header = unsupported_header(quant_format)
    header[key] = replacement
    with pytest.raises(QuantizationError, match=match):
        split_quantization(header, metadata({"block": {"format": quant_format}}))


@pytest.mark.parametrize(
    ("quant_format", "config", "match"),
    [
        ("mxfp8", {"future": 1}, "unexpected"),
        ("int8_tensorwise", {"convrot": "true"}, "convrot must be a bool"),
        ("int8_tensorwise", {"convrot": True, "convrot_groupsize": 1024}, "divisible"),
        ("convrot_w4a4", {"convrot_groupsize": 1024}, "divisible"),
        ("convrot_w4a4", {"linear_dtype": "float"}, "linear_dtype"),
        ("asym_w4a8_int8", {"group_size": float("nan")}, "group_size"),
        ("asym_w4a8_int8", {"group_size": 12}, "group_size"),
        ("asym_w4a8_int8", {"convrot_groupsize": 1024}, "inconsistent"),
    ],
)
def test_unsupported_contracts_reject_malformed_group_and_nonfinite_config(
    quant_format: str, config: dict[str, object], match: str
) -> None:
    with pytest.raises(QuantizationError, match=match):
        split_quantization(
            unsupported_header(quant_format),
            metadata({"block": {"format": quant_format, **config}}),
        )


def test_unsupported_contract_rejects_conflicting_parameter_spellings() -> None:
    with pytest.raises(QuantizationError, match="contradictory.*group_size"):
        split_quantization(
            unsupported_header("asym_w4a8_int8"),
            metadata(
                {
                    "block": {
                        "format": "asym_w4a8_int8",
                        "group_size": 16,
                        "params": {"group_size": 32},
                    }
                }
            ),
        )


def test_quant_payload_mappings_are_immutable_copies() -> None:
    payloads = {"weight_s_rel": "block.weight_s_rel"}
    quant = LayerQuant(
        layer="block",
        format="asym_w4a8_int8",
        weight="block.weight",
        weight_scale="",
        payloads=payloads,
        parameters={"group_size": 16, "convrot_groupsize": 256},
        logical_shape=(128, 256),
    )
    payloads["weight_s_channel"] = "block.weight_s_channel"
    assert dict(quant.payloads) == {"weight_s_rel": "block.weight_s_rel"}
    with pytest.raises(TypeError):
        quant.payloads["x"] = "y"  # type: ignore[index]


def prefixed(sd: dict[str, TensorGeometry], prefix: str) -> dict[str, TensorGeometry]:
    return {prefix + key: value for key, value in sd.items()}


def test_prefix_scopes_legacy_marker_in_combined_header() -> None:
    geometries = prefixed(legacy_header(), "model.diffusion_model.")
    geometries["vae.decoder.weight"] = geometry((4, 4))
    split = split_quantization(geometries, prefix="model.diffusion_model.")
    assert set(split.layers) == {"block"}
    layer = split.layers["block"]
    assert layer.weight == "block.weight"
    assert layer.weight_scale == "block.scale_weight"
    assert set(split.architecture) == {"block.weight"}


def test_prefix_scopes_metadata_layer_names() -> None:
    geometries = prefixed(new_header(), "model.diffusion_model.")
    geometries["vae.decoder.weight"] = geometry((4, 4))
    split = split_quantization(
        geometries,
        metadata({"model.diffusion_model.block": {"format": "float8_e4m3fn"}}),
        prefix="model.diffusion_model.",
    )
    assert set(split.layers) == {"block"}
    assert split.layers["block"].weight_scale == "block.weight_scale"


def test_prefix_with_out_of_scope_metadata_falls_through_to_legacy() -> None:
    # A combined file whose metadata names only DiT layers must not
    # block a text encoder's own legacy marker (comfy/sd.py converts
    # text-encoder scaled_fp8 per prefix with empty metadata).
    geometries = prefixed(legacy_header(), "text_encoders.t5xxl.transformer.")
    geometries.update(prefixed(new_header(), "model.diffusion_model."))
    dit_metadata = metadata({"model.diffusion_model.block": {"format": "float8_e4m3fn"}})
    split = split_quantization(geometries, dit_metadata, prefix="text_encoders.t5xxl.transformer.")
    assert set(split.layers) == {"block"}
    assert split.layers["block"].weight_scale == "block.scale_weight"


def test_prefix_scoping_drops_out_of_scope_keys_from_architecture() -> None:
    geometries = {
        "vae.decoder.weight": geometry((4, 4)),
        "model.diffusion_model.img_in.weight": geometry((64, 32)),
    }
    split = split_quantization(geometries, prefix="model.diffusion_model.")
    assert set(split.architecture) == {"img_in.weight"}


def test_artifact_keys_without_a_spelling_refuse() -> None:
    geometries = {
        "block.weight": geometry((64, 32), FLOAT8_E4M3),
        "block.weight_scale": geometry((), FLOAT32),
    }
    with pytest.raises(QuantizationError, match="no recognized spelling"):
        split_quantization(geometries)


@pytest.mark.skipif(not FLUX_NEW_FP8.exists(), reason="Flux new-fp8 checkpoint absent")
def test_real_flux_new_fp8_header() -> None:
    source = load_safetensors_header(FLUX_NEW_FP8)
    geometries = {key: source.entry(key).geometry for key in source.keys()}
    split = split_quantization(geometries, source.metadata())
    assert len(split.layers) == 266
    assert all(layer.format == "float8_e4m3fn" for layer in split.layers.values())
    assert all(layer.input_scale is not None for layer in split.layers.values())
    assert not any(key.endswith((".weight_scale", ".input_scale")) for key in split.architecture)
    assert (
        sum(
            geometry.dtype == FLOAT8_E4M3 and key.endswith(".weight")
            for key, geometry in split.architecture.items()
        )
        >= 266
    )


@pytest.mark.skipif(not FLUX_LEGACY_FP8.exists(), reason="Flux legacy-fp8 checkpoint absent")
def test_real_flux_legacy_fp8_header() -> None:
    source = load_safetensors_header(FLUX_LEGACY_FP8)
    geometries = {key: source.entry(key).geometry for key in source.keys()}
    split = split_quantization(geometries, source.metadata())
    assert len(split.layers) == 314
    assert all(layer.format == "float8_e4m3fn" for layer in split.layers.values())
    assert all(layer.input_scale is not None for layer in split.layers.values())
    assert not any(layer.full_precision_matmul for layer in split.layers.values())
    assert "scaled_fp8" not in split.architecture


@pytest.mark.skipif(not FLUX_PLAIN_FP8.exists(), reason="Flux plain-fp8 checkpoint absent")
def test_real_flux_plain_fp8_header_passes_through() -> None:
    source = load_safetensors_header(FLUX_PLAIN_FP8)
    geometries = {key: source.entry(key).geometry for key in source.keys()}
    split = split_quantization(geometries, source.metadata())
    assert split.layers == {}
    assert len(split.architecture) == len(source.keys())
