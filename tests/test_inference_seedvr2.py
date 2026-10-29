"""Torch-free checks for exact SeedVR2 model descriptions."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT16,
    FLOAT32,
    INT8,
    SEEDVR2,
    SEEDVR2_3B,
    SEEDVR2_7B,
    SEEDVR2_7B_MLP,
    SEEDVR2_CONFIGS,
    SEEDVR2_VAE_CONFIG,
    SEEDVR2_VAE_DETECTOR_KEY,
    UINT8,
    Parameterization,
    SeedVR2DetectError,
    SeedVR2VAEHeaderError,
    TensorGeometry,
    WeightEntry,
    WeightSource,
    builtin_families,
    detect_seedvr2,
    detect_seedvr2_config,
    detect_seedvr2_vae_config,
    plan_seedvr2_component,
    seedvr2_layout,
    seedvr2_vae_layout,
)


class HeaderSource:
    path = Path("seedvr2.safetensors")

    def __init__(self, shapes: Mapping[str, tuple[int, ...]], dtype: object = BFLOAT16) -> None:
        self.shapes = dict(shapes)
        self.dtype = dtype

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.shapes[key], self.dtype)  # type: ignore[arg-type]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        raise AssertionError("detection must not read metadata")


def geometries(layout: Mapping[str, tuple[int, ...]]) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, FLOAT32) for key, shape in layout.items()}


class QuantizedHeaderSource:
    path = Path("seedvr2-quantized.safetensors")

    def __init__(
        self,
        entries: Mapping[str, TensorGeometry],
        metadata: Mapping[str, str],
    ) -> None:
        self.entries = dict(entries)
        self.extra = dict(metadata)

    def keys(self) -> Sequence[str]:
        return tuple(self.entries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.entries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return self.extra


def quantized_source(quant_format: str) -> QuantizedHeaderSource:
    entries = geometries(seedvr2_layout(SEEDVR2_3B))
    layer = "blocks.0.attn.proj_qkv.vid"
    rows, columns = entries[f"{layer}.weight"].shape
    if quant_format == "int8_tensorwise":
        entries[f"{layer}.weight"] = TensorGeometry((rows, columns), INT8)
        entries[f"{layer}.weight_scale"] = TensorGeometry((rows, 1), FLOAT32)
        config: dict[str, object] = {
            "format": quant_format,
            "params": {"convrot": True, "convrot_groupsize": 256},
        }
    elif quant_format == "nvfp4":
        entries[f"{layer}.weight"] = TensorGeometry((rows, columns // 2), UINT8)
        entries[f"{layer}.weight_scale"] = TensorGeometry(
            (((rows + 127) // 128) * 128, ((columns // 16 + 3) // 4) * 4),
            FLOAT8_E4M3,
        )
        entries[f"{layer}.weight_scale_2"] = TensorGeometry((), FLOAT32)
        config = {"format": quant_format}
    else:
        raise AssertionError(quant_format)
    return QuantizedHeaderSource(
        entries,
        {"_quantization_metadata": json.dumps({"layers": {layer: config}})},
    )


@pytest.mark.parametrize(
    ("config", "count", "signature"),
    (
        (SEEDVR2_3B, 637, "blocks.31.mlp.all.proj_in_gate.weight"),
        (SEEDVR2_7B, 694, "blocks.35.mlp.all.proj_in_gate.weight"),
        (SEEDVR2_7B_MLP, 1130, "blocks.35.mlp.vid.proj_out.weight"),
    ),
)
def test_diffusion_layouts_and_detection(config: object, count: int, signature: str) -> None:
    assert config in SEEDVR2_CONFIGS
    layout = seedvr2_layout(config)  # type: ignore[arg-type]
    assert len(layout) == count
    assert signature in layout
    assert detect_seedvr2_config(geometries(layout)) is config
    with pytest.raises(TypeError):
        layout["foreign"] = (1,)  # type: ignore[index]


def test_variants_are_disambiguated_and_fail_closed() -> None:
    layout = dict(seedvr2_layout(SEEDVR2_7B))
    del layout["blocks.0.attn.proj_qkv.vid.weight"]
    with pytest.raises(SeedVR2DetectError, match="missing"):
        detect_seedvr2_config(geometries(layout))
    layout = dict(seedvr2_layout(SEEDVR2_3B))
    layout["foreign.weight"] = (1,)
    with pytest.raises(SeedVR2DetectError, match="unexpected"):
        detect_seedvr2_config(geometries(layout))
    layout = dict(seedvr2_layout(SEEDVR2_7B_MLP))
    layout["vid_in.proj.weight"] = (1, 132)
    with pytest.raises(SeedVR2DetectError, match="expected shape"):
        detect_seedvr2_config(geometries(layout))


@pytest.mark.parametrize(
    ("quant_format", "storage_dtype"),
    (("int8_tensorwise", INT8), ("nvfp4", UINT8)),
)
def test_diffusion_planner_accepts_generic_executable_quantized_weights(
    quant_format: str, storage_dtype: object
) -> None:
    plan = plan_seedvr2_component(
        cast("WeightSource", quantized_source(quant_format)),
        "diffusion",
    )
    layer = "blocks.0.attn.proj_qkv.vid"
    assert plan.config is SEEDVR2_3B
    assert plan.quant[layer].format == quant_format
    assert plan.dtypes[f"{layer}.weight"] is storage_dtype


def test_diffusion_planner_rejects_unquantized_integer_storage() -> None:
    source = quantized_source("int8_tensorwise")
    source.entries["vid_in.proj.bias"] = TensorGeometry((2560,), INT8)
    with pytest.raises(ValueError, match="require floating-point storage"):
        plan_seedvr2_component(cast("WeightSource", source), "diffusion")


def test_seedvr2_planner_accepts_float16_storage() -> None:
    diffusion_source = QuantizedHeaderSource(
        {key: TensorGeometry(shape, FLOAT16) for key, shape in seedvr2_layout(SEEDVR2_3B).items()},
        {},
    )
    vae_source = QuantizedHeaderSource(
        {key: TensorGeometry(shape, FLOAT16) for key, shape in seedvr2_vae_layout().items()},
        {},
    )
    diffusion = plan_seedvr2_component(cast("WeightSource", diffusion_source), "diffusion")
    vae = plan_seedvr2_component(cast("WeightSource", vae_source), "vae")
    assert set(diffusion.dtypes.values()) == {FLOAT16}
    assert set(vae.dtypes.values()) == {FLOAT16}


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_family_evidence_supports_bare_and_prefixed_files(prefix: str) -> None:
    source = HeaderSource(
        {prefix + key: shape for key, shape in seedvr2_layout(SEEDVR2_3B).items()}
    )
    evidence = detect_seedvr2(source)
    assert evidence is not None
    assert evidence.key_prefix == prefix
    assert evidence.config is SEEDVR2_3B


def test_vae_layout_and_detector_are_exact() -> None:
    layout = seedvr2_vae_layout()
    assert len(layout) == 250
    assert layout[SEEDVR2_VAE_DETECTOR_KEY] == (1024, 256, 1, 1, 1)
    assert detect_seedvr2_vae_config(geometries(layout)) is SEEDVR2_VAE_CONFIG
    damaged = geometries(layout)
    del damaged[SEEDVR2_VAE_DETECTOR_KEY]
    with pytest.raises(SeedVR2VAEHeaderError, match="detector key"):
        detect_seedvr2_vae_config(damaged)


def test_catalog_semantics() -> None:
    assert SEEDVR2 in builtin_families()
    latent = SEEDVR2.single_stream_latent()
    assert latent.channels == 16
    assert latent.dimensions == 3
    assert latent.temporal_causal
    assert latent.temporal_downscale == 4
    assert SEEDVR2.sampling.parameterization is Parameterization.FLOW
    assert SEEDVR2.sampling.shift == 1.0
    assert SEEDVR2.supported_dtypes == frozenset({BFLOAT16, FLOAT16, FLOAT32})
