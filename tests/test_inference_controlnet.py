"""Torch-free SD1.5 ControlNet layout, planning, and carrier proofs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from types import MappingProxyType

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    UINT8,
    ControlNetDetectError,
    ControlNetLayout,
    DependencyRef,
    PayloadReference,
    PercentRange,
    ReconstructionRecipe,
    RuntimeKnobs,
    SD15ControlNetConfig,
    SD15T2IAdapterConfig,
    SDControlMode,
    SDXLControlLoRAConfig,
    SDXLControlNetConfig,
    SDXLControlNetUnionConfig,
    T2IAdapterDetectError,
    TensorGeometry,
    WeightEntry,
    WeightSourceBinding,
    WeightSourceRef,
    default_diffusion_dtype,
    detect_sd15_controlnet,
    detect_sd15_t2i_adapter,
    detect_sdxl_control_lora,
    plan_sd15_controlnet,
    plan_sd15_t2i_adapter,
    plan_sdxl_control_lora,
    plan_sdxl_controlnet,
    plan_sdxl_controlnet_union,
    runtime_component_identity,
    sd15_controlnet_layout,
    sd15_t2i_adapter_layout,
    sdxl_control_lora_layout,
    sdxl_controlnet_layout,
    sdxl_controlnet_union_layout,
)
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.controlnet import ControlApplication, _sdxl_controlnet_diffusers_source_key

ASSET_DIGEST = "blake3:" + "a" * 64


@dataclass
class HeaderSource:
    path: Path
    geometries: dict[str, TensorGeometry]
    payload_reads: int = 0

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return {}

    def read_float_scalar(self, key: str) -> float:
        del key
        self.payload_reads += 1
        raise AssertionError("ControlNet planning must not read payloads")


@dataclass
class DuplicateKeySource(HeaderSource):
    def keys(self) -> tuple[str, ...]:
        keys = super().keys()
        return (*keys, keys[0])


def canonical_geometries() -> dict[str, TensorGeometry]:
    return {
        key: TensorGeometry(shape, FLOAT16)
        for key, shape in sd15_controlnet_layout(SD15ControlNetConfig()).keys.items()
    }


def adapter_geometries() -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, FLOAT32) for key, shape in sd15_t2i_adapter_layout().items()}


def control_lora_geometries(rank: int = 128) -> dict[str, TensorGeometry]:
    return {
        key: TensorGeometry(shape, FLOAT32 if key == "lora_controlnet" else FLOAT16)
        for key, shape in sdxl_control_lora_layout(SDXLControlLoRAConfig(rank=rank)).items()
    }


def controlnet_union_geometries(
    capacity: int = 8,
) -> dict[str, TensorGeometry]:
    config = SDXLControlNetUnionConfig(mode_capacity=capacity)  # type: ignore[arg-type]
    return {
        _sdxl_controlnet_diffusers_source_key(key): TensorGeometry(shape, FLOAT16)
        for key, shape in sdxl_controlnet_union_layout(config).items()
    }


def classic_sdxl_controlnet_geometries(
    layout: str = "diffusers",
) -> dict[str, TensorGeometry]:
    config = SDXLControlNetConfig()
    geometries: dict[str, TensorGeometry] = {}
    for key, shape in sdxl_controlnet_layout(config).items():
        source = _sdxl_controlnet_diffusers_source_key(key) if layout == "diffusers" else key
        if layout == "prefixed":
            source = "control_model." + source
        geometries[source] = TensorGeometry(shape, FLOAT16)
    return geometries


@pytest.mark.parametrize("layout", ["native", "prefixed", "diffusers"])
def test_classic_sdxl_controlnet_plans_all_comfyui_layouts_without_payload_reads(
    layout: str,
) -> None:
    geometries = classic_sdxl_controlnet_geometries(layout)
    source = HeaderSource(Path(f"sdxl-controlnet-{layout}.safetensors"), geometries)

    plan = plan_sdxl_controlnet(source, asset_digest=ASSET_DIGEST)

    assert len(geometries) == 844
    assert plan.family_id == "dinkster.sdxl_controlnet"
    assert plan.controlnet.config == SDXLControlNetConfig()
    assert plan.claims == tuple(sorted(geometries))
    assert set(plan.controlnet.keys.values()) == set(geometries)
    assert source.payload_reads == 0


def test_classic_sdxl_controlnet_plans_float64_storage_for_runtime_cast() -> None:
    geometries = classic_sdxl_controlnet_geometries()
    source_key = "controlnet_mid_block.weight"
    geometries[source_key] = replace(geometries[source_key], dtype=FLOAT64)

    plan = plan_sdxl_controlnet(
        HeaderSource(Path("sdxl-controlnet.safetensors"), geometries),
        asset_digest=ASSET_DIGEST,
    )

    assert plan.controlnet.dtypes["middle_block_out.0.weight"] is FLOAT64


def test_classic_sdxl_controlnet_nonfloating_storage_names_the_weight_requirement() -> None:
    geometries = classic_sdxl_controlnet_geometries()
    source_key = "controlnet_mid_block.weight"
    geometries[source_key] = replace(geometries[source_key], dtype=UINT8)

    with pytest.raises(
        ControlNetDetectError,
        match=(
            "SDXL ControlNet weight controlnet_mid_block.weight requires floating storage,"
            " got uint8"
        ),
    ):
        plan_sdxl_controlnet(
            HeaderSource(Path("sdxl-controlnet.safetensors"), geometries),
            asset_digest=ASSET_DIGEST,
        )


@pytest.mark.parametrize(
    "mutation,match", [("missing", "missing required"), ("foreign", "leftover")]
)
def test_classic_sdxl_controlnet_refuses_incomplete_or_foreign_layouts(
    mutation: str, match: str
) -> None:
    geometries = classic_sdxl_controlnet_geometries()
    if mutation == "missing":
        del geometries["controlnet_mid_block.bias"]
    else:
        geometries["foreign.weight"] = TensorGeometry((1,), FLOAT16)
    with pytest.raises(ControlNetDetectError, match=match):
        plan_sdxl_controlnet(
            HeaderSource(Path("sdxl-controlnet.safetensors"), geometries),
            asset_digest=ASSET_DIGEST,
        )


def test_sdxl_controlnet_union_layout_plan_and_modes_are_exact_and_torch_free() -> None:
    geometries = controlnet_union_geometries()
    source = HeaderSource(Path("diffusion_pytorch_model_promax.safetensors"), geometries)
    plan = plan_sdxl_controlnet_union(source, asset_digest=ASSET_DIGEST)

    assert len(geometries) == 863
    assert plan.family_id == "dinkster.sdxl_controlnet_union"
    assert plan.controlnet_union.config.mode_capacity == 8
    assert plan.claims == tuple(sorted(geometries))
    assert set(plan.controlnet_union.keys.values()) == set(geometries)
    assert source.payload_reads == 0
    assert SDControlMode("sdxl-controlnet-union", "hed") != SDControlMode(
        "sdxl-controlnet-union", "pidi"
    )
    with pytest.raises(ValueError, match="unknown SD control mode token"):
        SDControlMode("sdxl-controlnet-union", "unknown")  # type: ignore[arg-type]


def test_sdxl_controlnet_union_plans_float64_storage_for_runtime_cast() -> None:
    geometries = controlnet_union_geometries()
    source_key = "controlnet_mid_block.weight"
    geometries[source_key] = replace(geometries[source_key], dtype=FLOAT64)

    plan = plan_sdxl_controlnet_union(
        HeaderSource(Path("union.safetensors"), geometries), asset_digest=ASSET_DIGEST
    )

    assert plan.controlnet_union.dtypes["middle_block_out.0.weight"] is FLOAT64


def test_sdxl_controlnet_union_nonfloating_storage_names_the_weight_requirement() -> None:
    geometries = controlnet_union_geometries()
    source_key = "controlnet_mid_block.weight"
    geometries[source_key] = replace(geometries[source_key], dtype=UINT8)

    with pytest.raises(
        ControlNetDetectError,
        match=(
            "SDXL ControlNet Union weight controlnet_mid_block.weight requires floating storage,"
            " got uint8"
        ),
    ):
        plan_sdxl_controlnet_union(
            HeaderSource(Path("union.safetensors"), geometries), asset_digest=ASSET_DIGEST
        )


@pytest.mark.parametrize("capacity", [6, 8])
def test_sdxl_controlnet_union_accepts_only_detected_capacities(capacity: int) -> None:
    geometries = controlnet_union_geometries(capacity)
    plan = plan_sdxl_controlnet_union(
        HeaderSource(Path("union.safetensors"), geometries), asset_digest=ASSET_DIGEST
    )
    assert plan.controlnet_union.config.mode_capacity == capacity


def test_sdxl_controlnet_union_refuses_header_drift() -> None:
    geometries = controlnet_union_geometries()
    del geometries["controlnet_mid_block.bias"]
    with pytest.raises(ControlNetDetectError, match="missing required Union keys"):
        plan_sdxl_controlnet_union(
            HeaderSource(Path("union.safetensors"), geometries), asset_digest=ASSET_DIGEST
        )


@pytest.mark.parametrize("rank", [128, 256])
def test_sdxl_control_lora_layout_and_plan_are_exact_and_torch_free(rank: int) -> None:
    geometries = control_lora_geometries(rank)
    assert len(geometries) == 1228
    assert detect_sdxl_control_lora(geometries) == SDXLControlLoRAConfig(rank=rank)
    assert geometries["input_blocks.0.0.down"].shape == (4, 4, 3, 3)
    assert geometries["input_blocks.0.0.up"].shape == (320, 4, 1, 1)
    assert geometries["lora_controlnet"].shape == (0,)

    source = HeaderSource(Path(f"control-lora-canny-rank{rank}.safetensors"), geometries)
    plan = plan_sdxl_control_lora(
        source,
        asset_digest=ASSET_DIGEST,
        base_asset_digest="blake3:" + "b" * 64,
    )
    assert plan.family_id == "dinkster.sdxl_control_lora"
    assert plan.claims == tuple(sorted(geometries))
    assert plan.control_lora.keys == {key: key for key in geometries}
    assert source.payload_reads == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "missing required"),
        ("foreign", "leftover or foreign"),
        ("shape", "geometry mismatch"),
        ("dtype", "geometry mismatch"),
    ],
)
def test_sdxl_control_lora_refuses_header_drift(mutation: str, match: str) -> None:
    geometries = control_lora_geometries()
    if mutation == "missing":
        del geometries["middle_block_out.0.bias"]
    elif mutation == "foreign":
        geometries["foreign.weight"] = TensorGeometry((1,), FLOAT16)
    elif mutation == "shape":
        geometries["zero_convs.0.0.weight"] = TensorGeometry((321, 320, 1, 1), FLOAT16)
    else:
        geometries["zero_convs.0.0.weight"] = TensorGeometry((320, 320, 1, 1), FLOAT32)
    with pytest.raises(ControlNetDetectError, match=match):
        detect_sdxl_control_lora(geometries)


def test_sd15_full_adapter_layout_and_plan_are_exact_and_torch_free() -> None:
    geometries = adapter_geometries()
    assert len(geometries) == 38
    assert detect_sd15_t2i_adapter(geometries) == SD15T2IAdapterConfig()
    source = HeaderSource(Path("t2iadapter_canny_sd15v2.pth"), geometries)
    plan = plan_sd15_t2i_adapter(source, asset_digest=ASSET_DIGEST)
    assert plan.family_id == "dinkster.sd15_t2i_adapter"
    assert plan.claims == tuple(sorted(geometries))
    assert plan.adapter.keys == {key: key for key in geometries}
    assert source.payload_reads == 0
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(plan, claims=(plan.claims[0], plan.claims[0]))
    colliding_keys = dict(plan.adapter.keys)
    colliding_keys[plan.claims[1]] = plan.claims[0]
    with pytest.raises(ValueError, match="one-to-one"):
        replace(plan, adapter=replace(plan.adapter, keys=colliding_keys))

    malformed = dict(geometries)
    malformed["body.0.block2.weight"] = TensorGeometry((320, 320, 3, 3), FLOAT32)
    with pytest.raises(ValueError, match="geometry mismatch"):
        detect_sd15_t2i_adapter(malformed)


def test_sd15_adapter_accepts_generic_floating_storage_and_declares_compute_dtypes() -> None:
    geometries = {
        key: TensorGeometry(geometry.shape, FLOAT64)
        for key, geometry in adapter_geometries().items()
    }
    plan = plan_sd15_t2i_adapter(
        HeaderSource(Path("float64-adapter.safetensors"), geometries),
        asset_digest=ASSET_DIGEST,
    )

    assert set(plan.adapter.dtypes.values()) == {FLOAT64}
    descriptor = default_component_registry().get("dinkster.sd15_t2i_adapter")
    assert descriptor is not None
    assert descriptor.family.supported_dtypes == frozenset({FLOAT16, BFLOAT16, FLOAT32})
    assert default_diffusion_dtype("dinkster.sd15_t2i_adapter") is FLOAT16


def test_sd15_adapter_refuses_complete_integer_storage_layout() -> None:
    geometries = {
        key: TensorGeometry(geometry.shape, UINT8) for key, geometry in adapter_geometries().items()
    }

    with pytest.raises(T2IAdapterDetectError, match="storage must be floating"):
        detect_sd15_t2i_adapter(geometries)


def _diffusers_key(canonical: str) -> str:
    basics = {
        "time_embed.0": "time_embedding.linear_1",
        "time_embed.2": "time_embedding.linear_2",
        "input_blocks.0.0": "conv_in",
        "middle_block_out.0": "controlnet_mid_block",
    }
    for old, new in basics.items():
        if canonical.startswith(old + "."):
            return new + canonical[len(old) :]
    if canonical.startswith("input_hint_block."):
        index, parameter = canonical.removeprefix("input_hint_block.").split(".", 1)
        layer = int(index)
        if layer == 0:
            return f"controlnet_cond_embedding.conv_in.{parameter}"
        if layer == 14:
            return f"controlnet_cond_embedding.conv_out.{parameter}"
        return f"controlnet_cond_embedding.blocks.{layer // 2 - 1}.{parameter}"
    if canonical.startswith("zero_convs."):
        index, _, parameter = canonical.removeprefix("zero_convs.").split(".", 2)
        return f"controlnet_down_blocks.{index}.{parameter}"

    block_map = {
        1: (0, 0),
        2: (0, 1),
        4: (1, 0),
        5: (1, 1),
        7: (2, 0),
        8: (2, 1),
        10: (3, 0),
        11: (3, 1),
    }
    resnet_members = {
        "in_layers.0": "norm1",
        "in_layers.2": "conv1",
        "emb_layers.1": "time_emb_proj",
        "out_layers.0": "norm2",
        "out_layers.3": "conv2",
        "skip_connection": "conv_shortcut",
    }
    if canonical.startswith("input_blocks."):
        parts = canonical.split(".")
        block = int(parts[1])
        rest = ".".join(parts[3:])
        if block in (3, 6, 9):
            return f"down_blocks.{block // 3 - 1}.downsamplers.0.conv.{rest.removeprefix('op.')}"
        level, resnet = block_map[block]
        if parts[2] == "1":
            return f"down_blocks.{level}.attentions.{resnet}.{rest}"
        for old, new in resnet_members.items():
            if rest.startswith(old + "."):
                return f"down_blocks.{level}.resnets.{resnet}.{new}{rest[len(old) :]}"
    if canonical.startswith("middle_block."):
        parts = canonical.split(".")
        block = int(parts[1])
        rest = ".".join(parts[2:])
        if block == 1:
            return f"mid_block.attentions.0.{rest}"
        resnet = 0 if block == 0 else 1
        for old, new in resnet_members.items():
            if rest.startswith(old + "."):
                return f"mid_block.resnets.{resnet}.{new}{rest[len(old) :]}"
    raise AssertionError(f"test map does not cover {canonical}")


def diffusers_geometries() -> dict[str, TensorGeometry]:
    return {_diffusers_key(key): geometry for key, geometry in canonical_geometries().items()}


def source_ref(digit: str) -> WeightSourceRef:
    return WeightSourceRef(
        digest="blake3:" + digit * 64,
        name=f"{digit}.safetensors",
        size=123,
    )


def recipe(
    family: str,
    role: str,
    digit: str,
    dependencies: tuple[DependencyRef[ReconstructionRecipe], ...] = (),
) -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(WeightSourceBinding(role, source_ref(digit)),),
        family_id=family,
        component_identity=(f"family={family}", f"source-role={role}"),
        knobs=RuntimeKnobs("float16", "float32", "float32", False),
        dependencies=dependencies,
    )


def test_canonical_layout_golden_and_config_are_exact() -> None:
    config = SD15ControlNetConfig()
    layout = sd15_controlnet_layout(config)
    assert config == SD15ControlNetConfig(
        in_channels=4,
        model_channels=320,
        hint_channels=3,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=(2, 2, 2, 2),
        transformer_depth=(1, 1, 1, 1, 1, 1, 0, 0),
        transformer_depth_middle=1,
        context_dim=768,
        num_heads=8,
    )
    assert layout.zero_conv_channels == (
        320,
        320,
        320,
        320,
        640,
        640,
        640,
        1280,
        1280,
        1280,
        1280,
        1280,
    )
    assert layout.keys["input_hint_block.0.weight"] == (16, 3, 3, 3)
    assert layout.keys["input_hint_block.14.weight"] == (320, 256, 3, 3)
    assert layout.keys["zero_convs.11.0.weight"] == (1280, 1280, 1, 1)
    assert layout.keys["middle_block_out.0.weight"] == (1280, 1280, 1, 1)
    assert len(layout.keys) == 340
    assert detect_sd15_controlnet(canonical_geometries()) == config
    assert isinstance(layout.keys, MappingProxyType)
    with pytest.raises(TypeError):
        layout.keys["extra"] = (1,)  # type: ignore[index]
    altered = dict(layout.keys)
    altered["input_hint_block.0.weight"] = (17, 3, 3, 3)
    with pytest.raises(ValueError, match="exact classic"):
        ControlNetLayout(config, altered, layout.zero_conv_channels)
    with pytest.raises(ValueError, match="exact classic"):
        ControlNetLayout(config, layout.keys, (1,) * 12)


def test_diffusers_layout_maps_and_claims_every_source_key_once() -> None:
    geometries = diffusers_geometries()
    source = HeaderSource(Path("controlnet.safetensors"), geometries)
    plan = plan_sd15_controlnet(source, asset_digest=ASSET_DIGEST)
    assert plan.family_id == "dinkster.sd15_controlnet"
    assert plan.source_role == "controlnet"
    assert plan.source_layout == "diffusers"
    assert set(plan.controlnet.keys) == set(canonical_geometries())
    assert set(plan.claims) == set(geometries)
    assert len(plan.claims) == len(set(plan.claims)) == len(geometries)
    assert plan.controlnet.keys["input_hint_block.14.weight"] == (
        "controlnet_cond_embedding.conv_out.weight"
    )
    assert plan.controlnet.keys["zero_convs.11.0.bias"] == ("controlnet_down_blocks.11.bias")
    assert plan.controlnet.keys["middle_block.1.proj_in.weight"] == (
        "mid_block.attentions.0.proj_in.weight"
    )
    assert source.payload_reads == 0
    with pytest.raises(FrozenInstanceError):
        plan.source_layout = "canonical"  # type: ignore[misc]

    duplicate_map = dict(plan.controlnet.keys)
    duplicate_map["time_embed.0.bias"] = duplicate_map["time_embed.0.weight"]
    duplicate_component = replace(
        plan.controlnet,
        keys=duplicate_map,
        dtypes={key: FLOAT16 for key in duplicate_map},
    )
    with pytest.raises(ValueError, match="one-to-one"):
        replace(plan, controlnet=duplicate_component)
    with pytest.raises(ValueError, match="source_layout"):
        replace(plan, source_layout="unknown")  # type: ignore[arg-type]


def test_sd15_controlnet_plans_float64_storage_for_runtime_cast() -> None:
    geometries = canonical_geometries()
    source_key = "zero_convs.0.0.weight"
    geometries[source_key] = replace(geometries[source_key], dtype=FLOAT64)

    plan = plan_sd15_controlnet(
        HeaderSource(Path("controlnet.safetensors"), geometries), asset_digest=ASSET_DIGEST
    )

    assert plan.controlnet.dtypes[source_key] is FLOAT64


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda sd: {}, "empty"),
        (lambda sd: {k: v for k, v in sd.items() if k != "zero_convs.11.0.bias"}, "missing"),
        (lambda sd: {**sd, "foreign.weight": TensorGeometry((1,), FLOAT32)}, "leftover"),
        (
            lambda sd: {
                **sd,
                "control_model.zero_convs.0.0.weight": next(iter(sd.values())),
            },
            "packed",
        ),
        (
            lambda sd: {
                **sd,
                "controlnet_cond_embedding.conv_in.weight": next(iter(sd.values())),
            },
            "mixed",
        ),
        (lambda sd: {**sd, "zero_convs.12.0.weight": next(iter(sd.values()))}, "leftover"),
        (
            lambda sd: {**sd, "zero_convs.0.0.weight": TensorGeometry((320, 320, 1, 1), UINT8)},
            "ControlNet weight zero_convs.0.0.weight requires floating storage, got uint8",
        ),
    ],
)
def test_malformed_mixed_foreign_partial_duplicate_unknown_and_leftover_refuse(
    mutate: object, match: str
) -> None:
    source = HeaderSource(Path("bad.safetensors"), mutate(canonical_geometries()))  # type: ignore[operator]
    with pytest.raises(ControlNetDetectError, match=match):
        plan_sd15_controlnet(source, asset_digest=ASSET_DIGEST)
    assert source.payload_reads == 0


def test_duplicate_source_sequence_refuses_before_payload() -> None:
    source = DuplicateKeySource(Path("duplicate.safetensors"), canonical_geometries())
    with pytest.raises(ControlNetDetectError, match="duplicate"):
        plan_sd15_controlnet(source, asset_digest=ASSET_DIGEST)
    assert source.payload_reads == 0


@pytest.mark.parametrize("digest", ["", "a" * 64, "sha256:" + "a" * 64, "blake3:ABC"])
def test_plan_requires_asset_system_content_identity(digest: str) -> None:
    source = HeaderSource(Path("controlnet.safetensors"), canonical_geometries())
    with pytest.raises(ValueError, match="canonical blake3 asset digest"):
        plan_sd15_controlnet(source, asset_digest=digest)
    assert source.payload_reads == 0


def test_wrong_geometry_and_colliding_diffusers_alias_refuse_before_payload() -> None:
    wrong = canonical_geometries()
    wrong["input_blocks.0.0.weight"] = TensorGeometry((320, 9, 3, 3), FLOAT16)
    source = HeaderSource(Path("wrong.safetensors"), wrong)
    with pytest.raises(ControlNetDetectError, match="geometry mismatch"):
        plan_sd15_controlnet(source, asset_digest=ASSET_DIGEST)
    assert source.payload_reads == 0

    ambiguous = diffusers_geometries()
    ambiguous["zero_convs.0.0.weight"] = ambiguous["controlnet_down_blocks.0.weight"]
    source = HeaderSource(Path("ambiguous.safetensors"), ambiguous)
    with pytest.raises(ControlNetDetectError, match="mixed"):
        plan_sd15_controlnet(source, asset_digest=ASSET_DIGEST)
    assert source.payload_reads == 0


def test_control_application_is_frozen_strict_unique_and_acyclic() -> None:
    first = ControlApplication(
        "controlnet",
        PayloadReference("hint-a"),
        1.0,
        PercentRange(0.0, 1.0),
        None,
    )
    second = ControlApplication(
        "detail",
        PayloadReference("hint-b"),
        0.5,
        PercentRange(0.25, 0.75),
        first,
    )
    with pytest.raises(FrozenInstanceError):
        second.strength = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError, match="strength must be a float"):
        replace(first, strength=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[0, 10\]"):
        replace(first, strength=10.1)
    with pytest.raises(ValueError, match="unique"):
        replace(second, child_id="controlnet")
    object.__setattr__(first, "previous", second)
    with pytest.raises(ValueError, match="acyclic"):
        replace(second)


def test_dependency_append_preserves_base_and_exact_edge_identity() -> None:
    existing = DependencyRef(
        child_id="adapter",
        child=recipe("proof.adapter", "adapter", "3"),
        residency_group="adapter",
        scope="model",
        clone_mode="with-parent",
        accounting_owner="parent",
    )
    base = recipe("dinkster.sd15", "checkpoint", "1", dependencies=(existing,))
    child = recipe("dinkster.sd15_controlnet", "controlnet", "2")
    edge = DependencyRef(
        child_id="controlnet",
        child=child,
        residency_group="controlnet",
        scope="conditional",
        clone_mode="with-parent",
        accounting_owner="parent",
    )
    controlled = base.append_dependencies((edge,))
    assert base.dependencies == (existing,)
    assert controlled.dependencies == (existing, edge)
    assert controlled.dependencies[0] is existing
    assert controlled.dependencies[1].child_id == "controlnet"
    assert controlled.dependencies[1].residency_group == "controlnet"
    assert controlled.dependencies[1].scope == "conditional"
    assert controlled.dependencies[1].clone_mode == "with-parent"
    assert controlled.dependencies[1].accounting_owner == "parent"
    assert controlled.runtime_identity != base.runtime_identity
    assert controlled.runtime_identity == base.append_dependencies((edge,)).runtime_identity
    with pytest.raises(TypeError, match="tuple"):
        base.append_dependencies([edge])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="child_ids must be unique"):
        base.append_dependencies((replace(existing, child=recipe("proof.other", "other", "4")),))
    assert base.dependencies == (existing,)


def test_plan_child_and_parent_structural_identities_are_deterministic() -> None:
    canonical = plan_sd15_controlnet(
        HeaderSource(Path("canonical.safetensors"), canonical_geometries()),
        asset_digest=ASSET_DIGEST,
    )
    diffusers = plan_sd15_controlnet(
        HeaderSource(Path("diffusers.safetensors"), diffusers_geometries()),
        asset_digest=ASSET_DIGEST,
    )
    canonical_facts = runtime_component_identity(canonical.family_id, canonical.identity_components)
    diffusers_facts = runtime_component_identity(diffusers.family_id, diffusers.identity_components)
    assert canonical_facts == diffusers_facts

    child = ReconstructionRecipe(
        sources=(WeightSourceBinding(canonical.source_role, source_ref("2")),),
        family_id=canonical.family_id,
        component_identity=canonical_facts,
        knobs=RuntimeKnobs("float16", "float32", "float32", False),
    )
    base = recipe("dinkster.sd15", "checkpoint", "1")
    edge = DependencyRef(
        child_id="controlnet",
        child=child,
        residency_group="controlnet",
        scope="conditional",
        clone_mode="with-parent",
        accounting_owner="parent",
    )
    assert child.runtime_identity == replace(child).runtime_identity
    assert (
        base.append_dependencies((edge,)).runtime_identity
        == base.append_dependencies((edge,)).runtime_identity
    )
    assert base.append_dependencies(()).runtime_identity == base.runtime_identity


def test_controlnet_modules_import_without_framework_side_effects() -> None:
    import sys

    assert "torch" not in sys.modules
