"""MiniMax H3 torch-free DiT layout and assembly-plan proofs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, replace
from typing import cast

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    MINIMAX_H3_CONFIG,
    UINT8,
    AttentionPolicy,
    DType,
    MiniMaxH3DiTAssemblyError,
    MiniMaxH3DiTExecutionRefusal,
    NativeRefusalCategory,
    TensorGeometry,
    WeightEntry,
    builtin_families,
    minimax_h3_dit_layout,
    minimax_h3_dit_provider_facts,
    plan_minimax_h3_dit_assembly,
)


@dataclass
class HeaderSource:
    geometries: dict[str, TensorGeometry]
    entries_read: list[str]
    metadata_reads: int = 0

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        self.entries_read.append(key)
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        self.metadata_reads += 1
        raise AssertionError("H3 DiT planning must not read metadata or payloads")


class DuplicateKeySource(HeaderSource):
    def keys(self) -> Sequence[str]:
        keys = tuple(super().keys())
        return (*keys, keys[0])


def test_minimax_h3_dit_provider_facts_bind_behavior_versions() -> None:
    plain = minimax_h3_dit_provider_facts(
        "fl2va-dit", quantized=False, torch_version="2.13.0+cu130"
    )
    quantized = minimax_h3_dit_provider_facts(
        "ref2va-dit",
        quantized=True,
        torch_version="2.13.0+cu130",
        dinkster_kitchen_version="0.2.31",
    )

    assert "int8_provider=dinkster-kitchen.int8_linear" not in plain
    assert "torch_version=2.13.0+cu130" in plain
    assert "dinkster_kitchen_version=0.2.31" in quantized
    assert quantized[-1] == "artifact_role=ref2va-dit"
    assert quantized != minimax_h3_dit_provider_facts(
        "ref2va-dit",
        quantized=True,
        torch_version="2.13.0+cu130",
        dinkster_kitchen_version="0.2.32",
    )
    sol = minimax_h3_dit_provider_facts(
        "fl2va-dit",
        quantized=False,
        torch_version="2.13.0+cu130",
        dinkster_kitchen_version="0.2.32",
        attention_policy="sol",
    )
    assert "attention_provider=dinkster-kitchen.sol_attn" in sol
    assert "dinkster_kitchen_version=0.2.32" in sol
    with pytest.raises(ValueError, match="require a dinkster-kitchen version"):
        minimax_h3_dit_provider_facts(
            "fl2va-dit",
            quantized=False,
            torch_version="2.13.0+cu130",
            attention_policy="sol",
        )
    flash = minimax_h3_dit_provider_facts(
        "fl2va-dit", quantized=False, torch_version="2.13.0+cu130", attention_policy="flash"
    )
    assert "attention_provider=flash" in flash
    with pytest.raises(ValueError, match="unsupported attention policy"):
        minimax_h3_dit_provider_facts(
            "fl2va-dit",
            quantized=False,
            torch_version="2.13.0+cu130",
            attention_policy=cast("AttentionPolicy", "not-a-policy"),
        )


def expected_h3_layout() -> dict[str, tuple[int, ...]]:
    keys: dict[str, tuple[int, ...]] = {
        "video_patch_proj.weight": (5376, 96),
        "video_patch_proj.bias": (5376,),
        "audio_patch_proj.weight": (5376, 32),
        "audio_patch_proj.bias": (5376,),
        "condition_proj.weight": (5376, 5120),
        "condition_proj.bias": (5376,),
        "adaln_t_table": (1025, 8),
        "rope.inv_freq": (16,),
    }

    def add_attention(root: str) -> None:
        keys[f"{root}.qkv_proj.weight"] = (21504, 5376)
        keys[f"{root}.q_norm.weight"] = (128,)
        keys[f"{root}.k_norm.weight"] = (128,)
        keys[f"{root}.out_proj.weight"] = (5376, 7168)

    def add_mlp(root: str) -> None:
        keys[f"{root}.fc1.weight"] = (28672, 5376)
        keys[f"{root}.fc2.weight"] = (5376, 14336)

    for index in range(2):
        root = f"token_refiner.blocks.{index}"
        keys[f"{root}.norm1.weight"] = (5376,)
        keys[f"{root}.norm2.weight"] = (5376,)
        add_attention(f"{root}.attn")
        add_mlp(f"{root}.mlp")
    keys["token_refiner.final_norm.weight"] = (5376,)

    for index in range(50):
        root = f"blocks.{index}"
        keys[f"{root}.norm1.weight"] = (5376,)
        keys[f"{root}.norm2.weight"] = (5376,)
        add_attention(f"{root}.attn")
        add_mlp(f"{root}.mlp")
        keys[f"{root}.adaln_proj.linear.weight"] = (96768, 8)
        keys[f"{root}.adaln_proj.linear.bias"] = (96768,)

    keys["final_layer.norm.weight"] = (5376,)
    keys["final_layer.adaln_proj.linear.weight"] = (10752, 8)
    keys["final_layer.adaln_proj.linear.bias"] = (10752,)
    keys["final_layer.video_out.weight"] = (96, 5376)
    keys["final_layer.video_out.bias"] = (96,)
    keys["final_layer.audio_out.weight"] = (32, 5376)
    keys["final_layer.audio_out.bias"] = (32,)
    return keys


def expected_h3_fp32_storage_keys() -> frozenset[str]:
    keys = {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "adaln_t_table",
        "rope.inv_freq",
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
    for index in range(50):
        keys.add(f"blocks.{index}.adaln_proj.linear.weight")
        keys.add(f"blocks.{index}.adaln_proj.linear.bias")
    return frozenset(keys)


def h3_geometries(prefix: str = "", *, all_float32: bool = False) -> dict[str, TensorGeometry]:
    return {
        prefix + key: TensorGeometry(
            shape,
            FLOAT32 if all_float32 or key in expected_h3_fp32_storage_keys() else BFLOAT16,
        )
        for key, shape in expected_h3_layout().items()
    }


def test_minimax_h3_dit_layout_is_complete_exact_and_immutable() -> None:
    layout = minimax_h3_dit_layout()
    assert layout.config is MINIMAX_H3_CONFIG
    assert layout.depth == 50
    assert layout.hidden_width == 5376
    assert layout.attention_heads == 56
    assert layout.attention_head_dim == 128
    assert layout.attention_heads * layout.attention_head_dim == 7168
    assert layout.ffn_width == 14336
    assert layout.video_patch == (1, 2, 2)
    assert layout.video_patch_width == 96
    assert layout.audio_patch_width == 32
    assert layout.attention_kind == "full"
    assert layout.attention_mask is None
    assert layout.rope_axes == ("time", "height", "width")
    assert layout.rope_axis_dim == 16
    assert layout.rope_rotary_dim == 96
    assert layout.rope_style == "split_half"
    assert layout.sampler_stream_order == ("video", "audio")
    assert layout.packed_target_order == ("audio", "video")
    assert len(layout.keys) == 532
    assert dict(layout.keys) == expected_h3_layout()
    assert len(layout.fp32_storage_keys) == 112
    assert layout.fp32_storage_keys == expected_h3_fp32_storage_keys()
    assert layout.keys["video_patch_proj.weight"] == (5376, 96)
    assert layout.keys["audio_patch_proj.weight"] == (5376, 32)
    assert layout.keys["condition_proj.weight"] == (5376, 5120)
    assert layout.keys["token_refiner.blocks.1.mlp.fc2.weight"] == (5376, 14336)
    assert layout.keys["blocks.49.attn.qkv_proj.weight"] == (21504, 5376)
    assert layout.keys["blocks.49.attn.out_proj.weight"] == (5376, 7168)
    assert layout.keys["blocks.49.mlp.fc1.weight"] == (28672, 5376)
    assert layout.keys["blocks.49.mlp.fc2.weight"] == (5376, 14336)
    assert layout.keys["blocks.49.adaln_proj.linear.weight"] == (96768, 8)
    assert layout.keys["final_layer.adaln_proj.linear.weight"] == (10752, 8)
    assert layout.keys["final_layer.video_out.weight"] == (96, 5376)
    assert layout.keys["final_layer.audio_out.weight"] == (32, 5376)
    for index in range(50):
        assert sum(key.startswith(f"blocks.{index}.") for key in layout.keys) == 10
    with pytest.raises(TypeError):
        layout.keys["extra"] = (1,)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        layout.depth = 49  # type: ignore[misc]
    altered = dict(layout.keys)
    altered["blocks.49.attn.q_norm.weight"] = (127,)
    with pytest.raises(ValueError, match="exact H3 DiT layout"):
        replace(layout, keys=altered)
    with pytest.raises(ValueError, match="exact H3 FP32 islands"):
        replace(layout, fp32_storage_keys=frozenset())
    with pytest.raises(TypeError, match="frozenset"):
        replace(
            layout,
            fp32_storage_keys=set(layout.fp32_storage_keys),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="MiniMaxH3Config"):
        minimax_h3_dit_layout(object())  # type: ignore[arg-type]


def test_minimax_h3_mlp_time_embedding_layout_matches_official_split_artifact() -> None:
    layout = minimax_h3_dit_layout(time_embedding_kind="mlp")
    assert layout.time_embedding_kind == "mlp"
    assert "mappingproxy" not in repr(layout)
    assert "frozenset" not in repr(layout)
    assert len(layout.keys) == 535
    assert "adaln_t_table" not in layout.keys
    assert layout.keys["time_embedder.proj_in.weight"] == (5376, 256)
    assert layout.keys["time_embedder.proj_in.bias"] == (5376,)
    assert layout.keys["time_embedder.proj_out.weight"] == (2688, 5376)
    assert layout.keys["time_embedder.proj_out.bias"] == (2688,)
    assert all(
        key in layout.fp32_storage_keys
        for key in (
            "time_embedder.proj_in.weight",
            "time_embedder.proj_in.bias",
            "time_embedder.proj_out.weight",
            "time_embedder.proj_out.bias",
        )
    )
    assert "blocks.0.adaln_proj.linear.weight" not in layout.fp32_storage_keys
    assert "final_layer.adaln_proj.linear.weight" not in layout.fp32_storage_keys

    source = HeaderSource(
        {
            key: TensorGeometry(
                shape,
                FLOAT32 if key in layout.fp32_storage_keys else BFLOAT16,
            )
            for key, shape in layout.keys.items()
        },
        [],
    )
    plan = plan_minimax_h3_dit_assembly(source)
    assert plan.layout == layout
    assert plan.execution_format == "bfloat16"
    assert set(plan.claims) == set(layout.keys)


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_minimax_h3_dit_plan_claims_every_bare_or_prefixed_key_once(prefix: str) -> None:
    source = HeaderSource(h3_geometries(prefix), [])
    first = plan_minimax_h3_dit_assembly(source)
    second = plan_minimax_h3_dit_assembly(source)
    assert first == second
    assert first.layout == minimax_h3_dit_layout()
    assert first.source_prefix == prefix
    assert first.family_id == MINIMAX_H3_CONFIG.family_id
    assert first.source_role == "diffusion"
    assert first.execution_format == "bfloat16"
    assert first.runtime_provider is None
    assert first.runnable is False
    assert set(first.keys) == set(first.layout.keys)
    assert set(first.claims) == set(source.geometries)
    assert first.claims == tuple(sorted(first.claims))
    assert len(first.claims) == len(set(first.claims)) == len(first.keys)
    assert all(first.keys[key] == prefix + key for key in first.keys)
    assert {
        key for key, dtype in first.dtypes.items() if dtype == FLOAT32
    } == expected_h3_fp32_storage_keys()
    assert all(
        dtype == BFLOAT16
        for key, dtype in first.dtypes.items()
        if key not in expected_h3_fp32_storage_keys()
    )
    assert set(source.entries_read) == set(source.geometries)
    assert source.metadata_reads == 0
    with pytest.raises(TypeError):
        first.keys["extra"] = "extra"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first.source_prefix = ""  # type: ignore[misc]


def test_minimax_h3_dit_float32_plan_is_distinct_and_still_inert() -> None:
    bf16 = plan_minimax_h3_dit_assembly(HeaderSource(h3_geometries(), []))
    fp32 = plan_minimax_h3_dit_assembly(HeaderSource(h3_geometries(all_float32=True), []))
    assert fp32.execution_format == "float32"
    assert bf16 != fp32
    assert fp32.runtime_provider is None
    assert fp32.runnable is False


def test_minimax_h3_dit_plan_constructor_refuses_false_dtype_facts() -> None:
    plan = plan_minimax_h3_dit_assembly(HeaderSource(h3_geometries(), []))
    invalid_dtypes = dict(plan.dtypes)
    ordinary_key = "blocks.0.attn.qkv_proj.weight"
    fp32_island_key = "blocks.0.adaln_proj.linear.weight"
    invalid_dtypes[ordinary_key] = UINT8
    with pytest.raises(MiniMaxH3DiTAssemblyError, match="weights require floating storage"):
        replace(plan, dtypes=invalid_dtypes)
    ordinary_fp32 = dict(plan.dtypes)
    ordinary_fp32[ordinary_key] = FLOAT32
    assert replace(plan, dtypes=ordinary_fp32).dtypes[ordinary_key] is FLOAT32
    island_bfloat16 = dict(plan.dtypes)
    island_bfloat16[fp32_island_key] = BFLOAT16
    assert replace(plan, dtypes=island_bfloat16).dtypes[fp32_island_key] is BFLOAT16
    with pytest.raises(MiniMaxH3DiTAssemblyError, match="execution format"):
        replace(plan, execution_format="float32")


@pytest.mark.parametrize(
    ("key", "dtype"),
    (
        ("condition_proj.weight", FLOAT32),
        ("video_patch_proj.weight", BFLOAT16),
        ("blocks.0.attn.qkv_proj.weight", FLOAT16),
        ("blocks.49.attn.out_proj.weight", FLOAT32),
        ("blocks.49.adaln_proj.linear.bias", BFLOAT16),
        ("final_layer.audio_out.bias", BFLOAT16),
    ),
)
def test_minimax_h3_dit_plan_accepts_floating_storage_dtypes(key: str, dtype: DType) -> None:
    geometries = h3_geometries()
    geometries[key] = replace(geometries[key], dtype=dtype)
    plan = plan_minimax_h3_dit_assembly(HeaderSource(geometries, []))
    assert plan.dtypes[key] is dtype
    assert plan.execution_format == "bfloat16"


def test_minimax_h3_dit_execution_format_refusal_is_typed() -> None:
    plan = plan_minimax_h3_dit_assembly(HeaderSource(h3_geometries(), []))
    refusal = plan.execution_refusal()
    assert isinstance(refusal, MiniMaxH3DiTExecutionRefusal)
    assert refusal.execution_format == "bfloat16"
    assert refusal.runtime_provider is None
    assert refusal.category is NativeRefusalCategory.NATIVE_INELIGIBLE
    assert "no supported runtime provider" in str(refusal)
    assert MINIMAX_H3_CONFIG.family_id in {family.id for family in builtin_families()}


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda values: {}, "empty"),
        (
            lambda values: {
                key: value for key, value in values.items() if key != "blocks.49.attn.q_norm.weight"
            },
            "missing",
        ),
        (
            lambda values: {
                **values,
                "foreign.weight": TensorGeometry((1,), BFLOAT16),
            },
            "foreign",
        ),
        (
            lambda values: {
                **values,
                "model.diffusion_model.video_patch_proj.weight": values["video_patch_proj.weight"],
            },
            "mixed",
        ),
        (
            lambda values: {
                **values,
                "blocks.49.attn.q_norm.weight": TensorGeometry((127,), BFLOAT16),
            },
            "geometry mismatch",
        ),
    ),
)
def test_minimax_h3_dit_plan_refuses_incomplete_foreign_mixed_and_wrong_geometry(
    mutate: object, match: str
) -> None:
    source = HeaderSource(mutate(h3_geometries()), [])  # type: ignore[operator]
    with pytest.raises(MiniMaxH3DiTAssemblyError, match=match):
        plan_minimax_h3_dit_assembly(source)
    assert source.metadata_reads == 0


@pytest.mark.parametrize(
    ("key", "old_shape"),
    (
        ("video_patch_proj.weight", (7168, 96)),
        ("blocks.0.attn.qkv_proj.weight", (21504, 7168)),
        ("blocks.0.mlp.fc1.weight", (37888, 7168)),
        ("blocks.0.mlp.fc2.weight", (7168, 18944)),
        ("final_layer.video_out.weight", (96, 7168)),
    ),
)
def test_minimax_h3_dit_plan_rejects_old_outer_geometry(
    key: str, old_shape: tuple[int, ...]
) -> None:
    geometries = h3_geometries()
    geometries[key] = replace(geometries[key], shape=old_shape)
    with pytest.raises(MiniMaxH3DiTAssemblyError, match="geometry mismatch"):
        plan_minimax_h3_dit_assembly(HeaderSource(geometries, []))


def test_minimax_h3_dit_plan_refuses_duplicate_or_inconsistent_headers() -> None:
    duplicate = DuplicateKeySource(h3_geometries(), [])
    with pytest.raises(MiniMaxH3DiTAssemblyError, match="duplicate"):
        plan_minimax_h3_dit_assembly(duplicate)

    class InconsistentSource(HeaderSource):
        def entry(self, key: str) -> WeightEntry:
            raise KeyError(key)

    with pytest.raises(MiniMaxH3DiTAssemblyError, match="inconsistent"):
        plan_minimax_h3_dit_assembly(InconsistentSource(h3_geometries(), []))


def test_minimax_h3_dit_nonfloating_storage_names_the_weight_requirement() -> None:
    geometries = h3_geometries()
    geometries["blocks.0.attn.qkv_proj.weight"] = TensorGeometry((21504, 5376), UINT8)
    source = HeaderSource(geometries, [])
    with pytest.raises(
        MiniMaxH3DiTAssemblyError,
        match="H3 weight blocks.0.attn.qkv_proj.weight requires floating storage, got uint8",
    ):
        plan_minimax_h3_dit_assembly(source)
    assert source.metadata_reads == 0
