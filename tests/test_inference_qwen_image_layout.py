"""Qwen Image torch-free DiT layout and assembly-plan proofs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT32,
    NATIVE_WIRED_FAMILY_IDS,
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_2511_CONFIG,
    QWEN_IMAGE_LAYERED_CONFIG,
    UINT8,
    DType,
    QwenImageConfig,
    QwenImageDiTAssemblyError,
    TensorGeometry,
    WeightEntry,
    builtin_families,
    plan_qwen_image_dit_assembly,
    qwen_image_dit_layout,
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
        raise AssertionError("Qwen Image planning must not read metadata or payloads")


class DuplicateKeySource(HeaderSource):
    def keys(self) -> Sequence[str]:
        keys = tuple(super().keys())
        return (*keys, keys[0])


def expected_qwen_image_layout() -> dict[str, tuple[int, ...]]:
    keys: dict[str, tuple[int, ...]] = {
        "txt_norm.weight": (3584,),
        "time_text_embed.timestep_embedder.linear_1.weight": (3072, 256),
        "time_text_embed.timestep_embedder.linear_1.bias": (3072,),
        "time_text_embed.timestep_embedder.linear_2.weight": (3072, 3072),
        "time_text_embed.timestep_embedder.linear_2.bias": (3072,),
        "img_in.weight": (3072, 64),
        "img_in.bias": (3072,),
        "txt_in.weight": (3072, 3584),
        "txt_in.bias": (3072,),
    }
    block_suffixes = {
        "img_mod.1.weight": (18432, 3072),
        "img_mod.1.bias": (18432,),
        "img_mlp.net.0.proj.weight": (12288, 3072),
        "img_mlp.net.0.proj.bias": (12288,),
        "img_mlp.net.2.weight": (3072, 12288),
        "img_mlp.net.2.bias": (3072,),
        "txt_mod.1.weight": (18432, 3072),
        "txt_mod.1.bias": (18432,),
        "txt_mlp.net.0.proj.weight": (12288, 3072),
        "txt_mlp.net.0.proj.bias": (12288,),
        "txt_mlp.net.2.weight": (3072, 12288),
        "txt_mlp.net.2.bias": (3072,),
        "attn.norm_q.weight": (128,),
        "attn.norm_k.weight": (128,),
        "attn.norm_added_q.weight": (128,),
        "attn.norm_added_k.weight": (128,),
        "attn.to_q.weight": (3072, 3072),
        "attn.to_q.bias": (3072,),
        "attn.to_k.weight": (3072, 3072),
        "attn.to_k.bias": (3072,),
        "attn.to_v.weight": (3072, 3072),
        "attn.to_v.bias": (3072,),
        "attn.add_q_proj.weight": (3072, 3072),
        "attn.add_q_proj.bias": (3072,),
        "attn.add_k_proj.weight": (3072, 3072),
        "attn.add_k_proj.bias": (3072,),
        "attn.add_v_proj.weight": (3072, 3072),
        "attn.add_v_proj.bias": (3072,),
        "attn.to_out.0.weight": (3072, 3072),
        "attn.to_out.0.bias": (3072,),
        "attn.to_add_out.weight": (3072, 3072),
        "attn.to_add_out.bias": (3072,),
    }

    for index in range(60):
        root = f"transformer_blocks.{index}"
        keys.update({f"{root}.{suffix}": shape for suffix, shape in block_suffixes.items()})

    keys.update(
        {
            "norm_out.linear.weight": (6144, 3072),
            "norm_out.linear.bias": (6144,),
            "proj_out.weight": (64, 3072),
            "proj_out.bias": (64,),
        }
    )
    return keys


def qwen_image_geometries(
    prefix: str = "", *, dtype: DType = BFLOAT16
) -> dict[str, TensorGeometry]:
    return {
        prefix + key: TensorGeometry(shape, dtype)
        for key, shape in expected_qwen_image_layout().items()
    }


def test_qwen_image_dit_layout_is_complete_exact_and_immutable() -> None:
    layout = qwen_image_dit_layout()
    assert layout.config is QWEN_IMAGE_CONFIG
    assert layout.depth == 60
    assert layout.hidden_width == 3072
    assert layout.attention_heads == 24
    assert layout.attention_head_dim == 128
    assert layout.text_width == 3584
    assert layout.time_input_width == 256
    assert layout.time_embed_width == 3072
    assert layout.image_patch_width == 64
    assert layout.output_patch_width == 64
    assert layout.patch == (2, 2)
    assert layout.attention_kind == "joint_full"
    assert layout.stream_order == ("text", "image")
    assert len(layout.keys) == 1933
    assert dict(layout.keys) == expected_qwen_image_layout()
    assert layout.keys["time_text_embed.timestep_embedder.linear_1.weight"] == (
        3072,
        256,
    )
    assert layout.keys["img_in.weight"] == (3072, 64)
    assert layout.keys["txt_in.weight"] == (3072, 3584)
    assert layout.keys["transformer_blocks.59.img_mod.1.weight"] == (18432, 3072)
    assert layout.keys["transformer_blocks.59.img_mlp.net.0.proj.weight"] == (
        12288,
        3072,
    )
    assert layout.keys["transformer_blocks.59.attn.norm_added_k.weight"] == (128,)
    assert layout.keys["transformer_blocks.59.attn.to_add_out.weight"] == (
        3072,
        3072,
    )
    assert layout.keys["norm_out.linear.weight"] == (6144, 3072)
    assert layout.keys["proj_out.weight"] == (64, 3072)
    for index in range(60):
        assert sum(key.startswith(f"transformer_blocks.{index}.") for key in layout.keys) == 32
    with pytest.raises(TypeError):
        layout.keys["extra"] = (1,)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        layout.depth = 59  # type: ignore[misc]
    altered = dict(layout.keys)
    altered["transformer_blocks.59.attn.norm_q.weight"] = (127,)
    with pytest.raises(ValueError, match="exact Qwen Image DiT layout"):
        replace(layout, keys=altered)
    with pytest.raises(TypeError, match="QwenImageConfig"):
        qwen_image_dit_layout(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("config", "marker"),
    (
        (QWEN_IMAGE_EDIT_2511_CONFIG, "__index_timestep_zero__"),
        (
            QWEN_IMAGE_LAYERED_CONFIG,
            "time_text_embed.addition_t_embedding.weight",
        ),
    ),
)
def test_qwen_image_variant_layout_and_plan_include_exact_variant_state(
    config: QwenImageConfig, marker: str
) -> None:
    layout = qwen_image_dit_layout(config)
    assert layout.config is config
    assert marker in layout.keys
    assert len(layout.keys) == 1934
    source = HeaderSource(
        {key: TensorGeometry(shape, BFLOAT16) for key, shape in layout.keys.items()},
        [],
    )
    plan = plan_qwen_image_dit_assembly(source)
    assert plan.layout == layout
    assert set(plan.claims) == set(layout.keys)


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_qwen_image_dit_plan_claims_every_bare_or_prefixed_key_once(
    prefix: str,
) -> None:
    source = HeaderSource(qwen_image_geometries(prefix), [])
    first = plan_qwen_image_dit_assembly(source)
    second = plan_qwen_image_dit_assembly(source)
    assert first == second
    assert first.layout == qwen_image_dit_layout()
    assert first.source_prefix == prefix
    assert first.family_id == QWEN_IMAGE_CONFIG.family_id
    assert first.source_role == "diffusion"
    assert first.execution_format == "bfloat16"
    assert set(first.keys) == set(first.layout.keys)
    assert set(first.claims) == set(source.geometries)
    assert first.claims == tuple(sorted(first.claims))
    assert len(first.claims) == len(set(first.claims)) == len(first.keys) == 1933
    assert all(first.keys[key] == prefix + key for key in first.keys)
    assert set(first.dtypes.values()) == {BFLOAT16}
    assert source.entries_read == list(first.keys.values()) * 2
    assert source.metadata_reads == 0
    with pytest.raises(TypeError):
        first.keys["extra"] = "extra"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first.source_prefix = ""  # type: ignore[misc]


def test_qwen_image_dit_float32_plan_is_distinct() -> None:
    bf16 = plan_qwen_image_dit_assembly(HeaderSource(qwen_image_geometries(), []))
    fp32 = plan_qwen_image_dit_assembly(HeaderSource(qwen_image_geometries(dtype=FLOAT32), []))
    assert fp32.execution_format == "float32"
    assert set(fp32.dtypes.values()) == {FLOAT32}
    assert bf16 != fp32


def test_qwen_image_dit_plan_constructor_refuses_false_facts() -> None:
    plan = plan_qwen_image_dit_assembly(HeaderSource(qwen_image_geometries(), []))
    ordinary_key = "transformer_blocks.0.attn.to_q.weight"
    invalid_dtypes = dict(plan.dtypes)
    invalid_dtypes[ordinary_key] = UINT8
    with pytest.raises(QwenImageDiTAssemblyError, match="require floating storage"):
        replace(plan, dtypes=invalid_dtypes)
    mixed_dtypes = dict(plan.dtypes)
    mixed_dtypes[ordinary_key] = FLOAT32
    mixed = replace(plan, dtypes=mixed_dtypes)
    assert mixed.dtypes[ordinary_key] is FLOAT32
    with pytest.raises(QwenImageDiTAssemblyError, match="execution format"):
        replace(plan, execution_format="float16")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="every layout key"):
        replace(plan, keys={})
    with pytest.raises(ValueError, match="exactly cover"):
        replace(plan, claims=())


def test_qwen_image_dit_execution_format_is_natively_runnable() -> None:
    plan = plan_qwen_image_dit_assembly(HeaderSource(qwen_image_geometries(), []))
    assert plan.execution_format == "bfloat16"
    assert QWEN_IMAGE_CONFIG.family_id in {family.id for family in builtin_families()}
    assert QWEN_IMAGE_CONFIG.family_id in NATIVE_WIRED_FAMILY_IDS


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda values: {}, "empty"),
        (
            lambda values: {
                key: value
                for key, value in values.items()
                if key != "transformer_blocks.59.attn.norm_q.weight"
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
                "model.diffusion_model.img_in.weight": values["img_in.weight"],
            },
            "mixed",
        ),
        (
            lambda values: {
                **values,
                "transformer_blocks.59.attn.norm_q.weight": TensorGeometry((127,), BFLOAT16),
            },
            "geometry mismatch",
        ),
    ),
)
def test_qwen_image_dit_plan_refuses_incomplete_foreign_mixed_and_wrong_geometry(
    mutate: object, match: str
) -> None:
    source = HeaderSource(mutate(qwen_image_geometries()), [])  # type: ignore[operator]
    with pytest.raises(QwenImageDiTAssemblyError, match=match):
        plan_qwen_image_dit_assembly(source)
    assert source.metadata_reads == 0


def test_qwen_image_dit_plan_refuses_duplicate_or_inconsistent_headers() -> None:
    duplicate = DuplicateKeySource(qwen_image_geometries(), [])
    with pytest.raises(QwenImageDiTAssemblyError, match="duplicate"):
        plan_qwen_image_dit_assembly(duplicate)

    class InconsistentSource(HeaderSource):
        def entry(self, key: str) -> WeightEntry:
            raise KeyError(key)

    with pytest.raises(QwenImageDiTAssemblyError, match="inconsistent"):
        plan_qwen_image_dit_assembly(InconsistentSource(qwen_image_geometries(), []))

    class MiskeyedSource(HeaderSource):
        def entry(self, key: str) -> WeightEntry:
            entry = super().entry(key)
            return replace(entry, key="another.key")

    with pytest.raises(QwenImageDiTAssemblyError, match="returned another.key"):
        plan_qwen_image_dit_assembly(MiskeyedSource(qwen_image_geometries(), []))


def test_qwen_image_dit_mixed_storage_uses_default_compute_format() -> None:
    geometries = qwen_image_geometries()
    geometries["transformer_blocks.0.attn.to_q.weight"] = TensorGeometry((3072, 3072), FLOAT32)
    source = HeaderSource(geometries, [])
    plan = plan_qwen_image_dit_assembly(source)
    assert plan.execution_format == "bfloat16"
    assert source.metadata_reads == 0


def test_qwen_image_dit_nonfloating_storage_is_invalid_input() -> None:
    geometries = qwen_image_geometries()
    geometries["transformer_blocks.0.attn.to_q.weight"] = TensorGeometry((3072, 3072), UINT8)
    with pytest.raises(QwenImageDiTAssemblyError, match="requires floating storage"):
        plan_qwen_image_dit_assembly(HeaderSource(geometries, []))
