"""Torch-free Wan 2.1 Uni3C header and planning proofs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    WAN21_UNI3C,
    TensorGeometry,
    Wan21Uni3CConfig,
    WeightEntry,
    detect_wan21_uni3c,
    normalize_wan21_uni3c_key,
    plan_wan21_uni3c,
    require_wan21_uni3c_layout,
    wan21_uni3c_dtype,
    wan21_uni3c_layout,
    wan21_uni3c_model_layout,
)


@dataclass
class HeaderSource:
    geometries: dict[str, TensorGeometry]
    path: Path = Path("wan21_uni3c.safetensors")

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


def source_for() -> HeaderSource:
    return HeaderSource(
        {
            key: TensorGeometry(shape, wan21_uni3c_dtype(key))
            for key, shape in wan21_uni3c_layout().items()
        }
    )


def test_exact_layout_pins_published_geometry_dtypes_and_attention_spelling() -> None:
    source = source_for()
    layout = wan21_uni3c_layout()

    assert detect_wan21_uni3c(source) is WAN21_UNI3C
    assert require_wan21_uni3c_layout(source) == (WAN21_UNI3C, layout)
    assert len(layout) == 490
    assert layout["controlnet_patch_embedding.weight"] == (5120, 36, 1, 2, 2)
    assert layout["controlnet_blocks.19.norm1.linear.weight"] == (3072, 5120)
    assert layout["controlnet_blocks.19.ffn.0.weight"] == (4096, 1024)
    assert layout["proj_out.19.weight"] == (5120, 1024)
    assert wan21_uni3c_dtype("controlnet_patch_embedding.weight") == FLOAT32
    assert wan21_uni3c_dtype("controlnet_blocks.0.ffn.0.weight") == FLOAT16
    assert (
        normalize_wan21_uni3c_key("controlnet_blocks.0.self_attn.to_out.0.weight")
        == "controlnet_blocks.0.self_attn.o.weight"
    )
    assert "controlnet_blocks.0.self_attn.q.weight" in wan21_uni3c_model_layout()
    assert "controlnet_blocks.0.self_attn.to_q.weight" not in wan21_uni3c_model_layout()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "foreign",
        "shape",
        "patch-dtype",
        "block-dtype",
    ),
)
def test_detection_fails_closed_on_any_header_drift(mutation: str) -> None:
    source = source_for()
    if mutation == "missing":
        source.geometries.pop("proj_out.19.bias")
    elif mutation == "foreign":
        source.geometries["foreign.weight"] = TensorGeometry((1,), FLOAT16)
    elif mutation == "shape":
        source.geometries["controlnet_blocks.8.ffn.2.weight"] = TensorGeometry(
            (1023, 4096), FLOAT16
        )
    elif mutation == "patch-dtype":
        source.geometries["controlnet_patch_embedding.bias"] = TensorGeometry((5120,), FLOAT16)
    else:
        source.geometries["controlnet_blocks.4.norm2.norm.bias"] = TensorGeometry((1024,), FLOAT32)
    assert detect_wan21_uni3c(source) is None


def test_planner_maps_every_published_key_to_the_native_state_and_asset_identity() -> None:
    plan = plan_wan21_uni3c(source_for(), asset_digest="blake3:" + "a" * 64)

    assert plan.source_role == "wan21_uni3c"
    assert plan.patch.config is WAN21_UNI3C
    assert set(plan.patch.keys) == set(wan21_uni3c_model_layout())
    assert plan.patch.keys["controlnet_blocks.3.self_attn.q.weight"] == (
        "controlnet_blocks.3.self_attn.to_q.weight"
    )
    assert plan.patch.dtypes["controlnet_patch_embedding.weight"] == FLOAT32
    assert plan.patch.dtypes["controlnet_blocks.0.self_attn.o.weight"] == FLOAT16
    assert plan.identity_components == (plan.patch,)


def test_planner_and_profile_refuse_forged_or_incomplete_inputs() -> None:
    with pytest.raises(ValueError, match="canonical blake3"):
        plan_wan21_uni3c(source_for(), asset_digest="sha256:" + "0" * 64)
    source = source_for()
    source.geometries.pop("controlnet_mask_embedding.mask_zero_proj.bias")
    with pytest.raises(ValueError, match="exact maintained"):
        plan_wan21_uni3c(source, asset_digest="blake3:" + "0" * 64)
    forged = Wan21Uni3CConfig()
    with pytest.raises(ValueError, match="exact maintained"):
        wan21_uni3c_layout(forged)
