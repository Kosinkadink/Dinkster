"""Wan 2.2 TI2V 5B torch-free family contract tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from dinkster_inference.catalog import WAN22, builtin_family_registry
from dinkster_inference.devices import BFLOAT16, FLOAT8_E4M3, FLOAT16, INT64, DType
from dinkster_inference.sampling import Parameterization
from dinkster_inference.wan21 import (
    WAN22_FUN_CONTROL_5B,
    WAN22_FUN_INPAINT_5B,
    WAN22_LATENT,
    WAN22_SAMPLING,
    WAN22_SIGMAS,
    WAN22_TI2V_5B,
    Wan22Detector,
    detect_wan21,
    detect_wan22,
    wan21_layout,
)
from dinkster_inference.weights import TensorGeometry, WeightEntry


class HeaderSource:
    def __init__(
        self,
        shapes: Mapping[str, tuple[int, ...]],
        *,
        dtype: DType = FLOAT16,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.shapes = dict(shapes)
        self.dtype = dtype
        self._metadata = dict(metadata or {})

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.shapes[key], self.dtype)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return self._metadata


def wan22_shapes(prefix: str = "", profile: str = "ti2v-5b") -> dict[str, tuple[int, ...]]:
    configs = {
        "ti2v-5b": WAN22_TI2V_5B,
        "fun-control-5b": WAN22_FUN_CONTROL_5B,
        "fun-inpaint-5b": WAN22_FUN_INPAINT_5B,
    }
    return {prefix + key: shape for key, shape in wan21_layout(configs[profile]).items()}


def test_official_ti2v_layout_matches_checkpoint_geometry() -> None:
    layout = wan21_layout(WAN22_TI2V_5B)
    assert len(layout) == 825
    assert layout["patch_embedding.weight"] == (3072, 48, 1, 2, 2)
    assert layout["head.head.weight"] == (192, 3072)
    assert layout["head.modulation"] == (1, 2, 3072)
    assert layout["text_embedding.0.weight"] == (3072, 4096)
    assert layout["time_embedding.0.weight"] == (3072, 256)
    assert layout["blocks.0.ffn.0.weight"] == (14336, 3072)
    assert layout["blocks.29.ffn.2.weight"] == (3072, 14336)


def test_official_fun_layouts_match_checkpoint_geometry() -> None:
    control = wan21_layout(WAN22_FUN_CONTROL_5B)
    assert len(control) == 827
    assert control["patch_embedding.weight"] == (3072, 148, 1, 2, 2)
    assert control["ref_conv.weight"] == (3072, 48, 2, 2)
    inpaint = wan21_layout(WAN22_FUN_INPAINT_5B)
    assert len(inpaint) == 825
    assert inpaint["patch_embedding.weight"] == (3072, 100, 1, 2, 2)
    assert "ref_conv.weight" not in inpaint


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_official_profile_produces_exact_evidence(prefix: str) -> None:
    source = HeaderSource(wan22_shapes(prefix))
    evidence = detect_wan22(source)
    assert evidence == Wan22Detector().detect(source)
    assert evidence is not None
    assert evidence.family_id == "dinkster.wan22"
    assert evidence.fields == {
        "attention_head_dim": 128,
        "attention_heads": 24,
        "flf": False,
        "full_ref": False,
        "ffn_width": 14336,
        "hidden_width": 3072,
        "input_channels": 48,
        "key_prefix": prefix,
        "layers": 30,
        "model_type": "ti2v",
        "model_variant": "base",
        "output_channels": 48,
        "parameter_count": "5B",
        "patch": "1x2x2",
        "profile": "ti2v-5b",
        "ref_conv": False,
    }


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
@pytest.mark.parametrize(
    ("profile", "input_channels", "reference_channels"),
    (
        ("fun-control-5b", 148, 48),
        ("fun-inpaint-5b", 100, None),
    ),
)
def test_official_fun_profiles_produce_exact_evidence(
    prefix: str,
    profile: str,
    input_channels: int,
    reference_channels: int | None,
) -> None:
    source = HeaderSource(wan22_shapes(prefix, profile))
    evidence = detect_wan22(source)
    assert evidence is not None
    assert evidence.family_id == "dinkster.wan22"
    assert evidence.fields["profile"] == profile
    assert evidence.fields["input_channels"] == input_channels
    assert evidence.fields["output_channels"] == 48
    assert evidence.fields["full_ref"] is (reference_channels is not None)
    assert evidence.fields["ref_conv"] is (reference_channels is not None)
    if reference_channels is not None:
        assert evidence.fields["reference_channels"] == reference_channels


def test_wan21_does_not_steal_wan22_and_catalog_selects_it() -> None:
    source = HeaderSource(wan22_shapes())
    assert detect_wan21(source) is None
    result = builtin_family_registry().detect(source)
    assert result.best is not None
    assert result.best.family_id == "dinkster.wan22"
    assert not result.ambiguous


@pytest.mark.parametrize("dtype", (BFLOAT16, FLOAT8_E4M3))
def test_floating_checkpoint_storage_is_admitted(dtype: DType) -> None:
    assert detect_wan22(HeaderSource(wan22_shapes(), dtype=dtype)) is not None


def test_nonfloating_and_near_family_geometries_fail_closed() -> None:
    assert detect_wan22(HeaderSource(wan22_shapes(), dtype=INT64)) is None
    for key, shape in (
        ("patch_embedding.weight", (3072, 16, 1, 2, 2)),
        ("head.head.weight", (64, 3072)),
        ("blocks.0.ffn.0.weight", (13824, 3072)),
        ("blocks.29.modulation", (1, 6, 5120)),
    ):
        shapes = wan22_shapes()
        shapes[key] = shape
        assert detect_wan22(HeaderSource(shapes)) is None


@pytest.mark.parametrize(
    "marker",
    (
        "img_emb.emb_pos",
        "ref_conv.weight",
        "full_ref.weight",
        "vace_patch_embedding.weight",
        "control_adapter.conv.weight",
        "patch_embedding_mask.weight",
    ),
)
def test_other_wan_variants_fail_closed(marker: str) -> None:
    shapes = wan22_shapes()
    shapes[marker] = (1,)
    assert detect_wan22(HeaderSource(shapes)) is None


def test_duplicate_prefixes_and_behavior_metadata_are_refused() -> None:
    shapes = wan22_shapes()
    shapes.update(wan22_shapes("model.diffusion_model."))
    assert detect_wan22(HeaderSource(shapes)) is None
    assert (
        detect_wan22(
            HeaderSource(
                wan22_shapes(),
                metadata={"config": '{"transformer":{"model_type":"ti2v"}}'},
            )
        )
        is None
    )


def test_latent_sampling_and_family_facts_match_upstream() -> None:
    assert (
        WAN22_LATENT.channels,
        WAN22_LATENT.dimensions,
        WAN22_LATENT.temporal_causal,
        WAN22_LATENT.temporal_downscale,
        WAN22_LATENT.spatial_downscale,
        WAN22_LATENT.taesd_decoder,
    ) == (48, 3, True, 4, 16, "lighttaew2_2")
    assert WAN22_SIGMAS.shift == 8.0
    assert WAN22_SAMPLING.parameterization is Parameterization.FLOW
    assert WAN22_SAMPLING.sigma_min == WAN22_SIGMAS.sigma_min
    assert WAN22_SAMPLING.sigma_max == 1.0
    assert WAN22_SAMPLING.shift == 8.0
    assert WAN22.latent is WAN22_LATENT
    assert WAN22.sampling is WAN22_SAMPLING
    assert WAN22.memory_factor == 3072 / 2222
