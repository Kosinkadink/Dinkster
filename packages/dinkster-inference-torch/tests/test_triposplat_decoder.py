"""The native TripoSplat octree gaussian decoder against the executed
reference.

Every golden in goldens/triposplat_goldens.json was produced by
RUNNING the reference decoder stack (comfy/ldm/triposplat/vae.py and
gaussian.py @ the audited baseline, tools/gen_triposplat_goldens.py)
with attention forced to pytorch SDPA. The octree and elastic decoders
were instantiated STANDALONE there (no ``octree.`` / ``gs.`` prefix on
the state-dict keys), so the replay tests instantiate the Dinkster
submodules standalone the same way: the deterministic hash fill
(unet_fill.py) keys on the key names. Every random draw goes through
an explicitly seeded CPU generator.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import TripoSplatGaussianDecoderConfig
from dinkster_inference.triposplat import triposplat_gaussian_decoder_layout
from dinkster_inference_torch import (
    INITLESS,
    CastOperations,
    ElasticGaussianDecoder,
    OctreeGaussianDecoder,
    OctreeProbabilityDecoder,
    render_splat_tensors,
    sample_octree_points,
    select_attention,
)
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "triposplat_goldens.json").read_text())

_ATTENTION = select_attention("vae").kernel
_SPLAT_FIELDS = ("positions", "scales", "rotations", "opacities", "sh")


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def entries(listing: list[list[Any]]) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in listing]


def tiny_config(**overrides: Any) -> TripoSplatGaussianDecoderConfig:
    """The golden generator's tiny octree/elastic kwargs as a reduced
    TripoSplatGaussianDecoderConfig.

    The frozen config only represents the published architecture, so
    the tiny proof architecture is duck-typed with the same field
    names. ``latent_channels`` is the generator's ``cond_channels``
    (both decoders cross-attend over it); the elastic representation
    packs 8 gaussians into 120 feature channels.
    """
    spec = GOLDENS["cases"]["octree_logits"]["config"]
    representation = GOLDENS["cases"]["elastic_features"]["representation"]
    gaussians = representation["num_gaussians"]
    fields: dict[str, Any] = {
        "model_channels": spec["model_channels"],
        "latent_channels": spec["cond_channels"],
        "octree_blocks": spec["num_blocks"],
        "gaussian_blocks": GOLDENS["cases"]["elastic_features"]["config"]["num_blocks"],
        "attention_heads": spec["num_heads"],
        "attention_head_dim": spec["num_head_channels"],
        "mlp_ratio": int(spec["mlp_ratio"]),
        "gaussians_per_point": gaussians,
        "feature_channels": gaussians * 15,
        "max_voxel_level": 8,
    }
    fields.update(overrides)
    return cast(TripoSplatGaussianDecoderConfig, SimpleNamespace(**fields))


def build_octree() -> OctreeProbabilityDecoder:
    decoder = OctreeProbabilityDecoder(
        tiny_config(), operations=INITLESS, attention_kernel=_ATTENTION
    )
    decoder.load_state_dict(
        fill_state_dict(entries(GOLDENS["cases"]["octree_logits"]["state_dict"])), strict=True
    )
    return decoder


def build_elastic() -> ElasticGaussianDecoder:
    decoder = ElasticGaussianDecoder(
        tiny_config(), operations=INITLESS, attention_kernel=_ATTENTION
    )
    decoder.load_state_dict(
        fill_state_dict(entries(GOLDENS["cases"]["elastic_features"]["state_dict"])), strict=True
    )
    return decoder


def build_combined() -> OctreeGaussianDecoder:
    decoder = OctreeGaussianDecoder(tiny_config(), operations=INITLESS, attention_kernel=_ATTENTION)
    listing = sorted((key, list(value.shape)) for key, value in decoder.state_dict().items())
    decoder.load_state_dict(fill_state_dict(listing), strict=True)
    return decoder


def cond_input(case: str) -> torch.Tensor:
    spec = GOLDENS["cases"][case]
    channels = GOLDENS["cases"]["octree_logits"]["config"]["cond_channels"]
    return hashed_input(f"{case}:cond", (spec["batch"], spec["cond_rows"], channels))


# ------------------------------------------------------ key layout


def test_standalone_state_dicts_match_executed_reference() -> None:
    octree = OctreeProbabilityDecoder(
        tiny_config(), operations=INITLESS, attention_kernel=_ATTENTION
    )
    ours = sorted((key, list(value.shape)) for key, value in octree.state_dict().items())
    assert ours == entries(GOLDENS["cases"]["octree_logits"]["state_dict"])
    elastic = ElasticGaussianDecoder(
        tiny_config(), operations=INITLESS, attention_kernel=_ATTENTION
    )
    ours = sorted((key, list(value.shape)) for key, value in elastic.state_dict().items())
    assert ours == entries(GOLDENS["cases"]["elastic_features"]["state_dict"])


def test_full_size_module_matches_reference_layout() -> None:
    """The real published architecture, constructed on the meta device
    (initless factories never touch the storage), against the reference
    decoder's own full-size listing."""
    with torch.device("meta"):
        decoder = OctreeGaussianDecoder()
    ours = sorted((key, list(value.shape)) for key, value in decoder.state_dict().items())
    golden = entries(GOLDENS["layouts"]["octree_gaussian_decoder"])
    assert ours == golden
    predicted = sorted(
        (key, list(shape)) for key, shape in triposplat_gaussian_decoder_layout().items()
    )
    assert predicted == golden


# ---------------------------------------------------- golden replay


def test_octree_logits_match_executed_reference() -> None:
    spec = GOLDENS["cases"]["octree_logits"]
    octree = build_octree()
    points = hashed_input("octree_logits:points", (spec["batch"], spec["point_rows"], 3))
    points = points * 0.25 + 0.5
    levels = torch.tensor(spec["levels"], dtype=torch.long)
    with torch.no_grad():
        logits = octree(points, levels, cond_input("octree_logits"))
    torch.testing.assert_close(logits, dec(spec["logits"]), rtol=1e-4, atol=1e-5)


def test_octree_sample_matches_executed_reference() -> None:
    """The seeded descent replays exactly: systematic resampling and
    the leaf jitter consume the generator in the reference order, and
    the tiny logits sit far enough from the reference's bin edges that
    the searchsorted assignments cannot flip."""
    spec = GOLDENS["cases"]["octree_sample"]
    octree = build_octree()
    with torch.no_grad():
        sampled = sample_octree_points(
            octree,
            cond_input("octree_sample"),
            num_points=spec["num_points"],
            level=spec["level"],
            generator=torch.Generator().manual_seed(spec["seed"]),
        )
    torch.testing.assert_close(sampled.points, dec(spec["points"]), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sampled.log_probs, dec(spec["log_probs"]), rtol=1e-4, atol=1e-5)


def test_elastic_features_and_offsets_match_executed_reference() -> None:
    spec = GOLDENS["cases"]["elastic_features"]
    elastic = build_elastic()
    points = hashed_input("elastic_features:points", (spec["batch"], spec["point_rows"], 3))
    points = points * 0.25 + 0.5
    with torch.no_grad():
        features = elastic(points, cond_input("elastic_features"))
        offsets = elastic.offsets(features)
    torch.testing.assert_close(features, dec(spec["features"]), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(offsets, dec(spec["offsets"]), rtol=1e-4, atol=1e-5)


def test_splat_decode_matches_executed_reference() -> None:
    """The seeded end-to-end decode: octree descent, elastic features,
    and activation into render-ready splat tensors."""
    spec = GOLDENS["cases"]["splat_decode"]
    octree = build_octree()
    elastic = build_elastic()
    cond = cond_input("splat_decode")
    tokens = max(1, spec["num_gaussians"] // elastic.gaussians_per_point)
    with torch.no_grad():
        sampled = sample_octree_points(
            octree,
            cond,
            num_points=tokens,
            level=spec["level"],
            generator=torch.Generator().manual_seed(spec["seed"]),
        )
        features = elastic(sampled.points, cond)
        splats = render_splat_tensors(elastic, sampled.points, features)
    assert len(splats) == len(spec["splats"])
    for splat, golden in zip(splats, spec["splats"], strict=True):
        for field in _SPLAT_FIELDS:
            torch.testing.assert_close(
                getattr(splat, field), dec(golden[field]), rtol=1e-4, atol=1e-5
            )


# ------------------------------------------------- decode contract


def test_combined_decode_is_deterministic_under_one_seed() -> None:
    decoder = build_combined()
    channels = GOLDENS["cases"]["octree_logits"]["config"]["cond_channels"]
    latent = hashed_input("combined_decode:latent", (2, 7, channels))
    with torch.no_grad():
        first = decoder.decode(
            latent, num_gaussians=20, generator=torch.Generator().manual_seed(11), level=3
        )
        second = decoder.decode(
            latent, num_gaussians=20, generator=torch.Generator().manual_seed(11), level=3
        )
    for one, two in zip(first, second, strict=True):
        for field in _SPLAT_FIELDS:
            assert torch.equal(getattr(one, field), getattr(two, field))


def test_combined_decode_rounds_down_to_whole_tokens() -> None:
    """20 requested gaussians over 8 per point round down to 2 tokens,
    so 16 float32 gaussians come back per batch item."""
    decoder = build_combined()
    channels = GOLDENS["cases"]["octree_logits"]["config"]["cond_channels"]
    latent = hashed_input("combined_decode:latent", (2, 7, channels))
    with torch.no_grad():
        splats = decoder.decode(
            latent, num_gaussians=20, generator=torch.Generator().manual_seed(11), level=3
        )
    assert len(splats) == 2
    for splat in splats:
        assert splat.positions.shape == (16, 3)
        assert splat.scales.shape == (16, 3)
        assert splat.rotations.shape == (16, 4)
        assert splat.opacities.shape == (16, 1)
        assert splat.sh.shape == (16, 1, 3)
        for field in _SPLAT_FIELDS:
            value = getattr(splat, field)
            assert value.dtype == torch.float32
            assert torch.isfinite(value).all()


def test_combined_decode_casts_checkpoint_storage_to_compute_dtype() -> None:
    decoder = OctreeGaussianDecoder(
        tiny_config(), operations=CastOperations(torch.float32), attention_kernel=_ATTENTION
    )
    stored = {
        key: torch.zeros(value.shape, dtype=torch.float16)
        if value.is_floating_point()
        else torch.zeros(value.shape, dtype=value.dtype)
        for key, value in decoder.state_dict().items()
    }
    decoder.load_state_dict(stored, strict=True, assign=True)
    channels = GOLDENS["cases"]["octree_logits"]["config"]["cond_channels"]
    latent = torch.zeros((1, 7, channels), dtype=torch.float16)
    with torch.no_grad():
        splat = decoder.decode(
            latent, num_gaussians=8, generator=torch.Generator().manual_seed(11), level=1
        )[0]
    assert next(decoder.parameters()).dtype == torch.float16
    for field in _SPLAT_FIELDS:
        assert getattr(splat, field).dtype == torch.float32


# ------------------------------------------------------- refusals


def test_elastic_features_must_pack_fifteen_values_per_gaussian() -> None:
    with pytest.raises(ValueError, match="15 values per gaussian"):
        ElasticGaussianDecoder(
            tiny_config(feature_channels=121), operations=INITLESS, attention_kernel=_ATTENTION
        )


def test_decoder_width_must_factor_into_heads() -> None:
    with pytest.raises(ValueError, match="decoder width"):
        OctreeGaussianDecoder(
            tiny_config(attention_head_dim=17), operations=INITLESS, attention_kernel=_ATTENTION
        )


def test_decode_refuses_malformed_requests() -> None:
    decoder = build_combined()
    channels = GOLDENS["cases"]["octree_logits"]["config"]["cond_channels"]
    latent = hashed_input("combined_decode:latent", (2, 7, channels))
    generator = torch.Generator().manual_seed(11)
    with pytest.raises(ValueError, match="latent must be"):
        decoder.decode(latent[..., :-1], num_gaussians=20, generator=generator)
    with pytest.raises(ValueError, match="num_gaussians must be positive"):
        decoder.decode(latent, num_gaussians=0, generator=generator)
    with pytest.raises(ValueError, match="octree level"):
        decoder.decode(latent, num_gaussians=20, generator=generator, level=0)
    with pytest.raises(ValueError, match="octree level"):
        decoder.decode(latent, num_gaussians=20, generator=generator, level=9)


def test_render_refuses_malformed_tensors() -> None:
    elastic = build_elastic()
    points = torch.zeros(2, 6, 3)
    features = torch.zeros(2, 6, elastic.gaussians_per_point * 15)
    with pytest.raises(ValueError, match="points must be"):
        render_splat_tensors(elastic, points[..., :-1], features)
    with pytest.raises(ValueError, match="features must be"):
        render_splat_tensors(elastic, points, features[..., :-1])
    with pytest.raises(ValueError, match="features must be"):
        render_splat_tensors(elastic, points, features[:, :-1])
