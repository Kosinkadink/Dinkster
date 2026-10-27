"""CPU contracts for pre-encoded LTXV guides."""

from types import SimpleNamespace

import pytest
import torch
from dinkster_nodes_comfy.ltx_guides import LTXVAddLatentGuide


@pytest.mark.parametrize(
    ("index", "time_bounds"),
    [(-2, [(-16, -8), (-8, 0)]), (0, [(0, 1), (1, 9)]), (2, [(9, 17), (17, 25)])],
)
def test_latent_guide_preserves_timeline_conditioning_and_sources(index, time_bounds) -> None:
    samples = torch.arange(128 * 5 * 2 * 3, dtype=torch.float64).reshape(1, 128, 5, 2, 3)
    guide = torch.arange(128 * 2 * 2 * 3, dtype=torch.float64).reshape(1, 128, 2, 2, 3)
    mask = torch.full((1, 1, 5, 1, 1), 0.3)
    source = {"samples": samples, "noise_mask": mask, "extra": {"keep": 17}}
    before, guide_before, mask_before = samples.clone(), guide.clone(), mask.clone()
    old_positive = {"latent_shape": [1, 1, 1], "strength": 0.1}
    old_negative = {"latent_shape": [2, 1, 1], "strength": 0.9}
    positive = [[torch.ones(1), {"guide_attention_entries": [old_positive], "keep": "p"}]]
    negative = [[torch.zeros(1), {"guide_attention_entries": [old_negative], "keep": "n"}]]
    attention = torch.arange(6.0).reshape(1, 2, 3)
    output = LTXVAddLatentGuide.execute(
        positive=positive,
        negative=negative,
        vae=SimpleNamespace(downscale_index_formula=(8, 32, 16)),
        latent=source,
        guiding_latent={"samples": guide},
        latent_idx=index,
        strength=0.25,
        attention_mask=attention,
    )
    latent = output["latent"]
    assert latent["extra"] == {"keep": 17}
    assert latent["samples"].shape == (1, 128, 7, 2, 3)
    assert torch.equal(latent["samples"][:, :, :5], before)
    assert torch.equal(latent["samples"][:, :, 5:], guide_before)
    assert torch.equal(latent["noise_mask"][:, :, :5], mask_before)
    assert torch.equal(latent["noise_mask"][:, :, 5:], torch.full((1, 1, 2, 1, 1), 0.75))
    for name, old, label in (("positive", old_positive, "p"), ("negative", old_negative, "n")):
        metadata = output[name][0][1]
        assert metadata["keep"] == label
        assert metadata["guide_attention_entries"][0] == old
        entry = metadata["guide_attention_entries"][1]
        assert entry["latent_shape"] == [2, 2, 3]
        assert entry["pre_filter_count"] == 12
        assert entry["strength"] == 0.25
        assert torch.equal(entry["pixel_mask"], attention[None, None])
        coordinates = metadata["keyframe_idxs"]
        assert coordinates.shape == (1, 3, 12, 2)
        assert coordinates[0, 0, ::6].tolist() == [list(bounds) for bounds in time_bounds]
        assert coordinates[0, 1, :6].tolist() == [[0, 32]] * 3 + [[32, 64]] * 3
        assert coordinates[0, 2, :3].tolist() == [[0, 16], [16, 32], [32, 48]]
    assert positive[0][1] == {"guide_attention_entries": [old_positive], "keep": "p"}
    assert negative[0][1] == {"guide_attention_entries": [old_negative], "keep": "n"}
    negative_coordinates = output["negative"][0][1]["keyframe_idxs"].clone()
    output["positive"][0][1]["keyframe_idxs"].zero_()
    assert torch.equal(output["negative"][0][1]["keyframe_idxs"], negative_coordinates)
    latent["samples"].zero_()
    latent["noise_mask"].zero_()
    assert torch.equal(samples, before) and torch.equal(guide, guide_before)
    assert torch.equal(mask, mask_before)


def test_sparse_latent_guide_keeps_grid_and_expands_only_spatial_end_coordinates() -> None:
    source = {"samples": torch.zeros(1, 128, 4, 4, 6)}
    guide = torch.arange(128 * 2 * 3.0).reshape(1, 128, 1, 2, 3) + 1
    original_indices = torch.full((1, 3, 1, 2), -99)
    positive = [[torch.ones(1), {"keyframe_idxs": original_indices}]]
    output = LTXVAddLatentGuide.execute(
        positive=positive,
        negative=[[torch.zeros(1), {}]],
        vae=SimpleNamespace(downscale_index_formula=(8, 32, 16)),
        latent=source,
        guiding_latent={"samples": guide},
        latent_idx=1,
        strength=0.625,
    )
    samples, mask = output["latent"]["samples"], output["latent"]["noise_mask"]
    assert torch.equal(samples[:, :, -1:, ::2, ::2], guide)
    assert torch.count_nonzero(samples[:, :, -1:, 1::2]) == 0
    assert torch.count_nonzero(samples[:, :, -1:, :, 1::2]) == 0
    expected = torch.full((1, 1, 1, 4, 6), -1.625)
    expected[..., ::2, ::2] = 0.375
    torch.testing.assert_close(mask[:, :, -1:], expected, rtol=0, atol=0)
    assert torch.equal(mask[:, :, :4], torch.ones(1, 1, 4, 4, 6))
    pos, neg = output["positive"][0][1], output["negative"][0][1]
    assert torch.equal(pos["keyframe_idxs"][:, :, :1], original_indices)
    assert pos["keyframe_idxs"].shape[2] == 25
    assert neg["keyframe_idxs"].shape[2] == 24
    assert neg["keyframe_idxs"][0, :, 0].tolist() == [[1, 2], [0, 64], [0, 32]]
    assert neg["keyframe_idxs"][0, :, 1].tolist() == [[1, 2], [0, 64], [16, 48]]
    assert pos["guide_attention_entries"][0]["pre_filter_count"] == 24
    assert pos["guide_attention_entries"][0]["latent_shape"] == [1, 2, 3]
    assert torch.equal(original_indices, torch.full_like(original_indices, -99))


@pytest.mark.parametrize(
    ("target", "guide", "index", "message"),
    [
        ((1, 128, 3, 4), (1, 128, 1, 2, 2), 0, "latent must be a 5D"),
        ((1, 128, 3, 4, 6), (1, 128, 2, 2), 0, "guiding_latent must be a 5D"),
        ((1, 128, 3, 4, 6), (1, 128, 2, 2, 3), 2, "runs past"),
        ((1, 128, 3, 5, 6), (1, 128, 1, 2, 3), 0, "must divide"),
        ((1, 128, 3, 4, 6), (1, 128, 1, 2, 2), 0, "ratio must be square"),
        ((1, 129, 3, 4, 6), (1, 128, 1, 2, 3), 0, "combined AV"),
    ],
)
def test_invalid_latent_guides_reject_without_mutating(target, guide, index, message) -> None:
    source, reference = torch.ones(target), torch.full(guide, 2.0)
    with pytest.raises(ValueError, match=message):
        LTXVAddLatentGuide.execute(
            positive=[],
            negative=[],
            vae=SimpleNamespace(downscale_index_formula=(8, 32, 32)),
            latent={"samples": source},
            guiding_latent={"samples": reference},
            latent_idx=index,
            strength=1.0,
        )
    assert torch.equal(source, torch.ones(target))
    assert torch.equal(reference, torch.full(guide, 2.0))
