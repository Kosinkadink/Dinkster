"""Wan SCAIL model contracts and execution tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import WAN21_SCAIL2_14B, WAN21_SCAIL_14B, Wan21Config, wan21_layout
from dinkster_inference_torch.operations import ResidencyRouted
from dinkster_inference_torch.wan21_scail import WanScailModel
from golden_files import load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "wan21_scail_goldens.json")

# Fresh pinned ComfyUI outputs were bit-identical to Dinkster on the failing Linux
# host, isolating issue #564 to cross-host stored-golden drift. Each case
# measured at most 5.97e-7 absolute drift locally, while a second host measured
# about 5e-7 overall. The per-case limits leave at least 1.34x headroom.
SCAIL_GOLDEN_ATOL = {
    "scail_animation": 8e-7,
    "scail2_animation": 8e-7,
    "scail2_replacement": 8e-7,
}


def _decode(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=getattr(torch, payload["dtype"])).reshape(
        payload["shape"]
    )


def _golden_model(case: str) -> WanScailModel:
    spec = GOLDENS["cases"][case]
    config = Wan21Config(**GOLDENS["config"], model_variant=spec["variant"])
    model = WanScailModel(config)
    model.load_state_dict(fill_state_dict(spec["state_dict"]), strict=True)
    return model


def test_golden_provenance_is_exact_and_regeneration_is_documented() -> None:
    meta = GOLDENS["_meta"]
    assert meta["attention"] == "attention_pytorch"
    assert meta["commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    assert meta["reference"] == "ComfyUI SCAILWanModel/SCAIL2WanModel"
    assert meta["rope"] == "pure torch (model_management.in_training=True)"
    assert meta["source"] == "git archive of exact commit object"
    if sys.platform.startswith("linux"):
        assert meta == {
            "attention": "attention_pytorch",
            "commit": "b78cec879b9460d5cb25228a83a942fb78d2cd24",
            "python": "3.12.11",
            "reference": "ComfyUI SCAILWanModel/SCAIL2WanModel",
            "rope": "pure torch (model_management.in_training=True)",
            "source": "git archive of exact commit object",
            "torch": "2.13.0+cpu",
        }


@pytest.mark.parametrize("case", sorted(GOLDENS["cases"]))
def test_reduced_forward_matches_stored_reference(case: str) -> None:
    spec = GOLDENS["cases"][case]
    reference_mask_shape = spec["reference_mask_shape"]
    driving_mask_shape = spec["driving_mask_shape"]
    actual = _golden_model(case)(
        hashed_input(f"{case}:x", spec["input_shape"]),
        torch.tensor(spec["timesteps"], dtype=torch.float32),
        hashed_input(f"{case}:context", spec["context_shape"]),
        hashed_input(f"{case}:vision", spec["vision_shape"]),
        reference_latent=hashed_input(f"{case}:reference", spec["reference_shape"]),
        pose_latents=hashed_input(f"{case}:pose", spec["pose_shape"]),
        reference_mask=(
            None
            if reference_mask_shape is None
            else hashed_input(f"{case}:reference-mask", reference_mask_shape)
        ),
        driving_mask=(
            None
            if driving_mask_shape is None
            else hashed_input(f"{case}:driving-mask", driving_mask_shape)
        ),
        replacement=spec["replacement"],
    )

    torch.testing.assert_close(
        actual,
        _decode(spec["output"]),
        rtol=0.0,
        atol=SCAIL_GOLDEN_ATOL[case],
    )


def _config(variant: str) -> Wan21Config:
    return Wan21Config(
        model_type="i2v",
        model_variant=cast("Any", variant),
        in_channels=20,
        hidden_size=12,
        ffn_hidden_size=24,
        num_heads=1,
        num_layers=2,
        text_dim=8,
        time_freq_dim=4,
    )


def _model(variant: str) -> WanScailModel:
    model = WanScailModel(_config(variant))
    state = [(key, tuple(value.shape)) for key, value in sorted(model.state_dict().items())]
    model.load_state_dict(fill_state_dict(state), strict=True)
    return model


@pytest.mark.parametrize("config", (WAN21_SCAIL_14B, WAN21_SCAIL2_14B))
def test_official_state_dict_matches_header_contract(config: Wan21Config) -> None:
    with torch.device("meta"):
        model = WanScailModel(config)
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    owners = [
        module
        for module in model.modules()
        if tuple(module.parameters(recurse=False)) or tuple(module.buffers(recurse=False))
    ]

    assert actual == wan21_layout(config)
    assert len(actual) == (1307 if config.model_variant == "scail2" else 1305)
    assert all(isinstance(module, ResidencyRouted) for module in owners)


@pytest.mark.parametrize(("variant", "replacement"), (("scail", False), ("scail2", True)))
def test_reduced_model_runs_reference_and_half_resolution_pose_streams(
    variant: str, replacement: bool
) -> None:
    model = _model(variant)
    kwargs: dict[str, object] = {
        "reference_latent": hashed_input("scail:reference", (1, 20, 2, 5, 6)),
        "pose_latents": hashed_input("scail:pose", (1, 20, 2, 3, 3)),
        "replacement": replacement,
    }
    if variant == "scail2":
        kwargs.update(
            reference_mask=hashed_input("scail:reference-mask", (1, 28, 4, 5, 6)),
            driving_mask=hashed_input("scail:driving-mask", (1, 28, 2, 3, 3)),
        )

    output = model(
        hashed_input("scail:x", (1, 20, 2, 5, 6)),
        torch.tensor([0.375]),
        hashed_input("scail:context", (1, 4, 8)),
        hashed_input("scail:vision", (1, 3, 1280)),
        **kwargs,  # type: ignore[arg-type]
    )

    assert output.shape == (1, 16, 2, 5, 6)
    assert torch.isfinite(output).all()


def test_rope_layout_matches_animation_and_replacement_coordinates() -> None:
    model = _model("scail2")

    class Capture(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[torch.Tensor] = []

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            self.calls.append(ids.detach().clone())
            return ids[..., :1].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    capture = Capture()
    cast("Any", model).rope_embedder = capture
    x = torch.empty((1, 12, 5, 2, 2))

    animation = model._scail_rope(  # pyright: ignore[reportPrivateUsage]
        (3, 2, 2), 2, (3, 1, 1), (2.0, 2.0), x, replacement=False
    )
    main, pose = capture.calls
    assert animation.shape[1] == 23
    assert torch.equal(main[0, :, 0], torch.repeat_interleave(torch.arange(5.0), 4))
    assert torch.equal(pose[0, :, 0], torch.arange(2.0, 5.0))
    assert torch.equal(pose[0, :, 1], torch.full((3,), 0.5))
    assert torch.equal(pose[0, :, 2], torch.full((3,), 120.5))

    capture.calls.clear()
    replacement = model._scail_rope(  # pyright: ignore[reportPrivateUsage]
        (3, 2, 2), 2, (3, 1, 1), (2.0, 2.0), x, replacement=True
    )
    references, target, pose = capture.calls
    assert replacement.shape[1] == 23
    assert torch.equal(references[0, :, 0], torch.repeat_interleave(torch.arange(2.0), 4))
    assert torch.equal(references[0, :, 1], torch.tensor([120.0, 120.0, 121.0, 121.0] * 2))
    assert torch.equal(target[0, :, 0], torch.repeat_interleave(torch.arange(1.0, 4.0), 4))
    assert torch.equal(pose[0, :, 0], torch.arange(1.0, 4.0))


def test_scail_refuses_cross_variant_masks_and_framewise_timesteps() -> None:
    model = _model("scail")
    inputs = (
        torch.zeros((1, 20, 2, 4, 4)),
        torch.tensor([1.0]),
        torch.zeros((1, 3, 8)),
    )
    with pytest.raises(ValueError, match="only by SCAIL2"):
        model(*inputs, reference_mask=torch.zeros((1, 28, 2, 4, 4)))
    with pytest.raises(ValueError, match="one value per batch"):
        model(inputs[0], torch.ones((1, 2)), inputs[2])


def test_scail_model_requires_scail_configuration() -> None:
    with pytest.raises(ValueError, match="requires a SCAIL"):
        WanScailModel(Wan21Config())
