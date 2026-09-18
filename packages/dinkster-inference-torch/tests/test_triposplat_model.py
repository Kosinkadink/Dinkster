"""The native TripoSplat flow model against the executed reference.

Every golden in goldens/triposplat_goldens.json was produced by
RUNNING the reference TripoSplat flow denoiser
(comfy/ldm/triposplat/model.py LatentSeqMMFlowModel @ the audited
baseline, tools/gen_triposplat_goldens.py) with attention forced to
pytorch SDPA. Weights come from the shared deterministic hash
(unet_fill.py) and inputs from its ``hashed_input`` namespace.

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
from dinkster_inference import TRIPOSPLAT_CONFIG, TripoSplatConfig
from dinkster_inference.triposplat import triposplat_layout
from dinkster_inference_torch import ResidencyRouted, TripoSplatModel, enroll_component
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "triposplat_goldens.json").read_text())

CASES = sorted(name for name in GOLDENS["cases"] if name.startswith("dit_"))


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> TripoSplatConfig:
    """The golden case's generator kwargs as a reduced TripoSplatConfig.

    The frozen TripoSplatConfig only represents the published
    architecture, so the tiny proof architecture is duck-typed with the
    same field names.
    """
    spec = GOLDENS["cases"][case]["config"]
    return cast(
        TripoSplatConfig,
        SimpleNamespace(
            q_token_length=spec["q_token_length"],
            latent_channels=spec["in_channels"],
            model_channels=spec["model_channels"],
            cond_channels=spec["cond_channels"],
            cond2_channels=spec["cond2_channels"],
            num_blocks=spec["num_blocks"],
            num_refiner_blocks=spec["num_refiner_blocks"],
            attention_heads=spec["model_channels"] // spec["num_head_channels"],
            attention_head_dim=spec["num_head_channels"],
            cam_channels=spec["cam_channels"],
            mlp_ratio=spec["mlp_ratio"],
            repo_hidden_size=int(spec["model_channels"] * 0.125),
        ),
    )


def build_model(case: str) -> TripoSplatModel:
    model = TripoSplatModel(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def case_inputs(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    spec = GOLDENS["cases"][case]
    config = spec["config"]
    batch = spec["batch"]
    latent = hashed_input(
        f"{case}:latent", (batch, config["q_token_length"], config["in_channels"])
    )
    camera = hashed_input(f"{case}:camera", (batch, 1, config["cam_channels"]))
    timesteps = torch.tensor(spec["timesteps"], dtype=torch.float32)
    context = hashed_input(
        f"{case}:context", (batch, spec["context_rows"], config["cond_channels"])
    )
    reference_latent = None
    if spec["use_reference_latent"]:
        reference_latent = hashed_input(
            f"{case}:reference_latent", (batch, config["cond2_channels"], 2, 2)
        )
    return latent, camera, timesteps, context, reference_latent


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted(
        (key, list(value.shape))
        for key, value in TripoSplatModel(case_config(case)).state_dict().items()
    )
    assert ours == golden_entries(case)


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted(
        (key, list(shape)) for key, shape in triposplat_layout(case_config(case)).items()
    )
    assert predicted == golden_entries(case)


def test_full_size_module_matches_reference_layout() -> None:
    """The real published architecture, constructed on the meta device
    (initless factories never touch the storage), against the reference
    model's own full-size listing."""
    with torch.device("meta"):
        model = TripoSplatModel(TRIPOSPLAT_CONFIG)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["triposplat_dit"]]
    assert ours == golden
    predicted = sorted((key, list(shape)) for key, shape in triposplat_layout().items())
    assert predicted == golden


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_forward_refiners_and_block_match_executed_reference(case: str) -> None:
    model = build_model(case)
    latent, camera, timesteps, context, reference_latent = case_inputs(case)
    observed: dict[str, torch.Tensor] = {}
    hooks = [
        model.noise_refiner[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(
                noise_refiner=cast(torch.Tensor, output)
            )
        ),
        model.context_refiner[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(
                context_refiner=cast(torch.Tensor, output)
            )
        ),
        model.blocks[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(block=cast(torch.Tensor, output))
        ),
    ]
    try:
        with torch.no_grad():
            out_latent, out_camera = model(latent, camera, timesteps, context, reference_latent)
    finally:
        for hook in hooks:
            hook.remove()
    golden = GOLDENS["cases"][case]
    for name in ("noise_refiner", "context_refiner", "block"):
        torch.testing.assert_close(
            observed[name],
            dec(golden["intermediates"][name]),
            rtol=1e-4,
            atol=1e-5,
        )
    torch.testing.assert_close(out_latent, dec(golden["latent_output"]), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out_camera, dec(golden["camera_output"]), rtol=1e-4, atol=1e-5)


def test_fp16_forward_stays_finite() -> None:
    case = CASES[0]
    model = build_model(case).half()
    latent, camera, timesteps, context, reference_latent = case_inputs(case)
    with torch.no_grad():
        out_latent, out_camera = model(
            latent.half(),
            camera.half(),
            timesteps.half(),
            context.half(),
            None if reference_latent is None else reference_latent.half(),
        )
    assert out_latent.dtype == torch.float16
    assert out_camera.dtype == torch.float16
    assert torch.isfinite(out_latent).all()
    assert torch.isfinite(out_camera).all()


# ------------------------------------------------------ residency


def test_enrolled_prefetch_covers_exactly_the_persistent_state() -> None:
    """The anchor embedding is a derived constant outside the residency
    store; prefetch must request every stored key and nothing else."""
    case = CASES[0]
    model = build_model(case)
    assert "pos_emb" not in model.state_dict()

    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    latent, camera, timesteps, context, reference_latent = case_inputs(case)
    with torch.no_grad():
        resident = model(latent, camera, timesteps, context, reference_latent)

    mechanism.unload()
    requests: dict[str, torch.dtype | None] = {}
    for module in model.modules():
        if not isinstance(module, ResidencyRouted):
            continue
        prefetch = module.residency_prefetch()
        if prefetch is None:
            continue
        for key, dtype in prefetch[1]:
            assert key not in requests
            requests[key] = dtype
    assert requests == {key: torch.float32 for key in model.state_dict()}

    with torch.no_grad():
        offloaded = model(latent, camera, timesteps, context, reference_latent)
    assert torch.equal(offloaded[0], resident[0])
    assert torch.equal(offloaded[1], resident[1])


# ------------------------------------------------------- refusals


def _tiny_spec() -> SimpleNamespace:
    return cast(SimpleNamespace, case_config(CASES[0]))


def test_model_width_must_factor_into_heads() -> None:
    spec = _tiny_spec()
    spec.attention_head_dim += 1
    with pytest.raises(ValueError, match="model width"):
        TripoSplatModel(cast(TripoSplatConfig, spec))


def test_head_dimension_must_be_even_and_cover_three_rope_axes() -> None:
    spec = _tiny_spec()
    spec.attention_heads = 6
    spec.attention_head_dim = 4
    with pytest.raises(ValueError, match="at least 6"):
        TripoSplatModel(cast(TripoSplatConfig, spec))


def test_forward_refuses_malformed_streams() -> None:
    case = CASES[0]
    model = build_model(case)
    latent, camera, timesteps, context, _ = case_inputs(case)
    with pytest.raises(ValueError, match="latent must have shape"):
        model(latent[:, :-1], camera, timesteps, context)
    with pytest.raises(ValueError, match="camera must have shape"):
        model(latent, camera[..., :-1], timesteps, context)
    with pytest.raises(ValueError, match="timesteps must have shape"):
        model(latent, camera, timesteps[:1], context)
    with pytest.raises(ValueError, match="context must be"):
        model(latent, camera, timesteps, context[..., :-1])


def test_forward_refuses_malformed_reference_latents() -> None:
    case = CASES[0]
    model = build_model(case)
    latent, camera, timesteps, context, _ = case_inputs(case)
    config = case_config(case)
    bad_channels = torch.zeros(
        latent.shape[0], config.cond2_channels + 1, 2, 2, dtype=torch.float32
    )
    with pytest.raises(ValueError, match="reference latent must be"):
        model(latent, camera, timesteps, context, bad_channels)
    too_many_tokens = torch.zeros(latent.shape[0], config.cond2_channels, 4, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="must not exceed the context rows"):
        model(latent, camera, timesteps, context, too_many_tokens)
