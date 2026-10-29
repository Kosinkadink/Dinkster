"""The native Anima diffusion transformer against the executed reference.

Every golden in goldens/anima_goldens.json was produced by RUNNING
the reference Anima (comfy/ldm/anima/model.py: the Cosmos Predict2
MiniTrainDIT backbone plus the LLM adapter @ the audited baseline,
tools/gen_anima_goldens.py) with attention forced to pytorch SDPA;
the backbone's norm+rope runs dinkster_kitchen's rms_rope_split_half,
which dispatches to the deterministic eager kernel on CPU. Weights
come from the shared deterministic hash (unet_fill.py - every rank-1
weight is a norm scale, so the rank rule holds) and inputs from its
``hashed_input`` namespace.

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
from dinkster_inference import ANIMA_CONFIG, AnimaConfig
from dinkster_inference.anima import anima_layout
from dinkster_inference_torch import AnimaModel, CosmosPredict2Geometry
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "anima_goldens.json").read_text())

CASES = sorted(GOLDENS["cases"])


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> AnimaConfig:
    """The golden case's generator kwargs as a reduced AnimaConfig.

    The frozen AnimaConfig only represents the full 2B profile, so the
    tiny proof architecture is duck-typed with the same field names.
    """
    spec = GOLDENS["cases"][case]["config"]
    adapter = GOLDENS["cases"][case]["adapter_config"]
    patch = (spec["patch_temporal"], spec["patch_spatial"], spec["patch_spatial"])
    patch_volume = patch[0] * patch[1] * patch[2]
    return cast(
        AnimaConfig,
        SimpleNamespace(
            blocks=spec["num_blocks"],
            hidden_width=spec["model_channels"],
            attention_heads=spec["num_heads"],
            attention_head_dim=spec["model_channels"] // spec["num_heads"],
            context_width=spec["crossattn_emb_channels"],
            adaln_lora_dim=spec["adaln_lora_dim"],
            patchified_input_channels=(spec["in_channels"] + 1) * patch_volume,
            output_latent_channels=spec["out_channels"],
            patch=patch,
            latent_channels=spec["in_channels"],
            adapter_blocks=adapter["num_layers"],
            adapter_width=adapter["target_dim"],
            adapter_heads=adapter["num_heads"],
            adapter_head_dim=adapter["model_dim"] // adapter["num_heads"],
            adapter_source_width=adapter["source_dim"],
            adapter_vocabulary=32128,
        ),
    )


def build_model(case: str) -> AnimaModel:
    model = AnimaModel(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def case_inputs(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    spec = GOLDENS["cases"][case]
    config = spec["config"]
    adapter = spec["adapter_config"]
    batch = spec["batch"]
    x = hashed_input(
        f"{case}:x",
        (batch, config["in_channels"], spec["frames"], spec["height"], spec["width"]),
    )
    timesteps = torch.tensor(spec["timesteps"], dtype=torch.float32)
    context = hashed_input(
        f"{case}:context",
        (batch, spec["source_rows"], adapter["source_dim"]),
    )
    ids = torch.tensor(spec["t5xxl_ids"], dtype=torch.int64)
    weights = None
    if spec["t5xxl_weights"] is not None:
        weights = torch.tensor(spec["t5xxl_weights"], dtype=torch.float32)
    return x, timesteps, context, ids, weights


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted(
        (key, list(value.shape))
        for key, value in AnimaModel(case_config(case)).state_dict().items()
    )
    assert ours == golden_entries(case)


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in anima_layout(case_config(case)).items())
    assert predicted == golden_entries(case)


def test_full_size_module_matches_reference_layout() -> None:
    """The real 2B architecture, constructed on the meta device
    (initless factories never touch the storage), against the
    reference model's own full-size listing."""
    with torch.device("meta"):
        model = AnimaModel(ANIMA_CONFIG)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["anima_2b"]]
    assert ours == golden


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_forward_adapter_and_block_match_executed_reference(case: str) -> None:
    model = build_model(case)
    x, timesteps, context, ids, weights = case_inputs(case)
    observed: dict[str, torch.Tensor] = {}
    hooks = [
        model.llm_adapter.register_forward_hook(
            lambda _module, _inputs, output: observed.update(adapter=cast(torch.Tensor, output))
        ),
        model.blocks[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(block=cast(torch.Tensor, output))
        ),
    ]
    try:
        with torch.no_grad():
            output = model(x, timesteps, context, t5xxl_ids=ids, t5xxl_weights=weights)
    finally:
        for hook in hooks:
            hook.remove()
    golden = GOLDENS["cases"][case]
    for name in ("adapter", "block"):
        torch.testing.assert_close(
            observed[name],
            dec(golden["block_outputs"][name]),
            rtol=1e-4,
            atol=1e-5,
        )
    torch.testing.assert_close(output, dec(golden["output"]), rtol=1e-4, atol=1e-5)


def test_fused_and_autograd_rope_paths_agree() -> None:
    """The no-grad path runs dinkster_kitchen's fused norm+rope; under
    autograd the same math runs as eager norm then split-half
    rotation. Both must produce the same forward values."""
    case = CASES[0]
    model = build_model(case)
    x, timesteps, context, ids, weights = case_inputs(case)
    with torch.no_grad():
        fused = model(x, timesteps, context, t5xxl_ids=ids, t5xxl_weights=weights)
    eager = model(x, timesteps, context, t5xxl_ids=ids, t5xxl_weights=weights)
    torch.testing.assert_close(fused, eager.detach(), rtol=1e-5, atol=1e-6)


def test_fp16_blocks_keep_a_float32_residual_stream() -> None:
    """The residual stream carries large values: under an fp16 model
    the block outputs stay float32 and only the final projection
    returns to fp16."""
    case = CASES[0]
    model = build_model(case).half()
    x, timesteps, context, ids, weights = case_inputs(case)
    observed: dict[str, torch.Tensor] = {}
    hook = model.blocks[-1].register_forward_hook(
        lambda _module, _inputs, output: observed.update(block=cast(torch.Tensor, output))
    )
    try:
        with torch.no_grad():
            output = model(
                x.half(),
                timesteps.half(),
                context.half(),
                t5xxl_ids=ids,
                t5xxl_weights=None if weights is None else weights.half(),
            )
    finally:
        hook.remove()
    assert observed["block"].dtype == torch.float32
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


# ------------------------------------------------- forward contract


def test_short_adapter_output_pads_to_the_t5_context() -> None:
    """Seven target ids still condition all 512 cross-attention rows:
    the adapter output zero-pads to the T5-XXL context length."""
    case = CASES[0]
    model = build_model(case)
    x, _, context, ids, weights = case_inputs(case)
    with torch.no_grad():
        padded = model.preprocess_text_embeds(context, ids, weights)
    assert padded.shape == (x.shape[0], 512, case_config(case).context_width)
    assert torch.all(padded[:, ids.shape[1] :] == 0)


def test_preprocessed_context_matches_the_id_path() -> None:
    case = CASES[0]
    model = build_model(case)
    x, timesteps, context, ids, weights = case_inputs(case)
    with torch.no_grad():
        direct = model(x, timesteps, context, t5xxl_ids=ids, t5xxl_weights=weights)
        preprocessed = model.preprocess_text_embeds(context, ids, weights)
        replay = model(x, timesteps, preprocessed)
    torch.testing.assert_close(direct, replay)


# ------------------------------------------------------- refusals


def _tiny_spec() -> SimpleNamespace:
    return cast(SimpleNamespace, case_config(CASES[0]))


def test_adapter_width_must_feed_cross_attention_unprojected() -> None:
    spec = _tiny_spec()
    spec.context_width = spec.adapter_width * 2
    with pytest.raises(ValueError, match="adapter output width"):
        AnimaModel(cast(AnimaConfig, spec))


def test_hidden_width_must_factor_into_heads() -> None:
    spec = _tiny_spec()
    spec.attention_head_dim += 1
    with pytest.raises(ValueError, match="hidden width"):
        AnimaModel(cast(AnimaConfig, spec))


def test_spatial_patch_must_be_square() -> None:
    spec = _tiny_spec()
    spec.patch = (1, 2, 4)
    with pytest.raises(ValueError, match="square"):
        AnimaModel(cast(AnimaConfig, spec))


def test_patchified_channels_must_cover_the_padding_mask() -> None:
    spec = _tiny_spec()
    spec.patchified_input_channels += 1
    with pytest.raises(ValueError, match="patchified input channels"):
        AnimaModel(cast(AnimaConfig, spec))


def test_adapter_width_must_factor_into_adapter_heads() -> None:
    spec = _tiny_spec()
    spec.adapter_head_dim += 1
    with pytest.raises(ValueError, match="adapter width"):
        AnimaModel(cast(AnimaConfig, spec))


def test_geometry_refuses_head_dims_without_three_rope_axes() -> None:
    with pytest.raises(ValueError, match="three rope axes"):
        CosmosPredict2Geometry(
            in_channels=4,
            out_channels=4,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=16,
            num_blocks=1,
            num_heads=2,
            crossattn_emb_channels=16,
            adaln_lora_dim=8,
        )


def test_adapter_refuses_malformed_sources_and_ids() -> None:
    case = CASES[0]
    model = build_model(case)
    _, _, context, ids, _ = case_inputs(case)
    with pytest.raises(ValueError, match="source hidden states must be"):
        model.llm_adapter(context[..., :-1], ids)
    with pytest.raises(ValueError, match="floating-point"):
        model.llm_adapter(context.to(torch.int64), ids)
    with pytest.raises(ValueError, match="integer"):
        model.llm_adapter(context, ids.float())
    with pytest.raises(ValueError, match="source batch"):
        model.llm_adapter(torch.cat((context, context)), ids)


def test_context_rows_must_match_the_declared_width() -> None:
    case = CASES[0]
    model = build_model(case)
    x, timesteps, context, _, _ = case_inputs(case)
    with pytest.raises(ValueError, match="context must be"):
        model(x, timesteps, context)
