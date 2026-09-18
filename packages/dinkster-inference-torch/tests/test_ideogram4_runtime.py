"""Ideogram 4 native conditioning, runtime, and dual-model routing."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    Conditioning,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    SamplingGuidance,
)
from dinkster_inference_torch import (
    Ideogram4Conditioning,
    Ideogram4DiffusionRuntime,
    Ideogram4DiT,
    Ideogram4RuntimeError,
    RoutedConditioning,
    ideogram4_conditioning_to_carrier,
    materialize_ideogram4_conditioning,
)
from dinkster_inference_torch import ideogram4_runtime as runtime_module
from dinkster_inference_torch.solvers import torch_sampler_registry


class RecordingIdeogram(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.value = value
        self.calls: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        ] = []

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self.calls.append((latent, timestep, context, attention_mask))
        return torch.full_like(latent, self.value)


def diffusion_runtime(model: RecordingIdeogram, suffix: str = "0") -> Ideogram4DiffusionRuntime:
    return Ideogram4DiffusionRuntime(
        cast("Ideogram4DiT", model),
        runtime_identity="native:dinkster.ideogram4:" + suffix * 64,
        compute_dtype=torch.float32,
    )


def condition(fill: float = 0.0) -> Ideogram4Conditioning:
    return Ideogram4Conditioning(torch.full((1, 3, 53248), fill), None)


def custom_request() -> CustomSamplingRequest[torch.Tensor]:
    descriptor = torch_sampler_registry().get("dinkster.euler")
    assert descriptor is not None
    return CustomSamplingRequest(descriptor, (), (1.0, 0.0))


def test_conditioning_carrier_round_trip_preserves_optional_mask() -> None:
    original = Ideogram4Conditioning(
        torch.arange(2 * 3 * 53248, dtype=torch.float32).reshape(2, 3, 53248),
        None,
        torch.tensor(((1, 1, 0), (1, 1, 1)), dtype=torch.long),
    )
    materialized = materialize_ideogram4_conditioning(
        ideogram4_conditioning_to_carrier(original), device="cpu"
    )
    assert torch.equal(materialized.embeddings, original.embeddings)
    assert materialized.attention_mask is not None
    assert original.attention_mask is not None
    assert torch.equal(materialized.attention_mask, original.attention_mask)
    assert materialized.pooled is None


def test_conditioning_evaluator_supports_text_and_image_only_lanes() -> None:
    model = RecordingIdeogram(1.0)
    runtime = diffusion_runtime(model)
    evaluator = runtime.conditioning_evaluation(compute_dtype=torch.float32)
    latent = torch.full((2, 128, 2, 2), 4.0)
    prepared = evaluator.prepare(condition(), cast("Any", "positive"))
    result = evaluator.evaluate(latent, 0.25, prepared)
    torch.testing.assert_close(result, torch.full_like(latent, 3.75))
    _, timestep, context, attention = model.calls[-1]
    assert timestep.tolist() == [0.25, 0.25]
    assert context is not None and context.shape == (2, 3, 53248)
    assert attention is None

    image_only = evaluator.prepare(runtime.image_only_conditioning(), cast("Any", "negative"))
    evaluator.evaluate(latent, 0.5, image_only)
    assert model.calls[-1][2:] == (None, None)


def test_conditioning_batch_matches_separate_evaluation() -> None:
    model = RecordingIdeogram(1.0)
    evaluator = diffusion_runtime(model).conditioning_evaluation(compute_dtype=torch.float32)
    assert evaluator.evaluate_batch is not None
    latent = torch.arange(1024, dtype=torch.float32).reshape(2, 128, 2, 2)
    conditions = tuple(
        evaluator.prepare(condition(fill), cast("Any", role))
        for fill, role in ((1.0, "positive"), (-2.0, "negative"))
    )

    expected = tuple(evaluator.evaluate(latent, 0.25, value) for value in conditions)
    actual = evaluator.evaluate_batch(latent, 0.25, conditions)

    assert model.calls[-1][0].shape == (4, 128, 2, 2)
    for fused, separate in zip(actual, expected, strict=True):
        torch.testing.assert_close(fused, separate)


def test_conditioning_evaluator_refuses_malformed_values() -> None:
    evaluator = diffusion_runtime(RecordingIdeogram(0.0)).conditioning_evaluation()
    with pytest.raises(Ideogram4RuntimeError, match="Conditioning value"):
        evaluator.prepare(torch.zeros((1, 3, 53248)), cast("Any", "positive"))
    with pytest.raises(Ideogram4RuntimeError, match="pooled"):
        evaluator.prepare(
            Conditioning(torch.zeros((1, 3, 53248)), torch.zeros((1, 1))),
            cast("Any", "positive"),
        )
    with pytest.raises(Ideogram4RuntimeError, match="53248"):
        evaluator.prepare(Conditioning(torch.zeros((1, 3, 8)), None), cast("Any", "positive"))
    with pytest.raises(Ideogram4RuntimeError, match="binary"):
        evaluator.prepare(
            Ideogram4Conditioning(torch.zeros((1, 3, 53248)), None, torch.tensor(((1, 2, 0),))),
            cast("Any", "positive"),
        )


def test_sample_custom_routes_negative_lane_to_the_secondary_model() -> None:
    primary_model = RecordingIdeogram(0.25)
    secondary_model = RecordingIdeogram(-0.5)
    primary = diffusion_runtime(primary_model, "1")
    secondary = diffusion_runtime(secondary_model, "2")
    source = secondary.image_only_conditioning()
    routed = RoutedConditioning(
        embeddings=source.embeddings,
        pooled=None,
        evaluation=secondary.conditioning_evaluation(compute_dtype=torch.float32),
        source=source,
    )
    latent = torch.zeros((1, 128, 2, 2))
    result = primary.sample_custom(
        latent,
        noise=torch.zeros_like(latent),
        cond=condition(1.0),
        cfg=SamplingGuidance(routed, 7.0),
        request=custom_request(),
        compute_dtype=torch.float32,
    )
    assert isinstance(primary, CustomSamplingRuntime)
    assert result.output.shape == latent.shape
    assert len(primary_model.calls) == 1
    assert primary_model.calls[0][2] is not None
    assert len(secondary_model.calls) == 1
    assert secondary_model.calls[0][2] is None


def test_runtime_refuses_unsupported_modes_and_shapes() -> None:
    runtime = diffusion_runtime(RecordingIdeogram(0.0))
    request = custom_request()
    latent = torch.zeros((1, 128, 2, 2))

    def sample(**overrides: Any) -> Any:
        arguments: dict[str, Any] = {
            "noise": torch.zeros_like(latent),
            "cond": condition(),
            "request": request,
        }
        arguments.update(overrides)
        return runtime.sample_custom(arguments.pop("latent", latent), **arguments)

    with pytest.raises(Ideogram4RuntimeError, match="latent must have shape"):
        sample(latent=torch.zeros((1, 64, 2, 2)), noise=torch.zeros((1, 64, 2, 2)))
    with pytest.raises(Ideogram4RuntimeError, match="distilled-guidance"):
        sample(guidance=4.0)
    with pytest.raises(Ideogram4RuntimeError, match="inpaint"):
        sample(inpaint=cast("Any", object()))
    with pytest.raises(Ideogram4RuntimeError, match="context windows"):
        sample(context_windows=cast("Any", object()))


def test_denoiser_converts_flow_output_at_the_runtime_boundary() -> None:
    model = RecordingIdeogram(2.0)
    denoiser = runtime_module._Ideogram4Denoiser(  # pyright: ignore[reportPrivateUsage]
        cast("Any", model), compute_dtype=torch.float32
    )
    latent = torch.full((1, 128, 1, 1), 3.0)
    output = denoiser.evaluate_conditioning(
        latent, 0.25, denoiser.prepare_conditioning(condition())
    )
    torch.testing.assert_close(output, torch.full_like(latent, 2.5))
