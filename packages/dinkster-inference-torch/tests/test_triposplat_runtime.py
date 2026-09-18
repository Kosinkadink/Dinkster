from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    TRIPOSPLAT_CONFIG,
    TRIPOSPLAT_SIGMAS,
    Conditioning,
    ConditioningBatching,
    ConditioningBatchingMode,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    DualSamplingGuidance,
    MultiStreamLatent,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    SamplingCancelled,
    SamplingGuidance,
    SamplingSegment,
    sampling_sigmas,
)
from dinkster_inference_torch import (
    TripoSplatConditioning,
    TripoSplatDiffusionRuntime,
    TripoSplatRuntimeError,
    materialize_triposplat_conditioning,
    triposplat_conditioning_to_carrier,
)
from dinkster_inference_torch.conditioning_adapters import basic_conditioning_to_carrier
from dinkster_inference_torch.sampling_execution import run_ksampler_as_custom
from dinkster_inference_torch.schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry

BATCH = 1
TOKENS = TRIPOSPLAT_CONFIG.q_token_length
LATENT_CHANNELS = TRIPOSPLAT_CONFIG.latent_channels
COND_CHANNELS = TRIPOSPLAT_CONFIG.cond_channels
COND2_CHANNELS = TRIPOSPLAT_CONFIG.cond2_channels
CAM_CHANNELS = TRIPOSPLAT_CONFIG.cam_channels
FUSE_CFG_LANES = ConditioningBatching(ConditioningBatchingMode.MAX_FUSED_LANES, 2)


class FakeDiT:
    """Zero-velocity fake recording every forward call."""

    def __init__(self) -> None:
        self.input_layer = torch.nn.Linear(LATENT_CHANNELS, 8)
        self.calls: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []

    def __call__(
        self,
        latent: torch.Tensor,
        camera: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        reference_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((latent, camera, timesteps, context, reference_latent))
        return torch.zeros_like(latent), torch.zeros_like(camera)


def _runtime(model: FakeDiT | None = None) -> tuple[TripoSplatDiffusionRuntime, FakeDiT]:
    fake = FakeDiT() if model is None else model
    runtime = TripoSplatDiffusionRuntime(
        cast("Any", fake),
        runtime_identity="native:dinkster.triposplat:test",
        compute_dtype=torch.float32,
    )
    return runtime, fake


def _latent(batch: int = BATCH) -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent.from_pairs(
        (
            ("latent", torch.zeros((batch, TOKENS, LATENT_CHANNELS))),
            ("camera", torch.zeros((batch, 1, CAM_CHANNELS))),
        )
    )


def _features(fill: float = 0.0, rows: int = 4) -> torch.Tensor:
    return torch.full((BATCH, rows, COND_CHANNELS), fill)


def test_carrier_round_trip_preserves_features_and_reference() -> None:
    features = torch.arange(BATCH * 4 * COND_CHANNELS, dtype=torch.float32).reshape(
        BATCH, 4, COND_CHANNELS
    )
    reference = torch.arange(BATCH * COND2_CHANNELS * 4, dtype=torch.float32).reshape(
        BATCH, COND2_CHANNELS, 2, 2
    )
    carrier = triposplat_conditioning_to_carrier(TripoSplatConditioning(features, reference))
    out = materialize_triposplat_conditioning(carrier, device="cpu")
    assert torch.equal(out.features, features)
    assert out.reference_latent is not None
    assert torch.equal(out.reference_latent, reference)

    bare = materialize_triposplat_conditioning(
        triposplat_conditioning_to_carrier(TripoSplatConditioning(features)), device="cpu"
    )
    assert torch.equal(bare.features, features)
    assert bare.reference_latent is None


def test_materialize_refuses_foreign_carriers() -> None:
    carrier = basic_conditioning_to_carrier(Conditioning(torch.zeros((1, 3, 8)), None))
    with pytest.raises(TripoSplatRuntimeError, match="payloads"):
        materialize_triposplat_conditioning(carrier, device="cpu")


def test_prepare_conditioning_validates_geometry() -> None:
    runtime, _ = _runtime()
    good = runtime.prepare_conditioning(
        triposplat_conditioning_to_carrier(TripoSplatConditioning(_features()))
    )
    assert type(good) is TripoSplatConditioning

    with pytest.raises(TripoSplatRuntimeError, match="features must be nonempty"):
        runtime.prepare_conditioning(
            triposplat_conditioning_to_carrier(
                TripoSplatConditioning(torch.zeros((BATCH, 4, COND_CHANNELS + 1)))
            )
        )
    with pytest.raises(TripoSplatRuntimeError, match="must not exceed the features rows"):
        runtime.prepare_conditioning(
            triposplat_conditioning_to_carrier(
                TripoSplatConditioning(
                    _features(rows=3), torch.zeros((BATCH, COND2_CHANNELS, 2, 2))
                )
            )
        )


def test_conditioning_identity_is_the_pinned_architecture_string() -> None:
    runtime, _ = _runtime()
    assert runtime.conditioning_identity == (
        "dinkster.triposplat.conditioning:v1:dinkster.triposplat:8192:16:1280:128:5:1024:24:2"
    )
    assert runtime.runtime_identity == "native:dinkster.triposplat:test"


def test_runtime_names_the_catalog_family_for_preview_resolution() -> None:
    from dinkster_inference import TRIPOSPLAT

    runtime, _ = _runtime()
    assert runtime.family is TRIPOSPLAT


def test_sampling_reproduces_the_reference_nested_noise_draw() -> None:
    runtime, fake = _runtime()
    seed = 7
    result = runtime.sample_multistream(
        _latent(),
        conditioning=TripoSplatConditioning(_features()),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        denoise=1.0,
        seed=seed,
    )

    # sigma_max is 1.0 in the shift-3 flow space, the input latent is
    # zero, and the fake predicts zero velocity, so the euler output is
    # exactly the prepared noise: one CPU generator drawing the latent
    # stream then the camera stream.
    generator = torch.Generator("cpu")
    generator.manual_seed(seed)
    expected_latent = torch.randn(
        (BATCH, TOKENS, LATENT_CHANNELS), dtype=torch.float32, generator=generator, device="cpu"
    )
    expected_camera = torch.randn(
        (BATCH, 1, CAM_CHANNELS), dtype=torch.float32, generator=generator, device="cpu"
    )
    assert result.roles == ("latent", "camera")
    torch.testing.assert_close(result.by_role("latent"), expected_latent)
    torch.testing.assert_close(result.by_role("camera"), expected_camera)

    assert len(fake.calls) == 1
    latent_in, camera_in, timesteps, context, reference = fake.calls[0]
    assert latent_in.shape == (BATCH, TOKENS, LATENT_CHANNELS)
    assert camera_in.shape == (BATCH, 1, CAM_CHANNELS)
    assert timesteps.tolist() == [TRIPOSPLAT_SIGMAS.timestep(TRIPOSPLAT_SIGMAS.sigma_max)]
    assert context.shape == (BATCH, 4, COND_CHANNELS)
    assert reference is None


def test_sampling_forwards_the_reference_latent_and_cfg_lanes() -> None:
    runtime, fake = _runtime()
    reference = torch.ones((BATCH, COND2_CHANNELS, 2, 2))
    positive = TripoSplatConditioning(_features(1.0), reference)
    negative = TripoSplatConditioning(_features(0.0), torch.zeros_like(reference))

    runtime.sample_multistream(
        _latent(),
        conditioning=positive,
        cfg=SamplingGuidance(uncond=cast("Any", negative), scale=2.0, batching=FUSE_CFG_LANES),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        denoise=1.0,
        seed=3,
    )

    assert len(fake.calls) == 1
    fills = sorted(float(value) for value in fake.calls[0][3][:, 0, 0])
    assert fills == [0.0, 1.0]
    reference_call = fake.calls[0][4]
    assert reference_call is not None
    assert reference_call.shape == (BATCH * 2, COND2_CHANNELS, 2, 2)


def test_fused_cfg_expands_conditioning_batches_cyclically() -> None:
    runtime, fake = _runtime()
    latent_batch = 3

    def stacked(fills: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        return torch.stack([torch.full(shape, fill) for fill in fills])

    positive = TripoSplatConditioning(
        stacked((1.0, 2.0), (4, COND_CHANNELS)),
        stacked((5.0, 6.0), (COND2_CHANNELS, 2, 2)),
    )
    negative = TripoSplatConditioning(
        stacked((0.0, -1.0), (4, COND_CHANNELS)),
        stacked((7.0, 8.0), (COND2_CHANNELS, 2, 2)),
    )

    runtime.sample_multistream(
        _latent(batch=latent_batch),
        conditioning=positive,
        cfg=SamplingGuidance(uncond=cast("Any", negative), scale=2.0, batching=FUSE_CFG_LANES),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        denoise=1.0,
        seed=3,
    )

    assert len(fake.calls) == 1
    context = fake.calls[0][3]
    reference = fake.calls[0][4]
    assert context.shape == (latent_batch * 2, 4, COND_CHANNELS)
    assert reference is not None
    assert reference.shape == (latent_batch * 2, COND2_CHANNELS, 2, 2)
    # A batch-2 conditioning over a batch-3 latent repeats cyclically,
    # yielding rows [0, 1, 0] within each fused lane.
    lanes = {tuple(float(entry[0, 0]) for entry in chunk) for chunk in context.chunk(2)}
    assert lanes == {(1.0, 2.0, 1.0), (0.0, -1.0, 0.0)}
    reference_lanes = {
        tuple(float(entry[0, 0, 0]) for entry in chunk) for chunk in reference.chunk(2)
    }
    assert reference_lanes == {(5.0, 6.0, 5.0), (7.0, 8.0, 7.0)}


def test_sampling_refuses_foreign_topologies_and_inputs() -> None:
    runtime, _ = _runtime()
    conditioning = TripoSplatConditioning(_features())

    def sample(latent: MultiStreamLatent[torch.Tensor], **overrides: object) -> object:
        arguments: dict[str, Any] = {
            "conditioning": conditioning,
            "sampler_id": "dinkster.euler",
            "scheduler_id": "dinkster.simple",
            "steps": 1,
            "denoise": 1.0,
            "seed": 0,
        }
        arguments.update(overrides)
        return runtime.sample_multistream(latent, **arguments)

    with pytest.raises(TripoSplatRuntimeError, match="requires exactly the"):
        sample(
            MultiStreamLatent.from_pairs(
                (("latent", torch.zeros((BATCH, TOKENS, LATENT_CHANNELS))),)
            )
        )
    with pytest.raises(TripoSplatRuntimeError, match="latent stream must have shape"):
        sample(
            MultiStreamLatent.from_pairs(
                (
                    ("latent", torch.zeros((BATCH, TOKENS, LATENT_CHANNELS + 1))),
                    ("camera", torch.zeros((BATCH, 1, CAM_CHANNELS))),
                )
            )
        )
    with pytest.raises(TripoSplatRuntimeError, match="camera stream must have shape"):
        sample(
            MultiStreamLatent.from_pairs(
                (
                    ("latent", torch.zeros((BATCH, TOKENS, LATENT_CHANNELS))),
                    ("camera", torch.zeros((BATCH, 2, CAM_CHANNELS))),
                )
            )
        )
    with pytest.raises(TripoSplatRuntimeError, match="share one dtype"):
        sample(
            MultiStreamLatent.from_pairs(
                (
                    ("latent", torch.zeros((BATCH, TOKENS, LATENT_CHANNELS))),
                    ("camera", torch.zeros((BATCH, 1, CAM_CHANNELS), dtype=torch.float64)),
                )
            )
        )
    with pytest.raises(TripoSplatRuntimeError, match="no dual-guidance recipe"):
        sample(
            _latent(),
            cfg=DualSamplingGuidance(
                middle=cast("Any", conditioning),
                uncond=cast("Any", conditioning),
                scale=2.0,
                middle_scale=1.0,
            ),
        )
    masked = cast(
        "MultiStreamLatent[torch.Tensor]",
        sample(_latent(), denoise_mask=torch.zeros((BATCH, TOKENS, LATENT_CHANNELS))),
    )
    assert torch.equal(masked.by_role("latent"), _latent().by_role("latent"))
    with pytest.raises(TypeError, match="exact TripoSplatConditioning"):
        sample(_latent(), conditioning=object())
    with pytest.raises(TypeError, match="guidance lanes require exact TripoSplatConditioning"):
        sample(_latent(), cfg=SamplingGuidance(uncond=cast("Any", object()), scale=2.0))


def test_sampling_honors_cancellation() -> None:
    runtime, _ = _runtime()
    with pytest.raises(SamplingCancelled):
        runtime.sample_multistream(
            _latent(),
            conditioning=TripoSplatConditioning(_features()),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            denoise=1.0,
            seed=0,
            cancelled=lambda: True,
        )


class ArithmeticDiT(FakeDiT):
    """Input-dependent fake: the velocities couple latent, camera,
    timestep, and context so schedule, noise, and CFG-lane parity are
    all visible."""

    def __call__(
        self,
        latent: torch.Tensor,
        camera: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        reference_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((latent, camera, timesteps, context, reference_latent))
        scale = context.float().mean(dim=(1, 2)).reshape(-1, 1, 1)
        step = timesteps.reshape(-1, 1, 1)
        return latent * 0.5 + step * 0.25 + scale, camera * 0.5 + step * 0.25 + scale


def _random_latent(seed: int = 11, batch: int = BATCH) -> MultiStreamLatent[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return MultiStreamLatent.from_pairs(
        (
            ("latent", torch.rand((batch, TOKENS, LATENT_CHANNELS), generator=generator)),
            ("camera", torch.rand((batch, 1, CAM_CHANNELS), generator=generator)),
        )
    )


def _custom_request(
    sampler_id: str = "dinkster.euler",
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
) -> CustomSamplingRequest[torch.Tensor]:
    descriptor = torch_sampler_registry().get(sampler_id)
    assert descriptor is not None
    return CustomSamplingRequest(descriptor, (), sigmas)


def test_triposplat_runtime_satisfies_the_custom_sampling_protocol() -> None:
    runtime, _ = _runtime()
    assert isinstance(runtime, CustomSamplingRuntime)


@pytest.mark.parametrize(
    ("sampler_id", "scheduler_id", "steps", "cfg_scale", "segment"),
    [
        ("dinkster.euler", "dinkster.simple", 2, None, None),
        ("dinkster.res_multistep", "dinkster.simple", 2, 2.0, None),
        ("dinkster.dpmpp_sde", "dinkster.simple", 3, None, None),
        (
            "dinkster.euler",
            "dinkster.simple",
            3,
            None,
            SamplingSegment(
                steps=3,
                start_step=1,
                end_step=3,
                add_noise=False,
                return_with_leftover_noise=False,
            ),
        ),
    ],
    ids=("euler", "cfg", "brownian", "segment"),
)
def test_ksampler_surface_is_bit_equal_sugar_over_sample_custom(
    sampler_id: str,
    scheduler_id: str,
    steps: int,
    cfg_scale: float | None,
    segment: SamplingSegment | None,
) -> None:
    runtime, _ = _runtime(ArithmeticDiT())
    reference = torch.ones((BATCH, COND2_CHANNELS, 2, 2))
    positive = TripoSplatConditioning(_features(2.0), reference)
    negative = (
        None
        if cfg_scale is None
        else TripoSplatConditioning(_features(5.0), torch.zeros_like(reference))
    )
    latent = _random_latent()
    identity = runtime.conditioning_identity

    expected = runtime.sample_multistream(
        latent,
        conditioning=positive,
        cfg=SamplingGuidance(cast("Any", negative), 1.0 if cfg_scale is None else cfg_scale),
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        segment=segment,
    )
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=TRIPOSPLAT_SIGMAS,
        flow=True,
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        cond=PreparedMultiStreamConditioning(identity, positive),
        cfg=SamplingGuidance(
            None if negative is None else PreparedMultiStreamConditioning(identity, negative),
            1.0 if cfg_scale is None else cfg_scale,
        ),
        segment=segment,
        error=TripoSplatRuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("latent"), expected.by_role("latent"))
    assert torch.equal(output.by_role("camera"), expected.by_role("camera"))


def test_ksampler_sugar_parity_holds_for_low_precision_latents() -> None:
    """Seeded noise is drawn against float32 views on both surfaces,
    so half-precision latent streams must not round the draw on the
    decomposed path."""
    runtime, _ = _runtime(ArithmeticDiT())
    latent = _random_latent().map(lambda stream: stream.to(dtype=torch.float16))
    positive = TripoSplatConditioning(_features(2.0))

    expected = runtime.sample_multistream(
        latent,
        conditioning=positive,
        cfg=SamplingGuidance(None, 1.0),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
    )
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=TRIPOSPLAT_SIGMAS,
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=185,
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, positive),
        cfg=SamplingGuidance(None, 1.0),
        error=TripoSplatRuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("latent"), expected.by_role("latent"))
    assert torch.equal(output.by_role("camera"), expected.by_role("camera"))


def test_ksampler_sugar_parity_holds_for_cfg_pp_without_negative() -> None:
    """CFG++ samplers consume the guidance scale through a
    synthetic-zero unconditional lane even with no negative payload,
    so an empty negative must keep the scale on both surfaces."""
    runtime, _ = _runtime(ArithmeticDiT())
    latent = _random_latent()
    positive = TripoSplatConditioning(_features(2.0))
    guidance = SamplingGuidance(None, 3.0)

    expected = runtime.sample_multistream(
        latent,
        conditioning=positive,
        cfg=guidance,
        sampler_id="dinkster.euler_cfg_pp",
        scheduler_id="dinkster.simple",
        steps=3,
        denoise=0.75,
        seed=123,
    )
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=TRIPOSPLAT_SIGMAS,
        flow=True,
        sampler_id="dinkster.euler_cfg_pp",
        scheduler_id="dinkster.simple",
        steps=3,
        denoise=0.75,
        seed=123,
        cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, positive),
        cfg=guidance,
        error=TripoSplatRuntimeError,
    )

    output = result.output
    assert type(output) is MultiStreamLatent
    assert torch.equal(output.by_role("latent"), expected.by_role("latent"))
    assert torch.equal(output.by_role("camera"), expected.by_role("camera"))


def test_custom_sampling_sigma_surfaces_match_shift_three_flow_space() -> None:
    runtime, _ = _runtime()
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    assert runtime.custom_sampling_sigmas("dinkster.simple", 4, 0.5) == sampling_sigmas(
        scheduler, TRIPOSPLAT_SIGMAS, 4, denoise=0.5
    )
    with pytest.raises(TripoSplatRuntimeError, match="unknown scheduler"):
        runtime.custom_sampling_sigmas("test.missing", 4, 1.0)
    assert runtime.custom_sampling_beta_sigmas(4, 0.6, 0.6) == custom_beta_sigmas(
        TRIPOSPLAT_SIGMAS, 4, 0.6, 0.6
    )
    with pytest.raises(ValueError, match="discrete sigma space"):
        runtime.custom_sampling_sd_turbo_sigmas(2, 1.0)
    assert runtime.custom_sampling_percent_to_sigma(
        0.5, return_actual_sigma=False
    ) == custom_percent_to_sigma(
        TRIPOSPLAT_SIGMAS, TRIPOSPLAT_SIGMAS.percent_to_sigma, 0.5, return_actual_sigma=False
    )
    assert runtime.custom_sampling_percent_to_sigma(
        0.3, return_actual_sigma=True
    ) == custom_percent_to_sigma(
        TRIPOSPLAT_SIGMAS, TRIPOSPLAT_SIGMAS.percent_to_sigma, 0.3, return_actual_sigma=True
    )


def test_sample_custom_refuses_foreign_inputs_and_unsupported_modes() -> None:
    runtime, fake = _runtime()
    conditioning = TripoSplatConditioning(_features())
    prepared = PreparedMultiStreamConditioning(runtime.conditioning_identity, conditioning)
    latent = _latent()
    noise = latent.map(torch.zeros_like)
    request = _custom_request()

    def sample(**overrides: Any) -> Any:
        arguments: dict[str, Any] = {"noise": noise, "cond": prepared, "request": request}
        arguments.update(overrides)
        return runtime.sample_custom(arguments.pop("latent", latent), **arguments)

    with pytest.raises(TripoSplatRuntimeError, match="requires a MultiStreamLatent latent"):
        sample(latent=torch.zeros((BATCH, TOKENS, LATENT_CHANNELS)))
    with pytest.raises(TripoSplatRuntimeError, match="requires MultiStreamLatent noise"):
        sample(noise=torch.zeros((BATCH, TOKENS, LATENT_CHANNELS)))
    with pytest.raises(TripoSplatRuntimeError, match="dual-guidance"):
        sample(cfg=DualSamplingGuidance(cast("Any", prepared), cast("Any", prepared), 3.0, 1.5))
    with pytest.raises(TripoSplatRuntimeError, match="perp-neg"):
        sample(cfg=PerpNegSamplingGuidance(cast("Any", prepared), cast("Any", prepared), 3.0, 1.0))
    with pytest.raises(TripoSplatRuntimeError, match="distilled-guidance"):
        sample(guidance=4.0)
    with pytest.raises(TripoSplatRuntimeError, match="inpaint"):
        sample(inpaint=cast("Any", object()))
    euler = torch_sampler_registry().get("dinkster.euler")
    assert euler is not None
    unknown = replace(euler, id="test.missing", aliases=())
    with pytest.raises(TripoSplatRuntimeError, match="unknown sampler"):
        sample(request=CustomSamplingRequest(unknown, (), (1.0, 0.0)))
    with pytest.raises(TripoSplatRuntimeError, match="prepared multi-stream conditioning"):
        sample(cond=cast("Any", conditioning))
    with pytest.raises(TripoSplatRuntimeError, match="a different conditioner component"):
        sample(cond=PreparedMultiStreamConditioning("native:other", conditioning))
    with pytest.raises(TypeError, match="exact TripoSplatConditioning"):
        sample(
            cond=PreparedMultiStreamConditioning(
                runtime.conditioning_identity, cast("Any", object())
            )
        )
    with pytest.raises(TripoSplatRuntimeError, match="guidance requires prepared"):
        sample(cfg=SamplingGuidance(cast("Any", conditioning), 2.0))
    with pytest.raises(TripoSplatRuntimeError, match="different conditioner components"):
        sample(
            cfg=SamplingGuidance(
                cast("Any", PreparedMultiStreamConditioning("native:other", conditioning)), 2.0
            )
        )
    with pytest.raises(TypeError, match="guidance lanes require exact TripoSplatConditioning"):
        sample(
            cfg=SamplingGuidance(
                cast(
                    "Any",
                    PreparedMultiStreamConditioning(
                        runtime.conditioning_identity, cast("Any", object())
                    ),
                ),
                2.0,
            )
        )
    assert not fake.calls


def test_sample_custom_captures_denoised_output() -> None:
    runtime, _ = _runtime()
    latent = _latent()
    result = runtime.sample_custom(
        latent,
        noise=latent.map(torch.zeros_like),
        cond=PreparedMultiStreamConditioning(
            runtime.conditioning_identity, TripoSplatConditioning(_features())
        ),
        request=_custom_request(),
        seed=9,
    )
    output = result.output
    assert type(output) is MultiStreamLatent
    assert output.roles == ("latent", "camera")
    assert result.denoised_output is not None
    assert result.denoised_output.roles == ("latent", "camera")
