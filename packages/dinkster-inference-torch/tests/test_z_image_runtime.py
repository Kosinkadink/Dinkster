from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    Z_IMAGE,
    Z_IMAGE_CONTROL_RESIDUAL_SITES,
    Conditioning,
    ConstantGainCurve,
    ContributionGain,
    ControlApplication,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    DirectGainTableCurve,
    PayloadReference,
    PercentRange,
)
from dinkster_inference_torch import (
    ZImageControl,
    ZImageControlBindingError,
    ZImageControlConditioning,
    ZImageDenoiser,
    ZImageRuntime,
    ZImageRuntimeError,
    select_attention,
    z_image_control_hint_digest,
    z_image_control_resource_digest,
)
from dinkster_inference_torch import module_residency as residency_mod
from dinkster_inference_torch import sampling_execution as sampling_execution_mod
from dinkster_inference_torch import z_image_control as control_mod
from dinkster_inference_torch.denoise import prepare_noise
from dinkster_inference_torch.operations import INITLESS
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry


class RecordingZImage(torch.nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = value
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((latent, timestep, context))
        return torch.full_like(latent, self.value)


def _control_conditioning(
    hint: torch.Tensor,
    *,
    bind: bool = True,
    gain: ContributionGain | None = None,
) -> ZImageControlConditioning:
    with torch.device("meta"):
        model = ZImageControl(
            operations=INITLESS,
            attention_kernel=select_attention("flux").kernel,
        )
    model_digest = z_image_control_resource_digest("blake3:" + "0" * 64, torch.bfloat16)
    if bind:
        control_mod._bind_z_image_control_resource(  # pyright: ignore[reportPrivateUsage]
            model, model_digest
        )
    hint_digest = z_image_control_hint_digest(hint)
    return ZImageControlConditioning(
        ControlApplication(
            "z-image-fun",
            PayloadReference(hint_digest),
            1.0,
            PercentRange(0.0, 1.0),
        ),
        model,
        hint,
        model_digest,
        hint_digest,
        gain,
    )


def test_z_image_denoiser_converts_native_flow_output_and_batches_cfg() -> None:
    model = RecordingZImage()
    evaluator = ZImageDenoiser(cast("Any", model), compute_dtype=torch.float32)
    latent = torch.full((2, 16, 2, 2), 4.0)
    first = torch.zeros((1, 3, 2560))
    second = torch.ones((1, 3, 2560))
    outputs = evaluator.evaluate_conditioning_batch(
        latent, 0.25, ((first, "positive"), (second, "negative"))
    )
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0], torch.full_like(latent, 3.75))
    torch.testing.assert_close(outputs[1], torch.full_like(latent, 3.75))
    model_latent, timestep, context = model.calls[0]
    assert model_latent.shape == (4, 16, 2, 2)
    assert timestep.tolist() == [0.25] * 4
    assert context.shape == (4, 3, 2560)


def test_z_image_denoiser_refuses_pooled_or_wrong_width_conditioning() -> None:
    evaluator = ZImageDenoiser(cast("Any", RecordingZImage()))
    with pytest.raises(ZImageRuntimeError, match="pooled"):
        evaluator.prepare_conditioning(Conditioning(torch.zeros((1, 2, 2560)), torch.zeros((1, 1))))
    with pytest.raises(ZImageRuntimeError, match="2560"):
        evaluator.prepare_conditioning(Conditioning(torch.zeros((1, 2, 2304)), None))


def test_z_image_control_conditioning_requires_assembly_provenance() -> None:
    with pytest.raises(ZImageControlBindingError, match="assemble_z_image_control"):
        _control_conditioning(torch.zeros((1, 16, 2, 2)), bind=False)


def test_z_image_control_conditioning_refuses_changed_model_resource() -> None:
    hint = torch.zeros((1, 16, 2, 2))
    conditioning = _control_conditioning(hint)
    parameter = next(conditioning.model.parameters())
    with torch.no_grad():
        parameter.add_(1.0)
    with pytest.raises(ZImageControlBindingError, match="changed"):
        replace(conditioning)


def test_z_image_control_resource_accepts_only_residency_owned_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditioning = _control_conditioning(torch.zeros((1, 16, 2, 2)))
    model = conditioning.model
    name, original = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.ndim >= 2 and parameter.shape[-2] == parameter.shape[-1]
    )
    module_name, _, parameter_name = name.rpartition(".")
    owner = model.get_submodule(module_name)
    replacement = torch.nn.Parameter(torch.empty_like(original), requires_grad=False)
    with torch.no_grad():
        replacement.add_(1.0)
    setattr(owner, parameter_name, replacement)

    with pytest.raises(ZImageControlBindingError, match="changed"):
        control_mod.validate_z_image_control_resource(model, conditioning.model_digest)

    seen: list[str] = []

    def authorize(_model: torch.nn.Module, key: str, tensor: torch.Tensor) -> int | None:
        seen.append(key)
        return 1 if tensor is replacement else None

    monkeypatch.setattr(residency_mod, "_residency_assignment_generation", authorize)
    control_mod.validate_z_image_control_resource(model, conditioning.model_digest)
    assert seen == [name]

    with torch.no_grad():
        replacement.add_(1.0)
    with pytest.raises(ZImageControlBindingError, match="changed"):
        control_mod.validate_z_image_control_resource(model, conditioning.model_digest)

    setattr(owner, parameter_name, original)
    control_mod.validate_z_image_control_resource(model, conditioning.model_digest)
    original.data = original.data.transpose(-2, -1)
    with pytest.raises(ZImageControlBindingError, match="changed"):
        control_mod.validate_z_image_control_resource(model, conditioning.model_digest)
    original.data = original.data.transpose(-2, -1)
    control_mod.validate_z_image_control_resource(model, conditioning.model_digest)
    original.data = original.detach().clone()
    with pytest.raises(ZImageControlBindingError, match="changed"):
        control_mod.validate_z_image_control_resource(model, conditioning.model_digest)


def test_z_image_denoiser_revalidates_control_resource_before_execution() -> None:
    conditioning = _control_conditioning(torch.zeros((1, 16, 2, 2)))
    evaluator = ZImageDenoiser(
        cast("Any", RecordingZImage()), control=conditioning, compute_dtype=torch.float32
    )
    with torch.no_grad():
        next(conditioning.model.parameters()).add_(1.0)

    with pytest.raises(ZImageControlBindingError, match="changed"):
        evaluator.evaluate_conditioning_batch(
            torch.zeros((1, 16, 2, 2)),
            0.5,
            ((torch.zeros((1, 3, 2560)), "positive"),),
        )


def test_z_image_control_conditioning_refuses_changed_hint_and_chains() -> None:
    hint = torch.zeros((1, 16, 2, 2))
    conditioning = _control_conditioning(hint)
    hint.fill_(1.0)
    with pytest.raises(ZImageControlBindingError, match="materialized hint"):
        replace(conditioning, hint=hint)
    conditioning = _control_conditioning(torch.zeros((1, 16, 2, 2)))
    previous = conditioning.application
    chained = replace(
        conditioning.application,
        child_id="z-image-fun-next",
        previous=previous,
    )
    with pytest.raises(ValueError, match="admission limit of one"):
        replace(conditioning, application=chained)


def test_z_image_control_conditioning_binds_and_snapshots_hint() -> None:
    class ControlledRecordingZImage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hints: list[torch.Tensor] = []
            self.gains: list[tuple[torch.Tensor, ...]] = []
            self.plain_calls = 0

        def forward(
            self,
            latent: torch.Tensor,
            _timestep: torch.Tensor,
            _context: torch.Tensor,
            **kwargs: object,
        ) -> torch.Tensor:
            if not kwargs:
                self.plain_calls += 1
                return torch.zeros_like(latent)
            self.hints.append(cast("torch.Tensor", kwargs["control_latent"]))
            self.gains.append(cast("tuple[torch.Tensor, ...]", kwargs["control_gains"]))
            return torch.zeros_like(latent)

    hint = torch.zeros((1, 16, 2, 2))
    conditioning = _control_conditioning(hint)
    model = ControlledRecordingZImage()
    evaluator = ZImageDenoiser(
        cast("Any", model), control=conditioning, compute_dtype=torch.float32
    )
    hint.fill_(9.0)
    evaluator.set_control_gain_row(
        ("positive", "negative"),
        tuple((float(index), float(index + 10)) for index in range(6)),
    )
    evaluator.evaluate_conditioning_batch(
        torch.zeros((1, 16, 2, 2)),
        0.5,
        (
            (torch.zeros((1, 3, 2560)), "positive"),
            (torch.ones((1, 3, 2560)), "negative"),
        ),
    )
    assert torch.count_nonzero(model.hints[0]).item() == 0
    assert [gain.flatten().tolist() for gain in model.gains[0]] == [
        [float(index), float(index + 10)] for index in range(6)
    ]
    evaluator.set_control_gain_row(("positive", "negative"), ((0.0, 0.0),) * 6)
    evaluator.evaluate_conditioning_batch(
        torch.zeros((1, 16, 2, 2)),
        0.5,
        (
            (torch.zeros((1, 3, 2560)), "positive"),
            (torch.ones((1, 3, 2560)), "negative"),
        ),
    )
    assert model.plain_calls == 1
    assert len(model.hints) == 1
    evaluator.set_control_gain_row(("positive", "negative"), ((0.0, 1.0),) * 6)
    evaluator.evaluate_conditioning_batch(
        torch.zeros((1, 16, 2, 2)),
        0.5,
        (
            (torch.zeros((1, 3, 2560)), "positive"),
            (torch.ones((1, 3, 2560)), "negative"),
        ),
    )
    assert model.plain_calls == 2
    assert len(model.hints) == 2
    assert model.hints[-1].shape[0] == 1


def test_z_image_runtime_samples_in_shift_three_flow_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import sampling_execution

    model = RecordingZImage(value=0.0)
    assembled = type("Assembled", (), {"family": Z_IMAGE, "diffusion": model})()
    runtime = object.__new__(ZImageRuntime)
    runtime.assembled = cast("Any", assembled)
    runtime._samplers = torch_sampler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._schedulers = torch_scheduler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._guidance = None  # pyright: ignore[reportPrivateUsage]
    shifts: list[float] = []
    original = sampling_execution.build_sampling_schedule

    def capture(*args: Any, **kwargs: Any) -> Any:
        shifts.append(args[1].shift)
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling_execution, "build_sampling_schedule", capture)
    latent = torch.zeros((1, 16, 2, 2))
    result = runtime.sample(
        latent,
        cond=Conditioning(torch.zeros((1, 3, 2560)), None),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        seed=4,
        compute_dtype=torch.float32,
    )
    assert shifts == [3.0]
    assert result.shape == latent.shape
    assert model.calls


def test_z_image_runtime_conforms_to_custom_sampling_contract() -> None:
    runtime = object.__new__(ZImageRuntime)
    runtime.assembled = cast("Any", type("Assembled", (), {"family": Z_IMAGE})())
    runtime._runtime_identity = "test-z-image"  # pyright: ignore[reportPrivateUsage]
    assert isinstance(runtime, CustomSamplingRuntime)


def test_z_image_runtime_ksampler_and_custom_sampling_are_bit_identical() -> None:
    model = RecordingZImage(value=0.0)
    assembled = type("Assembled", (), {"family": Z_IMAGE, "diffusion": model})()
    runtime = object.__new__(ZImageRuntime)
    runtime.assembled = cast("Any", assembled)
    runtime._runtime_identity = "test-z-image"  # pyright: ignore[reportPrivateUsage]
    runtime._samplers = torch_sampler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._schedulers = torch_scheduler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._guidance = None  # pyright: ignore[reportPrivateUsage]
    latent = torch.zeros((1, 16, 2, 2))
    condition = Conditioning(torch.zeros((1, 3, 2560)), None)
    sampler = runtime._samplers.get("dinkster.euler")  # pyright: ignore[reportPrivateUsage]
    assert sampler is not None
    sigmas = runtime.custom_sampling_sigmas("dinkster.simple", 3, 1.0)
    custom = runtime.sample_custom(
        latent,
        noise=prepare_noise(latent, 4),
        cond=condition,
        request=CustomSamplingRequest(sampler, (), sigmas),
        seed=4,
        compute_dtype=torch.float32,
    )
    facade = runtime.sample(
        latent,
        cond=condition,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=3,
        seed=4,
        compute_dtype=torch.float32,
    )
    assert torch.equal(custom.output, facade)
    assert custom.denoised_output is not None


def test_z_image_sampling_paths_reject_unknown_adapter_options() -> None:
    model = RecordingZImage(value=0.0)
    assembled = type("Assembled", (), {"family": Z_IMAGE, "diffusion": model})()
    runtime = object.__new__(ZImageRuntime)
    runtime.assembled = cast("Any", assembled)
    runtime._runtime_identity = "test-z-image"  # pyright: ignore[reportPrivateUsage]
    runtime._samplers = torch_sampler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._schedulers = torch_scheduler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._guidance = None  # pyright: ignore[reportPrivateUsage]
    latent = torch.zeros((1, 16, 2, 2))
    condition = Conditioning(torch.zeros((1, 3, 2560)), None)
    sampler = runtime._samplers.get("dinkster.euler")  # pyright: ignore[reportPrivateUsage]
    assert sampler is not None

    with pytest.raises(ZImageRuntimeError, match="adapter options: bogus_option"):
        runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=condition,
            request=CustomSamplingRequest(sampler, (), (1.0, 0.0)),
            compute_dtype=torch.float32,
            bogus_option=True,
        )
    with pytest.raises(ZImageRuntimeError, match="adapter options: bogus_option"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            compute_dtype=torch.float32,
            bogus_option=True,
        )
    assert not model.calls


def test_z_image_runtime_refuses_unknown_control_site_before_execution() -> None:
    model = RecordingZImage(value=0.0)
    assembled = type("Assembled", (), {"family": Z_IMAGE, "diffusion": model})()
    runtime = object.__new__(ZImageRuntime)
    runtime.assembled = cast("Any", assembled)
    runtime._samplers = torch_sampler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._schedulers = torch_scheduler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._guidance = None  # pyright: ignore[reportPrivateUsage]
    control = _control_conditioning(
        torch.zeros((1, 16, 2, 2)),
        gain=ContributionGain(
            ConstantGainCurve(1.0),
            1.0,
            site_gains=(("unknown.site", 1.0),),
        ),
    )
    with pytest.raises(ZImageRuntimeError, match="operator_site_mismatch"):
        runtime.sample(
            torch.zeros((1, 16, 2, 2)),
            cond=Conditioning(torch.zeros((1, 3, 2560)), None),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            control=control,
            compute_dtype=torch.float32,
        )
    assert not model.calls


def test_z_image_runtime_realizes_site_lane_gains_and_binds_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingZImage(value=0.0)
    assembled = type("Assembled", (), {"family": Z_IMAGE, "diffusion": model})()
    runtime = object.__new__(ZImageRuntime)
    runtime.assembled = cast("Any", assembled)
    runtime._samplers = torch_sampler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._schedulers = torch_scheduler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._guidance = None  # pyright: ignore[reportPrivateUsage]
    identities: list[str] = []
    selected: list[tuple[tuple[float, ...], ...]] = []

    original_set_row = ZImageDenoiser.set_control_gain_row

    def capture_row(
        evaluator: ZImageDenoiser,
        lane_ids: tuple[str, ...],
        values: tuple[tuple[float, ...], ...],
    ) -> None:
        selected.append(values)
        original_set_row(evaluator, lane_ids, values)

    def capture_run(*args: object, **kwargs: object) -> torch.Tensor:
        denoiser = cast("Any", args[0])
        identities.append(denoiser.conditioning_plan.calls[0].evaluator_identity)
        callback = kwargs["on_step_begin"]
        assert callable(callback)
        for index in range(3):
            callback(index)
        return cast("torch.Tensor", kwargs["latent"])

    monkeypatch.setattr(ZImageDenoiser, "set_control_gain_row", capture_row)
    monkeypatch.setattr(sampling_execution_mod, "run_denoise", capture_run)
    gain = ContributionGain(
        DirectGainTableCurve((1.0, 0.5, 0.0)),
        2.0,
        site_gains=tuple(
            (site, float(index + 1)) for index, site in enumerate(Z_IMAGE_CONTROL_RESIDUAL_SITES)
        ),
        lane_gains=(("positive", 0.5),),
    )
    for hint_value in (0.0, 1.0):
        runtime.sample(
            torch.zeros((1, 16, 2, 2)),
            cond=Conditioning(torch.zeros((1, 3, 2560)), None),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            control=_control_conditioning(torch.full((1, 16, 2, 2), hint_value), gain=gain),
            compute_dtype=torch.float32,
        )
    assert selected[:3] == [
        tuple((float(index + 1) * timeline_gain,) for index in range(6))
        for timeline_gain in (1.0, 0.5, 0.0)
    ]
    assert identities[0].startswith("dinkster.z-image.conditioning.v1:intervention-plan=")
    assert identities[0] != identities[1]


def test_z_image_runtime_refuses_unknown_control_lane_before_execution() -> None:
    model = RecordingZImage(value=0.0)
    assembled = type("Assembled", (), {"family": Z_IMAGE, "diffusion": model})()
    runtime = object.__new__(ZImageRuntime)
    runtime.assembled = cast("Any", assembled)
    runtime._samplers = torch_sampler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._schedulers = torch_scheduler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._guidance = None  # pyright: ignore[reportPrivateUsage]
    control = _control_conditioning(
        torch.zeros((1, 16, 2, 2)),
        gain=ContributionGain(
            ConstantGainCurve(1.0),
            1.0,
            lane_gains=(("unknown", 1.0),),
        ),
    )
    with pytest.raises(ZImageRuntimeError, match="active guidance lanes"):
        runtime.sample(
            torch.zeros((1, 16, 2, 2)),
            cond=Conditioning(torch.zeros((1, 3, 2560)), None),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            control=control,
            compute_dtype=torch.float32,
        )
    assert not model.calls


@pytest.mark.parametrize("sampler_id", ["dinkster.dpm_fast", "dinkster.dpm_adaptive"])
def test_z_image_runtime_refuses_scheduled_gain_for_off_grid_sampler(
    sampler_id: str,
) -> None:
    model = RecordingZImage(value=0.0)
    assembled = type("Assembled", (), {"family": Z_IMAGE, "diffusion": model})()
    runtime = object.__new__(ZImageRuntime)
    runtime.assembled = cast("Any", assembled)
    runtime._samplers = torch_sampler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._schedulers = torch_scheduler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._guidance = None  # pyright: ignore[reportPrivateUsage]
    control = _control_conditioning(
        torch.zeros((1, 16, 2, 2)),
        gain=ContributionGain(DirectGainTableCurve((1.0, 0.5, 0.0)), 1.0),
    )
    with pytest.raises(ZImageRuntimeError, match=f"{sampler_id} supports only constant"):
        runtime.sample(
            torch.zeros((1, 16, 2, 2)),
            cond=Conditioning(torch.zeros((1, 3, 2560)), None),
            sampler_id=sampler_id,
            scheduler_id="dinkster.normal",
            steps=3,
            control=control,
            compute_dtype=torch.float32,
        )
    assert not model.calls


def test_z_image_runtime_refuses_nonlatent_and_unsupported_inputs() -> None:
    class Assembled:
        family = Z_IMAGE

        @staticmethod
        def compute_dtype(role: str) -> torch.dtype:
            del role
            return torch.bfloat16

    runtime = object.__new__(ZImageRuntime)
    runtime.assembled = cast("Any", Assembled())
    runtime._samplers = torch_sampler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._schedulers = torch_scheduler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._guidance = None  # pyright: ignore[reportPrivateUsage]
    condition = Conditioning(torch.zeros((1, 2, 2560)), None)
    with pytest.raises(ZImageRuntimeError, match="16"):
        runtime.sample(
            torch.zeros((1, 4, 2, 2)),
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
        )
    latent = torch.zeros((1, 16, 2, 2))
    with pytest.raises(TypeError, match="exact ZImageControlConditioning"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            control=cast("Any", object()),
        )
