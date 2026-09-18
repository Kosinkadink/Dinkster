"""Schedule construction is inherited; families supply only their sigma space."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from dinkster_inference import (
    FLUX_DEV,
    Conditioning,
    ContinuousEDMSigmas,
    CustomSamplingRequest,
    CustomSamplingResult,
    DiscreteSigmas,
    FlowSigmas,
    FluxFlowSigmas,
    ModelFamily,
    MultiStreamLatent,
    PreparedMultiStreamConditioning,
    SamplingGuidance,
    SamplingSegment,
    SigmaSpace,
    SparseLatent,
    sampling_sigmas,
)
from dinkster_inference_torch.denoise import prepare_multistream_noise, prepare_noise
from dinkster_inference_torch.sampling_runtime import (
    DenseOrSparseSamplingRuntime,
    MultiStreamSamplingRuntime,
    SamplingRuntime,
    SingleStreamSamplingRuntime,
)
from dinkster_inference_torch.schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    sd_turbo_sigmas,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry
from dinkster_inference_torch.sparse import make_sparse_support, pack_sparse_latent
from golden_files import assert_reference_schedule, load_platform_golden

# ComfyUI b78cec87, SDXL normal scheduler, 30 steps.
SDXL_NORMAL_30_CPU_B78CEC87 = (
    14.614640235900879,
    11.917510032653809,
    9.81415843963623,
    8.158439636230469,
    6.843043327331543,
    5.7885422706604,
    4.935636520385742,
    4.239697456359863,
    3.666886329650879,
    3.1913328170776367,
    2.793142318725586,
    2.456881284713745,
    2.170515537261963,
    1.9245861768722534,
    1.7116155624389648,
    1.5256513357162476,
    1.361922264099121,
    1.216577410697937,
    1.086482286453247,
    0.9690618515014648,
    0.862174391746521,
    0.7640063166618347,
    0.6729845404624939,
    0.5876831412315369,
    0.506723940372467,
    0.4286181628704071,
    0.35146668553352356,
    0.2721824645996094,
    0.18345095217227936,
    0.029167160391807556,
    0.0,
)

SDXL_NORMAL_30_CUDA_B78CEC87 = (
    *SDXL_NORMAL_30_CPU_B78CEC87[:8],
    3.6668860912323,
    SDXL_NORMAL_30_CPU_B78CEC87[9],
    2.793142080307007,
    *SDXL_NORMAL_30_CPU_B78CEC87[11:18],
    1.0864824056625366,
    0.9690617918968201,
    0.8621744513511658,
    *SDXL_NORMAL_30_CPU_B78CEC87[21:25],
    0.4286181926727295,
    *SDXL_NORMAL_30_CPU_B78CEC87[26:],
)


class _Runtime(SamplingRuntime):
    def __init__(self, space: SigmaSpace) -> None:
        self.space = space
        self._schedulers = torch_scheduler_registry()

    @property
    def family(self) -> ModelFamily:
        return FLUX_DEV

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return self.space


def test_add_noise_matches_flow_parameterization_and_latent_normalization() -> None:
    runtime = _Runtime(FlowSigmas())
    latent = torch.full((1, FLUX_DEV.single_stream_latent().channels, 2, 2), 0.25)
    noise = torch.full_like(latent, 2.0)
    descriptor = FLUX_DEV.single_stream_latent()
    normalized = (latent - descriptor.shift_factor) * descriptor.scale_factor
    expected_normalized = noise * 0.75 + normalized * 0.25
    expected = expected_normalized / descriptor.scale_factor + descriptor.shift_factor

    actual = runtime.custom_sampling_add_noise(latent, noise, 0.75)

    assert torch.equal(actual, expected)


def test_add_noise_does_not_shift_an_empty_latent_before_scaling() -> None:
    runtime = _Runtime(FlowSigmas())
    latent = torch.zeros((1, FLUX_DEV.single_stream_latent().channels, 2, 2))
    noise = torch.full_like(latent, 2.0)
    descriptor = FLUX_DEV.single_stream_latent()
    expected_normalized = noise * 0.25
    expected = expected_normalized / descriptor.scale_factor + descriptor.shift_factor

    actual = runtime.custom_sampling_add_noise(latent, noise, 0.25)

    assert torch.equal(actual, expected)


def test_add_noise_cleans_non_finite_output() -> None:
    runtime = _Runtime(FlowSigmas())
    latent = torch.zeros((1, FLUX_DEV.single_stream_latent().channels, 1, 1))
    noise = torch.full_like(latent, float("inf"))

    assert torch.equal(runtime.custom_sampling_add_noise(latent, noise, 1.0), latent)


def test_family_runtimes_inherit_all_four_sigma_helpers() -> None:
    import dinkster_inference_torch

    source = Path(dinkster_inference_torch.__file__).parent
    seen: set[type] = set()
    for path in (*source.glob("*_runtime.py"), source / "wiring.py"):
        module = importlib.import_module(f"dinkster_inference_torch.{path.stem}")
        for _, runtime in inspect.getmembers(module, inspect.isclass):
            if runtime.__module__ != module.__name__ or runtime is SamplingRuntime:
                continue
            if not hasattr(runtime, "custom_sampling_sigmas"):
                continue
            seen.add(runtime)
            for method in (
                "custom_sampling_sigmas",
                "custom_sampling_beta_sigmas",
                "custom_sampling_sd_turbo_sigmas",
                "custom_sampling_percent_to_sigma",
            ):
                assert getattr(runtime, method) is getattr(SamplingRuntime, method), (
                    runtime.__name__,
                    method,
                )
    assert len(seen) >= 22


@pytest.mark.parametrize(
    "name",
    (
        "AnimaDiffusionRuntime",
        "Krea2DiffusionRuntime",
        "Lumina2DiffusionRuntime",
        "MiniMaxMusic3DiffusionRuntime",
        "QwenImageRuntime",
        "QwenImageDiffusionRuntime",
        "SeedVR2DiffusionRuntime",
        "ZImageRuntime",
        "FluxRuntime",
        "SDRuntime",
        "Flux2Runtime",
        "Flux2DiffusionRuntime",
    ),
)
def test_dense_runtimes_inherit_one_ksampler_composition(name: str) -> None:
    import dinkster_inference_torch

    runtime = getattr(dinkster_inference_torch, name)
    assert "sample" not in runtime.__dict__
    assert runtime.sample is SingleStreamSamplingRuntime.sample


def test_multistream_runtimes_inherit_one_ksampler_composition() -> None:
    import dinkster_inference_torch

    source = Path(dinkster_inference_torch.__file__).parent
    seen: set[type] = set()
    for path in source.glob("*_runtime.py"):
        module = importlib.import_module(f"dinkster_inference_torch.{path.stem}")
        for _, runtime in inspect.getmembers(module, inspect.isclass):
            if (
                runtime.__module__ != module.__name__
                or runtime is MultiStreamSamplingRuntime
                or not issubclass(runtime, MultiStreamSamplingRuntime)
            ):
                continue
            seen.add(runtime)
            assert "sample_multistream" not in runtime.__dict__
            assert runtime.sample_multistream is MultiStreamSamplingRuntime.sample_multistream
            assert "run_ksampler_as_custom" not in runtime.__dict__
            assert (
                runtime.run_ksampler_as_custom is MultiStreamSamplingRuntime.run_ksampler_as_custom
            )
    assert len(seen) >= 7


def test_dense_or_sparse_runtime_inherits_one_ksampler_composition() -> None:
    from dinkster_inference_torch.trellis2_runtime import Trellis2DiffusionRuntime

    assert "sample" not in Trellis2DiffusionRuntime.__dict__
    assert Trellis2DiffusionRuntime.sample is DenseOrSparseSamplingRuntime.sample
    assert "custom_sampling_only" not in Trellis2DiffusionRuntime.__dict__
    assert Trellis2DiffusionRuntime.custom_sampling_only is True
    assert (
        Trellis2DiffusionRuntime.custom_sampling_only
        is DenseOrSparseSamplingRuntime.custom_sampling_only
    )


@pytest.mark.parametrize("representation", ("dense", "sparse"))
def test_dense_or_sparse_seam_inherits_ksampler_composition(representation: str) -> None:
    class SeamRuntime(DenseOrSparseSamplingRuntime):
        def __init__(self) -> None:
            self._samplers = torch_sampler_registry()
            self._schedulers = torch_scheduler_registry()
            self.arguments: dict[str, object] = {}

        @property
        def family(self) -> ModelFamily:
            return replace(FLUX_DEV, id="dinkster.synthetic")

        def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
            return FlowSigmas()

        def check_custom_sampling(self, request: object, **kwargs: object) -> None:
            pass

        def sample_custom(
            self,
            latent: torch.Tensor | SparseLatent[torch.Tensor],
            **kwargs: object,
        ) -> CustomSamplingResult[object]:
            self.arguments = dict(kwargs)
            return CustomSamplingResult(latent, None)

    dense = torch.zeros((1, 4, 2, 2))
    if representation == "dense":
        latent: torch.Tensor | SparseLatent[torch.Tensor] = dense
        mask: torch.Tensor | SparseLatent[torch.Tensor] = torch.ones_like(dense)
    else:
        support = make_sparse_support(
            torch.tensor(((0, 1, 2, 3),), dtype=torch.int32),
            (1,),
            8,
            (-0.5, -0.5, -0.5),
            (0.125, 0.125, 0.125),
        )
        latent = pack_sparse_latent(support, torch.zeros((1, 4)))
        mask = pack_sparse_latent(support, torch.ones((1, 4)))
    steps: list[object] = []
    states: list[object] = []
    runtime = SeamRuntime()
    result = runtime.sample(
        latent,
        cond=Conditioning(torch.zeros((1, 1, 4))),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        seed=11,
        denoise_mask=mask,
        on_step=steps.append,
        on_state=states.append,
        schedule_device="cpu",
        device="cpu",
        compute_dtype=torch.float32,
    )

    assert result is latent
    assert runtime.arguments["denoise_mask"] is mask
    assert runtime.arguments["on_step"] == steps.append
    assert runtime.arguments["on_state"] == states.append
    assert runtime.arguments["capture_denoised"] is False
    assert "device" not in runtime.arguments
    assert "compute_dtype" not in runtime.arguments


@pytest.mark.parametrize("add_noise", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_multistream_seam_inherits_noise_masks_and_callbacks(
    add_noise: bool, dtype: torch.dtype
) -> None:
    class SeamRuntime(MultiStreamSamplingRuntime):
        def __init__(self) -> None:
            self._samplers = torch_sampler_registry()
            self._schedulers = torch_scheduler_registry()
            self.arguments: dict[str, object] = {}
            self.admission: dict[str, object] = {}

        @property
        def family(self) -> ModelFamily:
            return replace(FLUX_DEV, id="dinkster.synthetic")

        @property
        def conditioning_identity(self) -> str:
            return "test:synthetic"

        def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
            return FluxFlowSigmas()

        def check_custom_sampling(self, request: object, **kwargs: object) -> None:
            self.admission = kwargs

        def sample_custom(
            self,
            latent: MultiStreamLatent[torch.Tensor],
            *,
            noise: object,
            cond: object,
            cfg: object,
            request: object,
            seed: int = 0,
            guidance: object = None,
            denoise_mask: object = None,
            inpaint: object = None,
            context_windows: object = None,
            on_step: object = None,
            on_state: object = None,
        ) -> CustomSamplingResult[MultiStreamLatent[torch.Tensor]]:
            self.arguments = {
                "noise": noise,
                "cond": cond,
                "mask": denoise_mask,
                "on_step": on_step,
                "on_state": on_state,
            }
            return CustomSamplingResult(latent, None)

    runtime = SeamRuntime()
    latent: MultiStreamLatent[torch.Tensor] = MultiStreamLatent.from_pairs(
        (
            ("video", torch.zeros((2, 4, 1, 2, 2), dtype=dtype)),
            ("audio", torch.zeros((2, 3, 7), dtype=dtype)),
        )
    )
    mask = latent.map(torch.ones_like)
    steps: list[object] = []
    states: list[object] = []
    conditioning = object()
    result = runtime.sample_multistream(
        latent,
        conditioning=conditioning,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=3,
        denoise=1.0,
        seed=71,
        segment=SamplingSegment(3, 0, 3, add_noise, False),
        noise_inds=(2, 2),
        denoise_mask=mask,
        on_step=steps.append,
        on_state=states.append,
    )
    assert result is latent
    noise = runtime.arguments["noise"]
    assert isinstance(noise, MultiStreamLatent)
    expected = (
        prepare_multistream_noise(latent, 71, (2, 2))
        if add_noise
        else latent.map(lambda stream: torch.zeros_like(stream, dtype=torch.float32))
    )
    for role in latent.roles:
        assert torch.equal(noise.by_role(role), expected.by_role(role))
        assert noise.by_role(role).dtype == torch.float32
    prepared = runtime.arguments["cond"]
    assert isinstance(prepared, PreparedMultiStreamConditioning)
    assert prepared.payload is conditioning
    assert prepared.runtime_identity == runtime.conditioning_identity
    assert runtime.arguments["mask"] is mask
    assert runtime.arguments["on_step"] == steps.append
    assert runtime.arguments["on_state"] == states.append
    assert runtime.admission["has_denoise_mask"] is True


@pytest.mark.parametrize("add_noise", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_seam_only_runtime_inherits_schedule_noise_and_callback_forwarding(
    add_noise: bool, dtype: torch.dtype
) -> None:
    class SeamRuntime(SingleStreamSamplingRuntime):
        def __init__(self) -> None:
            self._samplers = torch_sampler_registry()
            self._schedulers = torch_scheduler_registry()
            self.arguments: dict[str, object] = {}

        @property
        def family(self) -> ModelFamily:
            return replace(FLUX_DEV, id="dinkster.synthetic")

        def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
            return FluxFlowSigmas()

        def check_custom_sampling(self, request: object, **kwargs: object) -> None:
            pass

        def sample_custom(
            self,
            latent: torch.Tensor,
            *,
            noise: object,
            cond: object,
            cfg: object,
            request: object,
            seed: int = 0,
            guidance: object = None,
            denoise_mask: object = None,
            inpaint: object = None,
            context_windows: object = None,
            on_step: object = None,
            on_state: object = None,
        ) -> CustomSamplingResult[torch.Tensor]:
            self.arguments = {
                "noise": noise,
                "cond": cond,
                "request": request,
                "denoise_mask": denoise_mask,
                "on_step": on_step,
                "on_state": on_state,
            }
            return CustomSamplingResult(latent, None)

    runtime = SeamRuntime()
    latent = torch.zeros((2, 4, 2, 2), dtype=dtype)
    cond = Conditioning(torch.zeros((2, 3, 8)))
    mask = torch.ones_like(latent)
    steps: list[object] = []
    states: list[object] = []
    result = runtime.sample(
        latent,
        cond=cond,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=3,
        seed=71,
        segment=SamplingSegment(3, 0, 3, add_noise, False),
        noise_inds=(2, 2),
        denoise_mask=mask,
        on_step=steps.append,
        on_state=states.append,
    )
    assert result is latent
    args = runtime.arguments
    noise = args["noise"]
    assert isinstance(noise, torch.Tensor)
    expected = prepare_noise(latent, 71, (2, 2)) if add_noise else torch.zeros_like(latent)
    assert torch.equal(noise, expected)
    assert noise.dtype == dtype
    request = args["request"]
    assert isinstance(request, CustomSamplingRequest)
    assert request.sigmas == runtime.custom_sampling_sigmas("dinkster.simple", 3, 1.0)
    assert args["cond"] is cond
    assert args["denoise_mask"] is mask
    assert args["on_step"] == steps.append
    assert args["on_state"] == states.append


@pytest.mark.parametrize(
    "space",
    (
        FlowSigmas(shift=3.0),
        FlowSigmas(shift=1.0, multiplier=1.0),
        FluxFlowSigmas(shift=1.15, timesteps=10000),
        DiscreteSigmas.linear_beta(),
        ContinuousEDMSigmas(),
    ),
)
def test_sigma_helpers_preserve_reference_kernel_values(space: SigmaSpace) -> None:
    runtime = _Runtime(space)
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    assert runtime.custom_sampling_sigmas("dinkster.simple", 5, 0.7) == sampling_sigmas(
        scheduler, space, 5, denoise=0.7
    )
    assert runtime.custom_sampling_beta_sigmas(5, 0.6, 0.6) == custom_beta_sigmas(
        space, 5, 0.6, 0.6
    )
    if isinstance(space, DiscreteSigmas):
        assert runtime.custom_sampling_sd_turbo_sigmas(5, 0.7) == sd_turbo_sigmas(space, 5, 0.7)
    else:
        with pytest.raises(ValueError, match="require a discrete sigma space"):
            runtime.custom_sampling_sd_turbo_sigmas(5, 0.7)
        with pytest.raises(ValueError, match="require a discrete sigma space"):
            sd_turbo_sigmas(space, 5, 0.7)
    for percent in (0.0, 0.125, 0.5, 0.9, 1.0):
        for actual in (False, True):
            assert runtime.custom_sampling_percent_to_sigma(
                percent, return_actual_sigma=actual
            ) == custom_percent_to_sigma(
                space, space.percent_to_sigma, percent, return_actual_sigma=actual
            )


def test_sdxl_normal_30_schedule_matches_reference_contract() -> None:
    golden = load_platform_golden(
        Path(__file__).parent / "goldens" / "sdxl_normal_30_cpu_b78cec87.json",
        allow_portable_fallback=True,
    )
    assert golden["reference"]["commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    assert_reference_schedule(golden["sigmas"], SDXL_NORMAL_30_CPU_B78CEC87)
    assert_reference_schedule(
        _Runtime(DiscreteSigmas.linear_beta()).custom_sampling_sigmas(
            "dinkster.normal", 30, 1.0, device=torch.device("cpu")
        ),
        SDXL_NORMAL_30_CPU_B78CEC87,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sdxl_normal_30_schedule_matches_cuda_reference_exactly() -> None:
    actual = _Runtime(DiscreteSigmas.linear_beta()).custom_sampling_sigmas(
        "dinkster.normal", 30, 1.0, device=torch.device("cuda")
    )
    assert actual == SDXL_NORMAL_30_CUDA_B78CEC87
    assert tuple(
        index
        for index, (cpu_sigma, cuda_sigma) in enumerate(
            zip(SDXL_NORMAL_30_CPU_B78CEC87, actual, strict=True)
        )
        if cpu_sigma != cuda_sigma
    ) == (8, 10, 18, 19, 20, 25)


def test_sigma_helpers_refuse_unknown_schedulers_and_unsupported_shift() -> None:
    runtime = _Runtime(FlowSigmas())
    with pytest.raises(ValueError, match="unknown scheduler.*registered"):
        runtime.custom_sampling_sigmas("missing", 2, 1.0)
    with pytest.raises(ValueError, match="does not support a sampling shift"):
        runtime.custom_sampling_sigmas("dinkster.simple", 2, 1.0, sampling_shift=3.0)


def test_shift_is_resolved_by_the_runtime_space_hook() -> None:
    class ShiftedRuntime(_Runtime):
        supports_sampling_shift = True

        def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
            space = FlowSigmas(shift=1.0)
            return space if sampling_shift is None else replace(space, shift=sampling_shift)

    runtime = ShiftedRuntime(FlowSigmas())
    expected = _Runtime(FlowSigmas(shift=3.0))
    assert runtime.custom_sampling_sigmas(
        "dinkster.simple", 4, 1.0, sampling_shift=3.0
    ) == expected.custom_sampling_sigmas("dinkster.simple", 4, 1.0)
    assert runtime.custom_sampling_beta_sigmas(
        4, 0.6, 0.6, sampling_shift=3.0
    ) == expected.custom_sampling_beta_sigmas(4, 0.6, 0.6)
    with pytest.raises(ValueError, match="require a discrete sigma space"):
        runtime.custom_sampling_sd_turbo_sigmas(4, 1.0, sampling_shift=3.0)
    assert runtime.custom_sampling_percent_to_sigma(
        0.5, return_actual_sigma=True, sampling_shift=3.0
    ) == expected.custom_sampling_percent_to_sigma(0.5, return_actual_sigma=True)


@pytest.mark.parametrize(
    "module,factory,model,condition,shape",
    (
        ("anima", "_diffusion_runtime", "RecordingAnima", "_raw_conditioning", (1, 16, 1, 2, 2)),
        ("krea2", "_diffusion_runtime", "RecordingKrea2", "_filled_conditioning", (1, 16, 1, 2, 2)),
        ("lumina2", "diffusion_runtime", "RecordingLumina2", "conditioning", (1, 16, 2, 2)),
        ("chroma", "runtime_for", "RecordingChroma", "condition", (1, 16, 2, 2)),
        ("ideogram4", "diffusion_runtime", "RecordingIdeogram", "condition", (1, 128, 2, 2)),
    ),
)
def test_runtime_masks_preserve_protected_values_in_both_sampling_surfaces(
    module: str, factory: str, model: str, condition: str, shape: tuple[int, ...]
) -> None:
    fixtures = importlib.import_module(f"test_{module}_runtime")
    runtime = getattr(fixtures, factory)(getattr(fixtures, model)(value=0.5))
    conditioning = (
        getattr(fixtures, condition)(1.0) if module == "krea2" else getattr(fixtures, condition)()
    )
    latent = torch.ones(shape)
    mask = torch.ones_like(latent)
    mask[..., 0] = 0
    kwargs = {"cond": conditioning, "denoise_mask": mask}
    composed = runtime.sample(
        latent, sampler_id="dinkster.euler", scheduler_id="dinkster.simple", steps=2, **kwargs
    )
    sampler = torch_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    direct = runtime.sample_custom(
        latent,
        noise=prepare_noise(latent, 0),
        cfg=None,
        request=CustomSamplingRequest(
            sampler, (), runtime.custom_sampling_sigmas("dinkster.simple", 2, 1.0)
        ),
        **kwargs,
    ).output
    assert torch.equal(composed, direct)
    different_model = getattr(fixtures, factory)(getattr(fixtures, model)(value=9.0))
    different = different_model.sample(
        latent, sampler_id="dinkster.euler", scheduler_id="dinkster.simple", steps=2, **kwargs
    )
    assert torch.equal(composed[..., 0], different[..., 0])
    assert not torch.equal(composed[..., 1], different[..., 1])


@pytest.mark.parametrize("family", ("dinkster.minimax_h3", "dinkster.synthetic"))
def test_shared_guidance_admission_does_not_require_family_registration(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import distributed

    class Runtime(_Runtime):
        @property
        def family(self) -> ModelFamily:
            return replace(FLUX_DEV, id=family)

    config = distributed.DistributedSamplingConfig(0, 2, "guidance", "file:///group", "1" * 32)
    monkeypatch.setattr(distributed, "distributed_sampling_config", lambda: config)
    groups: list[int] = []

    def process_group() -> distributed.DistributedSamplingConfig:
        groups.append(config.world_size)
        return config

    monkeypatch.setattr(distributed, "ensure_process_group", process_group)
    sampler = torch_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    request = CustomSamplingRequest[torch.Tensor](sampler, (), (1.0, 0.0))
    cond = Conditioning(torch.zeros((1, 2, 3)))
    admission = Runtime(FluxFlowSigmas()).admit_distributed_guidance(
        request,
        cond=cond,
        cfg=SamplingGuidance(cond, 2.0),
        executor=None,
    )
    assert admission is not None
    assert admission.plan.needs_unconditional
    assert groups == [2]
