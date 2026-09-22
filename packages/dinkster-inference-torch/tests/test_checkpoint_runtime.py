from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    FLOAT32,
    FLUX_DEV,
    AttentionPolicy,
    ComponentPlan,
    Conditioning,
    CustomSamplingRequest,
    CustomSamplingResult,
    CustomSamplingRuntime,
    DenoiseMaskRuntime,
    FamilyRuntime,
    FlowSigmas,
    ModelFamily,
    Registry,
    SigmaSpace,
)
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.component_checkpoint import ComponentCheckpointPlan
from dinkster_inference_torch.assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from dinkster_inference_torch.attention import discover_attention_route_token
from dinkster_inference_torch.checkpoint_runtime import (
    ComponentAssembly,
    ComponentCheckpointRuntime,
    assemble_component_checkpoint,
)
from dinkster_inference_torch.sampling_execution import CustomSamplingCapabilities
from dinkster_inference_torch.sampling_runtime import SingleStreamSamplingRuntime
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry
from safetensors.torch import save_file


class Diffusion(SingleStreamSamplingRuntime):
    runtime_identity = "synthetic-checkpoint"
    supports_denoise_mask = True
    sampling_execution_registration = cast(
        "Any",
        SimpleNamespace(
            capabilities=CustomSamplingCapabilities(),
            forbidden_options=frozenset(),
            forbidden_options_message="",
        ),
    )

    def __init__(self) -> None:
        self._samplers = torch_sampler_registry()
        self._schedulers = torch_scheduler_registry()

        def dtype(role: str) -> torch.dtype | None:
            return torch.float32 if role == "diffusion" else None

        self.assembled = SimpleNamespace(
            diffusion=torch.nn.Linear(2, 2),
            compute_dtype=dtype,
        )
        self.calls: list[dict[str, Any]] = []

    @property
    def family(self) -> ModelFamily:
        return replace(FLUX_DEV, id="test.checkpoint")

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return FlowSigmas()

    def sample_custom(
        self, latent: torch.Tensor, *, noise: torch.Tensor, **options: Any
    ) -> CustomSamplingResult[torch.Tensor]:
        self.calls.append(options)
        return CustomSamplingResult(latent + noise, None)


def assembly(diffusion: Diffusion) -> ComponentAssembly:
    return ComponentAssembly(
        diffusion.family,
        "model",
        {"model": diffusion.assembled.diffusion, "words": torch.nn.Identity()},
        {"model": torch.float32, "words": torch.float32},
        {},
    )


def test_composed_runtime_preserves_inner_assembly_and_shared_sampling_engine() -> None:
    diffusion = Diffusion()
    original = diffusion.assembled
    assembled = assembly(diffusion)
    prepared = Conditioning(torch.zeros(1, 2, 2), None)

    def encode_text(_text: str) -> Conditioning[torch.Tensor]:
        return prepared

    def encode(tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 2

    def decode(tensor: torch.Tensor) -> torch.Tensor:
        return tensor - 2

    text = SimpleNamespace(encode_text=encode_text)
    codec = SimpleNamespace(encode=encode, decode=decode)
    runtime = ComponentCheckpointRuntime(diffusion, assembled, text_runtime=text, codec=codec)
    assert isinstance(runtime, FamilyRuntime)
    assert isinstance(runtime, CustomSamplingRuntime)
    assert isinstance(runtime, DenoiseMaskRuntime)
    assert runtime.sample.__self__ is diffusion
    assert runtime.sample_custom.__self__ is diffusion
    assert runtime.custom_sampling_sigmas.__self__ is diffusion
    assert runtime.encode_text("hello") is prepared
    latent = torch.zeros(1, 2, 2, 2)
    options: dict[str, Any] = dict(
        cond=prepared, sampler_id="dinkster.euler", scheduler_id="dinkster.simple", steps=3, seed=17
    )
    expected = diffusion.sample(latent, **options)
    actual = runtime.sample(latent, **options)
    assert torch.equal(actual, expected)
    assert len(diffusion.calls) == 2
    assert runtime.runtime_identity == diffusion.runtime_identity
    assert runtime.decode_latent(runtime.encode_content(latent)).equal(latent)
    assert diffusion.assembled is original
    assert assembled.diffusion is original.diffusion
    diffusion.supports_denoise_mask = False
    assert runtime.supports_denoise_mask is False
    replacement = Diffusion()
    replacement.assembled = original
    rebound = runtime.with_component_sampling_runtime(replacement)
    assert rebound.component_sampling_runtime is replacement
    assert rebound.sample_custom.__self__ is replacement
    assert rebound.encode_text("hello") is prepared
    assert rebound.decode_latent(rebound.encode_content(latent)).equal(latent)


def test_component_assembly_snapshots_named_maps() -> None:
    diffusion = Diffusion()
    original = assembly(diffusion)
    modules = dict(original.components)
    dtypes = dict(original.component_dtypes)
    current = replace(original, components=modules, component_dtypes=dtypes)
    modules.clear()
    dtypes.clear()
    assert tuple(current.components) == ("model", "words")
    assert current.compute_dtype("diffusion") is torch.float32
    assert current.compute_dtype("model") is torch.float32
    assert current.compute_dtype("missing") is None
    with pytest.raises(TypeError):
        cast("Any", current.components)["extra"] = torch.nn.Identity()
    with pytest.raises(ValueError, match="every realized component"):
        replace(current, component_dtypes={"model": torch.float32})
    with pytest.raises(ValueError, match="must not share"):
        replace(current, components=dict.fromkeys(("model", "words"), current.diffusion))


@pytest.mark.parametrize("mismatch", ["module", "dtype", "family"])
def test_composed_runtime_rejects_model_contract_mismatch(mismatch: str) -> None:
    diffusion = Diffusion()
    current = assembly(diffusion)
    if mismatch == "module":
        current = replace(
            current, components={**current.components, "model": torch.nn.Linear(2, 2)}
        )
    elif mismatch == "dtype":
        current = replace(
            current, component_dtypes={**current.component_dtypes, "model": torch.bfloat16}
        )
    else:
        current = replace(current, family=replace(current.family, id="test.other"))
    with pytest.raises(ValueError, match="checkpoint and diffusion runtime"):
        ComponentCheckpointRuntime(diffusion, current)


def test_missing_companions_are_diagnostic_not_guessed() -> None:
    diffusion = Diffusion()
    runtime = ComponentCheckpointRuntime(diffusion, assembly(diffusion))
    with pytest.raises(ValueError, match="no text binding.*model.*words"):
        runtime.encode_text("hello")
    with pytest.raises(ValueError, match="no codec binding.*model.*words"):
        runtime.decode_latent(torch.zeros(1))


def test_text_nodes_preserve_architecture_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    import dinkster_inference as inference
    from dinkster_compat_comfy import native_arm
    from dinkster_inference_torch.anima_runtime import AnimaConditioning, AnimaTextRuntime

    raw = AnimaConditioning(
        torch.zeros(1, 4, 12),
        None,
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, 1),
    )

    class Text(AnimaTextRuntime):
        def __init__(self) -> None:
            pass

        def encode_text(self, text: str) -> AnimaConditioning:
            return raw

    diffusion = Diffusion()
    runtime = ComponentCheckpointRuntime(
        diffusion,
        assembly(diffusion),
        text_runtime=Text(),
        prepare_conditioning="materialize_anima_conditioning",
    )

    def stage(_role: str) -> nullcontext[None]:
        return nullcontext()

    handle = SimpleNamespace(
        runtime=runtime,
        stage=stage,
        recipe=SimpleNamespace(runtime_identity=runtime.runtime_identity),
    )

    def resolve(*_args: Any) -> Any:
        return handle

    monkeypatch.setattr(native_arm, "_native_handle", resolve)
    monkeypatch.setattr(native_arm, "_require_provider_runtime", resolve)
    arm: Any = native_arm
    rows = arm.NativeClipTextEncode.execute(text="hello", clip=handle)["conditioning"]
    restored, inpaint = arm._conditioning(rows, "positive", torch, inference)
    assert restored is raw
    assert inpaint is None
    carrier = arm.GenerationClipTextEncode.execute(text="hello", clip=handle)["conditioning"]
    restored = runtime.prepare_single_stream_conditioning(carrier)
    assert isinstance(restored, AnimaConditioning)
    torch.testing.assert_close(restored.embeddings, raw.embeddings, atol=0, rtol=0)
    torch.testing.assert_close(restored.t5xxl_ids, raw.t5xxl_ids, atol=0, rtol=0)
    torch.testing.assert_close(restored.t5xxl_weights, raw.t5xxl_weights, atol=0, rtol=0)


@pytest.mark.parametrize("attention_policy", ["auto", "sdpa"])
@pytest.mark.parametrize("architecture", ["anima", "krea2"])
def test_checkpoint_construction_matches_existing_real_model_loading(
    tmp_path: Path, architecture: str, attention_policy: AttentionPolicy
) -> None:
    import test_anima_model
    import test_krea2_dit

    fixture: Any = test_anima_model if architecture == "anima" else test_krea2_dit
    case = fixture.CASES[0]
    original = fixture.build_model(case)
    path = tmp_path / "checkpoint.safetensors"
    prefix = "model.diffusion_model."
    state = original.state_dict()
    save_file({prefix + key: value.contiguous() for key, value in state.items()}, path)
    part = ComponentPlan(
        "diffusion",
        path,
        fixture.case_config(case),
        {key: prefix + key for key in state},
        dict.fromkeys(state, FLOAT32),
        {},
    )
    registration = default_component_registry().get(f"dinkster.{architecture}")
    assert registration is not None
    runtime = assemble_component_checkpoint(
        ComponentCheckpointPlan(registration, (("diffusion", part),)),
        diffusion_dtype=torch.float32,
        text_dtype=torch.float32,
        vae_dtype=torch.float32,
        attention_policy=attention_policy,
        attention_route_token=(
            None if attention_policy == "auto" else discover_attention_route_token(attention_policy)
        ),
    )
    baseline = _load_component(part, type(original), compute_dtype=torch.float32, fp8_matmul=False)
    assert type(runtime.assembled.diffusion) is type(baseline)
    assert runtime.assembled.diffusion is runtime.diffusion.assembled.diffusion
    assert runtime.assembled.compute_dtype("diffusion") == torch.float32
    assert cast("Any", runtime.attention_status["flux"]).requested_policy == attention_policy
    for key, actual in runtime.assembled.diffusion.state_dict().items():
        expected = baseline.state_dict()[key]
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    inputs = fixture.case_inputs(case)
    options: dict[str, Any] = {}
    if architecture == "anima":
        options = dict(t5xxl_ids=inputs[3], t5xxl_weights=inputs[4])
        inputs = inputs[:3]
    with torch.inference_mode():
        torch.testing.assert_close(
            runtime.assembled.diffusion(*inputs, **options),
            baseline(*inputs, **options),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("attention_policy", ["auto", "sdpa"])
def test_seedvr2_checkpoint_preserves_real_diffusion_weights_and_execution(
    tmp_path: Path, attention_policy: AttentionPolicy
) -> None:
    from dinkster_inference_torch import INITLESS, seedvr2_component
    from test_seedvr2_models import fill_module

    config: Any = SimpleNamespace(
        norm_eps=1e-5,
        layers=2,
        mlp_type="swiglu",
        width=128,
        heads=1,
        separate_layers=1,
        rope_type="mmrope3d",
        rope_dim=128,
        vid_out_norm=True,
    )
    builder = seedvr2_component._build_diffusion  # pyright: ignore[reportPrivateUsage]
    original = builder(config, operations=INITLESS)
    fill_module(original)
    state = original.state_dict()
    path = tmp_path / "seedvr2.safetensors"
    prefix = "model.diffusion_model."
    save_file({prefix + key: value.contiguous() for key, value in state.items()}, path)
    part = ComponentPlan(
        "diffusion",
        path,
        config,
        {key: prefix + key for key in state},
        dict.fromkeys(state, FLOAT32),
        {},
    )
    descriptor = default_component_registry().get("dinkster.seedvr2")
    assert descriptor is not None
    runtime = assemble_component_checkpoint(
        ComponentCheckpointPlan(descriptor, (("diffusion", part),)),
        diffusion_dtype=torch.float32,
        text_dtype=torch.float32,
        vae_dtype=torch.float16,
        attention_policy=attention_policy,
        attention_route_token=(
            None if attention_policy == "auto" else discover_attention_route_token(attention_policy)
        ),
    )
    baseline = _load_component(part, builder, compute_dtype=torch.float32, fp8_matmul=False)
    model = runtime.assembled.diffusion
    assert model is runtime.component_sampling_runtime.assembled.diffusion
    assert runtime.supports_denoised_capture
    assert runtime.sample_custom.__self__ is runtime.component_sampling_runtime
    assert "positive_conditioning" not in model.state_dict()
    parameters = dict(baseline.named_parameters())
    for key, actual in (*model.named_parameters(), *model.named_buffers()):
        expected = parameters[key] if key in parameters else baseline.get_buffer(key)
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    generator = torch.Generator().manual_seed(6)
    latent = torch.randn((1, 16, 1, 4, 4), generator=generator)
    timestep = torch.tensor([0.5])
    context = torch.randn((1, 3, 5120), generator=generator)
    condition = torch.randn((1, 17, 1, 4, 4), generator=generator)
    options = {"cond_or_uncond": [0]}
    with torch.inference_mode():
        torch.testing.assert_close(
            model(latent, timestep, context, condition=condition, transformer_options=options),
            baseline(latent, timestep, context, condition=condition, transformer_options=options),
            rtol=0,
            atol=0,
        )


def test_multistream_checkpoint_preserves_protocols_conditioning_and_sampling() -> None:
    from dinkster_inference import MultiStreamConditioningRuntime, MultiStreamFamilyRuntime
    from test_ltxv_runtime import (
        _prepared,  # pyright: ignore[reportPrivateUsage]
        _runtime,  # pyright: ignore[reportPrivateUsage]
        _sample,  # pyright: ignore[reportPrivateUsage]
        _video,  # pyright: ignore[reportPrivateUsage]
    )

    diffusion, model = _runtime()
    current = ComponentAssembly(
        diffusion.family, "diffusion", {"diffusion": model}, {"diffusion": torch.float32}, {}
    )
    runtime = ComponentCheckpointRuntime(diffusion, current)
    assert isinstance(runtime, MultiStreamFamilyRuntime)
    assert isinstance(runtime, MultiStreamConditioningRuntime)
    assert isinstance(runtime, CustomSamplingRuntime)
    assert not isinstance(runtime, FamilyRuntime)
    assert not hasattr(runtime, "sample")
    assert runtime.sample_multistream.__self__ is diffusion
    assert runtime.run_ksampler_as_custom.__self__ is diffusion
    assert runtime.prepare_conditioning.__self__ is diffusion
    assert runtime.conditioning_identity == diffusion.conditioning_identity
    assert runtime.conditioning_identity != runtime.runtime_identity
    torch.testing.assert_close(
        _sample(cast("Any", runtime), _video(), _prepared(), seed=17, steps=2),
        _sample(diffusion, _video(), _prepared(), seed=17, steps=2),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("roles", [("diffusion",), ("diffusion", "t5xxl", "vae")])
def test_ltxv_checkpoint_assembly_binds_optional_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, roles: tuple[str, ...]
) -> None:
    from dinkster_inference_torch import checkpoint_runtime, ltx_component
    from test_ltxv_runtime import (
        _VAE,  # pyright: ignore[reportPrivateUsage]
        _runtime,  # pyright: ignore[reportPrivateUsage]
    )

    class Text(torch.nn.Module):
        config = SimpleNamespace(model_type="t5")

    class VAE(_VAE, torch.nn.Module):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            _VAE.__init__(self)

    diffusion, model = _runtime()
    modules: dict[str, torch.nn.Module] = {
        "diffusion": model,
        "t5xxl": Text(),
        "vae": VAE(),
    }

    def realize(plan: ComponentPlan[object], **_options: object) -> torch.nn.Module:
        return modules[plan.component]

    def build_runtime(*_args: object, **_options: object) -> object:
        return diffusion

    monkeypatch.setattr(ltx_component, "realize_ltxv_component", realize)
    monkeypatch.setattr(checkpoint_runtime, "build_component_runtime", build_runtime)
    descriptor = default_component_registry().get("dinkster.ltxv")
    assert descriptor is not None
    plans = tuple(
        (
            role,
            ComponentPlan(role, tmp_path / f"{role}.safetensors", None, {}, {}, {}),
        )
        for role in roles
    )
    runtime = assemble_component_checkpoint(
        ComponentCheckpointPlan(descriptor, plans),
        diffusion_dtype=torch.float32,
        text_dtype=torch.bfloat16,
        vae_dtype=torch.float16,
    )
    assert tuple(runtime.assembled.components) == roles
    if roles == ("diffusion",):
        with pytest.raises(ValueError, match="no text binding"):
            runtime.encode_text("hello")
        with pytest.raises(ValueError, match="no codec binding"):
            runtime.decode_latent(torch.zeros(1))
    else:
        assert runtime.assembled.compute_dtype("t5xxl") is torch.bfloat16
        assert runtime.assembled.compute_dtype("vae") is torch.float16
        assert runtime.codec.vae is modules["vae"]
        assert runtime.codec.codec.compute_dtype is torch.float16


def test_ltxv_checkpoint_text_nodes_preserve_token_layout_and_frame_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy import native_arm
    from dinkster_inference import LTXV_2B_V09_CONFIG, PreparedMultiStreamConditioning
    from dinkster_inference_torch._conditioning_layout import declare_text_conditioning
    from dinkster_inference_torch.ltx_component import LTXVTextRuntime
    from dinkster_inference_torch.ltxv_runtime import LTXVPreparedConditioning
    from test_ltxv_runtime import _runtime  # pyright: ignore[reportPrivateUsage]

    raw = declare_text_conditioning(Conditioning(torch.ones(1, 4, 4096), None), 3)

    class Text(LTXVTextRuntime):
        def __init__(self) -> None:
            pass

        def encode_text(self, text: str, **_options: Any) -> Conditioning[torch.Tensor]:
            return raw

    diffusion, model = _runtime()
    model.config = cast("Any", LTXV_2B_V09_CONFIG)
    assembled = ComponentAssembly(
        diffusion.family, "diffusion", {"diffusion": model}, {"diffusion": torch.float32}, {}
    )
    runtime = ComponentCheckpointRuntime(diffusion, assembled, text_runtime=Text())

    def stage(_role: str) -> nullcontext[None]:
        return nullcontext()

    handle = SimpleNamespace(runtime=runtime, stage=stage)

    def resolve(*_args: Any) -> SimpleNamespace:
        return handle

    monkeypatch.setattr(native_arm, "_native_handle", resolve)
    monkeypatch.setattr(native_arm, "_require_provider_runtime", resolve)
    arm: Any = native_arm
    rows = arm.NativeClipTextEncode.execute(text="hello", clip=handle)["conditioning"]
    prepared = rows[0][0]
    assert isinstance(prepared, PreparedMultiStreamConditioning)
    assert prepared.runtime_identity == diffusion.conditioning_identity
    carrier = arm.GenerationClipTextEncode.execute(text="hello", clip=handle)["conditioning"]
    canonical = runtime.prepare_conditioning(carrier)
    assert isinstance(prepared.payload, LTXVPreparedConditioning)
    assert prepared.payload.attention_tokens == canonical.attention_tokens == 3
    assert prepared.payload.frame_rate == canonical.frame_rate == 25.0
    torch.testing.assert_close(prepared.payload.text, canonical.text, rtol=0, atol=0)
    assert runtime.prepare_conditioning(carrier, frame_rate=12.0).frame_rate == 12.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ltxv_checkpoint_codec_preserves_existing_float32_boundary(dtype: torch.dtype) -> None:
    from dinkster_inference_torch.ltxv_runtime import LTXVVideoCodecRuntime, checkpoint_codec
    from test_ltxv_runtime import _VAE  # pyright: ignore[reportPrivateUsage]

    diffusion = Diffusion()
    vae = _VAE()
    assembled = replace(
        assembly(diffusion),
        components={"model": diffusion.assembled.diffusion, "vae": cast("Any", vae)},
        component_dtypes={"model": torch.float32, "vae": dtype},
    )
    codec = checkpoint_codec(assembled)
    assert codec is not None
    reference = LTXVVideoCodecRuntime(cast("Any", vae), compute_dtype=dtype)
    runtime = ComponentCheckpointRuntime(diffusion, assembled, codec=codec)
    content = torch.linspace(0, 1, 33 * 65).reshape(1, 1, 1, 33, 65)
    actual = runtime.encode_content(content.clone())
    expected = reference.encode_content(content.clone())
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual = runtime.decode_latent(actual.clone())
    expected = reference.decode_latent(expected.clone())
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert checkpoint_codec(assembly(diffusion)) is None


@pytest.mark.parametrize("attention_policy", ["auto", "sdpa"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ltxv_realizer_preserves_real_diffusion_weights_and_execution(
    tmp_path: Path, attention_policy: AttentionPolicy, device: str
) -> None:
    from dinkster_inference_torch.assemble import (
        _select_attention_runtime,  # pyright: ignore[reportPrivateUsage]
    )
    from dinkster_inference_torch.ltx_component import realize_ltxv_component
    from dinkster_inference_torch.module_residency import enroll_component
    from test_ltx_model import CASES, build_model, case_inputs

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA residency requires the GPU validation environment")
    original = build_model(CASES[0])
    state = original.state_dict()
    path = tmp_path / "ltxv.safetensors"
    prefix = "model.diffusion_model."
    save_file({prefix + key: value.contiguous() for key, value in state.items()}, path)
    part: ComponentPlan[object] = ComponentPlan(
        "diffusion",
        path,
        original.config,
        {key: prefix + key for key in state},
        dict.fromkeys(state, FLOAT32),
        {},
    )
    token = None if attention_policy == "auto" else discover_attention_route_token(attention_policy)
    kernels, _ = _select_attention_runtime(attention_policy, token)
    model = realize_ltxv_component(part, compute_dtype=torch.float32, attention_kernels=kernels)
    baseline = _load_component(part, type(original), compute_dtype=torch.float32, fp8_matmul=False)
    for key, actual in model.state_dict().items():
        expected = baseline.state_dict()[key]
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    latent, timestep, context, mask, frame_rate, denoise_mask, guides = case_inputs(CASES[0])
    latent, timestep, context = (value.to(device) for value in (latent, timestep, context))
    options = dict(
        attention_mask=None if mask is None else mask.to(device),
        frame_rate=frame_rate,
        denoise_mask=None if denoise_mask is None else denoise_mask.to(device),
        guides=guides,
    )
    baseline.to(device)
    mechanism = enroll_component(model, load_device=device, offload_device="cpu")
    with torch.inference_mode():
        expected_output = baseline(latent, timestep, context, **options)
        for budget in (None, 0, None):
            mechanism.unload()
            mechanism.partially_load(budget)
            assert bool(mechanism.loaded_unit_names()) == (budget is None)
            torch.testing.assert_close(
                model(latent, timestep, context, **options), expected_output, rtol=0, atol=0
            )
    mechanism.unload()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
)
@pytest.mark.parametrize("attention_policy", ["auto", "sdpa"])
@pytest.mark.parametrize("architecture", ["chroma", "radiance", "radiance-conv-x0"])
def test_chroma_checkpoint_preserves_component_weights_and_execution(
    tmp_path: Path, architecture: str, attention_policy: AttentionPolicy, device: str
) -> None:
    from functools import partial

    import dinkster_inference as inference
    from dinkster_compat_comfy import native_arm
    from dinkster_inference import ChromaRadianceConfig
    from dinkster_inference_torch import INITLESS, chroma_component
    from dinkster_inference_torch.attention import select_attention
    from dinkster_inference_torch.z_image import PixelSpaceCodec
    from test_chroma import initialize, small_chroma_config, small_radiance_config

    config = (
        small_chroma_config()
        if architecture == "chroma"
        else small_radiance_config(
            nerf_final_head_type="conv" if architecture == "radiance-conv-x0" else "linear",
            use_x0=architecture == "radiance-conv-x0",
            use_sequential_txt_ids=architecture == "radiance-conv-x0",
        )
    )
    builder = partial(
        chroma_component._build_diffusion,  # pyright: ignore[reportPrivateUsage]
        attention_kernel=select_attention("flux").kernel,
    )
    original = builder(config, operations=INITLESS)
    initialize(original)
    state = original.state_dict()
    prefix = "model.diffusion_model."
    path = tmp_path / "chroma.safetensors"
    save_file({prefix + key: value.contiguous() for key, value in state.items()}, path)
    part = ComponentPlan(
        "diffusion",
        path,
        config,
        {key: prefix + key for key in state},
        dict.fromkeys(state, FLOAT32),
        {},
    )
    descriptor = default_component_registry().get("dinkster.chroma")
    assert descriptor is not None
    sampler = torch_sampler_registry().get("dinkster.euler")
    scheduler = torch_scheduler_registry().get("dinkster.normal")
    assert sampler is not None
    assert scheduler is not None
    sampler = replace(sampler, id="review.custom", aliases=())
    scheduler = replace(scheduler, id="review.custom", aliases=())
    samplers = Registry()
    schedulers = Registry()
    samplers.register(sampler)
    schedulers.register(scheduler)
    guidance_executor = cast("Any", object())
    runtime = assemble_component_checkpoint(
        ComponentCheckpointPlan(descriptor, (("diffusion", part),)),
        diffusion_dtype=torch.float32,
        text_dtype=torch.float32,
        vae_dtype=torch.float32,
        sampler_registry=samplers,
        scheduler_registry=schedulers,
        guidance_executor=guidance_executor,
        attention_policy=attention_policy,
        attention_route_token=(
            None if attention_policy == "auto" else discover_attention_route_token(attention_policy)
        ),
    )
    baseline = _load_component(part, builder, compute_dtype=torch.float32, fp8_matmul=False)
    model = runtime.assembled.diffusion
    assert model is runtime.component_sampling_runtime.assembled.diffusion
    assert runtime.sample_custom.__self__ is runtime.component_sampling_runtime
    assert runtime.family == runtime.diffusion.family
    assert runtime.assembled.compute_dtype("diffusion") == torch.float32
    assert runtime.diffusion.attention_status["flux"].requested_policy == attention_policy
    recipe = SimpleNamespace(
        family_id=runtime.family.id,
        runtime_identity=runtime.runtime_identity,
        sources=(SimpleNamespace(role="checkpoint"),),
    )
    handle = SimpleNamespace(
        runtime=runtime,
        recipe=recipe,
        load_device=torch.device(device),
        require_active=lambda: None,
    )
    option_windows = () if architecture == "chroma" else ("radiance-options",)
    resolved = native_arm.resolve_component_execution(
        cast("Any", handle),
        object(),
        object(),
        inference,
        sampling_shift=1.73,
        option_windows=option_windows,
    )
    assert resolved is not None
    sampling_runtime = cast("ComponentCheckpointRuntime", resolved[0])
    assert sampling_runtime is not runtime
    assert sampling_runtime.assembled is runtime.assembled
    configured = sampling_runtime.component_sampling_runtime
    assert configured._sampling_shift == 1.73
    assert configured._option_windows == option_windows
    assert configured._samplers.get("review.custom") is sampler
    assert configured._schedulers.get("review.custom") is scheduler
    assert configured._guidance is guidance_executor
    configured.check_custom_sampling(
        CustomSamplingRequest(sampler, (), (1.0, 0.0)),
        has_denoise_mask=False,
        has_inpaint=False,
        has_context_windows=False,
    )
    assert sampling_runtime.sample_custom.__self__ is sampling_runtime.component_sampling_runtime
    overlay = native_arm._NativeModelOverlay(  # pyright: ignore[reportPrivateUsage]
        cast("Any", handle),
        (),
        {},
        sampling_shift=2.5,
        chroma_radiance_options=option_windows,
    )
    scheduler_runtime, scheduler_shift, scheduler_device = (
        native_arm._require_custom_sampling_runtime(  # pyright: ignore[reportPrivateUsage]
            overlay, "BasicScheduler"
        )
    )
    assert isinstance(scheduler_runtime, ComponentCheckpointRuntime)
    assert scheduler_runtime.assembled is runtime.assembled
    configured = scheduler_runtime.component_sampling_runtime
    assert configured._sampling_shift == 2.5
    assert configured._option_windows == option_windows
    assert configured._samplers.get("review.custom") is sampler
    assert configured._schedulers.get("review.custom") is scheduler
    assert configured._guidance is guidance_executor
    assert scheduler_shift is None
    assert scheduler_device == torch.device(device)
    for key, actual in model.state_dict().items():
        expected = baseline.state_dict()[key]
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    radiance = isinstance(config, ChromaRadianceConfig)
    model.to(device)
    baseline.to(device)
    generator = torch.Generator().manual_seed(8)
    latent = torch.randn((1, 3 if radiance else 2, 4, 4), generator=generator).to(device)
    context = torch.randn((1, 3, 8), generator=generator).to(device)
    timestep = torch.tensor([0.5], device=device)
    guidance = torch.tensor([3.5], device=device)
    with torch.inference_mode():
        torch.testing.assert_close(
            model(latent, timestep, context, guidance),
            baseline(latent, timestep, context, guidance),
            rtol=0,
            atol=0,
        )
    if radiance:
        codec = PixelSpaceCodec(compute_dtype=torch.float32)
        assert torch.equal(runtime.decode_latent(latent), codec.decode(latent))
        assert torch.equal(runtime.encode_content(latent), codec.encode(latent))
    else:
        with pytest.raises(ValueError, match="no codec binding"):
            runtime.decode_latent(latent)


@pytest.mark.parametrize("radiance", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chroma_checkpoint_real_text_and_codec_companions(
    tmp_path: Path, radiance: bool, dtype: torch.dtype
) -> None:
    from functools import partial

    from dinkster_inference import KLConfig, T5Config
    from dinkster_inference_torch import AutoencoderKL, Chroma, ChromaRadiance, T5TextModel
    from dinkster_inference_torch.attention import AttentionRole, select_attention
    from dinkster_inference_torch.autoencoder_kl import kl_codec_plugin
    from dinkster_inference_torch.chroma_runtime import ChromaTextRuntime
    from dinkster_inference_torch.z_image import PixelSpaceCodec
    from test_chroma import initialize, small_chroma_config, small_radiance_config

    diffusion = (
        ChromaRadiance(small_radiance_config()) if radiance else Chroma(small_chroma_config())
    )
    text = T5TextModel(T5Config(8, 16, 4, 2, 1, 32128, "gelu_pytorch_tanh", True))
    originals: dict[str, Any] = {"diffusion": diffusion, "t5xxl": text}
    if not radiance:
        originals["vae"] = AutoencoderKL(KLConfig(3, 3, 32, 32, (1, 2), 1, 2, 2, quant_convs=False))
    prefixes = {
        "diffusion": "model.diffusion_model.",
        "t5xxl": "text_encoders.t5xxl.transformer.",
        "vae": "first_stage_model.",
    }
    path = tmp_path / "combined.safetensors"
    weights: dict[str, torch.Tensor] = {}
    parts: dict[str, ComponentPlan[object]] = {}
    for role, original in originals.items():
        initialize(original)
        state = original.state_dict()
        keys = {key: prefixes[role] + key for key in state}
        weights.update({keys[key]: value.contiguous() for key, value in state.items()})
        parts[role] = ComponentPlan(
            role, path, original.config, keys, dict.fromkeys(state, FLOAT32), {}
        )
    save_file(weights, path)
    descriptor = default_component_registry().get("dinkster.chroma")
    assert descriptor is not None
    runtime = assemble_component_checkpoint(
        ComponentCheckpointPlan(descriptor, tuple(parts.items())),
        diffusion_dtype=torch.float32,
        text_dtype=dtype,
        vae_dtype=dtype,
    )
    baselines: dict[str, Any] = {}
    attention_roles: dict[str, AttentionRole] = {"diffusion": "flux", "t5xxl": "t5", "vae": "vae"}
    for role, original in originals.items():
        baseline = _load_component(
            parts[role],
            partial(
                type(original), attention_kernel=select_attention(attention_roles[role]).kernel
            ),
            compute_dtype=torch.float32 if role == "diffusion" else dtype,
            fp8_matmul=False,
        )
        baselines[role] = baseline
        for key, actual in runtime.assembled.components[role].state_dict().items():
            expected = baseline.state_dict()[key]
            assert actual.dtype == expected.dtype
            assert actual.device == expected.device
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not any(
        parameter.requires_grad for parameter in runtime.assembled.components["t5xxl"].parameters()
    )
    with torch.inference_mode():
        expected_text = ChromaTextRuntime(baselines["t5xxl"]).encode_text("a red cat")
        actual_text = runtime.encode_text("a red cat")
        assert actual_text.pooled is None
        assert actual_text.embeddings.dtype == expected_text.embeddings.dtype
        torch.testing.assert_close(actual_text.embeddings, expected_text.embeddings, rtol=0, atol=0)
        prepared = runtime.prepare_single_stream_conditioning(
            runtime.text_conditioning_carrier(actual_text)
        )
        torch.testing.assert_close(prepared.embeddings, actual_text.embeddings, rtol=0, atol=0)
        codec = (
            PixelSpaceCodec(compute_dtype=torch.float32)
            if radiance
            else replace(kl_codec_plugin(baselines["vae"]), compute_dtype=dtype)
        )
        content = torch.linspace(0, 1, 3 * 8 * 8).reshape(1, 3, 8, 8)
        expected_latent = codec.encode(content)
        actual_latent = runtime.encode_content(content)
        assert actual_latent.dtype == expected_latent.dtype
        torch.testing.assert_close(actual_latent, expected_latent, rtol=0, atol=0)
        torch.testing.assert_close(
            runtime.decode_latent(actual_latent), codec.decode(expected_latent), rtol=0, atol=0
        )
