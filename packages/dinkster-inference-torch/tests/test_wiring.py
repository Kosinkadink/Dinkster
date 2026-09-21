"""Native runtime assembly and sampling composition.

FluxRuntime and load_runtime compose only already-pinned pieces
(tokenizers, text encoders, FluxDenoiser, run_denoise, the KL codec),
so these tests pin the assembly registry's lazy loader declarations,
the geometry-planning gate on load_runtime, the
runtime_identity stability contract, the FluxClipModel dual-encode
shape, the KSampler schedule/noise/drive recipe (including the SDE
pre-offset brownian-tree ordering), and the codec delegation. The
tiny text models carry REAL vocabulary sizes so the real BPE/spm
tokenizers drive them; full-size load_runtime against installed
checkpoints lives with the capability-gated GPU suite.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, TypeVar, cast

import dinkster_inference
import dinkster_inference_torch.sampling_execution as sampling_engine
import dinkster_inference_torch.wiring as wiring
import pytest
import torch
from clip_fill import fill_value
from dinkster_inference import (
    BFLOAT16,
    CHROMA,
    FLOAT16,
    FLOAT32,
    FLUX_DEV,
    FLUX_SCHNELL,
    KREA2,
    OVIS_QWEN3_2B_CONFIG,
    UMT5_XXL_CONFIG,
    WAN21,
    WAN21_ANIMATE2_14B,
    WAN21_CLIP_VISION,
    WAN21_VAE_CONFIG,
    CanonicalManifest,
    ClipTextConfig,
    ComponentPlan,
    CompositeWindowPlan,
    Conditioning,
    CustomSamplingRequest,
    CustomSamplingResult,
    CustomSamplingRuntime,
    FamilyRuntime,
    FluxAssemblyPlan,
    FluxConfig,
    FluxFlowSigmas,
    GuidanceContribution,
    GuidanceEvaluateNext,
    GuidanceEvaluationRequest,
    GuidanceEvaluationWrapperDescriptor,
    GuidancePostCFGContext,
    GuidancePostCFGDescriptor,
    GuidancePredictions,
    IntegerAffineIndexMap,
    KindAxisMap,
    KLConfig,
    LayerQuant,
    LayerWindow,
    ManifestConsensusToken,
    MediaAxis,
    MergeDeclaration,
    ModelFamily,
    ModelTokenLayout,
    ModelTokenSegment,
    NoiseKind,
    Registry,
    SamplerDescriptor,
    SamplingGuidance,
    SamplingStateEvent,
    StepEvent,
    T5Config,
    TensorGeometry,
    TokenGridTransform,
    Wan21AssemblyPlan,
    Wan21PoseBlockCacheSettings,
    Wan21PoseBlockCacheStorage,
    WeightEntry,
    WindowIndexList,
    WindowKind,
    WindowPlanBinding,
    WindowPlanLayer,
    WindowWeightKind,
    WindowWeightProfile,
    build_runtime_identity,
    build_windowed_evaluation_slot,
    builtin_assembly_registry,
    builtin_family_registry,
    builtin_sampler_registry,
    compile_window_plan,
    load_safetensors_header,
    offset_first_sigma_for_snr,
    sampling_execution_context,
    sampling_sigmas,
)
from dinkster_inference.runtime import (
    AssemblyRegistration,
    NativeAssemblyPlan,
)
from dinkster_inference_torch import (
    AssembledFlux,
    AutoencoderKL,
    BrownianTreeNoise,
    ClipTextModel,
    DenoiseError,
    Flux,
    FluxDenoiser,
    FluxRuntime,
    QwenTextModel,
    T5TextModel,
    WiringError,
    latent_process_out,
    load_runtime,
    prepare_noise,
    run_denoise,
    torch_sampler_registry,
    torch_scheduler_registry,
)
from dinkster_inference_torch._conditioning_layout import (
    declare_text_conditioning,
    declared_token_count,
)
from dinkster_inference_torch.distributed import DistributedSamplingConfig
from dinkster_inference_torch.flux_window import (
    derive_flux_window_layout,
    prepare_flux_window_plan,
)
from dinkster_inference_torch.guidance import (
    ConditioningEvaluation,
    ConditioningValidationPath,
    GuidanceExecutor,
    GuidanceRegistry,
    GuidedDenoiser,
)
from dinkster_inference_torch.sampling_execution import (
    build_custom_sampling_schedule,
    compile_guidance_plan,
    guided_denoiser,
)
from dinkster_protocol import ATTENTION_ROLES, AttentionRoute, AttentionRouteToken
from golden_files import (
    assert_reference_schedule,
    assert_reference_tensor,
    assert_reference_values,
    load_platform_golden,
)

CUSTOM_SIGMA_GOLDENS = load_platform_golden(
    Path(__file__).parents[3] / "tests" / "goldens" / "sampling_goldens.json",
    allow_portable_fallback=True,
)["custom_sigma_queries"]
FLUX_PATCH_GOLDENS = load_platform_golden(
    Path(__file__).parent / "goldens" / "model_sampling_flux.json",
    allow_portable_fallback=True,
)["cases"]

# Tiny everywhere EXCEPT the text vocabularies: encode_text drives the
# real CLIP BPE and T5 spm tokenizers, whose ids index real-sized
# embedding tables. Cross-component dims cohere like a real Flux:
# context_in_dim = T5 d_model, vec_in_dim = CLIP hidden, in_channels =
# VAE embed_dim.
TINY_CLIP = ClipTextConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    hidden_act="quick_gelu",
    vocab_size=49408,
    eos_token_id=49407,
)
TINY_T5 = T5Config(
    d_model=48,
    d_ff=96,
    d_kv=12,
    num_heads=4,
    num_layers=2,
    vocab_size=32128,
    dense_act_fn="gelu_pytorch_tanh",
    is_gated_act=True,
)
TINY_FLUX = FluxConfig(
    in_channels=16,
    out_channels=16,
    vec_in_dim=TINY_CLIP.hidden_size,
    context_in_dim=TINY_T5.d_model,
    hidden_size=32,
    depth=1,
    depth_single_blocks=1,
    num_heads=2,
    axes_dim=(4, 6, 6),
    guidance_embed=True,
)
TINY_KL = KLConfig(
    in_channels=3,
    out_channels=3,
    # ch rides the reference's fixed 32-group GroupNorm: every block
    # channel count (ch * mult) must divide by 32.
    ch=32,
    decoder_ch=32,
    ch_mult=(1, 2),
    num_res_blocks=1,
    z_channels=16,
    embed_dim=16,
)


ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


def filled(module: ModuleT) -> ModuleT:
    state = {
        key: fill_value(key, tuple(tensor.shape)) for key, tensor in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True)
    return module


@pytest.fixture(scope="module")
def assembled() -> AssembledFlux:
    torch.manual_seed(0)
    return AssembledFlux(
        family=FLUX_DEV,
        diffusion=filled(Flux(TINY_FLUX)),
        clip_l=filled(ClipTextModel(TINY_CLIP)),
        t5xxl=filled(T5TextModel(TINY_T5)),
        vae=filled(AutoencoderKL(TINY_KL)),
    )


@pytest.fixture(scope="module")
def runtime(assembled: AssembledFlux) -> FluxRuntime:
    return FluxRuntime(assembled, runtime_identity="native:test:feed")


def test_runtime_satisfies_the_protocol(runtime: FluxRuntime) -> None:
    seam: FamilyRuntime[torch.Tensor] = runtime
    assert seam.family.id == "dinkster.flux_dev"
    assert seam.runtime_identity == "native:test:feed"
    assert runtime.assembled.compute_dtype("vae") is None
    assert isinstance(runtime, CustomSamplingRuntime)


def test_runtime_retains_classic_text_offload_storage(runtime: FluxRuntime) -> None:
    assert runtime.retained_offload_storage_components == frozenset({"clip_l", "t5xxl"})


# --- load_runtime's assembly registry -------------------------------------


@dataclass
class FakeSource:
    """In-memory WeightSource (headers only, like the probe)."""

    path: Path
    geometries: dict[str, TensorGeometry]

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key=key, geometry=geometry, offset=0, nbytes=geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


@dataclass(frozen=True)
class SyntheticAssemblyPlan:
    family: ModelFamily
    identity_components: tuple[ComponentPlan[Any] | None, ...]


def synthetic_loader(plan: NativeAssemblyPlan, **options: object) -> FluxRuntime:
    raise AssertionError("tests must install the selected assembly's loader")


def synthetic_registry(
    planner: Callable[..., NativeAssemblyPlan],
    *,
    loader: str = f"{__name__}:synthetic_loader",
) -> Registry[AssemblyRegistration]:
    registry: Registry[AssemblyRegistration] = Registry()
    registry.register(AssemblyRegistration("test.checkpoint", planner, loader))
    return registry


@pytest.mark.parametrize("assembly", tuple(builtin_assembly_registry()), ids=lambda item: item.id)
def test_builtin_assembly_loader_declaration_resolves_callable(
    assembly: AssemblyRegistration,
) -> None:
    loader = wiring._resolve_assembly_loader(assembly)  # pyright: ignore[reportPrivateUsage]
    assert callable(loader)


def test_builtin_assembly_registries_are_independent() -> None:
    first = builtin_assembly_registry()
    second = builtin_assembly_registry()

    def accept(**_sources: object) -> FluxAssemblyPlan:
        return tiny_plan()

    registration = AssemblyRegistration("test.checkpoint", accept, f"{__name__}:synthetic_loader")
    first.register(registration)
    assert first.get(registration.id) is registration
    assert second.get(registration.id) is None
    assert builtin_assembly_registry().get(registration.id) is None


@pytest.mark.parametrize(
    "loader, reason",
    (
        ("dinkster_test_missing_assembly_module:load", "unavailable"),
        (f"{__name__}:missing_assembly_loader", "unavailable"),
        (f"{__name__}:TINY_FLUX", "not callable"),
    ),
)
def test_selected_assembly_loader_errors_name_registration_and_path(
    loader: str, reason: str
) -> None:
    def accept(**_sources: object) -> FluxAssemblyPlan:
        return tiny_plan()

    registry = synthetic_registry(accept, loader=loader)
    with pytest.raises(WiringError, match=reason) as caught:
        load_runtime(
            FakeSource(Path("/fake/loader.safetensors"), {}),
            assembly_registry=registry,
            registry_token="test-assembly",
        )
    assert "test.checkpoint" in str(caught.value)
    assert loader in str(caught.value)


def test_load_runtime_refuses_with_the_probe_reasons() -> None:
    source = FakeSource(
        Path("/fake/unknown.safetensors"),
        {"not_a_model.weight": TensorGeometry((1,), FLOAT16)},
    )
    with pytest.raises(WiringError, match="components do not match an executable architecture"):
        load_runtime(source)


def test_load_runtime_storage_dtype_policy_public_transport_defaults_false(
    runtime: FluxRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[bool] = []
    plan = tiny_plan()

    def accept(*args: object, **kwargs: object) -> FluxAssemblyPlan:
        return plan

    def load(planned: NativeAssemblyPlan, **kwargs: object) -> FluxRuntime:
        assert planned is plan
        captured.append(cast(bool, kwargs["storage_dtype_follows_compute"]))
        return runtime

    monkeypatch.setattr(sys.modules[__name__], "synthetic_loader", load)
    registry = synthetic_registry(accept)
    source = FakeSource(Path("/fake/transport.safetensors"), {})

    assert (
        load_runtime(source, assembly_registry=registry, registry_token="test-assembly") is runtime
    )
    assert (
        load_runtime(
            source,
            storage_dtype_follows_compute=True,
            assembly_registry=registry,
            registry_token="test-assembly",
        )
        is runtime
    )
    assert captured == [False, True]


def test_load_runtime_transports_wan_animate2_pose_cache_policy(
    runtime: FluxRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Wan21PoseBlockCacheSettings(storage=Wan21PoseBlockCacheStorage.INT8)
    captured: list[object] = []

    def accept(*args: object, **kwargs: object) -> Wan21AssemblyPlan:
        return Wan21AssemblyPlan(
            family=WAN21,
            diffusion=component_plan("diffusion", WAN21_ANIMATE2_14B),
            umt5xxl=component_plan("umt5xxl", UMT5_XXL_CONFIG),
            vae=component_plan("vae", WAN21_VAE_CONFIG),
            clip_vision=component_plan("clip_vision", WAN21_CLIP_VISION),
            tokenizer_source_key="spiece_model",
        )

    def load(planned: NativeAssemblyPlan, **kwargs: object) -> FluxRuntime:
        assert isinstance(planned, Wan21AssemblyPlan)
        captured.append(kwargs["pose_cache_settings"])
        return runtime

    monkeypatch.setattr(sys.modules[__name__], "synthetic_loader", load)
    registry = synthetic_registry(accept)

    assert (
        load_runtime(
            FakeSource(Path("/fake/animate2.safetensors"), {}),
            pose_cache_settings=settings,
            assembly_registry=registry,
            registry_token="test-assembly",
        )
        is runtime
    )
    assert captured == [settings]


def test_load_runtime_refuses_pose_cache_without_pose_blocks() -> None:
    def accept(*args: object, **kwargs: object) -> FluxAssemblyPlan:
        return tiny_plan()

    registry = synthetic_registry(accept, loader="dinkster_inference_torch.wiring:_load_flux")

    with pytest.raises(WiringError, match="pose cache settings require a Wan 2.1 runtime"):
        load_runtime(
            FakeSource(Path("/fake/flux.safetensors"), {}),
            pose_cache_settings=Wan21PoseBlockCacheSettings(),
            assembly_registry=registry,
            registry_token="test-assembly",
        )


def test_load_runtime_resolves_family_dtype_defaults_after_probe(
    runtime: FluxRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_probe: dict[str, object] = {}
    captured_load: dict[str, object] = {}
    plan = tiny_plan()

    def accept(*args: object, **kwargs: object) -> FluxAssemblyPlan:
        captured_probe.update(kwargs)
        return plan

    def load(planned: NativeAssemblyPlan, **kwargs: object) -> FluxRuntime:
        assert planned is plan
        captured_load.update(kwargs)
        return runtime

    monkeypatch.setattr(sys.modules[__name__], "synthetic_loader", load)
    registry = synthetic_registry(accept)
    source = FakeSource(Path("/fake/defaults.safetensors"), {})

    assert (
        load_runtime(source, assembly_registry=registry, registry_token="test-assembly") is runtime
    )

    assert "planning_context" not in captured_probe
    assert captured_probe["checkpoint"] is source
    assert set(captured_probe).isdisjoint(captured_load)
    assert captured_load["diffusion_dtype"] is torch.bfloat16
    assert captured_load["text_dtype"] is torch.bfloat16
    assert captured_load["vae_dtype"] is torch.bfloat16


def test_load_runtime_uses_family_from_active_registry(
    runtime: FluxRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tiny_plan()
    registered_family = replace(
        plan.family,
        engine=replace(plan.family.engine, regional_memory_factor=7.0),
    )
    family_registry: Registry[ModelFamily] = Registry()
    family_registry.register(registered_family)

    def accept(*_args: object, **_kwargs: object) -> FluxAssemblyPlan:
        return plan

    def load(planned: NativeAssemblyPlan, **_kwargs: object) -> FluxRuntime:
        assert planned.family is registered_family
        return runtime

    monkeypatch.setattr(sys.modules[__name__], "synthetic_loader", load)
    assembly_registry = synthetic_registry(accept)

    assert (
        load_runtime(
            FakeSource(Path("/fake/active-family.safetensors"), {}),
            assembly_registry=assembly_registry,
            family_registry=family_registry,
            registry_token="active-family",
        )
        is runtime
    )


@pytest.mark.parametrize("explicit_dtypes", (False, True))
def test_registered_noncatalogued_plan_and_options_reach_lazy_loader_once(
    explicit_dtypes: bool, runtime: FluxRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = replace(FLUX_DEV, id="test.noncatalogued")
    assert builtin_family_registry().get(family.id) is None
    plan = SyntheticAssemblyPlan(family, tiny_plan().identity_components)
    monkeypatch.setattr(runtime, "assembled", replace(runtime.assembled, family=family))
    sources = {
        role: FakeSource(Path(f"/fake/{role}.safetensors"), {})
        for role in ("checkpoint", "diffusion", "t5xxl", "vae")
    }
    calls: list[str] = []
    captured_sources: dict[str, object] = {}
    captured_options: dict[str, object] = {}

    def accept(**inputs: object) -> SyntheticAssemblyPlan:
        calls.append("plan")
        captured_sources.update(inputs)
        return plan

    def load(admitted: NativeAssemblyPlan, **options: object) -> FluxRuntime:
        calls.append("load")
        assert admitted is plan
        assert admitted.identity_components is plan.identity_components
        captured_options.update(options)
        return runtime

    # Install after registration to prove the loader is resolved at execution time.
    registry = synthetic_registry(accept)
    monkeypatch.setattr(sys.modules[__name__], "synthetic_loader", load)
    token = AttentionRouteToken(
        version=1,
        routes=tuple(AttentionRoute(role, "sdpa") for role in ATTENTION_ROLES),
        provider_versions=(("torch", "2.13.0"),),
        adapter_contract_revision="dinkster.attention-kernel.v1",
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        requested_policy="sdpa",
    )

    def lookup(_name: str) -> None:
        return None

    options: dict[str, Any] = dict(
        storage_dtype_follows_compute=True,
        fp8_matmul=True,
        sampler_registry=torch_sampler_registry(),
        scheduler_registry=torch_scheduler_registry(),
        registry_token="test-registries",
        extension_behavior_hash="a" * 64,
        patch_overlay_digests=("b" * 64,),
        guidance_executor=GuidanceExecutor(GuidanceRegistry()),
        embedding_lookups={"t5xxl": lookup},
        embedding_binding_digest="c" * 64,
        attention_policy="sdpa",
        attention_route_token=token,
        pose_cache_settings=Wan21PoseBlockCacheSettings(),
    )
    dtypes = dict(
        diffusion_dtype=torch.float16,
        text_dtype=torch.float32,
        vae_dtype=torch.bfloat16,
    )
    if explicit_dtypes:
        options.update(dtypes)
    loaded = load_runtime(
        **sources,
        **options,
        assembly_registry=registry,
        expected_identity=runtime.runtime_identity,
    )
    assert loaded is runtime
    assert loaded.family.id == family.id
    assert calls == ["plan", "load"]
    assert all(captured_sources[role] is source for role, source in sources.items())
    assert set(captured_sources).isdisjoint(captured_options)
    defaults = dict.fromkeys(dtypes, torch.bfloat16)
    assert captured_options == {**options, **(dtypes if explicit_dtypes else defaults)}


@pytest.mark.parametrize(
    "family_id",
    (
        "dinkster.wan21",
        "dinkster.wan22",
        "dinkster.z_image",
        "dinkster.z_image_pixel_space",
    ),
)
def test_load_runtime_resolves_reference_text_families_to_float32(
    family_id: str,
    runtime: FluxRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def accept(*args: object, **kwargs: object) -> FluxAssemblyPlan:
        family = builtin_family_registry().get(family_id)
        assert family is not None
        return replace(tiny_plan(), family=family)

    def load(planned: NativeAssemblyPlan, **kwargs: object) -> FluxRuntime:
        assert planned.family.id == family_id
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(sys.modules[__name__], "synthetic_loader", load)
    registry = synthetic_registry(accept)

    assert (
        load_runtime(
            FakeSource(Path("/fake/defaults.safetensors"), {}),
            assembly_registry=registry,
            registry_token="test-assembly",
        )
        is runtime
    )
    assert captured["text_dtype"] is torch.float32


OVIS_ARTIFACT_ROOT = Path("/home/kosin/model-artifacts/dinkster-w0-flux-vector-free")
REAL_OVIS_DIFFUSION = OVIS_ARTIFACT_ROOT / "diffusion_models/ovis_image_bf16.safetensors"
REAL_OVIS_TEXT = OVIS_ARTIFACT_ROOT / "text_encoders/ovis_2.5.safetensors"
REAL_OVIS_AE = OVIS_ARTIFACT_ROOT / "vae/ae.safetensors"


@pytest.mark.skipif(
    not all(path.exists() for path in (REAL_OVIS_DIFFUSION, REAL_OVIS_TEXT, REAL_OVIS_AE)),
    reason="digest-verified Ovis artifact set absent",
)
def test_load_runtime_dispatches_real_vector_free_ovis_headers_after_probe(
    runtime: FluxRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[FluxAssemblyPlan] = []

    def load(planned: NativeAssemblyPlan, **kwargs: object) -> FluxRuntime:
        assert isinstance(planned, FluxAssemblyPlan)
        captured.append(planned)
        assert not {"diffusion", "qwen3_2b", "vae"}.intersection(kwargs)
        return runtime

    monkeypatch.setattr(sys.modules[__name__], "synthetic_loader", load)
    registry: Registry[AssemblyRegistration] = Registry()
    for assembly in builtin_assembly_registry():
        registry.register(
            replace(assembly, load=f"{__name__}:synthetic_loader")
            if assembly.id == "dinkster.flux"
            else assembly
        )
    diffusion = load_safetensors_header(REAL_OVIS_DIFFUSION)
    qwen = load_safetensors_header(REAL_OVIS_TEXT)
    vae = load_safetensors_header(REAL_OVIS_AE)
    assert (
        load_runtime(
            diffusion=diffusion,
            qwen3_2b=qwen,
            vae=vae,
            assembly_registry=registry,
            registry_token="test-assembly",
        )
        is runtime
    )
    assert len(captured) == 1
    planned = captured[0]
    assert planned.family.id == FLUX_SCHNELL.id
    assert planned.diffusion.path == diffusion.path
    assert planned.qwen3_2b is not None
    assert planned.qwen3_2b.path == qwen.path
    assert planned.vae.path == vae.path


def test_unknown_identity_knob_dtype_refuses_loudly() -> None:
    with pytest.raises(WiringError, match="torch.float64"):
        wiring._identity_dtype(  # pyright: ignore[reportPrivateUsage]
            torch.float64
        )


def test_host_identity_matches_loader_and_expected_assertion(
    assembled: AssembledFlux, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tiny_plan()

    def return_assembled(planned: FluxAssemblyPlan, **kwargs: object) -> AssembledFlux:
        assert planned is plan
        return assembled

    monkeypatch.setattr(wiring, "assemble_flux", return_assembled)
    runtime = wiring._load_flux(  # pyright: ignore[reportPrivateUsage]
        plan,
        diffusion_dtype=None,
        text_dtype=torch.float32,
        vae_dtype=torch.float32,
        storage_dtype_follows_compute=False,
        fp8_matmul=False,
        sampler_registry=None,
        scheduler_registry=None,
        registry_token=None,
        embedding_lookups={
            "clip_l": lambda _name: None,
            "clip_g": lambda _name: None,
            "t5xxl": lambda _name: None,
        },
        embedding_binding_digest="1" * 64,
    )
    expected = build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
        embedding_binding_digest="1" * 64,
    )
    assert runtime.runtime_identity == expected

    def accept(*args: object, **kwargs: object) -> FluxAssemblyPlan:
        return plan

    def load(planned: NativeAssemblyPlan, **kwargs: object) -> FluxRuntime:
        assert planned is plan
        return runtime

    monkeypatch.setattr(sys.modules[__name__], "synthetic_loader", load)
    registry = synthetic_registry(accept)
    checkpoint = FakeSource(Path("/fake/host.safetensors"), {})
    assert (
        load_runtime(
            checkpoint,
            expected_identity=expected,
            assembly_registry=registry,
            registry_token="test-assembly",
        )
        is runtime
    )
    with pytest.raises(WiringError, match="wrong.*constructed"):
        load_runtime(
            checkpoint,
            expected_identity="wrong",
            assembly_registry=registry,
            registry_token="test-assembly",
        )


def test_storage_dtype_policy_intent_does_not_rotate_runtime_identity(
    assembled: AssembledFlux, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tiny_plan()

    def return_assembled(planned: FluxAssemblyPlan, **kwargs: object) -> AssembledFlux:
        assert planned is plan
        return assembled

    monkeypatch.setattr(wiring, "assemble_flux", return_assembled)

    def load(enabled: bool) -> FluxRuntime:
        return wiring._load_flux(  # pyright: ignore[reportPrivateUsage]
            plan,
            diffusion_dtype=None,
            text_dtype=torch.float32,
            vae_dtype=torch.float32,
            storage_dtype_follows_compute=enabled,
            fp8_matmul=False,
            sampler_registry=None,
            scheduler_registry=None,
            registry_token=None,
        )

    default = load(False)
    policy = load(True)

    assert not default.assembled._storage_dtype_follows_compute  # pyright: ignore[reportPrivateUsage]
    assert policy.assembled._storage_dtype_follows_compute  # pyright: ignore[reportPrivateUsage]
    assert policy.runtime_identity == default.runtime_identity


@pytest.mark.parametrize("family", (FLUX_DEV, FLUX_SCHNELL))
@pytest.mark.parametrize(
    ("diffusion_dtype", "fp8_matmul"),
    ((torch.bfloat16, False), (torch.float16, False), (torch.bfloat16, True)),
)
def test_flux_loader_does_not_claim_an_admission_receipt(
    assembled: AssembledFlux,
    monkeypatch: pytest.MonkeyPatch,
    family: ModelFamily,
    diffusion_dtype: torch.dtype,
    fp8_matmul: bool,
) -> None:
    from dinkster_inference_torch import distributed

    plan = replace(tiny_plan(), family=family)

    def return_assembled(planned: FluxAssemblyPlan, **kwargs: object) -> AssembledFlux:
        assert planned is plan
        return replace(assembled, family=family)

    monkeypatch.setattr(wiring, "assemble_flux", return_assembled)

    def load(
        selected_dtype: torch.dtype | None = diffusion_dtype,
        selected_fp8: bool = fp8_matmul,
    ) -> wiring.FluxRuntime:
        return wiring._load_flux(  # pyright: ignore[reportPrivateUsage]
            plan,
            diffusion_dtype=selected_dtype,
            text_dtype=torch.float32,
            vae_dtype=torch.float32,
            storage_dtype_follows_compute=False,
            fp8_matmul=selected_fp8,
            sampler_registry=None,
            scheduler_registry=None,
            registry_token=None,
        )

    runtime = load()
    assert runtime.family is family
    assert runtime.receipt_identity is None

    config = DistributedSamplingConfig(0, 2, "window", "file:///group", "1" * 32, "a:1")
    monkeypatch.setattr(distributed, "distributed_sampling_config", lambda: config)

    def unexpected_capability(*_args: object) -> None:
        raise AssertionError("assembly queried hardware")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_capability)
    assert load().receipt_identity is None
    assert load(torch.bfloat16).receipt_identity is None
    assert load(torch.float16).receipt_identity is None
    assert load(selected_fp8=True).receipt_identity is None


# --- runtime_identity ------------------------------------------------------


def component_plan(component: str, config: object) -> ComponentPlan[Any]:
    return ComponentPlan(
        component=component,
        path=Path(f"/fake/{component}.safetensors"),
        config=config,
        keys={"w": "w"},
        dtypes={"w": FLOAT32},
        quant={},
    )


def tiny_plan(**overrides: ComponentPlan[Any]) -> FluxAssemblyPlan:
    plans: dict[str, ComponentPlan[Any]] = {
        "diffusion": component_plan("diffusion", TINY_FLUX),
        "clip_l": component_plan("clip_l", TINY_CLIP),
        "t5xxl": component_plan("t5xxl", TINY_T5),
        "vae": component_plan("vae", TINY_KL),
    }
    plans.update(overrides)
    return FluxAssemblyPlan(
        family=FLUX_DEV,
        diffusion=plans["diffusion"],
        clip_l=plans["clip_l"],
        t5xxl=plans["t5xxl"],
        vae=plans["vae"],
    )


def identity(
    plan: FluxAssemblyPlan | None = None,
    *,
    diffusion_dtype: torch.dtype = torch.bfloat16,
    text_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.float32,
    fp8_matmul: bool = False,
    registry_token: str | None = None,
) -> str:
    """build_runtime_identity over tiny_plan() with _load_flux's knob
    defaults; override one knob to test its rotation."""
    if plan is None:
        plan = tiny_plan()
    dtypes = {
        torch.float16: FLOAT16,
        torch.float32: FLOAT32,
        torch.bfloat16: BFLOAT16,
    }
    return build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=dtypes[diffusion_dtype],
        text_dtype=dtypes[text_dtype],
        vae_dtype=dtypes[vae_dtype],
        fp8_matmul=fp8_matmul,
        registry_token=registry_token,
    )


class TestFluxIdentity:
    def test_format_and_cross_process_stability(self) -> None:
        """Same plan + knobs = same string (the contract's stability
        half); the family id rides in the clear so a
        dispatcher can log them without parsing the digest."""
        first = identity()
        second = identity()
        assert first == second
        prefix = "native:dinkster.flux_dev:"
        assert first.startswith(prefix)
        digest = first.removeprefix(prefix)
        # full sha256 - a correctness-critical cache key is never
        # truncated
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")

    def test_knob_changes_rotate(self) -> None:
        base = identity()
        assert identity(text_dtype=torch.bfloat16) != base
        assert identity(fp8_matmul=True) != base

    def test_plan_dtype_changes_rotate(self) -> None:
        fp16 = ComponentPlan(
            component="diffusion",
            path=Path("/fake/diffusion.safetensors"),
            config=TINY_FLUX,
            keys={"w": "w"},
            dtypes={"w": FLOAT16},
            quant={},
        )
        assert identity(tiny_plan(diffusion=fp16)) != identity()

    def test_paths_do_not_rotate(self) -> None:
        """Identity names the execution body, not file locations: the
        same components read from a moved file must not rotate cache
        keys (asset identity is the dispatcher's separate concern)."""
        moved = ComponentPlan(
            component="diffusion",
            path=Path("/elsewhere/renamed.safetensors"),
            config=TINY_FLUX,
            keys={"w": "w"},
            dtypes={"w": FLOAT32},
            quant={},
        )
        assert identity(tiny_plan(diffusion=moved)) == identity()

    @staticmethod
    def quant_plan(
        *, source_prefix: str = "", fmt: str | None = "float8_e4m3fn"
    ) -> FluxAssemblyPlan:
        """A diffusion plan with one quantized layer whose ARTIFACT
        keys carry ``source_prefix`` (combined checkpoints prefix
        them, split files do not - execution is identical)."""
        quantized = ComponentPlan(
            component="diffusion",
            path=Path("/fake/diffusion.safetensors"),
            config=TINY_FLUX,
            keys={"lin.weight": f"{source_prefix}lin.weight"},
            dtypes={"lin.weight": FLOAT16},
            quant={
                "lin": LayerQuant(
                    layer="lin",
                    format=fmt,
                    weight=f"{source_prefix}lin.weight",
                    weight_scale=f"{source_prefix}lin.scale_weight",
                    config=(None if fmt is not None else f"{source_prefix}lin.comfy_quant"),
                )
            },
        )
        return tiny_plan(diffusion=quantized)

    def test_quant_artifact_prefixes_do_not_rotate(self) -> None:
        """The same quantized component read split vs combined (whose
        artifact keys gain the checkpoint prefix) must hash
        identically - quant identity is the model layer's behavior
        facts, never source spellings."""
        split = identity(self.quant_plan())
        combined = identity(self.quant_plan(source_prefix="model.diffusion_model."))
        assert split == combined

    def test_quant_format_rotates(self) -> None:
        e4m3 = identity(self.quant_plan(fmt="float8_e4m3fn"))
        e5m2 = identity(self.quant_plan(fmt="float8_e5m2"))
        payload = identity(self.quant_plan(fmt=None))
        assert len({e4m3, e5m2, payload}) == 3

    def test_registry_token_rotates(self) -> None:
        assert identity(registry_token="mypack-v2") != identity()


@pytest.mark.parametrize("kind", ("assembly_registry", "sampler_registry", "scheduler_registry"))
def test_load_runtime_refuses_custom_registries_without_a_token(kind: str) -> None:
    registries: dict[str, object] = {
        "assembly_registry": builtin_assembly_registry(),
        "sampler_registry": builtin_sampler_registry(),
        "scheduler_registry": torch_scheduler_registry(),
    }
    with pytest.raises(WiringError, match="registry_token"):
        load_runtime(
            FakeSource(
                Path("/fake/unknown.safetensors"),
                {"not_a_model.weight": TensorGeometry((1,), FLOAT16)},
            ),
            **{kind: registries[kind]},
        )


def test_custom_assembly_token_rotates_constructed_runtime_identity(
    assembled: AssembledFlux, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tiny_plan()

    def accept(**_sources: object) -> FluxAssemblyPlan:
        return plan

    def build(planned: FluxAssemblyPlan, **_options: object) -> AssembledFlux:
        assert planned is plan
        return assembled

    monkeypatch.setattr(wiring, "assemble_flux", build)
    registry = synthetic_registry(accept, loader="dinkster_inference_torch.wiring:_load_flux")
    source = FakeSource(Path("/fake/custom-assembly.safetensors"), {})
    first = load_runtime(source, assembly_registry=registry, registry_token="assembly-v1")
    second = load_runtime(source, assembly_registry=registry, registry_token="assembly-v2")
    assert first.runtime_identity != second.runtime_identity
    assert first.runtime_identity == build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=BFLOAT16,
        vae_dtype=BFLOAT16,
        fp8_matmul=False,
        registry_token="assembly-v1",
    )


def test_load_runtime_refuses_a_token_without_custom_registries() -> None:
    with pytest.raises(WiringError, match="without a custom registry"):
        load_runtime(
            FakeSource(
                Path("/fake/unknown.safetensors"),
                {"not_a_model.weight": TensorGeometry((1,), FLOAT16)},
            ),
            registry_token="mypack-v2",
        )


# --- encode_text: the FluxClipModel dual encode ----------------------------


class TestEncodeText:
    def test_t5_sequence_with_clip_pooled(self, runtime: FluxRuntime) -> None:
        """FluxClipModel.encode_token_weights @ 947c2749: the T5
        sequence (Flux profile: final chunk padded to 256) is the
        conditioning, CLIP-L's raw pooled vector rides beside it."""
        cond = runtime.encode_text("a photo of a cat")
        assert cond.embeddings.shape == (1, 256, TINY_T5.d_model)
        assert cond.pooled is not None
        assert cond.pooled.shape == (1, TINY_CLIP.hidden_size)
        assert cond.embeddings.dtype == torch.float32
        assert cond.pooled.dtype == torch.float32

    def test_deterministic(self, runtime: FluxRuntime) -> None:
        first = runtime.encode_text("a photo of a cat")
        second = runtime.encode_text("a photo of a cat")
        assert torch.equal(first.embeddings, second.embeddings)
        assert first.pooled is not None and second.pooled is not None
        assert torch.equal(first.pooled, second.pooled)

    def test_empty_prompt_encodes(self, runtime: FluxRuntime) -> None:
        cond = runtime.encode_text("")
        assert cond.embeddings.shape == (1, 256, TINY_T5.d_model)

    def test_textual_inversion_resolves_each_flux_component_independently(
        self, assembled: AssembledFlux
    ) -> None:
        calls: list[tuple[str, str]] = []

        def clip_lookup(name: str) -> torch.Tensor | None:
            calls.append(("clip_l", name))
            return torch.ones((2, TINY_CLIP.hidden_size)) if name == "pair" else None

        def t5_lookup(name: str) -> torch.Tensor | None:
            calls.append(("t5xxl", name))
            return torch.ones((3, TINY_T5.d_model)) if name == "pair" else None

        runtime = FluxRuntime(
            assembled,
            runtime_identity="native:test:flux-embedding",
            embedding_lookups={"clip_l": clip_lookup, "t5xxl": t5_lookup},
        )
        cond = runtime.encode_text("embedding:pair")
        plain = runtime.encode_text("pair")
        zero_runtime = FluxRuntime(
            assembled,
            runtime_identity="native:test:flux-embedding-zero",
            embedding_lookups={
                "clip_l": lambda _name: torch.zeros((2, TINY_CLIP.hidden_size)),
                "t5xxl": lambda _name: torch.zeros((3, TINY_T5.d_model)),
            },
        )
        zero = zero_runtime.encode_text("embedding:pair")
        assert cond.embeddings.shape == (1, 256, TINY_T5.d_model)
        assert cond.pooled is not None
        assert cond.pooled.shape == (1, TINY_CLIP.hidden_size)
        assert not torch.equal(cond.embeddings, plain.embeddings)
        assert not torch.equal(cond.embeddings, zero.embeddings)
        assert zero.pooled is not None
        assert not torch.equal(cond.pooled, zero.pooled)
        assert calls.count(("clip_l", "pair")) >= 2
        assert calls.count(("t5xxl", "pair")) >= 2

    def test_ovis_slot_routes_raw_prompt_to_qwen_conditioning(
        self, assembled: AssembledFlux
    ) -> None:
        class RecordingQwen(torch.nn.Module):
            config = OVIS_QWEN3_2B_CONFIG

            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(1, 1)

            def forward(
                self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
            ) -> torch.Tensor:
                assert attention_mask is not None
                positions = torch.arange(ids.shape[1], dtype=torch.float32)
                return positions.reshape(1, -1, 1)

        qwen = RecordingQwen()
        ovis = AssembledFlux(
            family=assembled.family,
            diffusion=assembled.diffusion,
            clip_l=None,
            t5xxl=None,
            vae=assembled.vae,
            qwen3_2b=cast("QwenTextModel", qwen),
        )
        runtime = FluxRuntime(ovis, runtime_identity="native:test:ovis")
        condition = runtime.encode_text("cat")
        assert runtime.retained_offload_storage_components == frozenset()
        assert type(condition) is Conditioning
        assert condition.pooled is None
        assert condition.embeddings.shape == (1, 256, 1)
        assert condition.embeddings[0, 0, 0].item() == 28.0
        assert condition.embeddings[0, -1, 0].item() == 0.0


# --- sample: the KSampler recipe -------------------------------------------


def tiny_latent() -> torch.Tensor:
    generator = torch.Generator("cpu")
    generator.manual_seed(99)
    return torch.randn(1, TINY_FLUX.in_channels, 4, 4, generator=generator)


def flux_window_plan(*, split_height: bool) -> CompositeWindowPlan:
    axes = (MediaAxis("height", 2), MediaAxis("width", 2))
    windows = (
        (
            LayerWindow((WindowIndexList((0,)), WindowIndexList((0, 1)))),
            LayerWindow((WindowIndexList((1,)), WindowIndexList((0, 1)))),
        )
        if split_height
        else (LayerWindow((WindowIndexList((0, 1)), WindowIndexList((0, 1)))),)
    )
    return compile_window_plan(
        axes=axes,
        kinds=(
            WindowKind(
                "latent_image",
                tuple(
                    KindAxisMap(axis.name, axis.extent, IntegerAffineIndexMap(1)) for axis in axes
                ),
            ),
            WindowKind("text", invariant_axes=("height", "width")),
        ),
        layers=(
            WindowPlanLayer(
                ("height", "width"),
                windows,
                (
                    WindowWeightProfile(WindowWeightKind.FLAT),
                    WindowWeightProfile(WindowWeightKind.FLAT),
                ),
                MergeDeclaration(),
            ),
        ),
    )


def _local_preflight(failed: bool, _device: torch.device) -> bool:
    """Rank-local stand-in for the group-wide preflight all-reduce."""
    return failed


def _local_route_mismatch(_window: bool, _device: torch.device) -> bool:
    return False


class TestSample:
    @pytest.mark.parametrize("mode", ("window", "auto"))
    @pytest.mark.parametrize(
        ("family", "compute_dtype"),
        (
            (FLUX_DEV, torch.float8_e4m3fn),
            (FLUX_SCHNELL, torch.bfloat16),
            (KREA2, torch.bfloat16),
            (CHROMA, torch.bfloat16),
        ),
    )
    def test_window_scatter_admits_family_dtype_world_size_and_device_environment(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
        family: ModelFamily,
        compute_dtype: torch.dtype,
    ) -> None:
        class WindowAdmitted(Exception):
            pass

        candidate = FluxRuntime(
            replace(runtime.assembled, family=family),
            runtime_identity=f"native:test:{hashlib.sha256(family.id.encode()).hexdigest()}",
        )
        config = DistributedSamplingConfig(0, 3, mode, "file:///group", "1" * 32, "a:1")
        initialized: list[bool] = []
        monkeypatch.setattr(wiring, "distributed_sampling_config", lambda: config)

        def initialize() -> DistributedSamplingConfig:
            initialized.append(True)
            return config

        def preflight(failed: bool, _device: torch.device) -> bool:
            assert initialized
            return failed

        def route_mismatch(_window: bool, _device: torch.device) -> bool:
            assert initialized
            return False

        monkeypatch.setattr(wiring, "ensure_process_group", initialize)
        monkeypatch.setattr(wiring, "window_preflight_failed", preflight)
        monkeypatch.setattr(wiring, "window_route_mismatch", route_mismatch)

        def non_receipted_capability(*_args: object) -> tuple[int, int]:
            return (8, 9)

        def admit_manifest(**_kwargs: object) -> NoReturn:
            raise WindowAdmitted

        monkeypatch.setattr(
            torch.cuda,
            "get_device_capability",
            non_receipted_capability,
        )
        monkeypatch.setattr(wiring, "build_flux_window_manifest", admit_manifest)

        with pytest.raises(WindowAdmitted):
            candidate.sample(
                tiny_latent(),
                cond=candidate.encode_text("window admission"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                window_plan=flux_window_plan(split_height=True),
                compute_dtype=compute_dtype,
            )

    def test_auto_window_plan_failure_reaches_initialized_group_preflight(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = DistributedSamplingConfig(0, 2, "auto", "file:///group", "1" * 32, "a:1")
        events: list[str] = []
        monkeypatch.setattr(wiring, "distributed_sampling_config", lambda: config)

        def initialize() -> DistributedSamplingConfig:
            events.append("initialize")
            return config

        def preflight(failed: bool, _device: torch.device) -> bool:
            events.append(f"preflight:{failed}")
            return failed

        monkeypatch.setattr(wiring, "ensure_process_group", initialize)
        monkeypatch.setattr(wiring, "window_preflight_failed", preflight)

        with pytest.raises(WiringError, match="invalid-window-plan"):
            runtime.sample(
                tiny_latent(),
                cond=runtime.encode_text("invalid auto window"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                window_plan=cast(Any, object()),
                compute_dtype=torch.float32,
            )

        assert events == ["initialize", "preflight:True"]

    def test_auto_window_route_disagreement_refuses_before_manifest(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = DistributedSamplingConfig(0, 2, "auto", "file:///group", "1" * 32, "a:1")
        monkeypatch.setattr(wiring, "distributed_sampling_config", lambda: config)
        monkeypatch.setattr(wiring, "ensure_process_group", lambda: config)
        monkeypatch.setattr(wiring, "window_preflight_failed", _local_preflight)

        def route_mismatch(*_args: object) -> bool:
            return True

        monkeypatch.setattr(wiring, "window_route_mismatch", route_mismatch)

        with pytest.raises(WiringError, match="ranks disagree on Flux window route"):
            runtime.sample(
                tiny_latent(),
                cond=runtime.encode_text("divergent auto route"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                window_plan=flux_window_plan(split_height=True),
                compute_dtype=torch.float32,
            )

    @pytest.mark.parametrize("mode", ("guidance", "sequence"))
    def test_flux_non_window_distributed_modes_use_generic_admission(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
    ) -> None:
        class GenericAdmissionReached(Exception):
            pass

        config = DistributedSamplingConfig(0, 2, mode, "file:///group", "1" * 32, "a:1")
        monkeypatch.setattr(wiring, "distributed_sampling_config", lambda: config)

        def admit(*_args: object, **_kwargs: object) -> None:
            raise GenericAdmissionReached

        monkeypatch.setattr(runtime, "admit_distributed_guidance", admit)

        with pytest.raises(GenericAdmissionReached):
            runtime.sample(
                tiny_latent(),
                cond=runtime.encode_text("generic distributed admission"),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                compute_dtype=torch.float32,
            )

    @pytest.mark.parametrize("shift", [float("nan"), float("inf"), 100.0, -1000.0, 1000.0])
    def test_flux_sampling_patch_refuses_unrepresentable_sigmas(
        self, runtime: FluxRuntime, shift: float
    ) -> None:
        before = runtime.custom_sampling_sigmas("dinkster.simple", 4, 1.0)
        with pytest.raises(WiringError, match="Flux sampling"):
            runtime.with_sampling_space(FluxFlowSigmas(shift=shift))
        assert runtime.custom_sampling_sigmas("dinkster.simple", 4, 1.0) == before

    @pytest.mark.parametrize("family", [FLUX_DEV, FLUX_SCHNELL])
    @pytest.mark.parametrize("case", FLUX_PATCH_GOLDENS, ids=lambda case: case["name"])
    def test_flux_sampling_patch_matches_executed_reference(
        self, assembled: AssembledFlux, family: ModelFamily, case: dict[str, Any]
    ) -> None:
        base = FluxRuntime(replace(assembled, family=family), runtime_identity="test:flux-patch")
        original = base.custom_sampling_sigmas("dinkster.simple", 4, 1.0)
        space = FluxFlowSigmas(shift=case["shift"])
        patched = base.with_sampling_space(space)
        assert patched is not base
        assert patched.assembled is base.assembled
        assert base.custom_sampling_sigmas("dinkster.simple", 4, 1.0) == original
        sampler = builtin_sampler_registry().get("dinkster.dpmpp_2m_sde")
        assert sampler is not None and sampler.requires_snr_offset
        for name, expected in case["schedules"].items():
            sigmas = patched.custom_sampling_sigmas(f"dinkster.{name}", 4, 1.0)
            assert_reference_schedule(sigmas, expected["sigmas"])
            schedule = build_custom_sampling_schedule(sigmas, space, sampler, flow=True)
            # Torch 2.13 recomputation differs from the recorded float32 values by
            # at most 2.900e-8 relative across every AuraFlow and Flux case (#1538).
            assert_reference_schedule(schedule.sigmas, expected["snr_sigmas"], rel=1e-7)
            assert_reference_values(
                (
                    min(sigma for sigma in schedule.pre_offset if sigma > 0),
                    max(schedule.pre_offset),
                ),
                (expected["brownian_min"], expected["brownian_max"]),
            )
            noise = sampling_engine.brownian_step_noise(
                sampler, schedule, torch.zeros(1, 2, 2, 2), seed=23
            )
            if schedule.sigmas == schedule.pre_offset:
                assert noise is None
                noise = BrownianTreeNoise(
                    torch.zeros(1, 2, 2, 2),
                    min(sigma for sigma in schedule.sigmas if sigma > 0),
                    max(schedule.sigmas),
                    seed=23,
                    cpu=True,
                )
            assert noise is not None
            if "brownian_error" in expected:
                assert (case["name"], name, expected["brownian_error"]) == (
                    "large",
                    "normal",
                    "RecursionError",
                )
            else:
                for index, draw in enumerate(expected["brownian_draws"]):
                    actual = noise(schedule.sigmas[index], schedule.sigmas[index + 1])
                    assert_reference_tensor(actual, torch.tensor(draw, dtype=torch.float32))
        assert_reference_values(
            [
                patched.custom_sampling_percent_to_sigma(percent, return_actual_sigma=False)
                for percent in (0, 0.0001, 0.5, 1)
            ],
            case["percent_sigmas"],
        )

    @pytest.mark.parametrize("sampler_id", ["dinkster.euler", "dinkster.dpmpp_2m_sde"])
    def test_flux_sampling_patch_executes_shared_engine(
        self, assembled: AssembledFlux, sampler_id: str
    ) -> None:
        base = FluxRuntime(assembled, runtime_identity="test:flux-patch-sampling")
        patched = base.with_sampling_space(FluxFlowSigmas(shift=-0.5))
        latent = tiny_latent()
        cond = patched.encode_text("sampling patch")
        sampler = builtin_sampler_registry().get(sampler_id)
        assert sampler is not None
        sigmas = patched.custom_sampling_sigmas("dinkster.simple", 3, 1.0)
        expected = patched.sample(
            latent,
            cond=cond,
            sampler_id=sampler_id,
            scheduler_id="dinkster.simple",
            steps=3,
            seed=23,
            compute_dtype=torch.float32,
        )
        result = patched.sample_custom(
            latent,
            noise=prepare_noise(latent, 23),
            cond=cond,
            cfg=None,
            request=CustomSamplingRequest(sampler, (), sigmas),
            seed=23,
            compute_dtype=torch.float32,
        )
        assert torch.equal(result.output, expected)
        unpatched = base.sample(
            latent,
            cond=cond,
            sampler_id=sampler_id,
            scheduler_id="dinkster.simple",
            steps=3,
            seed=23,
            compute_dtype=torch.float32,
        )
        assert not torch.equal(expected, unpatched)

    def test_custom_sampling_uses_exact_sigmas_noise_options_and_denoised_state(
        self,
        assembled: AssembledFlux,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        built_options: list[dict[str, object]] = []

        def make(options: Mapping[str, object]) -> Any:
            built_options.append(dict(options))

            def solver(*_args: object, **_kwargs: object) -> torch.Tensor:
                raise AssertionError("the captured run_denoise must not execute the solver")

            return solver

        builtin = builtin_sampler_registry().get("dinkster.euler")
        assert builtin is not None
        sampler = replace(
            builtin,
            id="test.custom",
            aliases=(),
            make=make,
            noise=NoiseKind.BROWNIAN_GPU,
            requires_snr_offset=True,
        )
        samplers: dinkster_inference.Registry[SamplerDescriptor[Any]] = (
            dinkster_inference.Registry()
        )
        samplers.register(sampler)
        custom = FluxRuntime(
            assembled,
            runtime_identity="native:test:custom",
            sampler_registry=samplers,
        )
        latent = tiny_latent()
        noise = torch.full_like(latent, 0.25)
        pre_offset = (1.0, 0.7, 0.0)
        request = CustomSamplingRequest(
            sampler,
            (("s_churn", 0.5),),
            pre_offset,
        )
        captured: dict[str, object] = {}
        denoised = torch.full_like(latent, 0.4)
        output = torch.full_like(latent, 0.6)
        user_events: list[SamplingStateEvent[object]] = []
        step_noise = object()

        def capture_step_noise(
            resolved_sampler: SamplerDescriptor[Any],
            schedule: object,
            like: torch.Tensor,
            *,
            seed: int,
            device: torch.device,
        ) -> Any:
            assert resolved_sampler is sampler
            assert schedule is not None
            assert like is latent
            assert seed == 9
            assert device == next(assembled.diffusion.parameters()).device
            return step_noise

        def capture_run(_denoiser: object, _solver: object, **kwargs: object) -> torch.Tensor:
            captured.update(kwargs)
            callback = cast("Any", kwargs["on_state"])
            callback(
                SamplingStateEvent(
                    step=0,
                    total=2,
                    sigma=0.7,
                    phase="pre_update",
                    current=latent,
                    denoised=denoised,
                )
            )
            return output

        monkeypatch.setattr(sampling_engine, "brownian_step_noise", capture_step_noise)
        monkeypatch.setattr(sampling_engine, "run_denoise", capture_run)
        result = custom.sample_custom(
            latent,
            noise=noise,
            cond=custom.encode_text("a cat"),
            cfg=None,
            request=request,
            seed=9,
            on_state=user_events.append,
        )

        expected_sigmas = offset_first_sigma_for_snr(
            pre_offset,
            FluxFlowSigmas(shift=FLUX_DEV.sampling.shift),
            flow=True,
        )
        assert captured["noise"] is noise
        assert captured["sigmas"] == expected_sigmas
        assert captured["initial_sigma"] == pre_offset[0]
        assert captured["noise_sampler"] is step_noise
        assert built_options == [
            {
                "s_churn": 0.5,
                "s_tmin": 0.0,
                "s_tmax": float("inf"),
                "s_noise": 1.0,
            }
        ]
        assert result.output is output
        expected_denoised = latent_process_out(
            denoised,
            FLUX_DEV.single_stream_latent(),
        )
        assert result.denoised_output is not None
        assert torch.equal(result.denoised_output, expected_denoised)
        assert len(user_events) == 1

    def test_custom_sampling_sigmas_are_basic_scheduler_space(self, runtime: FluxRuntime) -> None:
        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert scheduler is not None
        expected = sampling_sigmas(
            scheduler,
            FluxFlowSigmas(shift=FLUX_DEV.sampling.shift),
            4,
            denoise=0.5,
        )
        assert runtime.custom_sampling_sigmas("dinkster.normal", 4, 0.5) == expected

    @pytest.mark.parametrize("family", [FLUX_DEV, FLUX_SCHNELL])
    def test_custom_sampling_model_sigma_queries_match_reference(
        self,
        assembled: AssembledFlux,
        family: ModelFamily,
    ) -> None:
        custom = FluxRuntime(
            replace(assembled, family=family),
            runtime_identity="native:test:model-sigma-queries",
        )
        options = CUSTOM_SIGMA_GOLDENS["beta_options"]
        assert_reference_schedule(
            custom.custom_sampling_beta_sigmas(options["steps"], options["alpha"], options["beta"]),
            CUSTOM_SIGMA_GOLDENS["beta"][family.id],
        )

        for key, expected in CUSTOM_SIGMA_GOLDENS["percent_to_sigma"][family.id].items():
            percent_text, actual_text = key.split(",")
            assert_reference_values(
                (
                    custom.custom_sampling_percent_to_sigma(
                        float(percent_text),
                        return_actual_sigma=actual_text == "True",
                    ),
                ),
                (expected,),
            )

        with pytest.raises(ValueError, match="require a discrete sigma space"):
            custom.custom_sampling_sd_turbo_sigmas(4, 0.65)

    def test_windowed_custom_sampling_matches_the_ksampler_facade(
        self, assembled: AssembledFlux
    ) -> None:
        custom = FluxRuntime(
            replace(assembled, _component_compute_dtypes={"diffusion": torch.float32}),
            runtime_identity="native:test:custom-parity",
        )
        latent = tiny_latent()
        cond = custom.encode_text("custom parity")
        seed = 23
        sigmas = custom.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        window_plan = flux_window_plan(split_height=True)

        expected = custom.sample(
            latent,
            cond=cond,
            sampler_id=sampler.id,
            scheduler_id="dinkster.normal",
            steps=2,
            seed=seed,
            window_plan=window_plan,
            compute_dtype=torch.float32,
        )
        result = custom.sample_custom(
            latent,
            noise=prepare_noise(latent, seed),
            cond=cond,
            cfg=None,
            request=CustomSamplingRequest(sampler, (), sigmas),
            seed=seed,
            window_plan=window_plan,
            compute_dtype=torch.float32,
        )

        assert torch.equal(result.output, expected)

    @pytest.mark.parametrize("windowed", (False, True))
    def test_masks_reach_both_sampling_surfaces(
        self, assembled: AssembledFlux, windowed: bool
    ) -> None:
        runtime = FluxRuntime(
            replace(assembled, _component_compute_dtypes={"diffusion": torch.float32}),
            runtime_identity="test:masked-flux",
        )
        latent = tiny_latent()
        cond = runtime.encode_text("masked sampling")
        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        plan = flux_window_plan(split_height=True) if windowed else None
        outputs: list[torch.Tensor] = []
        for mask in (None, torch.zeros_like(latent), torch.ones_like(latent)):
            composed = runtime.sample(
                latent,
                cond=cond,
                sampler_id=sampler.id,
                scheduler_id="dinkster.normal",
                steps=2,
                seed=23,
                window_plan=plan,
                denoise_mask=mask,
                compute_dtype=torch.float32,
            )
            direct = runtime.sample_custom(
                latent,
                noise=prepare_noise(latent, 23),
                cond=cond,
                cfg=None,
                request=CustomSamplingRequest(
                    sampler, (), runtime.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
                ),
                seed=23,
                window_plan=plan,
                denoise_mask=mask,
                compute_dtype=torch.float32,
            ).output
            assert torch.equal(composed, direct)
            outputs.append(composed)
        assert torch.equal(outputs[0], outputs[2])
        assert not torch.equal(outputs[0], outputs[1])

    def test_ksampler_facade_delegates_family_extras_to_custom_sampling(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dinkster_inference_torch import sampling_runtime

        captured: dict[str, object] = {}
        output = torch.zeros_like(tiny_latent())

        def capture(owner: object, latent: object, **kwargs: object) -> CustomSamplingResult[Any]:
            captured["owner"] = owner
            captured["latent"] = latent
            captured.update(kwargs)
            return CustomSamplingResult(output, None)

        monkeypatch.setattr(sampling_runtime, "run_ksampler_as_custom", capture)
        latent = tiny_latent()
        window_plan = flux_window_plan(split_height=True)

        result = runtime.sample(
            latent,
            cond=runtime.encode_text("facade composition"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=29,
            window_plan=window_plan,
            compute_dtype=torch.float32,
            device="cpu",
        )

        assert result is output
        assert captured["owner"] is runtime
        assert captured["latent"] is latent
        assert captured["sample_custom_kwargs"] == {
            "window_plan": window_plan,
            "compute_dtype": torch.float32,
            "device": "cpu",
            "capture_denoised": False,
        }

    def test_shape_and_determinism(self, runtime: FluxRuntime) -> None:
        latent = tiny_latent()
        cond = runtime.encode_text("a cat")
        out = runtime.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=3,
            compute_dtype=torch.float32,
        )
        assert out.shape == latent.shape
        again = runtime.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=3,
            compute_dtype=torch.float32,
        )
        assert torch.equal(out, again)

    def test_cfg_one_matches_positive_only(self, runtime: FluxRuntime) -> None:
        positive = runtime.encode_text("cfg one positive")
        output = runtime.sample(
            tiny_latent(),
            cond=positive,
            cfg=SamplingGuidance(runtime.encode_text("cfg one negative"), 1.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=17,
            compute_dtype=torch.float32,
        )
        assert torch.isfinite(output).all()
        assert output.amin() != output.amax()
        assert torch.equal(
            output,
            runtime.sample(
                tiny_latent(),
                cond=positive,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=2,
                seed=17,
                compute_dtype=torch.float32,
            ),
        )

    def test_none_and_one_full_window_are_bit_identical(
        self,
        runtime: FluxRuntime,
    ) -> None:
        latent = tiny_latent()
        cond = runtime.encode_text("full window identity")
        arguments: Any = {
            "cond": cond,
            "sampler_id": "dinkster.euler",
            "scheduler_id": "dinkster.normal",
            "steps": 2,
            "seed": 19,
            "compute_dtype": torch.float32,
        }

        ordinary = runtime.sample(latent, **arguments)
        explicit_none = runtime.sample(latent, window_plan=None, **arguments)
        windowed = runtime.sample(
            latent,
            window_plan=flux_window_plan(split_height=False),
            **arguments,
        )

        assert torch.equal(explicit_none, ordinary)
        assert torch.equal(windowed, ordinary)

    def test_multi_window_generation_is_deterministic_and_uses_global_positions(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[tuple[int, ...], torch.Tensor]] = []
        original = Flux.forward

        def capture(
            model: Flux,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            context: torch.Tensor,
            y: torch.Tensor | None = None,
            guidance: torch.Tensor | None = None,
            image_position_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            assert image_position_ids is not None
            calls.append((tuple(x.shape), image_position_ids.clone()))
            return original(
                model,
                x,
                timesteps,
                context,
                y,
                guidance,
                image_position_ids,
            )

        monkeypatch.setattr(Flux, "forward", capture)
        latent = tiny_latent()
        cond = runtime.encode_text("two declared windows")
        arguments: Any = {
            "cond": cond,
            "sampler_id": "dinkster.euler",
            "scheduler_id": "dinkster.normal",
            "steps": 1,
            "seed": 23,
            "window_plan": flux_window_plan(split_height=True),
            "compute_dtype": torch.float32,
        }

        first = runtime.sample(latent, **arguments)
        first_calls = tuple(calls)
        calls.clear()
        second = runtime.sample(latent, **arguments)

        assert torch.equal(first, second)
        assert torch.isfinite(first).all()
        assert first.amin() != first.amax()
        assert [shape for shape, _ids in calls] == [shape for shape, _ids in first_calls]
        assert all(
            torch.equal(actual_ids, expected_ids)
            for (_actual_shape, actual_ids), (_expected_shape, expected_ids) in zip(
                calls, first_calls, strict=True
            )
        )
        assert [shape[-2:] for shape, _ids in first_calls] == [(2, 4), (2, 4)]
        assert [ids.tolist() for _shape, ids in first_calls] == [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            [[[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]],
        ]

    def test_window_execution_plan_and_manifest_share_layout_digests(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[GuidedDenoiser] = []
        executed_layouts: list[tuple[str, ...]] = []
        original_guided_denoiser = sampling_engine.guided_denoiser
        original_validate_layout = wiring.FluxWindowConditioningEvaluation.validate_layout

        def capture_guided_denoiser(*args: Any, **kwargs: Any) -> GuidedDenoiser:
            guided = original_guided_denoiser(*args, **kwargs)
            captured.append(guided)
            return guided

        def capture_validate_layout(
            evaluation: Any,
            conditioning: Any,
            layout: ModelTokenLayout,
        ) -> None:
            executed_layouts.append(
                tuple(
                    window_layout.digest
                    for window_layout in conditioning.window_layouts
                    if window_layout is not None
                )
            )
            original_validate_layout(evaluation, conditioning, layout)

        monkeypatch.setattr(sampling_engine, "guided_denoiser", capture_guided_denoiser)
        monkeypatch.setattr(
            wiring.FluxWindowConditioningEvaluation,
            "validate_layout",
            capture_validate_layout,
        )
        latent = tiny_latent()
        cond = runtime.encode_text("window plan facts")
        plan = flux_window_plan(split_height=True)

        runtime.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=25,
            window_plan=plan,
            compute_dtype=torch.float32,
        )

        prepared = prepare_flux_window_plan(
            plan,
            latent_height=latent.shape[-2],
            latent_width=latent.shape[-1],
            patch_size=2,
        )
        assert tuple(window.declaration.index for window in prepared.windows) == tuple(
            range(len(prepared.windows))
        )
        text_token_count = declared_token_count(cond)
        assert text_token_count is not None
        expected = tuple(
            derive_flux_window_layout(
                text_token_count=text_token_count,
                latent_height=latent.shape[-2],
                latent_width=latent.shape[-1],
                patch_size=2,
                height_indices=window.height_indices,
                width_indices=window.width_indices,
            )[0].digest
            for window in prepared.windows
        )
        assert executed_layouts == [expected]
        compiled = captured[0].conditioning_plan
        assert compiled is not None
        full_layout = derive_flux_window_layout(
            text_token_count=text_token_count,
            latent_height=latent.shape[-2],
            latent_width=latent.shape[-1],
            patch_size=2,
            height_indices=(0, 1),
            width_indices=(0, 1),
        )[0]
        assert compiled.calls[0].layout_digest == full_layout.digest
        assert compiled.calls[0].inner_layout_digests == expected
        assert tuple(layout for layout, _transforms in compiled.lanes[0].inner_calls) == expected
        assert all(
            tuple(identity for identity, _digest in transforms)
            == ("flux.text-context-identity.v1", "flux.latent-window-crop.v1")
            for _layout, transforms in compiled.lanes[0].inner_calls
        )
        assert compiled.calls[0].layout_digest not in expected
        assert all(
            lane.validation is ConditioningValidationPath.LAYOUT_BACKED for lane in compiled.lanes
        )
        placeholder = tuple(
            hashlib.sha256(f"window-placeholder:{index}".encode()).hexdigest()
            for index in range(len(expected))
        )
        slot = build_windowed_evaluation_slot(
            derivation_identity="static-flux-window-plan.v1",
            derivation_facts_digest=hashlib.sha256(plan.digest.encode()).hexdigest(),
            plan_bindings=(WindowPlanBinding(plan, expected, placeholder, placeholder),),
        )
        assert all(
            f"plan[0].window[{index}].token_layout={digest}" in slot.facts
            for index, digest in enumerate(executed_layouts[0])
        )
        assert executed_layouts[0] == compiled.calls[0].inner_layout_digests

    def test_window_conditioning_prepares_once_per_window_not_per_step(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        preparations = 0
        original = FluxDenoiser.prepare_conditioning

        def prepare(
            evaluator: FluxDenoiser,
            conditioning: object,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            nonlocal preparations
            preparations += 1
            return cast(
                "tuple[torch.Tensor, torch.Tensor | None]",
                original(evaluator, conditioning),
            )

        monkeypatch.setattr(FluxDenoiser, "prepare_conditioning", prepare)
        runtime.sample(
            tiny_latent(),
            cond=runtime.encode_text("prepare per window"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=3,
            seed=27,
            window_plan=flux_window_plan(split_height=True),
            compute_dtype=torch.float32,
        )

        assert preparations == 2

    def test_windowed_cfg_pp_delivers_full_geometry_unconditional_prediction(
        self,
        runtime: FluxRuntime,
    ) -> None:
        output = runtime.sample(
            tiny_latent(),
            cond=runtime.encode_text("window cfg++ positive"),
            cfg=SamplingGuidance(runtime.encode_text("window cfg++ negative"), 2.0),
            sampler_id="dinkster.euler_cfg_pp",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=29,
            window_plan=flux_window_plan(split_height=True),
            compute_dtype=torch.float32,
        )

        assert output.shape == tiny_latent().shape
        assert torch.isfinite(output).all()

    def test_windowed_model_calls_stay_below_one_custom_guidance_execution(
        self,
        assembled: AssembledFlux,
    ) -> None:
        calls: list[tuple[str, tuple[int, ...], int]] = []

        def wrapper(
            request: GuidanceEvaluationRequest[torch.Tensor],
            next: GuidanceEvaluateNext[torch.Tensor],
        ) -> GuidancePredictions[torch.Tensor]:
            calls.append(
                (
                    "wrapper-enter",
                    tuple(request.input.shape),
                    request.execution.model_evaluation,
                )
            )
            result = next(request)
            calls.append(
                (
                    "wrapper-exit",
                    tuple(request.input.shape),
                    request.execution.model_evaluation,
                )
            )
            return result

        def post(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
            calls.append(
                (
                    "post",
                    tuple(context.reduced.shape),
                    context.request.execution.model_evaluation,
                )
            )
            return context.reduced

        contribution = GuidanceContribution(
            evaluation_wrappers=(GuidanceEvaluationWrapperDescriptor("test.wrapper", wrapper),),
            post_cfg=(
                GuidancePostCFGDescriptor(
                    "test.post",
                    post,
                ),
            ),
        )
        custom_runtime = FluxRuntime(
            assembled,
            runtime_identity="native:test:window-guidance",
            guidance_executor=GuidanceExecutor(GuidanceRegistry((("test", contribution),))),
        )

        output = custom_runtime.sample(
            tiny_latent(),
            cond=custom_runtime.encode_text("window guidance"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=31,
            window_plan=flux_window_plan(split_height=True),
            compute_dtype=torch.float32,
        )

        assert output.shape == tiny_latent().shape
        assert calls == [
            ("wrapper-enter", (1, TINY_FLUX.in_channels, 4, 4), 0),
            ("wrapper-exit", (1, TINY_FLUX.in_channels, 4, 4), 0),
            ("post", (1, TINY_FLUX.in_channels, 4, 4), 0),
        ]

    def test_window_plan_refuses_before_any_model_call(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0
        original = Flux.forward

        def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(Flux, "forward", capture)
        mismatched = compile_window_plan(
            axes=(MediaAxis("height", 3), MediaAxis("width", 2)),
            kinds=(
                WindowKind(
                    "latent_image",
                    (
                        KindAxisMap("height", 3, IntegerAffineIndexMap(1)),
                        KindAxisMap("width", 2, IntegerAffineIndexMap(1)),
                    ),
                ),
                WindowKind("text", invariant_axes=("height", "width")),
            ),
            layers=(
                WindowPlanLayer(
                    ("height", "width"),
                    (LayerWindow((WindowIndexList((0, 1, 2)), WindowIndexList((0, 1)))),),
                    (
                        WindowWeightProfile(WindowWeightKind.FLAT),
                        WindowWeightProfile(WindowWeightKind.FLAT),
                    ),
                    MergeDeclaration(),
                ),
            ),
        )
        arguments: Any = {
            "cond": runtime.encode_text("invalid window"),
            "sampler_id": "dinkster.euler",
            "scheduler_id": "dinkster.normal",
            "steps": 1,
            "denoise": 0.0,
            "compute_dtype": torch.float32,
        }

        with pytest.raises(WiringError, match="plan-latent-geometry-mismatch"):
            runtime.sample(tiny_latent(), window_plan=mismatched, **arguments)
        with pytest.raises(WiringError, match="invalid-window-plan"):
            runtime.sample(tiny_latent(), window_plan=cast(Any, object()), **arguments)
        assert calls == 0

    def test_window_plan_refuses_preencoded_conditioning_before_zero_step(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0
        original = Flux.forward

        def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(Flux, "forward", capture)
        conditioning = Conditioning(
            torch.zeros(1, 2, TINY_T5.d_model),
            torch.zeros(1, TINY_CLIP.hidden_size),
        )

        with pytest.raises(
            WiringError,
            match="window-layout-declaration-required: windowed execution requires"
            " declared conditioning",
        ):
            runtime.sample(
                tiny_latent(),
                cond=conditioning,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                denoise=0.0,
                window_plan=flux_window_plan(split_height=False),
                compute_dtype=torch.float32,
            )
        assert calls == 0

        ordinary = runtime.sample(
            tiny_latent(),
            cond=conditioning,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            denoise=0.0,
            compute_dtype=torch.float32,
        )
        assert torch.equal(ordinary, tiny_latent())

    @pytest.mark.parametrize("masked", (False, True))
    @pytest.mark.parametrize("world_size", (2, 3))
    def test_distributed_window_mode_preserves_masks_and_callbacks(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
        masked: bool,
        world_size: int,
    ) -> None:
        latent = tiny_latent()
        cond = runtime.encode_text("distributed window step events")
        mask = torch.ones_like(latent) if masked else None
        if mask is not None:
            mask[..., 0] = 0
        states: list[object] = []

        sampler = builtin_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        sigmas = runtime.custom_sampling_sigmas("dinkster.normal", 3, 1.0)
        window_plan = flux_window_plan(split_height=True)

        def run(*, direct: bool = False) -> tuple[list[tuple[int, int, float]], torch.Tensor]:
            events: list[tuple[int, int, float]] = []

            def on_step(event: StepEvent) -> None:
                events.append((event.step, event.total, event.sigma))

            if direct:
                output = runtime.sample_custom(
                    latent,
                    noise=prepare_noise(latent, 31),
                    cond=cond,
                    cfg=None,
                    request=CustomSamplingRequest(sampler, (), sigmas),
                    seed=31,
                    window_plan=window_plan,
                    on_step=on_step,
                    on_state=states.append,
                    denoise_mask=mask,
                    compute_dtype=torch.float32,
                ).output
            else:
                output = runtime.sample(
                    latent,
                    cond=cond,
                    sampler_id=sampler.id,
                    scheduler_id="dinkster.normal",
                    steps=3,
                    seed=31,
                    window_plan=window_plan,
                    on_step=on_step,
                    on_state=states.append,
                    denoise_mask=mask,
                    compute_dtype=torch.float32,
                )
            return events, output

        baseline_events, baseline_output = run()
        assert len(baseline_events) == 3

        class LocalTransport:
            """Unanimous digest exchange standing in for the group sideband."""

            def __init__(
                self,
                config: DistributedSamplingConfig,
                device: torch.device,
            ) -> None:
                del device
                self.physical_ranks = tuple(range(config.world_size))
                self.rank = config.rank
                self._world_size = config.world_size

            def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
                del rank
                return (digest,) * self._world_size

        constructed_scatters: list[object] = []

        class LocalWindowScatter:
            """Rank-local stand-in running the serial per-window loop."""

            def __init__(self, inner: Any, consensus_token: Any) -> None:
                assert type(consensus_token) is ManifestConsensusToken
                self.prepare_conditioning = inner.prepare_conditioning
                self.evaluate_conditioning = inner.evaluate_conditioning
                self.batchable = inner.batchable
                self.evaluate_conditioning_batch = inner.evaluate_conditioning_batch
                self.validate_layout = inner.validate_layout
                self.inner_calls = inner.inner_calls
                constructed_scatters.append(self)

        config = DistributedSamplingConfig(
            0, world_size, "window", "file:///group", "1" * 32, "a:1"
        )
        # The manifest demands a digest-bearing runtime identity; the
        # fixture's abbreviated one is fine everywhere else.
        monkeypatch.setattr(
            runtime,
            "_runtime_identity",
            f"native:test:{hashlib.sha256(b'window wiring').hexdigest()}",
        )
        monkeypatch.setattr(wiring, "distributed_sampling_config", lambda: config)
        monkeypatch.setattr(wiring, "ensure_process_group", lambda: config)
        monkeypatch.setattr(wiring, "window_preflight_failed", _local_preflight)
        monkeypatch.setattr(wiring, "window_route_mismatch", _local_route_mismatch)
        monkeypatch.setattr(wiring, "WindowDigestConsensusTransport", LocalTransport)
        monkeypatch.setattr(wiring, "DistributedFluxWindowEvaluation", LocalWindowScatter)

        distributed_events, distributed_output = run()
        custom_events, custom_output = run(direct=True)

        assert len(constructed_scatters) == 2
        assert distributed_events == baseline_events
        assert custom_events == baseline_events
        assert torch.equal(distributed_output, baseline_output)
        assert torch.equal(custom_output, baseline_output)
        assert len(states) == 9
        if masked:
            from dinkster_inference_torch.denoise import latent_process_in, latent_process_out

            descriptor = runtime.family.single_stream_latent()
            roundtrip = latent_process_out(latent_process_in(latent, descriptor), descriptor)
            assert torch.equal(distributed_output[..., 0], roundtrip[..., 0])

    def test_distributed_manifest_binds_pre_and_post_snr_offset_sigmas(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class ManifestCaptured(Exception):
            pass

        captured: dict[str, object] = {}

        def capture_manifest(**kwargs: object) -> CanonicalManifest:
            captured.update(kwargs)
            raise ManifestCaptured

        config = DistributedSamplingConfig(0, 2, "window", "file:///group", "1" * 32, "a:1")
        monkeypatch.setattr(wiring, "distributed_sampling_config", lambda: config)
        monkeypatch.setattr(wiring, "ensure_process_group", lambda: config)
        monkeypatch.setattr(wiring, "window_preflight_failed", _local_preflight)
        monkeypatch.setattr(wiring, "window_route_mismatch", _local_route_mismatch)
        monkeypatch.setattr(wiring, "build_flux_window_manifest", capture_manifest)

        sampler = builtin_sampler_registry().get("dinkster.exp_heun_2_x0")
        assert sampler is not None and sampler.requires_snr_offset
        sigmas = (1.0, 0.5, 0.0)
        latent = tiny_latent()
        with pytest.raises(ManifestCaptured):
            runtime.sample_custom(
                latent,
                noise=prepare_noise(latent, 37),
                cond=runtime.encode_text("snr offset manifest"),
                cfg=None,
                request=CustomSamplingRequest(sampler, (), sigmas),
                seed=37,
                window_plan=flux_window_plan(split_height=True),
                compute_dtype=torch.float32,
            )

        assert captured["pre_offset_sigmas"] == sigmas
        executed = cast("tuple[float, ...]", captured["sigmas"])
        assert executed != sigmas
        assert executed[1:] == sigmas[1:]

    @pytest.mark.parametrize("mode", ("guidance", "window", "auto"))
    def test_distributed_flux_without_windows_uses_shared_guidance(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
    ) -> None:
        from dinkster_inference_torch import distributed

        config = DistributedSamplingConfig(0, 2, mode, "file:///group", "1" * 32, "a:1")
        monkeypatch.setattr(wiring, "ensure_process_group", lambda: config)
        monkeypatch.setattr(wiring, "window_preflight_failed", _local_preflight)
        monkeypatch.setattr(wiring, "window_route_mismatch", _local_route_mismatch)
        cond = runtime.encode_text("shared distributed Flux")

        def sample() -> torch.Tensor:
            return runtime.sample(
                tiny_latent(),
                cond=cond,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=1,
                seed=33,
                denoise_mask=torch.ones_like(tiny_latent()),
                on_state=lambda _event: None,
                compute_dtype=torch.float32,
            )

        baseline = sample()
        transported: list[object] = []

        def evaluate(evaluator: Any, x: Any, sigma: Any, request: Any) -> Any:
            transported.append(request)
            return evaluator.evaluate(x, sigma, request)

        monkeypatch.setattr(wiring, "distributed_sampling_config", lambda: config)
        monkeypatch.setattr(distributed, "distributed_sampling_config", lambda: config)
        monkeypatch.setattr(distributed, "ensure_process_group", lambda: config)
        monkeypatch.setattr(distributed.DistributedGuidanceEvaluator, "evaluate_request", evaluate)
        assert torch.equal(sample(), baseline)
        assert transported

    def test_declares_text_and_packed_latent_layout_for_fused_cfg(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[GuidedDenoiser] = []
        original = sampling_engine.guided_denoiser

        def capture(*args: Any, **kwargs: Any) -> GuidedDenoiser:
            guided = original(*args, **kwargs)
            captured.append(guided)
            return guided

        monkeypatch.setattr(sampling_engine, "guided_denoiser", capture)
        latent = tiny_latent()
        cond = runtime.encode_text("layout positive")
        uncond = runtime.encode_text("layout negative")
        runtime.sample(
            latent,
            cond=cond,
            cfg=SamplingGuidance(uncond, 2.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=11,
            compute_dtype=torch.float32,
        )

        text_tokens = cond.embeddings.shape[1]
        image_grid = (latent.shape[2] // 2, latent.shape[3] // 2)
        image_tokens = image_grid[0] * image_grid[1]
        expected_layout = ModelTokenLayout(
            (
                ModelTokenSegment(
                    "text",
                    "text",
                    "context",
                    0,
                    text_tokens,
                    (text_tokens,),
                ),
                ModelTokenSegment(
                    "latent_image",
                    "image",
                    "latent",
                    text_tokens,
                    text_tokens + image_tokens,
                    image_grid,
                ),
            ),
            0,
        )
        expected_transforms = (
            TokenGridTransform(
                "flux.text-context-identity.v1",
                "text",
                "text",
                (text_tokens,),
                None,
            ),
            TokenGridTransform(
                "flux.latent-image-pack-2x2.v1",
                "image",
                "latent_image",
                tuple(latent.shape[-2:]),
                None,
            ),
        )
        compiled = captured[0].conditioning_plan
        assert compiled is not None
        assert len(compiled.calls) == 1
        assert compiled.calls[0].layout_digest == expected_layout.digest
        assert all(
            lane.validation is ConditioningValidationPath.LAYOUT_BACKED for lane in compiled.lanes
        )
        assert all(
            lane.token_transforms
            == tuple((transform.transform, transform.digest) for transform in expected_transforms)
            for lane in compiled.lanes
        )

    def test_pre_encoded_flux_conditioning_remains_layout_absent(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[GuidedDenoiser] = []
        original = sampling_engine.guided_denoiser

        def capture(*args: Any, **kwargs: Any) -> GuidedDenoiser:
            guided = original(*args, **kwargs)
            captured.append(guided)
            return guided

        monkeypatch.setattr(sampling_engine, "guided_denoiser", capture)
        cond = Conditioning(
            torch.zeros(1, 2, TINY_T5.d_model), torch.zeros(1, TINY_CLIP.hidden_size)
        )
        uncond = Conditioning(
            torch.zeros(1, 2, TINY_T5.d_model), torch.zeros(1, TINY_CLIP.hidden_size)
        )
        runtime.sample(
            tiny_latent(),
            cond=cond,
            cfg=SamplingGuidance(uncond, 2.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=12,
            compute_dtype=torch.float32,
        )

        compiled = captured[0].conditioning_plan
        assert compiled is not None
        assert compiled.calls[0].layout_digest is None
        assert all(
            lane.validation is ConditioningValidationPath.LEGACY_SHAPE_PREDICATE
            for lane in compiled.lanes
        )

    def test_unequal_declared_flux_text_lengths_use_one_layout_backed_call(
        self,
        runtime: FluxRuntime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[GuidedDenoiser] = []
        original = sampling_engine.guided_denoiser

        def capture(*args: Any, **kwargs: Any) -> GuidedDenoiser:
            guided = original(*args, **kwargs)
            captured.append(guided)
            return guided

        monkeypatch.setattr(sampling_engine, "guided_denoiser", capture)
        cond = declare_text_conditioning(
            Conditioning(torch.zeros(1, 2, TINY_T5.d_model), torch.zeros(1, TINY_CLIP.hidden_size)),
            2,
        )
        uncond = declare_text_conditioning(
            Conditioning(torch.zeros(1, 3, TINY_T5.d_model), torch.zeros(1, TINY_CLIP.hidden_size)),
            3,
        )
        runtime.sample(
            tiny_latent(),
            cond=cond,
            cfg=SamplingGuidance(uncond, 2.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
            seed=13,
            compute_dtype=torch.float32,
        )

        compiled = captured[0].conditioning_plan
        assert compiled is not None
        assert tuple(call.lane_ids for call in compiled.calls) == (("negative", "positive"),)
        lane_layouts = {lane.layout_digest for lane in compiled.lanes}
        assert None not in lane_layouts
        assert len(lane_layouts) == 2
        assert compiled.calls[0].layout_digest not in lane_layouts
        assert all(
            lane.validation is ConditioningValidationPath.LAYOUT_BACKED for lane in compiled.lanes
        )

    def test_euler_matches_the_documented_recipe(
        self, runtime: FluxRuntime, assembled: AssembledFlux
    ) -> None:
        """sample() is schedule -> prepare_noise -> FluxDenoiser ->
        run_denoise, nothing else: the composed call must reproduce
        the recipe assembled by hand from the pinned pieces."""
        latent = tiny_latent()
        cond = runtime.encode_text("a cat")
        uncond = runtime.encode_text("")
        actual = runtime.sample(
            latent,
            cond=cond,
            cfg=SamplingGuidance(uncond, 3.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=7,
            compute_dtype=torch.float32,
        )
        sampler = torch_sampler_registry().get("dinkster.euler")
        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert sampler is not None and scheduler is not None
        space = FluxFlowSigmas(shift=FLUX_DEV.sampling.shift)
        sigmas = sampling_sigmas(scheduler, space, 2)
        evaluator = FluxDenoiser(
            assembled.diffusion,
            compute_dtype=torch.float32,
        )
        denoiser = guided_denoiser(
            ConditioningEvaluation(
                lambda value, _role: evaluator.prepare_conditioning(value),
                evaluator.evaluate_conditioning,
                evaluator.batchable,
                evaluator.evaluate_conditioning_batch,
                standard_activation_memory_factor=1.0,
            ),
            input=latent,
            executor=None,
            plan=compile_guidance_plan(
                cond,
                SamplingGuidance(uncond, 3.0),
                sampler,
                None,
            ),
            execution=sampling_execution_context(sigmas, 7),
        )
        expected = run_denoise(
            denoiser,
            sampler.build(),
            latent=latent,
            noise=prepare_noise(latent, 7),
            sigmas=sigmas,
            family=FLUX_DEV,
            seed=7,
            noise_kind=sampler.noise,
        )
        assert torch.equal(actual, expected)

    def test_torch_euler_uses_device_float32_division_and_dt(self) -> None:
        sampler = torch_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        x = torch.tensor([-12.92857837677002], dtype=torch.float32)
        sigmas = (14.614640235900879, 10.746800422668457)

        def denoiser(x: torch.Tensor, sigma: float) -> torch.Tensor:
            assert sigma == sigmas[0]
            return torch.tensor([0.6638044118881226], dtype=torch.float32)

        actual = sampler.build()(
            denoiser,
            x,
            sigmas,
            dinkster_inference.SamplerInfo(dinkster_inference.Parameterization.EPS),
        )
        sigma = torch.tensor(sigmas[0], dtype=torch.float32)
        sigma_next = torch.tensor(sigmas[1], dtype=torch.float32)
        denoised = denoiser(x, sigmas[0])
        expected = x + ((x - denoised) / sigma) * (sigma_next - sigma)
        assert torch.equal(actual, expected)

    def test_sde_brownian_tree_is_built_before_the_snr_offset(
        self, runtime: FluxRuntime, assembled: AssembledFlux
    ) -> None:
        """The reference builds the initial state and brownian tree from
        the PRE-offset schedule, then offset_first_sigma_for_snr nudges
        sigmas[0] for flow model evaluations (comfy/k_diffusion/sampling.py
        @ 947c2749). sample() must replay that order."""
        latent = tiny_latent()
        cond = runtime.encode_text("a cat")
        actual = runtime.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.dpmpp_2m_sde",
            scheduler_id="dinkster.normal",
            steps=3,
            seed=5,
            compute_dtype=torch.float32,
        )
        sampler = torch_sampler_registry().get("dinkster.dpmpp_2m_sde")
        scheduler = torch_scheduler_registry().get("dinkster.normal")
        assert sampler is not None and scheduler is not None
        assert sampler.noise is NoiseKind.BROWNIAN
        space = FluxFlowSigmas(shift=FLUX_DEV.sampling.shift)
        sigmas = sampling_sigmas(scheduler, space, 3)
        assert sigmas[0] == 1.0  # the offset must actually fire
        positive = [sigma for sigma in sigmas if sigma > 0]
        tree = BrownianTreeNoise(latent.to(torch.float32), min(positive), max(sigmas), seed=5)
        offset = offset_first_sigma_for_snr(sigmas, space, flow=True)
        assert offset[0] != sigmas[0]
        denoiser = FluxDenoiser(assembled.diffusion, cond, compute_dtype=torch.float32)
        expected = run_denoise(
            denoiser,
            sampler.build(),
            latent=latent,
            noise=prepare_noise(latent, 5),
            sigmas=offset,
            initial_sigma=sigmas[0],
            family=FLUX_DEV,
            seed=5,
            noise_kind=sampler.noise,
            noise_sampler=tree,
        )
        assert torch.equal(actual, expected)

    def test_zero_denoise_returns_latent_untouched(self, runtime: FluxRuntime) -> None:
        latent = tiny_latent()
        out = runtime.sample(
            latent,
            cond=runtime.encode_text("a cat"),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            denoise=0.0,
        )
        assert torch.equal(out, latent)

    def test_zero_denoise_still_validates_conditioning(self, runtime: FluxRuntime) -> None:
        with pytest.raises(DenoiseError, match="embeddings must be"):
            runtime.sample(
                tiny_latent(),
                cond=Conditioning(torch.zeros(1, 4)),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=2,
                denoise=0.0,
            )

    def test_unknown_sampler_refuses_by_name(self, runtime: FluxRuntime) -> None:
        with pytest.raises(WiringError, match="unknown sampler 'nope'"):
            runtime.sample(
                tiny_latent(),
                cond=runtime.encode_text("a cat"),
                sampler_id="nope",
                scheduler_id="dinkster.normal",
                steps=1,
            )

    def test_unknown_scheduler_refuses_by_name(self, runtime: FluxRuntime) -> None:
        with pytest.raises(WiringError, match="unknown scheduler 'nope'"):
            runtime.sample(
                tiny_latent(),
                cond=runtime.encode_text("a cat"),
                sampler_id="dinkster.euler",
                scheduler_id="nope",
                steps=1,
            )


# --- the codec delegation ---------------------------------------------------


class TestCodec:
    def test_encode_decode_roundtrip_shapes(self, runtime: FluxRuntime) -> None:
        generator = torch.Generator("cpu")
        generator.manual_seed(11)
        content = torch.rand(1, 3, 16, 16, generator=generator)
        latent = runtime.encode_content(content)
        assert latent.shape == (1, TINY_KL.embed_dim, 8, 8)
        decoded = runtime.decode_latent(latent)
        assert decoded.shape == (1, 3, 16, 16)

    def test_delegates_to_the_codec_plugin(self, runtime: FluxRuntime) -> None:
        generator = torch.Generator("cpu")
        generator.manual_seed(12)
        content = torch.rand(1, 3, 16, 16, generator=generator)
        assert torch.equal(runtime.encode_content(content), runtime.codec.encode(content))
        latent = runtime.encode_content(content)
        assert torch.equal(runtime.decode_latent(latent), runtime.codec.decode(latent))
