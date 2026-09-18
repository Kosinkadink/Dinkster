"""Proving tests for the per-family runtime seam (stage 5 wiring).

probe_native must answer "can this checkpoint go native?" from headers
and modeled scalar config, with every "no" carrying its reasons; the FamilyRuntime
protocol must be satisfiable by a torch-free implementation (the
torch realization proves itself in the package suite). The seam is
pinned with the backend thread (STAGE-6 SEAM, 2026-07-26): shape
changes here are contract changes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypedDict, cast

import dinkster_inference.runtime as runtime_module
import pytest
from dinkster_inference import (
    BFLOAT16,
    CLIP_G_TEXT_CONFIG,
    CLIP_L_TEXT_CONFIG,
    FLOAT8_E5M2,
    FLOAT16,
    FLOAT32,
    FLUX_CLIP_L_PREFIX,
    FLUX_DEV,
    FLUX_DEV_CONFIG,
    FLUX_DIFFUSION_PREFIX,
    FLUX_T5XXL_PREFIX,
    FLUX_VAE_PREFIX,
    INT64,
    SD15_CLIP_L_PREFIX,
    SD15_UNET_CONFIG,
    SD_DIFFUSION_PREFIX,
    SD_VAE_PREFIX,
    SDXL_CLIP_G_PREFIX,
    SDXL_CLIP_L_PREFIX,
    SDXL_REFINER_CLIP_G_PREFIX,
    SDXL_REFINER_UNET_CONFIG,
    SDXL_UNET_CONFIG,
    T5_XXL_CONFIG,
    AssemblyError,
    AssemblyRegistration,
    ComponentPlan,
    ComponentWiring,
    Conditioning,
    ConditioningCarrier,
    ConditioningRuntime,
    CustomSamplingRuntime,
    DetectionEvidence,
    FamilyRegistry,
    FamilyRuntime,
    InpaintConditioning,
    LatentDescriptor,
    ModelFamily,
    MultiStreamConditioningRuntime,
    NativeAssemblyPlan,
    NativeCapability,
    NativeRefusalCategory,
    NativeRefusalError,
    Parameterization,
    SamplingDescriptor,
    SamplingGuidance,
    SamplingSegment,
    SamplingStateCallback,
    StepCallback,
    TensorGeometry,
    WeightEntry,
    WeightSource,
    builtin_assembly_registry,
    clip_text_layout,
    flux_layout,
    krea2_text_layout,
    openclip_text_layout,
    plan_native,
    probe_native,
    t5_layout,
    unet_layout,
)

KL_GOLDENS = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "kl_goldens.json"
)


# --- helpers (the assembly-test source builders) ------------------------


@dataclass
class FakeSource:
    """In-memory WeightSource with the path the planner requires."""

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
    identity_components: tuple[ComponentPlan[tuple[int, ...]], ...]


def plan_synthetic_assembly(
    *, checkpoint: WeightSource | None, **sources: WeightSource | None
) -> SyntheticAssemblyPlan:
    if (
        checkpoint is None
        or tuple(checkpoint.keys()) != ("weight",)
        or any(value is not None for value in sources.values())
    ):
        raise AssemblyError("synthetic checkpoint requires one weight and no split sources")
    geometry = checkpoint.entry("weight").geometry
    return SyntheticAssemblyPlan(
        replace(FLUX_DEV, id="test.synthetic"),
        (
            ComponentPlan(
                "diffusion",
                cast("FakeSource", checkpoint).path,
                geometry.shape,
                {"weight": "weight"},
                {"weight": geometry.dtype},
                {},
            ),
        ),
    )


def load_synthetic_runtime(plan: NativeAssemblyPlan, **_options: object) -> StubRuntime:
    return StubRuntime(plan.family)


def synthetic_assembly_registration() -> AssemblyRegistration:
    return AssemblyRegistration(
        "test.synthetic", plan_synthetic_assembly, f"{__name__}:load_synthetic_runtime"
    )


def test_registered_synthetic_assembly_plans_without_table_edits() -> None:
    checkpoint = FakeSource(
        Path("/fake/synthetic.safetensors"), {"weight": TensorGeometry((1,), FLOAT32)}
    )
    assert not probe_native(checkpoint).native
    registry = builtin_assembly_registry()
    registry.register(synthetic_assembly_registration())
    planned = plan_native(checkpoint, assembly_registry=registry)
    assert planned == plan_synthetic_assembly(checkpoint=checkpoint)
    assert probe_native(checkpoint, assembly_registry=registry).native
    assert planned.family.id == "test.synthetic"
    assert not probe_native(checkpoint).native


def test_ambiguous_assembly_registration_names_both_implementations() -> None:
    checkpoint = FakeSource(
        Path("/fake/synthetic.safetensors"), {"weight": TensorGeometry((1,), FLOAT32)}
    )
    registry = builtin_assembly_registry()
    entry = synthetic_assembly_registration()
    registry.register(entry)
    registry.register(replace(entry, id="test.other"))
    capability = probe_native(checkpoint, assembly_registry=registry)
    assert not capability.native
    assert "test.synthetic, test.other" in capability.reasons[0]


def test_registered_planner_contract_failure_is_not_an_unknown_family_refusal() -> None:
    checkpoint = FakeSource(
        Path("/fake/synthetic.safetensors"), {"weight": TensorGeometry((1,), FLOAT32)}
    )
    registry = builtin_assembly_registry()
    registry.register(
        replace(
            synthetic_assembly_registration(), plan=lambda **_: cast("NativeAssemblyPlan", object())
        )
    )
    with pytest.raises(TypeError, match="test.synthetic.*NativeAssemblyPlan"):
        probe_native(checkpoint, assembly_registry=registry)


def geometrize(
    layout: dict[str, tuple[int, ...]],
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, FLOAT16) for key, shape in layout.items()}


def prefixed(sd: dict[str, TensorGeometry], prefix: str) -> dict[str, TensorGeometry]:
    return {prefix + key: value for key, value in sd.items()}


def kl_geometries() -> dict[str, TensorGeometry]:
    payload = json.loads(KL_GOLDENS.read_text())
    return {
        key: TensorGeometry(tuple(shape), FLOAT32)
        for key, shape in payload["cases"]["standard"]["state_dict"]
    }


def source(sd: dict[str, TensorGeometry], name: str) -> FakeSource:
    return FakeSource(Path(f"/fake/{name}"), sd)


def dev_geometries() -> dict[str, TensorGeometry]:
    return geometrize(flux_layout(FLUX_DEV_CONFIG))


def schnell_geometries() -> dict[str, TensorGeometry]:
    return {
        key: value for key, value in dev_geometries().items() if not key.startswith("guidance_in.")
    }


def combined_dev_checkpoint() -> FakeSource:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(dev_geometries(), FLUX_DIFFUSION_PREFIX))
    sd.update(prefixed(geometrize(clip_text_layout(CLIP_L_TEXT_CONFIG)), FLUX_CLIP_L_PREFIX))
    sd.update(prefixed(geometrize(t5_layout(T5_XXL_CONFIG)), FLUX_T5XXL_PREFIX))
    sd.update(prefixed(kl_geometries(), FLUX_VAE_PREFIX))
    return source(sd, "combined.safetensors")


class SplitSources(TypedDict):
    """The split-file probe kwargs, typed so ``**`` unpacking checks."""

    diffusion: WeightSource
    clip_l: WeightSource
    t5xxl: WeightSource
    vae: WeightSource


def split_sources() -> SplitSources:
    return {
        "diffusion": source(dev_geometries(), "dit.safetensors"),
        "clip_l": source(geometrize(clip_text_layout(CLIP_L_TEXT_CONFIG)), "clip_l.safetensors"),
        "t5xxl": source(geometrize(t5_layout(T5_XXL_CONFIG)), "t5.safetensors"),
        "vae": source(kl_geometries(), "ae.safetensors"),
    }


def sd_clip_l_geometries() -> dict[str, TensorGeometry]:
    """SD-checkpoint CLIP-L: transformers format, no text projection,
    plus the inert position_ids buffer real checkpoints carry."""
    sd = {
        key: value
        for key, value in geometrize(clip_text_layout(CLIP_L_TEXT_CONFIG)).items()
        if key != "text_projection.weight"
    }
    sd["text_model.embeddings.position_ids"] = TensorGeometry((1, 77), INT64)
    return sd


def sd_clip_g_geometries() -> dict[str, TensorGeometry]:
    sd = geometrize(openclip_text_layout(CLIP_G_TEXT_CONFIG))
    sd["logit_scale"] = TensorGeometry((), FLOAT16)
    return sd


def sd15_checkpoint() -> FakeSource:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(geometrize(unet_layout(SD15_UNET_CONFIG)), SD_DIFFUSION_PREFIX))
    sd.update(prefixed(sd_clip_l_geometries(), SD15_CLIP_L_PREFIX))
    sd.update(prefixed(kl_geometries(), SD_VAE_PREFIX))
    return source(sd, "sd15.safetensors")


def sdxl_checkpoint() -> FakeSource:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(geometrize(unet_layout(SDXL_UNET_CONFIG)), SD_DIFFUSION_PREFIX))
    sd.update(prefixed(sd_clip_l_geometries(), SDXL_CLIP_L_PREFIX))
    sd.update(prefixed(sd_clip_g_geometries(), SDXL_CLIP_G_PREFIX))
    sd.update(prefixed(kl_geometries(), SD_VAE_PREFIX))
    return source(sd, "sdxl.safetensors")


def refiner_checkpoint() -> FakeSource:
    sd: dict[str, TensorGeometry] = {}
    sd.update(prefixed(geometrize(unet_layout(SDXL_REFINER_UNET_CONFIG)), SD_DIFFUSION_PREFIX))
    sd.update(prefixed(sd_clip_g_geometries(), SDXL_REFINER_CLIP_G_PREFIX))
    sd.update(prefixed(kl_geometries(), SD_VAE_PREFIX))
    return source(sd, "refiner.safetensors")


@pytest.mark.parametrize("checkpoint_factory", [combined_dev_checkpoint, sd15_checkpoint])
def test_component_geometry_plans_without_family_registry(
    checkpoint_factory: Callable[[], FakeSource], caplog: pytest.LogCaptureFixture
) -> None:
    checkpoint = checkpoint_factory()
    expected = plan_native(checkpoint)
    actual = plan_native(checkpoint, registry=FamilyRegistry())
    assert actual == expected
    assert "defaulting model, text, codec, sampling and dtype behavior" in caplog.text
    assert type(actual).__name__ in caplog.text


# --- NativeCapability invariants ----------------------------------------


class TestNativeCapability:
    def test_native_carries_no_reasons(self) -> None:
        with pytest.raises(ValueError, match="no reasons"):
            NativeCapability(family_id="dinkster.flux_dev", native=True, reasons=("x",))

    def test_refusal_must_name_reasons(self) -> None:
        with pytest.raises(ValueError, match="must name reasons"):
            NativeCapability(family_id="dinkster.flux_dev", native=False)

    def test_native_names_its_family(self) -> None:
        with pytest.raises(ValueError, match="names its family"):
            NativeCapability(family_id=None, native=True)

    def test_refusal_defaults_to_fail_closed_category(self) -> None:
        capability = NativeCapability(family_id=None, native=False, reasons=("strict refusal",))
        assert capability.refusal_category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY

    def test_native_requires_no_refusal_category(self) -> None:
        with pytest.raises(ValueError, match="no refusal category"):
            NativeCapability(
                family_id="dinkster.flux_dev",
                native=True,
                refusal_category=NativeRefusalCategory.NATIVE_INELIGIBLE,
            )

    def test_explicit_category_must_be_enum(self) -> None:
        invalid = cast(NativeRefusalCategory, "native_ineligible")
        with pytest.raises(TypeError, match="NativeRefusalCategory"):
            NativeCapability(
                family_id=None,
                native=False,
                reasons=("strict refusal",),
                refusal_category=invalid,
            )

    def test_refusal_error_rejects_non_enum_category(self) -> None:
        invalid = cast(NativeRefusalCategory, "bogus")
        with pytest.raises(TypeError, match="NativeRefusalCategory"):
            NativeRefusalError(("strict refusal",), invalid)

    def test_refusal_error_constructor_remains_strict_by_default(self) -> None:
        error = NativeRefusalError(("one", "two"))
        assert error.reasons == ("one", "two")
        assert str(error) == "one; two"
        assert error.category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY


# --- probe_native --------------------------------------------------------


class TestProbeNative:
    def test_combined_dev_checkpoint_is_native(self) -> None:
        capability = probe_native(combined_dev_checkpoint())
        assert capability == NativeCapability(family_id="dinkster.flux_dev", native=True)

    def test_split_sources_are_native(self) -> None:
        capability = probe_native(**split_sources())
        assert capability.native and capability.family_id == "dinkster.flux_dev"

    def test_schnell_detects_as_schnell(self) -> None:
        sources = split_sources()
        sources["diffusion"] = source(schnell_geometries(), "schnell.safetensors")
        capability = probe_native(**sources)
        assert capability.native and capability.family_id == "dinkster.flux_schnell"

    def test_diffusion_source_outranks_checkpoint_for_detection(self) -> None:
        """A split diffusion file overrides the combined checkpoint's
        DiT (plan semantics), so detection must follow it too."""
        capability = probe_native(
            combined_dev_checkpoint(),
            diffusion=source(schnell_geometries(), "schnell.safetensors"),
        )
        assert capability.native and capability.family_id == "dinkster.flux_schnell"

    def test_sd15_combined_checkpoint_is_native(self) -> None:
        capability = probe_native(sd15_checkpoint())
        assert capability == NativeCapability(family_id="dinkster.sd15", native=True)

    def test_sdxl_combined_checkpoint_is_native(self) -> None:
        capability = probe_native(sdxl_checkpoint())
        assert capability == NativeCapability(family_id="dinkster.sdxl", native=True)

    def test_refiner_combined_checkpoint_is_native(self) -> None:
        capability = probe_native(refiner_checkpoint())
        assert capability == NativeCapability(family_id="dinkster.sdxl_refiner", native=True)

    def test_flux_refuses_a_split_clip_g_source(self) -> None:
        capability = probe_native(
            combined_dev_checkpoint(),
            clip_g=source(sd_clip_g_geometries(), "clip_g.safetensors"),
        )
        assert not capability.native
        assert capability.family_id == "dinkster.flux_dev"
        assert capability.refusal_category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY
        (reason,) = capability.reasons
        assert "clip_g" in reason.lower() or "CLIP-G" in reason

    def test_sd_refuses_a_split_t5xxl_source(self) -> None:
        capability = probe_native(
            sd15_checkpoint(),
            t5xxl=source(geometrize(t5_layout(T5_XXL_CONFIG)), "t5.safetensors"),
        )
        assert not capability.native
        assert capability.family_id == "dinkster.sd15"
        (reason,) = capability.reasons
        assert "t5xxl" in reason

    def test_flux_refuses_a_split_qwen3vl_4b_source(self) -> None:
        capability = probe_native(
            combined_dev_checkpoint(),
            qwen3vl_4b=source(
                {
                    key: TensorGeometry(shape, BFLOAT16)
                    for key, shape in krea2_text_layout().items()
                },
                "qwen3vl_4b_bf16.safetensors",
            ),
        )
        assert not capability.native
        assert capability.family_id == "dinkster.flux_dev"
        (reason,) = capability.reasons
        assert "qwen3vl_4b" in reason

    def test_unrecognized_family_label_uses_component_configuration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:

        @dataclass(frozen=True)
        class KeyDetector:
            family_id: str
            key: str

            def detect(self, source: WeightSource) -> DetectionEvidence | None:
                if self.key not in source.keys():
                    return None
                return DetectionEvidence(
                    family_id=self.family_id, matched_keys=(self.key,), fields={}
                )

        registry = FamilyRegistry()
        registry.register(
            ModelFamily(
                id="dinkster.someday",
                display_name="someday",
                detector=KeyDetector("dinkster.someday", FLUX_DIFFUSION_PREFIX + "img_in.weight"),
                specificity=10,
                latent=LatentDescriptor(channels=4),
                sampling=SamplingDescriptor(Parameterization.EPS, sigma_min=0.03, sigma_max=14.6),
                wiring=ComponentWiring(),
                supported_dtypes=frozenset({FLOAT16, FLOAT32}),
            )
        )
        checkpoint = combined_dev_checkpoint()
        actual = plan_native(checkpoint, registry=registry)
        assert actual == plan_native(checkpoint)
        assert probe_native(checkpoint, registry=registry).native
        assert "dinkster.someday" in caplog.text
        assert "defaulting model, text, codec, sampling and dtype behavior" in caplog.text

    def test_unrecognized_source_is_refused_with_reason(self) -> None:
        capability = probe_native(
            source({"some.random.weight": TensorGeometry((3, 3), FLOAT16)}, "x.safetensors")
        )
        assert capability.family_id is None
        assert not capability.native
        assert "components do not match an executable architecture" in capability.reasons[0]
        assert capability.refusal_category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY

    def test_ambiguous_labels_do_not_override_unambiguous_geometry(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        @dataclass(frozen=True)
        class KeyDetector:
            family_id: str
            key: str

            def detect(self, source: WeightSource) -> DetectionEvidence | None:
                if self.key not in source.keys():
                    return None
                return DetectionEvidence(
                    family_id=self.family_id, matched_keys=(self.key,), fields={}
                )

        def family(family_id: str) -> ModelFamily:
            return ModelFamily(
                id=family_id,
                display_name=family_id,
                detector=KeyDetector(family_id, FLUX_DIFFUSION_PREFIX + "img_in.weight"),
                specificity=10,
                latent=LatentDescriptor(channels=4),
                sampling=SamplingDescriptor(Parameterization.EPS, sigma_min=0.03, sigma_max=14.6),
                wiring=ComponentWiring(),
                supported_dtypes=frozenset({FLOAT16, FLOAT32}),
            )

        registry = FamilyRegistry()
        registry.register(family("dinkster.twin_a"))
        registry.register(family("dinkster.twin_b"))
        checkpoint = combined_dev_checkpoint()
        assert plan_native(checkpoint, registry=registry) == plan_native(checkpoint)
        assert "dinkster.twin_a" in caplog.text and "dinkster.twin_b" in caplog.text

    def test_missing_component_carries_the_planner_refusal(self) -> None:
        capability = probe_native(diffusion=source(dev_geometries(), "dit.safetensors"))
        assert not capability.native
        assert capability.family_id == "dinkster.flux_dev"
        assert capability.refusal_category is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY
        (reason,) = capability.reasons
        assert "clip_l" in reason

    def test_no_detectable_source_is_a_caller_bug(self) -> None:
        with pytest.raises(
            ValueError,
            match="diffusion or checkpoint source",
        ):
            probe_native(vae=source(kl_geometries(), "ae.safetensors"))


def test_complete_e5m2_plan_is_ineligible_only_for_requested_fp8_matmul() -> None:
    checkpoint = combined_dev_checkpoint()
    storage_key = next(
        key for key in checkpoint.geometries if key.startswith(FLUX_DIFFUSION_PREFIX)
    )
    geometry = checkpoint.geometries[storage_key]
    checkpoint.geometries[storage_key] = TensorGeometry(geometry.shape, FLOAT8_E5M2)

    plan = plan_native(checkpoint, fp8_matmul=False)
    from dinkster_inference.assembly import FluxAssemblyPlan

    assert isinstance(plan, FluxAssemblyPlan)
    assert plan.family.id == "dinkster.flux_dev"
    assert FLOAT8_E5M2 in plan.diffusion.dtypes.values()

    with pytest.raises(NativeRefusalError) as caught:
        plan_native(checkpoint, fp8_matmul=True)
    assert caught.value.category is NativeRefusalCategory.NATIVE_INELIGIBLE
    assert caught.value.reasons == ("fp8 matmul does not support float8_e5m2 checkpoint storage",)
    assert caught.value.__dict__["family_id"] == "dinkster.flux_dev"

    capability = probe_native(checkpoint, fp8_matmul=True)
    assert capability == NativeCapability(
        family_id="dinkster.flux_dev",
        native=False,
        reasons=("fp8 matmul does not support float8_e5m2 checkpoint storage",),
        refusal_category=NativeRefusalCategory.NATIVE_INELIGIBLE,
    )


@pytest.mark.parametrize(
    "error",
    (
        ImportError("planner dependency unavailable"),
        MemoryError("planner out of memory"),
        asyncio.CancelledError("planner cancelled"),
        RuntimeError("planner invariant failed"),
        AssertionError("planner assertion failed"),
    ),
)
@pytest.mark.parametrize("operation", (plan_native, probe_native))
def test_non_refusal_planner_failures_propagate_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    operation: Callable[..., object],
) -> None:
    def fail(**_sources: object) -> object:
        raise error

    monkeypatch.setattr(runtime_module, "plan_flux_assembly", fail)
    with pytest.raises(type(error)) as caught:
        operation(combined_dev_checkpoint())
    assert caught.value is error


# --- FamilyRuntime protocol ----------------------------------------------


@dataclass(frozen=True)
class FakeTensor:
    """RuntimeTensor satisfied by a plain value: a shape for the
    Conditioning payloads, arithmetic for the denoise drive."""

    value: float

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def _other(self, other: FakeTensor | float) -> float:
        return other.value if isinstance(other, FakeTensor) else other

    def __add__(self, other: FakeTensor | float) -> FakeTensor:
        return FakeTensor(self.value + self._other(other))

    def __sub__(self, other: FakeTensor | float) -> FakeTensor:
        return FakeTensor(self.value - self._other(other))

    def __mul__(self, other: FakeTensor | float) -> FakeTensor:
        return FakeTensor(self.value * self._other(other))


class StubRuntime:
    """A FakeTensor runtime; its assignability to the protocol is
    checked by pyright, its behavior is immaterial."""

    def __init__(self, family: ModelFamily) -> None:
        self._family = family

    @property
    def family(self) -> ModelFamily:
        return self._family

    @property
    def runtime_identity(self) -> str:
        return "native:stub:0000000000000000"

    def encode_text(self, text: str) -> Conditioning[FakeTensor]:
        return Conditioning(embeddings=FakeTensor(float(len(text))))

    def sample(
        self,
        latent: FakeTensor,
        *,
        cond: Conditioning[FakeTensor],
        cfg: SamplingGuidance[Conditioning[FakeTensor]] | None = None,
        sampler_id: str,
        scheduler_id: str,
        steps: int,
        denoise: float | None = None,
        seed: int = 0,
        guidance: float | None = None,
        segment: SamplingSegment | None = None,
        denoise_mask: FakeTensor | None = None,
        inpaint: InpaintConditioning[FakeTensor] | None = None,
        noise_inds: Sequence[int] | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
    ) -> FakeTensor:
        return latent

    def decode_latent(self, latent: FakeTensor) -> FakeTensor:
        return latent

    def encode_content(self, content: FakeTensor) -> FakeTensor:
        return content


def test_protocol_is_satisfiable_without_torch() -> None:
    from dinkster_inference import builtin_family_registry

    family = builtin_family_registry().get("dinkster.flux_dev")
    assert family is not None
    runtime: FamilyRuntime[FakeTensor] = StubRuntime(family)
    assert runtime.family.id == "dinkster.flux_dev"
    assert runtime.runtime_identity.startswith("native:")
    assert not isinstance(runtime, CustomSamplingRuntime)
    assert runtime.sample(
        FakeTensor(1.0),
        cond=runtime.encode_text("a cat"),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.normal",
        steps=1,
    ) == FakeTensor(1.0)


class StubConditioningRuntime:
    """A carrier-preparing runtime; its assignability to the protocol is
    checked by pyright, its behavior is immaterial."""

    @property
    def runtime_identity(self) -> str:
        return "native:stub:0000000000000000"

    @property
    def conditioning_identity(self) -> str:
        return "native:stub:conditioning"

    def prepare_conditioning(self, carrier: ConditioningCarrier) -> object:
        return carrier


class StubSingleStreamConditioningRuntime:
    @property
    def runtime_identity(self) -> str:
        return "native:stub:0000000000000000"

    @property
    def conditioning_identity(self) -> str:
        return "native:stub:single-conditioning"

    def prepare_single_stream_conditioning(
        self, carrier: ConditioningCarrier
    ) -> Conditioning[FakeTensor]:
        del carrier
        return Conditioning(FakeTensor(1.0))


def test_conditioning_preparation_protocol_is_structural() -> None:
    from dinkster_inference import builtin_family_registry

    runtime: MultiStreamConditioningRuntime = StubConditioningRuntime()
    assert isinstance(runtime, MultiStreamConditioningRuntime)
    assert not isinstance(runtime, ConditioningRuntime)
    assert runtime.conditioning_identity == "native:stub:conditioning"

    family = builtin_family_registry().get("dinkster.flux_dev")
    assert family is not None
    assert not isinstance(StubRuntime(family), MultiStreamConditioningRuntime)


def test_single_stream_conditioning_preparation_protocol_is_structural() -> None:
    from dinkster_inference import builtin_family_registry

    runtime: ConditioningRuntime = StubSingleStreamConditioningRuntime()
    assert isinstance(runtime, ConditioningRuntime)
    assert not isinstance(runtime, MultiStreamConditioningRuntime)
    assert runtime.conditioning_identity == "native:stub:single-conditioning"

    family = builtin_family_registry().get("dinkster.flux_dev")
    assert family is not None
    assert not isinstance(StubRuntime(family), ConditioningRuntime)
