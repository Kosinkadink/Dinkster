"""A tiny registered architecture uses real loading, residency, and reconstruction."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import dinkster_inference as inference
import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_compat_comfy import native_arm
from dinkster_compat_comfy.native_residency import NativeRuntimeHandle
from dinkster_inference import (
    FLOAT32,
    LUMINA2,
    ComponentPlan,
    Conditioning,
    CustomSamplingResult,
    FlowSigmas,
    ModelFamily,
    SigmaSpace,
    build_runtime_identity,
    component_catalog,
    load_safetensors_header,
)
from dinkster_inference.assembly import _component_source  # pyright: ignore[reportPrivateUsage]
from dinkster_inference.component_checkpoint import ComponentCheckpointPlan
from dinkster_inference.component_registry import (
    ComponentDescriptor,
    ComponentRegistry,
)
from dinkster_inference.weights import AssetIdentifiedSource, WeightSource
from dinkster_inference_torch import CastOperations, ResidentWeights
from dinkster_inference_torch.checkpoint_runtime import ComponentAssembly
from dinkster_inference_torch.sampling_runtime import SingleStreamSamplingRuntime
from dinkster_values import Value, ValueMeta
from dinkster_values.model import PyObjPayload
from dinkster_workers import ExecutionContext
from dinkster_workers.execution import use_execution_context
from safetensors import safe_open
from safetensors.torch import save_file

PREFIXES = {
    "diffusion": "model.diffusion_model.",
    "words": "text_encoders.words.transformer.",
    "pixels": "vae.",
    "guide": "text_embedding_projection.",
}


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "off")


@dataclass
class TinyAssembly:
    diffusion: torch.nn.Module
    _storage_dtype_follows_compute: bool = False

    def compute_dtype(self, role: str) -> torch.dtype | None:
        return torch.float32 if role == "diffusion" else None


class TinyRuntime(SingleStreamSamplingRuntime):
    def __init__(
        self,
        diffusion: torch.nn.Module,
        family: ModelFamily,
        *,
        runtime_identity: str,
        **options: Any,
    ) -> None:
        self.assembled = TinyAssembly(diffusion)
        self._family = family
        self.runtime_identity = runtime_identity
        self.options = options

    @property
    def family(self) -> ModelFamily:
        return self._family

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return FlowSigmas()

    def sample_custom(
        self, latent: torch.Tensor, *, noise: torch.Tensor, **options: Any
    ) -> CustomSamplingResult[torch.Tensor]:
        return CustomSamplingResult(latent + noise, None)


def descriptor() -> ComponentDescriptor:
    def detect(
        source: WeightSource, path: Path, *, bind_asset_identity: bool = True
    ) -> tuple[tuple[str, ComponentPlan[str]], ...]:
        identity_facts: tuple[str, ...] = ()
        if bind_asset_identity and not isinstance(source, AssetIdentifiedSource):
            return ()
        if bind_asset_identity and isinstance(source, AssetIdentifiedSource):
            identity_facts = (
                f"asset_digest={source.asset_digest}",
                f"asset_size={source.asset_size}",
            )
        plans: list[tuple[str, ComponentPlan[str]]] = []
        for role, prefix in PREFIXES.items():
            if not any(key.startswith(prefix) for key in source.keys()):
                continue
            extracted = _component_source(role, source, None, prefix)
            if extracted.ignored or set(extracted.geometries) != {"weight"}:
                continue
            if extracted.geometries["weight"].shape != (2, 2):
                continue
            plans.append(
                (
                    role,
                    ComponentPlan(
                        role,
                        path,
                        "tiny-linear",
                        extracted.source_keys,
                        {key: geometry.dtype for key, geometry in extracted.geometries.items()},
                        extracted.quant,
                        identity_facts=identity_facts,
                    ),
                )
            )
        return tuple(plans)

    return ComponentDescriptor(
        replace(LUMINA2, id="test.tiny", display_name="Tiny"),
        detect,
        tuple(PREFIXES),
        ("words",),
        ("pixels",),
        f"{__name__}:load",
        f"{__name__}:TinyRuntime",
        runtime_with_family=True,
        checkpoint_loader=f"{__name__}:load_tiny_checkpoint",
    )


def load_linear(plan: ComponentPlan[Any], compute_dtype: torch.dtype) -> torch.nn.Module:
    module = CastOperations(compute_dtype).linear(2, 2, bias=False)
    with safe_open(plan.path, framework="pt", device="cpu") as weights:
        state = {key: weights.get_tensor(source_key) for key, source_key in plan.keys.items()}
    module.load_state_dict(state, strict=True, assign=True)
    return module


def realize_linear(
    plan: ComponentPlan[Any], *, compute_dtype: torch.dtype, **options: Any
) -> torch.nn.Module:
    assert set(options) == {"fp8_matmul", "attention_kernels"}
    assert options["fp8_matmul"] is False
    assert callable(options["attention_kernels"]["flux"])
    assert callable(options["attention_kernels"]["qwen"])
    return load_linear(plan, compute_dtype)


def text_factory(assembled: ComponentAssembly) -> SimpleNamespace | None:
    modules = assembled.components
    if "words" not in modules:
        return None

    def encode_text(text: str) -> Conditioning[torch.Tensor]:
        return Conditioning(modules["words"](torch.ones(1, 2)).unsqueeze(1), None)

    return SimpleNamespace(encode_text=encode_text)


def codec_factory(assembled: ComponentAssembly) -> SimpleNamespace | None:
    modules = assembled.components
    codec = modules.get("pixels")
    return None if codec is None else SimpleNamespace(encode=codec, decode=codec)


def load(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: str,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> SimpleNamespace:
    registration = descriptor()
    source = load_safetensors_header(path, asset_digest=asset.digest, asset_size=asset.size)
    plan = dict(registration.detector(source, path))[expected_role]
    assert (
        registration.component_identity(
            expected_role, plan, str(compute_dtype).removeprefix("torch.")
        )
        == expected_identity
    )
    return SimpleNamespace(role=expected_role, plan=plan, module=load_linear(plan, compute_dtype))


@dataclass(frozen=True)
class Resolver:
    path: Path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def select_native_context(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    input_name: str,
    asset: AssetRef,
    *,
    component_type: str | None = None,
) -> tuple[Any, ExecutionContext]:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "src"))
    native_policy = importlib.import_module("dinkster.native_policy")
    inputs = {
        input_name: Value(
            type_id="dinkster.asset",
            fingerprint=asset.digest,
            meta=ValueMeta({"digest": asset.digest, "name": asset.name}),
            payload=PyObjPayload(asset),
        )
    }
    if component_type is not None:
        inputs["type"] = Value(
            type_id="dinkster.string",
            fingerprint=component_type,
            meta=ValueMeta(),
            payload=PyObjPayload(component_type),
        )

    def ignore_diagnostic(_diagnostic: object) -> None:
        pass

    def locate(digest: str) -> Path | None:
        return asset.local_path() if digest == asset.digest else None

    policy = native_policy.NativeDispatchPolicy(
        locate,
        ignore_diagnostic,
        dtype_policy=lambda: {
            "diffusion": "float32",
            "textEncoder": "float32",
            "vae": "float32",
        },
    )
    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    selection = asyncio.run(
        policy.select(
            node_type,
            inputs,
            ("compat", {"compat": "compat-tag", "compat@native": "native-default"}),
        )
    )
    assert selection is not None and selection.target == "compat@native"
    return selection, ExecutionContext(
        selection.target,
        selection.cache_tag,
        diffusion_dtype=selection.diffusion_dtype,
        text_dtype=selection.text_dtype,
        vae_dtype=selection.vae_dtype,
        attention_policy=selection.attention_policy,
        attention_route_token=selection.attention_route_token,
    )


@dataclass(frozen=True)
class TinyCheckpointAssembly:
    components: Mapping[str, torch.nn.Module]
    _storage_dtype_follows_compute: bool = True

    def compute_dtype(self, component: str) -> torch.dtype:
        assert component in self.components
        return torch.float32


def load_tiny_checkpoint(plan: ComponentCheckpointPlan, **options: Any) -> SimpleNamespace:
    assert options["diffusion_dtype"] == options["text_dtype"] == options["vae_dtype"]
    assert options["diffusion_dtype"] == torch.float32
    assert options["fp8_matmul"] is False
    identity = build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=FLOAT32,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    )
    modules = {role: load_linear(part, torch.float32) for role, part in plan.components.items()}
    return SimpleNamespace(
        family=plan.family,
        runtime_identity=identity,
        assembled=TinyCheckpointAssembly(MappingProxyType(modules)),
    )


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize(
    "kind,role", [("model", "diffusion"), ("text", "words"), ("codec", "pixels")]
)
def test_new_architecture_loads_real_split_and_combined_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    combined: bool,
    kind: str,
    role: str,
) -> None:
    registration = descriptor()
    registry = ComponentRegistry()
    for builtin in component_catalog.default_component_registry():
        registry.register(builtin)
    registry.register(registration)
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    path = tmp_path / "weights.safetensors"
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    roles = registration.roles if combined else (role,)
    save_file({f"{PREFIXES[item]}weight": weight.clone() for item in roles}, path)
    asset = AssetRef(
        digest=digest_file(path), name=path.name, size=path.stat().st_size, resolver=Resolver(path)
    )
    source = load_safetensors_header(path, asset_digest=asset.digest, asset_size=asset.size)
    _selected, selected_role, plan = registry.select(source, path, kind)
    identity = registration.component_identity(selected_role, plan, "float32")
    input_name = {"model": "diffusion_model", "text": "text_encoder", "codec": "vae"}[kind]
    node_type = {
        "model": "dinkster.load_diffusion_model",
        "text": "dinkster.load_clip",
        "codec": "dinkster.load_vae",
    }[kind]
    selection, context = select_native_context(
        monkeypatch,
        node_type,
        input_name,
        asset,
        component_type="tiny" if kind == "text" else None,
    )
    assert selection.cache_tag == identity
    arm: Any = native_arm
    with use_execution_context(context):
        handle = (
            arm.NativeLoadClip.execute(text_encoder=asset, type="tiny", device="cpu")["clip"]
            if kind == "text"
            else arm.NativeLoadDiffusionModel.execute(
                diffusion_model=asset, weight_dtype="default"
            )["model"]
            if kind == "model"
            else arm.NativeLoadVae.execute(vae=asset)["vae"]
        )
    for rebuild in (False, True):
        current = handle.rebuild() if rebuild else handle
        try:
            assert current.recipe.runtime_identity == identity
            stage = (
                current.stage("diffusion")
                if isinstance(current, NativeRuntimeHandle)
                else current.stage()
            )
            with stage, torch.inference_mode():
                module = (
                    current.runtime.assembled.diffusion
                    if isinstance(current, NativeRuntimeHandle)
                    else current.component
                )
                assert module.weight.dtype == torch.bfloat16
                assert module.weight.device == torch.device("cpu")
                output = module(torch.tensor([[1.0, 1.0]], dtype=torch.float32))
                assert output.dtype == torch.float32
                torch.testing.assert_close(output, torch.tensor([[3.0, 7.0]]), rtol=0, atol=0)
        finally:
            current.terminal_release()


@pytest.mark.parametrize("dynamic_factory", [False, True])
@pytest.mark.parametrize("patch_mode", [None, "clone", "precalculate"])
def test_triposplat_model_loader_enrolls_and_rebuilds_declared_dit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dynamic_factory: bool, patch_mode: str | None
) -> None:
    import dinkster_inference_torch

    original = component_catalog.default_component_registry().get("dinkster.triposplat")
    assert original is not None
    path = tmp_path / "dit.safetensors"
    save_file({"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)}, path)
    asset = AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=Resolver(path))
    plan = ComponentPlan(
        "dit", path, "tiny-linear", {"weight": "weight"}, {"weight": inference.BFLOAT16}, {}
    )

    def detect(*_args: Any, **_kwargs: Any) -> tuple[tuple[str, ComponentPlan[str]], ...]:
        return (("dit", plan),)

    registration = replace(original, detector=detect)
    registry = ComponentRegistry()
    registry.register(registration)
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "off")
    if dynamic_factory:

        def factory(*_args: Any) -> tuple[type[ResidentWeights], None]:
            return ResidentWeights, None

        monkeypatch.setattr(native_arm, "_aimdo_mechanism_factory", factory)

    def load_dit(_path: Path, **options: Any) -> SimpleNamespace:
        assert options["asset"] == asset
        assert options["expected_role"] == "dit"
        return SimpleNamespace(
            role="dit",
            family_id=original.id,
            plan=plan,
            module=load_linear(plan, options["compute_dtype"]),
        )

    monkeypatch.setattr(dinkster_inference_torch, "load_triposplat_component", load_dit)
    identity = registration.component_identity("dit", plan, "float32")
    context = ExecutionContext(
        "native", identity, diffusion_dtype="float32", text_dtype="float32", vae_dtype="float32"
    )
    arm: Any = native_arm
    with use_execution_context(context):
        handle = arm.NativeLoadDiffusionModel.execute(
            diffusion_model=asset, weight_dtype="default"
        )["model"]
    base = handle
    if patch_mode is not None:
        patch_path = tmp_path / "delta.safetensors"
        save_file({"diffusion_model.diff": torch.ones((2, 2))}, patch_path)
        patch_asset = AssetRef(
            digest_file(patch_path),
            patch_path.name,
            patch_path.stat().st_size,
            resolver=Resolver(patch_path),
        )
        if patch_mode == "clone":
            overlay = arm._native_lora_overlay(handle, patch_asset, 1.0, 0.0)
            assert tuple(patch.component for patch in overlay.patches) == ("dit",)
            handle = handle.clone(
                (overlay,), source_resolvers={patch_asset.digest: patch_asset.resolver}
            )
        else:
            handle = arm.NativeLoadLoraModelOnly.execute(
                model=handle,
                lora=patch_asset,
                strength_model=1.0,
                execution_mode=patch_mode,
            )["model"]
        identity = handle.recipe.runtime_identity
        assert handle.recipe.overlays[0].strength_clip == "0"
    for rebuild in (False, True):
        current = handle.rebuild() if rebuild else handle
        try:
            runtime = current.runtime
            assert isinstance(runtime, dinkster_inference_torch.TripoSplatDiffusionRuntime)
            assert tuple(runtime.assembled.components) == ("dit",)
            assert runtime.assembled.components["dit"] is runtime.model
            assert runtime.assembled.diffusion is runtime.model
            assert runtime.assembled.compute_dtype("dit") is torch.float32
            assert runtime.assembled.compute_dtype("diffusion") is torch.float32
            assert runtime.assembled.compute_dtype("text") is None
            assert arm._lora_target_routes(runtime.assembled) == (("diffusion_model.", "dit"),)
            assert current.recipe.runtime_identity == identity
            with current.stage("diffusion"), torch.inference_mode():
                module = runtime.assembled.components["dit"]
                assert module.weight.dtype is torch.bfloat16
                assert module.weight.device.type == "cpu"
                output = module(torch.ones((1, 2)))
                expected = torch.tensor([[3.0, 7.0]]) + (2 if patch_mode is not None else 0)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
        finally:
            current.terminal_release()
    if base is not handle:
        base.terminal_release()


@pytest.mark.parametrize("generic_constructor", [False, True])
@pytest.mark.parametrize("dynamic_factory", [False, True])
@pytest.mark.parametrize(
    "roles",
    [
        ("diffusion",),
        ("diffusion", "words"),
        ("diffusion", "pixels"),
        ("diffusion", "words", "pixels", "guide"),
    ],
)
def test_descriptor_backed_assembly_contract_loads_and_rebuilds_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    roles: tuple[str, ...],
    dynamic_factory: bool,
    generic_constructor: bool,
) -> None:
    registration = descriptor()
    if generic_constructor:
        registration = replace(
            registration,
            checkpoint_loader="dinkster_inference_torch.checkpoint_runtime:assemble_component_checkpoint",
            component_realizer=f"{__name__}:realize_linear",
            checkpoint_text_factory=f"{__name__}:text_factory",
            checkpoint_codec_factory=f"{__name__}:codec_factory",
        )
    if dynamic_factory:

        def factory(*_args: Any) -> tuple[type[ResidentWeights], None]:
            return ResidentWeights, None

        monkeypatch.setattr(native_arm, "_aimdo_mechanism_factory", factory)
    monkeypatch.setattr(native_arm, "_freeze_embedding_resource", lambda: (None, None))
    path = tmp_path / "checkpoint.safetensors"
    weights = {
        role: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16) * (index + 1)
        for index, role in enumerate(roles)
    }
    save_file({f"{PREFIXES[role]}weight": weight for role, weight in weights.items()}, path)
    asset = AssetRef(
        digest=digest_file(path), name=path.name, size=path.stat().st_size, resolver=Resolver(path)
    )
    registry = ComponentRegistry()
    for builtin in component_catalog.default_component_registry():
        registry.register(builtin)
    registry.register(registration)
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    source = load_safetensors_header(path, asset_digest=asset.digest, asset_size=asset.size)
    plan = inference.plan_native(checkpoint=source)
    assert plan.family.id == registration.id
    assert {
        part.component: dict(part.keys) for part in plan.identity_components if part is not None
    } == {role: {"weight": f"{PREFIXES[role]}weight"} for role in roles}
    identity = build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=FLOAT32,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    )
    selection, context = select_native_context(
        monkeypatch,
        "dinkster.load_checkpoint",
        "checkpoint",
        asset,
    )
    assert selection.cache_tag == identity
    arm: Any = native_arm
    with use_execution_context(context):
        outputs = arm.NativeLoadCheckpoint.execute(checkpoint=asset)
    handle = outputs["model"]
    try:
        assert outputs["clip"] is outputs["vae"] is handle
        assert handle.recipe.runtime_identity == identity
        assert handle.recipe.sources[0].source.digest == asset.digest
        for rebuild in (False, True):
            current = handle.rebuild() if rebuild else handle
            try:
                assert current.recipe == handle.recipe
                assert tuple(current.runtime.assembled.components) == roles
                assert len(current.mechanisms) == len(roles)
                if dynamic_factory:
                    assert current.residency_route.dynamic_components == roles
                for role in roles:
                    stage = {"words": "text", "pixels": "vae"}.get(role, role)
                    with current.stage(stage), torch.inference_mode():
                        module = current.runtime.assembled.components[role]
                        assert module.weight.dtype == torch.bfloat16
                        assert module.weight.device == torch.device("cpu")
                        actual = module(torch.ones(1, 2, dtype=torch.float32))
                        torch.testing.assert_close(
                            actual, weights[role].float().sum(dim=1).unsqueeze(0), rtol=0, atol=0
                        )
                if generic_constructor:
                    if "words" in roles:
                        with current.stage("text"), torch.inference_mode():
                            encoded = current.runtime.encode_text("hello")
                            torch.testing.assert_close(
                                encoded.embeddings,
                                weights["words"].float().sum(dim=1).reshape(1, 1, 2),
                                rtol=0,
                                atol=0,
                            )
                    if "pixels" in roles:
                        with current.stage("vae"), torch.inference_mode():
                            decoded = current.runtime.decode_latent(torch.ones(1, 2))
                            torch.testing.assert_close(
                                decoded,
                                weights["pixels"].float().sum(dim=1).reshape(1, 2),
                                rtol=0,
                                atol=0,
                            )
            finally:
                if current is not handle:
                    current.terminal_release()

        def changed_detection(
            source: WeightSource, path: Path, *, bind_asset_identity: bool = True
        ) -> tuple[tuple[str, ComponentPlan[str]], ...]:
            return tuple(
                (role, replace(part, config="different-operator"))
                for role, part in registration.detector(
                    source, path, bind_asset_identity=bind_asset_identity
                )
            )

        changed_registry = ComponentRegistry()
        for current_descriptor in registry:
            changed_registry.register(
                replace(registration, detector=changed_detection)
                if current_descriptor is registration
                else current_descriptor
            )
        monkeypatch.setattr(
            component_catalog, "default_component_registry", lambda: changed_registry
        )
        with pytest.raises(RuntimeError, match="no longer match the reconstruction recipe"):
            handle.rebuild()
        handle.require_active()
    finally:
        handle.terminal_release()


@pytest.mark.parametrize(
    "kind,reverse,combined",
    [
        ("stable_diffusion", False, False),
        ("sdxl", False, False),
        ("sdxl", True, False),
        ("sdxl", False, True),
        ("flux", False, False),
        ("flux", True, False),
        ("flux", False, True),
    ],
)
def test_text_recipe_handle_loads_encodes_and_rebuilds_ordered_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, reverse: bool, combined: bool
) -> None:
    import gc
    import weakref

    from dinkster_inference.clip_text import CLIP_L_TEXT_CONFIG
    from dinkster_inference.t5_text import T5_XXL_CONFIG
    from dinkster_inference.text_recipes import resolve_text_recipe
    from dinkster_inference_torch import materialize_basic_conditioning
    from dinkster_inference_torch.clip_text import ClipTextModel
    from dinkster_inference_torch.t5_text import T5TextModel

    config = replace(
        CLIP_L_TEXT_CONFIG,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
    )
    roles = (
        ("clip_l",)
        if kind == "stable_diffusion"
        else ("clip_l", "t5xxl")
        if kind == "flux"
        else ("clip_l", "clip_g")
    )
    models = {
        role: (
            T5TextModel(
                replace(
                    T5_XXL_CONFIG,
                    d_model=8,
                    d_ff=16,
                    d_kv=4,
                    num_heads=2,
                    num_layers=2,
                )
            )
            if role == "t5xxl"
            else ClipTextModel(config)
        )
        for role in roles
    }
    generator = torch.Generator().manual_seed(1278)
    for model in models.values():
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.03)

    def detect(source: WeightSource, path: Path) -> tuple[tuple[str, ComponentPlan[Any]], ...]:
        assert isinstance(source, AssetIdentifiedSource)
        return tuple(
            (
                role,
                ComponentPlan(
                    role,
                    path,
                    model.config,
                    {key: f"{role}.{key}" for key in model.state_dict()},
                    {key: FLOAT32 for key in model.state_dict()},
                    {},
                    identity_facts=(
                        f"asset_digest={source.asset_digest}",
                        f"asset_size={source.asset_size}",
                    ),
                ),
            )
            for role, model in models.items()
            if any(f"{role}.{key}" in source.keys() for key in model.state_dict())
        )

    registration = replace(
        descriptor(),
        detector=detect,
        roles=roles,
        text_encoder_roles=roles,
        codec_roles=(),
        requires_text_recipe=True,
    )
    registry = ComponentRegistry()
    registry.register(registration)
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    embedding_digest = "e" * 64

    def missing_embedding(_name: str) -> None:
        return None

    embedding_lookups = {role: missing_embedding for role in roles}
    freeze_calls: list[tuple[str, ...]] = []

    def freeze_embedding_resource(
        *, component_roles: tuple[str, ...] = ("clip_l", "clip_g", "t5xxl")
    ) -> tuple[SimpleNamespace, dict[str, Any]]:
        freeze_calls.append(component_roles)
        return SimpleNamespace(binding_digest=embedding_digest), embedding_lookups

    monkeypatch.setattr(native_arm, "_freeze_embedding_resource", freeze_embedding_resource)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "src"))
    native_policy = importlib.import_module("dinkster.native_policy")
    monkeypatch.setattr(native_policy, "_embedding_binding_digest", lambda: embedding_digest)
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "off")
    assets = []
    groups = (models,) if combined else tuple({role: model} for role, model in models.items())
    for index, group in enumerate(groups):
        path = tmp_path / f"encoder-{index}.safetensors"
        save_file(
            {
                f"{role}.{key}": value
                for role, model in group.items()
                for key, value in model.state_dict().items()
            },
            path,
        )
        assets.append(
            AssetRef(
                digest=digest_file(path),
                name=path.name,
                size=path.stat().st_size,
                resolver=Resolver(path),
            )
        )
    if reverse:
        assets.reverse()
    binding = resolve_text_recipe(
        tuple(
            registry.detect(
                load_safetensors_header(
                    asset.local_path(), asset_digest=asset.digest, asset_size=asset.size
                ),
                asset.local_path(),
            )
            for asset in assets
        ),
        kind,
    )
    arm: Any = native_arm
    recipe = binding.recipe(
        tuple(arm._weight_source_ref(inference, asset) for asset in assets),
        "float32",
        embedding_binding_digest=embedding_digest,
    )
    if len(assets) == 1:
        selection, context = select_native_context(
            monkeypatch,
            "dinkster.load_clip",
            "text_encoder",
            assets[0],
            component_type=kind,
        )
        assert selection.cache_tag == recipe.runtime_identity
    else:
        paths = {asset.digest: asset.local_path() for asset in assets}

        def ignore_diagnostic(_diagnostic: object) -> None:
            pass

        policy = native_policy.NativeDispatchPolicy(
            paths.get,
            ignore_diagnostic,
            dtype_policy=lambda: {
                "diffusion": "float32",
                "textEncoder": "float32",
                "vae": "float32",
            },
        )
        monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
        inputs = {
            f"text_encoder{index}": Value(
                type_id="dinkster.asset",
                fingerprint=asset.digest,
                meta=ValueMeta({"digest": asset.digest, "name": asset.name}),
                payload=PyObjPayload(asset),
            )
            for index, asset in enumerate(assets, 1)
        }
        inputs["type"] = Value(
            type_id="dinkster.string",
            fingerprint=kind,
            meta=ValueMeta(),
            payload=PyObjPayload(kind),
        )
        selection = asyncio.run(
            policy.select(
                "dinkster.load_dual_clip",
                inputs,
                ("compat", {"compat": "compat-tag", "compat@native": "native-default"}),
            )
        )
        assert selection is not None and selection.target == "compat@native"
        assert selection.cache_tag == recipe.runtime_identity
        context = ExecutionContext(
            selection.target,
            selection.cache_tag,
            diffusion_dtype=selection.diffusion_dtype,
            text_dtype=selection.text_dtype,
            vae_dtype=selection.vae_dtype,
            attention_policy=selection.attention_policy,
            attention_route_token=selection.attention_route_token,
        )
    with use_execution_context(context):
        handle = (
            arm.NativeLoadClip.execute(text_encoder=assets[0], type=kind, device="cpu")["clip"]
            if len(assets) == 1
            else arm.NativeLoadDualClip.execute(
                text_encoder1=assets[0],
                text_encoder2=assets[1],
                type=kind,
                device="cpu",
            )["clip"]
        )
    try:
        assert handle.recipe == recipe
        assert [item.source.digest for item in recipe.sources] == [asset.digest for asset in assets]
        with handle.stage(), torch.inference_mode():
            expected = handle.runtime.encode_text("a bird")
        output = arm.GenerationClipTextEncode.execute(text="a bird", clip=handle)
        carrier, component_binding = inference.split_component_conditioning(output["conditioning"])
        assert component_binding == inference.ComponentBinding(
            "text", binding.family_id, recipe.runtime_identity
        )
        encoded = materialize_basic_conditioning(carrier, device="cpu")
        assert torch.equal(encoded.embeddings, expected.embeddings)
        assert encoded.pooled is not None
        assert torch.equal(encoded.pooled, expected.pooled)
        rebuilt = handle.rebuild()
        try:
            assert rebuilt.recipe == recipe
            assert rebuilt.recipe.knobs.embedding_binding_digest == embedding_digest
            for role, model in models.items():
                for key, value in model.state_dict().items():
                    assert torch.equal(rebuilt.module[role].state_dict()[key], value)
            with rebuilt.stage(), torch.inference_mode():
                actual = rebuilt.runtime.encode_text("a bird")
            assert torch.equal(actual.embeddings, expected.embeddings)
            assert torch.equal(actual.pooled, expected.pooled)
            runtime_ref = weakref.ref(rebuilt.runtime)
        finally:
            rebuilt.terminal_release()
        gc.collect()
        assert runtime_ref() is None
        patch_path = tmp_path / "text-patch.safetensors"
        delta = torch.full((16,), 0.125)
        save_file({"delta": delta}, patch_path)
        patch_asset = AssetRef(
            digest=digest_file(patch_path),
            name=patch_path.name,
            size=patch_path.stat().st_size,
            resolver=Resolver(patch_path),
        )
        overlay = inference.PatchOverlay.from_decoded(
            source=arm._weight_source_ref(inference, patch_asset),
            dialect="none",
            key_map="native.dinkster.test.v1",
            strength_model=0.0,
            strength_clip=1.0,
            patches=(
                inference.OverlayPatch(
                    "clip_l",
                    inference.PatchTarget("text_model.encoder.layers.0.mlp.fc1.bias"),
                    inference.DiffPatchRef("delta"),
                ),
            ),
        )
        patched = handle.clone(
            (overlay,), source_resolvers={patch_asset.digest: patch_asset.resolver}
        )
        try:
            assert patched.resource_identity != handle.resource_identity
            assert patched.recipe.knobs.embedding_binding_digest == embedding_digest
            with patched.stage(), torch.inference_mode():
                linear = patched.module["clip_l"].text_model.encoder.layers[0].mlp.fc1
                actual_bias = linear(torch.zeros(1, 8))
            expected_bias = (
                models["clip_l"].state_dict()["text_model.encoder.layers.0.mlp.fc1.bias"] + delta
            )
            torch.testing.assert_close(actual_bias, expected_bias.unsqueeze(0), rtol=0, atol=0)
        finally:
            patched.terminal_release()
        assert freeze_calls == [tuple(part.role for part in binding.components)]
        injected = arm.build_text_recipe_handle(
            tuple(assets),
            kind,
            recipe.runtime_identity,
            compute_dtype="float32",
            load_device="cpu",
            embedding_resource=(
                SimpleNamespace(binding_digest=embedding_digest),
                embedding_lookups,
            ),
        )
        try:
            assert injected.recipe == recipe
            assert freeze_calls == [tuple(part.role for part in binding.components)]
        finally:
            injected.terminal_release()
        with pytest.raises(RuntimeError, match="identity differs from dispatch"):
            arm.build_text_recipe_handle(
                tuple(assets), kind, "wrong-identity", compute_dtype="float32", load_device="cpu"
            )
        if len(assets) == 2:
            with (
                use_execution_context(
                    replace(context, expected_execution_identity="wrong-identity")
                ),
                pytest.raises(RuntimeError, match="identity differs from dispatch"),
            ):
                arm.NativeLoadDualClip.execute(
                    text_encoder1=assets[0],
                    text_encoder2=assets[1],
                    type=kind,
                    device="cpu",
                )
        with pytest.raises(RuntimeError, match="identity differs from dispatch"):
            arm.build_text_recipe_handle(
                tuple(assets),
                kind,
                recipe.runtime_identity,
                compute_dtype="float32",
                load_device="cpu",
                embedding_resource=(SimpleNamespace(binding_digest="1" * 64), None),
            )
        with (
            use_execution_context(context),
            pytest.raises(
                ValueError,
                match="unknown text recipe.*detected.*test.tiny/(clip_l|clip_g|t5xxl)",
            ),
        ):
            arm.NativeLoadClip.execute(text_encoder=assets[0], type="unknown", device="cpu")
        if len(assets) == 2:
            with (
                use_execution_context(context),
                pytest.raises(
                    ValueError,
                    match="unknown text recipe.*detected.*test.tiny/(clip_l|clip_g|t5xxl)",
                ),
            ):
                arm.NativeLoadDualClip.execute(
                    text_encoder1=assets[0],
                    text_encoder2=assets[1],
                    type="unknown",
                    device="cpu",
                )
        handle.require_active()
    finally:
        handle.terminal_release()
