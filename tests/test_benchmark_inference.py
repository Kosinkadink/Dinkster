"""CPU-testable contracts for the Dinkster benchmark command."""

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from blake3 import blake3
from dinkster_inference import (
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    PayloadBinding,
    PayloadDescriptor,
    PayloadReference,
    make_conditioning_carrier,
)
from PIL import Image

from tools.evidence_paths import EVIDENCE_ROOT

_MODULE_PATH = EVIDENCE_ROOT / "scripts" / "benchmark_inference.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_inference", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_inference = importlib.util.module_from_spec(_SPEC)
_PREVIOUS_TORCH = sys.modules.get("torch")
sys.modules["benchmark_inference"] = benchmark_inference
sys.modules["torch"] = ModuleType("torch")
try:
    _SPEC.loader.exec_module(benchmark_inference)
finally:
    if _PREVIOUS_TORCH is None:
        del sys.modules["torch"]
    else:
        sys.modules["torch"] = _PREVIOUS_TORCH


class _OffSampler:
    """Stand-in for the shared-usage sampler: resolved off, never reads a
    platform counter."""

    def __init__(self, spill_scope: str) -> None:
        del spill_scope
        self.scope = "off"

    def sample(self) -> int | None:
        return None


def test_cuda_device_identity_uses_compute_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        gcnArchName="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        major=12,
        minor=0,
        total_memory=102_642_761_728,
    )
    monkeypatch.setattr(
        benchmark_inference.torch,
        "cuda",
        SimpleNamespace(
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
        ),
        raising=False,
    )

    assert benchmark_inference._device_entries("cuda") == [
        {
            "index": 0,
            "name": properties.name,
            "architecture": "sm_120",
            "total_memory": properties.total_memory,
        }
    ]


def infinitetalk_arguments(tmp_path: Path) -> list[str]:
    arguments = ["--backend", "cuda", "--family", "wan21_infinitetalk"]
    for option in (
        "diffusion",
        "text-encoder",
        "vae",
        "lora",
        "model-patch",
        "audio-encoder",
        "clip-vision",
        "input-image",
        "input-audio-1",
        "input-audio-2",
    ):
        arguments.extend((f"--{option}", str(tmp_path / f"{option}.bin")))
    return arguments


def humo_arguments(tmp_path: Path) -> list[str]:
    arguments = ["--backend", "cuda", "--family", "wan21_humo"]
    for option in (
        "diffusion",
        "text-encoder",
        "vae",
        "lora",
        "audio-encoder",
        "input-image",
        "input-audio",
    ):
        arguments.extend((f"--{option}", str(tmp_path / f"{option}.bin")))
    return arguments


def anima_arguments(tmp_path: Path) -> list[str]:
    arguments = ["--backend", "cuda", "--family", "anima"]
    for option in ("diffusion", "text-encoder", "vae"):
        arguments.extend((f"--{option}", str(tmp_path / f"{option}.safetensors")))
    return arguments


def minimax_h3_arguments(tmp_path: Path) -> list[str]:
    arguments = ["--backend", "cuda", "--family", "minimax_h3"]
    for option in ("diffusion", "text-encoder", "vae", "audio-vae"):
        arguments.extend((f"--{option}", str(tmp_path / f"{option}.bin")))
    return arguments


def flux_arguments(tmp_path: Path) -> list[str]:
    arguments = ["--backend", "cuda", "--family", "flux"]
    for option in ("diffusion", "clip-l", "text-encoder", "vae"):
        arguments.extend((f"--{option}", str(tmp_path / f"{option}.safetensors")))
    return arguments


def chroma_arguments(tmp_path: Path) -> list[str]:
    arguments = ["--backend", "cuda", "--family", "chroma"]
    for option in ("diffusion", "text-encoder", "vae"):
        arguments.extend((f"--{option}", str(tmp_path / f"{option}.safetensors")))
    return arguments


def test_placement_cli_defaults_to_residency_and_labels_direct_as_diagnostic(
    tmp_path: Path,
) -> None:
    arguments = [
        "--backend",
        "cuda",
        "--family",
        "sd15",
        "--checkpoint",
        str(tmp_path / "checkpoint.safetensors"),
    ]

    production = benchmark_inference._parse_arguments(arguments)
    diagnostic = benchmark_inference._parse_arguments(
        [*arguments, "--placement", "direct-diagnostic"]
    )

    assert production.placement == "residency"
    assert diagnostic.placement == "direct-diagnostic"


@pytest.mark.parametrize("family", ["wan21_infinitetalk", "wan21_humo"])
def test_provider_workloads_require_their_existing_residency_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], family: str
) -> None:
    arguments_list = (
        infinitetalk_arguments(tmp_path)
        if family == "wan21_infinitetalk"
        else humo_arguments(tmp_path)
    )
    arguments = benchmark_inference._parse_arguments(arguments_list)

    assert arguments.placement == "residency"
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments([*arguments_list, "--placement", "direct-diagnostic"])
    assert "not supported for provider workloads" in capsys.readouterr().err


def test_anima_cli_defaults_to_the_pinned_primary_workload(tmp_path: Path) -> None:
    arguments = benchmark_inference._parse_arguments(anima_arguments(tmp_path))

    assert arguments.prompt == benchmark_inference.BENCHMARK_ANIMA_PROMPT
    assert arguments.negative_prompt == ""
    assert arguments.seed == 875817230929465
    assert arguments.steps == 30
    assert (arguments.width, arguments.height) == (1024, 1024)
    assert arguments.cfg == 4.0
    assert arguments.sampler == "dinkster.er_sde"
    assert arguments.scheduler == "dinkster.simple"
    assert arguments.warm_runs == 5
    assert arguments.mode == "eager"
    assert arguments.placement == "residency"
    assert arguments.fallback_768 is False


def test_minimax_h3_requires_production_residency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = minimax_h3_arguments(tmp_path)
    assert benchmark_inference._parse_arguments(arguments).placement == "residency"
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments([*arguments, "--placement", "direct-diagnostic"])
    assert "not supported for MiniMax H3" in capsys.readouterr().err


def test_benchmark_process_defaults_to_the_serving_aimdo_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_memory
    from dinkster_workers import aimdo_bootstrap

    calls: list[tuple[bool, int | None]] = []
    monkeypatch.delenv("DINKSTER_AIMDO_ARM", raising=False)
    monkeypatch.setattr(
        aimdo_bootstrap,
        "bootstrap_aimdo",
        lambda enabled, *, simple_vram_headroom=None: (
            calls.append((enabled, simple_vram_headroom)) or (True, simple_vram_headroom)
        ),
    )

    assert benchmark_inference._bootstrap_production_residency() == ("auto", True)
    assert os.environ["DINKSTER_AIMDO_ARM"] == "auto"
    assert calls == [(True, dinkster_memory.AcceleratorMemoryPolicy().minimum_free_bytes)]


@pytest.fixture
def aimdo_route(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "auto")
    monkeypatch.setattr(benchmark_inference, "_AIMDO_BOOTSTRAP_SUCCEEDED", True)
    return SimpleNamespace(
        requested="auto",
        mechanism="aimdo",
        fallback_reason=None,
        dynamic_components=("component",),
        resident_components=(),
        fallback_components=(),
    )


def test_minimax_h3_cuda_rejects_eager_fallback_route(aimdo_route: SimpleNamespace) -> None:
    arguments = SimpleNamespace(attention_policy="auto", backend="cuda")
    access = SimpleNamespace(synchronize=lambda: None)
    run = benchmark_inference.BenchmarkRun(arguments, access)
    for name in ("model_handle", "clip", "vae", "audio_vae"):
        setattr(run, name, SimpleNamespace(residency_route=aimdo_route))
    handles = {
        "diffusion": run.model_handle,
        "conditioner": run.clip,
        "video_vae": run.vae,
        "audio_vae": run.audio_vae,
    }

    run._capture_required_residency("MiniMax H3", handles)
    assert set(run.residency_routes) == {"diffusion", "conditioner", "video_vae", "audio_vae"}

    run.clip.residency_route = SimpleNamespace(
        requested="auto",
        mechanism="eager",
        fallback_reason="aimdo device activation gate failed",
        dynamic_components=(),
        resident_components=(),
        fallback_components=(),
    )
    with pytest.raises(RuntimeError, match="aimdo residency did not own components: conditioner"):
        run._capture_required_residency("MiniMax H3", handles)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--mode", "compile"), "not supported"),
        (("--placement", "direct-diagnostic"), "not supported for anima"),
        (("--sampler", "dinkster.euler"), "for anima"),
        (("--seed", "0"), "for anima"),
        (("--width", "768"), "Anima geometry"),
    ],
)
def test_anima_cli_rejects_noncanonical_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, str],
    message: str,
) -> None:
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments([*anima_arguments(tmp_path), *extra])
    assert message.lower() in capsys.readouterr().err.lower()


def test_anima_cli_rejects_unlabeled_symmetric_fallback_geometry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments(
            [*anima_arguments(tmp_path), "--width", "768", "--height", "768"]
        )
    assert "requires --fallback-768" in capsys.readouterr().err


def test_anima_cli_accepts_the_explicit_fallback(tmp_path: Path) -> None:
    arguments = benchmark_inference._parse_arguments([*anima_arguments(tmp_path), "--fallback-768"])
    assert (arguments.width, arguments.height) == (768, 768)
    assert arguments.fallback_768 is True


def test_anima_cli_rejects_a_missing_required_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = anima_arguments(tmp_path)
    index = arguments.index("--vae")
    del arguments[index : index + 2]
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments(arguments)
    assert "requires --vae" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("family", "placement_arguments", "executed", "expected_variant"),
    [
        ("sd15", (), "direct_placement_diagnostic", None),
        ("sd15", ("--placement", "direct-diagnostic"), "production_residency", None),
        ("anima", (), "production_residency", "primary"),
        ("anima", ("--fallback-768",), "production_residency", "fallback_768"),
    ],
)
def test_entrypoint_records_executed_placement_and_anima_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    placement_arguments: tuple[str, ...],
    executed: str,
    expected_variant: str | None,
) -> None:
    if family == "anima":
        artifact_arguments = [
            option
            for role in ("diffusion", "text-encoder", "vae")
            for option in (f"--{role}", str(tmp_path / f"{role}.safetensors"))
        ]
    else:
        checkpoint = tmp_path / "checkpoint.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        artifact_arguments = ["--checkpoint", str(checkpoint)]
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_inference.py",
            "--backend",
            "cuda",
            "--family",
            family,
            *artifact_arguments,
            "--json",
            str(output),
            *placement_arguments,
        ],
    )
    expected_preflights = {
        role: benchmark_inference._AssetPreflight(
            path=tmp_path / f"{role.replace('_', '-')}.safetensors",
            digest="blake3:" + "a" * 64,
            size=1,
            verification=object(),
        )
        for role in ("diffusion", "text_encoder", "vae")
    }

    class Access:
        backend_runtime = "test-runtime"

        def synchronize(self) -> None:
            pass

        def reset_peak(self) -> None:
            pass

        def peak_allocated(self) -> int:
            return 1

        def peak_reserved(self) -> int:
            return 2

    class Run:
        def __init__(
            self,
            arguments: object,
            access: object,
            preflight_assets: Mapping[str, object],
        ) -> None:
            del arguments, access
            if family == "anima":
                assert preflight_assets == expected_preflights
            else:
                assert preflight_assets == {}
            self.checks: dict[str, dict[str, object]] = {}
            self.family_id = "dinkster.anima" if family == "anima" else "dinkster.sd15"
            self.executed_placement: str | None = None
            self.cold: dict[str, object] = {}
            self.residual_allocated = 0

        def record(self, name: str, action: Callable[[], str], *, always: bool = False) -> None:
            del always
            action()
            self.checks[name] = {"ok": True, "detail": "test"}

        def load(self) -> str:
            self.executed_placement = executed
            return "test"

        def encode_text(self) -> str:
            return "test"

        def cold_run(self) -> str:
            return "test"

        def finite_output(self) -> str:
            return "test"

        def warm_runs(self) -> str:
            return "test"

        def unload(self) -> str:
            return "test"

        def warm_section(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(benchmark_inference, "_admit_backend", lambda backend: Access())
    monkeypatch.setattr(benchmark_inference, "_driver_identity", lambda backend: "test-driver")
    monkeypatch.setattr(benchmark_inference, "_device_entries", lambda backend: [])
    monkeypatch.setattr(benchmark_inference, "_peak_rss_bytes", lambda: 3)
    monkeypatch.delenv("DINKSTER_AIMDO_ARM", raising=False)
    monkeypatch.setattr(benchmark_inference, "_ResidencySampler", _OffSampler)
    monkeypatch.setattr(
        benchmark_inference,
        "_artifact_entry",
        lambda role, path, pin=None, *, include_asset_preflight=False: {
            "role": role,
            "path": str(path),
            "sha256": "a" * 64,
            "bytes": 1,
            **({"_asset_preflight": expected_preflights[role]} if include_asset_preflight else {}),
        },
    )
    monkeypatch.setattr(benchmark_inference, "BenchmarkRun", Run)
    validation_modes: list[bool] = []
    monkeypatch.setattr(
        benchmark_inference,
        "validate_benchmark_report",
        lambda report, *, accelerator, canonical_evidence: (
            validation_modes.append(canonical_evidence) or ()
        ),
    )
    monkeypatch.setattr(benchmark_inference.torch, "__version__", "test", raising=False)
    monkeypatch.setitem(
        sys.modules, "dinkster_inference_torch", ModuleType("dinkster_inference_torch")
    )
    import_start, import_end = 10.0, 10.000001
    clock = iter((import_start, import_end))
    monkeypatch.setattr(
        benchmark_inference, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )

    assert benchmark_inference.main() == 0
    assert validation_modes == ["direct-diagnostic" not in placement_arguments]
    report = json.loads(output.read_text())
    assert report["timings"]["import_s"] == import_end - import_start
    assert report["placement"] == executed
    if expected_variant is None:
        assert "variant" not in report
    else:
        assert report["variant"] == expected_variant
    assert report["residency"] == {
        "mechanism": "auto",
        "aimdo_bootstrap_succeeded": None,
        "routes": {},
        "regime": "open",
        "leave_free_mib": None,
        "ballast_bytes": None,
        "spill_scope": "off",
        "shared_before_bytes": None,
        "shared_warm_bytes": None,
        "shared_after_bytes": None,
        "shared_growth_bytes": None,
        "shared_spill_detected": None,
    }


def test_anima_main_reuses_one_descriptor_bound_preflight_per_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_assets.integrity
    import dinkster_inference
    import dinkster_inference.minimax_h3_assembly
    import dinkster_inference.qwen_image_assembly
    from dinkster_compat_comfy import native_arm
    from dinkster_inference.component_registry import ComponentDescriptor, ComponentRegistry

    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    payload = len(header).to_bytes(8, "little") + header + b"\0\0\0\0"
    paths = {
        role: tmp_path / f"{role.replace('_', '-')}.safetensors"
        for role in ("diffusion", "text_encoder", "vae")
    }
    for path in paths.values():
        path.write_bytes(payload)
    monkeypatch.setattr(
        benchmark_inference,
        "_ANIMA_ARTIFACT_PINS",
        {
            role: (
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                f"https://example.invalid/{path.name}",
            )
            for role, path in paths.items()
        },
    )

    production_started = False
    opens: dict[Path, list[tuple[int, bool]]] = {path: [] for path in paths.values()}
    reads: dict[Path, list[tuple[int, int, int, int, bool]]] = {path: [] for path in paths.values()}
    original_open = cast(Any, Path.open)

    class TrackedFile:
        def __init__(self, handle: Any, path: Path, open_id: int) -> None:
            self.handle = handle
            self.path = path
            self.open_id = open_id

        def read(self, size: int = -1) -> bytes:
            offset = self.handle.tell()
            data = self.handle.read(size)
            reads[self.path].append((self.open_id, offset, size, len(data), production_started))
            return data

        def __enter__(self) -> "TrackedFile":
            return self

        def __exit__(self, *exc_info: object) -> None:
            self.handle.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self.handle, name)

    def tracked_open(path: Path, mode: str = "r", *args: object, **kwargs: object) -> Any:
        handle = original_open(path, mode, *args, **kwargs)
        if path in opens and mode == "rb":
            open_id = len(opens[path])
            opens[path].append((open_id, production_started))
            return TrackedFile(handle, path, open_id)
        return handle

    monkeypatch.setattr(Path, "open", tracked_open)
    hash_calls: list[bool] = []
    original_hash_handle = dinkster_assets.integrity._hash_handle

    def tracked_hash_handle(handle: Any) -> str:
        hash_calls.append(production_started)
        return original_hash_handle(handle)

    monkeypatch.setattr(dinkster_assets.integrity, "_hash_handle", tracked_hash_handle)

    torch_module = ModuleType("torch")
    untyped_torch = cast(Any, torch_module)
    untyped_torch.bfloat16 = object()
    untyped_torch.float16 = object()
    untyped_torch.float32 = object()
    untyped_torch.nn = SimpleNamespace(Module=object)
    untyped_torch.device = lambda value: SimpleNamespace(type=str(value).split(":", 1)[0])
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    # The root venv is torch-free, so only tensor construction below asset.open is stubbed.
    package_root = (
        _MODULE_PATH.parent.parent
        / "packages"
        / "dinkster-inference-torch"
        / "src"
        / "dinkster_inference_torch"
    )
    inference_torch = ModuleType("dinkster_inference_torch")
    cast(Any, inference_torch).__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch", inference_torch)

    def dependency(name: str, **attributes: object) -> None:
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    class StubModule:
        pass

    dependency(
        "dinkster_inference_torch.assemble",
        AssembledQwenImage=object,
        _load_component=lambda *args, **kwargs: StubModule(),
    )
    dependency("dinkster_inference_torch.anima_model", AnimaModel=StubModule)
    dependency("dinkster_inference_torch.operations", Operations=object)
    dependency("dinkster_inference_torch.qwen_image", QwenImage=StubModule)
    dependency("dinkster_inference_torch.qwen_image_text", QwenImageTextModel=StubModule)
    dependency("dinkster_inference_torch.qwen_text", QwenTextModel=StubModule)
    dependency(
        "dinkster_inference_torch.wan21_vae",
        WanVAE=StubModule,
        WanVAEConfig=StubModule,
    )

    def load_source_module(name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    anima_loader = load_source_module(
        "dinkster_inference_torch.anima_component",
        package_root / "anima_component.py",
    )
    qwen_loader = load_source_module(
        "dinkster_inference_torch.qwen_image_assembly",
        package_root / "qwen_image_assembly.py",
    )

    def plan_component(source: object, *, role: str, path: Path) -> object:
        del source, path
        return SimpleNamespace(component=role, runtime_facts={})

    materialized_roles: list[str] = []

    def materialize_component(plan: object, _builder: object, **kwargs: object) -> object:
        assert production_started
        os.fstat(cast(Any, kwargs["source_file"]).fileno())
        materialized_roles.append(cast(Any, plan).component)
        return StubModule()

    monkeypatch.setattr(anima_loader, "plan_anima_split_component", plan_component)
    monkeypatch.setattr(
        anima_loader,
        "anima_component_runtime_identity",
        lambda plan, role, dtype: f"anima:{role}",
    )
    monkeypatch.setattr(anima_loader, "_load_component", materialize_component)
    monkeypatch.setattr(qwen_loader, "plan_qwen_image_official_component", plan_component)
    monkeypatch.setattr(
        qwen_loader,
        "qwen_image_component_runtime_identity",
        lambda plan, role, dtype: "qwen:vae",
    )
    monkeypatch.setattr(qwen_loader, "_load_component", materialize_component)
    untyped_inference_torch = cast(Any, inference_torch)
    untyped_inference_torch.load_anima_component = anima_loader.load_anima_component
    untyped_inference_torch.load_qwen_image_component = qwen_loader.load_qwen_image_component
    untyped_inference_torch.AnimaDiffusionRuntime = lambda *args, **kwargs: object()
    untyped_inference_torch.enroll_component = lambda *args, **kwargs: object()

    monkeypatch.setattr(
        dinkster_inference,
        "plan_anima_split_component",
        plan_component,
    )
    monkeypatch.setattr(
        dinkster_inference,
        "plan_qwen_image_official_component",
        lambda source, *, role, path: (source, role, path),
    )
    monkeypatch.setattr(
        dinkster_inference,
        "anima_component_runtime_identity",
        lambda plan, role, dtype: f"anima:{role}",
    )
    monkeypatch.setattr(
        dinkster_inference,
        "qwen_image_component_runtime_identity",
        lambda plan, role, dtype: f"qwen:{role}",
    )
    monkeypatch.setattr(
        ComponentRegistry,
        "select",
        lambda registry, source, path, kind, **kwargs: (
            registry.get("dinkster.qwen_image" if kind == "codec" else "dinkster.anima"),
            {"model": "diffusion", "text": "qwen3_06b", "codec": "vae"}[kind],
            None,
        ),
    )
    monkeypatch.setattr(
        ComponentRegistry,
        "select_detected",
        lambda registry, matches, kind, **kwargs: registry.select(None, None, kind, **kwargs),
    )
    monkeypatch.setattr(
        ComponentDescriptor,
        "recipe",
        lambda self, source, loaded, compute_dtype, **kwargs: SimpleNamespace(
            runtime_identity=loaded.runtime_identity, overlays=()
        ),
    )

    runtime = SimpleNamespace(
        assembled=SimpleNamespace(family=SimpleNamespace(id="dinkster.anima"))
    )

    class Handle:
        def __init__(self, *, with_runtime: bool = False) -> None:
            if with_runtime:
                self.runtime = runtime

        def terminal_release(self) -> None:
            pass

        def attach_pool(self, pool: object) -> None:
            pass

    monkeypatch.setattr(
        native_arm,
        "_build_runtime_handle",
        lambda *args, **kwargs: Handle(with_runtime=True),
    )
    monkeypatch.setattr(
        native_arm,
        "NativeComponentHandle",
        lambda *args, **kwargs: Handle(),
    )
    monkeypatch.setattr(native_arm, "select_load_device", lambda torch: torch.device("cpu"))
    monkeypatch.setattr(
        native_arm,
        "default_pool",
        lambda: SimpleNamespace(label=lambda handle, name: None),
    )

    class Coordinator:
        @staticmethod
        def enroll_component(module: object, *, enroller: Any, **kwargs: object) -> object:
            return enroller(module, **kwargs)

    monkeypatch.setattr(native_arm, "default_native_residency", Coordinator)
    monkeypatch.setattr(
        native_arm.GenerationClipTextEncode,
        "execute",
        staticmethod(lambda **kwargs: {"conditioning": object()}),
    )
    monkeypatch.setattr(
        native_arm.GenerationEmptyLatentImage,
        "execute",
        staticmethod(lambda **kwargs: {"latent": object()}),
    )
    monkeypatch.setattr(
        native_arm.GenerationKSampler,
        "execute",
        staticmethod(lambda **kwargs: {"latent": object()}),
    )

    class Tensor:
        shape = (1, 1, 1, 3)

    monkeypatch.setattr(
        native_arm.GenerationVAEDecode,
        "execute",
        staticmethod(lambda **kwargs: {"image": Tensor()}),
    )
    monkeypatch.setattr(benchmark_inference.torch, "Tensor", Tensor, raising=False)
    monkeypatch.setattr(
        benchmark_inference.torch,
        "isfinite",
        lambda value: SimpleNamespace(all=lambda: True),
        raising=False,
    )
    monkeypatch.setattr(benchmark_inference.torch, "__version__", "test", raising=False)

    class Access:
        backend_runtime = "test-runtime"
        device = SimpleNamespace(type="cpu")

        def synchronize(self) -> None:
            pass

        def reset_peak(self) -> None:
            pass

        def peak_allocated(self) -> int:
            return 1

        def peak_reserved(self) -> int:
            return 2

        def empty_cache(self) -> None:
            pass

        def allocated(self) -> int:
            return 0

    original_timed = benchmark_inference.BenchmarkRun._timed

    def tracked_timed(run: object, action: Callable[[], None]) -> float:
        nonlocal production_started
        production_started = True
        return original_timed(run, action)

    monkeypatch.setattr(benchmark_inference.BenchmarkRun, "_timed", tracked_timed)

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_inference.py", *anima_arguments(tmp_path), "--json", str(output)],
    )
    monkeypatch.setattr(benchmark_inference, "_admit_backend", lambda backend: Access())
    monkeypatch.setattr(benchmark_inference, "_driver_identity", lambda backend: "test-driver")
    monkeypatch.setattr(benchmark_inference, "_device_entries", lambda backend: [])
    monkeypatch.setattr(benchmark_inference, "_peak_rss_bytes", lambda: 3)
    monkeypatch.delenv("DINKSTER_AIMDO_ARM", raising=False)
    monkeypatch.setattr(benchmark_inference, "_ResidencySampler", _OffSampler)
    monkeypatch.setattr(
        benchmark_inference,
        "validate_benchmark_report",
        lambda report, *, accelerator, canonical_evidence: (),
    )
    assert benchmark_inference.main() == 0

    assert hash_calls == []
    assert materialized_roles == ["diffusion", "qwen3_06b", "vae"]
    for path in paths.values():
        preflight_reads = [
            (size, actual, during_production)
            for _, _, size, actual, during_production in reads[path]
            if size == 1 << 22
        ]
        assert preflight_reads == [(1 << 22, len(payload), False), (1 << 22, 0, False)]
        production_open_ids = [
            open_id for open_id, during_production in opens[path] if during_production
        ]
        assert len(production_open_ids) == 2  # Detection and verified materialization.
        production_reads = [
            (open_id, offset, size, actual)
            for open_id, offset, size, actual, during_production in reads[path]
            if during_production
        ]
        expected_reads = [
            event
            for open_id in production_open_ids
            for event in (
                (open_id, 0, 8, 8),
                (open_id, 8, len(header), len(header)),
            )
        ]
        assert production_reads == expected_reads
    report = json.loads(output.read_text())
    assert all("_asset_preflight" not in artifact for artifact in report["artifacts"])


def test_production_load_uses_package_native_handle_and_records_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aimdo_route: SimpleNamespace
) -> None:
    from dinkster_compat_comfy import native_arm
    from dinkster_workers import current_execution_context

    class Module:
        def parameters(self) -> tuple[object, ...]:
            return (SimpleNamespace(dtype="float16", numel=lambda: 1),)

    @dataclass
    class Assembled:
        family: object
        diffusion: object
        clip_l: object
        clip_g: object
        vae: object

    modules = (Module(), Module(), Module(), Module())
    assembled = Assembled(SimpleNamespace(id="dinkster.sd15"), *modules)
    runtime = SimpleNamespace(assembled=assembled)
    seen_assets: list[dict[str, object]] = []

    class Handle:
        def __init__(self) -> None:
            self.runtime = runtime
            self.residency_route = aimdo_route

        @contextmanager
        def stage(self, role: str):  # noqa: ANN202
            yield role

        def terminal_release(self) -> None:
            pass

    handle = Handle()

    def load_handle(assets: Mapping[str, object]) -> Handle:
        assert current_execution_context() is not None
        seen_assets.append(dict(assets))
        return handle

    monkeypatch.setattr(native_arm, "load_native_runtime_handle", load_handle)
    monkeypatch.setattr(
        benchmark_inference.torch,
        "nn",
        SimpleNamespace(Module=Module),
        raising=False,
    )
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    arguments = benchmark_inference._parse_arguments(
        [
            "--backend",
            "cuda",
            "--family",
            "sd15",
            "--checkpoint",
            str(checkpoint),
        ]
    )
    run = benchmark_inference.BenchmarkRun(
        arguments,
        SimpleNamespace(device="cuda:0", synchronize=lambda: None),
    )

    detail = run.load()

    assert tuple(seen_assets[0]) == ("checkpoint",)
    assert cast(Any, seen_assets[0]["checkpoint"]).name == checkpoint.name
    assert run.model_handle is handle
    assert run.executed_placement == "production_residency"
    assert run.residency_routes == {
        "runtime": {
            **vars(aimdo_route),
            "dynamic_components": ["component"],
            "resident_components": [],
            "fallback_components": [],
        }
    }
    assert "production native handle" in detail


@pytest.mark.parametrize(
    "invalid",
    ["missing", "eager", "reason", "fallback", "empty", "requested", "bootstrap"],
)
@pytest.mark.parametrize("selector", ["auto", "on"])
def test_generic_load_rejects_missing_or_degraded_residency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aimdo_route: SimpleNamespace,
    invalid: str,
    selector: str,
) -> None:
    from dinkster_compat_comfy import native_arm

    monkeypatch.setenv("DINKSTER_AIMDO_ARM", selector)
    aimdo_route.requested = selector
    handle = SimpleNamespace(runtime=None, residency_route=aimdo_route)
    if invalid == "missing":
        handle.residency_route = None
    elif invalid == "eager":
        aimdo_route.mechanism = "eager"
    elif invalid == "reason":
        aimdo_route.fallback_reason = "device admission failed"
    elif invalid == "fallback":
        aimdo_route.fallback_components = ("text",)
    elif invalid == "empty":
        aimdo_route.dynamic_components = ()
    elif invalid == "requested":
        aimdo_route.requested = "off"
    else:
        monkeypatch.setattr(benchmark_inference, "_AIMDO_BOOTSTRAP_SUCCEEDED", False)
    monkeypatch.setattr(native_arm, "load_native_runtime_handle", lambda assets: handle)
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    arguments = benchmark_inference._parse_arguments(
        ["--backend", "cuda", "--family", "sd15", "--checkpoint", str(checkpoint)]
    )
    run = benchmark_inference.BenchmarkRun(arguments, SimpleNamespace(synchronize=lambda: None))

    run.record("load", run.load)

    assert run.failed
    assert not run.checks["load"]["ok"]
    assert "residency" in str(run.checks["load"]["detail"])
    assert run.model_handle is handle
    if invalid != "missing":
        route = cast("Mapping[str, object]", run.residency_routes["runtime"])
        assert route["mechanism"] == aimdo_route.mechanism


def test_flux_production_load_binds_the_flux_asset_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aimdo_route: SimpleNamespace
) -> None:
    from dinkster_compat_comfy import native_arm

    class Module:
        def parameters(self) -> tuple[object, ...]:
            return (SimpleNamespace(dtype="bfloat16", numel=lambda: 1),)

    @dataclass
    class Assembled:
        family: object
        diffusion: object
        clip_l: object
        t5xxl: object
        vae: object

    assembled = Assembled(
        SimpleNamespace(id="dinkster.flux_dev"), Module(), Module(), Module(), Module()
    )
    runtime = SimpleNamespace(assembled=assembled)
    seen_assets: list[dict[str, object]] = []

    class Handle:
        def __init__(self) -> None:
            self.runtime = runtime
            self.residency_route = aimdo_route

        def terminal_release(self) -> None:
            pass

    handle = Handle()

    def load_handle(assets: Mapping[str, object]) -> Handle:
        seen_assets.append(dict(assets))
        return handle

    monkeypatch.setattr(native_arm, "load_native_runtime_handle", load_handle)
    monkeypatch.setattr(
        benchmark_inference.torch,
        "nn",
        SimpleNamespace(Module=Module),
        raising=False,
    )
    argv = flux_arguments(tmp_path)
    for value in argv[5::2]:
        Path(value).write_bytes(f"{Path(value).stem} bytes".encode())
    arguments = benchmark_inference._parse_arguments(argv)
    preflight_assets = {
        role: benchmark_inference._AssetPreflight(
            path=path,
            digest=f"blake3:{index:064x}",
            size=1,
            verification=object(),
        )
        for index, (role, path) in enumerate(
            {
                "diffusion": arguments.diffusion,
                "clip_l": arguments.clip_l,
                "text_encoder": arguments.text_encoder,
                "vae": arguments.vae,
            }.items()
        )
    }

    import dinkster_assets.integrity

    def rejected_rehash(path: Path) -> tuple[str, object]:
        raise AssertionError(f"flux load re-hashed an already-preflighted artifact: {path}")

    monkeypatch.setattr(dinkster_assets.integrity, "digest_file_with_record", rejected_rehash)
    run = benchmark_inference.BenchmarkRun(
        arguments,
        SimpleNamespace(device="cuda:0", synchronize=lambda: None),
        preflight_assets,
    )

    detail = run.load()

    assert tuple(seen_assets[0]) == ("diffusion", "clip_l", "t5xxl", "vae")
    assert cast(Any, seen_assets[0]["clip_l"]).name == "clip-l.safetensors"
    assert cast(Any, seen_assets[0]["t5xxl"]).name == "text-encoder.safetensors"
    assert cast(Any, seen_assets[0]["diffusion"]).digest == preflight_assets["diffusion"].digest
    assert cast(Any, seen_assets[0]["clip_l"]).digest == preflight_assets["clip_l"].digest
    assert cast(Any, seen_assets[0]["t5xxl"]).digest == preflight_assets["text_encoder"].digest
    assert cast(Any, seen_assets[0]["vae"]).digest == preflight_assets["vae"].digest
    assert run.model_handle is handle
    assert run.executed_placement == "production_residency"
    assert run.family_id == "dinkster.flux_dev"
    assert "production native handle" in detail


def test_failed_native_handle_enrollment_records_no_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy import native_arm

    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"checkpoint")

    def fail(_assets: object) -> object:
        raise RuntimeError("enrollment failed")

    monkeypatch.setattr(native_arm, "load_native_runtime_handle", fail)
    arguments = benchmark_inference._parse_arguments(
        ["--backend", "cuda", "--family", "sd15", "--checkpoint", str(checkpoint)]
    )
    run = benchmark_inference.BenchmarkRun(
        arguments, SimpleNamespace(device="cuda:0", synchronize=lambda: None)
    )

    with pytest.raises(RuntimeError, match="enrollment failed"):
        run.load()

    assert run.executed_placement is None


def test_anima_drives_production_component_nodes_and_releases_every_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy import native_arm
    from dinkster_workers import current_execution_context
    from dinkster_workers.execution import ExecutionContext

    events: list[object] = []
    runtime = SimpleNamespace(
        assembled=SimpleNamespace(family=SimpleNamespace(id="dinkster.anima"))
    )

    class Handle:
        def __init__(self, name: str, *, with_runtime: bool = False) -> None:
            self.name = name
            if with_runtime:
                self.runtime = runtime

        def terminal_release(self) -> None:
            events.append(("release", self.name))

    model = Handle("model", with_runtime=True)
    clip = Handle("clip")
    vae = Handle("vae")
    assets = {role: object() for role in ("diffusion", "text_encoder", "vae")}
    contexts = {
        role: ExecutionContext("native", f"anima-{role}")
        for role in ("diffusion", "text_encoder", "vae")
    }
    preflight_assets = {
        role: benchmark_inference._AssetPreflight(
            tmp_path / f"{role}.safetensors",
            "blake3:" + "a" * 64,
            1,
            object(),
        )
        for role in ("diffusion", "text_encoder", "vae")
    }
    preflight_roles: list[str] = []

    def component_load(
        path: Path,
        role: str,
        preflight: object,
    ) -> tuple[object, ExecutionContext]:
        del path
        assert preflight == preflight_assets[role]
        preflight_roles.append(role)
        return assets[role], contexts[role]

    monkeypatch.setattr(
        benchmark_inference,
        "_anima_component_load",
        component_load,
    )

    def load_model(**kwargs: object) -> dict[str, object]:
        events.append(("load_model", kwargs, current_execution_context()))
        return {"model": model}

    def load_clip(**kwargs: object) -> dict[str, object]:
        events.append(("load_clip", kwargs, current_execution_context()))
        return {"clip": clip}

    run: Any

    def load_vae(**kwargs: object) -> dict[str, object]:
        assert run.executed_placement is None
        events.append(("load_vae", kwargs, current_execution_context()))
        return {"vae": vae}

    monkeypatch.setattr(
        native_arm.GenerationLoadDiffusionModel, "execute", staticmethod(load_model)
    )
    monkeypatch.setattr(native_arm.NativeLoadClip, "execute", staticmethod(load_clip))
    monkeypatch.setattr(native_arm.NativeLoadVae, "execute", staticmethod(load_vae))

    def encode(**kwargs: object) -> dict[str, object]:
        events.append(("encode", kwargs))
        return {"conditioning": f"conditioning:{kwargs['text']}"}

    def empty_latent(**kwargs: object) -> dict[str, object]:
        events.append(("latent", kwargs))
        return {"latent": "empty-latent"}

    def sample(**kwargs: object) -> dict[str, object]:
        events.append(("sample", kwargs))
        return {"latent": "sampled-latent"}

    def decode(**kwargs: object) -> dict[str, object]:
        events.append(("decode", kwargs))
        return {"image": "decoded-image"}

    monkeypatch.setattr(native_arm.GenerationClipTextEncode, "execute", staticmethod(encode))
    monkeypatch.setattr(
        native_arm.GenerationEmptyLatentImage, "execute", staticmethod(empty_latent)
    )
    monkeypatch.setattr(native_arm.GenerationKSampler, "execute", staticmethod(sample))
    monkeypatch.setattr(native_arm.GenerationVAEDecode, "execute", staticmethod(decode))
    arguments = benchmark_inference._parse_arguments(anima_arguments(tmp_path))
    access = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        synchronize=lambda: None,
        empty_cache=lambda: None,
        allocated=lambda: 0,
    )
    run = benchmark_inference.BenchmarkRun(arguments, access, preflight_assets)

    timed_calls = 0

    def timed(action: Callable[[], None]) -> float:
        nonlocal timed_calls
        if timed_calls == 0:
            assert preflight_roles == ["diffusion", "text_encoder", "vae"]
        timed_calls += 1
        action()
        return 0.1

    monkeypatch.setattr(run, "_timed", timed)

    run.load()
    run.encode_text()
    image, _sample_s, _decode_s, _steps = run._run_once(arguments.seed)

    assert image == "decoded-image"
    assert run.executed_placement == "production_residency"
    assert events[:3] == [
        (
            "load_model",
            {"diffusion_model": assets["diffusion"], "weight_dtype": "default"},
            contexts["diffusion"],
        ),
        (
            "load_clip",
            {
                "text_encoder": assets["text_encoder"],
                "type": "stable_diffusion",
                "device": "default",
            },
            contexts["text_encoder"],
        ),
        ("load_vae", {"vae": assets["vae"]}, contexts["vae"]),
    ]
    assert [event for event in events if cast(Any, event)[0] == "encode"] == [
        ("encode", {"text": arguments.prompt, "clip": clip}),
        ("encode", {"text": "", "clip": clip}),
    ]
    sample_event = next(cast(Any, event)[1] for event in events if cast(Any, event)[0] == "sample")
    assert sample_event == {
        "model": model,
        "seed": 875817230929465,
        "steps": 30,
        "cfg": 4.0,
        "sampler_name": "dinkster.er_sde",
        "scheduler": "dinkster.simple",
        "positive": f"conditioning:{arguments.prompt}",
        "negative": "conditioning:",
        "latent_image": "empty-latent",
        "denoise": 1.0,
    }
    assert ("decode", {"samples": "sampled-latent", "vae": vae}) in events

    run.unload()
    assert events[-3:] == [("release", "clip"), ("release", "vae"), ("release", "model")]


def test_anima_partial_load_failure_releases_already_enrolled_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy import native_arm
    from dinkster_workers.execution import ExecutionContext

    released: list[str] = []

    class Handle:
        def __init__(self, name: str) -> None:
            self.name = name
            self.runtime = SimpleNamespace(
                assembled=SimpleNamespace(family=SimpleNamespace(id="dinkster.anima"))
            )

        def terminal_release(self) -> None:
            released.append(self.name)

    monkeypatch.setattr(
        benchmark_inference,
        "_anima_component_load",
        lambda path, role, digest: (object(), ExecutionContext("native", role)),
    )
    monkeypatch.setattr(
        native_arm.GenerationLoadDiffusionModel,
        "execute",
        staticmethod(lambda **kwargs: {"model": Handle("model")}),
    )
    monkeypatch.setattr(
        native_arm.NativeLoadClip,
        "execute",
        staticmethod(lambda **kwargs: {"clip": Handle("clip")}),
    )

    def fail_vae(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("VAE enrollment failed")

    monkeypatch.setattr(native_arm.NativeLoadVae, "execute", staticmethod(fail_vae))
    run = benchmark_inference.BenchmarkRun(
        benchmark_inference._parse_arguments(anima_arguments(tmp_path)),
        SimpleNamespace(synchronize=lambda: None),
        {
            role: benchmark_inference._AssetPreflight(
                tmp_path / f"{role}.safetensors",
                "blake3:" + "a" * 64,
                1,
                object(),
            )
            for role in ("diffusion", "text_encoder", "vae")
        },
    )

    with pytest.raises(RuntimeError, match="VAE enrollment failed"):
        run.load()

    assert released == ["clip", "model"]
    assert run.executed_placement is None


@pytest.mark.parametrize(
    ("role", "planned_role", "dtypes", "identity"),
    [
        ("diffusion", "diffusion", ("bfloat16", "unloaded", "unloaded"), "diffusion-id"),
        ("text_encoder", "qwen3_06b", ("unloaded", "float32", "unloaded"), "text-id"),
        ("vae", "vae", ("unloaded", "unloaded", "bfloat16"), "vae-id"),
    ],
)
def test_anima_component_load_reuses_preflight_identity_contexts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    planned_role: str,
    dtypes: tuple[str, str, str],
    identity: str,
) -> None:
    import dinkster_inference

    path = tmp_path / f"{role}.safetensors"
    path.write_bytes(b"small test artifact")
    preflight = benchmark_inference._AssetPreflight(
        path,
        "blake3:" + "a" * 64,
        len(b"small test artifact"),
        object(),
    )
    observed: dict[str, object] = {}

    def load_header(source_path: Path, *, asset_digest: str, asset_size: int) -> object:
        observed["header"] = (source_path, asset_digest, asset_size)
        return "source"

    def plan_anima(source: object, *, role: str, path: Path) -> object:
        observed["plan"] = (source, role, path)
        return f"plan:{role}"

    def plan_qwen(source: object, *, role: str, path: Path) -> object:
        observed["plan"] = (source, role, path)
        return f"plan:{role}"

    monkeypatch.setattr(dinkster_inference, "load_safetensors_header", load_header)
    monkeypatch.setattr(dinkster_inference, "plan_anima_split_component", plan_anima)
    monkeypatch.setattr(dinkster_inference, "plan_qwen_image_official_component", plan_qwen)
    monkeypatch.setattr(
        dinkster_inference,
        "anima_component_runtime_identity",
        lambda plan, component_role, dtype: (
            "text-id" if component_role == "qwen3_06b" else "diffusion-id"
        ),
    )
    monkeypatch.setattr(
        dinkster_inference,
        "qwen_image_component_runtime_identity",
        lambda plan, component_role, dtype: "vae-id",
    )

    asset, context = benchmark_inference._anima_component_load(path, role, preflight)

    assert asset.digest == preflight.digest
    assert observed["header"] == (path, asset.digest, len(b"small test artifact"))
    assert observed["plan"] == ("source", planned_role, path)
    assert context.expected_execution_identity == identity
    assert (context.diffusion_dtype, context.text_dtype, context.vae_dtype) == dtypes


def test_chroma_drives_production_component_nodes_and_releases_every_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy import native_arm
    from dinkster_workers import current_execution_context
    from dinkster_workers.execution import ExecutionContext

    events: list[object] = []
    runtime = SimpleNamespace(
        assembled=SimpleNamespace(family=SimpleNamespace(id="dinkster.chroma"))
    )

    class Handle:
        def __init__(self, name: str, *, with_runtime: bool = False) -> None:
            self.name = name
            if with_runtime:
                self.runtime = runtime

        def terminal_release(self) -> None:
            events.append(("release", self.name))

    model = Handle("model", with_runtime=True)
    clip = Handle("clip")
    vae = Handle("vae")
    assets = {role: object() for role in ("diffusion", "text_encoder", "vae")}
    contexts = {
        role: ExecutionContext("native", f"chroma-{role}")
        for role in ("diffusion", "text_encoder", "vae")
    }
    preflight_assets = {
        role: benchmark_inference._AssetPreflight(
            tmp_path / f"{role}.safetensors",
            "blake3:" + "a" * 64,
            1,
            object(),
        )
        for role in ("diffusion", "text_encoder", "vae")
    }
    preflight_roles: list[str] = []

    def component_load(
        path: Path,
        role: str,
        preflight: object,
    ) -> tuple[object, ExecutionContext]:
        del path
        assert preflight == preflight_assets[role]
        preflight_roles.append(role)
        return assets[role], contexts[role]

    monkeypatch.setattr(
        benchmark_inference,
        "_chroma_component_load",
        component_load,
    )

    def load_model(**kwargs: object) -> dict[str, object]:
        events.append(("load_model", kwargs, current_execution_context()))
        return {"model": model}

    def load_clip(**kwargs: object) -> dict[str, object]:
        events.append(("load_clip", kwargs, current_execution_context()))
        return {"clip": clip}

    run: Any

    def load_vae(**kwargs: object) -> dict[str, object]:
        assert run.executed_placement is None
        events.append(("load_vae", kwargs, current_execution_context()))
        return {"vae": vae}

    monkeypatch.setattr(
        native_arm.GenerationLoadDiffusionModel, "execute", staticmethod(load_model)
    )
    monkeypatch.setattr(native_arm.NativeLoadClip, "execute", staticmethod(load_clip))
    monkeypatch.setattr(native_arm.NativeLoadVae, "execute", staticmethod(load_vae))

    def tokenizer_options(**kwargs: object) -> dict[str, object]:
        events.append(("tokenizer_options", kwargs))
        return {"clip": "options-clip"}

    def encode(**kwargs: object) -> dict[str, object]:
        events.append(("encode", kwargs))
        return {"conditioning": f"conditioning:{kwargs['text']}"}

    def empty_latent(**kwargs: object) -> dict[str, object]:
        events.append(("latent", kwargs))
        return {"latent": "empty-latent"}

    def sample(**kwargs: object) -> dict[str, object]:
        events.append(("sample", kwargs))
        return {"latent": "sampled-latent"}

    def decode(**kwargs: object) -> dict[str, object]:
        events.append(("decode", kwargs))
        return {"image": "decoded-image"}

    monkeypatch.setattr(
        native_arm.GenerationT5TokenizerOptions, "execute", staticmethod(tokenizer_options)
    )
    monkeypatch.setattr(native_arm.GenerationClipTextEncode, "execute", staticmethod(encode))
    monkeypatch.setattr(
        native_arm.GenerationEmptySD3LatentImage, "execute", staticmethod(empty_latent)
    )
    monkeypatch.setattr(native_arm.GenerationKSampler, "execute", staticmethod(sample))
    monkeypatch.setattr(native_arm.GenerationVAEDecode, "execute", staticmethod(decode))
    arguments = benchmark_inference._parse_arguments(chroma_arguments(tmp_path))
    access = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        synchronize=lambda: None,
        empty_cache=lambda: None,
        allocated=lambda: 0,
    )
    run = benchmark_inference.BenchmarkRun(arguments, access, preflight_assets)

    timed_calls = 0

    def timed(action: Callable[[], None]) -> float:
        nonlocal timed_calls
        if timed_calls == 0:
            assert preflight_roles == ["diffusion", "text_encoder", "vae"]
        timed_calls += 1
        action()
        return 0.1

    monkeypatch.setattr(run, "_timed", timed)

    run.load()
    run.encode_text()
    image, _sample_s, _decode_s, step_wall_ms = run._run_once(arguments.seed)

    assert image == "decoded-image"
    assert step_wall_ms == []
    assert run.executed_placement == "production_residency"
    assert events[:3] == [
        (
            "load_model",
            {"diffusion_model": assets["diffusion"], "weight_dtype": "default"},
            contexts["diffusion"],
        ),
        (
            "load_clip",
            {
                "text_encoder": assets["text_encoder"],
                "type": "chroma",
                "device": "default",
            },
            contexts["text_encoder"],
        ),
        ("load_vae", {"vae": assets["vae"]}, contexts["vae"]),
    ]
    assert [event for event in events if cast(Any, event)[0] == "tokenizer_options"] == [
        ("tokenizer_options", {"clip": clip, "min_padding": 0, "min_length": 0}),
    ]
    assert [event for event in events if cast(Any, event)[0] == "encode"] == [
        ("encode", {"text": arguments.prompt, "clip": "options-clip"}),
        ("encode", {"text": "", "clip": "options-clip"}),
    ]
    sample_event = next(cast(Any, event)[1] for event in events if cast(Any, event)[0] == "sample")
    assert sample_event == {
        "model": model,
        "seed": 667,
        "steps": 26,
        "cfg": 3.5,
        "sampler_name": "dinkster.euler",
        "scheduler": "dinkster.beta",
        "positive": f"conditioning:{arguments.prompt}",
        "negative": "conditioning:",
        "latent_image": "empty-latent",
        "denoise": 1.0,
    }
    assert ("decode", {"samples": "sampled-latent", "vae": vae}) in events

    run.unload()
    assert events[-3:] == [("release", "clip"), ("release", "vae"), ("release", "model")]


def test_chroma_partial_load_failure_releases_already_enrolled_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy import native_arm
    from dinkster_workers.execution import ExecutionContext

    released: list[str] = []

    class Handle:
        def __init__(self, name: str) -> None:
            self.name = name
            self.runtime = SimpleNamespace(
                assembled=SimpleNamespace(family=SimpleNamespace(id="dinkster.chroma"))
            )

        def terminal_release(self) -> None:
            released.append(self.name)

    monkeypatch.setattr(
        benchmark_inference,
        "_chroma_component_load",
        lambda path, role, digest: (object(), ExecutionContext("native", role)),
    )
    monkeypatch.setattr(
        native_arm.GenerationLoadDiffusionModel,
        "execute",
        staticmethod(lambda **kwargs: {"model": Handle("model")}),
    )
    monkeypatch.setattr(
        native_arm.NativeLoadClip,
        "execute",
        staticmethod(lambda **kwargs: {"clip": Handle("clip")}),
    )

    def fail_vae(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("VAE enrollment failed")

    monkeypatch.setattr(native_arm.NativeLoadVae, "execute", staticmethod(fail_vae))
    run = benchmark_inference.BenchmarkRun(
        benchmark_inference._parse_arguments(chroma_arguments(tmp_path)),
        SimpleNamespace(synchronize=lambda: None),
        {
            role: benchmark_inference._AssetPreflight(
                tmp_path / f"{role}.safetensors",
                "blake3:" + "a" * 64,
                1,
                object(),
            )
            for role in ("diffusion", "text_encoder", "vae")
        },
    )

    with pytest.raises(RuntimeError, match="VAE enrollment failed"):
        run.load()

    assert released == ["clip", "model"]
    assert run.executed_placement is None


@pytest.mark.parametrize(
    ("role", "planned_role", "dtypes"),
    [
        ("diffusion", "diffusion", ("bfloat16", "unloaded", "unloaded")),
        ("text_encoder", "t5xxl", ("unloaded", "bfloat16", "unloaded")),
        ("vae", "vae", ("unloaded", "unloaded", "bfloat16")),
    ],
)
def test_chroma_component_load_reuses_preflight_identity_contexts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    planned_role: str,
    dtypes: tuple[str, str, str],
) -> None:
    import dinkster_inference

    path = tmp_path / f"{role}.safetensors"
    path.write_bytes(b"small test artifact")
    preflight = benchmark_inference._AssetPreflight(
        path,
        "blake3:" + "a" * 64,
        len(b"small test artifact"),
        object(),
    )
    observed: dict[str, object] = {}

    def load_header(source_path: Path, *, asset_digest: str, asset_size: int) -> object:
        observed["header"] = (source_path, asset_digest, asset_size)
        return "source"

    def plan_chroma(source: object, *, role: str, path: Path) -> object:
        observed["plan"] = (source, role, path)
        return f"plan:{role}"

    monkeypatch.setattr(dinkster_inference, "load_safetensors_header", load_header)
    monkeypatch.setattr(dinkster_inference, "plan_chroma_split_component", plan_chroma)
    monkeypatch.setattr(
        dinkster_inference,
        "chroma_component_runtime_identity",
        lambda plan, component_role, dtype: f"chroma-{component_role}-id",
    )

    asset, context = benchmark_inference._chroma_component_load(path, role, preflight)

    assert asset.digest == preflight.digest
    assert observed["header"] == (path, asset.digest, len(b"small test artifact"))
    assert observed["plan"] == ("source", planned_role, path)
    assert context.expected_execution_identity == f"chroma-{planned_role}-id"
    assert (context.diffusion_dtype, context.text_dtype, context.vae_dtype) == dtypes


def test_direct_diagnostic_load_places_modules_and_records_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dinkster_inference

    class Module:
        def __init__(self) -> None:
            self.to_calls: list[object] = []

        def parameters(self) -> tuple[object, ...]:
            return (SimpleNamespace(dtype="float16", numel=lambda: 1),)

        def to(self, device: object) -> None:
            self.to_calls.append(device)

    @dataclass
    class Assembled:
        family: object
        diffusion: object
        clip_l: object
        clip_g: object
        vae: object

    modules = (Module(), Module(), Module(), Module())
    runtime = SimpleNamespace(assembled=Assembled(SimpleNamespace(id="dinkster.sd15"), *modules))
    inference_torch = ModuleType("dinkster_inference_torch")
    inference_torch.load_runtime = lambda **kwargs: runtime  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch", inference_torch)
    monkeypatch.setattr(dinkster_inference, "load_safetensors_header", lambda path: object())
    monkeypatch.setattr(
        benchmark_inference.torch, "nn", SimpleNamespace(Module=Module), raising=False
    )
    checkpoint = tmp_path / "checkpoint.safetensors"
    arguments = benchmark_inference._parse_arguments(
        [
            "--backend",
            "cuda",
            "--family",
            "sd15",
            "--checkpoint",
            str(checkpoint),
            "--placement",
            "direct-diagnostic",
        ]
    )
    run = benchmark_inference.BenchmarkRun(
        arguments, SimpleNamespace(device="cuda:0", synchronize=lambda: None)
    )

    run.load()

    assert all(module.to_calls == ["cuda:0"] for module in modules)
    assert run.model_handle is None
    assert run.executed_placement == "direct_placement_diagnostic"


def test_production_diffusion_only_lora_forces_precalculated_native_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy import native_arm

    events: list[object] = []
    runtime = object()

    class Handle:
        def __init__(self, name: str) -> None:
            self.name = name
            self.runtime = runtime

        def terminal_release(self) -> None:
            events.append(("release", self.name))

    base = Handle("base")
    clone = Handle("clone")
    lora = tmp_path / "patch.safetensors"
    lora.write_bytes(b"lora")

    class DiffusionOverlay:
        def __init__(self, handle: Handle) -> None:
            self.handle = handle
            self.runtime = handle.runtime

        @contextmanager
        def stage(self, role: str):  # noqa: ANN202
            yield role

    def apply_lora(**kwargs: object) -> dict[str, object]:
        events.append(kwargs)
        if kwargs.get("execution_mode") == "precalculate":
            return {"model": clone, "clip": clone}
        return {"model": DiffusionOverlay(base), "clip": base}

    monkeypatch.setattr(native_arm.GenerationLoadLora, "execute", staticmethod(apply_lora))
    arguments = SimpleNamespace(
        placement="residency",
        lora=lora,
        lora_strength_model=0.8,
        lora_strength_clip=0.6,
    )
    run = benchmark_inference.BenchmarkRun(arguments, SimpleNamespace(synchronize=lambda: None))
    run.model_handle = base

    detail = run.lora_apply()

    applied = cast("dict[str, object]", events[0])
    assert applied["model"] is base and applied["clip"] is base
    assert cast(Any, applied["lora"]).name == lora.name
    assert applied["strength_model"] == 0.8
    assert applied["strength_clip"] == 0.6
    assert applied["execution_mode"] == "precalculate"
    assert run.model_handle is clone and run.runtime is runtime
    assert events[1] == ("release", "base")
    assert "production native handle" in detail


def test_encode_sample_and_decode_run_inside_residency_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class Tensor:
        def permute(self, *dimensions: int):
            return self

        def clamp(self, low: int, high: int):
            return self

    class Handle:
        @contextmanager
        def stage(
            self,
            role: str,
            *,
            unload_before: tuple[str, ...],
            observer_stage: str,
        ):  # noqa: ANN202
            events.append(("enter", role, unload_before, observer_stage))
            yield
            events.append(("exit", role))

    class Runtime:
        assembled = SimpleNamespace(
            family=SimpleNamespace(
                single_stream_latent=lambda: SimpleNamespace(channels=4, spatial_downscale=8)
            )
        )

        def encode_text(self, prompt: str) -> object:
            assert events[0] == ("enter", "text", (), "load")
            assert ("exit", "text") not in events
            events.append(("encode", prompt))
            return SimpleNamespace(embeddings=object(), pooled=None)

        def sample(self, latent: object, **kwargs: object) -> Tensor:
            assert events[-1] == ("enter", "diffusion", ("text",), "sample")
            events.append("sample")
            callback = kwargs["on_step"]
            assert callable(callback)
            callback(object())
            return Tensor()

        def decode_latent(self, latent: object) -> Tensor:
            assert events[-1] == ("enter", "vae", (), "load")
            events.append("decode")
            return Tensor()

    @contextmanager
    def inference_mode():
        yield

    monkeypatch.setattr(
        benchmark_inference.torch,
        "zeros",
        lambda *args, **kwargs: Tensor(),
        raising=False,
    )
    monkeypatch.setattr(benchmark_inference.torch, "float32", object(), raising=False)
    monkeypatch.setattr(benchmark_inference.torch, "float16", object(), raising=False)
    monkeypatch.setattr(benchmark_inference.torch, "inference_mode", inference_mode, raising=False)
    arguments = SimpleNamespace(
        family="sd15",
        width=64,
        height=64,
        cfg=1.0,
        sampler="dinkster.euler",
        scheduler="dinkster.simple",
        steps=1,
        prompt="positive",
        negative_prompt="negative",
    )
    access = SimpleNamespace(device="cuda:0", synchronize=lambda: None)
    run = benchmark_inference.BenchmarkRun(arguments, access)
    run.runtime = Runtime()
    run.model_handle = Handle()

    run.encode_text()
    image, _sample_s, _decode_s, step_wall_ms = run._run_image(0)

    assert isinstance(image, Tensor)
    assert len(step_wall_ms) == 1
    assert events == [
        ("enter", "text", (), "load"),
        ("encode", "positive"),
        ("encode", "negative"),
        ("exit", "text"),
        ("enter", "diffusion", ("text",), "sample"),
        "sample",
        ("exit", "diffusion"),
        ("enter", "vae", (), "load"),
        "decode",
        ("exit", "vae"),
    ]


def test_minimax_h3_drives_the_native_production_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy import native_arm
    from dinkster_protocol import ATTENTION_ROLES, AttentionRoute, AttentionRouteToken
    from dinkster_schema import report_progress
    from dinkster_workers import current_execution_context

    arguments_list = [
        *minimax_h3_arguments(tmp_path),
        "--attention-policy",
        "sage",
    ]
    for value in arguments_list:
        path = Path(value)
        if path.suffix == ".bin":
            path.write_bytes(b"component")
    arguments = benchmark_inference._parse_arguments(arguments_list)
    route_token = AttentionRouteToken(
        version=1,
        routes=tuple(
            AttentionRoute(role=role, primary="sage", fallback="sdpa") for role in ATTENTION_ROLES
        ),
        provider_versions=(("sageattention", "2.2.0"), ("torch", "2.9.1")),
        adapter_contract_revision="test",
        device_kind="cuda",
        device_sm=120,
        sdpa_torch_runtime="2.9.1",
        requested_policy="sage",
    )
    identities = {
        "diffusion": "diffusion-identity",
        "text_encoder": "text-identity",
        "video_vae": "video-vae-identity",
        "audio_vae": "audio-vae-identity",
    }
    identity_arguments: list[tuple[str, object]] = []

    def execution_identities(
        _assets: dict[str, object],
        *,
        attention_policy: str,
        attention_route_token: object,
    ) -> dict[str, str]:
        identity_arguments.append((attention_policy, attention_route_token))
        return identities

    monkeypatch.setattr(
        benchmark_inference,
        "_minimax_h3_execution_identities",
        execution_identities,
    )
    inference_torch = ModuleType("dinkster_inference_torch")
    inference_torch.discover_attention_route_token = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: route_token
    )
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch", inference_torch)
    events: list[object] = []

    class Handle:
        def __init__(self, role: str) -> None:
            self.role = role
            self.runtime = SimpleNamespace()
            self.residency_route = SimpleNamespace(
                requested="auto",
                mechanism="aimdo",
                fallback_reason=None,
                dynamic_components=(role,),
                resident_components=(),
                fallback_components=(),
            )

        def terminal_release(self) -> None:
            events.append(("release", self.role))

    handles = {
        "diffusion": Handle("diffusion"),
        "text_encoder": Handle("text_encoder"),
        "video_vae": Handle("video_vae"),
        "audio_vae": Handle("audio_vae"),
    }

    def context_event(role: str) -> None:
        context = current_execution_context()
        assert context is not None
        events.append(
            (
                "load",
                role,
                context.expected_execution_identity,
                context.diffusion_dtype,
                context.text_dtype,
                context.vae_dtype,
                context.attention_policy,
                context.attention_route_token,
            )
        )

    def load_diffusion(**kwargs: object) -> dict[str, object]:
        context_event("diffusion")
        asset = cast(Any, kwargs["diffusion_model"])
        assert asset.digest.startswith("blake3:")
        assert kwargs["weight_dtype"] == "default"
        return {"model": handles["diffusion"]}

    def load_clip(**kwargs: object) -> dict[str, object]:
        context_event("text_encoder")
        assert kwargs["type"] == "minimax"
        assert kwargs["device"] == "default"
        return {"clip": handles["text_encoder"]}

    def load_vae(**kwargs: object) -> dict[str, object]:
        asset = cast(Any, kwargs["vae"])
        role = "audio_vae" if "audio-vae" in asset.name else "video_vae"
        context_event(role)
        return {"vae": handles[role]}

    monkeypatch.setattr(
        native_arm.GenerationLoadDiffusionModel,
        "execute",
        staticmethod(load_diffusion),
    )
    monkeypatch.setattr(native_arm.NativeLoadClip, "execute", staticmethod(load_clip))
    monkeypatch.setattr(native_arm.NativeLoadVae, "execute", staticmethod(load_vae))

    class Tensor:
        pass

    frames = Tensor()
    audio = {"waveform": object(), "sample_rate": 32_000}
    monkeypatch.setattr(benchmark_inference.torch, "Tensor", Tensor, raising=False)
    monkeypatch.setattr(
        native_arm.NativeEmptyMiniMaxH3AV,
        "execute",
        staticmethod(lambda **kwargs: events.append(("empty", kwargs)) or {"latent": "empty-av"}),
    )
    monkeypatch.setattr(
        native_arm.NativeMiniMaxH3T2VAConditioning,
        "execute",
        staticmethod(
            lambda **kwargs: (
                events.append(("condition", kwargs)) or {"positive": "positive", "negative": []}
            )
        ),
    )

    def sample(**kwargs: object) -> dict[str, object]:
        context = current_execution_context()
        assert context is not None
        assert context.preview_mode == "off"
        events.append(("sample", kwargs))
        for step in range(1, arguments.steps + 1):
            report_progress(step, arguments.steps)
        return {"latent": "sampled-av"}

    monkeypatch.setattr(native_arm.GenerationKSampler, "execute", staticmethod(sample))
    monkeypatch.setattr(
        native_arm.NativeMiniMaxH3AVDecode,
        "execute",
        staticmethod(
            lambda **kwargs: events.append(("decode", kwargs)) or {"frames": frames, "audio": audio}
        ),
    )
    access = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        synchronize=lambda: None,
        empty_cache=lambda: None,
        allocated=lambda: 0,
    )
    run = benchmark_inference.BenchmarkRun(arguments, access)
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "auto")
    monkeypatch.setattr(benchmark_inference, "_AIMDO_BOOTSTRAP_SUCCEEDED", True)

    assert "production native handles" in run.load()
    assert run.executed_placement == "production_residency"
    assert identity_arguments == [("sage", route_token)]
    assert events[:4] == [
        (
            "load",
            "diffusion",
            "diffusion-identity",
            "bfloat16",
            "unloaded",
            "unloaded",
            "sage",
            route_token,
        ),
        (
            "load",
            "text_encoder",
            "text-identity",
            "unloaded",
            "float16",
            "unloaded",
            "auto",
            None,
        ),
        (
            "load",
            "video_vae",
            "video-vae-identity",
            "unloaded",
            "unloaded",
            "float16",
            "auto",
            None,
        ),
        (
            "load",
            "audio_vae",
            "audio-vae-identity",
            "unloaded",
            "unloaded",
            "float32",
            "auto",
            None,
        ),
    ]
    run.encode_text()
    detail = run.cold_run()
    assert run.first_image is frames
    assert run.decoded_audio is audio
    assert run.execution_path == "generation_ksampler_multistream"
    assert len(cast("list[float]", run.cold["step_wall_ms"])) == 20
    assert "GenerationKSampler production path" in detail
    assert events[4] == (
        "empty",
        {"width": 1344, "height": 768, "frame_count": 124},
    )
    assert events[5] == (
        "condition",
        {
            "clip": handles["text_encoder"],
            "target": "empty-av",
            "prompt": "A red square centered on a black background.",
        },
    )
    assert events[6] == (
        "sample",
        {
            "model": handles["diffusion"],
            "seed": 20260813,
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "dinkster.res_multistep",
            "scheduler": "dinkster.simple",
            "positive": "positive",
            "negative": [],
            "latent_image": "empty-av",
            "denoise": 1.0,
        },
    )
    assert events[7] == (
        "decode",
        {
            "video_vae": handles["video_vae"],
            "audio_vae": handles["audio_vae"],
            "latent": "sampled-av",
        },
    )
    run.unload()
    assert events[-4:] == [
        ("release", "audio_vae"),
        ("release", "video_vae"),
        ("release", "text_encoder"),
        ("release", "diffusion"),
    ]


def test_minimax_h3_partial_load_releases_enrolled_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy import native_arm

    arguments_list = minimax_h3_arguments(tmp_path)
    for value in arguments_list:
        path = Path(value)
        if path.suffix == ".bin":
            path.write_bytes(b"component")
    arguments = benchmark_inference._parse_arguments(arguments_list)
    monkeypatch.setattr(
        benchmark_inference,
        "_minimax_h3_execution_identities",
        lambda assets: {
            "diffusion": "d",
            "text_encoder": "t",
            "video_vae": "v",
            "audio_vae": "a",
        },
    )
    releases: list[str] = []

    class Handle:
        runtime = SimpleNamespace()

        def __init__(self, role: str) -> None:
            self.role = role

        def terminal_release(self) -> None:
            releases.append(self.role)

    model = Handle("diffusion")
    clip = Handle("text_encoder")
    monkeypatch.setattr(
        native_arm.GenerationLoadDiffusionModel,
        "execute",
        staticmethod(lambda **kwargs: {"model": model}),
    )
    monkeypatch.setattr(
        native_arm.NativeLoadClip,
        "execute",
        staticmethod(lambda **kwargs: {"clip": clip}),
    )
    monkeypatch.setattr(
        native_arm.NativeLoadVae,
        "execute",
        staticmethod(lambda **kwargs: (_ for _ in ()).throw(RuntimeError("VAE failed"))),
    )
    run = benchmark_inference.BenchmarkRun(
        arguments,
        SimpleNamespace(synchronize=lambda: None),
    )
    with pytest.raises(RuntimeError, match="VAE failed"):
        run.load()
    assert releases == ["text_encoder", "diffusion"]
    assert run.executed_placement is None


def test_minimax_h3_output_requires_exact_finite_av_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        def __init__(self, shape: tuple[int, ...], *, finite: bool = True) -> None:
            self.shape = shape
            self.ndim = len(shape)
            self.finite = finite

    monkeypatch.setattr(benchmark_inference.torch, "Tensor", Tensor, raising=False)
    monkeypatch.setattr(
        benchmark_inference.torch,
        "isfinite",
        lambda value: SimpleNamespace(all=lambda: value.finite),
        raising=False,
    )
    run = benchmark_inference.BenchmarkRun(
        SimpleNamespace(family="minimax_h3", length=124, height=768, width=1344),
        SimpleNamespace(),
    )
    frames = Tensor((124, 768, 1344, 3))
    audio = {"waveform": Tensor((1, 2, 165_333)), "sample_rate": 32_000}

    assert "all values finite" in run._validate_minimax_h3_output(frames, audio, "cold")
    with pytest.raises(RuntimeError, match="video shape"):
        run._validate_minimax_h3_output(Tensor((123, 768, 1344, 3)), audio, "cold")
    audio["sample_rate"] = 44_100
    with pytest.raises(RuntimeError, match="sample rate"):
        run._validate_minimax_h3_output(frames, audio, "cold")
    audio["sample_rate"] = 32_000
    audio["waveform"] = Tensor((1, 2, 0))
    with pytest.raises(RuntimeError, match="batch-one stereo"):
        run._validate_minimax_h3_output(frames, audio, "cold")
    audio["waveform"] = Tensor((1, 2, 165_333), finite=False)
    with pytest.raises(RuntimeError, match="non-finite"):
        run._validate_minimax_h3_output(frames, audio, "cold")


def test_infinitetalk_cli_defaults_are_the_pinned_workload(tmp_path: Path) -> None:
    arguments = benchmark_inference._parse_arguments(infinitetalk_arguments(tmp_path))

    assert arguments.prompt == "The camera zooms in. Two characters are talking."
    assert arguments.negative_prompt == ""
    assert arguments.seed == 0
    assert arguments.steps == 6
    assert arguments.width == 832
    assert arguments.height == 480
    assert arguments.length == 81
    assert arguments.cfg == 1.0
    assert arguments.sampler == "dinkster.euler"
    assert arguments.scheduler == "dinkster.normal"
    assert arguments.warm_runs == 3
    assert arguments.motion_frame_count == 9
    assert arguments.audio_scale == 1.0
    assert arguments.lora_strength_model == 1.0


def test_infinitetalk_cli_rejects_a_missing_required_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = infinitetalk_arguments(tmp_path)
    index = arguments.index("--clip-vision")
    del arguments[index : index + 2]

    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments(arguments)

    assert "requires --clip-vision" in capsys.readouterr().err


def test_infinitetalk_cli_rejects_non_4k_plus_1_length(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [*infinitetalk_arguments(tmp_path), "--length", "82"]

    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments(arguments)

    assert "4k+1" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value"),
    [("--sampler", "dinkster.uni_pc"), ("--lora-strength-model", "0.5")],
)
def test_infinitetalk_cli_rejects_non_pinned_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments([*infinitetalk_arguments(tmp_path), option, value])

    assert "for wan21_infinitetalk" in capsys.readouterr().err


def test_humo_cli_defaults_are_the_pinned_workload(tmp_path: Path) -> None:
    arguments = benchmark_inference._parse_arguments(humo_arguments(tmp_path))

    assert arguments.prompt == (
        "A young boy in sci-fi style clothing is talking to the camera in an alien desert."
    )
    assert arguments.negative_prompt.startswith("\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd")
    assert arguments.negative_prompt.endswith(
        "\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012\u7740\u8d70"
    )
    assert arguments.seed == 0
    assert arguments.steps == 6
    assert arguments.width == 640
    assert arguments.height == 640
    assert arguments.length == 97
    assert arguments.cfg == 1.0
    assert arguments.sampler == "dinkster.uni_pc"
    assert arguments.scheduler == "dinkster.simple"
    assert arguments.warm_runs == 3
    assert arguments.lora_strength_model == 1.0


def test_humo_cli_rejects_a_missing_required_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = humo_arguments(tmp_path)
    index = arguments.index("--input-audio")
    del arguments[index : index + 2]

    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments(arguments)

    assert "requires --input-audio" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--sampler", "dinkster.euler"),
        ("--length", "93"),
        ("--lora-strength-model", "0.5"),
    ],
)
def test_humo_cli_rejects_non_pinned_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments([*humo_arguments(tmp_path), option, value])

    assert "for wan21_humo" in capsys.readouterr().err


def test_minimax_h3_cli_defaults_are_the_pinned_workload(tmp_path: Path) -> None:
    arguments = benchmark_inference._parse_arguments(minimax_h3_arguments(tmp_path))
    assert arguments.prompt == "A red square centered on a black background."
    assert arguments.negative_prompt == ""
    assert arguments.seed == 20260813
    assert arguments.steps == 20
    assert arguments.width == 1344
    assert arguments.height == 768
    assert arguments.length == 124
    assert arguments.cfg == 1.0
    assert arguments.sampler == "dinkster.res_multistep"
    assert arguments.scheduler == "dinkster.simple"
    assert arguments.warm_runs == 3
    assert arguments.attention_policy == "auto"
    assert arguments.quality_output_dir is None
    assert arguments.quality_spatial_stride == 4


@pytest.mark.parametrize("policy", ["sdpa", "dinkster_kitchen_int8", "sage"])
def test_minimax_h3_cli_accepts_explicit_attention_evidence(tmp_path: Path, policy: str) -> None:
    output = tmp_path / "quality"
    arguments = benchmark_inference._parse_arguments(
        [
            *minimax_h3_arguments(tmp_path),
            "--attention-policy",
            policy,
            "--quality-output-dir",
            str(output),
            "--require-commit",
            "abc123",
        ]
    )

    assert arguments.attention_policy == policy
    assert arguments.quality_output_dir == output
    assert arguments.require_commit == "abc123"


def test_attention_evidence_options_are_rejected_for_other_families(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments(
            [*flux_arguments(tmp_path), "--attention-policy", "sage"]
        )
    assert "only with MiniMax H3" in capsys.readouterr().err


def test_quality_capture_writes_strided_float32_npy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Tensor:
        def __init__(self, array: np.ndarray) -> None:
            self.array = array
            self.shape = array.shape

        def __getitem__(self, item: Any) -> "Tensor":
            return Tensor(self.array[item])

        def detach(self) -> "Tensor":
            return self

        def to(self, **kwargs: Any) -> "Tensor":
            assert kwargs["device"] == "cpu"
            return self

        def numpy(self) -> np.ndarray:
            return self.array

    monkeypatch.setattr(benchmark_inference.torch, "float32", object(), raising=False)
    source = np.arange(2 * 8 * 12 * 3, dtype=np.float64).reshape(2, 8, 12, 3)
    path = tmp_path / "capture.npy"

    metadata = benchmark_inference._capture_quality_tensor(Tensor(source), path, spatial_stride=4)

    captured = np.load(path)
    np.testing.assert_array_equal(captured, source[:, ::4, ::4, :].astype(np.float32))
    assert metadata["source_shape"] == [2, 8, 12, 3]
    assert metadata["captured_shape"] == [2, 2, 3, 3]
    assert metadata["spatial_stride"] == 4
    assert metadata["bytes"] == path.stat().st_size
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        benchmark_inference._capture_quality_tensor(Tensor(source), path, spatial_stride=4)


def test_quality_capture_uses_a_fresh_post_measurement_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = SimpleNamespace(
        family="minimax_h3",
        seed=20260813,
        warm_runs=3,
        quality_output_dir=tmp_path,
        quality_spatial_stride=4,
        attention_policy="sdpa",
    )
    run = benchmark_inference.BenchmarkRun(arguments, SimpleNamespace())
    image = object()
    waveform = object()
    captured: list[tuple[object, Path, int | None]] = []
    raw_outputs: list[tuple[object, object, int, Path]] = []

    def run_once(seed: int) -> tuple[object, float, float, list[float]]:
        assert seed == 20260817
        run.decoded_audio = {"waveform": waveform, "sample_rate": 32_000}
        return image, 0.0, 0.0, []

    def capture(tensor: object, path: Path, *, spatial_stride: int | None) -> dict[str, object]:
        captured.append((tensor, path, spatial_stride))
        return {"path": str(path)}

    def capture_raw(
        frames: object,
        audio: object,
        sample_rate: int,
        output_dir: Path,
    ) -> dict[str, object]:
        raw_outputs.append((frames, audio, sample_rate, output_dir))
        return {"frames": {}, "audio": {}}

    monkeypatch.setattr(run, "_run_once", run_once)
    monkeypatch.setattr(
        run,
        "_validate_minimax_h3_output",
        lambda frames, audio, label: f"{label} valid",
    )
    monkeypatch.setattr(benchmark_inference, "_capture_quality_tensor", capture)
    monkeypatch.setattr(benchmark_inference, "_capture_raw_outputs", capture_raw)

    assert run.capture_quality() == "quality capture valid"
    assert run.quality_capture == {
        "version": 1,
        "seed": 20260817,
        "image": {"path": str(tmp_path / "capture_image.npy")},
        "audio": {"path": str(tmp_path / "capture_audio.npy")},
        "audio_sample_rate": 32_000,
        "raw_outputs": {"frames": {}, "audio": {}},
    }
    assert captured == [
        (image, tmp_path / "capture_image.npy", 4),
        (waveform, tmp_path / "capture_audio.npy", None),
    ]
    assert raw_outputs == [(image, waveform, 32_000, tmp_path)]
    assert run.decoded_audio is None


def test_minimax_h3_cli_requires_the_audio_vae(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = minimax_h3_arguments(tmp_path)
    index = arguments.index("--audio-vae")
    del arguments[index : index + 2]
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments(arguments)
    assert "requires --audio-vae" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--seed", "0"),
        ("--length", "107"),
        ("--sampler", "dinkster.euler"),
        ("--mode", "compile"),
    ],
)
def test_minimax_h3_cli_rejects_non_pinned_or_unsupported_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments([*minimax_h3_arguments(tmp_path), option, value])
    error = capsys.readouterr().err
    if option == "--mode":
        assert "not supported" in error
    else:
        assert "for minimax_h3" in error


def test_flux_cli_defaults_are_the_pinned_workload(tmp_path: Path) -> None:
    arguments = benchmark_inference._parse_arguments(flux_arguments(tmp_path))

    assert arguments.prompt == "a photograph of an astronaut riding a horse"
    assert arguments.negative_prompt == ""
    assert arguments.seed == 667
    assert arguments.steps == 20
    assert arguments.width == 1024
    assert arguments.height == 1024
    assert arguments.cfg == 1.0
    assert arguments.guidance == 3.5
    assert arguments.sampler == "dinkster.euler"
    assert arguments.scheduler == "dinkster.simple"
    assert arguments.warm_runs == 5
    assert arguments.mode == "eager"
    assert arguments.placement == "residency"


def test_flux_cli_requires_the_clip_l_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = flux_arguments(tmp_path)
    index = arguments.index("--clip-l")
    del arguments[index : index + 2]
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments(arguments)
    assert "requires --clip-l" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--seed", "0"),
        ("--steps", "19"),
        ("--guidance", "4.0"),
        ("--sampler", "dinkster.uni_pc"),
        ("--mode", "compile"),
        ("--placement", "direct-diagnostic"),
    ],
)
def test_flux_cli_rejects_non_pinned_or_unsupported_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments([*flux_arguments(tmp_path), option, value])
    error = capsys.readouterr().err
    if option in ("--mode", "--placement"):
        assert "not supported" in error
    else:
        assert "for flux" in error


def test_chroma_cli_defaults_are_the_pinned_workload(tmp_path: Path) -> None:
    arguments = benchmark_inference._parse_arguments(chroma_arguments(tmp_path))

    assert arguments.prompt == "a photograph of an astronaut riding a horse"
    assert arguments.negative_prompt == ""
    assert arguments.seed == 667
    assert arguments.steps == 26
    assert arguments.width == 1024
    assert arguments.height == 1024
    assert arguments.cfg == 3.5
    assert arguments.sampler == "dinkster.euler"
    assert arguments.scheduler == "dinkster.beta"
    assert arguments.warm_runs == 5
    assert arguments.mode == "eager"
    assert arguments.placement == "residency"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--seed", "0"),
        ("--steps", "20"),
        ("--cfg", "1.0"),
        ("--sampler", "dinkster.uni_pc"),
        ("--mode", "compile"),
        ("--placement", "direct-diagnostic"),
    ],
)
def test_chroma_cli_rejects_non_pinned_or_unsupported_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments([*chroma_arguments(tmp_path), option, value])
    error = capsys.readouterr().err
    if option in ("--mode", "--placement"):
        assert "not supported" in error
    else:
        assert "for chroma" in error


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--clip-l", "clip_l.safetensors"), "--clip-l is only meaningful"),
        (("--guidance", "3.5"), "--guidance is only meaningful"),
    ],
)
def test_flux_only_options_are_rejected_elsewhere(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, str],
    message: str,
) -> None:
    with pytest.raises(SystemExit):
        benchmark_inference._parse_arguments([*anima_arguments(tmp_path), *extra])
    assert message in capsys.readouterr().err


def test_speaker_masks_are_exact_non_overlapping_halves() -> None:
    first, second = benchmark_inference.speaker_mask_arrays(832, 480)
    first = np.asarray(first)
    second = np.asarray(second)

    assert first.shape == second.shape == (1, 480, 832)
    assert np.all(first[:, :, :416] == 1.0)
    assert np.all(first[:, :, 416:] == 0.0)
    assert np.all(second[:, :, :416] == 0.0)
    assert np.all(second[:, :, 416:] == 1.0)
    assert np.all(first + second == 1.0)


def test_image_input_uses_the_production_asset_decoder(tmp_path: Path) -> None:
    path = tmp_path / "input.png"
    Image.new("RGB", (2, 1), (255, 0, 128)).save(path)

    decoded = np.asarray(benchmark_inference._decode_image_file(path))

    assert decoded.shape == (1, 1, 2, 3)
    assert decoded.dtype == np.float32
    assert decoded[0, 0, 0].tolist() == pytest.approx([1.0, 0.0, 128 / 255])


def test_zero_conditioning_preserves_descriptors_and_zeros_payloads() -> None:
    descriptor = PayloadDescriptor(
        PayloadReference("text"),
        (1, 2, 2),
        "F32",
        "conditioning-text",
    )
    source = make_conditioning_carrier(
        ConditioningSet((ConditioningRecord(channels=((ConditioningChannel.TEXT, descriptor),)),)),
        (
            PayloadBinding(
                "text",
                (1, 2, 2),
                "F32",
                "conditioning-text",
                bytes(range(16)),
            ),
        ),
    )

    zeroed = benchmark_inference._zero_conditioning(source)

    assert zeroed.conditioning.records[0].channels[0][1].shape == (1, 2, 2)
    assert len(zeroed.bindings) == 1
    assert zeroed.bindings[0].data == bytes(16)
    assert zeroed != source


def test_humo_audio_conditioning_drives_the_production_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = ModuleType("dinkster_model_wan.provider")
    observed: dict[str, object] = {}

    def encode_audio(**kwargs: object) -> dict[str, object]:
        observed["encode"] = kwargs
        return {"audio_encoder_output": "whisper-features"}

    def humo(**kwargs: object) -> dict[str, object]:
        observed["humo"] = kwargs
        return {"positive": "humo-positive", "negative": "humo-negative", "latent": "latent"}

    typed_provider: Any = provider
    typed_provider.execute_encode_wav2vec2_audio = encode_audio
    typed_provider.execute_wan21_humo = humo
    monkeypatch.setitem(sys.modules, "dinkster_model_wan.provider", provider)
    monkeypatch.setattr(
        benchmark_inference,
        "_decode_audio_file",
        lambda path: {"waveform": "decoded", "sample_rate": 48_000},
    )
    monkeypatch.setattr(benchmark_inference, "_decode_image_file", lambda path: "decoded-image")
    arguments = SimpleNamespace(
        family="wan21_humo",
        input_audio=tmp_path / "input.wav",
        input_image=tmp_path / "input.png",
        width=640,
        height=640,
        length=97,
    )
    access = SimpleNamespace(synchronize=lambda: None)
    run = benchmark_inference.BenchmarkRun(arguments, access)
    run.audio_encoder = "whisper-handle"
    run.vae = "codec-handle"
    run.cond = "positive"
    run.uncond = "negative"

    detail = run.encode_audio()

    assert observed["encode"] == {
        "audio_encoder": "whisper-handle",
        "audio": {"waveform": "decoded", "sample_rate": 48_000},
    }
    assert observed["humo"] == {
        "positive": "positive",
        "negative": "negative",
        "vae": "codec-handle",
        "width": 640,
        "height": 640,
        "length": 97,
        "batch_size": 1,
        "audio_encoder_output": "whisper-features",
        "ref_image": "decoded-image",
    }
    assert (run.cond, run.uncond, run.latent) == ("humo-positive", "humo-negative", "latent")
    assert "HuMo conditioning" in detail


def test_artifact_pin_rejects_wrong_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="does not match its pin"):
        benchmark_inference._artifact_entry(
            "input_image",
            path,
            (5, "0" * 64, "https://example.invalid/artifact.bin"),
        )


def test_anima_artifact_preflight_computes_both_digests_in_one_read(tmp_path: Path) -> None:
    payload = b"pinned anima artifact"
    path = tmp_path / "artifact.safetensors"
    path.write_bytes(payload)

    entry = benchmark_inference._artifact_entry(
        "diffusion",
        path,
        (
            len(payload),
            hashlib.sha256(payload).hexdigest(),
            "https://example.invalid/artifact.safetensors",
        ),
        include_asset_preflight=True,
    )

    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    preflight = cast(Any, entry["_asset_preflight"])
    assert preflight.digest == f"blake3:{blake3(payload).hexdigest()}"
    assert preflight.size == len(payload)
    assert preflight.verification.digest == preflight.digest


def test_artifact_preflight_rejects_a_file_changed_during_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.safetensors"
    path.write_bytes(b"original")
    original_open = cast(Any, Path.open)

    class MutatingFile:
        def __init__(self, handle: Any) -> None:
            self.handle = handle
            self.mutated = False

        def read(self, size: int = -1) -> bytes:
            data = self.handle.read(size)
            if data and not self.mutated:
                self.mutated = True
                stat = os.stat(path)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            return data

        def __enter__(self) -> "MutatingFile":
            return self

        def __exit__(self, *exc_info: object) -> None:
            self.handle.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self.handle, name)

    def mutating_open(candidate: Path, mode: str = "r", *args: object, **kwargs: object) -> Any:
        handle = original_open(candidate, mode, *args, **kwargs)
        return MutatingFile(handle) if candidate == path and mode == "rb" else handle

    monkeypatch.setattr(Path, "open", mutating_open)

    with pytest.raises(ValueError, match="changed while its digests were computed"):
        benchmark_inference._artifact_entry("diffusion", path, include_asset_preflight=True)


def test_artifact_preflight_rejects_a_missing_verification_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_assets.integrity

    path = tmp_path / "artifact.safetensors"
    path.write_bytes(b"artifact")
    monkeypatch.setattr(dinkster_assets.integrity, "verification_record", lambda digest, stat: None)

    with pytest.raises(RuntimeError, match="cannot bind diffusion artifact verification"):
        benchmark_inference._artifact_entry("diffusion", path, include_asset_preflight=True)


def test_artifact_preflight_record_rejects_a_replaced_path(tmp_path: Path) -> None:
    from dinkster_assets.integrity import AssetIntegrityError

    path = tmp_path / "artifact.safetensors"
    path.write_bytes(b"original")
    entry = benchmark_inference._artifact_entry(
        "diffusion",
        path,
        include_asset_preflight=True,
    )
    preflight = cast(Any, entry["_asset_preflight"])
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(b"replacement")
    replacement.replace(path)

    asset = cast(Any, benchmark_inference._asset_ref(path, preflight))
    with pytest.raises(AssetIntegrityError, match="stale_ingest_record"):
        asset.open()


def test_infinitetalk_entrypoint_refuses_unpinned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = infinitetalk_arguments(tmp_path)
    for value in arguments:
        path = Path(value)
        if path.suffix == ".bin":
            path.write_bytes(b"wrong")
    monkeypatch.setattr(sys, "argv", ["benchmark_inference.py", *arguments])
    monkeypatch.setattr(
        benchmark_inference,
        "_admit_backend",
        lambda backend: SimpleNamespace(backend_runtime=f"test-{backend}"),
    )
    monkeypatch.setattr(benchmark_inference.torch, "__version__", "test", raising=False)

    with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
        benchmark_inference.main()


def test_humo_entrypoint_refuses_unpinned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = humo_arguments(tmp_path)
    for value in arguments:
        path = Path(value)
        if path.suffix == ".bin":
            path.write_bytes(b"wrong")
    monkeypatch.setattr(sys, "argv", ["benchmark_inference.py", *arguments])
    monkeypatch.setattr(
        benchmark_inference,
        "_admit_backend",
        lambda backend: SimpleNamespace(backend_runtime=f"test-{backend}"),
    )
    monkeypatch.setattr(benchmark_inference.torch, "__version__", "test", raising=False)

    with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
        benchmark_inference.main()


def test_anima_entrypoint_refuses_unpinned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = anima_arguments(tmp_path)
    for value in arguments:
        path = Path(value)
        if path.suffix == ".safetensors":
            path.write_bytes(b"wrong")
    monkeypatch.setattr(sys, "argv", ["benchmark_inference.py", *arguments])
    monkeypatch.setattr(
        benchmark_inference,
        "_admit_backend",
        lambda backend: SimpleNamespace(backend_runtime=f"test-{backend}"),
    )
    monkeypatch.setattr(benchmark_inference.torch, "__version__", "test", raising=False)

    with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
        benchmark_inference.main()


def test_minimax_h3_entrypoint_refuses_unpinned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = minimax_h3_arguments(tmp_path)
    for value in arguments:
        path = Path(value)
        if path.suffix == ".bin":
            path.write_bytes(b"wrong")
    monkeypatch.setattr(sys, "argv", ["benchmark_inference.py", *arguments])
    monkeypatch.setattr(
        benchmark_inference,
        "_admit_backend",
        lambda backend: SimpleNamespace(backend_runtime=f"test-{backend}"),
    )
    monkeypatch.setattr(benchmark_inference.torch, "__version__", "test", raising=False)

    with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
        benchmark_inference.main()


def test_infinitetalk_pins_cover_every_report_role() -> None:
    assert set(benchmark_inference._INFINITETALK_ARTIFACT_PINS) == {
        "diffusion",
        "text_encoder",
        "vae",
        "lora",
        "model_patch",
        "audio_encoder",
        "clip_vision",
        "input_image",
        "input_audio_1",
        "input_audio_2",
    }
    for size, sha256, url in benchmark_inference._INFINITETALK_ARTIFACT_PINS.values():
        assert size > 0
        assert len(sha256) == 64
        assert url.startswith("https://")


def test_humo_pins_cover_every_report_role() -> None:
    assert set(benchmark_inference._HUMO_ARTIFACT_PINS) == {
        "diffusion",
        "text_encoder",
        "vae",
        "lora",
        "audio_encoder",
        "input_image",
        "input_audio",
    }
    for size, sha256, url in benchmark_inference._HUMO_ARTIFACT_PINS.values():
        assert size > 0
        assert len(sha256) == 64
        assert "resolve/main/" not in url


def test_anima_pins_cover_every_report_role() -> None:
    assert set(benchmark_inference._ANIMA_ARTIFACT_PINS) == {
        "diffusion",
        "text_encoder",
        "vae",
    }
    for size, sha256, url in benchmark_inference._ANIMA_ARTIFACT_PINS.values():
        assert size > 0
        assert len(sha256) == 64
        assert "resolve/main/" not in url


def test_minimax_h3_pins_cover_every_report_role() -> None:
    assert set(benchmark_inference._MINIMAX_H3_ARTIFACT_PINS) == {
        "diffusion",
        "text_encoder",
        "video_vae",
        "audio_vae",
    }
    for size, sha256, url in benchmark_inference._MINIMAX_H3_ARTIFACT_PINS.values():
        assert size > 0
        assert len(sha256) == 64
        assert "resolve/main/" not in url


def test_flux_pins_cover_every_report_role() -> None:
    assert set(benchmark_inference._FLUX_ARTIFACT_PINS) == {
        "diffusion",
        "clip_l",
        "text_encoder",
        "vae",
    }
    for size, sha256, url in benchmark_inference._FLUX_ARTIFACT_PINS.values():
        assert size > 0
        assert len(sha256) == 64
        assert "resolve/main/" not in url


def test_chroma_pins_cover_every_report_role() -> None:
    assert set(benchmark_inference._CHROMA_ARTIFACT_PINS) == {
        "diffusion",
        "text_encoder",
        "vae",
    }
    for size, sha256, url in benchmark_inference._CHROMA_ARTIFACT_PINS.values():
        assert size > 0
        assert len(sha256) == 64
        assert "resolve/main/" not in url


def test_minimax_h3_identities_are_accepted_by_the_production_dit_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planned DiT identity must be the exact string the loader constructs.

    load_minimax_h3_model rejects any expected identity it does not itself
    derive, so this drives the real _minimax_h3_execution_identities (no
    monkeypatch of the function under test) and feeds its diffusion identity
    to the real loader. A planning-facts mismatch between the harness and the
    loader fails here instead of at benchmark runtime. Requires torch: it
    skips in the torch-free root venv and executes in CI's torch-cpu job
    (and any local torch venv).
    """
    torch = pytest.importorskip("torch")
    assembly = pytest.importorskip("dinkster_inference_torch.minimax_h3_assembly")
    import dinkster_assets.model
    import dinkster_inference
    from dinkster_assets import AssetRef, digest_file
    from dinkster_inference import BFLOAT16, FLOAT16, FLOAT32
    from dinkster_inference.minimax_h3_conditioner import minimax_h3_conditioner_layout
    from dinkster_inference.minimax_h3_dit import minimax_h3_dit_layout
    from dinkster_inference.weights import TensorGeometry, WeightEntry
    from dinkster_inference_torch.minimax_h3_audio import MiniMaxH3AudioVAE
    from dinkster_inference_torch.minimax_h3_video_vae import MiniMaxH3VideoVAE

    # The module under test was executed with a stub torch (the root suite is
    # torch-free); identity planning needs the real one.
    monkeypatch.setattr(benchmark_inference, "torch", torch)

    class HeaderSource:
        def __init__(self, path: Path, geometries: Mapping[str, Any]) -> None:
            self.path = path
            self.geometries = dict(geometries)

        def keys(self) -> tuple[str, ...]:
            return tuple(self.geometries)

        def entry(self, key: str) -> Any:
            geometry = self.geometries[key]
            return WeightEntry(key, geometry, 0, geometry.nbytes)

        def metadata(self) -> Mapping[str, str]:
            return {}

        def read_uint8_configuration(self, key: str) -> bytes:
            raise KeyError(key)

    @dataclass(frozen=True)
    class FixedResolver:
        path: Path

        def resolve(self, digest: str) -> Path:
            del digest
            return self.path

    paths = {
        "fl2va-dit": tmp_path / "dit.safetensors",
        "qwen3vl-32b-conditioner": tmp_path / "conditioner.safetensors",
        "video-vae": tmp_path / "video.safetensors",
        "audio-vae": tmp_path / "audio.safetensors",
    }
    for role, path in paths.items():
        path.write_bytes(role.encode())
    dit_path = paths["fl2va-dit"]
    dit_size = dit_path.stat().st_size
    dit_digest = digest_file(dit_path)
    dit_layout = minimax_h3_dit_layout()
    with torch.device("meta"):
        video_state = MiniMaxH3VideoVAE().state_dict()
        audio_state = MiniMaxH3AudioVAE().state_dict()
    sources = {
        "fl2va-dit": HeaderSource(
            dit_path,
            {
                key: TensorGeometry(
                    shape,
                    FLOAT32 if key in dit_layout.fp32_storage_keys else BFLOAT16,
                )
                for key, shape in dit_layout.keys.items()
            },
        ),
        "qwen3vl-32b-conditioner": HeaderSource(
            paths["qwen3vl-32b-conditioner"],
            {
                key: TensorGeometry(shape, BFLOAT16)
                for key, shape in minimax_h3_conditioner_layout().keys.items()
            },
        ),
        "video-vae": HeaderSource(
            paths["video-vae"],
            {
                key: TensorGeometry(tuple(value.shape), FLOAT16)
                for key, value in video_state.items()
            },
        ),
        "audio-vae": HeaderSource(
            paths["audio-vae"],
            {
                key: TensorGeometry(tuple(value.shape), FLOAT32)
                for key, value in audio_state.items()
            },
        ),
    }
    by_path = {source.path: source for source in sources.values()}

    def fake_header(path: Path, *, asset_digest: str, asset_size: int) -> Any:
        del asset_digest, asset_size
        return by_path[Path(path)]

    monkeypatch.setattr(dinkster_inference, "load_safetensors_header", fake_header)
    # The fake files stand in for the pinned multi-gigabyte artifacts, so
    # skip content verification exactly like the assembly loader tests do.
    monkeypatch.setattr(
        dinkster_assets.model,
        "verified_local_path",
        lambda path, digest, verification: path,
    )
    monkeypatch.setattr(
        dinkster_assets.model,
        "open_verified",
        lambda path, digest, verification: path.open("rb"),
    )

    def common_asset(role: str) -> Any:
        path = paths[role]
        return AssetRef(
            digest_file(path), path.name, path.stat().st_size, resolver=FixedResolver(path)
        )

    assets = {
        "diffusion": AssetRef(
            dit_digest, dit_path.name, dit_size, resolver=FixedResolver(dit_path)
        ),
        "text_encoder": common_asset("qwen3vl-32b-conditioner"),
        "video_vae": common_asset("video-vae"),
        "audio_vae": common_asset("audio-vae"),
    }
    identities = benchmark_inference._minimax_h3_execution_identities(assets)

    def read_header(_handle: Any, *, path: Path) -> Any:
        assert path == dit_path
        return sources["fl2va-dit"]

    monkeypatch.setattr(assembly, "load_safetensors_header_from_file", read_header)
    monkeypatch.setattr(
        assembly,
        "_load_component",
        lambda component, _build, **_kwargs: torch.nn.Identity(),
    )
    attention_backend = dinkster_inference.MINIMAX_H3.engine.attention_backend("diffusion")
    assert attention_backend == "flux"
    model = assembly.load_minimax_h3_model(
        dit_path,
        asset=assets["diffusion"],
        role="fl2va-dit",
        expected_identity=identities["diffusion"],
        attention_backend=attention_backend,
    )
    assert model.model_role == "fl2va-dit"
    assert model.runtime_identity == identities["diffusion"]


def _provider_wan_artifact_paths(arguments: list[str]) -> dict[str, Path]:
    roles: dict[str, Path] = {}
    for option, value in zip(arguments[::2], arguments[1::2], strict=True):
        role = option.removeprefix("--").replace("-", "_")
        if role in ("backend", "family"):
            continue
        path = Path(value)
        path.write_bytes(f"{role} bytes".encode())
        roles[role] = path
    return roles


def _provider_wan_pin_map(paths: Mapping[str, Path]) -> dict[str, tuple[int, str, str]]:
    return {
        role: (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            f"https://example.invalid/{path.name}",
        )
        for role, path in paths.items()
    }


@pytest.mark.parametrize(
    ("pin_attribute", "arguments_builder"),
    [
        ("_INFINITETALK_ARTIFACT_PINS", infinitetalk_arguments),
        ("_HUMO_ARTIFACT_PINS", humo_arguments),
    ],
)
def test_provider_wan_main_binds_descriptor_preflights(
    pin_attribute: str,
    arguments_builder: Callable[[Path], list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = arguments_builder(tmp_path)
    paths = _provider_wan_artifact_paths(arguments)
    monkeypatch.setattr(benchmark_inference, pin_attribute, _provider_wan_pin_map(paths))
    monkeypatch.setattr(sys, "argv", ["benchmark_inference.py", *arguments])
    monkeypatch.setattr(
        benchmark_inference,
        "_admit_backend",
        lambda backend: SimpleNamespace(
            backend_runtime=f"test-{backend}",
            synchronize=lambda: None,
            reset_peak=lambda: None,
        ),
    )
    monkeypatch.setattr(benchmark_inference.torch, "__version__", "test", raising=False)
    monkeypatch.delenv("DINKSTER_AIMDO_ARM", raising=False)
    monkeypatch.setattr(benchmark_inference, "_ResidencySampler", _OffSampler)
    for name in ("dinkster_inference_torch", "dinkster_model_wan", "dinkster_model_wan.provider"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    captured: dict[str, Any] = {}

    class _SpyExit(Exception):
        pass

    class SpyRun:
        def __init__(
            self,
            arguments: Any,
            access: Any,
            preflight_assets: Mapping[str, Any] | None = None,
        ) -> None:
            captured["preflights"] = dict(preflight_assets or {})
            raise _SpyExit

    monkeypatch.setattr(benchmark_inference, "BenchmarkRun", SpyRun)

    with pytest.raises(_SpyExit):
        benchmark_inference.main()

    preflights = captured["preflights"]
    assert set(preflights) == set(paths)
    for role, path in paths.items():
        preflight = preflights[role]
        assert preflight.path == path
        assert preflight.size == path.stat().st_size
        assert preflight.digest == f"blake3:{blake3(path.read_bytes()).hexdigest()}"
        assert preflight.verification is not None


def test_infinitetalk_load_consumes_descriptor_preflights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_assets.integrity
    from dinkster_compat_comfy import native_arm

    argv = infinitetalk_arguments(tmp_path)
    paths = _provider_wan_artifact_paths(argv)
    pin_map = _provider_wan_pin_map(paths)
    monkeypatch.setattr(benchmark_inference, "_INFINITETALK_ARTIFACT_PINS", pin_map)
    arguments = benchmark_inference._parse_arguments(argv)
    preflights = {
        role: benchmark_inference._artifact_entry(
            role, path, pin_map[role], include_asset_preflight=True
        ).pop("_asset_preflight")
        for role, path in paths.items()
    }

    def forbid_rehash(path: Path) -> Any:
        raise AssertionError(f"descriptor preflight ignored; re-hashed {path}")

    monkeypatch.setattr(dinkster_assets.integrity, "digest_file_with_record", forbid_rehash)

    captured: dict[str, Any] = {}
    handle = SimpleNamespace(
        runtime=SimpleNamespace(
            assembled=SimpleNamespace(
                family=SimpleNamespace(id="dinkster.wan21"),
                diffusion=object(),
            )
        ),
        terminal_release=lambda: None,
    )

    def load_native_runtime_handle(runtime_assets: Mapping[str, Any]) -> Any:
        captured["runtime_assets"] = dict(runtime_assets)
        return handle

    def execute_lora(**kwargs: Any) -> dict[str, Any]:
        captured["lora"] = kwargs["lora"]
        return {"model": kwargs["model"]}

    def execute_model_patch(**kwargs: Any) -> dict[str, Any]:
        captured["model_patch"] = kwargs["model_patch"]
        return {"model_patch": object()}

    monkeypatch.setattr(native_arm, "load_native_runtime_handle", load_native_runtime_handle)
    monkeypatch.setattr(
        native_arm, "GenerationLoadLoraModelOnly", SimpleNamespace(execute=execute_lora)
    )
    monkeypatch.setattr(
        native_arm, "NativeLoadZImageControlPatch", SimpleNamespace(execute=execute_model_patch)
    )
    monkeypatch.setattr(native_arm, "_NativeCodecHandle", lambda handle: SimpleNamespace())
    monkeypatch.setattr(benchmark_inference, "_module_dtype", lambda module: "stub-dtype")

    def execute_load_wav2vec2_audio_encoder(*, audio_encoder: Any) -> dict[str, Any]:
        captured["audio_encoder"] = audio_encoder
        return {"audio_encoder": object()}

    provider = ModuleType("dinkster_model_wan.provider")
    cast(Any, provider).execute_load_wav2vec2_audio_encoder = execute_load_wav2vec2_audio_encoder
    package = ModuleType("dinkster_model_wan")
    cast(Any, package).provider = provider
    monkeypatch.setitem(sys.modules, "dinkster_model_wan", package)
    monkeypatch.setitem(sys.modules, "dinkster_model_wan.provider", provider)

    inference_torch = ModuleType("dinkster_inference_torch")

    @contextmanager
    def use_component_publisher(publisher: Any) -> Any:
        yield

    cast(Any, inference_torch).use_component_publisher = use_component_publisher
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch", inference_torch)

    class StubResidencyManager:
        def __init__(self, **kwargs: Any) -> None:
            pass

    residency = ModuleType("dinkster_inference_torch.residency")
    cast(Any, residency).ResidencyManager = StubResidencyManager
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch.residency", residency)

    run = benchmark_inference.BenchmarkRun(
        arguments,
        cast(Any, SimpleNamespace(backend_runtime="test", synchronize=lambda: None)),
        preflights,
    )
    run.load()

    assert run.family_id == "dinkster.wan21"
    runtime_assets = captured["runtime_assets"]
    slot_roles = {
        "diffusion": "diffusion",
        "t5xxl": "text_encoder",
        "vae": "vae",
        "clip_vision": "clip_vision",
    }
    for slot, role in slot_roles.items():
        assert runtime_assets[slot].digest == preflights[role].digest
        assert runtime_assets[slot].size == preflights[role].size
        assert runtime_assets[slot].name == paths[role].name
    for capture_key in ("lora", "model_patch", "audio_encoder"):
        assert captured[capture_key].digest == preflights[capture_key].digest
        assert captured[capture_key].size == preflights[capture_key].size


@pytest.mark.parametrize("family", ["wan21_infinitetalk", "wan21_humo"])
def test_provider_wan_unload_clears_cublas_workspaces_before_the_residual_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    argv = (
        infinitetalk_arguments(tmp_path)
        if family == "wan21_infinitetalk"
        else humo_arguments(tmp_path)
    )
    arguments = benchmark_inference._parse_arguments(argv)

    class Access:
        """CUDA access whose only residual bytes are cuBLAS workspaces."""

        device = SimpleNamespace(type="cuda")

        def __init__(self) -> None:
            self.workspace_bytes = 2 * 32 * 1024 * 1024
            self.leaked_bytes = 0

        def empty_cache(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def allocated(self) -> int:
            return self.workspace_bytes + self.leaked_bytes

    access = Access()
    monkeypatch.setattr(
        benchmark_inference.torch,
        "_C",
        SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: setattr(access, "workspace_bytes", 0)),
        raising=False,
    )

    run = benchmark_inference.BenchmarkRun(arguments, cast(Any, access))
    assert run.unload() == "residual_allocated=0 B"
    assert run.residual_allocated == 0

    leak = benchmark_inference.FAMILY_RESIDUAL_CEILING_BYTES + 1
    access.workspace_bytes = 2 * 32 * 1024 * 1024
    access.leaked_bytes = leak
    leaking_run = benchmark_inference.BenchmarkRun(arguments, cast(Any, access))
    with pytest.raises(RuntimeError, match=f"allocator still holds {leak} B"):
        leaking_run.unload()


_MIB = 1024 * 1024


def _sd15_arguments(tmp_path: Path) -> list[str]:
    return [
        "--backend",
        "cuda",
        "--family",
        "sd15",
        "--checkpoint",
        str(tmp_path / "sd15.safetensors"),
    ]


def test_residency_cli_defaults_and_bounds(tmp_path: Path) -> None:
    arguments = benchmark_inference._parse_arguments(_sd15_arguments(tmp_path))
    assert arguments.regime == "open"
    assert arguments.leave_free_mib == 768
    assert arguments.spill_scope == "auto"
    for option, value in (
        ("--regime", "cramped"),
        ("--spill-scope", "device"),
        ("--leave-free-mib", "0"),
        ("--spill-threshold-mib", "64"),
    ):
        with pytest.raises(SystemExit):
            benchmark_inference._parse_arguments([*_sd15_arguments(tmp_path), option, value])


def test_ballast_size_pins_free_memory_down_to_the_target() -> None:
    assert benchmark_inference._ballast_size_bytes(4096 * _MIB, 768) == 3328 * _MIB
    assert benchmark_inference._ballast_size_bytes(512 * _MIB, 768) == 0


@pytest.mark.parametrize("growth", [-100, 0, 64 * _MIB + 1])
@pytest.mark.parametrize("scope", ["process", "machine"])
def test_residency_section_growth_does_not_assess_spill(growth: int, scope: str) -> None:
    section = benchmark_inference._residency_section(
        mechanism="auto",
        regime="open",
        leave_free_mib=None,
        ballast_bytes=None,
        spill_scope=scope,
        shared_before_bytes=0,
        shared_warm_bytes=1_000_000_000,
        shared_after_bytes=1_000_000_000 + growth,
    )
    assert section["shared_growth_bytes"] == growth
    assert section["shared_spill_detected"] is None
    assert "spill_threshold_mib" not in section


def test_residency_section_propagates_unreachable_samples_as_null() -> None:
    section = benchmark_inference._residency_section(
        mechanism="off",
        regime="open",
        leave_free_mib=None,
        ballast_bytes=None,
        spill_scope="machine",
        shared_before_bytes=None,
        shared_warm_bytes=None,
        shared_after_bytes=None,
    )
    assert section["shared_growth_bytes"] is None
    assert section["shared_spill_detected"] is None


def test_residency_sampler_resolves_scope_and_reads_the_probe_counter() -> None:
    off = benchmark_inference._ResidencySampler("off")
    assert off.scope == "off"
    assert off.sample() is None

    sampler = benchmark_inference._ResidencySampler("process")
    assert sampler.scope == "process"
    reads: list[tuple[str, int]] = []

    def read(scope: str, pid: int) -> int:
        reads.append((scope, pid))
        return 123

    sampler._read = read
    assert sampler.sample() == 123
    assert reads == [("process", os.getpid())]


def test_entrypoint_rejects_an_unknown_offload_mechanism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark_inference.py", *_sd15_arguments(tmp_path)])
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "banana")

    def refuse_admission(backend: str) -> SimpleNamespace:
        raise AssertionError("an unknown mechanism must exit before backend admission")

    monkeypatch.setattr(benchmark_inference, "_admit_backend", refuse_admission)
    with pytest.raises(SystemExit, match="DINKSTER_AIMDO_ARM"):
        benchmark_inference.main()


def test_entrypoint_constrained_regime_ballast_and_sample_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_inference.py",
            "--backend",
            "cuda",
            "--family",
            "sd15",
            "--checkpoint",
            str(checkpoint),
            "--json",
            str(output),
            "--regime",
            "constrained",
            "--leave-free-mib",
            "1024",
            "--spill-scope",
            "process",
        ],
    )
    monkeypatch.delenv("DINKSTER_AIMDO_ARM", raising=False)
    events: list[str] = []

    class Sampler:
        def __init__(self, spill_scope: str) -> None:
            assert spill_scope == "process"
            self.scope = "process"
            self._samples = iter((5, 1_000_000_000, 1_000_000_000 + 64 * _MIB + 1))

        def sample(self) -> int:
            value = next(self._samples)
            events.append(f"sample:{value}")
            return value

    class Access:
        backend_runtime = "test-runtime"
        device = SimpleNamespace(type="cuda")

        def synchronize(self) -> None:
            pass

        def reset_peak(self) -> None:
            events.append("reset_peak")

        def peak_allocated(self) -> int:
            return 1

        def peak_reserved(self) -> int:
            return 2

        def empty_cache(self) -> None:
            events.append("empty_cache")

    class Run:
        def __init__(self, arguments: object, access: object, preflights: object) -> None:
            del arguments, access, preflights
            self.checks: dict[str, dict[str, object]] = {}
            self.family_id = "dinkster.sd15"
            self.executed_placement = "production_residency"
            self.cold: dict[str, object] = {}
            self.residual_allocated = 0

        def record(self, name: str, action: Callable[[], str], *, always: bool = False) -> None:
            del always
            events.append(f"record:{name}")
            action()
            self.checks[name] = {"ok": True, "detail": "test"}

        def load(self) -> str:
            return "test"

        def encode_text(self) -> str:
            return "test"

        def cold_run(self) -> str:
            return "test"

        def finite_output(self) -> str:
            return "test"

        def warm_runs(self) -> str:
            return "test"

        def unload(self) -> str:
            return "test"

        def warm_section(self) -> dict[str, object]:
            return {}

    free_bytes = 3 * 1024 * _MIB + 5

    def fake_empty(size: int, *, dtype: object, device: object) -> object:
        del dtype, device
        events.append(f"ballast:{size}")
        return object()

    monkeypatch.setattr(
        benchmark_inference.torch,
        "cuda",
        SimpleNamespace(mem_get_info=lambda device: (free_bytes, 32 * 1024 * _MIB)),
        raising=False,
    )
    monkeypatch.setattr(benchmark_inference.torch, "uint8", "uint8", raising=False)
    monkeypatch.setattr(benchmark_inference.torch, "empty", fake_empty, raising=False)
    monkeypatch.setattr(benchmark_inference.torch, "__version__", "test", raising=False)
    monkeypatch.setattr(benchmark_inference, "_ResidencySampler", Sampler)
    monkeypatch.setattr(benchmark_inference, "_admit_backend", lambda backend: Access())
    monkeypatch.setattr(benchmark_inference, "_driver_identity", lambda backend: "test-driver")
    monkeypatch.setattr(benchmark_inference, "_device_entries", lambda backend: [])
    monkeypatch.setattr(benchmark_inference, "_peak_rss_bytes", lambda: 3)
    monkeypatch.setattr(
        benchmark_inference,
        "_artifact_entry",
        lambda role, path, pin=None, *, include_asset_preflight=False: {
            "role": role,
            "path": str(path),
            "sha256": "a" * 64,
            "bytes": 1,
        },
    )
    monkeypatch.setattr(benchmark_inference, "BenchmarkRun", Run)
    monkeypatch.setattr(
        benchmark_inference, "validate_benchmark_report", lambda *args, **kwargs: ()
    )
    monkeypatch.setitem(
        sys.modules, "dinkster_inference_torch", ModuleType("dinkster_inference_torch")
    )

    assert benchmark_inference.main() == 0

    # Ballast is chunked and allocated before the pre-load sample; the
    # warm sample follows the cold run; ballast is freed after the final
    # sample and before unload.
    assert events == [
        f"ballast:{1024 * _MIB}",
        f"ballast:{1024 * _MIB}",
        "ballast:5",
        "reset_peak",
        "sample:5",
        "record:load",
        "record:encode_text",
        "record:cold_run",
        "sample:1000000000",
        "record:finite_output",
        "record:warm_runs",
        f"sample:{1_000_000_000 + 64 * _MIB + 1}",
        "empty_cache",
        "record:unload",
    ]
    report = json.loads(output.read_text())
    assert report["residency"] == {
        "mechanism": "auto",
        "aimdo_bootstrap_succeeded": None,
        "routes": {},
        "regime": "constrained",
        "leave_free_mib": 1024,
        "ballast_bytes": 2 * 1024 * _MIB + 5,
        "spill_scope": "process",
        "shared_before_bytes": 5,
        "shared_warm_bytes": 1_000_000_000,
        "shared_after_bytes": 1_000_000_000 + 64 * _MIB + 1,
        "shared_growth_bytes": 64 * _MIB + 1,
        "shared_spill_detected": None,
    }
