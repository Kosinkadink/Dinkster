"""Legacy checkpoint conversion at the compat boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from dinkster_compat_comfy import legacy_sources
from dinkster_compat_comfy.legacy_sources import (
    LegacyCheckpointError,
    classify_conversion_error,
    classify_weight_source,
    convert_legacy_checkpoint,
    discover_converted_sidecar,
    resolve_weight_source,
)
from dinkster_inference import load_safetensors_header


@pytest.fixture
def torch_modules() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    inference_torch = pytest.importorskip("dinkster_inference_torch")
    return torch, inference_torch


def _empty_safetensors(path: Path) -> Path:
    header = json.dumps({}, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


@pytest.mark.parametrize("suffix", [".safetensors", ".sft", ".SAFETENSORS"])
def test_safetensors_passes_through_by_identity_without_sidecar(
    tmp_path: Path, suffix: str
) -> None:
    source = _empty_safetensors(tmp_path / f"weights{suffix}")
    original = source.read_bytes()

    assert resolve_weight_source(source) is source
    assert list(tmp_path.iterdir()) == [source]
    assert source.read_bytes() == original


@pytest.mark.parametrize("name", ["digestwithoutanextension", "weights.ckpt"])
def test_safetensors_content_passes_through_regardless_of_physical_name(
    tmp_path: Path, name: str
) -> None:
    source = _empty_safetensors(tmp_path / name)

    assert resolve_weight_source(source) is source
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize(
    "name", ["weights.safetensors", "digestwithoutanextension", "weights.ckpt"]
)
def test_classify_weight_source_accepts_valid_safetensors_by_content(
    tmp_path: Path, name: str
) -> None:
    source = _empty_safetensors(tmp_path / name)

    assert classify_weight_source(source, "logical.pth") == "safetensors"


@pytest.mark.parametrize("logical_name", ["model.ckpt", "MODEL.PT", "folder/model.pth"])
def test_classify_weight_source_uses_logical_name_for_legacy_bytes(
    tmp_path: Path, logical_name: str
) -> None:
    source = tmp_path / "extensionless-digest"
    source.write_bytes(b"not safetensors")

    assert classify_weight_source(source, logical_name) == "legacy-convertible"


@pytest.mark.parametrize("logical_name", [None, "", "model.bin", "model.safetensors"])
def test_classify_weight_source_refuses_other_invalid_sources(
    tmp_path: Path, logical_name: str | None
) -> None:
    source = tmp_path / "extensionless-digest"
    source.write_bytes(b"not safetensors")

    assert classify_weight_source(source, logical_name) == "unsupported"


@pytest.mark.parametrize(
    "message",
    [
        "legacy-checkpoint-converter-unavailable: torch missing",
        "legacy-checkpoint-sidecar-directory-not-writable: /models",
        "legacy-checkpoint-sidecar-write-failed: disk full",
        "legacy-checkpoint-sidecar-cleanup-failed: busy",
        "legacy-checkpoint-source-read-failed: disappeared",
        "legacy-checkpoint-source-reverification-failed: disappeared",
        "legacy-checkpoint-source-changed: replaced",
    ],
)
def test_classify_conversion_error_owns_retryable_vocabulary(message: str) -> None:
    assert classify_conversion_error(LegacyCheckpointError(message)) == "retryable"


def test_classify_conversion_error_refuses_safe_loader_rejection_only() -> None:
    refusal = LegacyCheckpointError("legacy-checkpoint-safe-load-failed: unsafe globals")

    assert classify_conversion_error(refusal) == "refused"
    assert classify_conversion_error(RuntimeError("worker runtime broke")) == "retryable"


def test_discover_converted_sidecar_returns_valid_deterministic_sibling(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"legacy")
    sidecar = _empty_safetensors(tmp_path / "model.0123456789abcdef.dinkster.safetensors")

    assert discover_converted_sidecar(source) == sidecar


def test_discover_converted_sidecar_rejects_corrupt_and_malformed_names(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"legacy")
    (tmp_path / "model.0123456789abcdef.dinkster.safetensors").write_bytes(b"corrupt")
    _empty_safetensors(tmp_path / "model.not-a-digest.dinkster.safetensors")

    assert discover_converted_sidecar(source) is None


def test_conversion_round_trips_and_logs_actual_conversion(
    tmp_path: Path,
    torch_modules: tuple[Any, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    torch, inference_torch = torch_modules
    source = tmp_path / "model.ckpt"
    tensors = {
        "z.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "a.bias": torch.tensor([1, 2], dtype=torch.int64),
    }
    torch.save(tensors, source)

    with caplog.at_level("WARNING", logger="dinkster.compat_comfy.legacy_sources"):
        sidecar = resolve_weight_source(source)

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sidecar.name == f"model.{digest[:16]}.dinkster.safetensors"
    header = load_safetensors_header(sidecar)
    loaded = inference_torch.load_tensors(sidecar, header.keys())
    assert tuple(header.keys()) == ("a.bias", "z.weight")
    for key, expected in tensors.items():
        assert loaded[key].shape == expected.shape
        assert loaded[key].dtype == expected.dtype
        assert loaded[key].contiguous().numpy().tobytes() == expected.numpy().tobytes()
    assert f"source={source}" in caplog.text
    assert f"sidecar={sidecar}" in caplog.text
    assert f"source_sha256={digest}" in caplog.text
    assert f"source_bytes={source.stat().st_size}" in caplog.text
    assert f"output_bytes={sidecar.stat().st_size}" in caplog.text


def test_conversion_is_byte_deterministic(tmp_path: Path, torch_modules: tuple[Any, Any]) -> None:
    torch, _ = torch_modules
    source = tmp_path / "model.pt"
    torch.save(
        {
            "later": torch.arange(4, dtype=torch.float16),
            "earlier": torch.tensor([[True, False]]),
        },
        source,
    )
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"

    first_result = convert_legacy_checkpoint(source, first)
    second_result = convert_legacy_checkpoint(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.output_sha256 == second_result.output_sha256
    assert first_result.output_bytes == second_result.output_bytes


def test_valid_sidecar_cache_hit_does_not_rewrite(
    tmp_path: Path,
    torch_modules: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, _ = torch_modules
    source = tmp_path / "model.pth"
    torch.save({"weight": torch.ones(2)}, source)
    sidecar = resolve_weight_source(source)
    before = sidecar.stat()

    def unexpected_conversion(_source: Path, _output: Path) -> Any:
        raise AssertionError("cache hit attempted conversion")

    monkeypatch.setattr(legacy_sources, "convert_legacy_checkpoint", unexpected_conversion)
    assert resolve_weight_source(source) == sidecar
    after = sidecar.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_corrupt_existing_sidecar_is_reconverted(
    tmp_path: Path, torch_modules: tuple[Any, Any]
) -> None:
    torch, _ = torch_modules
    source = tmp_path / "model.ckpt"
    torch.save({"weight": torch.arange(3)}, source)
    sidecar = resolve_weight_source(source)
    expected = sidecar.read_bytes()
    sidecar.write_bytes(b"corrupt")

    assert resolve_weight_source(source) == sidecar
    assert sidecar.read_bytes() == expected
    load_safetensors_header(sidecar)


def test_nested_state_dict_extracts_and_accounts_for_exact_dropped_keys(
    tmp_path: Path, torch_modules: tuple[Any, Any]
) -> None:
    torch, inference_torch = torch_modules
    source = tmp_path / "nested.ckpt"
    torch.save(
        {
            "state_dict": {
                "weight": torch.ones(1),
                "description": "not a tensor",
            },
            "optimizer": {"step": 1},
            "epoch": 4,
        },
        source,
    )
    output = tmp_path / "nested.safetensors"

    result = convert_legacy_checkpoint(source, output)

    assert result.dropped_keys == ("optimizer", "epoch", "description")
    header = load_safetensors_header(output)
    assert tuple(header.keys()) == ("weight",)
    assert inference_torch.load_tensors(output)["weight"].shape == (1,)


def test_non_mapping_root_refuses(tmp_path: Path, torch_modules: tuple[Any, Any]) -> None:
    torch, _ = torch_modules
    source = tmp_path / "list.ckpt"
    torch.save([torch.ones(1)], source)

    with pytest.raises(LegacyCheckpointError, match="root-not-mapping"):
        convert_legacy_checkpoint(source, tmp_path / "output.safetensors")


def test_non_tensor_entries_are_dropped(tmp_path: Path, torch_modules: tuple[Any, Any]) -> None:
    torch, _ = torch_modules
    source = tmp_path / "mixed.ckpt"
    torch.save({"weight": torch.ones(1), "name": "model", "step": 3}, source)
    output = tmp_path / "mixed.safetensors"

    result = convert_legacy_checkpoint(source, output)

    assert result.dropped_keys == ("name", "step")
    assert tuple(load_safetensors_header(output).keys()) == ("weight",)


def test_tensor_normalization_failure_is_named_with_key(
    tmp_path: Path, torch_modules: tuple[Any, Any]
) -> None:
    torch, _ = torch_modules
    source = tmp_path / "meta.ckpt"
    torch.save({"meta.weight": torch.empty(1, device="meta")}, source)

    with pytest.raises(LegacyCheckpointError, match="conversion-failed.*'meta.weight'.*normalized"):
        convert_legacy_checkpoint(source, tmp_path / "output.safetensors")


def test_non_string_tensor_key_refuses(tmp_path: Path, torch_modules: tuple[Any, Any]) -> None:
    torch, _ = torch_modules
    source = tmp_path / "bad-key.ckpt"
    torch.save({1: torch.ones(1)}, source)

    with pytest.raises(LegacyCheckpointError, match="key-not-string"):
        convert_legacy_checkpoint(source, tmp_path / "output.safetensors")


def test_non_string_wrapper_key_refuses(tmp_path: Path, torch_modules: tuple[Any, Any]) -> None:
    torch, _ = torch_modules
    source = tmp_path / "bad-wrapper-key.ckpt"
    torch.save({"state_dict": {"weight": torch.ones(1)}, 1: "metadata"}, source)

    with pytest.raises(LegacyCheckpointError, match="key-not-string"):
        convert_legacy_checkpoint(source, tmp_path / "output.safetensors")


def test_unsupported_extension_refuses_with_supported_formats(tmp_path: Path) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(b"anything")

    with pytest.raises(LegacyCheckpointError, match="unsupported-weight-source-format") as error:
        resolve_weight_source(source)

    for suffix in (".ckpt", ".pt", ".pth", ".safetensors", ".sft"):
        assert suffix in str(error.value)
    assert "convert_ckpt_to_safetensors.py" in str(error.value)
    assert error.value.__cause__ is not None


def test_malformed_extensionless_source_never_reaches_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "digestwithoutanextension"
    source.write_bytes(b"not safetensors")

    def unexpected_conversion(_source: Path, _output: Path) -> Any:
        raise AssertionError("extensionless bytes reached torch conversion")

    monkeypatch.setattr(legacy_sources, "convert_legacy_checkpoint", unexpected_conversion)
    with pytest.raises(LegacyCheckpointError, match="must use .ckpt, .pt, or .pth"):
        resolve_weight_source(source)


def test_logical_legacy_name_allows_extensionless_source_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "extensionless-digest"
    source.write_bytes(b"legacy checkpoint")

    def convert(_source: Path, output: Path) -> legacy_sources.LegacyConversionResult:
        _empty_safetensors(output)
        return legacy_sources.LegacyConversionResult((), output.stat().st_size, "0" * 64)

    monkeypatch.setattr(legacy_sources, "convert_legacy_checkpoint", convert)

    sidecar = resolve_weight_source(source, "logical.ckpt")

    assert sidecar.name.startswith("extensionless-digest.")
    assert sidecar.name.endswith(".dinkster.safetensors")
    load_safetensors_header(sidecar)


def test_unwritable_sidecar_directory_refuses_actionably(
    tmp_path: Path, torch_modules: tuple[Any, Any]
) -> None:
    torch, _ = torch_modules
    directory = tmp_path / "read-only"
    directory.mkdir()
    source = directory / "model.ckpt"
    torch.save({"weight": torch.ones(1)}, source)
    directory.chmod(0o555)
    try:
        with pytest.raises(LegacyCheckpointError, match="directory-not-writable") as error:
            resolve_weight_source(source)
    finally:
        directory.chmod(0o755)

    assert "tools/convert_ckpt_to_safetensors.py" in str(error.value)


def test_save_failure_cleans_atomic_temp_file(
    tmp_path: Path,
    torch_modules: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, _ = torch_modules
    source = tmp_path / "model.ckpt"
    torch.save({"weight": torch.ones(1)}, source)

    def failing_save(_tensors: object, output: Path) -> None:
        output.write_bytes(b"partial")
        raise OSError("simulated save failure")

    monkeypatch.setattr(legacy_sources, "_save_file", failing_save)
    with pytest.raises(LegacyCheckpointError, match="conversion-failed"):
        resolve_weight_source(source)

    assert list(tmp_path.glob("*.dinkster.safetensors")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_temp_cleanup_failure_is_named(
    tmp_path: Path,
    torch_modules: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, _ = torch_modules
    source = tmp_path / "model.ckpt"
    torch.save({"weight": torch.ones(1)}, source)

    def failing_save(_tensors: object, _output: Path) -> None:
        raise OSError("simulated save failure")

    def failing_cleanup(_path: Path) -> None:
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(legacy_sources, "_save_file", failing_save)
    monkeypatch.setattr(legacy_sources, "_remove_file", failing_cleanup)
    with pytest.raises(LegacyCheckpointError, match="sidecar-cleanup-failed"):
        resolve_weight_source(source)


def test_source_mutation_during_conversion_refuses_without_publication(
    tmp_path: Path,
    torch_modules: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, _ = torch_modules
    source = tmp_path / "model.ckpt"
    torch.save({"weight": torch.ones(1)}, source)
    real_convert = legacy_sources.convert_legacy_checkpoint

    def mutate_after_conversion(input_path: Path, output_path: Path) -> Any:
        result = real_convert(input_path, output_path)
        torch.save({"weight": torch.zeros(2)}, input_path)
        return result

    monkeypatch.setattr(legacy_sources, "convert_legacy_checkpoint", mutate_after_conversion)
    with pytest.raises(LegacyCheckpointError, match="source-changed"):
        resolve_weight_source(source)

    assert list(tmp_path.glob("*.dinkster.safetensors")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_source_reverification_failure_is_named_and_cleans_temp(
    tmp_path: Path,
    torch_modules: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, _ = torch_modules
    source = tmp_path / "model.ckpt"
    torch.save({"weight": torch.ones(1)}, source)
    real_hash = legacy_sources._sha256_file
    source_hashes = 0

    def fail_second_source_hash(path: Path) -> str:
        nonlocal source_hashes
        if path == source:
            source_hashes += 1
            if source_hashes == 2:
                raise OSError("simulated source re-read failure")
        return real_hash(path)

    monkeypatch.setattr(legacy_sources, "_sha256_file", fail_second_source_hash)
    with pytest.raises(LegacyCheckpointError, match="source-reverification-failed"):
        resolve_weight_source(source)

    assert list(tmp_path.glob("*.dinkster.safetensors")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_competing_valid_publication_wins_without_temp_leak(
    tmp_path: Path,
    torch_modules: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, _ = torch_modules
    source = tmp_path / "model.ckpt"
    torch.save({"weight": torch.ones(1)}, source)

    def competing_replace(temporary: Path, sidecar: Path) -> None:
        shutil.copyfile(temporary, sidecar)
        raise PermissionError("simulated competing publication")

    monkeypatch.setattr(legacy_sources, "_replace_file", competing_replace)
    sidecar = resolve_weight_source(source)

    load_safetensors_header(sidecar)
    assert list(tmp_path.glob(".*.tmp")) == []


def test_conversion_has_no_unsafe_load_fallback() -> None:
    source = inspect.getsource(legacy_sources)
    compact = source.replace(" ", "")

    assert "weights_only=True" in compact
    assert "weights_only=False" not in compact
    assert "pickle_module" not in source


def test_converter_import_failure_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"checkpoint")

    def fail_torch_import() -> Any:
        raise RuntimeError("broken native torch install")

    monkeypatch.setattr(legacy_sources, "_load_torch_module", fail_torch_import)
    with pytest.raises(LegacyCheckpointError, match="converter-unavailable"):
        convert_legacy_checkpoint(source, tmp_path / "output.safetensors")


def test_output_hash_failure_is_named(
    tmp_path: Path,
    torch_modules: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, _ = torch_modules
    source = tmp_path / "model.ckpt"
    output = tmp_path / "output.safetensors"
    torch.save({"weight": torch.ones(1)}, source)

    def fail_output_hash(_path: Path) -> str:
        raise OSError("simulated output read failure")

    monkeypatch.setattr(legacy_sources, "_sha256_file", fail_output_hash)
    with pytest.raises(LegacyCheckpointError, match="conversion-failed"):
        convert_legacy_checkpoint(source, output)


def test_native_arm_legacy_resolution_is_bounded_to_compatible_sources() -> None:
    import dinkster_compat_comfy.native_arm as native_arm

    source = inspect.getsource(native_arm)

    # Each prevalidated overlay and canonical role source is resolved before
    # any residency graph mutation.
    assert source.count("resolve_weight_source(") == 8
    assert "load_safetensors_header(checkpoint_asset.local_path())" not in source
    assert "load_safetensors_header(lora.local_path())" not in source
    component_loader = inspect.getsource(native_arm._load_detected_component)
    assert "_component_candidate_path(asset)" in component_loader
    assert "asset_digest=asset.digest, asset_size=asset.size" in component_loader
    assert "resolve_weight_source(" not in component_loader
    assert component_loader.index("load_safetensors_header(") < component_loader.index(
        "_build_component_runtime_handle("
    )
    control_loader = inspect.getsource(native_arm.NativeControlNetLoader.execute)
    assert control_loader.index("resolve_weight_source(") < control_loader.index(
        "_enroll_control_module("
    )


def test_module_import_is_torch_free() -> None:
    source = inspect.getsource(legacy_sources)

    assert "import torch" not in source
    assert os.path.basename(legacy_sources.__file__) == "legacy_sources.py"


def test_latent_served_graph_and_host_import_are_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import importlib.abc
import os
import sys
class HostOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'dinkster_inference_torch', 'comfy',
                                     'comfy_extras', 'folder_paths', 'nodes'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, HostOnly())
os.environ['DINKSTER_COMFY_NATIVE_ONLY'] = '1'
import dinkster.compat_api
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetRef
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy.entry import COMFY_NODES, register_types
from dinkster_compat_comfy.native import LATENT, LoadLatent
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, validate
from dinkster_schema import build_node_types, build_schemas, schema_from_wire
from dinkster_server import create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
registry = TypeRegistry()
register_core_types(registry)
register_types(registry)
schemas = build_schemas(COMFY_NODES)
assert 'comfy.LoadLatent' not in schemas
assert schemas['dinkster.load_latent'] == LoadLatent.schema()
assert tuple(output.id for output in LoadLatent.schema().outputs) == ('samples', 'vae_hint')
assert LATENT.types == ('comfy.LATENT',)
selected = {key: schemas[key] for key in (
    'dinkster.load_latent', 'dinkster.save_latent', 'dinkster.load_vae', 'dinkster.vae_decode',
)}
def make_engine(on_event):
    return Engine(
        schemas=selected, registry=registry,
        worker=InProcessWorker(build_node_types(COMFY_NODES), registry),
        cache=MemoryLRUCache(), on_event=on_event,
    )
async def scenario():
    async with TestClient(TestServer(create_app(make_engine, selected))) as client:
        async with client.get('/api/nodes') as response:
            assert response.status == 200
            data = await response.json()
    served = {key: schema_from_wire(value) for key, value in data['nodes'].items()}
    assert served == selected
    asset = AssetRef('blake3:' + 'a' * 64, 'sample.latent', 4).to_wire()
    for with_hint in (False, True):
        save_inputs = {'samples': Link('load', 'samples')}
        if with_hint:
            save_inputs['vae'] = Link('vae', 'vae')
        graph = Graph(nodes={
            'load': GraphNode('dinkster.load_latent', {'asset': asset}),
            'vae': GraphNode('dinkster.load_vae', {'pixel_space': True}),
            'save': GraphNode('dinkster.save_latent', save_inputs),
            'reload': GraphNode('dinkster.load_latent', {'asset': Link('save', 'asset')}),
            'decode': GraphNode('dinkster.vae_decode', {
                'samples': Link('load', 'samples'), 'vae': Link('vae', 'vae'),
            }),
            'decode_saved': GraphNode('dinkster.vae_decode', {
                'samples': Link('save', 'samples'), 'vae': Link('vae', 'vae'),
            }),
        })
        for current in (schemas, served):
            assert validate(
                graph, current, ['reload', 'decode', 'decode_saved'], known_types=registry,
            ) == []
asyncio.run(scenario())
assert 'torch' not in sys.modules
assert 'dinkster_inference_torch' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
