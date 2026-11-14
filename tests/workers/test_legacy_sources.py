"""Legacy checkpoint conversion at the worker boundary."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from dinkster_inference_wire import load_safetensors_header
from dinkster_workers import legacy_sources
from dinkster_workers.legacy_sources import (
    LegacyCheckpointError,
    convert_legacy_checkpoint,
    resolve_weight_source,
)


@pytest.fixture
def torch_modules() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    return torch, safetensors_torch


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


def test_conversion_round_trips_and_logs_actual_conversion(
    tmp_path: Path,
    torch_modules: tuple[Any, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    torch, safetensors_torch = torch_modules
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
    loaded = safetensors_torch.load_file(sidecar)
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
    torch, safetensors_torch = torch_modules
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
    assert safetensors_torch.load_file(output)["weight"].shape == (1,)


def test_unsupported_extension_refuses_with_supported_formats(tmp_path: Path) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(b"anything")

    with pytest.raises(LegacyCheckpointError, match="unsupported-weight-source-format") as error:
        resolve_weight_source(source)

    for suffix in (".ckpt", ".pt", ".pth", ".safetensors", ".sft"):
        assert suffix in str(error.value)
    assert "convert_ckpt_to_safetensors.py" in str(error.value)
    assert error.value.__cause__ is not None


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
