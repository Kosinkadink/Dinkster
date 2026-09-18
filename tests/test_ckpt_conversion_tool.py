"""The manual checkpoint CLI delegates all conversion to the compat seam."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import pytest
from dinkster_compat_comfy.legacy_sources import LegacyConversionResult

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = REPO_ROOT / "tools" / "convert_ckpt_to_safetensors.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_ckpt_to_safetensors", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ckpt_conversion_tool_delegates_to_canonical_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _tool()
    source = tmp_path / "model.ckpt"
    output = tmp_path / "model.safetensors"
    calls: list[tuple[Path, Path]] = []

    def convert(source_path: Path, output_path: Path) -> LegacyConversionResult:
        calls.append((source_path, output_path))
        return LegacyConversionResult(
            dropped_keys=("optimizer", "epoch"),
            output_bytes=1234,
            output_sha256="a" * 64,
        )

    monkeypatch.setattr(tool, "convert_legacy_checkpoint", convert)
    tool.main([str(source), str(output)])

    assert calls == [(source, output)]
    assert capsys.readouterr().out == (
        f"converted {source} -> {output} "
        f"(dropped_keys=2, output_bytes=1234, output_sha256={'a' * 64})\n"
    )


def test_ckpt_conversion_tool_contains_no_conversion_logic() -> None:
    source = inspect.getsource(_tool())

    assert "from dinkster_compat_comfy.legacy_sources import convert_legacy_checkpoint" in source
    assert "torch.load" not in source
    assert "save_file" not in source
    assert "weights_only" not in source
