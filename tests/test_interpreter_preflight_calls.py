from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest
from dinkster_values import TypeRegistry
from dinkster_workers import doctor, isolated, provision
from dinkster_workers.interpreter import InterpreterPreflightError
from dinkster_workers.manifest import load_manifest
from dinkster_workers.provision import ProvisionError

from dinkster import comfy_compose, port
from dinkster.compose import CompositionError
from dinkster.port import PortError


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"


def _reject(_interpreter: Path | str) -> tuple[int, int]:
    raise InterpreterPreflightError("unsupported selected interpreter")


def _unexpected(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("child import, probe, or launcher was reached")


def _manifest(root: Path, name: str = "test-pack") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dinkster-pack.toml"
    path.write_text(
        f'[pack]\nname = "{name}"\n[pack.sandbox]\n[pack.entry]\nnodes = "nodes:NODES"\n'
    )
    return path


def test_comfy_model_roots_preflights_before_import_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comfy_compose, "preflight_interpreter", _reject, raising=False)
    monkeypatch.setattr(comfy_compose.subprocess, "run", _unexpected)
    with pytest.raises(CompositionError, match="unsupported selected interpreter"):
        comfy_compose.comfy_model_roots(tmp_path, python="/selected/python")


@pytest.mark.parametrize("with_legacy", [False, True])
def test_comfy_specs_preflight_before_core_or_legacy_blake3_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_legacy: bool
) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    monkeypatch.setattr(comfy_compose, "preflight_interpreter", _reject, raising=False)
    monkeypatch.setattr(comfy_compose, "_probe_comfy_blake3", _unexpected)
    with pytest.raises(CompositionError, match="unsupported selected interpreter"):
        comfy_compose.comfy_compat_specs(
            tmp_path,
            python="/selected/python",
            legacy_packs=(legacy,) if with_legacy else (),
        )


def test_dinkster_port_preflights_before_legacy_pack_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(port, "preflight_interpreter", _reject, raising=False)
    monkeypatch.setattr(port.subprocess, "run", _unexpected)
    with pytest.raises(PortError, match="unsupported selected interpreter"):
        port._run_probe(tmp_path / "pack", tmp_path, python="/selected/python")


@pytest.mark.parametrize("reused", [True, False])
def test_pack_venv_preflights_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reused: bool
) -> None:
    manifest = load_manifest(_manifest(tmp_path / "pack"))
    venv_root = tmp_path / "venvs"
    python = _venv_python(venv_root / manifest.name)
    if reused:
        python.parent.mkdir(parents=True)
        python.write_text("")
        (python.parent.parent / ".dinkster-complete").write_text("complete\n")

    def run(command: list[str]) -> None:
        if command[1] == "venv":
            python.parent.mkdir(parents=True)
            python.write_text("")

    monkeypatch.setattr(provision, "preflight_interpreter", _reject, raising=False)
    monkeypatch.setattr(provision, "_run", run)
    with pytest.raises(ProvisionError, match="unsupported selected interpreter"):
        provision.ensure_pack_venv(manifest, venv_root=venv_root)


@pytest.mark.parametrize("reused", [True, False])
def test_group_venv_preflights_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reused: bool
) -> None:
    manifests = (
        load_manifest(_manifest(tmp_path / "one", "pack-one")),
        load_manifest(_manifest(tmp_path / "two", "pack-two")),
    )
    venv_root = tmp_path / "venvs"
    venv = venv_root / "group"
    python = _venv_python(venv)
    if reused:
        python.parent.mkdir(parents=True)
        python.write_text("")
        (venv / ".dinkster-complete").write_text("complete\n")

    def run(command: list[str]) -> None:
        if command[1] == "venv":
            python.parent.mkdir(parents=True)
            python.write_text("")

    monkeypatch.setattr(provision, "preflight_interpreter", _reject, raising=False)
    monkeypatch.setattr(provision, "_run", run)
    with pytest.raises(ProvisionError, match="unsupported selected interpreter"):
        provision.ensure_group_venv(manifests, "group", venv_root=venv_root)


def test_doctor_preflights_before_child_import_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "pack")
    monkeypatch.setattr(doctor, "preflight_interpreter", _reject, raising=False)
    monkeypatch.setattr("dinkster_workers.doctor.subprocess.run", _unexpected)
    report = doctor.diagnose(manifest, interpreter="/selected/python")
    assert {finding.code for finding in report.findings} == {"doctor.interpreter"}
    assert "unsupported selected interpreter" in report.findings[0].message


def test_doctor_reports_interpreter_timeout_before_child_import_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "pack")

    def timeout(command: list[str], **kwargs: object) -> None:
        assert command[:4] == ["/selected/python", "-I", "-S", "-c"]
        assert kwargs["timeout"] == 5.0
        raise subprocess.TimeoutExpired(command, 5.0)

    monkeypatch.setattr("dinkster_workers.interpreter.subprocess.run", timeout)
    report = doctor.diagnose(manifest, interpreter="/selected/python")
    assert not report.ok
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.code == "doctor.interpreter"
    assert finding.severity == "error"
    assert "version check timed out after 5s" in finding.message
    assert "Python >=3.12" in finding.fix


class _UnexpectedLauncher:
    async def launch(self, _spec: object) -> None:
        _unexpected()


def test_isolated_worker_preflights_before_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated, "preflight_interpreter", _reject, raising=False)
    worker = isolated.IsolatedWorker(
        _manifest(tmp_path / "pack"),
        TypeRegistry(),
        python="/selected/python",
        launcher=_UnexpectedLauncher(),  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="unsupported selected interpreter"):
        asyncio.run(worker.start())


def test_isolated_worker_group_preflights_before_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated, "preflight_interpreter", _reject, raising=False)
    worker = isolated.GroupIsolatedWorker(
        "group",
        (_manifest(tmp_path / "one", "pack-one"), _manifest(tmp_path / "two", "pack-two")),
        TypeRegistry(),
        python="/selected/python",
        launcher=_UnexpectedLauncher(),  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="unsupported selected interpreter"):
        asyncio.run(worker.start())
