from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import pytest
from dinkster_workers import interpreter
from dinkster_workers.doctor import prepare_catalog
from dinkster_workers.interpreter import InterpreterPreflightError, preflight_interpreter
from dinkster_workers.launch import LaunchSpec, SubprocessLauncher
from test_doctor import HEALTHY_MANIFEST, HEALTHY_NODES


def test_preflight_accepts_current_supported_interpreter() -> None:
    assert preflight_interpreter(sys.executable) == sys.version_info[:2]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows executable search order")
def test_worker_bare_interpreter_uses_native_executable_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alternative = tmp_path / "other"
    venv.EnvBuilder(with_pip=False).create(alternative)
    environment = {
        "PATH": str(alternative / "Scripts") + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(tmp_path),
    }
    monkeypatch.setenv("PATH", environment["PATH"])
    assert Path(shutil.which("python") or "") == alternative / "Scripts/python.exe"
    (tmp_path / "selection_probe.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps([sys.executable,sys.prefix]))\n"
    )
    plain = tmp_path / "plain.json"
    subprocess.run(
        ["python", "-m", "selection_probe", str(plain)],
        env={**os.environ, **environment},
        check=True,
        timeout=30,
    )
    expected = json.loads(plain.read_text())
    assert Path(expected[1]) != alternative
    result = tmp_path / "launched.json"

    async def scenario() -> None:
        process = await SubprocessLauncher().launch(
            LaunchSpec(
                command=("python", "-m", "selection_probe", str(result)),
                env=environment,
                endpoint_dir=tmp_path,
                pack_root=tmp_path,
                python="python",
                use_shm=False,
            )
        )
        assert await asyncio.wait_for(process.wait(), timeout=30) == 0

    asyncio.run(scenario())
    assert json.loads(result.read_text()) == expected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native extension loader contract")
@pytest.mark.parametrize("long_root", [False, True])
@pytest.mark.parametrize(
    "pack", ["dinkster-vision-depth-anything-v2", "dinkster-vision-depth-anything-v3"]
)
def test_worker_launch_imports_native_extension_at_ordinary_data_roots(
    long_root: bool, pack: str
) -> None:
    import sqlalchemy
    import typing_extensions

    with tempfile.TemporaryDirectory(prefix="pack-path-", dir=Path.home()) as temporary:
        root = Path(temporary)
        if long_root:
            root = root / ("data" + "x" * max(0, 59 - len(str(root)) - 5))
        root.mkdir(exist_ok=True)
        venv_root = root / "library/venvs/win-nvidia" / ("a" * 64) / pack
        venv.EnvBuilder(with_pip=False).create(venv_root)
        site = venv_root / "Lib/site-packages"
        shutil.copytree(Path(sqlalchemy.__file__).parent, site / "sqlalchemy")
        shutil.copy2(typing_extensions.__file__, site / "typing_extensions.py")
        python = venv_root / "Scripts/python.exe"
        result_path = root / "result.json"
        script = (
            "import json, os, sys; from pathlib import Path; "
            "import sqlalchemy.util._collections_cy as native; "
            "Path(sys.argv[1]).write_text(json.dumps("
            "{'prefix': sys.prefix, 'native': native.__file__, 'argv': sys.argv[1:], "
            "'environment': os.environ['DINKSTER_PATH_TEST'], 'module': __spec__.name}))"
        )
        (site / "native_path_probe.py").write_text(script)

        async def scenario() -> None:
            process = await SubprocessLauncher().launch(
                LaunchSpec(
                    command=(str(python), "-m", "native_path_probe", str(result_path), "two words"),
                    env={"DINKSTER_PATH_TEST": "selected"},
                    endpoint_dir=root,
                    pack_root=root,
                    python=str(python),
                    use_shm=False,
                )
            )
            assert await asyncio.wait_for(process.wait(), timeout=30) == 0

        asyncio.run(scenario())
        result = json.loads(result_path.read_text())
        assert Path(result["prefix"]) == venv_root
        assert result["argv"] == [str(result_path), "two words"]
        assert result["environment"] == "selected"
        assert result["module"] == "native_path_probe"
        extension = Path(result["native"])
        assert extension.suffix == ".pyd"
        assert extension.samefile(next((site / "sqlalchemy/util").glob("_collections_cy*.pyd")))
        if long_root:
            assert len(str(root).encode("utf-16-le")) // 2 >= 59
            assert len(str(extension).removeprefix("\\\\?\\").encode("utf-16-le")) // 2 >= 251

        (site / "fixture.pth").write_text(
            "\n".join(str(Path(path).resolve()) for path in sys.path if path) + "\n"
        )
        source = root / "source"
        source.mkdir()
        manifest = source / "dinkster-pack.toml"
        manifest.write_text(HEALTHY_MANIFEST)
        (source / "healthy_nodes.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            "import sqlalchemy.util._collections_cy as native\n"
            f"assert Path(sys.prefix) == Path({str(venv_root)!r})\n"
            f"assert Path(native.__file__).samefile(Path({str(extension)!r}))\n" + HEALTHY_NODES
        )
        report = prepare_catalog(manifest, interpreter=python)
        assert report.ok, report.to_json()
        assert set(report.node_types) == {"healthy.doubler", "healthy.tagger"}


def test_version_script_requires_no_additional_imports() -> None:
    command = (
        "import sys\n"
        "def reject_import(event, args):\n"
        "    if event == 'import':\n"
        "        raise RuntimeError('version probe imported ' + args[0])\n"
        "sys.addaudithook(reject_import)\n" + interpreter._VERSION_SCRIPT
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    assert result.stdout.strip() == f"[{sys.version_info.major},{sys.version_info.minor}]"


@pytest.mark.parametrize("hook_name", ["startup.pth", "sitecustomize.py"])
def test_preflight_does_not_execute_site_hooks(tmp_path: Path, hook_name: str) -> None:
    root = tmp_path / "selected"
    venv.EnvBuilder(with_pip=False, symlinks=sys.platform != "win32").create(root)
    python = root / "Scripts" / "python.exe" if sys.platform == "win32" else root / "bin" / "python"
    site = subprocess.run(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    site_packages = Path(site.stdout.strip())
    (site_packages / hook_name).write_text("import os; os._exit(61)\n")
    if hook_name == "sitecustomize.py":
        # Some distributions ship a standard-library sitecustomize ahead of the venv.
        (site_packages / "startup.pth").write_text(
            f"import sys; sys.path.insert(0, {str(site_packages)!r})\n"
        )

    assert preflight_interpreter(python) == sys.version_info[:2]
    normal_start = subprocess.run(
        [str(python), "-I", "-c", "pass"], capture_output=True, timeout=30, check=False
    )
    assert normal_start.returncode == 61


@pytest.mark.parametrize("version", [(3, 12), (3, 13), (3, 14)])
def test_preflight_accepts_supported_version(
    version: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, f"[{version[0]},{version[1]}]\n", "")

    monkeypatch.setattr("dinkster_workers.interpreter.subprocess.run", run)
    assert preflight_interpreter("/selected/python") == version
    command, kwargs = seen[0]
    assert command[:4] == ["/selected/python", "-I", "-S", "-c"]
    assert "sys.version_info" in command[4]
    assert kwargs == {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 5.0,
        "check": False,
        "shell": False,
    }


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (subprocess.CompletedProcess([], 0, "[3,11]\n", ""), "Python 3.11"),
        (subprocess.CompletedProcess([], 7, "", "broken"), "exited 7"),
        (subprocess.CompletedProcess([], 0, "not-json", ""), "malformed"),
        (subprocess.CompletedProcess([], 0, "\N{REPLACEMENT CHARACTER}", ""), "malformed"),
        (subprocess.CompletedProcess([], 0, "[3,12,1]", ""), "malformed"),
        (subprocess.CompletedProcess([], 0, "[true,12]", ""), "malformed"),
    ],
)
def test_preflight_fails_closed_for_bad_result(
    result: subprocess.CompletedProcess[str],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dinkster_workers.interpreter.subprocess.run", lambda *_args, **_kwargs: result
    )
    with pytest.raises(InterpreterPreflightError, match=match):
        preflight_interpreter("/selected/python")


def test_preflight_failures_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dinkster_workers.interpreter.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "failure"),
    )
    with pytest.raises(InterpreterPreflightError, match=r"configure an executable Python >=3\.12"):
        preflight_interpreter("/selected/python")


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        (OSError("secret path details"), "could not start"),
        (subprocess.TimeoutExpired(["python"], 5), "timed out"),
    ],
)
def test_preflight_fails_closed_when_probe_does_not_complete(
    failure: BaseException, match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("dinkster_workers.interpreter.subprocess.run", fail)
    with pytest.raises(InterpreterPreflightError, match=match) as caught:
        preflight_interpreter(Path("/selected/python"))
    assert "secret path details" not in str(caught.value)


def test_preflight_bounds_selected_interpreter_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dinkster_workers.interpreter.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "x" * 1_000),
    )
    with pytest.raises(InterpreterPreflightError) as caught:
        preflight_interpreter("/" + "selected" * 1_000)
    assert len(str(caught.value)) < 700
