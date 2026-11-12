from __future__ import annotations

import json
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.mark.parametrize("published", ["pass", "fail", "fail-only", "absent"])
def test_selected_archive_runner_preserves_execution_and_unavailable_exports(
    tmp_path: Path, published: str
) -> None:
    runner = Path(__file__).parents[1] / "scripts/_selected_pack_test_child.py"
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "-n 2 --max-worker-restart=0"\n', encoding="ascii"
    )
    source = tmp_path / "test_selected.py"
    code = (
        "from _selected_pack_test_child import UnavailableExport\n"
        "def test_unavailable():\n"
        "    bool(UnavailableExport('fixture:unpublished'))\n"
    )
    if published != "absent":
        code += f"def test_published():\n    assert {published == 'pass'}\n"
    if published == "fail":
        code += "def test_keeper():\n    assert 17 + 4 == 21\n"
    source.write_text(code, encoding="ascii")
    companion = tmp_path / "test_unavailable.py"
    companion.write_text(
        "from _selected_pack_test_child import UnavailableExport\n"
        "def test_unavailable_companion():\n"
        "    bool(UnavailableExport('fixture:unpublished'))\n",
        encoding="ascii",
    )
    config = tmp_path / "selection.json"
    config.write_text(
        json.dumps({"modules": {}, "packs": [], "tests": [str(source), str(companion)]})
    )
    result = subprocess.run(
        [sys.executable, str(runner), str(config)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    expected = {"pass": 0, "fail": 1, "fail-only": 1, "absent": 5}[published]
    assert result.returncode == expected, result.stdout + result.stderr
    assert "1 skipped" in result.stdout
    if published != "absent":
        assert ("1 passed" if published == "pass" else "1 failed") in result.stdout


def test_selected_files_run_concurrently_and_exactly_once(tmp_path: Path) -> None:
    runner = Path(__file__).parents[1] / "scripts/_selected_pack_test_child.py"
    sources = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        listener.settimeout(30)
        for index in range(3):
            source = tmp_path / f"test_selected_{index}.py"
            source.write_text(
                "import socket\nfrom pathlib import Path\n"
                "def test_selected():\n"
                + (
                    "    with socket.create_connection(\n"
                    f"        {listener.getsockname()!r}, timeout=30\n"
                    "    ) as peer:\n"
                    "        assert peer.recv(1) == b'!'\n"
                    if index < 2
                    else ""
                )
                + f"    with Path({str(tmp_path / f'executed-{index}')!r}).open('a') as output:\n"
                f"        output.write('{index}')\n",
                encoding="ascii",
            )
            sources.append(str(source))
        config = tmp_path / "selection.json"
        config.write_text(json.dumps({"modules": {}, "packs": [], "tests": sources}))

        def release_both() -> None:
            with listener.accept()[0] as first, listener.accept()[0] as second:
                first.sendall(b"!")
                second.sendall(b"!")

        with ThreadPoolExecutor(max_workers=1) as executor:
            rendezvous = executor.submit(release_both)
            result = subprocess.run(
                [sys.executable, str(runner), str(config)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            rendezvous.result()
    assert result.returncode == 0, result.stdout + result.stderr
    for index in range(3):
        assert (tmp_path / f"executed-{index}").read_text() == str(index)


@pytest.mark.parametrize("outcome", ["pass", "fail", "skip", "missing-runtime"])
def test_selected_cross_pack_cases_use_the_owning_interpreter(
    tmp_path: Path, selected_pack_generation: Path, outcome: str
) -> None:
    runner = Path(__file__).parents[1] / "scripts/_selected_pack_test_child.py"
    runtime = tmp_path / "pack"
    subprocess.run(
        [
            "uv",
            "venv",
            "--system-site-packages",
            "--python",
            str(selected_pack_generation),
            str(runtime),
        ],
        check=True,
    )
    python = runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    source = tmp_path / "test_cross_pack.py"
    source.write_text(
        "import sys\nimport pytest\nfrom pathlib import Path\n"
        "@pytest.mark.pack_runtime('other-pack')\n"
        "def test_registration():\n"
        f"    assert Path(sys.prefix) == Path({str(runtime)!r})\n"
        + (
            "    pytest.skip('unavailable')\n"
            if outcome == "skip"
            else f"    assert {outcome != 'fail'}\n"
        )
        + "def test_local():\n"
        f"    assert Path(sys.prefix) != Path({str(runtime)!r})\n",
        encoding="ascii",
    )
    config = tmp_path / "selection.json"
    config.write_text(
        json.dumps(
            {
                "modules": {},
                "packs": [],
                "tests": [str(source)],
                "runtimes": {} if outcome == "missing-runtime" else {"other-pack": str(python)},
            }
        )
    )
    result = subprocess.run(
        [str(selected_pack_generation), str(runner), str(config)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == (0 if outcome in {"pass", "skip"} else 1), (
        result.stdout + result.stderr
    )
    if outcome == "pass":
        assert "2 passed" in result.stdout
    elif outcome == "skip":
        assert "1 skipped" in result.stdout
    elif outcome == "fail":
        assert "assert False" in result.stdout
    else:
        assert "KeyError: 'other-pack'" in result.stdout
