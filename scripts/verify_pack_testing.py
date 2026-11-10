"""Exercise the public pack-testing example using only built wheels."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def verify(wheels: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    example = root / "tests/fixtures/pack-testing"
    companions = [next(wheels.glob(f"dinkster_{name}-*.whl")) for name in ("api", "engine")]
    with tempfile.TemporaryDirectory(prefix="pack-testing-", dir=wheels.parent) as directory:
        work = Path(directory)
        environment = work / "venv"
        subprocess.run(["uv", "venv", "--python", "3.12", str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), *map(str, companions), "pytest"],
            cwd=work,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.metadata as m, importlib.util as u; "
                "assert u.find_spec('torch') is None; "
                "from dinkster_engine.testing import load_pack; "
                "print('Lean install packages:', len(list(m.distributions())))",
            ],
            cwd=work,
            check=True,
        )
        print(
            "Lean install file bytes:",
            sum(path.stat().st_size for path in environment.rglob("*") if path.is_file()),
            flush=True,
        )
        if os.name == "posix":
            print(
                "Lean install allocated disk bytes:",
                sum(
                    path.lstat().st_blocks * 512 for path in [environment, *environment.rglob("*")]
                ),
                flush=True,
            )
        built = work / "example-wheel"
        subprocess.run(
            ["uv", "build", str(example), "--wheel", "--out-dir", str(built)],
            cwd=work,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                str(next(built.glob("*.whl"))),
            ],
            cwd=work,
            check=True,
        )
        shutil.copyfile(example / "example.py", work / "test_pack.py")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PACK_TEST_IN_PROCESS", None)
        for in_process in (False, True):
            if in_process:
                env["PACK_TEST_IN_PROCESS"] = "1"
            started = time.monotonic()
            subprocess.run(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import os,sys,pytest; starts=[]; "
                    "sys.addaudithook(lambda event,args: starts.append(args[1]) "
                    "if event == 'subprocess.Popen' and 'dinkster_workers.host' in args[1] "
                    "else None); "
                    "status=pytest.main(['-q','test_pack.py']); "
                    "print('Worker process starts:',len(starts)); "
                    "expected=0 if os.environ.get('PACK_TEST_IN_PROCESS') == '1' else 1; "
                    "assert len(starts) == expected; "
                    "sys.exit(status)",
                ],
                cwd=work,
                env=env,
                check=True,
            )
            print(f"in_process={in_process}: {time.monotonic() - started:.3f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", type=Path)
    verify(parser.parse_args().wheels.resolve())
