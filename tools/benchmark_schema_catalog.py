"""Measure complete installed-pack composition through the dinkster-serve entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import statistics
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import aiohttp
import psutil
from dinkster_registry import Lockfile

from dinkster.installer import Installer, lock_local_pack

BUDGET_SECONDS = 10.0

# Reject execution imports and expose aiohttp's post-bind callback. Composition and
# serving configuration follow the real CLI, without readiness polling.
ENTRY = f"""
import importlib.abc
import os
import sys
import threading
import traceback

print('BENCH_STARTED', os.getpid(), flush=True)
def dump_startup_stack():
    print('BENCH_STACK', flush=True)
    traceback.print_stack(sys._current_frames()[threading.main_thread().ident])
watchdog = threading.Timer({BUDGET_SECONDS / 2}, dump_startup_stack)
watchdog.daemon = True
watchdog.start()

blocked = ('dinkster_compat_comfy.native_arm', 'dinkster_compat_comfy.entry',
           'dinkster_inference_torch', 'torch')
class NoExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            print('BLOCKED_EXECUTION_IMPORT', fullname, flush=True)
            raise AssertionError('cold host imported execution module: ' + fullname)
assert not any(name in sys.modules for name in blocked)
sys.meta_path.insert(0, NoExecutionImports())

from dinkster import serve
print('BENCH_IMPORTED', flush=True)
run_app = serve.web.run_app
def announced(app, **kwargs):
    kwargs['print'] = lambda _: print('BENCH_BOUND', os.getpid(), flush=True)
    return run_app(app, **kwargs)
serve.web.run_app = announced
serve.main()
"""


def bound_server(launcher_pid: int, lines: list[str]) -> psutil.Process:
    """Identify the serving interpreter, not a Windows virtualenv redirector."""
    pid = int(next(line.split()[1] for line in lines if line.startswith("BENCH_BOUND ")))
    launcher = psutil.Process(launcher_pid)
    server = psutil.Process(pid)
    assert server == launcher or server in launcher.children(recursive=True)
    return server


def terminate_children(launcher_pid: int) -> list[psutil.Process]:
    """Stop owned descendants, including an interpreter behind a launcher."""
    try:
        children = psutil.Process(launcher_pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    return children


def install_packs(root: Path, count: int) -> Path:
    installer = Installer(root / "install", accelerator="cpu", runtime_probe=lambda _: ())
    entries = []
    for index in range(count):
        name = f"bootpack{index}"
        pack = root / "sources" / name
        pack.mkdir(parents=True)
        (pack / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{name}"\n[pack.sandbox]\n[pack.entry]\nnodes = "{name}:NODES"\n',
            encoding="utf-8",
        )
        (pack / f"{name}.py").write_text(
            "import os\n"
            "from pathlib import Path\n"
            "from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr\n"
            "if marker := os.environ.get('DINKSTER_BOOT_IMPORTS'):\n"
            "    with Path(marker).open('a') as stream:\n"
            f"        stream.write('{name}\\n')\n"
            "class Constant(Node):\n"
            "    @classmethod\n"
            "    def define_schema(cls):\n"
            f"        return NodeSchema(node_type='{name}.constant', "
            "outputs=(OutputSpec('value', TypeExpr.concrete('core.int')),))\n"
            "    @classmethod\n"
            "    def execute(cls):\n"
            "        return cls.outputs(value=1)\n"
            "NODES = [Constant]\n",
            encoding="utf-8",
        )
        entry, _ = lock_local_pack(pack, installer.artifacts_dir)
        entries.append(entry)
    installer.apply(Lockfile.of(entries), venvs=False)
    return installer.root


async def measure(root: Path, install: Path, count: int) -> float:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    marker = root / "imports.txt"
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("DINKSTER_")
    }
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        DINKSTER_BOOT_IMPORTS=str(marker),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        ENTRY,
        "--no-default-packs",
        "--strict-packs",
        "--install-root",
        str(install),
        "--library-root",
        "",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        cwd=root,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    lines: list[str] = []
    timeline = [f"{time.perf_counter() - start:.3f}s subprocess created"]
    bound = composed = False
    stage = "startup stdout"
    pending_error: BaseException | None = None
    try:
        async with asyncio.timeout(BUDGET_SECONDS):
            assert proc.stdout is not None
            while not (bound and composed):
                line = (await proc.stdout.readline()).decode()
                if not line:
                    raise RuntimeError("server exited before composition: " + "".join(lines))
                lines.append(line)
                timeline.append(f"{time.perf_counter() - start:.3f}s {line.rstrip()}")
                bound |= "BENCH_BOUND" in line
                composed |= "node types" in line and " composed" in line
            assert not any("BLOCKED_EXECUTION_IMPORT" in line for line in lines), "".join(lines)
            async with aiohttp.ClientSession() as client:
                stage = "/api/composition"
                async with client.get(f"http://127.0.0.1:{port}/api/composition") as response:
                    response.raise_for_status()
                    composition = await response.json()
                    elapsed = time.perf_counter() - start
                assert not composition.get("composing"), composition
                assert len(composition["packs"]) == count, composition
                assert all(item["state"] == "announced" for item in composition["packs"].values())
                stage = "/api/nodes"
                async with client.get(f"http://127.0.0.1:{port}/api/nodes") as response:
                    response.raise_for_status()
                    nodes = await response.json()
                    for index in range(count):
                        assert f"bootpack{index}.constant" in str(nodes)
                assert not marker.exists(), "pack imported while serving the persisted catalog"
                assert not bound_server(proc.pid, lines).children(recursive=True), (
                    "server spawned a child"
                )
                return elapsed
    except BaseException as error:
        pending_error = error
        error.add_note(
            f"boot failed at {time.perf_counter() - start:.3f}s: "
            f"stage={stage}, bound={bound}, composed={composed}\n" + "\n".join(timeline)
        )
        raise
    finally:
        children = terminate_children(proc.pid)
        try:
            _, alive = await asyncio.to_thread(psutil.wait_procs, children, timeout=15)
        finally:
            if proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.terminate()
            output, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if pending_error is not None and output:
            pending_error.add_note("remaining server output:\n" + output.decode(errors="replace"))
        assert not alive, f"server descendants survived teardown: {alive}"
        assert b"BLOCKED_EXECUTION_IMPORT" not in output, output.decode(errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 10, 50, 200])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 0 for count in args.counts):
        parser.error("counts must be nonnegative and repeats must be positive")
    print(
        json.dumps(
            {"platform": platform.platform(), "python": sys.version, "budget_s": BUDGET_SECONDS}
        ),
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="dinkster-catalog-boot-") as directory:
        for count in args.counts:
            root = Path(directory) / str(count)
            root.mkdir()
            install = install_packs(root, count)
            samples = []
            loads = []
            for index in range(args.repeats):
                loads.append(os.getloadavg() if hasattr(os, "getloadavg") else None)
                sample = {"packs": count, "sample": index + 1, "load_average_before": loads[-1]}
                started = time.perf_counter()
                try:
                    elapsed = asyncio.run(measure(root, install, count))
                except BaseException as error:
                    print(
                        json.dumps(
                            sample
                            | {
                                "outcome": "failed",
                                "error": type(error).__name__,
                                "wall_s_including_cleanup": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
                    raise
                samples.append(elapsed)
                print(
                    json.dumps(
                        sample | {"outcome": "passed", "seconds": elapsed, "worker_starts": 0}
                    ),
                    flush=True,
                )
            print(
                json.dumps(
                    {
                        "packs": count,
                        "seconds": samples,
                        "load_average_before_samples": loads,
                        "median_s": statistics.median(samples),
                        "max_s": max(samples),
                        "worker_starts": 0,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
