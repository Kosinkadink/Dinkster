"""Measure real pack workers on WSL; run GPU mode under the host's exclusive GPU lock.

Run through scripts/develop.py run. --work-dir must be a new child of an
authorized WSL-disk root. The Windows sentinel must already exist. No host
files are modified: driver write access is tested by opening without truncation
or writing, and its digest is checked before and after. Results remain in the
work directory, including failures. --cpu-only never imports torch in a worker.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dinkster_api.v1 import CORE_STRING, Node, NodeSchema, Output, TypeExpr


def file_digest(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
    except OSError as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def tcp_probe(host: str, port: int) -> dict[str, object]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=2):
            return {"connected": True, "seconds": time.perf_counter() - started}
    except OSError as exc:
        return {
            "connected": False,
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def inventory() -> dict[str, object]:
    from dinkster_workers.pack_wheel import installed_pack_manifests

    manifests = [
        {"name": dist.metadata["Name"], "version": dist.version, "manifest": file_digest(path)}
        for dist, path in installed_pack_manifests()
    ]
    return {
        "interpreter": file_digest(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "packages": sorted(
            (
                {"name": d.metadata["Name"], "version": d.version}
                for d in importlib.metadata.distributions()
            ),
            key=lambda d: (d["name"] or "", d["version"]),
        ),
        "installed_pack_count": len(manifests),
        "installed_pack_manifests": manifests,
        "pack_lock_comparison": (
            "no installed packs; temporary probe is not a published first-party pack"
            if not manifests
            else "BLOCKED: installed packs require committed lock/content comparison"
        ),
    }


class Inspect(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="wslprobe.inspect",
            inputs=(),
            outputs=(Output("report", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    async def execute(cls):
        config = json.loads(Path(__file__).with_name("config.json").read_text())
        if config["inventory_only"]:
            return cls.outputs(report=json.dumps(inventory(), sort_keys=True))
        report: dict[str, object] = {}
        sentinel = Path(config["windows_sentinel"])
        try:
            report["windows_sentinel"] = {"readable": True, "bytes": len(sentinel.read_bytes())}
        except OSError as exc:
            report["windows_sentinel"] = {"readable": False, "error": str(exc)}
        report["windows_drive_visible"] = Path(*sentinel.parts[:3]).exists()
        report["run_wsl_visible"] = Path("/run/WSL").exists()
        interop = config["interop_socket"]
        if interop:
            try:
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(2)
                    client.connect(interop)
                report["interop_socket"] = {"connected": True}
            except OSError as exc:
                report["interop_socket"] = {"connected": False, "error": str(exc)}
        else:
            report["interop_socket"] = {"error": "no exact interop socket supplied"}
        try:
            proc = subprocess.run(["/init", "--help"], capture_output=True, timeout=2)
            report["init"] = {"executed": True, "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            report["init"] = {"executed": True, "error": "execution exceeded two seconds"}
        except OSError as exc:
            report["init"] = {"executed": False, "error": str(exc)}
        report["host_loopback"] = tcp_probe("127.0.0.1", config["host_port"])
        report["external_tcp"] = tcp_probe(config["external_host"], config["external_port"])
        try:
            descriptor = os.open(config["driver_library"], os.O_WRONLY)
            os.close(descriptor)
            report["driver_write_open"] = {"opened": True}
        except OSError as exc:
            report["driver_write_open"] = {"opened": False, "error": str(exc)}
        try:
            report["driver_mount_readonly"] = bool(
                os.statvfs(config["driver_library"]).f_flag & os.ST_RDONLY
            )
        except OSError as exc:
            report["driver_mount_readonly"] = {"error": str(exc)}
        report["dxg_visible"] = Path("/dev/dxg").exists()
        if config["cpu_only"]:
            report["compute"] = {"not_run": "--cpu-only"}
        else:
            try:
                import torch

                report["torch"] = {
                    "version": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                }
                try:
                    a = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
                    b = torch.randn_like(a)
                    for _ in range(10):
                        c = torch.mm(a, b)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    completed = 0
                    while time.perf_counter() - started < config["seconds"]:
                        for _ in range(20):
                            c = torch.mm(a, b)
                        torch.cuda.synchronize()
                        completed += 20
                    elapsed = time.perf_counter() - started
                    report["matmul"] = {
                        "shape": [8192, 8192],
                        "dtype": "float16",
                        "iterations": completed,
                        "warmups": 10,
                        "seconds": elapsed,
                        "per_second": completed / elapsed,
                        "finite": bool(torch.isfinite(c).all().item()),
                    }
                    del a, b, c
                    torch.cuda.empty_cache()
                except Exception as exc:
                    report["matmul"] = {"error": f"{type(exc).__name__}: {exc}"}
                try:
                    from safetensors.torch import load_file

                    loads = []
                    for _ in range(config["loads"]):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        tensors = load_file(config["model"], device="cuda")
                        torch.cuda.synchronize()
                        loads.append(
                            {
                                "seconds": time.perf_counter() - started,
                                "tensor_count": len(tensors),
                                "bytes": sum(
                                    t.numel() * t.element_size() for t in tensors.values()
                                ),
                            }
                        )
                        del tensors
                        torch.cuda.empty_cache()
                    report["model_load"] = {"device": "cuda", "loads": loads}
                except Exception as exc:
                    report["model_load"] = {"error": f"{type(exc).__name__}: {exc}"}
            except Exception as exc:
                report["compute"] = {"error": f"{type(exc).__name__}: {exc}"}
        return cls.outputs(report=json.dumps(report, sort_keys=True))


NODES = (Inspect,)


async def run_workers(args, work: Path) -> dict[str, object]:
    from dinkster_caches import MemoryLRUCache
    from dinkster_engine import Engine
    from dinkster_graph import Graph, GraphNode
    from dinkster_values import TypeRegistry, register_core_types
    from dinkster_workers import BubblewrapLauncher, IsolatedWorker, SandboxPolicy
    from dinkster_workers.launch import SubprocessLauncher

    pack = work / "pack"
    pack.mkdir()
    shutil.copyfile(__file__, pack / "probe_wsl_sandbox.py")
    manifest = pack / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "wsl-probe"\npublisher = "dinkster"\n'
        'namespaces = ["wslprobe"]\n[pack.entry]\nnodes = "probe_wsl_sandbox:NODES"\n'
    )

    async def accept(reader, writer):
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    config = vars(args).copy()
    config["host_port"] = server.sockets[0].getsockname()[1]
    (pack / "config.json").write_text(json.dumps(config))
    results = {}
    try:
        modes = ("plain",) if args.inventory_only else ("plain", "sandbox_gpu", "sandbox_no_gpu")
        for mode in modes:
            for repeat in range(1 if args.inventory_only else args.repeats):
                cell_name = f"{mode}_{repeat + 1}"
                registry = TypeRegistry()
                register_core_types(registry)
                policy = SandboxPolicy(
                    gpu=mode == "sandbox_gpu",
                    ro_binds=tuple(args.ro_bind) + ((args.model,) if args.model else ()),
                )
                launcher = SubprocessLauncher() if mode == "plain" else BubblewrapLauncher(policy)
                worker = IsolatedWorker(
                    manifest,
                    registry,
                    launcher=launcher,
                    extra_env={"CUDA_VISIBLE_DEVICES": "" if args.cpu_only else args.cuda_device},
                )
                started = time.perf_counter()
                try:
                    await worker.start()
                    engine = Engine(
                        schemas=dict(worker.schemas),
                        registry=registry,
                        worker=worker,
                        cache=MemoryLRUCache(),
                    )
                    result = await engine.run(
                        Graph(nodes={"probe": GraphNode("wslprobe.inspect", {})}), ["probe"]
                    )
                    observed = json.loads(str(result.outputs["probe"]["report"].resolve()))
                    results[cell_name] = {
                        "mode": mode,
                        "seconds": time.perf_counter() - started,
                        "node": observed,
                    }
                except Exception as exc:
                    results[cell_name] = {
                        "mode": mode,
                        "seconds": time.perf_counter() - started,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                finally:
                    await worker.close()
                (work / f"{cell_name}.json").write_text(
                    json.dumps(results[cell_name], indent=2) + "\n"
                )
    finally:
        server.close()
        await server.wait_closed()
    return results


def assess(workers, cpu_only: bool, unchanged_driver: bool) -> dict[str, str]:
    checks = {"driver_digest_unchanged": "pass" if unchanged_driver else "fail"}
    plain = workers.get("plain_1", {}).get("node", {})
    for cell_name, cell in workers.items():
        mode = cell["mode"]
        if "error" in cell:
            checks[cell_name] = "fail"
            continue
        node = cell["node"]
        if mode == "plain":
            checks[f"{cell_name}_controls"] = (
                "pass"
                if node["windows_sentinel"]["readable"] and node["host_loopback"]["connected"]
                else "fail"
            )
        else:
            checks[f"{cell_name}_escape"] = (
                "pass"
                if not any(
                    (
                        node["windows_sentinel"]["readable"],
                        node["windows_drive_visible"],
                        node["run_wsl_visible"],
                        node["init"]["executed"],
                        node["host_loopback"]["connected"],
                        node["external_tcp"]["connected"],
                        node["driver_write_open"]["opened"],
                        node["interop_socket"].get("connected", False),
                    )
                )
                else "fail"
            )
            checks[f"{cell_name}_external_control"] = (
                "pass" if plain.get("external_tcp", {}).get("connected") else "inconclusive"
            )
            checks[f"{cell_name}_interop_control"] = (
                "pass" if plain.get("interop_socket", {}).get("connected") else "inconclusive"
            )
            checks[f"{cell_name}_device"] = (
                "pass" if node["dxg_visible"] == (mode == "sandbox_gpu") else "fail"
            )
            if mode == "sandbox_gpu":
                checks[f"{cell_name}_driver_mount_readonly"] = (
                    "pass" if node["driver_mount_readonly"] is True else "fail"
                )
        if cpu_only:
            checks[f"{cell_name}_compute"] = "not_run"
        elif mode == "sandbox_no_gpu":
            checks[f"{cell_name}_compute"] = (
                "pass"
                if "error" in node.get("matmul", {}) and "error" in node.get("model_load", {})
                else "fail"
            )
        else:
            checks[f"{cell_name}_compute"] = (
                "pass"
                if node.get("matmul", {}).get("finite") and node.get("model_load", {}).get("loads")
                else "fail"
            )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--windows-sentinel", required=True)
    parser.add_argument("--model")
    parser.add_argument("--interop-socket")
    parser.add_argument("--driver-library", default="/usr/lib/wsl/lib/libcuda.so.1")
    parser.add_argument("--external-host", default="1.1.1.1")
    parser.add_argument("--external-port", type=int, default=443)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--loads", type=int, default=3)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--ro-bind", action="append", default=[])
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--lock", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.lock = [str(path.resolve()) for path in args.lock]
    args.preflight = str(args.preflight.resolve()) if args.preflight else None
    if args.inventory_only:
        args.cpu_only = True
    if sys.platform != "linux":
        parser.error("run inside WSL Linux")
    if not args.cpu_only and not args.model:
        parser.error("GPU measurements require --model on the WSL disk")
    if args.seconds < 5 or args.loads < 1 or args.repeats < 3:
        parser.error("seconds must be >=5, repeats >=3 and loads positive")
    if not Path(args.windows_sentinel).is_file():
        parser.error("the exact Windows sentinel must exist for the plain control")
    sentinel = Path(args.windows_sentinel)
    if len(sentinel.parts) < 4 or sentinel.parts[1] != "mnt":
        parser.error("Windows sentinel must be an absolute /mnt/<drive>/... path")
    if args.model and not Path(args.model).is_file():
        parser.error("model must be an existing safetensors file")
    work = Path(args.work_dir).resolve()
    work.mkdir()
    os.environ["TMPDIR"] = str(work)
    tempfile.tempdir = str(work)
    before = file_digest(Path(args.driver_library))
    supervisor = inventory()
    source = file_digest(Path(__file__))
    model = file_digest(Path(args.model)) if args.model else None
    bwrap = file_digest(Path(shutil.which("bwrap") or "/missing-bwrap"))
    locks = [file_digest(Path(path)) for path in args.lock]
    if not args.cpu_only:
        if not args.preflight:
            parser.error("GPU runs require a saved --inventory-only receipt via --preflight")
        preflight = json.loads(Path(args.preflight).read_text())
        for key, actual in (
            ("source", source),
            ("model", model),
            ("supervisor", supervisor),
            ("bubblewrap", bwrap),
            ("locks", locks),
        ):
            if preflight.get(key) != actual:
                parser.error(f"preflight {key} differs; rerun CPU inventory before GPU admission")
        worker_inventory = preflight.get("workers", {}).get("plain_1", {}).get("node", {})
        if worker_inventory != supervisor or supervisor["installed_pack_count"]:
            parser.error(
                "worker inventory mismatch or installed packs need committed lock comparison"
            )
    receipt = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source": source,
        "model": model,
        "model_provenance": "existing local file; no historical hash/official revision asserted",
        "supervisor": supervisor,
        "bubblewrap": bwrap,
        "locks": locks,
        "driver_before": before,
        "kernel": os.uname().release,
        "arguments": vars(args),
        "workers": asyncio.run(run_workers(args, work)),
        "driver_after": file_digest(Path(args.driver_library)),
    }
    receipt["checks"] = (
        {
            "inventory": "pass"
            if receipt["workers"].get("plain_1", {}).get("node") == supervisor
            and not supervisor["installed_pack_count"]
            else "fail"
        }
        if args.inventory_only
        else assess(
            receipt["workers"],
            args.cpu_only,
            before == receipt["driver_after"] and "sha256" in before,
        )
    )
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return int(any(value in ("fail", "inconclusive") for value in receipt["checks"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
