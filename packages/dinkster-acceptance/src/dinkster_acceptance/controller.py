from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import RemoteWorker


async def run(host: str, port: int, token_file: Path, output: Path) -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    worker = RemoteWorker(
        host,
        port,
        token_file.read_text(encoding="utf-8").strip(),
        registry,
        name="model-sampling-flux-worker",
        connect_timeout=60.0,
    )
    await worker.start()
    try:
        engine = Engine(
            schemas=dict(worker.schemas),
            registry=registry,
            worker=worker,
            cache=MemoryLRUCache(),
        )
        result = await engine.run(
            Graph(nodes={"acceptance": GraphNode("acceptance.model_sampling_flux", {})}),
            ["acceptance"],
            run_id="model-sampling-flux-cross-machine",
        )
        report_value = result.outputs["acceptance"]["report"].resolve()
        if not isinstance(report_value, str):
            raise TypeError("acceptance worker returned a non-string report")
        report = json.loads(report_value)
        report["remote_worker"] = {
            "name": "model-sampling-flux-worker",
            "instance_token": worker.instance_token,
            "endpoint": f"{host}:{port}",
            "executed": list(result.executed),
            "cached": list(result.cached),
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
    finally:
        await worker.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.host, args.port, args.token_file, args.output))


if __name__ == "__main__":
    main()
