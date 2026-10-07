"""Serve real CUDA allocations through the memory governor for panel inspection.

Run under the assigned GPU flock with UUID-scoped CUDA_VISIBLE_DEVICES.
The tensors are pressure buffers, not inference models. Unload controls use
the production /cache/trim endpoint and the governor's item contract.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from aiohttp import web
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_memory import BudgetExceeded, ConsumerItem, MemoryGovernor, PressureSignal
from dinkster_native.devices import torch_vram_telemetry
from dinkster_server import create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from dinkster.compose import default_pack_ids, default_pack_spec

DEVICE = "vram:cuda:0"
GIB = 1024**3


def inventory(output: Path) -> None:
    packs = []
    for name in default_pack_ids():
        spec = default_pack_spec(name)
        info = spec.packs[name]
        packs.append({"name": name, "version": info.version, "digest": info.artifact_digest})
    output.write_text(json.dumps({"python": sys.executable, "packs": packs}, indent=2) + "\n")


async def run(arguments: argparse.Namespace) -> None:
    import torch

    class PressureBuffers:
        def __init__(self) -> None:
            self.buffers: dict[str, torch.Tensor] = {}

        def footprint(self, device: str) -> int:
            return (
                sum(tensor.numel() for tensor in self.buffers.values()) if device == DEVICE else 0
            )

        def details(self) -> list[ConsumerItem]:
            return [
                ConsumerItem(key, key, {DEVICE: tensor.numel()})
                for key, tensor in self.buffers.items()
            ]

        async def shed(self, pressure: PressureSignal) -> int:
            if pressure.device != DEVICE:
                return 0
            freed = 0
            for key in list(self.buffers):
                if pressure.items is not None and key not in pressure.items:
                    continue
                freed += self.buffers.pop(key).numel()
                if freed >= pressure.bytes_needed:
                    break
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return freed

    buffers = PressureBuffers()
    governor = MemoryGovernor({DEVICE: 11 * GIB}, telemetry=torch_vram_telemetry)
    governor.register_shedder(buffers, name="CUDA pressure buffers")
    records: list[dict[str, object]] = []

    def record(state: str) -> None:
        records.append(
            {"state": state, "timestamp": time.time(), "memoryGovernor": governor.status()}
        )
        arguments.output.write_text(json.dumps(records, indent=2) + "\n")

    record("idle")
    for name, size in (("Pressure buffer A", 6 * GIB), ("Pressure buffer B", 4 * GIB)):
        async with governor.reserve(DEVICE, size):
            buffers.buffers[name] = torch.zeros(size, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    record("loaded")
    try:
        async with governor.reserve(DEVICE, 12 * GIB):
            raise AssertionError("over-budget reservation was admitted")
    except BudgetExceeded:
        record("over-budget reservation refused")

    def engine(on_event=None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        return Engine(
            schemas={},
            registry=registry,
            worker=InProcessWorker({}, registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    app = create_app(engine, {}, governor=governor)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", arguments.port).start()
    print(f"Pressure server on port {arguments.port}", flush=True)
    try:
        for _ in range(arguments.seconds):
            await asyncio.sleep(1)
            record("live")
    finally:
        await runner.cleanup()
        await buffers.shed(PressureSignal(DEVICE, buffers.footprint(DEVICE)))
        record("released")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=5397)
    parser.add_argument("--seconds", type=int, default=180)
    arguments = parser.parse_args()
    if arguments.inventory is not None:
        inventory(arguments.inventory)
        return
    if arguments.output is None:
        parser.error("--output is required for GPU use")
    asyncio.run(run(arguments))


if __name__ == "__main__":
    main()
