"""CPU compact-storage memory gate: python tools/media_storage_conformance.py."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import TypeRegistry, Value, register_core_types
from dinkster_values.image_codec import image_array_fingerprint, image_array_meta
from dinkster_values.storage import image_input
from dinkster_workers import InProcessWorker


class TrackingMemoryCache(MemoryLRUCache):
    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        super().__init__(max_entries=max_entries, max_bytes=max_bytes)
        self.peak_ram = 0

    async def put(self, key: str, outputs: Mapping[str, Value]) -> None:
        await super().put(key, outputs)
        self.peak_ram = max(self.peak_ram, self.footprint("ram"))


def _peak_rss() -> int:
    if sys.platform == "win32":
        import psutil

        return cast(Any, psutil.Process().memory_info()).peak_wset
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


async def measure(frames: int, height: int, width: int) -> dict[str, int]:
    shape = (frames, height, width, 3)
    batch_bytes = frames * height * width * 3
    image_type = TypeExpr.concrete("comfy.IMAGE")
    integer_type = TypeExpr.concrete("core.int")

    class Source(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="measure.source", outputs=(OutputSpec("images", image_type),)
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            images = np.full(shape, 127, dtype=np.uint8)
            return cls.outputs(images=images)

    class View(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="measure.view",
                inputs=(InputSpec("images", image_type, accepts_storage=True),),
                outputs=(OutputSpec("images", image_type),),
            )

        @classmethod
        def execute(cls, images: object) -> Mapping[str, object]:
            array = np.asarray(images)
            if array.dtype != np.uint8 or array.shape != shape:
                raise AssertionError(
                    "storage-aware consumer received an expanded or reshaped batch"
                )
            return cls.outputs(images=array.view())

    class Inspect(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="measure.inspect",
                inputs=(InputSpec("images", image_type, accepts_storage=True),),
                outputs=(OutputSpec("sum", integer_type),),
            )

        @classmethod
        def execute(cls, images: object) -> Mapping[str, object]:
            array = np.asarray(images)
            if array.dtype != np.uint8:
                raise AssertionError("storage-aware consumer received an expanded batch")
            return cls.outputs(sum=int(array.sum(dtype=np.uint64)))

    registry = TypeRegistry()
    register_core_types(registry)
    registry.register(
        "comfy.IMAGE",
        fingerprint=image_array_fingerprint("comfy.IMAGE"),
        meta=image_array_meta,
        input_convert=image_input,
    )
    nodes = [Source, View, Inspect]
    cache = TrackingMemoryCache(max_entries=1, max_bytes=batch_bytes * 3 // 2)
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=cache,
    )
    graph = Graph(
        nodes={
            "source": GraphNode("measure.source", {}),
            "view": GraphNode("measure.view", {"images": Link("source", "images")}),
            "inspect": GraphNode("measure.inspect", {"images": Link("view", "images")}),
        }
    )
    baseline = _peak_rss()
    result = await engine.run(graph, ["inspect"])
    if result.outputs["inspect"]["sum"].resolve() != batch_bytes * 127:
        raise AssertionError("three-node chain changed the source bytes")
    peak = _peak_rss()
    return {
        "frames": frames,
        "height": height,
        "width": width,
        "channels": 3,
        "batch_bytes": batch_bytes,
        "baseline_rss": baseline,
        "peak_rss": peak,
        "rss_growth": peak - baseline,
        "limit": 2 * batch_bytes,
        "cache_budget": cache.max_bytes,
        "cache_peak_resident": cache.peak_ram,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    if min(args.frames, args.height, args.width) < 1:
        parser.error("dimensions and frame count must be positive")
    measurements = asyncio.run(measure(args.frames, args.height, args.width))
    print(json.dumps(measurements, sort_keys=True), flush=True)
    if measurements["rss_growth"] >= measurements["limit"]:
        raise SystemExit("compact batch chain exceeded its peak RSS bound")
    if measurements["cache_peak_resident"] > measurements["cache_budget"]:
        raise SystemExit("compact batch chain exceeded its cache byte budget")


if __name__ == "__main__":
    main()
