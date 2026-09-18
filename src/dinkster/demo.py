"""Demo: a toy graph run end-to-end through the real engine.

Run 1 executes everything; run 2 is served entirely from cache; run 3 changes
one literal and re-executes only the dirty suffix. The save node is
non-idempotent and never cached. Engine events are printed as they arrive -
the seed of dev mode (DESIGN 3.9). Run 4 shows dynamic inputs: an autogrow
family elaborated from the document, growing the graph without a schema change.
Run 5 shows dynamic outputs: document-stored output members flowing into
dynamic inputs - the whole topology is known before anything executes.
Run 6 shows parallelism: independent branches overlap and identical work
coalesces (the delay nodes are io_bound - waits, not compute - so they are
exempt from the default compute lane). Run 7 shows resource admission:
gpu-occupying nodes serialize on the default capacity-1 lane while io-bound
nodes overlap freely.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_dev import PACK_NODES, register_dev_types
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, load_manifest
from dinkster_workers.host import load_pack

from .compose import default_pack_specs


class GpuDelay(Node):
    """Demo-only Delay that declares it occupies the gpu lane while running.
    The declaration is the node's whole involvement (hazard H12): admission -
    how many may occupy "gpu" at once - is engine configuration."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="demo.gpu_delay",
            display_name="GPU Delay",
            category="demo",
            inputs=(
                InputSpec("value", TypeExpr.concrete("core.string")),
                InputSpec("seconds", TypeExpr.concrete("core.float"), default=0.1),
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete("core.string")),),
            occupies=("gpu",),
        )

    @classmethod
    async def execute(cls, *, value: str, seconds: float) -> Mapping[str, object]:
        await asyncio.sleep(seconds)
        return cls.outputs(value=value)


def grant_output_mount(base: Path, mount_id: str = "demo-output") -> Path:
    """Demo-scale mount plumbing: publish a one-mount snapshot (the same
    file MountTable.write_snapshot produces in a real server) and point
    DINKSTER_MOUNTS_SNAPSHOT at it, so save nodes may write - into exactly
    this folder, nowhere else."""
    root = base / "output"
    root.mkdir(parents=True, exist_ok=True)
    snapshot = base / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": mount_id, "root": str(root), "mode": "readwrite"}]}),
        "utf-8",
    )
    os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = str(snapshot)
    return root


def build_graph(blend_ratio: float, prefix: str) -> Graph:
    return Graph(
        nodes={
            "gradient": GraphNode("dev.image.gradient", {"width": 320, "height": 200}),
            "inverted": GraphNode("dev.image.invert", {"image": Link("gradient", "image")}),
            "blended": GraphNode(
                "dev.image.blend",
                {
                    "a": Link("gradient", "image"),
                    "b": Link("inverted", "image"),
                    "ratio": blend_ratio,
                },
            ),
            "stats": GraphNode("dev.image.stats", {"image": Link("blended", "image")}),
            "save": GraphNode(
                "dev.image.save_pgm",
                {
                    "image": Link("blended", "image"),
                    "target": {"mount": "demo-output", "prefix": prefix},
                },
            ),
        }
    )


def print_event(event: EngineEvent) -> None:
    detail = dict(event.detail)
    detail.pop("cache_key", None)
    extras = f" {detail}" if detail else ""
    print(f"  [{event.kind:>13}] {event.node_id or '-'}{extras}")


async def run_demo() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    default_nodes: list[type[Node]] = []
    for default_spec in default_pack_specs():
        if not default_spec.in_process:
            continue
        _, _, pack_nodes, _ = load_pack(
            load_manifest(Path(default_spec.manifest)),
            import_from_pack_root=True,
            registry=registry,
        )
        default_nodes.extend(pack_nodes)
    register_dev_types(registry)

    nodes = [*default_nodes, *PACK_NODES, GpuDelay]
    schemas = build_schemas(nodes)
    worker = InProcessWorker(build_node_types(nodes), registry)
    cache = MemoryLRUCache()
    engine = Engine(
        schemas=schemas, registry=registry, worker=worker, cache=cache, on_event=print_event
    )

    out_dir = grant_output_mount(Path(tempfile.mkdtemp(prefix="dinkster-demo-")))
    targets = ["stats", "save", "blended"]

    print("run 1: cold - everything executes")
    graph = build_graph(0.25, "blend")
    result = await engine.run(graph, targets)
    stats = {k: round(cast(float, v.resolve()), 4) for k, v in result.outputs["stats"].items()}
    print(f"  stats={stats} executed={len(result.executed)} cached={len(result.cached)}")

    print("run 2: identical - compute nodes cached, save re-runs (non-idempotent)")
    graph = build_graph(0.25, "blend")
    result = await engine.run(graph, targets)
    print(f"  executed={list(result.executed)} cached={list(result.cached)}")

    print("run 3: blend ratio changed - only the dirty suffix re-executes")
    graph = build_graph(0.75, "blend")
    result = await engine.run(graph, targets)
    print(f"  executed={list(result.executed)} cached={list(result.cached)}")

    print("run 4: dynamic inputs - autogrow family, members from the document")
    join_graph = Graph(
        nodes={
            "join": GraphNode(
                "std.string.join",
                {"pieces.first": "dinkster", "pieces.second": "elaborates", "separator": " "},
            )
        }
    )
    join = await engine.run(join_graph, ["join"])
    print(f"  2 members -> {join.outputs['join']['text'].resolve()!r}")
    join_graph = Graph(
        nodes={
            "join": GraphNode(
                "std.string.join",
                {
                    "pieces.first": "dinkster",
                    "pieces.second": "elaborates",
                    "pieces.third": "dynamically",
                    "separator": " ",
                },
            )
        }
    )
    join = await engine.run(join_graph, ["join"])
    print(
        f"  3 members -> {join.outputs['join']['text'].resolve()!r} "
        f"(executed={list(join.executed)}: new member set = new cache key)"
    )

    print("run 5: dynamic outputs - document-stored members, split feeding join")
    split_graph = Graph(
        nodes={
            "split": GraphNode(
                "std.string.split",
                {"text": "dinkster plans dynamic topology upfront", "separator": " "},
                output_members={"parts": ("a", "b", "c", "d", "e")},
            ),
            "rejoin": GraphNode(
                "std.string.join",
                {
                    "pieces.x": Link("split", "parts.d"),
                    "pieces.y": Link("split", "parts.e"),
                    "separator": " ",
                },
            ),
        }
    )
    rejoined = await engine.run(split_graph, ["rejoin"])
    print(f"  parts.d + parts.e -> {rejoined.outputs['rejoin']['text'].resolve()!r}")

    print("run 6: parallelism - independent branches overlap, identical work coalesces")
    fan = Graph(
        nodes={
            "a": GraphNode("dev.util.delay", {"value": "a", "seconds": 0.15}),
            "b": GraphNode("dev.util.delay", {"value": "b", "seconds": 0.15}),
            "c": GraphNode("dev.util.delay", {"value": "c", "seconds": 0.15}),
        }
    )
    started = time.perf_counter()
    await engine.run(fan, ["a", "b", "c"])
    wall_ms = (time.perf_counter() - started) * 1000.0
    print(f"  three 150ms nodes in {wall_ms:.0f}ms wall time (ready-set scheduler)")

    shared = Graph(nodes={"d": GraphNode("dev.util.delay", {"value": "shared", "seconds": 0.15})})
    started = time.perf_counter()
    w1, w2 = await asyncio.gather(engine.run(shared, ["d"]), engine.run(shared, ["d"]))
    wall_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"  two concurrent workflows, same computation: executed="
        f"{len(w1.executed) + len(w2.executed)} coalesced="
        f"{len(w1.cached) + len(w2.cached)} in {wall_ms:.0f}ms (single-flight)"
    )

    print("run 7: resource admission - the gpu lane serializes, io-bound work overlaps")
    lanes = Graph(
        nodes={
            "g1": GraphNode("demo.gpu_delay", {"value": "g1", "seconds": 0.1}),
            "g2": GraphNode("demo.gpu_delay", {"value": "g2", "seconds": 0.1}),
            "i1": GraphNode("dev.util.delay", {"value": "i1", "seconds": 0.1}),
            "i2": GraphNode("dev.util.delay", {"value": "i2", "seconds": 0.1}),
        }
    )
    started = time.perf_counter()
    await engine.run(lanes, ["g1", "g2", "i1", "i2"])
    wall_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"  2 gpu + 2 io-bound 100ms nodes in {wall_ms:.0f}ms wall time "
        "(gpu capacity defaults to 1: gpu nodes back-to-back, io free)"
    )

    blended = result.outputs["blended"]["image"]
    print(
        "value interrogation: blended.image is "
        f"type={blended.type_id} fingerprint={blended.fingerprint[:12]}... "
        f"meta={dict(blended.meta.entries)}"
    )
    print(f"cache: {cache.hits} hits / {cache.misses} misses, {len(cache)} entries")
    print(f"images written under {out_dir}")


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
