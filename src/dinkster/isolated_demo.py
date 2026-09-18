"""Isolation demo: the toy image graph, with the whole node pack out-of-process.

What this proves (the boundary proves itself):

- The engine is unchanged: it drives an IsolatedWorker through the same
  Worker protocol as InProcessWorker (hazard H3).
- The parent process NEVER imports the pack. It learns the node interface
  from the hello handshake, in the schema wire format (hazard H1). This
  process does not even register the dev.image type - yet it can cache,
  route, and interrogate image values (meta crossed the boundary), it just
  cannot resolve() them locally.
- Fingerprints computed in the worker key the parent's cache: run 2 is
  served without a single boundary crossing (hazard H4).
- Dev-mode diagnostics (DESIGN 3.9): every crossing reports per-edge
  transport, payload size, codec time, and fallback-codec hits.

Run: ``uv run dinkster-isolated``
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

from dinkster_assets.save_target import register_save_target_type
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import TypeRegistry, register_core_types
from dinkster_values.model import UnresolvablePayload
from dinkster_workers import BoundaryDiagnostic, IsolatedWorker


def find_dev_manifest() -> Path:
    """Locate the nodes-dev pack manifest WITHOUT importing the pack."""
    spec = importlib.util.find_spec("dinkster_nodes_dev")
    if spec is None or spec.origin is None:
        raise RuntimeError("dinkster-nodes-dev is not installed")
    package_dir = Path(spec.origin).parent  # .../src/dinkster_nodes_dev
    manifest = package_dir.parents[1] / "dinkster-pack.toml"
    if not manifest.exists():
        raise RuntimeError(f"pack manifest not found at {manifest}")
    return manifest


def build_graph() -> Graph:
    return Graph(
        nodes={
            "gradient": GraphNode("dev.image.gradient", {"width": 320, "height": 200}),
            "inverted": GraphNode("dev.image.invert", {"image": Link("gradient", "image")}),
            "blended": GraphNode(
                "dev.image.blend",
                {
                    "a": Link("gradient", "image"),
                    "b": Link("inverted", "image"),
                    "ratio": 0.25,
                },
            ),
            "stats": GraphNode("dev.image.stats", {"image": Link("blended", "image")}),
            "save": GraphNode(
                "dev.image.save_pgm",
                {
                    "image": Link("blended", "image"),
                    "target": {"mount": "demo-output", "prefix": "blend"},
                },
            ),
        }
    )


def print_diagnostic(diag: BoundaryDiagnostic) -> None:
    print(
        f"  [boundary] {diag.node_id} ({diag.node_type}) "
        f"execute={diag.execute_ms:.1f}ms boundary={diag.boundary_ms:.1f}ms"
    )
    for direction, costs in (("in", diag.inputs), ("out", diag.outputs)):
        for cost in costs:
            notes: list[str] = []
            if cost.reused:
                notes.append("bytes relayed, no re-encode")
            elif not cost.declared_codec:
                notes.append("fallback codec")
            suffix = f"  ({'; '.join(notes)})" if notes else ""
            print(
                f"    {direction:>3} {cost.edge_id:<8} {cost.type_id:<12} "
                f"{cost.transport:<6} {cost.size_bytes:>8}B "
                f"codec={cost.codec_ms:.2f}ms{suffix}"
            )


async def run() -> None:
    manifest = find_dev_manifest()
    assert "dinkster_nodes_dev" not in sys.modules, "the parent must not import the pack"

    registry = TypeRegistry()
    register_core_types(registry)  # dev.image stays foreign here
    # dinkster.save_target is host-owned (std composition registers it in a
    # real server); the parent needs it to wrap the save node's target
    # input. The pack's own types still stay foreign.
    register_save_target_type(registry)

    # Grant one writable mount BEFORE the worker spawns: the child inherits
    # DINKSTER_MOUNTS_SNAPSHOT at spawn (the snapshot file is the same shape a
    # real server publishes, and its CONTENT stays live-editable afterward).
    base = Path(tempfile.mkdtemp(prefix="dinkster-demo-"))
    out_root = base / "output"
    out_root.mkdir()
    snapshot = base / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "demo-output", "root": str(out_root), "mode": "readwrite"}]}),
        "utf-8",
    )
    os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = str(snapshot)

    print(f"starting isolated worker from {manifest} ...")
    worker = IsolatedWorker(
        manifest,
        registry,
        shm_threshold=64 * 1024,  # 320x200 float32 images take the shm transport
        on_diagnostic=print_diagnostic,
    )
    await worker.start()
    try:
        print(f"pack '{worker.pack}' announced {len(worker.schemas)} node schemas over the wire\n")
        engine = Engine(
            schemas=dict(worker.schemas),
            registry=registry,
            worker=worker,
            cache=MemoryLRUCache(),
        )
        graph = build_graph()

        print("run 1: everything crosses the boundary")
        result = await engine.run(graph, ["stats", "save", "blended"])
        mean = result.outputs["stats"]["mean"].resolve()
        print(f"\n  stats.mean = {mean} (core.float: decodable here)")

        image = result.outputs["blended"]["image"]
        print(
            f"  blended.image: type={image.type_id} "
            f"shape={image.meta.get('shape')} dtype={image.meta.get('dtype')} "
            f"fingerprint={image.fingerprint[:12]}..."
        )
        try:
            image.resolve()
        except UnresolvablePayload as exc:
            print(f"  blended.image.resolve() -> {exc}")
        saved = result.outputs["save"]["path"].resolve()
        print(f"  saved (in the worker process): {saved} under {out_root}")

        print("\nrun 2: served from the parent's cache, zero crossings")
        again = await engine.run(graph, ["stats", "blended"])
        print(f"  executed={list(again.executed)} cached={sorted(again.cached)}")
    finally:
        await worker.close()
    print("\nworker closed; shm handoffs all acknowledged and unlinked")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
