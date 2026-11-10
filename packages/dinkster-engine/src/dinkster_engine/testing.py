"""Synchronous, single-pack tests through the real engine and workers."""

from __future__ import annotations

import asyncio
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from dinkster_caches import MemoryLRUCache
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import build_schemas
from dinkster_workers import IsolatedWorker
from dinkster_workers.host import load_pack as load_worker_pack
from dinkster_workers.manifest import add_pack_root_to_import_path, load_manifest

from .engine import Engine

__all__ = ["load_pack"]


@dataclass(frozen=True)
class PackRunResult:
    """Decoded output values and the engine's execution/cache observations."""

    outputs: Mapping[str, Mapping[str, object]]
    executed: frozenset[str]
    cached: frozenset[str]

    def value(self, node: str, output: str) -> object:
        return self.outputs[node][output]


class PackRunner:
    """One engine, cache and worker shared by synchronous test calls."""

    def __init__(self, runner: asyncio.Runner, engine: Engine) -> None:
        self._runner = runner
        self._engine = engine

    @staticmethod
    def link(node: str, output: str) -> Link:
        return Link(node, output)

    def run(
        self,
        nodes: Mapping[str, tuple[str, Mapping[str, object]]],
        targets: Sequence[str] | None = None,
    ) -> PackRunResult:
        """Run Python graph declarations; by default return every node's outputs."""
        graph = Graph(
            nodes={
                name: GraphNode(node_type, dict(inputs))
                for name, (node_type, inputs) in nodes.items()
            }
        )

        async def execute() -> PackRunResult:
            result = await self._engine.run(graph, list(nodes) if targets is None else targets)
            outputs = MappingProxyType(
                {
                    node: MappingProxyType(
                        {name: value.resolve() for name, value in values.items()}
                    )
                    for node, values in result.outputs.items()
                }
            )
            return PackRunResult(outputs, frozenset(result.executed), frozenset(result.cached))

        return self._runner.run(execute())

    def run_node(self, node_type: str, inputs: Mapping[str, object]) -> PackRunResult:
        """Run one node, whose result is available under the node id ``node``."""
        return self.run({"node": (node_type, inputs)})


@contextmanager
def load_pack(manifest_path: Path | str, *, in_process: bool = False) -> Generator[PackRunner]:
    """Load a manifest and keep its worker alive until the context exits.

    Use this context in a module- or session-scoped pytest fixture to share
    worker startup across tests. Dependencies must already be installed in
    the test interpreter. ``in_process=True`` opts out of the process boundary.
    """
    import sys

    manifest = load_manifest(manifest_path)
    add_pack_root_to_import_path(manifest, sys.path)
    with asyncio.Runner() as runner:

        async def start():
            local, registry, classes, _arms = load_worker_pack(manifest)
            if in_process:
                await local.start()
                return local, registry, build_schemas(classes)
            # Register pack-owned codecs in the caller as well as in the child.
            await local.close()
            isolated = IsolatedWorker(manifest.path, registry)
            try:
                await isolated.start()
            except BaseException:
                await isolated.close()
                raise
            return isolated, registry, dict(isolated.schemas)

        worker, registry, schemas = runner.run(start())
        try:
            yield PackRunner(
                runner,
                Engine(schemas=schemas, registry=registry, worker=worker, cache=MemoryLRUCache()),
            )
        finally:
            runner.run(worker.close())
