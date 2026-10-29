"""A persisted declaration surface with runtime activation on demand."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from dinkster_protocol import Invocation, InvocationResult, OnInvocationEvent
from dinkster_values import TypeRegistry
from dinkster_workers.catalog import (
    CatalogTypes,
    PackCatalog,
    source_digest,
    worker_declarations_match_catalog,
)
from dinkster_workers.manifest import PackManifest


class CatalogTypeRegistry:
    """Validation metadata only; runtime codecs stay in the live registry."""

    def __init__(self, registry: TypeRegistry, declarations: tuple[CatalogTypes, ...]) -> None:
        self._registry = registry
        self._declarations = declarations

    def __contains__(self, type_id: str) -> bool:
        return type_id in self._registry or any(
            type_id in item.type_ids for item in self._declarations
        )

    def equivalent_type(self, type_id: str) -> str | None:
        return self._registry.equivalent_type(type_id) or next(
            (other for item in self._declarations if (other := item.equivalences.get(type_id))),
            None,
        )

    def asset_decoder_for(self, type_id: str) -> object | None:
        return self._registry.asset_decoder_for(type_id) or next(
            (type_id for item in self._declarations if type_id in item.asset_decoders), None
        )

    def batch_merge_for(self, type_id: str) -> object | None:
        return self._registry.batch_merge_for(type_id) or next(
            (type_id for item in self._declarations if type_id in item.batch_merges), None
        )


class LazyWorker:
    def __init__(
        self,
        manifest: PackManifest,
        catalog: PackCatalog,
        start: Callable[[], Awaitable[Any]],
        close: Callable[[], Awaitable[None]],
        on_declarations_changed: Callable[[], None] | None = None,
    ) -> None:
        self.pack = manifest.name
        self._manifest = manifest
        self.catalog = catalog
        self._start = start
        self._close = close
        self._on_declarations_changed = on_declarations_changed
        self.declarations_changed = False
        self._owns_worker = True
        self._worker: Any = None
        self._activation: asyncio.Task[Any] | None = None
        self._closing: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def cold(self) -> bool:
        return (
            not self._closed
            and self._worker is None
            and (self._activation is None or not self._activation.done())
        )

    @property
    def alive(self) -> bool:
        return not self._closed and self._worker is not None and self._worker.alive

    async def start(self) -> None:
        """Publish declarations without activating the execution runtime."""

    @property
    def instance_token(self) -> str | None:
        return None if self._worker is None else self._worker.instance_token

    @property
    def workgroup_capabilities(self) -> frozenset[str]:
        return frozenset() if self._worker is None else self._worker.workgroup_capabilities

    def bind_workgroup_endpoint(self, *args: Any, **kwargs: Any) -> Any:
        return self._worker.bind_workgroup_endpoint(*args, **kwargs)

    def unbind_workgroup_endpoint(self, *args: Any, **kwargs: Any) -> None:
        self._worker.unbind_workgroup_endpoint(*args, **kwargs)

    async def prepare(self, node_types: Sequence[str]) -> None:
        missing = set(node_types) - self.catalog.schemas.keys()
        if missing:
            raise KeyError(f"worker has no implementation for: {', '.join(sorted(missing))}")

    async def invoke(
        self, invocation: Invocation, on_event: OnInvocationEvent | None = None
    ) -> InvocationResult:
        worker = await self.ensure_started()
        self.validate_schema(invocation.node_type)
        return await worker.invoke(invocation, on_event=on_event)

    def validate_schema(self, node_type: str) -> None:
        if self.declarations_changed and self.catalog.schemas.get(
            node_type
        ) != self._worker.schemas.get(node_type):
            raise RuntimeError(f"pack {self.pack!r} schema changed; refresh the graph and retry")

    def __getattr__(self, name: str) -> Any:
        if name in (
            "schemas",
            "combo_choices",
            "lazy_choice_ids",
            "compat_skips",
            "body_arms",
            "extension_contributions",
            "renditions",
        ):
            return getattr(self.catalog, name)
        if name in ("attention_capabilities", "attention_route_token"):
            return None if self._worker is None else getattr(self._worker, name)
        if name == "can_convert_legacy_checkpoint":
            return self._worker is not None and self._worker.can_convert_legacy_checkpoint
        if name in (
            "check_lazy_status",
            "fetch_choices",
            "call_pack_route",
            "resolve_rendition",
            "resolve_rendition_mime",
            "render_rendition",
            "materialize_sampler_registry",
            "materialize_inference_generation",
            "compile_graph",
            "convert_legacy_checkpoint",
        ):

            async def call(*args: Any, **kwargs: Any) -> Any:
                worker = await self.ensure_started()
                return await getattr(worker, name)(*args, **kwargs)

            return call
        if self._worker is None:
            raise AttributeError(name)
        return getattr(self._worker, name)

    async def ensure_started(self) -> Any:
        if self._closed:
            raise RuntimeError(f"pack {self.pack!r} is closed")
        if self._worker is not None:
            return self._worker
        if self._activation is None:
            self._activation = asyncio.create_task(self._activate())
        return await asyncio.shield(self._activation)

    def validate_source(self) -> None:
        if source_digest(self._manifest) != self.catalog.source:
            raise RuntimeError(f"pack {self.pack!r} changed; refresh its catalog and restart")

    def adopt(self, worker: Any) -> None:
        if not worker_declarations_match_catalog(worker, self.catalog):
            if self._on_declarations_changed is None:
                raise RuntimeError(
                    f"pack {self.pack!r} declarations changed; run dinkster-doctor and restart"
                )
            self.declarations_changed = True
        self._worker = worker
        if self.declarations_changed:
            assert self._on_declarations_changed is not None
            self._on_declarations_changed()

    def detach(self) -> None:
        """Keep pinned routes usable while the composition owns the live worker."""
        self._owns_worker = False

    async def _activate(self) -> Any:
        try:
            self.validate_source()
            worker = await self._start()
            self.adopt(worker)
            return worker
        except BaseException:
            await self._close()
            raise

    async def release_inference_generation(self, key: str) -> None:
        if self.alive:
            await self._worker.release_inference_generation(key)

    async def close(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._finish_close())
        try:
            await asyncio.shield(self._closing)
        except asyncio.CancelledError as cancelled:
            while not self._closing.done():
                try:
                    await asyncio.shield(self._closing)
                except asyncio.CancelledError:
                    continue
            await self._closing
            raise cancelled

    async def _finish_close(self) -> None:
        self._closed = True
        if self._activation is not None:
            try:
                await asyncio.shield(self._activation)
            except BaseException:
                pass
        if self._owns_worker:
            await self._close()
