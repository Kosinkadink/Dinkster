"""Task-local facts supplied by the worker host for one node execution."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

from dinkster_assets import AssetRef
from dinkster_protocol import (
    AttentionPolicy,
    AttentionRouteToken,
    ExportSnapshot,
    MediaSourceAuthority,
    PreviewAnimation,
    PreviewMode,
    is_extension_snapshot_digest,
    resolve_attention_runtime_status,
    validate_preview_animation,
    validate_preview_mode,
)


def _not_cancelled() -> bool:
    return False


class SourceMaterializer(Protocol):
    def __call__(self, asset: AssetRef, kind: str, category: str) -> str: ...


class ArtifactSink(Protocol):
    def __call__(self, node_id: str, filename: str, subfolder: str, folder_type: str) -> None: ...


class SourceStagingSession(Protocol):
    def materialize(self, asset: AssetRef, kind: str, category: str) -> str: ...

    def close(self) -> None: ...


class SourceStagingProvider(Protocol):
    def sweep(self) -> None: ...

    def open(
        self, invocation_id: str, authorities: Sequence[MediaSourceAuthority]
    ) -> SourceStagingSession: ...


@dataclass(frozen=True)
class ExecutionContext:
    """Host-authoritative facts for one worker invocation.

    ``node_id`` is the executing graph node's id. It is host-authoritative
    and None when the context is not invocation-scoped.
    """

    arm: str | None
    expected_execution_identity: str | None
    fp8_matmul: bool = False
    diffusion_dtype: str | None = None
    text_dtype: str | None = None
    vae_dtype: str | None = None
    attention_policy: AttentionPolicy = "auto"
    attention_route_token: AttentionRouteToken | None = None
    extension_snapshot_digest: str | None = None
    preview_mode: PreviewMode = "off"
    preview_animation: PreviewAnimation = "ring"
    cancelled: Callable[[], bool] = _not_cancelled
    node_id: str | None = None
    export_snapshot: ExportSnapshot | None = None
    materialize_source: SourceMaterializer | None = None
    artifact_sink: ArtifactSink | None = None
    started_at_ns: int = 0

    def __post_init__(self) -> None:
        if self.extension_snapshot_digest is not None and not is_extension_snapshot_digest(
            self.extension_snapshot_digest
        ):
            raise ValueError("ExecutionContext.extension_snapshot_digest must be a sha256 digest")
        if self.fp8_matmul and self.expected_execution_identity is None:
            raise ValueError("ExecutionContext.fp8_matmul requires expected_execution_identity")
        dtypes = (self.diffusion_dtype, self.text_dtype, self.vae_dtype)
        if any(dtype is not None for dtype in dtypes) and (
            self.expected_execution_identity is None
            or not all(isinstance(dtype, str) and dtype for dtype in dtypes)
        ):
            raise ValueError(
                "ExecutionContext component dtypes must be complete and require expected identity"
            )
        resolve_attention_runtime_status(self.attention_policy, self.attention_route_token)
        validate_preview_mode(self.preview_mode)
        validate_preview_animation(self.preview_animation)
        if self.started_at_ns < 0:
            raise ValueError("ExecutionContext.started_at_ns must be nonnegative")


_execution_context: ContextVar[ExecutionContext | None] = ContextVar(
    "dinkster_execution_context", default=None
)


def current_execution_context() -> ExecutionContext | None:
    """The current worker invocation's host-authoritative execution facts."""
    return _execution_context.get()


@contextmanager
def use_execution_context(context: ExecutionContext) -> Generator[None]:
    token = _execution_context.set(context)
    try:
        yield
    finally:
        _execution_context.reset(token)
