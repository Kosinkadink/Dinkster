"""Invocation-local execution of declared attention and residual transforms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
from dinkster_inference import (
    AttentionCallContext,
    AttentionContribution,
    AttentionTokenSpan,
    GuidanceCondition,
    GuidanceContractError,
    GuidanceExtensionError,
    SamplingCancelled,
    SamplingExecutionContext,
)

Kernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class _Owned:
    owner: str
    descriptor: Any


class AttentionRegistry:
    """Callbacks retain their pack owner; selection never mutates a model."""

    def __init__(
        self,
        contributions: tuple[tuple[str, AttentionContribution[torch.Tensor]], ...] = (),
    ) -> None:
        self.contributions = contributions
        self.qkv = self._ordered("qkv")
        self.wrappers = self._ordered("wrappers")
        self.outputs = self._ordered("outputs")
        self.blocks = self._ordered("blocks")
        self.backends: dict[str, _Owned] = {}
        for owner, contribution in contributions:
            for descriptor in contribution.backends:
                previous = self.backends.get(descriptor.family)
                if previous is not None:
                    raise GuidanceContractError(
                        f"attention backend for {descriptor.family}: "
                        f"{previous.owner}:{previous.descriptor.id} conflicts with "
                        f"{owner}:{descriptor.id}"
                    )
                self.backends[descriptor.family] = _Owned(owner, descriptor)

    def _ordered(self, name: str) -> tuple[_Owned, ...]:
        return tuple(
            sorted(
                (
                    _Owned(owner, descriptor)
                    for owner, contribution in self.contributions
                    for descriptor in getattr(contribution, name)
                ),
                key=lambda item: (item.descriptor.order, item.descriptor.id),
            )
        )

    @property
    def active(self) -> bool:
        return bool(self.contributions)


@dataclass(frozen=True)
class AttentionExecution:
    registry: AttentionRegistry
    execution: SamplingExecutionContext
    lanes: tuple[GuidanceCondition[torch.Tensor], ...]
    batch_size: int

    def validate_sites(
        self,
        family: str,
        sites: tuple[tuple[str, str], ...],
        block_sites: tuple[tuple[str, str], ...],
    ) -> None:
        for owned in (
            *self.registry.qkv,
            *self.registry.wrappers,
            *self.registry.outputs,
            *self.registry.blocks,
        ):
            selector = owned.descriptor.selector
            candidates = block_sites if owned in self.registry.blocks else sites
            if selector.family == family and not any(
                selector.matches(family, block, kind) for block, kind in candidates
            ):
                raise GuidanceContractError(
                    f"extension={owned.owner} contribution={owned.descriptor.id}: "
                    f"attention selector {selector} matches no declared model point"
                )

    def context(
        self,
        *,
        family: str,
        block: str,
        kind: str,
        heads: int,
        spatial_shape: tuple[int, int],
        query_tokens: int,
        key_tokens: int,
        text_tokens: int = 0,
        reference_tokens: tuple[int, ...] = (),
    ) -> AttentionCallContext:
        spans: list[AttentionTokenSpan] = []
        for index, lane in enumerate(self.lanes):
            for axis, count in (("query", query_tokens), ("key", key_tokens)):
                ranges = (
                    ((0, count, "text" if axis == "key" else "image"),)
                    if kind == "cross"
                    else (
                        (0, text_tokens, "text"),
                        (text_tokens, count - sum(reference_tokens), "image"),
                    )
                    if kind == "joint"
                    else ((0, count, "image"),)
                )
                for start, end, stream in ranges:
                    if start < end:
                        spans.append(
                            AttentionTokenSpan(
                                axis=axis,
                                start=start,
                                end=end,
                                condition_id=lane.id,
                                role=lane.role.value,
                                batch_start=index * self.batch_size,
                                batch_end=(index + 1) * self.batch_size,
                                stream=stream,
                            )
                        )
                start = count - sum(reference_tokens)
                for reference_index, length in enumerate(reference_tokens):
                    spans.append(
                        AttentionTokenSpan(
                            axis=axis,
                            start=start,
                            end=start + length,
                            condition_id=f"{lane.id}:reference:{reference_index}",
                            role=lane.role.value,
                            batch_start=index * self.batch_size,
                            batch_end=(index + 1) * self.batch_size,
                            stream="reference",
                        )
                    )
                    start += length
        return AttentionCallContext(
            family=family,
            block=block,
            kind=kind,
            heads=heads,
            spatial_shape=spatial_shape,
            spans=tuple(spans),
            execution=self.execution,
            state={},
        )

    def _call(self, owned: _Owned, fn: Callable[..., Any], *args: Any) -> Any:
        self.execution.cancellation.check()
        try:
            value = fn(*args)
        except Exception as error:
            if isinstance(
                error, (SamplingCancelled, GuidanceContractError, GuidanceExtensionError)
            ):
                raise
            raise GuidanceExtensionError(
                f"extension={owned.owner} contribution={owned.descriptor.id}: {error}"
            ) from error
        self.execution.cancellation.check()
        return value

    def _context(self, context: AttentionCallContext, owned: _Owned) -> AttentionCallContext:
        return replace(context, state=self.execution.extension_state[owned.owner])

    @staticmethod
    def _tensor(value: object, reference: torch.Tensor, label: str) -> torch.Tensor:
        if (
            type(value) is not torch.Tensor
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise GuidanceContractError(f"{label} returned an incompatible tensor")
        return value

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel: Kernel,
        context: AttentionCallContext,
    ) -> torch.Tensor:
        def selected(items: tuple[_Owned, ...]) -> tuple[_Owned, ...]:
            return tuple(
                item
                for item in items
                if item.descriptor.selector.matches(context.family, context.block, context.kind)
            )

        for owned in selected(self.registry.qkv):
            values = self._call(
                owned, owned.descriptor.transform, q, k, v, self._context(context, owned)
            )
            if type(values) is not tuple or len(values) != 3:
                raise GuidanceContractError(f"{owned.descriptor.id} must return (q, k, v)")
            q, k, v = (
                self._tensor(value, reference, owned.descriptor.id)
                for value, reference in zip(values, (q, k, v), strict=True)
            )
        wrappers = selected(self.registry.wrappers)

        def invoke(index: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            if index == len(wrappers):
                backend = self.registry.backends.get(context.family)
                if backend is None:
                    return kernel(q, k, v)
                return self._tensor(
                    self._call(
                        backend, backend.descriptor.kernel, q, k, v, self._context(context, backend)
                    ),
                    q,
                    backend.descriptor.id,
                )
            owned = wrappers[index]
            calls = 0
            opened = True

            def next_(
                next_q: torch.Tensor, next_k: torch.Tensor, next_v: torch.Tensor
            ) -> torch.Tensor:
                nonlocal calls
                if not opened:
                    raise GuidanceContractError(
                        f"{owned.descriptor.id} used next outside its callback"
                    )
                calls += 1
                if calls != 1:
                    raise GuidanceContractError(f"{owned.descriptor.id} violated next contract")
                return invoke(
                    index + 1,
                    self._tensor(next_q, q, owned.descriptor.id),
                    self._tensor(next_k, k, owned.descriptor.id),
                    self._tensor(next_v, v, owned.descriptor.id),
                )

            try:
                result = self._call(
                    owned, owned.descriptor.wrapper, q, k, v, self._context(context, owned), next_
                )
            finally:
                opened = False
            if calls == 0 and not owned.descriptor.terminal:
                raise GuidanceContractError(f"{owned.descriptor.id} violated next contract")
            return self._tensor(result, q, owned.descriptor.id)

        output = invoke(0, q, k, v)
        for owned in selected(self.registry.outputs):
            output = self._tensor(
                self._call(
                    owned, owned.descriptor.transform, output, self._context(context, owned)
                ),
                output,
                owned.descriptor.id,
            )
        return output

    def block(self, value: torch.Tensor, context: AttentionCallContext, phase: str) -> torch.Tensor:
        for owned in self.registry.blocks:
            descriptor = owned.descriptor
            if descriptor.phase == phase and descriptor.selector.matches(
                context.family, context.block, context.kind
            ):
                value = self._tensor(
                    self._call(owned, descriptor.transform, value, self._context(context, owned)),
                    value,
                    descriptor.id,
                )
        return value
