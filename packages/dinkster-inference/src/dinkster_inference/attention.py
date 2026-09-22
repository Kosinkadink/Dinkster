"""Torch-free authoring contract for attention and block-level extensions.

Packs declare runtime-adapter attention backends, Q/K/V transforms, attention
wrappers, output transforms, and block-level residual/state injections
through :class:`AttentionContribution`. Declarations are projected to
canonical keyed surfaces at composition and executed inside the inference
worker by the torch runtime; this module never imports torch - tensors
appear only through the ``SizedTensor`` structural contract.

Selectors match whole ``*`` wildcards or exact names, never patterns, and
resolve against stable block names (``middle_block.1.transformer_blocks.0``)
rather than module paths, so nothing here reaches model internals by
reflection.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast

from dinkster_protocol import (
    ATTENTION_BACKEND_SURFACE,
    ATTENTION_OUTPUT_SURFACE,
    ATTENTION_QKV_SURFACE,
    ATTENTION_WRAPPER_SURFACE,
    BLOCK_INJECTION_SURFACE,
    BehaviorValue,
    KeyedContribution,
)

from .guidance import validate_descriptor, validate_order
from .patches import SizedTensor
from .sampling import SamplingExecutionContext

T = TypeVar("T", bound=SizedTensor)

ATTENTION_AXES = ("query", "key")
"""The two token axes a span can name."""

ATTENTION_STREAMS = ("text", "image", "reference")
"""Closed vocabulary for the conditioning stream a span belongs to."""

ATTENTION_KINDS = ("self", "cross", "joint")
"""Concrete attention kinds; selectors additionally accept the ``*`` wildcard."""

TORCH_DISTRIBUTION = "torch"
AIMDO_DISTRIBUTION = "dinkster-aimdo"


@dataclass(frozen=True)
class AttentionSelector:
    """Where one attention extension point applies.

    ``family`` is the runtime attention adapter key (matched exactly, never
    a pattern), such as ``unet`` or ``flux``. ``block`` is a stable block
    name or the whole ``*`` wildcard; ``kind`` is ``*`` or one of
    :data:`ATTENTION_KINDS`. Only a standalone ``*`` is a wildcard - a name
    containing it elsewhere matches nothing.
    """

    family: str
    block: str = "*"
    kind: str = "*"

    def __post_init__(self) -> None:
        if type(self.family) is not str or not self.family.strip():
            raise ValueError("attention selector family must be a nonempty exact name")
        if not self._selector_word(self.block):
            raise ValueError("attention selector block must be '*' or a nonempty stable block name")
        if self.kind not in ("*", *ATTENTION_KINDS):
            raise ValueError(
                "attention selector kind must be '*' or one of: " + ", ".join(ATTENTION_KINDS)
            )

    @staticmethod
    def _selector_word(value: str) -> bool:
        return value == "*" or (type(value) is str and value.strip() != "")

    def matches(self, family: str, block: str, kind: str) -> bool:
        """Whether this selector covers one concrete attention site.

        Only a whole ``*`` wildcards; every other value is an exact
        comparison, so a block name that merely contains ``*`` never
        matches.
        """
        if self.family != family:
            return False
        if self.block != "*" and self.block != block:
            return False
        return self.kind == "*" or self.kind == kind


@dataclass(frozen=True)
class AttentionTokenSpan:
    """One half-open token range with its conditioning identity.

    ``start``/``end`` index tokens along ``axis``; ``batch_start``/
    ``batch_end`` index the batch. All ranges are half-open, non-negative,
    and nonempty. ``condition_id`` names the conditioning entry the span
    was resolved from and ``role`` its role in the fused call.
    """

    axis: str
    start: int
    end: int
    condition_id: str
    role: str
    batch_start: int
    batch_end: int
    stream: str

    def __post_init__(self) -> None:
        if type(self.condition_id) is not str or not self.condition_id:
            raise ValueError("span condition_id must be a nonempty string")
        if type(self.role) is not str or not self.role:
            raise ValueError("span role must be a nonempty string")
        if self.axis not in ATTENTION_AXES:
            raise ValueError("span axis must be 'query' or 'key'")
        for name, value in (("start", self.start), ("end", self.end)):
            if type(value) is not int:
                raise TypeError(f"span {name} must be an int")
        for name, value in (("batch_start", self.batch_start), ("batch_end", self.batch_end)):
            if type(value) is not int:
                raise TypeError(f"span {name} must be an int")
        if not 0 <= self.start < self.end:
            raise ValueError("span token range must be half-open, non-negative, and nonempty")
        if not 0 <= self.batch_start < self.batch_end:
            raise ValueError("span batch range must be half-open, non-negative, and nonempty")
        if not self.condition_id:
            raise ValueError("span condition_id must be nonempty")
        if not self.role:
            raise ValueError("span role must be nonempty")
        if self.stream not in ATTENTION_STREAMS:
            raise ValueError("span stream must be one of: " + ", ".join(ATTENTION_STREAMS))


@dataclass(frozen=True)
class AttentionCallContext:
    """Invocation-local facts for one attention site evaluation.

    ``state`` is the owning extension's invocation scratch namespace from
    ``SamplingExecutionContext.extension_state``; there is no global
    mutable state anywhere in this contract.
    """

    family: str
    block: str
    kind: str
    heads: int
    spatial_shape: tuple[int, int]
    spans: tuple[AttentionTokenSpan, ...]
    execution: SamplingExecutionContext
    state: MutableMapping[str, object]

    def __post_init__(self) -> None:
        if type(self.family) is not str or not self.family:
            raise ValueError("attention context family must be a nonempty string")
        if type(self.block) is not str or not self.block:
            raise ValueError("attention context block must be a nonempty string")
        if self.kind not in ATTENTION_KINDS:
            raise ValueError("attention context kind must be one of: " + ", ".join(ATTENTION_KINDS))
        if type(self.heads) is not int or self.heads < 1:
            raise ValueError("attention context heads must be a positive int")
        shape = cast("object", self.spatial_shape)
        if not isinstance(shape, tuple):
            raise ValueError("attention context spatial_shape must be two positive ints")
        dims = cast("tuple[object, ...]", shape)
        if len(dims) != 2 or any(type(dim) is not int or dim < 1 for dim in dims):
            raise ValueError("attention context spatial_shape must be two positive ints")
        spans = cast("object", self.spans)
        if not isinstance(spans, tuple) or not all(
            isinstance(span, AttentionTokenSpan) for span in cast("tuple[object, ...]", spans)
        ):
            raise TypeError("spans must contain AttentionTokenSpan values")
        execution = cast("object", self.execution)
        if not isinstance(execution, SamplingExecutionContext):
            raise TypeError("execution must be a SamplingExecutionContext")
        state = cast("object", self.state)
        if not isinstance(state, MutableMapping):
            raise TypeError("state must be a MutableMapping")


class AttentionQKVTransform(Protocol[T]):
    def __call__(self, q: T, k: T, v: T, context: AttentionCallContext) -> tuple[T, T, T]: ...


class AttentionWrapperNext(Protocol[T]):
    def __call__(self, q: T, k: T, v: T) -> T: ...


class AttentionWrapperFn(Protocol[T]):
    def __call__(
        self, q: T, k: T, v: T, context: AttentionCallContext, next: AttentionWrapperNext[T]
    ) -> T: ...


class AttentionOutputTransform(Protocol[T]):
    def __call__(self, output: T, context: AttentionCallContext) -> T: ...


class AttentionKernelFn(Protocol[T]):
    def __call__(self, q: T, k: T, v: T, context: AttentionCallContext) -> T: ...


class BlockInjectionTransform(Protocol[T]):
    def __call__(self, value: T, context: AttentionCallContext) -> T: ...


BLOCK_INJECTION_PHASES = ("before", "after")
"""Where a block injection applies relative to the block's residual path."""


def _validate_selector(selector: object) -> None:
    if not isinstance(selector, AttentionSelector):
        raise TypeError("selector must be an AttentionSelector")


@dataclass(frozen=True)
class AttentionQKVDescriptor(Generic[T]):
    """Rewrite the query/key/value tensors of one selected attention site."""

    id: str
    selector: AttentionSelector
    transform: AttentionQKVTransform[T]
    order: int = 0
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        validate_descriptor(self.id, self.transform, self.behavior_metadata, label="attention")
        validate_order(self.order)
        _validate_selector(self.selector)


@dataclass(frozen=True)
class AttentionWrapperDescriptor(Generic[T]):
    """Wrap one selected attention site around the kernel call.

    ``wrapper`` receives the site's q/k/v, the call context, and ``next``,
    which invokes the rest of the chain. ``terminal`` grants permission to
    stop the chain: a terminal wrapper may call ``next`` zero or one time
    (so it can delegate baseline calls and only rewrite its auxiliary
    ones), while a non-terminal wrapper must call ``next`` exactly once.
    Both reject more than one call - the runtime enforces the counts and
    refuses the composition otherwise.
    """

    id: str
    selector: AttentionSelector
    wrapper: AttentionWrapperFn[T]
    order: int = 0
    terminal: bool = False
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        validate_descriptor(self.id, self.wrapper, self.behavior_metadata, label="attention")
        validate_order(self.order)
        _validate_selector(self.selector)
        if type(self.terminal) is not bool:
            raise TypeError("terminal must be bool")


@dataclass(frozen=True)
class AttentionOutputDescriptor(Generic[T]):
    """Rewrite the attention output of one selected attention site."""

    id: str
    selector: AttentionSelector
    transform: AttentionOutputTransform[T]
    order: int = 0
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        validate_descriptor(self.id, self.transform, self.behavior_metadata, label="attention")
        validate_order(self.order)
        _validate_selector(self.selector)


@dataclass(frozen=True)
class AttentionBackendDescriptor(Generic[T]):
    """Replace the attention kernel of one runtime attention adapter.

    ``family`` must be an exact adapter key - no wildcard. Each adapter
    admits exactly one backend; a second declaration for the same adapter
    refuses at composition naming both packs and descriptor ids.
    """

    id: str
    family: str
    kernel: AttentionKernelFn[T]
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        validate_descriptor(self.id, self.kernel, self.behavior_metadata, label="attention")
        if type(self.family) is not str or not self.family or self.family == "*":
            raise ValueError("attention backend family must be an exact nonempty family id")


@dataclass(frozen=True)
class BlockInjectionDescriptor(Generic[T]):
    """Inject state at a block boundary of one selected attention site."""

    id: str
    selector: AttentionSelector
    transform: BlockInjectionTransform[T]
    phase: str = "after"
    order: int = 0
    behavior_metadata: tuple[tuple[str, BehaviorValue], ...] = ()

    def __post_init__(self) -> None:
        validate_descriptor(self.id, self.transform, self.behavior_metadata, label="attention")
        validate_order(self.order)
        _validate_selector(self.selector)
        if self.phase not in BLOCK_INJECTION_PHASES:
            raise ValueError("phase must be 'before' or 'after'")


@dataclass(frozen=True)
class AttentionContribution(Generic[T]):
    """Additive attention and block-level result of one pack's entry point.

    ``torch_version`` and ``aimdo_version`` are required keyword-only exact
    pins. Before any execution, the inference worker checks both against
    the distributions installed in its own environment via
    ``importlib.metadata`` (torch is never imported here) and refuses a
    mismatch with reinstall/recreate-environment guidance.
    """

    qkv: tuple[AttentionQKVDescriptor[T], ...] = ()
    wrappers: tuple[AttentionWrapperDescriptor[T], ...] = ()
    outputs: tuple[AttentionOutputDescriptor[T], ...] = ()
    backends: tuple[AttentionBackendDescriptor[T], ...] = ()
    blocks: tuple[BlockInjectionDescriptor[T], ...] = ()
    torch_version: str = field(kw_only=True)
    aimdo_version: str = field(kw_only=True)

    def __post_init__(self) -> None:
        collections = (
            ("qkv", self.qkv, AttentionQKVDescriptor),
            ("wrappers", self.wrappers, AttentionWrapperDescriptor),
            ("outputs", self.outputs, AttentionOutputDescriptor),
            ("backends", self.backends, AttentionBackendDescriptor),
            ("blocks", self.blocks, BlockInjectionDescriptor),
        )
        ids: list[str] = []
        for name, values, descriptor_type in collections:
            raw_values = cast("object", values)
            if not isinstance(raw_values, tuple) or not all(
                isinstance(value, descriptor_type)
                for value in cast("tuple[object, ...]", raw_values)
            ):
                raise TypeError(f"{name} must be a tuple of {descriptor_type.__name__} values")
            ids.extend(item.id for item in values)
        if not ids:
            raise ValueError("attention contribution must be nonempty")
        if len(ids) != len(set(ids)):
            raise ValueError("attention descriptor ids must be unique within a contribution")
        for name, pin in (
            ("torch_version", self.torch_version),
            ("aimdo_version", self.aimdo_version),
        ):
            if type(pin) is not str or not pin.strip():
                raise ValueError(f"{name} must be an explicit nonempty version string")


class AttentionPinError(RuntimeError):
    """A declared attention pin does not match the inference worker."""


def _installed_version(distribution: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return ""


def check_attention_pins(extension_id: str, contribution: AttentionContribution[Any]) -> None:
    """Refuse unless the declared pins match this interpreter's distributions.

    Runs inside the inference worker before execution, reading installed
    distribution metadata only - torch is never imported here.
    """
    for distribution, declared in (
        (TORCH_DISTRIBUTION, contribution.torch_version),
        (AIMDO_DISTRIBUTION, contribution.aimdo_version),
    ):
        installed = _installed_version(distribution)
        if installed == declared:
            continue
        actual = installed or "not installed"
        raise AttentionPinError(
            f"extension {extension_id!r} attention contribution declares exact "
            f"{distribution} pin {declared!r} but the inference worker environment has "
            f"{actual!r}; reinstall the pack with a pin matching the execution "
            "environment, or recreate the execution environment to match the pack's "
            "declared pins"
        )


def attention_declaration_metadata(
    contribution: AttentionContribution[Any],
    descriptor: (
        AttentionQKVDescriptor[T]
        | AttentionWrapperDescriptor[T]
        | AttentionOutputDescriptor[T]
        | AttentionBackendDescriptor[T]
        | BlockInjectionDescriptor[T]
    ),
) -> tuple[tuple[str, BehaviorValue], ...]:
    """The RPC-clean identity facts of one attention descriptor.

    Binds the selector, ordering, pins, and the descriptor's own
    ``config.*`` metadata so behavior identity covers the declared
    attention behavior.
    """
    selector = getattr(descriptor, "selector", None)
    pairs: list[tuple[str, BehaviorValue]] = [
        ("aimdoVersion", contribution.aimdo_version),
        ("contractVersion", 1),
        ("torchVersion", contribution.torch_version),
    ]
    if selector is not None:
        pairs += [
            ("selectorBlock", selector.block),
            ("selectorFamily", selector.family),
            ("selectorKind", selector.kind),
        ]
    if isinstance(descriptor, AttentionBackendDescriptor):
        pairs.append(("family", descriptor.family))
    elif isinstance(descriptor, AttentionWrapperDescriptor):
        pairs += [("order", descriptor.order), ("terminal", descriptor.terminal)]
    elif isinstance(descriptor, BlockInjectionDescriptor):
        pairs += [("order", descriptor.order), ("phase", descriptor.phase)]
    else:
        pairs.append(("order", descriptor.order))
    pairs.extend(descriptor.behavior_metadata)
    return tuple(sorted(pairs))


def attention_declarations(
    contribution: AttentionContribution[Any],
) -> tuple[KeyedContribution, ...]:
    """Project worker-local callbacks to canonical RPC-clean declarations."""
    grouped = (
        (ATTENTION_QKV_SURFACE, contribution.qkv),
        (ATTENTION_WRAPPER_SURFACE, contribution.wrappers),
        (ATTENTION_OUTPUT_SURFACE, contribution.outputs),
        (ATTENTION_BACKEND_SURFACE, contribution.backends),
        (BLOCK_INJECTION_SURFACE, contribution.blocks),
    )
    declarations: list[KeyedContribution] = []
    for surface, descriptors in grouped:
        for descriptor in sorted(
            descriptors,
            key=lambda item: (getattr(item, "order", 0), item.id),
        ):
            declarations.append(
                KeyedContribution(
                    surface,
                    descriptor.id,
                    behavior_metadata=attention_declaration_metadata(contribution, descriptor),
                )
            )
    surface_order = {
        surface: index
        for index, surface in enumerate(
            (
                ATTENTION_QKV_SURFACE,
                ATTENTION_WRAPPER_SURFACE,
                ATTENTION_OUTPUT_SURFACE,
                ATTENTION_BACKEND_SURFACE,
                BLOCK_INJECTION_SURFACE,
            )
        )
    }
    return tuple(
        sorted(
            declarations,
            key=lambda item: (
                surface_order[item.surface_id],
                dict(item.behavior_metadata).get("order", 0),
                item.id,
            ),
        )
    )


__all__ = [
    "AIMDO_DISTRIBUTION",
    "ATTENTION_AXES",
    "ATTENTION_KINDS",
    "ATTENTION_STREAMS",
    "ATTENTION_BACKEND_SURFACE",
    "ATTENTION_QKV_SURFACE",
    "ATTENTION_WRAPPER_SURFACE",
    "ATTENTION_OUTPUT_SURFACE",
    "BLOCK_INJECTION_SURFACE",
    "BLOCK_INJECTION_PHASES",
    "TORCH_DISTRIBUTION",
    "AttentionBackendDescriptor",
    "AttentionCallContext",
    "AttentionContribution",
    "AttentionKernelFn",
    "AttentionOutputDescriptor",
    "AttentionOutputTransform",
    "AttentionPinError",
    "AttentionQKVDescriptor",
    "AttentionQKVTransform",
    "AttentionSelector",
    "AttentionTokenSpan",
    "AttentionWrapperDescriptor",
    "AttentionWrapperFn",
    "AttentionWrapperNext",
    "BlockInjectionDescriptor",
    "BlockInjectionTransform",
    "attention_declarations",
    "check_attention_pins",
]
