"""Test pack for combo choice-list tests. Imported by the worker HOST
process (via a manifest entry) AND by the test process directly for the
load_choices unit tests - keep it dependency-light and side-effect-free
at import (hazard H5)."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from dinkster_protocol import CompatGateDiagnostic
from dinkster_schema import (
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_values import TypeRegistry

STRING = TypeExpr.concrete("core.string")


def register_types(registry: TypeRegistry) -> None:
    pass


class Pick(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cp.pick",
            display_name="Choice Pick",
            inputs=(
                InputSpec(
                    "value",
                    TypeExpr.concrete("core.combo"),
                    widget=ComboWidget(remote_route="/api/choices/cp.samplers"),
                ),
            ),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(out=value)


NODES = [Pick]


class Pick2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cp2.pick",
            display_name="Choice Pick 2",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(out=value)


NODES2 = [Pick2]


class ReservedPick(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cp.reserved-pick",
            inputs=(
                InputSpec(
                    "value",
                    TypeExpr.concrete("core.combo"),
                    widget=ComboWidget(remote_route="/api/choices/comfy.samplers"),
                ),
            ),
        )


RESERVED_NODES = [ReservedPick]


class MissingRemoteChoice(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cp.missing-remote",
            inputs=(
                InputSpec(
                    "value",
                    TypeExpr.concrete("core.combo"),
                    widget=ComboWidget(remote_route="/api/choices/cp.missing"),
                ),
            ),
        )


MISSING_REMOTE_NODES = [MissingRemoteChoice]


class NestedMissingRemoteChoice(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cp.nested-missing-remote",
            combos=(
                DynamicComboSpec(
                    "mode",
                    options=(
                        DynamicComboOption(
                            "active",
                            inputs=(
                                InputSpec(
                                    "value",
                                    TypeExpr.concrete("core.combo"),
                                    widget=ComboWidget(
                                        remote_route="/api/choices/cp.nested-missing"
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )


NESTED_MISSING_REMOTE_NODES = [NestedMissingRemoteChoice]


def choices_reserved() -> Mapping[str, Sequence[str]]:
    """A choice id under the shared reserved root - two trusted packs may
    both claim "comfy", which is exactly where a cross-pack choice-id
    collision becomes possible."""
    return {"comfy.samplers": ("euler",)}


def choices_reserved_lazy() -> Mapping[str, object]:
    """The same reserved id as choices_reserved but declared lazy - the
    subject for cross-pack lazy-vs-static and lazy-vs-lazy collisions."""
    return {"comfy.samplers": lambda: ("euler",)}


class LazyPick(Node):
    """A node whose remote combo route targets a LAZY choice id - remote
    authority must accept lazy ids exactly like static ones."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cp.lazy-pick",
            inputs=(
                InputSpec(
                    "value",
                    TypeExpr.concrete("core.combo"),
                    widget=ComboWidget(remote_route="/api/choices/cp.devices", refresh_button=True),
                ),
            ),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(out=value)


LAZY_NODES = [Pick, LazyPick]


def _bump_counter() -> int:
    """Increment the invocation counter file named by CHOICEPACK_LAZY_COUNTER
    and return the new count. Providers run inside the worker process, so a
    file is the observable cross-process record of invocations."""
    path = Path(os.environ["CHOICEPACK_LAZY_COUNTER"])
    count = (int(path.read_text()) if path.exists() else 0) + 1
    path.write_text(str(count))
    return count


def lazy_probe_choices() -> Mapping[str, object]:
    """One table exercising every lazy outcome the server can surface, next
    to a static list. Only cp.devices bumps the counter, so tests can prove
    invocation counts: zero at load/hello (providers never run at startup)
    and exactly one per fetch, with count-dependent values proving no cache."""

    def devices() -> Sequence[str]:
        return (f"dev-{_bump_counter()}", "common")

    def boom() -> Sequence[str]:
        raise RuntimeError("device scan failed")

    return {
        "cp.samplers": ("euler", "ddim"),
        "cp.devices": devices,
        "cp.empty-lazy": lambda: (),
        "cp.boom": boom,
        "cp.bad": lambda: ("ok", 3),
    }


def lazy_slow_choices() -> Mapping[str, object]:
    """A provider that hangs on its first invocation only: fetch one times
    out under a shortened server budget, fetch two returns promptly - proving
    the late first reply is discarded and the channel stays healthy."""

    def slow() -> Sequence[str]:
        count = _bump_counter()
        if count == 1:
            time.sleep(2.0)
        return (f"slow-{count}",)

    return {"cp.slow": slow}


def lazy_bad_id_choices() -> Mapping[str, object]:
    """A lazy provider under a grammar-invalid id - the host must refuse at
    load, identically to static ids."""
    return {"Bad Id": lambda: ("euler",)}


def combo_choices() -> Mapping[str, Sequence[str]]:
    """The happy path: two lists, one with a duplicate (dedupe preserves
    first-seen order) and one empty (a registered but empty source)."""
    return {
        "cp.samplers": ("euler", "ddim", "euler", "heun"),
        "cp.empty": (),
    }


def _diagnostic(source_node: str, reason: str) -> CompatGateDiagnostic:
    return CompatGateDiagnostic(
        code="compat.dynamic.unsupported",
        source_node=source_node,
        reason=reason,
        source_generation="v1",
        path_kind="declared",
        input_id="value",
        input_path=("value",),
        lazy=False,
        input_is_list=False,
        output_is_list=False,
        raw_link=False,
        accept_all=False,
    )


def translation_skips() -> Mapping[str, CompatGateDiagnostic]:
    return {"OpaqueNode": _diagnostic("OpaqueNode", "unsupported dynamic marker")}


def legacy_translation_skips() -> Mapping[str, CompatGateDiagnostic]:
    source = "comfy.legacy.OpaqueNode"
    return {source: _diagnostic(source, "legacy dynamic marker")}


def choices_outside_namespace() -> Mapping[str, Sequence[str]]:
    """A choice id outside the pack's claims - composition must refuse."""
    return {"other.samplers": ("euler",)}


# -- host-validation subjects --------------------------------------------------

NOT_CALLABLE = "not a function"


def choices_not_mapping() -> object:
    return [("cp.samplers", ("euler",))]


def choices_bad_id() -> Mapping[str, Sequence[str]]:
    return {"Bad Id": ("euler",)}


def choices_bare_string() -> Mapping[str, object]:
    # A bare str is a Sequence[str]; the host must refuse it anyway.
    return {"cp.samplers": "euler"}


def choices_non_string_value() -> Mapping[str, object]:
    return {"cp.samplers": ("euler", 3)}


def choices_empty_value() -> Mapping[str, Sequence[str]]:
    return {"cp.samplers": ("euler", "")}


def choices_nul_value() -> Mapping[str, Sequence[str]]:
    return {"cp.samplers": ("nul\0value",)}


def choices_surrogate_value() -> Mapping[str, Sequence[str]]:
    return {"cp.samplers": ("\ud800",)}


def choices_long_value() -> Mapping[str, Sequence[str]]:
    return {"cp.samplers": ("x" * 4097,)}


def choices_too_many() -> Mapping[str, Sequence[str]]:
    return {"cp.samplers": tuple(f"v{index}" for index in range(10_001))}


def choices_response_too_large() -> Mapping[str, Sequence[str]]:
    values = tuple(f"{index:04d}" + "x" * 4092 for index in range(511))
    return {"cp.samplers": (*values, "tail" + "x" * 2556)}


def skips_not_mapping() -> object:
    return [("OpaqueNode", "reason")]


def skips_bad_name() -> Mapping[object, str]:
    return {3: "reason"}


def skips_bad_reason() -> Mapping[str, object]:
    return {"OpaqueNode": 3}
