"""Development nodes whose rows come from lookups the host calls.

The LoRA names are a fixed list so the examples need no models. ``delay`` and
``fail`` are ordinary arguments, so a slow or failing answer is part of what
the node asks for rather than hidden state.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_STRING,
    ComboWidget,
    Input,
    Lookup,
    Node,
    NodeSchema,
    Note,
    NumberWidget,
    Output,
    TypeExpr,
    lookup,
)

STRING = TypeExpr.concrete(CORE_STRING)
COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)

DEV_LORAS = (
    "a.safetensors",
    "sd/x.safetensors",
    "sd/detail.safetensors",
    "sdxl/y.safetensors",
    "sdxl/style.safetensors",
)


class LookupFailed(RuntimeError):
    """What a development lookup raises when asked to fail."""


def _wait_or_fail(delay: float, fail: bool) -> None:
    if delay > 0:
        time.sleep(min(delay, 30.0))
    if fail:
        raise LookupFailed("asked to fail")


@lookup
def lora_categories() -> list[str]:
    return ["All", *sorted({name.rpartition("/")[0] for name in DEV_LORAS} - {""})]


@lookup
def loras_in(category: str) -> list[str]:
    return [name for name in DEV_LORAS if category == "All" or name.startswith(category)]


@lookup
def lora_summary(category: str, delay: float, fail: bool) -> str:
    """A note's text: how many LoRAs the category holds."""
    _wait_or_fail(delay, fail)
    count = len(loras_in(category))
    return f"{count} LoRA{'s' if count != 1 else ''} in {category}"


@lookup
def quick_note() -> str:
    return "Answered at once."


_asked = itertools.count(1)


@lookup(watches=("media/image",), after_run=True)
def times_asked() -> str:
    """Asked again whenever the host's images change or the node runs."""
    count = next(_asked)
    return f"Asked {count} time{'s' if count != 1 else ''}."


@lookup
def empty_names(fail: bool) -> list[str]:
    _wait_or_fail(0.0, fail)
    return []


class LookupNoteExample(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.lookups.note",
            display_name="Lookup: Note",
            category="dev/lookups",
            description=(
                "A note whose text a lookup works out from the category. Delay makes the "
                "lookup slow; Fail makes it raise."
            ),
            output_node=True,
            body=(
                Input(
                    "category",
                    COMBO,
                    default="sd",
                    widget=ComboWidget(options=("All", "sd", "sdxl")),
                ),
                Input(
                    "delay", FLOAT, default=0.0, widget=NumberWidget(min=0.0, max=20.0, step=0.5)
                ),
                Input("fail", BOOLEAN, default=False),
                Note(
                    "summary",
                    Lookup(lora_summary, args=("category", "delay", "fail")),
                    unknown="Counting LoRAs...",
                ),
                Note("quick", Lookup(quick_note), unknown="Waiting for the pack..."),
                Note("asked", Lookup(times_asked), unknown="Not asked yet."),
                Output("category_out", STRING),
            ),
        )

    @classmethod
    def execute(cls, *, category: str, delay: float, fail: bool) -> Mapping[str, object]:
        del delay, fail
        return cls.outputs(category_out=category)


LOOKUP_NODES = [LookupNoteExample]
