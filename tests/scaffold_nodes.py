"""The engine-test node set: default first-party nodes plus dev scaffolding.

Engine/worker/server tests exercise machinery (values, renditions, lists,
scheduling, saves) through the dev pack's nodes; user-facing composition
tests use the component pack node sets to pin what a real user surface contains.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence

from dinkster_api.v1 import TypeRegistry
from dinkster_nodes_dev import PACK_NODES, register_dev_types
from dinkster_nodes_dev import combo_choices as dev_combo_choices
from dinkster_nodes_foundation import FOUNDATION_NODES, register_foundation_types
from dinkster_nodes_image import IMAGE_NODES, image_choices, register_image_types
from dinkster_nodes_media_io import (
    AUDIO_DEVICE_CHOICES_ID,
    MEDIA_IO_NODES,
    VIDEO_DEVICE_CHOICES_ID,
    register_media_types,
)

SCAFFOLD_NODES = [*FOUNDATION_NODES, *MEDIA_IO_NODES, *IMAGE_NODES, *PACK_NODES]


def scaffold_choices() -> Mapping[str, Sequence[str]]:
    """Return every static remote choice owned by the scaffold packs."""
    return {**dev_combo_choices(), **image_choices()}


def scaffold_lazy_choices() -> dict[str, Callable[[], Awaitable[Sequence[str]]]]:
    """The lazy routes the scaffold schemas reference (the capture device
    dropdowns), served empty - the same answer a fetch gets when no capture
    provider is configured. Hand-built servers over SCAFFOLD_NODES need
    these registered because every published remote route must have an
    owner; real device enumeration is composition-tested elsewhere."""

    async def empty() -> Sequence[str]:
        return ()

    return {AUDIO_DEVICE_CHOICES_ID: empty, VIDEO_DEVICE_CHOICES_ID: empty}


def register_scaffold_types(registry: TypeRegistry) -> None:
    """Register every non-core value type used by the scaffold set."""
    register_foundation_types(registry)
    register_media_types(registry)
    register_image_types(registry)
    register_dev_types(registry)
