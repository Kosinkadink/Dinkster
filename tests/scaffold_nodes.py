"""The engine-test node set: default first-party nodes plus dev scaffolding.

Engine/worker/server tests exercise machinery (values, renditions, lists,
scheduling, saves) through the dev pack's nodes; user-facing composition
tests use the component pack node sets to pin what a real user surface contains.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence

from dinkster_api.v1 import TypeRegistry
from dinkster_audio import NODES as AUDIO_NODES
from dinkster_audio import register_types as register_audio_types
from dinkster_nodes_dev import PACK_NODES, register_dev_types
from dinkster_nodes_dev import combo_choices as dev_combo_choices
from dinkster_image import NODES as IMAGE_NODES
from dinkster_image import register_types as register_image_types
from dinkster_utilities import NODES as UTILITY_NODES
from dinkster_utilities import register_types as register_utility_types
from dinkster_video import NODES as VIDEO_NODES
from dinkster_video import register_types as register_video_types

SCAFFOLD_NODES = [*UTILITY_NODES, *AUDIO_NODES, *VIDEO_NODES, *IMAGE_NODES, *PACK_NODES]


def scaffold_choices() -> Mapping[str, Sequence[str]]:
    """Return every static remote choice owned by the scaffold packs."""
    return dev_combo_choices()


def scaffold_lazy_choices() -> dict[str, Callable[[], Awaitable[Sequence[str]]]]:
    """First-party scaffold nodes do not publish capture-device choice routes."""
    return {}


def register_scaffold_types(registry: TypeRegistry) -> None:
    """Register every non-core value type used by the scaffold set."""
    register_utility_types(registry)
    register_audio_types(registry)
    register_video_types(registry)
    register_image_types(registry)
    register_dev_types(registry)
