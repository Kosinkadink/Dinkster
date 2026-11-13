"""Engine-test nodes and value types from the nonpublishing development pack."""

from collections.abc import Awaitable, Callable, Mapping, Sequence

from dinkster_api.v1 import TypeRegistry
from dinkster_nodes_dev import PACK_NODES, register_dev_types
from dinkster_nodes_dev import combo_choices as dev_combo_choices

SCAFFOLD_NODES = [*PACK_NODES]


def scaffold_choices() -> Mapping[str, Sequence[str]]:
    """Return every static remote choice owned by the scaffold packs."""
    return dev_combo_choices()


def scaffold_lazy_choices() -> dict[str, Callable[[], Awaitable[Sequence[str]]]]:
    """First-party scaffold nodes do not publish capture-device choice routes."""
    return {}


def register_scaffold_types(registry: TypeRegistry) -> None:
    """Register every non-core value type used by the scaffold set."""
    register_dev_types(registry)
