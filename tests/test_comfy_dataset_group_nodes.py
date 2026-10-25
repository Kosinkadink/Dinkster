"""Dataset list contracts remain visible without loading Torch."""

from typing import Any, cast

from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.dataset_groups import MergeImageLists, ResolutionBucket


def test_dataset_group_interfaces_preserve_list_elements() -> None:
    for node, names, elements in (
        (MergeImageLists, ("images", "images"), ("IMAGE", "IMAGE")),
        (
            ResolutionBucket,
            ("latents", "conditioning", "latents", "conditioning"),
            ("LATENT", "CONDITIONING", "LATENT", "CONDITIONING"),
        ),
    ):
        interface = cast(list[dict[str, Any]], schema_to_wire(node.schema())["interface"])
        assert tuple(item["id"] for item in interface) == names
        for item, element in zip(interface, elements, strict=True):
            assert item["type"] == {
                "kind": "list",
                "element": {"kind": "concrete", "types": [element]},
            }
