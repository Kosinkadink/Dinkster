"""Post-processing nodes keep their source schemas and CPU behavior."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.post_processing import (
    POST_PROCESSING_NODES,
    BatchImagesNode,
    BatchLatentsNode,
    BatchMasksNode,
    ColorTransfer,
    ResizeImageMaskNode,
)


def _wire(node: type[Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", schema_to_wire(node.schema()))


def test_post_processing_schemas_use_exact_core_ids_and_dynamic_interfaces() -> None:
    schemas = {node.schema().node_type: _wire(node) for node in POST_PROCESSING_NODES}
    assert tuple(schemas) == (
        "ResizeImageMaskNode",
        "BatchImagesNode",
        "BatchMasksNode",
        "BatchLatentsNode",
        "ColorTransfer",
    )
    resize = schemas["ResizeImageMaskNode"]
    assert [item["id"] for item in resize["interface"]] == [
        "input",
        "resize_type",
        "scale_method",
        "resized",
    ]
    assert resize["interface"][0]["type"] == {
        "kind": "variable",
        "templateId": "input_type",
        "allowed": ["IMAGE", "MASK"],
    }
    assert [option["key"] for option in resize["interface"][1]["options"]] == [
        "scale dimensions",
        "scale by multiplier",
        "scale longer dimension",
        "scale shorter dimension",
        "scale width",
        "scale height",
        "scale total pixels",
        "match size",
        "scale to multiple",
    ]
    for node_type, family, prefix, value_type in (
        ("BatchImagesNode", "images", "image", "IMAGE"),
        ("BatchMasksNode", "masks", "mask", "MASK"),
        ("BatchLatentsNode", "latents", "latent", "LATENT"),
    ):
        interface = schemas[node_type]["interface"]
        assert interface[0]["id"] == family
        assert interface[0]["memberPrefix"] == prefix
        assert interface[0]["minMembers"] == 1 and interface[0]["maxMembers"] == 50
        assert interface[0]["template"][0]["type"]["types"] == [value_type]
        assert interface[1]["type"]["types"] == [value_type]
    color = schemas["ColorTransfer"]
    assert [item["id"] for item in color["interface"]] == [
        "image_target",
        "image_ref",
        "method",
        "source_stats",
        "strength",
        "image",
    ]
    assert [option["key"] for option in color["interface"][3]["options"]] == [
        "per_frame",
        "uniform",
        "target_frame",
    ]


@pytest.fixture
def tensor_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = importlib.import_module("torch")

    def common_upscale(value: Any, width: int, height: int, method: str, crop: str) -> Any:
        kwargs: dict[str, object] = {
            "size": (height, width),
            "mode": "nearest" if method == "nearest-exact" else method,
        }
        if kwargs["mode"] in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        if kwargs["mode"] == "lanczos":
            kwargs["mode"] = "bicubic"
            kwargs["align_corners"] = False
        return torch.nn.functional.interpolate(value, **kwargs)

    monkeypatch.setitem(
        sys.modules, "dinkster_inference.utils", SimpleNamespace(common_upscale=common_upscale)
    )
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.model_management",
        SimpleNamespace(
            get_torch_device=lambda: torch.device("cpu"),
            intermediate_device=lambda: torch.device("cpu"),
            intermediate_dtype=lambda: torch.float32,
        ),
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="requires CPU inference environment"
)
def test_resize_preserves_image_and_mask_shapes_across_asymmetric_boundaries(
    tensor_backend: None,
) -> None:
    torch = importlib.import_module("torch")

    image = torch.arange(1 * 3 * 5 * 3, dtype=torch.float32).reshape(1, 3, 5, 3) / 45
    mask = torch.arange(1 * 3 * 5, dtype=torch.float32).reshape(1, 3, 5) / 15
    assert cast(
        "Any",
        ResizeImageMaskNode.execute(
            input=image,
            scale_method="nearest-exact",
            resize_type={"resize_type": "scale width", "width": 4},
        )["resized"],
    ).shape == (1, 2, 4, 3)
    assert cast(
        "Any",
        ResizeImageMaskNode.execute(
            input=mask,
            scale_method="nearest-exact",
            resize_type={"resize_type": "scale shorter dimension", "shorter_size": 4},
        )["resized"],
    ).shape == (1, 4, 7)
    multiple_input = torch.arange(1 * 5 * 6 * 3, dtype=torch.float32).reshape(1, 5, 6, 3) / 90
    result = cast(
        "Any",
        ResizeImageMaskNode.execute(
            input=multiple_input,
            scale_method="nearest-exact",
            resize_type={"resize_type": "scale to multiple", "multiple": 4},
        )["resized"],
    )
    assert result.shape == (1, 4, 4, 3)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="requires CPU inference environment"
)
def test_batching_reconciles_asymmetric_images_masks_and_latent_metadata(
    tensor_backend: None,
) -> None:
    torch = importlib.import_module("torch")

    image = torch.zeros((1, 2, 3, 3))
    rgba = torch.ones((2, 4, 5, 4))
    images = cast("Any", BatchImagesNode.execute(images={"image0": image, "image1": rgba})["IMAGE"])
    assert images.shape == (3, 2, 3, 4)
    assert torch.equal(images[0, ..., 3], torch.ones((2, 3)))
    masks = cast(
        "Any",
        BatchMasksNode.execute(
            masks={"mask0": torch.zeros((1, 2, 3)), "mask1": torch.ones((2, 4, 5))}
        )["MASK"],
    )
    assert masks.shape == (3, 2, 3)
    latents = cast(
        "dict[str, Any]",
        BatchLatentsNode.execute(
            latents={
                "latent0": {"samples": torch.ones((1, 4, 2, 3)), "batch_index": [7]},
                "latent1": {"samples": torch.zeros((2, 4, 4, 5))},
            }
        )["LATENT"],
    )
    assert latents["samples"].shape == (3, 4, 2, 3)
    assert latents["batch_index"] == [7, 0, 1]


def test_color_transfer_identity_strength_does_not_materialize_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = object()
    monkeypatch.setattr(
        "dinkster_nodes_comfy.post_processing.materialize_image", lambda value: value
    )
    assert (
        ColorTransfer.execute(
            image_target=target,
            image_ref=None,
            method="histogram",
            source_stats={"source_stats": "per_frame"},
            strength=0.0,
        )["image"]
        is target
    )
