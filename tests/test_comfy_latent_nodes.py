"""Core latent node interfaces and metadata behavior stay exact."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.latent import (
    LATENT_NODES,
    LatentFromBatch,
    LatentUpscale,
    RepeatLatentBatch,
    SetLatentNoiseMask,
)
from dinkster_values import EncodedLatentTensor, decode_latent, encode_latent

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("node_id", "display", "category", "inputs"),
    [
        ("EmptyLatentImage", "Empty Latent Image", "", ("width", "height", "batch_size")),
        (
            "LatentFromBatch",
            "Get Latent From Batch",
            "/batch",
            ("samples", "batch_index", "length"),
        ),
        ("RepeatLatentBatch", "Repeat Latent Batch", "/batch", ("samples", "amount")),
        (
            "LatentUpscale",
            "Upscale Latent",
            "",
            ("samples", "upscale_method", "width", "height", "crop"),
        ),
        ("LatentUpscaleBy", "Upscale Latent By", "", ("samples", "upscale_method", "scale_by")),
        ("LatentRotate", "Rotate Latent", "/transform", ("samples", "rotation")),
        ("LatentFlip", "Flip Latent", "/transform", ("samples", "flip_method")),
        ("SetLatentNoiseMask", "Set Latent Noise Mask", "", ("samples", "mask")),
    ],
)
def test_latent_schemas_use_exact_core_interfaces(
    node_id: str, display: str, category: str, inputs: tuple[str, ...]
) -> None:
    node = next(node for node in LATENT_NODES if node.schema().node_type == node_id)
    schema = node.schema()
    assert schema.display_name == display
    assert schema.category == "model/latent" + category
    assert tuple(spec.id for spec in schema.inputs) == inputs
    assert tuple((spec.id, spec.type.types) for spec in schema.outputs) == (
        ("LATENT", ("LATENT",)),
    )


class ArrayTensor:
    """CPU array fixture for tensor-only slicing, cloning and repetition."""

    def __init__(self, values: object) -> None:
        self.array = np.asarray(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    def __getitem__(self, key: object) -> ArrayTensor:
        return ArrayTensor(self.array[cast(Any, key)])

    def clone(self) -> ArrayTensor:
        return ArrayTensor(self.array.copy())

    def repeat(self, *shape: Any) -> ArrayTensor:
        dimensions = shape[0] if len(shape) == 1 else shape
        return ArrayTensor(np.tile(self.array, dimensions))

    def reshape(self, shape: tuple[int, ...]) -> ArrayTensor:
        return ArrayTensor(self.array.reshape(shape))

    def detach(self) -> ArrayTensor:
        return self


@pytest.mark.parametrize(("index", "expected"), [(-2, [20, 30]), (-99, [10, 20]), (99, [30])])
def test_batch_selection_clamps_indices_tiles_masks_and_copies(
    index: int, expected: list[int]
) -> None:
    source = ArrayTensor(np.array([10, 20, 30]).reshape(3, 1, 1, 1))
    masks = ArrayTensor(np.array([2, 7]).reshape(2, 1, 1, 1))
    latent = {"samples": source, "noise_mask": masks, "batch_index": [11, 19, 23], "note": "keep"}
    result = cast(
        dict[str, Any],
        LatentFromBatch.execute(samples=latent, batch_index=index, length=2)["LATENT"],
    )
    assert result["samples"].array.flatten().tolist() == expected
    expected_indices = {10: 11, 20: 19, 30: 23}
    assert result["batch_index"] == [expected_indices[value] for value in expected]
    expected_masks = {10: 2, 20: 7, 30: 2}
    assert result["noise_mask"].array.flatten().tolist() == [
        expected_masks[value] for value in expected
    ]
    result["samples"].array.fill(-1)
    result["noise_mask"].array.fill(-1)
    assert source.array.flatten().tolist() == [10, 20, 30]
    assert masks.array.flatten().tolist() == [2, 7]
    assert result["note"] == "keep"


def test_repeat_offsets_nonconsecutive_indices_and_repeats_original_masks() -> None:
    source = ArrayTensor(np.array([10, 20, 30]).reshape(3, 1, 1, 1))
    masks = ArrayTensor(np.array([2, 7]).reshape(2, 1, 1, 1))
    latent = {"samples": source, "noise_mask": masks, "batch_index": [4, 9, 6]}
    result = cast(dict[str, Any], RepeatLatentBatch.execute(samples=latent, amount=3)["LATENT"])
    assert result["samples"].array.flatten().tolist() == [10, 20, 30] * 3
    assert result["noise_mask"].array.flatten().tolist() == [2, 7] * 3
    assert result["batch_index"] == [4, 9, 6, 10, 15, 12, 16, 21, 18]
    assert latent["batch_index"] == [4, 9, 6]


def test_mask_shape_is_batch_channel_height_width_without_changing_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(as_tensor=ArrayTensor))
    samples = object()
    mask = np.arange(24).reshape(2, 3, 4)
    latent = {"samples": samples, "note": "preserved"}
    result = cast(dict[str, Any], SetLatentNoiseMask.execute(samples=latent, mask=mask)["LATENT"])
    assert result["samples"] is samples
    assert result["note"] == "preserved"
    assert isinstance(result["noise_mask"], ArrayTensor)
    np.testing.assert_array_equal(result["noise_mask"].array, mask[:, None, :, :])
    assert "noise_mask" not in latent


def test_zero_size_upscale_returns_the_existing_latent_unchanged() -> None:
    latent = {"samples": object(), "note": "preserved"}
    assert (
        LatentUpscale.execute(
            samples=latent, upscale_method="bislerp", width=0, height=0, crop="center"
        )["LATENT"]
        is latent
    )


class PortableLatentSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="latent-fixture.source",
            outputs=(
                OutputSpec("samples", TypeExpr.concrete("LATENT")),
                OutputSpec("mask", TypeExpr.concrete("MASK")),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return {
            "samples": {
                "samples": EncodedLatentTensor(
                    "float32", (1, 2, 2, 3), np.arange(12, dtype=np.float32).tobytes()
                ),
                "note": "keep",
            },
            "mask": np.arange(6, dtype=np.float32).reshape(1, 2, 3) / 5,
        }


@pytest.mark.parametrize("in_process", (True, False))
def test_manifest_executes_latent_and_mask_without_host_types(
    tmp_path: Path, in_process: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(as_tensor=np.asarray))

    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = (
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
            )
            (tmp_path / "latent_fixture_sources.py").write_text(
                "from test_comfy_latent_nodes import PortableLatentSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "import sys, numpy as np\n"
                "from types import SimpleNamespace\n"
                "sys.modules.setdefault('torch', SimpleNamespace(as_tensor=np.asarray))\n"
                "NODES = (*COMFY_NODES, PortableLatentSource)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["latent-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "latent_fixture_sources:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "latent_fixture_sources:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            tensor = EncodedLatentTensor(
                dtype="float32", shape=(1, 2, 2, 3), data=np.arange(12, dtype=np.float32).tobytes()
            )
            mask = np.arange(6, dtype=np.float32).reshape(1, 2, 3) / 5
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "source": GraphNode("latent-fixture.source"),
                        "masked": GraphNode(
                            "SetLatentNoiseMask",
                            {"samples": Link("source", "samples"), "mask": Link("source", "mask")},
                        ),
                    }
                ),
                ["masked"],
            )
            output = cast(
                dict[str, Any],
                decode_latent(encode_latent(result.outputs["masked"]["LATENT"].resolve())),
            )
            assert output["samples"] == tensor
            assert output["noise_mask"].shape == (1, 1, 2, 3)
            assert output["noise_mask"].data == mask.tobytes()
            assert output["note"] == "keep"
        finally:
            await composer.close()

    asyncio.run(scenario())
