"""Dataset grouping preserves pixels, latent order and paired conditions."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import PORTS_NODE_ID, Graph, GraphNode, Link, RegionNode, RegionOutput
from dinkster_nodes_comfy.dataset_groups import MergeImageLists, ResolutionBucket

from dinkster.compose import PackSpec, ServingComposer

_IMAGES = TypeExpr.list_of(TypeExpr.concrete(IMAGE_TYPE))
_LATENTS = TypeExpr.list_of(TypeExpr.concrete(LATENT_TYPE))
_CONDITIONS = TypeExpr.list_of(TypeExpr.concrete(CONDITIONING_TYPE))
_STRINGS = TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))


def test_merge_images_flattens_batches_without_resizing_or_reordering() -> None:
    first = torch.arange(36, dtype=torch.float32).reshape(2, 2, 3, 3) / 36
    second = np.ones((1, 3, 1, 4), dtype=np.float32)
    result = MergeImageLists.execute(images=[first, second])["images"]
    assert [tuple(frame.shape) for frame in result] == [(1, 2, 3, 3), (1, 2, 3, 3), (1, 3, 1, 4)]
    assert torch.equal(result[0], first[:1])
    assert torch.equal(result[1], first[1:])
    assert torch.equal(result[2], torch.ones((1, 3, 1, 4)))
    assert len(MergeImageLists.execute(images=first)["images"]) == 2
    assert MergeImageLists.execute(images=[])["images"] == []
    with pytest.raises(ValueError, match="4D"):
        MergeImageLists.execute(images=[torch.zeros(2, 3, 3)])


def _dataset() -> tuple[list[dict[str, Any]], list[Any]]:
    latents = [
        {
            "samples": torch.stack((torch.full((4, 3, 2), 11.0), torch.full((4, 3, 2), 12.0))),
            "noise_mask": torch.ones(1, 3, 2),
        },
        {"samples": torch.full((1, 4, 2, 3), 21.0)},
        {"samples": torch.full((1, 4, 3, 2), 31.0), "batch_index": [9]},
    ]
    entries = [
        [torch.full((1, 2, 3), float(value)), {"label": label}]
        for value, label in ((11, "first"), (12, "second"), (21, "third"), (31, "fourth"))
    ]
    conditions = [entries[:2], entries[2:3], entries[3:]]
    return latents, conditions


def test_resolution_buckets_preserve_first_seen_order_and_pairing() -> None:
    latents, conditions = _dataset()
    before = [item["samples"].clone() for item in latents]
    result = ResolutionBucket.execute(latents=latents, conditioning=conditions)
    assert [tuple(item["samples"].shape) for item in result["latents"]] == [
        (3, 4, 3, 2),
        (1, 4, 2, 3),
    ]
    assert [item["samples"][:, 0, 0, 0].tolist() for item in result["latents"]] == [
        [11, 12, 31],
        [21],
    ]
    assert [[entry[1]["label"] for entry in group] for group in result["conditioning"]] == [
        ["first", "second", "fourth"],
        ["third"],
    ]
    assert [entry[0].flatten()[0].item() for entry in result["conditioning"][0]] == [11, 12, 31]
    assert all(set(item) == {"samples"} for item in result["latents"])
    for item, original in zip(latents, before, strict=True):
        assert torch.equal(item["samples"], original)
    assert "noise_mask" in latents[0] and latents[2]["batch_index"] == [9]
    assert ResolutionBucket.execute(latents=[], conditioning=[]) == {
        "latents": [],
        "conditioning": [],
    }
    with pytest.raises(ValueError, match="does not match"):
        ResolutionBucket.execute(latents=latents, conditioning=conditions[:1])


class DatasetGroupSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dataset-group-fixture.source",
            outputs=(
                OutputSpec("images", _IMAGES),
                OutputSpec("latents", _LATENTS),
                OutputSpec("conditioning", _CONDITIONS),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        latents, conditions = _dataset()
        return cls.outputs(
            images=[
                np.full((2, 2, 3, 3), 0.25, dtype=np.float32),
                np.full((1, 3, 1, 4), 0.75, dtype=np.float32),
            ],
            latents=latents,
            conditioning=conditions,
        )


class DatasetBucketInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dataset-group-fixture.bucket-inspect",
            inputs=(
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
            outputs=(OutputSpec("pairs", _STRINGS),),
        )

    @classmethod
    def execute(cls, *, latent: Any, conditioning: Any) -> Any:
        samples = materialize_latent(latent)["samples"]
        assert samples.shape[0] == len(conditioning)
        pairs = []
        for sample, (embedding, metadata) in zip(samples, conditioning, strict=True):
            value = int(sample[0, 0, 0])
            assert torch.all(sample == value)
            assert torch.all(embedding == value)
            pairs.append(f"{metadata['label']}:{value}")
        return cls.outputs(pairs=pairs)


class DatasetGroupInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dataset-group-fixture.inspect",
            inputs=(
                InputSpec("images", _IMAGES),
                InputSpec("latents", _LATENTS),
                InputSpec("conditioning", _CONDITIONS),
                InputSpec("inverted_images", _IMAGES),
                InputSpec("pairs", _STRINGS),
            ),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(
        cls, *, images: Any, latents: Any, conditioning: Any, inverted_images: Any, pairs: Any
    ) -> Any:
        assert [tuple(materialize_image(image).shape) for image in images] == [
            (1, 2, 3, 3),
            (1, 2, 3, 3),
            (1, 3, 1, 4),
        ]
        assert [float(materialize_image(image).flatten()[0]) for image in images] == [
            0.25,
            0.25,
            0.75,
        ]
        samples = [materialize_latent(item)["samples"] for item in latents]
        assert [item[:, 0, 0, 0].tolist() for item in samples] == [[11, 12, 31], [21]]
        assert [[entry[1]["label"] for entry in group] for group in conditioning] == [
            ["first", "second", "fourth"],
            ["third"],
        ]
        assert [entry[0].flatten()[0].item() for entry in conditioning[0]] == [11, 12, 31]
        assert len(inverted_images) == 3
        for original, inverted in zip(images, inverted_images, strict=True):
            original = materialize_image(original)
            inverted = materialize_image(inverted)
            assert inverted.shape == original.shape
            assert torch.equal(inverted[..., :3], 1.0 - original[..., :3])
            if original.shape[-1] == 4:
                assert torch.equal(inverted[..., 3], original[..., 3])
        assert pairs == ["first:11", "second:12", "fourth:31", "third:21"]
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_dataset_groups_ordinary_manifest_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(Path.cwd() / "src")))
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "dataset_group_fixture.py").write_text(
                "from test_dataset_groups_torch import (\n"
                "    DatasetGroupSource, DatasetGroupInspect, DatasetBucketInspect,\n"
                ")\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (\n"
                "    *COMFY_NODES, DatasetGroupSource, DatasetGroupInspect, DatasetBucketInspect,\n"
                ")\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["dataset-group-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "dataset_group_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "dataset_group_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("dataset-group-fixture.source"),
                    "merge": GraphNode("MergeImageLists", {"images": Link("source", "images")}),
                    "bucket": GraphNode(
                        "ResolutionBucket",
                        {
                            "latents": Link("source", "latents"),
                            "conditioning": Link("source", "conditioning"),
                        },
                    ),
                    "per_image": RegionNode(
                        kind="map",
                        body=Graph(
                            nodes={
                                "invert": GraphNode(
                                    "ImageInvert", {"image": Link(PORTS_NODE_ID, "image")}
                                )
                            }
                        ),
                        ports={"image": TypeExpr.concrete(IMAGE_TYPE)},
                        inputs={"image": Link("merge", "images")},
                        element_ports=("image",),
                        outputs={"images": RegionOutput(Link("invert", "IMAGE"))},
                    ),
                    "per_bucket": RegionNode(
                        kind="map",
                        binding="zip",
                        body=Graph(
                            nodes={
                                "inspect": GraphNode(
                                    "dataset-group-fixture.bucket-inspect",
                                    {
                                        "latent": Link(PORTS_NODE_ID, "latent"),
                                        "conditioning": Link(PORTS_NODE_ID, "conditioning"),
                                    },
                                )
                            }
                        ),
                        ports={
                            "latent": TypeExpr.concrete(LATENT_TYPE),
                            "conditioning": TypeExpr.concrete(CONDITIONING_TYPE),
                        },
                        inputs={
                            "latent": Link("bucket", "latents"),
                            "conditioning": Link("bucket", "conditioning"),
                        },
                        element_ports=("latent", "conditioning"),
                        outputs={"pairs": RegionOutput(Link("inspect", "pairs"), mode="flatten")},
                    ),
                    "inspect": GraphNode(
                        "dataset-group-fixture.inspect",
                        {
                            "images": Link("merge", "images"),
                            "latents": Link("bucket", "latents"),
                            "conditioning": Link("bucket", "conditioning"),
                            "inverted_images": Link("per_image", "images"),
                            "pairs": Link("per_bucket", "pairs"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
        finally:
            await composer.close()

    asyncio.run(scenario())
