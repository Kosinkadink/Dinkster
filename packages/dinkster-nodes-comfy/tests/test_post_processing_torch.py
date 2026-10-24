"""CPU tensor and ordinary-pack execution behavior for post-processing nodes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_inference.cli_args import args

args.cpu = True

from dinkster_api.v1 import (  # noqa: E402
    CORE_FLOAT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link  # noqa: E402
from dinkster_nodes_comfy.post_processing import (  # noqa: E402
    BatchImagesNode,
    BatchLatentsNode,
    BatchMasksNode,
    ColorTransfer,
    ResizeImageMaskNode,
)

from dinkster.compose import PackSpec, ServingComposer  # noqa: E402

_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)


def test_post_processing_uses_real_cpu_inference_utilities_at_asymmetric_boundaries() -> None:
    image = torch.arange(45, dtype=torch.float32).reshape(1, 3, 5, 3) / 45
    mask = torch.arange(15, dtype=torch.float32).reshape(1, 3, 5) / 15
    image_before, mask_before = image.clone(), mask.clone()

    resized_image = ResizeImageMaskNode.execute(
        input=image,
        scale_method="nearest-exact",
        resize_type={"resize_type": "scale width", "width": 4},
    )["resized"]
    resized_mask = ResizeImageMaskNode.execute(
        input=mask,
        scale_method="nearest-exact",
        resize_type={"resize_type": "scale shorter dimension", "shorter_size": 4},
    )["resized"]
    assert resized_image.shape == (1, 2, 4, 3)
    assert resized_mask.shape == (1, 4, 7)
    torch.testing.assert_close(image, image_before)
    torch.testing.assert_close(mask, mask_before)

    images = BatchImagesNode.execute(
        images={"image0": torch.zeros((1, 2, 3, 3)), "image1": torch.ones((2, 4, 5, 4))}
    )["IMAGE"]
    masks = BatchMasksNode.execute(
        masks={"mask0": torch.zeros((1, 2, 3)), "mask1": torch.ones((2, 4, 5))}
    )["MASK"]
    latents = BatchLatentsNode.execute(
        latents={
            "latent0": {"samples": torch.ones((1, 4, 2, 3)), "batch_index": [7]},
            "latent1": {"samples": torch.zeros((2, 4, 4, 5))},
        }
    )["LATENT"]
    assert images.shape == (3, 2, 3, 4)
    assert torch.equal(images[0, ..., 3], torch.ones((2, 3)))
    assert masks.shape == (3, 2, 3)
    assert latents["samples"].shape == (3, 4, 2, 3)
    assert latents["batch_index"] == [7, 0, 1]

    target = torch.tensor(
        [[[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], [[0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]]]
    )
    reference = 1.0 - target
    target_before = target.clone()
    transfer = ColorTransfer.execute(
        image_target=target,
        image_ref=reference,
        method="histogram",
        source_stats={"source_stats": "per_frame"},
        strength=1.0,
    )["image"]
    assert torch.count_nonzero(transfer - target) > 0
    torch.testing.assert_close(target, target_before)


@pytest.mark.parametrize(("stats", "target_index"), (("uniform", 0), ("target_frame", 1)))
def test_mkl_lab_shared_statistics_preserve_reference_operation_order(
    stats: str, target_index: int
) -> None:
    generator = torch.Generator().manual_seed(511)
    target = torch.rand((2, 7, 11, 3), generator=generator)
    reference = torch.rand((1, 5, 9, 3), generator=generator)
    device = torch.device("cpu")
    epsilon = 1e-6
    channels, pixels = target.shape[-1], target.shape[1] * target.shape[2]
    reference_mean, reference_scale = ColorTransfer._pool_stats(reference, device, False, epsilon)
    if stats == "uniform":
        source_mean, source_scale = ColorTransfer._pool_stats(target, device, False, epsilon)
    else:
        source_lab = ColorTransfer._to_lab(target, target_index, device).view(channels, -1)
        source_mean, source_scale = ColorTransfer._frame_stats(source_lab, pixels, False, epsilon)
    matrix = ColorTransfer._mkl_matrix(source_scale, reference_scale, epsilon)
    source = ColorTransfer._to_lab(target, 0, device).view(channels, -1)
    expected = matrix @ source + (reference_mean - matrix @ source_mean)
    legacy = matrix @ source + reference_mean - matrix @ source_mean
    actual = ColorTransfer._lab_transform(target, reference, device, stats, target_index, False)(
        source, 0
    )

    assert not torch.equal(expected, legacy)
    assert torch.equal(actual, expected)


class PostProcessingSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="post-processing-fixture.source",
            outputs=(
                OutputSpec("target", _IMAGE),
                OutputSpec("reference", _IMAGE),
                OutputSpec("image", _IMAGE),
                OutputSpec("mask", _MASK),
                OutputSpec("latent", _LATENT),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        target = np.arange(45, dtype=np.float32).reshape(1, 3, 5, 3) / 45
        return cls.outputs(
            target=target,
            reference=1.0 - target,
            image=np.full((1, 2, 3, 3), 0.25, dtype=np.float32),
            mask=np.arange(6, dtype=np.float32).reshape(1, 2, 3) / 5,
            latent={"samples": np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3)},
        )


class PostProcessingInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="post-processing-fixture.inspect",
            inputs=(
                InputSpec("source", _IMAGE),
                InputSpec("resized", _IMAGE),
                InputSpec("images", _IMAGE),
                InputSpec("masks", _MASK),
                InputSpec("latents", _LATENT),
                InputSpec("transfer", _IMAGE),
            ),
            outputs=(
                OutputSpec("source_pixel", _FLOAT),
                OutputSpec("resized_width", _FLOAT),
                OutputSpec("image_batch", _FLOAT),
                OutputSpec("mask_batch", _FLOAT),
                OutputSpec("latent_batch", _FLOAT),
                OutputSpec("transfer_delta", _FLOAT),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        source: Any,
        resized: Any,
        images: Any,
        masks: Any,
        latents: Any,
        transfer: Any,
    ) -> dict[str, object]:
        source = materialize_image(source)
        resized = materialize_image(resized)
        images = materialize_image(images)
        masks = materialize_image(masks)
        latents = materialize_latent(latents)
        transfer = materialize_image(transfer)
        return cls.outputs(
            source_pixel=float(source[0, 0, 0, 0]),
            resized_width=float(resized.shape[2]),
            image_batch=float(images.shape[0]),
            mask_batch=float(masks.shape[0]),
            latent_batch=float(latents["samples"].shape[0]),
            transfer_delta=float(torch.abs(transfer - source).sum()),
        )


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_post_processing_dynamic_nodes_with_decoded_numpy_values(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), str(source_root), str(Path.cwd() / "src"))
                )
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "post_processing_fixture_sources.py").write_text(
                "from test_post_processing_torch import (\n"
                "    PostProcessingInspect, PostProcessingSource,\n"
                ")\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, PostProcessingSource, PostProcessingInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["post-processing-fixture",')
                .replace(
                    "dinkster_nodes_comfy:COMFY_NODES", "post_processing_fixture_sources:NODES"
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "post_processing_fixture_sources:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "source": GraphNode("post-processing-fixture.source"),
                        "resize": GraphNode(
                            "ResizeImageMaskNode",
                            {
                                "input": Link("source", "target"),
                                "scale_method": "nearest-exact",
                                "resize_type.width": 4,
                            },
                            slot_variants={"resize_type": "scale width"},
                        ),
                        "images": GraphNode(
                            "BatchImagesNode",
                            {
                                "images.image0": Link("source", "image"),
                                "images.image1": Link("source", "target"),
                            },
                        ),
                        "masks": GraphNode(
                            "BatchMasksNode",
                            {
                                "masks.mask0": Link("source", "mask"),
                                "masks.mask1": Link("source", "mask"),
                            },
                        ),
                        "latents": GraphNode(
                            "BatchLatentsNode",
                            {
                                "latents.latent0": Link("source", "latent"),
                                "latents.latent1": Link("source", "latent"),
                            },
                        ),
                        "transfer": GraphNode(
                            "ColorTransfer",
                            {
                                "image_target": Link("source", "target"),
                                "image_ref": Link("source", "reference"),
                                "method": "histogram",
                                "strength": 1.0,
                            },
                            slot_variants={"source_stats": "uniform"},
                        ),
                        "inspect": GraphNode(
                            "post-processing-fixture.inspect",
                            {
                                "source": Link("source", "target"),
                                "resized": Link("resize", "resized"),
                                "images": Link("images", "IMAGE"),
                                "masks": Link("masks", "MASK"),
                                "latents": Link("latents", "LATENT"),
                                "transfer": Link("transfer", "image"),
                            },
                        ),
                    }
                ),
                ["inspect"],
            )
            output = result.outputs["inspect"]
            assert output["source_pixel"].resolve() == 0.0
            assert output["resized_width"].resolve() == 4.0
            assert output["image_batch"].resolve() == 2.0
            assert output["mask_batch"].resolve() == 2.0
            assert output["latent_batch"].resolve() == 2.0
            assert output["transfer_delta"].resolve() > 0.0
        finally:
            await composer.close()

    asyncio.run(scenario())
