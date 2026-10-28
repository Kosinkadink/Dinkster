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
from kornia import color

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
    total_pixels = ResizeImageMaskNode.execute(
        input=torch.zeros((1, 7, 11, 3)),
        scale_method="nearest-exact",
        resize_type={"resize_type": "scale total pixels", "megapixels": 0.013746166229248047},
    )["resized"]
    assert total_pixels.shape == (1, 96, 150, 3)

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
    target_before = target.clone()
    epsilon = 1e-6
    channels, pixels = target.shape[-1], target.shape[1] * target.shape[2]

    def to_lab(images: torch.Tensor, index: int) -> torch.Tensor:
        return color.rgb_to_lab(
            images[index : index + 1].to(dtype=torch.float32).permute(0, 3, 1, 2)
        )

    def pooled_statistics(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame_count = images.shape[0]
        frame_pixels = images.shape[1] * images.shape[2]
        mean = torch.zeros((channels, 1), dtype=torch.float32)
        for index in range(frame_count):
            mean += to_lab(images, index).view(channels, -1).mean(dim=-1, keepdim=True)
        mean /= frame_count
        covariance = torch.zeros((channels, channels), dtype=torch.float32)
        for index in range(frame_count):
            centered = to_lab(images, index).view(channels, -1) - mean
            covariance += centered @ centered.T / frame_pixels
        return mean, covariance / frame_count

    def frame_statistics(lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = lab.mean(dim=-1, keepdim=True)
        centered = lab - mean
        return mean, centered @ centered.T / pixels

    def mkl_matrix(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        source_values, source_vectors = torch.linalg.eigh(source)
        source_root = torch.sqrt(source_values.clamp_min(0)).clamp_min(epsilon)
        scaled_vectors = source_vectors * source_root.unsqueeze(0)
        middle_values, middle_vectors = torch.linalg.eigh(
            scaled_vectors.T @ reference @ scaled_vectors
        )
        middle_root = torch.sqrt(middle_values.clamp_min(0))
        inverse_scaled = source_vectors * (1.0 / source_root).unsqueeze(0)
        return (
            inverse_scaled
            @ ((middle_vectors * middle_root.unsqueeze(0)) @ middle_vectors.T)
            @ inverse_scaled.T
        )

    reference_mean, reference_covariance = pooled_statistics(reference)
    if stats == "uniform":
        source_mean, source_covariance = pooled_statistics(target)
    else:
        source_mean, source_covariance = frame_statistics(
            to_lab(target, target_index).view(channels, -1)
        )
    transform = mkl_matrix(source_covariance, reference_covariance)
    offset = reference_mean - transform @ source_mean

    def render(*, legacy_association: bool) -> torch.Tensor:
        frames = []
        for index in range(target.shape[0]):
            source = to_lab(target, index)
            flattened = source.view(channels, -1)
            corrected = (
                transform @ flattened + reference_mean - transform @ source_mean
                if legacy_association
                else transform @ flattened + offset
            )
            frames.append(
                color.lab_to_rgb(corrected.view_as(source)).squeeze(0).permute(1, 2, 0).clamp(0, 1)
            )
        return torch.stack(frames)

    expected = render(legacy_association=False)
    legacy = render(legacy_association=True)
    actual = ColorTransfer.execute(
        image_target=target,
        image_ref=reference,
        method="mkl_lab",
        source_stats={"source_stats": stats, "target_index": target_index},
        strength=1.0,
    )["image"]

    assert not torch.equal(expected, legacy)
    assert torch.equal(actual, expected)
    assert torch.equal(target, target_before)


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


@pytest.mark.parametrize("in_process", (True,))
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
