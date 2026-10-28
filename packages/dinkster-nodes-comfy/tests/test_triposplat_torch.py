"""CPU TripoSplat foreground geometry and ordinary host transport."""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.triposplat import TripoSplatConditioning, TripoSplatPreprocessImage

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize("erode_radius", (0, 2))
@pytest.mark.parametrize("mask_size", ((64, 96), (32, 48)))
def test_primary_colors_repeat_resize_masks_and_preserve_inputs(
    erode_radius: int, mask_size: tuple[int, int]
) -> None:
    # Hand-derived: RGB values 2/-1 clamp to primary colors at the center;
    # a centered rectangular alpha region leaves black corners after square padding.
    # The declared stride rounds size 271 down to 256, independently of input size.
    colors = torch.eye(3)
    image = (colors * 3 - 1)[:, None, None, :].expand(3, 64, 96, 3).clone()
    mask = torch.zeros(1, *mask_size)
    height, width = mask_size
    mask[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
    original_image, original_mask = image.clone(), mask.clone()
    result = TripoSplatPreprocessImage.execute(
        image=image.numpy(), mask=mask.numpy(), erode_radius=erode_radius, size=271
    )["image"]
    assert result.shape == (3, 256, 256, 3) and result.dtype == torch.float32
    torch.testing.assert_close(result[:, 128, 128], colors, rtol=0, atol=0)
    assert torch.count_nonzero(result[:, 0, 0]) == 0
    assert torch.all((result >= 0) & (result <= 1))
    torch.testing.assert_close(image, original_image, rtol=0, atol=0)
    torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)


def test_empty_mask_reports_foreground_error() -> None:
    # Hand-derived: an all-zero mask has no foreground, regardless of RGB values.
    with pytest.raises(ValueError, match="mask is empty .*no foreground pixels"):
        TripoSplatPreprocessImage.execute(
            image=torch.ones(1, 48, 80, 3),
            mask=torch.zeros(1, 48, 80),
            erode_radius=1,
            size=256,
        )


def test_erosion_removes_a_foreground_strip_narrower_than_the_kernel() -> None:
    # Hand-derived: a 5x5 minimum filter removes a three-row strip. Input and
    # requested size are both 256, so the first resize cannot widen the strip.
    image = torch.ones(1, 256, 256, 3)
    mask = torch.zeros(1, 256, 256)
    mask[:, 100:103, 50:200] = 1
    un_eroded = TripoSplatPreprocessImage.execute(image=image, mask=mask, erode_radius=0, size=256)[
        "image"
    ]
    assert torch.count_nonzero(un_eroded) > 0
    with pytest.raises(ValueError, match="mask is empty .*no foreground pixels"):
        TripoSplatPreprocessImage.execute(image=image, mask=mask, erode_radius=2, size=256)


class PreprocessSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.source",
            outputs=(
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        image = torch.zeros(1, 64, 96, 3)
        image[..., 0] = 1
        mask = torch.zeros(1, 64, 96)
        mask[:, 16:48, 24:72] = 1
        return cls.outputs(image=image.numpy(), mask=mask.numpy())


class PreprocessInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.inspect",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, image: Any) -> Any:
        # Hand-derived from PreprocessSource's uniform red RGB, centered alpha
        # rectangle and requested size 256: red center, black corners, no green/blue.
        image = materialize_image(image)
        assert image.shape == (1, 256, 256, 3)
        torch.testing.assert_close(
            image[0, 128, 128], torch.tensor([1.0, 0.0, 0.0]), rtol=0, atol=0
        )
        assert torch.count_nonzero(image[0, 0, 0]) == 0
        assert torch.count_nonzero(image[..., 1:]) == 0
        return cls.outputs(summary="foreground crop transported")


@pytest.mark.parametrize("in_process", (True, False))
def test_native_manifest_preprocesses_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "triposplat_fixture.py").write_text(
            "from test_triposplat_torch import PreprocessSource, PreprocessInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (PreprocessSource, PreprocessInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="triposplat-fixture"\npublisher="dinkster"\n'
            'namespaces=["triposplat-fixture"]\n[pack.entry]\n'
            'nodes="triposplat_fixture:NODES"\ntypes="triposplat_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "triposplat-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("triposplat-fixture.source"),
                    "preprocess": GraphNode(
                        "TripoSplatPreprocessImage",
                        {
                            "image": Link("source", "image"),
                            "mask": Link("source", "mask"),
                            "size": 256,
                        },
                    ),
                    "inspect": GraphNode(
                        "triposplat-fixture.inspect", {"image": Link("preprocess", "image")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve() == "foreground crop transported"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())


class ConditioningVision:
    patcher = None
    load_device = torch.device("cpu")

    def __init__(self) -> None:
        self.model = self

    def __call__(self, *, pixel_values: Any) -> tuple[Any]:
        assert pixel_values.dtype == torch.float32
        return (pixel_values.movedim(1, -1).flatten(1, 2),)


class ConditioningVAE:
    def __init__(self, first_only: bool = False) -> None:
        self.first_only = first_only

    def encode(self, image: Any) -> Any:
        return (image[:1] if self.first_only else image)[..., :2] * 7 + 3


def _conditioning_image(dtype: Any = torch.float32) -> Any:
    return torch.arange(36, dtype=dtype).reshape(2, 2, 3, 3) / 35


def _assert_conditioning(positive: Any, negative: Any, latent: Any, first_only: bool) -> None:
    # Hand-derived: fixture pixel i is i/35. Apply ImageNet normalization,
    # then population-variance normalization with epsilon1e-5 in Python floats.
    # Source constants: TripoSplatConditioning F bc5f880d, image arange(36)/35
    # shaped(2,2,3,3), mean(.485,.456,.406), std(.229,.224,.225), shape8192x16/camera1x5.
    expected = []
    for pixel in range(12):
        rgb = [
            ((pixel * 3 + channel) / 35 - mean) / std
            for channel, (mean, std) in enumerate(
                zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), strict=True)
            )
        ]
        average = sum(rgb) / 3
        denominator = math.sqrt(sum((channel - average) ** 2 for channel in rgb) / 3 + 1e-5)
        expected.append([(channel - average) / denominator for channel in rgb])
    expected = torch.tensor(expected).reshape(2, 6, 3)
    torch.testing.assert_close(positive[0][0], expected, rtol=1e-5, atol=2e-6)
    reference = _conditioning_image()[..., :2] * 7 + 3
    reference = reference[:1] if first_only else reference
    torch.testing.assert_close(
        positive[0][1]["reference_latents"][0], reference, rtol=1e-6, atol=1e-6, check_dtype=False
    )
    assert torch.count_nonzero(negative[0][0]) == 0
    assert negative[0][0].shape == (2, 6, 3)
    assert torch.equal(negative[0][1]["reference_latents"][0], torch.zeros_like(reference))
    samples = latent["samples"]
    assert samples.roles == ("shape", "camera")
    batch = 1 if first_only else 2
    assert samples.by_role("shape").shape == (batch, 8192, 16)
    assert samples.by_role("camera").shape == (batch, 1, 5)
    assert torch.count_nonzero(samples.by_role("shape")) == 0
    assert torch.count_nonzero(samples.by_role("camera")) == 0


@pytest.mark.parametrize("first_only", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_conditioning_normalization_reference_batch_and_zero_streams(
    monkeypatch: pytest.MonkeyPatch, first_only: bool, dtype: Any
) -> None:
    initialize_inference_process()
    import dinkster_inference.model_management as management

    monkeypatch.setattr(management, "load_model_gpu", lambda *_args: None)
    monkeypatch.setattr(management, "intermediate_device", lambda: torch.device("cpu"))
    image = _conditioning_image(dtype)
    original = image.clone()
    result = TripoSplatConditioning.execute(
        image=image.numpy(), clip_vision=ConditioningVision(), vae=ConditioningVAE(first_only)
    )
    _assert_conditioning(result["positive"], result["negative"], result["latent"], first_only)
    assert torch.equal(image, original)
    # Hand-derived independence: writing the negative reference does not
    # change the positive reference or original input.
    result["negative"][0][1]["reference_latents"][0].fill_(99)
    assert not torch.any(result["positive"][0][1]["reference_latents"][0] == 99)
    assert torch.equal(image, original)


class ConditioningSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.conditioning-source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("image", IMAGE_TYPE),
                    ("vision", CLIP_VISION_TYPE),
                    ("vae", VAE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        initialize_inference_process()
        import dinkster_inference.model_management as management

        management.load_model_gpu = lambda *_args: None
        management.intermediate_device = lambda: torch.device("cpu")
        return cls.outputs(
            image=_conditioning_image().numpy(), vision=ConditioningVision(), vae=ConditioningVAE()
        )


class ConditioningInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.conditioning-inspect",
            inputs=(
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any, latent: Any) -> Any:
        _assert_conditioning(
            materialize_latent(positive),
            materialize_latent(negative),
            materialize_latent(latent),
            False,
        )
        return cls.outputs(summary="conditioning and ordered noise targets transported")


@pytest.mark.parametrize("in_process", (True, False))
def test_native_manifest_conditions_twice(tmp_path: Path, in_process: bool) -> None:
    initialize_inference_process()
    import dinkster_inference.model_management as management

    original_load = management.load_model_gpu
    original_device = management.intermediate_device

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "conditioning_fixture.py").write_text(
            "from test_triposplat_torch import ConditioningSource, ConditioningInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (ConditioningSource, ConditioningInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "conditioning.toml"
        fixture.write_text(
            '[pack]\nname="conditioning-fixture"\npublisher="dinkster"\n'
            'namespaces=["triposplat-fixture"]\n[pack.entry]\n'
            'nodes="conditioning_fixture:NODES"\ntypes="conditioning_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "conditioning-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("triposplat-fixture.conditioning-source"),
                    "conditioning": GraphNode(
                        "TripoSplatConditioning",
                        {
                            "image": Link("source", "image"),
                            "clip_vision": Link("source", "vision"),
                            "vae": Link("source", "vae"),
                        },
                    ),
                    "inspect": GraphNode(
                        "triposplat-fixture.conditioning-inspect",
                        {
                            name: Link("conditioning", name)
                            for name in ("positive", "negative", "latent")
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "conditioning and ordered noise targets transported"
                )
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        management.load_model_gpu = original_load
        management.intermediate_device = original_device
