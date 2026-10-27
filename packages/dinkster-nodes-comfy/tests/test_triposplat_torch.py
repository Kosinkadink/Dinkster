"""CPU TripoSplat foreground geometry and ordinary host transport."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    SPLAT_TYPE,
    VAE_TYPE,
    InputSpec,
    MultiStreamLatent,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.triposplat import TripoSplatPreprocessImage, VAEDecodeTripoSplat

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


class SplatDecoder:
    gaussians_per_point = 96
    gs = SimpleNamespace(model_channels=7)

    def decode(self, latent: Any, *, num_gaussians: int, generator: Any) -> list[Any]:
        # Hand-derived fixture: each field exposes a distinct input to detect
        # dropped/reordered batch rows, camera selection, count and RNG mistakes.
        parts = []
        for row in latent:
            positions = row[:2, :3].clone()
            tensors = (
                positions,
                torch.full_like(positions, num_gaussians),
                torch.arange(8, dtype=latent.dtype).reshape(2, 4),
                torch.rand(2, 1, generator=generator, dtype=latent.dtype),
                torch.full((2, 1, 3), 0.25, dtype=latent.dtype),
            )
            parts.append(SimpleNamespace(render_tensors=lambda tensors=tensors: tensors))
        return parts


def _decoder_vae() -> Any:
    return SimpleNamespace(
        first_stage_model=SplatDecoder(),
        patcher=object(),
        vae_dtype=torch.float32,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("requested, expected", ((1, 32736), (32816, 32832), (2000000, 1048608)))
@pytest.mark.parametrize("streams", (False, True))
def test_decode_count_stream_selection_batch_and_rng(
    monkeypatch: pytest.MonkeyPatch, requested: int, expected: int, streams: bool
) -> None:
    initialize_inference_process()
    import dinkster_inference.model_management as management

    monkeypatch.setattr(management, "load_models_gpu", lambda *_args, **_kwargs: None)
    # Hand-derived clamp then nearest-96 rounding: 32768 -> 32736,
    # 32816 -> 32832, 1048576 -> 1048608 (rounding follows clamping).
    latent = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    before = latent.clone()
    payload = (
        MultiStreamLatent.from_pairs((("shape", latent), ("camera", torch.full((2, 1, 5), -500.0))))
        if streams
        else latent
    )
    samples = {"samples": payload}
    rng = torch.random.get_rng_state().clone()
    outputs = [
        VAEDecodeTripoSplat.execute(
            samples=samples, vae=_decoder_vae(), num_gaussians=requested, seed=123
        )["splat"]
        for _ in range(2)
    ]
    left, right = outputs
    for name in ("positions", "scales", "rotations", "opacities", "sh"):
        assert torch.equal(left[name], right[name])
        assert left[name].dtype == torch.float32
    assert torch.equal(left["positions"], latent[:, :2, :3].float())
    assert torch.equal(left["scales"], torch.full((2, 2, 3), float(expected)))
    assert torch.equal(left["rotations"], torch.arange(8).reshape(1, 2, 4).expand(2, 2, 4))
    assert torch.equal(left["sh"], torch.full((2, 2, 1, 3), 0.25))
    other = VAEDecodeTripoSplat.execute(
        samples=samples, vae=_decoder_vae(), num_gaussians=requested, seed=124
    )["splat"]
    assert not torch.equal(left["opacities"], other["opacities"])
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(latent, before)


class SplatSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.splat-source",
            outputs=(
                OutputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        initialize_inference_process()
        import dinkster_inference.model_management as management

        management.load_models_gpu = lambda *_args, **_kwargs: None
        latent = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        return cls.outputs(
            samples={
                "samples": MultiStreamLatent.from_pairs(
                    (("shape", latent), ("camera", torch.full((2, 1, 5), -500.0)))
                )
            },
            vae=_decoder_vae(),
        )


class SplatInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.splat-inspect",
            inputs=(InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, splat: Any) -> Any:
        # Hand-derived from SplatSource: shape stream's first two rows, not
        # the camera stream; default density 262144 rounds to 262176 at gpp96.
        splat = materialize_latent(splat)
        assert torch.equal(splat["positions"], torch.arange(24).reshape(2, 3, 4)[:, :2, :3].float())
        assert torch.equal(splat["scales"], torch.full((2, 2, 3), 262176.0))
        assert splat["rotations"].shape == (2, 2, 4)
        assert splat["opacities"].shape == (2, 2, 1)
        assert torch.equal(splat["sh"], torch.full((2, 2, 1, 3), 0.25))
        return cls.outputs(summary="splat batch transported")


@pytest.mark.parametrize("in_process", (True, False))
def test_native_manifest_decodes_splats_twice(tmp_path: Path, in_process: bool) -> None:
    initialize_inference_process()
    import dinkster_inference.model_management as management

    original_load = management.load_models_gpu

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "splat_fixture.py").write_text(
            "from test_triposplat_torch import SplatSource, SplatInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (SplatSource, SplatInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "splat.toml"
        fixture.write_text(
            '[pack]\nname="splat-fixture"\npublisher="dinkster"\n'
            'namespaces=["triposplat-fixture"]\n[pack.entry]\n'
            'nodes="splat_fixture:NODES"\ntypes="splat_fixture:register_inference_types"\n',
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
                        worker_group=None if in_process else "splat-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("triposplat-fixture.splat-source"),
                    "decode": GraphNode(
                        "VAEDecodeTripoSplat",
                        {"samples": Link("source", "samples"), "vae": Link("source", "vae")},
                    ),
                    "inspect": GraphNode(
                        "triposplat-fixture.splat-inspect", {"splat": Link("decode", "splat")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "splat batch transported"
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        management.load_models_gpu = original_load
