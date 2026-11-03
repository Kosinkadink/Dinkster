"""Pixal3D public conditioning preserves view order, patch layout and camera rigs."""

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
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.pixal_conditioning import (
    Pixal3DConditioning,
    Pixal3DMultiViewConditioning,
)

from dinkster.compose import PackSpec, ServingComposer


class TinyDino(torch.nn.Module):
    def forward(self, pixels: Any, *, skip_norm_elementwise: bool) -> Any:
        assert skip_norm_elementwise
        channels = pixels.mean(dim=(-2, -1))
        side = pixels.shape[-1] // 16
        globals_ = channels[:, None] + torch.tensor([0, 10, 20])[None, :, None]
        patches = channels[:, None] + torch.arange(side * side)[None, :, None] / 10000
        return (torch.cat((globals_, patches), dim=1),)


class TinyNaf(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, image: Any, features: Any, target: Any, *, output: Any) -> None:
        assert image.dtype == features.dtype == output.dtype == self.weight.dtype
        output.copy_(features[..., :1, :1].expand(-1, -1, *target) + image.mean())


class Vision:
    def __init__(self, *, custom: bool = True, naf: bool = False) -> None:
        self.model = TinyDino()
        self.patcher = object()
        self.image_mean = [0, 0, 0] if custom else None
        self.image_std = [1, 1, 1] if custom else None
        self.naf = type("NafPatcher", (), {"model": TinyNaf()})() if naf else None


def image_batch() -> Any:
    # Alpha must be ignored for single-view, unlike multi-view compositing.
    return (
        torch.tensor([[2.0, -1.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.5]])[:, None, None]
        .expand(2, 7, 11, 4)
        .clone()
    )


def check_single(positive: Any, negative: Any, *, custom: bool = True, naf: bool = False) -> None:
    colors = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    if not custom:
        colors = (colors - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
            [0.229, 0.224, 0.225]
        )
    expected = colors[:, None] + torch.tensor([0, 10, 20])[None, :, None]
    torch.testing.assert_close(positive[0][0], expected)
    torch.testing.assert_close(positive[0][1]["embeds"], expected)
    assert torch.count_nonzero(negative[0][0]) == 0
    assert torch.count_nonzero(negative[0][1]["embeds"]) == 0
    pack = positive[0][1]["proj_feat_pack"]
    assert pack["num_views"] == 1 and pack["patch_size"] == 16
    torch.testing.assert_close(pack["mesh_scale"], torch.ones(2))
    torch.testing.assert_close(pack["camera_angle_x"], torch.full((2,), math.pi / 2))
    assert positive[0][1]["trellis2_proj_feats"].shape == (2, 4096, 3)
    assert torch.isfinite(positive[0][1]["trellis2_proj_feats"]).all()
    for stage, side, size, hr_side in (
        ("ss", 32, 512, None),
        ("shape_512", 32, 512, 512),
        ("shape_1024", 64, 1024, 512),
        ("tex_1024", 64, 1024, 1024),
    ):
        entry = pack["stages"][stage]
        assert entry["image_resolution"] == size
        patches = entry["feature_map"]
        assert patches.shape == (2, 3, side, side)
        torch.testing.assert_close(patches[:, :, 0, 1], colors + 0.0001)
        torch.testing.assert_close(patches[:, :, -1, -1], colors + (side * side - 1) / 10000)
        high = entry["feature_map_hr"]
        if naf and hr_side is not None:
            assert high.shape == (2, 3, hr_side, hr_side) and high.dtype == torch.float64
            torch.testing.assert_close(high[:, :, 0, 0], colors.double() + 1 / 3)
        else:
            assert high is None


@pytest.mark.parametrize("custom,naf", ((False, False), (True, False), (True, True)))
def test_single_view_patch_layout_projection_and_naf(
    monkeypatch: pytest.MonkeyPatch, custom: bool, naf: bool
) -> None:
    initialize_inference_process()
    import dinkster_inference.model_management as management

    monkeypatch.setattr(management, "load_model_gpu", lambda _model: None)
    image = image_batch().double()
    before = image.clone()
    result = Pixal3DConditioning.execute(
        clip_vision_model=Vision(custom=custom, naf=naf), image=image.numpy(), camera_angle_x=90
    )
    check_single(result["positive"], result["negative"], custom=custom, naf=naf)
    result["negative"][0][0].fill_(123)
    assert torch.count_nonzero(result["negative"][0][1]["embeds"]) == 0
    assert not torch.any(result["positive"][0][0] == 123)
    assert torch.equal(image, before)


def test_multiview_rebases_camera_and_cycles_shorter_view(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    initialize_inference_process()
    import dinkster_inference.model_management as management

    monkeypatch.setattr(management, "load_model_gpu", lambda _model: None)
    left = (
        torch.tensor([[1.0, 0, 0, 0.5], [0, 1.0, 0, 0.25]])[:, None, None]
        .expand(2, 5, 9, 4)
        .clone()
    )
    back = torch.tensor([0.0, 0, 1.0])[None, None, None].expand(1, 3, 7, 3).clone()
    result = Pixal3DMultiViewConditioning.execute(
        clip_vision_model=Vision(), fov=90, left=left, back=back
    )
    positive, negative = result["positive"], result["negative"]
    # The pinned Lanczos path truncates to uint8: .5 -> 127, .25 -> 63.
    colors = torch.tensor([[127 / 510, 0, 0.5], [0, 63 / 510, 0.5]])
    torch.testing.assert_close(positive[0][0][:, 0], colors)
    assert torch.count_nonzero(negative[0][0]) == 0
    pack = positive[0][1]["proj_feat_pack"]
    assert pack["num_views"] == 2
    torch.testing.assert_close(
        pack["stages"]["ss"]["feature_map"][:, :, 0, 0],
        torch.tensor([[127 / 255, 0, 0], [0, 0, 1.0], [0, 63 / 255, 0], [0, 0, 1.0]]),
    )
    positions = pack["transform_matrix"][:, :3, 3]
    torch.testing.assert_close(
        positions,
        torch.tensor([[0, -0.55, 0], [0.55, 0, 0], [0, -0.55, 0], [0.55, 0, 0]]),
        atol=1e-7,
        rtol=0,
    )
    torch.testing.assert_close(
        pack["transform_matrix"][0, :3, :3], torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    )
    assert "left view as its front" in caplog.text


def test_multiview_requires_a_view() -> None:
    with pytest.raises(ValueError, match="needs at least one view"):
        Pixal3DMultiViewConditioning.execute(clip_vision_model=Vision(), fov=20)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="pixal-fixture.source",
            outputs=(
                OutputSpec("vision", TypeExpr.concrete(CLIP_VISION_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        initialize_inference_process()
        import dinkster_inference.model_management as management

        management.load_model_gpu = lambda _model: None
        return cls.outputs(vision=Vision(), image=image_batch().numpy()[..., :3])


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="pixal-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("positive", "negative")
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any) -> Any:
        check_single(materialize_latent(positive), materialize_latent(negative))
        return cls.outputs(summary="pixal projections transported")


@pytest.mark.parametrize("in_process", (False, True))
@pytest.mark.parametrize("multi_view", (False, True))
def test_native_manifest_conditions_twice(
    tmp_path: Path, in_process: bool, multi_view: bool
) -> None:
    initialize_inference_process()
    import dinkster_inference.model_management as management

    original_load = management.load_model_gpu

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        fixture_module = f"pixal_fixture_{int(multi_view)}"
        (tmp_path / f"{fixture_module}.py").write_text(
            "from test_pixal_conditioning_torch import Source, Inspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (Source, Inspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="pixal-fixture"\npublisher="dinkster"\n'
            'namespaces=["pixal-fixture"]\n[pack.entry]\n'
            f'nodes="{fixture_module}:NODES"\n'
            f'types="{fixture_module}:register_inference_types"\n',
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
                        worker_group=None if in_process else "pixal-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("pixal-fixture.source"),
                    "conditioning": GraphNode(
                        "Pixal3DMultiViewConditioning" if multi_view else "Pixal3DConditioning",
                        {
                            "clip_vision_model": Link("source", "vision"),
                            "front" if multi_view else "image": Link("source", "image"),
                            "fov" if multi_view else "camera_angle_x": 90.0,
                        },
                    ),
                    "inspect": GraphNode(
                        "pixal-fixture.inspect",
                        {name: Link("conditioning", name) for name in ("positive", "negative")},
                    ),
                }
            )
            for _ in range(2):
                engine = composer.composition.make_engine(lambda _event: None)
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "pixal projections transported"
                )
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        management.load_model_gpu = original_load
