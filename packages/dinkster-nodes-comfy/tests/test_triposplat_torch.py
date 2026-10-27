"""CPU TripoSplat foreground geometry and ordinary host transport."""

from __future__ import annotations

import asyncio
import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.triposplat import TripoSplatPreprocessImage, TripoSplatSamplingPreview
from dinkster_schema import use_reporter
from PIL import Image

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


class PreviewModel:
    def __init__(self) -> None:
        self.model = SimpleNamespace(diffusion_model=SimpleNamespace(q_token_length=7))
        self.wrapper: Any = None

    def clone(self) -> PreviewModel:
        return PreviewModel()

    def add_wrapper_with_key(self, kind: Any, key: str, wrapper: Any) -> None:
        self.wrapper = wrapper


def _preview_vae() -> Any:
    return SimpleNamespace(
        first_stage_model=SimpleNamespace(
            gaussians_per_point=6, gs=SimpleNamespace(model_channels=3)
        ),
        patcher="fixture-vae",
        vae_dtype=torch.float32,
    )


@pytest.mark.parametrize("positional", (False, True))
@pytest.mark.parametrize("fail_after", (None, 0, 1))
def test_sampling_preview_preserves_callbacks_and_disables_only_this_run(
    monkeypatch: pytest.MonkeyPatch, positional: bool, fail_after: int | None
) -> None:
    initialize_inference_process()
    import dinkster_inference.ldm.triposplat.preview as preview
    import dinkster_inference.model_management as management

    loads: list[Any] = []
    decodes: list[Any] = []
    original_calls: list[Any] = []
    monkeypatch.setattr(management, "loaded_models", lambda **kwargs: ["existing"])
    monkeypatch.setattr(
        management, "load_models_gpu", lambda models, **kwargs: loads.append((models, kwargs))
    )

    def decode(vae: Any, x0: Any, config: Any) -> Any:
        decodes.append((x0, config))
        if fail_after is not None and len(decodes) > fail_after:
            raise RuntimeError("fixture preview failure")
        return Image.new("RGB", (37, 19), (30, 90, 150))

    monkeypatch.setattr(preview, "decode_x0_to_image", decode)
    model = PreviewModel()
    patched = TripoSplatSamplingPreview.execute(
        model=model,
        vae=_preview_vae(),
        num_gaussians=1030,
        octree_level=3,
        yaw=-27.0,
        pitch=41.0,
        point_size=4,
    )["model"]
    assert model.wrapper is None and patched is not model

    def executor(*args: Any, **kwargs: Any) -> str:
        callback = args[5] if positional else kwargs["callback"]
        for step in range(3):
            callback(step, "shape", "noisy", 3)
        return "sampler result"

    def original(*args: Any) -> None:
        original_calls.append(args)

    events: list[Any] = []
    for _ in range(2):
        decodes.clear()
        events.clear()
        with use_reporter(lambda name, data, blob: events.append((name, data, blob))):
            result = (
                patched.wrapper(executor, 0, 1, 2, 3, 4, original)
                if positional
                else patched.wrapper(executor, callback=original)
            )
        assert result == "sampler result"
        expected = 3 if fail_after is None else fail_after
        previews = [item for item in events if item[0] == "preview"]
        assert len(previews) == expected
        assert len(decodes) == (3 if fail_after is None else fail_after + 1)
        # Hand-derived fixture memory: (7*4 + floor(1030/6)*10)*3*4 = 20856.
        assert loads[-1] == (["existing", "fixture-vae"], {"memory_required": 20856})
        for _, data, blob in previews:
            # TripoSplatSamplingPreview F bc5f880d sends JPEG/max512 to server.py.
            # For fixture RGB 37x19, contain gives 512 x round(19/37*512) = 263.
            assert data == {"mime": "image/jpeg", "width": 512, "height": 263}
            assert Image.open(BytesIO(blob)).size == (512, 263)
        assert [item[1]["step"] for item in events if item[0] == "progress"] == list(
            range(1, expected + 1)
        )
        assert decodes[0] == (
            "shape",
            {"gaussians": 1030, "level": 3, "yaw": -27.0, "pitch": 41.0, "point_size": 4},
        )
    assert len(loads) == 2
    assert original_calls == [(step, "shape", "noisy", 3) for step in range(3)] * 2


class SamplingPreviewSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.preview-source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        initialize_inference_process()
        import dinkster_inference.ldm.triposplat.preview as preview
        import dinkster_inference.model_management as management

        management.loaded_models = lambda **kwargs: []
        management.load_models_gpu = lambda *_args, **_kwargs: None
        preview.decode_x0_to_image = lambda *_args: Image.new("RGB", (37, 19), (30, 90, 150))
        return cls.outputs(model=PreviewModel(), vae=_preview_vae())


class SamplingPreviewInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.preview-inspect",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> Any:
        calls: list[Any] = []

        def executor(*args: Any, **kwargs: Any) -> str:
            kwargs["callback"](0, "shape", "noise", 2)
            kwargs["callback"](1, "shape", "noise", 2)
            return "sampling continued"

        assert (
            model.wrapper(executor, callback=lambda *args: calls.append(args))
            == "sampling continued"
        )
        assert len(calls) == 2
        return cls.outputs(summary="patched model and sampler previews transported")


@pytest.mark.parametrize("in_process", (True, False))
def test_native_manifest_sampling_preview_twice(tmp_path: Path, in_process: bool) -> None:
    initialize_inference_process()
    import dinkster_inference.ldm.triposplat.preview as preview
    import dinkster_inference.model_management as management

    originals = management.loaded_models, management.load_models_gpu, preview.decode_x0_to_image

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "sampling_preview_fixture.py").write_text(
            "from test_triposplat_torch import SamplingPreviewSource, SamplingPreviewInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (SamplingPreviewSource, SamplingPreviewInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "sampling-preview.toml"
        fixture.write_text(
            '[pack]\nname="preview-fixture"\npublisher="dinkster"\n'
            'namespaces=["triposplat-fixture"]\n[pack.entry]\n'
            'nodes="sampling_preview_fixture:NODES"\ntypes="sampling_preview_fixture:register_inference_types"\n',
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
                        worker_group=None if in_process else "preview-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("triposplat-fixture.preview-source"),
                    "preview": GraphNode(
                        "TripoSplatSamplingPreview",
                        {"model": Link("source", "model"), "vae": Link("source", "vae")},
                    ),
                    "inspect": GraphNode(
                        "triposplat-fixture.preview-inspect", {"model": Link("preview", "model")}
                    ),
                }
            )
            events: list[Any] = []
            for _ in range(2):
                events.clear()
                engine = composer.composition.make_engine(events.append)
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "patched model and sampler previews transported"
                )
                previews = [
                    event
                    for event in events
                    if event.kind == "node_event" and event.detail["name"] == "preview"
                ]
                assert len(previews) == 2
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        management.loaded_models, management.load_models_gpu, preview.decode_x0_to_image = originals
