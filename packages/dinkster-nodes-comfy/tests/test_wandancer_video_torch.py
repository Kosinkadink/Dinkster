"""CPU contracts and host transport for native WanDancer video conditioning."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)
from dinkster_api.v1 import register_inference_types as register_inference_types
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.wandancer_video import WanDancerVideo

from dinkster.compose import PackSpec, ServingComposer


class TinyVAE:
    latent_channels = 16

    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return image[::4, ::8, ::8, 0].unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def _conditioning(label: str) -> list[list[Any]]:
    return [[torch.tensor([[1.0]], dtype=torch.float64), {"label": label}]]


def _image(values: tuple[float, ...], channels: int = 5) -> torch.Tensor:
    frames = []
    for value in values:
        frame = torch.arange(6 * 10 * channels, dtype=torch.float64).reshape(6, 10, channels)
        frames.append(frame / 100 + value)
    return torch.stack(frames)


def _vision(values: tuple[float, ...]) -> dict[str, torch.Tensor]:
    return {
        "penultimate_hidden_states": torch.tensor(values, dtype=torch.float64).reshape(1, -1, 1),
    }


def _expected_image(source: torch.Tensor, *, length: int, width: int, height: int) -> torch.Tensor:
    resized = torch.nn.functional.interpolate(
        source[:length].movedim(-1, 1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)
    result = torch.zeros((length, height, width, source.shape[-1]), dtype=source.dtype)
    result[: resized.shape[0]] = resized
    return result[:, :, :, :3]


def test_schema_and_cold_source_tree_discovery() -> None:
    schema = WanDancerVideo.define_schema()
    assert schema.node_type == "WanDancerVideo"
    assert schema.category == "model/conditioning/wan/dancer"
    assert [input.id for input in schema.inputs] == [
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "clip_vision_output",
        "clip_vision_output_ref",
        "start_image",
        "mask",
        "audio_encoder_output",
    ]
    assert [input.type.types for input in schema.inputs] == [
        (CONDITIONING_TYPE,),
        (CONDITIONING_TYPE,),
        (VAE_TYPE,),
        (CORE_INT,),
        (CORE_INT,),
        (CORE_INT,),
        (CLIP_VISION_OUTPUT_TYPE,),
        (CLIP_VISION_OUTPUT_TYPE,),
        (IMAGE_TYPE,),
        (MASK_TYPE,),
        (AUDIO_ENCODER_OUTPUT_TYPE,),
    ]
    assert [input.default for input in schema.inputs[3:6]] == [480, 832, 149]
    assert [input.widget for input in schema.inputs[3:6]] == [
        NumberWidget(min=16, max=16384, step=16),
        NumberWidget(min=16, max=16384, step=16),
        NumberWidget(min=1, max=16384, step=4),
    ]
    assert all(not input.required for input in schema.inputs[6:])
    assert (
        schema.inputs[5].doc
        == "The number of frames in the generated video. Should stay 149 for WanDancer."
    )
    assert schema.inputs[6].doc == "The CLIP vision embeds for the first frame."
    assert schema.inputs[7].doc == "The CLIP vision embeds for the reference image."
    assert (
        schema.inputs[8].doc == "The initial image(s) to be encoded, can be any number of frames."
    )
    assert schema.inputs[9].doc == (
        "Image conditioning mask for the start image(s). White is kept, black is generated. "
        "Used for the local generations."
    )
    assert [
        (output.id, output.type.types, output.display_name, output.doc) for output in schema.outputs
    ] == [
        ("positive", (CONDITIONING_TYPE,), "positive", ""),
        ("negative", (CONDITIONING_TYPE,), "negative", ""),
        ("latent", (LATENT_TYPE,), "latent", "Empty latent."),
    ]

    root = Path(__file__).resolve().parents[3]
    python_path = os.pathsep.join(
        (str(root), str(root / "src"), *(str(path) for path in sorted(root.glob("packages/*/src"))))
    )
    script = "\n".join(
        (
            "import json",
            "import sys",
            "import dinkster_nodes_comfy.wandancer_video as module",
            "module.WanDancerVideo.define_schema()",
            "forbidden = sorted(name for name in sys.modules if (",
            "    name == 'torch' or name.startswith('torch.')",
            "    or name == 'torchvision' or name.startswith('torchvision.')",
            "    or name == 'dinkster_inference' or name.startswith('dinkster_inference.')",
            "))",
            "print(json.dumps({'path': module.__file__, 'forbidden': forbidden}))",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": python_path},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    discovery = json.loads(result.stdout)
    assert Path(discovery["path"]).resolve() == (
        root / "packages/dinkster-nodes-comfy/src/dinkster_nodes_comfy/wandancer_video.py"
    )
    assert discovery["forbidden"] == []


def test_empty_inputs_and_reference_only_preserve_conditioning() -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    reference = _vision((7.0, 8.0))
    output = WanDancerVideo.execute(
        positive=positive,
        negative=negative,
        vae=TinyVAE(),
        width=16,
        height=8,
        length=9,
        mask=torch.ones((9, 3, 5)),
        clip_vision_output_ref=reference,
    )
    assert output["positive"] is positive and output["negative"] is negative
    assert output["positive"][0][1] == {"label": "positive"}
    assert output["negative"][0][1] == {"label": "negative"}
    samples = output["latent"]["samples"]
    assert samples.shape == (1, 16, 3, 1, 2)
    assert samples.dtype is torch.float32 and samples.device.type == "cpu"
    assert not samples.count_nonzero()


@pytest.mark.parametrize(
    ("frames", "length", "width", "crop_rows", "mask_values"),
    ((6, 5, 16, slice(None), (0.0, 0.0)), (2, 9, 24, slice(1, 5), (0.0, 1.0, 1.0))),
)
def test_start_image_truncates_resizes_pads_and_drops_extra_channels(
    frames: int, length: int, width: int, crop_rows: slice, mask_values: tuple[float, ...]
) -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    source = _image(tuple(float(index + 1) for index in range(frames)))
    before = source.clone()
    vae = TinyVAE()
    output = WanDancerVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=width,
        height=8,
        length=length,
        start_image=source,
    )
    expected = _expected_image(source[:, crop_rows], length=length, width=width, height=8)
    assert vae.images[0].shape == (length, 8, width, 3)
    assert vae.images[0].dtype is torch.float64
    torch.testing.assert_close(vae.images[0], expected, rtol=0, atol=0)
    metadata = output["positive"][0][1]
    assert metadata["concat_latent_image"].shape == (1, 16, len(mask_values), 1, width // 8)
    assert metadata["concat_mask"].dtype is torch.float64
    torch.testing.assert_close(
        metadata["concat_mask"],
        torch.tensor(mask_values, dtype=torch.float64)
        .reshape(1, 1, -1, 1, 1)
        .expand(1, 1, len(mask_values), 1, width // 8),
        rtol=0,
        atol=0,
    )
    assert output["negative"][0][1]["concat_latent_image"] is metadata["concat_latent_image"]
    assert output["negative"][0][1]["concat_mask"] is metadata["concat_mask"]
    assert torch.equal(source, before)
    assert positive == _conditioning("positive") and negative == _conditioning("negative")


def test_explicit_mask_vision_audio_and_sources_are_preserved() -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    source = _image((1.0, 2.0, 3.0, 4.0, 5.0), channels=3)
    mask = torch.tensor(
        [
            [[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0]],
            [[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    first, reference = _vision((2.0, 3.0)), _vision((9.0, 10.0, 11.0))
    feature = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4).transpose(1, 2)
    audio = {"audio_feature": feature, "fps": 23.75, "audio_inject_scale": -2.5}
    source_before, mask_before, first_before = (
        source.clone(),
        mask.clone(),
        first["penultimate_hidden_states"].clone(),
    )
    reference_before, feature_before = (
        reference["penultimate_hidden_states"].clone(),
        feature.clone(),
    )
    output = WanDancerVideo.execute(
        positive=positive,
        negative=negative,
        vae=TinyVAE(),
        width=16,
        height=8,
        length=5,
        start_image=source,
        mask=mask,
        clip_vision_output=first,
        clip_vision_output_ref=reference,
        audio_encoder_output=audio,
    )
    # Swapped 2x1 nearest-exact resize selects source rows 0/2 and column 2.
    expected_mask = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    ).reshape(1, 4, 2, 1, 2)
    for conditioning, label in ((output["positive"], "positive"), (output["negative"], "negative")):
        metadata = conditioning[0][1]
        assert metadata["label"] == label
        torch.testing.assert_close(metadata["concat_mask"], expected_mask, rtol=0, atol=0)
        torch.testing.assert_close(
            metadata["clip_vision_output"].penultimate_hidden_states,
            first_before,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            metadata["clip_vision_output_ref"].penultimate_hidden_states,
            reference_before,
            rtol=0,
            atol=0,
        )
        assert metadata["audio_embed"] is feature
        assert metadata["audio_embed"].dtype is torch.float64
        assert metadata["fps"] == 23.75 and metadata["audio_inject_scale"] == -2.5
    default_audio = {"audio_feature": feature, "fps": 30.0}
    default = WanDancerVideo.execute(
        positive=_conditioning("positive"),
        negative=_conditioning("negative"),
        vae=TinyVAE(),
        width=16,
        height=8,
        length=1,
        audio_encoder_output=default_audio,
    )
    assert default["positive"][0][1]["audio_inject_scale"] == 1.0
    assert torch.equal(source, source_before) and torch.equal(mask, mask_before)
    assert torch.equal(first["penultimate_hidden_states"], first_before)
    assert torch.equal(reference["penultimate_hidden_states"], reference_before)
    assert torch.equal(feature, feature_before)
    assert positive == _conditioning("positive") and negative == _conditioning("negative")


class HostSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wandancer-video-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("vae", VAE_TYPE),
                    ("image", IMAGE_TYPE),
                    ("mask", MASK_TYPE),
                    ("vision", CLIP_VISION_OUTPUT_TYPE),
                    ("reference", CLIP_VISION_OUTPUT_TYPE),
                    ("audio", AUDIO_ENCODER_OUTPUT_TYPE),
                    ("latent", LATENT_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        image = torch.stack([torch.full((8, 16, 3), value) for value in (0.25, 0.5, 0.75, 1.0)])
        return cls.outputs(
            positive=[[torch.ones((1, 2)), {"label": "positive"}]],
            negative=[[torch.ones((1, 2)), {"label": "negative"}]],
            vae=TinyVAE(),
            image=image,
            mask=torch.tensor(
                [
                    [[0.0] * 16] * 8,
                    [[1.0] * 16] * 8,
                    [[0.0] * 16] * 8,
                    [[0.0] * 16] * 8,
                    [[1.0] * 16] * 8,
                ]
            ),
            vision=_vision((1.0, 2.0)),
            reference=_vision((3.0,)),
            audio={
                "audio_feature": torch.arange(12, dtype=torch.float64).reshape(1, 3, 4),
                "fps": 29.5,
            },
            latent={"samples": torch.zeros((1, 16, 2, 1, 2))},
        )


class HostConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wandancer-video-fixture.consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("node_latent", LATENT_TYPE),
                    ("source_image", IMAGE_TYPE),
                    ("source_mask", MASK_TYPE),
                    ("source_vision", CLIP_VISION_OUTPUT_TYPE),
                    ("source_audio", AUDIO_ENCODER_OUTPUT_TYPE),
                    ("source_latent", LATENT_TYPE),
                )
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **values: Any) -> dict[str, object]:
        image, mask = (
            materialize_image(values["source_image"]),
            materialize_image(values["source_mask"]),
        )
        assert image.shape == (4, 8, 16, 3) and mask.shape == (5, 8, 16)
        torch.testing.assert_close(
            image,
            torch.tensor([0.25, 0.5, 0.75, 1.0]).reshape(4, 1, 1, 1).expand(4, 8, 16, 3),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            mask,
            torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0]).reshape(5, 1, 1).expand(5, 8, 16),
            rtol=0,
            atol=0,
        )
        assert materialize_latent(values["source_latent"])["samples"].shape == (1, 16, 2, 1, 2)
        samples = materialize_latent(values["node_latent"])["samples"]
        assert samples.shape == (1, 16, 2, 1, 2)
        assert samples.dtype is torch.float32 and samples.device.type == "cpu"
        assert not samples.count_nonzero()
        for key, label in (("positive", "positive"), ("negative", "negative")):
            metadata = values[key][0][1]
            assert metadata["label"] == label
            torch.testing.assert_close(
                metadata["concat_latent_image"],
                torch.tensor([0.25, 0.0]).reshape(1, 1, 2, 1, 1).expand(1, 16, 2, 1, 2),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                metadata["concat_mask"],
                torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
                .reshape(1, 4, 2, 1, 1)
                .expand(1, 4, 2, 1, 2),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                metadata["clip_vision_output"].penultimate_hidden_states,
                torch.tensor([1.0, 2.0], dtype=torch.float64).reshape(1, 2, 1),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                metadata["clip_vision_output_ref"].penultimate_hidden_states,
                torch.tensor([3.0], dtype=torch.float64).reshape(1, 1, 1),
                rtol=0,
                atol=0,
            )
            assert metadata["fps"] == 29.5 and metadata["audio_inject_scale"] == 1.0
            torch.testing.assert_close(
                metadata["audio_embed"],
                torch.arange(12, dtype=torch.float64).reshape(1, 3, 4),
                rtol=0,
                atol=0,
            )
        return cls.outputs(summary="wandancer-video-host-values")


HOST_NODES = (HostSource, HostConsumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_wandancer_video_runs_twice_in_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        root = package.parents[1]
        source_paths = (
            str(root),
            str(root / "src"),
            *(str(path) for path in sorted(root.glob("packages/*/src"))),
        )
        source_module = package / "src/dinkster_nodes_comfy/wandancer_video.py"
        entry = tmp_path / "wandancer_video_host_entry.py"
        entry.write_text(
            "from pathlib import Path\n"
            "import dinkster_nodes_comfy.wandancer_video as native\n"
            f"assert Path(native.__file__).resolve() == Path({str(source_module)!r}).resolve()\n"
            "from test_wandancer_video_torch import HOST_NODES, register_inference_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="wandancer-video-fixture"\npublisher="dinkster"\n'
            'namespaces=["wandancer-video-fixture"]\n[pack.entry]\n'
            'nodes="wandancer_video_host_entry:HOST_NODES"\n'
            'types="wandancer_video_host_entry:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), *source_paths)
                ),
            }
        )
        try:
            native = package / "dinkster-pack.toml"
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "wandancer-video-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("wandancer-video-fixture.source"),
                    "dancer": GraphNode(
                        "WanDancerVideo",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "vae": Link("source", "vae"),
                            "width": 16,
                            "height": 8,
                            "length": 5,
                            "start_image": Link("source", "image"),
                            "mask": Link("source", "mask"),
                            "clip_vision_output": Link("source", "vision"),
                            "clip_vision_output_ref": Link("source", "reference"),
                            "audio_encoder_output": Link("source", "audio"),
                        },
                    ),
                    "consumer": GraphNode(
                        "wandancer-video-fixture.consumer",
                        {
                            "positive": Link("dancer", "positive"),
                            "negative": Link("dancer", "negative"),
                            "node_latent": Link("dancer", "latent"),
                            "source_image": Link("source", "image"),
                            "source_mask": Link("source", "mask"),
                            "source_vision": Link("source", "vision"),
                            "source_audio": Link("source", "audio"),
                            "source_latent": Link("source", "latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert (
                    result.outputs["consumer"]["summary"].resolve() == "wandancer-video-host-values"
                )
                assert set(result.cached) == (
                    set() if attempt == 0 else {"source", "dancer", "consumer"}
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
