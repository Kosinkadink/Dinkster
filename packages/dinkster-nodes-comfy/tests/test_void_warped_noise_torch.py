"""CPU contracts for VOID optical-flow-warped latents."""

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
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    OPTICAL_FLOW_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_api.v1 import register_inference_types as register_inference_types
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.void_warped_noise import VOIDWarpedNoise

from dinkster.compose import PackSpec, ServingComposer


class _AsymmetricRaft(torch.nn.Module):
    def forward(self, first: torch.Tensor, second: torch.Tensor) -> list[torch.Tensor]:
        _, _, height, width = first.shape
        flow = torch.empty((1, 2, height, width), dtype=first.dtype, device=first.device)
        flow[:, 0].fill_(0.75)
        flow[:, 1].fill_(-0.5)
        return [flow]


def _patcher() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import CoreModelPatcher

    cpu = torch.device("cpu")
    return CoreModelPatcher(_AsymmetricRaft(), cpu, cpu)


def _video(frames: int, height: int = 32, width: int = 48) -> torch.Tensor:
    values = torch.arange(frames * height * width * 3, dtype=torch.float32)
    return (values.reshape(frames, height, width, 3) % 251) / 250


def _python_path(root: Path) -> str:
    return os.pathsep.join(
        (str(root), str(root / "src"), *(str(path) for path in sorted(root.glob("packages/*/src"))))
    )


def test_schema_is_canonical_and_cold_import_avoids_tensor_runtime(tmp_path: Path) -> None:
    assert VOIDWarpedNoise in COMFY_NODES
    schema = VOIDWarpedNoise.schema()
    assert schema.node_type == "VOIDWarpedNoise"
    assert schema.display_name == ""
    assert schema.category == "model/latent/void"
    assert [input.id for input in schema.inputs] == [
        "optical_flow",
        "video",
        "width",
        "height",
        "length",
        "batch_size",
    ]
    assert [input.type.types for input in schema.inputs[:2]] == [
        (OPTICAL_FLOW_TYPE,),
        (IMAGE_TYPE,),
    ]
    assert [input.default for input in schema.inputs[2:]] == [672, 384, 45, 1]
    assert [input.widget.min for input in schema.inputs[2:]] == [16, 16, 1, 1]
    assert [input.widget.max for input in schema.inputs[2:]] == [16384, 16384, 16384, 64]
    assert [input.widget.step for input in schema.inputs[2:]] == [8, 8, 1, None]
    assert [(output.id, output.type.types, output.display_name) for output in schema.outputs] == [
        ("LATENT", (LATENT_TYPE,), "warped_noise")
    ]

    root = Path(__file__).resolve().parents[3]
    discovery = "\n".join(
        (
            "import json",
            "import sys",
            "import dinkster_nodes_comfy.void_warped_noise as module",
            "module.VOIDWarpedNoise.schema()",
            "forbidden = sorted(name for name in sys.modules if (",
            "    name == 'torch' or name.startswith('torch.')",
            "    or name == 'torchvision' or name.startswith('torchvision.')",
            "    or name == 'dinkster_inference' or name.startswith('dinkster_inference.')",
            "))",
            "print(json.dumps({'path': module.__file__, 'forbidden': forbidden}))",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", discovery],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": _python_path(root)},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["path"]).resolve() == (
        root / "packages/dinkster-nodes-comfy/src/dinkster_nodes_comfy/void_warped_noise.py"
    )
    assert payload["forbidden"] == []


@pytest.mark.parametrize(
    ("requested_length", "expected_time"),
    ((49, 12), (1, 2)),
)
def test_node_uses_real_warp_math_and_corrects_even_latent_time(
    requested_length: int, expected_time: int
) -> None:
    video = _video(49)
    before = video.clone()
    torch.manual_seed(918)
    result = VOIDWarpedNoise.execute(
        optical_flow=_patcher(),
        video=video,
        width=48,
        height=32,
        length=requested_length,
        batch_size=2,
    )["LATENT"]
    samples = result["samples"]
    assert samples.shape == (2, 16, expected_time, 4, 6)
    assert samples.dtype is torch.float32 and samples.device.type == "cpu"
    assert torch.isfinite(samples).all()
    assert torch.equal(samples[0], samples[1])
    torch.testing.assert_close(video, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("length", "frame_count", "time_indices"),
    ((49, 45, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4]), (1, 5, [0, 4])),
)
def test_node_quantizes_video_and_resamples_controlled_warped_noise(
    monkeypatch: pytest.MonkeyPatch,
    length: int,
    frame_count: int,
    time_indices: list[int],
) -> None:
    initialize_inference_process()
    video = _video(49, height=32, width=48) * 1.2 - 0.1
    before = video.clone()
    captured: dict[str, torch.Tensor] = {}
    warped = torch.arange(5 * 2 * 3 * 16, dtype=torch.float32).reshape(5, 2, 3, 16) / 13

    def controlled_noise(frames: torch.Tensor, *_: object, **__: object) -> torch.Tensor:
        captured["frames"] = frames.clone()
        return warped

    monkeypatch.setattr(
        "dinkster_inference.model_management.get_torch_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr("dinkster_inference.model_management.load_model_gpu", lambda _: None)
    monkeypatch.setattr(
        "dinkster_inference.model_management.intermediate_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        "dinkster_nodes_comfy._void_noise_warp.get_noise_from_video", controlled_noise
    )

    samples = VOIDWarpedNoise.execute(
        optical_flow=_patcher(), video=video, width=48, height=32, length=length, batch_size=3
    )["LATENT"]["samples"]

    expected_frames = (video[:frame_count].clamp(0, 1) * 255).to(torch.uint8)
    torch.testing.assert_close(captured["frames"], expected_frames, rtol=0, atol=0)
    expected = torch.nn.functional.interpolate(
        warped[time_indices].permute(0, 3, 1, 2), size=(4, 6), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)
    expected = expected.permute(3, 0, 1, 2).unsqueeze(0).repeat(3, 1, 1, 1, 1)
    torch.testing.assert_close(samples, expected, rtol=0, atol=0)
    assert samples.shape == (3, 16, len(time_indices), 4, 6)
    torch.testing.assert_close(video, before, rtol=0, atol=0)


class _FixtureSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="void-warped-noise-fixture.source",
            outputs=(
                OutputSpec("flow", TypeExpr.concrete(OPTICAL_FLOW_TYPE)),
                OutputSpec("video", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        import dinkster_nodes_comfy.void_warped_noise as module

        assert Path(module.__file__).resolve() == (
            Path(__file__).resolve().parents[1] / "src/dinkster_nodes_comfy/void_warped_noise.py"
        )
        return cls.outputs(flow=_patcher(), video=_video(5))


class _FixtureConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="void-warped-noise-fixture.consumer",
            inputs=(InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: Any) -> dict[str, object]:
        samples = materialize_latent(latent)["samples"]
        assert samples.shape == (2, 16, 2, 4, 6)
        assert samples.dtype is torch.float32 and samples.device.type == "cpu"
        assert torch.isfinite(samples).all() and torch.equal(samples[0], samples[1])
        return cls.outputs(summary="cpu-flow-latent")


FIXTURE_NODES = (_FixtureSource, _FixtureConsumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_node_crosses_ordinary_hosts_twice_with_native_manifest(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "void_warped_noise_fixture.py").write_text(
            "from test_void_warped_noise_torch import FIXTURE_NODES, register_inference_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "void-warped-noise-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["void-warped-noise-fixture"]\n[pack.entry]\n'
            'nodes = "void_warped_noise_fixture:FIXTURE_NODES"\n'
            'types = "void_warped_noise_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), _python_path(package.parent.parent))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest,
                        worker_group="void-warped-noise-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("void-warped-noise-fixture.source"),
                    "noise": GraphNode(
                        "VOIDWarpedNoise",
                        {
                            "optical_flow": Link("source", "flow"),
                            "video": Link("source", "video"),
                            "width": 48,
                            "height": 32,
                            "length": 1,
                            "batch_size": 2,
                        },
                    ),
                    "consumer": GraphNode(
                        "void-warped-noise-fixture.consumer", {"latent": Link("noise", "LATENT")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for run in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "cpu-flow-latent"
                if run == 1:
                    assert set(result.cached) == {"source", "noise", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
