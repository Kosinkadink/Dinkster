"""CPU contracts for native PiD conditioning."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_inference.latent_formats import SD3, SDXL, Flux, Flux2, Wan21
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.pid_conditioning import PiDConditioning

from dinkster.compose import PackSpec, ServingComposer

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)


def _positive() -> list[list[object]]:
    return [[torch.arange(6, dtype=torch.float32).reshape(1, 2, 3), {"preserved": "metadata"}]]


def _latent(channels: int, *, frames: int | None = None) -> dict[str, object]:
    shape = (1, channels, 3, 5) if frames is None else (1, channels, frames, 3, 5)
    return {
        "samples": torch.arange(torch.tensor(shape).prod(), dtype=torch.float64).reshape(shape)
        / 10.0,
        "unrelated": {"asymmetric": (3, 5)},
    }


def _expected_lq_latent(format_class: type[Any], samples: torch.Tensor) -> torch.Tensor:
    result = format_class().process_in(samples)
    return result[:, :, 0] if result.ndim == 5 else result


@pytest.mark.parametrize(
    ("latent_format", "channels", "format_class"),
    (
        ("flux", 16, Flux),
        ("flux", 128, Flux2),
        ("sd3", 16, SD3),
        ("sdxl", 4, SDXL),
        ("qwenimage", 16, Wan21),
    ),
)
def test_pid_conditioning_selects_the_reference_latent_format_and_preserves_inputs(
    latent_format: str, channels: int, format_class: type[Any]
) -> None:
    latent = _latent(channels)
    positive = _positive()
    original_samples = latent["samples"].clone()
    output = PiDConditioning.execute(
        positive=positive,
        latent=latent,
        latent_format=latent_format,
        degrade_sigma=0.37,
    )["CONDITIONING"]
    metadata = output[0][1]

    assert PiDConditioning in COMFY_NODES
    assert output[0][0] is positive[0][0]
    assert metadata is not positive[0][1]
    assert metadata["preserved"] == "metadata"
    torch.testing.assert_close(
        metadata["lq_latent"], _expected_lq_latent(format_class, original_samples)
    )
    assert metadata["degrade_sigma"].dtype == torch.float32
    assert metadata["degrade_sigma"].device.type == "cpu"
    torch.testing.assert_close(metadata["degrade_sigma"], torch.tensor([0.37], dtype=torch.float32))
    assert latent["unrelated"] == {"asymmetric": (3, 5)}
    torch.testing.assert_close(latent["samples"], original_samples)


def test_pid_conditioning_uses_the_first_video_latent_frame() -> None:
    latent = _latent(128, frames=2)
    original_samples = latent["samples"].clone()
    output = PiDConditioning.execute(
        positive=_positive(), latent=latent, latent_format="flux", degrade_sigma=0.0
    )["CONDITIONING"]

    torch.testing.assert_close(
        output[0][1]["lq_latent"], _expected_lq_latent(Flux2, original_samples)
    )
    torch.testing.assert_close(latent["samples"], original_samples)


def test_pid_conditioning_schema_matches_the_fixed_reference() -> None:
    schema = PiDConditioning.define_schema()
    assert schema.node_type == "PiDConditioning"
    assert schema.display_name == "PiD Conditioning"
    assert schema.category == "model/conditioning"
    assert [input.id for input in schema.inputs] == [
        "positive",
        "latent",
        "latent_format",
        "degrade_sigma",
    ]
    assert schema.inputs[2].default == "flux"
    assert schema.inputs[2].widget.options == ("flux", "sd3", "sdxl", "qwenimage")
    assert schema.inputs[3].default == 0.0
    assert schema.inputs[3].widget.min == 0.0
    assert schema.inputs[3].widget.max == 1.0
    assert schema.inputs[3].widget.step == 0.01
    assert schema.outputs[0].id == "CONDITIONING"


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="pid-conditioning-fixture.source",
            outputs=(OutputSpec("positive", _CONDITIONING), OutputSpec("latent", _LATENT)),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(positive=_positive(), latent=_latent(128, frames=2))


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="pid-conditioning-fixture.inspect",
            inputs=(InputSpec("conditioning", _CONDITIONING),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: object) -> dict[str, object]:
        metadata = conditioning[0][1]
        lq_latent = materialize_latent(metadata["lq_latent"])
        expected = _expected_lq_latent(Flux2, _latent(128, frames=2)["samples"])
        torch.testing.assert_close(lq_latent, expected)
        assert metadata["degrade_sigma"].dtype == torch.float32
        return cls.outputs(summary="pid-conditioning")


@pytest.mark.parametrize("in_process", (True, False))
def test_pid_conditioning_executes_twice_through_both_production_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(source_root))
                ),
            }
        )
        try:
            native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "pid_conditioning_fixture.py").write_text(
                "from test_pid_conditioning_torch import Inspect, Source\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (Source, Inspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "pid-conditioning-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["pid-conditioning-fixture"]\n\n[pack.entry]\n'
                'nodes = "pid_conditioning_fixture:NODES"\n'
                'types = "pid_conditioning_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for manifest in (native_manifest, fixture_manifest):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "pid-conditioning-fixture",
                        group_manifests=() if in_process else (native_manifest, fixture_manifest),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("pid-conditioning-fixture.source"),
                    "conditioning": GraphNode(
                        "PiDConditioning",
                        {
                            "positive": Link("source", "positive"),
                            "latent": Link("source", "latent"),
                            "latent_format": "flux",
                            "degrade_sigma": 0.37,
                        },
                    ),
                    "inspect": GraphNode(
                        "pid-conditioning-fixture.inspect",
                        {"conditioning": Link("conditioning", "CONDITIONING")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "pid-conditioning"
        finally:
            await composer.close()

    asyncio.run(scenario())
