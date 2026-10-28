"""SeedVR2 geometry, color correction and ordinary worker execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.seedvr_postprocess import SeedVR2PostProcessing

from dinkster.compose import PackSpec, ServingComposer


def pixels() -> torch.Tensor:
    return torch.linspace(-0.2, 1.3, 6 * 5 * 7 * 3).reshape(6, 5, 7, 3)


def original() -> torch.Tensor:
    rgb = torch.linspace(0.1, 0.9, 2 * 2 * 3 * 5 * 3).reshape(2, 2, 3, 5, 3)
    alpha = torch.linspace(0.2, 0.8, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5, 1)
    return torch.cat((rgb, alpha), dim=-1)


def expected_none() -> torch.Tensor:
    source, reference = pixels(), original()
    return torch.stack(
        [
            torch.cat((source[b * 3 + t, :2, :4], reference[b, t, :2, :4, 3:4]), dim=-1)
            for b in range(2)
            for t in range(2)
        ]
    )


@pytest.mark.parametrize("five_dimensional", (False, True))
def test_geometry_restores_batch_time_and_alpha_without_mutating_sources(
    five_dimensional: bool,
) -> None:
    source = pixels().reshape(1, 6, 5, 7, 3) if five_dimensional else pixels()
    reference = original()
    before, before_ref = source.clone(), reference.clone()
    output = SeedVR2PostProcessing.execute(
        images=source, original_resized_images=reference, color_correction_method="none"
    )["images"]
    expected = expected_none().reshape(2, 2, 2, 4, 4) if five_dimensional else expected_none()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.dtype == source.dtype and output.device == source.device
    assert torch.equal(source, before) and torch.equal(reference, before_ref)


@pytest.mark.parametrize("method", ("lab", "wavelet", "adain"))
def test_color_transfer_matches_constant_reference_and_preserves_sources(method: str) -> None:
    source = torch.tensor([0.15, 0.35, 0.7]).expand(2, 16, 18, 3).clone()
    reference = torch.tensor([0.8, 0.2, 0.4]).expand(2, 16, 18, 3).clone()
    before, before_ref = source.clone(), reference.clone()
    output = SeedVR2PostProcessing.execute(
        images=source, original_resized_images=reference, color_correction_method=method
    )["images"]
    torch.testing.assert_close(output, reference, rtol=0, atol=2e-5)
    assert torch.equal(source, before) and torch.equal(reference, before_ref)


def test_color_transfer_resizes_the_whole_reference_before_even_crop() -> None:
    source = torch.tensor([0.15, 0.35, 0.7]).expand(2, 5, 7, 3).clone()
    ramp = torch.linspace(-0.2, 0.2, 9 * 11).reshape(1, 9, 11, 1)
    reference = (ramp + torch.tensor([0.45, 0.25, 0.55])).expand(2, 9, 11, 3).clone()
    output = SeedVR2PostProcessing.execute(
        images=source, original_resized_images=reference, color_correction_method="adain"
    )["images"]
    torch.testing.assert_close(
        output, torch.tensor([0.45, 0.25, 0.55]).expand(2, 4, 6, 3), rtol=0, atol=2e-5
    )


def test_adain_retries_smaller_chunks_and_rejects_non_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_inference import model_management
    from dinkster_inference.ldm.seedvr import color_fix

    transfer = color_fix.adain_color_transfer

    def limited(decoded: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if decoded.shape[0] > 1:
            raise torch.OutOfMemoryError("controlled CPU allocation failure")
        return transfer(decoded, reference)

    monkeypatch.setattr(color_fix, "adain_color_transfer", limited)
    source = torch.tensor([0.15, 0.35, 0.7]).expand(5, 4, 6, 3).clone()
    reference = torch.tensor([0.8, 0.2, 0.4]).expand_as(source).clone()
    output = SeedVR2PostProcessing.execute(
        images=source, original_resized_images=reference, color_correction_method="adain"
    )["images"]
    torch.testing.assert_close(output, reference, rtol=0, atol=2e-5)

    def bad(*args: Any) -> Any:
        raise ValueError("not an allocation failure")

    monkeypatch.setattr(color_fix, "adain_color_transfer", bad)
    with pytest.raises(ValueError, match="not an allocation failure"):
        SeedVR2PostProcessing.execute(
            images=source, original_resized_images=reference, color_correction_method="adain"
        )
    assert model_management.vae_device().type == "cpu"


class PostprocessSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="seedvr-postprocess-fixture.source",
            outputs=(
                OutputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("reference", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("color", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("color_reference", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            images=pixels(),
            reference=original(),
            color=torch.tensor([0.15, 0.35, 0.7]).expand(2, 4, 6, 3).clone(),
            color_reference=torch.tensor([0.8, 0.2, 0.4]).expand(2, 4, 6, 3).clone(),
        )


class PostprocessConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="seedvr-postprocess-fixture.consumer",
            inputs=(
                InputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("color", TypeExpr.concrete(IMAGE_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, images: object, color: object) -> dict[str, object]:
        torch.testing.assert_close(materialize_image(images), expected_none(), rtol=0, atol=0)
        torch.testing.assert_close(
            materialize_image(color),
            torch.tensor([0.8, 0.2, 0.4]).expand(2, 4, 6, 3),
            rtol=0,
            atol=2e-5,
        )
        return cls.outputs(summary="seedvr-postprocess")


@pytest.mark.parametrize("in_process", (True,))
def test_postprocess_runs_through_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        module = "seedvr_postprocess_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from test_seedvr_postprocess_torch import PostprocessSource, PostprocessConsumer\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES=(PostprocessSource,PostprocessConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="seedvr-postprocess-fixture"\npublisher="dinkster"\n'
            'namespaces=["seedvr-postprocess-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "seedvr-postprocess-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("seedvr-postprocess-fixture.source", {}),
                    "post": GraphNode(
                        "SeedVR2PostProcessing",
                        {
                            "images": Link("source", "images"),
                            "original_resized_images": Link("source", "reference"),
                            "color_correction_method": "none",
                        },
                    ),
                    "color": GraphNode(
                        "SeedVR2PostProcessing",
                        {
                            "images": Link("source", "color"),
                            "original_resized_images": Link("source", "color_reference"),
                            "color_correction_method": "adain",
                        },
                    ),
                    "consumer": GraphNode(
                        "seedvr-postprocess-fixture.consumer",
                        {"images": Link("post", "images"), "color": Link("color", "images")},
                    ),
                }
            )
            for _ in range(2):
                engine = composer.composition.make_engine(lambda _event: None)
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "seedvr-postprocess"
        finally:
            await composer.close()

    asyncio.run(scenario())
