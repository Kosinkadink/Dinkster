"""Latent operations preserve metadata, schedule boundaries, and frame ownership."""

import gc
import os
import sys
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_graph import Graph, GraphNode, Link
from dinkster_latent.latent_operations import (
    LatentApplyOperation,
    LatentApplyOperationCFG,
    LatentOperationBlend,
    LatentOperationSharpen,
    LatentOperationTonemapReinhard,
    ReplaceVideoLatentFrames,
)
from test_comfy_latent_nodes import ArrayTensor

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario

_STORAGE: list[weakref.ReferenceType[Any]] = []


class ResidentSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="resident-fixture.source",
            outputs=(
                OutputSpec("operation", TypeExpr.concrete("LATENT_OPERATION")),
                OutputSpec("conditioning", TypeExpr.concrete("CONDITIONING")),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        captured = np.array([3.0, -8.0])
        conditioning = np.array([2.0, 9.0])
        _STORAGE.extend((weakref.ref(captured), weakref.ref(conditioning)))

        def operation(*, latent: object) -> Any:
            return captured

        return cls.outputs(operation=operation, conditioning=[[conditioning, {}]])


class ResidentInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="resident-fixture.inspect",
            inputs=(
                InputSpec("operation", TypeExpr.concrete("LATENT_OPERATION")),
                InputSpec("conditioning", TypeExpr.concrete("CONDITIONING")),
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete("FLOAT")),),
        )

    @classmethod
    def execute(cls, *, operation: Any, conditioning: Any) -> Mapping[str, object]:
        return cls.outputs(value=float(operation(latent=None).sum() + conditioning[0][0].sum()))


class ResidentStorage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="resident-fixture.storage",
            idempotent=False,
            outputs=(OutputSpec("live", TypeExpr.concrete("INT")),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        gc.collect()
        return cls.outputs(live=sum(reference() is not None for reference in _STORAGE))


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_pack_full_release_collects_operation_and_conditioning_storage(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        (tmp_path / "resident_fixture_nodes.py").write_text(
            "from test_comfy_latent_operation_nodes import (\n"
            "ResidentSource, ResidentInspect, ResidentStorage)\n"
            "from dinkster_latent.types import register_types\n"
            "NODES = (ResidentSource, ResidentInspect, ResidentStorage)\n",
            encoding="ascii",
        )
        source_manifest = tmp_path / "dinkster-pack.toml"
        source_manifest.write_text(
            '[pack]\nname="resident-fixture"\npublisher="dinkster"\n'
            'namespaces=["resident-fixture"]\n[pack.entry]\n'
            'nodes="resident_fixture_nodes:NODES"\n'
            'types="resident_fixture_nodes:register_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), *sys.path))}
        )
        try:
            await composer.add_pack(PackSpec(source_manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("resident-fixture.source"),
                    "inspect": GraphNode(
                        "resident-fixture.inspect",
                        {
                            "operation": Link("source", "operation"),
                            "conditioning": Link("source", "conditioning"),
                        },
                    ),
                    "storage": GraphNode("resident-fixture.storage"),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["value"].resolve() == 6.0
            live = await engine.run(graph, ["storage"])
            assert cast(int, live.outputs["storage"]["live"].resolve()) >= 2
            released = await composer.full_free("ordinary-resident-release")
            assert all(
                consumer["status"] == "complete"
                for worker in released
                for consumer in cast(list[dict[str, object]], worker["consumers"])
            )
            live = await engine.run(graph, ["storage"])
            assert live.outputs["storage"]["live"].resolve() == 0
            result = await engine.run(graph, ["inspect"])
            assert result.outputs["inspect"]["value"].resolve() == 6.0
        finally:
            await composer.full_free("ordinary-resident-cleanup")
            await composer.close()

    run_scenario(scenario())


def test_latent_operation_interfaces_keep_exact_input_order_and_optional_groups() -> None:
    expected = {
        "dinkster.latent.latent_apply_operation": ("samples", "operation"),
        "dinkster.latent.latent_apply_operation_cfg": (
            "model",
            "operation",
            "start_percent",
            "end_percent",
        ),
        "dinkster.latent.latent_operation_blend": ("reference", "strength"),
        "dinkster.latent.latent_operation_tonemap_reinhard": ("multiplier",),
        "dinkster.latent.latent_operation_sharpen": ("sharpen_radius", "sigma", "alpha"),
        "dinkster.latent.replace_video_latent_frames": ("destination", "source", "index"),
    }
    assert {
        node.schema().node_type: tuple(port.id for port in node.schema().inputs)
        for node in (
            LatentApplyOperation,
            LatentApplyOperationCFG,
            LatentOperationBlend,
            LatentOperationTonemapReinhard,
            LatentOperationSharpen,
            ReplaceVideoLatentFrames,
        )
    } == expected
    assert not ReplaceVideoLatentFrames.schema().inputs[1].required
    assert ReplaceVideoLatentFrames.schema().inputs[2].required
    assert all(not port.required for port in LatentApplyOperationCFG.schema().inputs[-2:])


def test_apply_preserves_metadata_and_zero_strength_operation_keeps_input() -> None:
    samples = {"samples": np.array([2, -7]), "note": "keep"}
    result = cast(
        dict[str, Any],
        LatentApplyOperation.execute(samples=samples, operation=lambda latent: latent * 3)[
            "LATENT"
        ],
    )
    np.testing.assert_array_equal(result["samples"], [6, -21])
    np.testing.assert_array_equal(samples["samples"], [2, -7])
    assert result["note"] == "keep"
    operation = cast(
        Any, LatentOperationBlend.execute(reference=samples, strength=0)["LATENT_OPERATION"]
    )
    assert operation(latent=samples["samples"]) is samples["samples"]


class ScalarModel:
    def __init__(self) -> None:
        self.transform: Any = lambda args: args["conds_out"]

    def clone(self) -> "ScalarModel":
        return ScalarModel()

    def get_model_object(self, name: str) -> "ScalarModel":
        return self

    def percent_to_sigma(self, percent: float) -> float:
        return 10 * (1 - percent)

    def set_model_sampler_pre_cfg_function(self, transform: Any) -> None:
        self.transform = transform

    def predict(self, sigma: float, predictions: list[float]) -> list[float]:
        return self.transform({"sigma": np.array([sigma]), "conds_out": predictions.copy()})


@pytest.mark.parametrize(
    ("sigma", "expected"), [(8.01, [11, 3]), (8, [27, 3]), (4, [27, 3]), (3.99, [11, 3])]
)
def test_cfg_operation_applies_to_conditional_delta_inside_inclusive_schedule(
    sigma: float, expected: list[float]
) -> None:
    source = ScalarModel()
    model = cast(
        ScalarModel,
        LatentApplyOperationCFG.execute(
            model=source, operation=lambda latent: latent * 3, start_percent=0.2, end_percent=0.6
        )["MODEL"],
    )
    assert model.predict(sigma, [11, 3]) == expected
    assert source.predict(sigma, [11, 3]) == [11, 3]
    assert model.predict(6, [11]) == [33]


def test_video_replacement_uses_source_metadata_and_copies_destination_storage() -> None:
    destination = {"samples": ArrayTensor(np.arange(7).reshape(1, 1, 7, 1, 1)), "note": "dest"}
    source = {"samples": ArrayTensor(np.array([-2, -9]).reshape(1, 1, 2, 1, 1)), "note": "source"}
    result = cast(
        dict[str, Any],
        ReplaceVideoLatentFrames.execute(destination=destination, source=source, index=-2)[
            "LATENT"
        ],
    )
    np.testing.assert_array_equal(result["samples"].array.flatten(), [0, 1, 2, 3, 4, -2, -9])
    np.testing.assert_array_equal(
        cast(ArrayTensor, destination["samples"]).array.flatten(), np.arange(7)
    )
    assert result["note"] == "source"
    assert (
        ReplaceVideoLatentFrames.execute(destination=destination, source=source, index=6)["LATENT"]
        is destination
    )
    assert ReplaceVideoLatentFrames.execute(destination=destination)["LATENT"] is destination
