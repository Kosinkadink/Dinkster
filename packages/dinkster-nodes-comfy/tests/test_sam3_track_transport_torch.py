"""CPU coverage for portable SAM3 track data through ordinary pack hosts."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    SAM3_TRACK_DATA_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    materialize_latent,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer


def _track(case: str) -> dict[str, object]:
    if case == "asymmetric-present-scores":
        return {
            "packed_masks": torch.arange(30, dtype=torch.uint8).reshape(2, 3, 5),
            "orig_size": [13, 29],
            "n_frames": 7,
            "scores": [0.25, (0.5, 0.75)],
            "metadata": {"labels": ("left", "right"), "indices": [3, 1]},
        }
    if case == "zero-objects":
        return {
            "packed_masks": torch.empty((0, 0, 5), dtype=torch.uint8),
            "orig_size": (3, 5),
            "n_frames": 0,
            "scores": (),
            "metadata": {"empty": [(), []]},
        }
    if case == "tensor-scores":
        return {
            "packed_masks": torch.tensor([[[7, 3, 11]]], dtype=torch.uint8),
            "orig_size": [5, 3],
            "n_frames": 1,
            "scores": torch.tensor([0.125, 0.875], dtype=torch.float32),
            "metadata": {"score_names": ("first", "second")},
        }
    if case == "none-scores":
        return {
            "packed_masks": None,
            "orig_size": (17, 11),
            "n_frames": 2,
            "scores": None,
            "metadata": {"source": ["prompt", ("point", 4)]},
        }
    if case == "none-absent-scores":
        return {
            "packed_masks": None,
            "orig_size": (17, 11),
            "n_frames": 2,
            "metadata": {"source": ["prompt", ("point", 4)]},
        }
    raise ValueError(f"unknown SAM3 track case: {case}")


def _track_summary(value: dict[str, object]) -> str:
    packed = value["packed_masks"]
    if packed is None:
        packed_summary = "none"
    else:
        tensor = packed
        assert isinstance(tensor, torch.Tensor)
        packed_summary = ";".join(
            (
                str(tensor.dtype),
                str(tuple(tensor.shape)),
                hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
            )
        )
    return repr(
        (
            packed_summary,
            value["orig_size"],
            value["n_frames"],
            value.get("scores", "absent"),
            tuple(sorted(value["metadata"].items())),
        )
    )


def _assert_structure(actual: object, expected: object) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_structure(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple))
        assert len(actual) == len(expected)
        for received, wanted in zip(actual, expected, strict=True):
            _assert_structure(received, wanted)
    else:
        assert actual == expected


def _assert_track(value: dict[str, object], case: str) -> None:
    expected = _track(case)
    assert set(value) == set(expected)
    _assert_structure(value["orig_size"], expected["orig_size"])
    assert value["n_frames"] == expected["n_frames"]
    assert type(value["n_frames"]) is int
    _assert_structure(value.get("scores", "absent"), expected.get("scores", "absent"))
    if "scores" in expected:
        assert type(value["scores"]) is type(expected["scores"])
    _assert_structure(value["metadata"], expected["metadata"])
    packed = value["packed_masks"]
    expected_packed = expected["packed_masks"]
    if expected_packed is None:
        assert packed is None
        return
    assert isinstance(packed, torch.Tensor)
    assert isinstance(expected_packed, torch.Tensor)
    assert packed.dtype == expected_packed.dtype
    assert packed.shape == expected_packed.shape
    assert torch.equal(packed, expected_packed)


@pytest.mark.parametrize(
    "case",
    (
        "asymmetric-present-scores",
        "zero-objects",
        "tensor-scores",
        "none-scores",
        "none-absent-scores",
    ),
)
def test_public_sam3_track_codec_materializes_torch_data_without_mutating_source(case: str) -> None:
    source = _track(case)
    packed = source["packed_masks"]
    before = packed.clone() if isinstance(packed, torch.Tensor) else None
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SAM3_TRACK_DATA_TYPE)

    restored = materialize_latent(spec.decode(spec.encode(source)))

    assert isinstance(restored, dict)
    _assert_track(restored, case)
    _assert_track(source, case)
    if before is not None:
        assert isinstance(packed, torch.Tensor)
        assert torch.equal(packed, before)


SAM3_TRACK = TypeExpr.concrete(SAM3_TRACK_DATA_TYPE)
STRING = TypeExpr.concrete(CORE_STRING)


class SAM3TrackSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3-public-source.source",
            inputs=(InputSpec("case", STRING),),
            outputs=(OutputSpec("track", SAM3_TRACK),),
        )

    @classmethod
    def execute(cls, *, case: str) -> dict[str, object]:
        return cls.outputs(track=_track(case))


class SAM3TrackConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3-public-consumer.consumer",
            inputs=(InputSpec("track", SAM3_TRACK), InputSpec("case", STRING)),
            outputs=(OutputSpec("summary", STRING),),
        )

    @classmethod
    def execute(cls, *, track: Any, case: str) -> dict[str, object]:
        restored = materialize_latent(track)
        assert isinstance(restored, dict)
        _assert_track(restored, case)
        return cls.outputs(summary=_track_summary(restored))


@pytest.mark.parametrize(
    "in_process,case",
    (
        (True, "asymmetric-present-scores"),
        (False, "asymmetric-present-scores"),
        (True, "zero-objects"),
        (True, "tensor-scores"),
        (True, "none-scores"),
        (True, "none-absent-scores"),
    ),
)
def test_public_sam3_track_producer_consumer_repeats_through_ordinary_hosts(
    tmp_path: Path, case: str, in_process: bool
) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        module = "sam3_public_fixture_" + case.replace("-", "_") + str(in_process)
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_sam3_track_transport_torch import SAM3TrackConsumer, SAM3TrackSource\n"
            "SOURCE_NODES = (SAM3TrackSource,)\n"
            "CONSUMER_NODES = (SAM3TrackConsumer,)\n",
            encoding="ascii",
        )
        source_manifest = tmp_path / "source-dinkster-pack.toml"
        consumer_manifest = tmp_path / "consumer-dinkster-pack.toml"
        for manifest, name, namespace, nodes in (
            (source_manifest, "sam3-public-source", "sam3-public-source", "SOURCE_NODES"),
            (consumer_manifest, "sam3-public-consumer", "sam3-public-consumer", "CONSUMER_NODES"),
        ):
            manifest.write_text(
                "[pack]\n"
                f'name = "{name}"\n'
                'publisher = "dinkster"\n'
                f'namespaces = ["{namespace}"]\n'
                "\n[pack.entry]\n"
                f'nodes = "{module}:{nodes}"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
        native_manifest = root / "dinkster-pack.toml"
        manifests = (native_manifest, source_manifest, consumer_manifest)
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            specs = (
                tuple(PackSpec(manifest=manifest, in_process=True) for manifest in manifests)
                if in_process
                else tuple(
                    PackSpec(
                        manifest=manifest,
                        worker_group="sam3-public-fixture",
                        group_manifests=manifests,
                    )
                    for manifest in manifests
                )
            )
            for spec in specs:
                await composer.add_pack(spec)
            graph = Graph(
                nodes={
                    "source": GraphNode("sam3-public-source.source", {"case": case}),
                    "consumer": GraphNode(
                        "sam3-public-consumer.consumer",
                        {"track": Link("source", "track"), "case": case},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            expected = _track_summary(_track(case))
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == expected
                assert set(result.cached) == ({"source", "consumer"} if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
