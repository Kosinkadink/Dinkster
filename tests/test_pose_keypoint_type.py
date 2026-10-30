"""Public pose frames retain source structure through ordinary pack hosts."""

from __future__ import annotations

import asyncio
import copy
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    POSE_KEYPOINT_TYPE,
    POSE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import LATENT_TYPE, encode_latent
from dinkster_workers import load_manifest

from dinkster.compose import PackSpec, ServingComposer

PERSON = {
    field: [number + offset / 8 for number in range(count * 3)]
    for field, offset, count in (
        ("pose_keypoints_2d", -13, 18),
        ("foot_keypoints_2d", 7, 6),
        ("face_keypoints_2d", 29, 70),
        ("hand_right_keypoints_2d", -41, 21),
        ("hand_left_keypoints_2d", 53, 21),
    )
}
FRAMES = [
    {
        "canvas_width": 641,
        "canvas_height": 479,
        "people": [PERSON, {"pose_keypoints_2d": [-5, 12.5, 2.25], "foot_keypoints_2d": []}],
        "metadata": {"label": "first", "index": 17},
    },
    {"canvas_width": 173, "canvas_height": 251, "people": []},
    {
        "canvas_width": 89,
        "canvas_height": 47,
        "people": [{"face_keypoints_2d": [], "foot_keypoints_2d": None}],
    },
]


@pytest.mark.parametrize("value", (FRAMES, []))
def test_public_pose_keypoint_roundtrip_and_identity(value: list[dict[str, object]]) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(POSE_KEYPOINT_TYPE)
    register_inference_types(registry)
    assert registry.spec(POSE_KEYPOINT_TYPE) is spec
    assert POSE_KEYPOINT_TYPE != POSE_TYPE
    assert spec.validate_encoded is not None
    before = copy.deepcopy(value)
    encoded = spec.encode(value)
    spec.validate_encoded(encoded, {})
    assert spec.decode(encoded) == before
    assert value == before
    assert spec.fingerprint is not None
    assert spec.fingerprint(spec.decode(encoded)) == spec.fingerprint(value)
    assert spec.fingerprint([dict(reversed(tuple(frame.items()))) for frame in value]) == (
        spec.fingerprint(value)
    )
    latent_fingerprint = registry.spec(LATENT_TYPE).fingerprint
    assert latent_fingerprint is not None
    assert spec.fingerprint(value) != latent_fingerprint({"samples": value})
    if value:
        changed = copy.deepcopy(value)
        changed[0]["canvas_width"] = 479
        assert spec.fingerprint(value) != spec.fingerprint(changed)
        assert spec.fingerprint(value) != spec.fingerprint(list(reversed(value)))


@pytest.mark.parametrize(
    "value",
    (
        {},
        [None],
        [{"canvas_width": True, "canvas_height": 3, "people": []}],
        [{"canvas_width": 7, "canvas_height": 3, "people": {}}],
        [{"canvas_width": 7, "canvas_height": 3, "people": [None]}],
        [{"canvas_width": 7, "canvas_height": 3, "people": [{"pose_keypoints_2d": [1, 2]}]}],
        [{"canvas_width": 7, "canvas_height": 3, "people": [{"face_keypoints_2d": [1, 2, True]}]}],
    ),
)
def test_public_pose_keypoint_rejects_malformed_structure(value: object) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(POSE_KEYPOINT_TYPE)
    assert spec.validate_encoded is not None
    malformed = encode_latent({"samples": value})
    with pytest.raises(ValueError):
        spec.encode(value)
    with pytest.raises(ValueError):
        spec.decode(malformed)
    with pytest.raises(ValueError):
        spec.validate_encoded(malformed, {})


def test_public_pose_registration_preserves_first_authority_and_existing_pose() -> None:
    registry = TypeRegistry()
    pose = registry.register(POSE_TYPE)
    keypoints = registry.register(POSE_KEYPOINT_TYPE)
    from dinkster_nodes_comfy import register_inference_types as register_native_types

    register_native_types(registry)
    register_native_types(registry)
    assert registry.spec(POSE_KEYPOINT_TYPE) is keypoints
    assert registry.spec(POSE_TYPE) is pose
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert manifest.types_entry == "dinkster_nodes_comfy:register_inference_types"


def test_public_pose_keypoint_registration_and_codec_are_torch_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from dinkster_api.v1 import POSE_KEYPOINT_TYPE, TypeRegistry\n"
            "from dinkster_values.builtin_types import POSE_KEYPOINT_TYPE as canonical\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "assert POSE_KEYPOINT_TYPE is canonical\n"
            "registry = TypeRegistry()\n"
            "register_inference_types(registry)\n"
            "value = [{'canvas_width': 7, 'canvas_height': 13, 'people': []}]\n"
            "spec = registry.spec(canonical)\n"
            "data = spec.encode(value)\n"
            "spec.validate_encoded(data, {})\n"
            "assert spec.decode(data) == value\n"
            "assert not any(name == 'torch' or name.startswith('torch.')\n"
            "    or name == 'dinkster_inference' or name.startswith('dinkster_inference.')\n"
            "    for name in sys.modules)\n",
        ],
        check=True,
    )


class PoseSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="pose-keypoint-source.emit",
            outputs=(OutputSpec("frames", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(frames=FRAMES)


class PoseObserver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="pose-keypoint-observer.inspect",
            inputs=(InputSpec("frames", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),),
            outputs=(OutputSpec("frames", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, frames: object) -> Mapping[str, object]:
        assert frames == FRAMES
        return cls.outputs(frames=frames)


@pytest.mark.parametrize("in_process", (True, False))
def test_pose_keypoints_cross_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        checkout = Path(__file__).resolve().parents[1]
        native = checkout / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        manifests = []
        for name, node in (("source", "PoseSource"), ("observer", "PoseObserver")):
            (tmp_path / f"pose_{name}.py").write_text(
                f"from test_pose_keypoint_type import {node}\n"
                "from dinkster_api.v1 import register_inference_types\n"
                f"NODES = ({node},)\n",
                encoding="ascii",
            )
            manifest = tmp_path / f"{name}-pack.toml"
            manifest.write_text(
                f'[pack]\nname="pose-keypoint-{name}"\npublisher="dinkster"\n'
                f'namespaces=["pose-keypoint-{name}"]\n[pack.entry]\n'
                f'nodes="pose_{name}:NODES"\ntypes="pose_{name}:register_inference_types"\n',
                encoding="ascii",
            )
            manifests.append(manifest)
        source, observer = manifests
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    [str(tmp_path), str(Path(__file__).parent), str(checkout / "src")]
                    + [str(path) for path in sorted((checkout / "packages").glob("*/src"))]
                    + sys.path
                ),
            },
        )
        try:
            for manifest in (native, source, observer):
                group = (observer,) if manifest == observer else (native, source)
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else f"pose-{group[0].stem}",
                        group_manifests=() if in_process else group,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("pose-keypoint-source.emit"),
                    "observer": GraphNode(
                        "pose-keypoint-observer.inspect", {"frames": Link("source", "frames")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["observer"])
            assert first.outputs["observer"]["frames"].resolve() == FRAMES
            assert set(first.executed) == {"source", "observer"}
            replay = await engine.run(graph, ["observer"])
            assert replay.outputs["observer"]["frames"].resolve() == FRAMES
            assert set(replay.cached) == {"source", "observer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
