from __future__ import annotations

import base64
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dinkster_api.v1 import CORE_STRING, TypeExpr, TypeRegistry
from dinkster_nodes_image import (
    IMAGE_NODES,
    POSE_NODES,
    POSE_TYPE,
    ExportPoseJson,
    ImportPoseJson,
    Pose,
    PoseKeypoint,
    PoseSkeleton,
    RenderPose,
    register_image_types,
)
from dinkster_nodes_image.pose import poses_from_json, poses_to_json

GOLDEN_PATH = Path(__file__).parent / "goldens" / "pose_controlnet_aux_59b1fc4.json"
GOLDEN = cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))
GOLDEN_CASES = cast("dict[str, dict[str, object]]", GOLDEN["cases"])


def _skeleton(**overrides: object) -> PoseSkeleton:
    values: dict[str, object] = {
        "kind": "person",
        "layout": "test",
        "subject": 0,
        "normalized": True,
        "keypoints": (PoseKeypoint(0.25, 0.5, 0.75), None, PoseKeypoint(0.75, 0.5)),
        "edges": ((0, 2),),
    }
    values.update(overrides)
    return PoseSkeleton(**values)  # type: ignore[arg-type]


def _decode_output(record: object) -> np.ndarray:
    output = cast("dict[str, object]", record)
    return np.frombuffer(
        base64.b64decode(cast("str", output["uint8Base64"])), dtype=np.uint8
    ).reshape(cast("list[int]", output["shape"]))


def test_pose_value_is_frozen_validated_and_has_a_canonical_codec() -> None:
    pose = Pose(64, 48, (_skeleton(),))
    with pytest.raises(FrozenInstanceError):
        pose.canvas_width = 32  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        pose.skeletons[0].subject = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        point = cast("PoseKeypoint", pose.skeletons[0].keypoints[0])
        point.x = 0.5  # type: ignore[misc]

    registry = TypeRegistry()
    register_image_types(registry)
    register_image_types(registry)
    spec = registry.spec(POSE_TYPE)
    encoded = spec.encode(pose)
    assert encoded == spec.encode(pose)
    assert spec.decode(encoded) == pose
    assert spec.coerce is not None
    assert spec.coerce(pose.to_record()) == pose
    assert spec.meta is not None
    assert spec.meta(pose) == {
        "canvas_width": 64,
        "canvas_height": 48,
        "skeletons": 1,
        "keypoints": 3,
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PoseKeypoint(math.inf, 0),
        lambda: PoseKeypoint(32_768, 0),
        lambda: PoseKeypoint(0, 0, 0),
        lambda: PoseKeypoint(0, 0, 1.1),
        lambda: _skeleton(kind="unknown"),
        lambda: _skeleton(layout=""),
        lambda: _skeleton(subject=-1),
        lambda: _skeleton(edges=((0, 3),)),
        lambda: Pose(0, 64, ()),
    ],
)
def test_pose_value_rejects_invalid_records(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        cast("object", factory)()  # type: ignore[operator]


def test_pose_codec_rejects_malformed_payloads() -> None:
    registry = TypeRegistry()
    register_image_types(registry)
    spec = registry.spec(POSE_TYPE)
    assert spec.coerce is not None
    with pytest.raises(TypeError, match="expects an object"):
        spec.coerce([])
    with pytest.raises(ValueError, match="requires exactly"):
        spec.coerce({"canvas_width": 1, "canvas_height": 1})
    with pytest.raises(ValueError, match="invalid POSE payload"):
        spec.decode(b"not json")
    with pytest.raises(ValueError, match="invalid POSE payload"):
        spec.decode(b"[" * 10_000 + b"]" * 10_000)


def test_pose_boundary_preserves_values_and_declared_dependencies() -> None:
    from dinkster_assets import AssetRef, digest_bytes
    from dinkster_assets.value import value_asset_references
    from dinkster_values import default_encode
    from dinkster_workers.boundary import ValueCodec

    registry = TypeRegistry()
    register_image_types(registry)
    pose = Pose(
        16_384,
        1,
        (
            _skeleton(
                normalized=False,
                keypoints=(PoseKeypoint(-32_767, 12.5, 0.25), None),
                edges=((0, 1),),
            ),
        ),
    )
    wrapped = registry.wrap(POSE_TYPE, pose)
    assert value_asset_references({"pose": wrapped}, complete_registry=registry) == {}
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    reference = AssetRef(digest_bytes(b"pose-source"), "pose.json", 11)
    blobs[cast(int, wire["metaBlob"])] = default_encode(
        {**wrapped.meta.entries, "asset_refs": [reference.to_wire()]}
    )
    received, _ = codec.decode(wire, blobs, [])
    assert received.resolve() == pose
    assert received.fingerprint == wrapped.fingerprint
    assert registry.spec(POSE_TYPE).encode(received.resolve()) == (
        b'{"canvas_height":1,"canvas_width":16384,"skeletons":[{"edges":[[0,1]],'
        b'"keypoints":[{"confidence":0.25,"x":-32767.0,"y":12.5},null],'
        b'"kind":"person","layout":"test","normalized":false,"subject":0}]}'
    )
    assert value_asset_references({"pose": received}, complete_registry=registry) == {
        reference.digest: reference
    }
    payload = cast("dict[str, object]", wire["payload"])
    blobs[cast(int, payload["blob"])] = (
        b'{ "skeletons": [], "canvas_width": 1, "canvas_height": 16384 }'
    )
    empty, _ = codec.decode(wire, blobs, [])
    assert empty.resolve() == Pose(1, 16_384, ())
    first = TypeRegistry()
    first_spec = first.register(POSE_TYPE)
    register_image_types(first)
    assert first.spec(POSE_TYPE) is first_spec
    assert not first.spec(POSE_TYPE).asset_references_complete


@pytest.mark.parametrize(
    "invalid",
    [
        b"not json",
        b"[" * 10_000 + b"]" * 10_000,
        b'{"canvas_width":true,"canvas_height":1,"skeletons":[]}',
        b'{"canvas_width":16385,"canvas_height":1,"skeletons":[]}',
        b'{"canvas_width":1,"canvas_height":0,"skeletons":[]}',
        b'{"canvas_width":1,"canvas_height":1,"skeletons":[],"asset":{"digest":"hidden"}}',
        b'{"canvas_width":1,"canvas_height":1,"skeletons":[{"kind":"person",'
        b'"layout":"test","subject":0,"normalized":false,"edges":[],"keypoints":['
        b'{"x":{"digest":"hidden"},"y":0,"confidence":1}]}]}',
        b'{"canvas_width":1,"canvas_height":1,"skeletons":[{"kind":"person",'
        b'"layout":"test","subject":0,"normalized":false,"edges":[],"keypoints":['
        b'{"x":32768,"y":0,"confidence":1}]}]}',
        b'{"canvas_width":1,"canvas_height":1,"skeletons":[{"kind":"person",'
        b'"layout":"test","subject":0,"normalized":false,"edges":[],"keypoints":['
        b'{"x":0,"y":0,"confidence":0}]}]}',
        b'{"canvas_width":1,"canvas_height":1,"skeletons":[{"kind":"person",'
        b'"layout":"test","subject":0,"normalized":false,"edges":[],"keypoints":['
        b'{"x":NaN,"y":0,"confidence":1}]}]}',
    ],
    ids=[
        "json",
        "depth",
        "bool",
        "width",
        "height",
        "extra",
        "asset",
        "coordinate",
        "confidence",
        "nan",
    ],
)
def test_pose_boundary_refuses_invalid_payload_before_exposure(invalid: bytes) -> None:
    from dinkster_workers.boundary import EncodedPayloadValidationError, ValueCodec

    registry = TypeRegistry()
    register_image_types(registry)
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(registry.wrap(POSE_TYPE, Pose(1, 1, ())), blobs, segments)
    payload = cast("dict[str, object]", wire["payload"])
    blobs[cast(int, payload["blob"])] = invalid
    with pytest.raises(EncodedPayloadValidationError):
        codec.decode(wire, blobs, [])


def test_pose_json_import_preserves_human_animal_and_coordinate_metadata() -> None:
    human_source = GOLDEN_CASES["human"]["source"]
    animal_source = GOLDEN_CASES["animal"]["source"]
    human, animal = poses_from_json(json.dumps([human_source, animal_source]))

    assert (human.canvas_width, human.canvas_height) == (80, 64)
    assert [skeleton.kind for skeleton in human.skeletons] == ["person", "left_hand", "face"]
    assert all(skeleton.normalized for skeleton in human.skeletons)
    assert human.skeletons[0].layout == "openpose_body"
    assert len(human.skeletons[0].keypoints) == 18
    assert human.skeletons[0].keypoints[-1] is None

    assert (animal.canvas_width, animal.canvas_height) == (72, 64)
    assert len(animal.skeletons) == 1
    assert animal.skeletons[0].kind == "animal"
    assert animal.skeletons[0].layout == "ap10k"
    assert animal.skeletons[0].normalized is False
    assert len(animal.skeletons[0].keypoints) == 17


def test_pose_json_round_trip_preserves_frames_parts_and_feet() -> None:
    body = [[index / 20, index / 20, 1] for index in range(18)]
    feet = [[0.1 + index / 20, 0.8, 1] for index in range(6)]
    source = {
        "canvas_width": 100,
        "canvas_height": 80,
        "people": [{"pose_keypoints_2d": body, "foot_keypoints_2d": feet}],
    }
    poses = poses_from_json(json.dumps(source))
    assert poses[0].skeletons[0].layout == "openpose_body_feet"
    assert len(poses[0].skeletons[0].keypoints) == 24
    assert poses_from_json(poses_to_json(poses)) == poses
    assert ExportPoseJson.execute(poses=poses) == {"json": poses_to_json(poses)}
    assert ImportPoseJson.execute(json=json.dumps(source)) == {"poses": poses}

    animal = poses_from_json(json.dumps(GOLDEN_CASES["animal"]["source"]))
    animal_record = json.loads(poses_to_json(animal))
    assert animal_record["version"] == "ap10k"
    assert poses_from_json(json.dumps(animal_record)) == animal


def test_pose_json_export_fails_instead_of_losing_internal_metadata() -> None:
    human = poses_from_json(json.dumps(GOLDEN_CASES["human"]["source"]))[0]
    body = human.skeletons[0]
    cases = (
        Pose(human.canvas_width, human.canvas_height, (replace(body, layout="custom"),)),
        Pose(human.canvas_width, human.canvas_height, (replace(body, subject=2),)),
        Pose(human.canvas_width, human.canvas_height, (replace(body, normalized=False),)),
        Pose(human.canvas_width, human.canvas_height, (replace(body, edges=((0, 1),)),)),
    )
    for pose in cases:
        with pytest.raises(ValueError, match="cannot be represented losslessly"):
            poses_to_json([pose])


@pytest.mark.parametrize("name", ["human", "animal"])
def test_pose_rendering_matches_pinned_controlnet_aux(name: str) -> None:
    case = GOLDEN_CASES[name]
    poses = poses_from_json(json.dumps(case["source"]))
    image = cast("np.ndarray", RenderPose.execute(poses=poses)["image"])
    expected = _decode_output(case["output"])
    assert image.shape == (1, *expected.shape)
    assert image.dtype == np.float32
    np.testing.assert_array_equal(np.rint(image[0] * 255.0).astype(np.uint8), expected)


def test_pose_rendering_options_and_batch_shape_validation() -> None:
    poses = poses_from_json(json.dumps(GOLDEN_CASES["human"]["source"]))
    blank = cast(
        "np.ndarray",
        RenderPose.execute(
            poses=poses,
            render_body=False,
            render_hand=False,
            render_face=False,
            render_animal=False,
        )["image"],
    )
    assert not np.any(blank)
    with pytest.raises(ValueError, match="same canvas dimensions"):
        RenderPose.execute(poses=[poses[0], Pose(1, 1, ())])
    with pytest.raises(ValueError, match="confidence_threshold"):
        RenderPose.execute(poses=poses, confidence_threshold=math.nan)
    with pytest.raises(ValueError, match="at least one frame"):
        RenderPose.execute(poses=[])


def test_pose_json_rejects_invalid_and_non_finite_input() -> None:
    with pytest.raises(ValueError, match="invalid pose JSON"):
        poses_from_json("{")
    with pytest.raises(ValueError, match="invalid pose JSON"):
        poses_from_json("[" * 10_000 + "]" * 10_000)
    with pytest.raises(ValueError, match="non-finite"):
        poses_from_json('{"canvas_width": 1, "canvas_height": NaN}')
    with pytest.raises(ValueError, match="at least one frame"):
        poses_from_json("[]")
    with pytest.raises(TypeError, match="canvas_width"):
        poses_from_json('{"canvas_width": true, "canvas_height": 64}')
    with pytest.raises(ValueError, match="confidence"):
        poses_from_json(
            '{"canvas_width": 64, "canvas_height": 64, "version": "ap10k", '
            '"animals": [[[1, 2, 2]]]}'
        )
    with pytest.raises(ValueError, match="animal version"):
        poses_from_json(
            '{"canvas_width": 64, "canvas_height": 64, "version": "other", '
            '"animals": [[[1, 2, 1]]]}'
        )
    with pytest.raises(ValueError, match="animal version"):
        poses_from_json('{"canvas_width": 64, "canvas_height": 64, "animals": [[[1, 2, 1]]]}')


def test_pose_json_bounds_frames_subjects_and_keypoints_before_conversion() -> None:
    with pytest.raises(ValueError, match="1024 frames"):
        poses_from_json(json.dumps([{}] * 1025))
    with pytest.raises(ValueError, match="1024 frames"):
        poses_to_json([Pose(1, 1, ())] * 1025)
    with pytest.raises(ValueError, match="1024 frames"):
        RenderPose.execute(poses=[Pose(1, 1, ())] * 1025)
    with pytest.raises(ValueError, match="1024 people"):
        poses_from_json(
            json.dumps({"canvas_width": 64, "canvas_height": 64, "people": [{}] * 1025})
        )
    with pytest.raises(ValueError, match="4096 keypoints"):
        poses_from_json(
            json.dumps(
                {
                    "canvas_width": 64,
                    "canvas_height": 64,
                    "version": "ap10k",
                    "animals": [[[1, 2, 1]] * 4097],
                }
            )
        )


def test_pose_nodes_are_registered_with_typed_interfaces() -> None:
    assert {node.schema().node_type for node in POSE_NODES} == {
        "dinkster.pose.import_json",
        "dinkster.pose.export_json",
        "dinkster.pose.render",
    }
    assert all(node in IMAGE_NODES for node in POSE_NODES)
    assert ImportPoseJson.schema().outputs[0].type == TypeExpr.list_of(TypeExpr.concrete(POSE_TYPE))
    assert ExportPoseJson.schema().outputs[0].type == TypeExpr.concrete(CORE_STRING)
