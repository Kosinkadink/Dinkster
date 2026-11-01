"""Temporal pose results and canonical transport through both ordinary hosts."""

from __future__ import annotations

import asyncio
import copy
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    MHR_POSE_DATA_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_vision_packet,
)
from dinkster_assets import MountDef, MountTable
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_comfy.sam3d_body import SAM3DBody_Loader, SAM3DBody_Smooth
from safetensors.torch import save_file

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("storage_dtype", "mixed", "missing", "unexpected"),
    (
        (torch.float32, False, [], []),
        (torch.float16, False, [], []),
        (torch.float32, True, [], []),
        (torch.float32, False, ["z", "a"], []),
        (torch.float32, False, [], ["y", "b"]),
        (torch.float32, False, ["z", "a"], ["y", "b"]),
    ),
)
def test_bound_sam3d_loader_state_policy_and_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_dtype: Any,
    mixed: bool,
    missing: list[str],
    unexpected: list[str],
) -> None:
    """Real checkpoint IO and CPU policy, with only model construction controlled."""
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    class ControlledModel(torch.nn.Module):
        def __init__(self, *, dtype: Any, operations: Any) -> None:
            super().__init__()
            self.constructor_dtype = dtype
            self.operations = operations

        def load_state_dict(self, state_dict: Any, strict: bool = True, **kwargs: Any) -> Any:
            assert strict is False
            assert sorted(state_dict) == (
                ["probe.layers.0.comfy_quant", "probe.layers.0.weight"]
                if mixed
                else ["probe.layers.0.weight"]
            )
            torch.testing.assert_close(
                state_dict["probe.layers.0.weight"].float(),
                torch.tensor([[1.0, -2.0, 5.0], [8.0, 0.5, -4.0]]),
                rtol=0,
                atol=0,
            )
            assert hasattr(self.operations, "_quant_config") == mixed
            if mixed:
                assert self.operations._quant_config == {"mixed_ops": True}
            return missing, unexpected

    original_import = importlib.import_module

    def controlled_import(name: str, package: str | None = None) -> Any:
        if name == "dinkster_inference.ldm.sam3d_body.model.model":
            return SimpleNamespace(SAM3DBody=ControlledModel)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", controlled_import)
    state = {
        "probe.layers.0.0.weight": torch.tensor(
            [[1.0, -2.0, 5.0], [8.0, 0.5, -4.0]], dtype=storage_dtype
        ),
        "hand_cls_embed.weight": torch.tensor([17.0], dtype=storage_dtype),
        "hand_cls_embed.bias": torch.tensor([19.0], dtype=storage_dtype),
    }
    if mixed:
        state["probe.layers.0.0.comfy_quant"] = torch.tensor([123, 125], dtype=torch.uint8)
    path = tmp_path / "model.safetensors"
    save_file(state, path)
    before = path.read_bytes()
    mounts = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    mounts.add(MountDef(id="detection", path=tmp_path), kind="model/detection")
    asset = mounts.bind_name(path.name, "model/detection")
    if missing or unexpected:
        with pytest.raises(RuntimeError) as error:
            SAM3DBody_Loader.execute(model_file=asset)
        assert str(error.value) == (
            f"SAM3D-Body checkpoint key mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    else:
        patcher = SAM3DBody_Loader.execute(model_file=asset)["sam3d_body_model"]
        assert isinstance(patcher, ModelPatcher)
        assert patcher.load_device.type == patcher.offload_device.type == "cpu"
        assert patcher.model.backbone_dtype == patcher.model.constructor_dtype == torch.float32
        if not mixed:
            assert patcher.model.operations.__name__ == "disable_weight_init"
    assert path.read_bytes() == before


def test_sam3d_loader_rejects_unbound_file() -> None:
    with pytest.raises(TypeError, match="model_file must be an AssetRef or LocalAssetRef"):
        SAM3DBody_Loader.execute(model_file="model.safetensors")


class LoaderModel(torch.nn.Module):
    def __init__(self, *, dtype: Any, operations: Any) -> None:
        super().__init__()
        self.probe = torch.nn.Module()
        self.probe.layers = torch.nn.ModuleList([torch.nn.Linear(3, 2, bias=False, dtype=dtype)])


class ModelInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3d-loader-fixture.inspect",
            inputs=(InputSpec("model", TypeExpr.concrete(SAM3D_BODY_MODEL_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> Any:
        torch.testing.assert_close(
            model.model.probe.layers[0].weight,
            torch.tensor([[1.0, -2.0, 5.0], [8.0, 0.5, -4.0]]),
            atol=0,
            rtol=0,
        )
        assert model.model.backbone_dtype == torch.float32
        assert model.load_device.type == model.offload_device.type == "cpu"
        return cls.outputs(result="checkpoint values and CPU patcher preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_sam3d_loader_ordinary_hosts_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    """Controlled architecture only; this does not certify the unmocked model import."""
    model_module = "dinkster_inference.ldm.sam3d_body.model.model"
    monkeypatch.setitem(sys.modules, model_module, SimpleNamespace(SAM3DBody=LoaderModel))
    path = tmp_path / "model.safetensors"
    save_file({"probe.layers.0.0.weight": torch.tensor([[1.0, -2.0, 5.0], [8.0, 0.5, -4.0]])}, path)
    before = path.read_bytes()
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="detection", path=tmp_path), kind="model/detection")
    table.scan("detection")
    asset = table.bind_name(path.name, "model/detection")
    assert asset is not None
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))
    module = f"sam3d_loader_fixture_{int(in_process)}"
    (tmp_path / f"{module}.py").write_text(
        "import sys\nfrom types import SimpleNamespace\n"
        "from test_sam3d_body_torch import LoaderModel, ModelInspect\n"
        "from dinkster_api.v1 import register_inference_types\n"
        f"sys.modules[{model_module!r}] = SimpleNamespace(SAM3DBody=LoaderModel)\n"
        "NODES = (ModelInspect,)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="sam3d-loader-fixture"\npublisher="dinkster"\n'
        'namespaces=["sam3d-loader-fixture"]\n'
        f'[pack.entry]\nnodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "mounts.json"),
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest, worker_group="loader", group_manifests=(native, fixture)
                    )
                )
            graph = graph_from_wire(
                graph_to_wire(
                    Graph(
                        nodes={
                            "load": GraphNode("SAM3DBody_Loader", {"model_file": asset.to_wire()}),
                            "inspect": GraphNode(
                                "sam3d-loader-fixture.inspect",
                                {"model": Link("load", "sam3d_body_model")},
                            ),
                        }
                    )
                )
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == (
                    "checkpoint values and CPU patcher preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert path.read_bytes() == before


def _pose() -> dict[str, Any]:
    return {
        "frames": [
            [
                {
                    "pred_vertices": np.array([[x, 2 * x, -x]], dtype=np.float64),
                    "pred_cam_t": np.array([x, 10 + x, -2 + x], dtype=np.float32),
                    "global_rot": np.array([0, 0, 0], dtype=np.float32),
                    "tag": "preserved",
                }
            ]
            for x in (0, 6, 0, 12)
        ],
        "image_size": (3, 5),
        "faces": np.array([[0, 0, 0]], dtype=np.uint32),
        "fps": 24,
    }


def _smooth(pose: Any, **options: Any) -> Any:
    return SAM3DBody_Smooth.execute(
        mhr_pose_data=pose,
        **(
            {"strength": 1.0, "method": "savgol", "window": 3, "rotation_threshold_degrees": 0.0}
            | options
        ),
    )["mhr_pose_data"]


def _assert_result(pose: Any) -> None:
    pose = materialize_vision_packet(pose)
    # Three-point linear least squares has weights 1/3, including repeated endpoints.
    expected = [2, 2, 6, 8]
    for frame, x in zip(pose["frames"], expected, strict=True):
        np.testing.assert_allclose(frame[0]["pred_vertices"], [[x, 2 * x, -x]], atol=1e-6)
        np.testing.assert_allclose(frame[0]["pred_cam_t"], [x, 10 + x, -2 + x], atol=1e-6)
        assert frame[0]["pred_vertices"].dtype == np.float64
        assert frame[0]["tag"] == "preserved"
    assert pose["image_size"] == (3, 5) and pose["fps"] == 24
    np.testing.assert_array_equal(pose["faces"], [[0, 0, 0]])
    assert pose["faces"].dtype == np.uint32


def test_smoothing_values_noops_and_source_preservation() -> None:
    pose = _pose()
    before = copy.deepcopy(pose)
    _assert_result(_smooth(pose))
    assert _smooth(pose, strength=0.0) is pose
    assert _smooth(pose, window=1) is pose
    for actual, original in zip(pose["frames"], before["frames"], strict=True):
        for key in ("pred_vertices", "pred_cam_t", "global_rot"):
            np.testing.assert_array_equal(actual[0][key], original[0][key])
    empty = {**pose, "frames": []}
    assert _smooth(empty) is empty
    singleton = {**pose, "frames": pose["frames"][:1]}
    assert _smooth(singleton) is singleton


def test_fast_spin_backs_off_geometry_but_not_camera() -> None:
    pose = _pose()
    for index, frame in enumerate(pose["frames"]):
        frame[0]["global_rot"][0] = index * np.pi / 2
    result = _smooth(pose, method="gaussian", rotation_threshold_degrees=30.0)
    for actual, original in zip(result["frames"], pose["frames"], strict=True):
        np.testing.assert_array_equal(actual[0]["pred_vertices"], original[0]["pred_vertices"])
    # Gaussian sigma=1 at window3; independent normalized endpoint weight.
    weight = np.exp(-0.5) / (1 + 2 * np.exp(-0.5))
    np.testing.assert_allclose(
        result["frames"][0][0]["pred_cam_t"],
        [6 * weight, 10 + 6 * weight, -2 + 6 * weight],
        atol=1e-6,
    )


def test_wrapped_angle_ranges_and_rotation_projection() -> None:
    pose = _pose()
    angles = [3.0, -3.0, 3.0, -3.0]
    for frame, angle in zip(pose["frames"], angles, strict=True):
        frame[0]["global_rot"][0] = angle
        frame[0]["body_pose_params"] = np.full(126, angle, dtype=np.float32)
        frame[0]["mhr_model_params"] = np.full(132, angle, dtype=np.float32)
        frame[0]["pred_global_rots"] = np.diag([1.0, 2.0, -1.0])[None]
    result = _smooth(pose)
    root = result["frames"][0][0]
    # First window repeats the first angle; only -3 crosses to 2*pi-3.
    unwrapped = (3 + 2 * np.pi) / 3
    linear = 1.0
    np.testing.assert_allclose(root["global_rot"][0], unwrapped, atol=1e-6)
    np.testing.assert_allclose(root["body_pose_params"][:124], unwrapped, atol=1e-6)
    np.testing.assert_allclose(root["body_pose_params"][124:], linear, atol=1e-6)
    np.testing.assert_allclose(root["mhr_model_params"][:3], linear, atol=1e-6)
    np.testing.assert_allclose(root["mhr_model_params"][3:130], unwrapped, atol=1e-6)
    np.testing.assert_allclose(root["mhr_model_params"][130:], linear, atol=1e-6)
    for frame in result["frames"]:
        rotation = frame[0]["pred_global_rots"][0]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-6)


def test_missing_people_fields_even_window_and_partial_strength() -> None:
    pose = _pose()
    pose["frames"][1] = []
    del pose["frames"][2][0]["pred_cam_t"]
    result = _smooth(pose)
    assert result["frames"][1] == []
    np.testing.assert_allclose(result["frames"][2][0]["pred_cam_t"], [4, 14, 2], atol=1e-6)
    # An even requested window normalizes to five; half strength blends with raw values.
    result = _smooth(_pose(), window=4, method="gaussian", strength=0.5)
    weights = np.array([np.exp(-2), np.exp(-0.5), 1, np.exp(-0.5), np.exp(-2)])
    expected = 0.5 * (6 * weights[3]) / weights.sum()
    np.testing.assert_allclose(result["frames"][0][0]["pred_vertices"][0, 0], expected, atol=1e-6)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3d-fixture.source",
            inputs=(),
            outputs=(OutputSpec("pose", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(pose=_pose())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3d-fixture.inspect",
            inputs=(InputSpec("pose", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, pose: Any) -> Any:
        _assert_result(pose)
        return cls.outputs(result="pose values preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_smoothing_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    module = f"sam3d_fixture_{int(in_process)}"
    (tmp_path / f"{module}.py").write_text(
        "from test_sam3d_body_torch import Source, Inspect\n"
        "from dinkster_api.v1 import register_inference_types\n"
        "NODES = (Source, Inspect)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="sam3d-fixture"\npublisher="dinkster"\nnamespaces=["sam3d-fixture"]\n'
        f'[pack.entry]\nnodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest, worker_group="sam3d", group_manifests=(native, fixture)
                    )
                )
            graph = graph_from_wire(
                graph_to_wire(
                    Graph(
                        nodes={
                            "source": GraphNode("sam3d-fixture.source"),
                            "smooth": GraphNode(
                                "SAM3DBody_Smooth",
                                {
                                    "mhr_pose_data": Link("source", "pose"),
                                    "window": 3,
                                    "rotation_threshold_degrees": 0.0,
                                },
                            ),
                            "inspect": GraphNode(
                                "sam3d-fixture.inspect",
                                {
                                    "pose": Link("smooth", "mhr_pose_data"),
                                },
                            ),
                        }
                    )
                )
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == "pose values preserved"
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
