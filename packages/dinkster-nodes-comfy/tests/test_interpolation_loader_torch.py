"""Real checkpoint loading through the public interpolation node."""

import ast
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    INTERP_MODEL_TYPE,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)
from dinkster_assets import MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.interpolation_loader import FrameInterpolationModelLoader
from safetensors.torch import save_file

from dinkster.compose import PackSpec, ServingComposer

_FIXTURES = Path(__file__).parent / "fixtures/interpolation"


def _reference_state(kind: str) -> dict[str, torch.Tensor]:
    metadata = json.loads((_FIXTURES / "reference.json").read_text())
    generator = torch.Generator().manual_seed(metadata["seed"])
    return {
        key: torch.randn(shape, generator=generator) * metadata["weight_scale"]
        for key, shape in metadata["models"][kind]["shapes"].items()
    }


def _reference_film(state: dict[str, torch.Tensor]) -> Any:
    metadata = json.loads((_FIXTURES / "reference.json").read_text())
    # Keep the archived numerical fixture intact even though FILM now uses a runtime oracle.
    assert hashlib.sha256((_FIXTURES / "outputs.npz").read_bytes()).hexdigest() == (
        "776e1d72ebff058a994eb8682a6242809bdd6030c472bd4028eadec8bd370e52"
    )
    path = _FIXTURES / "film_net.py.txt"
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == metadata["models"]["film"]["source_sha256"]
    tree = ast.parse(source, filename=str(path))
    # Only the operations provider changes; the frozen reference arithmetic is untouched.
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Import)
            and any(alias.name == "comfy.ops" for alias in node.names)
            or isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "ops" for target in node.targets)
        )
    ]
    scope: dict[str, Any] = {"ops": torch.nn}
    exec(compile(tree, str(path), "exec"), scope)
    model = scope["FILMNet"]()
    model.load_state_dict(state)
    return model.eval()


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, _digest: str) -> Path:
        return self.path


@pytest.mark.parametrize(
    ("kind", "mkldnn_enabled"),
    (("film", True), ("film", False), ("rife", True), ("raw-rife", True)),
    ids=("film", "film-native", "rife", "raw-rife"),
)
def test_real_checkpoint_loads_exact_weights_and_configuration(
    tmp_path: Path, kind: str, mkldnn_enabled: bool
) -> None:
    initialize_inference_process()
    from dinkster_inference.model_patcher import CoreModelPatcher

    architecture = "film" if kind == "film" else "rife"
    expected = _reference_state(architecture)
    stored = expected
    if kind == "raw-rife":
        stored = {
            "module." + key.replace("blocks.", "block"): value for key, value in expected.items()
        }
        stored["module.teacher.discarded"] = torch.tensor([73.0])
        stored["module.caltime.discarded"] = torch.tensor([-19.0])
    path = tmp_path / "interpolation.safetensors"
    save_file(stored, path)
    digest = digest_file(path)
    asset = AssetRef(digest, path.name, path.stat().st_size, resolver=_Resolver(path))
    result = FrameInterpolationModelLoader.execute(model_name=asset)["INTERP_MODEL"]
    assert isinstance(result, CoreModelPatcher)
    assert not result.model.training
    assert result.model.get_dtype() == torch.float32
    actual = result.model.state_dict()
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    with (
        np.load(_FIXTURES / "outputs.npz") as fixture,
        torch.inference_mode(),
        torch.backends.mkldnn.flags(enabled=mkldnn_enabled),
    ):
        images = torch.from_numpy(fixture["images"])
        output = result.model(images[:1].movedim(-1, 1), images[1:2].movedim(-1, 1), timestep=0.375)
        if kind == "film":
            # Float32 convolution accumulation varies across CPU backends. Compare the
            # independent fixed-source model on this backend, retaining bit equality.
            reference = _reference_film(expected)
            expected_output = reference(
                images[:1].movedim(-1, 1), images[1:2].movedim(-1, 1), timestep=0.375
            )
        else:
            expected_output = torch.from_numpy(fixture[architecture])
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    assert digest_file(path) == digest
    if kind != "film":
        assert result.model.encode.cnn3.out_channels == 6
        assert [block.conv0[1][0].out_channels for block in result.model.blocks] == [
            8,
            10,
            12,
            14,
            16,
        ]


def test_unrecognized_checkpoint_has_reference_error(tmp_path: Path) -> None:
    initialize_inference_process()
    path = tmp_path / "unknown.safetensors"
    save_file({"unrelated.weight": torch.tensor([[1.0, -3.0]])}, path)
    asset = AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=_Resolver(path))
    with pytest.raises(ValueError, match="^Unrecognized frame interpolation model format$"):
        FrameInterpolationModelLoader.execute(model_name=asset)


def test_unbound_checkpoint_is_rejected() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        FrameInterpolationModelLoader.execute(model_name="checkpoint.safetensors")


class InspectInterpolationModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="interpolation-fixture.inspect",
            inputs=(InputSpec("model", TypeExpr.concrete(INTERP_MODEL_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: object) -> dict[str, object]:
        from dinkster_inference.model_patcher import CoreModelPatcher

        assert isinstance(model, CoreModelPatcher)
        assert model.load_device == model.offload_device == torch.device("cpu")
        assert not model.model.training
        assert model.model.encode.cnn3.out_channels == 6
        assert [block.conv0[1][0].out_channels for block in model.model.blocks] == [
            8,
            10,
            12,
            14,
            16,
        ]
        weight = model.model.state_dict()["encode.cnn0.weight"]
        assert weight.dtype == torch.float32
        torch.testing.assert_close(
            weight,
            (torch.arange(432, dtype=torch.float32).reshape(16, 3, 3, 3) - 137) / 1024,
            rtol=0,
            atol=0,
        )
        return cls.outputs(summary="resident RIFE has exact weights, channels, dtype and devices")


class InterpolationImages(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="interpolation-fixture.images",
            outputs=(OutputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            images=torch.tensor([0.125, 0.75, 0.375])
            .reshape(3, 1, 1, 1)
            .expand(3, 64, 64, 3)
            .clone()
        )


class InspectInterpolatedImages(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="interpolation-fixture.frames",
            inputs=(InputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, images: object) -> dict[str, object]:
        actual = materialize_image(images)
        expected = torch.tensor([0.125, 0.4375, 0.4375, 0.75, 0.5625, 0.5625, 0.375])
        expected = expected.reshape(7, 1, 1, 1).expand(7, 64, 64, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return cls.outputs(summary="native loader feeds native interpolation with ordered frames")


@pytest.mark.parametrize("verified", (False, True))
@pytest.mark.parametrize("in_process", (True, False))
def test_mounted_checkpoint_crosses_ordinary_hosts_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verified: bool, in_process: bool
) -> None:
    initialize_inference_process()

    state = {key: torch.zeros_like(value) for key, value in _reference_state("rife").items()}
    state["encode.cnn0.weight"] = (
        torch.arange(432, dtype=torch.float32).reshape(16, 3, 3, 3) - 137
    ) / 1024
    path = tmp_path / "rife.safetensors"
    save_file(state, path)
    before = digest_file(path)
    snapshot = tmp_path / "mounts.json"
    mounts = MountTable(snapshot, index_root=tmp_path / "index")
    mounts.add(MountDef(id="interpolation", path=tmp_path), kind="model/frame-interpolation")
    if verified:
        mounts.scan("interpolation")
    asset = mounts.bind_name(path.name, "model/frame-interpolation")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    module = "interpolation_fixture_" + tmp_path.name.replace("-", "_")
    (tmp_path / f"{module}.py").write_text(
        "from test_interpolation_loader_torch import (InspectInterpolationModel, "
        "InterpolationImages, InspectInterpolatedImages)\n"
        "from dinkster_nodes_comfy import register_inference_types\n"
        "NODES = (InspectInterpolationModel, InterpolationImages, InspectInterpolatedImages)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname = "interpolation-fixture"\npublisher = "dinkster"\n'
        'namespaces = ["interpolation-fixture"]\n[pack.entry]\n'
        f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
        encoding="ascii",
    )
    package = Path(__file__).resolve().parents[1]
    checkout = package.parents[1]
    native = package / "dinkster-pack.toml"

    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(checkout / "src"),
                        *(str(p) for p in sorted(checkout.glob("packages/*/src"))),
                        *(p for p in sys.path if p),
                    )
                ),
            }
        )
        try:
            manifests = (native, fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "interpolation-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "FrameInterpolationModelLoader", {"model_name": asset.to_wire()}
                    ),
                    "inspect": GraphNode(
                        "interpolation-fixture.inspect", {"model": Link("load", "INTERP_MODEL")}
                    ),
                    "images": GraphNode("interpolation-fixture.images"),
                    "interpolate": GraphNode(
                        "FrameInterpolate",
                        {
                            "interp_model": Link("load", "INTERP_MODEL"),
                            "images": Link("images", "images"),
                            "multiplier": 3,
                        },
                    ),
                    "frames": GraphNode(
                        "interpolation-fixture.frames",
                        {
                            "images": Link("interpolate", "IMAGE"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect", "frames"])
                assert result.outputs["inspect"]["summary"].resolve() == (
                    "resident RIFE has exact weights, channels, dtype and devices"
                )
                assert result.outputs["frames"]["summary"].resolve() == (
                    "native loader feeds native interpolation with ordered frames"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert digest_file(path) == before
