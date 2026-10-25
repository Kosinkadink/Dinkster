"""CPU hypernetwork loading and ordinary-host attention execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    MODEL_TYPE,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    initialize_inference_process,
    register_asset_type,
    resolver_from_env,
)
from dinkster_assets import MountDef, MountTable
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hypernetwork import HypernetworkLoader
from dinkster_values import register_core_types

from dinkster.compose import PackSpec, ServingComposer, default_pack_spec


def _model() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))


def _state() -> dict[Any, Any]:
    return {
        2: [
            {
                "0.weight": torch.tensor([[2.0, -1.0], [0.5, 3.0]]),
                "0.bias": torch.tensor([1.0, -2.0]),
            },
            {
                "0.weight": torch.tensor([[-1.0, 4.0], [2.0, 0.0]]),
                "0.bias": torch.tensor([-3.0, 0.5]),
            },
        ]
    }


def _asset(tmp_path: Path, *, verified: bool = False, state: dict[Any, Any] | None = None) -> Any:
    path = tmp_path / "network.pt"
    torch.save(_state() if state is None else state, path)
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="hypernetworks", path=tmp_path), kind="model/hypernetwork")
    if verified:
        table.scan("hypernetworks")
    asset = table.bind_name(path.name, "model/hypernetwork")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    return asset


def _check(source: Any, patched: Any, strength: float) -> None:
    assert source is not patched and source.model is patched.model
    assert "patches" not in source.model_options["transformer_options"]
    patches = patched.model_options["transformer_options"]["patches"]
    q = torch.tensor([[[13.0, -9.0]]])
    k = torch.tensor([[[3.0, -2.0], [-1.0, 4.0]]])
    v = torch.tensor([[[-5.0, 1.0], [2.0, -3.0]]])
    expected_k = k + strength * torch.tensor([[[9.0, -6.5], [-5.0, 9.5]]])
    expected_v = v + strength * torch.tensor([[[6.0, -9.5], [-17.0, 4.5]]])
    for key in ("attn1_patch", "attn2_patch"):
        patch = patches[key][0]
        result = patch(q, k, v, {})
        assert result[0] is q
        torch.testing.assert_close(result[1], expected_k, rtol=0, atol=0)
        torch.testing.assert_close(result[2], expected_v, rtol=0, atol=0)
        assert patch.to(torch.device("cpu")) is patch
        unmatched = torch.ones(1, 1, 3)
        assert patch(q, unmatched, unmatched, {}) == (q, unmatched, unmatched)
    assert patches["attn1_patch"][0] is patches["attn2_patch"][0]


@pytest.mark.parametrize("verified", (False, True))
@pytest.mark.parametrize("strength", (-0.5, 0.0, 1.25))
def test_hypernetwork_bound_weight_loading(tmp_path: Path, verified: bool, strength: float) -> None:
    source = _model()
    result = HypernetworkLoader.execute(
        model=source, hypernetwork_name=_asset(tmp_path, verified=verified), strength=strength
    )["MODEL"]
    _check(source, result, strength)


def test_hypernetwork_unsupported_activation_and_unbound_filename(tmp_path: Path) -> None:
    source = _model()
    state = {**_state(), "activation_func": "unsupported"}
    result = HypernetworkLoader.execute(
        model=source, hypernetwork_name=_asset(tmp_path, state=state), strength=1.0
    )["MODEL"]
    assert result is not source and result.model_options == source.model_options
    with pytest.raises(TypeError, match="AssetRef"):
        HypernetworkLoader.execute(model=source, hypernetwork_name="network.pt", strength=1.0)


@pytest.mark.parametrize(
    "activation", ("relu", "leakyrelu", "elu", "swish", "tanh", "sigmoid", "softsign", "mish")
)
def test_hypernetwork_output_activation(tmp_path: Path, activation: str) -> None:
    functions = {
        "relu": torch.nn.functional.relu,
        "leakyrelu": torch.nn.functional.leaky_relu,
        "elu": torch.nn.functional.elu,
        "swish": torch.nn.functional.hardswish,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "softsign": torch.nn.functional.softsign,
        "mish": torch.nn.functional.mish,
    }
    asset = _asset(
        tmp_path, state={**_state(), "activation_func": activation, "activate_output": True}
    )
    result = HypernetworkLoader.execute(model=_model(), hypernetwork_name=asset, strength=0.5)[
        "MODEL"
    ]
    patch = result.model_options["transformer_options"]["patches"]["attn1_patch"][0]
    value = torch.tensor([[[3.0, -2.0]]])
    q, k, v = patch(value, value, value, {})
    assert q is value
    torch.testing.assert_close(
        k, value + 0.5 * functions[activation](torch.tensor([[[9.0, -6.5]]]))
    )
    torch.testing.assert_close(
        v, value + 0.5 * functions[activation](torch.tensor([[[-14.0, 6.5]]]))
    )


@pytest.mark.parametrize("last_layer_dropout", (False, True))
def test_hypernetwork_layer_norm_and_dropout(tmp_path: Path, last_layer_dropout: bool) -> None:
    branches = []
    for initial in _state()[2]:
        branches.append(
            {
                **initial,
                "1.weight": torch.tensor([1.5, 0.5]),
                "1.bias": torch.tensor([-0.25, 0.75]),
                "2.weight": torch.eye(2) * 0.5,
                "2.bias": torch.ones(2) * 0.25,
                "3.weight": torch.ones(2),
                "3.bias": torch.zeros(2),
            }
        )
    state = {
        2: branches,
        "activation_func": "relu",
        "is_layer_norm": True,
        "use_dropout": True,
        "last_layer_dropout": last_layer_dropout,
    }
    asset = _asset(tmp_path, state=state)
    result = HypernetworkLoader.execute(model=_model(), hypernetwork_name=asset, strength=0.5)[
        "MODEL"
    ]
    patch = result.model_options["transformer_options"]["patches"]["attn1_patch"][0]
    value = torch.tensor([[[3.0, -2.0], [-1.0, 4.0]]])

    def expected(initial: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.layer_norm(
            initial.relu(), (2,), torch.tensor([1.5, 0.5]), torch.tensor([-0.25, 0.75])
        )
        hidden = torch.nn.functional.dropout(hidden, 0.3, training=True)
        hidden = torch.nn.functional.layer_norm((hidden * 0.5 + 0.25).relu(), (2,))
        if last_layer_dropout:
            hidden = torch.nn.functional.dropout(hidden, 0.3, training=True)
        return value + hidden * 0.5

    torch.manual_seed(441)
    expected_k = expected(torch.tensor([[[9.0, -6.5], [-5.0, 9.5]]]))
    expected_v = expected(torch.tensor([[[-14.0, 6.5], [14.0, -1.5]]]))
    torch.manual_seed(441)
    _, k, v = patch(value, value, value, {})
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected_v)


class HypernetworkSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hypernetwork-fixture.source",
            outputs=(OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=_model())


class HypernetworkInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hypernetwork-fixture.inspect",
            inputs=(
                InputSpec("source", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("patched", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, source: Any, patched: Any) -> dict[str, object]:
        _check(source, patched, -0.5)
        return cls.outputs(summary="asymmetric attention preserved")


HYPERNETWORK_FIXTURES = (HypernetworkSource, HypernetworkInspect)


@pytest.mark.parametrize("in_process", (True, False))
def test_hypernetwork_both_ordinary_hosts_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    asset = _asset(tmp_path)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        native = root / "dinkster-pack.toml"
        fixture = tmp_path / "hypernetwork-pack.toml"
        fixture.write_text(
            '[pack]\nname="hypernetwork-fixture"\npublisher="dinkster"\n'
            'namespaces=["hypernetwork-fixture"]\n[pack.entry]\n'
            'nodes="hypernetwork_fixture:HYPERNETWORK_FIXTURES"\n'
            'types="hypernetwork_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        (tmp_path / "hypernetwork_fixture.py").write_text(
            "from test_hypernetwork_torch import HYPERNETWORK_FIXTURES\n"
            "from dinkster_nodes_comfy import register_inference_types\n",
            encoding="ascii",
        )
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, resolver_from_env())
        composer = ServingComposer(
            registry=registry,
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "mounts.json"),
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(root / "tests"), str(root / "src"))
                ),
            },
        )
        try:
            generation = default_pack_spec("dinkster-nodes-generation")
            manifests = (
                Path(generation.manifest),
                root.parent / "dinkster-native/dinkster-pack.toml",
                native,
                fixture,
            )
            for manifest in manifests:
                packs = generation.packs if manifest == Path(generation.manifest) else None
                await composer.add_pack(
                    PackSpec(manifest=manifest, packs=packs, in_process=True, trust_reserved=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        packs=packs,
                        trust_reserved=True,
                        worker_group="hypernetwork-fixture",
                        group_manifests=manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("hypernetwork-fixture.source"),
                    "patch": GraphNode(
                        "HypernetworkLoader",
                        {
                            "model": Link("source", "model"),
                            "hypernetwork_name": asset.to_wire(),
                            "strength": -0.5,
                        },
                    ),
                    "inspect": GraphNode(
                        "hypernetwork-fixture.inspect",
                        {
                            "source": Link("source", "model"),
                            "patched": Link("patch", "MODEL"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "asymmetric attention preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
