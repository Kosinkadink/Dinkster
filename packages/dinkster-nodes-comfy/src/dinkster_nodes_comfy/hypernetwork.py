"""Hypernetwork attention patches loaded from bound model assets."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_FLOAT,
    MODEL_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


class _HypernetworkPatch:
    def __init__(self, networks: dict[int, Any], strength: float) -> None:
        self.networks = networks
        self.strength = strength

    def __call__(self, q: Any, k: Any, v: Any, extra_options: Any) -> tuple[Any, Any, Any]:
        network = self.networks.get(k.shape[-1])
        if network is not None:
            k = k + network[0](k) * self.strength
            v = v + network[1](v) * self.strength
        return q, k, v

    def to(self, device: Any) -> _HypernetworkPatch:
        for dimension in self.networks:
            self.networks[dimension] = self.networks[dimension].to(device)
        return self


def _load_patch(path: str, strength: float) -> _HypernetworkPatch | None:
    torch = importlib.import_module("torch")
    state = importlib.import_module("dinkster_inference.utils").load_torch_file(
        path, safe_load=True
    )
    activation = state.get("activation_func", "linear")
    layer_norm = state.get("is_layer_norm", False)
    dropout = state.get("use_dropout", False)
    activate_output = state.get("activate_output", False)
    last_layer_dropout = state.get("last_layer_dropout", False)
    activations = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softsign": torch.nn.Softsign,
        "mish": torch.nn.Mish,
    }
    if activation not in activations:
        logging.error("Unsupported Hypernetwork activation: %s", activation)
        return None
    networks: dict[int, Any] = {}
    for key in state:
        try:
            dimension = int(key)
        except (ValueError, TypeError):
            continue
        branches: list[Any] = []
        for branch in (0, 1):
            weights = state[dimension][branch]
            names = [name.removesuffix(".weight") for name in weights if name.endswith(".weight")]
            layers: list[Any] = []
            index = 0
            while index < len(names):
                name = names[index]
                last = index == len(names) - 1
                penultimate = index == len(names) - 2
                weight, bias = weights[f"{name}.weight"], weights[f"{name}.bias"]
                linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
                linear.load_state_dict({"weight": weight, "bias": bias})
                layers.append(linear)
                if activation != "linear" and (not last or activate_output):
                    layers.append(activations[activation]())
                if layer_norm:
                    index += 1
                    name = names[index]
                    weight, bias = weights[f"{name}.weight"], weights[f"{name}.bias"]
                    norm = torch.nn.LayerNorm(weight.shape[0])
                    norm.load_state_dict({"weight": weight, "bias": bias})
                    layers.append(norm)
                if dropout and not last and (not penultimate or last_layer_dropout):
                    layers.append(torch.nn.Dropout(p=0.3))
                index += 1
            branches.append(torch.nn.Sequential(*layers))
        networks[dimension] = torch.nn.ModuleList(branches)
    return _HypernetworkPatch(networks, strength)


class HypernetworkLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="HypernetworkLoader",
            display_name="Load Hypernetwork",
            category="model/loaders",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec(
                    "hypernetwork_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(
                        kind="model/hypernetwork", accept=("application/octet-stream",)
                    ),
                ),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, hypernetwork_name: object, strength: float
    ) -> Mapping[str, object]:
        if not isinstance(hypernetwork_name, (AssetRef, LocalAssetRef)):
            raise TypeError("hypernetwork_name must be an AssetRef or LocalAssetRef")
        path = str(hypernetwork_name.local_path())
        initialize_inference_process()
        patched = model.clone()
        patch = _load_patch(path, strength)
        if patch is not None:
            patched.set_model_attn1_patch(patch)
            patched.set_model_attn2_patch(patch)
        return cls.outputs(MODEL=patched)
