"""CPU ControlNet chaining, masked hints, and ordinary-pack execution."""

from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CONTROL_NET_TYPE,
    CORE_FLOAT,
    IMAGE_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.controlnet import (
    ControlNetApply,
    ControlNetApplyAdvanced,
    ControlNetInpaintingAliMamaApply,
    SetUnionControlNetType,
)

from dinkster.compose import PackSpec, ServingComposer


class Control:
    def __init__(self, *, concat_mask: bool = True) -> None:
        self.concat_mask = concat_mask
        self.cond_hint_original: Any = None
        self.strength = 0.0
        self.timestep_range = (0.0, 1.0)
        self.previous_controlnet: Any = None
        self.extra_args: dict[str, object] = {}
        self.extra_concat: list[Any] = []
        self.vae: Any = None

    def copy(self) -> Control:
        result = copy.copy(self)
        result.extra_args = self.extra_args.copy()
        return result

    def set_cond_hint(
        self,
        hint: torch.Tensor,
        strength: float,
        timestep_range: tuple[float, float] = (0.0, 1.0),
        *,
        vae: Any = None,
        extra_concat: list[Any] | None = None,
    ) -> Control:
        self.cond_hint_original = hint
        self.strength = strength
        self.timestep_range = timestep_range
        self.vae = vae
        self.extra_concat = [] if extra_concat is None else extra_concat
        return self

    def set_previous_controlnet(self, previous: Any) -> None:
        self.previous_controlnet = previous

    def set_extra_arg(self, name: str, value: object) -> None:
        self.extra_args[name] = value


def test_legacy_copies_each_entry_while_advanced_reuses_prior_control_chains() -> None:
    image = torch.arange(2 * 7 * 11 * 4, dtype=torch.float32).reshape(2, 7, 11, 4) / 1024
    before = image.clone()
    previous = Control()
    source = Control()
    positive = [
        [torch.tensor([1.0]), {"control": previous, "strength": 0.7}],
        [torch.tensor([2.0]), {}],
    ]
    negative = [[torch.tensor([-1.0]), {"control": previous}], [torch.tensor([-2.0]), {}]]
    vae = object()
    advanced: Any = ControlNetApplyAdvanced.execute(
        positive=positive,
        negative=negative,
        control_net=source,
        image=image.numpy(),
        strength=1.25,
        start_percent=0.17,
        end_percent=0.83,
        vae=vae,
    )
    p, n = advanced["positive"], advanced["negative"]
    assert p[0][1]["control"] is n[0][1]["control"]
    assert p[1][1]["control"] is n[1][1]["control"]
    assert p[0][1]["control"] is not p[1][1]["control"]
    control = p[0][1]["control"]
    assert control.previous_controlnet is previous
    assert control.timestep_range == (0.17, 0.83)
    assert control.vae is vae
    assert control.strength == 1.25
    assert torch.equal(control.cond_hint_original, image.movedim(-1, 1))
    assert not p[0][1]["control_apply_to_uncond"]
    legacy: Any = ControlNetApply.execute(
        conditioning=positive + negative, control_net=source, image=image.numpy(), strength=0.5
    )["CONDITIONING"]
    assert len({id(entry[1]["control"]) for entry in legacy}) == 4
    assert all(entry[1]["control_apply_to_uncond"] for entry in legacy)
    assert positive[0][1] == {"control": previous, "strength": 0.7}
    assert positive[1][1] == {}
    assert source.cond_hint_original is None
    assert previous.cond_hint_original is None
    assert torch.equal(image, before)


@pytest.mark.parametrize("concat_mask", (False, True))
def test_alimama_preserves_original_inverse_mask_and_only_masks_when_required(
    concat_mask: bool,
) -> None:
    image = torch.arange(5 * 7 * 4, dtype=torch.float32).reshape(1, 5, 7, 4) / 256
    mask = torch.zeros(1, 5, 7)
    mask[:, 1:4, 2:5] = 1
    before = image.clone()
    source = Control(concat_mask=concat_mask)
    result: Any = ControlNetInpaintingAliMamaApply.execute(
        positive=[[torch.tensor([1.0]), {}]],
        negative=[],
        control_net=source,
        vae=object(),
        image=image.numpy(),
        mask=mask.numpy(),
        strength=0.75,
        start_percent=0.13,
        end_percent=0.91,
    )
    control = result["positive"][0][1]["control"]
    expected = image.clone()
    if concat_mask:
        expected[:, 1:4, 2:5, :] = 0
        assert torch.equal(control.extra_concat[0], (1 - mask).reshape(1, 1, 5, 7))
    else:
        assert control.extra_concat == []
    assert torch.equal(control.cond_hint_original, expected.movedim(-1, 1))
    assert control.strength == 0.75
    assert control.timestep_range == (0.13, 0.91)
    assert torch.equal(image, before)
    assert torch.equal(mask[:, 1:4, 2:5], torch.ones(1, 3, 3))
    assert source.cond_hint_original is None


@pytest.mark.parametrize("name, expected", (("auto", []), ("openpose", [0]), ("repaint", [7])))
def test_union_type_selection_copies_source(name: str, expected: list[int]) -> None:
    source = Control()
    source.set_extra_arg("retained", 4)
    output: Any = SetUnionControlNetType.execute(control_net=source, type=name)["CONTROL_NET"]
    assert output.extra_args == {"retained": 4, "control_type": expected}
    assert source.extra_args == {"retained": 4}


class ControlSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="control-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("control", CONTROL_NET_TYPE),
                    ("image", IMAGE_TYPE),
                    ("mask", MASK_TYPE),
                    ("vae", VAE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        mask = torch.zeros(1, 5, 7)
        mask[:, 1:4, 2:5] = 1
        return cls.outputs(
            positive=[[torch.tensor([1.0]), {"strength": 0.7}]],
            negative=[[torch.tensor([-1.0]), {}]],
            control=Control(),
            image=torch.full((1, 5, 7, 4), 0.25).numpy(),
            mask=mask.numpy(),
            vae=Control(),
        )


class ControlInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="control-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("legacy", "positive", "negative", "masked")
            ),
            outputs=(OutputSpec("strength", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(
        cls, *, legacy: Any, positive: Any, negative: Any, masked: Any
    ) -> dict[str, object]:
        p, n = positive[0][1]["control"], negative[0][1]["control"]
        assert p is n
        assert p.extra_args["control_type"] == [7]
        assert p.timestep_range == (0.17, 0.83)
        assert not positive[0][1]["control_apply_to_uncond"]
        assert legacy[0][1]["control_apply_to_uncond"]
        control = masked[0][1]["control"]
        assert torch.equal(control.cond_hint_original[:, :, 1:4, 2:5], torch.zeros(1, 4, 3, 3))
        assert float(control.cond_hint_original[0, 0, 0, 0]) == 0.25
        assert control.extra_concat[0].shape == (1, 1, 5, 7)
        return cls.outputs(strength=p.strength)


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_controlnet_nodes(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src")))}
        )
        try:
            (tmp_path / "control_fixture.py").write_text(
                "from test_controlnet_torch import ControlSource, ControlInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, ControlSource, ControlInspect)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (root / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["control-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "control_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "control_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            base = {
                "control_net": Link("union", "CONTROL_NET"),
                "image": Link("source", "image"),
                "strength": 1.25,
            }
            scheduled = {
                **base,
                "positive": Link("source", "positive"),
                "negative": Link("source", "negative"),
                "vae": Link("source", "vae"),
                "start_percent": 0.17,
                "end_percent": 0.83,
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("control-fixture.source"),
                    "union": GraphNode(
                        "SetUnionControlNetType",
                        {"control_net": Link("source", "control"), "type": "repaint"},
                    ),
                    "legacy": GraphNode(
                        "ControlNetApply", {**base, "conditioning": Link("source", "positive")}
                    ),
                    "advanced": GraphNode("ControlNetApplyAdvanced", scheduled),
                    "masked": GraphNode(
                        "ControlNetInpaintingAliMamaApply",
                        {**scheduled, "mask": Link("source", "mask")},
                    ),
                    "inspect": GraphNode(
                        "control-fixture.inspect",
                        {
                            "legacy": Link("legacy", "CONDITIONING"),
                            "positive": Link("advanced", "positive"),
                            "negative": Link("advanced", "negative"),
                            "masked": Link("masked", "positive"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["strength"].resolve() == 1.25
        finally:
            await composer.close()

    asyncio.run(scenario())
