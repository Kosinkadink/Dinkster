"""Independent public types and numerical consumers for image-control patches."""

import importlib
from types import SimpleNamespace
from typing import Any

import torch
from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    register_inference_types,
)
from test_image_control_patches_torch import _DiffSynth, _Host, _patcher, _Projector, _Vae, _ZImage

initialize_inference_process()
ZImage_Control = importlib.import_module("dinkster_inference.ldm.lumina.controlnet").ZImage_Control

_PORTS = (
    ("model", MODEL_TYPE),
    ("weights", MODEL_PATCH_TYPE),
    ("vae", VAE_TYPE),
    ("image", IMAGE_TYPE),
    ("vision", CLIP_VISION_OUTPUT_TYPE),
)


class Control(ZImage_Control):
    additional_in_dim = 0
    n_control_layers = 3

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    forward = _ZImage.forward
    forward_control_block = _ZImage.forward_control_block
    forward_noise_refiner_block = _ZImage.forward_noise_refiner_block


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="image-control-fixture.source",
            inputs=(InputSpec("kind", TypeExpr.concrete(CORE_STRING)),),
            outputs=tuple(OutputSpec(name, TypeExpr.concrete(kind)) for name, kind in _PORTS),
        )

    @classmethod
    def execute(cls, *, kind: str) -> dict[str, object]:
        weights = {"diffsynth": _DiffSynth, "zimage": Control, "uso": _Projector}[kind]()
        return cls.outputs(
            model=_patcher(_Host()),
            weights=_patcher(weights),
            vae=_Vae(),
            image=torch.tensor([[[[1.0, 2.0, 3.0, 99.0]]]]),
            vision=SimpleNamespace(
                all_hidden_states=torch.arange(126, dtype=torch.float32).reshape(1, 21, 2, 3),
                penultimate_hidden_states=torch.full((1, 2, 3), 99.0),
            ),
        )


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="image-control-fixture.inspect",
            inputs=(
                InputSpec("kind", TypeExpr.concrete(CORE_STRING)),
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("patched", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("weights", TypeExpr.concrete(MODEL_PATCH_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, kind: str, model: Any, patched: Any, weights: Any, image: Any
    ) -> dict[str, object]:
        assert model is not patched
        assert model.model_options == {"transformer_options": {}}
        assert torch.equal(materialize_image(image), torch.tensor([[[[1.0, 2.0, 3.0, 99.0]]]]))
        patches = patched.model_options["transformer_options"]["patches"]
        if kind == "uso":
            patch = patches["post_input"][0]
            text = torch.tensor([[[7.0, 8.0, 9.0]]])
            result = patch({"txt": text, "txt_ids": torch.ones(1, 1, 3)})
            assert torch.equal(
                result["txt"],
                torch.tensor([[[165.0, 167.0, 169.0], [171.0, 173.0, 175.0], [7.0, 8.0, 9.0]]]),
            )
            assert torch.equal(result["txt_ids"], torch.tensor([[[0.0] * 3, [0.0] * 3, [1.0] * 3]]))
            assert torch.equal(text, torch.tensor([[[7.0, 8.0, 9.0]]]))
        else:
            patch = patches["double_block"][0]
            values = torch.zeros(1, 17, 1, 1) if kind == "diffsynth" else torch.zeros(1, 4, 3)
            common = {
                "transformer_options": {"sigmas": torch.tensor([9.5])},
                "x": torch.zeros(1, 3, 1, 1),
                "img": values,
                "block_index": 1 if kind == "diffsynth" else 0,
                "img_input": torch.zeros(1, 4, 3),
                "txt": torch.ones(1, 1, 3),
                "pe": torch.ones(1, 1, 3),
                "vec": torch.ones(1, 3),
            }
            patch(dict(common))
            if kind == "diffsynth":
                expected = torch.tensor([0.75, 1.0, 1.25] + [0.5] * 13 + [0.75]).reshape(
                    1, 17, 1, 1
                )
            else:
                expected = torch.full_like(values, 0.25)
                assert patches["noise_refiner"][0] is patch
            assert torch.equal(values, expected)
            inactive = torch.full_like(values, 13.0)
            patch(
                {**common, "img": inactive, "transformer_options": {"sigmas": torch.tensor([8.0])}}
            )
            assert torch.equal(inactive, torch.full_like(inactive, 13.0))
        assert patch.models() == [weights]
        assert model.model_options == {"transformer_options": {}}
        return cls.outputs(summary="independent image-control values and source preserved")


FIXTURE_NODES = (Source, Inspect)
__all__ = ["FIXTURE_NODES", "register_inference_types"]
