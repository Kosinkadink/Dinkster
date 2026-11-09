"""PhotoMaker identity conditioning over the pinned vision primitives."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    ASSET_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    PHOTOMAKER_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

_VISION_CONFIG = {
    "hidden_size": 1024,
    "image_size": 224,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768,
    "hidden_act": "quick_gelu",
    "model_type": "clip_vision_model",
}


def _fusion_module(operations: Any) -> type[Any]:
    # The application composite is lazy so catalog composition stays Torch-free.
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    module_base = cast(type[object], nn.Module)

    # PhotoMaker model composition: TencentARC/PhotoMaker, Apache-2.0.
    class MLP(module_base):
        def __init__(
            self, in_dim: int, out_dim: int, hidden_dim: int, *, use_residual: bool = True
        ) -> None:
            super().__init__()
            if use_residual:
                assert in_dim == out_dim
            self.layernorm = operations.LayerNorm(in_dim)
            self.fc1 = operations.Linear(in_dim, hidden_dim)
            self.fc2 = operations.Linear(hidden_dim, out_dim)
            self.use_residual = use_residual
            self.act_fn = nn.GELU()

        def forward(self, x: Any) -> Any:
            residual = x
            x = self.fc2(self.act_fn(self.fc1(self.layernorm(x))))
            if self.use_residual:
                x = x + residual
            return x

    class FuseModule(module_base):
        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.mlp1: Any = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
            self.mlp2: Any = MLP(embed_dim, embed_dim, embed_dim)
            self.layer_norm = operations.LayerNorm(embed_dim)

        def fuse_fn(self, prompt_embeds: Any, id_embeds: Any) -> Any:
            stacked = self.mlp1(torch.cat([prompt_embeds, id_embeds], dim=-1)) + prompt_embeds
            return self.layer_norm(self.mlp2(stacked))

        def forward(self, prompt_embeds: Any, id_embeds: Any, class_tokens_mask: Any) -> Any:
            id_embeds = id_embeds.to(prompt_embeds.dtype)
            num_inputs = class_tokens_mask.sum().unsqueeze(0)
            batch_size, max_num_inputs = id_embeds.shape[:2]
            seq_length = prompt_embeds.shape[1]
            flat_ids = id_embeds.view(-1, id_embeds.shape[-2], id_embeds.shape[-1])
            valid = (
                torch.arange(max_num_inputs, device=flat_ids.device)[None, :] < num_inputs[:, None]
            )
            valid_ids = flat_ids[valid.flatten()]
            prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
            class_tokens_mask = class_tokens_mask.view(-1)
            valid_ids = valid_ids.view(-1, valid_ids.shape[-1])
            stacked = self.fuse_fn(prompt_embeds[class_tokens_mask], valid_ids)
            assert class_tokens_mask.sum() == stacked.shape[0], (
                f"{class_tokens_mask.sum()} != {stacked.shape[0]}"
            )
            prompt_embeds.masked_scatter_(
                class_tokens_mask[:, None], stacked.to(prompt_embeds.dtype)
            )
            return prompt_embeds.view(batch_size, seq_length, -1)

    return FuseModule


def _id_encoder() -> Any:
    clip_model = importlib.import_module("dinkster_inference.clip_model")
    management = importlib.import_module("dinkster_inference.model_management")
    ops = importlib.import_module("dinkster_inference.ops")
    torch = importlib.import_module("torch")
    projection_base = cast(type[object], clip_model.CLIPVisionModelProjection)

    class PhotoMakerIDEncoder(projection_base):
        vision_model: Any
        visual_projection: Any

        def __init__(self) -> None:
            self.load_device = management.text_encoder_device()
            offload_device = management.text_encoder_offload_device()
            dtype = management.text_encoder_dtype(self.load_device)
            cast(Any, super()).__init__(_VISION_CONFIG, dtype, offload_device, ops.manual_cast)
            self.visual_projection_2 = ops.manual_cast.Linear(1024, 1280, bias=False)
            self.fuse_module = _fusion_module(ops.manual_cast)(2048)

        def forward(self, id_pixel_values: Any, prompt_embeds: Any, class_tokens_mask: Any) -> Any:
            b, num_inputs, c, h, w = id_pixel_values.shape
            pixels = id_pixel_values.view(b * num_inputs, c, h, w)
            shared = self.vision_model(pixels)[2]
            first = self.visual_projection(shared).view(b, num_inputs, 1, -1)
            second = self.visual_projection_2(shared).view(b, num_inputs, 1, -1)
            return self.fuse_module(
                prompt_embeds, torch.cat((first, second), dim=-1), class_tokens_mask
            )

    return PhotoMakerIDEncoder()


class PhotoMakerLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PhotoMakerLoader",
            display_name="Load PhotoMaker Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "photomaker_model_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(kind="model/photomaker"),
                ),
            ),
            outputs=(OutputSpec("PHOTOMAKER", TypeExpr.concrete(PHOTOMAKER_TYPE)),),
        )

    @classmethod
    def execute(cls, *, photomaker_model_name: object) -> Mapping[str, object]:
        if not isinstance(photomaker_model_name, (AssetRef, LocalAssetRef)):
            raise TypeError("photomaker_model_name must be an AssetRef or LocalAssetRef")
        path = photomaker_model_name.local_path()
        initialize_inference_process()
        model = _id_encoder()
        utils = importlib.import_module("dinkster_inference.utils")
        data = utils.load_torch_file(str(path), safe_load=True)
        if "id_encoder" in data:
            data = data["id_encoder"]
        model.load_state_dict(data)
        return cls.outputs(PHOTOMAKER=model)


class PhotoMakerEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PhotoMakerEncode",
            display_name="PhotoMaker Encode",
            category="model/conditioning/photomaker",
            inputs=(
                InputSpec("photomaker", TypeExpr.concrete(PHOTOMAKER_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec(
                    "text",
                    TypeExpr.concrete(CORE_STRING),
                    default="photograph of photomaker",
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, photomaker: Any, image: object, clip: Any, text: str
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        clip_vision = importlib.import_module("dinkster_inference.clip_vision")
        pixels = clip_vision.clip_preprocess(
            materialize_image(image).to(photomaker.load_device)
        ).float()
        try:
            index = text.split(" ").index("photomaker") + 1
        except ValueError:
            index = -1
        tokens = clip.tokenize(text, return_word_ids=True)
        out_tokens: dict[str, list[Any]] = {}
        for key, chunks in tokens.items():
            out_tokens[key] = []
            for chunk in chunks:
                filtered = [token for token in chunk if token[2] != index]
                while len(filtered) < len(chunk):
                    filtered.append(chunk[-1])
                out_tokens[key].append(filtered)
        cond, pooled = clip.encode_from_tokens(out_tokens, return_pooled=True)
        if index > 0:
            mask = [index - 1 <= i < index for i in range(77)]
            out = photomaker(
                id_pixel_values=pixels.unsqueeze(0),
                prompt_embeds=cond.to(photomaker.load_device),
                class_tokens_mask=torch.tensor(
                    mask, dtype=torch.bool, device=photomaker.load_device
                ).unsqueeze(0),
            )
        else:
            out = cond
        return cls.outputs(CONDITIONING=[[out, {"pooled_output": pooled}]])
