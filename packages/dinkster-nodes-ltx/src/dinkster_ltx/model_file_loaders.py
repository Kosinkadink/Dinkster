from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    CLIP_TYPE,
    VAE_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    first_party_schema,
    initialize_inference_process,
)
from dinkster_pack_helpers.model_file_loaders import (
    asset_input as _asset,
)
from dinkster_pack_helpers.model_file_loaders import (
    asset_path as _path,
)
from dinkster_pack_helpers.model_file_loaders import (
    clip_options as _clip_options,
)
from dinkster_pack_helpers.model_file_loaders import (
    combo_input as _combo,
)
from dinkster_pack_helpers.model_file_loaders import (
    embedding_roots as _embedding_roots,
)

_CLIP = TypeExpr.concrete(CLIP_TYPE)

_VAE = TypeExpr.concrete(VAE_TYPE)


class LTXVAudioVAELoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="LTXVAudioVAELoader",
            display_name="Load LTXV Audio VAE",
            category="model/loaders",
            inputs=(_asset("ckpt_name", "model/checkpoint", doc="Audio VAE checkpoint to load."),),
            outputs=(OutputSpec("VAE", _VAE, display_name="Audio VAE"),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, ckpt_name: object) -> Mapping[str, object]:
        path = _path(ckpt_name, "ckpt_name")
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        sd = importlib.import_module("dinkster_inference.sd")
        state, metadata = utils.load_torch_file(path, return_metadata=True)
        state = utils.state_dict_prefix_replace(
            state, {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True
        )
        vae = sd.VAE(sd=state, metadata=metadata)
        vae.throw_exception_if_invalid()
        return cls.outputs(VAE=vae)


class LTXAVTextEncoderLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="LTXAVTextEncoderLoader",
            display_name="Load LTXV Audio Text Encoder",
            category="model/loaders",
            description="Recipes:\nltxav: gemma 3 12B or matching gemma 4 model",
            inputs=(
                _asset("text_encoder", "model/text-encoder"),
                _asset("ckpt_name", "model/checkpoint"),
                _combo("device", ("default", "cpu"), "default", advanced=True),
            ),
            outputs=(OutputSpec("CLIP", _CLIP),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(
        cls, *, text_encoder: object, ckpt_name: object, device: str = "default"
    ) -> Mapping[str, object]:
        paths = [_path(text_encoder, "text_encoder"), _path(ckpt_name, "ckpt_name")]
        initialize_inference_process()
        sd = importlib.import_module("dinkster_inference.sd")
        clip = sd.load_clip(
            ckpt_paths=paths,
            embedding_directory=_embedding_roots(),
            clip_type=sd.CLIPType.LTXV,
            model_options=_clip_options("cpu") if device == "cpu" else {},
        )
        return cls.outputs(CLIP=clip)
