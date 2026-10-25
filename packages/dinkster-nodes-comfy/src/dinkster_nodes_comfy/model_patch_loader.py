"""Load model patch checkpoints into Dinkster inference patchers."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    ASSET_TYPE,
    MODEL_PATCH_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_ASSET = TypeExpr.concrete(ASSET_TYPE)
_MODEL_PATCH = TypeExpr.concrete(MODEL_PATCH_TYPE)


def _z_image_convert(state_dict: dict[str, Any], torch: Any) -> dict[str, Any]:
    replacements = {
        ".attention.to_out.0.bias": ".attention.out.bias",
        ".attention.norm_k.weight": ".attention.k_norm.weight",
        ".attention.norm_q.weight": ".attention.q_norm.weight",
        ".attention.to_out.0.weight": ".attention.out.weight",
    }
    converted: dict[str, Any] = {}
    qk: list[Any] = []
    for key in sorted(state_dict):
        value = state_dict[key]
        output_key = key
        if output_key.endswith(".attention.to_k.weight"):
            qk = [value]
            continue
        if output_key.endswith(".attention.to_q.weight"):
            qk = [value, *qk]
            continue
        if output_key.endswith(".attention.to_v.weight"):
            value = torch.cat([*qk, value], dim=0)
            output_key = output_key.replace(".attention.to_v.weight", ".attention.qkv.weight")
        for source, target in replacements.items():
            output_key = output_key.replace(source, target)
        converted[output_key] = value
    return converted


def _qwen_blockwise_controlnet(operations: Any) -> type[Any]:
    nn = importlib.import_module("torch.nn")
    module_base = cast(type[object], nn.Module)

    class BlockWiseControlBlock(module_base):
        def __init__(self, dim: int, *, device: Any, dtype: Any) -> None:
            super().__init__()
            self.x_rms = operations.RMSNorm(dim, eps=1e-6)
            self.y_rms = operations.RMSNorm(dim, eps=1e-6)
            self.input_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
            self.act = nn.GELU()
            self.output_proj = operations.Linear(dim, dim, device=device, dtype=dtype)

        def forward(self, x: Any, y: Any) -> Any:
            return self.output_proj(self.act(self.input_proj(self.x_rms(x) + self.y_rms(y))))

    class QwenImageBlockWiseControlNet(module_base):
        def __init__(
            self,
            *,
            additional_in_dim: int,
            num_layers: int = 60,
            in_dim: int = 64,
            dim: int = 3072,
            device: Any,
            dtype: Any,
        ) -> None:
            super().__init__()
            self.additional_in_dim = additional_in_dim
            self.img_in = operations.Linear(
                in_dim + additional_in_dim, dim, device=device, dtype=dtype
            )
            self.controlnet_blocks = nn.ModuleList(
                BlockWiseControlBlock(dim, device=device, dtype=dtype) for _ in range(num_layers)
            )

        def process_input_latent_image(self, latent_image: Any) -> Any:
            latent_formats = importlib.import_module("dinkster_inference.latent_formats")
            common_dit = importlib.import_module("dinkster_inference.ldm.common_dit")
            latent_image[:, :16] = latent_formats.Wan21().process_in(latent_image[:, :16])
            hidden_states = common_dit.pad_to_patch_size(latent_image, (1, 2, 2))
            shape = hidden_states.shape
            hidden_states = hidden_states.view(
                shape[0], shape[1], shape[-2] // 2, 2, shape[-1] // 2, 2
            )
            hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
            hidden_states = hidden_states.reshape(
                shape[0], (shape[-2] // 2) * (shape[-1] // 2), shape[1] * 4
            )
            return self.img_in(hidden_states)

        def control_block(self, image: Any, conditioning: Any, block_id: int) -> Any:
            return self.controlnet_blocks[block_id](image, conditioning)

    return QwenImageBlockWiseControlNet


def _siglip_multi_feat_projector(operations: Any) -> type[Any]:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    module_base = cast(type[object], nn.Module)

    class SigLIPMultiFeatProjModel(module_base):
        def __init__(self, *, device: Any, dtype: Any) -> None:
            super().__init__()
            self.high_embedding_linear = nn.Sequential(operations.Linear(729, 64), nn.SiLU())
            self.high_layer_norm = operations.LayerNorm(1152)
            self.high_projection = operations.Linear(
                1152, 3072, bias=True, device=device, dtype=dtype
            )
            self.mid_embedding_linear = nn.Sequential(operations.Linear(729, 64), nn.SiLU())
            self.mid_layer_norm = operations.LayerNorm(1152)
            self.mid_projection = operations.Linear(
                1152, 3072, bias=True, device=device, dtype=dtype
            )
            self.low_embedding_linear = nn.Sequential(operations.Linear(729, 64), nn.SiLU())
            self.low_layer_norm = operations.LayerNorm(1152)
            self.low_projection = operations.Linear(
                1152, 3072, bias=True, device=device, dtype=dtype
            )

        def forward(self, siglip_outputs: Any) -> Any:
            dtype = next(self.high_embedding_linear.parameters()).dtype
            outputs: list[Any] = []
            for hidden_states, embedder, layer_norm, projection in (
                (
                    siglip_outputs[2],
                    self.high_embedding_linear,
                    self.high_layer_norm,
                    self.high_projection,
                ),
                (
                    siglip_outputs[1],
                    self.mid_embedding_linear,
                    self.mid_layer_norm,
                    self.mid_projection,
                ),
                (
                    siglip_outputs[0],
                    self.low_embedding_linear,
                    self.low_layer_norm,
                    self.low_projection,
                ),
            ):
                embedding = embedder(hidden_states.to(dtype).transpose(1, 2)).transpose(1, 2)
                outputs.append(projection(layer_norm(embedding)))
            return torch.cat(outputs, dim=1)

    return SigLIPMultiFeatProjModel


def _multi_talk_patch(multitalk: Any, operations: Any) -> type[Any]:
    nn = importlib.import_module("torch.nn")
    module_base = cast(type[object], nn.Module)

    class MultiTalkModelPatch(module_base):
        def __init__(
            self, *, in_dim: int, intermediate_dim: int, out_dim: int, device: Any
        ) -> None:
            super().__init__()
            self.audio_proj = multitalk.MultiTalkAudioProjModel(
                seq_len=5,
                seq_len_vf=8,
                intermediate_dim=intermediate_dim,
                out_dim=out_dim,
                context_tokens=32,
                device=device,
                operations=operations,
            )
            self.blocks = nn.ModuleList(
                multitalk.WanMultiTalkAttentionBlock(
                    in_dim, out_dim, device=device, operations=operations
                )
                for _ in range(40)
            )

    return MultiTalkModelPatch


class ModelPatchLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ModelPatchLoader",
            category="model/loaders",
            inputs=(InputSpec("name", _ASSET, widget=AssetWidget(kind="model/model-patch")),),
            outputs=(OutputSpec("MODEL_PATCH", _MODEL_PATCH),),
        )

    @classmethod
    def execute(cls, *, name: object) -> Mapping[str, object]:
        if not isinstance(name, (AssetRef, LocalAssetRef)):
            raise TypeError("name must be an AssetRef or LocalAssetRef")
        initialize_inference_process()
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        management = importlib.import_module("dinkster_inference.model_management")
        ops = importlib.import_module("dinkster_inference.ops")
        state_dict, metadata = utils.load_torch_file(
            str(name.local_path()), safe_load=True, return_metadata=True
        )
        dtype = utils.weight_dtype(state_dict)
        offload_device = management.unet_offload_device()
        patch_model: Any

        if "lllite_conditioning1.conv1.weight" in state_dict:
            patch_model = importlib.import_module(
                "dinkster_inference.ldm.anima.lllite"
            ).AnimaLLLite(
                state_dict, metadata, device=offload_device, dtype=dtype, operations=ops.manual_cast
            )
        elif "controlnet_blocks.0.y_rms.weight" in state_dict:
            patch_model = _qwen_blockwise_controlnet(ops.manual_cast)(
                additional_in_dim=state_dict["img_in.weight"].shape[1] - 64,
                device=offload_device,
                dtype=dtype,
            )
        elif "feature_embedder.mid_layer_norm.bias" in state_dict:
            state_dict = utils.state_dict_prefix_replace(
                state_dict, {"feature_embedder.": ""}, filter_keys=True
            )
            patch_model = _siglip_multi_feat_projector(ops.manual_cast)(
                device=offload_device, dtype=dtype
            )
        elif "control_all_x_embedder.2-1.weight" in state_dict:
            state_dict = _z_image_convert(state_dict, torch)
            config: dict[str, object] = {}
            if "control_layers.4.adaLN_modulation.0.weight" not in state_dict:
                config.update(n_control_layers=3, additional_in_dim=17, refiner_control=True)
            if "control_layers.14.adaLN_modulation.0.weight" in state_dict:
                config.update(n_control_layers=15, additional_in_dim=17, refiner_control=True)
                ref_weight = state_dict.get("control_noise_refiner.0.after_proj.weight")
                if ref_weight is not None and torch.count_nonzero(ref_weight) == 0:
                    config["broken"] = True
            patch_model = importlib.import_module(
                "dinkster_inference.ldm.lumina.controlnet"
            ).ZImage_Control(
                device=offload_device, dtype=dtype, operations=ops.manual_cast, **config
            )
        elif (
            "control_img_in.weight" in state_dict
            and "control_blocks.0.img_mlp.out.weight" in state_dict
        ):
            dtype, operations = cls._dit_operations(state_dict, torch, utils, management, ops)
            gate_up = state_dict.get("control_blocks.0.img_mlp.gate_up.weight")
            inner_dim = state_dict["control_img_in.weight"].shape[0]
            hidden_dim = (
                gate_up.shape[0] // 2
                if gate_up is not None
                else state_dict["control_blocks.0.img_mlp.proj.weight"].shape[0]
            )
            patch_model = importlib.import_module(
                "dinkster_inference.ldm.qwen_image21.model"
            ).QwenImage21FunControl(
                num_blocks=cls._block_count(state_dict, "control_blocks.{}.after_proj.weight"),
                control_in_dim=129,
                inner_dim=inner_dim,
                attention_head_dim=state_dict["control_blocks.0.attn.norm_q.weight"].shape[0],
                mlp_ratio=hidden_dim // inner_dim,
                fused_mlp=gate_up is not None,
                operations=operations,
                device=offload_device,
                dtype=dtype,
            )
        elif importlib.import_module(
            "dinkster_inference.ldm.minimax.controlnet"
        ).is_minimax_h3_fun_state_dict(state_dict):
            minimax = importlib.import_module("dinkster_inference.ldm.minimax.controlnet")
            dtype, operations = cls._dit_operations(state_dict, torch, utils, management, ops)
            num_blocks = cls._block_count(state_dict, "control_blocks.{}.after_proj.weight")
            injection_layers = tuple(range(0, 50, 50 // num_blocks))
            if metadata is not None and "control_blocks_places" in metadata:
                injection_layers = tuple(json.loads(metadata["control_blocks_places"]))
                if len(injection_layers) != num_blocks:
                    raise ValueError(
                        "MiniMax H3 Fun control_blocks_places metadata does not match "
                        "the checkpoint"
                    )
            qkv = state_dict["control_blocks.0.attn.qkv_proj.weight"]
            head_dim = state_dict["control_blocks.0.attn.q_norm.weight"].shape[0]
            curves = (
                metadata is not None and metadata.get("minimax_h3_fun_controlnet") == "adaln_basis"
            )
            patch_model = minimax.MiniMaxH3FunControl(
                control_in_dim=49,
                injection_layers=injection_layers,
                inpaint_post_norm=metadata is not None
                and metadata.get("inpaint_masked_pixel_mode") == "post_norm",
                hidden_size=state_dict["control_proj_in.weight"].shape[0],
                num_attention_heads=qkv.shape[0] // (3 * head_dim),
                attention_head_dim=head_dim,
                ffn_hidden_size=state_dict["control_blocks.0.mlp.fc1.weight"].shape[0] // 2,
                time_embed_dim=8 if curves else 2688,
                use_adaln_curves=curves,
                operations=operations,
                device=offload_device,
                dtype=dtype,
            )
            patch_model.requires_grad_(False)
        elif "controlnet_patch_embedding.weight" in state_dict:
            replacements = {
                ".self_attn.to_q.": ".self_attn.q.",
                ".self_attn.to_k.": ".self_attn.k.",
                ".self_attn.to_v.": ".self_attn.v.",
                ".self_attn.to_out.0.": ".self_attn.o.",
            }
            state_dict = {
                cls._replace_key(key, replacements): value for key, value in state_dict.items()
            }
            conv_out_dim = state_dict["controlnet_patch_embedding.weight"].shape[0]
            dim = state_dict.get(
                "proj_in.weight", state_dict["controlnet_patch_embedding.weight"]
            ).shape[0]
            patch_model = importlib.import_module(
                "dinkster_inference.ldm.wan.uni3c"
            ).WanUni3CControlnet(
                in_channels=state_dict["controlnet_patch_embedding.weight"].shape[1],
                conv_out_dim=conv_out_dim,
                dim=dim,
                ffn_dim=state_dict["controlnet_blocks.0.ffn.0.bias"].shape[0],
                num_layers=sum(
                    key.startswith("proj_out.") and key.endswith(".weight") for key in state_dict
                ),
                time_embed_dim=state_dict["controlnet_blocks.0.norm1.linear.weight"].shape[1],
                out_proj_dim=state_dict["proj_out.0.weight"].shape[0],
                add_channels=state_dict["controlnet_mask_embedding.mask_proj.0.weight"].shape[1],
                mid_channels=state_dict["controlnet_mask_embedding.mask_proj.0.weight"].shape[0],
                device=offload_device,
                dtype=dtype,
                operations=ops.manual_cast,
            )
        elif any(
            key.endswith("duration_head.attention_pooler.query_tokens") for key in state_dict
        ) or ("attention_pooler.query_tokens" in state_dict):
            duration = importlib.import_module("dinkster_inference.ldm.lightricks.duration_head")
            state_dict = {
                key: value.float()
                for key, value in duration.normalize_state_dict(state_dict).items()
            }
            patch_model = duration.DurationHead()
        elif "audio_proj.proj1.weight" in state_dict:
            multitalk = importlib.import_module("dinkster_inference.ldm.wan.model_multitalk")
            patch_model = _multi_talk_patch(multitalk, ops.manual_cast)(
                in_dim=state_dict["blocks.0.audio_cross_attn.proj.weight"].shape[0],
                intermediate_dim=state_dict["audio_proj.proj1.weight"].shape[0],
                out_dim=state_dict["audio_proj.norm.weight"].shape[0],
                device=offload_device,
            )
        elif (
            "model.control_model.input_hint_block.0.weight" in state_dict
            or "control_model.input_hint_block.0.weight" in state_dict
        ):
            prefixes = (
                {
                    "model.control_model.": "control_model.",
                    "model.diffusion_model.project_modules.": "project_modules.",
                }
                if "model.control_model.input_hint_block.0.weight" in state_dict
                else {"control_model.": "control_model.", "project_modules.": "project_modules."}
            )
            denoise_prefix = "first_stage_model.denoise_encoder."
            denoise = {
                key[len(denoise_prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(denoise_prefix)
            }
            state_dict = utils.state_dict_prefix_replace(state_dict, prefixes, filter_keys=True)
            state_dict.pop("control_model.mask_LQ", None)
            patch_model = importlib.import_module(
                "dinkster_inference.ldm.supir.supir_modules"
            ).SUPIR(device=offload_device, dtype=dtype, operations=ops.manual_cast)
            if denoise:
                patch_model.denoise_encoder_sd = denoise
        else:
            raise ValueError("unrecognized model patch checkpoint")

        patcher = importlib.import_module("dinkster_inference.model_patcher").CoreModelPatcher(
            patch_model,
            load_device=management.get_torch_device(),
            offload_device=offload_device,
            fast_disk=importlib.import_module("dinkster_inference.storage").state_dict_fast_disk(
                state_dict
            ),
        )
        patch_model.load_state_dict(state_dict, assign=patcher.is_dynamic())
        return cls.outputs(MODEL_PATCH=patcher)

    @staticmethod
    def _block_count(state_dict: Mapping[str, object], pattern: str) -> int:
        count = 0
        while pattern.format(count) in state_dict:
            count += 1
        return count

    @staticmethod
    def _replace_key(key: str, replacements: Mapping[str, str]) -> str:
        for source, target in replacements.items():
            key = key.replace(source, target)
        return key

    @staticmethod
    def _dit_operations(
        state_dict: Mapping[str, object], torch: Any, utils: Any, management: Any, ops: Any
    ) -> tuple[Any, Any]:
        quant = utils.detect_layer_quantization(state_dict, "")
        if quant is not None:
            return torch.bfloat16, ops.mixed_precision_ops(quant, torch.bfloat16)
        dtype = management.unet_dtype(
            model_params=-1,
            supported_dtypes=[torch.bfloat16, torch.float32],
            weight_dtype=utils.weight_dtype(state_dict),
        )
        manual = management.unet_manual_cast(
            dtype,
            management.get_torch_device(),
            supported_dtypes=[torch.bfloat16, torch.float32],
        )
        return dtype, ops.pick_operations(dtype, manual, load_device=management.get_torch_device())


MODEL_PATCH_LOADER_NODES = (ModelPatchLoader,)
