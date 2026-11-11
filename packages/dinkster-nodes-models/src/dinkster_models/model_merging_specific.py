"""Model-family-specific block merge schemas."""

from __future__ import annotations

from dinkster_api.v1 import NodeSchema, with_identity_replacement

from .model_merging import ModelMergeBlocks, block_merge_schema

_CATEGORY = "model/merging/model specific"


def _numbered(prefix: str, count: int, *, suffix: str = ".") -> tuple[str, ...]:
    return tuple(f"{prefix}{index}{suffix}" for index in range(count))


_SD1_BLOCK_NAMES = (
    "time_embed.",
    "label_emb.",
    *_numbered("input_blocks.", 12),
    *_numbered("middle_block.", 3),
    *_numbered("output_blocks.", 12),
    "out.",
)


class ModelMergeSD1(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_sd1",
                _SD1_BLOCK_NAMES,
                category=_CATEGORY,
            ),
            "ModelMergeSD1",
        )


class ModelMergeSD2(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_sd2",
                _SD1_BLOCK_NAMES,
                category=_CATEGORY,
            ),
            "ModelMergeSD2",
        )


class ModelMergeSDXL(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_sdxl",
                (
                    "time_embed.",
                    "label_emb.",
                    *_numbered("input_blocks.", 9, suffix=""),
                    *_numbered("middle_block.", 3, suffix=""),
                    *_numbered("output_blocks.", 9, suffix=""),
                    "out.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeSDXL",
        )


class ModelMergeSD3_2B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_sd3_2_b",
                (
                    "pos_embed.",
                    "x_embedder.",
                    "context_embedder.",
                    "y_embedder.",
                    "t_embedder.",
                    *_numbered("joint_blocks.", 24),
                    "final_layer.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeSD3_2B",
        )


class ModelMergeAuraflow(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_auraflow",
                (
                    "init_x_linear.",
                    "positional_encoding",
                    "cond_seq_linear.",
                    "register_tokens",
                    "t_embedder.",
                    *_numbered("double_layers.", 4),
                    *_numbered("single_layers.", 32),
                    "modF.",
                    "final_linear.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeAuraflow",
        )


class ModelMergeFlux1(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_flux1",
                (
                    "img_in.",
                    "time_in.",
                    "guidance_in",
                    "vector_in.",
                    "txt_in.",
                    *_numbered("double_blocks.", 19),
                    *_numbered("single_blocks.", 38),
                    "final_layer.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeFlux1",
        )


class ModelMergeSD35_Large(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_sd35_large",
                (
                    "pos_embed.",
                    "x_embedder.",
                    "context_embedder.",
                    "y_embedder.",
                    "t_embedder.",
                    *_numbered("joint_blocks.", 38),
                    "final_layer.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeSD35_Large",
        )


class ModelMergeMochiPreview(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_mochi_preview",
                (
                    "pos_frequencies.",
                    "t_embedder.",
                    "t5_y_embedder.",
                    "t5_yproj.",
                    *_numbered("blocks.", 48),
                    "final_layer.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeMochiPreview",
        )


class ModelMergeLTXV(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_ltxv",
                (
                    "patchify_proj.",
                    "adaln_single.",
                    "caption_projection.",
                    *_numbered("transformer_blocks.", 28),
                    "scale_shift_table",
                    "proj_out.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeLTXV",
        )


class ModelMergeCosmos7B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_cosmos7_b",
                (
                    "pos_embedder.",
                    "extra_pos_embedder.",
                    "x_embedder.",
                    "t_embedder.",
                    "affline_norm.",
                    *_numbered("blocks.block", 28),
                    "final_layer.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeCosmos7B",
        )


class ModelMergeCosmos14B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_cosmos14_b",
                (
                    "pos_embedder.",
                    "extra_pos_embedder.",
                    "x_embedder.",
                    "t_embedder.",
                    "affline_norm.",
                    *_numbered("blocks.block", 36),
                    "final_layer.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeCosmos14B",
        )


class ModelMergeWAN2_1(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_wan2_1",
                (
                    "patch_embedding.",
                    "time_embedding.",
                    "time_projection.",
                    "text_embedding.",
                    "img_emb.",
                    *_numbered("blocks.", 40),
                    "head.",
                ),
                category=_CATEGORY,
                description=(
                    "1.3B model has 30 blocks, 14B model has 40 blocks. "
                    "Image to video model has the extra img_emb."
                ),
            ),
            "ModelMergeWAN2_1",
        )


class ModelMergeCosmosPredict2_2B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_cosmos_predict2_2_b",
                (
                    "pos_embedder.",
                    "x_embedder.",
                    "t_embedder.",
                    "t_embedding_norm.",
                    *_numbered("blocks.", 28),
                    "final_layer.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeCosmosPredict2_2B",
        )


class ModelMergeCosmosPredict2_14B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_cosmos_predict2_14_b",
                (
                    "pos_embedder.",
                    "x_embedder.",
                    "t_embedder.",
                    "t_embedding_norm.",
                    *_numbered("blocks.", 36),
                    "final_layer.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeCosmosPredict2_14B",
        )


class ModelMergeQwenImage(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_qwen_image",
                (
                    "pos_embeds.",
                    "img_in.",
                    "txt_norm.",
                    "txt_in.",
                    "time_text_embed.",
                    *_numbered("transformer_blocks.", 60),
                    "proj_out.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeQwenImage",
        )


class ModelMergeKrea2(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            block_merge_schema(
                "dinkster.models.model_merge_krea2",
                (
                    "first.",
                    "tmlp.",
                    "txtmlp.",
                    "tproj.",
                    *_numbered("txtfusion.layerwise_blocks.", 2),
                    "txtfusion.projector.",
                    *_numbered("txtfusion.refiner_blocks.", 2),
                    *_numbered("blocks.", 28),
                    "last.",
                ),
                category=_CATEGORY,
            ),
            "ModelMergeKrea2",
        )
