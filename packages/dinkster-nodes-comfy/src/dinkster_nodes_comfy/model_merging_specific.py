"""Model-family-specific block merge schemas."""

from __future__ import annotations

from dinkster_api.v1 import NodeSchema

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
        return block_merge_schema(
            "ModelMergeSD1",
            _SD1_BLOCK_NAMES,
            category=_CATEGORY,
        )


class ModelMergeSD2(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeSD2",
            _SD1_BLOCK_NAMES,
            category=_CATEGORY,
        )


class ModelMergeSDXL(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeSDXL",
            (
                "time_embed.",
                "label_emb.",
                *_numbered("input_blocks.", 9, suffix=""),
                *_numbered("middle_block.", 3, suffix=""),
                *_numbered("output_blocks.", 9, suffix=""),
                "out.",
            ),
            category=_CATEGORY,
        )


class ModelMergeSD3_2B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeSD3_2B",
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
        )


class ModelMergeAuraflow(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeAuraflow",
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
        )


class ModelMergeFlux1(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeFlux1",
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
        )


class ModelMergeSD35_Large(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeSD35_Large",
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
        )


class ModelMergeMochiPreview(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeMochiPreview",
            (
                "pos_frequencies.",
                "t_embedder.",
                "t5_y_embedder.",
                "t5_yproj.",
                *_numbered("blocks.", 48),
                "final_layer.",
            ),
            category=_CATEGORY,
        )


class ModelMergeLTXV(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeLTXV",
            (
                "patchify_proj.",
                "adaln_single.",
                "caption_projection.",
                *_numbered("transformer_blocks.", 28),
                "scale_shift_table",
                "proj_out.",
            ),
            category=_CATEGORY,
        )


class ModelMergeCosmos7B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeCosmos7B",
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
        )


class ModelMergeCosmos14B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeCosmos14B",
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
        )


class ModelMergeWAN2_1(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeWAN2_1",
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
        )


class ModelMergeCosmosPredict2_2B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeCosmosPredict2_2B",
            (
                "pos_embedder.",
                "x_embedder.",
                "t_embedder.",
                "t_embedding_norm.",
                *_numbered("blocks.", 28),
                "final_layer.",
            ),
            category=_CATEGORY,
        )


class ModelMergeCosmosPredict2_14B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeCosmosPredict2_14B",
            (
                "pos_embedder.",
                "x_embedder.",
                "t_embedder.",
                "t_embedding_norm.",
                *_numbered("blocks.", 36),
                "final_layer.",
            ),
            category=_CATEGORY,
        )


class ModelMergeQwenImage(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeQwenImage",
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
        )


class ModelMergeKrea2(ModelMergeBlocks):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema(
            "ModelMergeKrea2",
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
        )


MODEL_SPECIFIC_MERGE_NODES = (
    ModelMergeSD1,
    ModelMergeSD2,
    ModelMergeSDXL,
    ModelMergeSD3_2B,
    ModelMergeAuraflow,
    ModelMergeFlux1,
    ModelMergeSD35_Large,
    ModelMergeMochiPreview,
    ModelMergeLTXV,
    ModelMergeCosmos7B,
    ModelMergeCosmos14B,
    ModelMergeWAN2_1,
    ModelMergeCosmosPredict2_2B,
    ModelMergeCosmosPredict2_14B,
    ModelMergeQwenImage,
    ModelMergeKrea2,
)
