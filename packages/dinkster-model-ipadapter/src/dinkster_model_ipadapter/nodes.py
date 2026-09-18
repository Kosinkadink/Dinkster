"""Native standard SD1.5 IP-Adapter nodes."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_FLOAT,
    AssetWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

ASSET = TypeExpr.concrete("dinkster.asset")
MODEL = TypeExpr.concrete("dinkster.model")
IMAGE = TypeExpr.concrete("dinkster.image")
MASK = TypeExpr.concrete("dinkster.mask")
IPADAPTER = TypeExpr.concrete("dinkster.sd15-ipadapter")
FLOAT = TypeExpr.concrete(CORE_FLOAT)


class LoadSD15IPAdapter(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_sd15_ipadapter",
            display_name="Load SD1.5 IP-Adapter",
            category="model/loaders/ipadapter",
            description="Loads the standard SD1.5 IP-Adapter and its CLIP ViT-H encoder.",
            inputs=(
                InputSpec(
                    "adapter",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/ipadapter",
                    ),
                ),
                InputSpec(
                    "clip_vision",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/clip-vision",
                    ),
                ),
            ),
            outputs=(OutputSpec("ipadapter", IPADAPTER),),
            search_terms=("ip-adapter", "sd1.5", "reference image"),
        )

    @classmethod
    def execute(cls, adapter: object, clip_vision: object) -> Mapping[str, object]:
        from .provider import execute_load_sd15_ipadapter

        return execute_load_sd15_ipadapter(adapter=adapter, clip_vision=clip_vision)


class ApplySD15IPAdapter(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_sd15_ipadapter",
            display_name="Apply SD1.5 IP-Adapter",
            category="model/conditioning/ipadapter",
            description="Applies one reference image to a native SD1.5 model.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("ipadapter", IPADAPTER),
                InputSpec("image", IMAGE),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-1.0, max=3.0, step=0.05),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("mask", MASK, required=False, default=None),
            ),
            outputs=(OutputSpec("model", MODEL),),
            search_terms=("ip-adapter", "sd1.5", "reference image", "image prompt"),
        )

    @classmethod
    def execute(
        cls,
        model: object,
        ipadapter: object,
        image: object,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        mask: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_apply_sd15_ipadapter

        return execute_apply_sd15_ipadapter(
            model=model,
            ipadapter=ipadapter,
            image=image,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
            mask=mask,
        )


IPADAPTER_MODEL_NODES: tuple[type[Node], ...] = (
    LoadSD15IPAdapter,
    ApplySD15IPAdapter,
)
IPADAPTER_MODEL_NODE_IDS = tuple(node.schema().node_type for node in IPADAPTER_MODEL_NODES)

__all__ = ["IPADAPTER_MODEL_NODE_IDS", "IPADAPTER_MODEL_NODES"]
