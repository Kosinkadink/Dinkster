"""Native execution for the stable line and edge preprocessor schemas."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    prepare_image_array_encoding,
)

IMAGE_TYPE = "dinkster.image"
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)


def _provider_input(choice: str) -> InputSpec:
    return InputSpec(
        "provider",
        COMBO,
        required=False,
        widget=ComboWidget(remote_route=f"/api/choices/{choice}"),
        hidden=True,
    )


def _number_input(
    input_id: str,
    type_expr: TypeExpr,
    default: int | float,
    *,
    minimum: int | float,
    maximum: int | float,
    step: int | float,
) -> InputSpec:
    return InputSpec(
        input_id,
        type_expr,
        required=False,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _resolution_input(*, default: int = 512, step: int = 64) -> InputSpec:
    return _number_input(
        "resolution",
        INT,
        default,
        minimum=64,
        maximum=16_384,
        step=step,
    )


class ModelEdgePreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.model_edges",
            display_name="Preprocess Model Edges",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _provider_input("dinkster.preprocess.model_edges.providers"),
                InputSpec("safe", BOOLEAN, required=False, default=True),
                InputSpec("scribble", BOOLEAN, required=False, default=False),
                _resolution_input(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("HED", "soft edge", "fake scribble", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        provider: str,
        safe: bool = True,
        scribble: bool = False,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        del provider
        from .model import execute_hed

        return cls.outputs(
            image=execute_hed(image, safe=safe, scribble=scribble, resolution=resolution)
        )


class RealisticLineartPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.lineart_realistic",
            display_name="Preprocess Realistic Line Art",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _provider_input("dinkster.preprocess.lineart_realistic.providers"),
                InputSpec("coarse", BOOLEAN, required=False, default=False),
                _resolution_input(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("realistic lineart", "coarse lineart", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        provider: str,
        coarse: bool = False,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        del provider
        from .lineart import execute_realistic

        return cls.outputs(image=execute_realistic(image, coarse=coarse, resolution=resolution))


class AnimeLineartPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.lineart_anime",
            display_name="Preprocess Anime Line Art",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _provider_input("dinkster.preprocess.lineart_anime.providers"),
                _resolution_input(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("anime lineart", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        provider: str,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        del provider
        from .lineart import execute_anime

        return cls.outputs(image=execute_anime(image, resolution=resolution))


class MangaLineartPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.lineart_manga",
            display_name="Preprocess Manga Line Art",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _provider_input("dinkster.preprocess.lineart_manga.providers"),
                _resolution_input(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("manga lineart", "anime denoise", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        provider: str,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        del provider
        from .lineart import execute_manga

        return cls.outputs(image=execute_manga(image, resolution=resolution))


class AnyLinePreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.anyline",
            display_name="Preprocess AnyLine",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _provider_input("dinkster.preprocess.anyline.providers"),
                InputSpec(
                    "merge_with_lineart",
                    COMBO,
                    required=False,
                    default="lineart_standard",
                    widget=ComboWidget(
                        options=(
                            "lineart_standard",
                            "lineart_realisitic",
                            "lineart_anime",
                            "manga_line",
                        )
                    ),
                ),
                _resolution_input(default=1280, step=8),
                _number_input(
                    "lineart_lower_bound",
                    FLOAT,
                    0.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                ),
                _number_input(
                    "lineart_upper_bound",
                    FLOAT,
                    1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                ),
                _number_input(
                    "object_min_size",
                    INT,
                    36,
                    minimum=1,
                    maximum=16_384,
                    step=1,
                ),
                _number_input(
                    "object_connectivity",
                    INT,
                    1,
                    minimum=1,
                    maximum=16_384,
                    step=1,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("AnyLine", "MTEED", "lineart", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        provider: str,
        merge_with_lineart: str = "lineart_standard",
        resolution: int = 1280,
        lineart_lower_bound: float = 0.0,
        lineart_upper_bound: float = 1.0,
        object_min_size: int = 36,
        object_connectivity: int = 1,
    ) -> Mapping[str, object]:
        del provider
        from .anyline import execute_anyline

        return cls.outputs(
            image=execute_anyline(
                image,
                merge_with_lineart=merge_with_lineart,
                resolution=resolution,
                lineart_lower_bound=lineart_lower_bound,
                lineart_upper_bound=lineart_upper_bound,
                object_min_size=object_min_size,
                object_connectivity=object_connectivity,
            )
        )


class TEEDPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.teed",
            display_name="Preprocess TEED Edges",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _provider_input("dinkster.preprocess.teed.providers"),
                _number_input("safe_steps", INT, 2, minimum=0, maximum=10, step=1),
                _resolution_input(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("TEED", "soft edge", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        provider: str,
        safe_steps: int = 2,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        del provider
        from .teed import execute_teed

        return cls.outputs(image=execute_teed(image, safe_steps=safe_steps, resolution=resolution))


class MLSDPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.mlsd",
            display_name="Preprocess M-LSD Lines",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _provider_input("dinkster.preprocess.mlsd.providers"),
                _number_input(
                    "score_threshold",
                    FLOAT,
                    0.1,
                    minimum=0.01,
                    maximum=2.0,
                    step=0.01,
                ),
                _number_input(
                    "distance_threshold",
                    FLOAT,
                    0.1,
                    minimum=0.01,
                    maximum=20.0,
                    step=0.01,
                ),
                _resolution_input(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("M-LSD", "line segment", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        provider: str,
        score_threshold: float = 0.1,
        distance_threshold: float = 0.1,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        del provider
        from .mlsd import execute_mlsd

        return cls.outputs(
            image=execute_mlsd(
                image,
                score_threshold=score_threshold,
                distance_threshold=distance_threshold,
                resolution=resolution,
            )
        )


def register_types(registry: TypeRegistry) -> None:
    if IMAGE_TYPE not in registry:
        registry.register(
            IMAGE_TYPE,
            encode=encode_image_array,
            decode=decode_image_array,
            prepare_buffer_encoding=prepare_image_array_encoding,
            fingerprint=image_array_fingerprint(IMAGE_TYPE),
            meta=image_array_meta,
        )


HED_PROVIDER_NODES: tuple[type[Node], ...] = (
    ModelEdgePreprocessor,
    RealisticLineartPreprocessor,
    AnimeLineartPreprocessor,
    MangaLineartPreprocessor,
    AnyLinePreprocessor,
    TEEDPreprocessor,
    MLSDPreprocessor,
)


__all__ = [
    "HED_PROVIDER_NODES",
    "AnimeLineartPreprocessor",
    "AnyLinePreprocessor",
    "MLSDPreprocessor",
    "MangaLineartPreprocessor",
    "ModelEdgePreprocessor",
    "RealisticLineartPreprocessor",
    "TEEDPreprocessor",
    "register_types",
]
