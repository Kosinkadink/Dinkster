# ruff: noqa: E501
"""Wan partner contracts and schemas pinned to ComfyUI e651b7be."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

from .opspec import (
    BatchMapJoin,
    Check,
    CheckInputs,
    Cond,
    DownloadDecode,
    EncodeMedia,
    FixedField,
    FormatField,
    HttpSyncJson,
    InputBinding,
    MediaConstraints,
    OpSpec,
    ProxyUpload,
    Segment,
    SubmitPoll,
)
from .partner_runtime import run_op, worker_runtime_context

STRING = TypeExpr.concrete(CORE_STRING)
COMBO = TypeExpr.concrete(CORE_COMBO)
INT = TypeExpr.concrete(CORE_INT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
IMAGE = TypeExpr.concrete("comfy.IMAGE")
VIDEO = TypeExpr.concrete("comfy.VIDEO")
AUDIO = TypeExpr.concrete("comfy.AUDIO")

_MAX_SEED = 2147483647


def _validate_seed(seed: int) -> None:
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")


def _validate_duration(duration: int, minimum: int) -> None:
    if not minimum <= duration <= 15:
        raise ValueError(f"duration must be between {minimum} and 15")


@dataclass(frozen=True, kw_only=True)
class Text2ImageInputField:
    prompt: str
    negative_prompt: str | None = None


@dataclass(frozen=True, kw_only=True)
class Image2ImageInputField:
    prompt: str
    negative_prompt: str | None = None
    images: list[str]

    def __post_init__(self) -> None:
        if not 1 <= len(self.images) <= 2:
            raise ValueError("images must contain between 1 and 2 items")


@dataclass(frozen=True, kw_only=True)
class Text2VideoInputField:
    prompt: str
    negative_prompt: str | None = None
    audio_url: str | None = None


@dataclass(frozen=True, kw_only=True)
class Image2VideoInputField:
    prompt: str
    negative_prompt: str | None = None
    img_url: str
    audio_url: str | None = None


@dataclass(frozen=True, kw_only=True)
class Reference2VideoInputField:
    prompt: str
    negative_prompt: str | None = None
    reference_video_urls: list[str]


@dataclass(frozen=True, kw_only=True)
class Txt2ImageParametersField:
    size: str
    n: int = 1
    seed: int
    prompt_extend: bool = True
    watermark: bool = False

    def __post_init__(self) -> None:
        _validate_seed(self.seed)


@dataclass(frozen=True, kw_only=True)
class Image2ImageParametersField:
    size: str | None = None
    n: int = 1
    seed: int
    watermark: bool = False

    def __post_init__(self) -> None:
        _validate_seed(self.seed)


@dataclass(frozen=True, kw_only=True)
class Text2VideoParametersField:
    size: str
    seed: int
    duration: int = 5
    prompt_extend: bool = True
    watermark: bool = False
    audio: bool = False
    shot_type: str = "single"

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_duration(self.duration, 5)


@dataclass(frozen=True, kw_only=True)
class Image2VideoParametersField:
    resolution: str
    seed: int
    duration: int = 5
    prompt_extend: bool = True
    watermark: bool = False
    audio: bool = False
    shot_type: str = "single"

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_duration(self.duration, 5)


@dataclass(frozen=True, kw_only=True)
class Reference2VideoParametersField:
    size: str
    duration: int = 5
    shot_type: str = "single"
    seed: int
    watermark: bool = False

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_duration(self.duration, 5)


@dataclass(frozen=True, kw_only=True)
class Text2ImageTaskCreationRequest:
    model: str
    input: Text2ImageInputField
    parameters: Txt2ImageParametersField


@dataclass(frozen=True, kw_only=True)
class Image2ImageTaskCreationRequest:
    model: str
    input: Image2ImageInputField
    parameters: Image2ImageParametersField


@dataclass(frozen=True, kw_only=True)
class Text2VideoTaskCreationRequest:
    model: str
    input: Text2VideoInputField
    parameters: Text2VideoParametersField


@dataclass(frozen=True, kw_only=True)
class Image2VideoTaskCreationRequest:
    model: str
    input: Image2VideoInputField
    parameters: Image2VideoParametersField


@dataclass(frozen=True, kw_only=True)
class Reference2VideoTaskCreationRequest:
    model: str
    input: Reference2VideoInputField
    parameters: Reference2VideoParametersField


@dataclass(frozen=True, kw_only=True)
class Wan27MediaItem:
    type: str
    url: str


@dataclass(frozen=True, kw_only=True)
class Wan27ReferenceVideoInputField:
    prompt: str
    negative_prompt: str | None = None
    media: list[Wan27MediaItem]


@dataclass(frozen=True, kw_only=True)
class Wan27ReferenceVideoParametersField:
    resolution: str
    ratio: str | None = None
    duration: int = 5
    watermark: bool = False
    seed: int

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_duration(self.duration, 2)


@dataclass(frozen=True, kw_only=True)
class Wan27ReferenceVideoTaskCreationRequest:
    model: str
    input: Wan27ReferenceVideoInputField
    parameters: Wan27ReferenceVideoParametersField


@dataclass(frozen=True, kw_only=True)
class Wan27ImageToVideoInputField:
    prompt: str | None = None
    negative_prompt: str | None = None
    media: list[Wan27MediaItem]


@dataclass(frozen=True, kw_only=True)
class Wan27ImageToVideoParametersField:
    resolution: str
    duration: int = 5
    prompt_extend: bool = True
    watermark: bool = False
    seed: int

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_duration(self.duration, 2)


@dataclass(frozen=True, kw_only=True)
class Wan27ImageToVideoTaskCreationRequest:
    model: str
    input: Wan27ImageToVideoInputField
    parameters: Wan27ImageToVideoParametersField


@dataclass(frozen=True, kw_only=True)
class Wan27VideoEditInputField:
    prompt: str
    media: list[Wan27MediaItem]


@dataclass(frozen=True, kw_only=True)
class Wan27VideoEditParametersField:
    resolution: str
    ratio: str | None = None
    duration: int | None = 0
    audio_setting: str = "auto"
    watermark: bool = False
    seed: int

    def __post_init__(self) -> None:
        _validate_seed(self.seed)


@dataclass(frozen=True, kw_only=True)
class Wan27VideoEditTaskCreationRequest:
    model: str
    input: Wan27VideoEditInputField
    parameters: Wan27VideoEditParametersField


@dataclass(frozen=True, kw_only=True)
class Wan27Text2VideoParametersField:
    resolution: str
    ratio: str | None = None
    duration: int = 5
    prompt_extend: bool = True
    watermark: bool = False
    seed: int

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_duration(self.duration, 2)


@dataclass(frozen=True, kw_only=True)
class Wan27Text2VideoTaskCreationRequest:
    model: str
    input: Text2VideoInputField
    parameters: Wan27Text2VideoParametersField


@dataclass(frozen=True, kw_only=True)
class TaskCreationOutputField:
    task_id: str
    task_status: str


@dataclass(frozen=True, kw_only=True)
class TaskCreationResponse:
    output: TaskCreationOutputField | None = None
    request_id: str
    code: str | None = None
    message: str | None = None


@dataclass(frozen=True, kw_only=True)
class TaskResult:
    url: str | None = None
    code: str | None = None
    message: str | None = None


@dataclass(frozen=True, kw_only=True)
class ImageTaskStatusOutputField(TaskCreationOutputField):
    task_id: str
    task_status: str
    results: list[TaskResult] | None = None


@dataclass(frozen=True, kw_only=True)
class VideoTaskStatusOutputField(TaskCreationOutputField):
    task_id: str
    task_status: str
    video_url: str | None = None
    code: str | None = None
    message: str | None = None


@dataclass(frozen=True, kw_only=True)
class ImageTaskStatusResponse:
    output: ImageTaskStatusOutputField | None = None
    request_id: str


@dataclass(frozen=True, kw_only=True)
class VideoTaskStatusResponse:
    output: VideoTaskStatusOutputField | None = None
    request_id: str


WAN_CONTRACTS: Mapping[str, type[object]] = {
    "Text2ImageInputField": Text2ImageInputField,
    "Image2ImageInputField": Image2ImageInputField,
    "Text2VideoInputField": Text2VideoInputField,
    "Image2VideoInputField": Image2VideoInputField,
    "Reference2VideoInputField": Reference2VideoInputField,
    "Txt2ImageParametersField": Txt2ImageParametersField,
    "Image2ImageParametersField": Image2ImageParametersField,
    "Text2VideoParametersField": Text2VideoParametersField,
    "Image2VideoParametersField": Image2VideoParametersField,
    "Reference2VideoParametersField": Reference2VideoParametersField,
    "Text2ImageTaskCreationRequest": Text2ImageTaskCreationRequest,
    "Image2ImageTaskCreationRequest": Image2ImageTaskCreationRequest,
    "Text2VideoTaskCreationRequest": Text2VideoTaskCreationRequest,
    "Image2VideoTaskCreationRequest": Image2VideoTaskCreationRequest,
    "Reference2VideoTaskCreationRequest": Reference2VideoTaskCreationRequest,
    "Wan27MediaItem": Wan27MediaItem,
    "Wan27ReferenceVideoInputField": Wan27ReferenceVideoInputField,
    "Wan27ReferenceVideoParametersField": Wan27ReferenceVideoParametersField,
    "Wan27ReferenceVideoTaskCreationRequest": Wan27ReferenceVideoTaskCreationRequest,
    "Wan27ImageToVideoInputField": Wan27ImageToVideoInputField,
    "Wan27ImageToVideoParametersField": Wan27ImageToVideoParametersField,
    "Wan27ImageToVideoTaskCreationRequest": Wan27ImageToVideoTaskCreationRequest,
    "Wan27VideoEditInputField": Wan27VideoEditInputField,
    "Wan27VideoEditParametersField": Wan27VideoEditParametersField,
    "Wan27VideoEditTaskCreationRequest": Wan27VideoEditTaskCreationRequest,
    "Wan27Text2VideoParametersField": Wan27Text2VideoParametersField,
    "Wan27Text2VideoTaskCreationRequest": Wan27Text2VideoTaskCreationRequest,
    "TaskCreationOutputField": TaskCreationOutputField,
    "TaskCreationResponse": TaskCreationResponse,
    "TaskResult": TaskResult,
    "ImageTaskStatusOutputField": ImageTaskStatusOutputField,
    "VideoTaskStatusOutputField": VideoTaskStatusOutputField,
    "ImageTaskStatusResponse": ImageTaskStatusResponse,
    "VideoTaskStatusResponse": VideoTaskStatusResponse,
}


WAN_VIDEO_PATH = "/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis"
SIZE_VALUES = {
    "480p: 1:1 (624x624)": "624*624",
    "480p: 16:9 (832x480)": "832*480",
    "480p: 9:16 (480x832)": "480*832",
    "720p: 1:1 (960x960)": "960*960",
    "720p: 16:9 (1280x720)": "1280*720",
    "720p: 9:16 (720x1280)": "720*1280",
    "720p: 4:3 (1088x832)": "1088*832",
    "720p: 3:4 (832x1088)": "832*1088",
    "1080p: 1:1 (1440x1440)": "1440*1440",
    "1080p: 16:9 (1920x1080)": "1920*1080",
    "1080p: 9:16 (1080x1920)": "1080*1920",
    "1080p: 4:3 (1632x1248)": "1632*1248",
    "1080p: 3:4 (1248x1632)": "1248*1632",
}


class WanNode(Node):
    SPEC: OpSpec

    @classmethod
    async def execute(cls, **inputs: object) -> Mapping[str, object]:
        return await run_op(cls.SPEC, inputs, worker_runtime_context())


def _wan_result(
    source: str, output: str, media_family: str, interval: float
) -> tuple[SubmitPoll, DownloadDecode]:
    url_path = (
        ("output", "results", 0, "url") if media_family == "image" else ("output", "video_url")
    )
    return (
        SubmitPoll(
            "poll",
            source,
            status_path=("output", "task_status"),
            interval=interval,
            path_template="/proxy/wan/api/v1/tasks/{value}",
            path_value_path=("output", "task_id"),
        ),
        DownloadDecode(
            "download",
            "poll",
            output=output,
            media_family=media_family,  # type: ignore[arg-type]
            url_path=url_path,
        ),
    )


def _optional_audio() -> tuple[MediaConstraints, EncodeMedia]:
    return (
        MediaConstraints(
            "validate_audio",
            "audio",
            min_duration=3.0,
            max_duration=29.0,
            duration_media="audio",
            optional=True,
        ),
        EncodeMedia(
            "encoded_audio",
            "audio",
            media_family="audio",
            format="MP3",
            output="data_url",
            optional=True,
        ),
    )


class WanTextToImageApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.text-to-image",
            display_name="Wan Text to Image",
            category="partner/image/Wan",
            description="Generates an image based on a text prompt.",
            inputs=(
                InputSpec(
                    "model",
                    COMBO,
                    default="wan2.5-t2i-preview",
                    required=True,
                    doc="Model to use.",
                    widget=ComboWidget(options=("wan2.5-t2i-preview",)),
                ),
                InputSpec(
                    "prompt",
                    STRING,
                    default="",
                    required=True,
                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "negative_prompt",
                    STRING,
                    default="",
                    required=False,
                    doc="Negative prompt describing what to avoid.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "width", INT, default=1024, required=False, widget=NumberWidget(768, 1440, 32)
                ),
                InputSpec(
                    "height", INT, default=1024, required=False, widget=NumberWidget(768, 1440, 32)
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=False,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "prompt_extend",
                    BOOLEAN,
                    default=True,
                    required=False,
                    doc="Whether to enhance the prompt with AI assistance.",
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=False,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("image", IMAGE),),
            aliases=("WanTextToImageApi",),
            combos=(),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            HttpSyncJson(
                "submit",
                "/proxy/wan/api/v1/services/aigc/text2image/image-synthesis",
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "prompt"),
                    InputBinding("input.negative_prompt", "negative_prompt"),
                    InputBinding("parameters.seed", "seed"),
                    InputBinding("parameters.prompt_extend", "prompt_extend"),
                    InputBinding("parameters.watermark", "watermark"),
                ),
                fixed=(FixedField("parameters.n", 1),),
                formatted=(FormatField("parameters.size", "{width}*{height}"),),
            ),
            *_wan_result("submit", "image", "image", 3),
        )
    )


class WanImageToImageApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.image-to-image",
            display_name="Wan Image to Image",
            category="partner/image/Wan",
            description="Generates an image from one or two input images and a text prompt. The output image is currently fixed at 1.6 MP, and its aspect ratio matches the input image(s).",
            inputs=(
                InputSpec(
                    "model",
                    COMBO,
                    default="wan2.5-i2i-preview",
                    required=True,
                    doc="Model to use.",
                    widget=ComboWidget(options=("wan2.5-i2i-preview",)),
                ),
                InputSpec(
                    "image",
                    IMAGE,
                    default=None,
                    required=True,
                    doc="Single-image editing or multi-image fusion. Maximum 2 images.",
                ),
                InputSpec(
                    "prompt",
                    STRING,
                    default="",
                    required=True,
                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "negative_prompt",
                    STRING,
                    default="",
                    required=False,
                    doc="Negative prompt describing what to avoid.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=False,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=False,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("image", IMAGE),),
            aliases=("WanImageToImageApi",),
            combos=(),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_images",
                (
                    Check(
                        "Expected 1 or 2 input images, but got {count}.",
                        require=(Cond("image", "count_ge", 1), Cond("image", "count_le", 2)),
                    ),
                ),
            ),
            EncodeMedia(
                "encoded_images",
                "image",
                output="data_url",
                max_pixels=4096 * 4096,
                batch_targets=("image_1", "image_2"),
            ),
            BatchMapJoin(
                "images",
                segments=(Segment("encoded_images", wrap_key=None, mode="mapping_values"),),
            ),
            HttpSyncJson(
                "submit",
                "/proxy/wan/api/v1/services/aigc/image2image/image-synthesis",
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "prompt"),
                    InputBinding("input.negative_prompt", "negative_prompt"),
                    InputBinding("input.images", "images"),
                    InputBinding("parameters.seed", "seed"),
                    InputBinding("parameters.watermark", "watermark"),
                ),
                fixed=(FixedField("parameters.n", 1),),
            ),
            *_wan_result("submit", "image", "image", 4),
        )
    )


class WanTextToVideoApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.text-to-video",
            display_name="Wan Text to Video",
            category="partner/video/Wan",
            description="Generates a video based on a text prompt.",
            inputs=(
                InputSpec(
                    "model",
                    COMBO,
                    default="wan2.6-t2v",
                    required=True,
                    doc="Model to use.",
                    widget=ComboWidget(options=("wan2.5-t2v-preview", "wan2.6-t2v")),
                ),
                InputSpec(
                    "prompt",
                    STRING,
                    default="",
                    required=True,
                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "negative_prompt",
                    STRING,
                    default="",
                    required=False,
                    doc="Negative prompt describing what to avoid.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "size",
                    COMBO,
                    default="720p: 1:1 (960x960)",
                    required=False,
                    widget=ComboWidget(
                        options=(
                            "480p: 1:1 (624x624)",
                            "480p: 16:9 (832x480)",
                            "480p: 9:16 (480x832)",
                            "720p: 1:1 (960x960)",
                            "720p: 16:9 (1280x720)",
                            "720p: 9:16 (720x1280)",
                            "720p: 4:3 (1088x832)",
                            "720p: 3:4 (832x1088)",
                            "1080p: 1:1 (1440x1440)",
                            "1080p: 16:9 (1920x1080)",
                            "1080p: 9:16 (1080x1920)",
                            "1080p: 4:3 (1632x1248)",
                            "1080p: 3:4 (1248x1632)",
                        )
                    ),
                ),
                InputSpec(
                    "duration",
                    INT,
                    default=5,
                    required=False,
                    doc="A 15-second duration is available only for the Wan 2.6 model.",
                    widget=NumberWidget(5, 15, 5),
                ),
                InputSpec(
                    "audio",
                    AUDIO,
                    default=None,
                    required=False,
                    doc="Audio must contain a clear, loud voice, without extraneous noise or background music.",
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=False,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "generate_audio",
                    BOOLEAN,
                    default=False,
                    required=False,
                    doc="If no audio input is provided, generate audio automatically.",
                ),
                InputSpec(
                    "prompt_extend",
                    BOOLEAN,
                    default=True,
                    required=False,
                    doc="Whether to enhance the prompt with AI assistance.",
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=False,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
                InputSpec(
                    "shot_type",
                    COMBO,
                    default="single",
                    required=False,
                    doc="Specifies the shot type for the generated video, that is, whether the video is a single continuous shot or multiple shots with cuts. This parameter takes effect only when prompt_extend is True.",
                    widget=ComboWidget(options=("single", "multi")),
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("WanTextToVideoApi",),
            combos=(),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_model_options",
                (
                    Check(
                        "The Wan 2.6 model does not support 480p.",
                        when=(Cond("model", "eq", "wan2.6-t2v"),),
                        require=tuple(
                            Cond("size", "ne", size)
                            for size in (
                                "480p: 1:1 (624x624)",
                                "480p: 16:9 (832x480)",
                                "480p: 9:16 (480x832)",
                            )
                        ),
                    ),
                    Check(
                        "A 15-second duration is supported only by the Wan 2.6 model.",
                        when=(Cond("duration", "eq", 15),),
                        require=(Cond("model", "eq", "wan2.6-t2v"),),
                    ),
                ),
            ),
            *_optional_audio(),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "prompt"),
                    InputBinding("input.negative_prompt", "negative_prompt"),
                    InputBinding("input.audio_url", "encoded_audio"),
                    InputBinding("parameters.size", "size", value_map=SIZE_VALUES),
                    InputBinding("parameters.duration", "duration"),
                    InputBinding("parameters.seed", "seed"),
                    InputBinding("parameters.audio", "generate_audio"),
                    InputBinding("parameters.prompt_extend", "prompt_extend"),
                    InputBinding("parameters.watermark", "watermark"),
                    InputBinding("parameters.shot_type", "shot_type"),
                ),
            ),
            *_wan_result("submit", "video", "video", 6),
        )
    )


class WanImageToVideoApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.image-to-video",
            display_name="Wan Image to Video",
            category="partner/video/Wan",
            description="Generates a video from the first frame and a text prompt.",
            inputs=(
                InputSpec(
                    "model",
                    COMBO,
                    default="wan2.6-i2v",
                    required=True,
                    doc="Model to use.",
                    widget=ComboWidget(options=("wan2.5-i2v-preview", "wan2.6-i2v")),
                ),
                InputSpec("image", IMAGE, default=None, required=True),
                InputSpec(
                    "prompt",
                    STRING,
                    default="",
                    required=True,
                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "negative_prompt",
                    STRING,
                    default="",
                    required=False,
                    doc="Negative prompt describing what to avoid.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "resolution",
                    COMBO,
                    default="720P",
                    required=False,
                    widget=ComboWidget(options=("480P", "720P", "1080P")),
                ),
                InputSpec(
                    "duration",
                    INT,
                    default=5,
                    required=False,
                    doc="Duration 15 available only for WAN2.6 model.",
                    widget=NumberWidget(5, 15, 5),
                ),
                InputSpec(
                    "audio",
                    AUDIO,
                    default=None,
                    required=False,
                    doc="Audio must contain a clear, loud voice, without extraneous noise or background music.",
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=False,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "generate_audio",
                    BOOLEAN,
                    default=False,
                    required=False,
                    doc="If no audio input is provided, generate audio automatically.",
                ),
                InputSpec(
                    "prompt_extend",
                    BOOLEAN,
                    default=True,
                    required=False,
                    doc="Whether to enhance the prompt with AI assistance.",
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=False,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
                InputSpec(
                    "shot_type",
                    COMBO,
                    default="single",
                    required=False,
                    doc="Specifies the shot type for the generated video, that is, whether the video is a single continuous shot or multiple shots with cuts. This parameter takes effect only when prompt_extend is True.",
                    widget=ComboWidget(options=("single", "multi")),
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("WanImageToVideoApi",),
            combos=(),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_image_and_options",
                (
                    Check(
                        "Exactly one input image is required.",
                        require=(Cond("image", "count_eq", 1),),
                    ),
                    Check(
                        "The Wan 2.6 model does not support 480P.",
                        when=(Cond("model", "eq", "wan2.6-i2v"),),
                        require=(Cond("resolution", "ne", "480P"),),
                    ),
                    Check(
                        "A 15-second duration is supported only by the Wan 2.6 model.",
                        when=(Cond("duration", "eq", 15),),
                        require=(Cond("model", "eq", "wan2.6-i2v"),),
                    ),
                ),
            ),
            EncodeMedia("encoded_image", "image", output="data_url", max_pixels=2000 * 2000),
            *_optional_audio(),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "prompt"),
                    InputBinding("input.negative_prompt", "negative_prompt"),
                    InputBinding("input.img_url", "encoded_image"),
                    InputBinding("input.audio_url", "encoded_audio"),
                    InputBinding("parameters.resolution", "resolution"),
                    InputBinding("parameters.duration", "duration"),
                    InputBinding("parameters.seed", "seed"),
                    InputBinding("parameters.audio", "generate_audio"),
                    InputBinding("parameters.prompt_extend", "prompt_extend"),
                    InputBinding("parameters.watermark", "watermark"),
                    InputBinding("parameters.shot_type", "shot_type"),
                ),
            ),
            *_wan_result("submit", "video", "video", 6),
        )
    )


class WanReferenceVideoApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.reference-video",
            display_name="Wan Reference to Video",
            category="partner/video/Wan",
            description="Use the character and voice from input videos, combined with a prompt, to generate a new video that maintains character consistency.",
            inputs=(
                InputSpec(
                    "model",
                    COMBO,
                    default="wan2.6-r2v",
                    required=True,
                    widget=ComboWidget(options=("wan2.6-r2v",)),
                ),
                InputSpec(
                    "prompt",
                    STRING,
                    default="",
                    required=True,
                    doc="Prompt describing the elements and visual features. Supports English and Chinese. Use identifiers such as `character1` and `character2` to refer to the reference characters.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "negative_prompt",
                    STRING,
                    default="",
                    required=True,
                    doc="Negative prompt describing what to avoid.",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "size",
                    COMBO,
                    default="720p: 1:1 (960x960)",
                    required=True,
                    widget=ComboWidget(
                        options=(
                            "720p: 1:1 (960x960)",
                            "720p: 16:9 (1280x720)",
                            "720p: 9:16 (720x1280)",
                            "720p: 4:3 (1088x832)",
                            "720p: 3:4 (832x1088)",
                            "1080p: 1:1 (1440x1440)",
                            "1080p: 16:9 (1920x1080)",
                            "1080p: 9:16 (1080x1920)",
                            "1080p: 4:3 (1632x1248)",
                            "1080p: 3:4 (1248x1632)",
                        )
                    ),
                ),
                InputSpec("duration", INT, default=5, required=True, widget=NumberWidget(5, 10, 5)),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "shot_type",
                    COMBO,
                    default="single",
                    required=True,
                    doc="Specifies the shot type for the generated video, that is, whether the video is a single continuous shot or multiple shots with cuts.",
                    widget=ComboWidget(options=("single", "multi")),
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("WanReferenceVideoApi",),
            input_families=(
                InputFamilySpec(
                    "reference_videos",
                    (InputSpec("reference_video", VIDEO, default=None, required=True),),
                    min_members=1,
                    member_names=("character1", "character2", "character3"),
                ),
            ),
            combos=(),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            MediaConstraints(
                "validate_reference_videos",
                "reference_videos",
                min_duration=2,
                max_duration=30,
                batch=True,
            ),
            ProxyUpload(
                "uploaded_reference_videos",
                "reference_videos",
                "upload.mp4",
                "video/mp4",
                batch=True,
            ),
            BatchMapJoin(
                "reference_video_urls",
                segments=(
                    Segment("uploaded_reference_videos", wrap_key=None, mode="mapping_values"),
                ),
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "prompt"),
                    InputBinding("input.negative_prompt", "negative_prompt"),
                    InputBinding("input.reference_video_urls", "reference_video_urls"),
                    InputBinding("parameters.size", "size", value_map=SIZE_VALUES),
                    InputBinding("parameters.duration", "duration"),
                    InputBinding("parameters.shot_type", "shot_type"),
                    InputBinding("parameters.watermark", "watermark"),
                    InputBinding("parameters.seed", "seed"),
                ),
            ),
            *_wan_result("submit", "video", "video", 6),
        )
    )


class Wan2TextToVideoApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.2-text-to-video",
            display_name="Wan 2.7 Text to Video",
            category="partner/video/Wan",
            description="Generates a video based on a text prompt using the Wan 2.7 model.",
            inputs=(
                InputSpec(
                    "audio",
                    AUDIO,
                    default=None,
                    required=False,
                    doc="Audio for driving video generation (e.g., lip sync, beat-matched motion). Duration: 3s-30s. If not provided, the model automatically generates matching background music or sound effects.",
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "prompt_extend",
                    BOOLEAN,
                    default=True,
                    required=True,
                    doc="Whether to enhance the prompt with AI assistance.",
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("Wan2TextToVideoApi",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "wan2.7-t2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "negative_prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Negative prompt describing what to avoid.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "ratio",
                                    COMBO,
                                    default="16:9",
                                    required=True,
                                    widget=ComboWidget(
                                        options=("16:9", "9:16", "1:1", "4:3", "3:4")
                                    ),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    widget=NumberWidget(2, 15, 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_prompt",
                (
                    Check(
                        "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
                        require=(Cond("model.prompt", "ne", ""),),
                    ),
                ),
            ),
            MediaConstraints(
                "validate_audio",
                "audio",
                min_duration=1.5,
                max_duration=60,
                duration_media="audio",
                optional=True,
            ),
            EncodeMedia(
                "encoded_audio",
                "audio",
                media_family="audio",
                format="MP3",
                output="bytes",
                optional=True,
            ),
            ProxyUpload(
                "uploaded_audio", "encoded_audio", "upload.mp3", "audio/mpeg", optional=True
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "model.prompt"),
                    InputBinding("input.negative_prompt", "model.negative_prompt", omit_if=""),
                    InputBinding("input.audio_url", "uploaded_audio"),
                    InputBinding("parameters.resolution", "model.resolution"),
                    InputBinding("parameters.ratio", "model.ratio"),
                    InputBinding("parameters.duration", "model.duration"),
                    InputBinding("parameters.seed", "seed"),
                    InputBinding("parameters.prompt_extend", "prompt_extend"),
                    InputBinding("parameters.watermark", "watermark"),
                ),
            ),
            *_wan_result("submit", "video", "video", 7),
        )
    )


class Wan2ImageToVideoApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.2-image-to-video",
            display_name="Wan 2.7 Image to Video",
            category="partner/video/Wan",
            description="Generate a video from a first-frame image, with optional last-frame image and audio.",
            inputs=(
                InputSpec(
                    "first_frame",
                    IMAGE,
                    default=None,
                    required=True,
                    doc="First frame image. The output aspect ratio is derived from this image.",
                ),
                InputSpec(
                    "last_frame",
                    IMAGE,
                    default=None,
                    required=False,
                    doc="Last frame image. The model generates a video transitioning from first to last frame.",
                ),
                InputSpec(
                    "audio",
                    AUDIO,
                    default=None,
                    required=False,
                    doc="Audio for driving video generation (e.g., lip sync, beat-matched motion). Duration: 2s-30s. If not provided, the model automatically generates matching background music or sound effects.",
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "prompt_extend",
                    BOOLEAN,
                    default=True,
                    required=True,
                    doc="Whether to enhance the prompt with AI assistance.",
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("Wan2ImageToVideoApi",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "wan2.7-i2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "negative_prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Negative prompt describing what to avoid.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    widget=NumberWidget(2, 15, 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            MediaConstraints(
                "validate_audio",
                "audio",
                min_duration=2.0,
                max_duration=30,
                duration_media="audio",
                optional=True,
            ),
            EncodeMedia("encoded_first_frame", "first_frame", output="bytes"),
            EncodeMedia("encoded_last_frame", "last_frame", output="bytes", optional=True),
            EncodeMedia(
                "encoded_audio",
                "audio",
                media_family="audio",
                format="MP3",
                output="bytes",
                optional=True,
            ),
            ProxyUpload("uploaded_first_frame", "encoded_first_frame"),
            ProxyUpload("uploaded_last_frame", "encoded_last_frame", optional=True),
            ProxyUpload(
                "uploaded_audio", "encoded_audio", "upload.mp3", "audio/mpeg", optional=True
            ),
            BatchMapJoin(
                "media",
                segments=(
                    Segment("uploaded_first_frame", fixed={"type": "first_frame"}),
                    Segment(
                        "uploaded_last_frame", fixed={"type": "last_frame"}, mode="single_optional"
                    ),
                    Segment(
                        "uploaded_audio", fixed={"type": "driving_audio"}, mode="single_optional"
                    ),
                ),
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "model.prompt", omit_if=""),
                    InputBinding("input.negative_prompt", "model.negative_prompt", omit_if=""),
                    InputBinding("input.media", "media"),
                    InputBinding("parameters.resolution", "model.resolution"),
                    InputBinding("parameters.duration", "model.duration"),
                    InputBinding("parameters.seed", "seed"),
                    InputBinding("parameters.prompt_extend", "prompt_extend"),
                    InputBinding("parameters.watermark", "watermark"),
                ),
            ),
            *_wan_result("submit", "video", "video", 7),
        )
    )


class Wan2VideoContinuationApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.2-video-continuation",
            display_name="Wan 2.7 Video Continuation",
            category="partner/video/Wan",
            description="Continue a video from where it left off, with optional last-frame control.",
            inputs=(
                InputSpec(
                    "first_clip",
                    VIDEO,
                    default=None,
                    required=True,
                    doc="Input video to continue from. Duration: 2s-10s. The output aspect ratio is derived from this video.",
                ),
                InputSpec(
                    "last_frame",
                    IMAGE,
                    default=None,
                    required=False,
                    doc="Last frame image. The continuation will transition towards this frame.",
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "prompt_extend",
                    BOOLEAN,
                    default=True,
                    required=True,
                    doc="Whether to enhance the prompt with AI assistance.",
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("Wan2VideoContinuationApi",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "wan2.7-i2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "negative_prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Negative prompt describing what to avoid.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    doc="Total output duration in seconds. The model generates continuation to fill the remaining time after the input clip.",
                                    widget=NumberWidget(2, 15, 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            MediaConstraints("validate_first_clip", "first_clip", min_duration=2, max_duration=10),
            ProxyUpload("uploaded_first_clip", "first_clip", "upload.mp4", "video/mp4"),
            EncodeMedia("encoded_last_frame", "last_frame", output="bytes", optional=True),
            ProxyUpload("uploaded_last_frame", "encoded_last_frame", optional=True),
            BatchMapJoin(
                "media",
                segments=(
                    Segment("uploaded_first_clip", fixed={"type": "first_clip"}),
                    Segment(
                        "uploaded_last_frame", fixed={"type": "last_frame"}, mode="single_optional"
                    ),
                ),
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "model.prompt", omit_if=""),
                    InputBinding("input.negative_prompt", "model.negative_prompt", omit_if=""),
                    InputBinding("input.media", "media"),
                    InputBinding("parameters.resolution", "model.resolution"),
                    InputBinding("parameters.duration", "model.duration"),
                    InputBinding("parameters.seed", "seed"),
                    InputBinding("parameters.prompt_extend", "prompt_extend"),
                    InputBinding("parameters.watermark", "watermark"),
                ),
            ),
            *_wan_result("submit", "video", "video", 7),
        )
    )


class Wan2VideoEditApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.2-video-edit",
            display_name="Wan 2.7 Video Edit",
            category="partner/video/Wan",
            description="Edit a video using text instructions, reference images, or style transfer.",
            inputs=(
                InputSpec("video", VIDEO, default=None, required=True, doc="The video to edit."),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "audio_setting",
                    COMBO,
                    default="auto",
                    required=True,
                    doc="'auto': model decides whether to regenerate audio based on the prompt. 'origin': preserve the original audio from the input video.",
                    widget=ComboWidget(options=("auto", "origin")),
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("Wan2VideoEditApi",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "wan2.7-videoedit",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Editing instructions or style transfer requirements.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "ratio",
                                    COMBO,
                                    default="16:9",
                                    required=True,
                                    doc="Aspect ratio. If not changed, approximates the input video ratio.",
                                    widget=ComboWidget(
                                        options=("16:9", "9:16", "1:1", "4:3", "3:4")
                                    ),
                                ),
                                InputSpec(
                                    "duration",
                                    COMBO,
                                    default="auto",
                                    required=True,
                                    doc="Output duration in seconds. 'auto' matches the input video duration. A specific value truncates from the start of the video.",
                                    widget=ComboWidget(
                                        options=(
                                            "auto",
                                            "2",
                                            "3",
                                            "4",
                                            "5",
                                            "6",
                                            "7",
                                            "8",
                                            "9",
                                            "10",
                                        )
                                    ),
                                ),
                                InputFamilySpec(
                                    "reference_images",
                                    (
                                        InputSpec(
                                            "reference_image", IMAGE, default=None, required=True
                                        ),
                                    ),
                                    min_members=0,
                                    member_names=("image1", "image2", "image3", "image4"),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_prompt",
                (
                    Check(
                        "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
                        require=(Cond("model.prompt", "ne", ""),),
                    ),
                ),
            ),
            MediaConstraints("validate_video", "video", min_duration=2, max_duration=10),
            ProxyUpload("uploaded_video", "video", "upload.mp4", "video/mp4"),
            EncodeMedia(
                "encoded_reference_images",
                "model.reference_images",
                output="bytes",
                batch_targets=("image1", "image2", "image3", "image4"),
            ),
            ProxyUpload("uploaded_reference_images", "encoded_reference_images", batch=True),
            BatchMapJoin(
                "media",
                segments=(
                    Segment("uploaded_video", fixed={"type": "video"}),
                    Segment(
                        "uploaded_reference_images",
                        fixed={"type": "reference_image"},
                        mode="mapping",
                    ),
                ),
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "model.prompt"),
                    InputBinding("input.media", "media"),
                    InputBinding("parameters.resolution", "model.resolution"),
                    InputBinding("parameters.ratio", "model.ratio"),
                    InputBinding(
                        "parameters.duration",
                        "model.duration",
                        value_map={
                            "auto": 0,
                            "2": 2,
                            "3": 3,
                            "4": 4,
                            "5": 5,
                            "6": 6,
                            "7": 7,
                            "8": 8,
                            "9": 9,
                            "10": 10,
                        },
                    ),
                    InputBinding("parameters.audio_setting", "audio_setting"),
                    InputBinding("parameters.watermark", "watermark"),
                    InputBinding("parameters.seed", "seed"),
                ),
            ),
            *_wan_result("submit", "video", "video", 7),
        )
    )


class Wan2ReferenceVideoApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.2-reference-video",
            display_name="Wan 2.7 Reference to Video",
            category="partner/video/Wan",
            description="Generate a video featuring a person or object from reference materials. Supports single-character performances and multi-character interactions.",
            inputs=(
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("Wan2ReferenceVideoApi",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "wan2.7-r2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the video. Use identifiers such as 'character1' and 'character2' to refer to the reference characters.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "negative_prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Negative prompt describing what to avoid.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "ratio",
                                    COMBO,
                                    default="16:9",
                                    required=True,
                                    widget=ComboWidget(
                                        options=("16:9", "9:16", "1:1", "4:3", "3:4")
                                    ),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    widget=NumberWidget(2, 10, 1),
                                ),
                                InputFamilySpec(
                                    "reference_videos",
                                    (
                                        InputSpec(
                                            "reference_video", VIDEO, default=None, required=True
                                        ),
                                    ),
                                    min_members=0,
                                    member_names=("video1", "video2", "video3"),
                                ),
                                InputFamilySpec(
                                    "reference_images",
                                    (
                                        InputSpec(
                                            "reference_image", IMAGE, default=None, required=True
                                        ),
                                    ),
                                    min_members=0,
                                    member_names=("image1", "image2", "image3", "image4", "image5"),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_prompt",
                (
                    Check(
                        "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
                        require=(Cond("model.prompt", "ne", ""),),
                    ),
                ),
            ),
            ProxyUpload(
                "uploaded_reference_videos",
                "model.reference_videos",
                "upload.mp4",
                "video/mp4",
                batch=True,
            ),
            EncodeMedia(
                "encoded_reference_images",
                "model.reference_images",
                output="bytes",
                batch_targets=("image1", "image2", "image3", "image4", "image5"),
            ),
            ProxyUpload("uploaded_reference_images", "encoded_reference_images", batch=True),
            BatchMapJoin(
                "media",
                segments=(
                    Segment(
                        "uploaded_reference_videos",
                        fixed={"type": "reference_video"},
                        mode="mapping",
                    ),
                    Segment(
                        "uploaded_reference_images",
                        fixed={"type": "reference_image"},
                        mode="mapping",
                    ),
                ),
                min_items=1,
                max_items=5,
                min_message="At least one reference video or reference image must be provided.",
                max_message="Too many references ({count}). The maximum total of reference videos and images is 5.",
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "model.prompt"),
                    InputBinding("input.negative_prompt", "model.negative_prompt", omit_if=""),
                    InputBinding("input.media", "media"),
                    InputBinding("parameters.resolution", "model.resolution"),
                    InputBinding("parameters.ratio", "model.ratio"),
                    InputBinding("parameters.duration", "model.duration"),
                    InputBinding("parameters.watermark", "watermark"),
                    InputBinding("parameters.seed", "seed"),
                ),
            ),
            *_wan_result("submit", "video", "video", 7),
        )
    )


class HappyHorseTextToVideoApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.happy-horse-text-to-video",
            display_name="HappyHorse Text to Video",
            category="partner/video/Wan",
            description="Generates a video based on a text prompt using the HappyHorse model.",
            inputs=(
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("HappyHorseTextToVideoApi",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "happyhorse-1.1-t2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "ratio",
                                    COMBO,
                                    default="16:9",
                                    required=True,
                                    widget=ComboWidget(
                                        options=(
                                            "16:9",
                                            "9:16",
                                            "1:1",
                                            "4:3",
                                            "3:4",
                                            "21:9",
                                            "9:21",
                                            "5:4",
                                            "4:5",
                                        )
                                    ),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    widget=NumberWidget(3, 15, 1),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "happyhorse-1.0-t2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "ratio",
                                    COMBO,
                                    default="16:9",
                                    required=True,
                                    widget=ComboWidget(
                                        options=("16:9", "9:16", "1:1", "4:3", "3:4")
                                    ),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    widget=NumberWidget(3, 15, 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_prompt",
                (
                    Check(
                        "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
                        require=(Cond("model.prompt", "ne", ""),),
                    ),
                ),
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "model.prompt"),
                    InputBinding("parameters.resolution", "model.resolution"),
                    InputBinding("parameters.ratio", "model.ratio"),
                    InputBinding("parameters.duration", "model.duration"),
                    InputBinding("parameters.seed", "seed"),
                    InputBinding("parameters.watermark", "watermark"),
                ),
                fixed=(FixedField("parameters.prompt_extend", True),),
            ),
            *_wan_result("submit", "video", "video", 7),
        )
    )


class HappyHorseImageToVideoApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.happy-horse-image-to-video",
            display_name="HappyHorse Image to Video",
            category="partner/video/Wan",
            description="Generate a video from a first-frame image using the HappyHorse model.",
            inputs=(
                InputSpec(
                    "first_frame",
                    IMAGE,
                    default=None,
                    required=True,
                    doc="First frame image. The output aspect ratio is derived from this image.",
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("HappyHorseImageToVideoApi",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "happyhorse-1.1-i2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    widget=NumberWidget(3, 15, 1),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "happyhorse-1.0-i2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the elements and visual features. Supports English and Chinese.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    widget=NumberWidget(3, 15, 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            MediaConstraints(
                "validate_first_frame",
                "first_frame",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
            ),
            EncodeMedia("encoded_first_frame", "first_frame", output="bytes"),
            ProxyUpload("uploaded_first_frame", "encoded_first_frame"),
            BatchMapJoin(
                "media", segments=(Segment("uploaded_first_frame", fixed={"type": "first_frame"}),)
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "model.prompt", omit_if=""),
                    InputBinding("input.media", "media"),
                    InputBinding("parameters.resolution", "model.resolution"),
                    InputBinding("parameters.duration", "model.duration"),
                    InputBinding("parameters.seed", "seed"),
                    InputBinding("parameters.watermark", "watermark"),
                ),
                fixed=(FixedField("parameters.prompt_extend", True),),
            ),
            *_wan_result("submit", "video", "video", 7),
        )
    )


class HappyHorseVideoEditApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.happy-horse-video-edit",
            display_name="HappyHorse Video Edit",
            category="partner/video/Wan",
            description="Edit a video using text instructions or reference images with the HappyHorse model. Output duration is 3-15s and matches the input video; inputs longer than 15s are truncated.",
            inputs=(
                InputSpec("video", VIDEO, default=None, required=True, doc="The video to edit."),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("HappyHorseVideoEditApi",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "happyhorse-1.0-video-edit",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Editing instructions or style transfer requirements.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "ratio",
                                    COMBO,
                                    default="16:9",
                                    required=True,
                                    doc="Aspect ratio. If not changed, approximates the input video ratio.",
                                    widget=ComboWidget(
                                        options=("16:9", "9:16", "1:1", "4:3", "3:4")
                                    ),
                                ),
                                InputFamilySpec(
                                    "reference_images",
                                    (
                                        InputSpec(
                                            "reference_image", IMAGE, default=None, required=True
                                        ),
                                    ),
                                    min_members=0,
                                    member_names=("image1", "image2", "image3", "image4", "image5"),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_prompt",
                (
                    Check(
                        "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
                        require=(Cond("model.prompt", "ne", ""),),
                    ),
                ),
            ),
            MediaConstraints("validate_video", "video", min_duration=3, max_duration=60),
            ProxyUpload("uploaded_video", "video", "upload.mp4", "video/mp4"),
            EncodeMedia(
                "encoded_reference_images",
                "model.reference_images",
                output="bytes",
                batch_targets=("image1", "image2", "image3", "image4", "image5"),
            ),
            ProxyUpload("uploaded_reference_images", "encoded_reference_images", batch=True),
            BatchMapJoin(
                "media",
                segments=(
                    Segment("uploaded_video", fixed={"type": "video"}),
                    Segment(
                        "uploaded_reference_images",
                        fixed={"type": "reference_image"},
                        mode="mapping",
                    ),
                ),
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "model.prompt"),
                    InputBinding("input.media", "media"),
                    InputBinding("parameters.resolution", "model.resolution"),
                    InputBinding("parameters.ratio", "model.ratio"),
                    InputBinding("parameters.watermark", "watermark"),
                    InputBinding("parameters.seed", "seed"),
                ),
                fixed=(FixedField("parameters.audio_setting", "auto"),),
            ),
            *_wan_result("submit", "video", "video", 7),
        )
    )


class HappyHorseReferenceVideoApi(WanNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.wan.happy-horse-reference-video",
            display_name="HappyHorse Reference to Video",
            category="partner/video/Wan",
            description="Generate a video featuring a person or object from reference materials with the HappyHorse model. Supports single-character performances and multi-character interactions.",
            inputs=(
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    required=True,
                    doc="Seed to use for generation.",
                    widget=NumberWidget(0, 2147483647, 1, control_after_generate="randomize"),
                ),
                InputSpec(
                    "watermark",
                    BOOLEAN,
                    default=False,
                    required=True,
                    doc="Whether to add an AI-generated watermark to the result.",
                ),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("HappyHorseReferenceVideoApi",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "happyhorse-1.1-r2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the video. Use identifiers such as 'character1' and 'character2' to refer to the reference characters.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "ratio",
                                    COMBO,
                                    default="16:9",
                                    required=True,
                                    widget=ComboWidget(
                                        options=(
                                            "16:9",
                                            "9:16",
                                            "1:1",
                                            "4:3",
                                            "3:4",
                                            "21:9",
                                            "9:21",
                                            "5:4",
                                            "4:5",
                                        )
                                    ),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    widget=NumberWidget(3, 15, 1),
                                ),
                                InputFamilySpec(
                                    "reference_images",
                                    (
                                        InputSpec(
                                            "reference_image", IMAGE, default=None, required=True
                                        ),
                                    ),
                                    min_members=1,
                                    member_names=(
                                        "image1",
                                        "image2",
                                        "image3",
                                        "image4",
                                        "image5",
                                        "image6",
                                        "image7",
                                        "image8",
                                        "image9",
                                    ),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "happyhorse-1.0-r2v",
                            (
                                InputSpec(
                                    "prompt",
                                    STRING,
                                    default="",
                                    required=True,
                                    doc="Prompt describing the video. Use identifiers such as 'character1' and 'character2' to refer to the reference characters.",
                                    widget=StringWidget(multiline=True),
                                ),
                                InputSpec(
                                    "resolution",
                                    COMBO,
                                    default="720P",
                                    required=True,
                                    widget=ComboWidget(options=("720P", "1080P")),
                                ),
                                InputSpec(
                                    "ratio",
                                    COMBO,
                                    default="16:9",
                                    required=True,
                                    widget=ComboWidget(
                                        options=("16:9", "9:16", "1:1", "4:3", "3:4")
                                    ),
                                ),
                                InputSpec(
                                    "duration",
                                    INT,
                                    default=5,
                                    required=True,
                                    widget=NumberWidget(3, 15, 1),
                                ),
                                InputFamilySpec(
                                    "reference_images",
                                    (
                                        InputSpec(
                                            "reference_image", IMAGE, default=None, required=True
                                        ),
                                    ),
                                    min_members=1,
                                    member_names=(
                                        "image1",
                                        "image2",
                                        "image3",
                                        "image4",
                                        "image5",
                                        "image6",
                                        "image7",
                                        "image8",
                                        "image9",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_prompt",
                (
                    Check(
                        "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
                        require=(Cond("model.prompt", "ne", ""),),
                    ),
                ),
            ),
            MediaConstraints(
                "validate_reference_images",
                "model.reference_images",
                min_width=400,
                min_height=400,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                batch=True,
            ),
            EncodeMedia(
                "encoded_reference_images",
                "model.reference_images",
                output="bytes",
                batch_targets=(
                    "image1",
                    "image2",
                    "image3",
                    "image4",
                    "image5",
                    "image6",
                    "image7",
                    "image8",
                    "image9",
                ),
            ),
            ProxyUpload("uploaded_reference_images", "encoded_reference_images", batch=True),
            BatchMapJoin(
                "media",
                segments=(
                    Segment(
                        "uploaded_reference_images",
                        fixed={"type": "reference_image"},
                        mode="mapping",
                    ),
                ),
                min_items=1,
                min_message="At least one reference image must be provided.",
            ),
            HttpSyncJson(
                "submit",
                WAN_VIDEO_PATH,
                body=(
                    InputBinding("model", "model"),
                    InputBinding("input.prompt", "model.prompt"),
                    InputBinding("input.media", "media"),
                    InputBinding("parameters.resolution", "model.resolution"),
                    InputBinding("parameters.ratio", "model.ratio"),
                    InputBinding("parameters.duration", "model.duration"),
                    InputBinding("parameters.watermark", "watermark"),
                    InputBinding("parameters.seed", "seed"),
                ),
            ),
            *_wan_result("submit", "video", "video", 7),
        )
    )


WAN_NODES = (
    WanTextToImageApi,
    WanImageToImageApi,
    WanTextToVideoApi,
    WanImageToVideoApi,
    WanReferenceVideoApi,
    Wan2TextToVideoApi,
    Wan2ImageToVideoApi,
    Wan2VideoContinuationApi,
    Wan2VideoEditApi,
    Wan2ReferenceVideoApi,
    HappyHorseTextToVideoApi,
    HappyHorseImageToVideoApi,
    HappyHorseVideoEditApi,
    HappyHorseReferenceVideoApi,
)
