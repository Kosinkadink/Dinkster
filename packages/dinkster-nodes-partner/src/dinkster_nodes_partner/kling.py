"""Genuinely declarative Kling partner nodes pinned to ComfyUI e651b7be."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

from .kling_helpers import KLING_HELPERS
from .opspec import (
    Adapter,
    BatchMapJoin,
    Check,
    CheckInputs,
    Class3Spec,
    Cond,
    DownloadDecode,
    EncodeMedia,
    FixedField,
    FormatField,
    HelperBinding,
    HelperCall,
    HttpSyncJson,
    InputBinding,
    MediaConstraints,
    OpSpec,
    ProxyUpload,
    ResponseSelect,
    Segment,
    SubmitPoll,
    ValueConstruct,
)
from .partner_runtime import run_class3_op, run_op, worker_runtime_context

COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
IMAGE = TypeExpr.concrete("comfy.IMAGE")
VIDEO = TypeExpr.concrete("comfy.VIDEO")
AUDIO = TypeExpr.concrete("comfy.AUDIO")
CAMERA_CONTROL = TypeExpr.concrete("partner.kling.camera-control")
KLING_STATUSES = ("submitted", "processing", "succeed", "failed")
MAX_IMAGE_PIXELS = 2048 * 2048

MODE_TEXT2VIDEO = {
    "standard mode / 5s duration / kling-v1-6": ("std", "5", "kling-v1-6"),
    "standard mode / 10s duration / kling-v1-6": ("std", "10", "kling-v1-6"),
    "pro mode / 5s duration / kling-v2-master": ("pro", "5", "kling-v2-master"),
    "pro mode / 10s duration / kling-v2-master": ("pro", "10", "kling-v2-master"),
    "standard mode / 5s duration / kling-v2-master": ("std", "5", "kling-v2-master"),
    "standard mode / 10s duration / kling-v2-master": ("std", "10", "kling-v2-master"),
    "pro mode / 5s duration / kling-v2-1-master": ("pro", "5", "kling-v2-1-master"),
    "pro mode / 10s duration / kling-v2-1-master": ("pro", "10", "kling-v2-1-master"),
    "pro mode / 5s duration / kling-v2-5-turbo": ("pro", "5", "kling-v2-5-turbo"),
    "pro mode / 10s duration / kling-v2-5-turbo": ("pro", "10", "kling-v2-5-turbo"),
}
MODE_START_END_FRAME = {
    "pro mode / 5s duration / kling-v1-5": ("pro", "5", "kling-v1-5"),
    "pro mode / 10s duration / kling-v1-5": ("pro", "10", "kling-v1-5"),
    "pro mode / 5s duration / kling-v1-6": ("pro", "5", "kling-v1-6"),
    "pro mode / 10s duration / kling-v1-6": ("pro", "10", "kling-v1-6"),
    "pro mode / 5s duration / kling-v2-1": ("pro", "5", "kling-v2-1"),
    "pro mode / 10s duration / kling-v2-1": ("pro", "10", "kling-v2-1"),
    "pro mode / 5s duration / kling-v2-5-turbo": ("pro", "5", "kling-v2-5-turbo"),
    "pro mode / 10s duration / kling-v2-5-turbo": ("pro", "10", "kling-v2-5-turbo"),
}
# Keep non-ASCII upstream display names as \u escapes in source and fixtures.
VOICES_CONFIG = {
    "Melody": ("girlfriend_4_speech02", "en"),
    "Sunny": ("genshin_vindi2", "en"),
    "Sage": ("zhinen_xuesheng", "en"),
    "Ace": ("AOT", "en"),
    "Blossom": ("ai_shatang", "en"),
    "Peppy": ("genshin_klee2", "en"),
    "Dove": ("genshin_kirara", "en"),
    "Shine": ("ai_kaiya", "en"),
    "Anchor": ("oversea_male1", "en"),
    "Lyric": ("ai_chenjiahao_712", "en"),
    "Tender": ("chat1_female_new-3", "en"),
    "Siren": ("chat_0407_5-1", "en"),
    "Zippy": ("cartoon-boy-07", "en"),
    "Bud": ("uk_boy1", "en"),
    "Sprite": ("cartoon-girl-01", "en"),
    "Candy": ("PeppaPig_platform", "en"),
    "Beacon": ("ai_huangzhong_712", "en"),
    "Rock": ("ai_huangyaoshi_712", "en"),
    "Titan": ("ai_laoguowang_712", "en"),
    "Grace": ("chengshu_jiejie", "en"),
    "Helen": ("you_pingjing", "en"),
    "Lore": ("calm_story1", "en"),
    "Crag": ("uk_man2", "en"),
    "Prattle": ("laopopo_speech02", "en"),
    "Hearth": ("heainainai_speech02", "en"),
    "The Reader": ("reader_en_m-v1", "en"),
    "Commercial Lady": ("commercial_lady_en_f-v1", "en"),
    "\u9633\u5149\u5c11\u5e74": ("genshin_vindi2", "zh"),
    "\u61c2\u4e8b\u5c0f\u5f1f": ("zhinen_xuesheng", "zh"),
    "\u8fd0\u52a8\u5c11\u5e74": ("tiyuxi_xuedi", "zh"),
    "\u9752\u6625\u5c11\u5973": ("ai_shatang", "zh"),
    "\u6e29\u67d4\u5c0f\u59b9": ("genshin_klee2", "zh"),
    "\u5143\u6c14\u5c11\u5973": ("genshin_kirara", "zh"),
    "\u9633\u5149\u7537\u751f": ("ai_kaiya", "zh"),
    "\u5e7d\u9ed8\u5c0f\u54e5": ("tiexin_nanyou", "zh"),
    "\u6587\u827a\u5c0f\u54e5": ("ai_chenjiahao_712", "zh"),
    "\u751c\u7f8e\u90bb\u5bb6": ("girlfriend_1_speech02", "zh"),
    "\u6e29\u67d4\u59d0\u59d0": ("chat1_female_new-3", "zh"),
    "\u804c\u573a\u5973\u9752": ("girlfriend_2_speech02", "zh"),
    "\u6d3b\u6cfc\u7537\u7ae5": ("cartoon-boy-07", "zh"),
    "\u4fcf\u76ae\u5973\u7ae5": ("cartoon-girl-01", "zh"),
    "\u7a33\u91cd\u8001\u7238": ("ai_huangyaoshi_712", "zh"),
    "\u6e29\u67d4\u5988\u5988": ("you_pingjing", "zh"),
    "\u4e25\u8083\u4e0a\u53f8": ("ai_laoguowang_712", "zh"),
    "\u4f18\u96c5\u8d35\u5987": ("chengshu_jiejie", "zh"),
    "\u6148\u7965\u7237\u7237": ("zhuxi_speech02", "zh"),
    "\u5520\u53e8\u7237\u7237": ("uk_oldman3", "zh"),
    "\u5520\u53e8\u5976\u5976": ("laopopo_speech02", "zh"),
    "\u548c\u853c\u5976\u5976": ("heainainai_speech02", "zh"),
    "\u4e1c\u5317\u8001\u94c1": ("dongbeilaotie_speech02", "zh"),
    "\u91cd\u5e86\u5c0f\u4f19": ("chongqingxiaohuo_speech02", "zh"),
    "\u56db\u5ddd\u59b9\u5b50": ("chuanmeizi_speech02", "zh"),
    "\u6f6e\u6c55\u5927\u53d4": ("chaoshandashu_speech02", "zh"),
    "\u53f0\u6e7e\u7537\u751f": ("ai_taiwan_man2_speech02", "zh"),
    "\u897f\u5b89\u638c\u67dc": ("xianzhanggui_speech02", "zh"),
    "\u5929\u6d25\u59d0\u59d0": ("tianjinjiejie_speech02", "zh"),
    "\u65b0\u95fb\u64ad\u62a5\u7537": ("diyinnansang_DB_CN_M_04-v2", "zh"),
    "\u8bd1\u5236\u7247\u7537": ("yizhipiannan-v1", "zh"),
    "\u6492\u5a07\u5973\u53cb": ("tianmeixuemei-v1", "zh"),
    "\u5200\u7247\u70df\u55d3": ("daopianyansang-v1", "zh"),
    "\u4e56\u5de7\u6b63\u592a": ("mengwa-v1", "zh"),
}


@dataclass(frozen=True, kw_only=True)
class Data:
    created_at: int | None = None
    task_id: str | None = None
    task_info: TaskInfo | None = None
    task_result: TaskResult | None = None
    task_status: str | None = None
    updated_at: int | None = None


@dataclass(frozen=True, kw_only=True)
class Data1:
    created_at: int | None = None
    task_id: str | None = None
    task_result: TaskResult1 | None = None
    task_status: str | None = None
    task_status_msg: str | None = None
    updated_at: int | None = None


@dataclass(frozen=True, kw_only=True)
class Data2:
    created_at: int | None = None
    task_id: str | None = None
    task_info: TaskInfo | None = None
    task_result: TaskResult2 | None = None
    task_status: str | None = None
    updated_at: int | None = None


@dataclass(frozen=True, kw_only=True)
class Data4:
    created_at: int | None = None
    task_id: str | None = None
    task_info: TaskInfo | None = None
    task_result: TaskResult2 | None = None
    task_status: str | None = None
    updated_at: int | None = None


@dataclass(frozen=True, kw_only=True)
class Data5:
    created_at: int | None = None
    task_id: str | None = None
    task_info: TaskInfo | None = None
    task_result: TaskResult2 | None = None
    task_status: str | None = None
    updated_at: int | None = None


@dataclass(frozen=True, kw_only=True)
class Data6:
    created_at: int | None = None
    task_id: str | None = None
    task_info: TaskInfo | None = None
    task_result: TaskResult2 | None = None
    task_status: str | None = None
    updated_at: int | None = None


@dataclass(frozen=True, kw_only=True)
class Data7:
    created_at: int | None = None
    task_id: str | None = None
    task_result: TaskResult6 | None = None
    task_status: str | None = None
    task_status_msg: str | None = None
    updated_at: int | None = None


@dataclass(frozen=True, kw_only=True)
class DynamicMask:
    mask: str | None = None
    trajectories: list[Trajectory] | None = None


@dataclass(frozen=True, kw_only=True)
class ImageToVideoWithAudioRequest:
    model_name: str
    image: str
    duration: str
    prompt: str | None
    sound: str
    image_tail: str | None = None
    negative_prompt: str | None = None
    mode: str = "pro"
    multi_shot: bool | None = None
    multi_prompt: list[MultiPromptEntry] | None = None
    shot_type: str | None = None


@dataclass(frozen=True, kw_only=True)
class Kling3TurboContent:
    type: str
    text: str | None = None
    url: str | None = None


@dataclass(frozen=True, kw_only=True)
class Kling3TurboCreateData:
    id: str | None = None
    status: str | None = None
    message: str | None = None


@dataclass(frozen=True, kw_only=True)
class Kling3TurboCreateResponse:
    code: int | None = None
    message: str | None = None
    request_id: str | None = None
    data: Kling3TurboCreateData | None = None


@dataclass(frozen=True, kw_only=True)
class Kling3TurboImage2VideoRequest:
    contents: list[Kling3TurboContent]
    settings: Kling3TurboSettings | None = None


@dataclass(frozen=True, kw_only=True)
class Kling3TurboOutput:
    type: str | None = None
    id: str | None = None
    url: str | None = None
    duration: str | None = None


@dataclass(frozen=True, kw_only=True)
class Kling3TurboQueryResponse:
    code: int | None = None
    message: str | None = None
    request_id: str | None = None
    data: list[Kling3TurboTaskData] | None = None


@dataclass(frozen=True, kw_only=True)
class Kling3TurboSettings:
    resolution: str = "720p"
    aspect_ratio: str | None = None
    duration: int = 5


@dataclass(frozen=True, kw_only=True)
class Kling3TurboTaskData:
    id: str | None = None
    status: str | None = None
    message: str | None = None
    outputs: list[Kling3TurboOutput] | None = None


@dataclass(frozen=True, kw_only=True)
class Kling3TurboText2VideoRequest:
    prompt: str
    settings: Kling3TurboSettings | None = None


@dataclass(frozen=True, kw_only=True)
class KlingAvatarRequest:
    image: str
    sound_file: str
    mode: str
    prompt: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingCameraConfig:
    horizontal: float | None = None
    pan: float | None = None
    roll: float | None = None
    tilt: float | None = None
    vertical: float | None = None
    zoom: float | None = None


@dataclass(frozen=True, kw_only=True)
class KlingCameraControl:
    config: KlingCameraConfig | None = None
    type: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingDualCharacterEffectInput:
    duration: str
    images: KlingDualCharacterImages
    mode: str | None = "std"
    model_name: str | None = "kling-v1"


@dataclass(frozen=True, kw_only=True)
class KlingDualCharacterImages:
    root: list[str]


@dataclass(frozen=True, kw_only=True)
class KlingImage2VideoRequest:
    aspect_ratio: str | None = "16:9"
    callback_url: str | None = None
    camera_control: KlingCameraControl | None = None
    cfg_scale: float | None = 0.5
    duration: str | None = "5"
    dynamic_masks: list[DynamicMask] | None = None
    external_task_id: str | None = None
    image: str | None = None
    image_tail: str | None = None
    mode: str | None = "std"
    model_name: str | None = "kling-v2-master"
    negative_prompt: str | None = None
    prompt: str | None = None
    static_mask: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingImage2VideoResponse:
    code: int | None = None
    data: Data | None = None
    message: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingImageGenerationsRequest:
    model_name: str
    prompt: str
    aspect_ratio: str | None = "16:9"
    callback_url: str | None = None
    human_fidelity: float | None = 0.45
    image: str | None = None
    image_fidelity: float | None = 0.5
    image_reference: str | None = None
    n: int | None = 1
    negative_prompt: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingImageGenerationsResponse:
    code: int | None = None
    data: Data1 | None = None
    message: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingImageResult:
    index: int | None = None
    url: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingLipSyncInputObject:
    mode: str
    audio_file: str | None = None
    audio_type: str | None = None
    audio_url: str | None = None
    text: str | None = None
    video_id: str | None = None
    video_url: str | None = None
    voice_id: str | None = None
    voice_language: str | None = "en"
    voice_speed: float | None = 1


@dataclass(frozen=True, kw_only=True)
class KlingLipSyncRequest:
    input: KlingLipSyncInputObject
    callback_url: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingLipSyncResponse:
    code: int | None = None
    data: Data2 | None = None
    message: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingSingleImageEffectInput:
    duration: str
    image: str
    model_name: str

    def __post_init__(self) -> None:
        if self.duration != "5":
            raise ValueError("Input should be '5'")


@dataclass(frozen=True, kw_only=True)
class KlingText2VideoRequest:
    aspect_ratio: str | None = "16:9"
    callback_url: str | None = None
    camera_control: KlingCameraControl | None = None
    cfg_scale: float | None = 0.5
    duration: str | None = "5"
    external_task_id: str | None = None
    mode: str | None = "std"
    model_name: str | None = "kling-v1"
    negative_prompt: str | None = None
    prompt: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingText2VideoResponse:
    code: int | None = None
    data: Data4 | None = None
    message: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingVideoEffectsInput:
    root: KlingSingleImageEffectInput | KlingDualCharacterEffectInput


@dataclass(frozen=True, kw_only=True)
class KlingVideoEffectsRequest:
    effect_scene: str | str
    input: KlingVideoEffectsInput
    callback_url: str | None = None
    external_task_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingVideoEffectsResponse:
    code: int | None = None
    data: Data5 | None = None
    message: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingVideoExtendRequest:
    callback_url: str | None = None
    cfg_scale: float | None = 0.5
    negative_prompt: str | None = None
    prompt: str | None = None
    video_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingVideoExtendResponse:
    code: int | None = None
    data: Data6 | None = None
    message: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingVideoGenCfgScale:
    root: float


@dataclass(frozen=True, kw_only=True)
class KlingVideoResult:
    duration: str | None = None
    id: str | None = None
    url: str | None = None


@dataclass(frozen=True, kw_only=True)
class KlingVirtualTryOnRequest:
    human_image: str
    callback_url: str | None = None
    cloth_image: str | None = None
    model_name: str | None = "kolors-virtual-try-on-v1"


@dataclass(frozen=True, kw_only=True)
class KlingVirtualTryOnResponse:
    code: int | None = None
    data: Data7 | None = None
    message: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class MotionControlRequest:
    prompt: str
    image_url: str
    video_url: str
    keep_original_sound: str
    character_orientation: str
    mode: str
    model_name: str


@dataclass(frozen=True, kw_only=True)
class MultiPromptEntry:
    index: int
    prompt: str
    duration: str


@dataclass(frozen=True, kw_only=True)
class OmniImageParamImage:
    image: str


@dataclass(frozen=True, kw_only=True)
class OmniParamImage:
    image_url: str
    type: str | None = None


@dataclass(frozen=True, kw_only=True)
class OmniParamVideo:
    video_url: str
    refer_type: str | None
    keep_original_sound: str


@dataclass(frozen=True, kw_only=True)
class OmniProFirstLastFrameRequest:
    model_name: str
    image_list: list[OmniParamImage]
    duration: str
    prompt: str
    mode: str = "pro"
    sound: str | None = None
    multi_shot: bool | None = None
    multi_prompt: list[MultiPromptEntry] | None = None
    shot_type: str | None = None


@dataclass(frozen=True, kw_only=True)
class OmniProImageRequest:
    model_name: str
    resolution: str
    aspect_ratio: str | None
    prompt: str
    image_list: list[OmniImageParamImage] | None
    mode: str = "pro"
    n: int | None = 1
    result_type: str | None = None
    series_amount: int | None = None


@dataclass(frozen=True, kw_only=True)
class OmniProReferences2VideoRequest:
    model_name: str
    aspect_ratio: str | None
    duration: str | None
    prompt: str
    image_list: list[OmniParamImage] | None = None
    video_list: list[OmniParamVideo] | None = None
    mode: str = "pro"
    sound: str | None = None
    multi_shot: bool | None = None
    multi_prompt: list[MultiPromptEntry] | None = None
    shot_type: str | None = None


@dataclass(frozen=True, kw_only=True)
class OmniProText2VideoRequest:
    model_name: str
    aspect_ratio: str
    duration: str
    prompt: str
    sound: str
    mode: str = "pro"
    multi_shot: bool | None = None
    multi_prompt: list[MultiPromptEntry] | None = None
    shot_type: str | None = None


@dataclass(frozen=True, kw_only=True)
class TaskInfo:
    external_task_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class TaskResult:
    videos: list[KlingVideoResult] | None = None


@dataclass(frozen=True, kw_only=True)
class TaskResult1:
    images: list[KlingImageResult] | None = None


@dataclass(frozen=True, kw_only=True)
class TaskResult2:
    videos: list[KlingVideoResult] | None = None


@dataclass(frozen=True, kw_only=True)
class TaskResult6:
    images: list[KlingImageResult] | None = None


@dataclass(frozen=True, kw_only=True)
class TaskStatusImageResult:
    index: int
    url: str


@dataclass(frozen=True, kw_only=True)
class TaskStatusResponse:
    code: int | None = None
    message: str | None = None
    request_id: str | None = None
    data: TaskStatusResponseData | None = None


@dataclass(frozen=True, kw_only=True)
class TaskStatusResponseData:
    created_at: int | None = None
    updated_at: int | None = None
    task_status: str | None = None
    task_status_msg: str | None = None
    task_id: str | None = None
    task_result: TaskStatusResults | None = None


@dataclass(frozen=True, kw_only=True)
class TaskStatusResults:
    videos: list[TaskStatusVideoResult] | None = None
    images: list[TaskStatusImageResult] | None = None
    series_images: list[TaskStatusImageResult] | None = None


@dataclass(frozen=True, kw_only=True)
class TaskStatusVideoResult:
    duration: str | None = None
    id: str | None = None
    url: str | None = None


@dataclass(frozen=True, kw_only=True)
class TextToVideoWithAudioRequest:
    model_name: str
    aspect_ratio: str
    duration: str
    prompt: str | None
    sound: str
    negative_prompt: str | None = None
    mode: str = "pro"
    multi_shot: bool | None = None
    multi_prompt: list[MultiPromptEntry] | None = None
    shot_type: str | None = None


@dataclass(frozen=True, kw_only=True)
class Trajectory:
    x: int | None = None
    y: int | None = None


KLING_CONTRACTS: Mapping[str, type[object]] = {
    "Data": Data,
    "Data1": Data1,
    "Data2": Data2,
    "Data4": Data4,
    "Data5": Data5,
    "Data6": Data6,
    "Data7": Data7,
    "DynamicMask": DynamicMask,
    "ImageToVideoWithAudioRequest": ImageToVideoWithAudioRequest,
    "Kling3TurboContent": Kling3TurboContent,
    "Kling3TurboCreateData": Kling3TurboCreateData,
    "Kling3TurboCreateResponse": Kling3TurboCreateResponse,
    "Kling3TurboImage2VideoRequest": Kling3TurboImage2VideoRequest,
    "Kling3TurboOutput": Kling3TurboOutput,
    "Kling3TurboQueryResponse": Kling3TurboQueryResponse,
    "Kling3TurboSettings": Kling3TurboSettings,
    "Kling3TurboTaskData": Kling3TurboTaskData,
    "Kling3TurboText2VideoRequest": Kling3TurboText2VideoRequest,
    "KlingAvatarRequest": KlingAvatarRequest,
    "KlingCameraConfig": KlingCameraConfig,
    "KlingCameraControl": KlingCameraControl,
    "KlingDualCharacterEffectInput": KlingDualCharacterEffectInput,
    "KlingDualCharacterImages": KlingDualCharacterImages,
    "KlingImage2VideoRequest": KlingImage2VideoRequest,
    "KlingImage2VideoResponse": KlingImage2VideoResponse,
    "KlingImageGenerationsRequest": KlingImageGenerationsRequest,
    "KlingImageGenerationsResponse": KlingImageGenerationsResponse,
    "KlingImageResult": KlingImageResult,
    "KlingLipSyncInputObject": KlingLipSyncInputObject,
    "KlingLipSyncRequest": KlingLipSyncRequest,
    "KlingLipSyncResponse": KlingLipSyncResponse,
    "KlingSingleImageEffectInput": KlingSingleImageEffectInput,
    "KlingText2VideoRequest": KlingText2VideoRequest,
    "KlingText2VideoResponse": KlingText2VideoResponse,
    "KlingVideoEffectsInput": KlingVideoEffectsInput,
    "KlingVideoEffectsRequest": KlingVideoEffectsRequest,
    "KlingVideoEffectsResponse": KlingVideoEffectsResponse,
    "KlingVideoExtendRequest": KlingVideoExtendRequest,
    "KlingVideoExtendResponse": KlingVideoExtendResponse,
    "KlingVideoGenCfgScale": KlingVideoGenCfgScale,
    "KlingVideoResult": KlingVideoResult,
    "KlingVirtualTryOnRequest": KlingVirtualTryOnRequest,
    "KlingVirtualTryOnResponse": KlingVirtualTryOnResponse,
    "MotionControlRequest": MotionControlRequest,
    "MultiPromptEntry": MultiPromptEntry,
    "OmniImageParamImage": OmniImageParamImage,
    "OmniParamImage": OmniParamImage,
    "OmniParamVideo": OmniParamVideo,
    "OmniProFirstLastFrameRequest": OmniProFirstLastFrameRequest,
    "OmniProImageRequest": OmniProImageRequest,
    "OmniProReferences2VideoRequest": OmniProReferences2VideoRequest,
    "OmniProText2VideoRequest": OmniProText2VideoRequest,
    "TaskInfo": TaskInfo,
    "TaskResult": TaskResult,
    "TaskResult1": TaskResult1,
    "TaskResult2": TaskResult2,
    "TaskResult6": TaskResult6,
    "TaskStatusImageResult": TaskStatusImageResult,
    "TaskStatusResponse": TaskStatusResponse,
    "TaskStatusResponseData": TaskStatusResponseData,
    "TaskStatusResults": TaskStatusResults,
    "TaskStatusVideoResult": TaskStatusVideoResult,
    "TextToVideoWithAudioRequest": TextToVideoWithAudioRequest,
    "Trajectory": Trajectory,
}


def _combo(id: str, options: tuple[str, ...], default: str | None = None) -> InputSpec:
    return InputSpec(
        id,
        COMBO,
        default=options[0] if default is None else default,
        widget=ComboWidget(options=options),
    )


def _float(id: str, default: float) -> InputSpec:
    return InputSpec(id, FLOAT, default=default, widget=NumberWidget(-10.0, 10.0, 0.25))


def _schema(
    upstream_id: str,
    display_name: str,
    category: str,
    inputs: tuple[InputSpec, ...],
    outputs: tuple[OutputSpec, ...],
) -> NodeSchema:
    stem = upstream_id.removeprefix("Kling").removesuffix("Node")
    operation = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", stem).lower()
    return NodeSchema(
        node_type=f"partner.kling.{operation}",
        display_name=display_name,
        category=category,
        inputs=inputs,
        outputs=outputs,
        aliases=(upstream_id,),
        io_bound=upstream_id != "KlingCameraControls",
    )


class KlingNode(Node):
    SPEC: OpSpec | Class3Spec

    @classmethod
    async def execute(cls, **inputs: object) -> Mapping[str, object]:
        if isinstance(cls.SPEC, Class3Spec):
            result = await run_class3_op(cls.SPEC, inputs, worker_runtime_context(), KLING_HELPERS)
        else:
            result = await run_op(cls.SPEC, inputs, worker_runtime_context(), KLING_HELPERS)
        declared = tuple(output.id for output in cls.define_schema().outputs)
        missing = tuple(output for output in declared if output not in result)
        if missing:
            raise RuntimeError(f"Kling runtime omitted declared outputs: {missing!r}")
        return {output: result[output] for output in declared}


class KlingCameraControls(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingCameraControls",
            "Kling Camera Controls",
            "partner/video/Kling",
            (
                _combo(
                    "camera_control_type",
                    (
                        "simple",
                        "down_back",
                        "forward_up",
                        "right_turn_forward",
                        "left_turn_forward",
                    ),
                ),
                _float("horizontal_movement", 0.0),
                _float("vertical_movement", 0.0),
                _float("pan", 0.5),
                _float("tilt", 0.0),
                _float("roll", 0.0),
                _float("zoom", 0.0),
            ),
            (OutputSpec("camera_control", CAMERA_CONTROL),),
        )

    SPEC = OpSpec(
        (
            CheckInputs(
                "validate",
                (
                    Check(
                        "Invalid camera control configs: at least one of the values "
                        "must be non-zero",
                        when=tuple(
                            Cond(name, "eq", 0.0)
                            for name in (
                                "horizontal_movement",
                                "vertical_movement",
                                "pan",
                                "tilt",
                                "roll",
                                "zoom",
                            )
                        ),
                        require=(Cond("pan", "ne", 0.0),),
                    ),
                ),
            ),
            ValueConstruct(
                "camera",
                "camera_control",
                (
                    InputBinding("type", "camera_control_type"),
                    InputBinding("config.horizontal", "horizontal_movement"),
                    InputBinding("config.vertical", "vertical_movement"),
                    InputBinding("config.pan", "pan"),
                    InputBinding("config.tilt", "tilt"),
                    InputBinding("config.roll", "roll"),
                    InputBinding("config.zoom", "zoom"),
                ),
            ),
        )
    )


def _poll(
    source: str,
    path_template: str,
    *,
    media_family: Literal["image", "video"],
    result_path: tuple[str | int, ...],
    output: str,
) -> tuple[SubmitPoll, DownloadDecode]:
    return (
        SubmitPoll(
            "poll",
            source,
            status_path=("data", "task_status"),
            completed=("succeed",),
            failed=("failed",),
            queued=("submitted",),
            allowed=KLING_STATUSES,
            path_template=path_template,
            path_value_path=("data", "task_id"),
        ),
        DownloadDecode(
            "download",
            "poll",
            output=output,
            media_family=media_family,
            items_path=result_path,
            item_url_path=("url",),
        ),
    )


class KlingVirtualTryOnNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingVirtualTryOnNode",
            "Kling Virtual Try On",
            "partner/image/Kling",
            (
                InputSpec("human_image", IMAGE),
                InputSpec("cloth_image", IMAGE),
                _combo(
                    "model_name",
                    ("kolors-virtual-try-on-v1", "kolors-virtual-try-on-v1-5"),
                ),
            ),
            (OutputSpec("image", IMAGE),),
        )

    SPEC = OpSpec(
        (
            EncodeMedia("human", "human_image", max_pixels=MAX_IMAGE_PIXELS),
            EncodeMedia("cloth", "cloth_image", max_pixels=MAX_IMAGE_PIXELS),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/images/kolors-virtual-try-on",
                body=(
                    InputBinding("human_image", "human"),
                    InputBinding("cloth_image", "cloth"),
                    InputBinding("model_name", "model_name"),
                ),
            ),
            *_poll(
                "submit",
                "/proxy/kling/v1/images/kolors-virtual-try-on/{value}",
                media_family="image",
                result_path=("data", "task_result", "images"),
                output="image",
            ),
        )
    )


def _string(id: str, default: str = "", *, required: bool = True) -> InputSpec:
    return InputSpec(
        id,
        STRING,
        default=default,
        required=required,
        widget=StringWidget(multiline=True),
    )


def _int(id: str, default: int, minimum: int, maximum: int, *, required: bool = True) -> InputSpec:
    return InputSpec(
        id,
        INT,
        default=default,
        required=required,
        widget=NumberWidget(minimum, maximum, 1),
    )


def _bool(id: str, default: bool, *, required: bool = True) -> InputSpec:
    return InputSpec(id, BOOLEAN, default=default, required=required)


VIDEO_OUTPUTS = (
    OutputSpec("video", VIDEO),
    OutputSpec("video_id", STRING),
    OutputSpec("duration", STRING),
)


def _helper(
    id: str,
    helper_id: str,
    stage: str,
    placement: Literal["before", "after", "select"],
    outputs: tuple[str, ...],
    *,
    anchor: str = "",
    inputs: tuple[str, ...] = (),
    states: tuple[str, ...] = (),
    present: tuple[str, ...] = (),
    families: tuple[str, ...] = (),
    fixed: Mapping[str, object] | None = None,
) -> HelperCall:
    return HelperCall(
        id,
        helper_id,
        stage,
        placement,
        anchor,
        tuple(HelperBinding(name, name) for name in inputs)
        + tuple(HelperBinding(name, name, "state") for name in states)
        + tuple(HelperBinding(f"{name}_present", name, mode="present") for name in present)
        + tuple(HelperBinding(name, name, mode="family", optional=True) for name in families),
        {} if fixed is None else fixed,
        outputs,
    )


def _result_calls(
    creation_stage: str = "creation",
    result_stage: str = "video-result",
) -> tuple[HelperCall, HelperCall]:
    return (
        HelperCall(
            "created",
            "kling.task",
            creation_stage,
            "after",
            "submit",
            (HelperBinding("response", "submit", "state"),),
            {},
            ("task_id",),
        ),
        HelperCall(
            "result",
            "kling.task",
            result_stage,
            "after",
            "poll",
            (HelperBinding("response", "poll", "state"),),
            {},
            ("video" if "video" in result_stage else "images",),
        ),
    )


def _video_tail(path: str, *, max_attempts: int = 480) -> tuple[Adapter, ...]:
    return (
        SubmitPoll(
            "poll",
            "created.task_id",
            status_path=("data", "task_status"),
            completed=("succeed",),
            failed=("failed",),
            queued=("submitted",),
            allowed=KLING_STATUSES,
            path_template=path,
            path_value_path=("value",),
            max_attempts=max_attempts,
        ),
        DownloadDecode("download", "result.video", ("url",), "video", "video"),
        ResponseSelect("video_id", "result.video", ("id",), "video_id"),
        ResponseSelect("duration", "result.video", ("duration",), "duration"),
    )


def _legacy_schema(
    upstream: str,
    display: str,
    inputs: tuple[InputSpec, ...],
) -> NodeSchema:
    return _schema(upstream, display, "partner/video/Kling", inputs, VIDEO_OUTPUTS)


class KlingTextToVideoNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        modes = tuple(MODE_TEXT2VIDEO)
        return _legacy_schema(
            "KlingTextToVideoNode",
            "Kling Text to Video",
            (
                _string("prompt"),
                _string("negative_prompt"),
                InputSpec("cfg_scale", FLOAT, default=1.0, widget=NumberWidget(0.0, 1.0, 0.01)),
                _combo("aspect_ratio", ("16:9", "9:16", "1:1"), "16:9"),
                _combo("mode", modes, modes[8]),
            ),
        )

    SPEC = OpSpec(
        (
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/text2video",
                body=(
                    InputBinding("prompt", "prompt"),
                    InputBinding("negative_prompt", "negative_prompt", omit_if=""),
                    InputBinding("cfg_scale", "cfg_scale"),
                    InputBinding("aspect_ratio", "aspect_ratio"),
                    InputBinding("mode", "prepare.mode"),
                    InputBinding("duration", "prepare.duration"),
                    InputBinding("model_name", "prepare.model_name"),
                ),
            ),
            *_video_tail("/proxy/kling/v1/videos/text2video/{value}"),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.legacy-video",
                "text",
                "before",
                ("mode", "duration", "model_name"),
                anchor="submit",
                inputs=("prompt", "negative_prompt", "mode"),
                fixed={"modes": {key: list(value) for key, value in MODE_TEXT2VIDEO.items()}},
            ),
            *_result_calls(),
        ),
    )


def _image_video_spec(
    *,
    model_name: str | None = None,
    mode: str | None = None,
    duration: str | None = None,
    end_frame: bool = False,
) -> OpSpec:
    inputs = ("prompt", "negative_prompt", "model_name", "mode")
    fixed: tuple[FixedField, ...] = ()
    if model_name is not None:
        inputs += ("camera_control",)
        inputs = tuple(name for name in inputs if name != "model_name")
        fixed += (FixedField("model_name", model_name),)
    if mode is not None:
        inputs = tuple(name for name in inputs if name != "mode")
    if duration is not None:
        fixed += (FixedField("duration", duration),)
    body = (
        InputBinding("image", "image"),
        InputBinding("image_tail", "tail") if end_frame else InputBinding("prompt", "prompt"),
        InputBinding("prompt", "prompt")
        if end_frame
        else InputBinding("negative_prompt", "negative_prompt", omit_if=""),
        InputBinding("negative_prompt", "negative_prompt", omit_if="")
        if end_frame
        else InputBinding("cfg_scale", "cfg_scale"),
        InputBinding("cfg_scale", "cfg_scale")
        if end_frame
        else InputBinding("mode", "prepare.mode"),
        InputBinding("mode", "mode_values.mode")
        if end_frame
        else InputBinding("camera_control", "prepare.camera_control"),
        InputBinding("duration", "mode_values.duration")
        if end_frame
        else InputBinding("duration", "duration"),
        InputBinding("model_name", "mode_values.model_name")
        if end_frame
        else InputBinding("model_name", "model_name"),
    )
    if duration is not None:
        body = tuple(binding for binding in body if binding.target != "duration")
    if model_name is not None:
        body = tuple(binding for binding in body if binding.target != "model_name")
    helper_fixed: dict[str, object] = {}
    if mode is not None:
        helper_fixed["mode"] = mode
    if model_name is not None:
        helper_fixed["model_name"] = model_name
    return OpSpec(
        (
            MediaConstraints(
                "validate_image",
                "start_frame",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            EncodeMedia("image", "start_frame"),
            *((EncodeMedia("tail", "end_frame"),) if end_frame else ()),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/image2video",
                body=body,
                fixed=fixed,
            ),
            *_video_tail("/proxy/kling/v1/videos/image2video/{value}"),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.legacy-video",
                "image",
                "before",
                ("mode", "camera_control"),
                anchor="validate_image",
                inputs=inputs,
                fixed=helper_fixed,
            ),
            *_result_calls(),
        ),
    )


IMAGE_VIDEO_MODELS = (
    "kling-v1",
    "kling-v1-5",
    "kling-v1-6",
    "kling-v2-master",
    "kling-v2-1",
    "kling-v2-1-master",
    "kling-v2-5-turbo",
)


class KlingImage2VideoNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _legacy_schema(
            "KlingImage2VideoNode",
            "Kling Image(First Frame) to Video",
            (
                InputSpec("start_frame", IMAGE),
                _string("prompt"),
                _string("negative_prompt"),
                _combo("model_name", IMAGE_VIDEO_MODELS, "kling-v2-master"),
                InputSpec("cfg_scale", FLOAT, default=0.8, widget=NumberWidget(0.0, 1.0, 0.01)),
                _combo("mode", ("std", "pro"), "std"),
                _combo("aspect_ratio", ("16:9", "9:16", "1:1"), "16:9"),
                _combo("duration", ("5", "10"), "5"),
            ),
        )

    SPEC = _image_video_spec()


class KlingCameraControlI2VNode(KlingImage2VideoNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _legacy_schema(
            "KlingCameraControlI2VNode",
            "Kling Image to Video (Camera Control)",
            (
                InputSpec("start_frame", IMAGE),
                _string("prompt"),
                _string("negative_prompt"),
                InputSpec("cfg_scale", FLOAT, default=0.75, widget=NumberWidget(0.0, 1.0, 0.01)),
                _combo("aspect_ratio", ("16:9", "9:16", "1:1"), "16:9"),
                InputSpec("camera_control", CAMERA_CONTROL),
            ),
        )

    SPEC = _image_video_spec(model_name="kling-v1-5", mode="pro", duration="5")


class KlingCameraControlT2VNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _legacy_schema(
            "KlingCameraControlT2VNode",
            "Kling Text to Video (Camera Control)",
            (
                _string("prompt"),
                _string("negative_prompt"),
                InputSpec("cfg_scale", FLOAT, default=0.75, widget=NumberWidget(0.0, 1.0, 0.01)),
                _combo("aspect_ratio", ("16:9", "9:16", "1:1"), "16:9"),
                InputSpec("camera_control", CAMERA_CONTROL),
            ),
        )

    SPEC = OpSpec(
        (
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/text2video",
                body=(
                    InputBinding("prompt", "prompt"),
                    InputBinding("negative_prompt", "negative_prompt", omit_if=""),
                    InputBinding("cfg_scale", "cfg_scale"),
                    InputBinding("aspect_ratio", "aspect_ratio"),
                    InputBinding("camera_control", "camera_control"),
                ),
                fixed=(
                    FixedField("model_name", "kling-v1"),
                    FixedField("mode", "std"),
                    FixedField("duration", "5"),
                ),
            ),
            *_video_tail("/proxy/kling/v1/videos/text2video/{value}"),
        ),
        helper_calls=(
            _helper(
                "prevalidate",
                "kling.legacy-video",
                "text",
                "before",
                ("mode", "duration", "model_name"),
                anchor="submit",
                inputs=("prompt", "negative_prompt"),
                fixed={"mode": "fixed", "modes": {"fixed": ["std", "5", "kling-v1"]}},
            ),
            *_result_calls(),
        ),
    )


class KlingStartEndFrameNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        modes = tuple(MODE_START_END_FRAME)
        return _legacy_schema(
            "KlingStartEndFrameNode",
            "Kling Start-End Frame to Video",
            (
                InputSpec("start_frame", IMAGE),
                InputSpec("end_frame", IMAGE),
                _string("prompt"),
                _string("negative_prompt"),
                InputSpec("cfg_scale", FLOAT, default=0.5, widget=NumberWidget(0.0, 1.0, 0.01)),
                _combo("aspect_ratio", ("16:9", "9:16", "1:1")),
                _combo("mode", modes, modes[6]),
            ),
        )

    SPEC = OpSpec(
        _image_video_spec(end_frame=True).adapters,
        helper_calls=(
            _helper(
                "mode_values",
                "kling.legacy-video",
                "text",
                "before",
                ("mode", "duration", "model_name"),
                anchor="validate_image",
                inputs=("prompt", "negative_prompt", "mode"),
                fixed={"modes": {key: list(value) for key, value in MODE_START_END_FRAME.items()}},
            ),
            _helper(
                "prepare",
                "kling.legacy-video",
                "image",
                "before",
                ("mode", "camera_control"),
                anchor="validate_image",
                inputs=("prompt", "negative_prompt"),
                fixed={"mode": "pro", "camera_control": None},
            ),
            *_result_calls(),
        ),
    )


class KlingVideoExtendNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _legacy_schema(
            "KlingVideoExtendNode",
            "Kling Video Extend",
            (
                _string("prompt"),
                _string("negative_prompt"),
                InputSpec("cfg_scale", FLOAT, default=0.5, widget=NumberWidget(0.0, 1.0, 0.01)),
                _string("video_id"),
            ),
        )

    SPEC = OpSpec(
        (
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/video-extend",
                body=(
                    InputBinding("prompt", "prepare.prompt"),
                    InputBinding("negative_prompt", "prepare.negative_prompt"),
                    InputBinding("cfg_scale", "cfg_scale"),
                    InputBinding("video_id", "video_id"),
                ),
            ),
            *_video_tail("/proxy/kling/v1/videos/video-extend/{value}"),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.legacy-video",
                "extend",
                "before",
                ("prompt", "negative_prompt"),
                anchor="submit",
                inputs=("prompt", "negative_prompt"),
            ),
            *_result_calls(),
        ),
    )


def _lip_spec(stage: Literal["audio", "text"]) -> OpSpec:
    adapters: tuple[Adapter, ...] = (
        MediaConstraints(
            "validate_video",
            "video",
            min_width=720,
            max_width=1920,
            min_duration=2,
            max_duration=10,
        ),
        ProxyUpload("video_url", "video", "video.mp4", "video/mp4"),
    )
    if stage == "audio":
        adapters += (
            EncodeMedia("audio_mp3", "audio", "audio", "MP3", "bytes"),
            ProxyUpload("audio_url", "audio_mp3", "audio.mp3", "audio/mpeg"),
        )
        bindings = (
            HelperBinding("video_url", "video_url", "state"),
            HelperBinding("audio_url", "audio_url", "state"),
            HelperBinding("voice_language", "voice_language"),
        )
    else:
        bindings = (
            HelperBinding("video_url", "video_url", "state"),
            HelperBinding("text", "text"),
            HelperBinding("voice", "voice"),
            HelperBinding("voice_speed", "voice_speed"),
        )
    adapters += (
        HttpSyncJson(
            "submit",
            "/proxy/kling/v1/videos/lip-sync",
            body=(InputBinding("input", "prepare.input"),),
        ),
        *_video_tail("/proxy/kling/v1/videos/lip-sync/{value}"),
    )
    return OpSpec(
        adapters,
        helper_calls=(
            *(
                (
                    HelperCall(
                        "prevalidate",
                        "kling.lip-sync",
                        "text",
                        "before",
                        "validate_video",
                        tuple(
                            HelperBinding(name, name) for name in ("text", "voice", "voice_speed")
                        )
                        + (HelperBinding("video_url", "video_url", optional=True),),
                        {"voices": {key: list(value) for key, value in VOICES_CONFIG.items()}},
                        ("input",),
                    ),
                )
                if stage == "text"
                else ()
            ),
            HelperCall(
                "prepare",
                "kling.lip-sync",
                stage,
                "before",
                "submit",
                bindings,
                {"voices": {key: list(value) for key, value in VOICES_CONFIG.items()}}
                if stage == "text"
                else {},
                ("input",),
            ),
            *_result_calls(),
        ),
    )


class KlingLipSyncAudioToVideoNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _legacy_schema(
            "KlingLipSyncAudioToVideoNode",
            "Kling Lip Sync Video with Audio",
            (
                InputSpec("video", VIDEO),
                InputSpec("audio", AUDIO),
                _combo("voice_language", ("en", "zh"), "en"),
            ),
        )

    SPEC = _lip_spec("audio")


class KlingLipSyncTextToVideoNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _legacy_schema(
            "KlingLipSyncTextToVideoNode",
            "Kling Lip Sync Video with Text",
            (
                InputSpec("video", VIDEO),
                _string("text"),
                _combo("voice", tuple(VOICES_CONFIG), "Melody"),
                InputSpec("voice_speed", FLOAT, default=1.0, widget=NumberWidget(0.8, 2.0, 0.1)),
            ),
        )

    SPEC = _lip_spec("text")


def _effect_spec(stage: Literal["single", "dual"]) -> OpSpec:
    adapters: tuple[Adapter, ...]
    bindings: tuple[HelperBinding, ...]
    if stage == "single":
        adapters = (EncodeMedia("image", "image"),)
        bindings = (
            HelperBinding("image", "image", "state"),
            HelperBinding("model_name", "model_name"),
            HelperBinding("duration", "duration"),
        )
    else:
        adapters = (
            EncodeMedia("left", "image_left"),
            EncodeMedia("right", "image_right"),
            ValueConstruct(
                "images",
                "encoded_images",
                (
                    InputBinding("left", "left"),
                    InputBinding("right", "right"),
                ),
            ),
            BatchMapJoin(
                "image_list",
                segments=(Segment("images", wrap_key=None, mode="mapping_values"),),
            ),
        )
        bindings = (
            HelperBinding("images", "image_list", "state"),
            HelperBinding("model_name", "model_name"),
            HelperBinding("mode", "mode"),
            HelperBinding("duration", "duration"),
        )
    adapters += (
        HttpSyncJson(
            "submit",
            "/proxy/kling/v1/videos/effects",
            body=(
                InputBinding("effect_scene", "effect_scene"),
                InputBinding("input", "prepare.input"),
            ),
        ),
        *_video_tail("/proxy/kling/v1/videos/effects/{value}"),
    )
    return OpSpec(
        adapters,
        helper_calls=(
            *(
                (
                    HelperCall(
                        "prevalidate",
                        "kling.video-effect",
                        "single",
                        "before",
                        "image",
                        (
                            HelperBinding("model_name", "model_name"),
                            HelperBinding("duration", "duration"),
                        ),
                        {"image": "validation-placeholder"},
                        ("input",),
                    ),
                )
                if stage == "single"
                else ()
            ),
            HelperCall(
                "prepare",
                "kling.video-effect",
                stage,
                "before",
                "submit",
                bindings,
                {},
                ("input",),
            ),
            *_result_calls(),
        ),
    )


class KlingSingleImageVideoEffectNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _legacy_schema(
            "KlingSingleImageVideoEffectNode",
            "Kling Video Effects",
            (
                InputSpec("image", IMAGE),
                _combo(
                    "effect_scene",
                    ("squish", "expansion", "fuzzyfuzzy", "dizzydizzy", "bloombloom"),
                ),
                _combo("model_name", ("kling-v1-6",)),
                _combo("duration", ("5", "10")),
            ),
        )

    SPEC = _effect_spec("single")


class KlingDualCharacterVideoEffectNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingDualCharacterVideoEffectNode",
            "Kling Dual Character Video Effects",
            "partner/video/Kling",
            (
                InputSpec("image_left", IMAGE),
                InputSpec("image_right", IMAGE),
                _combo("effect_scene", ("hug", "kiss", "heart_gesture")),
                _combo("model_name", ("kling-v1", "kling-v1-5", "kling-v1-6"), "kling-v1"),
                _combo("mode", ("std", "pro"), "std"),
                _combo("duration", ("5", "10")),
            ),
            (OutputSpec("video", VIDEO), OutputSpec("duration", STRING)),
        )

    SPEC = _effect_spec("dual")


class KlingImageGenerationNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingImageGenerationNode",
            "Kling 3.0 Image",
            "partner/image/Kling",
            (
                _string("prompt"),
                _string("negative_prompt"),
                _combo("image_type", ("subject", "face")),
                InputSpec(
                    "image_fidelity", FLOAT, default=0.5, widget=NumberWidget(0.0, 1.0, 0.01)
                ),
                InputSpec(
                    "human_fidelity", FLOAT, default=0.45, widget=NumberWidget(0.0, 1.0, 0.01)
                ),
                _combo("model_name", ("kling-v3", "kling-v2", "kling-v1-5")),
                _combo(
                    "aspect_ratio",
                    ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"),
                    "16:9",
                ),
                _int("n", 1, 1, 9),
                InputSpec("image", IMAGE, default=None, required=False),
                _int("seed", 0, 0, 2147483647, required=False),
            ),
            (OutputSpec("image", IMAGE),),
        )

    SPEC = OpSpec(
        (
            EncodeMedia("image_data", "image", optional=True),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/images/generations",
                body=(
                    InputBinding("model_name", "model_name"),
                    InputBinding("prompt", "prompt"),
                    InputBinding("negative_prompt", "negative_prompt"),
                    InputBinding("image", "image_data"),
                    InputBinding("image_reference", "prepare.image_reference"),
                    InputBinding("image_fidelity", "image_fidelity"),
                    InputBinding("human_fidelity", "human_fidelity"),
                    InputBinding("n", "n"),
                    InputBinding("aspect_ratio", "aspect_ratio"),
                ),
            ),
            SubmitPoll(
                "poll",
                "created.task_id",
                status_path=("data", "task_status"),
                completed=("succeed",),
                failed=("failed",),
                queued=("submitted",),
                allowed=KLING_STATUSES,
                path_template="/proxy/kling/v1/images/generations/{value}",
                path_value_path=("value",),
            ),
            DownloadDecode(
                "download",
                "poll",
                output="image",
                media_family="image",
                items_path=("data", "task_result", "images"),
                item_url_path=("url",),
            ),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.image-generation",
                "prepare",
                "before",
                ("image_reference",),
                anchor="image_data",
                inputs=("prompt", "negative_prompt", "image_type"),
                present=("image",),
            ),
            *_result_calls(result_stage="image-result"),
        ),
    )


def _storyboard_combo(
    name: str = "storyboards", *, disabled_fields: bool = False
) -> DynamicComboSpec:
    options: list[DynamicComboOption] = []
    if disabled_fields:
        options.append(
            DynamicComboOption(
                "disabled",
                (
                    _string("prompt"),
                    _string("negative_prompt"),
                    _int("duration", 5, 3, 15),
                ),
            )
        )
    else:
        options.append(DynamicComboOption("disabled", ()))
    for count in range(1, 7):
        nested: list[InputSpec] = []
        for index in range(1, count + 1):
            nested.extend(
                (
                    _string(f"storyboard_{index}_prompt"),
                    _int(f"storyboard_{index}_duration", 4, 1, 15),
                )
            )
        options.append(
            DynamicComboOption(f"{count} storyboard" + ("s" if count > 1 else ""), tuple(nested))
        )
    return DynamicComboSpec(name, tuple(options))


def _omni_result_calls(*, image: bool = False) -> tuple[HelperCall, HelperCall]:
    return _result_calls(
        creation_stage="omni-creation",
        result_stage="omni-image-result" if image else "omni-video-result",
    )


def _omni_video_tail(path: str) -> tuple[Adapter, ...]:
    return (
        SubmitPoll(
            "poll",
            "created.task_id",
            status_path=("data", "task_status"),
            completed=("succeed",),
            failed=("failed",),
            queued=("submitted",),
            allowed=KLING_STATUSES,
            path_template=path,
            path_value_path=("value",),
        ),
        DownloadDecode("download", "result.video", ("url",), "video", "video"),
    )


OMNI_MODEL = ("kling-v3-omni", "kling-video-o1")
OMNI_RATIO = ("16:9", "9:16", "1:1")
OMNI_RESOLUTION = ("4k", "1080p", "720p")
OMNI_COMMON_OUTPUTS = (
    "prompt",
    "duration",
    "mode",
    "sound",
    "multi_shot",
    "multi_prompt",
    "shot_type",
)


class OmniProTextToVideoNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.kling.omni-pro-text-to-video",
            display_name="Kling 3.0 Omni Text to Video",
            category="partner/video/Kling",
            inputs=(
                _combo("model_name", OMNI_MODEL),
                _string("prompt"),
                _combo("aspect_ratio", OMNI_RATIO),
                _int("duration", 5, 3, 15),
                _combo("resolution", OMNI_RESOLUTION, "1080p"),
                _bool("generate_audio", False, required=False),
                _int("seed", 0, 0, 2147483647, required=False),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("KlingOmniProTextToVideoNode",),
            combos=(_storyboard_combo(),),
            io_bound=True,
        )

    _OUTPUTS = OMNI_COMMON_OUTPUTS
    SPEC = OpSpec(
        (
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/omni-video",
                body=(
                    InputBinding("model_name", "model_name"),
                    InputBinding("aspect_ratio", "aspect_ratio"),
                    *(InputBinding(name, f"prepare.{name}") for name in _OUTPUTS),
                ),
            ),
            *_omni_video_tail("/proxy/kling/v1/videos/omni-video/{value}"),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.omni-video",
                "text",
                "before",
                _OUTPUTS,
                anchor="submit",
                inputs=("model_name", "prompt", "duration", "resolution", "generate_audio"),
                families=("storyboards",),
            ),
            *_omni_result_calls(),
        ),
    )


def _encoded_image_list(
    source: str,
    count: int,
    *,
    optional: bool = False,
    fixed: Mapping[str, str] | None = None,
) -> tuple[Adapter, ...]:
    targets = tuple(f"image_{index}" for index in range(1, count + 1))
    return (
        EncodeMedia(
            "encoded_" + source,
            source,
            output="bytes",
            optional=optional,
            batch_targets=targets,
        ),
        ProxyUpload("uploaded_" + source, "encoded_" + source, batch=True),
        BatchMapJoin(
            "list_" + source,
            segments=(
                Segment(
                    "uploaded_" + source,
                    {} if fixed is None else fixed,
                    "image_url",
                    "mapping",
                ),
            ),
        ),
    )


class OmniProFirstLastFrameNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.kling.omni-pro-first-last-frame",
            display_name="Kling 3.0 Omni First-Last-Frame to Video",
            category="partner/video/Kling",
            inputs=(
                _combo("model_name", OMNI_MODEL),
                _string("prompt"),
                _int("duration", 5, 3, 15),
                InputSpec("first_frame", IMAGE),
                InputSpec("end_frame", IMAGE, default=None, required=False),
                InputSpec("reference_images", IMAGE, default=None, required=False),
                _combo("resolution", OMNI_RESOLUTION, "1080p"),
                _bool("generate_audio", False, required=False),
                _int("seed", 0, 0, 2147483647, required=False),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("KlingOmniProFirstLastFrameNode",),
            combos=(_storyboard_combo(),),
            io_bound=True,
        )

    _OUTPUTS = OMNI_COMMON_OUTPUTS + ("image_list",)
    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_reference_count",
                (
                    Check(
                        "The maximum number of reference images allowed is 6.",
                        require=(Cond("reference_images", "count_le", 6),),
                    ),
                ),
            ),
            MediaConstraints(
                "validate_first",
                "first_frame",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            MediaConstraints(
                "validate_end",
                "end_frame",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                optional=True,
                aspect_strict=True,
            ),
            MediaConstraints(
                "validate_references",
                "reference_images",
                min_width=300,
                min_height=300,
                max_count=6,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                optional=True,
                aspect_strict=True,
            ),
            EncodeMedia("first_bytes", "first_frame", output="bytes"),
            ProxyUpload("first_url", "first_bytes"),
            EncodeMedia("end_bytes", "end_frame", output="bytes", optional=True),
            ProxyUpload("end_url", "end_bytes", optional=True),
            *_encoded_image_list("reference_images", 6, optional=True),
            BatchMapJoin(
                "image_list",
                segments=(
                    Segment("first_url", {"type": "first_frame"}, "image_url", "single"),
                    Segment("end_url", {"type": "end_frame"}, "image_url", "single_optional"),
                    Segment("uploaded_reference_images", wrap_key="image_url", mode="mapping"),
                ),
            ),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/omni-video",
                body=(
                    InputBinding("model_name", "model_name"),
                    *(InputBinding(name, f"prepare.{name}") for name in _OUTPUTS),
                ),
            ),
            *_omni_video_tail("/proxy/kling/v1/videos/omni-video/{value}"),
        ),
        helper_calls=(
            _helper(
                "prevalidate",
                "kling.omni-video",
                "first-last",
                "before",
                _OUTPUTS,
                anchor="validate_reference_count",
                inputs=("model_name", "prompt", "duration", "resolution", "generate_audio"),
                present=("end_frame", "reference_images"),
                families=("storyboards",),
                fixed={"image_list": None},
            ),
            _helper(
                "prepare",
                "kling.omni-video",
                "first-last",
                "before",
                _OUTPUTS,
                anchor="submit",
                inputs=("model_name", "prompt", "duration", "resolution", "generate_audio"),
                states=("image_list",),
                present=("end_frame", "reference_images"),
                families=("storyboards",),
            ),
            *_omni_result_calls(),
        ),
    )


class OmniProImageToVideoNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.kling.omni-pro-image-to-video",
            display_name="Kling 3.0 Omni Image to Video",
            category="partner/video/Kling",
            inputs=(
                _combo("model_name", OMNI_MODEL),
                _string("prompt"),
                _combo("aspect_ratio", OMNI_RATIO),
                _int("duration", 5, 3, 15),
                InputSpec("reference_images", IMAGE),
                _combo("resolution", OMNI_RESOLUTION, "1080p"),
                _bool("generate_audio", False, required=False),
                _int("seed", 0, 0, 2147483647, required=False),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("KlingOmniProImageToVideoNode",),
            combos=(_storyboard_combo(),),
            io_bound=True,
        )

    _OUTPUTS = OMNI_COMMON_OUTPUTS + ("image_list",)
    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_reference_count",
                (
                    Check(
                        "The maximum number of reference images is 7.",
                        require=(Cond("reference_images", "count_le", 7),),
                    ),
                ),
            ),
            MediaConstraints(
                "validate_images",
                "reference_images",
                min_width=300,
                min_height=300,
                max_count=7,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            *_encoded_image_list("reference_images", 7),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/omni-video",
                body=(
                    InputBinding("model_name", "model_name"),
                    InputBinding("aspect_ratio", "aspect_ratio"),
                    *(InputBinding(name, f"prepare.{name}") for name in _OUTPUTS),
                ),
            ),
            *_omni_video_tail("/proxy/kling/v1/videos/omni-video/{value}"),
        ),
        helper_calls=(
            _helper(
                "prevalidate",
                "kling.omni-video",
                "images",
                "before",
                _OUTPUTS,
                anchor="validate_reference_count",
                inputs=("model_name", "prompt", "duration", "resolution", "generate_audio"),
                families=("storyboards",),
                fixed={"image_list": None, "list_reference_images": None},
            ),
            _helper(
                "prepare",
                "kling.omni-video",
                "images",
                "before",
                _OUTPUTS,
                anchor="submit",
                inputs=("model_name", "prompt", "duration", "resolution", "generate_audio"),
                states=("list_reference_images",),
                families=("storyboards",),
                fixed={"image_list": None},
            ),
            *_omni_result_calls(),
        ),
    )


def _omni_reference_spec(stage: Literal["video", "edit"]) -> OpSpec:
    video_input = "reference_video" if stage == "video" else "video"
    body = (
        (
            InputBinding("model_name", "model_name"),
            InputBinding("prompt", "prepare.prompt"),
            InputBinding("aspect_ratio", "aspect_ratio"),
            InputBinding("image_list", "prepare.image_list"),
            InputBinding("video_list", "prepare.video_list"),
            InputBinding("mode", "prepare.mode"),
        )
        if stage == "video"
        else (
            InputBinding("model_name", "model_name"),
            InputBinding("prompt", "prepare.prompt"),
            InputBinding("image_list", "prepare.image_list"),
            InputBinding("video_list", "prepare.video_list"),
            InputBinding("mode", "prepare.mode"),
        )
    )
    adapters: tuple[Adapter, ...] = (
        MediaConstraints(
            "validate_video",
            video_input,
            min_width=720,
            min_height=720,
            max_width=2160,
            max_height=2160,
            min_duration=3,
            max_duration=10.05,
        ),
        CheckInputs(
            "validate_reference_count",
            (
                Check(
                    "The maximum number of reference images allowed with a video input is 4.",
                    require=(Cond("reference_images", "count_le", 4),),
                ),
            ),
        ),
        MediaConstraints(
            "validate_references",
            "reference_images",
            min_width=300,
            min_height=300,
            max_count=4,
            min_aspect_ratio=0.4,
            max_aspect_ratio=2.5,
            optional=True,
            aspect_strict=True,
        ),
        ProxyUpload("video_url", video_input, "video.mp4", "video/mp4"),
        ValueConstruct(
            "video_param",
            "video_param_output",
            (
                InputBinding("video_url", "video_url"),
                InputBinding(
                    "keep_original_sound",
                    "keep_original_sound",
                    value_map={"true": "yes", "false": "no"},
                ),
            ),
            (FixedField("refer_type", "feature" if stage == "video" else "base"),),
        ),
        BatchMapJoin(
            "video_list",
            segments=(Segment("video_param", wrap_key=None, mode="verbatim"),),
        ),
        *_encoded_image_list("reference_images", 4, optional=True),
        HttpSyncJson(
            "submit",
            "/proxy/kling/v1/videos/omni-video",
            body=body,
            formatted=(FormatField("duration", "{duration}"),) if stage == "video" else (),
        ),
        *_omni_video_tail("/proxy/kling/v1/videos/omni-video/{value}"),
    )
    return OpSpec(
        adapters,
        helper_calls=(
            _helper(
                "prevalidate",
                "kling.omni-video",
                stage,
                "before",
                ("prompt", "mode", "image_list", "video_list"),
                anchor="validate_video",
                inputs=("prompt", "resolution"),
            ),
            _helper(
                "prepare",
                "kling.omni-video",
                stage,
                "before",
                ("prompt", "mode", "image_list", "video_list"),
                anchor="submit",
                inputs=("prompt", "resolution"),
                states=("list_reference_images", "video_list"),
            ),
            *_omni_result_calls(),
        ),
    )


class OmniProVideoToVideoNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingOmniProVideoToVideoNode",
            "Kling 3.0 Omni Video to Video",
            "partner/video/Kling",
            (
                _combo("model_name", OMNI_MODEL),
                _string("prompt"),
                _combo("aspect_ratio", OMNI_RATIO),
                _int("duration", 3, 3, 10),
                InputSpec("reference_video", VIDEO),
                _bool("keep_original_sound", True),
                InputSpec("reference_images", IMAGE, default=None, required=False),
                _combo("resolution", ("1080p", "720p"), "1080p"),
                _int("seed", 0, 0, 2147483647, required=False),
            ),
            (OutputSpec("video", VIDEO),),
        )

    SPEC = _omni_reference_spec("video")


class OmniProEditVideoNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingOmniProEditVideoNode",
            "Kling 3.0 Omni Edit Video",
            "partner/video/Kling",
            (
                _combo("model_name", OMNI_MODEL),
                _string("prompt"),
                InputSpec("video", VIDEO),
                _bool("keep_original_sound", True),
                InputSpec("reference_images", IMAGE, default=None, required=False),
                _combo("resolution", ("1080p", "720p"), "1080p"),
                _int("seed", 0, 0, 2147483647, required=False),
            ),
            (OutputSpec("video", VIDEO),),
        )

    SPEC = _omni_reference_spec("edit")


class OmniProImageNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingOmniProImageNode",
            "Kling 3.0 Omni Image",
            "partner/image/Kling",
            (
                _combo("model_name", ("kling-v3-omni", "kling-image-o1")),
                _string("prompt"),
                _combo("resolution", ("1K", "2K", "4K")),
                _combo("aspect_ratio", ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9")),
                _combo("series_amount", ("disabled", "2", "3", "4", "5", "6", "7", "8", "9")),
                InputSpec("reference_images", IMAGE, default=None, required=False),
                _int("seed", 0, 0, 2147483647, required=False),
            ),
            (OutputSpec("image", IMAGE),),
        )

    _OUTPUTS = ("prompt", "resolution", "result_type", "series_amount")
    SPEC = OpSpec(
        (
            CheckInputs(
                "validate_reference_count",
                (
                    Check(
                        "The maximum number of reference images is 10.",
                        require=(Cond("reference_images", "count_le", 10),),
                    ),
                ),
            ),
            MediaConstraints(
                "validate_references",
                "reference_images",
                min_width=300,
                min_height=300,
                max_count=10,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                optional=True,
                aspect_strict=True,
            ),
            *_encoded_image_list("reference_images", 10, optional=True),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/images/omni-image",
                body=(
                    InputBinding("model_name", "model_name"),
                    InputBinding("aspect_ratio", "aspect_ratio"),
                    InputBinding(
                        "image_list", "list_reference_images", present_if="reference_images"
                    ),
                    *(InputBinding(name, f"prepare.{name}") for name in _OUTPUTS),
                ),
            ),
            SubmitPoll(
                "poll",
                "created.task_id",
                status_path=("data", "task_status"),
                completed=("succeed",),
                failed=("failed",),
                queued=("submitted",),
                allowed=KLING_STATUSES,
                path_template="/proxy/kling/v1/images/omni-image/{value}",
                path_value_path=("value",),
            ),
            DownloadDecode(
                "download",
                "poll",
                output="image",
                media_family="image",
                url_paths=(
                    ("data", "task_result", "series_images"),
                    ("data", "task_result", "images"),
                ),
                item_url_path=("url",),
            ),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.omni-image",
                "prepare",
                "before",
                _OUTPUTS,
                anchor="validate_reference_count",
                inputs=("model_name", "prompt", "resolution", "series_amount"),
            ),
            *_omni_result_calls(image=True),
        ),
    )


def _new_video_tail(path: str, *, max_attempts: int = 480) -> tuple[Adapter, ...]:
    return (
        SubmitPoll(
            "poll",
            "created.task_id",
            status_path=("data", "task_status"),
            completed=("succeed",),
            failed=("failed",),
            queued=("submitted",),
            allowed=KLING_STATUSES,
            path_template=path,
            path_value_path=("value",),
            max_attempts=max_attempts,
        ),
        DownloadDecode("download", "result.video", ("url",), "video", "video"),
    )


class TextToVideoWithAudio(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingTextToVideoWithAudio",
            "Kling 2.6 Text to Video with Audio",
            "partner/video/Kling",
            (
                _combo("model_name", ("kling-v2-6",)),
                _string("prompt"),
                _combo("mode", ("pro",)),
                _combo("aspect_ratio", OMNI_RATIO),
                _combo("duration", ("5", "10")),
                _bool("generate_audio", True),
            ),
            (OutputSpec("video", VIDEO),),
        )

    SPEC = OpSpec(
        (
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/text2video",
                body=(
                    InputBinding("model_name", "model_name"),
                    InputBinding("prompt", "prompt"),
                    InputBinding("mode", "mode"),
                    InputBinding("aspect_ratio", "aspect_ratio"),
                    InputBinding("sound", "prepare.sound"),
                ),
                formatted=(FormatField("duration", "{duration}"),),
            ),
            *_new_video_tail("/proxy/kling/v1/videos/text2video/{value}"),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.audio-video",
                "text",
                "before",
                ("sound",),
                anchor="submit",
                inputs=("prompt", "generate_audio"),
            ),
            *_result_calls(),
        ),
    )


class ImageToVideoWithAudio(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingImageToVideoWithAudio",
            "Kling 2.6 Image(First Frame) to Video with Audio",
            "partner/video/Kling",
            (
                _combo("model_name", ("kling-v2-6",)),
                InputSpec("start_frame", IMAGE),
                _string("prompt"),
                _combo("mode", ("pro",)),
                _combo("duration", ("5", "10")),
                _bool("generate_audio", True),
            ),
            (OutputSpec("video", VIDEO),),
        )

    SPEC = OpSpec(
        (
            MediaConstraints(
                "validate",
                "start_frame",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            EncodeMedia("image_bytes", "start_frame", output="bytes"),
            ProxyUpload("image", "image_bytes"),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/image2video",
                body=(
                    InputBinding("model_name", "model_name"),
                    InputBinding("image", "image"),
                    InputBinding("prompt", "prompt"),
                    InputBinding("mode", "mode"),
                    InputBinding("sound", "prepare.sound"),
                ),
                formatted=(FormatField("duration", "{duration}"),),
            ),
            *_new_video_tail("/proxy/kling/v1/videos/image2video/{value}"),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.audio-video",
                "image",
                "before",
                ("sound",),
                anchor="validate",
                inputs=("prompt", "generate_audio"),
            ),
            *_result_calls(),
        ),
    )


def _motion_spec(max_duration: int) -> OpSpec:
    return OpSpec(
        (
            MediaConstraints(
                "validate_image",
                "reference_image",
                min_width=340,
                min_height=340,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            MediaConstraints(
                "validate_video",
                "reference_video",
                min_width=340,
                min_height=340,
                max_width=3850,
                max_height=3850,
                min_duration=3,
                max_duration=max_duration,
            ),
            EncodeMedia("image", "reference_image", output="bytes"),
            ProxyUpload("image_url", "image"),
            ProxyUpload("video_url", "reference_video", "video.mp4", "video/mp4"),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/motion-control",
                body=(
                    InputBinding("prompt", "prompt"),
                    InputBinding("image_url", "image_url"),
                    InputBinding("video_url", "video_url"),
                    InputBinding(
                        "keep_original_sound",
                        "keep_original_sound",
                        value_map={"true": "yes", "false": "no"},
                    ),
                    InputBinding("character_orientation", "character_orientation"),
                    InputBinding("mode", "mode"),
                    InputBinding("model_name", "model"),
                ),
            ),
            *_new_video_tail("/proxy/kling/v1/videos/motion-control/{value}"),
        ),
        helper_calls=_result_calls(),
    )


class MotionControl(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingMotionControl",
            "Kling Motion Control",
            "partner/video/Kling",
            (
                _string("prompt"),
                InputSpec("reference_image", IMAGE),
                InputSpec("reference_video", VIDEO),
                _bool("keep_original_sound", True),
                _combo("character_orientation", ("video", "image")),
                _combo("mode", ("pro", "std")),
                InputSpec(
                    "model",
                    COMBO,
                    default="kling-v2-6",
                    required=False,
                    widget=ComboWidget(("kling-v3", "kling-v2-6")),
                ),
            ),
            (OutputSpec("video", VIDEO),),
        )

    SPEC = Class3Spec(
        _helper(
            "select",
            "kling.motion-control",
            "prepare",
            "select",
            ("max_duration",),
            inputs=("prompt", "character_orientation"),
        ),
        {"image-orientation": _motion_spec(10), "video-orientation": _motion_spec(30)},
    )


class KlingFirstLastFrameNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.kling.first-last-frame",
            display_name="Kling 3.0 First-Last-Frame to Video",
            category="partner/video/Kling",
            inputs=(
                _string("prompt"),
                _int("duration", 5, 3, 15),
                InputSpec("first_frame", IMAGE),
                InputSpec("end_frame", IMAGE),
                _bool("generate_audio", True),
                _int("seed", 0, 0, 2147483647),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("KlingFirstLastFrameNode",),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "kling-v3", (_combo("resolution", ("4k", "1080p", "720p"), "1080p"),)
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = OpSpec(
        (
            MediaConstraints(
                "validate_first",
                "first_frame",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            MediaConstraints(
                "validate_end",
                "end_frame",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            EncodeMedia("image_bytes", "first_frame", output="bytes"),
            ProxyUpload("image", "image_bytes"),
            EncodeMedia("image_tail_bytes", "end_frame", output="bytes"),
            ProxyUpload("image_tail", "image_tail_bytes"),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/image2video",
                body=(
                    InputBinding("model_name", "prepare.model_name"),
                    InputBinding("image", "image"),
                    InputBinding("image_tail", "image_tail"),
                    InputBinding("prompt", "prompt"),
                    InputBinding("mode", "prepare.mode"),
                    InputBinding(
                        "sound", "generate_audio", value_map={"true": "on", "false": "off"}
                    ),
                ),
                formatted=(FormatField("duration", "{duration}"),),
            ),
            *_new_video_tail("/proxy/kling/v1/videos/image2video/{value}"),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.first-last-frame",
                "prepare",
                "before",
                ("mode", "model_name"),
                anchor="validate_first",
                inputs=("prompt",),
                families=("model",),
            ),
            *_result_calls(),
        ),
    )


class KlingAvatarNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "KlingAvatarNode",
            "Kling Avatar 2.0",
            "partner/video/Kling",
            (
                InputSpec("image", IMAGE),
                InputSpec("sound_file", AUDIO),
                _combo("mode", ("std", "pro")),
                _string("prompt", "", required=False),
                _int("seed", 0, 0, 2147483647),
            ),
            (OutputSpec("video", VIDEO),),
        )

    SPEC = OpSpec(
        (
            MediaConstraints(
                "validate_image",
                "image",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            MediaConstraints(
                "validate_audio",
                "sound_file",
                duration_media="audio",
                min_duration=2,
                max_duration=300,
            ),
            EncodeMedia("image_bytes", "image", output="bytes"),
            ProxyUpload("image_url", "image_bytes"),
            EncodeMedia("audio_bytes", "sound_file", "audio", "MP3", "bytes"),
            ProxyUpload("sound_url", "audio_bytes", "audio.mp3", "audio/mpeg"),
            HttpSyncJson(
                "submit",
                "/proxy/kling/v1/videos/avatar/image2video",
                body=(
                    InputBinding("image", "image_url"),
                    InputBinding("sound_file", "sound_url"),
                    InputBinding("prompt", "prepare.prompt"),
                    InputBinding("mode", "mode"),
                ),
            ),
            *_new_video_tail("/proxy/kling/v1/videos/avatar/image2video/{value}", max_attempts=800),
        ),
        helper_calls=(
            _helper(
                "prepare",
                "kling.avatar",
                "prepare",
                "before",
                ("prompt",),
                anchor="submit",
                inputs=("prompt",),
            ),
            *_result_calls(),
        ),
    )


_VIDEO_SELECTOR_OUTPUTS = (
    "prompt",
    "negative_prompt",
    "duration",
    "duration_string",
    "resolution",
    "aspect_ratio",
    "model_name",
    "mode",
    "multi_shot",
    "multi_prompt",
    "shot_type",
)


def _kling_v3_video_spec(*, image: bool) -> OpSpec:
    prefix: tuple[Adapter, ...] = ()
    path = "/proxy/kling/v1/videos/text2video"
    body: tuple[InputBinding, ...] = (
        InputBinding("model_name", "select.model_name"),
        InputBinding("aspect_ratio", "select.aspect_ratio"),
        InputBinding("prompt", "select.prompt"),
        InputBinding("negative_prompt", "select.negative_prompt"),
        InputBinding("mode", "select.mode"),
        InputBinding("duration", "select.duration_string"),
        InputBinding("sound", "generate_audio", value_map={"true": "on", "false": "off"}),
        InputBinding("multi_shot", "select.multi_shot"),
        InputBinding("multi_prompt", "select.multi_prompt"),
        InputBinding("shot_type", "select.shot_type"),
    )
    if image:
        path = "/proxy/kling/v1/videos/image2video"
        prefix = (
            MediaConstraints(
                "validate",
                "start_frame",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            EncodeMedia("image_bytes", "start_frame", output="bytes"),
            ProxyUpload("image", "image_bytes"),
        )
        body = (InputBinding("image", "image"),) + tuple(
            item for item in body if item.target != "aspect_ratio"
        )
    return OpSpec(
        (
            *prefix,
            HttpSyncJson(
                "submit",
                path,
                body=body,
            ),
            *_new_video_tail(path + "/{value}"),
        ),
        helper_calls=_result_calls(),
    )


def _kling_turbo_spec(*, image: bool) -> OpSpec:
    settings = ValueConstruct(
        "settings",
        "settings_value",
        (
            InputBinding("resolution", "select.resolution"),
            InputBinding("duration", "select.duration"),
        )
        + (() if image else (InputBinding("aspect_ratio", "select.aspect_ratio"),)),
    )
    prefix: tuple[Adapter, ...] = (settings,)
    path = "/proxy/kling/text-to-video/kling-3.0-turbo"
    body: tuple[InputBinding, ...] = (
        InputBinding("prompt", "select.prompt"),
        InputBinding("settings", "settings"),
    )
    if image:
        path = "/proxy/kling/image-to-video/kling-3.0-turbo"
        prefix = (
            MediaConstraints(
                "validate",
                "start_frame",
                min_width=300,
                min_height=300,
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
            EncodeMedia("image", "start_frame"),
            ValueConstruct(
                "prompt_content",
                "prompt_content_value",
                (InputBinding("text", "select.prompt"),),
                (FixedField("type", "prompt"),),
            ),
            ValueConstruct(
                "frame_content",
                "frame_content_value",
                (InputBinding("url", "image"),),
                (FixedField("type", "first_frame"),),
            ),
            BatchMapJoin(
                "contents",
                segments=(
                    Segment("prompt_content", wrap_key=None, mode="verbatim"),
                    Segment("frame_content", wrap_key=None, mode="verbatim"),
                ),
            ),
            settings,
        )
        body = (InputBinding("contents", "contents"), InputBinding("settings", "settings"))
    return OpSpec(
        (
            *prefix,
            HttpSyncJson("submit", path, body=body),
            SubmitPoll(
                "poll",
                "created.task_id",
                status_path=("data", 0, "status"),
                completed=("succeeded",),
                failed=("failed",),
                queued=("submitted",),
                allowed=("submitted", "processing", "succeeded", "failed"),
                path_template="/proxy/kling/tasks?task_ids={value}",
                path_value_path=("value",),
            ),
            DownloadDecode("download", "result.video", ("url",), "video", "video"),
        ),
        helper_calls=_result_calls("turbo-creation", "turbo-video-result"),
    )


class KlingVideoNode(KlingNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="partner.kling.video",
            display_name="Kling 3.0 Video",
            category="partner/video/Kling",
            inputs=(
                _bool("generate_audio", True),
                _int("seed", 0, 0, 2147483647),
                InputSpec("start_frame", IMAGE, default=None, required=False),
            ),
            outputs=(OutputSpec("video", VIDEO),),
            aliases=("KlingVideoNode",),
            combos=(
                _storyboard_combo("multi_shot", disabled_fields=True),
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "kling-v3",
                            (
                                _combo("resolution", ("4k", "1080p", "720p"), "1080p"),
                                _combo("aspect_ratio", OMNI_RATIO),
                            ),
                        ),
                        DynamicComboOption(
                            "kling-3.0-turbo",
                            (
                                _combo("resolution", ("1080p", "720p"), "720p"),
                                _combo("aspect_ratio", OMNI_RATIO),
                            ),
                        ),
                    ),
                ),
            ),
            io_bound=True,
        )

    SPEC = Class3Spec(
        _helper(
            "select",
            "kling.video",
            "prepare",
            "select",
            _VIDEO_SELECTOR_OUTPUTS,
            present=("start_frame",),
            families=("multi_shot", "model"),
        ),
        {
            "turbo-image": _kling_turbo_spec(image=True),
            "turbo-text": _kling_turbo_spec(image=False),
            "v3-image": _kling_v3_video_spec(image=True),
            "v3-text": _kling_v3_video_spec(image=False),
        },
    )


KLING_NODES: list[type[Node]] = [
    KlingCameraControls,
    KlingTextToVideoNode,
    KlingImage2VideoNode,
    KlingCameraControlI2VNode,
    KlingCameraControlT2VNode,
    KlingStartEndFrameNode,
    KlingVideoExtendNode,
    KlingLipSyncAudioToVideoNode,
    KlingLipSyncTextToVideoNode,
    KlingVirtualTryOnNode,
    KlingImageGenerationNode,
    KlingSingleImageVideoEffectNode,
    KlingDualCharacterVideoEffectNode,
    OmniProTextToVideoNode,
    OmniProFirstLastFrameNode,
    OmniProImageToVideoNode,
    OmniProVideoToVideoNode,
    OmniProEditVideoNode,
    OmniProImageNode,
    TextToVideoWithAudio,
    ImageToVideoWithAudio,
    MotionControl,
    KlingFirstLastFrameNode,
    KlingAvatarNode,
    KlingVideoNode,
]
