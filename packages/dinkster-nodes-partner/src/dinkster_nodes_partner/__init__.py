"""Partner API nodes and their pack-local runtime."""

from dinkster_api.v1 import (
    PNG_CONTAINER_VERSION,
    TypeRegistry,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    render_image_png,
)

from .bfl import BFL_NODES
from .grok import GROK_NODES
from .kling import KLING_NODES
from .opspec import (
    ADAPTER_KINDS,
    Adapter,
    BatchMapJoin,
    Check,
    CheckInputs,
    Cond,
    DownloadDecode,
    EncodeMedia,
    FixedField,
    FormatField,
    HttpSyncBinary,
    HttpSyncJson,
    InputBinding,
    LocalProgress,
    MaskPrepare,
    MediaConstraints,
    MultipartMap,
    MultiStage,
    OpSpec,
    ProxyUpload,
    ResponseSelect,
    Segment,
    SubmitPoll,
    ValueConstruct,
)
from .partner_runtime import (
    ApiServerError,
    LocalNetworkError,
    MissingApiKeyError,
    PartnerError,
    RuntimeContext,
    TrustPolicyError,
    run_op,
    worker_runtime_context,
)
from .wan import WAN_NODES

PARTNER_NODES: list[type[object]] = [*BFL_NODES, *GROK_NODES, *KLING_NODES, *WAN_NODES]


def register_partner_types(registry: TypeRegistry) -> None:
    registry.register("partner.kling.camera-control")
    for type_id in ("comfy.IMAGE", "comfy.MASK"):
        registry.register(
            type_id,
            encode=encode_image_array,
            decode=decode_image_array,
            fingerprint=image_array_fingerprint(type_id),
            meta=image_array_meta,
        )
    registry.register_rendition(
        "comfy.IMAGE",
        "png",
        mime="image/png",
        render=render_image_png,
        version=PNG_CONTAINER_VERSION,
    )


__all__ = [
    "ADAPTER_KINDS",
    "PARTNER_NODES",
    "Adapter",
    "ApiServerError",
    "BatchMapJoin",
    "Check",
    "CheckInputs",
    "Cond",
    "DownloadDecode",
    "EncodeMedia",
    "FixedField",
    "FormatField",
    "HttpSyncBinary",
    "HttpSyncJson",
    "InputBinding",
    "LocalNetworkError",
    "LocalProgress",
    "MaskPrepare",
    "MediaConstraints",
    "MissingApiKeyError",
    "MultiStage",
    "MultipartMap",
    "OpSpec",
    "PartnerError",
    "ProxyUpload",
    "ResponseSelect",
    "Segment",
    "RuntimeContext",
    "SubmitPoll",
    "TrustPolicyError",
    "ValueConstruct",
    "register_partner_types",
    "run_op",
    "worker_runtime_context",
]
