"""Dinkster's first-party asset-backed media I/O nodes."""

import json

from dinkster_api.v1 import (
    ASSET_TYPE,
    IMAGE_BATCH_MERGER_ID,
    PNG_CONTAINER_VERSION,
    SAVE_TARGET_TYPE,
    SPLAT_FILE_DECODER_ID,
    SPLAT_PLY_MIME,
    TypeRegistry,
    decode_image_array,
    decode_splat,
    decode_splat_file,
    encode_image_array,
    encode_splat,
    image_array_fingerprint,
    image_array_meta,
    image_input,
    mask_array_meta,
    merge_image_batches,
    prepare_image_array_encoding,
    register_asset_type,
    register_audio_value_type,
    register_curve_type,
    register_model3d_type,
    register_save_target_type,
    register_video_value_type,
    render_image_png,
    render_mask_png,
    render_splat_ply,
    resolver_from_env,
    splat_fingerprint,
    splat_meta,
    validate_image_encoded,
    validate_splat_encoded,
)
from dinkster_api.v1 import video_document as document_codec

from .audio import (
    AUDIO_IO_NODES,
    EmptyAudio,
    LoadAudio,
    PreviewAudio,
    SaveAudio,
    SaveAudioMP3,
    SaveAudioOpus,
)
from .audio_ops import (
    AUDIO_OPS_NODES,
    AdjustAudioVolume,
    AudioOnsets,
    ConcatAudio,
    DownmixAudio,
    EqualizeAudio,
    ExtractAudioEnvelope,
    FadeAudio,
    JoinAudioChannels,
    MergeAudio,
    ResampleAudio,
    SplitAudioChannels,
    TrimAudio,
)
from .capture import (
    AUDIO_DEVICE_CHOICES_ID,
    CAPTURE_NODES,
    VIDEO_DEVICE_CHOICES_ID,
    AudioCaptureDevice,
    AudioCaptureProvider,
    CaptureCancelled,
    CaptureDeviceNotFound,
    CaptureError,
    CapturePermissionDenied,
    CaptureTimeout,
    RecordAudio,
    VideoCaptureDevice,
    VideoCaptureProvider,
    WebcamCapture,
    capture_device_choices,
)
from .image import (
    IMAGE_FILE_DECODER_ID,
    IMAGE_IO_NODES,
    MASK_FILE_DECODER_ID,
    LoadImage,
    LoadImageOutput,
    LoadMask,
    PaintMask,
    PreviewImage,
    ReadImageMetadata,
    SaveAnimatedImage,
    SaveImage,
    SaveMask,
    decode_image_file,
    decode_mask_file,
)
from .image_document import IMAGE_DOCUMENT_NODES, RenderImageDocument
from .model3d import (
    MODEL3D_NODES,
    MODEL3D_TYPE,
    LoadModel3D,
    PreviewModel3D,
    SaveModel3D,
)
from .primitives import SAVE_TARGET_NODES, SetSaveTargetPrefix
from .splat import SPLAT_NODES, SPLAT_TYPE, LoadGaussianSplat, SaveGaussianSplat
from .text import TEXT_IO_NODES, SaveText
from .video import (
    AUDIO_TYPE,
    COMPAT_IMAGE_TYPE,
    IMAGE_TYPE,
    VIDEO_NODES,
    VIDEO_TYPE,
    LoadVideo,
    LoadVideoValue,
    SaveVideo,
    SaveVideoValue,
)
from .video_document import VIDEO_DOCUMENT_NODES
from .video_ops import (
    VIDEO_OPS_NODES,
    AssembleVideo,
    CropVideo,
    DisassembleVideo,
    TrimVideo,
    VideoFrameRate,
    VideoFrameWindow,
    VideoInfo,
)

MASK_TYPE = "dinkster.mask"

MEDIA_IO_NODES = [
    *SAVE_TARGET_NODES,
    *IMAGE_IO_NODES,
    *IMAGE_DOCUMENT_NODES,
    *VIDEO_NODES,
    *VIDEO_OPS_NODES,
    *VIDEO_DOCUMENT_NODES,
    *AUDIO_IO_NODES,
    *AUDIO_OPS_NODES,
    *MODEL3D_NODES,
    *SPLAT_NODES,
    *TEXT_IO_NODES,
    *CAPTURE_NODES,
]


def register_media_types(registry: TypeRegistry) -> None:
    """Register the asset, save-target, and media values used by this pack."""
    register_curve_type(registry)
    if document_codec.DOCUMENT_TYPE not in registry:
        registry.register(
            document_codec.DOCUMENT_TYPE,
            encode=document_codec.encode_document,
            decode=document_codec.decode_document,
            coerce=document_codec.document,
            fingerprint=document_codec.document_fingerprint,
            meta=document_codec.document_meta,
        )
    if "comfy.VIDEO_EDIT" not in registry:
        registry.register(
            "comfy.VIDEO_EDIT",
            encode=lambda obj: json.dumps(obj, sort_keys=True, allow_nan=False).encode(),
            decode=json.loads,
        )
    if SAVE_TARGET_TYPE not in registry:
        register_save_target_type(registry)
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    for type_id, renderer in (
        (IMAGE_TYPE, render_image_png),
        (COMPAT_IMAGE_TYPE, render_image_png),
        (MASK_TYPE, render_mask_png),
    ):
        if type_id not in registry:
            registry.register(
                type_id,
                encode=encode_image_array,
                decode=decode_image_array,
                prepare_buffer_encoding=prepare_image_array_encoding,
                fingerprint=image_array_fingerprint(type_id),
                meta=mask_array_meta if type_id == MASK_TYPE else image_array_meta,
                input_convert=image_input,
                validate_encoded=validate_image_encoded,
                validate_encoded_buffer=validate_image_encoded,
            )
            registry.register_rendition(
                type_id,
                "png",
                mime="image/png",
                render=renderer,
                version=PNG_CONTAINER_VERSION,
            )
    if registry.asset_decoder_for(IMAGE_TYPE) is None:
        registry.register_asset_decoder(
            IMAGE_TYPE,
            provider_id=IMAGE_FILE_DECODER_ID,
            decode=decode_image_file,
        )
    if registry.batch_merge_for(IMAGE_TYPE) is None:
        registry.register_batch_merge(
            IMAGE_TYPE,
            provider_id=IMAGE_BATCH_MERGER_ID,
            merge=merge_image_batches,
        )
    if registry.asset_decoder_for(MASK_TYPE) is None:
        registry.register_asset_decoder(
            MASK_TYPE,
            provider_id=MASK_FILE_DECODER_ID,
            decode=decode_mask_file,
        )
    if AUDIO_TYPE not in registry:
        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())
    if VIDEO_TYPE not in registry:
        register_video_value_type(registry, VIDEO_TYPE, resolver_from_env())
    register_model3d_type(registry, MODEL3D_TYPE)
    if SPLAT_TYPE not in registry:
        registry.register(
            SPLAT_TYPE,
            encode=encode_splat,
            decode=decode_splat,
            fingerprint=splat_fingerprint(SPLAT_TYPE),
            meta=splat_meta,
            validate_encoded_buffer=validate_splat_encoded,
        )
        registry.register_rendition(
            SPLAT_TYPE,
            "ply",
            mime=SPLAT_PLY_MIME,
            render=render_splat_ply,
        )
        registry.register_asset_decoder(
            SPLAT_TYPE,
            provider_id=SPLAT_FILE_DECODER_ID,
            decode=decode_splat_file,
        )


__all__ = [
    "AUDIO_DEVICE_CHOICES_ID",
    "AUDIO_IO_NODES",
    "AUDIO_OPS_NODES",
    "AUDIO_TYPE",
    "CAPTURE_NODES",
    "COMPAT_IMAGE_TYPE",
    "IMAGE_TYPE",
    "IMAGE_IO_NODES",
    "IMAGE_DOCUMENT_NODES",
    "MASK_TYPE",
    "MEDIA_IO_NODES",
    "MODEL3D_NODES",
    "MODEL3D_TYPE",
    "SAVE_TARGET_NODES",
    "SPLAT_NODES",
    "SPLAT_TYPE",
    "TEXT_IO_NODES",
    "VIDEO_DEVICE_CHOICES_ID",
    "VIDEO_NODES",
    "VIDEO_OPS_NODES",
    "VIDEO_TYPE",
    "AdjustAudioVolume",
    "AssembleVideo",
    "AudioOnsets",
    "AudioCaptureDevice",
    "AudioCaptureProvider",
    "CaptureCancelled",
    "CaptureDeviceNotFound",
    "CaptureError",
    "CapturePermissionDenied",
    "CaptureTimeout",
    "ConcatAudio",
    "CropVideo",
    "DisassembleVideo",
    "DownmixAudio",
    "EmptyAudio",
    "EqualizeAudio",
    "ExtractAudioEnvelope",
    "FadeAudio",
    "JoinAudioChannels",
    "LoadAudio",
    "LoadGaussianSplat",
    "LoadImage",
    "LoadImageOutput",
    "LoadMask",
    "PaintMask",
    "LoadModel3D",
    "LoadVideo",
    "LoadVideoValue",
    "MergeAudio",
    "PreviewAudio",
    "PreviewModel3D",
    "PreviewImage",
    "ReadImageMetadata",
    "RenderImageDocument",
    "RecordAudio",
    "ResampleAudio",
    "SaveAnimatedImage",
    "SaveAudio",
    "SaveAudioMP3",
    "SaveAudioOpus",
    "SaveGaussianSplat",
    "SaveImage",
    "SaveMask",
    "SaveModel3D",
    "SaveText",
    "SaveVideo",
    "SaveVideoValue",
    "SetSaveTargetPrefix",
    "SplitAudioChannels",
    "TrimAudio",
    "TrimVideo",
    "VideoCaptureDevice",
    "VideoCaptureProvider",
    "VideoFrameRate",
    "VideoFrameWindow",
    "VideoInfo",
    "WebcamCapture",
    "capture_device_choices",
    "register_media_types",
]
