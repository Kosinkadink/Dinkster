"""Asset-native model loading: the properly-ported loader.

v1 CheckpointLoaderSimple takes a filename from a folder listing - a
location-dependent name that breaks the moment a file moves, renames, or
lives on another machine. This node takes a ``dinkster.asset`` instead: the
graph names the checkpoint by content digest, cache identity *is* that
digest (location-independent by construction), and the AssetRef becomes a
real path only inside the worker, via its configured asset store.

Loader, LoRA, hook, Wan image-conditioning, and sampler bodies here use
ComfyUI objects on the compatibility arm. The native arm overrides those
bodies rather than passing native resident handles to ComfyUI. Empty image
and Hunyuan video latents also run standalone; absent ComfyUI, they use its
default CPU intermediate placement.

Environment contract (child process, in addition to bootstrap's):

- ``DINKSTER_ASSET_VAULT``: optional upload vault in AssetVault's sharded CAS
  layout. This is checked first so server uploads execute without copying.
- ``DINKSTER_MOUNTS_SNAPSHOT``: optional path to the engine-published mount
  snapshot (MountTable.write_snapshot). The resolver re-reads it whenever
  it changes, so mounts granted at runtime materialize here live.
- ``DINKSTER_ASSET_ROOT``: optional directory holding this worker's local
  asset content (typically the ComfyUI ``models/`` dir). Its
  ``.dinkster-asset-index.json`` must already exist - written by a scanning
  LocalAssetLibrary where Dinkster proper runs; the worker only resolves
  digests, it never scans or hashes. Unset means asset inputs cannot be
  materialized here and fail with a clear AssetError when a node asks
  for bytes.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from dinkster_assets import (
    ASSET_TYPE,
    KIND_MODEL_DIFFUSION,
    KIND_MODEL_LORA,
    KIND_MODEL_TEXT_ENCODER,
    KIND_MODEL_VAE,
    LATENT_ASSET_KIND,
    LATENT_MEDIA_TYPE,
    LATENT_SUFFIX,
    MAX_LATENT_DATA_BYTES,
    MAX_LATENT_HEADER_BYTES,
    SAVE_TARGET_TYPE,
    AssetError,
    AssetRef,
    AssetWriter,
    MountSnapshotWriter,
    register_asset_type,
    register_save_target_type,
    resolver_from_env,
)
from dinkster_inference import register_inference_types
from dinkster_inference.sampling_wire import register_sampling_type
from dinkster_schema import (
    AssetWidget,
    ComboWidget,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SaveTargetWidget,
    SourceFilenameSpec,
    StringWidget,
    TypeExpr,
    WidgetRepresentation,
    WidgetRepresentations,
)
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    TypeRegistry,
    register_curve_type,
    register_model3d_type,
)
from dinkster_workers import current_execution_context

from .audio import register_audio_type
from .devices import comfy_resident_meta
from .image import register_image_asset_providers, register_image_type
from .latent import register_latent_type
from .native_residency import select_intermediate_device
from .pool import default_pool
from .resident import register_resident_type
from .type_ids import comfy_type_id
from .usdu import USDU_CARRIER_NODES
from .video import register_video_type

ASSET = TypeExpr.concrete(ASSET_TYPE)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)
LORA_EXECUTION_MODES = ("auto", "attach", "precalculate")
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
COMBO = TypeExpr.concrete(CORE_COMBO)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
MODEL = TypeExpr.concrete(comfy_type_id("MODEL"))
CLIP = TypeExpr.concrete(comfy_type_id("CLIP"))
CLIP_VISION = TypeExpr.concrete(comfy_type_id("CLIP_VISION"))
VAE = TypeExpr.concrete(comfy_type_id("VAE"))
DINKSTER_CLIP = TypeExpr.concrete("dinkster.clip")
DINKSTER_CLIP_VISION = TypeExpr.concrete("dinkster.clip-vision")
DINKSTER_VAE = TypeExpr.concrete("dinkster.vae")
DINKSTER_LATENT = TypeExpr.concrete("dinkster.latent")
DINKSTER_CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
CONTROL_NET = TypeExpr.concrete(comfy_type_id("CONTROL_NET"))
MODEL_PATCH = TypeExpr.concrete(comfy_type_id("MODEL_PATCH"))
IMAGE = TypeExpr.concrete(comfy_type_id("IMAGE"))
AUDIO = TypeExpr.concrete(comfy_type_id("AUDIO"))
VIDEO = TypeExpr.concrete(comfy_type_id("VIDEO"))
MASK = TypeExpr.concrete(comfy_type_id("MASK"))
LATENT = TypeExpr.concrete(comfy_type_id("LATENT"))
CONDITIONING = TypeExpr.concrete(comfy_type_id("CONDITIONING"))
CLIP_VISION_OUTPUT = TypeExpr.concrete(comfy_type_id("CLIP_VISION_OUTPUT"))
WAN_CAMERA_EMBEDDING = TypeExpr.concrete(comfy_type_id("WAN_CAMERA_EMBEDDING"))
TRACKS = TypeExpr.concrete(comfy_type_id("TRACKS"))
MINIMAX_H3_REFERENCE = TypeExpr.concrete("dinkster.minimax_h3.reference")
_CHECKPOINT_KIND = "model/checkpoint"
SCHEDULED_HOOKS_KEY = "dinkster.native/hooks"
WAN_CAMERA_POSES = (
    "Static",
    "Pan Up",
    "Pan Down",
    "Pan Left",
    "Pan Right",
    "Zoom In",
    "Zoom Out",
    "Anti Clockwise (ACW)",
    "ClockWise (CW)",
)


@dataclass(frozen=True, slots=True)
class ScheduledHookKeyframes:
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class ScheduledLoraHook:
    lora: AssetRef
    strength_model: float
    strength_clip: float
    keyframes: ScheduledHookKeyframes | None = None


@dataclass(frozen=True, slots=True)
class ScheduledHooks:
    loras: tuple[ScheduledLoraHook, ...]


@dataclass(frozen=True, slots=True)
class MiniMaxH3ImageReferenceValue:
    image: object


@dataclass(frozen=True, slots=True)
class MiniMaxH3AudioReferenceValue:
    waveform: object
    sample_rate: int


@dataclass(frozen=True, slots=True)
class MiniMaxH3VideoReferenceValue:
    frames: object
    audio: MiniMaxH3AudioReferenceValue | None


def intermediate_dtype(torch: Any, arguments: Iterable[str] | None = None) -> Any:
    """ComfyUI's intermediate dtype from the sanitized worker arguments."""
    supplied = sys.argv[1:] if arguments is None else arguments
    return torch.float16 if "--fp16-intermediates" in supplied else torch.float32


class LoadCheckpoint(Node):
    """Load a checkpoint named by content digest, not by filename."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_checkpoint",
            # The "(asset)" suffix died with the translated twin: once the
            # alias claim evicts comfy.CheckpointLoaderSimple, this IS the
            # checkpoint loader.
            display_name="Load Checkpoint",
            category="loaders",
            description=(
                "Loads MODEL/CLIP/VAE from a checkpoint asset. Identity is "
                "the file's content digest: the same bytes hit the same "
                "cache entries on any machine, under any filename."
            ),
            inputs=(
                InputSpec(
                    "checkpoint",
                    ASSET,
                    widget=AssetWidget(accept=("application/octet-stream",), kind=_CHECKPOINT_KIND),
                ),
            ),
            outputs=(
                OutputSpec("model", MODEL),
                OutputSpec("clip", CLIP),
                OutputSpec("vae", VAE),
            ),
            # Claiming the legacy name evicts translated
            # comfy.CheckpointLoaderSimple (merge_native_nodes): API
            # prompts naming "CheckpointLoaderSimple" reach this port,
            # with the prompt boundary converting legacy ckpt_name
            # strings to digest-backed refs against the ordered checkpoint
            # roots (make_load_checkpoint_adapter).
            aliases=("CheckpointLoaderSimple",),
            search_terms=("CheckpointLoaderSimple",),
        )

    @classmethod
    def execute(cls, *, checkpoint: object) -> Mapping[str, object]:
        assert isinstance(checkpoint, AssetRef)  # schema-typed input
        path = checkpoint.local_path()  # materialize inside the worker only
        latent_space: str | None = None
        try:
            inference = cast("Any", importlib.import_module("dinkster_inference"))
            context = current_execution_context()
            registries = cast(
                "Any",
                context.inference_registries
                if context is not None and context.inference_registries is not None
                else inference.builtin_registries(),
            )
            detection = registries.families.detect(inference.load_safetensors_header(path))
            if detection.best is not None:
                latent_space = detection.best.family_id
        except Exception:  # noqa: BLE001 - optional provenance never gates loading
            pass
        comfy_sd = cast("Any", importlib.import_module("comfy.sd"))
        folder_paths = cast("Any", importlib.import_module("folder_paths"))
        model, clip, vae, _clip_vision = comfy_sd.load_checkpoint_guess_config(
            str(path),
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        # v1 installs path-backed multigpu reload factories on all three
        # residents; a retained mount path bypasses digest verification
        # on reload, so the asset port drops them.
        _drop_path_reload_factories(model, clip, vae)
        # Name the residents for the memory panel while we still know the
        # asset: downstream only ever sees stubs, and "sd15.safetensors
        # (CLIP)" beats a Python class name that collides across
        # checkpoints sharing an architecture.
        pool = default_pool()
        pool.label(model, checkpoint.name)
        pool.label(clip, f"{checkpoint.name} (CLIP)")
        pool.label(vae, f"{checkpoint.name} (VAE)")
        pool.label_source(
            vae,
            digest=checkpoint.digest,
            name=checkpoint.name,
            latent_space=latent_space,
        )
        return cls.outputs(model=model, clip=clip, vae=vae)


class Wan21ClipVisionEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan21_clip_vision_encode",
            display_name="Wan 2.1 CLIP Vision Encode",
            category="model/conditioning/wan",
            inputs=(InputSpec("model", MODEL), InputSpec("image", IMAGE)),
            outputs=(OutputSpec("clip_vision_output", CLIP_VISION_OUTPUT),),
            search_terms=("wan", "clip vision", "image encode"),
        )

    @classmethod
    def execute(cls, *, model: object, image: object) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan21_clip_vision_encode requires the native execution arm")


class BerniniConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.bernini_conditioning",
            display_name="Bernini Conditioning",
            category="model/conditioning/bernini",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=8192, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=8192, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=8192, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec("source_video", IMAGE, required=False, default=None),
                InputSpec("reference_video", IMAGE, required=False, default=None),
                InputSpec(
                    "ref_max_size",
                    INT,
                    required=False,
                    default=848,
                    widget=NumberWidget(min=16, max=8192, step=16),
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            input_families=(
                InputFamilySpec(
                    "reference_images",
                    IMAGE,
                    max_members=8,
                    required=False,
                    member_prefix="reference_image_",
                ),
            ),
            aliases=("BerniniConditioning",),
            search_terms=("wan", "bernini", "reference", "video", "in context"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        source_video: object = None,
        reference_video: object = None,
        reference_images: Mapping[str, object] | None = None,
        ref_max_size: int = 848,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.bernini_conditioning requires the native execution arm")


class Wan21ImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan21_image_to_video",
            display_name="WanImageToVideo",
            category="model/conditioning/wan",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("model", MODEL),
                InputSpec("start_image", IMAGE),
                InputSpec(
                    "clip_vision_output",
                    CLIP_VISION_OUTPUT,
                    required=False,
                    default=None,
                ),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanImageToVideo",),
            search_terms=("wan", "wan 2.1", "wan 2.2", "image to video", "i2v"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        model: object,
        start_image: object,
        clip_vision_output: object = None,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan21_image_to_video requires the native execution arm")


class WanCameraEmbedding(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_camera_embedding",
            display_name="WanCameraEmbedding",
            category="model/conditioning/wan/camera",
            inputs=(
                InputSpec(
                    "camera_pose",
                    COMBO,
                    default="Static",
                    widget=ComboWidget(options=WAN_CAMERA_POSES),
                ),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "speed",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.1),
                ),
                InputSpec(
                    "fx",
                    FLOAT,
                    required=False,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=1e-9),
                    advanced=True,
                ),
                InputSpec(
                    "fy",
                    FLOAT,
                    required=False,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=1e-9),
                    advanced=True,
                ),
                InputSpec(
                    "cx",
                    FLOAT,
                    required=False,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "cy",
                    FLOAT,
                    required=False,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    advanced=True,
                ),
            ),
            outputs=(
                OutputSpec("camera_embedding", WAN_CAMERA_EMBEDDING),
                OutputSpec("width", INT),
                OutputSpec("height", INT),
                OutputSpec("length", INT),
            ),
            aliases=("WanCameraEmbedding",),
            search_terms=("wan", "camera", "trajectory", "plucker"),
        )

    @classmethod
    def execute(
        cls,
        *,
        camera_pose: str,
        width: int,
        height: int,
        length: int,
        speed: float = 1.0,
        fx: float = 0.5,
        fy: float = 0.5,
        cx: float = 0.5,
        cy: float = 0.5,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_camera_embedding requires the native execution arm")


class WanCameraImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_camera_image_to_video",
            display_name="WanCameraImageToVideo",
            category="model/conditioning/wan/camera",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec(
                    "clip_vision_output",
                    CLIP_VISION_OUTPUT,
                    required=False,
                    default=None,
                ),
                InputSpec("start_image", IMAGE, required=False, default=None),
                InputSpec(
                    "camera_conditions",
                    WAN_CAMERA_EMBEDDING,
                    required=False,
                    default=None,
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanCameraImageToVideo",),
            search_terms=("wan", "camera", "image to video", "trajectory"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_output: object = None,
        start_image: object = None,
        camera_conditions: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_camera_image_to_video requires the native execution arm")


class WanPhantomSubjectToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_phantom_subject_to_video",
            display_name="WanPhantomSubjectToVideo",
            category="model/conditioning/wan/phantom subject",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec("images", IMAGE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative_text", CONDITIONING),
                OutputSpec("negative_img_text", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanPhantomSubjectToVideo",),
            search_terms=("wan", "phantom", "subject", "reference", "video"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        images: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError(
            "dinkster.wan_phantom_subject_to_video requires the native execution arm"
        )


class WanTrackToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_track_to_video",
            display_name="WanTrackToVideo",
            category="model/conditioning/wan/move",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec("tracks", STRING, default="[]", widget=StringWidget(multiline=True)),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec(
                    "temperature",
                    FLOAT,
                    default=220.0,
                    widget=NumberWidget(min=1.0, max=1000.0, step=0.1),
                    advanced=True,
                ),
                InputSpec(
                    "topk",
                    INT,
                    default=2,
                    widget=NumberWidget(min=1, max=10, step=1),
                    advanced=True,
                ),
                InputSpec("start_image", IMAGE),
                InputSpec(
                    "clip_vision_output",
                    CLIP_VISION_OUTPUT,
                    required=False,
                    default=None,
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanTrackToVideo",),
            search_terms=("wan", "ati", "motion tracking", "trajectory", "point tracking"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        tracks: str,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        temperature: float,
        topk: int,
        start_image: object,
        clip_vision_output: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_track_to_video requires the native execution arm")


class WanMoveTracksFromCoords(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_move_tracks_from_coords",
            display_name="WanMoveTracksFromCoords",
            category="model/conditioning/wan/move",
            inputs=(
                InputSpec(
                    "track_coords",
                    STRING,
                    required=False,
                    default="[]",
                    force_input=True,
                ),
                InputSpec("track_mask", MASK, required=False, default=None),
            ),
            outputs=(
                OutputSpec("tracks", TRACKS),
                OutputSpec("track_length", INT),
            ),
            aliases=("WanMoveTracksFromCoords",),
            search_terms=("wan", "move", "motion paths", "trajectory"),
        )

    @classmethod
    def execute(
        cls,
        *,
        track_coords: str = "[]",
        track_mask: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_move_tracks_from_coords requires the native execution arm")


class WanMoveConcatTrack(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_move_concat_track",
            display_name="WanMoveConcatTrack",
            category="model/conditioning/wan/move",
            inputs=(
                InputSpec("tracks_1", TRACKS),
                InputSpec("tracks_2", TRACKS, required=False, default=None),
            ),
            outputs=(OutputSpec("tracks", TRACKS),),
            aliases=("WanMoveConcatTrack",),
            search_terms=("wan", "move", "combine tracks", "trajectory"),
        )

    @classmethod
    def execute(
        cls,
        *,
        tracks_1: object,
        tracks_2: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_move_concat_track requires the native execution arm")


class WanMoveGenerateTracks(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_move_generate_tracks",
            display_name="Generate Video Tracks",
            category="model/conditioning/wan/move",
            inputs=(
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=4096, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=4096, step=16)
                ),
                InputSpec(
                    "start_x", FLOAT, default=0.0, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
                InputSpec(
                    "start_y", FLOAT, default=0.0, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
                InputSpec(
                    "end_x", FLOAT, default=1.0, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
                InputSpec(
                    "end_y", FLOAT, default=1.0, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
                InputSpec("num_frames", INT, default=81, widget=NumberWidget(min=1, max=1024)),
                InputSpec("num_tracks", INT, default=5, widget=NumberWidget(min=1, max=100)),
                InputSpec(
                    "track_spread",
                    FLOAT,
                    default=0.025,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("bezier", BOOLEAN, default=False),
                InputSpec(
                    "mid_x", FLOAT, default=0.5, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
                InputSpec(
                    "mid_y", FLOAT, default=0.5, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
                InputSpec(
                    "interpolation",
                    COMBO,
                    default="linear",
                    widget=ComboWidget(
                        options=("linear", "ease_in", "ease_out", "ease_in_out", "constant")
                    ),
                ),
                InputSpec("track_mask", MASK, required=False, default=None),
            ),
            outputs=(
                OutputSpec("tracks", TRACKS),
                OutputSpec("track_length", INT),
            ),
            aliases=("GenerateTracks",),
            search_terms=("motion paths", "camera movement", "trajectory"),
        )

    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        num_frames: int,
        num_tracks: int,
        track_spread: float,
        bezier: bool = False,
        mid_x: float = 0.5,
        mid_y: float = 0.5,
        interpolation: str = "linear",
        track_mask: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_move_generate_tracks requires the native execution arm")


class WanMoveVisualizeTracks(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_move_visualize_tracks",
            display_name="WanMoveVisualizeTracks",
            category="model/conditioning/wan/move",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec("tracks", TRACKS, required=False, default=None),
                InputSpec(
                    "line_resolution",
                    INT,
                    default=24,
                    widget=NumberWidget(min=1, max=1024),
                ),
                InputSpec(
                    "circle_size",
                    INT,
                    default=12,
                    widget=NumberWidget(min=1, max=128),
                ),
                InputSpec(
                    "opacity",
                    FLOAT,
                    default=0.75,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "line_width",
                    INT,
                    default=16,
                    widget=NumberWidget(min=1, max=128),
                ),
            ),
            outputs=(OutputSpec("images", IMAGE),),
            aliases=("WanMoveVisualizeTracks",),
            search_terms=("wan", "move", "track preview", "trajectory"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        line_resolution: int,
        circle_size: int,
        opacity: float,
        line_width: int,
        tracks: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_move_visualize_tracks requires the native execution arm")


class WanMoveTrackToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_move_track_to_video",
            display_name="WanMoveTrackToVideo",
            category="model/conditioning/wan/move",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec("tracks", TRACKS, required=False, default=None),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec("start_image", IMAGE),
                InputSpec(
                    "clip_vision_output",
                    CLIP_VISION_OUTPUT,
                    required=False,
                    default=None,
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanMoveTrackToVideo",),
            search_terms=("wan", "move", "motion paths", "trajectory"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        strength: float,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object,
        tracks: object = None,
        clip_vision_output: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_move_track_to_video requires the native execution arm")


class WanFirstLastFrameToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_first_last_frame_to_video",
            display_name="WanFirstLastFrameToVideo",
            category="model/conditioning/wan",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec(
                    "clip_vision_start_image",
                    CLIP_VISION_OUTPUT,
                    required=False,
                    default=None,
                ),
                InputSpec(
                    "clip_vision_end_image",
                    CLIP_VISION_OUTPUT,
                    required=False,
                    default=None,
                ),
                InputSpec("start_image", IMAGE, required=False, default=None),
                InputSpec("end_image", IMAGE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanFirstLastFrameToVideo",),
            search_terms=("wan", "first frame", "last frame", "flf", "video"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_start_image: object = None,
        clip_vision_end_image: object = None,
        start_image: object = None,
        end_image: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError(
            "dinkster.wan_first_last_frame_to_video requires the native execution arm"
        )


class WanFunControlToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_fun_control_to_video",
            display_name="WanFunControlToVideo",
            category="model/conditioning/wan/fun control",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec(
                    "clip_vision_output",
                    CLIP_VISION_OUTPUT,
                    required=False,
                    default=None,
                ),
                InputSpec("start_image", IMAGE, required=False, default=None),
                InputSpec("control_video", IMAGE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanFunControlToVideo",),
            search_terms=("wan", "fun", "control", "video"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_output: object = None,
        start_image: object = None,
        control_video: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_fun_control_to_video requires the native execution arm")


class Wan22FunControlToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = WanFunControlToVideo.define_schema()
        return NodeSchema(
            node_type="dinkster.wan22_fun_control_to_video",
            display_name="Wan22FunControlToVideo",
            category=schema.category,
            inputs=(
                *schema.inputs[:7],
                InputSpec("ref_image", IMAGE, required=False, default=None),
                InputSpec("control_video", IMAGE, required=False, default=None),
            ),
            outputs=schema.outputs,
            aliases=("Wan22FunControlToVideo",),
            search_terms=("wan", "wan 2.2", "fun", "control", "video"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        ref_image: object = None,
        start_image: object = None,
        control_video: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan22_fun_control_to_video requires the native execution arm")


class WanFunInpaintToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = WanFirstLastFrameToVideo.define_schema()
        return NodeSchema(
            node_type="dinkster.wan_fun_inpaint_to_video",
            display_name="WanFunInpaintToVideo",
            category="model/conditioning/wan/fun inpaint",
            inputs=(
                *schema.inputs[:7],
                InputSpec(
                    "clip_vision_output",
                    CLIP_VISION_OUTPUT,
                    required=False,
                    default=None,
                ),
                InputSpec("start_image", IMAGE, required=False, default=None),
                InputSpec("end_image", IMAGE, required=False, default=None),
            ),
            outputs=schema.outputs,
            aliases=("WanFunInpaintToVideo",),
            search_terms=("wan", "fun", "inpaint", "first frame", "last frame"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_output: object = None,
        start_image: object = None,
        end_image: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_fun_inpaint_to_video requires the native execution arm")


class WanVaceToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_vace_to_video",
            display_name="WanVaceToVideo",
            category="model/conditioning/wan/vace",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1000.0, step=0.01),
                ),
                InputSpec("control_video", IMAGE, required=False, default=None),
                InputSpec("control_masks", MASK, required=False, default=None),
                InputSpec("reference_image", IMAGE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
                OutputSpec("trim_latent", INT),
            ),
            aliases=("WanVaceToVideo",),
            search_terms=("wan", "vace", "video conditioning", "video control"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        strength: float,
        control_video: object = None,
        control_masks: object = None,
        reference_image: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.wan_vace_to_video requires the native execution arm")


class TrimVideoLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.trim_video_latent",
            display_name="Trim Video Latent",
            category="model/latent",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec(
                    "trim_amount",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=99999),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("TrimVideoLatent",),
            search_terms=("trim video latent", "wan", "video latent"),
        )

    @classmethod
    def execute(
        cls,
        *,
        samples: object,
        trim_amount: int,
    ) -> Mapping[str, object]:
        source = cast("Any", samples)
        output = source.copy()
        output["samples"] = source["samples"][:, :, trim_amount:]
        return cls.outputs(latent=output)


class Wan22ImageToVideoLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan22_image_to_video_latent",
            display_name="Wan22ImageToVideoLatent",
            category="model/conditioning/wan",
            inputs=(
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=1280, widget=NumberWidget(min=32, max=16384, step=32)
                ),
                InputSpec(
                    "height", INT, default=704, widget=NumberWidget(min=32, max=16384, step=32)
                ),
                InputSpec("length", INT, default=49, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec("start_image", IMAGE, required=False, default=None),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("Wan22ImageToVideoLatent",),
            search_terms=("wan", "wan 2.2", "image to video", "ti2v"),
        )

    @classmethod
    def execute(
        cls,
        *,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object = None,
    ) -> Mapping[str, object]:
        if width < 32 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 32 and 16384")
        if height < 32 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 32 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        torch = cast("Any", importlib.import_module("torch"))
        latent = torch.zeros((1, 48, ((length - 1) // 4) + 1, height // 16, width // 16))
        if start_image is None:
            return cls.outputs(latent={"samples": latent})
        if not hasattr(vae, "encode"):
            raise TypeError("vae must provide encode")
        loaded_vae = cast("Any", vae)
        image = cast("Any", start_image)
        resized = (
            importlib.import_module("dinkster_inference_torch.resize")
            .common_upscale(image[:length].movedim(-1, 1), width, height, "bilinear", "center")
            .movedim(1, -1)
        )
        reference = loaded_vae.encode(resized)
        mask = torch.ones((1, 1, *latent.shape[2:]), dtype=reference.dtype)
        latent = latent.to(dtype=reference.dtype)
        latent[:, :, : reference.shape[2]] = reference
        mask[:, :, : reference.shape[2]] = 0.0
        latent_format = cast("Any", importlib.import_module("comfy.latent_formats")).Wan22()
        latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)
        return cls.outputs(
            latent={
                "samples": latent.repeat(batch_size, 1, 1, 1, 1),
                "noise_mask": mask.repeat(batch_size, 1, 1, 1, 1),
            }
        )


class LoadZImageControlPatch(Node):
    """Load one supported standalone model patch."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_z_image_control_patch",
            display_name="Load Model Patch",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model_patch",
                    ASSET,
                    widget=AssetWidget(accept=("application/octet-stream",), kind="model/patch"),
                ),
            ),
            outputs=(OutputSpec("model_patch", MODEL_PATCH),),
            aliases=("ModelPatchLoader",),
            search_terms=("z-image", "fun controlnet", "wan", "infinite talk", "model patch"),
        )

    @classmethod
    def execute(cls, *, model_patch: object) -> Mapping[str, object]:
        raise RuntimeError("dinkster.load_z_image_control_patch requires the native execution arm")


class ApplyZImageControlPatch(Node):
    """Attach Z-Image Fun ControlNet conditioning to a native model."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_z_image_control_patch",
            display_name="Apply Z-Image Fun ControlNet",
            category="model/patch/z-image",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("model_patch", MODEL_PATCH),
                InputSpec("vae", VAE),
                InputSpec("image", IMAGE),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            aliases=("ZImageFunControlnet",),
            search_terms=("z-image", "fun controlnet", "control image"),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        model_patch: object,
        vae: object,
        image: object,
        strength: float,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.apply_z_image_control_patch requires the native execution arm")


def _drop_path_reload_factories(*objects: object) -> None:
    """Remove v1's path-backed multigpu reload factories from loaded models.

    The pinned loaders (load_clip, load_diffusion_model,
    load_checkpoint_guess_config @ b78cec87) install
    ``cached_patcher_init`` factories that retain the load path so
    deepclone_multigpu can re-read weights per device. A mount path
    retained beyond this invocation bypasses digest verification on
    reload - refused under the AssetRef.local_path contract, so the
    asset ports drop the factories. Asset-safe replacements are
    ledgered in ROADMAP ("Asset-safe model reload factories")."""
    for obj in objects:
        patcher = cast("Any", getattr(obj, "patcher", obj))
        if getattr(patcher, "cached_patcher_init", None) is not None:
            patcher.cached_patcher_init = None


_LORA_SD_CACHE_BUDGET = 4 * 1024**3
"""v1's LoraLoader keeps ONE parsed state dict per live node instance,
for the node's whole lifetime - a chain of N loaders keeps N parsed
loras. Dinkster nodes are stateless, so the worker keeps a digest-keyed
LRU instead, bounded by BYTES (the asset's size, ~ the parsed tensor
payload) rather than entry count: ordinary chains cache fully whatever
their length, and eviction only fires as a genuine memory tradeoff.
Eviction just re-reads the file - correctness never depends on a hit."""

_lora_sd_cache: OrderedDict[str, tuple[int, Any, Any]] = OrderedDict()
_lora_sd_cache_bytes = 0
_lora_sd_cache_lock = threading.Lock()


def _load_lora_file(lora: AssetRef) -> tuple[Any, Any]:
    """Parsed (state dict, metadata) for a LoRA asset, cached by digest.

    Upstream parity floor: v1 reuses the parsed lora across executions
    of the same node (strength tweaks never re-read the file). The
    engine's output cache cannot provide that - its key includes model,
    clip, and strengths - so the worker keeps this digest-keyed cache.
    Digest keys make it exact by construction: same bytes, same entry,
    any filename, any node."""
    global _lora_sd_cache_bytes
    with _lora_sd_cache_lock:
        cached = _lora_sd_cache.get(lora.digest)
        if cached is not None:
            _lora_sd_cache.move_to_end(lora.digest)
            return cached[1], cached[2]
    path = lora.local_path()
    checkpoint = cast("Any", importlib.import_module("dinkster_inference_torch.checkpoint"))
    lora_sd, lora_metadata = checkpoint.load_checkpoint_with_metadata(path)
    with _lora_sd_cache_lock:
        previous = _lora_sd_cache.pop(lora.digest, None)
        if previous is not None:
            _lora_sd_cache_bytes -= previous[0]
        _lora_sd_cache[lora.digest] = (lora.size, lora_sd, lora_metadata)
        _lora_sd_cache_bytes += lora.size
        # Keep at least the entry just parsed, even oversized: the caller
        # is about to use it, and evicting it buys nothing.
        while _lora_sd_cache_bytes > _LORA_SD_CACHE_BUDGET and len(_lora_sd_cache) > 1:
            _, (evicted_size, _sd, _meta) = _lora_sd_cache.popitem(last=False)
            _lora_sd_cache_bytes -= evicted_size
    return lora_sd, lora_metadata


def _apply_lora(
    model: object,
    clip: object,
    lora: object,
    strength_model: float,
    strength_clip: float,
) -> tuple[object, object]:
    """The shared LoRA application path for both LoRA ports: reference
    LoraLoader.load_lora semantics against a digest-named asset.

    v1's per-node-instance ``loaded_lora`` cache ports as the module's
    digest-keyed ``_load_lora_file`` cache (Dinkster nodes are stateless).
    The v1 FLOAT range contract ([-100, 100]) survives as execute-side
    validation (the EmptyLatentImage precedent)."""
    assert isinstance(lora, AssetRef)  # schema-typed input
    for name, value in (
        ("strength_model", strength_model),
        ("strength_clip", strength_clip),
    ):
        if not -100.0 <= value <= 100.0:
            raise ValueError(f"{name} must be in [-100.0, 100.0], got {value}")
    if strength_model == 0 and strength_clip == 0:
        return model, clip  # reference parity: no load, no clone
    lora_sd, lora_metadata = _load_lora_file(lora)
    comfy_sd = cast("Any", importlib.import_module("comfy.sd"))
    # The lora_metadata kwarg is recent upstream API; older installs'
    # load_lora_for_models does not take it; the compat pack runs inside
    # whatever ComfyUI the user points it at.
    kwargs: dict[str, Any] = {}
    if "lora_metadata" in inspect.signature(comfy_sd.load_lora_for_models).parameters:
        kwargs["lora_metadata"] = lora_metadata
    model_lora, clip_lora = comfy_sd.load_lora_for_models(
        model, clip, lora_sd, strength_model, strength_clip, **kwargs
    )
    return model_lora, clip_lora


def validate_lora_execution_mode(execution_mode: str) -> None:
    if execution_mode not in LORA_EXECUTION_MODES:
        raise ValueError(f"unknown LoRA execution mode {execution_mode!r}")


class LoadLora(Node):
    """Apply a LoRA named by content digest to a model and CLIP.

    The canonical Dinkster port of ComfyUI's LoraLoader; claiming the
    legacy name as an alias evicts the translated filename/combo node
    (merge_native_nodes). The ``lora`` input is a ``dinkster.asset``:
    legacy prompts' ``lora_name`` strings convert to digest-backed refs
    at the compat boundary (make_load_lora_adapter)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_lora",
            display_name="Load LoRA (Model and CLIP)",
            category="model/loaders",
            description=(
                "Modifies both the diffusion and CLIP models with a LoRA, "
                "altering the way in which latents are denoised such as "
                "applying styles. Multiple LoRA nodes can be linked "
                "together. Identity is the file's content digest: the same "
                "bytes hit the same cache entries on any machine, under "
                "any filename."
            ),
            inputs=(
                InputSpec(
                    "model",
                    MODEL,
                    doc="The diffusion model the LoRA will be applied to.",
                ),
                InputSpec(
                    "clip",
                    CLIP,
                    doc="The CLIP model the LoRA will be applied to.",
                ),
                InputSpec(
                    "lora",
                    ASSET,
                    doc="The LoRA file to apply.",
                    widget=AssetWidget(accept=("application/octet-stream",), kind=KIND_MODEL_LORA),
                ),
                InputSpec(
                    "strength_model",
                    FLOAT,
                    required=False,
                    default=1.0,
                    doc=("How strongly to modify the diffusion model. This value can be negative."),
                    widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "strength_clip",
                    FLOAT,
                    required=False,
                    default=1.0,
                    doc=("How strongly to modify the CLIP model. This value can be negative."),
                    widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "execution_mode",
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    default="auto",
                    doc=(
                        "Native execution strategy. Auto attaches diffusion-only LoRAs and "
                        "precalculates LoRAs that patch the text encoder. Attach avoids a "
                        "model rebuild but adds work to every sampling step."
                    ),
                    widget=ComboWidget(options=LORA_EXECUTION_MODES),
                ),
            ),
            outputs=(
                OutputSpec("model", MODEL, doc="The modified diffusion model."),
                OutputSpec("clip", CLIP, doc="The modified CLIP model."),
            ),
            # Legacy API prompts naming "LoraLoader" reach this port, with
            # the prompt boundary converting legacy lora_name strings to
            # digest-backed refs against the ordered LoRA roots
            # (make_load_lora_adapter).
            aliases=("LoraLoader",),
            # The legacy class name plus ComfyUI's SEARCH_ALIASES.
            search_terms=(
                "LoraLoader",
                "lora",
                "load lora",
                "apply lora",
                "lora loader",
                "lora model",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        clip: object,
        lora: object,
        strength_model: float,
        strength_clip: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        validate_lora_execution_mode(execution_mode)
        if execution_mode == "attach":
            raise ValueError("LoRA attach mode requires native model execution")
        model_lora, clip_lora = _apply_lora(model, clip, lora, strength_model, strength_clip)
        return cls.outputs(model=model_lora, clip=clip_lora)


class LoadLoraModelOnly(Node):
    """Apply a LoRA named by content digest to the model only (no CLIP).

    The canonical Dinkster port of ComfyUI's LoraLoaderModelOnly: the same
    application path as dinkster.load_lora with no CLIP flowing through
    (reference semantics: clip=None, strength_clip=0)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_lora_model_only",
            display_name="Load LoRA",
            category="model/loaders",
            description=(
                "Modifies the diffusion model with a LoRA, altering the "
                "way in which latents are denoised such as applying "
                "styles. Multiple LoRA nodes can be linked together."
            ),
            inputs=(
                InputSpec(
                    "model",
                    MODEL,
                    doc="The diffusion model the LoRA will be applied to.",
                ),
                InputSpec(
                    "lora",
                    ASSET,
                    doc="The LoRA file to apply.",
                    widget=AssetWidget(accept=("application/octet-stream",), kind=KIND_MODEL_LORA),
                ),
                InputSpec(
                    "strength_model",
                    FLOAT,
                    required=False,
                    default=1.0,
                    doc=("How strongly to modify the diffusion model. This value can be negative."),
                    widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "execution_mode",
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    default="auto",
                    doc=(
                        "Native execution strategy. Attach avoids a model rebuild but adds "
                        "work to every sampling step."
                    ),
                    widget=ComboWidget(options=LORA_EXECUTION_MODES),
                ),
            ),
            outputs=(OutputSpec("model", MODEL, doc="The modified diffusion model."),),
            # Same legacy input id as LoraLoader, so the same prompt
            # adapter serves both ports.
            aliases=("LoraLoaderModelOnly",),
            search_terms=("LoraLoaderModelOnly", "lora", "load lora"),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        lora: object,
        strength_model: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        validate_lora_execution_mode(execution_mode)
        if execution_mode == "attach":
            raise ValueError("LoRA attach mode requires native model execution")
        model_lora, _ = _apply_lora(model, None, lora, strength_model, 0.0)
        return cls.outputs(model=model_lora)


class LoadVae(Node):
    """Load a digest-backed VAE or the synthetic pixel-space codec."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_vae",
            display_name="Load VAE",
            category="model/loaders",
            description=(
                "Loads a VAE from a file asset, or creates the pixel-space "
                "codec used by RGB latent models."
            ),
            inputs=(
                InputSpec(
                    "vae",
                    ASSET,
                    required=False,
                    doc="The VAE file to load.",
                    widget=AssetWidget(accept=("application/octet-stream",), kind=KIND_MODEL_VAE),
                ),
                InputSpec("pixel_space", BOOLEAN, required=False, default=False, advanced=True),
            ),
            outputs=(OutputSpec("vae", DINKSTER_VAE, doc="The loaded VAE model."),),
            aliases=("VAELoader",),
            search_terms=("VAELoader", "vae", "load vae"),
        )

    @classmethod
    def execute(
        cls, *, vae: object | None = None, pixel_space: bool = False
    ) -> Mapping[str, object]:
        if pixel_space:
            raise RuntimeError("pixel-space codec loading requires native execution")
        if not isinstance(vae, AssetRef):
            raise TypeError("vae must be an AssetRef")
        path = vae.local_path()
        comfy_sd = cast("Any", importlib.import_module("comfy.sd"))
        checkpoint = cast("Any", importlib.import_module("dinkster_inference_torch.checkpoint"))
        sd, metadata = checkpoint.load_checkpoint_with_metadata(path)
        loaded = comfy_sd.VAE(sd=sd, metadata=metadata)
        loaded.throw_exception_if_invalid()
        # DELIBERATE non-port: v1 registers a path-backed reload factory
        # (patcher.cached_patcher_init) for multigpu deepclones. That
        # retains a mount path beyond this invocation, bypassing digest
        # verification on reload - refused under the AssetRef.local_path
        # contract. Asset-safe reload factories are ledgered in ROADMAP
        # ("Asset-safe model reload factories").
        default_pool().label(loaded, vae.name)
        default_pool().label_source(loaded, digest=vae.digest, name=vae.name)
        return cls.outputs(vae=loaded)


class LoadVision(Node):
    """Load one digest-backed vision encoder."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_vision",
            display_name="Load Vision Encoder",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "vision_encoder",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/clip-vision",
                    ),
                ),
            ),
            outputs=(OutputSpec("vision", DINKSTER_CLIP_VISION),),
            aliases=("CLIPVisionLoader",),
            search_terms=("CLIPVisionLoader", "clip vision", "vision encoder"),
        )

    @classmethod
    def execute(cls, *, vision_encoder: object) -> Mapping[str, object]:
        if not isinstance(vision_encoder, AssetRef):
            raise TypeError("vision_encoder must be an AssetRef")
        clip_vision = cast("Any", importlib.import_module("comfy.clip_vision"))
        loaded = clip_vision.load(str(vision_encoder.local_path()))
        default_pool().label(loaded, vision_encoder.name)
        default_pool().label_source(
            loaded,
            digest=vision_encoder.digest,
            name=vision_encoder.name,
        )
        return cls.outputs(vision=loaded)


class LoadDiffusionModel(Node):
    """Load a diffusion model (UNET/DiT) named by content digest.

    The canonical Dinkster port of ComfyUI's UNETLoader; claiming the
    legacy name as an alias evicts the translated filename/combo node
    (merge_native_nodes). ``weight_dtype`` keeps the v1 vocabulary as a
    static COMBO (a closed set the port itself maps to torch dtypes -
    unlike sampler lists it cannot drift, because unrecognized values
    mean 'default' exactly as v1 behaves)."""

    WEIGHT_DTYPES: tuple[str, ...] = (
        "default",
        "fp8_e4m3fn",
        "fp8_e4m3fn_fast",
        "fp8_e5m2",
    )

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_diffusion_model",
            display_name="Load Diffusion Model",
            category="model/loaders",
            description=(
                "Loads a diffusion model (UNET or DiT) from a file asset. "
                "Identity is the file's content digest: the same bytes hit "
                "the same cache entries on any machine, under any filename."
            ),
            inputs=(
                InputSpec(
                    "diffusion_model",
                    ASSET,
                    doc="The diffusion model file to load.",
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind=KIND_MODEL_DIFFUSION,
                    ),
                ),
                InputSpec(
                    "weight_dtype",
                    COMBO,
                    required=False,
                    default="default",
                    doc=(
                        "The dtype to store the model weights in; fp8 "
                        "variants trade quality for memory."
                    ),
                    widget=ComboWidget(options=cls.WEIGHT_DTYPES),
                ),
            ),
            outputs=(OutputSpec("model", MODEL, doc="The loaded diffusion model."),),
            # Legacy API prompts naming "UNETLoader" reach this port, with
            # the prompt boundary converting legacy unet_name strings to
            # digest-backed refs (make_load_diffusion_model_adapter);
            # weight_dtype rides through unchanged.
            aliases=("UNETLoader",),
            search_terms=(
                "UNETLoader",
                "unet",
                "load unet",
                "diffusion model",
                "load diffusion model",
            ),
        )

    @classmethod
    def execute(cls, *, diffusion_model: object, weight_dtype: str) -> Mapping[str, object]:
        assert isinstance(diffusion_model, AssetRef)  # schema-typed input
        path = diffusion_model.local_path()
        torch = cast("Any", importlib.import_module("torch"))
        comfy_sd = cast("Any", importlib.import_module("comfy.sd"))
        # Reference parity: only the three fp8 spellings set options; any
        # other value (including unknown strings) means default. Choices
        # are UI vocabulary, never execution identity.
        model_options: dict[str, Any] = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        model = comfy_sd.load_diffusion_model(str(path), model_options=model_options)
        # v1's load_diffusion_model installs a path-backed multigpu reload
        # factory; dropped for the same reason as LoadCheckpoint's.
        _drop_path_reload_factories(model)
        default_pool().label(model, diffusion_model.name)
        return cls.outputs(model=model)


class LoadClip(Node):
    """Load a CLIP/text-encoder model named by content digest.

    The canonical Dinkster port of ComfyUI's CLIPLoader (the single-file
    loader; DualCLIPLoader's two-file recipes are not this port);
    claiming the legacy name as an alias evicts the translated
    filename/combo node (merge_native_nodes). ``type`` keeps the v1
    vocabulary as a static COMBO with v1's exact fallback semantics: an
    unrecognized value means STABLE_DIFFUSION (getattr default), so
    choices stay UI vocabulary, never execution identity."""

    # The reference CLIPLoader "type" list at 947c2749, plus native family options.
    CLIP_TYPES: tuple[str, ...] = (
        "stable_diffusion",
        "stable_cascade",
        "sd3",
        "stable_audio",
        "mochi",
        "ltxv",
        "pixart",
        "cosmos",
        "lumina2",
        "wan",
        "hidream",
        "chroma",
        "ace",
        "omnigen2",
        "qwen_image",
        "hunyuan_image",
        "flux2",
        "ovis",
        "longcat_image",
        "cogvideox",
        "lens",
        "pixeldit",
        "ideogram4",
        "boogu",
        "krea2",
        "anima",
        "joyimage",
        "minimax",
    )

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_clip",
            display_name="Load CLIP",
            category="model/loaders",
            description=(
                "Loads a CLIP or other text-encoder model from a file "
                "asset. Identity is the file's content digest: the same "
                "bytes hit the same cache entries on any machine, under "
                "any filename.\n"
                "Recipes:\n"
                "sd: clip-l\n"
                "stable cascade: clip-g\n"
                "sd3: t5 xxl / clip-g / clip-l\n"
                "stable audio: t5 base\n"
                "mochi: t5 xxl\n"
                "cogvideox: t5 xxl (226-token padding)\n"
                "cosmos: old t5 xxl\n"
                "lumina2: gemma 2 2B\n"
                "wan: umt5 xxl\n"
                "hidream: llama-3.1 (Recommend) or t5\n"
                "omnigen2: qwen vl 2.5 3B\n"
                "joyimage: qwen3-vl 8B\n"
                "lens: gpt-oss-20b\n"
                "pixeldit: gemma 2 2B elm"
            ),
            inputs=(
                InputSpec(
                    "text_encoder",
                    ASSET,
                    doc="The text encoder model file to load.",
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind=KIND_MODEL_TEXT_ENCODER,
                    ),
                ),
                InputSpec(
                    "type",
                    COMBO,
                    required=False,
                    default="stable_diffusion",
                    doc=(
                        "The model family the text encoder drives; picks "
                        "the tokenizer/embedding recipe."
                    ),
                    widget=ComboWidget(options=cls.CLIP_TYPES),
                ),
                InputSpec(
                    "device",
                    COMBO,
                    required=False,
                    default="default",
                    doc=(
                        "Force the text encoder onto the CPU instead of "
                        "the management-chosen device."
                    ),
                    widget=ComboWidget(options=("default", "cpu")),
                ),
            ),
            outputs=(OutputSpec("clip", DINKSTER_CLIP, doc="The loaded text encoder."),),
            # Legacy API prompts naming "CLIPLoader" reach this port, with
            # the prompt boundary converting legacy clip_name strings to
            # digest-backed refs (make_load_clip_adapter); type and
            # device ride through unchanged.
            aliases=("CLIPLoader",),
            search_terms=(
                "CLIPLoader",
                "clip",
                "load clip",
                "text encoder",
                "load text encoder",
            ),
        )

    @classmethod
    def execute(cls, *, text_encoder: object, type: str, device: str) -> Mapping[str, object]:
        assert isinstance(text_encoder, AssetRef)  # schema-typed input
        path = text_encoder.local_path()
        torch = cast("Any", importlib.import_module("torch"))
        comfy_sd = cast("Any", importlib.import_module("comfy.sd"))
        folder_paths = cast("Any", importlib.import_module("folder_paths"))
        # Reference parity: unknown type strings fall back to
        # STABLE_DIFFUSION exactly as v1's getattr default does.
        clip_type = getattr(comfy_sd.CLIPType, type.upper(), comfy_sd.CLIPType.STABLE_DIFFUSION)
        model_options: dict[str, Any] = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        clip = comfy_sd.load_clip(
            ckpt_paths=[str(path)],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        # v1's load_clip installs a path-backed multigpu reload factory on
        # clip.patcher; dropped for the same reason as LoadCheckpoint's.
        _drop_path_reload_factories(clip)
        default_pool().label(clip, text_encoder.name)
        return cls.outputs(clip=clip)


class LoadDualClip(Node):
    """Load an ordered pair of text encoders as one recipe."""

    CLIP_TYPES: tuple[str, ...] = (
        "sdxl",
        "sd3",
        "flux",
        "hunyuan_video",
        "hidream",
        "hunyuan_image",
        "hunyuan_video_15",
        "kandinsky5",
        "kandinsky5_image",
        "ltxv",
        "newbie",
        "ace",
    )

    @classmethod
    def define_schema(cls) -> NodeSchema:
        asset_widget = AssetWidget(
            accept=("application/octet-stream",),
            kind=KIND_MODEL_TEXT_ENCODER,
        )
        return NodeSchema(
            node_type="dinkster.load_dual_clip",
            display_name="Load Dual CLIP",
            category="model/loaders",
            description=(
                "Loads two ordered CLIP or other text-encoder model files as one recipe.\n"
                "Recipes:\n"
                "sdxl: clip-l, clip-g\n"
                "sd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\n"
                "flux: clip-l, t5\n"
                "hidream: at least one of t5 or llama, recommended t5 and llama\n"
                "hunyuan_image: qwen2.5vl 7b and byt5 small\n"
                "newbie: gemma-3-4b-it, jina clip v2"
            ),
            inputs=(
                InputSpec(
                    "text_encoder1",
                    ASSET,
                    doc="The first text encoder model file to load.",
                    widget=asset_widget,
                ),
                InputSpec(
                    "text_encoder2",
                    ASSET,
                    doc="The second text encoder model file to load.",
                    widget=asset_widget,
                ),
                InputSpec(
                    "type",
                    COMBO,
                    required=False,
                    default="sdxl",
                    doc="The text encoding recipe to assemble.",
                    widget=ComboWidget(options=cls.CLIP_TYPES),
                ),
                InputSpec(
                    "device",
                    COMBO,
                    required=False,
                    default="default",
                    doc=(
                        "Force both text encoders onto the CPU instead of "
                        "the management-chosen device."
                    ),
                    widget=ComboWidget(options=("default", "cpu")),
                ),
            ),
            outputs=(OutputSpec("clip", CLIP, doc="The loaded text encoding recipe."),),
            aliases=("DualCLIPLoader",),
            search_terms=(
                "DualCLIPLoader",
                "dual clip",
                "load dual clip",
                "text encoder",
                "load text encoder",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        text_encoder1: object,
        text_encoder2: object,
        type: str,
        device: str,
    ) -> Mapping[str, object]:
        assert isinstance(text_encoder1, AssetRef)  # schema-typed input
        assert isinstance(text_encoder2, AssetRef)  # schema-typed input
        if type not in cls.CLIP_TYPES:
            raise ValueError(f"unknown dual text encoding recipe {type!r}")
        paths = (text_encoder1.local_path(), text_encoder2.local_path())
        torch = cast("Any", importlib.import_module("torch"))
        comfy_sd = cast("Any", importlib.import_module("comfy.sd"))
        folder_paths = cast("Any", importlib.import_module("folder_paths"))
        clip_type = getattr(comfy_sd.CLIPType, type.upper())
        model_options: dict[str, Any] = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        clip = comfy_sd.load_clip(
            ckpt_paths=[str(path) for path in paths],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        _drop_path_reload_factories(clip)
        default_pool().label(clip, f"{text_encoder1.name} + {text_encoder2.name}")
        return cls.outputs(clip=clip)


class EmptyLatentImage(Node):
    """Comfy-shaped implementation used by the native generation provider."""

    MIN_DIMENSION = 16
    MAX_DIMENSION = 16384  # ComfyUI's MAX_RESOLUTION
    MAX_BATCH = 4096

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_latent_image",
            display_name="Empty Latent Image",
            category="model/latent",
            description=("Create a new batch of empty latent images to be denoised via sampling."),
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=512,
                    doc="The width of the latent images in pixels.",
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=512,
                    doc="The height of the latent images in pixels.",
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    required=False,
                    default=1,
                    doc="The number of latent images in the batch.",
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            # Standalone users retain the legacy lookup vocabulary.
            aliases=("EmptyLatentImage",),
            # The legacy class name plus ComfyUI's search vocabulary.
            search_terms=(
                "EmptyLatentImage",
                "empty",
                "empty latent",
                "new latent",
                "create latent",
                "blank latent",
                "blank",
            ),
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int) -> Mapping[str, object]:
        for name, value, low, high in (
            ("width", width, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("height", height, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("batch_size", batch_size, 1, cls.MAX_BATCH),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        torch = cast("Any", importlib.import_module("torch"))
        device = select_intermediate_device(torch)
        samples = torch.zeros(
            [batch_size, 4, height // 8, width // 8],
            device=device,
            dtype=intermediate_dtype(torch),
        )
        return cls.outputs(latent={"samples": samples, "downscale_ratio_spacial": 8})


class EmptyHunyuanLatentVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_hunyuan_latent_video",
            display_name="Empty HunyuanVideo 1.0 Latent",
            category="model/latent/hunyuan video",
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=848,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=480,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "length",
                    INT,
                    required=False,
                    default=25,
                    widget=NumberWidget(min=1, max=16384, step=4),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("latent", DINKSTER_LATENT),),
            aliases=("EmptyHunyuanLatentVideo",),
            search_terms=("EmptyHunyuanLatentVideo",),
        )

    @classmethod
    def execute(
        cls, *, width: int, height: int, length: int, batch_size: int
    ) -> Mapping[str, object]:
        if width < 16 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 16 and 16384")
        if height < 16 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 16 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        torch = cast("Any", importlib.import_module("torch"))
        device = select_intermediate_device(torch)
        samples = torch.zeros(
            [
                batch_size,
                16,
                ((length - 1) // 4) + 1,
                height // 8,
                width // 8,
            ],
            device=device,
        )
        return cls.outputs(latent={"samples": samples, "downscale_ratio_spacial": 8})


class CLIPTextEncode(Node):
    """Encode a text prompt with a CLIP model into conditioning.

    The canonical Dinkster port of ComfyUI's CLIPTextEncode: the same
    tokenize-then-encode path against the worker-resident CLIP object.
    Claiming the legacy name as an alias evicts the translated node
    (merge_native_nodes). ComfyUI's dynamicPrompts flag is frontend
    vocabulary preserved in schema/signature identity; the frontend expands
    it before submitting, and backend execution receives only that literal."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.clip_text_encode",
            display_name="CLIP Text Encode",
            category="model/conditioning",
            description=(
                "Encodes a text prompt using a CLIP model into an embedding "
                "that can be used to guide the diffusion model towards "
                "generating specific images."
            ),
            inputs=(
                InputSpec(
                    "text",
                    STRING,
                    doc="The text to be encoded.",
                    widget=WidgetRepresentations(
                        representations=(
                            WidgetRepresentation(
                                "single-line",
                                StringWidget(multiline=False, dynamic_prompts=True),
                                display_name="Single line",
                            ),
                            WidgetRepresentation(
                                "multiline",
                                StringWidget(multiline=True, dynamic_prompts=True),
                                display_name="Multiline",
                            ),
                        ),
                        default="multiline",
                        user_switchable=True,
                    ),
                ),
                InputSpec(
                    "clip",
                    CLIP,
                    doc="The CLIP model used for encoding the text.",
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            # Legacy API prompts naming "CLIPTextEncode" resolve here; the
            # inputs are a plain string and a linked CLIP, so no
            # prompt-boundary adapter is needed.
            aliases=("CLIPTextEncode",),
            # The legacy class name plus ComfyUI's search vocabulary.
            search_terms=(
                "CLIPTextEncode",
                "text",
                "prompt",
                "text prompt",
                "positive prompt",
                "negative prompt",
                "encode text",
                "text encoder",
                "encode prompt",
            ),
        )

    @classmethod
    def execute(cls, *, text: str, clip: object) -> Mapping[str, object]:
        if clip is None:
            # A checkpoint without a text encoder loads with clip=None;
            # name the real cause instead of an AttributeError on None.
            raise ValueError(
                "clip input is invalid: the checkpoint it came from does "
                "not contain a valid CLIP or text encoder model"
            )
        clip_model = cast("Any", clip)
        tokens = clip_model.tokenize(text)
        return cls.outputs(conditioning=clip_model.encode_from_tokens_scheduled(tokens))


class MiniMaxMusic3TextEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.minimax_music3_text_encode",
            display_name="MiniMax Music3 Text Encode",
            category="model/conditioning/minimax music",
            description=(
                "Uses a MiniMax Music3 CLIP model to generate the acoustic conditioning sequence."
            ),
            inputs=(
                InputSpec("clip", DINKSTER_CLIP),
                InputSpec(
                    "caption",
                    STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "lyrics",
                    STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "seed",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        step=1,
                        control_after_generate="randomize",
                    ),
                ),
                InputSpec(
                    "max_duration",
                    FLOAT,
                    required=False,
                    default=120.0,
                    widget=NumberWidget(min=0.04, max=360.0, step=0.04),
                ),
                InputSpec(
                    "cfg_scale",
                    FLOAT,
                    required=False,
                    default=1.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
                ),
                InputSpec(
                    "top_k",
                    INT,
                    required=False,
                    default=50,
                    widget=NumberWidget(min=1, max=16384, step=1),
                ),
            ),
            outputs=(
                OutputSpec("conditioning", DINKSTER_CONDITIONING),
                OutputSpec("seconds", FLOAT),
            ),
            aliases=("MiniMaxMusic3TextEncode",),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        caption: str,
        lyrics: str,
        seed: int,
        max_duration: float,
        cfg_scale: float,
        top_k: int,
    ) -> Mapping[str, object]:
        raise RuntimeError("MiniMaxMusic3TextEncode requires the native execution arm")


class ControlNetLoader(Node):
    """ComfyUI-compatible ControlNet asset loader schema."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.ControlNetLoader",
            display_name="Load ControlNet Model",
            category="loaders",
            inputs=(
                InputSpec(
                    "control_net_name",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",), kind="model/controlnet"
                    ),
                ),
            ),
            outputs=(OutputSpec("control_net", CONTROL_NET),),
        )

    @classmethod
    def execute(cls, *, control_net_name: object) -> Mapping[str, object]:
        raise RuntimeError("ControlNetLoader requires the native execution arm")


class ControlNetApply(Node):
    """ComfyUI-compatible deprecated ControlNet apply schema."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.ControlNetApply",
            display_name="Apply ControlNet (DEPRECATED)",
            category="conditioning/controlnet",
            inputs=(
                InputSpec("conditioning", CONDITIONING),
                InputSpec("control_net", CONTROL_NET),
                InputSpec("image", IMAGE),
                InputSpec(
                    "strength",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        conditioning: object,
        control_net: object,
        image: object,
        strength: float,
    ) -> Mapping[str, object]:
        raise RuntimeError("ControlNetApply requires the native execution arm")


class ControlNetApplyAdvanced(Node):
    """ComfyUI-compatible advanced ControlNet apply schema."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.ControlNetApplyAdvanced",
            display_name="Apply ControlNet",
            category="conditioning/controlnet",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("control_net", CONTROL_NET),
                InputSpec("image", IMAGE),
                InputSpec(
                    "strength",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("vae", VAE, required=False),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        control_net: object,
        image: object,
        strength: float,
        start_percent: float,
        end_percent: float,
        vae: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("ControlNetApplyAdvanced requires the native execution arm")


class VAEDecode(Node):
    """Decode a latent into pixel-space images with a VAE.

    The canonical Dinkster port of ComfyUI's VAEDecode: the same decode
    path against the worker-resident VAE object, including the nested-
    latent unbind and the 5-dim video-batch flatten. Claiming the legacy
    name as an alias evicts the translated node (merge_native_nodes)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_decode",
            display_name="VAE Decode",
            category="model/latent",
            description="Decodes latent images back into pixel space images.",
            inputs=(
                InputSpec("samples", LATENT, doc="The latent to be decoded."),
                InputSpec(
                    "vae",
                    VAE,
                    doc="The VAE model used for decoding the latent.",
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, doc="The decoded image."),),
            # Legacy API prompts naming "VAEDecode" resolve here; both
            # inputs arrive over links, so no prompt-boundary adapter is
            # needed.
            aliases=("VAEDecode",),
            # The legacy class name plus ComfyUI's SEARCH_ALIASES.
            search_terms=(
                "VAEDecode",
                "decode",
                "decode latent",
                "latent to image",
                "render latent",
            ),
        )

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        latent = cast("Any", samples)["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        images = cast("Any", vae).decode(latent)
        if len(images.shape) == 5:  # combine video batches, as v1 does
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return cls.outputs(image=images)


class VAEDecodeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_decode_audio",
            display_name="VAE Decode Audio",
            category="model/latent",
            inputs=(InputSpec("samples", DINKSTER_LATENT), InputSpec("vae", DINKSTER_VAE)),
            outputs=(OutputSpec("audio", AUDIO),),
            aliases=("VAEDecodeAudio",),
            search_terms=("latent to audio",),
        )

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        raise RuntimeError("VAEDecodeAudio requires the native execution arm")


class VAEDecodeAudioTiled(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_decode_audio_tiled",
            display_name="VAE Decode Audio (Tiled)",
            category="model/latent",
            inputs=(
                InputSpec("samples", DINKSTER_LATENT),
                InputSpec("vae", DINKSTER_VAE),
                InputSpec(
                    "tile_size",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=32, max=8192, step=8),
                ),
                InputSpec(
                    "overlap",
                    INT,
                    required=False,
                    default=64,
                    widget=NumberWidget(min=0, max=1024, step=8),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO),),
            aliases=("VAEDecodeAudioTiled",),
            search_terms=("latent to audio",),
        )

    @classmethod
    def execute(
        cls, *, samples: object, vae: object, tile_size: int, overlap: int
    ) -> Mapping[str, object]:
        raise RuntimeError("VAEDecodeAudioTiled requires the native execution arm")


class VAEEncode(Node):
    """Encode pixel-space images into a latent with a VAE.

    The canonical Dinkster port of ComfyUI's VAEEncode: ``vae.encode``
    wrapped in the LATENT dict shape every sampler consumes. Claiming
    the legacy name as an alias evicts the translated node
    (merge_native_nodes)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_encode",
            display_name="VAE Encode",
            category="model/latent",
            description="Encodes pixel space images into latent images.",
            inputs=(
                InputSpec("pixels", IMAGE, doc="The image to be encoded."),
                InputSpec(
                    "vae",
                    VAE,
                    doc="The VAE model used for encoding the image.",
                ),
            ),
            outputs=(OutputSpec("latent", LATENT, doc="The encoded latent."),),
            aliases=("VAEEncode",),
            # The legacy class name plus ComfyUI's SEARCH_ALIASES.
            search_terms=(
                "VAEEncode",
                "encode",
                "encode image",
                "image to latent",
            ),
        )

    @classmethod
    def execute(cls, *, pixels: object, vae: object) -> Mapping[str, object]:
        return cls.outputs(latent={"samples": cast("Any", vae).encode(pixels)})


# Static COMBO options for dinkster.ksampler: a SNAPSHOT of ComfyUI's core
# sampler/scheduler lists, baked so the dropdown renders immediately (the
# frontend-agreed wire-v9 contract wants at least one static option here).
# The remote route is authoritative - a fetched /api/choices/* result
# REPLACES these - so drift against a newer ComfyUI costs nothing at
# runtime; refresh the snapshot when it gets embarrassing. Execution never
# validates against either list: a name outside it (a remote-registered
# sampler, a stale workflow) goes to comfy.samplers verbatim and fails
# loudly there if genuinely unknown - choices are UI vocabulary, never
# execution identity.
SAMPLER_CHOICES: tuple[str, ...] = (
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
    "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde",
    "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
    "ipndm", "ipndm_v", "deis",
    "res_multistep", "res_multistep_cfg_pp",
    "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde",
    "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece",
    "ddim", "uni_pc", "uni_pc_bh2",
)  # fmt: skip
SCHEDULER_CHOICES: tuple[str, ...] = (
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform",
    "beta", "normal", "linear_quadratic", "kl_optimal",
)  # fmt: skip


class KSampler(Node):
    """Denoise a latent with a model - the canonical sampling node.

    The canonical Dinkster port of ComfyUI's KSampler keeps its inputs and
    defaults. Execution delegates to the same ComfyUI sampler kernels while
    adapting their callback state. Claiming the legacy name as an alias evicts
    the translated node (merge_native_nodes).
    sampler_name/scheduler are concrete core.combo sockets wearing wire-v9
    COMBO widgets: baked static options for immediate render, an authoritative
    /api/choices/* remote route, and a refresh button. Truthful v1 numeric
    presentation rides NumberWidget metadata, while the full input contract
    remains execute-side validation (the EmptyLatentImage precedent)."""

    MAX_SEED = 0xFFFFFFFFFFFFFFFF
    MAX_STEPS = 10000
    MAX_CFG = 100.0

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ksampler",
            display_name="KSampler",
            category="model/sampling",
            description=(
                "Uses the provided model, positive and negative "
                "conditioning to denoise the latent image."
            ),
            inputs=(
                InputSpec(
                    "model",
                    MODEL,
                    doc="The model used for denoising the input latent.",
                ),
                InputSpec(
                    "seed",
                    INT,
                    required=False,
                    default=0,
                    doc="The random seed used for creating the noise.",
                    widget=NumberWidget(
                        min=0,
                        max=cls.MAX_SEED,
                        step=1,
                        control_after_generate="randomize",
                    ),
                ),
                InputSpec(
                    "steps",
                    INT,
                    required=False,
                    default=20,
                    doc="The number of steps used in the denoising process.",
                    widget=NumberWidget(min=1, max=10_000),
                ),
                InputSpec(
                    "cfg",
                    FLOAT,
                    required=False,
                    default=8.0,
                    doc=(
                        "The Classifier-Free Guidance scale balances "
                        "creativity and adherence to the prompt. Higher "
                        "values result in images more closely matching the "
                        "prompt however too high values will negatively "
                        "impact quality."
                    ),
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "sampler_name",
                    COMBO,
                    required=False,
                    default="euler",
                    doc=(
                        "The algorithm used when sampling, this can affect "
                        "the quality, speed, and style of the generated "
                        "output."
                    ),
                    widget=ComboWidget(
                        options=SAMPLER_CHOICES,
                        remote_route="/api/choices/comfy.samplers",
                        refresh_button=True,
                    ),
                ),
                InputSpec(
                    "scheduler",
                    COMBO,
                    required=False,
                    default="simple",
                    doc=(
                        "The scheduler controls how noise is gradually removed to form the image."
                    ),
                    widget=ComboWidget(
                        options=SCHEDULER_CHOICES,
                        remote_route="/api/choices/comfy.schedulers",
                        refresh_button=True,
                    ),
                ),
                InputSpec(
                    "positive",
                    CONDITIONING,
                    doc=(
                        "The conditioning describing the attributes you "
                        "want to include in the image."
                    ),
                ),
                InputSpec(
                    "negative",
                    CONDITIONING,
                    doc=(
                        "The conditioning describing the attributes you "
                        "want to exclude from the image."
                    ),
                ),
                InputSpec(
                    "latent_image",
                    LATENT,
                    doc="The latent image to denoise.",
                ),
                InputSpec(
                    "denoise",
                    FLOAT,
                    required=False,
                    default=1.0,
                    doc=(
                        "The amount of denoising applied, lower values will "
                        "maintain the structure of the initial image "
                        "allowing for image to image sampling."
                    ),
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT, doc="The denoised latent."),),
            # Legacy API prompts naming "KSampler" resolve here; the
            # original input ids retain the v1 signature, so no
            # prompt-boundary adapter is needed.
            aliases=("KSampler",),
            # The legacy class name plus ComfyUI's SEARCH_ALIASES.
            search_terms=(
                "KSampler",
                "sampler",
                "sample",
                "generate",
                "denoise",
                "diffuse",
                "txt2img",
                "img2img",
            ),
            emits_previews=True,
        )


class KSamplerAdvanced(Node):
    """Run one exact step range of a full Comfy-compatible sampling schedule."""

    MAX_SEED = KSampler.MAX_SEED
    MAX_STEPS = KSampler.MAX_STEPS
    MAX_CFG = KSampler.MAX_CFG

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ksampler_advanced",
            display_name="KSampler (Advanced)",
            category="model/sampling",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "add_noise",
                    COMBO,
                    default="enable",
                    widget=ComboWidget(options=("enable", "disable")),
                    advanced=True,
                ),
                InputSpec(
                    "noise_seed",
                    INT,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=cls.MAX_SEED,
                        step=1,
                        control_after_generate="randomize",
                    ),
                ),
                InputSpec("steps", INT, default=20, widget=NumberWidget(min=1, max=10_000)),
                InputSpec(
                    "cfg",
                    FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "sampler_name",
                    COMBO,
                    default="euler",
                    widget=ComboWidget(
                        options=SAMPLER_CHOICES,
                        remote_route="/api/choices/comfy.samplers",
                        refresh_button=True,
                    ),
                ),
                InputSpec(
                    "scheduler",
                    COMBO,
                    default="simple",
                    widget=ComboWidget(
                        options=SCHEDULER_CHOICES,
                        remote_route="/api/choices/comfy.schedulers",
                        refresh_button=True,
                    ),
                ),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("latent_image", LATENT),
                InputSpec(
                    "start_at_step",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=10_000),
                    advanced=True,
                ),
                InputSpec(
                    "end_at_step",
                    INT,
                    default=10_000,
                    widget=NumberWidget(min=0, max=10_000),
                    advanced=True,
                ),
                InputSpec(
                    "return_with_leftover_noise",
                    COMBO,
                    default="disable",
                    widget=ComboWidget(options=("disable", "enable")),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("KSamplerAdvanced",),
            search_terms=("KSamplerAdvanced", "advanced sampler", "sample"),
            emits_previews=True,
        )


def _minimax_h3_tensor(value: object, name: str) -> Any:
    torch = importlib.import_module("torch")
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    tensor = cast("Any", value)
    if not tensor.is_floating_point() or tensor.layout != torch.strided:
        raise TypeError(f"{name} must be a strided floating tensor")
    return tensor


def _minimax_h3_audio(value: object, name: str = "audio") -> MiniMaxH3AudioReferenceValue:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be the standard waveform/sample_rate mapping")
    audio = cast("Mapping[object, object]", value)
    if set(audio) != {"waveform", "sample_rate"}:
        raise TypeError(f"{name} must be the standard waveform/sample_rate mapping")
    waveform = _minimax_h3_tensor(audio["waveform"], f"{name}.waveform")
    if (
        waveform.ndim != 3
        or waveform.shape[0] <= 0
        or waveform.shape[1] != 2
        or waveform.shape[2] <= 0
    ):
        raise ValueError(f"{name}.waveform must be nonempty [batch,2,samples]")
    sample_rate = audio["sample_rate"]
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError(f"{name}.sample_rate must be a positive integer")
    return MiniMaxH3AudioReferenceValue(waveform, sample_rate)


class EmptyMiniMaxMusic3LatentAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_minimax_music3_latent_audio",
            display_name="Empty MiniMax Music3 Latent Audio",
            category="model/latent/minimax music",
            description="Creates an empty MiniMax Music3 audio latent for the requested duration.",
            inputs=(
                InputSpec(
                    "seconds",
                    FLOAT,
                    required=False,
                    default=120.0,
                    widget=NumberWidget(min=0.04, max=360.0, step=0.04),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=4096, step=1),
                ),
            ),
            outputs=(OutputSpec("latent", DINKSTER_LATENT),),
            aliases=("EmptyMiniMaxMusic3LatentAudio",),
            dispatch_affinity="native",
        )

    @classmethod
    def execute(cls, *, seconds: float, batch_size: int) -> Mapping[str, object]:
        raise RuntimeError("EmptyMiniMaxMusic3LatentAudio requires the native execution arm")


class EmptyMiniMaxH3AV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_minimax_h3_av",
            display_name="Empty MiniMax H3 AV Latent",
            category="minimax h3",
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    default=1344,
                    widget=NumberWidget(min=32, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=768,
                    widget=NumberWidget(min=32, max=16384, step=32),
                ),
                InputSpec(
                    "frame_count",
                    INT,
                    default=124,
                    widget=NumberWidget(min=5, max=3600, step=17),
                ),
            ),
            outputs=(OutputSpec("latent", DINKSTER_LATENT),),
            aliases=("EmptyMiniMaxH3LatentAV",),
            dispatch_affinity="native",
        )

    @classmethod
    def execute(cls, *, width: int, height: int, frame_count: int) -> Mapping[str, object]:
        raise RuntimeError("dinkster.empty_minimax_h3_av requires the native execution arm")


class FrameRangeMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.frame_range_mask",
            display_name="Frame Range Mask",
            category="mask/timeline",
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    default=832,
                    widget=NumberWidget(min=1, max=16384, step=8),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=480,
                    widget=NumberWidget(min=1, max=16384, step=8),
                ),
                InputSpec(
                    "frames",
                    INT,
                    default=81,
                    widget=NumberWidget(min=1, max=16384, step=1),
                ),
                InputSpec(
                    "ranges",
                    STRING,
                    default="0:24",
                    widget=StringWidget(multiline=True),
                ),
            ),
            outputs=(OutputSpec("mask", MASK),),
        )

    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        frames: int,
        ranges: str,
    ) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        torch = importlib.import_module("torch")
        selected = inference.parse_frame_ranges(ranges, frames)
        mask = torch.zeros((frames, height, width), dtype=torch.float32)
        if selected:
            mask[list(selected)] = 1.0
        return cls.outputs(mask=mask)


class SetLatentMaskFromFrames(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.set_latent_mask_from_frames",
            display_name="Set Latent Mask from Frames",
            category="latent/mask",
            inputs=(
                InputSpec("latent", LATENT),
                InputSpec("vae", VAE),
                InputSpec("mask", MASK),
                InputSpec(
                    "spatial_reduction",
                    COMBO,
                    default="max",
                    widget=ComboWidget(options=("max", "min", "mean")),
                ),
                InputSpec(
                    "temporal_reduction",
                    COMBO,
                    default="max",
                    widget=ComboWidget(options=("max", "min", "mean", "first", "last")),
                ),
                InputSpec(
                    "operation",
                    COMBO,
                    default="replace",
                    widget=ComboWidget(options=("replace", "max", "min", "multiply")),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        latent: object,
        vae: object,
        mask: object,
        spatial_reduction: str,
        temporal_reduction: str,
        operation: str,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.set_latent_mask_from_frames requires the native execution arm")


class SetLatentMaskFromTimeRanges(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.set_latent_mask_from_time_ranges",
            display_name="Set Latent Mask from Time Ranges",
            category="latent/mask",
            inputs=(
                InputSpec("latent", LATENT),
                InputSpec("vae", VAE),
                InputSpec(
                    "ranges",
                    STRING,
                    default="0:1",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "selected",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "unselected",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "operation",
                    COMBO,
                    default="replace",
                    widget=ComboWidget(options=("replace", "max", "min", "multiply")),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        latent: object,
        vae: object,
        ranges: str,
        selected: float,
        unselected: float,
        operation: str,
    ) -> Mapping[str, object]:
        raise RuntimeError(
            "dinkster.set_latent_mask_from_time_ranges requires the native execution arm"
        )


class InspectLatentMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.inspect_latent_mask",
            display_name="Inspect Latent Mask",
            category="latent/mask",
            inputs=(InputSpec("latent", LATENT), InputSpec("vae", VAE)),
            outputs=(OutputSpec("mask", MASK), OutputSpec("report", STRING)),
        )

    @classmethod
    def execute(cls, *, latent: object, vae: object) -> Mapping[str, object]:
        raise RuntimeError("dinkster.inspect_latent_mask requires the native execution arm")


class MiniMaxH3ImageReference(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.minimax_h3_image_reference",
            display_name="MiniMax H3 Image Reference",
            category="minimax h3/references",
            inputs=(InputSpec("image", IMAGE),),
            outputs=(OutputSpec("reference", MINIMAX_H3_REFERENCE),),
        )

    @classmethod
    def execute(cls, *, image: object) -> Mapping[str, object]:
        tensor = _minimax_h3_tensor(image, "image")
        if tensor.ndim != 4 or tensor.shape[0] <= 0 or tensor.shape[-1] != 3:
            raise ValueError("image must be nonempty [batch,height,width,3]")
        return cls.outputs(reference=MiniMaxH3ImageReferenceValue(tensor))


class MiniMaxH3AudioReference(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.minimax_h3_audio_reference",
            display_name="MiniMax H3 Audio Reference",
            category="minimax h3/references",
            inputs=(InputSpec("audio", AUDIO),),
            outputs=(OutputSpec("reference", MINIMAX_H3_REFERENCE),),
        )

    @classmethod
    def execute(cls, *, audio: object) -> Mapping[str, object]:
        return cls.outputs(reference=_minimax_h3_audio(audio))


class MiniMaxH3VideoReference(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.minimax_h3_video_reference",
            display_name="MiniMax H3 Video Reference",
            category="minimax h3/references",
            inputs=(
                InputSpec("frames", IMAGE),
                InputSpec("audio", AUDIO, required=False, default=None),
            ),
            outputs=(OutputSpec("reference", MINIMAX_H3_REFERENCE),),
        )

    @classmethod
    def execute(cls, *, frames: object, audio: object = None) -> Mapping[str, object]:
        tensor = _minimax_h3_tensor(frames, "frames")
        if tensor.ndim != 4 or tensor.shape[0] <= 0 or tensor.shape[-1] != 3:
            raise ValueError("frames must be nonempty [time,height,width,3]")
        soundtrack = None if audio is None else _minimax_h3_audio(audio)
        return cls.outputs(reference=MiniMaxH3VideoReferenceValue(tensor, soundtrack))


def _minimax_h3_conditioning_schema(
    node_type: str,
    display_name: str,
    component_inputs: tuple[InputSpec, ...],
    extra_inputs: tuple[InputSpec, ...],
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="minimax h3/conditioning",
        inputs=(
            *component_inputs,
            InputSpec("target", LATENT),
            InputSpec("prompt", STRING, widget=StringWidget(multiline=True)),
            InputSpec(
                "negative_prompt",
                STRING,
                required=False,
                default=None,
                widget=StringWidget(multiline=True),
            ),
            *extra_inputs,
        ),
        outputs=(OutputSpec("positive", CONDITIONING), OutputSpec("negative", CONDITIONING)),
    )


class MiniMaxH3T2VAConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _minimax_h3_conditioning_schema(
            "dinkster.minimax_h3_t2va_conditioning",
            "MiniMax H3 T2VA Conditioning",
            (InputSpec("clip", CLIP),),
            (),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        target: object,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("MiniMax H3 conditioning requires the native execution arm")


class MiniMaxH3FL2VAConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _minimax_h3_conditioning_schema(
            "dinkster.minimax_h3_fl2va_conditioning",
            "MiniMax H3 FL2VA Conditioning",
            (InputSpec("clip", CLIP), InputSpec("video_vae", VAE)),
            (
                InputSpec("first_image", IMAGE, required=False, default=None),
                InputSpec("last_image", IMAGE, required=False, default=None),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        video_vae: object,
        target: object,
        prompt: str,
        negative_prompt: str | None = None,
        first_image: object = None,
        last_image: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("MiniMax H3 conditioning requires the native execution arm")


class MiniMaxH3REF2VAConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _minimax_h3_conditioning_schema(
            "dinkster.minimax_h3_ref2va_conditioning",
            "MiniMax H3 REF2VA Conditioning",
            (
                InputSpec("clip", CLIP),
                InputSpec("video_vae", VAE),
                InputSpec("audio_vae", VAE),
            ),
            (
                InputSpec("references", TypeExpr.list_of(MINIMAX_H3_REFERENCE)),
                InputSpec(
                    "ref_image_size",
                    COMBO,
                    default="match",
                    widget=ComboWidget(options=("match", "max")),
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        video_vae: object,
        audio_vae: object,
        target: object,
        prompt: str,
        references: object,
        ref_image_size: str,
        negative_prompt: str | None = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("MiniMax H3 conditioning requires the native execution arm")


class MiniMaxH3AddGuide(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.minimax_h3_add_guide",
            display_name="Add Guide for MiniMax H3",
            category="minimax h3/conditioning",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("vae", VAE, required=False, default=None),
                InputSpec("audio_vae", VAE, required=False, default=None),
                InputSpec("latent", LATENT),
                InputSpec("image", IMAGE, required=False, default=None),
                InputSpec("audio", AUDIO, required=False, default=None),
                InputSpec(
                    "frame_idx",
                    INT,
                    default=0,
                    widget=NumberWidget(min=-9999, max=9999, step=1),
                ),
            ),
            outputs=(OutputSpec("positive", CONDITIONING),),
            aliases=("MiniMaxH3AddGuide",),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        latent: object,
        frame_idx: int,
        vae: object = None,
        audio_vae: object = None,
        image: object = None,
        audio: object = None,
    ) -> Mapping[str, object]:
        raise RuntimeError("MiniMax H3 guides require the native execution arm")


class MiniMaxH3MotionContext(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.minimax_h3_motion_context",
            display_name="Add Motion Context to MiniMax H3",
            category="minimax h3/conditioning",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("latent", LATENT),
                InputSpec("previous_latent", LATENT),
                InputSpec(
                    "context_length",
                    INT,
                    default=22,
                    widget=NumberWidget(min=5, max=3600, step=17),
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("trim_time", FLOAT),
            ),
            aliases=("MiniMaxH3MotionContext",),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        latent: object,
        previous_latent: object,
        context_length: int,
    ) -> Mapping[str, object]:
        raise RuntimeError("MiniMax H3 motion context requires the native execution arm")


class MiniMaxH3AVEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.minimax_h3_av_encode",
            display_name="MiniMax H3 AV Encode",
            category="minimax h3/codec",
            inputs=(
                InputSpec("video_vae", VAE),
                InputSpec("audio_vae", VAE),
                InputSpec("frames", IMAGE),
                InputSpec("audio", AUDIO),
            ),
            outputs=(OutputSpec("latent", LATENT),),
        )

    @classmethod
    def execute(
        cls, *, video_vae: object, audio_vae: object, frames: object, audio: object
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.minimax_h3_av_encode requires the native execution arm")


class MiniMaxH3AVDecode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.minimax_h3_av_decode",
            display_name="MiniMax H3 AV Decode",
            category="minimax h3/codec",
            inputs=(
                InputSpec("video_vae", VAE),
                InputSpec("audio_vae", VAE),
                InputSpec("latent", LATENT),
            ),
            outputs=(OutputSpec("frames", IMAGE), OutputSpec("audio", AUDIO)),
        )

    @classmethod
    def execute(
        cls, *, video_vae: object, audio_vae: object, latent: object
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.minimax_h3_av_decode requires the native execution arm")


class EmptyLTXAVLatent(Node):
    """Blank video and audio latent streams sized for one LTX-2 clip.

    The latent geometry is model-derived (video VAE ratios, audio latent
    rate), so the node takes the loaded model rather than hardcoding one
    checkpoint's constants.
    """

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_ltxav_latent",
            display_name="Empty LTX-2 AV Latent",
            category="latent/multi-stream",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "width",
                    INT,
                    default=768,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=512,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "length",
                    INT,
                    default=97,
                    widget=NumberWidget(min=1, max=9999, step=8),
                ),
                InputSpec(
                    "frame_rate",
                    INT,
                    default=25,
                    widget=NumberWidget(min=1, max=240, step=1),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=4096, step=1),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        width: int,
        height: int,
        length: int,
        frame_rate: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.empty_ltxav_latent requires the native execution arm")


class EmptyLTXVLatent(Node):
    """Blank video latent stream sized for one LTX-Video clip.

    The latent geometry is model-derived (video VAE ratios), so the node
    takes the loaded model rather than hardcoding one checkpoint's
    constants.
    """

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_ltxv_latent",
            display_name="Empty LTX-Video Latent",
            category="latent/multi-stream",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec(
                    "width",
                    INT,
                    default=768,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=512,
                    widget=NumberWidget(min=64, max=16384, step=32),
                ),
                InputSpec(
                    "length",
                    INT,
                    default=97,
                    widget=NumberWidget(min=1, max=9999, step=8),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=4096, step=1),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        raise RuntimeError("dinkster.empty_ltxv_latent requires the native execution arm")


class ConcatAVLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.concat_av_latent",
            display_name="Concat Audio/Video Latent",
            category="latent/multi-stream",
            inputs=(
                InputSpec("video_latent", DINKSTER_LATENT),
                InputSpec("audio_latent", DINKSTER_LATENT),
            ),
            outputs=(OutputSpec("latent", DINKSTER_LATENT),),
        )

    @classmethod
    def execute(cls, *, video_latent: object, audio_latent: object) -> Mapping[str, object]:
        raise RuntimeError("dinkster.concat_av_latent requires the native execution arm")


class SeparateAVLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.separate_av_latent",
            display_name="Separate Audio/Video Latent",
            category="latent/multi-stream",
            inputs=(InputSpec("latent", DINKSTER_LATENT),),
            outputs=(
                OutputSpec("video_latent", DINKSTER_LATENT),
                OutputSpec("audio_latent", DINKSTER_LATENT),
            ),
        )

    @classmethod
    def execute(cls, *, latent: object) -> Mapping[str, object]:
        raise RuntimeError("dinkster.separate_av_latent requires the native execution arm")


class PreviewLatentVisual(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preview_latent_visual",
            display_name="Preview Visual Latent Stream",
            category="latent/multi-stream",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("latent", LATENT),
                InputSpec("role", STRING, default="video", widget=StringWidget()),
            ),
            outputs=(OutputSpec("image", IMAGE),),
        )

    @classmethod
    def execute(cls, *, model: object, latent: object, role: str) -> Mapping[str, object]:
        raise RuntimeError("dinkster.preview_latent_visual requires the native execution arm")


class PreviewLatentAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preview_latent_audio",
            display_name="Preview Audio Latent Stream",
            category="latent/multi-stream",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("latent", LATENT),
                InputSpec("role", STRING, default="audio", widget=StringWidget()),
            ),
            outputs=(OutputSpec("audio", AUDIO),),
        )

    @classmethod
    def execute(cls, *, model: object, latent: object, role: str) -> Mapping[str, object]:
        raise RuntimeError("dinkster.preview_latent_audio requires the native execution arm")


def mount_writer() -> AssetWriter:
    """The worker's write gate: authority is the engine-published mount
    snapshot, re-read per save - grants and revokes made while this worker
    runs apply to the very next save, no restart."""
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError(
            "saving requires filesystem mounts, but this worker has no "
            "DINKSTER_MOUNTS_SNAPSHOT configured (run with a library root)"
        )
    return AssetWriter(MountSnapshotWriter(snapshot))


class LoadLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        latent = TypeExpr.concrete("dinkster.latent")
        return NodeSchema(
            node_type="dinkster.load_latent",
            display_name="Load Latent",
            category="latent",
            aliases=("LoadLatent",),
            inputs=(
                InputSpec(
                    "asset",
                    TypeExpr.asset_of(latent),
                    widget=AssetWidget(
                        accept=(LATENT_MEDIA_TYPE, "application/octet-stream"),
                        kind=LATENT_ASSET_KIND,
                        allow_upload=True,
                    ),
                    source_filename=SourceFilenameSpec("data/latent", "input"),
                ),
            ),
            outputs=(
                OutputSpec("samples", latent),
                OutputSpec("vae_hint", STRING),
            ),
        )

    @classmethod
    def execute(cls, *, asset: object) -> Mapping[str, object]:
        assert isinstance(asset, AssetRef)
        codec = importlib.import_module("dinkster_inference_torch.latent_assets")
        torch = cast("Any", importlib.import_module("torch"))
        with asset.open() as source:
            samples, hint = codec.load_latent(source, torch)
        return cls.outputs(samples={"samples": samples}, vae_hint=hint)


class SaveLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        latent = TypeExpr.concrete("dinkster.latent")
        return NodeSchema(
            node_type="dinkster.save_latent",
            display_name="Save Latent",
            category="latent",
            inputs=(
                InputSpec("samples", latent),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default={"mount": "comfy-output", "prefix": "latents/ComfyUI"},
                    widget=SaveTargetWidget(LATENT_SUFFIX),
                ),
                InputSpec("vae", DINKSTER_VAE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("samples", latent),
                OutputSpec("asset", TypeExpr.asset_of(latent)),
            ),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls, *, samples: object, target: object, vae: object = None
    ) -> Mapping[str, object]:
        if not isinstance(samples, Mapping) or "samples" not in samples:
            raise AssetError("LATENT input must contain samples")
        codec = importlib.import_module("dinkster_inference_torch.latent_assets")
        source = default_pool().source_for(vae) if vae is not None else None
        context = current_execution_context()
        spool = codec.serialize_native_latent(
            cast("Mapping[str, object]", samples)["samples"],
            snapshot=None if context is None else context.export_snapshot,
            vae_hint=None if source is None else source.hint(),
        )
        try:
            asset = mount_writer().save_stream(
                target,
                spool,
                suffix=LATENT_SUFFIX,
                media_type=LATENT_MEDIA_TYPE,
                limit=MAX_LATENT_DATA_BYTES + MAX_LATENT_HEADER_BYTES + 8,
            )
        finally:
            spool.close()
        return cls.outputs(samples=cast(object, samples), asset=asset)


NATIVE_NODES: tuple[type[Node], ...] = (
    *USDU_CARRIER_NODES,
    EmptyMiniMaxMusic3LatentAudio,
    MiniMaxMusic3TextEncode,
    VAEDecodeAudio,
    VAEDecodeAudioTiled,
    EmptyMiniMaxH3AV,
    FrameRangeMask,
    SetLatentMaskFromFrames,
    SetLatentMaskFromTimeRanges,
    InspectLatentMask,
    MiniMaxH3ImageReference,
    MiniMaxH3AudioReference,
    MiniMaxH3VideoReference,
    MiniMaxH3T2VAConditioning,
    MiniMaxH3FL2VAConditioning,
    MiniMaxH3REF2VAConditioning,
    MiniMaxH3AddGuide,
    MiniMaxH3MotionContext,
    MiniMaxH3AVEncode,
    MiniMaxH3AVDecode,
    ConcatAVLatent,
    SeparateAVLatent,
    PreviewLatentVisual,
    PreviewLatentAudio,
    Wan21ClipVisionEncode,
    BerniniConditioning,
    Wan21ImageToVideo,
    WanCameraEmbedding,
    WanCameraImageToVideo,
    WanPhantomSubjectToVideo,
    WanTrackToVideo,
    WanMoveTracksFromCoords,
    WanMoveConcatTrack,
    WanMoveGenerateTracks,
    WanMoveVisualizeTracks,
    WanMoveTrackToVideo,
    WanFirstLastFrameToVideo,
    WanFunControlToVideo,
    Wan22FunControlToVideo,
    WanFunInpaintToVideo,
    WanVaceToVideo,
    TrimVideoLatent,
    Wan22ImageToVideoLatent,
    LoadZImageControlPatch,
    ApplyZImageControlPatch,
    LoadVae,
    LoadVision,
    LoadClip,
    LoadDualClip,
    LoadLatent,
    SaveLatent,
    EmptyHunyuanLatentVideo,
)

# Legacy v1 names claimed by ALWAYS-composed foundation nodes. merge_native_nodes
# must evict this worker's translated twins for them exactly as it does for
# NATIVE_NODES claims, but the one-way dependency rule (H6) forbids
# importing another pack here - so the names are hand-kept and
# tests/test_primitives.py pins them to the aliases dinkster-nodes-foundation
# actually declares. "PreviewAny" is deliberately absent: the translated
# node runs beside torch and stringifies resident values the
# engine-process dinkster.preview_any cannot, so it keeps the name.
STD_CLAIMED_V1_NAMES: tuple[str, ...] = (
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "PrimitiveBoolean",
    "CreateList",
)

# The media pack owns these native IDs; compat retains only their legacy prompt
# adapters and translated-node eviction claims.
MEDIA_IO_CLAIMED_V1_NAMES: tuple[str, ...] = (
    "LoadImage",
    "SaveImage",
)

IMAGE_CLAIMED_V1_NAMES: tuple[str, ...] = ("ResizeImageMaskNode",)

GENERATION_CLAIMED_V1_NAMES: tuple[str, ...] = (
    "EmptyTrellis2LatentStructure",
    "Trellis2Conditioning",
    "Pixal3DConditioning",
    "VaeDecodeStructureTrellis2",
    "Trellis2ShapeStage",
    "Trellis2UpsampleStage",
    "VaeDecodeShapeTrellis",
    "Trellis2TextureStage",
    "VaeDecodeTextureTrellis",
    "LoadMoGeModel",
    "MoGeInference",
    "MoGeGeometryToFOV",
    "LoadBackgroundRemovalModel",
    "RemoveBackground",
    "ImageCropToMask",
    "MaskPreview",
    "VoxelToMesh",
    "GetMeshInfo",
    "RemeshMesh",
    "DecimateMesh",
    "MeshSmoothNormals",
    "UnwrapMesh",
    "PaintMesh",
    "BakeTextureFromVoxel",
    "BakeNormalMapFromMesh",
    "BakeAmbientOcclusion",
    "RenderUVAtlas",
    "ApplyTextureToMesh",
    "MeshToFile3D",
    "CheckpointLoaderSimple",
    "ControlNetLoader",
    "ControlNetApply",
    "ControlNetApplyAdvanced",
    "SetUnionControlNetType",
    "UNETLoader",
    "LTXAVTextEncoderLoader",
    "LTXVAudioVAELoader",
    "LatentUpscaleModelLoader",
    "LoraLoader",
    "LoraLoaderModelOnly",
    "CLIPTextEncode",
    "CLIPTextEncodeLumina2",
    "ModelSamplingAuraFlow",
    "TextGenerate",
    "TextGenerateLTX2Prompt",
    "CLIPSetLastLayer",
    "T5TokenizerOptions",
    "CLIPTextEncodeControlnet",
    "FluxGuidance",
    "FluxDisableGuidance",
    "ReferenceLatent",
    "CFGZeroStar",
    "CFGNorm",
    "TCFG",
    "FreSca",
    "ContextWindowsManual",
    "WanContextWindowsManual",
    "LTXVContextWindows",
    "APG",
    "Mahiro",
    "Epsilon Scaling",
    "CFGOverride",
    "RescaleCFG",
    "RenormCFG",
    "TemporalScoreRescaling",
    "NAGuidance",
    "LTXVConditioning",
    "LTXVReferenceAudio",
    "LTXVSpatioTemporalGuidance",
    "LTXVModalityGuidance",
    "LTXVDurationPredictor",
    "LTXVDualCFGGuider",
    "LTXVImgToVideo",
    "LTXVImgToVideoInplace",
    "LTXVAddGuide",
    "LTXVCropGuides",
    "LTXVLatentUpsampler",
    "ConditioningCombine",
    "ConditioningAverage",
    "ConditioningConcat",
    "ConditioningMultiply",
    "ConditioningSetArea",
    "ConditioningSetAreaPercentage",
    "ConditioningSetAreaPercentageVideo",
    "ConditioningSetMask",
    "ConditioningSetTimestepRange",
    "ConditioningZeroOut",
    "ChromaRadianceOptions",
    "ModelSamplingSD3",
    "ModelSamplingLTXV",
    "ModelSamplingFlux",
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyChromaRadianceLatentImage",
    "EmptyFlux2LatentImage",
    "KSampler",
    "KSamplerAdvanced",
    "KSamplerSelect",
    "SamplerDPMPP_3M_SDE",
    "SamplerDPMPP_2M_SDE",
    "SamplerDPMPP_SDE",
    "SamplerDPMPP_2S_Ancestral",
    "SamplerEulerAncestral",
    "SamplerEulerAncestralCFGPP",
    "SamplerLMS",
    "SamplerDPMAdaptative",
    "SamplerER_SDE",
    "SamplerSEEDS2",
    "SamplerSASolver",
    "BasicScheduler",
    "BetaSamplingScheduler",
    "SDTurboScheduler",
    "KarrasScheduler",
    "ExponentialScheduler",
    "PolyexponentialScheduler",
    "LaplaceScheduler",
    "VPScheduler",
    "AlignYourStepsScheduler",
    "GITSScheduler",
    "OptimalStepsScheduler",
    "Flux2Scheduler",
    "ManualSigmas",
    "SplitSigmas",
    "SplitSigmasDenoise",
    "FlipSigmas",
    "SetFirstSigma",
    "ExtendIntermediateSigmas",
    "SamplingPercentToSigma",
    "BasicGuider",
    "CFGGuider",
    "DualCFGGuider",
    "PerpNegGuider",
    "DisableModelCfg1Optimization",
    "DisableNoise",
    "RandomNoise",
    "AddNoise",
    "SamplerCustom",
    "SamplerCustomAdvanced",
    "LTXVAudioVAEDecode",
    "VAEDecode",
    "VAEDecodeTiled",
    "VAEEncode",
    "VAEEncodeTiled",
    "SeedVR2Preprocess",
    "SeedVR2PostProcessing",
    "SeedVR2Conditioning",
    "SeedVR2TemporalChunk",
    "SeedVR2TemporalMerge",
)


def _explicit_legacy_node(node: type[Node]) -> type[Node]:
    schema = replace(node.schema(), aliases=())

    def define_schema(cls: type[Node]) -> NodeSchema:
        return schema

    return type(node.__name__, (node,), {"define_schema": classmethod(define_schema)})


def merge_native_nodes(
    translated: Iterable[type[Node]],
) -> tuple[type[Node], ...]:
    """Translated nodes with native replacements evicted, natives appended.

    A native node replaces a translated one when it claims the legacy name:
    either by reusing the translated node id outright or by carrying the v1
    class name as an alias. Eviction keeps aliases unambiguous because
    translated nodes carry their v1 name too. An additive native may
    intentionally claim no legacy name and therefore evict nothing.

    Separately composed schema owners count too. Their claims arrive through
    the hand-kept claim lists rather than pack imports; schema-owner tests pin
    each list so drift fails CI. The legacy empty latent remains addressable
    by its explicit id for graphs consuming comfy.LATENT; its unqualified
    alias belongs only to the native generation provider."""
    claimed: set[str] = {
        comfy_type_id(name)
        for name in (
            *STD_CLAIMED_V1_NAMES,
            *MEDIA_IO_CLAIMED_V1_NAMES,
            *IMAGE_CLAIMED_V1_NAMES,
            *GENERATION_CLAIMED_V1_NAMES,
        )
    }
    for node in NATIVE_NODES:
        schema = node.schema()
        claimed.add(schema.node_type)
        claimed.update(comfy_type_id(alias) for alias in schema.aliases)
    kept: list[type[Node]] = []
    for node in translated:
        if node.schema().node_type == "comfy.EmptyLatentImage":
            kept.append(_explicit_legacy_node(node))
        elif node.schema().node_type not in claimed:
            kept.append(node)
    return (*kept, *NATIVE_NODES)


def register_native_types(registry: TypeRegistry) -> None:
    """Register what the native nodes need beyond the v1 translation:
    ``dinkster.asset`` (resolver-bound when DINKSTER_ASSET_ROOT is set) and the
    resident model types, when the translated node set didn't already
    bring them in."""
    if ASSET_TYPE not in registry:
        # The one shared chain assembly (vault CAS, live mounts snapshot,
        # indexed library root) - see dinkster_assets.declared.
        register_asset_type(registry, resolver_from_env())
    if SAVE_TARGET_TYPE not in registry:
        register_save_target_type(registry)
    register_model3d_type(registry, "dinkster.model3d")
    for expr in (MINIMAX_H3_REFERENCE,):
        (type_id,) = expr.types
        if type_id not in registry:
            registry.register(type_id)
    for expr in (
        MODEL,
        CLIP,
        CLIP_VISION,
        VAE,
        CONTROL_NET,
        TypeExpr.concrete("dinkster.model"),
        TypeExpr.concrete("dinkster.clip"),
        TypeExpr.concrete("dinkster.clip-vision"),
        TypeExpr.concrete("dinkster.vae"),
    ):
        (type_id,) = expr.types
        if type_id not in registry:
            # Same pool the loader labels into: codec and label() must
            # agree on the table, or they would assign different rids to
            # the same object.
            register_resident_type(
                registry, type_id, table=default_pool(), meta=comfy_resident_meta
            )
    # LATENT has a structural tensor codec; CONDITIONING remains opaque data.
    (latent_type,) = LATENT.types
    if latent_type not in registry:
        register_latent_type(registry, latent_type)
    for expr in (CONDITIONING, CLIP_VISION_OUTPUT, WAN_CAMERA_EMBEDDING, TRACKS):
        (type_id,) = expr.types
        if type_id not in registry:
            registry.register(type_id)
    from .native_arm import sampler_wire_value

    for type_id in ("dinkster.sampler", "dinkster.sigmas", "dinkster.noise"):
        register_sampling_type(registry, type_id, sampler_coerce=sampler_wire_value)
    if "dinkster.guider" not in registry:
        register_resident_type(registry, "dinkster.guider", table=default_pool())
    register_inference_types(registry)
    register_curve_type(registry)
    for type_id in (
        "comfy.BACKGROUND_REMOVAL",
        "comfy.LATENT_UPSCALE_MODEL",
        "comfy.MODEL_PATCH",
        "comfy.MOGE_MODEL",
    ):
        if type_id not in registry:
            register_resident_type(
                registry, type_id, table=default_pool(), meta=comfy_resident_meta
            )
    for type_id in (
        "comfy.BASIC_PIPE",
        "comfy.HOOKS",
        "comfy.HOOK_KEYFRAMES",
        "comfy.MESH",
        "comfy.MOGE_GEOMETRY",
        "comfy.SHAPE_SUBDIVIDES",
        "comfy.TIMESTEPS_RANGE",
        "comfy.VOXEL",
        "dinkster.latent-operation",
    ):
        if type_id not in registry:
            registry.register(type_id)
    # IMAGE and MASK cross as npy bytes (image.py) so the torchless engine
    # can decode and render previews. Normally the v1 translation already
    # registered them; the guard covers filtered node sets (DINKSTER_COMFY_NODES)
    # where only the native ports mention them.
    (image_type,) = IMAGE.types
    if image_type not in registry:
        register_image_type(registry, image_type)
    if "dinkster.image" not in registry:
        register_image_type(registry, "dinkster.image")
    (mask_type,) = MASK.types
    if mask_type not in registry:
        register_image_type(registry, mask_type)
    if "dinkster.mask" not in registry:
        register_image_type(registry, "dinkster.mask")
    # Typed assets: asset<comfy.IMAGE> decode + comfy.IMAGE batch merge,
    # torch-producing worker halves of the host registrations in
    # comfy_compose.register_comfy_host_types (same provider identities).
    if registry.asset_decoder_for(image_type) is None:
        register_image_asset_providers(registry, image_type)
    for expr, register in (
        (AUDIO, register_audio_type),
        (VIDEO, register_video_type),
    ):
        (type_id,) = expr.types
        if type_id not in registry:
            register(registry, type_id)
