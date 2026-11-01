"""Compat suite for the extension door (DESIGN 3.6, M7).

dinkster_api.v1 is the one import path packs may rely on. These tests are the
enforcement half of the promise:

- The golden surface test freezes the exported names: removing or renaming
  anything here fails loudly - within v1 the surface only grows, and every
  addition is a deliberate diff to GOLDEN_V1_SURFACE.
- The identity test pins re-exports to the internal objects themselves - the
  door adds zero wrappers, so isinstance/identity agree on both sides.
- The authoring test writes a small pack THROUGH THE DOOR ONLY (node with a
  custom codec'd type, progress reporting, deliberate absence) and runs it on
  the real engine: what the door exposes is sufficient to author a pack.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Mapping

import dinkster_api.v1 as api
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import register_core_types
from dinkster_workers import InProcessWorker

GOLDEN_V1_SURFACE = (
    "TriangleMeshBatch",
    "materialize_mesh",
    "IMAGE_DOCUMENT_MEDIA_TYPE",
    "BLEND_MODES",
    "ImageDocument",
    "append_raster",
    "apply_commands",
    "blend",
    "canonical_json",
    "current_execution_context",
    "dissolve_alpha",
    "document_meta",
    "validate_image_document_encoded",
    "empty_document",
    "export_ora",
    "export_psd",
    "flatten",
    "from_comfy_layers",
    "linear_to_srgb",
    "load_document",
    "ordered_layer_ids",
    "raster_png",
    "register_comfy_compositor",
    "register_comfy_layers",
    "srgb_to_linear",
    "obtain_resident",
    "LatentStream",
    "MultiStreamLatent",
    "initialize_inference_process",
    "prepare_sampling_model",
    "materialize_audio",
    "materialize_clip_vision_output",
    "materialize_image",
    "materialize_latent",
    "materialize_shape_subdivides",
    "materialize_splat",
    "materialize_svg",
    "materialize_vision_packet",
    "materialize_voxel",
    "register_inference_types",
    "FrontendContribution",
    "FrontendModule",
    "JsonField",
    "JsonObjectSchema",
    "PackEvent",
    "PackRoute",
    "report_pack_event",
    "curve",
    "timeline_video",
    "video_document",
    "image_math",
    "timeline_document",
    "timeline_render",
    "timeline_runtime",
    # -- extension declarations and behavior identity (dinkster-protocol)
    "EXTENSION_CAPABILITIES",
    "EXTENSION_SCOPES",
    "ActiveExtension",
    "BehaviorValue",
    "CompositionMode",
    "ContributionSurfaceDescriptor",
    "ExtensionDeclaration",
    "ExtensionEntryPoints",
    "ExtensionScope",
    "ExtensionSnapshot",
    "GuidancePhase",
    "GuidancePhaseParticipation",
    "GuidanceRegistrySnapshot",
    "canonical_extension_snapshot",
    "extension_behavior_hash",
    # -- training session boundary values (dinkster-protocol)
    "TrainingSessionHandle",
    "ExportSnapshot",
    "current_export_snapshot",
    "MirrorSpec",
    "MirrorTolerance",
    "SourceFilenameSpec",
    # -- node authoring (dinkster-schema)
    "ABSENT",
    "BLOCKED",
    "AbsentOutput",
    "AbsentPolicy",
    "Expansion",
    "ExpansionBuilder",
    "ExpansionNode",
    "ExpansionOutput",
    "BooleanWidget",
    "ColorWidget",
    "CompositorWidget",
    "ConditionalWidgetCondition",
    "ConditionalWidgetGroup",
    "CurveWidget",
    "ComboOption",
    "ComboWidget",
    "ControlAfterGenerate",
    "CustomWidgetDescriptor",
    "Deprecation",
    "DynamicComboOption",
    "DynamicComboSpec",
    "DynamicSlotSpec",
    "InputFamilyOptionSource",
    "InputFamilyMapping",
    "InputFamilyMember",
    "InputFamilySpec",
    "InputSpec",
    "MappingSource",
    "MountWidget",
    "MultiComboWidget",
    "Node",
    "NodeOutputError",
    "NodeSchema",
    "NumberWidget",
    "OutputCountSpec",
    "OutputDescriptorsSpec",
    "OutputFamilyMapping",
    "OutputFamilyMember",
    "OutputFamilySpec",
    "OutputInterface",
    "OutputKnownValue",
    "OutputProbeSpec",
    "OutputRepresents",
    "OutputSpec",
    "ReplacementCase",
    "ReplacementLink",
    "ReplacementMigration",
    "ReplacementNode",
    "ReplacementPredicate",
    "ReplacementRule",
    "RunExecutionContext",
    "SearchVisibility",
    "SelectorSpec",
    "SlotValue",
    "SlotVariant",
    "StringWidget",
    "TextCompletionItem",
    "TextCompletionKind",
    "TextCompletions",
    "TypeExpr",
    "ValueTransform",
    "WidgetRepresentation",
    "WidgetRepresentations",
    "SCHEMA_WIRE_VERSION",
    "output_descriptor_entries",
    "schema_from_wire",
    "schema_signature",
    "schema_to_wire",
    # -- reporting and logging (dinkster-schema)
    "LOG_EVENT",
    "PREVIEW_EVENT",
    "PROGRESS_EVENT",
    "pack_logger",
    "attach_saved_artifact",
    "publish_preview_artifact",
    "publish_saved_file_preview",
    "report_event",
    "report_log",
    "report_preview",
    "report_progress",
    "report_value_diagnostic",
    # -- value types and codecs (dinkster-values)
    "CORE_BOOLEAN",
    "CORE_COMBO",
    "CORE_FLOAT",
    "CORE_INT",
    "CORE_STRING",
    "ARRAY_TYPE",
    "AUDIO_RECORD_TYPE",
    "AUDIO_ENCODER_OUTPUT_TYPE",
    "AUDIO_ENCODER_TYPE",
    "AUDIO_TYPE",
    "BACKGROUND_REMOVAL_TYPE",
    "BOUNDING_BOX_TYPE",
    "BOUNDING_BOXES_TYPE",
    "CAMERA_INFO_STATE_TYPE",
    "CLIP_TYPE",
    "CLIP_VISION_OUTPUT_TYPE",
    "CLIP_VISION_TYPE",
    "COLORS_TYPE",
    "COLOR_TYPE",
    "COMFY_ARRAY_TYPE",
    "COMPOSITOR_TYPE",
    "CONDITIONING_TYPE",
    "CONTEXT_WINDOWS_TYPE",
    "CONTROL_NET_TYPE",
    "DICT_TYPE",
    "FLOATS_TYPE",
    "GLIGEN_TYPE",
    "HISTOGRAM_TYPE",
    "PHOTOMAKER_TYPE",
    "GUIDER_TYPE",
    "HOOK_KEYFRAMES_TYPE",
    "HOOKS_TYPE",
    "IC_LORA_PARAMETERS_TYPE",
    "IMAGE_DOCUMENT_TYPE",
    "IMAGE_TYPE",
    "IMAGECOMPARE_TYPE",
    "IMPACT_BASIC_PIPE_TYPE",
    "INFERENCE_LATENT_OPERATION_TYPE",
    "INPAINT_CONDITIONING_TYPE",
    "INTERP_MODEL_TYPE",
    "LATENT_OPERATION_TYPE",
    "LATENT_TYPE",
    "LATENT_UPSCALE_MODEL_TYPE",
    "LAYERS_TYPE",
    "LOAD3D_CAMERA_TYPE",
    "LOAD3D_MODEL_INFO_TYPE",
    "LOAD_3D_TYPE",
    "MASK_TYPE",
    "MESH_TYPE",
    "MODEL3D_TYPE",
    "FILE_3D_TYPE",
    "FILE_3D_GLB_TYPE",
    "FILE_3D_GLTF_TYPE",
    "FILE_3D_FBX_TYPE",
    "FILE_3D_OBJ_TYPE",
    "FILE_3D_STL_TYPE",
    "FILE_3D_USDZ_TYPE",
    "FILE_3D_PLY_TYPE",
    "FILE_3D_SPLAT_TYPE",
    "FILE_3D_SPZ_TYPE",
    "FILE_3D_KSPLAT_TYPE",
    "FILE_3D_SPLAT_ANY_TYPE",
    "FILE_3D_POINT_CLOUD_ANY_TYPE",
    "MODEL_PATCH_TYPE",
    "MODEL_TYPE",
    "MOGE_GEOMETRY_TYPE",
    "MOGE_MODEL_TYPE",
    "DA3_MODEL_TYPE",
    "DA3_GEOMETRY_TYPE",
    "FACE_DETECTION_MODEL_TYPE",
    "FACE_LANDMARKS_TYPE",
    "SAM3D_BODY_MODEL_TYPE",
    "MHR_POSE_DATA_TYPE",
    "KIMODO_POSE_DATA_TYPE",
    "NOISE_TYPE",
    "OPTICAL_FLOW_TYPE",
    "POSE_KEYPOINT_TYPE",
    "POSE_TYPE",
    "RESOLUTION_PREVIEW_TYPE",
    "RESOURCE_HANDLE_TYPE",
    "SAMPLER_TYPE",
    "SAM3_TRACK_DATA_TYPE",
    "SEGMENTATION_TYPE",
    "SHAPE_SUBDIVIDES_TYPE",
    "SIGMAS_TYPE",
    "SPLAT_TYPE",
    "STYLE_MODEL_TYPE",
    "SVG_TYPE",
    "TIMESTEPS_RANGE_TYPE",
    "TRACKS_TYPE",
    "UPSCALE_MODEL_TYPE",
    "WAN_CAMERA_EMBEDDING_TYPE",
    "VAE_TYPE",
    "VECTOR_TYPE",
    "VIDEO_DOCUMENT_TYPE",
    "VIDEO_EDIT_TYPE",
    "VIDEO_TYPE",
    "VOXEL_TYPE",
    "COST_META_KEY",
    "CURVE_TYPE",
    "ENCODED_MEDIA_LIMIT_BYTES",
    "GIBIBYTE",
    "IMAGE_WORKING_SET_LIMIT_BYTES",
    "INLINE_STRING_CAP",
    "LENGTH_META_KEY",
    "MAX_CURVE_POINTS",
    "MEBIBYTE",
    "RESOURCES_META_KEY",
    "RESOURCE_ID_META_KEY",
    "VIDEO_AUDIO_WORKING_SET_LIMIT_BYTES",
    "VIDEO_FRAME_WORKING_SET_LIMIT_BYTES",
    "BufferEncoding",
    "Curve",
    "Rendition",
    "RenditionSpec",
    "ResourceHandle",
    "TypeId",
    "TypeRegistry",
    "TypeSpec",
    "AudioWindowReader",
    "append_audio_edit",
    "audio_channel_layout",
    "audio_fingerprint",
    "audio_from_source",
    "audio_meta",
    "audio_window",
    "array_storage_meta",
    "audio_input",
    "byte_storage_meta",
    "coerce_audio",
    "list_children",
    "Value",
    "effective_audio_facts",
    "decode_audio",
    "encode_audio",
    "register_box_types",
    "register_curve_type",
    "register_resource_handle_type",
    "render_audio_wav",
    "stable_hash",
    "storage_dtype",
    # -- shared image-array codec (dinkster-values.image_codec): npy bytes
    # across interpreter boundaries, PNG renditions out. Added alongside
    # the comfy.IMAGE preview contract so packs register image types the
    # same way dev.image and the compat surface do.
    "IMAGE_BATCH_MERGER_ID",
    "PNG_CONTAINER_VERSION",
    "ImageComponents",
    "ReadableImageSource",
    "decode_image_array",
    "decode_image_components",
    "open_image_components",
    "encode_image_array",
    "image_array_fingerprint",
    "image_array_meta",
    "image_input",
    "merge_image_batches",
    "prepare_image_array_encoding",
    "render_image_png",
    "render_mask_png",
    # -- region and detection codec (dinkster-values.detection_codec): JSON
    # region records and framed detection bytes across interpreter
    # boundaries, so detection provider packs produce the same values the
    # image pack's nodes consume.
    "DETECTION_TYPE",
    "Detection",
    "REGION_TYPE",
    "Region",
    "coerce_detection",
    "coerce_region",
    "decode_detection",
    "decode_region",
    "detection_meta",
    "encode_detection",
    "encode_region",
    "region_meta",
    "validate_detection_encoded",
    "validate_region_encoded",
    # -- model3d codec (dinkster-values.model3d_codec): raw GLB bytes across
    # boundaries, the original container as the browser rendition, and the
    # asset<MODEL3D> file decode provider.
    "MODEL3D_FILE_DECODER_ID",
    "decode_model3d",
    "decode_model3d_file",
    "encode_model3d",
    "model3d_fingerprint",
    "model3d_format",
    "model3d_meta",
    "register_model3d_type",
    "render_model3d_original",
    "validate_model3d_encoded",
    # -- gaussian splat codec (dinkster-values.splat_codec): framed tensor
    # buffers across boundaries, PLY interchange renditions, and the
    # asset<SPLAT> file decode provider.
    "SPLAT_FILE_DECODER_ID",
    "SPLAT_PLY_MIME",
    "decode_splat",
    "decode_splat_file",
    "encode_splat",
    "parse_ply_splat",
    "register_splat_type",
    "render_splat_ply",
    "splat_fingerprint",
    "splat_meta",
    "validate_splat_encoded",
    # -- video codec (dinkster-values.video_codec): encoded container bytes
    # across boundaries and the original container as the browser
    # rendition. Added so packs can register and operate on VIDEO
    # values the same way the compat surface does.
    "decode_video",
    "encode_video",
    "render_video_original",
    "validate_video_encoded",
    "video_fingerprint",
    "video_meta",
    "video_rendition_mime",
    # -- memory policy (dinkster-memory)
    "ConsumerItem",
    "FullReleaseResult",
    "InvocationView",
    "MeasuredMemory",
    "PressureSignal",
    "ReleaseCandidate",
    "ReservationPlanner",
    "ReservationRequest",
    "Shedder",
    # -- assets (dinkster-assets)
    "ASSET_TYPE",
    "AssetEntry",
    "AssetError",
    "AssetIntegrityError",
    "AssetRef",
    "AssetResolver",
    "AssetVault",
    "FolderListing",
    "IndexedAssetResolver",
    "LocalAssetRef",
    "declared_asset",
    "digest_bytes",
    "digest_file",
    "is_digest",
    "mount_roots_for_kind",
    "produced_output_scratch",
    "asset_selection_names",
    "asset_selection_options",
    "resolve_asset_selection",
    "register_asset_type",
    "register_load3d_type",
    "resolver_from_env",
    # -- mounted saves (dinkster-assets + current dinkster-schema widgets)
    "AssetWidget",
    "AssetWriter",
    "admit_video_source",
    "bind_video_value",
    "MountSnapshotWriter",
    "SAVE_TARGET_TYPE",
    "SaveTarget",
    "SaveTargetWidget",
    "register_save_target_type",
    "register_audio_value_type",
    "register_video_value_type",
    "register_file3d_types",
    "register_video_edit_type",
    "DITHERS",
    "FRAME_FORMATS",
    "read_video_metadata",
    "save_video_frames",
    "assemble_video",
    "concatenate_video",
    "disassemble_video",
    "iter_video_pixels",
    "save_frame_records",
    "save_video_stream",
    "coerce_video",
    "edit_video",
    "effective_video_facts",
    "video_from_source",
    "annotate_image",
    "annotate_mask",
    "copy_media_semantics",
    "mask_array_meta",
    "media_semantics",
    "validate_bounding_box_encoded",
    "validate_image_encoded",
    "validate_placement_array_encoded",
)


def test_authoring_import_does_not_initialize_media_runtimes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dinkster_api.v1; "
            "loaded = {'av', 'numpy'} & sys.modules.keys(); assert not loaded, loaded",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_node_authoring_import_and_discovery_do_not_load_inference() -> None:
    script = """
import sys
from dinkster_api.v1 import (
    Node, NodeSchema, InputSpec, OutputSpec, TypeExpr, TypeRegistry, report_progress,
)
import dinkster_api.v1 as api

assert set(api.__all__) <= set(dir(api))
for name in ("not_an_export", "__path__"):
    try:
        getattr(api, name)
    except AttributeError as exc:
        assert str(exc) == f"module 'dinkster_api.v1' has no attribute {name!r}"
    else:
        raise AssertionError(name)
    assert not hasattr(api, name)
loaded = {
    name
    for name in sys.modules
    if name.split(".")[0] in {"dinkster_inference_wire", "torch"}
}
assert not loaded, sorted(loaded)
from dinkster_api.v1 import initialize_inference_process, materialize_audio
assert callable(initialize_inference_process)
assert callable(materialize_audio)
assert not any(name.split('.')[0] in {'torch', 'dinkster_inference'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_public_multi_stream_latents_are_cold_structural_identities() -> None:
    script = """
import importlib.abc
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'dinkster_inference'}:
            raise AssertionError(f'runtime imported: {fullname}')

sys.meta_path.insert(0, BlockRuntime())
import dinkster_api.v1 as api
assert {'LatentStream', 'MultiStreamLatent'} <= set(dir(api))
from dinkster_api.v1 import LatentStream, MultiStreamLatent
from dinkster_inference_wire.latents import LatentStream as InternalStream
from dinkster_inference_wire.latents import MultiStreamLatent as InternalMulti
assert LatentStream is InternalStream
assert MultiStreamLatent is InternalMulti
audio, video = object(), object()
value = MultiStreamLatent((LatentStream('audio', audio), LatentStream('video', video)))
assert value.roles == ('audio', 'video')
assert value.by_role('audio') is audio
assert value.streams[-1].payload is video
reversed_streams = MultiStreamLatent.from_pairs((('video', video), ('audio', audio)))
assert reversed_streams.roles == ('video', 'audio')
assert not any(name.split('.')[0] in {'torch', 'dinkster_inference'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_controlnet_registration_preserves_resident_identity_without_torch() -> None:
    script = """
import sys
from dinkster_api.v1 import CONTROL_NET_TYPE, TypeRegistry, register_inference_types
from dinkster_memory.resident import default_pool
from dinkster_values import RESOURCE_ID_META_KEY, ResidentLookupError

assert CONTROL_NET_TYPE == "CONTROL_NET"
registry = TypeRegistry()
register_inference_types(registry)
spec = registry.spec(CONTROL_NET_TYPE)
register_inference_types(registry)
assert registry.spec(CONTROL_NET_TYPE) is spec
assert "comfy.CONTROL_NET" not in registry
source = object()
encoded = spec.encode(source)
assert spec.decode(encoded) is source
assert spec.fingerprint(source) == spec.fingerprint(spec.decode(encoded))
assert spec.meta(source)[RESOURCE_ID_META_KEY].startswith("resident:")
assert default_pool().release_resident(source)
try:
    spec.decode(encoded)
except ResidentLookupError:
    pass
else:
    raise AssertionError("released ControlNet remained resolvable")
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_upscale_model_registration_preserves_resident_identity_without_torch() -> None:
    script = """
import sys
from dinkster_api.v1 import UPSCALE_MODEL_TYPE, TypeRegistry, register_inference_types
from dinkster_memory.resident import default_pool
from dinkster_values import RESOURCE_ID_META_KEY, ResidentLookupError

assert UPSCALE_MODEL_TYPE == "UPSCALE_MODEL"
registry = TypeRegistry()
register_inference_types(registry)
spec = registry.spec(UPSCALE_MODEL_TYPE)
register_inference_types(registry)
assert registry.spec(UPSCALE_MODEL_TYPE) is spec
descriptor = object()
encoded = spec.encode(descriptor)
assert spec.decode(encoded) is descriptor
assert spec.fingerprint(descriptor) == spec.fingerprint(spec.decode(encoded))
assert spec.meta(descriptor)[RESOURCE_ID_META_KEY].startswith("resident:")
assert default_pool().release_resident(descriptor)
try:
    spec.decode(encoded)
except ResidentLookupError:
    pass
else:
    raise AssertionError("released upscale descriptor remained resolvable")
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("native_first", (True, False))
@pytest.mark.parametrize("type_name", ("GLIGEN", "PHOTOMAKER", "BACKGROUND_REMOVAL"))
@pytest.mark.parametrize("preexisting", (False, True))
def test_model_registration_preserves_resident_identity_without_torch(
    native_first: bool, type_name: str, preexisting: bool
) -> None:
    script = f"""
import sys
from dinkster_api.v1 import {type_name}_TYPE, TypeRegistry, register_inference_types
from dinkster_memory.resident import ResidentPool, default_pool
from dinkster_native.native import register_native_types
from dinkster_values import RESOURCE_ID_META_KEY, ResidentLookupError, register_resident_type

model_type = {type_name}_TYPE
assert model_type == {type_name!r}
registry = TypeRegistry()
table = ResidentPool() if {preexisting!r} else default_pool()
source = object()
original = register_resident_type(registry, model_type, table=table) if {preexisting!r} else None
original_encoded = original.encode(source) if original is not None else None
registrations = (register_native_types, register_inference_types)
if not {native_first!r}:
    registrations = registrations[::-1]
registrations[0](registry)
spec = registry.spec(model_type)
if original is not None:
    assert spec is original
    assert spec.decode(original_encoded) is source
registrations[1](registry)
assert registry.spec(model_type) is spec
register_inference_types(registry)
assert registry.spec(model_type) is spec
assert table in registry.resident_tables
assert "comfy.{type_name}" not in registry
encoded = spec.encode(source)
assert spec.decode(encoded) is source
assert spec.fingerprint(source) == spec.fingerprint(spec.decode(encoded))
assert spec.meta(source)[RESOURCE_ID_META_KEY].startswith("resident:")
assert table.release_resident(source)
try:
    spec.decode(encoded)
except ResidentLookupError:
    pass
else:
    raise AssertionError("released {type_name} remained resolvable")
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("type_name", ("OPTICAL_FLOW", "INTERP_MODEL"))
@pytest.mark.parametrize("native_first", (True, False))
def test_model_registration_uses_shared_holding_pool_without_torch(
    type_name: str, native_first: bool
) -> None:
    script = f"""
import json
import subprocess
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import {type_name}_TYPE as model_type, TypeRegistry, register_inference_types
from dinkster_memory.resident import ResidentPool, default_pool
from dinkster_native.native import register_native_types
from dinkster_values import ResidentLookupError, register_resident_type
from dinkster_values.builtin_types import {type_name}_TYPE as builtin_type, BUILTIN_TYPE_IDS
from dinkster_values import {type_name}_TYPE as values_type
from dinkster_workers.boundary import ValueCodec

assert model_type == values_type == builtin_type == {type_name!r}
assert model_type in BUILTIN_TYPE_IDS
holding_pool = default_pool()
registry = TypeRegistry()
registrations = (register_native_types, register_inference_types)
if not {native_first!r}:
    registrations = registrations[::-1]
registrations[0](registry)
spec = registry.spec(model_type)
for registration in (*registrations, *registrations):
    registration(registry)
    assert registry.spec(model_type) is spec
assert holding_pool in registry.resident_tables
assert 'comfy.' + model_type not in registry
model_patcher = object()
encoded = spec.encode(model_patcher)
assert spec.decode(encoded) is model_patcher
value = registry.wrap(model_type, model_patcher)
assert value.resolve() is model_patcher
assert value.fingerprint == registry.wrap(model_type, model_patcher).fingerprint
codec = ValueCodec(registry)
blobs = []
wire, _ = codec.encode(value, blobs, [])
foreign_script = '''
import json
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import TypeRegistry, register_inference_types
from dinkster_values import ResidentLookupError
from dinkster_workers.boundary import ValueCodec
registry = TypeRegistry()
register_inference_types(registry)
wire, blob_hex = json.loads(sys.stdin.read())
codec = ValueCodec(registry)
value, _ = codec.decode(wire, [bytes.fromhex(blob) for blob in blob_hex], [])
try:
    value.resolve()
except ResidentLookupError:
    pass
else:
    raise AssertionError('foreign process resolved a holding-pool value')
blobs = []
wire, _ = codec.encode(value, blobs, [])
print(json.dumps([wire, [blob.hex() for blob in blobs]]))
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
'''
relay = subprocess.run([sys.executable, '-c', foreign_script],
                       input=json.dumps([wire, [blob.hex() for blob in blobs]]),
                       capture_output=True, text=True, timeout=15)
assert relay.returncode == 0, relay.stderr
wire, blob_hex = json.loads(relay.stdout)
returned, _ = codec.decode(wire, [bytes.fromhex(blob) for blob in blob_hex], [])
assert returned.type_id == value.type_id
assert returned.fingerprint == value.fingerprint
assert returned.meta == value.meta
assert returned.resolve() is model_patcher

foreign_pool = ResidentPool()
foreign = TypeRegistry()
first_authority = register_resident_type(foreign, model_type, table=foreign_pool)
for registration in (*registrations, *registrations):
    registration(foreign)
    assert foreign.spec(model_type) is first_authority
try:
    first_authority.decode(encoded)
except ResidentLookupError:
    pass
else:
    raise AssertionError('foreign pool resolved a holding-pool value')
assert holding_pool.release_resident(model_patcher)
try:
    spec.decode(encoded)
except ResidentLookupError:
    pass
else:
    raise AssertionError('released model remained resolvable')
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_voxel_registration_is_public_cold_and_preserves_first_authority() -> None:
    script = """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
import dinkster_api.v1 as api
from dinkster_inference_wire import materialize_voxel
from dinkster_native.native import register_native_types
assert materialize_voxel is api.materialize_voxel
for registrations in ((api.register_inference_types, register_native_types),
                      (register_native_types, api.register_inference_types)):
    for preexisting in (True, False):
        registry = api.TypeRegistry()
        first = registry.register(api.VOXEL_TYPE) if preexisting else None
        registrations[0](registry)
        spec = registry.spec(api.VOXEL_TYPE)
        registrations[1](registry)
        registrations[0](registry)
        assert registry.spec(api.VOXEL_TYPE) is spec
        assert first is None or spec is first
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_bounding_box_registration_is_shared_cold_and_preserves_first_authority() -> None:
    script = """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import BOUNDING_BOX_TYPE, TypeRegistry, register_inference_types
from dinkster_nodes_comfy import register_inference_types as register_pack_types

# Hand-derived plain JSON values: an optional default, one frame, and
# multiple frames including an empty frame must retain structure and order.
values = (
    {'x': 0, 'y': 0, 'width': 512, 'height': 512},
    [{'x': 7, 'y': 3, 'width': 11, 'height': 5}],
    [[], [{'x': 7, 'y': 3, 'width': 11, 'height': 5, 'score': 0.75}]],
)
for registrations in (
    (register_inference_types, register_pack_types),
    (register_pack_types, register_inference_types),
):
    for preexisting in (False, True):
        registry = TypeRegistry()
        first = registry.register(BOUNDING_BOX_TYPE) if preexisting else None
        registrations[0](registry)
        spec = registry.spec(BOUNDING_BOX_TYPE)
        assert first is None or spec is first
        registrations[1](registry)
        register_inference_types(registry)
        register_pack_types(registry)
        assert registry.spec(BOUNDING_BOX_TYPE) is spec
        for value in values:
            assert spec.decode(spec.encode(value)) == value
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_editor_value_registration_is_public_cold_and_preserves_first_authority() -> None:
    script = """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
import dinkster_api.v1 as api
import dinkster_values as values
from dinkster_values import builtin_types
from dinkster_native.native import register_native_types

payloads = {
    'CAMERA_INFO_STATE': {'position': {'x': -3.25, 'y': 7, 'z': 0.5},
                         'target': [2, -1.5, 8], 'zoom': 1.25,
                         'extension': {'active': False, 'selection': None,
                                       'labels': ['right', 'left']}},
    'BOUNDING_BOXES': [{'x': 13, 'y': 7, 'width': 29, 'height': 11,
                        'metadata': {'label': 'left', 'score': 0.75}},
                       {'x': -2, 'y': 5, 'width': 3, 'height': 17, 'metadata': {}}],
    'IMAGECOMPARE': {'selected': 1, 'divider': 0.375, 'labels': ['right', 'left']},
    'COMPOSITOR': {'version': 1, 'canvas': {'width': 19, 'height': 7},
                   'order': [4, 1], 'layers': [{'id': 4, 'opacity': 0.25}],
                   'extension': {'enabled': False}},
    'LAYERS': {'version': 1, 'canvas': (19, 7), 'layers': []},
}
for type_id in payloads:
    name = type_id + '_TYPE'
    assert getattr(api, name) == getattr(values, name) == getattr(builtin_types, name) == type_id
    assert type_id in builtin_types.BUILTIN_TYPE_IDS
for registrations in ((api.register_inference_types, register_native_types),
                      (register_native_types, api.register_inference_types)):
    for preexisting in (False, True):
        registry = api.TypeRegistry()
        first = {t: registry.register(t) for t in payloads} if preexisting else {}
        registrations[0](registry)
        specs = {t: registry.spec(t) for t in payloads}
        registrations[1](registry)
        registrations[0](registry)
        for type_id, value in payloads.items():
            spec = registry.spec(type_id)
            assert spec is specs[type_id]
            assert not preexisting or spec is first[type_id]
            if not preexisting:
                assert spec.decode(spec.encode(value)) == value
                assert registry.wrap(type_id, spec.decode(spec.encode(value))).resolve() == value
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_golden_surface_is_exact() -> None:
    """The frozen v1 surface. A failure here is a compat event: additions
    update this list deliberately; removals/renames are forbidden within v1
    (they mean a v2 module, not an edit)."""
    assert sorted(api.__all__) == sorted(GOLDEN_V1_SURFACE)
    assert len(set(api.__all__)) == len(api.__all__)
    assert len(set(GOLDEN_V1_SURFACE)) == len(GOLDEN_V1_SURFACE)
    for name in GOLDEN_V1_SURFACE:
        assert hasattr(api, name), f"exported name missing on module: {name}"


def test_reexports_are_the_internal_objects() -> None:
    """Zero wrappers: the door hands out the internal objects themselves."""
    import dinkster_assets
    import dinkster_image_document.blend as document_blend
    import dinkster_image_document.compat as document_compat
    import dinkster_image_document.document as document
    import dinkster_image_document.format as document_format
    import dinkster_image_document.interchange as document_interchange
    import dinkster_inference_wire.latents as latent_values
    import dinkster_inference_wire.process as inference_process
    import dinkster_inference_wire.sparse as sparse_values
    import dinkster_inference_wire.types as inference_types
    import dinkster_memory
    import dinkster_protocol
    import dinkster_schema
    import dinkster_values
    import dinkster_video
    import dinkster_workers.execution as worker_execution

    document_sources = {
        "IMAGE_DOCUMENT_MEDIA_TYPE": document_format,
        "BLEND_MODES": document_format,
        "canonical_json": document_format,
        "ordered_layer_ids": document_format,
        "ImageDocument": document,
        "apply_commands": document,
        "append_raster": document,
        "document_meta": document,
        "validate_image_document_encoded": document,
        "empty_document": document,
        "flatten": document,
        "raster_png": document,
        "blend": document_blend,
        "dissolve_alpha": document_blend,
        "linear_to_srgb": document_blend,
        "srgb_to_linear": document_blend,
        "from_comfy_layers": document_compat,
        "register_comfy_compositor": document_compat,
        "register_comfy_layers": document_compat,
        "export_ora": document_interchange,
        "export_psd": document_interchange,
        "load_document": document_interchange,
        "current_execution_context": worker_execution,
    }

    sources = (
        latent_values,
        inference_process,
        sparse_values,
        inference_types,
        dinkster_schema,
        dinkster_values,
        dinkster_protocol,
        dinkster_memory,
        dinkster_assets,
        dinkster_video,
    )
    for name in GOLDEN_V1_SURFACE:
        exported = getattr(api, name)
        if name in document_sources:
            assert exported is getattr(document_sources[name], name), f"{name} is a wrapper/copy"
            continue
        owners = [m for m in sources if name in m.__all__]
        assert owners, f"{name} is not exported by any internal package"
        expected_owners = (
            2 if name in {"CustomWidgetDescriptor", "ASSET_TYPE", "SAVE_TARGET_TYPE"} else 1
        )
        assert len(owners) == expected_owners, f"{name} exported by unexpected internal packages"
        assert all(getattr(owner, name) is exported for owner in owners), (
            f"{name} is a wrapper/copy"
        )


def test_image_document_round_trip_through_v1() -> None:
    import numpy as np

    pixels = np.array([[[1, 0, 0], [0, 0.5, 1]]], dtype=np.float32)
    document = api.append_raster(api.empty_document(2, 1), api.raster_png(pixels))
    restored = api.ImageDocument(document.data)
    rendered, transparency = api.flatten(restored)
    np.testing.assert_allclose(rendered, [[[[1, 0, 0], [0, 128 / 255, 1]]]])
    np.testing.assert_array_equal(transparency, [[[0, 0]]])


def test_remote_client_reads_host_cancellation_through_v1() -> None:
    from dinkster_workers.execution import ExecutionContext, use_execution_context

    assert api.current_execution_context() is None
    cancelled = False
    context = ExecutionContext(None, None, cancelled=lambda: cancelled)
    with use_execution_context(context):
        active = api.current_execution_context()
        assert active is not None
        assert active is context
        assert not active.cancelled()
        cancelled = True
        assert active.cancelled()
    assert api.current_execution_context() is None


def test_dynamic_combo_specs_construct_through_v1() -> None:
    from dinkster_api.v1 import DynamicComboOption, DynamicComboSpec

    option = DynamicComboOption("Flux.2 [pro]")
    combo = DynamicComboSpec("model", (option,), default="Flux.2 [pro]")
    assert combo.options == (option,)
    assert combo.default == "Flux.2 [pro]"


def test_buffer_encoding_constructs_through_v1() -> None:
    registry = api.TypeRegistry()

    def prepare(_obj: object) -> api.BufferEncoding:
        return api.BufferEncoding(0, lambda _buffer: 0)

    spec = registry.register(
        "pack.buffered",
        encode=lambda _obj: b"",
        decode=lambda _data: object(),
        prepare_buffer_encoding=prepare,
    )

    assert spec.prepare_buffer_encoding is prepare
    assert callable(api.prepare_image_array_encoding)


def test_one_to_n_replacement_cases_construct_through_v1() -> None:
    from dinkster_api.v1 import (
        MappingSource,
        OutputFamilyMapping,
        OutputFamilyMember,
        ReplacementCase,
        ReplacementLink,
        ReplacementNode,
    )

    helper = ReplacementNode.build("helper.join", values={"count": 2})
    case = ReplacementCase.build(
        "new.node",
        nodes={"join": helper},
        inputs={"join:item": MappingSource.copy("picture")},
        output_families={
            "items": OutputFamilyMapping.from_members(OutputFamilyMember("0", "family_result"))
        },
        links=(ReplacementLink("join:items", "image"),),
        outputs={"join:preview": "result"},
    )
    assert case.nodes == (("join", helper),)
    assert case.links == (ReplacementLink("join:items", "image"),)
    assert case.output_families == (
        ("items", OutputFamilyMapping.from_members(OutputFamilyMember("0", "family_result"))),
    )


# -- a pack authored through the door only ------------------------------------


class Celsius:
    """A pack-defined value type: nothing but a wrapped float."""

    def __init__(self, degrees: float) -> None:
        self.degrees = degrees


class ReadThermometer(api.Node):
    @classmethod
    def define_schema(cls) -> api.NodeSchema:
        return api.NodeSchema(
            node_type="weather.read",
            display_name="Read Thermometer",
            category="weather",
            inputs=(api.InputSpec("degrees", api.TypeExpr.concrete(api.CORE_FLOAT)),),
            outputs=(
                api.OutputSpec("reading", api.TypeExpr.concrete("weather.celsius")),
                api.OutputSpec(
                    "warning",
                    api.TypeExpr.concrete(api.CORE_STRING),
                    optional=True,
                ),
            ),
        )

    @classmethod
    def execute(cls, *, degrees: float) -> Mapping[str, object]:
        api.report_progress(1, 2, text="reading")
        api.report_event("weather.station", {"sensor": "main"})
        api.report_progress(2, 2)
        warning: object = "heat" if degrees > 35.0 else api.AbsentOutput("all calm")
        return cls.outputs(reading=Celsius(degrees), warning=warning)


class DescribeReading(api.Node):
    @classmethod
    def define_schema(cls) -> api.NodeSchema:
        return api.NodeSchema(
            node_type="weather.describe",
            display_name="Describe Reading",
            category="weather",
            inputs=(api.InputSpec("reading", api.TypeExpr.concrete("weather.celsius")),),
            outputs=(api.OutputSpec("text", api.TypeExpr.concrete(api.CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, reading: Celsius) -> Mapping[str, object]:
        return cls.outputs(text=f"{reading.degrees:.1f} C")


def register_weather_types(registry: api.TypeRegistry) -> None:
    def encode(obj: object) -> bytes:
        assert isinstance(obj, Celsius)
        return repr(obj.degrees).encode("ascii")

    registry.register(
        "weather.celsius",
        encode=encode,
        decode=lambda data: Celsius(float(data.decode("ascii"))),
    )


def test_pack_authored_through_the_door_runs() -> None:
    """Everything a basic pack needs - schema, custom codec'd type,
    reporting, deliberate absence - is reachable through v1 alone."""
    pack_nodes = [ReadThermometer, DescribeReading]

    registry = api.TypeRegistry()
    register_core_types(registry)  # host-side composition, not pack code
    register_weather_types(registry)

    events: list[EngineEvent] = []
    engine = Engine(
        schemas=build_schemas(pack_nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(pack_nodes), registry),
        cache=MemoryLRUCache(),
        on_event=events.append,
    )
    graph = Graph(
        nodes={
            "read": GraphNode("weather.read", {"degrees": 21.5}),
            "say": GraphNode("weather.describe", {"reading": Link("read", "reading")}),
        }
    )
    result = asyncio.run(engine.run(graph, ["read", "say"]))

    assert result.outputs["say"]["text"].resolve() == "21.5 C"
    reading = result.outputs["read"]["reading"]
    assert reading.type_id == "weather.celsius"
    warning = result.outputs["read"]["warning"]
    assert warning.type_id == "core.absent"  # calm day: deliberate absence

    reports = [e for e in events if e.kind == "node_event"]
    names = [e.detail["name"] for e in reports]
    assert api.PROGRESS_EVENT in names
    assert "weather.station" in names
