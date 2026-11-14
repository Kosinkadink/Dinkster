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

from collections.abc import Mapping

import dinkster_api.v1 as api
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

GOLDEN_V1_SURFACE = (
    "MAX_COMPOSITOR_DIMENSION",
    "MAX_COMPOSITOR_LAYERS",
    "CompositorLayer",
    "CompositorRecipe",
    "CompositorTransform",
    "coerce_compositor_recipe",
    "coerce_delta",
    "register_compositor_type",
    "TriangleMeshBatch",
    "materialize_mesh",
    "register_load3d_camera_type",
    "register_load3d_model_info_type",
    "register_mesh_type",
    "register_shape_subdivides_type",
    "register_voxel_type",
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
    "default_pool",
    "comfy_resident_meta",
    "register_resident_type",
    "LatentStream",
    "MultiStreamLatent",
    "inference_nodes",
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
    "register_media_types",
    "register_vision_packet_types",
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
    "parse_color",
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
    "with_identity_replacement",
    # -- reporting and logging (dinkster-schema)
    "LOG_EVENT",
    "PREVIEW_EVENT",
    "PROGRESS_EVENT",
    "pack_logger",
    "attach_saved_artifact",
    "publish_image_document",
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
    "WEBCAM_TYPE",
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
    "decode_latent",
    "encode_latent",
    "latent_fingerprint",
    "validate_latent_encoded",
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
    "load_file3d",
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


def test_color_parser_through_v1() -> None:
    import numpy as np

    np.testing.assert_array_equal(
        api.parse_color("#12345678"), np.array([18, 52, 86, 120], dtype=np.float32) / 255
    )
    np.testing.assert_array_equal(api.parse_color("blue"), [0, 0, 1, 1])
    with pytest.raises(ValueError, match="color"):
        api.parse_color("#12345")


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
    result = run_scenario(engine.run(graph, ["read", "say"]))

    assert result.outputs["say"]["text"].resolve() == "21.5 C"
    reading = result.outputs["read"]["reading"]
    assert reading.type_id == "weather.celsius"
    warning = result.outputs["read"]["warning"]
    assert warning.type_id == "core.absent"  # calm day: deliberate absence

    reports = [e for e in events if e.kind == "node_event"]
    names = [e.detail["name"] for e in reports]
    assert api.PROGRESS_EVENT in names
    assert "weather.station" in names
