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

import ast
import asyncio
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import dinkster_api.v1 as api
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import register_core_types
from dinkster_workers import InProcessWorker

GOLDEN_V1_SURFACE = (
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
    # -- sampling extension authoring (dinkster-inference)
    "CancellationToken",
    "CompilerEmission",
    "ConditionScaleVector",
    "ControlApplication",
    "GraphCompilerDescriptor",
    "GuidanceCondition",
    "GuidanceContribution",
    "GuidanceEvaluationPlan",
    "GuidanceEvaluationRequest",
    "GuidanceEvaluationWrapperDescriptor",
    "GuidancePlanContext",
    "GuidancePostCFGContext",
    "GuidancePostCFGDescriptor",
    "GuidancePreCFGContext",
    "GuidancePreCFGDescriptor",
    "GuidancePrediction",
    "GuidancePredictionSource",
    "GuidancePredictions",
    "GuidanceReduceContext",
    "GuidanceResult",
    "GuidanceRole",
    "GuidanceScaleDescriptor",
    "GuidanceStrategyDescriptor",
    "InferenceContribution",
    "InputRewrite",
    "ContextDenoiser",
    "ContextSolverFn",
    "MirrorSpec",
    "MirrorTolerance",
    "ModelEvaluation",
    "NoiseKind",
    "NoiseSampler",
    "OptionKind",
    "OptionSpec",
    "OptionValue",
    "Parameterization",
    "ProgressScope",
    "SamplerContribution",
    "SamplerDescriptor",
    "SamplerInfo",
    "SchedulerDescriptor",
    "SamplingCancelled",
    "SamplingExecutionContext",
    "SolverFn",
    "StepCallback",
    "SourceFilenameSpec",
    "StepEvent",
    # -- node authoring (dinkster-schema)
    "ABSENT",
    "AbsentOutput",
    "AbsentPolicy",
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
    "COST_META_KEY",
    "CURVE_TYPE",
    "INLINE_STRING_CAP",
    "LENGTH_META_KEY",
    "MAX_CURVE_POINTS",
    "RESOURCES_META_KEY",
    "RESOURCE_ID_META_KEY",
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
    "effective_audio_facts",
    "decode_audio",
    "encode_audio",
    "register_curve_type",
    "register_resource_handle_type",
    "render_audio_wav",
    "stable_hash",
    # -- shared image-array codec (dinkster-values.image_codec): npy bytes
    # across interpreter boundaries, PNG renditions out. Added alongside
    # the comfy.IMAGE preview contract so packs register image types the
    # same way dev.image and the compat surface do.
    "IMAGE_BATCH_MERGER_ID",
    "PNG_CONTAINER_VERSION",
    "decode_image_array",
    "encode_image_array",
    "image_array_fingerprint",
    "image_array_meta",
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
    # -- model3d codec (dinkster-values.model3d_codec): raw GLB bytes across
    # boundaries, the original container as the browser rendition, and the
    # asset<dinkster.model3d> file decode provider.
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
    # asset<dinkster.splat> file decode provider.
    "SPLAT_FILE_DECODER_ID",
    "SPLAT_PLY_MIME",
    "decode_splat",
    "decode_splat_file",
    "encode_splat",
    "parse_ply_splat",
    "render_splat_ply",
    "splat_fingerprint",
    "splat_meta",
    "validate_splat_encoded",
    # -- video codec (dinkster-values.video_codec): encoded container bytes
    # across boundaries and the original container as the browser
    # rendition. Added so packs can register and operate on comfy.VIDEO
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
    "declared_asset",
    "digest_bytes",
    "digest_file",
    "is_digest",
    "register_asset_type",
    "resolver_from_env",
    # -- mounted saves (dinkster-assets + dinkster-schema widgets, wire v5)
    "AssetWidget",
    "AssetWriter",
    "bind_video_value",
    "MountSnapshotWriter",
    "SAVE_TARGET_TYPE",
    "SaveTarget",
    "SaveTargetWidget",
    "register_save_target_type",
    "register_audio_value_type",
    "register_video_value_type",
    "DITHERS",
    "FRAME_FORMATS",
    "read_video_metadata",
    "save_video_frames",
    "assemble_video",
    "disassemble_video",
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
    "validate_image_encoded",
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
assert not api._INFERENCE_EXPORTS.intersection(vars(api))
for name in ("not_an_export", "__path__"):
    try:
        getattr(api, name)
    except AttributeError as exc:
        assert str(exc) == f"module 'dinkster_api.v1' has no attribute {name!r}"
    else:
        raise AssertionError(name)
    assert not hasattr(api, name)
loaded = {name for name in sys.modules if name.split(".")[0] in {"dinkster_inference", "torch"}}
assert not loaded, sorted(loaded)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_deferred_exports_match_type_checking_declarations() -> None:
    assert api.__file__ is not None
    tree = ast.parse(Path(api.__file__).read_text(encoding="utf-8"))
    declarations = [
        alias
        for guard in tree.body
        if isinstance(guard, ast.If)
        and isinstance(guard.test, ast.Name)
        and guard.test.id == "TYPE_CHECKING"
        for statement in guard.body
        if isinstance(statement, ast.ImportFrom) and statement.module == "dinkster_inference"
        for alias in statement.names
    ]
    assert declarations
    assert all(alias.asname in (None, alias.name) for alias in declarations)
    declared_names = [alias.name for alias in declarations]
    assert len(declared_names) == len(set(declared_names))
    assert set(declared_names) == api._INFERENCE_EXPORTS
    assert api._INFERENCE_EXPORTS <= set(api.__all__)


@pytest.mark.parametrize("style", ["attribute", "from", "star", "inference-first"])
def test_deferred_export_imports_preserve_identity(style: str) -> None:
    script = """
import sys

style = sys.argv[1]
if style == "inference-first":
    import dinkster_inference
import dinkster_api.v1 as api

exports = list(api.__all__)
discovered = dir(api)
assert not api._INFERENCE_EXPORTS.intersection(vars(api))
if style != "inference-first":
    assert "dinkster_inference" not in sys.modules
namespace = {}
if style == "star":
    exec("from dinkster_api.v1 import *", namespace)
    assert namespace.keys() - {"__builtins__"} == set(exports)
for name in sorted(api._INFERENCE_EXPORTS):
    if style == "from":
        exec(f"from dinkster_api.v1 import {name}", namespace)
        value = namespace[name]
    elif style == "star":
        value = namespace[name]
    else:
        value = getattr(api, name)
    import dinkster_inference
    assert value is getattr(dinkster_inference, name), name
    assert value is getattr(api, name), name
    assert value is vars(api)[name], name
assert api.__all__ == exports
assert dir(api) == discovered
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, style], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("style", ["attribute", "from"])
def test_reload_discards_cached_inference_exports(style: str) -> None:
    script = """
import importlib
import sys
import dinkster_api.v1 as api

cached = {name: getattr(api, name) for name in api._INFERENCE_EXPORTS}
exports = list(api.__all__)
import dinkster_inference

sentinel = object()
try:
    dinkster_inference.CancellationToken = sentinel
    importlib.reload(api)
    assert not api._INFERENCE_EXPORTS.intersection(vars(api))
    assert api.__all__ == exports
    if sys.argv[1] == "from":
        from dinkster_api.v1 import CancellationToken as value
    else:
        value = getattr(api, "CancellationToken")
    assert value is sentinel
    from dinkster_api.v1 import CancellationToken
    assert CancellationToken is value
    assert api.CancellationToken is value
    assert vars(api)["CancellationToken"] is value
finally:
    dinkster_inference.CancellationToken = cached["CancellationToken"]
    importlib.reload(api)
assert api.CancellationToken is cached["CancellationToken"]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, style], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_golden_surface_is_exact() -> None:
    """The frozen v1 surface. A failure here is a compat event: additions
    update this list deliberately; removals/renames are forbidden within v1
    (they mean a v2 module, not an edit)."""
    assert sorted(api.__all__) == sorted(GOLDEN_V1_SURFACE)
    assert len(set(GOLDEN_V1_SURFACE)) == len(GOLDEN_V1_SURFACE)
    for name in GOLDEN_V1_SURFACE:
        assert hasattr(api, name), f"exported name missing on module: {name}"


def test_reexports_are_the_internal_objects() -> None:
    """Zero wrappers: the door hands out the internal objects themselves."""
    import dinkster_assets
    import dinkster_inference
    import dinkster_memory
    import dinkster_protocol
    import dinkster_schema
    import dinkster_values
    import dinkster_video

    sources = (
        dinkster_schema,
        dinkster_values,
        dinkster_protocol,
        dinkster_inference,
        dinkster_memory,
        dinkster_assets,
        dinkster_video,
    )
    for name in GOLDEN_V1_SURFACE:
        exported = getattr(api, name)
        owners = [m for m in sources if name in m.__all__]
        assert owners, f"{name} is not exported by any internal package"
        assert len(owners) == 1, f"{name} exported by multiple internal packages"
        assert getattr(owners[0], name) is exported, f"{name} is a wrapper/copy"


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
