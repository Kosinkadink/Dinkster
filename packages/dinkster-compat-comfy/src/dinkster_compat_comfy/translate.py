"""ComfyUI v1 -> Dinkster translation, pure and importable without ComfyUI.

The compat strategy (DESIGN 3.10): legacy nodes stay quarantined behind
the same Worker boundary as everything else. This module is the whole
trick - given v1 node classes (INPUT_TYPES / RETURN_TYPES / FUNCTION),
it manufactures Dinkster Node subclasses whose schemas are honest V3-shaped
descriptions and whose execute() adapts calling conventions. The engine
never learns v1 existed.

Translation rules:

- v1 primitive type strings map to core envelope types; combo lists map
  to core.combo (choice vocabulary is UI affordance within that identity).
  When
  every choice is a non-empty string the list rides along as a
  ComboWidget with baked options, so the native wire renders the same
  dropdown v1 did - except under INPUT_IS_LIST, where the socket is
  list<core.combo> and a widget on a list socket is undefined in the
  wire contract. A combo whose choice list IS a filesystem listing (the
  very object folder_paths.get_filename_list returned, observed by an
  injected probe) additionally carries a remote route
  (/api/choices/comfy.files.<category>) so the vocabulary is
  re-fetchable instead of startup-frozen; see _remote_listing_combo for
  the honesty rules.
- Two-option combos that are disguised booleans (enable/disable, on/off,
  true/false, yes/no - case-insensitive, either order) become honest
  core.boolean inputs wearing a BooleanWidget whose labels are the
  original option strings. execute() maps the boolean back to the v1
  string, and a legacy prompt still sending the string passes through
  verbatim. v1 BOOLEAN inputs keep their label_on/label_off config as
  the same widget.
- Usable v1 INT/FLOAT min/max/step metadata and supported v1 INT
  control-after-generate metadata become NumberWidget presentation. Exact
  recognized lowercase display values survive without inference from bounds.
  Malformed, non-finite, unsafe, or contradictory constraints are omitted
  without changing the primitive socket, default, execution value, or schema
  identity. INPUT_IS_LIST carries no widget because the wire has no
  per-element widget contract.
- Exact FLOAT round, STRING placeholder/dynamicPrompts, combo
  control_after_generate, and COLOR input declarations survive as wire-v19
  schema facts. Unknown fields are omitted independently and no field
  is inferred from names, values, bounds, options, or multiline state.
- ``*`` maps to a wildcard TypeExpr.
- Any other type string T becomes the opaque envelope type ``comfy.T``,
  registered with default codec/fingerprint (correct-everywhere-possibly-
  slow); MODEL/CLIP/VAE stay interrogable envelopes without Dinkster knowing
  their internals. IMAGE is the exception: it registers the shared npy
  image codec (image.py) so the torchless engine can decode and render
  previews from the same bytes.
- v1 outputs are positional tuples; Dinkster outputs are id-keyed. Ids come
  from RETURN_NAMES when present, else lowercased RETURN_TYPES with
  positional suffixes on collision. execute() maps the returned tuple
  back onto those ids in order.
- V3 ``io.ComfyNode`` classes arrive through the SAME mapping (nodes.py
  registers them into NODE_CLASS_MAPPINGS directly; classproperties fake
  the v1 surface and FUNCTION names EXECUTE_NORMALIZED, which always
  returns ``io.NodeOutput``). MatchType variables, both Autogrow naming
  forms, DynamicCombo, DynamicSlot, nested dynamics, and static COMBO
  declarations survive the INPUT_TYPES shim. Unsupported structural markers
  refuse the node rather than becoming fake opaque atoms. execute() unwraps
  the NodeOutput shape
  (detected by base-class name, keeping this module ComfyUI-free):
  positional results become the v1 tuple, ``ui`` is dropped like the v1
  ui-dict half, and expansion/execution-blocking refuse loudly. Async
  functions keep the ordinary worker contract: their awaitable result is
  normalized only after it completes, with cancellation and failures owned
  by the worker invocation that already runs every native async node.
- OUTPUT_NODE, IS_CHANGED, or a hidden UNIQUE_ID marks the node
  non-idempotent: Dinkster will never cache it rather than emulate IS_CHANGED
  guessing or share identity-dependent results across graph nodes.
- v1 hidden inputs (PROMPT, UNIQUE_ID, EXTRA_PNGINFO) never enter the
  schema. execute() synthesizes UNIQUE_ID from the lowered graph node id and
  export metadata for output-node wrappers from the opaque compat snapshot;
  other declared hidden names receive None so wrapped signatures still bind.
- INPUT_IS_LIST / OUTPUT_IS_LIST translate to honest ``list<T>`` sockets
  (DESIGN 3.13). INPUT_IS_LIST is class-wide in v1 - the function is
  called once with every input as a whole list, widget/hidden values
  arriving length-1-wrapped - so every input type wraps in list<T> and a
  declared default becomes ``[default]``. OUTPUT_IS_LIST is per-output:
  only flagged positions wrap. execute() passes list values through
  verbatim, so the function is called exactly the way v1 calls it; the
  schema now *says* what v1's executor kept invisible. A flagged output
  returning a non-list raises at the cause instead of v1's silent
  ``extend()`` of whatever iterable came back.

Untranslatable nodes do not poison the pack: ``translate_mappings``
records them as skips-with-reasons when translating everything available,
and only fails loud when the caller named the node explicitly (asking for
a node you cannot have is an error; a pack containing one is a diagnostic).
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import math
import os
import sys
import textwrap
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, cast

from dinkster_assets import (
    ASSET_TYPE,
    AssetError,
    AssetRef,
    MountSnapshotResolver,
    register_asset_type,
    resolver_from_env,
    verified_local_path,
)
from dinkster_schema import (
    AssetWidget,
    BooleanWidget,
    ColorWidget,
    ComboWidget,
    ControlAfterGenerate,
    DynamicComboOption,
    DynamicComboSpec,
    DynamicEntry,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    MultiComboWidget,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SelectorSpec,
    SlotValue,
    SourceFilenameSpec,
    StringWidget,
    TypeExpr,
    Widget,
    validate_name,
)
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    TypeRegistry,
)
from dinkster_workers import CompatGateDiagnostic, current_execution_context

from .audio import AUDIO_V1_NAME, register_audio_type
from .image import IMAGE_V1_NAME, MASK_V1_NAME, register_image_type
from .latent import register_latent_type
from .resident import (
    DEFAULT_RESIDENT_V1_TYPES,
    ResidencyTable,
    register_resident_type,
)
from .saved_results import capture_saved_results
from .video import VIDEO_V1_NAME, register_video_type

COMFY_TYPE_PREFIX = "comfy."
V3_AUTOGROW_IO_TYPE = "COMFY_AUTOGROW_V3"
V3_DYNAMICCOMBO_IO_TYPE = "COMFY_DYNAMICCOMBO_V3"
V3_DYNAMICSLOT_IO_TYPE = "COMFY_DYNAMICSLOT_V3"
V3_MATCHTYPE_IO_TYPE = "COMFY_MATCHTYPE_V3"
V3_COMBO_IO_TYPE = "COMBO"

CUSTOM_COMBO_NODE_ID = "CustomCombo"
CUSTOM_COMBO_FAMILY_ID = "options"
CUSTOM_COMBO_OPTION_NAMES = tuple(f"option{index}" for index in range(1, 101))

#: Choice-list id root for filesystem-derived combo vocabularies: the
#: folder_paths category name rides behind it verbatim, so the id is a
#: pure function of upstream's own stable key (``comfy.files.checkpoints``,
#: ``comfy.files.loras``) - never of the listing's contents. The compat
#: pack's choices entry serves these behind /api/choices/{id}.
LISTING_CHOICE_PREFIX = "comfy.files."


class ModelFileCategory(NamedTuple):
    """One approved ComfyUI model-file category and its picker kind."""

    kind: str


MODEL_FILE_CATEGORIES: Mapping[str, ModelFileCategory] = MappingProxyType(
    {
        "audio_encoders": ModelFileCategory("model/audio-encoder"),
        "background_removal": ModelFileCategory("model/background-removal"),
        "checkpoints": ModelFileCategory("model/checkpoint"),
        "clip_vision": ModelFileCategory("model/clip-vision"),
        "controlnet": ModelFileCategory("model/controlnet"),
        "detection": ModelFileCategory("model/detection"),
        "diffusion_models": ModelFileCategory("model/diffusion"),
        "embeddings": ModelFileCategory("model/embedding"),
        "frame_interpolation": ModelFileCategory("model/frame-interpolation"),
        "geometry_estimation": ModelFileCategory("model/geometry-estimation"),
        "gligen": ModelFileCategory("model/gligen"),
        "hypernetworks": ModelFileCategory("model/hypernetwork"),
        "latent_upscale_models": ModelFileCategory("model/latent-upscaler"),
        "loras": ModelFileCategory("model/lora"),
        "model_patches": ModelFileCategory("model/patch"),
        "optical_flow": ModelFileCategory("model/optical-flow"),
        "photomaker": ModelFileCategory("model/photomaker"),
        "style_models": ModelFileCategory("model/style"),
        "text_encoders": ModelFileCategory("model/text-encoder"),
        "upscale_models": ModelFileCategory("model/upscaler"),
        "vae": ModelFileCategory("model/vae"),
    }
)


class ModelFileSelector(NamedTuple):
    """One exact first-party file selector approved for asset promotion."""

    module: str
    class_name: str
    input_id: str
    category: str
    provenance: str = "identity"


# Pinned ComfyUI 947c2749 inventory: exactly 38 translated selectors.
# A custom or future first-party node does not become an asset merely by
# asking for an approved category; it needs an explicit row here.
MODEL_FILE_SELECTORS: tuple[ModelFileSelector, ...] = (
    ModelFileSelector("nodes", "CheckpointLoader", "ckpt_name", "checkpoints"),
    ModelFileSelector("nodes", "unCLIPCheckpointLoader", "ckpt_name", "checkpoints"),
    ModelFileSelector("nodes", "ControlNetLoader", "control_net_name", "controlnet"),
    ModelFileSelector("nodes", "DiffControlNetLoader", "control_net_name", "controlnet"),
    ModelFileSelector("nodes", "DualCLIPLoader", "clip_name1", "text_encoders"),
    ModelFileSelector("nodes", "DualCLIPLoader", "clip_name2", "text_encoders"),
    ModelFileSelector("nodes", "CLIPVisionLoader", "clip_name", "clip_vision"),
    ModelFileSelector("nodes", "StyleModelLoader", "style_model_name", "style_models"),
    ModelFileSelector("nodes", "GLIGENLoader", "gligen_name", "gligen"),
    ModelFileSelector(
        "comfy_extras.nodes_audio_encoder",
        "AudioEncoderLoader",
        "audio_encoder_name",
        "audio_encoders",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_bg_removal",
        "LoadBackgroundRemovalModel",
        "bg_removal_name",
        "background_removal",
        "stable-sorted-source",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_depth_anything_3",
        "LoadDA3Model",
        "model_name",
        "geometry_estimation",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_frame_interpolation",
        "FrameInterpolationModelLoader",
        "model_name",
        "frame_interpolation",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_hidream",
        "QuadrupleCLIPLoader",
        "clip_name1",
        "text_encoders",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_hidream",
        "QuadrupleCLIPLoader",
        "clip_name2",
        "text_encoders",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_hidream",
        "QuadrupleCLIPLoader",
        "clip_name3",
        "text_encoders",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_hidream",
        "QuadrupleCLIPLoader",
        "clip_name4",
        "text_encoders",
    ),
    ModelFileSelector("comfy_extras.nodes_hooks", "CreateHookLora", "lora_name", "loras"),
    ModelFileSelector(
        "comfy_extras.nodes_hooks",
        "CreateHookLoraModelOnly",
        "lora_name",
        "loras",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_hooks",
        "CreateHookModelAsLora",
        "ckpt_name",
        "checkpoints",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_hooks",
        "CreateHookModelAsLoraModelOnly",
        "ckpt_name",
        "checkpoints",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_hunyuan",
        "LatentUpscaleModelLoader",
        "model_name",
        "latent_upscale_models",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_hypernetwork",
        "HypernetworkLoader",
        "hypernetwork_name",
        "hypernetworks",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_lora_debug",
        "LoraLoaderBypass",
        "lora_name",
        "loras",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_lora_debug",
        "LoraLoaderBypassModelOnly",
        "lora_name",
        "loras",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_lt_audio",
        "LTXVAudioVAELoader",
        "ckpt_name",
        "checkpoints",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_lt_audio",
        "LTXAVTextEncoderLoader",
        "text_encoder",
        "text_encoders",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_lt_audio",
        "LTXAVTextEncoderLoader",
        "ckpt_name",
        "checkpoints",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_mediapipe",
        "LoadMediaPipeFaceLandmarker",
        "model_name",
        "detection",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_model_patch",
        "ModelPatchLoader",
        "name",
        "model_patches",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_moge",
        "LoadMoGeModel",
        "model_name",
        "geometry_estimation",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_photomaker",
        "PhotoMakerLoader",
        "photomaker_model_name",
        "photomaker",
    ),
    ModelFileSelector("comfy_extras.nodes_sd3", "TripleCLIPLoader", "clip_name1", "text_encoders"),
    ModelFileSelector("comfy_extras.nodes_sd3", "TripleCLIPLoader", "clip_name2", "text_encoders"),
    ModelFileSelector("comfy_extras.nodes_sd3", "TripleCLIPLoader", "clip_name3", "text_encoders"),
    ModelFileSelector(
        "comfy_extras.nodes_upscale_model",
        "UpscaleModelLoader",
        "model_name",
        "upscale_models",
    ),
    ModelFileSelector(
        "comfy_extras.nodes_video_model",
        "ImageOnlyCheckpointLoader",
        "ckpt_name",
        "checkpoints",
    ),
    ModelFileSelector("comfy_extras.nodes_void", "OpticalFlowLoader", "model_name", "optical_flow"),
)

_MODEL_FILE_SELECTOR_INDEX = MappingProxyType(
    {
        (selector.module, selector.class_name, selector.input_id): selector
        for selector in MODEL_FILE_SELECTORS
    }
)
_MODEL_FILE_SELECTOR_PATH_INDEX = MappingProxyType(
    {
        key: tuple(
            selector
            for selector in MODEL_FILE_SELECTORS
            if selector.module.startswith("comfy_extras.")
            and (selector.class_name, selector.input_id) == key
        )
        for key in {
            (selector.class_name, selector.input_id)
            for selector in MODEL_FILE_SELECTORS
            if selector.module.startswith("comfy_extras.")
        }
    }
)

_BACKGROUND_REMOVAL_SCHEMA_SHA256 = (
    "3541f97340ecf59930d0c52fcd1be2b0363a1d94a6217f7562677455f3a5aad4"
)


class ListingObservation(NamedTuple):
    """One folder_paths.get_filename_list call observed during a node's
    INPUT_TYPES(): the category asked for, the values it returned, and
    the very object it returned (``source``). Identity of ``source`` is
    the provenance test - a static enum or copied/sorted list that merely
    EQUALS a listing is never mistaken for one."""

    category: str
    values: tuple[str, ...]
    source: object


#: Wraps one ``INPUT_TYPES()`` invocation and reports which filesystem
#: listings fed it: returns the raw result plus every
#: get_filename_list call observed during it. Injected by the
#: ComfyUI-aware bootstrap (this module stays importable without
#: ComfyUI); ``None`` translates statically, exactly as before the
#: probe existed.
InputTypesProbe = Callable[[Callable[[], object]], tuple[object, Sequence[ListingObservation]]]


def listing_choice_id(category: str) -> str | None:
    """The /api/choices id for a folder_paths category, or None when the
    category name does not survive the choice-id grammar (an exotic
    custom category; its combos then simply stay frozen)."""
    choice_id = f"{LISTING_CHOICE_PREFIX}{category}"
    return choice_id if validate_name(choice_id) is None else None


def _observed_listing(
    v1_type: object, observations: Sequence[ListingObservation]
) -> list[ListingObservation]:
    """Every observation whose returned object IS this input's choice
    list. Identity, never equality: a static enum that coincidentally
    equals a listing, or a ``sorted(...)`` copy of one, is not the
    listing itself and must stay frozen."""
    return [obs for obs in observations if obs.source is v1_type]


def _model_file_selector(v1_type: type, input_id: str) -> ModelFileSelector | None:
    module_name = getattr(v1_type, "__module__", None)
    if not isinstance(module_name, str):
        return None
    exact = _MODEL_FILE_SELECTOR_INDEX.get((module_name, v1_type.__name__, input_id))
    if exact is not None:
        return exact

    # ComfyUI loads its built-in comfy_extras through load_custom_node(),
    # whose module name is the absolute source path without '.py'. Admit
    # that spelling only when every independent identity names the exact
    # frozen built-in source. This is deliberately not basename matching:
    # custom nodes and another comfy_extras tree remain ordinary combos.
    candidates = _MODEL_FILE_SELECTOR_PATH_INDEX.get((v1_type.__name__, input_id), ())
    if len(candidates) != 1:
        return None
    selector = candidates[0]
    module_path = Path(module_name)
    if (
        not module_path.is_absolute()
        or module_path.suffix
        or str(module_path) != module_name
        or os.path.abspath(module_name) != module_name
        or any(part in {".", ".."} for part in module_name.split(os.sep))
    ):
        return None
    root_text = os.environ.get("DINKSTER_COMFYUI_ROOT", "")
    if not root_text:
        return None
    try:
        configured_root = Path(os.path.abspath(root_text))
        root = configured_root.resolve(strict=True)
        if not (root / "folder_paths.py").is_file() or not (root / "nodes.py").is_file():
            return None
        extras_root = (root / "comfy_extras").resolve(strict=True)
        extras_root.relative_to(root)
        if not extras_root.is_dir():
            return None
        expected_spelling = root.joinpath(*selector.module.split(".")).with_suffix(".py")
        candidate_source = module_path.parent / f"{module_path.name}.py"
        if candidate_source != expected_spelling:
            return None
        expected_source = expected_spelling.resolve(strict=True)
        expected_source.relative_to(extras_root)
        source_module = sys.modules.get(module_name)
        if source_module is None or getattr(source_module, v1_type.__name__, None) is not v1_type:
            return None
        class_source_name = inspect.getsourcefile(v1_type)
        if class_source_name is None:
            return None
        class_source = Path(class_source_name)
        if class_source.suffix != ".py" or class_source.resolve(strict=True) != expected_source:
            return None
        if expected_source.suffix != ".py" or not expected_source.is_file():
            return None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return selector


def _stable_sorted_listing(
    v1_type: type,
    input_id: str,
    combo_source: object,
    observations: Sequence[ListingObservation],
) -> list[ListingObservation]:
    """Recognize the sole approved copied-list transform.

    Source identity remains the normal rule. This exception first proves
    the exact pinned method bytes and AST shape, then checks that the combo
    is the stable sort of that observed call. Equal values alone never
    authorize promotion.
    """
    method = getattr(v1_type, "define_schema", None)
    if not callable(method):
        return []
    try:
        source = textwrap.dedent(inspect.getsource(method))
    except (OSError, TypeError):
        return []
    if hashlib.sha256(source.encode()).hexdigest() != _BACKGROUND_REMOVAL_SCHEMA_SHA256:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    assignments: dict[str, str] = {}
    sorted_inputs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "folder_paths"
            and node.value.func.attr == "get_filename_list"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)
        ):
            assignments[node.targets[0].id] = node.value.args[0].value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Input"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == input_id
        ):
            options = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "options"),
                None,
            )
            if (
                isinstance(options, ast.Call)
                and isinstance(options.func, ast.Name)
                and options.func.id == "sorted"
                and len(options.args) == 1
                and isinstance(options.args[0], ast.Name)
                and options.args[0].id in assignments
            ):
                sorted_inputs.add((input_id, assignments[options.args[0].id]))
    if sorted_inputs != {(input_id, "background_removal")}:
        return []
    if not isinstance(combo_source, (list, tuple)):
        return []
    return [
        ListingObservation(
            observation.category,
            tuple(cast("Sequence[str]", combo_source)),
            observation.source,
        )
        for observation in observations
        if observation.category == "background_removal"
        and tuple(cast("Sequence[str]", combo_source)) == tuple(sorted(observation.values))
    ]


def _asset_to_model_filename(value: object, category: str) -> str:
    """Convert one digest-backed asset to the exact relative name v1 expects.

    ComfyUI loader functions do not accept arbitrary paths: they accept a
    relative filename and resolve it again through ``folder_paths``. The
    conversion therefore proves all of the following before returning a
    name: the category is allowlisted, the verified asset path is under
    exactly one configured category root, and ComfyUI's own lookup of the
    resulting name resolves back to the same file. Assets in a vault,
    another model category, a shadowed later root, or overlapping roots fail
    closed instead of making v1 open bytes other than the selected digest."""
    if not isinstance(value, AssetRef):
        raise AssetError(
            f"translated model input for {category!r} requires an AssetRef, "
            f"got {type(value).__name__}"
        )
    if category not in MODEL_FILE_CATEGORIES:
        raise AssetError(f"unsupported translated model-file category: {category!r}")

    # Child-process-only import: translate.py itself remains importable
    # without ComfyUI, exactly like the generated node body it serves.
    import importlib

    folder_paths = importlib.import_module("folder_paths")
    get_roots = getattr(folder_paths, "get_folder_paths", None)
    get_full_path = getattr(folder_paths, "get_full_path", None)
    if not callable(get_roots) or not callable(get_full_path):
        raise AssetError(
            "ComfyUI folder_paths lacks the category lookup APIs required "
            "for exact asset conversion"
        )
    raw_roots: object = get_roots(category)
    if not isinstance(raw_roots, (list, tuple)):
        raise AssetError(f"folder_paths returned invalid roots for category {category!r}")
    root_items = cast("Sequence[object]", raw_roots)
    if not all(isinstance(root, str) and root for root in root_items):
        raise AssetError(f"folder_paths returned invalid roots for category {category!r}")

    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if snapshot:
        kind = MODEL_FILE_CATEGORIES[category].kind
        resolution = MountSnapshotResolver(snapshot).resolve_asset_for_kind(value.digest, kind)
        if resolution is None:
            raise AssetError(f"asset {value.digest} is not materializable from a {kind!r} mount")
        asset_path = verified_local_path(
            resolution.path, value.digest, resolution.verification
        ).resolve()
    else:
        # Direct pack use and unit tests may bind a resolver without the
        # serving mount snapshot. The category-root checks below remain the
        # authority in that mode.
        asset_path = value.local_path().resolve()
    roots = tuple(dict.fromkeys(Path(root).resolve() for root in cast("Sequence[str]", raw_roots)))
    matches = [root for root in roots if asset_path.is_relative_to(root)]
    if not matches:
        raise AssetError(f"asset {value.digest} is not under a configured {category!r} model root")
    if len(matches) != 1:
        raise AssetError(f"asset {value.digest} is ambiguous across {category!r} model roots")
    relative = asset_path.relative_to(matches[0]).as_posix()
    resolved = get_full_path(category, relative)
    if not isinstance(resolved, str) or Path(resolved).resolve() != asset_path:
        raise AssetError(
            f"asset {value.digest} is shadowed or unresolved as {category!r}/{relative}"
        )
    return relative


def _remote_listing_combo(
    widget: ComboWidget,
    matched: Sequence[ListingObservation],
    committed: Mapping[str, tuple[str, ...]],
    pending: dict[str, tuple[str, ...]],
) -> ComboWidget:
    """Attach the per-category remote route when the combo's vocabulary
    is verbatim one recorded filesystem listing.

    The static options stay as the baked snapshot - the same shape native
    nodes ship (a successful /api/choices fetch replaces them). The
    combo stays frozen instead of guessing whenever the story is not
    airtight: values mutated since the listing call (an in-place
    ``insert(0, "None")``), the same object recorded under two
    categories (a custom cache), an ungrammatical category name, or a
    snapshot already staged/committed under this id with DIFFERENT
    values (the first successfully translated observer is canonical;
    the provider and every routed widget must tell one story)."""
    options = tuple(option for option in widget.options if isinstance(option, str))
    if len(options) != len(widget.options):
        return widget
    categories = {obs.category for obs in matched if obs.values == options}
    if len(categories) != 1:
        return widget
    choice_id = listing_choice_id(next(iter(categories)))
    if choice_id is None:
        return widget
    known = pending.get(choice_id, committed.get(choice_id))
    if known is not None and known != options:
        return widget
    pending[choice_id] = options
    return ComboWidget(
        options=options,
        remote_route=f"/api/choices/{choice_id}",
        refresh_button=True,
        control_after_generate=widget.control_after_generate,
    )


#: ComfyUI primitive type names -> core Dinkster type ids. Shared verbatim
#: by the v1 translator here and the V3 translator (translate_v3.py):
#: both APIs spell primitives with the same strings.
PRIMITIVES: dict[str, str] = {
    "INT": CORE_INT,
    "FLOAT": CORE_FLOAT,
    "STRING": CORE_STRING,
    "BOOLEAN": CORE_BOOLEAN,
}


class GateRef(NamedTuple):
    code: str
    input_path: tuple[str, ...] = ()
    path_kind: Literal["declared", "dynamic-family"] = "declared"
    lazy: bool | None = None
    raw_link: bool | None = None


class CompatError(Exception):
    """A source class does not have a safely translatable shape."""

    def __init__(self, reason: str, *, gate: GateRef | None = None) -> None:
        self.gate = gate
        super().__init__(reason)


MULTI_STREAM_ROLES_KEY = "dinkster.multi_stream_roles@1"
_MULTI_STREAM_INPUTS: Mapping[str, frozenset[str]] = {
    "KSampler": frozenset({"latent_image"}),
    "LTXVConcatAVLatent": frozenset({"video_latent", "audio_latent"}),
    "LTXVSeparateAVLatent": frozenset({"av_latent"}),
}
_FIXED_AV_PRODUCERS = frozenset(
    {
        "EmptyMiniMaxH3LatentAV",
        "LTXVConcatAVLatent",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
    }
)


def _contains_multistream(value: object) -> bool:
    if type(value).__name__ == "MultiStreamLatent" and hasattr(value, "streams"):
        return True
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value).values()
        return any(_contains_multistream(item) for item in items)
    if isinstance(value, (list, tuple)):
        return any(_contains_multistream(item) for item in cast("Sequence[object]", value))
    return False


def to_comfy_multistream(value: object) -> object:
    if isinstance(value, list):
        return [to_comfy_multistream(item) for item in cast("list[object]", value)]
    if isinstance(value, tuple):
        return tuple(to_comfy_multistream(item) for item in cast("tuple[object, ...]", value))
    if not isinstance(value, Mapping):
        return value
    latent = cast("Mapping[object, object]", value)
    samples = latent.get("samples")
    if type(samples).__name__ != "MultiStreamLatent" or not hasattr(samples, "streams"):
        return cast("object", value)
    streams = cast("Any", samples)
    roles = tuple(streams.roles)
    if not roles or any(type(role) is not str or not role for role in roles):
        raise CompatError("multi-stream LATENT roles must be nonempty strings")
    nested_type = importlib.import_module("comfy.nested_tensor").NestedTensor
    output = dict(latent)
    output["samples"] = nested_type(tuple(stream.payload for stream in streams.streams))
    mask = output.get("noise_mask")
    if type(mask).__name__ == "MultiStreamLatent" and hasattr(mask, "roles"):
        structural_mask = cast("Any", mask)
        if tuple(structural_mask.roles) != roles:
            raise CompatError("multi-stream LATENT mask roles must match sample roles")
        output["noise_mask"] = nested_type(
            tuple(stream.payload for stream in structural_mask.streams)
        )
    output[MULTI_STREAM_ROLES_KEY] = {"version": 1, "roles": roles}
    return output


def from_comfy_multistream(value: object) -> object:
    if isinstance(value, list):
        return [from_comfy_multistream(item) for item in cast("list[object]", value)]
    if isinstance(value, tuple):
        return tuple(from_comfy_multistream(item) for item in cast("tuple[object, ...]", value))
    if not isinstance(value, Mapping):
        return value
    latent = cast("Mapping[object, object]", value)
    samples = latent.get("samples")
    if type(samples).__name__ != "NestedTensor":
        return cast("object", value)
    nested_type = importlib.import_module("comfy.nested_tensor").NestedTensor
    if type(samples) is not nested_type:
        return cast("object", value)
    sidecar = latent.get(MULTI_STREAM_ROLES_KEY)
    if not isinstance(sidecar, Mapping):
        raise CompatError("NestedTensor LATENT output lost its multi-stream role sidecar")
    sidecar_map = cast("Mapping[object, object]", sidecar)
    roles = sidecar_map.get("roles")
    if sidecar_map.get("version") != 1 or not isinstance(roles, (list, tuple)):
        raise CompatError("NestedTensor LATENT output has an invalid role sidecar")
    role_tuple = tuple(cast("Sequence[object]", roles))
    payloads = tuple(cast("Any", samples).unbind())
    if len(role_tuple) != len(payloads) or any(
        type(role) is not str or not role for role in role_tuple
    ):
        raise CompatError("NestedTensor LATENT output role count does not match its streams")
    multi_stream = importlib.import_module("dinkster_inference").MultiStreamLatent
    output = dict(latent)
    output["samples"] = multi_stream.from_pairs(zip(role_tuple, payloads, strict=True))
    mask = output.get("noise_mask")
    if type(mask) is nested_type:
        mask_payloads = tuple(cast("Any", mask).unbind())
        if len(mask_payloads) != len(role_tuple):
            raise CompatError("NestedTensor LATENT mask count does not match its streams")
        output["noise_mask"] = multi_stream.from_pairs(zip(role_tuple, mask_payloads, strict=True))
    output.pop(MULTI_STREAM_ROLES_KEY, None)
    return output


def _declare_fixed_av_output(v1_name: str, value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    output = dict(cast("Mapping[object, object]", value))
    if v1_name in _FIXED_AV_PRODUCERS:
        output[MULTI_STREAM_ROLES_KEY] = {
            "version": 1,
            "roles": ("video", "audio"),
        }
    elif v1_name == "LTXVSeparateAVLatent":
        output.pop(MULTI_STREAM_ROLES_KEY, None)
    return output


_SOURCE_UPLOAD_KINDS = {
    "image_upload": "media/image",
    "audio_upload": "media/audio",
    "video_upload": "media/video",
}
_SOURCE_UPLOAD_ACCEPT = {
    "media/image": ("image/png", "image/jpeg", "image/webp"),
    "media/audio": (
        "audio/wav",
        "audio/flac",
        "audio/mpeg",
        "audio/ogg",
        "audio/webm",
        "audio/mp4",
    ),
    "media/video": ("video/mp4", "video/webm"),
}


def _source_filename_config(
    node_id: str,
    input_id: str,
    config: Mapping[str, object] | None,
    *,
    input_is_list: bool,
) -> tuple[SourceFilenameSpec, bool] | None:
    if config is None:
        return None
    known_upload_fields = {*_SOURCE_UPLOAD_KINDS, "file_upload", "mesh_upload", "animated_upload"}
    unknown_upload_fields = sorted(
        field for field in config if field.endswith("_upload") and field not in known_upload_fields
    )
    if unknown_upload_fields:
        raise CompatError(
            f"{node_id}: input {input_id!r} has unsupported source upload declarations "
            f"{unknown_upload_fields!r}"
        )
    enabled: list[str] = []
    for field in (*_SOURCE_UPLOAD_KINDS, "file_upload"):
        if field not in config:
            continue
        value = config[field]
        if type(value) is not bool:
            raise CompatError(f"{node_id}: input {input_id!r} {field} must be a Boolean")
        if value:
            if field == "file_upload":
                raise CompatError(
                    f"{node_id}: input {input_id!r} file_upload/model sources are not supported"
                )
            enabled.append(field)
    for field in ("mesh_upload", "animated_upload", "animated", "multiple"):
        if field not in config:
            continue
        value = config[field]
        if type(value) is not bool:
            raise CompatError(f"{node_id}: input {input_id!r} {field} must be a Boolean")
        if value:
            raise CompatError(f"{node_id}: input {input_id!r} {field} sources are not supported")
    if len(enabled) > 1:
        raise CompatError(
            f"{node_id}: input {input_id!r} has contradictory source upload declarations"
        )
    raw_category = config.get("image_folder")
    if raw_category is not None and (
        type(raw_category) is not str or raw_category not in {"input", "output", "temp"}
    ):
        raise CompatError(
            f"{node_id}: input {input_id!r} image_folder must be input, output, or temp"
        )
    if not enabled:
        if raw_category is not None:
            raise CompatError(
                f"{node_id}: input {input_id!r} image_folder has no source upload declaration"
            )
        return None
    for field in ("lazy", "rawLink", "control_after_generate"):
        value = config.get(field)
        if value is not None and value is not False:
            raise CompatError(
                f"{node_id}: input {input_id!r} source upload does not support {field}"
            )
    remote = config.get("remote")
    if remote is not None and remote is not False:
        if not isinstance(remote, Mapping):
            raise CompatError(f"{node_id}: input {input_id!r} remote must be an object")
        remote_config = cast("Mapping[object, object]", remote)
        unknown_remote = sorted(
            str(field)
            for field in remote_config
            if field not in {"route", "refresh_button", "control_after_refresh"}
        )
        route = remote_config.get("route")
        refresh_button = remote_config.get("refresh_button")
        control_after_refresh = remote_config.get("control_after_refresh")
        if unknown_remote:
            raise CompatError(
                f"{node_id}: input {input_id!r} remote has unsupported fields {unknown_remote!r}"
            )
        if type(route) is not str or not route.startswith("/"):
            raise CompatError(
                f"{node_id}: input {input_id!r} remote route must be an absolute route"
            )
        if refresh_button is not None and type(refresh_button) is not bool:
            raise CompatError(
                f"{node_id}: input {input_id!r} remote refresh_button must be a Boolean"
            )
        if control_after_refresh is not None and (
            type(control_after_refresh) is not str or not control_after_refresh
        ):
            raise CompatError(
                f"{node_id}: input {input_id!r} remote control_after_refresh must be a string"
            )
    if input_is_list:
        raise CompatError(
            f"{node_id}: input {input_id!r} source upload conflicts with INPUT_IS_LIST"
        )
    raw_multiselect = config.get("multiselect", False)
    if type(raw_multiselect) is not bool:
        raise CompatError(f"{node_id}: input {input_id!r} multiselect must be a Boolean")
    if config.get("default") is not None:
        raise CompatError(
            f"{node_id}: input {input_id!r} source upload cannot declare an ambient default"
        )
    return SourceFilenameSpec(
        cast("Any", _SOURCE_UPLOAD_KINDS[enabled[0]]),
        cast("Any", raw_category or "input"),
    ), raw_multiselect


def source_asset_widget(binding: SourceFilenameSpec) -> AssetWidget:
    return AssetWidget(
        accept=_SOURCE_UPLOAD_ACCEPT[binding.kind],
        kind=binding.kind,
        allow_upload=True,
    )


def _materialize_source_value(
    node_id: str,
    input_id: str,
    value: object,
    binding: SourceFilenameSpec,
    *,
    listed: bool,
) -> str | list[str]:
    context = current_execution_context()
    materialize = context.materialize_source if context is not None else None
    if materialize is None:
        raise CompatError(
            f"{node_id}: input {input_id!r} requires invocation-scoped source staging"
        )

    def one(item: object) -> str:
        if not isinstance(item, AssetRef):
            raise CompatError(
                f"{node_id}: input {input_id!r} requires an AssetRef, got {type(item).__name__}"
            )
        result = materialize(item, binding.kind, binding.category)
        if type(result) is not str or not result:
            raise CompatError(
                f"{node_id}: source staging returned an invalid filename for input {input_id!r}"
            )
        return result

    if listed:
        if type(value) is not list:
            raise CompatError(
                f"{node_id}: input {input_id!r} requires its declared asset list, "
                f"got {type(value).__name__}"
            )
        items = cast("list[object]", value)
        if any(not isinstance(item, AssetRef) for item in items):
            raise CompatError(
                f"{node_id}: input {input_id!r} requires a list containing only AssetRef values"
            )
        return [one(item) for item in items]
    if isinstance(value, list):
        raise CompatError(f"{node_id}: scalar input {input_id!r} does not accept an asset list")
    return one(value)


def _gate_error(
    reason: str,
    code: str,
    *,
    input_path: tuple[str, ...] = (),
    path_kind: Literal["declared", "dynamic-family"] = "declared",
    lazy: bool | None = None,
    raw_link: bool | None = None,
) -> CompatError:
    return CompatError(
        reason,
        gate=GateRef(code, input_path, path_kind, lazy, raw_link),
    )


def _exact_bool(value: object) -> bool | None:
    return value if type(value) is bool else None


def _static_attr(source: type, name: str, default: object) -> object:
    """Read class-declared gate facts without invoking pack descriptors."""
    return inspect.getattr_static(source, name, default)


def _output_list_fact(source: type) -> bool | None:
    declared = _static_attr(source, "OUTPUT_IS_LIST", False)
    if type(declared) is bool:
        return declared
    if not isinstance(declared, (list, tuple)):
        return None
    values = tuple(cast("Sequence[object]", declared))
    if any(type(value) is not bool for value in values):
        return None
    return any(cast("tuple[bool, ...]", values))


def _skip_diagnostic(
    source_node: str,
    source: type,
    reason: str,
    gate: GateRef | None,
) -> CompatGateDiagnostic:
    generation_marker: object = _static_attr(source, "GET_NODE_INFO_V1", None)
    is_v3 = isinstance(generation_marker, (classmethod, staticmethod)) or callable(
        generation_marker
    )
    source_generation: Literal["v1", "v3"] = "v3" if is_v3 else "v1"
    accept_all = _static_attr(source, "_ACCEPT_ALL_INPUTS", source)
    if accept_all is source:
        accept_all = _static_attr(source, "ACCEPT_ALL_INPUTS", False)
    path = gate.input_path if gate is not None else ()
    return CompatGateDiagnostic(
        code=gate.code if gate is not None else "compat.translation.refused",
        source_node=source_node,
        reason=reason,
        source_generation=source_generation,
        path_kind=gate.path_kind if gate is not None else "declared",
        input_id=path[-1] if path else None,
        input_path=path,
        lazy=gate.lazy if gate is not None else None,
        input_is_list=_exact_bool(_static_attr(source, "INPUT_IS_LIST", False)),
        output_is_list=_output_list_fact(source),
        raw_link=gate.raw_link if gate is not None else None,
        accept_all=_exact_bool(accept_all),
    )


def _custom_combo_runtime_shape(
    v1_name: str,
    v1_class: type,
    raw_inputs: Mapping[str, object],
    return_types: object,
    *,
    namespace: str,
    raw_input_is_list: object,
) -> bool:
    """Whether a V3 shim class is the one closed core CustomCombo shape."""
    if (
        type(v1_name) is not str
        or type(namespace) is not str
        or v1_name != CUSTOM_COMBO_NODE_ID
        or namespace
        or raw_input_is_list is not False
    ):
        return False
    info_fn = getattr(v1_class, "GET_NODE_INFO_V1", None)
    if not callable(info_fn):
        return False
    info = info_fn()
    if not isinstance(info, Mapping):
        return False
    info_map = cast("Mapping[str, object]", info)
    info_name = info_map.get("name")
    if type(info_name) is not str or info_name != CUSTOM_COMBO_NODE_ID:
        return False
    rows = iter_v1_inputs(raw_inputs)
    if len(rows) != 1:
        return False
    name, v1_type, config, required = rows[0]
    if (
        type(name) is not str
        or name != "choice"
        or not required
        or not _is_exact_v3_type(v1_type, V3_COMBO_IO_TYPE)
        or config is None
        or type(config.get("options")) is not list
        or len(cast("list[object]", config.get("options"))) != 0
        or config.get("multiselect") is not False
        or any(type(key) is not str for key in config)
        or set(config) != {"multiselect", "options"}
    ):
        return False
    if any(type(key) is not str for key in raw_inputs) or set(raw_inputs) != {"required"}:
        return False
    if not isinstance(return_types, (list, tuple)):
        return False
    return_type_values = tuple(cast("Sequence[object]", return_types))
    if any(type(value) is not str for value in return_type_values) or return_type_values != (
        "STRING",
        "INT",
    ):
        return False
    return_names = getattr(v1_class, "RETURN_NAMES", None)
    if not isinstance(return_names, (list, tuple)):
        return False
    return_name_values = tuple(cast("Sequence[object]", return_names))
    if any(type(value) is not str for value in return_name_values) or return_name_values != (
        "STRING",
        "INDEX",
    ):
        return False
    output_is_list = getattr(v1_class, "OUTPUT_IS_LIST", None)
    if not isinstance(output_is_list, (list, tuple)):
        return False
    output_list_flags = cast("Sequence[object]", output_is_list)
    if len(output_list_flags) != 2 or any(value is not False for value in output_list_flags):
        return False
    function_name = getattr(v1_class, "FUNCTION", None)
    if type(function_name) is not str or function_name != "EXECUTE_NORMALIZED":
        return False
    execute = getattr(v1_class, "execute", None)
    if not callable(execute):
        return False
    parameters = tuple(inspect.signature(execute).parameters.values())
    return (
        len(parameters) == 3
        and parameters[0].name == "choice"
        and parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameters[0].default is inspect.Parameter.empty
        and parameters[1].name == "index"
        and parameters[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and type(parameters[1].default) is int
        and parameters[1].default == 0
        and parameters[2].kind is inspect.Parameter.VAR_KEYWORD
    )


def _closed_custom_combo_options(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CompatError("CustomCombo: input family 'options' expected a mapping")
    options = dict(cast("Mapping[str, object]", value))
    if not options:
        return {}
    expected = CUSTOM_COMBO_OPTION_NAMES[: len(options)]
    if set(options) != set(expected):
        raise CompatError("CustomCombo: options must be the contiguous family option1..optionN")
    if not all(isinstance(option, str) for option in options.values()):
        raise CompatError("CustomCombo: every option family value must be a string")
    return {name: cast(str, options[name]) for name in expected}


def comfy_type_id(v1_type: str) -> str:
    return COMFY_TYPE_PREFIX + v1_type


def _is_v3_marker(v1_type: object) -> bool:
    return type(v1_type) is str and v1_type.startswith("COMFY_") and v1_type.endswith("_V3")


def _is_exact_v3_type(v1_type: object, expected: str) -> bool:
    """Marker equality that cannot trigger AnyType's match-everything hack."""
    return type(v1_type) is str and v1_type == expected


def selector_lazy_inputs(
    node_id: str,
    source: object,
    inputs: object,
    outputs: object,
    *,
    input_is_list: bool,
) -> frozenset[str]:
    """Return the verified lazy branch ids for the one supported switch shape."""
    if node_id != "ComfySwitchNode" or input_is_list:
        return frozenset()
    output_seq = cast("Sequence[object]", outputs) if isinstance(outputs, Sequence) else ()
    if isinstance(inputs, Mapping):
        entries = list(iter_v1_inputs(cast("Mapping[str, object]", inputs)))
        by_id = {name: (value, config, required) for name, value, config, required in entries}
        if set(by_id) != {"switch", "on_false", "on_true"} or len(entries) != 3:
            return frozenset()
        switch_type, switch_config, switch_required = by_id["switch"]
        branches = (by_id["on_false"], by_id["on_true"])
        output_flags = getattr(source, "OUTPUT_IS_LIST", False)
        if (
            switch_type != "BOOLEAN"
            or not switch_required
            or (switch_config is not None and switch_config.get("lazy"))
            or len(output_seq) != 1
            or not _is_exact_v3_type(output_seq[0], V3_MATCHTYPE_IO_TYPE)
            or output_flags not in (False, None, [False], (False,))
        ):
            return frozenset()
        expressions: list[TypeExpr] = []
        for branch_type, config, required in branches:
            if (
                not required
                or not _is_exact_v3_type(branch_type, V3_MATCHTYPE_IO_TYPE)
                or config is None
                or not config.get("lazy")
            ):
                return frozenset()
            expression, _, _ = _matchtype_expr(config.get("template"))
            expressions.append(expression)
        schema = getattr(source, "SCHEMA", None)
        schema_outputs = getattr(schema, "outputs", ())
        if len(schema_outputs) != 1:
            return frozenset()
        template = getattr(schema_outputs[0], "template", None)
        output_template_id = getattr(template, "template_id", None)
        if not output_template_id or any(
            expr.template_id != output_template_id for expr in expressions
        ):
            return frozenset()
        return frozenset({"on_false", "on_true"})

    input_seq = cast("Sequence[object]", inputs) if isinstance(inputs, Sequence) else ()
    by_id = {str(getattr(item, "id", "") or ""): item for item in input_seq}
    if set(by_id) != {"switch", "on_false", "on_true"} or len(input_seq) != 3:
        return frozenset()
    switch = by_id["switch"]
    branches = (by_id["on_false"], by_id["on_true"])

    def io_type(item: object) -> str:
        getter = getattr(item, "get_io_type", None)
        return str(getter()) if callable(getter) else ""

    if (
        io_type(switch) != "BOOLEAN"
        or bool(getattr(switch, "optional", False))
        or bool(getattr(switch, "lazy", False))
        or len(output_seq) != 1
        or any(io_type(branch) != V3_MATCHTYPE_IO_TYPE for branch in branches)
        or any(bool(getattr(branch, "optional", False)) for branch in branches)
        or any(not bool(getattr(branch, "lazy", False)) for branch in branches)
        or io_type(output_seq[0]) != V3_MATCHTYPE_IO_TYPE
        or bool(getattr(output_seq[0], "is_output_list", False))
    ):
        return frozenset()
    template_ids = [
        getattr(getattr(item, "template", None), "template_id", None)
        for item in (*branches, output_seq[0])
    ]
    if not template_ids[0] or len(set(template_ids)) != 1:
        return frozenset()
    return frozenset({"on_false", "on_true"})


def supported_lazy_inputs(
    node_id: str,
    source: object,
    inputs: Mapping[str, object],
    outputs: object,
    *,
    input_is_list: bool,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return supported lazy ids and the subset eligible for switch lowering."""
    entries = iter_v1_inputs(inputs)
    selector_ids = selector_lazy_inputs(
        node_id,
        source,
        inputs,
        outputs,
        input_is_list=input_is_list,
    )
    lazy_ids: set[str] = set()
    for name, v1_type, config, _required in entries:
        if config is None or "lazy" not in config:
            continue
        lazy = config["lazy"]
        if type(lazy) is not bool:
            raise _gate_error(
                f"{node_id}: input {name!r} lazy must be a Boolean",
                "compat.lazy.malformed",
                input_path=(name,),
                raw_link=_exact_bool(config.get("rawLink", False)),
            )
        if not lazy:
            continue
        if (
            _is_v3_marker(v1_type)
            and name not in selector_ids
            and not (input_is_list and _is_exact_v3_type(v1_type, V3_MATCHTYPE_IO_TYPE))
        ):
            raise _gate_error(
                f"{node_id}: V3 input {name!r} uses unsupported lazy semantics",
                "compat.lazy.unsupported",
                input_path=(name,),
                lazy=True,
                raw_link=_exact_bool(config.get("rawLink", False)),
            )
        lazy_ids.add(name)

    if not lazy_ids:
        return frozenset(), frozenset()

    hook = getattr(source, "check_lazy_status", None)
    if not callable(hook):
        raise _gate_error(
            f"{node_id}: lazy inputs require check_lazy_status",
            "compat.lazy.missing-hook",
            lazy=True,
        )

    return frozenset(lazy_ids), selector_ids


def _member_type_id(io_type: str) -> tuple[str, tuple[str, ...]]:
    if _is_v3_marker(io_type):
        raise CompatError(f"unsupported V3 dynamic type marker {io_type}")
    if io_type == V3_COMBO_IO_TYPE:
        return CORE_COMBO, ()
    if io_type == "COLOR":
        return CORE_STRING, ()
    primitive = PRIMITIVES.get(io_type)
    if primitive is not None:
        return primitive, ()
    opaque = comfy_type_id(io_type)
    return opaque, (opaque,)


def translate_type(v1_type: object) -> tuple[TypeExpr, tuple[str, ...]]:
    """v1 type declaration -> (TypeExpr, opaque comfy type ids to register)."""
    if isinstance(v1_type, (list, tuple)):
        # Combo: plain-string payload, distinct socket identity. The choice
        # vocabulary remains presentation and does not refine the type.
        return TypeExpr.concrete(CORE_COMBO), ()
    if not isinstance(v1_type, str):
        raise CompatError(f"unsupported v1 input type declaration: {v1_type!r}")
    # The ecosystem's wildcard hack is a str subclass whose __eq__ always
    # matches (AnyType/AlwaysEqualProxy, usually spelled "*") - and it
    # defines no __hash__, so it explodes in any set/dict lookup. Collapse
    # to the plain string; the wildcard branch below handles "*" honestly.
    if type(v1_type) is not str:
        v1_type = str(v1_type)
    if _is_v3_marker(v1_type):
        raise CompatError(f"unsupported V3 dynamic type marker {v1_type}")
    members = [part.strip() for part in v1_type.split(",") if part.strip()]
    if not members:
        raise CompatError(f"empty v1 type declaration: {v1_type!r}")
    marker = next((member for member in members if _is_v3_marker(member)), None)
    if marker is not None:
        raise CompatError(f"unsupported V3 dynamic type marker {marker}")
    if "*" in members:
        return TypeExpr.wildcard(), ()
    if len(members) == 1:
        type_id, single_opaque = _member_type_id(members[0])
        return TypeExpr.concrete(type_id), single_opaque
    seen: dict[str, None] = {}
    opaque_types: list[str] = []
    for member in members:
        type_id, member_opaque = _member_type_id(member)
        opaque_types.extend(member_opaque)
        seen.setdefault(type_id, None)
    ids = list(seen)
    if len(ids) == 1:
        return TypeExpr.concrete(ids[0]), tuple(opaque_types)
    return TypeExpr.union(*ids), tuple(opaque_types)


#: Case-insensitive truthy option word -> its falsy partner. A two-option
#: combo matching one of these pairs is a disguised boolean.
_BOOLEAN_COMBO_PAIRS: dict[str, str] = {
    "enable": "disable",
    "on": "off",
    "true": "false",
    "yes": "no",
}


def boolean_combo(v1_type: object) -> tuple[str, str] | None:
    """(on_option, off_option) in original spelling when the v1 combo is
    a disguised boolean - exactly two string choices matching one of the
    recognized truthy/falsy word pairs, either order - else None.

    These translate to core.boolean with the original strings as toggle
    labels; execute() maps the boolean back to the v1 string, so the
    wrapped function never learns the schema got honest."""
    if not isinstance(v1_type, (list, tuple)):
        return None
    entries = cast("Sequence[object]", v1_type)
    if len(entries) != 2 or not all(isinstance(entry, str) for entry in entries):
        return None
    first, second = cast("Sequence[str]", entries)
    for on_word, off_word in _BOOLEAN_COMBO_PAIRS.items():
        if first.lower() == on_word and second.lower() == off_word:
            return first, second
        if first.lower() == off_word and second.lower() == on_word:
            return second, first
    return None


def _restore_boolean_vocabulary(
    value: object,
    pair: tuple[str, str] | None,
) -> object:
    if pair is None:
        return value
    if isinstance(value, bool):
        return pair[0] if value else pair[1]
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [
            (pair[0] if item else pair[1]) if isinstance(item, bool) else item for item in items
        ]
    return value


def _type_occupies_gpu(type_expr: TypeExpr) -> bool:
    if any(
        type_id.removeprefix(COMFY_TYPE_PREFIX) in DEFAULT_RESIDENT_V1_TYPES
        for type_id in type_expr.types
    ):
        return True
    return type_expr.element is not None and _type_occupies_gpu(type_expr.element)


def dynamic_entries_occupy_gpu(entries: Sequence[DynamicEntry]) -> bool:
    for entry in entries:
        if isinstance(entry, InputSpec):
            if _type_occupies_gpu(entry.type):
                return True
        elif isinstance(entry, InputFamilySpec):
            if dynamic_entries_occupy_gpu(entry.template):
                return True
        elif isinstance(entry, DynamicComboSpec):
            if any(dynamic_entries_occupy_gpu(option.inputs) for option in entry.options):
                return True
        else:
            if entry.slot_type is not None and _type_occupies_gpu(entry.slot_type):
                return True
            if dynamic_entries_occupy_gpu(entry.inputs):
                return True
            if any(
                _type_occupies_gpu(variant.type) or dynamic_entries_occupy_gpu(variant.inputs)
                for variant in entry.variants or ()
            ):
                return True
    return False


def _boolean_labels_widget(
    v1_type: object, config: Mapping[str, object] | None
) -> BooleanWidget | None:
    """v1 BOOLEAN inputs carry custom toggle labels in config
    (label_on/label_off); those survive as a BooleanWidget. A label-less
    boolean carries no widget - the type alone means "render a toggle"."""
    if v1_type != "BOOLEAN" or config is None:
        return None
    label_on = config.get("label_on")
    label_off = config.get("label_off")
    on_text = label_on if isinstance(label_on, str) else ""
    off_text = label_off if isinstance(label_off, str) else ""
    if not on_text and not off_text:
        return None
    return BooleanWidget(label_on=on_text, label_off=off_text)


def number_widget(
    name: str,
    v1_type: object,
    config: Mapping[str, object] | None,
) -> NumberWidget | None:
    """Translate only numeric metadata accepted by the public schema contract."""
    if config is None or not isinstance(v1_type, str):
        return None
    legacy_type = str(v1_type)
    if legacy_type not in {"INT", "FLOAT"}:
        return None
    socket = TypeExpr.concrete(CORE_INT if legacy_type == "INT" else CORE_FLOAT)

    accepted: dict[str, int | float] = {}
    fields = ("min", "max", "step", "round") if legacy_type == "FLOAT" else ("min", "max", "step")
    for field in fields:
        if field not in config:
            continue
        raw_value = config[field]
        if type(raw_value) not in (int, float):
            continue
        value = cast("int | float", raw_value)
        if legacy_type == "INT" and abs(value) > 2**53 - 1:
            continue
        try:
            candidate = NumberWidget(**{field: value})  # type: ignore[arg-type]
            InputSpec(name, socket, widget=candidate)
        except (TypeError, ValueError):
            continue
        accepted[field] = value

    minimum = accepted.get("min")
    maximum = accepted.get("max")
    if minimum is not None and maximum is not None and minimum > maximum:
        accepted.pop("min")
        accepted.pop("max")

    control: object = config.get("control_after_generate")
    if legacy_type != "INT":
        control = None
    elif control is True:
        control = "randomize"
    elif type(control) is not str:
        control = None
    if control is not None:
        try:
            candidate = NumberWidget(control_after_generate=control)  # type: ignore[arg-type]
            InputSpec(name, socket, widget=candidate)
        except (TypeError, ValueError):
            control = None

    display: object = config.get("display")
    if type(display) is not str:
        display = None
    if display is not None:
        try:
            NumberWidget(display=display)  # type: ignore[arg-type]
        except ValueError:
            display = None

    if not accepted and control is None and display is None:
        return None
    return NumberWidget(
        min=accepted.get("min"),
        max=accepted.get("max"),
        step=accepted.get("step"),
        round=accepted.get("round"),
        control_after_generate=control,  # type: ignore[arg-type]
        display=display,  # type: ignore[arg-type]
    )


def combo_widget(v1_type: object, config: Mapping[str, object] | None = None) -> ComboWidget | None:
    """The v1 choice list as a static ComboWidget, when it can be one.

    Only a list whose every entry is a non-empty string qualifies; packs
    in the wild declare combos of ints/floats and empty folder listings,
    and those render as plain typed fields rather than lying about their
    vocabulary. Choices stay verbatim - order and duplicates are the
    pack author's UI, not ours to clean."""
    if not isinstance(v1_type, (list, tuple)) or not v1_type:
        return None
    entries = cast("Sequence[object]", v1_type)
    if not all(isinstance(entry, str) and entry for entry in entries):
        return None
    control: ControlAfterGenerate | None = None
    if config is not None:
        declared = config.get("control_after_generate")
        if declared is True:
            control = "randomize"
        elif type(declared) is str and declared in {
            "fixed",
            "increment",
            "decrement",
            "randomize",
        }:
            control = cast("ControlAfterGenerate", declared)
    return ComboWidget(
        options=tuple(cast("Sequence[str]", entries)),
        control_after_generate=control,
    )


def multi_combo_widget(
    options: object,
    config: Mapping[str, object] | None,
    *,
    input_is_list: bool,
    trusted_choice_ids: set[str],
) -> MultiComboWidget | None:
    """Translate only an explicit, structurally complete MultiCombo declaration."""
    multiselect = config.get("multiselect") if config is not None else None
    has_multi_select = config is not None and "multi_select" in config
    if multiselect is not True:
        if has_multi_select or (multiselect is not None and multiselect is not False):
            raise CompatError("MultiCombo multiselect flag is missing or malformed")
        return None
    if input_is_list:
        raise CompatError("MultiCombo conflicts with class INPUT_IS_LIST")
    assert config is not None
    multi_select = config.get("multi_select")
    if not isinstance(multi_select, Mapping):
        raise CompatError("MultiCombo multi_select must be an object")
    presentation = cast("Mapping[str, object]", multi_select)
    unknown = sorted(set(presentation) - {"placeholder", "chip"})
    if unknown:
        raise CompatError(f"MultiCombo multi_select has unknown fields: {unknown}")
    placeholder = presentation.get("placeholder")
    chip = presentation.get("chip")
    if placeholder is not None and type(placeholder) is not str:
        raise CompatError("MultiCombo multi_select placeholder must be a string")
    if chip is not None and type(chip) is not bool:
        raise CompatError("MultiCombo multi_select chip must be a bool")
    if not isinstance(options, (list, tuple)) or any(
        type(option) is not str or not option for option in cast("Sequence[object]", options)
    ):
        raise CompatError("MultiCombo options must be non-empty strings")
    default = config.get("default")
    if default is not None and (
        type(default) is not list
        or any(type(value) is not str for value in cast("list[object]", default))
    ):
        raise CompatError("MultiCombo default must be an array of strings")
    if config.get("control_after_generate") is not None:
        raise CompatError("MultiCombo control_after_generate is not supported")
    option_values = tuple(cast("Sequence[str]", options))
    remote = config.get("remote")
    if not option_values and remote is None:
        raise CompatError("V1 MultiCombo requires static options or a remote source")
    widget = MultiComboWidget(
        options=option_values or (("remote",) if remote is not None else ()),
        placeholder=placeholder,
        chip=chip,
    )
    if remote is not None:
        widget = trusted_v3_remote_multicombo(widget, remote, trusted_choice_ids)
        if not option_values:
            widget = MultiComboWidget(
                remote_route=widget.remote_route,
                refresh_button=widget.refresh_button,
                control_after_refresh=widget.control_after_refresh,
                remote_timeout_ms=widget.remote_timeout_ms,
                remote_max_retries=widget.remote_max_retries,
                remote_refresh_ms=widget.remote_refresh_ms,
                placeholder=widget.placeholder,
                chip=widget.chip,
            )
    return widget


def trusted_v3_remote_combo(
    widget: ComboWidget,
    remote: object,
    trusted_choice_ids: set[str],
) -> ComboWidget:
    """Translate V3 RemoteOptions only through an already registered Dinkster
    choice id. The upstream route is never copied as a browser URL."""
    if isinstance(remote, Mapping):
        declaration = cast("Mapping[str, object]", remote)

        def get_remote_field(field_name: str) -> object:
            return declaration.get(field_name)

    else:

        def get_remote_field(field_name: str) -> object:
            return getattr(remote, field_name, None)

    route = get_remote_field("route")
    prefix = "/api/choices/"
    choice_id = route.removeprefix(prefix) if type(route) is str else ""
    if (
        type(route) is not str
        or route != f"{prefix}{choice_id}"
        or validate_name(choice_id) is not None
        or choice_id not in trusted_choice_ids
    ):
        raise CompatError(
            "V3 RemoteOptions.route cannot resolve to a trusted registered Dinkster choice id"
        )
    refresh_button = get_remote_field("refresh_button")
    if type(refresh_button) is not bool:
        raise CompatError("V3 RemoteOptions.refresh_button must be a bool")
    control_after_refresh = get_remote_field("control_after_refresh")
    timeout = get_remote_field("timeout")
    max_retries = get_remote_field("max_retries")
    refresh = get_remote_field("refresh")
    try:
        return ComboWidget(
            options=widget.options,
            remote_route=route,
            refresh_button=refresh_button,
            control_after_generate=widget.control_after_generate,
            control_after_refresh=cast("Literal['first', 'last'] | None", control_after_refresh),
            remote_timeout_ms=cast("int | None", timeout),
            remote_max_retries=cast("int | None", max_retries),
            remote_refresh_ms=cast("int | None", refresh),
        )
    except ValueError as exc:
        raise CompatError(f"V3 RemoteOptions {exc}") from exc


def trusted_v3_remote_multicombo(
    widget: MultiComboWidget,
    remote: object,
    trusted_choice_ids: set[str],
) -> MultiComboWidget:
    """Apply the settled trusted remote policy without scalar controller state."""
    scalar = trusted_v3_remote_combo(
        ComboWidget(options=widget.options or ("remote",)),
        remote,
        trusted_choice_ids,
    )
    return MultiComboWidget(
        options=widget.options,
        remote_route=scalar.remote_route,
        refresh_button=scalar.refresh_button,
        control_after_refresh=scalar.control_after_refresh,
        remote_timeout_ms=scalar.remote_timeout_ms,
        remote_max_retries=scalar.remote_max_retries,
        remote_refresh_ms=scalar.remote_refresh_ms,
        placeholder=widget.placeholder,
        chip=widget.chip,
    )


def string_widget(
    v1_type: object, config: Mapping[str, object] | None
) -> StringWidget | ColorWidget | None:
    if type(v1_type) is not str:
        return None
    if v1_type == "COLOR":
        return ColorWidget()
    if v1_type != "STRING" or config is None:
        return None
    raw_multiline = config.get("multiline")
    raw_placeholder = config.get("placeholder")
    raw_dynamic = config.get("dynamicPrompts")
    multiline = raw_multiline if type(raw_multiline) is bool else None
    placeholder = raw_placeholder if type(raw_placeholder) is str else None
    dynamic = raw_dynamic if type(raw_dynamic) is bool else None
    if multiline is None and placeholder is None and dynamic is None:
        return None
    return StringWidget(
        multiline=multiline,
        placeholder=placeholder,
        dynamic_prompts=dynamic,
    )


def _input_default(config: Mapping[str, object] | None, v1_type: object) -> object:
    if config is not None and "default" in config:
        return config["default"]
    if isinstance(v1_type, (list, tuple)) and v1_type:
        first = cast("Sequence[object]", v1_type)[0]
        if isinstance(first, str):
            return first
    return None


def _v3_combo_options(config: Mapping[str, object] | None) -> object:
    return config.get("options") if config is not None else None


def _combo_token(value: object) -> object:
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise CompatError("numeric combo values must be finite")
        return str(value)
    return value


def _numeric_combo_vocabulary(options: object) -> dict[str, object]:
    if not isinstance(options, (list, tuple)):
        return {}
    choices = cast("Sequence[object]", options)
    if not any(type(choice) in (int, float) for choice in choices):
        return {}
    vocabulary: dict[str, object] = {}
    for choice in choices:
        token = _combo_token(choice)
        if not isinstance(token, str):
            raise CompatError("numeric combo choices must be strings or finite numbers")
        if token in vocabulary and type(vocabulary[token]) is not type(choice):
            raise CompatError(f"numeric combo has ambiguous string key {token!r}")
        vocabulary[token] = choice
    return vocabulary


def _restore_combo_value(value: object, vocabulary: Mapping[str, object]) -> object:
    return vocabulary.get(value, value) if isinstance(value, str) else value


def _matchtype_expr(
    template: object,
) -> tuple[TypeExpr, tuple[str, ...], str]:
    if not isinstance(template, Mapping):
        raise CompatError("V3 MatchType lacks a template mapping")
    template_map = cast("Mapping[str, object]", template)
    template_id = template_map.get("template_id")
    allowed_types = template_map.get("allowed_types")
    if not isinstance(template_id, str) or not template_id:
        raise CompatError("V3 MatchType template lacks a template_id")
    if not isinstance(allowed_types, str):
        raise CompatError(f"V3 MatchType template {template_id!r} lacks allowed_types")
    members = [part.strip() for part in allowed_types.split(",") if part.strip()]
    if not members:
        raise CompatError(f"V3 MatchType template {template_id!r} has empty allowed_types")
    if "*" in members:
        return TypeExpr.variable(template_id), (), template_id
    allowed: dict[str, None] = {}
    opaque_types: list[str] = []
    for member in members:
        if _is_v3_marker(member):
            raise CompatError(
                f"V3 MatchType template {template_id!r} contains nested dynamic marker {member}"
            )
        type_id, member_opaque = _member_type_id(member)
        allowed.setdefault(type_id, None)
        opaque_types.extend(member_opaque)
    return (
        TypeExpr.variable(template_id, tuple(allowed)),
        tuple(opaque_types),
        template_id,
    )


def _output_matchtype_expr(
    v1_name: str,
    v1_class: type,
    output_index: int,
    input_templates: Mapping[str, tuple[TypeExpr, tuple[str, ...]]],
) -> tuple[TypeExpr, tuple[str, ...]]:
    schema = getattr(v1_class, "SCHEMA", None)
    outputs = getattr(schema, "outputs", None)
    if isinstance(outputs, Sequence):
        output_seq = cast("Sequence[object]", outputs)
    else:
        output_seq = ()
    if output_index < len(output_seq):
        template = getattr(output_seq[output_index], "template", None)
        template_id = getattr(template, "template_id", None)
        allowed_decl = getattr(template, "allowed_types", None)
        if isinstance(template_id, str) and template_id and allowed_decl is not None:
            allowed_io_types: list[str] = []
            try:
                for entry in cast("Sequence[object]", allowed_decl):
                    io_type = getattr(entry, "io_type", None)
                    if not isinstance(io_type, str) or not io_type:
                        raise CompatError(
                            f"{v1_name}: V3 MatchType output {output_index} has "
                            "malformed allowed_types"
                        )
                    allowed_io_types.append(io_type)
            except TypeError as exc:
                raise CompatError(
                    f"{v1_name}: V3 MatchType output {output_index} has malformed allowed_types"
                ) from exc
            expr, opaque, _ = _matchtype_expr(
                {
                    "template_id": template_id,
                    "allowed_types": ",".join(allowed_io_types),
                }
            )
            return expr, opaque

    get_info = getattr(v1_class, "GET_NODE_INFO_V1", None)
    if callable(get_info):
        try:
            info = get_info()
        except Exception:  # noqa: BLE001 - defensive recovery from upstream schema
            info = None
        if isinstance(info, Mapping):
            info_map = cast("Mapping[str, object]", info)
            output_matchtypes = info_map.get("output_matchtypes")
            if isinstance(output_matchtypes, Sequence):
                matchtype_seq = cast("Sequence[object]", output_matchtypes)
            else:
                matchtype_seq = ()
            if output_index < len(matchtype_seq):
                template_id = matchtype_seq[output_index]
                if isinstance(template_id, str) and template_id in input_templates:
                    return input_templates[template_id]
    raise CompatError(f"{v1_name}: V3 MatchType output {output_index} template is unrecoverable")


def _autogrow_family(
    name: str,
    config: Mapping[str, object] | None,
    *,
    input_is_list: bool,
    path: tuple[str, ...],
) -> tuple[
    InputFamilySpec,
    str,
    tuple[str, ...],
    tuple[str, str] | None,
    tuple[str, TypeExpr] | None,
]:
    template = config.get("template") if config is not None else None
    if not isinstance(template, Mapping):
        raise CompatError(f"V3 Autogrow input {name!r} lacks a template mapping")
    template_map = cast("Mapping[str, object]", template)
    names_obj = template_map.get("names")
    names: tuple[str, ...] | None = None
    if names_obj is not None:
        if not isinstance(names_obj, Sequence) or isinstance(names_obj, str):
            raise CompatError(f"V3 Autogrow input {name!r} has invalid names")
        names = tuple(str(item) for item in cast("Sequence[object]", names_obj))
    prefix = template_map.get("prefix")
    nested = template_map.get("input")
    if names is None and (not isinstance(prefix, str) or not prefix):
        raise CompatError(f"V3 Autogrow input {name!r} lacks a prefix")
    if names is not None and prefix is not None:
        raise CompatError(f"V3 Autogrow input {name!r} has both names and prefix")
    if not isinstance(nested, Mapping):
        raise CompatError(f"V3 Autogrow input {name!r} lacks a nested input mapping")
    rows = iter_v1_inputs(cast("Mapping[str, object]", nested))
    if len(rows) != 1:
        raise CompatError(
            f"V3 Autogrow input {name!r} expected exactly one nested input, got {len(rows)}"
        )
    member_name, member_v1_type, member_config, _ = rows[0]
    match_template: tuple[str, TypeExpr] | None = None
    if _is_exact_v3_type(member_v1_type, V3_MATCHTYPE_IO_TYPE):
        member_template = member_config.get("template") if member_config is not None else None
        type_expr, opaque_types, template_id = _matchtype_expr(member_template)
        match_template = (template_id, type_expr)
        pair = None
    elif _is_v3_marker(member_v1_type):
        raise _gate_error(
            f"V3 Autogrow input {name!r} has unsupported nested dynamic marker {member_v1_type}",
            "compat.dynamic.unsupported",
            input_path=(*path, member_name),
            path_kind="dynamic-family",
            lazy=_exact_bool((member_config or {}).get("lazy", False)),
            raw_link=_exact_bool((member_config or {}).get("rawLink", False)),
        )
    else:
        combo_source = (
            _v3_combo_options(member_config)
            if _is_exact_v3_type(member_v1_type, V3_COMBO_IO_TYPE)
            else member_v1_type
        )
        pair = boolean_combo(combo_source)
        if pair is not None:
            type_expr = TypeExpr.concrete(CORE_BOOLEAN)
            opaque_types = ()
        else:
            type_expr, opaque_types = translate_type(member_v1_type)
    if input_is_list:
        type_expr = TypeExpr.list_of(type_expr)
    min_members = template_map.get("min", 0)
    max_members = template_map.get("max")
    if not isinstance(min_members, int) or isinstance(min_members, bool):
        raise CompatError(f"V3 Autogrow input {name!r} has invalid min")
    if names is None and (not isinstance(max_members, int) or isinstance(max_members, bool)):
        raise CompatError(f"V3 Autogrow input {name!r} has invalid max")
    return (
        InputFamilySpec(
            name,
            type_expr,
            min_members=min_members,
            max_members=cast("int | None", max_members) if names is None else None,
            member_prefix=cast("str | None", prefix),
            member_names=names,
        ),
        cast("str", prefix) if names is None else "",
        opaque_types,
        pair,
        match_template,
    )


def iter_v1_inputs(
    input_types: Mapping[str, object],
) -> list[tuple[str, object, Mapping[str, object] | None, bool]]:
    """Yield (name, v1_type, config, required) from a v1 INPUT_TYPES dict.

    v1 tolerates the same input name appearing in both ``required`` and
    ``optional`` (execution passes one value per name regardless); packs in
    the wild rely on that, so the first declaration wins - required is
    iterated first - and later duplicates are dropped instead of producing
    a NodeSchema with duplicate input ids."""
    rows: list[tuple[str, object, Mapping[str, object] | None, bool]] = []
    seen: set[str] = set()
    for section, required in (("required", True), ("optional", False)):
        table = input_types.get(section)
        if table is None:
            continue
        if not isinstance(table, Mapping):
            raise CompatError(f"INPUT_TYPES[{section!r}] must be a mapping")
        for name, declared in cast("Mapping[str, object]", table).items():
            if name in seen:
                continue
            seen.add(name)
            v1_type: object = declared
            config: Mapping[str, object] | None = None
            if isinstance(declared, tuple):
                declared_tuple = cast("tuple[object, ...]", declared)
                if not declared_tuple:
                    raise CompatError(f"input {name!r}: empty declaration tuple")
                v1_type = declared_tuple[0]
                if len(declared_tuple) > 1:
                    if not isinstance(declared_tuple[1], Mapping):
                        raise CompatError(f"input {name!r}: config must be a mapping")
                    config = cast("Mapping[str, object]", declared_tuple[1])
            rows.append((name, v1_type, config, required))
    return rows


def _dynamic_entries_from_mapping(
    input_types: Mapping[str, object],
    translation: CompatTranslation,
    input_match_templates: dict[str, tuple[TypeExpr, tuple[str, ...]]],
    boolean_pairs: dict[int, tuple[str, str]],
    *,
    input_is_list: bool,
    parent_path: tuple[str, ...],
) -> tuple[DynamicEntry, ...]:
    return tuple(
        _dynamic_entry(
            name,
            v1_type,
            config,
            required,
            translation,
            input_match_templates,
            boolean_pairs,
            input_is_list=input_is_list,
            path=(*parent_path, name),
        )
        for name, v1_type, config, required in iter_v1_inputs(input_types)
    )


def _dynamic_entry(
    name: str,
    v1_type: object,
    config: Mapping[str, object] | None,
    required: bool,
    translation: CompatTranslation,
    input_match_templates: dict[str, tuple[TypeExpr, tuple[str, ...]]],
    boolean_pairs: dict[int, tuple[str, str]],
    *,
    input_is_list: bool,
    path: tuple[str, ...],
) -> DynamicEntry:
    path_kind: Literal["declared", "dynamic-family"] = (
        "declared" if len(path) == 1 else "dynamic-family"
    )
    lazy_fact = _exact_bool((config or {}).get("lazy", False))
    raw_link_fact = _exact_bool((config or {}).get("rawLink", False))
    if config is not None and config.get("lazy"):
        raise _gate_error(
            f"V3 input {name!r} uses unsupported lazy semantics",
            "compat.lazy.unsupported",
            input_path=path,
            path_kind=path_kind,
            lazy=lazy_fact,
            raw_link=raw_link_fact,
        )
    if config is not None and config.get("rawLink"):
        raise _gate_error(
            f"V3 input {name!r} uses unsupported rawLink semantics",
            "compat.raw-link.unsupported",
            input_path=path,
            path_kind=path_kind,
            lazy=lazy_fact,
            raw_link=raw_link_fact,
        )
    if _is_exact_v3_type(v1_type, V3_AUTOGROW_IO_TYPE):
        family, _, opaque, pair, match_template = _autogrow_family(
            name, config, input_is_list=input_is_list, path=path
        )
        translation.opaque_types.update(opaque)
        if pair is not None:
            boolean_pairs[id(family)] = pair
        if match_template is not None:
            template_id, expression = match_template
            input_match_templates[template_id] = (expression, opaque)
        return family
    if _is_exact_v3_type(v1_type, V3_DYNAMICCOMBO_IO_TYPE):
        options_obj = config.get("options") if config is not None else None
        if not isinstance(options_obj, Sequence) or isinstance(options_obj, str):
            raise _gate_error(
                f"V3 DynamicCombo input {name!r} has invalid options",
                "compat.dynamic.malformed",
                input_path=path,
                path_kind=path_kind,
                lazy=lazy_fact,
                raw_link=raw_link_fact,
            )
        options: list[DynamicComboOption] = []
        for raw_option in cast("Sequence[object]", options_obj):
            if not isinstance(raw_option, Mapping):
                raise _gate_error(
                    f"V3 DynamicCombo input {name!r} has malformed option",
                    "compat.dynamic.malformed",
                    input_path=path,
                    path_kind=path_kind,
                    lazy=lazy_fact,
                    raw_link=raw_link_fact,
                )
            option = cast("Mapping[str, object]", raw_option)
            key = option.get("key")
            nested = option.get("inputs")
            if not isinstance(key, str) or not isinstance(nested, Mapping):
                raise _gate_error(
                    f"V3 DynamicCombo input {name!r} has malformed option",
                    "compat.dynamic.malformed",
                    input_path=path,
                    path_kind=path_kind,
                    lazy=lazy_fact,
                    raw_link=raw_link_fact,
                )
            option_inputs = _dynamic_entries_from_mapping(
                cast("Mapping[str, object]", nested),
                translation,
                input_match_templates,
                boolean_pairs,
                input_is_list=input_is_list,
                parent_path=path,
            )
            options.append(DynamicComboOption(key, option_inputs))
        if required and not options:
            raise _gate_error(
                f"V3 DynamicCombo input {name!r} is required but has zero options",
                "compat.dynamic.malformed",
                input_path=path,
                path_kind=path_kind,
                lazy=lazy_fact,
                raw_link=raw_link_fact,
            )
        default = config.get("default") if config is not None else None
        if default is not None and not isinstance(default, str):
            raise _gate_error(
                f"V3 DynamicCombo input {name!r} has invalid default",
                "compat.dynamic.malformed",
                input_path=path,
                path_kind=path_kind,
                lazy=lazy_fact,
                raw_link=raw_link_fact,
            )
        return DynamicComboSpec(
            name,
            tuple(options),
            default=default,
            required=required,
            doc=str(config.get("tooltip", "") if config is not None else ""),
            display_name=str(config.get("display_name", "") if config is not None else ""),
        )
    if _is_exact_v3_type(v1_type, V3_DYNAMICSLOT_IO_TYPE):
        slot_type = config.get("slotType") if config is not None else None
        nested = config.get("inputs") if config is not None else None
        if not isinstance(slot_type, str) or not isinstance(nested, Mapping):
            raise _gate_error(
                f"V3 DynamicSlot input {name!r} is malformed",
                "compat.dynamic.malformed",
                input_path=path,
                path_kind=path_kind,
                lazy=lazy_fact,
                raw_link=raw_link_fact,
            )
        assert config is not None
        type_expr, opaque = translate_type(slot_type)
        translation.opaque_types.update(opaque)
        if input_is_list:
            type_expr = TypeExpr.list_of(type_expr)
        dependents = _dynamic_entries_from_mapping(
            cast("Mapping[str, object]", nested),
            translation,
            input_match_templates,
            boolean_pairs,
            input_is_list=input_is_list,
            parent_path=path,
        )
        return DynamicSlotSpec(
            name,
            slot_type=type_expr,
            required=False,
            inputs=dependents,
            force_input=bool(config.get("forceInput", False)),
            doc=str(config.get("tooltip", "")),
            display_name=str(config.get("display_name", "")),
        )
    combo_source = v1_type
    pair: tuple[str, str] | None = None
    if _is_exact_v3_type(v1_type, V3_MATCHTYPE_IO_TYPE):
        template = config.get("template") if config is not None else None
        type_expr, opaque, template_id = _matchtype_expr(template)
        input_match_templates[template_id] = (type_expr, opaque)
    elif _is_v3_marker(v1_type):
        raise _gate_error(
            f"unsupported V3 dynamic input kind {v1_type}",
            "compat.dynamic.unsupported",
            input_path=path,
            path_kind=path_kind,
            lazy=lazy_fact,
            raw_link=raw_link_fact,
        )
    else:
        combo_source = (
            _v3_combo_options(config) if _is_exact_v3_type(v1_type, V3_COMBO_IO_TYPE) else v1_type
        )
        pair = boolean_combo(combo_source)
        if pair is not None:
            type_expr = TypeExpr.concrete(CORE_BOOLEAN)
            opaque = ()
        else:
            type_expr, opaque = translate_type(v1_type)
    translation.opaque_types.update(opaque)
    if input_is_list:
        type_expr = TypeExpr.list_of(type_expr)
    default = _input_default(config, combo_source)
    if pair is not None and isinstance(default, str):
        default = default == pair[0]
    if input_is_list and default is not None:
        default = [default]
    widget: Widget | None = None
    if not input_is_list:
        widget = (
            BooleanWidget(label_on=pair[0], label_off=pair[1])
            if pair is not None
            else combo_widget(combo_source, config)
        )
        if widget is None:
            widget = string_widget(v1_type, config) or number_widget(name, v1_type, config)
    spec = InputSpec(
        name,
        type_expr,
        required=required and default is None,
        default=default,
        doc=str(config.get("tooltip", "") if config is not None else ""),
        widget=widget,
        force_input=bool(config.get("forceInput", False)) if config else False,
        advanced=bool(config.get("advanced", False)) if config else False,
        display_name=str(config.get("display_name", "")) if config else "",
    )
    if pair is not None:
        boolean_pairs[id(spec)] = pair
    return spec


def _output_ids(return_types: Sequence[object], return_names: Sequence[str] | None) -> list[str]:
    """Derive stable output ids from v1 RETURN_TYPES/RETURN_NAMES.

    In v1, outputs are positional and RETURN_NAMES are only display labels,
    so packs ship duplicates and wrong arities without noticing. Dinkster
    outputs are id-addressed, so ids are derived tolerantly: extra names
    are dropped, missing ones are filled from the type, and duplicates get
    a positional suffix - v1's positional wiring is unaffected either way."""
    names: list[str | None] = [None] * len(return_types)
    if return_names is not None:
        for i, name in enumerate(return_names[: len(return_types)]):
            names[i] = str(name)
    ids: list[str] = []
    seen: dict[str, int] = {}
    for name, rtype in zip(names, return_types, strict=True):
        if name is None:
            base = str(rtype).lower() if isinstance(rtype, str) else "out"
        else:
            base = name
        count = seen.get(base, 0)
        seen[base] = count + 1
        ids.append(base if count == 0 else f"{base}_{count + 1}")
    return ids


def _output_list_flags(v1_name: str, v1_class: type, output_count: int) -> tuple[bool, ...]:
    """Per-output OUTPUT_IS_LIST flags, aligned to RETURN_TYPES.

    v1 zips the declared flags against results, so a too-short tuple
    silently *drops* the trailing outputs (a wrong-output-count bug that
    detonates downstream); packs shipping short tuples mean False for the
    rest, so missing entries pad False and extras are ignored - tolerant
    the same way _output_ids is."""
    declared = getattr(v1_class, "OUTPUT_IS_LIST", None)
    if declared is None:
        return (False,) * output_count
    if not isinstance(declared, (list, tuple)):
        raise _gate_error(
            f"{v1_name}: OUTPUT_IS_LIST must be a tuple",
            "compat.output-list.malformed",
        )
    flags = [bool(x) for x in cast("Sequence[object]", declared)[:output_count]]
    flags.extend([False] * (output_count - len(flags)))
    return tuple(flags)


def _is_node_output(value: object) -> bool:
    """Whether a v1 function result is a ComfyUI V3 ``io.NodeOutput``.

    V3 ``io.ComfyNode`` classes ride NODE_CLASS_MAPPINGS behind a v1
    shim (nodes.py registers them directly; classproperties fake
    INPUT_TYPES/RETURN_TYPES and FUNCTION names EXECUTE_NORMALIZED,
    which ALWAYS returns a NodeOutput - comfy_api/latest/_io.py
    @ b78cec87), so the v1 wrapper must recognize the shape. Detection
    is by base-class name, mirroring execution.py's
    ``isinstance(r, _NodeOutputInternal)`` without importing ComfyUI -
    this module stays pure."""
    return any(base.__name__ == "_NodeOutputInternal" for base in type(value).__mro__)


def _is_execution_blocker(value: object) -> bool:
    """Whether a v1 output value is ComfyUI's ExecutionBlocker sentinel."""
    return any(base.__name__ == "ExecutionBlocker" for base in type(value).__mro__)


def _unwrap_node_output(v1_name: str, output: object) -> tuple[object, ...]:
    """A V3 NodeOutput's positional results, as the v1 result tuple
    (execution.py's V3 branch @ b78cec87: ``r.result`` or nothing).
    ``ui`` is dropped exactly like the v1 ``{"ui": ..., "result": ...}``
    convention's ui half; expansion and execution-blocking have no
    compat equivalent and refuse loudly instead of misexecuting."""
    if getattr(output, "expand", None) is not None:
        raise CompatError(
            f"{v1_name}: V3 NodeOutput requested graph expansion, which"
            " compat does not support (ROADMAP: Compat surface)"
        )
    block = getattr(output, "block_execution", None)
    if block is not None:
        raise CompatError(f"{v1_name}: V3 NodeOutput blocked execution: {block}")
    result = getattr(output, "result", None)
    if result is None:
        return ()
    if not isinstance(result, tuple):
        raise CompatError(
            f"{v1_name}: V3 NodeOutput.result is {type(result).__name__}, expected a tuple"
        )
    return cast("tuple[object, ...]", result)


_MISSING = object()


def _lower_dynamic_entry(
    entry: DynamicEntry,
    path: str,
    inputs: dict[str, object],
    boolean_pairs: Mapping[int, tuple[str, str]],
    *,
    input_is_list: bool,
) -> object:
    if isinstance(entry, InputSpec):
        value = inputs.pop(path, _MISSING)
        if value is _MISSING:
            return value
        return _restore_boolean_vocabulary(value, boolean_pairs.get(id(entry)))
    if isinstance(entry, InputFamilySpec):
        grouped = inputs.pop(path, _MISSING)
        if isinstance(grouped, Mapping):
            suffixes = list(cast("Mapping[str, object]", grouped))
            values = dict(cast("Mapping[str, object]", grouped))
        else:
            prefix = path + "."
            suffixes: list[str] = []
            for input_id in inputs:
                if input_id.startswith(prefix):
                    suffix = input_id[len(prefix) :].split(".", 1)[0]
                    if suffix not in suffixes:
                        suffixes.append(suffix)
            values = {}
        if entry.member_names is not None:
            suffixes = [name for name in entry.member_names if name in suffixes]
        nested: dict[str, object] = {}
        for index, suffix in enumerate(suffixes):
            member_path = f"{path}.{suffix}"
            if len(entry.template) != 1 or not isinstance(entry.template[0], InputSpec):
                raise CompatError(f"V3 Autogrow input {path!r} has unsupported grouped template")
            value = values[suffix] if suffix in values else inputs.pop(member_path, _MISSING)
            if value is _MISSING:
                continue
            value = _restore_boolean_vocabulary(value, boolean_pairs.get(id(entry)))
            upstream_name = (
                suffix if entry.member_names is not None else f"{entry.member_prefix}{index}"
            )
            nested[upstream_name] = value
        return nested
    if isinstance(entry, DynamicComboSpec):
        choice = inputs.pop(path, _MISSING)
        if choice is _MISSING:
            return _MISSING
        if not isinstance(choice, str):
            raise CompatError(f"V3 DynamicCombo input {path!r} expected a string choice")
        option = entry.option(choice)
        if option is None:
            raise CompatError(f"V3 DynamicCombo input {path!r} has unknown option {choice!r}")
        nested = {entry.id: [choice] if input_is_list else choice}
        for child in option.inputs:
            child_path = f"{path}.{child.id}"
            value = _lower_dynamic_entry(
                child,
                child_path,
                inputs,
                boolean_pairs,
                input_is_list=input_is_list,
            )
            if value is not _MISSING:
                nested[child.id] = value
        return nested
    assert isinstance(entry, DynamicSlotSpec)
    raw = inputs.pop(path, _MISSING)
    nested: dict[str, object] = {}
    if isinstance(raw, SlotValue):
        nested[entry.id] = raw.value
        nested.update(raw.options)
    elif raw is not _MISSING:
        nested[entry.id] = raw
    for child in entry.inputs:
        child_path = f"{path}.{child.id}"
        value = _lower_dynamic_entry(
            child,
            child_path,
            inputs,
            boolean_pairs,
            input_is_list=input_is_list,
        )
        if value is not _MISSING:
            nested[child.id] = value
    return nested if nested else _MISSING


class CompatTranslation:
    """The output of translating a v1 mapping: Dinkster node classes plus the
    opaque comfy.* type ids they need registered."""

    def __init__(self) -> None:
        self.node_classes: list[type[Node]] = []
        self.opaque_types: set[str] = set()
        # Namespaced node identity -> reason. This prevents separate legacy
        # packs with the same v1 name from overwriting each other's diagnostic.
        # A skip is a diagnostic, not a failure: the rest of the pack loads.
        self.skipped: dict[str, str] = {}
        self.diagnostics: dict[str, CompatGateDiagnostic] = {}
        # Choice-list id (comfy.files.<category>) -> the listing values
        # recorded when a probe matched a combo to a filesystem category.
        # The pack's choices entry serves these behind /api/choices/{id};
        # enumerating here (worker startup, same moment the schemas baked
        # their snapshots) keeps schema and choice list telling one story.
        # First successfully translated observer wins: a later node
        # observing different values for the same id keeps its combo
        # frozen instead of overwriting the served provider, and a node
        # that fails translation commits nothing (staged per node).
        self.listing_snapshots: dict[str, tuple[str, ...]] = {}
        # Translated model-file selectors speak dinkster.asset. Legacy custom
        # pack workers do not compose native.py, so register_types() must
        # bind the asset resolver when at least one such input lands.
        self._requires_asset_type = False

    def require_asset_type(self) -> None:
        """Mark that at least one translated input consumes an AssetRef."""
        self._requires_asset_type = True

    def register_types(
        self,
        registry: TypeRegistry,
        *,
        resident: frozenset[str] | None = None,
        resident_meta: Callable[[object], Mapping[str, object]] | None = None,
        table: ResidencyTable | None = None,
    ) -> None:
        """Register every opaque comfy.* type. Loaded-hardware-state types
        (MODEL/CLIP/VAE/...) get the resident codec - the value stays in
        this process, a stub crosses (resident.py) - with ``resident_meta``
        publishing device residency/cost on their envelopes; everything
        else gets correct-everywhere defaults (default codec, content-hash
        fingerprint of encoded bytes)."""
        if self._requires_asset_type and ASSET_TYPE not in registry:
            register_asset_type(registry, resolver_from_env())
        resident_types = resident if resident is not None else DEFAULT_RESIDENT_V1_TYPES
        for type_id in sorted(self.opaque_types):
            v1_name = type_id.removeprefix(COMFY_TYPE_PREFIX)
            if v1_name in resident_types:
                register_resident_type(registry, type_id, table=table, meta=resident_meta)
            elif v1_name in (IMAGE_V1_NAME, MASK_V1_NAME):
                # Images and masks cross as npy bytes, not pickle, so the
                # torchless engine can decode and render them, and the two
                # spellings of each (comfy.IMAGE/dinkster.image, comfy.MASK/
                # dinkster.mask) stay one value type (image.py has the story).
                register_image_type(registry, type_id)
            elif v1_name == AUDIO_V1_NAME:
                register_audio_type(registry, type_id)
            elif v1_name == VIDEO_V1_NAME:
                register_video_type(registry, type_id)
            elif v1_name in {"LAYERS", "COMPOSITOR"}:
                from dinkster_image_document.compat import (
                    register_comfy_compositor,
                    register_comfy_layers,
                )

                if v1_name == "LAYERS":
                    register_comfy_layers(registry)
                else:
                    register_comfy_compositor(registry)
            elif v1_name == "LATENT":
                register_latent_type(registry, type_id)
            else:
                registry.register(type_id)


def translate_node(
    v1_name: str,
    v1_class: type,
    translation: CompatTranslation,
    *,
    display_name: str = "",
    namespace: str = "",
    probe: InputTypesProbe | None = None,
) -> type[Node]:
    """Manufacture a Dinkster Node subclass wrapping one v1 node class.

    ``namespace`` scopes the Dinkster node type id: core ComfyUI nodes get
    ``comfy.<name>``; legacy custom packs get ``comfy.<pack>.<name>`` so two
    packs declaring the same v1 key cannot collide (v1's flat global
    NODE_CLASS_MAPPINGS namespace is a bug, not a feature). Opaque *type*
    ids stay unnamespaced on purpose - a MODEL is the same MODEL whichever
    pack produced it."""
    input_types_fn = getattr(v1_class, "INPUT_TYPES", None)
    if not callable(input_types_fn):
        raise CompatError(f"{v1_name}: v1 class has no INPUT_TYPES classmethod")
    # INPUT_IS_LIST/OUTPUT_IS_LIST switch v1 to a list-batch calling
    # convention. Dinkster has that shape as data (list<T>, DESIGN 3.13), so
    # the flags translate to honest list-typed sockets: the schema says
    # what v1's executor kept invisible, and execute() passes the lists
    # through verbatim - the function is called exactly the way v1 calls
    # it (once, with whole lists), never miscalled.
    raw_input_is_list = getattr(v1_class, "INPUT_IS_LIST", False)
    input_is_list = bool(raw_input_is_list)
    observations: Sequence[ListingObservation] = ()
    if probe is None:
        raw_inputs = input_types_fn()
    else:
        raw_inputs, observations = probe(input_types_fn)
    if not isinstance(raw_inputs, Mapping):
        raise CompatError(f"{v1_name}: INPUT_TYPES() must return a mapping")
    return_types_obj = getattr(v1_class, "RETURN_TYPES", ())
    custom_combo = getattr(v1_class, "_ACCEPT_ALL_INPUTS", False) is True and (
        _custom_combo_runtime_shape(
            v1_name,
            v1_class,
            cast("Mapping[str, object]", raw_inputs),
            return_types_obj,
            namespace=namespace,
            raw_input_is_list=raw_input_is_list,
        )
    )
    if bool(getattr(v1_class, "_ACCEPT_ALL_INPUTS", False)) and not custom_combo:
        raise _gate_error(
            f"{v1_name}: V3 accept_all_inputs is not supported",
            "compat.accept-all.unsupported",
        )
    allowed_lazy, selector_lazy = supported_lazy_inputs(
        v1_name,
        v1_class,
        cast("Mapping[str, object]", raw_inputs),
        return_types_obj,
        input_is_list=input_is_list,
    )

    specs: list[InputSpec] = []
    families: list[InputFamilySpec] = []
    combos: list[DynamicComboSpec] = []
    slots: list[DynamicSlotSpec] = []
    family_prefixes: dict[str, str] = {}
    family_names: dict[str, tuple[str, ...]] = {}
    family_boolean_combo_strings: dict[str, tuple[str, str]] = {}
    dynamic_boolean_combo_strings: dict[int, tuple[str, str]] = {}
    input_match_templates: dict[str, tuple[TypeExpr, tuple[str, ...]]] = {}
    model_asset_inputs: dict[str, str] = {}
    source_filename_inputs: dict[str, tuple[SourceFilenameSpec, bool]] = {}
    #: input id -> (on_string, off_string) for disguised-boolean combos;
    #: execute() maps the honest boolean back onto these v1 strings.
    boolean_combo_strings: dict[str, tuple[str, str]] = {}
    numeric_combo_values: dict[str, dict[str, object]] = {}
    #: Listing snapshots staged by this node's combos; committed onto the
    #: translation only when the whole node translates successfully, so a
    #: node that records a listing and then fails validation cannot leave
    #: a snapshot behind (or overwrite a valid node's).
    pending_snapshots: dict[str, tuple[str, ...]] = {}
    occupies_gpu = False
    for name, v1_type, config, required in iter_v1_inputs(cast("Mapping[str, object]", raw_inputs)):
        source_filename = _source_filename_config(
            v1_name,
            name,
            config,
            input_is_list=input_is_list,
        )
        if source_filename is not None:
            assert config is not None
            if not (
                isinstance(v1_type, (list, tuple)) or _is_exact_v3_type(v1_type, V3_COMBO_IO_TYPE)
            ):
                raise CompatError(
                    f"{v1_name}: input {name!r} source upload requires an exact COMBO declaration"
                )
            binding, listed = source_filename
            source_type = TypeExpr.concrete(ASSET_TYPE)
            specs.append(
                InputSpec(
                    id=name,
                    type=TypeExpr.list_of(source_type) if listed else source_type,
                    required=required,
                    widget=source_asset_widget(binding),
                    doc=str(config.get("tooltip", "")),
                    display_name=str(config.get("display_name", "")),
                    force_input=bool(config.get("forceInput", False)),
                    advanced=bool(config.get("advanced", False)),
                    source_filename=binding,
                )
            )
            source_filename_inputs[name] = (binding, listed)
            continue
        if config is not None and config.get("lazy") and name not in allowed_lazy:
            raise _gate_error(
                f"{v1_name}: V3 input {name!r} uses unsupported lazy semantics",
                "compat.lazy.unsupported",
                input_path=(name,),
                lazy=_exact_bool(config.get("lazy")),
                raw_link=_exact_bool(config.get("rawLink", False)),
            )
        if config is not None and config.get("rawLink"):
            raise _gate_error(
                f"{v1_name}: V3 input {name!r} uses unsupported rawLink semantics",
                "compat.raw-link.unsupported",
                input_path=(name,),
                lazy=_exact_bool(config.get("lazy", False)),
                raw_link=_exact_bool(config.get("rawLink")),
            )
        if _is_exact_v3_type(v1_type, V3_DYNAMICCOMBO_IO_TYPE) or _is_exact_v3_type(
            v1_type, V3_DYNAMICSLOT_IO_TYPE
        ):
            dynamic = _dynamic_entry(
                name,
                v1_type,
                config,
                required,
                translation,
                input_match_templates,
                dynamic_boolean_combo_strings,
                input_is_list=input_is_list,
                path=(name,),
            )
            if isinstance(dynamic, DynamicComboSpec):
                combos.append(dynamic)
            else:
                assert isinstance(dynamic, DynamicSlotSpec)
                slots.append(dynamic)
            if dynamic_entries_occupy_gpu((dynamic,)):
                occupies_gpu = True
            continue
        if _is_exact_v3_type(v1_type, V3_AUTOGROW_IO_TYPE):
            family, prefix, opaque_types, family_pair, match_template = _autogrow_family(
                name, config, input_is_list=input_is_list, path=(name,)
            )
            families.append(family)
            family_prefixes[name] = prefix
            if family.member_names is not None:
                family_names[name] = family.member_names
            if family_pair is not None:
                family_boolean_combo_strings[name] = family_pair
            if match_template is not None:
                template_id, template_expr = match_template
                input_match_templates[template_id] = (template_expr, opaque_types)
            translation.opaque_types.update(opaque_types)
            if dynamic_entries_occupy_gpu((family,)):
                occupies_gpu = True
            continue
        if _is_exact_v3_type(v1_type, V3_MATCHTYPE_IO_TYPE):
            template = config.get("template") if config is not None else None
            type_expr, opaque_types, template_id = _matchtype_expr(template)
            input_match_templates[template_id] = (type_expr, opaque_types)
        elif _is_v3_marker(v1_type):
            raise _gate_error(
                f"{v1_name}: unsupported V3 dynamic input kind {v1_type}",
                "compat.dynamic.unsupported",
                input_path=(name,),
                lazy=_exact_bool((config or {}).get("lazy", False)),
                raw_link=_exact_bool((config or {}).get("rawLink", False)),
            )
        else:
            type_expr, opaque_types = translate_type(v1_type)
        translation.opaque_types.update(opaque_types)
        # Which GPU is a fact of the model value received (its envelope's
        # residency meta binds the concrete lane); *that* the node occupies
        # a GPU follows from taking loaded hardware state as input at all.
        if isinstance(v1_type, str) and any(
            part.strip() in DEFAULT_RESIDENT_V1_TYPES for part in str(v1_type).split(",")
        ):
            occupies_gpu = True
        combo_source = (
            _v3_combo_options(config) if _is_exact_v3_type(v1_type, V3_COMBO_IO_TYPE) else v1_type
        )
        multi_widget = multi_combo_widget(
            combo_source,
            config,
            input_is_list=input_is_list,
            trusted_choice_ids=set(translation.listing_snapshots) | set(pending_snapshots),
        )
        if multi_widget is not None:
            default = config.get("default") if config is not None else None
            default_values = None if default is None else list(cast("list[str]", default))
            specs.append(
                InputSpec(
                    id=name,
                    type=TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO)),
                    required=required and default is None,
                    default=default_values,
                    widget=multi_widget,
                    doc=str(config.get("tooltip", "") if config is not None else ""),
                    display_name=str(config.get("display_name", "") if config is not None else ""),
                    force_input=bool(config.get("forceInput", False)) if config else False,
                    advanced=bool(config.get("advanced", False)) if config else False,
                    lazy=name in allowed_lazy,
                )
            )
            continue
        vocabulary = _numeric_combo_vocabulary(combo_source)
        if vocabulary:
            numeric_combo_values[name] = vocabulary
            combo_source = tuple(vocabulary)
        default = _input_default(config, combo_source)
        if vocabulary:
            default = _combo_token(default)
        required_spec = required and default is None
        # The v1 choice list survives as a static dropdown - except under
        # INPUT_IS_LIST, where the socket is list<core.combo> and a
        # widget on a list socket is undefined in the wire contract.
        widget: Widget | None = None
        if not input_is_list:
            listing_sources = _observed_listing(combo_source, observations)
            selector = _model_file_selector(v1_class, name)
            if selector is not None and selector.provenance == "stable-sorted-source":
                listing_sources = _stable_sorted_listing(v1_class, name, combo_source, observations)
            if listing_sources:
                # This choice list IS a filesystem listing - the very
                # object get_filename_list returned, except for the one
                # source-proven stable-sort marker above. A listing is
                # never a disguised boolean. Only an exact selector row
                # becomes an asset; every other observed listing retains
                # its remote combo instead of widening by category.
                exact_categories = {
                    observation.category
                    for observation in listing_sources
                    if observation.values == tuple(cast("Sequence[str]", combo_source))
                }
                if len(exact_categories) > 1:
                    raise CompatError(
                        f"{v1_name}: input {name!r} is the same listing object "
                        f"for multiple categories: {sorted(exact_categories)!r}"
                    )
                if selector is not None and exact_categories == {selector.category}:
                    category = selector.category
                    model_asset_inputs[name] = category
                    type_expr = TypeExpr.concrete(ASSET_TYPE)
                    default = None
                    required_spec = required
                    widget = AssetWidget(
                        accept=("application/octet-stream",),
                        kind=MODEL_FILE_CATEGORIES[category].kind,
                    )
                else:
                    widget = combo_widget(combo_source, config)
                    if widget is not None:
                        widget = _remote_listing_combo(
                            widget,
                            listing_sources,
                            translation.listing_snapshots,
                            pending_snapshots,
                        )
            else:
                pair = boolean_combo(combo_source)
                if pair is not None:
                    # A disguised boolean gets an honest type; the
                    # original option strings survive as toggle labels
                    # and as the vocabulary execute() converts back into.
                    on_string, off_string = pair
                    boolean_combo_strings[name] = pair
                    type_expr = TypeExpr.concrete(CORE_BOOLEAN)
                    widget = BooleanWidget(label_on=on_string, label_off=off_string)
                    if isinstance(default, str):
                        default = default == on_string
                else:
                    widget = combo_widget(combo_source, config) or _boolean_labels_widget(
                        v1_type, config
                    )
            if widget is None:
                widget = string_widget(v1_type, config) or number_widget(name, v1_type, config)
        if input_is_list:
            # v1 with INPUT_IS_LIST receives every input as a whole list;
            # widget/constant values arrive length-1-wrapped upstream, so
            # the socket is list<T> and a declared default becomes [default].
            type_expr = TypeExpr.list_of(type_expr)
            if default is not None:
                default = [default]
        remote = config.get("remote") if config is not None else None
        if remote is not None and _is_exact_v3_type(v1_type, V3_COMBO_IO_TYPE):
            if not isinstance(widget, ComboWidget):
                raise CompatError(f"{v1_name}: V3 remote combo {name!r} is not representable")
            widget = trusted_v3_remote_combo(
                widget,
                remote,
                set(translation.listing_snapshots) | set(pending_snapshots),
            )
        specs.append(
            InputSpec(
                id=name,
                type=type_expr,
                required=required_spec,
                default=default,
                widget=widget,
                doc=str(config.get("tooltip", "") if config is not None else ""),
                display_name=str(config.get("display_name", "") if config is not None else ""),
                force_input=bool(config.get("forceInput", False)) if config else False,
                advanced=bool(config.get("advanced", False)) if config else False,
                lazy=name in allowed_lazy,
            )
        )

    if custom_combo:
        specs.append(
            InputSpec(
                id="index",
                type=TypeExpr.concrete(CORE_INT),
                required=False,
                default=0,
            )
        )
        families.append(
            InputFamilySpec(
                CUSTOM_COMBO_FAMILY_ID,
                TypeExpr.concrete(CORE_STRING),
                member_names=CUSTOM_COMBO_OPTION_NAMES,
            )
        )
        family_names[CUSTOM_COMBO_FAMILY_ID] = CUSTOM_COMBO_OPTION_NAMES

    if not isinstance(return_types_obj, (list, tuple)):
        raise CompatError(f"{v1_name}: RETURN_TYPES must be a tuple")
    return_types = tuple(cast("Sequence[object]", return_types_obj))
    return_names_obj = getattr(v1_class, "RETURN_NAMES", None)
    return_names: Sequence[str] | None = None
    if return_names_obj is not None:
        if not isinstance(return_names_obj, (list, tuple)):
            raise CompatError(f"{v1_name}: RETURN_NAMES must be a tuple")
        return_names = [str(x) for x in cast("Sequence[object]", return_names_obj)]
    output_ids = _output_ids(return_types, return_names)
    output_list_flags = _output_list_flags(v1_name, v1_class, len(return_types))
    combo_output_ids = {
        output_id
        for output_id, rtype in zip(output_ids, return_types, strict=True)
        if _is_exact_v3_type(rtype, V3_COMBO_IO_TYPE) or isinstance(rtype, (list, tuple))
    }

    outputs: list[OutputSpec] = []
    for output_index, (output_id, rtype, is_list) in enumerate(
        zip(output_ids, return_types, output_list_flags, strict=True)
    ):
        if _is_exact_v3_type(rtype, V3_MATCHTYPE_IO_TYPE):
            type_expr, opaque_types = _output_matchtype_expr(
                v1_name, v1_class, output_index, input_match_templates
            )
        elif _is_v3_marker(rtype):
            raise _gate_error(
                f"{v1_name}: unsupported V3 dynamic output kind {rtype}",
                "compat.dynamic.unsupported",
            )
        else:
            type_expr, opaque_types = translate_type(rtype)
        translation.opaque_types.update(opaque_types)
        if is_list:
            type_expr = TypeExpr.list_of(type_expr)
        outputs.append(OutputSpec(id=output_id, type=type_expr))

    function_name = getattr(v1_class, "FUNCTION", None)
    if not isinstance(function_name, str) or not hasattr(v1_class, function_name):
        raise CompatError(f"{v1_name}: FUNCTION does not name a method")
    function_is_async = inspect.iscoroutinefunction(getattr(v1_class, function_name))

    hidden_types: Mapping[str, object] = {}
    hidden_table = cast("Mapping[str, object]", raw_inputs).get("hidden")
    if isinstance(hidden_table, Mapping):
        hidden_types = dict(cast("Mapping[str, object]", hidden_table))

    unique_id_names = frozenset(
        hidden
        for hidden, hidden_type in hidden_types.items()
        if hidden_type == "UNIQUE_ID" or hidden_type == ("UNIQUE_ID",)
    )
    prompt_names = frozenset(
        hidden
        for hidden, hidden_type in hidden_types.items()
        if hidden_type == "PROMPT" or hidden_type == ("PROMPT",)
    )
    extra_pnginfo_names = frozenset(
        hidden
        for hidden, hidden_type in hidden_types.items()
        if hidden_type == "EXTRA_PNGINFO" or hidden_type == ("EXTRA_PNGINFO",)
    )

    is_output_node = bool(getattr(v1_class, "OUTPUT_NODE", False))
    has_is_changed = getattr(v1_class, "IS_CHANGED", None) is not None

    schema = NodeSchema(
        node_type=comfy_type_id(f"{namespace}.{v1_name}" if namespace else v1_name),
        display_name=display_name or v1_name,
        category="comfy/" + str(getattr(v1_class, "CATEGORY", "uncategorized")),
        description=str(getattr(v1_class, "DESCRIPTION", "") or ""),
        inputs=tuple(specs),
        input_families=tuple(families),
        combos=tuple(combos),
        slots=tuple(slots),
        outputs=tuple(outputs),
        # v1 cannot declare idempotence; OUTPUT_NODE and IS_CHANGED both
        # signal "do not trust replay". UNIQUE_ID is also an execution input
        # that is intentionally absent from engine cache identity.
        idempotent=not (is_output_node or has_is_changed or unique_id_names),
        occupies=("gpu",) if occupies_gpu else (),
        # The v1 class_type stays resolvable: submission-format adapters
        # (the Comfy API prompt endpoint) map class_type -> node_type
        # through aliases instead of parsing namespaced type ids back apart.
        aliases=(v1_name,),
        # Any v1 function may return a runtime graph expansion payload from
        # runtime data; compat refuses those payloads loudly (normalize_result),
        # so every translated schema declares the capability up front.
        may_expand_graph=True,
        output_node=is_output_node,
        selector=(
            SelectorSpec("switch", {"false": "on_false", "true": "on_true"})
            if selector_lazy
            else None
        ),
    )

    def restore_inputs(inputs: Mapping[str, object]) -> dict[str, object]:
        restored = dict(inputs)
        for name, vocabulary in numeric_combo_values.items():
            if name not in restored:
                continue
            value = restored[name]
            restored[name] = (
                [_restore_combo_value(item, vocabulary) for item in cast("list[object]", value)]
                if input_is_list and isinstance(value, list)
                else _restore_combo_value(value, vocabulary)
            )
        for name, (binding, listed) in source_filename_inputs.items():
            if name in restored:
                restored[name] = _materialize_source_value(
                    v1_name,
                    name,
                    restored[name],
                    binding,
                    listed=listed,
                )
        for name, category in model_asset_inputs.items():
            if name in restored:
                restored[name] = _asset_to_model_filename(restored[name], category)
        for name, (on_string, off_string) in boolean_combo_strings.items():
            # The schema says core.boolean; the v1 function compares
            # strings. Convert honest booleans back, and pass a legacy
            # prompt's original string through verbatim.
            if name in restored:
                restored[name] = _restore_boolean_vocabulary(
                    restored[name], (on_string, off_string)
                )
        for family_id, prefix in family_prefixes.items():
            members = restored.get(family_id, {})
            if not isinstance(members, Mapping):
                raise CompatError(f"{v1_name}: input family {family_id!r} expected a mapping")
            member_map = cast("Mapping[str, object]", members)
            pair = family_boolean_combo_strings.get(family_id)
            nested: dict[str, object] = {}
            ordered_members = (
                (
                    (suffix, member_map[suffix])
                    for suffix in family_names[family_id]
                    if suffix in member_map
                )
                if family_id in family_names
                else member_map.items()
            )
            for index, (suffix, value) in enumerate(ordered_members):
                value = _restore_boolean_vocabulary(value, pair)
                nested[suffix if family_id in family_names else f"{prefix}{index}"] = value
            restored[family_id] = nested
        for dynamic in (*combos, *slots):
            value = _lower_dynamic_entry(
                dynamic,
                dynamic.id,
                restored,
                dynamic_boolean_combo_strings,
                input_is_list=input_is_list,
            )
            if value is not _MISSING:
                restored[dynamic.id] = value
        context = current_execution_context()
        node_id = context.node_id if context is not None else None
        export_snapshot = context.export_snapshot if context is not None else None
        for hidden in hidden_types:
            # Under INPUT_IS_LIST v1 wraps hidden values like everything
            # else, so a node indexing a synthesized hidden value still binds.
            if hidden in unique_id_names:
                value = node_id
            elif is_output_node and export_snapshot is not None and hidden in prompt_names:
                value = export_snapshot.prompt
            elif is_output_node and export_snapshot is not None and hidden in extra_pnginfo_names:
                value = export_snapshot.extra_pnginfo
            else:
                value = None
            restored.setdefault(hidden, [value] if input_is_list else value)
        return restored

    def normalize_result(result: object) -> Mapping[str, object]:
        if isinstance(result, Mapping):
            # v1 UI convention: {"ui": {...}, "result": (...)}
            if "expand" in result:
                raise CompatError(
                    f"{v1_name}: v1 result requested graph expansion, which"
                    " compat does not support (docs/compat-porting-recipes.md,"
                    " 'Graph expansion dispositions')"
                )
            result_mapping = cast("Mapping[str, object]", result)
            if is_output_node and "ui" in result_mapping:
                capture_saved_results(result_mapping["ui"])
            result = result_mapping.get("result", ())
        elif _is_node_output(result):
            # V3 io.ComfyNode riding NODE_CLASS_MAPPINGS via the shim.
            if is_output_node:
                capture_saved_results(getattr(result, "ui", None))
            result = _unwrap_node_output(v1_name, result)
        if not isinstance(result, (list, tuple)):
            raise CompatError(
                f"{v1_name}: v1 function returned {type(result).__name__}, expected a tuple"
            )
        result_tuple = cast("Sequence[object]", result)
        if len(result_tuple) != len(output_ids):
            raise CompatError(
                f"{v1_name}: v1 function returned {len(result_tuple)} values for "
                f"{len(output_ids)} declared outputs"
            )
        values: list[object] = []
        for output_id, value, is_list in zip(
            output_ids, result_tuple, output_list_flags, strict=True
        ):
            if is_list:
                # v1's merge would extend() whatever iterable came back (a
                # string would splat into characters); a non-list here is a
                # node bug, surfaced at the cause instead.
                if not isinstance(value, (list, tuple)):
                    raise CompatError(
                        f"{v1_name}: output {output_id!r} is OUTPUT_IS_LIST "
                        f"but the v1 function returned "
                        f"{type(value).__name__}, expected a list"
                    )
                value = list(cast("Sequence[object]", value))
                blocked = any(_is_execution_blocker(item) for item in value)
            else:
                blocked = _is_execution_blocker(value)
            if blocked:
                raise CompatError(
                    f"{v1_name}: output {output_id!r} returned an ExecutionBlocker, which"
                    " compat does not support (docs/compat-porting-recipes.md,"
                    " 'ExecutionBlocker to Dinkster absence semantics')"
                )
            if output_id in combo_output_ids:
                value = (
                    [_combo_token(item) for item in cast("list[object]", value)]
                    if is_list
                    else _combo_token(value)
                )
            values.append(from_comfy_multistream(_declare_fixed_av_output(v1_name, value)))
        return dict(zip(output_ids, values, strict=True))

    async def await_result(result: Awaitable[object]) -> Mapping[str, object]:
        return normalize_result(await result)

    def invoke(inputs: Mapping[str, object]) -> object:
        custom_combo_options = (
            _closed_custom_combo_options(inputs.get(CUSTOM_COMBO_FAMILY_ID, {}))
            if custom_combo
            else None
        )
        native_inputs = restore_inputs(inputs)
        structural_inputs = {
            name for name, value in native_inputs.items() if _contains_multistream(value)
        }
        if structural_inputs:
            allowed = _MULTI_STREAM_INPUTS.get(v1_name)
            if allowed is None or not structural_inputs <= allowed:
                raise CompatError(
                    f"{v1_name}: multi-stream LATENT inputs are not declared for "
                    f"{sorted(structural_inputs)!r}"
                )
        restored = {name: to_comfy_multistream(value) for name, value in native_inputs.items()}
        prepare_v3 = getattr(v1_class, "PREPARE_CLASS_CLONE", None)
        if callable(prepare_v3):
            hidden_inputs: dict[str, object] = {}
            for name, declared in hidden_types.items():
                hidden_type = (
                    cast("tuple[object, ...]", declared)[0]
                    if isinstance(declared, tuple)
                    else declared
                )
                if isinstance(hidden_type, str):
                    hidden_inputs[hidden_type] = restored.pop(name)
            prepared = prepare_v3({"hidden_inputs": hidden_inputs})
            fn = cast("Callable[..., object]", getattr(prepared, function_name))
        else:
            instance = v1_class()
            fn = cast("Callable[..., object]", getattr(instance, function_name))
        if custom_combo:
            unknown = set(restored) - {"choice", "index", CUSTOM_COMBO_FAMILY_ID}
            if unknown:
                raise CompatError(
                    "CustomCombo: undeclared execution inputs are not supported: "
                    f"{sorted(unknown)!r}"
                )
            restored.pop(CUSTOM_COMBO_FAMILY_ID, None)
            assert custom_combo_options is not None
            choice = restored.pop("choice")
            index = restored.pop("index", 0)
            return fn(choice=choice, index=index, options=custom_combo_options)
        return fn(**restored)

    async def execute_async(cls: type[Node], **inputs: object) -> Mapping[str, object]:
        del cls
        result = invoke(inputs)
        if not inspect.isawaitable(result):
            raise CompatError(f"{v1_name}: async function returned a non-awaitable result")
        return await await_result(cast("Awaitable[object]", result))

    def execute_sync(
        cls: type[Node], **inputs: object
    ) -> Mapping[str, object] | Awaitable[Mapping[str, object]]:
        del cls
        result = invoke(inputs)
        if inspect.iscoroutine(result):
            # A synchronous callable may still return a coroutine. The
            # ordinary worker awaits that continuation without a nested loop
            # or detached task.
            return await_result(cast("Awaitable[object]", result))
        return normalize_result(result)

    def define_schema(cls: type[Node]) -> NodeSchema:
        return schema

    def check_lazy_status(cls: type[Node], **inputs: object) -> object:
        del cls
        hook = getattr(v1_class(), "check_lazy_status", None)
        if not callable(hook):
            raise CompatError(f"{v1_name}: lazy inputs require check_lazy_status")
        return hook(**restore_inputs(inputs))

    members: dict[str, object] = {
        "define_schema": classmethod(define_schema),
        "execute": classmethod(execute_async if function_is_async else execute_sync),
        "__doc__": f"Dinkster compat wrapper for ComfyUI v1 node {v1_name!r}.",
    }
    if allowed_lazy:
        members["check_lazy_status"] = classmethod(check_lazy_status)
    node_class = type(
        f"Compat_{v1_name}",
        (Node,),
        members,
    )
    # The node translated end to end: only now do its staged listing
    # snapshots become servable (a failed node's observations vanish
    # with it, never overwriting a valid node's provider values).
    translation.listing_snapshots.update(pending_snapshots)
    if model_asset_inputs or source_filename_inputs:
        translation.require_asset_type()
    translation.node_classes.append(node_class)
    return node_class


def translate_mappings(
    node_class_mappings: Mapping[str, type],
    *,
    display_names: Mapping[str, str] | None = None,
    only: Sequence[str] | None = None,
    namespace: str = "",
    translation: CompatTranslation | None = None,
    probe: InputTypesProbe | None = None,
) -> CompatTranslation:
    """Translate a NODE_CLASS_MAPPINGS dict. ``only`` restricts to the named
    v1 nodes (missing names raise: a filter that silently drops is a trap).

    Explicitly requested nodes must translate or the whole call fails.
    Without ``only`` (translate whatever the pack offers), untranslatable
    nodes are recorded in ``CompatTranslation.skipped`` with the reason and
    the rest of the pack still loads - one exotic node must not take down
    the pack, but the skip must never be silent.

    Pass an existing ``translation`` to accumulate several packs into one
    (opaque types unify; namespaced node ids keep the packs distinct)."""
    if translation is None:
        translation = CompatTranslation()
    names: Sequence[str]
    if only is not None:
        missing = [name for name in only if name not in node_class_mappings]
        if missing:
            raise CompatError(f"unknown v1 node names: {missing}")
        names = list(only)
    else:
        names = list(node_class_mappings)
    display = display_names or {}
    for name in names:
        skip_id = f"{namespace}.{name}" if namespace else name
        opaque_before = set(translation.opaque_types)
        try:
            translate_node(
                name,
                node_class_mappings[name],
                translation,
                display_name=display.get(name, ""),
                namespace=namespace,
                probe=probe,
            )
            translation.skipped.pop(skip_id, None)
            translation.diagnostics.pop(skip_id, None)
        except CompatError as exc:
            translation.opaque_types.clear()
            translation.opaque_types.update(opaque_before)
            if only is not None:
                raise
            reason = str(exc)
            translation.skipped[skip_id] = reason
            translation.diagnostics[skip_id] = _skip_diagnostic(
                skip_id, node_class_mappings[name], reason, exc.gate
            )
        except Exception as exc:  # noqa: BLE001 - INPUT_TYPES() runs pack code
            # A pack's INPUT_TYPES()/schema code can raise anything (version
            # skew against its ComfyUI install, plain bugs). In sweep mode
            # that is a per-node diagnostic, not a pack-fatal error.
            translation.opaque_types.clear()
            translation.opaque_types.update(opaque_before)
            if only is not None:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            translation.skipped[skip_id] = reason
            translation.diagnostics[skip_id] = _skip_diagnostic(
                skip_id, node_class_mappings[name], reason, None
            )
    return translation


__all__ = [
    "COMFY_TYPE_PREFIX",
    "PRIMITIVES",
    "CompatError",
    "CompatTranslation",
    "InputTypesProbe",
    "ListingObservation",
    "comfy_type_id",
    "translate_mappings",
    "translate_node",
    "translate_type",
]
