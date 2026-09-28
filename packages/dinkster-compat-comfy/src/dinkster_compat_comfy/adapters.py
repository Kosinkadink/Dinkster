"""Input adapters for natively-ported nodes: legacy v1 input shapes ->
native structured inputs, converted once at the prompt boundary.

A native port that changes a node's input contract (SaveImage's raw
``filename_prefix`` string becoming a structured ``dinkster.save_target``)
would otherwise break every existing API prompt naming the old input.
These adapters keep those prompts working - by CONVERTING the legacy
value into the native shape, never by teaching the engine to accept raw
strings. Unsafe legacy values (absolute paths, traversal, %date%
substitution patterns) are refused with anchored problems, not preserved:
bug-for-bug compatibility explicitly excludes the bugs that were holes.

Only compat submissions pass through here (compat_api hands
``COMFY_INPUT_ADAPTERS`` to translate_prompt); native graph submissions
never do.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Mapping

from dinkster_assets import AssetError, AssetRef, SaveTarget, normalize_guess_query
from dinkster_graph.model import Link

from .prompt import InputAdapter, PromptProblem

__all__ = [
    "COMFY_INPUT_ADAPTERS",
    "adapt_save_image_inputs",
    "make_load_checkpoint_adapter",
    "make_load_clip_adapter",
    "make_load_dual_clip_adapter",
    "make_load_diffusion_model_adapter",
    "make_load_image_adapter",
    "make_load_latent_adapter",
    "make_load_lora_adapter",
    "make_load_model_patch_adapter",
    "make_load_vae_adapter",
    "make_load_vision_adapter",
    "make_model_asset_inputs_adapter",
]

_PREFIX_INPUT = "filename_prefix"
_TARGET_INPUT = "target"
_LEGACY_OUTPUT_MOUNT = "comfy-output"
_DEFAULT_PREFIX = "ComfyUI"

_CUSTOM_COMBO_FAMILY = "options"
_CUSTOM_COMBO_OPTIONS = frozenset(f"option{index}" for index in range(1, 101))


def adapt_custom_combo_inputs(
    node_id: str, inputs: dict[str, object]
) -> tuple[dict[str, object], list[PromptProblem]]:
    """Nest the closed literal ``option1..N`` prompt surface into one family."""
    legacy: dict[str, object] = {}
    native: dict[str, object] = {}
    problems: list[PromptProblem] = []
    for input_id, value in inputs.items():
        native_prefix = _CUSTOM_COMBO_FAMILY + "."
        if input_id.startswith(native_prefix):
            option_id = input_id.removeprefix(native_prefix)
            target = native
        elif input_id.startswith("option"):
            option_id = input_id
            target = legacy
        else:
            continue
        if option_id not in _CUSTOM_COMBO_OPTIONS:
            problems.append(
                PromptProblem(
                    code="prompt.custom_combo.bad_option",
                    message=(
                        f"CustomCombo input {input_id!r} is not a canonical option1..option100 key"
                    ),
                    node_id=node_id,
                    input_id=input_id,
                )
            )
            continue
        target[input_id] = value
    native_option_ids = {
        input_id: input_id.removeprefix(_CUSTOM_COMBO_FAMILY + ".") for input_id in native
    }
    if legacy and native:
        problems.append(
            PromptProblem(
                code="prompt.custom_combo.conflict",
                message="CustomCombo mixes legacy optionN keys with native options.optionN keys",
                node_id=node_id,
                input_id=next(iter(legacy)),
            )
        )
    option_values = legacy or {
        option_id: native[input_id] for input_id, option_id in native_option_ids.items()
    }
    ordered = sorted(option_values, key=lambda name: int(name.removeprefix("option")))
    expected = [f"option{index}" for index in range(1, len(ordered) + 1)]
    if option_values and ordered != expected:
        problems.append(
            PromptProblem(
                code="prompt.custom_combo.noncontiguous",
                message="CustomCombo options must be contiguous option1..optionN",
                node_id=node_id,
                input_id=ordered[0] if ordered else "",
            )
        )
    for input_id, value in (*legacy.items(), *native.items()):
        if isinstance(value, Link):
            problems.append(
                PromptProblem(
                    code="prompt.custom_combo.linked_option",
                    message=f"CustomCombo input {input_id!r} must be a literal string, not a link",
                    node_id=node_id,
                    input_id=input_id,
                )
            )
        elif not isinstance(value, str):
            problems.append(
                PromptProblem(
                    code="prompt.custom_combo.bad_value",
                    message=f"CustomCombo input {input_id!r} must be a string",
                    node_id=node_id,
                    input_id=input_id,
                )
            )
    if problems or not legacy:
        return inputs, problems
    adapted = {key: value for key, value in inputs.items() if key not in legacy}
    adapted.update({f"{_CUSTOM_COMBO_FAMILY}.{key}": legacy[key] for key in ordered})
    return adapted, []


def adapt_save_image_inputs(
    node_id: str, inputs: dict[str, object]
) -> tuple[dict[str, object], list[PromptProblem]]:
    """``filename_prefix`` -> ``target`` on the derived ``comfy-output``
    mount - exactly where classic SaveImage always wrote, now through the
    guarded writer.

    Conversion handles literals only: a LINKED filename_prefix (some node
    computing the string at runtime) cannot become a structured target at
    translation time and is refused loudly rather than half-converted.
    Backslash separators are normalized to ``/`` (a Windows-idiom prefix
    is not a traversal attempt); absolute paths, ``.``/``..`` segments,
    and ``%...%`` substitution patterns are refused - the first two were
    never safe, and date/counter substitution is deliberately not ported
    (a native prefix is a literal path fragment)."""
    if _PREFIX_INPUT not in inputs:
        return inputs, []
    adapted = dict(inputs)
    raw = adapted.pop(_PREFIX_INPUT)
    if _TARGET_INPUT in adapted:
        return adapted, [
            PromptProblem(
                code="prompt.save_target.conflict",
                message=(
                    "node has both legacy 'filename_prefix' and native 'target'; send exactly one"
                ),
                node_id=node_id,
                input_id=_PREFIX_INPUT,
            )
        ]
    if isinstance(raw, Link):
        return adapted, [
            PromptProblem(
                code="prompt.save_target.linked",
                message=(
                    "legacy 'filename_prefix' is a link; a computed prefix "
                    "cannot be converted to a structured save target - "
                    "submit a native graph with a 'target' input instead"
                ),
                node_id=node_id,
                input_id=_PREFIX_INPUT,
            )
        ]
    if not isinstance(raw, str):
        return adapted, [
            PromptProblem(
                code="prompt.save_target.invalid",
                message=f"legacy 'filename_prefix' must be a string, got {type(raw).__name__}",
                node_id=node_id,
                input_id=_PREFIX_INPUT,
            )
        ]
    prefix = raw.replace("\\", "/") or _DEFAULT_PREFIX
    if "%" in prefix:
        return adapted, [
            PromptProblem(
                code="prompt.save_target.substitution",
                message=(
                    f"filename_prefix {raw!r} uses %...% substitution, which "
                    "is not ported; use a literal prefix"
                ),
                node_id=node_id,
                input_id=_PREFIX_INPUT,
            )
        ]
    try:
        target = SaveTarget(mount=_LEGACY_OUTPUT_MOUNT, prefix=prefix)
    except AssetError as exc:
        return adapted, [
            PromptProblem(
                code="prompt.save_target.invalid",
                message=f"filename_prefix {raw!r} cannot become a save target: {exc}",
                node_id=node_id,
                input_id=_PREFIX_INPUT,
            )
        ]
    adapted[_TARGET_INPUT] = target.to_wire()
    return adapted, []


_IMAGE_INPUT = "image"
_CKPT_INPUT = "ckpt_name"
_CHECKPOINT_INPUT = "checkpoint"
_LORA_NAME_INPUT = "lora_name"
_LORA_INPUT = "lora"
_MODEL_PATCH_NAME_INPUT = "name"
_MODEL_PATCH_INPUT = "model_patch"
_VAE_NAME_INPUT = "vae_name"
_VAE_INPUT = "vae"
_UNET_NAME_INPUT = "unet_name"
_DIFFUSION_MODEL_INPUT = "diffusion_model"
_CLIP_NAME_INPUT = "clip_name"
_TEXT_ENCODER_INPUT = "text_encoder"
_DUAL_CLIP_INPUTS = (
    ("clip_name1", "text_encoder1"),
    ("clip_name2", "text_encoder2"),
)
_VISION_ENCODER_INPUT = "vision_encoder"
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")

AssetPathResolver = Callable[[str], "AssetRef | None"]
"""Resolves a normalized '/'-separated path RELATIVE to the legacy
directory the adapter covers (LoadImage's input dir, or one model
category's ordered roots) to a digest-backed AssetRef, or None when
nothing is cataloged there. The host binds this to matching mount
catalogs; the adapter itself never sees a filesystem."""


def _legacy_relative_name(
    raw: object,
    *,
    node_id: str,
    input_id: str,
    code: str,
    legacy_kind: str,
    relative_to: str,
) -> tuple[str | None, list[PromptProblem]]:
    """Validate a legacy filename value into a normalized relative path.

    The legacy widgets named files RELATIVE to a well-known directory;
    absolute paths, drive letters, traversal, and non-strings were never
    valid names there, so they refuse loudly (before any catalog lookup)
    rather than converting into something the prompt never meant."""
    if not isinstance(raw, str):
        return None, [
            PromptProblem(
                code=code,
                message=(
                    f"legacy '{input_id}' must be a filename string, got {type(raw).__name__}"
                ),
                node_id=node_id,
                input_id=input_id,
            )
        ]
    if raw.replace("\\", "/").startswith("/") or _WINDOWS_DRIVE.match(raw):
        return None, [
            PromptProblem(
                code=code,
                message=(
                    f"{input_id} {raw!r} is an absolute path; legacy "
                    f"{legacy_kind} names are relative to {relative_to}"
                ),
                node_id=node_id,
                input_id=input_id,
            )
        ]
    relative = normalize_guess_query(raw)
    if not relative:
        return None, [
            PromptProblem(
                code=code,
                message=(
                    f"{input_id} {raw!r} is not a plain relative path (empty or contains '..')"
                ),
                node_id=node_id,
                input_id=input_id,
            )
        ]
    return relative, []


def make_load_image_adapter(resolve: AssetPathResolver) -> InputAdapter:
    """Build the ``dinkster.load_image`` adapter: legacy ``image`` filename
    strings -> digest-backed asset wire values, resolved through the
    host-supplied ``resolve`` at the prompt boundary.

    Classic LoadImage's ``image`` widget named a file relative to the
    input directory; the native port takes an ``asset<dinkster.image>``.
    Conversion is EXACT-path only against what ``resolve`` can see (the derived
    ``comfy-input`` mount): the legacy semantic was an exact relative
    path, and a name-similarity fallback would silently substitute
    same-named content - digest identity stays the only authority. A
    filename that does not resolve is refused with an anchored problem
    pointing at POST /api/assets/guess, where the client can pick a
    ranked candidate explicitly.

    Already-native values pass through untouched: a link (some node
    producing the asset) and a mapping (asset wire form) are not legacy
    shapes. Backslash separators normalize to '/'; absolute paths and
    traversal are refused - they were never valid input names."""

    def adapt(
        node_id: str, inputs: dict[str, object]
    ) -> tuple[dict[str, object], list[PromptProblem]]:
        if _IMAGE_INPUT not in inputs:
            return inputs, []
        raw = inputs[_IMAGE_INPUT]
        if isinstance(raw, Link | Mapping):
            return inputs, []
        relative, problems = _legacy_relative_name(
            raw,
            node_id=node_id,
            input_id=_IMAGE_INPUT,
            code="prompt.load_image.invalid",
            legacy_kind="image",
            relative_to="the input directory",
        )
        if relative is None:
            return inputs, problems
        ref = resolve(relative)
        if ref is None:
            return inputs, [
                PromptProblem(
                    code="prompt.load_image.unresolved",
                    message=(
                        f"image {raw!r} is not cataloged on the "
                        "'comfy-input' mount; resolve the name to a digest "
                        "via POST /api/assets/guess and submit a native "
                        "asset input"
                    ),
                    node_id=node_id,
                    input_id=_IMAGE_INPUT,
                )
            ]
        adapted = dict(inputs)
        adapted[_IMAGE_INPUT] = ref.to_wire()
        return adapted, []

    return adapt


def make_load_latent_adapter(resolve: AssetPathResolver) -> InputAdapter:
    """Lower classic latent inputs to native assets, resolving output filenames."""

    def adapt(
        node_id: str, inputs: dict[str, object]
    ) -> tuple[dict[str, object], list[PromptProblem]]:
        if "latent" not in inputs:
            return inputs, []
        if "asset" in inputs:
            return inputs, [
                PromptProblem(
                    code="prompt.load_latent.invalid",
                    message="LoadLatent cannot supply both latent and asset inputs",
                    node_id=node_id,
                    input_id="latent",
                )
            ]
        raw = inputs["latent"]
        adapted = dict(inputs)
        del adapted["latent"]
        if raw is None or isinstance(raw, Link | Mapping | AssetRef):
            adapted["asset"] = raw.to_wire() if isinstance(raw, AssetRef) else raw
            return adapted, []
        relative, problems = _legacy_relative_name(
            raw,
            node_id=node_id,
            input_id="latent",
            code="prompt.load_latent.invalid",
            legacy_kind="latent",
            relative_to="the output directory",
        )
        if relative is None:
            return inputs, problems
        ref = resolve(relative)
        if ref is None:
            return inputs, [
                PromptProblem(
                    code="prompt.load_latent.unresolved",
                    message=f"latent {raw!r} is not cataloged on the 'comfy-output' mount",
                    node_id=node_id,
                    input_id="latent",
                )
            ]
        adapted["asset"] = ref.to_wire()
        return adapted, []

    return adapt


def _make_model_name_adapter(
    resolve: AssetPathResolver,
    *,
    legacy_input: str,
    native_input: str,
    code_prefix: str,
    legacy_kind: str,
    subtree: str,
    refuse_unported: Callable[[str], str | None] | None = None,
) -> InputAdapter:
    """The shared model-loader adapter shape (the CheckpointLoaderSimple
    recipe): a legacy ``<name>`` filename string relative to a models
    subtree becomes a digest-backed asset wire value on the RENAMED
    native input.

    Because the input renames, already-native shapes never appear under
    the legacy key: a mapping or link there is refused, not passed
    through - native values belong on ``native_input``. Conversion is
    EXACT-path only against what ``resolve`` can see in the category's
    ordered derived roots; a name that does not
    resolve is refused with an anchored problem pointing at POST
    /api/assets/guess, never silently substituted by name similarity -
    digest identity stays the only authority.

    ``refuse_unported`` lets a loader refuse legacy names that were never
    files at all (VAELoader's taesd/pixel_space arms) with an honest
    message instead of a misleading not-cataloged refusal."""

    def adapt(
        node_id: str, inputs: dict[str, object]
    ) -> tuple[dict[str, object], list[PromptProblem]]:
        if legacy_input not in inputs:
            return inputs, []
        adapted = dict(inputs)
        raw = adapted.pop(legacy_input)
        if native_input in adapted:
            return adapted, [
                PromptProblem(
                    code=f"{code_prefix}.conflict",
                    message=(
                        f"node has both legacy '{legacy_input}' and native "
                        f"'{native_input}'; send exactly one"
                    ),
                    node_id=node_id,
                    input_id=legacy_input,
                )
            ]
        if isinstance(raw, Link | Mapping):
            return adapted, [
                PromptProblem(
                    code=f"{code_prefix}.invalid",
                    message=(
                        f"legacy '{legacy_input}' converts literal filename "
                        "strings only; a computed or already-native value "
                        f"belongs on the native '{native_input}' asset input"
                    ),
                    node_id=node_id,
                    input_id=legacy_input,
                )
            ]
        relative, problems = _legacy_relative_name(
            raw,
            node_id=node_id,
            input_id=legacy_input,
            code=f"{code_prefix}.invalid",
            legacy_kind=legacy_kind,
            relative_to=f"the {subtree} model directory",
        )
        if relative is None:
            return adapted, problems
        if refuse_unported is not None:
            unported = refuse_unported(relative)
            if unported is not None:
                return adapted, [
                    PromptProblem(
                        code=f"{code_prefix}.unported",
                        message=unported,
                        node_id=node_id,
                        input_id=legacy_input,
                    )
                ]
        ref = resolve(relative)
        if ref is None:
            return adapted, [
                PromptProblem(
                    code=f"{code_prefix}.unresolved",
                    message=(
                        f"{legacy_kind} {raw!r} is not cataloged in the "
                        f"derived ComfyUI {subtree} roots; resolve "
                        "the name to a digest via POST /api/assets/guess "
                        "and submit a native asset input"
                    ),
                    node_id=node_id,
                    input_id=legacy_input,
                )
            ]
        adapted[native_input] = ref.to_wire()
        return adapted, []

    return adapt


def make_model_asset_inputs_adapter(
    resolvers: Mapping[str, tuple[AssetPathResolver, str]],
) -> InputAdapter:
    """Build an adapter for translated model-file inputs now typed as assets.

    Native graphs and the frontend already submit AssetRef wire mappings;
    legacy Comfy API prompts still submit category-relative filename
    strings. Resolve only those strings through the ordered derived roots
    assigned to each input's category. Links and mappings pass through unchanged,
    while malformed or uncataloged names refuse before graph validation.
    ``category`` is carried only for anchored diagnostics; the caller binds
    each resolver to that category's finite registry entry."""

    def adapt(
        node_id: str, inputs: dict[str, object]
    ) -> tuple[dict[str, object], list[PromptProblem]]:
        adapted = dict(inputs)
        problems: list[PromptProblem] = []
        for input_id, (resolve, category) in resolvers.items():
            if input_id not in adapted:
                continue
            raw = adapted[input_id]
            if isinstance(raw, Link | Mapping):
                continue
            relative, invalid = _legacy_relative_name(
                raw,
                node_id=node_id,
                input_id=input_id,
                code="prompt.model_asset.invalid",
                legacy_kind=f"{category} model",
                relative_to=f"the {category} model directory",
            )
            if relative is None:
                problems.extend(invalid)
                continue
            ref = resolve(relative)
            if ref is None:
                problems.append(
                    PromptProblem(
                        code="prompt.model_asset.unresolved",
                        message=(
                            f"{category} model {raw!r} is not cataloged in its "
                            "derived ComfyUI category roots; resolve the name to "
                            "a digest via POST /api/assets/guess and submit an "
                            "asset input"
                        ),
                        node_id=node_id,
                        input_id=input_id,
                    )
                )
                continue
            adapted[input_id] = ref.to_wire()
        return adapted, problems

    return adapt


def make_load_checkpoint_adapter(resolve: AssetPathResolver) -> InputAdapter:
    """Build the ``dinkster.load_checkpoint`` adapter: legacy ``ckpt_name``
    filename strings -> digest-backed asset wire values on the native
    ``checkpoint`` input, resolved through the host-supplied ``resolve``
    at the prompt boundary. The first instance of the model-loader
    recipe; see ``_make_model_name_adapter`` for the shared semantics."""
    return _make_model_name_adapter(
        resolve,
        legacy_input=_CKPT_INPUT,
        native_input=_CHECKPOINT_INPUT,
        code_prefix="prompt.load_checkpoint",
        legacy_kind="checkpoint",
        subtree="checkpoints",
    )


def make_load_lora_adapter(resolve: AssetPathResolver) -> InputAdapter:
    """Build the shared ``dinkster.load_lora`` / ``dinkster.load_lora_model_only``
    adapter: legacy ``lora_name`` filename strings -> digest-backed asset
    wire values on the native ``lora`` input. Both LoRA ports carry the
    same legacy input id, so one adapter serves both node types."""
    return _make_model_name_adapter(
        resolve,
        legacy_input=_LORA_NAME_INPUT,
        native_input=_LORA_INPUT,
        code_prefix="prompt.load_lora",
        legacy_kind="lora",
        subtree="loras",
    )


def make_load_model_patch_adapter(resolve: AssetPathResolver) -> InputAdapter:
    return _make_model_name_adapter(
        resolve,
        legacy_input=_MODEL_PATCH_NAME_INPUT,
        native_input=_MODEL_PATCH_INPUT,
        code_prefix="prompt.load_model_patch",
        legacy_kind="model patch",
        subtree="model_patches",
    )


# VAELoader's legacy combo mixed three vocabularies: real files under the
# vae model directory (ported), image-TAE names assembled from SEPARATE
# encoder/decoder files under vae_approx (taesd/taesdxl/taesd3/taef1/
# taef2), video-TAE files under vae_approx, and the "pixel_space"
# sentinel that never touched disk. Image and video TAEs cannot become
# one digest and are refused honestly rather than reported as
# not-cataloged. The name lists mirror the reference's
# VAELoader.image_taes / video_taes at 947c2749.
_IMAGE_TAES = ("taesd", "taesdxl", "taesd3", "taef1", "taef2")
_VIDEO_TAES = ("taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2")


def _unported_vae_name(relative: str) -> str | None:
    # Reference dispatch, mirrored exactly: pixel_space and the image
    # TAEs match the whole name, the video TAEs match the extension-less
    # stem (splitext) - so a real file like taehv2.safetensors still
    # resolves under vae/ exactly as v1 loads it.
    if relative in _IMAGE_TAES or posixpath.splitext(relative)[0] in _VIDEO_TAES:
        return (
            f"vae {relative!r} names VAELoader's taesd/vae_approx arm, "
            "which does not resolve to a single file "
            "in the ComfyUI vae roots and is not ported; "
            "submit a native asset input naming a VAE file"
        )
    return None


def make_load_vae_adapter(resolve: AssetPathResolver) -> InputAdapter:
    """Build the ``dinkster.load_vae`` adapter: legacy ``vae_name`` filename
    strings -> digest-backed asset wire values on the native ``vae``
    input. The ``pixel_space`` sentinel maps to the native synthetic-codec
    selector; composite taesd arms refuse as unported."""
    file_adapter = _make_model_name_adapter(
        resolve,
        legacy_input=_VAE_NAME_INPUT,
        native_input=_VAE_INPUT,
        code_prefix="prompt.load_vae",
        legacy_kind="vae",
        subtree="vae",
        refuse_unported=_unported_vae_name,
    )

    def adapt(
        node_id: str, inputs: dict[str, object]
    ) -> tuple[dict[str, object], list[PromptProblem]]:
        if inputs.get(_VAE_NAME_INPUT) != "pixel_space":
            return file_adapter(node_id, inputs)
        adapted = dict(inputs)
        adapted.pop(_VAE_NAME_INPUT)
        if _VAE_INPUT in adapted:
            return adapted, [
                PromptProblem(
                    code="prompt.load_vae.conflict",
                    message=("node has both legacy 'vae_name' and native 'vae'; send exactly one"),
                    node_id=node_id,
                    input_id=_VAE_NAME_INPUT,
                )
            ]
        adapted["pixel_space"] = True
        return adapted, []

    return adapt


def make_load_diffusion_model_adapter(resolve: AssetPathResolver) -> InputAdapter:
    """Build the ``dinkster.load_diffusion_model`` adapter: legacy
    ``unet_name`` filename strings -> digest-backed asset wire values on
    the native ``diffusion_model`` input. ``weight_dtype`` passes through
    untouched - the native port keeps that input."""
    return _make_model_name_adapter(
        resolve,
        legacy_input=_UNET_NAME_INPUT,
        native_input=_DIFFUSION_MODEL_INPUT,
        code_prefix="prompt.load_diffusion_model",
        legacy_kind="diffusion model",
        subtree="diffusion_models",
    )


def make_load_clip_adapter(resolve: AssetPathResolver) -> InputAdapter:
    """Build the ``dinkster.load_clip`` adapter: legacy ``clip_name``
    filename strings -> digest-backed asset wire values on the native
    ``text_encoder`` input. The legacy ``type`` and ``device`` selectors
    pass through untouched - the native port keeps both inputs."""
    return _make_model_name_adapter(
        resolve,
        legacy_input=_CLIP_NAME_INPUT,
        native_input=_TEXT_ENCODER_INPUT,
        code_prefix="prompt.load_clip",
        legacy_kind="text encoder",
        subtree="text_encoders",
    )


def make_load_dual_clip_adapter(resolve: AssetPathResolver) -> InputAdapter:
    """Convert both DualCLIPLoader filenames to ordered text encoder assets."""
    field_adapters = tuple(
        _make_model_name_adapter(
            resolve,
            legacy_input=legacy_input,
            native_input=native_input,
            code_prefix=f"prompt.load_dual_clip.{legacy_input}",
            legacy_kind="text encoder",
            subtree="text_encoders",
        )
        for legacy_input, native_input in _DUAL_CLIP_INPUTS
    )

    def adapt(
        node_id: str, inputs: dict[str, object]
    ) -> tuple[dict[str, object], list[PromptProblem]]:
        adapted = inputs
        problems: list[PromptProblem] = []
        for field_adapter in field_adapters:
            adapted, field_problems = field_adapter(node_id, adapted)
            problems.extend(field_problems)
        return adapted, problems

    return adapt


def make_load_vision_adapter(resolve: AssetPathResolver) -> InputAdapter:
    """Convert CLIPVisionLoader filenames to generic vision assets."""

    return _make_model_name_adapter(
        resolve,
        legacy_input=_CLIP_NAME_INPUT,
        native_input=_VISION_ENCODER_INPUT,
        code_prefix="prompt.load_vision",
        legacy_kind="vision encoder",
        subtree="clip_vision",
    )


def _make_legacy_combo_adapter(
    class_name: str,
    code_prefix: str,
    choice_id: str,
    choice: str,
    renames: Mapping[str, str],
) -> InputAdapter:
    """Nest one legacy flat conditioning class under its merged native
    node's combo option: pin the combo choice and move each legacy input
    to its dotted per-option id. Keyed by the submitted class_type, so
    sibling classes folded into the same native node each get their own
    option. Native keys on a legacy class_type refuse loudly - the prompt
    mixes two vocabularies and its intent is ambiguous."""

    def adapt(
        node_id: str, inputs: dict[str, object]
    ) -> tuple[dict[str, object], list[PromptProblem]]:
        conflicts = sorted(key for key in (choice_id, *renames.values()) if key in inputs)
        if conflicts:
            return inputs, [
                PromptProblem(
                    code=f"{code_prefix}.conflict",
                    message=(
                        f"{class_name} takes legacy inputs only; native keys "
                        f"({', '.join(conflicts)}) belong on the native node type"
                    ),
                    node_id=node_id,
                    input_id=conflicts[0],
                )
            ]
        adapted = dict(inputs)
        adapted[choice_id] = choice
        for legacy, native in renames.items():
            if legacy in adapted:
                adapted[native] = adapted.pop(legacy)
        return adapted, []

    return adapt


def adapt_create_video_inputs(
    _node_id: str, inputs: dict[str, object]
) -> tuple[dict[str, object], list[PromptProblem]]:
    if "codec" in inputs:
        inputs = {name: value for name, value in inputs.items() if name != "codec"}
    bit_depth = inputs.get("bit_depth")
    if type(bit_depth) is not int:
        return inputs, []
    adapted = dict(inputs)
    adapted["bit_depth"] = str(bit_depth)
    return adapted, []


def adapt_comfy_switch_inputs(
    _node_id: str, inputs: dict[str, object]
) -> tuple[dict[str, object], list[PromptProblem]]:
    if "switch" not in inputs:
        return inputs, []
    adapted = dict(inputs)
    adapted["condition"] = adapted.pop("switch")
    return adapted, []


def adapt_save_video_inputs(
    node_id: str, inputs: dict[str, object]
) -> tuple[dict[str, object], list[PromptProblem]]:
    adapted, problems = adapt_save_image_inputs(node_id, inputs)
    if problems:
        return adapted, problems
    for input_id in ("format.codec", "codec"):
        if adapted.get(input_id) == "auto":
            adapted.pop(input_id)
    if adapted.get("format") == "auto":
        adapted.pop("format")
    return adapted, []


_LEGACY_AREA_RENAMES = {name: f"units.{name}" for name in ("width", "height", "x", "y")}
_LEGACY_VIDEO_AREA_RENAMES = {
    name: f"units.{name}" for name in ("width", "height", "temporal", "x", "y", "z")
}

COMFY_INPUT_ADAPTERS: dict[str, InputAdapter] = {
    "comfy.CustomCombo": adapt_custom_combo_inputs,
    "CreateVideo": adapt_create_video_inputs,
    "comfy.CreateVideo": adapt_create_video_inputs,
    "dinkster.video.assemble": adapt_create_video_inputs,
    "SaveVideo": adapt_save_video_inputs,
    "comfy.SaveVideo": adapt_save_video_inputs,
    "dinkster.save_video": adapt_save_video_inputs,
    "ComfySwitchNode": adapt_comfy_switch_inputs,
    "comfy.ComfySwitchNode": adapt_comfy_switch_inputs,
    "dinkster.value.select": adapt_comfy_switch_inputs,
    "dinkster.save_image": adapt_save_image_inputs,
    "ConditioningCombine": _make_legacy_combo_adapter(
        "ConditioningCombine",
        "prompt.conditioning_merge",
        "mode",
        "combine",
        {f"conditioning_{index}": f"mode.inputs.conditioning_{index}" for index in range(1, 9)},
    ),
    "ConditioningAverage": _make_legacy_combo_adapter(
        "ConditioningAverage",
        "prompt.conditioning_merge",
        "mode",
        "average",
        {
            "conditioning_to": "mode.conditioning_to",
            "conditioning_from": "mode.conditioning_from",
            "conditioning_to_strength": "mode.conditioning_to_strength",
        },
    ),
    "ConditioningConcat": _make_legacy_combo_adapter(
        "ConditioningConcat",
        "prompt.conditioning_merge",
        "mode",
        "concat",
        {
            "conditioning_to": "mode.conditioning_to",
            "conditioning_from": "mode.conditioning_from",
        },
    ),
    "ConditioningSetArea": _make_legacy_combo_adapter(
        "ConditioningSetArea",
        "prompt.conditioning_set_area",
        "units",
        "pixels",
        _LEGACY_AREA_RENAMES,
    ),
    "ConditioningSetAreaPercentage": _make_legacy_combo_adapter(
        "ConditioningSetAreaPercentage",
        "prompt.conditioning_set_area",
        "units",
        "percent",
        _LEGACY_AREA_RENAMES,
    ),
    "ConditioningSetAreaPercentageVideo": _make_legacy_combo_adapter(
        "ConditioningSetAreaPercentageVideo",
        "prompt.conditioning_set_area",
        "units",
        "percent-video",
        _LEGACY_VIDEO_AREA_RENAMES,
    ),
}
"""The static (context-free) adapter table. Keys are either resolved node
types or submitted legacy class_type names (translate_prompt prefers the
class_type match). The compat endpoint extends this with contextual
adapters - ``make_load_image_adapter`` and ``make_load_checkpoint_adapter``
bound to the host's mount catalogs - before handing the table to
translate_prompt."""
