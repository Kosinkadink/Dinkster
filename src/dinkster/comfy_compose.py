"""ComfyUI compat workers as composable packs (DESIGN 3.8 meets 3.6).

Builds the PackSpecs that put the compat surface behind the same
compose_serving path as any native pack: the translated core surface as
one worker, the legacy custom-pack quarantine as another, both on the
ComfyUI install's own interpreter. This module never imports
dinkster_compat_comfy - it only locates its manifests and writes the
environment contract legacy.py documents; the bootstrap happens in the
children (hazard H5).

Provenance follows the agreed frontend contract: the compat layer itself
is the "comfy" pack (a truthful chip for translated core nodes), and each
legacy pack attributes as "comfy.<pack>" derived from the loading record -
the same directory names the operator configured, never anything a pack
claims about itself. A legacy node that matches no configured pack (never
expected; the quarantine namespaces every node it loads) falls back to
"comfy" rather than lying with a specific pack id.

Presentation for legacy packs is a two-layer story. Every unported pack
gets the DEFAULT legacy badge (shared mark + color declared here as host
data) so "this node is a legacy ComfyUI pack, not ported to Dinkster" is
visible at a glance. A pack that ships - or whose user drops in - a
``dinkster-pack.toml`` with ``[pack.presentation]`` in its directory
overrides that default wholesale with its own badge; the same file grows
``[pack.entry]`` later when the pack actually ports, so declaring a badge
today is the first step of the migration, not a dead end.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple, cast

from dinkster_assets import (
    ASSET_TYPE,
    SAVE_TARGET_TYPE,
    is_asset_kind,
    register_asset_type,
    register_save_target_type,
    register_video_value_type,
    resolver_from_env,
)
from dinkster_assets.audio import register_audio_value_type
from dinkster_server import PackInfo
from dinkster_values import (
    IMAGE_BATCH_MERGER_ID,
    IMAGE_FILE_DECODER_ID,
    PNG_CONTAINER_VERSION,
    TypeRegistry,
    decode_image_array,
    decode_image_file,
    decode_latent,
    encode_image_array,
    encode_latent,
    image_array_fingerprint,
    image_array_meta,
    image_input,
    latent_fingerprint,
    mask_array_meta,
    merge_image_batches,
    prepare_image_array_encoding,
    render_image_png,
    render_mask_png,
    validate_image_encoded,
    validate_latent_encoded,
)
from dinkster_workers import (
    SingleJobMultiGpuConfig,
    load_manifest,
    load_pack_assets,
    load_pack_blueprints,
    load_pack_presentation,
    load_pack_templates,
)
from dinkster_workers.interpreter import InterpreterPreflightError, preflight_interpreter

from .compose import CompositionError, PackSpec, cuda_vram_budgets, default_pack_spec
from .packs import blueprint_assets, pack_info_from_presentation, template_assets

__all__ = [
    "COMFY_AUDIO_TYPE",
    "COMFY_IMAGE_TYPE",
    "COMFY_MASK_TYPE",
    "COMFY_VIDEO_TYPE",
    "ComfyModelRoot",
    "comfy_compat_specs",
    "comfy_model_roots",
    "dinkster_pythonpath",
    "execution_python",
    "find_compat_manifest",
    "legacy_pack_info",
    "register_comfy_host_types",
]

COMFY_PACK_ID = "comfy"
_CORE_START_TIMEOUT = 180.0  # first torch + ComfyUI import is slow
_LEGACY_START_TIMEOUT = 300.0  # torch + ComfyUI + every legacy pack

COMFY_IMAGE_TYPE = "comfy.IMAGE"
"""The host-side spelling of the compat workers' image type. Kept as a
literal because this module must not import dinkster_compat_comfy; it must
match comfy_type_id("IMAGE") over there (translate.py's COMFY_TYPE_PREFIX
plus image.py's IMAGE_V1_NAME)."""

COMFY_MASK_TYPE = "comfy.MASK"
"""MASK shares the image-array byte contract with IMAGE (``[H, W]`` /
``[B, H, W]`` layouts), so ``comfy.MASK`` and ``dinkster.mask`` are one
value type; only the PNG renderer differs (render_mask_png)."""

COMFY_AUDIO_TYPE = "comfy.AUDIO"
COMFY_VIDEO_TYPE = "comfy.VIDEO"
COMFY_LATENT_TYPE = "comfy.LATENT"
_IMAGE_TYPE_EQUIVALENCE_PROVIDER = "dinkster.image-array-type-equivalence@1"
_COMFY_TYPE_EQUIVALENCES = (
    (COMFY_IMAGE_TYPE, "dinkster.image"),
    (COMFY_MASK_TYPE, "dinkster.mask"),
)


class ComfyModelRoot(NamedTuple):
    """One ordered category root from the isolated ComfyUI path probe."""

    mount_id: str
    category: str
    kind: str
    path: Path


class _ExecutionPythonSelection(NamedTuple):
    interpreter: str
    step: str


def register_comfy_host_types(registry: TypeRegistry) -> None:
    """The torch-free host halves of compat value contracts.

    The base asset atom is host-owned and uses the process resolver chain.
    In-process native nodes materialize assets through this registration when
    catalog-backed packs activate lazily after compat composition. Save targets
    are pure structured data, so the host and worker use the same codec.

    IMAGE uses numpy plus its existing PNG/typed-asset providers; MASK
    shares those bytes with a mask PNG renderer. AUDIO uses
    the pinned numpy mapping and VIDEO the pinned raw-container mapping.
    Compat workers register torch/upstream forms over the same bytes.
    Idempotent because core and legacy PackSpecs share this hook."""
    from dinkster_image_document.compat import register_comfy_compositor, register_comfy_layers

    register_comfy_layers(registry)
    register_comfy_compositor(registry)
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if SAVE_TARGET_TYPE not in registry:
        register_save_target_type(registry)
    if COMFY_IMAGE_TYPE not in registry:
        registry.register(
            COMFY_IMAGE_TYPE,
            encode=encode_image_array,
            decode=decode_image_array,
            prepare_buffer_encoding=prepare_image_array_encoding,
            fingerprint=image_array_fingerprint(COMFY_IMAGE_TYPE),
            meta=image_array_meta,
            input_convert=image_input,
            validate_encoded=validate_image_encoded,
            validate_encoded_buffer=validate_image_encoded,
        )
        registry.register_rendition(
            COMFY_IMAGE_TYPE,
            "png",
            mime="image/png",
            render=render_image_png,
            version=PNG_CONTAINER_VERSION,
        )
        # Typed assets (joint contract 2026-07-26): asset<comfy.IMAGE> decodes
        # and list<asset<comfy.IMAGE>> merges. Host and workers use the same
        # provider identities even though their runtime forms differ.
        registry.register_asset_decoder(
            COMFY_IMAGE_TYPE,
            provider_id=IMAGE_FILE_DECODER_ID,
            decode=decode_image_file,
        )
        registry.register_batch_merge(
            COMFY_IMAGE_TYPE,
            provider_id=IMAGE_BATCH_MERGER_ID,
            merge=merge_image_batches,
        )
    if COMFY_MASK_TYPE not in registry:
        registry.register(
            COMFY_MASK_TYPE,
            encode=encode_image_array,
            decode=decode_image_array,
            prepare_buffer_encoding=prepare_image_array_encoding,
            fingerprint=image_array_fingerprint(COMFY_MASK_TYPE),
            meta=mask_array_meta,
            input_convert=image_input,
            validate_encoded=validate_image_encoded,
            validate_encoded_buffer=validate_image_encoded,
        )
        registry.register_rendition(
            COMFY_MASK_TYPE,
            "png",
            mime="image/png",
            render=render_mask_png,
            version=PNG_CONTAINER_VERSION,
        )
    if COMFY_AUDIO_TYPE not in registry:
        register_audio_value_type(registry, COMFY_AUDIO_TYPE, resolver_from_env())
    if COMFY_VIDEO_TYPE not in registry:
        register_video_value_type(registry, COMFY_VIDEO_TYPE, resolver_from_env())
    if COMFY_LATENT_TYPE not in registry:
        registry.register(
            COMFY_LATENT_TYPE,
            encode=encode_latent,
            decode=decode_latent,
            fingerprint=latent_fingerprint(COMFY_LATENT_TYPE),
            validate_encoded=validate_latent_encoded,
        )
    for compat_type, native_type in _COMFY_TYPE_EQUIVALENCES:
        if compat_type in registry and native_type in registry:
            registry.register_type_equivalence(
                compat_type,
                native_type,
                provider_id=_IMAGE_TYPE_EQUIVALENCE_PROVIDER,
            )


# Declared host data, not synthesis: one shared badge marks every unported
# legacy pack (the per-pack identity within it is the displayName / the
# frontend's derived initials). The compat layer's own chip shares the
# color but not the mark, so "translated core surface" and "legacy custom
# pack" read differently at a glance.
_COMFY_COLOR = "#4a7ab5"
_COMFY_INFO = PackInfo(display_name="ComfyUI Compat", abbr="C1", color=_COMFY_COLOR)
_LEGACY_MARK = "\U0001f9e9"


def find_compat_manifest(name: str = "dinkster-pack.toml") -> Path:
    """Locate a dinkster-compat-comfy manifest WITHOUT importing the package
    (importing its entry modules would bootstrap ComfyUI in this process)."""
    spec = importlib.util.find_spec("dinkster_compat_comfy")
    if spec is None or spec.origin is None:
        raise CompositionError("dinkster-compat-comfy is not installed")
    package_dir = Path(spec.origin).parent  # .../src/dinkster_compat_comfy
    manifest = package_dir.parents[1] / name
    if not manifest.is_file():
        distribution = importlib.metadata.distribution("dinkster-compat-comfy")
        manifest = Path(str(distribution.locate_file(f"dinkster_compat_comfy_pack/{name}")))
    if not manifest.is_file():
        raise CompositionError(f"compat manifest not found at {manifest}")
    return manifest


def find_native_manifest() -> Path:
    """Locate the native provider manifest without importing execution bodies."""
    spec = importlib.util.find_spec("dinkster_native")
    if spec is None or spec.origin is None:
        raise CompositionError("dinkster-native is not installed")
    package_dir = Path(spec.origin).parent
    manifest = package_dir.parents[1] / "dinkster-pack.toml"
    if not manifest.is_file():
        distribution = importlib.metadata.distribution("dinkster-native")
        manifest = Path(str(distribution.locate_file("dinkster_native_pack/dinkster-pack.toml")))
    if not manifest.is_file():
        raise CompositionError(f"native manifest not found at {manifest}")
    return manifest


RETIRED_EXECUTION_PYTHON_ENV = "DINKSTER_COMFYUI_PYTHON"
RETIRED_EXECUTION_PYTHON_ENV_MESSAGE = (
    "DINKSTER_COMFYUI_PYTHON is retired; set DINKSTER_EXECUTION_PYTHON instead"
)
RETIRED_EXECUTION_PYTHON_FLAG_MESSAGE = "--comfy-python is retired; pass --execution-python instead"


def _reject_retired_execution_python_env() -> None:
    """The pre-rename environment variable is retired without an alias; a set
    but ignored variable would silently run the wrong interpreter."""
    if RETIRED_EXECUTION_PYTHON_ENV in os.environ:
        raise CompositionError(RETIRED_EXECUTION_PYTHON_ENV_MESSAGE)


def _execution_python_selection(
    comfy_root: Path, explicit: str | None = None
) -> _ExecutionPythonSelection:
    """Resolve the execution-arm interpreter and retain the selecting configuration step."""
    _reject_retired_execution_python_env()
    if explicit:
        return _ExecutionPythonSelection(explicit, "--execution-python")
    env = os.environ.get("DINKSTER_EXECUTION_PYTHON", "")
    if env:
        return _ExecutionPythonSelection(env, "DINKSTER_EXECUTION_PYTHON")
    venv_python = comfy_root / "venv" / "bin" / "python"
    if venv_python.exists():
        return _ExecutionPythonSelection(str(venv_python), "<comfy-root>/venv/bin/python")
    return _ExecutionPythonSelection(sys.executable, "current Python")


def execution_python(comfy_root: Path, explicit: str | None = None) -> str:
    """The interpreter used by legacy and compatibility children."""
    return _execution_python_selection(comfy_root, explicit).interpreter


_COMFY_REQUIREMENTS_SCRIPT = r"""
import importlib
import importlib.metadata
import json
import re
import sys

canonical = lambda value: re.sub(r"[-_.]+", "-", value).lower()
requirements = []
for raw in open(sys.argv[1], encoding="utf-8"):
    line = raw.partition("#")[0].strip()
    if not line:
        continue
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    if match is None:
        print(json.dumps({"invalid": line}, separators=(",", ":")))
        raise SystemExit(2)
    requirements.append(match.group(1))

by_distribution = {}
for module, distributions in importlib.metadata.packages_distributions().items():
    if module.startswith("_"):
        continue
    for distribution in distributions:
        by_distribution.setdefault(canonical(distribution), []).append(module)

for requirement in requirements:
    key = canonical(requirement)
    candidates = sorted(
        set(by_distribution.get(key, ())),
        key=lambda module: (canonical(module) != key, len(module), module),
    )
    module = candidates[0] if candidates else requirement.replace("-", "_").replace(".", "_")
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        print(json.dumps({"missing": exc.name or module}, separators=(",", ":")))
        raise SystemExit(1) from None
    except Exception:
        print(json.dumps({"missing": module}, separators=(",", ":")))
        raise SystemExit(1) from None
print("{}")
"""


def _probe_comfy_requirements(comfy_root: Path, selection: _ExecutionPythonSelection) -> None:
    requirements = comfy_root / "requirements.txt"
    if not requirements.is_file():
        return
    command = (
        selection.interpreter,
        "-I",
        "-c",
        _COMFY_REQUIREMENTS_SCRIPT,
        str(requirements),
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise CompositionError(
            f"ComfyUI requirements probe could not start interpreter "
            f"'{selection.interpreter}' selected by {selection.step}: {str(exc)[:400]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CompositionError(
            f"ComfyUI requirements probe timed out in interpreter "
            f"'{selection.interpreter}' selected by {selection.step}"
        ) from exc
    if completed.returncode == 0:
        return
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {}
    missing = payload.get("missing") if isinstance(payload, dict) else None
    if isinstance(missing, str) and missing:
        raise CompositionError(
            f"ComfyUI requirement module {missing!r} is unavailable in interpreter "
            f"'{selection.interpreter}' selected by {selection.step}"
        )
    detail = (completed.stderr.strip() or completed.stdout.strip()).replace("\n", " ")
    detail = detail[:400] or f"exit status {completed.returncode}"
    raise CompositionError(
        f"ComfyUI requirements probe failed in interpreter '{selection.interpreter}' "
        f"selected by {selection.step}: {detail}"
    )


def _probe_comfy_blake3(interpreter: str) -> None:
    command = (interpreter, "-I", "-c", "import blake3")
    remedy = f"{interpreter} -m pip install blake3"
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            shell=False,
        )
    except OSError as exc:
        detail = f"could not start: {str(exc)[:400]}"
        raise CompositionError(
            f"ComfyUI interpreter {interpreter!r} cannot import blake3 ({detail}); run `{remedy}`"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CompositionError(
            f"ComfyUI interpreter {interpreter!r} cannot import blake3 "
            f"(probe timed out); run `{remedy}`"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip()).replace("\n", " ")
        detail = detail[:400] or f"exit status {completed.returncode}"
        raise CompositionError(
            f"ComfyUI interpreter {interpreter!r} cannot import blake3 ({detail}); run `{remedy}`"
        )


def dinkster_pythonpath(manifest: Path) -> str:
    """The compat child imports Dinkster's pure-stdlib packages from source
    (the ComfyUI venv has torch but no dinkster packages). Empty when the
    manifest is not in a source checkout - an installed world puts the
    packages on the child's own path instead."""
    packages_dir = manifest.parents[1]
    src_dirs = sorted(str(p) for p in packages_dir.glob("*/src"))
    return os.pathsep.join(src_dirs)


def _decode_comfy_model_roots(payload: object) -> tuple[ComfyModelRoot, ...]:
    if not isinstance(payload, dict):
        raise CompositionError("ComfyUI model-root probe returned a non-object")
    rows: list[ComfyModelRoot] = []
    for category, raw_entry in sorted(cast("dict[object, object]", payload).items()):
        if not isinstance(category, str) or not category:
            raise CompositionError("ComfyUI model-root probe returned an invalid category")
        if not isinstance(raw_entry, dict):
            raise CompositionError(
                f"ComfyUI model-root probe returned invalid data for {category!r}"
            )
        entry = cast("dict[object, object]", raw_entry)
        if set(entry) != {"kind", "roots"}:
            raise CompositionError(
                f"ComfyUI model-root probe returned unexpected fields for {category!r}"
            )
        kind, raw_roots = entry["kind"], entry["roots"]
        if not isinstance(kind, str) or not is_asset_kind(kind):
            raise CompositionError(
                f"ComfyUI model-root probe returned invalid kind for {category!r}"
            )
        if not isinstance(raw_roots, list) or not all(
            isinstance(root, str) and root for root in raw_roots
        ):
            raise CompositionError(
                f"ComfyUI model-root probe returned invalid roots for {category!r}"
            )
        for index, root_text in enumerate(cast("list[str]", raw_roots), 1):
            root = Path(root_text)
            if not root.is_absolute():
                raise CompositionError(f"ComfyUI model root for {category!r} is not absolute")
            root = root.resolve()
            rows.append(
                ComfyModelRoot(
                    mount_id=f"comfy-model-{category.replace('_', '-')}-{index}",
                    category=category,
                    kind=kind,
                    path=root,
                )
            )
    return tuple(rows)


def comfy_model_roots(
    comfy_root: Path | str,
    *,
    python: str | None = None,
    comfy_args: tuple[str, ...] = (),
) -> tuple[ComfyModelRoot, ...]:
    """Probe live ``folder_paths`` roots in the ComfyUI interpreter.

    The engine never imports ComfyUI. A short isolated child initializes
    the same sanitized args and extra-model-path configs as the real compat
    worker, then returns only category/kind/root data as JSON.
    """
    root = Path(comfy_root).resolve()
    manifest = find_compat_manifest("dinkster-pack.toml")
    selection = _execution_python_selection(root, python)
    interpreter = selection.interpreter
    try:
        preflight_interpreter(interpreter)
    except InterpreterPreflightError as exc:
        raise CompositionError(f"ComfyUI model-root probe refused: {exc}") from exc
    _probe_comfy_requirements(root, selection)
    env = dict(os.environ)
    env["DINKSTER_COMFYUI_ROOT"] = str(root)
    pythonpath = dinkster_pythonpath(manifest)
    if pythonpath:
        inherited = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = pythonpath + (os.pathsep + inherited if inherited else "")
    command = (
        interpreter,
        "-c",
        "import json; from dinkster_compat_comfy.bootstrap import comfy_model_roots; "
        "print(json.dumps(comfy_model_roots(), separators=(',', ':')))",
        *comfy_args,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CompositionError(f"ComfyUI model-root probe failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CompositionError(f"ComfyUI model-root probe exited {completed.returncode}: {detail}")
    try:
        payload: object = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CompositionError("ComfyUI model-root probe returned invalid JSON") from exc
    return _decode_comfy_model_roots(payload)


def _legacy_pack_id(path: Path) -> str:
    """Mirror legacy.py's namespace rule: directory name, or file stem for
    single-file packs. Node types arrive as ``comfy.<pack id>.<v1 name>``."""
    name = path.stem if path.is_file() else path.name
    return f"{COMFY_PACK_ID}.{name}"


def legacy_pack_info(path: Path) -> PackInfo:
    """The packs-table entry for one unported legacy pack: its declared
    ``[pack.presentation]`` when the directory carries a ``dinkster-pack.toml``
    (author-shipped or user-dropped), otherwise the shared legacy default
    badge. Blueprints ride the same presentation-only on-ramp: a legacy
    pack can ship starter workflows without being a loadable Dinkster pack
    (its blueprints will reference ``comfy.<pack>.*`` node types, which
    resolve only when the compat worker loads the pack - valid data that
    cannot instantiate until then). Single-file packs have no directory
    of their own to declare in, so they always get the default."""
    name = path.stem if path.is_file() else path.name
    fallback = f"{name} (ComfyUI)"
    if path.is_dir():
        manifest_path = path / "dinkster-pack.toml"
        declared = load_pack_presentation(manifest_path, pack_name=fallback)
        blueprints = blueprint_assets(load_pack_blueprints(manifest_path))
        # Declared assets ride the same on-ramp: a legacy pack that today
        # downloads models with custom code can declare them as data
        # instead, and preflight takes over - digest-pinned, consented,
        # verified. Packaged sources resolve under the legacy pack id
        # (the spec's asset_roots maps it to this directory).
        assets = load_pack_assets(manifest_path, pack=_legacy_pack_id(path))
        # Templates too: starter workflows for old packs without porting
        # them. Their asset references validate against the same file's
        # [[pack.assets]] declarations.
        templates = template_assets(load_pack_templates(manifest_path, pack=_legacy_pack_id(path)))
        if declared is not None:
            info = pack_info_from_presentation(declared, fallback)
        else:
            info = PackInfo(display_name=fallback, mark=_LEGACY_MARK, color=_COMFY_COLOR)
        if blueprints:
            info = replace(info, blueprints=blueprints)
        if templates:
            info = replace(info, templates=templates)
        if assets:
            info = replace(info, assets=assets)
        return info
    return PackInfo(display_name=fallback, mark=_LEGACY_MARK, color=_COMFY_COLOR)


def comfy_compat_specs(
    comfy_root: Path | str | None = None,
    *,
    python: str | None = None,
    _requirements_checked: bool = False,
    legacy_packs: Sequence[Path | str] = (),
    comfy_nodes: Sequence[str] | None = None,
    asset_vault: Path | str | None = None,
    mounts_snapshot: Path | str | None = None,
    aimdo: str = "off",
    memory_budgets: Mapping[str, int] | None = None,
    reserve_vram: int | None = None,
    comfy_args: tuple[str, ...] = (),
    multi_device_cuda_indices: tuple[int, ...] = (),
    single_job_multi_gpu: SingleJobMultiGpuConfig | None = None,
) -> list[PackSpec]:
    """PackSpecs for the universal generation owner and ComfyUI compat surface.

    Without a root the provider uses native bodies and pinned import schemas.
    The generation schema owner precedes the execution provider.
    The legacy quarantine worker follows when ``legacy_packs``
    names custom pack directories (or single-file packs). ``comfy_nodes``
    optionally filters the v1 translation (DINKSTER_COMFY_NODES) - mostly a
    test/bring-up knob.
    """
    _reject_retired_execution_python_env()
    root = Path(comfy_root) if comfy_root is not None else None
    if root is not None and not root.is_dir():
        raise CompositionError(f"ComfyUI root not found: {root}")
    if legacy_packs and root is None:
        raise CompositionError("legacy packs require --comfy-root")
    core_manifest = find_native_manifest() if root is None else find_compat_manifest()
    generation_spec = default_pack_spec("dinkster-nodes-generation")
    if root is None:
        native_executes = load_manifest(core_manifest).executes
        generation_spec = replace(
            generation_spec,
            optional_execution=tuple(
                node_type
                for node_type in load_manifest(find_compat_manifest()).executes
                if node_type not in native_executes
            ),
        )
    selection = (
        _execution_python_selection(root, python)
        if root is not None
        else _ExecutionPythonSelection(
            python or os.environ.get("DINKSTER_EXECUTION_PYTHON") or sys.executable,
            (
                "--execution-python"
                if python
                else "DINKSTER_EXECUTION_PYTHON"
                if os.environ.get("DINKSTER_EXECUTION_PYTHON")
                else "current Python"
            ),
        )
    )
    interpreter = selection.interpreter
    try:
        preflight_interpreter(interpreter)
    except InterpreterPreflightError as exc:
        raise CompositionError(f"ComfyUI compat interpreter refused: {exc}") from exc
    if root is not None and not _requirements_checked:
        _probe_comfy_requirements(root, selection)
    _probe_comfy_blake3(interpreter)
    base_env = (
        {"DINKSTER_COMFYUI_ROOT": str(root), "DINKSTER_COMFY_NATIVE_ONLY": "0"}
        if root is not None
        else {"DINKSTER_COMFY_NATIVE_ONLY": "1", "DINKSTER_COMFYUI_ROOT": ""}
    )
    pythonpath = dinkster_pythonpath(core_manifest)
    if pythonpath:
        base_env["PYTHONPATH"] = pythonpath
    # ServingComposer maps each replica's child-local cuda:0 back to the
    # corresponding parent-visible index.
    vram_budgets = cuda_vram_budgets(memory_budgets or {})

    comfy_info = replace(
        _COMFY_INFO,
        comfy_aliases=load_manifest(find_compat_manifest()).comfy_aliases,
    )
    core_env = dict(base_env)
    if asset_vault:
        core_env["DINKSTER_ASSET_VAULT"] = str(asset_vault)
    if mounts_snapshot:
        # Runtime mounts reach the worker through the snapshot FILE, not
        # this env var's value: the var only names where to look, and the
        # worker's resolver re-reads it whenever the engine rewrites it -
        # so granting a folder mid-session needs no worker restart.
        core_env["DINKSTER_MOUNTS_SNAPSHOT"] = str(mounts_snapshot)
    if comfy_nodes is not None:
        core_env["DINKSTER_COMFY_NODES"] = ",".join(comfy_nodes)
    specs = [
        generation_spec,
        PackSpec(
            manifest=core_manifest,
            python=interpreter,
            env=core_env,
            aimdo=aimdo,
            vram_budgets=vram_budgets,
            reserve_vram=reserve_vram,
            comfy_args=comfy_args,
            runtime_settings=True,
            replica_cuda_indices=multi_device_cuda_indices,
            single_job_cuda_indices=(
                () if single_job_multi_gpu is None else single_job_multi_gpu.cuda_indices
            ),
            single_job_mode=("auto" if single_job_multi_gpu is None else single_job_multi_gpu.mode),
            start_timeout=_CORE_START_TIMEOUT,
            packs={COMFY_PACK_ID: comfy_info},
            attribute=lambda _node_type: COMFY_PACK_ID,
            # The compat manifests claim the reserved "comfy" root; the
            # host wiring them up IS the trust grant.
            trust_reserved=True,
            host_types=register_comfy_host_types,
        ),
    ]

    if legacy_packs:
        pack_paths: list[Path] = []
        table: dict[str, PackInfo] = {COMFY_PACK_ID: comfy_info}
        asset_roots: dict[str, Path] = {}
        for entry in legacy_packs:
            path = Path(entry)
            if not path.exists():
                raise CompositionError(f"legacy pack not found: {path}")
            pack_id = _legacy_pack_id(path)
            if pack_id in table and pack_id != COMFY_PACK_ID:
                raise CompositionError(f"two legacy packs derive the same pack id {pack_id!r}")
            table[pack_id] = legacy_pack_info(path)
            if path.is_dir():
                # Packaged asset declarations resolve against the legacy
                # pack's own directory (single-file packs have none).
                asset_roots[pack_id] = path
            pack_paths.append(path)
        # Longest prefix first: a pack named "a.b" must win over "a" for
        # node type "comfy.a.b.x" (dots in directory names are legal).
        prefixes = sorted((pid for pid in table if pid != COMFY_PACK_ID), key=len, reverse=True)

        def attribute_legacy(node_type: str) -> str:
            for pack_id in prefixes:
                if node_type.startswith(pack_id + "."):
                    return pack_id
            return COMFY_PACK_ID

        legacy_env = {
            **base_env,
            "DINKSTER_LEGACY_PACKS": os.pathsep.join(str(p) for p in pack_paths),
        }
        # Translated custom-pack model selectors consume the same AssetRefs
        # as core compat nodes. Give the quarantine worker the same read-only
        # resolver chain so it can verify a selected digest before converting
        # it back to a category-relative v1 filename.
        if asset_vault:
            legacy_env["DINKSTER_ASSET_VAULT"] = str(asset_vault)
        if mounts_snapshot:
            legacy_env["DINKSTER_MOUNTS_SNAPSHOT"] = str(mounts_snapshot)
        specs.append(
            PackSpec(
                manifest=find_compat_manifest("dinkster-legacy-pack.toml"),
                python=interpreter,
                env=legacy_env,
                aimdo=aimdo,
                vram_budgets=vram_budgets,
                reserve_vram=reserve_vram,
                comfy_args=comfy_args,
                runtime_settings=True,
                start_timeout=_LEGACY_START_TIMEOUT,
                packs=table,
                attribute=attribute_legacy,
                trust_reserved=True,
                asset_roots=asset_roots,
                host_types=register_comfy_host_types,
            )
        )
    return specs
