"""Import a real ComfyUI installation and translate its nodes.

This module is only ever executed inside a compat worker child process
(hazard H5: the engine's interpreter never imports ComfyUI, torch, or any
v1 node code). The child is expected to run on the ComfyUI install's own
interpreter (its venv has torch and friends); Dinkster's pure-stdlib packages
ride in via PYTHONPATH.

Environment contract:

- ``DINKSTER_COMFYUI_ROOT``: path to the ComfyUI checkout/install (required).
- ``DINKSTER_COMFY_NODES``: optional comma-separated v1 node names to
  translate; unset translates everything in NODE_CLASS_MAPPINGS.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import itertools
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .native_residency import select_load_device
from .translate import (
    MODEL_FILE_CATEGORIES,
    CompatError,
    CompatTranslation,
    InputTypesProbe,
    ListingObservation,
    translate_mappings,
)

_comfy_args_initialized: tuple[str, ...] | None = None
_comfy_paths_initialized = False


def _comfy_root_on_path() -> Path:
    root_text = os.environ.get("DINKSTER_COMFYUI_ROOT", "")
    if not root_text:
        raise CompatError(
            "DINKSTER_COMFYUI_ROOT is not set; the compat pack needs a ComfyUI "
            "installation to translate"
        )
    root = Path(root_text)
    if not (root / "folder_paths.py").is_file():
        raise CompatError(f"not a ComfyUI installation (no folder_paths.py): {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def initialize_comfy_args() -> object:
    """Parse only the worker host's sanitized ComfyUI argv, exactly once."""
    global _comfy_args_initialized
    supplied = tuple(sys.argv[1:])
    if _comfy_args_initialized is not None:
        if supplied != _comfy_args_initialized:
            raise CompatError(
                "ComfyUI arguments changed after cli_args initialization: "
                f"{supplied!r} != {_comfy_args_initialized!r}"
            )
        return importlib.import_module("comfy.cli_args")
    if "comfy.cli_args" in sys.modules:
        raise CompatError(
            "ComfyUI cli_args was imported before Dinkster could initialize its "
            "explicit worker arguments"
        )
    options = importlib.import_module("comfy.options")
    enable = getattr(options, "enable_args_parsing", None)
    if not callable(enable):
        raise CompatError("ComfyUI comfy.options has no enable_args_parsing hook")
    enable(True)
    try:
        cli_args = importlib.import_module("comfy.cli_args")
    except SystemExit as exc:
        rendered = ", ".join(repr(argument) for argument in supplied) or "<empty>"
        raise CompatError(f"ComfyUI rejected supplied argument list: {rendered}") from exc
    _comfy_args_initialized = supplied
    return cli_args


def _initialize_comfy_device() -> None:
    """Select CPU before ComfyUI's import-time memory policy probes CUDA.

    Resolve against the worker's torch build, not the engine host. The same
    DINKSTER_ACCELERATOR selection governs native execution and remote workers.
    Path-only probes stay torch-free because they never call this function.
    """
    cli_args = cast(Any, initialize_comfy_args())
    torch = importlib.import_module("torch")
    device = select_load_device(torch)
    if device.type == "cpu":
        cli_args.args.cpu = True


def initialize_comfy_paths() -> object:
    """Apply ComfyUI's model-path startup contract exactly once.

    The compat worker does not import ``main.py`` because that module owns
    the web server and process startup. Reproduce only ``apply_custom_paths``
    after ComfyUI has parsed the worker's sanitized argv, including default
    and explicit extra-model-path configs and output model directories.
    """
    global _comfy_paths_initialized
    root = _comfy_root_on_path()
    cli_args = initialize_comfy_args()
    folder_paths = importlib.import_module("folder_paths")
    if _comfy_paths_initialized:
        return folder_paths
    args = getattr(cli_args, "args", None)
    if args is None:
        raise CompatError("ComfyUI cli_args exposes no parsed args object")
    default_config = root / "extra_model_paths.yaml"
    configured = getattr(args, "extra_model_paths_config", None)
    configs = [str(default_config)] if default_config.is_file() else []
    if configured:
        configs.extend(itertools.chain.from_iterable(configured))
    if configs:
        extra_config = importlib.import_module("utils.extra_config")
        load_extra = getattr(extra_config, "load_extra_path_config", None)
        if not callable(load_extra):
            raise CompatError("ComfyUI utils.extra_config exposes no path loader")
        for config_path in configs:
            load_extra(config_path)

    output_directory = getattr(args, "output_directory", None)
    if output_directory:
        folder_paths.set_output_directory(os.path.abspath(output_directory))
    output_root = folder_paths.get_output_directory()
    for category, subdir in (
        ("checkpoints", "checkpoints"),
        ("clip", "clip"),
        ("vae", "vae"),
        ("diffusion_models", "diffusion_models"),
        ("loras", "loras"),
    ):
        folder_paths.add_model_folder_path(category, os.path.join(output_root, subdir))
    input_directory = getattr(args, "input_directory", None)
    if input_directory:
        folder_paths.set_input_directory(os.path.abspath(input_directory))
    temp_directory = getattr(args, "temp_directory", None)
    if temp_directory:
        folder_paths.set_temp_directory(os.path.abspath(temp_directory))
    user_directory = getattr(args, "user_directory", None)
    if user_directory:
        folder_paths.set_user_directory(os.path.abspath(user_directory))
    _comfy_paths_initialized = True
    return folder_paths


def comfy_model_roots() -> dict[str, dict[str, object]]:
    """The exact live roots and semantic kinds for approved categories.

    Called both in workers and by the host's isolated root probe. Host path
    authority therefore comes from the same initialized ``folder_paths``
    table that translated execution uses, including extra_model_paths.
    """
    folder_paths = initialize_comfy_paths()
    get_roots = getattr(folder_paths, "get_folder_paths", None)
    if not callable(get_roots):
        raise CompatError("ComfyUI folder_paths exposes no get_folder_paths")
    result: dict[str, dict[str, object]] = {}
    for category, descriptor in MODEL_FILE_CATEGORIES.items():
        try:
            raw: object = get_roots(category)
        except KeyError:
            # Older ComfyUI installs do not define categories introduced by
            # later first-party nodes. No selector on that install can have
            # observed such a listing, so it has no live root to publish.
            continue
        if not isinstance(raw, (list, tuple)):
            raise CompatError(f"folder_paths returned invalid roots for category {category!r}")
        items = cast("Sequence[object]", raw)
        if not all(isinstance(root, str) and root for root in items):
            raise CompatError(f"folder_paths returned invalid roots for category {category!r}")
        roots = tuple(
            dict.fromkeys(str(Path(root).resolve()) for root in cast("Sequence[str]", raw))
        )
        result[category] = {"kind": descriptor.kind, "roots": roots}
    return result


def _await_sync(value: object) -> None:
    """Run a maybe-coroutine to completion from synchronous code.

    ComfyUI's init_extra_nodes is a plain function in some versions and a
    coroutine in others. Pack loading happens synchronously inside the
    worker host's already-running event loop, so a nested asyncio.run()
    here would blow up; a private loop on a short-lived thread runs the
    coroutine without touching the host's loop."""
    if not inspect.iscoroutine(value):
        return
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            asyncio.run(value)
        except BaseException as exc:  # noqa: BLE001 - reraised below
            failure.append(exc)

    thread = threading.Thread(target=runner, name="dinkster-comfy-bootstrap")
    thread.start()
    thread.join()
    if failure:
        raise failure[0]


def ensure_prompt_server() -> object | None:
    """Construct ComfyUI's real PromptServer, headless, if none exists yet.

    Idempotent; returns the instance or None when server.py itself cannot
    load (a ComfyUI too old or too broken to have one - callers treat that
    as "no route surface to observe", not an error)."""
    try:
        server_module = importlib.import_module("server")
    except Exception:
        return None
    prompt_server = getattr(server_module, "PromptServer", None)
    if prompt_server is None:
        return None
    instance = getattr(prompt_server, "instance", None)
    if instance is not None:
        return instance
    try:
        return prompt_server(asyncio.new_event_loop())
    except Exception:
        return None


def bootstrap_comfyui() -> object:
    """Put the ComfyUI install on sys.path, import its ``nodes`` module, and
    initialize the deterministic core surface (comfy_extras; no custom or
    API nodes). Idempotent per process; returns the ``nodes`` module.

    This is the one place ComfyUI enters the interpreter - both the core
    translation below and the legacy custom-pack loader (legacy.py) start
    here so packs see the same environment either way."""
    root = _comfy_root_on_path()
    if not (root / "nodes.py").is_file():
        raise CompatError(f"not a ComfyUI installation (no nodes.py): {root}")

    # comfy.cli_args parses at import time only when comfy.options has
    # enabled parsing. The worker host has replaced sys.argv with exactly
    # PackSpec.comfy_args, including an explicit empty tuple; initialize
    # this before nodes.py or any other ComfyUI module can observe defaults.
    initialize_comfy_paths()
    _initialize_comfy_device()

    # ComfyUI's nodes.py prepends its comfy/ directory to sys.path, after
    # which comfy/utils.py shadows the install's root utils/ package for
    # every later import (server.py -> app.* -> utils.install_util). Real
    # ComfyUI survives only because main.py imports utils before nodes;
    # cache the real package under the same accident of import order.
    with contextlib.suppress(ImportError):
        importlib.import_module("utils")

    # ComfyUI import, child process only; importlib keeps the engine-side
    # type checker from needing ComfyUI on its path.
    nodes = importlib.import_module("nodes")

    # main.py's startup order is import nodes -> construct PromptServer ->
    # init_extra_nodes, and both comfy_extras modules and custom packs read
    # PromptServer.instance during load. Reproduce the order with a real
    # but headless server: ComfyUI's own class, never bound to a socket,
    # its loop never run - so route registrations land somewhere inert
    # instead of crashing imports.
    ensure_prompt_server()

    init_extra = getattr(nodes, "init_extra_nodes", None)
    if callable(init_extra):
        # Core "extra" nodes (comfy_extras) without custom nodes or API
        # nodes: deterministic core surface only. Async in current ComfyUI,
        # plain in older ones; _await_sync handles both.
        try:
            parameters = inspect.signature(init_extra).parameters
        except (TypeError, ValueError):
            # Some extension callables expose no signature. Prefer the current
            # API and call once: a body TypeError must not cause double init.
            accepts_api_nodes = True
        else:
            accepts_api_nodes = "init_api_nodes" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
        kwargs = {"init_custom_nodes": False}
        if accepts_api_nodes:
            kwargs["init_api_nodes"] = False
        _await_sync(init_extra(**kwargs))
    return nodes


def _string_tuple(value: object) -> tuple[str, ...] | None:
    """The value as a tuple of strings, or None when it is not a
    homogeneous string list/tuple (a custom get_filename_list override
    returning something exotic simply is not recorded)."""
    if not isinstance(value, (list, tuple)):
        return None
    items = cast("Sequence[object]", value)
    if not all(isinstance(v, str) for v in items):
        return None
    return tuple(cast("Sequence[str]", items))


def filename_listing_probe() -> InputTypesProbe | None:
    """An InputTypesProbe recording folder_paths.get_filename_list calls.

    Upstream nodes universally spell filesystem vocabularies as
    ``folder_paths.get_filename_list("<category>")`` inside INPUT_TYPES()
    (module-attribute call, so a swap of the attribute observes them all).
    The probe wraps ONE invocation: patch, call, restore - translation is
    single-threaded at pack import, and the restore is try/finally so a
    raising INPUT_TYPES() (routine in the wild; translate_mappings records
    it as a skip) cannot leave the patch behind. Returns None when
    folder_paths or the function is missing (a ComfyUI too old/broken to
    have it): translation then stays fully static, exactly as before."""
    try:
        folder_paths = cast("Any", importlib.import_module("folder_paths"))
    except Exception:
        return None
    real_fn = getattr(folder_paths, "get_filename_list", None)
    if not callable(real_fn):
        return None

    def probe(
        invoke: Callable[[], object],
    ) -> tuple[object, Sequence[ListingObservation]]:
        observed: list[ListingObservation] = []

        def recording(folder_name: object, *args: object, **kwargs: object) -> object:
            values: object = real_fn(folder_name, *args, **kwargs)
            if isinstance(folder_name, str):
                entries = _string_tuple(values)
                if entries is not None:
                    # ``values`` (the very returned object) is the
                    # provenance token: translation remote-ifies only
                    # the combo holding THIS object, never a lookalike.
                    observed.append(ListingObservation(folder_name, entries, values))
            return values

        folder_paths.get_filename_list = recording
        try:
            result = invoke()
        finally:
            folder_paths.get_filename_list = real_fn
        return result, tuple(observed)

    return probe


def load_comfyui_nodes(*, required: Sequence[str] = ()) -> CompatTranslation:
    nodes = bootstrap_comfyui()
    mappings = cast("Mapping[str, type]", getattr(nodes, "NODE_CLASS_MAPPINGS", {}))
    if not mappings:
        raise CompatError(
            "ComfyUI at "
            f"{os.environ.get('DINKSTER_COMFYUI_ROOT', '')} exposes no NODE_CLASS_MAPPINGS"
        )
    display = cast(
        "Mapping[str, str]",
        getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}),
    )

    only_text = os.environ.get("DINKSTER_COMFY_NODES", "").strip()
    only = [name.strip() for name in only_text.split(",") if name.strip()] if only_text else None
    if only is not None:
        only = list(dict.fromkeys((*only, *required)))
    return translate_mappings(
        mappings,
        display_names=display,
        only=only,
        probe=filename_listing_probe(),
    )
