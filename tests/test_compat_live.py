"""Live compat test: real ComfyUI v1 nodes on the ComfyUI install's own
interpreter, driven by a Dinkster engine whose interpreter has no torch.

Skipped unless DINKSTER_COMFYUI_ROOT points at a ComfyUI installation with a
``venv/`` (or DINKSTER_COMFYUI_PYTHON names the interpreter). What it proves:

- the compat pack bootstraps inside the child (hazard H5: the engine
  process imports neither ComfyUI nor torch);
- v1 schemas arrive over the hello handshake as honest Dinkster schemas;
- LATENT values (torch tensors) flow between v1 nodes across the
  boundary while the parent merely carries them - it never resolves,
  never needs torch, but can still interrogate type and fingerprint.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import aiohttp
import numpy as np
import pytest
from dinkster_assets import AssetRef, LocalAssetLibrary, digest_file, register_asset_type
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import COMFY_INPUT_ADAPTERS, translate_prompt
from dinkster_compat_comfy.translate import MODEL_FILE_SELECTORS
from dinkster_engine import (
    Engine,
    EngineEvent,
    ExecutionError,
    ExecutionSelection,
    Invocation,
    InvocationResult,
    OnInvocationEvent,
    Worker,
)
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_memory import GovernorReservationService, MemoryGovernor
from dinkster_protocol import LazyStatusInvocation
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_STRING,
    RESOURCE_PRODUCER_ARM_META_KEY,
    EncodedLatentTensor,
    EncodedPayload,
    TypeRegistry,
    Value,
    register_core_types,
)
from dinkster_workers import DeviceMap, IsolatedWorker, PlacementWorker, SingleJobMultiGpuConfig

from dinkster.comfy_compose import comfy_compat_specs
from dinkster.compose import PackSpec, ServingComposer, default_pack_spec
from dinkster.native_policy import (
    NativeDispatchPolicy,
    NativePolicyDiagnostic,
)
from tests.platform_support import symlink_or_skip

COMFY_ROOT = os.environ.get("DINKSTER_COMFYUI_ROOT", "")

pytestmark = pytest.mark.skipif(
    not COMFY_ROOT, reason="DINKSTER_COMFYUI_ROOT not set (live ComfyUI test)"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_PARITY_RECORDS = Path(
    os.environ.get(
        "DINKSTER_INFERENCE_PARITY_RECORDS",
        REPO_ROOT.parent / "dinkster-evidence" / "inference-parity" / "records",
    )
)
COMPAT_MANIFEST = REPO_ROOT / "packages" / "dinkster-compat-comfy" / "dinkster-pack.toml"


def comfy_python() -> str:
    explicit = os.environ.get("DINKSTER_COMFYUI_PYTHON", "")
    if explicit:
        return explicit
    venv_python = Path(COMFY_ROOT) / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def aimdo_python() -> str:
    """The Dinkster GPU venv carrying the current comfy-aimdo API."""
    return str(REPO_ROOT / ".venv-gpu" / "bin" / "python")


def dinkster_pythonpath() -> str:
    """The child imports Dinkster's pure-stdlib packages from source."""
    src_dirs = sorted(str(p) for p in (REPO_ROOT / "packages").glob("*/src"))
    return os.pathsep.join(src_dirs)


def test_real_comfy_builtin_catalog_preserves_selectors_clip_and_batch_types(
    tmp_path: Path,
) -> None:
    """The actual ComfyUI loader proves selectors and dynamic type parity."""
    result_path = tmp_path / "model-selectors.json"
    script = r"""
import inspect
import json
import os
import sys
from pathlib import Path

from dinkster_assets import ASSET_TYPE
from dinkster_compat_comfy import bootstrap
from dinkster_compat_comfy.native import CLIPTextEncode, NATIVE_NODES
from dinkster_compat_comfy.translate import (
    MODEL_FILE_CATEGORIES,
    MODEL_FILE_SELECTORS,
)
from dinkster_schema import (
    AssetWidget,
    InputSpec,
    StringWidget,
    TypeExpr,
    WidgetRepresentation,
    WidgetRepresentations,
    schema_to_wire,
)

sys.argv = [sys.argv[0], "--cpu"]
root = Path(os.environ["DINKSTER_COMFYUI_ROOT"]).resolve(strict=True)
nodes = bootstrap.bootstrap_comfyui()
mappings = nodes.NODE_CLASS_MAPPINGS
translation = bootstrap.load_comfyui_nodes()
schemas = {
    node_class.schema().node_type: node_class.schema()
    for node_class in translation.node_classes
}

# The actual pinned Comfy class declares multiline prompt text and keeps
# dynamic-prompts behavior as client serialization metadata. The native
# replacement truthfully offers the inherited multiline default plus an
# explicit one-line representation without changing the string socket/value.
source_text = mappings["CLIPTextEncode"].INPUT_TYPES()["required"]["text"]
assert source_text[0] == nodes.IO.STRING
assert source_text[1]["multiline"] is True
assert source_text[1]["dynamicPrompts"] is True
native_text = CLIPTextEncode.schema().inputs[0]
assert native_text.type == TypeExpr.concrete("core.string")
assert native_text.widget == WidgetRepresentations(
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
)

# These are the exact live upstream registrations. The three concrete batch
# nodes preserve their member types through the v3 translator and wire; the
# upstream combined MatchType node is deliberately not registered yet.
autogrow_rows = []
for class_name, family_id, member_id, type_id in (
    ("BatchImagesNode", "images", "image", "comfy.IMAGE"),
    ("BatchMasksNode", "masks", "mask", "comfy.MASK"),
    ("BatchLatentsNode", "latents", "latent", "comfy.LATENT"),
):
    assert class_name in mappings
    schema = schemas[f"comfy.{class_name}"]
    assert len(schema.input_families) == 1
    family = schema.input_families[0]
    assert family.id == family_id
    assert family.min_members == 1
    assert family.max_members == 50
    assert family.member_prefix == member_id
    assert len(family.template) == 1
    member = family.template[0]
    assert isinstance(member, InputSpec)
    # InputFamilySpec's canonical repeated template id is value; the
    # upstream socket name is preserved as the memberPrefix.
    assert member.id == "value"
    assert member.type == TypeExpr.concrete(type_id)
    assert schema.outputs[0].type == TypeExpr.concrete(type_id)
    wire_family = next(
        entry for entry in schema_to_wire(schema)["interface"]
        if entry["role"] == "inputFamily"
    )
    assert wire_family["id"] == family_id
    assert wire_family["minMembers"] == 1
    assert wire_family["maxMembers"] == 50
    assert wire_family["memberPrefix"] == member_id
    assert wire_family["template"] == [
        {
            "role": "input",
            "id": "value",
            "type": {"kind": "concrete", "types": [type_id]},
            "required": True,
        }
    ]
    autogrow_rows.append([class_name, family_id, member_id, type_id])
assert "BatchImagesMasksLatentsNode" not in mappings
assert "comfy.BatchImagesMasksLatentsNode" not in schemas

rows = []
for selector in MODEL_FILE_SELECTORS:
    source_class = mappings[selector.class_name]
    source_name = inspect.getsourcefile(source_class)
    assert source_name is not None
    source = Path(source_name).resolve(strict=True)
    if selector.module == "nodes":
        expected_source = (root / "nodes.py").resolve(strict=True)
        assert source_class.__module__ == "nodes"
    else:
        expected_source = root.joinpath(*selector.module.split(".")).with_suffix(".py")
        expected_source = expected_source.resolve(strict=True)
        assert source_class.__module__ == str(expected_source.with_suffix(""))
    assert source == expected_source

    schema = schemas[f"comfy.{selector.class_name}"]
    input_spec = next(item for item in schema.inputs if item.id == selector.input_id)
    assert input_spec.type == TypeExpr.concrete(ASSET_TYPE)
    assert input_spec.widget == AssetWidget(
        accept=("application/octet-stream",),
        kind=MODEL_FILE_CATEGORIES[selector.category].kind,
    )
    rows.append(
        [
            selector.module,
            selector.class_name,
            selector.input_id,
            selector.category,
            str(source.relative_to(root)),
        ]
    )

translated_model_assets = {
    (schema.node_type, input_spec.id)
    for schema in schemas.values()
    for input_spec in schema.inputs
    if isinstance(input_spec.widget, AssetWidget)
    and input_spec.widget.kind.startswith("model/")
}
expected_translated = {
    (f"comfy.{selector.class_name}", selector.input_id)
    for selector in MODEL_FILE_SELECTORS
}
assert len(translated_model_assets) == 38
assert translated_model_assets == expected_translated

native_model_assets = {
    (schema.node_type, input_spec.id)
    for node_class in NATIVE_NODES
    for schema in (node_class.schema(),)
    for input_spec in schema.inputs
    if isinstance(input_spec.widget, AssetWidget)
    and input_spec.widget.kind.startswith("model/")
}
assert len(rows) == 38
assert sum(row[0].startswith("comfy_extras.") for row in rows) == 29
assert len(native_model_assets) == 7
assert len(rows) + len(native_model_assets) == 45
Path(os.environ["DINKSTER_RESULT_PATH"]).write_text(
    json.dumps(
        {
            "rows": rows,
            "native": sorted(native_model_assets),
            "autogrow": autogrow_rows,
        },
        sort_keys=True,
    )
)
"""
    env = os.environ.copy()
    env.update(
        {
            "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
            "DINKSTER_COMFY_NODES": ",".join(
                dict.fromkeys(
                    [selector.class_name for selector in MODEL_FILE_SELECTORS]
                    + ["BatchImagesNode", "BatchMasksNode", "BatchLatentsNode"]
                )
            ),
            "DINKSTER_RESULT_PATH": str(result_path),
            "PYTHONPATH": dinkster_pythonpath(),
        }
    )
    completed = subprocess.run(
        [comfy_python(), "-c", script],
        cwd=COMFY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(result_path.read_text())
    assert len(payload["rows"]) == 38
    assert len(payload["native"]) == 6
    assert payload["autogrow"] == [
        ["BatchImagesNode", "images", "image", "comfy.IMAGE"],
        ["BatchMasksNode", "masks", "mask", "comfy.MASK"],
        ["BatchLatentsNode", "latents", "latent", "comfy.LATENT"],
    ]


def test_aimdo_bootstrap_precedes_real_comfy_torch_import(tmp_path: Path) -> None:
    """Fresh compat-shaped workers prove both sides of the argv opt-in:
    bootstrap succeeds before the real entry imports torch, while the
    default worker reaches that same import without initializing aimdo."""
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "aimdo-probe"\n\n[pack.entry]\nnodes = "aimdo_live_nodes:NODES"\n'
    )

    async def run_probe(*, aimdo_init: bool) -> int:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = IsolatedWorker(
            manifest,
            registry,
            python=comfy_python(),
            aimdo_init=aimdo_init,
            extra_env={
                "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
                "DINKSTER_COMFY_NODES": "EmptyImage",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), dinkster_pythonpath())),
            },
            start_timeout=180.0,
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            result = await engine.run(
                Graph(nodes={"probe": GraphNode("aimdo.bootstrap_probe", {})}),
                ["probe"],
            )
            initialized = result.outputs["probe"]["initialized"].resolve()
            assert isinstance(initialized, int)
            return initialized
        finally:
            await worker.close()

    async def scenario() -> None:
        assert await run_probe(aimdo_init=True) == 1
        assert await run_probe(aimdo_init=False) == 0

    asyncio.run(scenario())


def test_real_worker_comfy_args_parse_supplied_value_and_explicit_defaults(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "comfy-args-probe"\n\n[pack.entry]\nnodes = "aimdo_live_nodes:NODES"\n'
    )

    async def run_probe(comfy_args: tuple[str, ...]) -> int:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = IsolatedWorker(
            manifest,
            registry,
            python=comfy_python(),
            comfy_args=comfy_args,
            extra_env={
                "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
                "DINKSTER_COMFY_NODES": "EmptyImage",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), dinkster_pythonpath())),
            },
            start_timeout=180.0,
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            result = await engine.run(
                Graph(nodes={"probe": GraphNode("comfy.args_probe", {})}),
                ["probe"],
            )
            value = result.outputs["probe"]["preview_size"].resolve()
            assert isinstance(value, int)
            return value
        finally:
            await worker.close()

    async def scenario() -> None:
        assert await run_probe(("--preview-size", "321")) == 321
        assert await run_probe(()) == 512

    asyncio.run(scenario())


def test_aimdo_arm_activates_in_gpu_worker_interpreter(tmp_path: Path) -> None:
    """Real IsolatedWorker argv bootstraps and arms the .venv-gpu child."""
    interpreter = Path(aimdo_python())
    if not interpreter.is_file():
        pytest.skip(f"Dinkster GPU interpreter is absent: {interpreter}")
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "aimdo-arm-probe"\n\n[pack.entry]\nnodes = "aimdo_live_nodes:NODES"\n'
    )
    pyheaders = REPO_ROOT / ".venv-gpu-extras" / "pyheaders" / "usr" / "include"
    cpath = os.pathsep.join((str(pyheaders / "python3.12"), str(pyheaders)))
    if os.environ.get("CPATH"):
        cpath += os.pathsep + os.environ["CPATH"]

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = IsolatedWorker(
            manifest,
            registry,
            python=str(interpreter),
            aimdo_arm="on",
            extra_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), dinkster_pythonpath())),
                "CPATH": cpath,
            },
            start_timeout=180.0,
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            result = await engine.run(
                Graph(nodes={"probe": GraphNode("aimdo.arm_probe", {})}),
                ["probe"],
            )
            assert result.outputs["probe"]["ready"].resolve() == 1
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_aimdo_governor_reservation_mirrors_native_headroom(tmp_path: Path) -> None:
    """Live base edits and governor extras compose in the native control."""
    from dinkster_memory import MemoryGovernor
    from dinkster_workers import HeadroomMirror

    interpreter = Path(aimdo_python())
    if not interpreter.is_file():
        pytest.skip(f"Dinkster GPU interpreter is absent: {interpreter}")
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "aimdo-headroom-probe"\n\n[pack.entry]\nnodes = "aimdo_live_nodes:NODES"\n'
    )
    pyheaders = REPO_ROOT / ".venv-gpu-extras" / "pyheaders" / "usr" / "include"
    cpath = os.pathsep.join((str(pyheaders / "python3.12"), str(pyheaders)))
    if os.environ.get("CPATH"):
        cpath += os.pathsep + os.environ["CPATH"]
    base = 128 * 1024**2
    new_base = 192 * 1024**2
    extra = 64 * 1024**2

    async def scenario() -> None:
        governor = MemoryGovernor()
        mirror = HeadroomMirror(governor, base_bytes=base)
        registry = TypeRegistry()
        register_core_types(registry)
        worker = IsolatedWorker(
            manifest,
            registry,
            python=str(interpreter),
            aimdo_arm="on",
            reserve_vram=base,
            headroom_mirror=mirror,
            extra_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), dinkster_pythonpath())),
                "CPATH": cpath,
            },
            start_timeout=180.0,
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )

            async def read(nonce: int) -> tuple[int, int]:
                result = await engine.run(
                    Graph(nodes={"probe": GraphNode("aimdo.headroom_probe", {"nonce": nonce})}),
                    ["probe"],
                )
                return (
                    cast(int, result.outputs["probe"]["native_headroom"].resolve()),
                    cast(int, result.outputs["probe"]["pending"].resolve()),
                )

            assert await read(0) == (base, 0)
            mirror.set_base(new_base)
            await asyncio.sleep(0.1)
            assert await read(1) == (new_base, 0)
            async with governor.reserve("vram:cuda:0", extra):
                await asyncio.sleep(0.1)
                assert await read(2) == (new_base + extra, 0)
            await asyncio.sleep(0.1)
            assert await read(3) == (new_base, 0)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_real_comfy_latent_nodes_flow_through_torchless_engine() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)  # comfy.* types stay unregistered here
        worker = IsolatedWorker(
            COMPAT_MANIFEST,
            registry,
            python=comfy_python(),
            extra_env={
                "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
                "DINKSTER_COMFY_NODES": "EmptyLatentImage,LatentUpscale",
                "PYTHONPATH": dinkster_pythonpath(),
            },
            start_timeout=180.0,  # first torch + ComfyUI import is slow
        )
        await worker.start()
        try:
            # The Comfy twin remains available for legacy Comfy-typed graphs,
            # while the generation provider publishes the native boundary.
            from dinkster_compat_comfy.native import NATIVE_NODES
            from dinkster_compat_comfy.native_arm import GENERATION_PROVIDER_NODES

            native_ids = {node.schema().node_type for node in NATIVE_NODES}
            provider_ids = {node.schema().node_type for node in GENERATION_PROVIDER_NODES}
            assert (
                set(worker.schemas)
                == {
                    "comfy.EmptyLatentImage",
                    "comfy.LatentUpscale",
                    "dinkster.create_hook_lora",
                    "dinkster.create_hook_keyframe",
                    "dinkster.set_hook_keyframes",
                    "dinkster.conditioning_timesteps_range",
                    "dinkster.conditioning_set_properties_and_combine",
                    "dinkster.pair_conditioning_set_properties",
                }
                | native_ids
                | provider_ids
            )
            empty = worker.schemas["dinkster.empty_latent_image"]
            assert [out.id for out in empty.outputs] == ["latent"]
            assert empty.outputs[0].type.types == ("dinkster.latent",)

            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=_Recorder(worker),
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                nodes={
                    "native": GraphNode(
                        "dinkster.empty_latent_image",
                        {"width": 64, "height": 64, "batch_size": 1},
                    ),
                }
            )
            result = await engine.run(graph, ["native"])
            latent = result.outputs["native"]["latent"]
            # The parent carries the envelope without resolving: it has no
            # torch, yet knows the type and the location-independent identity.
            assert latent.type_id == "dinkster.latent"
            assert latent.fingerprint

            # Same graph again: pure cache hits, the child does nothing.
            again = await engine.run(graph, ["native"])
            assert again.executed == ()
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_real_comfy_image_renders_png_in_torchless_engine() -> None:
    """The comfy.IMAGE preview contract end to end: a real v1 node produces
    a torch image tensor in the child, the npy bytes cross the boundary,
    and the torchless parent - carrying the host-side registration exactly
    as comfy_compat_specs installs it - decodes with numpy and renders the
    declared PNG rendition."""
    from dinkster.comfy_compose import COMFY_IMAGE_TYPE, register_comfy_host_types

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_comfy_host_types(registry)  # the PackSpec.host_types hook
        worker = IsolatedWorker(
            COMPAT_MANIFEST,
            registry,
            python=comfy_python(),
            extra_env={
                "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
                "DINKSTER_COMFY_NODES": "EmptyImage",
                "PYTHONPATH": dinkster_pythonpath(),
            },
            start_timeout=180.0,
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                nodes={
                    "i": GraphNode(
                        "comfy.EmptyImage",
                        {
                            "width": 8,
                            "height": 4,
                            "batch_size": 1,
                            "color": 0xFF0000,
                        },
                    )
                }
            )
            result = await engine.run(graph, ["i"])
            image = result.outputs["i"]["image"]
            assert image.type_id == COMFY_IMAGE_TYPE
            # Meta declares dimensions without touching the payload.
            assert image.meta.entries.get("shape") == [1, 4, 8, 3]

            specs = registry.renditions_of(image.type_id)
            assert [(s.kind, s.default) for s in specs] == [("png", True)]
            rendition = registry.render(image, "png")
            assert rendition.mime == "image/png"
            assert rendition.data[:8] == b"\x89PNG\r\n\x1a\n"
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_real_comfy_mask_renders_png_in_torchless_engine() -> None:
    """The comfy.MASK preview contract end to end: a real v1 node produces
    a torch mask tensor in the child, the npy bytes cross the boundary (the
    image-array codec, never pickle), and the torchless parent decodes with
    numpy and renders the declared mask PNG rendition."""
    from dinkster.comfy_compose import COMFY_MASK_TYPE, register_comfy_host_types

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_comfy_host_types(registry)  # the PackSpec.host_types hook
        worker = IsolatedWorker(
            COMPAT_MANIFEST,
            registry,
            python=comfy_python(),
            extra_env={
                "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
                "DINKSTER_COMFY_NODES": "SolidMask",
                "PYTHONPATH": dinkster_pythonpath(),
            },
            start_timeout=180.0,
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            schema = worker.schemas["comfy.SolidMask"]
            (output,) = schema.outputs
            graph = Graph(
                nodes={
                    "m": GraphNode(
                        "comfy.SolidMask",
                        {"value": 0.5, "width": 8, "height": 4},
                    )
                }
            )
            result = await engine.run(graph, ["m"])
            mask = result.outputs["m"][output.id]
            assert mask.type_id == COMFY_MASK_TYPE
            # Meta declares dimensions without touching the payload.
            assert mask.meta.entries.get("shape") == [1, 4, 8]

            specs = registry.renditions_of(mask.type_id)
            assert [(s.kind, s.default) for s in specs] == [("png", True)]
            rendition = registry.render(mask, "png")
            assert rendition.mime == "image/png"
            assert rendition.data[:8] == b"\x89PNG\r\n\x1a\n"
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_composed_load_image_routes_typed_asset_without_manual_base_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production composition resolves the first-party media loader once."""
    import PIL.Image
    from dinkster_schema import AssetWidget, SourceFilenameSpec

    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    image_path = tmp_path / "input.png"
    PIL.Image.fromarray(pixels, mode="RGB").save(image_path)
    library = LocalAssetLibrary(tmp_path)
    library.scan()
    ref = library.ref("models/input.png")
    monkeypatch.setenv("DINKSTER_ASSET_ROOT", str(tmp_path))

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await _add_composed_compat(
                composer,
                asset_root=tmp_path,
                comfy_nodes=("LoadImage", "ImageInvert"),
            )
            schema = composer.composition.schemas["dinkster.load_image"]
            assert schema.inputs[0].widget == AssetWidget(
                ("image/png", "image/jpeg", "image/webp"),
                kind="media/image",
                allow_upload=True,
            )
            assert schema.inputs[0].source_filename == SourceFilenameSpec("media/image", "input")
            registry = composer.composition._registry
            wrapped = registry.wrap("asset<dinkster.image>", ref.to_wire())
            assert wrapped.type_id == "asset<dinkster.image>"
            assert wrapped.fingerprint == ref.digest
            resolved = wrapped.resolve()
            assert isinstance(resolved, AssetRef)
            assert resolved.read_bytes() == image_path.read_bytes()

            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode("dinkster.load_image", {"image": ref.to_wire()}),
                    "invert": GraphNode(
                        "comfy.ImageInvert",
                        {"image": Link("load", "image")},
                    ),
                }
            )
            result = await engine.run(graph, ["load", "invert"])
            image = result.outputs["load"]["image"]
            assert image.type_id == "dinkster.image"
            assert image.meta.entries["shape"] == (1, 4, 8, 3)
            inverted = result.outputs["invert"]["image"]
            assert inverted.type_id == "comfy.IMAGE"
            assert inverted.meta.entries["shape"] == [1, 4, 8, 3]
            rendered = registry.render(inverted, "png")
            with PIL.Image.open(io.BytesIO(rendered.data)) as preview:
                assert preview.convert("RGB").getpixel((0, 0)) == (0, 255, 255)
        finally:
            await composer.close()

    asyncio.run(scenario())


def first_checkpoint() -> str | None:
    ckpt_dir = Path(COMFY_ROOT) / "models" / "checkpoints"
    names = sorted(p.name for p in ckpt_dir.glob("*.safetensors"))
    return names[0] if names else None


def models_root() -> Path:
    return Path(COMFY_ROOT) / "models"


def checkpoint_ref_wire(ckpt_name: str, *, asset_root: Path | None = None) -> dict[str, object]:
    """Digest-backed AssetRef wire literal for a checkpoint under the
    install's models/checkpoints - the native loader's input form. The
    literal is pure identity + metadata, no path."""
    library = LocalAssetLibrary(asset_root or models_root())
    library.scan()
    return library.ref(f"models/checkpoints/{ckpt_name}").to_wire()


def _signal_serve_process_group(process: subprocess.Popen[bytes], *, force: bool) -> None:
    """Stop the daemon and its worker process group; plain kill on Windows."""
    if sys.platform == "win32":
        if force:
            process.kill()
        else:
            process.terminate()
    else:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)


def test_daemon_runs_classic_sd15_controlnet_on_the_native_arm(tmp_path: Path) -> None:
    checkpoint = Path(COMFY_ROOT) / "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
    controlnet = Path(COMFY_ROOT) / "models/controlnet/control_v11p_sd15_canny.safetensors"
    hint = Path(COMFY_ROOT) / "input/dinkster29-canny-hint.png"
    for artifact in (checkpoint, controlnet, hint):
        if not artifact.is_file():
            pytest.skip(f"SD1.5 ControlNet daemon artifact not present: {artifact}")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = cast("tuple[str, int]", listener.getsockname())[1]
    library_root = tmp_path / "library"
    run_name = f"controlnet-{time.time_ns()}"
    output_prefix = f"dinkster-tests/{run_name}"
    output_root = Path(COMFY_ROOT) / "output"
    log_path = tmp_path / "serve.log"
    log = log_path.open("wb")
    process = subprocess.Popen(
        (
            sys.executable,
            "-m",
            "dinkster.serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--library-root",
            str(library_root),
            "--comfy-root",
            COMFY_ROOT,
            "--comfy-python",
            comfy_python(),
            "--aimdo",
            "off",
            "--strict-packs",
        ),
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "DINKSTER_SERVING_PYTHON": sys.executable,
        },
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def workflow(strength: float, label: str) -> dict[str, object]:
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint.name},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "a stone castle on a hill"},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": ""},
            },
            "4": {"class_type": "LoadImage", "inputs": {"image": hint.name}},
            "5": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": controlnet.name},
            },
            "6": {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "control_net": ["5", 0],
                    "image": ["4", 0],
                    "strength": strength,
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                },
            },
            "7": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1},
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "seed": 29,
                    "steps": 2,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "positive": ["6", 0],
                    "negative": ["6", 1],
                    "latent_image": ["7", 0],
                    "denoise": 1.0,
                },
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["8", 0], "vae": ["1", 2]},
            },
            "10": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["9", 0],
                    "filename_prefix": f"{output_prefix}-{label}",
                },
            },
        }

    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=900)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with asyncio.timeout(180):
                while True:
                    assert process.poll() is None, log_path.read_text("utf-8", errors="replace")
                    try:
                        async with session.get(base + "/api/composition") as response:
                            composition = await response.json()
                        async with session.get(base + "/api/mounts") as response:
                            mounts = await response.json()
                        ready = {
                            row["id"]
                            for row in mounts.get("mounts", [])
                            if row.get("state") == "ready"
                        }
                        if (
                            composition.get("packs", {})
                            .get("dinkster-compat-comfy", {})
                            .get("state")
                            == "announced"
                            and "comfy-model-checkpoints-1" in ready
                            and "comfy-model-controlnet-1" in ready
                        ):
                            break
                    except (aiohttp.ClientError, ValueError):
                        pass
                    await asyncio.sleep(0.1)

            async def submit(
                strength: float, label: str
            ) -> tuple[dict[str, object], dict[str, object]]:
                async with session.post(
                    base + "/api/compat/comfy/prompt",
                    json={
                        "client_id": "controlnet-daemon-test",
                        "prompt": workflow(strength, label),
                    },
                ) as response:
                    submitted = await response.json()
                    assert response.status == 202, submitted
                job_ref = cast("str", submitted["jobRef"])
                async with asyncio.timeout(600):
                    while True:
                        async with session.get(base + f"/api/jobs/by-ref/{job_ref}") as response:
                            status = await response.json()
                        if status["state"] in ("completed", "failed", "cancelled"):
                            break
                        await asyncio.sleep(0.1)
                async with session.get(base + f"/api/jobs/by-ref/{job_ref}/events") as response:
                    events = await response.json()
                assert status["state"] == "completed", status.get("error")
                arms = {
                    event.get("nodeId"): event.get("detail", {}).get("executionArm")
                    for event in events["events"]
                    if event["type"] in ("node_finished", "node_cached")
                }
                assert {
                    node_id: arms.get(node_id) for node_id in ("1", "2", "3", "5", "6", "8", "9")
                } == {node_id: "native" for node_id in ("1", "2", "3", "5", "6", "8", "9")}
                return status, events

            controlled, _ = await submit(1.0, "gain1")
            plain, _ = await submit(0.0, "gain0")
            return controlled, plain

    try:
        controlled, plain = asyncio.run(scenario())
        output_fingerprints = []
        for status in (controlled, plain):
            node_output = cast(
                "dict[str, object]", cast("dict[str, object]", status["outputs"])["10"]
            )
            descriptor = cast("dict[str, object]", node_output["assets"])
            element = cast("dict[str, object]", cast("list[object]", descriptor["elements"])[0])
            output_fingerprints.append(cast("str", element["fingerprint"]))
        assert output_fingerprints[0] != output_fingerprints[1]
    finally:
        if process.poll() is None:
            _signal_serve_process_group(process, force=False)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _signal_serve_process_group(process, force=True)
                process.wait(timeout=30)
        log.close()
        for output in output_root.glob(f"{output_prefix}-*_*.png"):
            output.unlink()


def test_real_checkpoint_stays_resident_in_the_worker() -> None:
    """Model, text-encoder, and codec objects stay inside their worker."""
    ckpt_name = first_checkpoint()
    if ckpt_name is None:
        pytest.skip("no .safetensors checkpoint under models/checkpoints")

    async def scenario() -> None:
        composer, engine = await _generation_composer(ckpt_name)
        try:
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "dinkster.load_checkpoint",
                        {"checkpoint": checkpoint_ref_wire(ckpt_name)},
                    ),
                    "enc": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": "a photo of a cat"},
                    ),
                }
            )
            result = await engine.run(graph, ["load", "enc"])

            # Gigabytes of loaded model state stayed in the child; the
            # parent got a resident stub per output, kilobytes at most.
            for output_id in ("model", "clip", "vae"):
                value = result.outputs["load"][output_id]
                assert value.fingerprint.startswith("resident:")
                assert len(value.payload.data) < 128  # type: ignore[attr-defined]

            # The torch-less parent still knows *where* the model lives
            # and what using it costs: residency/cost meta crossed even
            # though the object never will. This is what binds GPU
            # admission lanes and VRAM budgets.
            model = result.outputs["load"]["model"]
            resources = model.meta.get("resources")
            assert isinstance(resources, dict)
            assert str(resources["gpu"]).startswith("cuda")
            cost = model.meta.get("cost")
            assert isinstance(cost, dict)
            assert any(key.startswith("vram:cuda") for key in cost)
            assert all(nbytes > 0 for nbytes in cost.values())

            # Conditioning is data, not residency: it crossed as real
            # canonical bytes the parent carries without resolving.
            cond = result.outputs["enc"]["conditioning"]
            assert cond.type_id == "dinkster.conditioning"
            assert not cond.fingerprint.startswith("resident:")

            # Resident fingerprints are stable, so a re-run is pure cache.
            again = await engine.run(graph, ["load", "enc"])
            assert again.executed == ()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_real_inference_end_to_end_through_torchless_engine() -> None:
    """The whole SD pipeline - load, encode, sample on the GPU, decode -
    driven by an engine process that never imports torch. Models stay
    resident in the worker; latents/conditioning/images cross as bytes."""
    ckpt_name = first_checkpoint()
    if ckpt_name is None:
        pytest.skip("no .safetensors checkpoint under models/checkpoints")

    async def scenario() -> None:
        composer, engine = await _generation_composer(ckpt_name)
        try:
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "dinkster.load_checkpoint",
                        {"checkpoint": checkpoint_ref_wire(ckpt_name)},
                    ),
                    "pos": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": "a photo of a cat"},
                    ),
                    "neg": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": ""},
                    ),
                    "lat": GraphNode(
                        "dinkster.empty_latent_image",
                        {"width": 512, "height": 512, "batch_size": 1},
                    ),
                    "sample": GraphNode(
                        "dinkster.ksampler",
                        {
                            "model": Link("load", "model"),
                            "seed": 7,
                            "steps": 4,
                            "cfg": 7.0,
                            "sampler_name": "euler",
                            "scheduler": "normal",
                            "positive": Link("pos", "conditioning"),
                            "negative": Link("neg", "conditioning"),
                            "latent_image": Link("lat", "latent"),
                            "denoise": 1.0,
                        },
                    ),
                    "decode": GraphNode(
                        "dinkster.vae_decode",
                        {
                            "samples": Link("sample", "latent"),
                            "vae": Link("load", "vae"),
                        },
                    ),
                }
            )
            result = await engine.run(graph, ["decode"])

            # A real 512x512 image came back: megabytes of pixels the
            # torch-less parent carries but has no need (or way) to open.
            image = result.outputs["decode"]["image"]
            assert image.type_id == "dinkster.image"
            assert len(image.payload.data) > 512 * 512 * 3  # type: ignore[attr-defined]

            # Everything upstream is deterministic + resident-stable:
            # a re-run executes nothing at all.
            again = await engine.run(graph, ["decode"])
            assert again.executed == ()
        finally:
            await composer.close()

    asyncio.run(scenario())


class _RecordingNativePolicy:
    def __init__(self, checkpoint: Path) -> None:
        digest = digest_file(checkpoint)
        self.diagnostics: list[NativePolicyDiagnostic] = []
        self.selections: list[tuple[str, ExecutionSelection | None]] = []
        self.inner = NativeDispatchPolicy(
            lambda candidate: checkpoint if candidate == digest else None,
            self.diagnostics.append,
        )

    async def select(
        self,
        node_type: str,
        inputs: Mapping[str, Value],
        arms: tuple[str, Mapping[str, str]],
        **kwargs: Any,
    ) -> ExecutionSelection | None:
        selection = await self.inner.select(node_type, inputs, arms, **kwargs)
        self.selections.append((node_type, selection))
        return selection


def _composed_compat_specs(
    *,
    asset_root: Path | None = None,
    comfy_nodes: tuple[str, ...] = ("CLIPTextEncode",),
) -> list[PackSpec]:
    generation, spec = comfy_compat_specs(
        COMFY_ROOT,
        python=comfy_python(),
        comfy_nodes=comfy_nodes,
    )
    return [
        generation,
        replace(
            spec,
            env={
                **spec.env,
                "DINKSTER_ASSET_ROOT": str(asset_root or models_root()),
            },
        ),
    ]


async def _add_composed_compat(
    composer: ServingComposer,
    *,
    asset_root: Path | None = None,
    comfy_nodes: tuple[str, ...] = ("CLIPTextEncode",),
) -> None:
    generation, compat = _composed_compat_specs(
        asset_root=asset_root,
        comfy_nodes=comfy_nodes,
    )
    await _add_generation_compat_specs(composer, generation, compat)


async def _add_generation_compat_specs(
    composer: ServingComposer,
    generation: PackSpec,
    compat: PackSpec,
) -> None:
    await composer.add_pack(default_pack_spec("dinkster-nodes-media-io"))
    await composer.add_pack(generation)
    await composer.add_pack(compat)


async def _generation_composer(checkpoint_name: str) -> tuple[ServingComposer, Engine]:
    checkpoint = Path(COMFY_ROOT) / "models" / "checkpoints" / checkpoint_name
    composer = ServingComposer(native_policy=_RecordingNativePolicy(checkpoint))  # type: ignore[arg-type]
    register_asset_type(composer.composition._registry)
    try:
        await _add_composed_compat(composer)
    except BaseException:
        await composer.close()
        raise
    return composer, composer.composition.make_engine(lambda _event: None)


def test_generation_provider_real_sd15_gpu_end_to_end() -> None:
    """Real admission policy -> generation provider -> native SD1.5 runtime."""
    checkpoint = (
        Path(COMFY_ROOT) / "models" / "checkpoints" / "v1-5-pruned-emaonly-fp16.safetensors"
    )
    if not checkpoint.is_file():
        pytest.skip(f"required native checkpoint is absent: {checkpoint}")

    async def scenario() -> None:
        policy = _RecordingNativePolicy(checkpoint)
        composer = ServingComposer(native_policy=policy)  # type: ignore[arg-type]
        try:
            register_asset_type(composer.composition._registry)
            await _add_composed_compat(composer)
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "dinkster.load_checkpoint",
                        {"checkpoint": checkpoint_ref_wire(checkpoint.name)},
                    ),
                    "pos": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": "a photo of a cat"},
                    ),
                    "neg": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": ""},
                    ),
                    "lat": GraphNode(
                        "dinkster.empty_latent_image",
                        {"width": 64, "height": 64, "batch_size": 1},
                    ),
                    "sample": GraphNode(
                        "dinkster.ksampler",
                        {
                            "model": Link("load", "model"),
                            "seed": 7,
                            "steps": 2,
                            "cfg": 7.0,
                            "sampler_name": "euler",
                            "scheduler": "normal",
                            "positive": Link("pos", "conditioning"),
                            "negative": Link("neg", "conditioning"),
                            "latent_image": Link("lat", "latent"),
                            "denoise": 1.0,
                        },
                    ),
                    "decode": GraphNode(
                        "dinkster.vae_decode",
                        {
                            "samples": Link("sample", "latent"),
                            "vae": Link("load", "vae"),
                        },
                    ),
                }
            )

            result = await engine.run(graph, ["load", "decode"])

            selected = {
                node_type: selection.target
                for node_type, selection in policy.selections
                if selection is not None
            }
            assert selected == {
                "dinkster.load_checkpoint": "dinkster-compat-comfy",
            }
            for output_id in ("model", "clip", "vae"):
                assert (
                    result.outputs["load"][output_id].meta.get(RESOURCE_PRODUCER_ARM_META_KEY)
                    == "dinkster-compat-comfy"
                )
            image = result.outputs["decode"]["image"]
            assert isinstance(image.payload, EncodedPayload)
            array = cast("Any", image.payload.load())
            assert array.shape == (1, 64, 64, 3)
            assert bool(np.isfinite(array).all())
            assert float(array.min()) >= 0.0
            assert float(array.max()) <= 1.0
            assert policy.diagnostics == []
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("sampler_name", "scheduler"),
    (
        ("dinkster.lcm", "dinkster.normal"),
        ("dinkster.dpmpp_2m_sde", "dinkster.karras"),
    ),
)
def test_generation_provider_real_sd15_lora_gpu_end_to_end(
    sampler_name: str, scheduler: str
) -> None:
    checkpoint = (
        Path(COMFY_ROOT) / "models" / "checkpoints" / "v1-5-pruned-emaonly-fp16.safetensors"
    )
    lora = Path(COMFY_ROOT) / "models" / "loras" / "dinkster-native-linear-proof.safetensors"
    if not checkpoint.is_file() or not lora.is_file():
        pytest.skip(f"required native checkpoint or LoRA is absent: {checkpoint}, {lora}")

    async def scenario() -> None:
        library = LocalAssetLibrary(models_root())
        library.scan()
        policy = _RecordingNativePolicy(checkpoint)
        composer = ServingComposer(native_policy=policy)  # type: ignore[arg-type]
        try:
            await _add_composed_compat(composer)
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "dinkster.load_checkpoint",
                        {"checkpoint": checkpoint_ref_wire(checkpoint.name)},
                    ),
                    "lora": GraphNode(
                        "dinkster.load_lora",
                        {
                            "model": Link("load", "model"),
                            "clip": Link("load", "clip"),
                            "lora": library.ref(f"models/loras/{lora.name}").to_wire(),
                            "strength_model": 1.0,
                            "strength_clip": 0.0,
                            "execution_mode": "auto",
                        },
                    ),
                    "pos": GraphNode(
                        "dinkster.clip_text_encode",
                        {
                            "clip": Link("lora", "clip"),
                            "text": "a pokemon style cat",
                        },
                    ),
                    "neg": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": ""},
                    ),
                    "lat": GraphNode(
                        "dinkster.empty_latent_image",
                        {"width": 64, "height": 64, "batch_size": 1},
                    ),
                    "sample": GraphNode(
                        "dinkster.ksampler",
                        {
                            "model": Link("lora", "model"),
                            "seed": 89,
                            "steps": 2,
                            "cfg": 7.0,
                            "sampler_name": sampler_name,
                            "scheduler": scheduler,
                            "positive": Link("pos", "conditioning"),
                            "negative": Link("neg", "conditioning"),
                            "latent_image": Link("lat", "latent"),
                            "denoise": 1.0,
                        },
                    ),
                    "decode": GraphNode(
                        "dinkster.vae_decode",
                        {
                            "samples": Link("sample", "latent"),
                            "vae": Link("load", "vae"),
                        },
                    ),
                }
            )

            result = await engine.run(graph, ["lora", "pos", "decode"])

            assert result.outputs["lora"]["model"].type_id == "dinkster.model"
            assert result.outputs["lora"]["clip"].type_id == "dinkster.clip"
            assert result.outputs["pos"]["conditioning"].type_id == "dinkster.conditioning"
            image = result.outputs["decode"]["image"]
            assert isinstance(image.payload, EncodedPayload)
            array = cast("Any", image.payload.load())
            assert array.shape == (1, 64, 64, 3)
            assert bool(np.isfinite(array).all())
            assert policy.diagnostics == []
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_native_dispatch_arm_combined_flux_fp8_gpu_end_to_end(
    tmp_path: Path,
) -> None:
    """Combined Flux header -> native policy -> full native runtime graph."""
    checkpoint = Path(COMFY_ROOT) / "models" / "diffusion_models" / "flux1-dev-fp8.safetensors"
    if not checkpoint.is_file():
        pytest.skip(f"required combined Flux checkpoint is absent: {checkpoint}")
    asset_root = tmp_path / "models"
    checkpoint_dir = asset_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    exposed = checkpoint_dir / checkpoint.name
    symlink_or_skip(exposed, checkpoint)
    checkpoint_wire = checkpoint_ref_wire(
        checkpoint.name,
        asset_root=asset_root,
    )

    async def scenario() -> None:
        policy = _RecordingNativePolicy(checkpoint)
        composer = ServingComposer(native_policy=policy)  # type: ignore[arg-type]
        try:
            register_asset_type(composer.composition._registry)
            await _add_composed_compat(
                composer,
                asset_root=asset_root,
                comfy_nodes=("CLIPTextEncode", "EmptySD3LatentImage"),
            )
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "dinkster.load_checkpoint",
                        {"checkpoint": checkpoint_wire},
                    ),
                    "pos": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": "a cat"},
                    ),
                    "neg": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": ""},
                    ),
                    "lat": GraphNode(
                        "comfy.EmptySD3LatentImage",
                        {"width": 64, "height": 64, "batch_size": 1},
                    ),
                    "sample": GraphNode(
                        "dinkster.ksampler",
                        {
                            "model": Link("load", "model"),
                            "seed": 17,
                            "steps": 1,
                            "cfg": 1.0,
                            "sampler_name": "euler",
                            "scheduler": "simple",
                            "positive": Link("pos", "conditioning"),
                            "negative": Link("neg", "conditioning"),
                            "latent_image": Link("lat", "LATENT"),
                            "denoise": 1.0,
                        },
                    ),
                    "decode": GraphNode(
                        "dinkster.vae_decode",
                        {
                            "samples": Link("sample", "latent"),
                            "vae": Link("load", "vae"),
                        },
                    ),
                }
            )

            result = await engine.run(graph, ["decode"])

            selected = {
                node_type: selection.target
                for node_type, selection in policy.selections
                if selection is not None
            }
            assert selected == {
                "dinkster.load_checkpoint": "dinkster-compat-comfy",
            }
            assert policy.diagnostics == []
            image = result.outputs["decode"]["image"]
            assert isinstance(image.payload, EncodedPayload)
            array = cast("Any", image.payload.load())
            assert array.shape == (1, 64, 64, 3)
            assert bool(np.isfinite(array).all())
            assert float(array.min()) >= 0.0
            assert float(array.max()) <= 1.0
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_native_dispatch_small_budget_advisory_evicts_and_reloads() -> None:
    """Exact declared-weight budget: unload to zero, then reload each stage."""
    checkpoint = (
        Path(COMFY_ROOT) / "models" / "checkpoints" / "v1-5-pruned-emaonly-fp16.safetensors"
    )
    if not checkpoint.is_file():
        pytest.skip(f"required native checkpoint is absent: {checkpoint}")

    async def scenario() -> None:
        governor = MemoryGovernor()
        policy = _RecordingNativePolicy(checkpoint)
        composer = ServingComposer(
            native_policy=policy,  # type: ignore[arg-type]
            governor=governor,
            reservations=GovernorReservationService(governor),
        )
        consumer = "dinkster-compat-comfy:comfy-models"

        async def wait_footprint(device: str, expected: int) -> None:
            for _ in range(500):
                if governor.footprint(device) == expected:
                    return
                await asyncio.sleep(0.01)
            raise AssertionError(
                f"timed out waiting for {device} footprint {expected}; got "
                f"{governor.footprint(device)}"
            )

        try:
            register_asset_type(composer.composition._registry)
            await _add_composed_compat(composer)
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "dinkster.load_checkpoint",
                        {"checkpoint": checkpoint_ref_wire(checkpoint.name)},
                    ),
                    "pos": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": "a cat"},
                    ),
                    "neg": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": ""},
                    ),
                    "lat": GraphNode(
                        "dinkster.empty_latent_image",
                        {"width": 64, "height": 64, "batch_size": 1},
                    ),
                    "sample": GraphNode(
                        "dinkster.ksampler",
                        {
                            "model": Link("load", "model"),
                            "seed": 13,
                            "steps": 2,
                            "cfg": 7.0,
                            "sampler_name": "euler",
                            "scheduler": "normal",
                            "positive": Link("pos", "conditioning"),
                            "negative": Link("neg", "conditioning"),
                            "latent_image": Link("lat", "latent"),
                            "denoise": 1.0,
                        },
                    ),
                    "decode": GraphNode(
                        "dinkster.vae_decode",
                        {
                            "samples": Link("sample", "latent"),
                            "vae": Link("load", "vae"),
                        },
                    ),
                }
            )

            loaded = await engine.run(graph, ["load"])
            model = loaded.outputs["load"]["model"]
            cost = model.meta.get("cost")
            assert isinstance(cost, dict)
            [(vram_lane, declared_bytes)] = [
                (lane, nbytes) for lane, nbytes in cost.items() if lane.startswith("vram:")
            ]
            assert isinstance(declared_bytes, int) and declared_bytes > 0
            governor.set_budget(vram_lane, declared_bytes)
            await wait_footprint(vram_lane, 0)

            await engine.run(graph, ["pos"])
            await wait_footprint(vram_lane, declared_bytes)
            (item,) = governor.details()[consumer]
            assert item.bytes_by_residency[vram_lane] == declared_bytes

            first_freed = await governor.shed(
                vram_lane,
                declared_bytes,
                consumers=[consumer],
                items=[item.item_id],
            )
            assert first_freed == declared_bytes
            await wait_footprint(vram_lane, 0)

            await engine.run(graph, ["sample"])
            await wait_footprint(vram_lane, declared_bytes)
            second_freed = await governor.shed(
                vram_lane,
                declared_bytes,
                consumers=[consumer],
                items=[item.item_id],
            )
            assert second_freed == declared_bytes
            await wait_footprint(vram_lane, 0)

            decoded = await engine.run(graph, ["decode"])
            await wait_footprint(vram_lane, declared_bytes)
            image = cast("Any", decoded.outputs["decode"]["image"].payload).load()
            assert image.shape == (1, 64, 64, 3)
            assert bool(np.isfinite(image).all())
            print(
                "NATIVE_RESIDENCY_E2E "
                f"budget={declared_bytes} initial=0 text={declared_bytes} "
                f"evict1={first_freed} sample_reload={declared_bytes} "
                f"evict2={second_freed} vae_reload={declared_bytes}"
            )
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_generation_provider_refuses_unrecognized_checkpoint_before_execution() -> None:
    """The native-only generation provider cannot silently fall back to comfy.*."""
    checkpoint = (
        Path(COMFY_ROOT)
        / "models"
        / "checkpoints"
        / "NetaYumev35_pretrained_all_in_one.safetensors"
    )
    if not checkpoint.is_file():
        pytest.skip(f"required fallback checkpoint is absent: {checkpoint}")

    async def scenario() -> None:
        policy = _RecordingNativePolicy(checkpoint)
        composer = ServingComposer(native_policy=policy)  # type: ignore[arg-type]
        try:
            register_asset_type(composer.composition._registry)
            await _add_composed_compat(composer)
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "dinkster.load_checkpoint",
                        {"checkpoint": checkpoint_ref_wire(checkpoint.name)},
                    )
                }
            )

            with pytest.raises(RuntimeError, match="native admission did not recognize"):
                await engine.run(graph, ["load"])

            assert policy.selections == []
            assert policy.diagnostics
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_checkpoint_loads_by_digest_identity_not_filename() -> None:
    """The asset-native loader: the graph names the checkpoint by content
    digest; the worker materializes bytes from its own store. Cache
    identity is the digest - a rename or another machine changes nothing."""
    ckpt_name = first_checkpoint()
    if ckpt_name is None:
        pytest.skip("no .safetensors checkpoint under models/checkpoints")

    ref_wire = checkpoint_ref_wire(ckpt_name)

    async def scenario() -> None:
        composer, engine = await _generation_composer(ckpt_name)
        try:
            assert "dinkster.load_checkpoint" in composer.composition.schemas
            graph = Graph(
                nodes={
                    # The literal is pure identity + metadata - no path.
                    "load": GraphNode("dinkster.load_checkpoint", {"checkpoint": ref_wire}),
                    "enc": GraphNode(
                        "dinkster.clip_text_encode",
                        {"clip": Link("load", "clip"), "text": "a photo of a cat"},
                    ),
                }
            )
            result = await engine.run(graph, ["load", "enc"])
            for output_id in ("model", "clip", "vae"):
                value = result.outputs["load"][output_id]
                assert value.fingerprint.startswith("resident:")
            assert result.outputs["enc"]["conditioning"].type_id == "dinkster.conditioning"

            # The loader's cache key derives from the digest fingerprint.
            again = await engine.run(graph, ["load", "enc"])
            assert again.executed == ()
        finally:
            await composer.close()

    asyncio.run(scenario())


def gpu_count() -> int:
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if proc.returncode != 0:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("GPU "))


def gpu_compute_capability() -> tuple[int, int] | None:
    """The compute capability of the first CUDA-visible GPU, torch-free.

    nvidia-smi ignores CUDA_VISIBLE_DEVICES, so the first entry of that
    variable (an index or UUID) picks the queried board explicitly.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    command = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    if visible:
        command.append(f"--id={visible}")
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        return None
    major, _, minor = lines[0].strip().partition(".")
    try:
        return (int(major), int(minor))
    except ValueError:
        return None


def test_native_residency_partial_sd15_forward_is_bitwise() -> None:
    """A partially placed enrolled SD1.5 stage executes bitwise."""
    if gpu_count() < 1:
        pytest.skip("needs a CUDA GPU")
    code = r"""
import json

import torch

from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    KLConfig,
    SD15,
    UNetConfig,
    ClipTextConfig,
    ReconstructionRecipe,
    RuntimeKnobs,
    WeightSourceBinding,
    WeightSourceRef,
)
from dinkster_compat_comfy.native_residency import (
    NativeResidencyCoordinator,
    NativeRuntimeHandle,
)
from dinkster_inference_torch import (
    AssembledSD,
    AutoencoderKL,
    ClipTextModel,
    SDRuntime,
    UNetModel,
)
from dinkster_inference_torch.memory import DeviceMemory, MemoryPolicy
from dinkster_inference_torch.residency import ResidencyManager

device = torch.device("cuda:0")
clip_config = ClipTextConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    hidden_act="quick_gelu",
    vocab_size=49408,
    eos_token_id=49407,
)
unet_config = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1, 1),
    channel_mult=(1, 2),
    transformer_depth=(1, 1),
    transformer_depth_output=(1, 1, 1, 1),
    transformer_depth_middle=1,
    context_dim=clip_config.hidden_size,
    use_linear_in_transformer=False,
    num_heads=8,
)
kl_config = KLConfig(
    in_channels=3,
    out_channels=3,
    ch=32,
    decoder_ch=32,
    ch_mult=(1, 2),
    num_res_blocks=1,
    z_channels=4,
    embed_dim=4,
)

def filled(module):
    with torch.no_grad():
        for index, tensor in enumerate(module.state_dict().values()):
            if tensor.is_floating_point():
                tensor.fill_((index % 17 - 8) * 0.001)
            else:
                tensor.zero_()
    return module

assembled = AssembledSD(
    family=SD15,
    diffusion=filled(UNetModel(unet_config)),
    clip_l=filled(ClipTextModel(clip_config)),
    clip_g=None,
    vae=filled(AutoencoderKL(kl_config)),
)
recipe = ReconstructionRecipe(
    sources=(
        WeightSourceBinding(
            "checkpoint",
            WeightSourceRef("blake3:" + "0" * 64, "live-test.safetensors", 0),
        ),
    ),
    family_id="dinkster.sd15",
    component_identity=("family=dinkster.sd15",),
    knobs=RuntimeKnobs(
        diffusion_dtype=FLOAT16.name,
        text_dtype=FLOAT32.name,
        vae_dtype=FLOAT32.name,
        fp8_matmul=False,
    ),
)
runtime = SDRuntime(assembled, runtime_identity=recipe.runtime_identity)
mechanisms = []
capacity = [0]

def free_memory(_device):
    loaded = sum(mechanism.loaded_bytes() for mechanism in mechanisms)
    return DeviceMemory(free_total=max(0, capacity[0] - loaded), free_torch=0)

manager = ResidencyManager(
    policy=MemoryPolicy(
        inference_reserve=0,
        physical_headroom=0,
        min_weight_memory_ratio=1.0,
        load_inflation=1.0,
    ),
    free_memory=free_memory,
)
coordinator = NativeResidencyCoordinator(manager)
handle = NativeRuntimeHandle(
    runtime, device, recipe=recipe, coordinator=coordinator
)
mechanisms.extend(handle.mechanisms)
text = handle.mechanisms[1]
total = text.total_bytes()
capacity[0] = total + 1

with handle.stage("text"):
    expected = runtime.encode_text("a photo of a cat")
    assert text.loaded_bytes() == total

text.unload()
torch.cuda.empty_cache()
capacity[0] = total - 1
with handle.stage("text"):
    loaded = text.loaded_bytes()
    assert 0 < loaded < total
    actual = runtime.encode_text("a photo of a cat")

assert torch.equal(actual.embeddings, expected.embeddings)
assert actual.pooled is not None and expected.pooled is not None
assert torch.equal(actual.pooled, expected.pooled)
print(json.dumps({
    "total": total,
    "loaded": loaded,
    "offloaded": total - loaded,
}))
"""
    env = {**os.environ, "PYTHONPATH": dinkster_pythonpath()}
    completed = subprocess.run(
        [comfy_python(), "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout.splitlines()[-1])
    assert 0 < evidence["loaded"] < evidence["total"]
    assert evidence["offloaded"] == evidence["total"] - evidence["loaded"]


def test_native_aimdo_sd15_text_forward_is_bitwise_and_demand_paged() -> None:
    """An armed real worker selects AimdoWeights and stays bitwise exact."""
    if gpu_count() < 1:
        pytest.skip("needs a CUDA GPU")
    checkpoint = (
        Path(COMFY_ROOT) / "models" / "checkpoints" / "v1-5-pruned-emaonly-fp16.safetensors"
    )
    if not checkpoint.is_file():
        pytest.skip(f"required native checkpoint is absent: {checkpoint}")
    interpreter = Path(aimdo_python())
    if not interpreter.is_file():
        pytest.skip(f"Dinkster GPU interpreter is absent: {interpreter}")
    measured_total = int(
        subprocess.check_output(
            [
                str(interpreter),
                "-c",
                "import torch; print(torch.cuda.mem_get_info(torch.device('cuda', 0))[1])",
            ],
            text=True,
        ).strip()
    )
    expected_activation_headroom = min(64 * 1024**2, measured_total)
    declared_budget = measured_total - expected_activation_headroom
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest = Path(temp_dir) / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "aimdo-native-proof"\n\n[pack.entry]\n'
            'nodes = "aimdo_live_nodes:NODES"\n'
        )
        pyheaders = REPO_ROOT / ".venv-gpu-extras" / "pyheaders" / "usr" / "include"
        cpath = os.pathsep.join((str(pyheaders / "python3.12"), str(pyheaders)))
        if os.environ.get("CPATH"):
            cpath += os.pathsep + os.environ["CPATH"]

        async def scenario() -> dict[str, int]:
            registry = TypeRegistry()
            register_core_types(registry)
            worker = IsolatedWorker(
                manifest,
                registry,
                python=str(interpreter),
                aimdo_arm="on",
                reserve_vram=128 * 1024**2,
                vram_budgets={"vram:cuda:0": declared_budget},
                extra_env={
                    "PYTHONPATH": os.pathsep.join(
                        (str(Path(__file__).parent), dinkster_pythonpath())
                    ),
                    "CPATH": cpath,
                    "DINKSTER_AIMDO_TEST_CHECKPOINT": str(checkpoint),
                    "DINKSTER_AIMDO_TEST_CHECKPOINT_DIGEST": digest_file(checkpoint),
                },
                start_timeout=180.0,
            )
            await worker.start()
            try:
                engine = Engine(
                    schemas=dict(worker.schemas),
                    registry=registry,
                    worker=worker,
                    cache=MemoryLRUCache(),
                )
                result = await engine.run(
                    Graph(nodes={"proof": GraphNode("aimdo.native_proof", {})}),
                    ["proof"],
                )
                return {
                    name: cast(int, value.resolve())
                    for name, value in result.outputs["proof"].items()
                }
            finally:
                await worker.close()

        evidence = asyncio.run(scenario())
        assert evidence["off_mechanisms"] > 0
        assert evidence["off_demand_paged"] == 0
        assert evidence["on_mechanisms"] == evidence["on_demand_paged"]
        assert evidence["embeddings_equal"] == 1
        assert evidence["pooled_equal"] == 1
        assert evidence["resident_bytes"] > 0
        assert (
            evidence["dynamic_evictable_bytes"] + evidence["dynamic_pinned_bytes"]
            == evidence["resident_bytes"]
        )
        assert evidence["telemetry_free"] == min(
            evidence["telemetry_total"],
            evidence["raw_free"]
            + evidence["allocator_reclaimable_bytes"]
            + evidence["dynamic_evictable_bytes"],
        )
        assert evidence["telemetry_total"] == evidence["raw_total"]
        assert evidence["activation_headroom"] == expected_activation_headroom
        print(json.dumps(evidence, sort_keys=True))


def checkpoints_smallest_first() -> list[str]:
    ckpt_dir = Path(COMFY_ROOT) / "models" / "checkpoints"
    paths = sorted(ckpt_dir.glob("*.safetensors"), key=lambda p: p.stat().st_size)
    return [p.name for p in paths]


def smallest_checkpoint() -> str | None:
    names = checkpoints_smallest_first()
    return names[0] if names else None


class _Recorder:
    """Wraps a worker to observe which node_ids landed on it and when."""

    def __init__(self, inner: Worker) -> None:
        self.inner = inner
        self.spans: list[tuple[str, float, float]] = []  # (node_id, start, end)

    @property
    def node_ids(self) -> list[str]:
        return [node_id for node_id, _, _ in self.spans]

    def span(self, node_id: str) -> tuple[float, float]:
        return next((s, e) for n, s, e in self.spans if n == node_id)

    async def prepare(self, node_types) -> None:  # noqa: ANN001
        await self.inner.prepare(node_types)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        start = time.monotonic()
        route = cast("Any", self.inner).attention_route_token
        if route is not None:
            invocation = replace(
                invocation,
                attention_policy=route.requested_policy,
                attention_route_token=route,
            )
        result = await self.inner.invoke(invocation)
        self.spans.append((invocation.node_id, start, time.monotonic()))
        return result


def pinned_worker(device: int) -> IsolatedWorker:
    """An isolated Comfy worker pinned to one physical GPU. The child
    truthfully sees its GPU as cuda:0; DeviceMap puts its device facts
    into the parent namespace."""
    registry = TypeRegistry()
    register_core_types(registry)
    register_asset_type(registry)
    return IsolatedWorker(
        COMPAT_MANIFEST,
        registry,
        python=comfy_python(),
        extra_env={
            "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
            # sd_branch is all-native, loader included (see the comment
            # in test_real_inference_end_to_end_through_torchless_engine).
            "DINKSTER_COMFY_NODES": "CLIPTextEncode",
            "DINKSTER_ASSET_ROOT": str(models_root()),
            "PYTHONPATH": dinkster_pythonpath(),
            "CUDA_VISIBLE_DEVICES": str(device),
        },
        device_map=DeviceMap({"cuda:0": f"cuda:{device}"}),
        start_timeout=180.0,
    )


def sd_branch(
    suffix: str,
    checkpoint: dict[str, object],
    *,
    seed: int,
    steps: int = 4,
    width: int = 512,
    height: int = 512,
) -> dict[str, GraphNode]:
    """One load -> encode -> sample -> decode chain, node ids suffixed.
    ``checkpoint`` is an AssetRef wire literal (see checkpoint_ref_wire)."""
    return {
        f"load{suffix}": GraphNode("dinkster.load_checkpoint", {"checkpoint": checkpoint}),
        f"pos{suffix}": GraphNode(
            "dinkster.clip_text_encode",
            {"clip": Link(f"load{suffix}", "clip"), "text": "a photo of a cat"},
        ),
        f"neg{suffix}": GraphNode(
            "dinkster.clip_text_encode",
            {"clip": Link(f"load{suffix}", "clip"), "text": ""},
        ),
        f"lat{suffix}": GraphNode(
            "dinkster.empty_latent_image",
            {"width": width, "height": height, "batch_size": 1},
        ),
        f"sample{suffix}": GraphNode(
            "dinkster.ksampler",
            {
                "model": Link(f"load{suffix}", "model"),
                "seed": seed,
                "steps": steps,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": Link(f"pos{suffix}", "conditioning"),
                "negative": Link(f"neg{suffix}", "conditioning"),
                "latent_image": Link(f"lat{suffix}", "latent"),
                "denoise": 1.0,
            },
        ),
        f"decode{suffix}": GraphNode(
            "dinkster.vae_decode",
            {
                "samples": Link(f"sample{suffix}", "latent"),
                "vae": Link(f"load{suffix}", "vae"),
            },
        ),
    }


def test_placement_spans_two_pinned_gpu_workers() -> None:
    """Two isolated Comfy workers pinned to different physical GPUs
    (CUDA_VISIBLE_DEVICES=0/1), one placement layer over both. Proves live:

    - each pinned child honestly reports cuda:0 and DeviceMap translates
      its residency meta into distinct parent devices;
    - placement pins consumers of resident values to the owning worker
      (a CLIP encode never lands where its CLIP does not live);
    - two workflows sample concurrently, one per GPU, through the same
      homogeneous worker pool.
    """
    if gpu_count() < 2:
        pytest.skip("needs two CUDA GPUs")
    ckpt_name = smallest_checkpoint()
    if ckpt_name is None:
        pytest.skip("no .safetensors checkpoint under models/checkpoints")
    ckpt = checkpoint_ref_wire(ckpt_name)

    def sd_graph(suffix: str, seed: int) -> Graph:
        return Graph(nodes=sd_branch(suffix, ckpt, seed=seed))

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry)
        worker0, worker1 = pinned_worker(0), pinned_worker(1)
        await asyncio.gather(worker0.start(), worker1.start())
        try:
            rec0, rec1 = _Recorder(worker0), _Recorder(worker1)
            schemas = dict(worker0.schemas)
            assert set(worker1.schemas) == set(schemas)  # homogeneous pool

            def make_engine(fresh_load_target: str) -> Engine:
                # One placement view per workflow: residency pins invocations
                # to owners; only fresh (unpinned) loads follow the default.
                placement = PlacementWorker(
                    {"w0": rec0, "w1": rec1},
                    devices={"cuda:0": "w0", "cuda:1": "w1"},
                    default=fresh_load_target,
                )
                return Engine(
                    schemas=schemas,
                    registry=registry,
                    worker=placement,
                    cache=MemoryLRUCache(),
                )

            graph0, graph1 = sd_graph("_a", seed=7), sd_graph("_b", seed=11)
            result0, result1 = await asyncio.gather(
                make_engine("w0").run(graph0, ["load_a", "decode_a"]),
                make_engine("w1").run(graph1, ["load_b", "decode_b"]),
            )

            # DeviceMap live: two children both honestly said cuda:0; the
            # parent sees the models on distinct silicon.
            model0 = result0.outputs["load_a"]["model"]
            model1 = result1.outputs["load_b"]["model"]
            resources0 = model0.meta.get("resources")
            resources1 = model1.meta.get("resources")
            assert isinstance(resources0, dict) and resources0["gpu"] == "cuda:0"
            assert isinstance(resources1, dict) and resources1["gpu"] == "cuda:1"
            cost1 = model1.meta.get("cost")
            assert isinstance(cost1, dict)
            assert any(key.startswith("vram:cuda:1") for key in cost1)

            # Real images came out of both GPUs.
            for result, out in ((result0, "decode_a"), (result1, "decode_b")):
                image = result.outputs[out]["image"]
                assert image.type_id == "comfy.IMAGE"
                assert len(image.payload.data) > 512 * 512 * 3  # type: ignore[attr-defined]

            # Placement kept every workflow with its owner: nothing from
            # graph0 touched worker1 and vice versa.
            assert {n for n in rec0.node_ids} == {
                "load_a",
                "pos_a",
                "neg_a",
                "lat_a",
                "sample_a",
                "decode_a",
            }
            assert {n for n in rec1.node_ids} == {
                "load_b",
                "pos_b",
                "neg_b",
                "lat_b",
                "sample_b",
                "decode_b",
            }
        finally:
            await asyncio.gather(worker0.close(), worker1.close())

    asyncio.run(scenario())


def test_serving_composer_runs_concurrent_jobs_on_cuda_replicas() -> None:
    if gpu_count() < 2:
        pytest.skip("needs two CUDA GPUs")
    ckpt_name = smallest_checkpoint()
    if ckpt_name is None:
        pytest.skip("no .safetensors checkpoint under models/checkpoints")
    checkpoint = Path(COMFY_ROOT) / "models" / "checkpoints" / ckpt_name
    checkpoint_ref = checkpoint_ref_wire(ckpt_name)
    digest = cast("str", checkpoint_ref["digest"])

    async def scenario() -> None:
        specs = comfy_compat_specs(
            COMFY_ROOT,
            python=comfy_python(),
            comfy_nodes=("CLIPTextEncode",),
            multi_device_cuda_indices=(0, 1),
        )
        generation, compat = specs
        compat = replace(
            compat,
            env={**compat.env, "DINKSTER_ASSET_ROOT": str(models_root())},
        )
        policy = NativeDispatchPolicy(
            lambda candidate: checkpoint if candidate == digest else None,
            lambda _diagnostic: None,
        )
        composer = ServingComposer(native_policy=policy)
        try:
            await _add_generation_compat_specs(composer, generation, compat)
            engine = composer.composition.make_engine(lambda _event: None)
            graph_a = Graph(nodes=sd_branch("_a", checkpoint_ref, seed=7, steps=2))
            graph_b = Graph(nodes=sd_branch("_b", checkpoint_ref, seed=11, steps=2))
            result_a, result_b = await asyncio.gather(
                engine.run(graph_a, ["load_a", "decode_a"], run_id="replica-a"),
                engine.run(graph_b, ["load_b", "decode_b"], run_id="replica-b"),
            )

            resources_a = result_a.outputs["load_a"]["model"].meta.get("resources")
            resources_b = result_b.outputs["load_b"]["model"].meta.get("resources")
            assert isinstance(resources_a, dict) and resources_a["gpu"] == "cuda:0"
            assert isinstance(resources_b, dict) and resources_b["gpu"] == "cuda:1"
            for result, output in ((result_a, "decode_a"), (result_b, "decode_b")):
                image = result.outputs[output]["image"]
                assert image.type_id == "comfy.IMAGE"
                assert len(image.payload.data) > 512 * 512 * 3  # type: ignore[attr-defined]
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(("size", "steps"), ((512, 20), (1024, 30)))
def test_sd15_single_job_guidance_matches_split_reference_golden(
    size: int,
    steps: int,
) -> None:
    if gpu_count() < 2:
        pytest.skip("needs two CUDA GPUs")
    ckpt_name = "v1-5-pruned-emaonly-fp16.safetensors"
    checkpoint = Path(COMFY_ROOT) / "models" / "checkpoints" / ckpt_name
    if not checkpoint.is_file():
        pytest.skip(f"SD1.5 checkpoint not present: {checkpoint}")
    checkpoint_ref = checkpoint_ref_wire(ckpt_name)
    digest = cast("str", checkpoint_ref["digest"])

    async def execute(single_job: bool) -> tuple[bytes, bytes]:
        specs = comfy_compat_specs(
            COMFY_ROOT,
            python=comfy_python(),
            comfy_nodes=(
                "CLIPTextEncode",
                "CreateHookLora",
                "CreateHookKeyframe",
                "SetHookKeyframes",
                "ConditioningTimestepsRange",
                "ConditioningSetPropertiesAndCombine",
                "PairConditioningSetProperties",
            ),
            single_job_multi_gpu=(
                SingleJobMultiGpuConfig((0, 1), "guidance") if single_job else None
            ),
        )
        generation, compat = specs
        compat = replace(
            compat,
            env={**compat.env, "DINKSTER_ASSET_ROOT": str(models_root())},
        )
        policy = NativeDispatchPolicy(
            lambda candidate: checkpoint if candidate == digest else None,
            lambda _diagnostic: None,
            # The split-reference records were generated at float32 text;
            # the "auto" text default is bfloat16 and would move every
            # conditioning value under the pinned production sha256s.
            dtype_policy=lambda: {
                "diffusion": "auto",
                "textEncoder": "float32",
                "vae": "auto",
            },
        )
        governor = MemoryGovernor()
        composer = ServingComposer(
            native_policy=policy,
            governor=governor,
            reservations=GovernorReservationService(governor),
        )
        try:
            await _add_generation_compat_specs(composer, generation, compat)
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes=sd_branch(
                    "",
                    checkpoint_ref,
                    seed=264,
                    steps=steps,
                    width=size,
                    height=size,
                )
            )
            result = await engine.run(graph, ["sample"])
            latent = result.outputs["sample"]["latent"]
            assert isinstance(latent.payload, EncodedPayload)
            decoded = latent.payload.load()
            assert isinstance(decoded, Mapping)
            samples = decoded.get("samples")
            assert isinstance(samples, EncodedLatentTensor)
            assert samples.dtype == "float32"
            assert samples.shape == (1, 4, size // 8, size // 8)
            return samples.data, latent.payload.data
        finally:
            await composer.close()

    record_dir = {
        (8, 9): "distributed-sd15-fp16-guidance-d1",
        (12, 0): "distributed-sd15-fp16-guidance-sm120-d1",
    }.get(gpu_compute_capability() or (0, 0))
    if record_dir is None:
        pytest.skip("no SD1.5 split-reference record for this device capability")
    single, distributed = asyncio.run(execute(False)), asyncio.run(execute(True))
    expected = {
        (case["width"], case["steps"]): case
        for case in json.loads(
            (INFERENCE_PARITY_RECORDS / record_dir / "split-reference.json").read_text()
        )["cases"]
    }
    case = expected[(size, steps)]
    assert hashlib.sha256(single[0]).hexdigest() == case["production_fused_samples_sha256"]
    assert hashlib.sha256(single[1]).hexdigest() == case["production_fused_encoded_sha256"]
    assert hashlib.sha256(distributed[0]).hexdigest() == case["production_split_samples_sha256"]
    assert hashlib.sha256(distributed[1]).hexdigest() == case["production_split_encoded_sha256"]


def test_two_checkpoints_sample_in_parallel_within_one_workflow() -> None:
    """One graph, two different checkpoints, two GPUs. Proves live that
    parallelism *within* a single workflow works end to end:

    - the ready-set scheduler dispatches both independent branches at once
      (compute lane widened deliberately so the two loads overlap);
    - the placement policy spreads the unpinned loads across workers, and
      residency pins everything downstream to its model's owner;
    - the two KSamplers occupy distinct concrete lanes (gpu:cuda:0 vs
      gpu:cuda:1), so their executions genuinely overlap in time.

    Needs two distinct checkpoints: identical loads are one computation by
    content-addressed identity, and coalescing them is correct behavior.
    """
    if gpu_count() < 2:
        pytest.skip("needs two CUDA GPUs")
    ckpts = checkpoints_smallest_first()[:2]
    if len(ckpts) < 2:
        pytest.skip("needs two distinct .safetensors checkpoints")
    ckpt_a, ckpt_b = (checkpoint_ref_wire(name) for name in ckpts)

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry)
        worker0, worker1 = pinned_worker(0), pinned_worker(1)
        await asyncio.gather(worker0.start(), worker1.start())
        try:
            rec0, rec1 = _Recorder(worker0), _Recorder(worker1)
            placement = PlacementWorker(
                {"w0": rec0, "w1": rec1},
                devices={"cuda:0": "w0", "cuda:1": "w1"},
                # Only unpinned invocations (the loads, the empty latents)
                # ever reach the policy; residency owns the rest.
                place=lambda inv: "w0" if inv.node_id.endswith("_a") else "w1",
            )
            engine = Engine(
                schemas=dict(worker0.schemas),
                registry=registry,
                worker=placement,
                cache=MemoryLRUCache(),
                # Plain nodes default to one shared compute permit; widening
                # it is the deliberate opt-in that lets both branches' loads
                # run at once. GPU lanes need nothing: they are per-device.
                resource_capacities={"compute": 4},
            )
            graph = Graph(
                nodes={
                    **sd_branch("_a", ckpt_a, seed=7, steps=24),
                    **sd_branch("_b", ckpt_b, seed=11, steps=24),
                }
            )
            result = await engine.run(graph, ["load_a", "decode_a", "load_b", "decode_b"])

            # The policy spread the loads; the parent sees two models on
            # distinct silicon within one run.
            resources_a = result.outputs["load_a"]["model"].meta.get("resources")
            resources_b = result.outputs["load_b"]["model"].meta.get("resources")
            assert isinstance(resources_a, dict) and resources_a["gpu"] == "cuda:0"
            assert isinstance(resources_b, dict) and resources_b["gpu"] == "cuda:1"

            # Residency kept each branch whole on its owner. The two empty
            # latents are the *same computation* by content-addressed
            # identity (identical inputs), so single-flight coalesces them:
            # exactly one executes, on whichever branch won the race.
            executed_lats = {"lat_a", "lat_b"} & set(rec0.node_ids + rec1.node_ids)
            assert len(executed_lats) == 1
            assert {n for n in rec0.node_ids} - {"lat_a"} == {
                "load_a",
                "pos_a",
                "neg_a",
                "sample_a",
                "decode_a",
            }
            assert {n for n in rec1.node_ids} - {"lat_b"} == {
                "load_b",
                "pos_b",
                "neg_b",
                "sample_b",
                "decode_b",
            }

            # Genuine concurrency, not just correct routing: the two loads
            # overlapped (widened compute lane), and the two samplers
            # overlapped (distinct concrete GPU lanes).
            load_a, load_b = rec0.span("load_a"), rec1.span("load_b")
            assert max(load_a[0], load_b[0]) < min(load_a[1], load_b[1])
            sample_a, sample_b = rec0.span("sample_a"), rec1.span("sample_b")
            assert max(sample_a[0], sample_b[0]) < min(sample_a[1], sample_b[1])

            # Two real images from two different models.
            image_a = result.outputs["decode_a"]["image"]
            image_b = result.outputs["decode_b"]["image"]
            for image in (image_a, image_b):
                assert image.type_id == "comfy.IMAGE"
                assert len(image.payload.data) > 512 * 512 * 3  # type: ignore[attr-defined]
            assert image_a.fingerprint != image_b.fingerprint
        finally:
            await asyncio.gather(worker0.close(), worker1.close())

    asyncio.run(scenario())


LEGACY_MANIFEST = REPO_ROOT / "packages" / "dinkster-compat-comfy" / "dinkster-legacy-pack.toml"
RGTHREE_PACK = os.environ.get("DINKSTER_LEGACY_RGTHREE", "/tmp/dinkster-legacy-packs/rgthree-comfy")


def test_current_upstream_async_pack_executes_and_times_out() -> None:
    """Drive current ComfyUI's real async execution fixtures end to end."""
    from dinkster.compose import compose_serving

    testing_pack = Path(COMFY_ROOT) / "tests" / "execution" / "testing_nodes" / "testing-pack"
    if not testing_pack.is_dir():
        pytest.skip(f"current upstream async testing pack not found at {testing_pack}")

    async def scenario() -> None:
        composition = await compose_serving(
            comfy_compat_specs(
                COMFY_ROOT,
                python=comfy_python(),
                comfy_nodes=["EmptyImage"],
                legacy_packs=[testing_pack],
            )
        )
        try:
            async_type = "comfy.testing-pack.TestAsyncBatchProcessing"
            timeout_type = "comfy.testing-pack.TestAsyncTimeout"
            assert async_type in composition.schemas
            assert timeout_type in composition.schemas
            engine = composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode(
                        "comfy.EmptyImage",
                        {"width": 48, "height": 32, "batch_size": 2, "color": 0},
                    ),
                    "async": GraphNode(
                        async_type,
                        {
                            "images": Link("source", "image"),
                            "process_time_per_item": 0.01,
                        },
                    ),
                }
            )
            result = await engine.run(graph, ["async"])
            image = result.outputs["async"]["image"].resolve()
            assert isinstance(image, np.ndarray)
            assert image.shape == (2, 32, 48, 3)
            assert np.all(image == 1.0)

            timeout_graph = Graph(
                nodes={
                    "timeout": GraphNode(
                        timeout_type,
                        {
                            "value": TypedLiteral("core.int", 1),
                            "timeout": 0.1,
                            "operation_time": 0.2,
                        },
                    )
                }
            )
            with pytest.raises(ExecutionError, match="Operation timed out after 0.1 seconds"):
                await engine.run(timeout_graph, ["timeout"])
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_current_upstream_async_lazy_check_executes() -> None:
    """Drive current ComfyUI's real TestAsyncLazyCheck through M1."""
    from dinkster.compose import compose_serving

    testing_pack = Path(COMFY_ROOT) / "tests" / "execution" / "testing_nodes" / "testing-pack"
    if not testing_pack.is_dir():
        pytest.skip(f"current upstream async testing pack not found at {testing_pack}")

    async def scenario() -> None:
        composition = await compose_serving(
            comfy_compat_specs(
                COMFY_ROOT,
                python=comfy_python(),
                comfy_nodes=["EmptyImage"],
                legacy_packs=[testing_pack],
            )
        )
        try:
            lazy_type = "comfy.testing-pack.TestAsyncLazyCheck"
            assert lazy_type in composition.schemas
            engine = composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "selected": GraphNode(
                        "comfy.EmptyImage",
                        {"width": 32, "height": 24, "batch_size": 1, "color": 0},
                    ),
                    "unselected": GraphNode(
                        "comfy.EmptyImage",
                        {"width": 16, "height": 16, "batch_size": 1, "color": 0},
                    ),
                    "lazy": GraphNode(
                        lazy_type,
                        {
                            "input1": Link("selected", "image"),
                            "input2": Link("unselected", "image"),
                            "condition": True,
                        },
                    ),
                }
            )
            result = await engine.run(graph, ["lazy"])
            image = result.outputs["lazy"]["image"].resolve()
            assert isinstance(image, np.ndarray)
            assert image.shape == (1, 512, 512, 3)
            assert np.all(image == 1.0)
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_composed_server_serves_comfy_compat_with_provenance() -> None:
    """The serve wiring end-to-end: compose_serving + comfy_compat_specs
    put std nodes, the translated comfy core surface, and (when the clone
    is present) an unmodified legacy pack behind ONE server - and
    /api/nodes attributes each surface truthfully ('core' / 'comfy' /
    'comfy.<pack>')."""
    from aiohttp.test_utils import TestClient, TestServer
    from dinkster_graph import graph_to_wire
    from dinkster_server import create_app

    from dinkster.comfy_compose import comfy_compat_specs
    from dinkster.compose import compose_serving

    with_legacy = Path(RGTHREE_PACK).is_dir()

    async def scenario() -> None:
        specs = comfy_compat_specs(
            COMFY_ROOT,
            python=comfy_python(),
            comfy_nodes=["EmptyLatentImage", "LatentUpscale"],
            legacy_packs=[RGTHREE_PACK] if with_legacy else (),
        )
        composition = await compose_serving(specs)
        try:
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                data = await (await client.get("/api/nodes")).json()
                assert data["packs"]["comfy"] == {
                    "displayName": "ComfyUI Compat",
                    "abbr": "C1",
                    "color": "#4a7ab5",
                }
                # Legacy and native latent sources remain separate typed
                # contracts instead of blessing comfy.LATENT as native.
                assert data["nodes"]["comfy.LatentUpscale"]["pack"] == "comfy"
                assert data["nodes"]["comfy.EmptyLatentImage"]["pack"] == "comfy"
                assert data["nodes"]["dinkster.empty_latent_image"]["pack"] == (
                    "dinkster-nodes-generation"
                )
                assert data["nodes"]["dinkster.load_checkpoint"]["pack"] == (
                    "dinkster-nodes-generation"
                )
                assert data["nodes"]["dinkster.load_checkpoint"]["executionArms"] == ["native"]
                assert data["nodes"]["dinkster.ksampler"]["pack"] == "dinkster-nodes-generation"
                assert data["nodes"]["std.math.add_ints"]["pack"] == "dinkster-nodes-foundation"
                if with_legacy:
                    # rgthree ships no dinkster-pack.toml, so it wears the
                    # shared default legacy badge.
                    assert data["packs"]["comfy.rgthree-comfy"] == {
                        "displayName": "rgthree-comfy (ComfyUI)",
                        "mark": "\U0001f9e9",
                        "color": "#4a7ab5",
                    }
                    legacy_nodes = [
                        node
                        for node, entry in data["nodes"].items()
                        if entry["pack"] == "comfy.rgthree-comfy"
                    ]
                    assert len(legacy_nodes) > 20

                # One job through the composed queue: a comfy latent
                # crosses between compat nodes; the server never resolves
                # it (no torch in this process).
                graph = Graph(
                    nodes={
                        "e": GraphNode(
                            "comfy.EmptyLatentImage",
                            {"width": 64, "height": 64, "batch_size": 1},
                        ),
                        "u": GraphNode(
                            "comfy.LatentUpscale",
                            {
                                "samples": Link("e", "latent"),
                                "upscale_method": "nearest-exact",
                                "width": 128,
                                "height": 128,
                                "crop": "disabled",
                            },
                        ),
                    }
                )
                body = {
                    "clientId": "c1",
                    "jobId": "j1",
                    "graph": graph_to_wire(graph),
                    "targets": ["u"],
                }
                resp = await client.post("/api/jobs", json=body)
                assert resp.status == 202, await resp.text()
                status: dict[str, object] = {}
                for _ in range(600):
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] in ("completed", "failed"):
                        break
                    await asyncio.sleep(0.1)
                assert status.get("state") == "completed", json.dumps(status, indent=2)
            finally:
                await client.close()
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_unmodified_rgthree_pack_runs_in_quarantine(tmp_path: Path) -> None:
    """DESIGN 3.8 spike: a real, unmodified custom-node pack (rgthree-comfy,
    top-tier by actual usage) loads in the legacy quarantine worker and one
    of its nodes executes end-to-end - while its server/execution hooks are
    counted and reported, not emulated."""
    if not Path(RGTHREE_PACK).is_dir():
        pytest.skip(f"rgthree-comfy clone not found at {RGTHREE_PACK}")

    node_type = "comfy.rgthree-comfy.SDXL Empty Latent Image (rgthree)"

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        report_path = tmp_path / "legacy-report.json"
        worker = IsolatedWorker(
            LEGACY_MANIFEST,
            registry,
            python=comfy_python(),
            extra_env={
                "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
                "DINKSTER_LEGACY_PACKS": RGTHREE_PACK,
                "DINKSTER_LEGACY_REPORT": str(report_path),
                "PYTHONPATH": dinkster_pythonpath(),
            },
            start_timeout=240.0,
        )
        await worker.start()
        try:
            # Every announced node is namespaced to the pack; nothing from
            # the pack leaked into an unnamespaced id.
            assert node_type in worker.schemas
            assert all(t.startswith("comfy.rgthree-comfy.") for t in worker.schemas)

            # The structured report is the diagnostic surface: the pack
            # loaded, its web-route registrations were counted (Dinkster will
            # not serve them), and its runtime imports are on record.
            import json as json_module

            (report,) = json_module.loads(report_path.read_text(encoding="utf-8"))
            assert report["status"] == "loaded"
            assert report["pack_id"] == "rgthree-comfy"
            assert report["nodes_translated"] > 20
            assert report["server_routes_added"] > 0
            assert "execution" in report["hook_imports"]

            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                nodes={
                    "lat": GraphNode(
                        node_type,
                        {
                            "dimensions": "1024 x 1024  (square)",
                            "clip_scale": 2.0,
                            "batch_size": 1,
                        },
                    ),
                }
            )
            result = await engine.run(graph, ["lat"])
            latent = result.outputs["lat"]["LATENT"]
            assert latent.type_id == "comfy.LATENT"
            assert latent.fingerprint
            assert result.outputs["lat"]["CLIP_WIDTH"].resolve() == 2048
            assert result.outputs["lat"]["CLIP_HEIGHT"].resolve() == 2048

            # Same graph again: cache hits; the quarantined pack idles.
            again = await engine.run(graph, ["lat"])
            assert again.executed == ()
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_real_v3_nodes_execute_through_compat() -> None:
    """Regression for the frontend-reported V3 breakage: ComfyUI's V3
    io.ComfyNode classes ride NODE_CLASS_MAPPINGS behind the v1 shim, so
    their FUNCTION returns io.NodeOutput. Before the unwrap fix every one
    of them translated fine and then died at execute with 'returned
    NodeOutput, expected a tuple'. The original repro nodes (the V3
    Primitive* classes) are claimed by dinkster-nodes-foundation's natives now
    (STD_CLAIMED_V1_NAMES evicts their translated twins), so this drives
    two unclaimed core V3 nodes instead - same shim, same NodeOutput
    return path."""

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = IsolatedWorker(
            COMPAT_MANIFEST,
            registry,
            python=comfy_python(),
            extra_env={
                "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
                "DINKSTER_COMFY_NODES": "StringConcatenate,StringLength",
                "PYTHONPATH": dinkster_pythonpath(),
            },
            start_timeout=180.0,
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            concat_out = worker.schemas["comfy.StringConcatenate"].outputs[0].id
            length_out = worker.schemas["comfy.StringLength"].outputs[0].id
            graph = Graph(
                nodes={
                    "c": GraphNode(
                        "comfy.StringConcatenate",
                        {"string_a": "dom", "string_b": "fy", "delimiter": ""},
                    ),
                    "n": GraphNode(
                        "comfy.StringLength",
                        {"string": Link("c", concat_out)},
                    ),
                }
            )
            result = await engine.run(graph, ["c", "n"])
            assert result.outputs["c"][concat_out].resolve() == "dinkster"
            assert result.outputs["n"][length_out].resolve() == 5
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_comfy_switch_catalog_and_pruning() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = IsolatedWorker(
            COMPAT_MANIFEST,
            registry,
            python=comfy_python(),
            extra_env={
                "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
                "PYTHONPATH": dinkster_pythonpath(),
            },
            start_timeout=180.0,
        )
        await worker.start()
        events: list[EngineEvent] = []
        try:
            assert "comfy.ComfySwitchNode" in worker.schemas
            assert "comfy.CustomCombo" in worker.schemas, worker.compat_skips
            assert worker.compat_skips == {}
            schemas = dict(worker.schemas)
            switch_schema = schemas["comfy.ComfySwitchNode"]
            waiting = await worker.check_lazy_status(
                LazyStatusInvocation(
                    request_id="current-switch-waiting",
                    node_id="switch",
                    node_type=switch_schema.node_type,
                    available_inputs={
                        "switch": registry.wrap(CORE_BOOLEAN, True),
                    },
                    connected_undemanded_inputs=("on_false", "on_true"),
                    effective_schema=switch_schema,
                )
            )
            ready = await worker.check_lazy_status(
                LazyStatusInvocation(
                    request_id="current-switch-ready",
                    node_id="switch",
                    node_type=switch_schema.node_type,
                    available_inputs={
                        "switch": registry.wrap(CORE_BOOLEAN, True),
                        "on_true": registry.wrap(CORE_STRING, "selected"),
                    },
                    connected_undemanded_inputs=("on_false",),
                    effective_schema=switch_schema,
                )
            )
            assert waiting.requested_inputs == ("on_true",)
            assert ready.error is None
            assert ready.requested_inputs == ()
            custom_schema = schemas["comfy.CustomCombo"]
            assert [item.id for item in custom_schema.inputs] == ["choice", "index"]
            assert custom_schema.input_families[0].member_names == tuple(
                f"option{index}" for index in range(1, 101)
            )
            custom_schemas = dict(schemas)
            custom_schemas["comfy.CustomCombo"] = replace(custom_schema, output_node=True)
            custom_prompt = translate_prompt(
                {
                    "custom": {
                        "class_type": "CustomCombo",
                        "inputs": {
                            "choice": "second",
                            "index": 1,
                            "option1": "first",
                            "option2": "second",
                        },
                    }
                },
                custom_schemas,
                input_adapters=COMFY_INPUT_ADAPTERS,
            )
            custom_engine = Engine(
                schemas=custom_schemas,
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            custom_result = await custom_engine.run(custom_prompt.graph, custom_prompt.targets)
            assert custom_result.outputs["custom"]["STRING"].resolve() == "second"
            assert custom_result.outputs["custom"]["INDEX"].resolve() == 1
            schemas["comfy.StringLength"] = replace(schemas["comfy.StringLength"], output_node=True)
            length_out = schemas["comfy.StringLength"].outputs[0].id
            prompt = {
                "inactive": {
                    "class_type": "StringConcatenate",
                    "inputs": {
                        "string_a": "never",
                        "string_b": "runs",
                        "delimiter": "",
                    },
                },
                "active": {
                    "class_type": "StringConcatenate",
                    "inputs": {"string_a": "dom", "string_b": "fy", "delimiter": ""},
                },
                "switch": {
                    "class_type": "ComfySwitchNode",
                    "inputs": {
                        "switch": True,
                        "on_false": ["inactive", 0],
                        "on_true": ["active", 0],
                    },
                },
                "length": {
                    "class_type": "StringLength",
                    "inputs": {"string": ["switch", 0]},
                },
            }
            translated = translate_prompt(prompt, schemas)
            assert set(translated.graph.nodes) == {"active", "length"}
            engine = Engine(
                schemas=schemas,
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
                on_event=events.append,
            )
            result = await engine.run(translated.graph, translated.targets)
            assert result.outputs["length"][length_out].resolve() == 5
            assert {e.node_id for e in events if e.kind == "node_started"} == {
                "active",
                "length",
            }
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_resize_image_mask_defaults_to_native_semantics_end_to_end() -> None:
    """Compatibility composition retires the source node in favor of native resize."""
    from dinkster_schema import TypeExpr

    from dinkster.compose import compose_serving

    def nearest_exact(image: np.ndarray, width: int, height: int) -> np.ndarray:
        # torch F.interpolate(mode="nearest-exact"):
        # src = min(floor((dst + 0.5) * in / out), in - 1) per axis.
        _, in_h, in_w, _ = image.shape
        rows = np.minimum(((np.arange(height) + 0.5) * in_h / height).astype(int), in_h - 1)
        cols = np.minimum(((np.arange(width) + 0.5) * in_w / width).astype(int), in_w - 1)
        return image[:, rows][:, :, cols]

    def center_narrow(image: np.ndarray, width: int, height: int) -> np.ndarray:
        # comfy.utils.common_upscale crop="center": narrow the SOURCE to the
        # target aspect before interpolating.
        _, in_h, in_w, _ = image.shape
        old_aspect = in_w / in_h
        new_aspect = width / height
        if old_aspect > new_aspect:
            x = round((in_w - in_w * new_aspect / old_aspect) / 2)
            return image[:, :, x : in_w - x, :]
        if old_aspect < new_aspect:
            y = round((in_h - in_h * old_aspect / new_aspect) / 2)
            return image[:, y : in_h - y, :, :]
        return image

    async def scenario() -> None:
        composition = await compose_serving(
            comfy_compat_specs(
                COMFY_ROOT,
                python=comfy_python(),
                comfy_nodes=["ResizeImageMaskNode", "EmptyImage", "SolidMask", "MaskToImage"],
            )
        )
        try:
            schemas = composition.schemas
            assert "comfy.ResizeImageMaskNode" not in schemas
            schema = schemas["dinkster.image.resize"]

            # The MatchType template covers images and masks, in and out.
            image_or_mask = TypeExpr.variable("input_type", ("dinkster.image", "dinkster.mask"))
            input_spec = next(spec for spec in schema.inputs if spec.id == "image")
            assert input_spec.type == image_or_mask
            assert len(schema.outputs) == 2
            assert schema.outputs[0].type == image_or_mask
            out = schema.outputs[0].id

            # The dynamic combo carries all nine upstream resize modes with
            # their exact space-containing keys and per-mode inputs.
            combo = next(c for c in schema.combos if c.id == "target")
            assert {option.key for option in combo.options} == {
                "dimensions",
                "factor",
                "height",
                "longest",
                "match",
                "multiple_cover",
                "shortest",
                "total_pixels",
                "width",
            }
            dims_option = combo.option("dimensions")
            assert dims_option is not None
            assert [entry.id for entry in dims_option.inputs] == ["width", "height"]
            match_option = combo.option("match")
            assert match_option is not None
            assert [entry.id for entry in match_option.inputs] == ["reference"]

            engine = composition.make_engine(lambda _event: None)

            # Value parity on a gradient literal: nearest-exact resampling
            # is pure pixel selection, so outputs must be exactly equal to
            # upstream's documented math. Graph literals are JSON-shaped, so
            # the gradient rides as nested lists (integer-valued, hence
            # exact in any float width the boundary picks).
            gradient = np.arange(1 * 5 * 7 * 3, dtype=np.float32).reshape(1, 5, 7, 3)

            def literal_resize(target: str, mode: str = "stretch", **extra: object) -> GraphNode:
                dotted = {f"target.{key}": value for key, value in extra.items()}
                return GraphNode(
                    "dinkster.image.resize",
                    {
                        "image": TypedLiteral("dinkster.image", gradient.tolist()),
                        "interpolation": "nearest-exact",
                        **dotted,
                    },
                    slot_variants={"target": target, "mode": mode, "divisibility": "none"},
                )

            value_graph = Graph(
                nodes={
                    "g_by": literal_resize("factor", factor=1.5),
                    "g_crop": literal_resize("dimensions", "fill", width=4, height=4),
                    "g_mult": literal_resize("multiple_cover", multiple_of=4),
                }
            )
            values = await engine.run(value_graph, ["g_by", "g_crop", "g_mult"])

            def resolved(node_id: str) -> np.ndarray:
                value = values.outputs[node_id][out].resolve()
                assert isinstance(value, np.ndarray), node_id
                return value

            # scale by 1.5: no crop, resample straight to 10x8.
            assert np.array_equal(resolved("g_by"), nearest_exact(gradient, 10, 8))
            # scale dimensions 4x4 center: narrow 7x5 to the square aspect
            # first (x=round((7-5)/2)=1 -> columns 1..5), then resample.
            assert np.array_equal(
                resolved("g_crop"),
                nearest_exact(center_narrow(gradient, 4, 4), 4, 4),
            )
            # scale to multiple 4: targets (7//4*4, 5//4*4) = 4x4;
            # cover-scale wins on height (s_h=0.8, ceil(7*0.8)=6) so the
            # intermediate is 6x4, then center-crop x0=(6-4)//2=1, y0=0.
            assert np.array_equal(
                resolved("g_mult"),
                nearest_exact(gradient, 6, 4)[:, 0:4, 1:5, :],
            )
        finally:
            await composition.close()

    asyncio.run(scenario())
