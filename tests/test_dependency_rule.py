"""Hazard H6: one-way package dependencies, enforced.

Parses every module in packages/*/src and asserts no package imports a dinkster
package outside its allowed set. This is the CI teeth behind the layering
diagram in DESIGN.md.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import yaml

from tools.evidence_paths import EVIDENCE_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWED: dict[str, set[str]] = {
    # Acceptance is an installable end-to-end harness, not a reusable runtime
    # layer. It deliberately composes the production engine, worker, pack, and
    # inference boundaries that a deployed acceptance run must cross.
    "dinkster_acceptance": {
        "dinkster_caches",
        "dinkster_compat_comfy",
        "dinkster_engine",
        "dinkster_graph",
        "dinkster_inference",
        "dinkster_inference_torch",
        "dinkster_schema",
        "dinkster_values",
        "dinkster_workers",
    },
    "dinkster_values": set(),
    # Sideways edge within the bottom layer: the list type-id grammar is
    # owned by dinkster_values (type ids are registry namespace) and schema's
    # TypeExpr must use the same one (DESIGN 3.13).
    "dinkster_schema": {"dinkster_values"},
    "dinkster_graph": {"dinkster_schema"},
    # Native inference contracts contain protocols and frozen data only.
    # dinkster_schema supplies the closed name grammar for registry ids;
    # The conditioning carrier uses the bottom-layer dinkster_values package.
    # Nothing else (and never torch) belongs here.
    "dinkster_inference": {"dinkster_schema", "dinkster_protocol", "dinkster_values"},
    # The torch half provides adapter tensor math, weight materialization,
    # and PatchSet application. It consumes torch-free contracts from
    # dinkster_inference and the asset system's concrete verified-open boundary;
    # accepting structural lookalikes would let low-level loaders bypass
    # content verification. Both dependencies point downward.
    "dinkster_inference_torch": {
        "dinkster_schema",
        "dinkster_inference",
        "dinkster_assets",
        "dinkster_memory",
        "dinkster_kernels",
    },
    # Fused GPU kernels are a leaf: torch/triton only, no dinkster imports,
    # so consumers can probe for them without pulling anything upward.
    "dinkster_kernels": set(),
    # The execution boundary itself (hazard H3): Worker/CacheStore protocols
    # plus the frozen data they exchange. A leaf on purpose - both the engine
    # (consumer) and workers/caches (implementations) depend on it, so a
    # worker child interpreter never transitively installs the scheduler.
    "dinkster_protocol": {"dinkster_schema", "dinkster_values"},
    "dinkster_engine": {
        "dinkster_schema",
        "dinkster_values",
        "dinkster_graph",
        "dinkster_protocol",
    },
    # dinkster_assets is allowed because the manifest owns [[pack.assets]]
    # parsing (DeclaredAsset/AssetNeed are the canonical shapes - a
    # workers-local mirror type would duplicate them across layers).
    # Downward edge only: dinkster_assets sits in the bottom layer, depending
    # on nothing but dinkster_values.
    # Workers implement the boundary, never the engine behind it: an
    # import of dinkster_engine here would put the scheduler back into every
    # worker child environment (the layering bug this split fixed).
    # dinkster_caches is allowed so the service daemon composes its persistent
    # value store from the canonical CAS machinery instead of duplicating
    # it. The edge stays downward: caches never imports workers, and its
    # own dependencies are already in this set, so worker child
    # interpreters still never transitively install the scheduler.
    "dinkster_workers": {
        "dinkster_schema",
        "dinkster_values",
        "dinkster_protocol",
        "dinkster_memory",
        "dinkster_assets",
        "dinkster_caches",
    },
    "dinkster_memory": {"dinkster_values"},
    # dinkster_assets is allowed so DiskCAS reuses the canonical blake3 digest
    # identity instead of inventing a second namespace.
    "dinkster_caches": {
        "dinkster_values",
        "dinkster_protocol",
        "dinkster_memory",
        "dinkster_assets",
    },
    "dinkster_assets": {"dinkster_values"},
    "dinkster_native": {
        "dinkster_image_document",
        "dinkster_schema",
        "dinkster_values",
        "dinkster_video",
        "dinkster_assets",
        "dinkster_memory",
        "dinkster_graph",
        "dinkster_inference",
        "dinkster_inference_torch",
        "dinkster_protocol",
        "dinkster_nodes_generation",
        "dinkster_workers",
    },
    # Shared materialization consumes portable values, never packs or the scheduler.
    "dinkster_video": {"dinkster_values", "dinkster_image_document"},
    "dinkster_image_document": {"dinkster_assets", "dinkster_values"},
    # The host-managed libtorrent sidecar consumes strict asset descriptors
    # and the established private process boundary without importing the server.
    "dinkster_p2p": {"dinkster_assets", "dinkster_workers"},
    # Packs may declare reservation policy (dinkster_memory's planner contracts)
    # but never touch the engine: planners observe InvocationView, not
    # Invocation. dinkster_graph is allowed for the Comfy API prompt adapter:
    # translating a v1 prompt means *producing* a native Graph, and the
    # graph model is a pure document layer (schema-only deps) - the edge
    # stays acyclic and the engine still never learns v1 existed.
    # The built-in compat pack also consumes native inference contracts and
    # dinkster_workers' invocation-local execution context so a same-session arm
    # can assert and execute the host's expected runtime identity. The torch
    # executor edge stays lazy so importing pack schemas remains torch-free.
    # These are body-wiring edges, not scheduler access; third-party packs
    # still author through dinkster_api below.
    "dinkster_compat_comfy": {
        "dinkster_image_document",
        "dinkster_schema",
        "dinkster_values",
        "dinkster_video",
        "dinkster_assets",
        "dinkster_memory",
        "dinkster_graph",
        "dinkster_inference",
        "dinkster_inference_torch",
        "dinkster_native",
        "dinkster_protocol",
        "dinkster_nodes_generation",
        "dinkster_workers",
    },
    # The door (DESIGN 3.6): re-exports the pack-author surface from the
    # bottom layers and the RPC-clean protocol leaf - never engine, workers,
    # caches, or server.
    "dinkster_api": {
        "dinkster_schema",
        "dinkster_values",
        "dinkster_video",
        "dinkster_protocol",
        "dinkster_inference",
        "dinkster_memory",
        "dinkster_assets",
    },
    # The install suite contains metadata only; component packs author
    # through the door exclusively and never import one another.
    "dinkster_nodes_std": set(),
    "dinkster_nodes_foundation": {"dinkster_api"},
    "dinkster_nodes_media_io": {"dinkster_api", "dinkster_image_document"},
    "dinkster_nodes_image": {"dinkster_api", "dinkster_image_document"},
    "dinkster_nodes_remote": {"dinkster_api", "dinkster_workers"},
    "dinkster_nodes_generation": {"dinkster_api"},
    "dinkster_nodes_generation_openai": {
        "dinkster_inference",
        "dinkster_nodes_generation",
        "dinkster_workers",
    },
    "dinkster_model_wan": {"dinkster_api", "dinkster_inference", "dinkster_inference_torch"},
    "dinkster_model_qwen_image": {"dinkster_api", "dinkster_inference", "dinkster_inference_torch"},
    "dinkster_model_triposplat": {"dinkster_api", "dinkster_inference", "dinkster_inference_torch"},
    "dinkster_model_ipadapter": {"dinkster_api", "dinkster_inference", "dinkster_inference_torch"},
    # Model-backed vision providers execute stable owner schemas through the
    # pack-author door and stay independent of the host scheduler.
    "dinkster_nodes_vision": {"dinkster_api", "dinkster_inference_torch"},
    # Dev scaffolding is a pack like any other: the same door, nothing
    # more. Separation from std is compositional (--dev), not structural.
    "dinkster_nodes_dev": {"dinkster_api"},
    # Partner/API providers keep their descriptor interpreter and transport
    # inside the independently movable pack, authored through the same door.
    "dinkster_nodes_partner": {"dinkster_api"},
    # Training nodes are thin adapters over a host-bound service protocol;
    # the session handle and its digest grammar arrive through the door.
    "dinkster_nodes_training": {"dinkster_api"},
    # The optional torch trainer implements the training service against the
    # native model runtime and the durable session store. The schema package
    # remains torch-free; only a selected training worker imports this package.
    "dinkster_training_torch": {
        "dinkster_api",
        "dinkster_inference",
        "dinkster_inference_torch",
        "dinkster_nodes_training",
        "dinkster_server",
    },
    # The isolated executor binds a training service to those schema adapters.
    # It owns the concrete durable store dependency and keeps it out of the
    # schema package loaded by the server process.
    "dinkster_training_worker": {
        "dinkster_api",
        "dinkster_nodes_training",
        "dinkster_server",
        "dinkster_training_torch",
        "dinkster_workers",
    },
    # The pure registry model (DESIGN M8): shares only the closed name
    # grammar with the rest of the stack. Doctor evidence arrives as the
    # report JSON, never a dinkster_workers import - the registry consumes
    # the machine interface, keeping it deployable without host machinery.
    "dinkster_registry": {"dinkster_schema"},
    # Layer 0, above even the server: the supervisor spawns engine hosts
    # and speaks HTTP to them, NEVER importing engine code - the layer
    # that will manage multiple installations (different code versions,
    # different venvs) cannot share a Python surface with any one of
    # them. Empty ON PURPOSE; any addition here is a design regression.
    "dinkster_supervisor": set(),
    # Collaboration sessions (platform plan): a separate additive surface
    # beside the job queue. Empty ON PURPOSE - it orders and relays
    # patches it never interprets, so it needs no schema, engine, or
    # server import; extractable to its own process without a wire
    # change. Any addition here braids collaboration into execution.
    "dinkster_collab": set(),
    # The top of the stack: the server serves what the layers below produce
    # but is imported by nothing (node packs included). Worker composition
    # lives in the src/dinkster app layer, so even dinkster_workers stays out. The
    # external token verifier is a lower-level authentication boundary with
    # no imports back into Dinkster.
    "dinkster_server": {
        "dinkster_schema",
        "dinkster_values",
        "dinkster_graph",
        "dinkster_protocol",
        "dinkster_engine",
        "dinkster_caches",
        "dinkster_memory",
        "dinkster_assets",
        "dinkster_image_document",
        "dinkster_token_verifier",
        "dinkster_p2p",
    },
}


def dinkster_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found.intersection(ALLOWED)


def test_every_package_is_governed() -> None:
    """A package absent from ALLOWED escapes the layering rule entirely -
    every packages/dinkster-* directory must have an explicit allowed set."""
    on_disk = {
        path.name.replace("-", "_")
        for path in (
            *(REPO_ROOT / "packages").iterdir(),
            EVIDENCE_ROOT / "packages/dinkster-acceptance",
        )
        if path.is_dir() and path.name.startswith("dinkster-")
    }
    assert on_disk == set(ALLOWED), (
        f"packages missing from ALLOWED: {sorted(on_disk - set(ALLOWED))}; "
        f"ALLOWED entries with no package: {sorted(set(ALLOWED) - on_disk)}"
    )


def test_one_way_dependencies() -> None:
    violations: list[str] = []
    checked = 0
    for package, allowed in ALLOWED.items():
        root = EVIDENCE_ROOT if package == "dinkster_acceptance" else REPO_ROOT
        src = root / "packages" / package.replace("_", "-") / "src" / package
        if not src.exists():
            assert not allowed, f"metadata-only package {package} declares code dependencies"
            continue
        assert src.is_dir(), f"missing package source: {src}"
        for module in src.rglob("*.py"):
            checked += 1
            for imported in dinkster_imports(module) - {package}:
                if imported not in allowed:
                    violations.append(f"{module.relative_to(root)} imports {imported}")
    assert checked > 0
    assert not violations, "one-way dependency rule violated:\n" + "\n".join(violations)


def test_bundled_video_preview_imports_only_the_pack_api() -> None:
    source = REPO_ROOT / "packages/dinkster-video/preview/src/dinkster_video_preview"
    modules = list(source.rglob("*.py"))
    assert modules
    for module in modules:
        assert dinkster_imports(module) <= {"dinkster_api", "dinkster_video_preview"}


def test_gguf_dependency_is_locked_only_behind_inference_extra() -> None:
    root_project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    inference_project = tomllib.loads(
        (REPO_ROOT / "packages/dinkster-inference/pyproject.toml").read_text()
    )["project"]
    assert root_project.get("optional-dependencies") is None
    assert all(not dependency.startswith("gguf") for dependency in root_project["dependencies"])
    assert all(
        not dependency.startswith("gguf") for dependency in inference_project["dependencies"]
    )
    assert inference_project["optional-dependencies"] == {"gguf": ["gguf==0.19.0"]}

    locked = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    packages = {package["name"]: package for package in locked["package"]}
    root_locked = packages["dinkster"]
    inference_locked = packages["dinkster-inference"]
    gguf_locked = packages["gguf"]
    assert "gguf" not in {dependency["name"] for dependency in root_locked["dependencies"]}
    assert "gguf" not in {dependency["name"] for dependency in inference_locked["dependencies"]}
    assert inference_locked["optional-dependencies"] == {"gguf": [{"name": "gguf"}]}
    assert gguf_locked["version"] == "0.19.0"
    assert gguf_locked["source"] == {"registry": "https://pypi.org/simple"}
    assert gguf_locked["wheels"] == [
        {
            "url": "https://files.pythonhosted.org/packages/b3/bb/d71d6da82763528c2c2ed6b59a9d6142c6595545a4c448e2085d155e88c2/gguf-0.19.0-py3-none-any.whl",
            "hash": "sha256:70bcd10edfe697fb2dad6e40af2234b9d8ece9a41a99761405121ebda1c3c1cd",
            "size": 118475,
            "upload-time": "2026-05-06T13:04:02.588Z",
        }
    ]


def test_torch_runtime_backends_are_constrained_behind_torch_extra() -> None:
    project = tomllib.loads(
        (REPO_ROOT / "packages/dinkster-inference-torch/pyproject.toml").read_text()
    )["project"]
    assert project["optional-dependencies"]["torch"] == [
        "torch>=2.5",
        "torchvision>=0.20",
        "packaging",
        "numpy>=1.26",
        "scipy>=1.11",
        "pillow>=10",
        "tqdm>=4.66",
        "dinkster-kitchen==0.2.35.post1",
        "dinkster-aimdo==0.5.5.post2",
        "sentencepiece==0.2.1",
        "tokenizers==0.23.1",
        "dinkster-kernels",
    ]

    locked = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    packages = {package["name"]: package for package in locked["package"]}
    torch_runtime = packages["dinkster-inference-torch"]
    assert torch_runtime["optional-dependencies"]["torch"] == [
        {"name": "dinkster-aimdo"},
        {"name": "dinkster-kernels"},
        {"name": "dinkster-kitchen"},
        {"name": "numpy"},
        {"name": "packaging"},
        {"name": "pillow"},
        {"name": "scipy"},
        {"name": "sentencepiece"},
        {"name": "tokenizers"},
        {"name": "torch"},
        {"name": "torchvision"},
        {"name": "tqdm"},
    ]
    assert packages["dinkster-kernels"]["source"] == {"editable": "packages/dinkster-kernels"}
    assert packages["dinkster-kitchen"]["version"] == "0.2.35.post1"
    assert packages["dinkster-aimdo"]["version"] == "0.5.5.post2"
    assert packages["sentencepiece"]["version"] == "0.2.1"
    assert packages["tokenizers"]["version"] == "0.23.1"
    assert packages["dinkster-kitchen"]["source"] == {"registry": "https://pypi.org/simple"}
    assert packages["dinkster-aimdo"]["source"] == {"registry": "https://pypi.org/simple"}
    assert packages["sentencepiece"]["source"] == {"registry": "https://pypi.org/simple"}
    assert packages["tokenizers"]["source"] == {"registry": "https://pypi.org/simple"}


def test_windows_ci_installs_published_engine_dependencies_without_torch() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/full-validation.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["test"]["steps"]
    (install_step,) = [
        step
        for step in steps
        if step.get("name") == "Install and verify published Windows engine dependencies"
    ]

    assert install_step["if"] == "matrix.os == 'windows'"
    assert install_step["shell"] == "pwsh"
    command = install_step["run"]
    assert "uv run --no-sync python" in command
    assert (
        "uv pip install --python $python --no-deps --index-url https://pypi.org/simple" in command
    )
    assert '"dinkster-kitchen==0.2.35.post1"' in command
    assert '"dinkster-aimdo==0.5.5.post2"' in command
    assert "find_spec('dinkster_kitchen') is not None" in command
    assert "find_spec('dinkster_aimdo') is not None" in command
