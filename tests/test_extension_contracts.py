"""S0-A declarative extension and snapshot identity contracts."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import dinkster_protocol as protocol
import pytest
from dinkster_protocol import (
    EXTENSION_CAPABILITIES,
    EXTENSION_SCOPES,
    ActiveExtension,
    CompositionMode,
    ContributionSurfaceDescriptor,
    ExtensionDeclaration,
    ExtensionEntryPoints,
    ExtensionScope,
    ExtensionSnapshot,
    GraphCompilerRegistrySnapshot,
    KeyedContribution,
    canonical_compile_reply_bytes,
    canonical_extension_snapshot,
    extension_behavior_hash,
    generated_node_id,
)
from dinkster_workers import ManifestError, load_manifest
from dinkster_workers.host import load_extension_contributions


def _manifest(tmp_path: Path, extension: str = "") -> Path:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "example"\n[pack.entry]\nnodes = "example.nodes:NODES"\n' + extension,
        encoding="utf-8",
    )
    return path


def test_manifest_extension_entries_are_scope_separated_and_code_free(
    tmp_path: Path,
) -> None:
    module_name = "must_not_import_extension_pack"
    sys.modules.pop(module_name, None)
    manifest = load_manifest(
        _manifest(
            tmp_path,
            "[pack.extension]\n"
            f'schema = "{module_name}:schema"\n'
            f'server = "{module_name}:server"\n'
            f'inference = "{module_name}:inference"\n'
            f'frontend = "{module_name}:frontend"\n'
            f'training = "{module_name}:training"\n'
            'privileges = ["server", "schema", "inference", "frontend", "training"]\n'
            'capabilities = ["routes", "filesystem", "downloads", '
            '"background-jobs", "model-family-registration", "accelerator", "artifacts"]\n',
        )
    )

    assert module_name not in sys.modules
    assert manifest.extension.entries.schema == f"{module_name}:schema"
    assert manifest.extension.entries.server == f"{module_name}:server"
    assert manifest.extension.entries.inference == f"{module_name}:inference"
    assert manifest.extension.entries.frontend == f"{module_name}:frontend"
    assert manifest.extension.entries.training == f"{module_name}:training"
    assert manifest.extension.privileges == EXTENSION_SCOPES
    assert manifest.extension.capabilities == EXTENSION_CAPABILITIES


def test_manifest_extension_is_additive_and_legacy_entries_are_unchanged(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(_manifest(tmp_path))
    assert manifest.nodes_entry == "example.nodes:NODES"
    assert manifest.extension.entries.schema is None
    assert manifest.extension.privileges == ()
    assert manifest.extension.capabilities == ()


@pytest.mark.parametrize("scope", EXTENSION_SCOPES)
def test_manifest_extension_entry_requires_matching_privilege(tmp_path: Path, scope: str) -> None:
    with pytest.raises(ManifestError, match=f"matching privileges: {scope}"):
        load_manifest(
            _manifest(
                tmp_path,
                f'[pack.extension]\n{scope} = "extension:{scope}"\n',
            )
        )


def test_manifest_extension_unknown_vocabulary_is_loud_and_deterministic(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ManifestError,
        match=r"unknown capabilities: network, subprocess; known values: "
        r"accelerator, artifacts, background-jobs, downloads, filesystem, "
        r"model-family-registration, routes",
    ):
        load_manifest(
            _manifest(
                tmp_path,
                '[pack.extension]\ncapabilities = ["subprocess", "network"]\n',
            )
        )

    with pytest.raises(ManifestError, match="duplicate capabilities: routes"):
        load_manifest(
            _manifest(
                tmp_path,
                '[pack.extension]\ncapabilities = ["routes", "routes"]\n',
            )
        )

    with pytest.raises(ManifestError, match="unknown privileges: desktop"):
        load_manifest(
            _manifest(
                tmp_path,
                '[pack.extension]\nprivileges = ["desktop"]\n',
            )
        )


def test_pack_worker_never_resolves_inference_or_training_entries(tmp_path: Path) -> None:
    module_name = "must_not_import_worker_scoped_pack"
    sys.modules.pop(module_name, None)
    manifest = load_manifest(
        _manifest(
            tmp_path,
            "[pack.extension]\n"
            f'inference = "{module_name}:inference"\n'
            f'training = "{module_name}:training"\n'
            'privileges = ["inference", "training"]\n',
        )
    )
    assert load_extension_contributions(manifest) == ()
    assert module_name not in sys.modules


def test_manifest_extension_rejects_bad_entry_and_unknown_field(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="schema must be a 'module:attr'"):
        load_manifest(
            _manifest(
                tmp_path,
                '[pack.extension]\nschema = "missing_attr"\nprivileges = ["schema"]\n',
            )
        )
    with pytest.raises(ManifestError, match="unknown fields: scheam"):
        load_manifest(_manifest(tmp_path, '[pack.extension]\nscheam = "module:attr"\n'))


def test_composition_mode_contract_is_closed_and_immutable() -> None:
    assert tuple(mode.value for mode in CompositionMode) == (
        "ordered_list",
        "wrapper_chain",
        "exclusive",
        "observers",
        "keyed_registry",
    )
    assert tuple(scope.value for scope in ExtensionScope) == EXTENSION_SCOPES
    descriptor = ContributionSurfaceDescriptor("sampling.pre-cfg", CompositionMode.ORDERED_LIST)
    with pytest.raises(FrozenInstanceError):
        descriptor.surface_id = "sampling.post-cfg"  # type: ignore[misc]
    with pytest.raises(TypeError, match="mode must be a CompositionMode"):
        ContributionSurfaceDescriptor("sampling.driver", "exclusive")  # type: ignore[arg-type]


def test_direct_extension_declarations_enforce_manifest_invariants() -> None:
    entries = ExtensionEntryPoints(schema="example.extensions:schema")
    declaration = ExtensionDeclaration(entries=entries, privileges=("schema",))
    assert declaration.entries.for_scope(ExtensionScope.SCHEMA) == ("example.extensions:schema")

    with pytest.raises(ValueError, match="matching privileges: schema"):
        ExtensionDeclaration(entries=entries)
    with pytest.raises(ValueError, match="unknown privileges: desktop"):
        ExtensionDeclaration(privileges=("desktop",))
    with pytest.raises(ValueError, match="unknown capabilities: network"):
        ExtensionDeclaration(capabilities=("network",))
    with pytest.raises(ValueError, match="vocabulary order without duplicates"):
        ExtensionDeclaration(privileges=("server", "schema"))
    with pytest.raises(ValueError, match="module:attr"):
        ExtensionEntryPoints(server="example.extensions: server")


def _snapshot(
    *,
    mode: str = "strict",
    contributions: tuple[str, ...],
    selector_point: str = "double-block/0/attn",
    provider: str = "example.guidance/reducer",
) -> ExtensionSnapshot:
    return ExtensionSnapshot(
        (
            ActiveExtension(
                id="example.guidance",
                version="2.1.0",
                package_digest="sha256:" + "a" * 64,
                contribution_ids=contributions,
                selector_resolutions=(("attention:cross", (selector_point,)),),
                service_providers=(("guidance.reducer", provider),),
                capabilities=("background-jobs", "routes"),
                behavior_configuration=(
                    ("enabled", True),
                    ("limit", 3),
                    ("mode", mode),
                    ("optional", None),
                ),
            ),
        )
    )


def test_snapshot_serialization_and_hash_are_deterministic_and_order_sensitive() -> None:
    first = _snapshot(contributions=("guidance/pre-cfg", "guidance/post-cfg"))
    same = _snapshot(contributions=("guidance/pre-cfg", "guidance/post-cfg"))
    reordered = _snapshot(contributions=("guidance/post-cfg", "guidance/pre-cfg"))
    reconfigured = _snapshot(
        mode="fast",
        contributions=("guidance/pre-cfg", "guidance/post-cfg"),
    )
    reresolved = _snapshot(
        contributions=("guidance/pre-cfg", "guidance/post-cfg"),
        selector_point="single-block/0/attn",
    )
    provider_changed = _snapshot(
        contributions=("guidance/pre-cfg", "guidance/post-cfg"),
        provider="builtin/guidance-reducer",
    )

    assert canonical_extension_snapshot(first) == canonical_extension_snapshot(same)
    assert extension_behavior_hash(first) == extension_behavior_hash(same)
    assert extension_behavior_hash(first) != extension_behavior_hash(reordered)
    assert extension_behavior_hash(first) != extension_behavior_hash(reconfigured)
    assert extension_behavior_hash(first) != extension_behavior_hash(reresolved)
    assert extension_behavior_hash(first) != extension_behavior_hash(provider_changed)
    assert len(extension_behavior_hash(first)) == 64


def test_snapshot_refuses_noncanonical_or_non_rpc_clean_state() -> None:
    with pytest.raises(ValueError, match="capabilities must be sorted and unique"):
        ActiveExtension(
            id="example.guidance",
            version="1.0.0",
            package_digest="sha256:" + "b" * 64,
            capabilities=("routes", "downloads"),
        )
    with pytest.raises(ValueError, match="unknown capabilities: network"):
        ActiveExtension(
            id="example.guidance",
            version="1.0.0",
            package_digest="sha256:" + "b" * 64,
            capabilities=("network",),
        )
    with pytest.raises(TypeError, match="contribution_ids must be a tuple"):
        ActiveExtension(
            id="example.guidance",
            version="1.0.0",
            package_digest="sha256:" + "b" * 64,
            contribution_ids=["guidance/reducer"],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="configuration keys must be sorted and unique"):
        ActiveExtension(
            id="example.guidance",
            version="1.0.0",
            package_digest="sha256:" + "b" * 64,
            behavior_configuration=(("z", True), ("a", False)),
        )
    with pytest.raises(TypeError, match="values must be str, int, bool, or None"):
        ActiveExtension(
            id="example.guidance",
            version="1.0.0",
            package_digest="sha256:" + "b" * 64,
            behavior_configuration=(("bad", 1.5),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="selector resolution points must be sorted"):
        ActiveExtension(
            id="example.guidance",
            version="1.0.0",
            package_digest="sha256:" + "b" * 64,
            selector_resolutions=(("attention:cross", ("z", "a")),),
        )
    with pytest.raises(ValueError, match="service_providers must be sorted by unique key"):
        ActiveExtension(
            id="example.guidance",
            version="1.0.0",
            package_digest="sha256:" + "b" * 64,
            service_providers=(
                ("guidance.reducer", "provider-a"),
                ("guidance.reducer", "provider-b"),
            ),
        )
    with pytest.raises(ValueError, match="sorted by id with unique ids"):
        ExtensionSnapshot(
            (
                ActiveExtension(id="z", version="1", package_digest="sha256:z"),
                ActiveExtension(id="a", version="1", package_digest="sha256:a"),
            )
        )


def _compiler(
    id: str,
    order: object,
    *,
    surface_id: str = protocol.GRAPH_COMPILERS_SURFACE,
    aliases: tuple[str, ...] = (),
    extra_metadata: tuple[tuple[str, object], ...] = (),
) -> KeyedContribution:
    metadata = tuple(sorted((("contractVersion", 1), *extra_metadata, ("order", order))))
    return KeyedContribution(
        surface_id=surface_id,
        id=id,
        aliases=aliases,
        behavior_metadata=metadata,  # type: ignore[arg-type]
    )


def test_graph_compile_constants_and_errors_are_exact() -> None:
    assert protocol.GRAPH_COMPILERS_SURFACE == "inference.graph-compilers"
    assert (
        protocol.GRAPH_COMPILE_REQUEST_TYPE,
        protocol.GRAPH_COMPILE_RESULT_TYPE,
        protocol.GRAPH_COMPILE_CANCEL_TYPE,
    ) == ("compileGraph", "compileGraphResult", "cancelCompile")
    assert (
        protocol.GRAPH_COMPILE_MAX_NODES,
        protocol.GRAPH_COMPILE_MAX_LINKS,
        protocol.GRAPH_COMPILE_MAX_PASSES,
        protocol.GRAPH_COMPILE_MAX_DEPTH,
        protocol.GRAPH_COMPILE_MAX_GENERATED_PER_PASS,
        protocol.GRAPH_COMPILE_MAX_REPLY_BYTES,
        protocol.GRAPH_COMPILE_TIMEOUT_SECONDS,
        protocol.GENERATED_NODE_ID_PREFIX,
    ) == (4096, 16384, 32, 16, 1024, 8 * 1024 * 1024, 30.0, "$gen-")
    assert {
        protocol.GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
        protocol.GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION,
        protocol.GRAPH_COMPILE_ERROR_TIMEOUT,
        protocol.GRAPH_COMPILE_ERROR_REPLY_OVERSIZE,
        protocol.GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
        protocol.GRAPH_COMPILE_ERROR_NODE_LIMIT,
        protocol.GRAPH_COMPILE_ERROR_LINK_LIMIT,
        protocol.GRAPH_COMPILE_ERROR_PASS_LIMIT,
        protocol.GRAPH_COMPILE_ERROR_DEPTH_LIMIT,
        protocol.GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
        protocol.GRAPH_COMPILE_ERROR_TARGET_MISMATCH,
        protocol.GRAPH_COMPILE_ERROR_SELECTOR_INPUT,
        protocol.GRAPH_COMPILE_ERROR_SELECTOR_EMITTED,
        protocol.GRAPH_COMPILE_ERROR_ID_COLLISION,
        protocol.GRAPH_COMPILE_ERROR_ID_FORMAT,
        protocol.GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
        protocol.GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
    } == {
        "compile-compiler-failure",
        "compile-unknown-generation",
        "compile-timeout",
        "compile-reply-oversize",
        "compile-malformed-reply",
        "compile-node-limit",
        "compile-link-limit",
        "compile-pass-limit",
        "compile-depth-limit",
        "compile-generated-limit",
        "compile-target-mismatch",
        "compile-selector-input",
        "compile-selector-emitted",
        "compile-id-collision",
        "compile-id-format",
        "compile-origin-coverage",
        "compile-generation-mismatch",
    }


def test_graph_compiler_registry_is_frozen_rpc_clean_and_canonical() -> None:
    snapshot = GraphCompilerRegistrySnapshot(
        (
            _compiler("example.first", -1),
            _compiler("example.second", 0, extra_metadata=(("config.mode", "strict"),)),
        )
    )
    assert snapshot.contributions[1].id == "example.second"
    with pytest.raises(FrozenInstanceError):
        snapshot.contributions = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="tuple of KeyedContribution"):
        GraphCompilerRegistrySnapshot([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple of KeyedContribution"):
        GraphCompilerRegistrySnapshot(("example.compiler",))  # type: ignore[arg-type]


@pytest.mark.parametrize("id", ["compiler", ".compiler", "compiler."])
def test_graph_compiler_registry_requires_namespace_qualified_ids(id: str) -> None:
    with pytest.raises(ValueError, match="namespace-qualified"):
        GraphCompilerRegistrySnapshot((_compiler(id, 0),))


def test_graph_compiler_registry_refuses_surface_alias_duplicate_and_order_errors() -> None:
    with pytest.raises(ValueError, match="unknown graph compiler surface"):
        GraphCompilerRegistrySnapshot((_compiler("example.compiler", 0, surface_id="other"),))
    with pytest.raises(ValueError, match="globally unique"):
        GraphCompilerRegistrySnapshot(
            (_compiler("example.compiler", 0), _compiler("example.compiler", 1))
        )
    with pytest.raises(ValueError, match="aliases must be empty"):
        GraphCompilerRegistrySnapshot((_compiler("example.compiler", 0, aliases=("compiler",)),))
    with pytest.raises(ValueError, match="canonical order"):
        GraphCompilerRegistrySnapshot(
            (_compiler("example.second", 0), _compiler("example.first", 0))
        )
    with pytest.raises(ValueError, match="canonical order"):
        GraphCompilerRegistrySnapshot(
            (_compiler("example.first", 1), _compiler("example.second", 0))
        )


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ((("order", 0),), "metadata is malformed"),
        ((("contractVersion", 1),), "metadata is malformed"),
        ((("contractVersion", 1), ("order", 0), ("unexpected", True)), "metadata is malformed"),
        ((("contractVersion", 2), ("order", 0)), "contractVersion must be 1"),
        ((("contractVersion", True), ("order", 0)), "contractVersion must be 1"),
        ((("contractVersion", 1), ("order", True)), "signed 32-bit integer"),
        ((("contractVersion", 1), ("order", -(2**31) - 1)), "signed 32-bit integer"),
        ((("contractVersion", 1), ("order", 2**31)), "signed 32-bit integer"),
    ],
)
def test_graph_compiler_registry_refuses_malformed_metadata(
    metadata: tuple[tuple[str, object], ...], match: str
) -> None:
    contribution = KeyedContribution(
        surface_id=protocol.GRAPH_COMPILERS_SURFACE,
        id="example.compiler",
        behavior_metadata=metadata,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match=match):
        GraphCompilerRegistrySnapshot((contribution,))


def test_generated_node_id_is_fixed_order_sensitive_and_grammar_safe() -> None:
    generated = generated_node_id(
        "example.compiler",
        ["node-a", "node-b"],
        "branch-1",
    )
    assert generated == "$gen-e0dac1ff3d38b5d5"
    assert generated_node_id("example.compiler", ("node-a", "node-b"), "branch-1") == generated
    assert generated_node_id("example.compiler", ("node-b", "node-a"), "branch-1") != generated
    assert generated_node_id("example.caf\u00e9", ("n\u00f6de",), "\u03ba") == (
        "$gen-01850836bc49f2a4"
    )
    assert len(generated) == len("$gen-") + 16
    assert generated.startswith("$gen-")
    assert all(char in "0123456789abcdef" for char in generated[len("$gen-") :])


@pytest.mark.parametrize(
    ("compiler_id", "sources", "local_key", "error"),
    [
        (1, ("node-a",), "key", ValueError),
        ("compiler", ("node-a",), "key", ValueError),
        (".compiler", ("node-a",), "key", ValueError),
        ("compiler.", ("node-a",), "key", ValueError),
        ("example.compiler", ("node-a",), "", ValueError),
        ("example.compiler", ("node-a",), 1, ValueError),
        ("example.compiler", "node-a", "key", TypeError),
        ("example.compiler", (), "key", ValueError),
        ("example.compiler", ("",), "key", ValueError),
        ("example.compiler", (1,), "key", ValueError),
    ],
)
def test_generated_node_id_refuses_invalid_inputs(
    compiler_id: object,
    sources: object,
    local_key: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        generated_node_id(compiler_id, sources, local_key)  # type: ignore[arg-type]


def test_canonical_compile_reply_bytes_pins_worker_reply_shape() -> None:
    reply: dict[str, object] = {
        "attemptedGeneratedCounts": [0, 3],
        "generationKey": "sha256:" + "a" * 64,
        "graph": {
            "nodes": {"node-a": {"type": "example.Input", "inputs": {}}},
            "links": [],
        },
        "targets": ["node-a", "node-b"],
        "passCount": 2,
        "origins": [
            {
                "nodeId": "$gen-0123456789abcdef",
                "compilerId": "example.compiler",
                "passIndex": 1,
                "sources": ["node-a", "node-b"],
                "localKey": "branch-1",
            }
        ],
    }
    assert canonical_compile_reply_bytes(reply) == (
        b'{"attemptedGeneratedCounts":[0,3],'
        b'"generationKey":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"graph":{"links":[],"nodes":{"node-a":{"inputs":{},"type":"example.Input"}}},'
        b'"origins":[{"compilerId":"example.compiler","localKey":"branch-1",'
        b'"nodeId":"$gen-0123456789abcdef","passIndex":1,'
        b'"sources":["node-a","node-b"]}],"passCount":2,'
        b'"targets":["node-a","node-b"]}'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_compile_reply_bytes_refuses_nonfinite_json(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_compile_reply_bytes({"graph": {"value": value}})


def test_pack_route_event_manifest_and_canonical_identity(tmp_path: Path) -> None:
    import json
    from dataclasses import replace

    from dinkster_api.v1 import JsonField, JsonObjectSchema, PackEvent, PackRoute
    from dinkster_protocol.pack_surfaces import pack_surfaces_from_wire

    route = PackRoute(
        "echo",
        "POST",
        "not_imported:echo",
        JsonObjectSchema((JsonField("text", "string"),)),
        JsonObjectSchema((JsonField("text", "string"),)),
    )
    event = PackEvent("example.updated", JsonObjectSchema((JsonField("count", "integer"),)))
    manifest = load_manifest(
        _manifest(
            tmp_path,
            """
[pack.extension]
privileges = ["schema", "server"]
capabilities = ["routes"]
[[pack.extension.routes]]
id = "echo"
method = "POST"
handler = "not_imported:echo"
request = { text = "string" }
response = { text = "string" }
[[pack.extension.events]]
name = "example.updated"
payload = { count = "integer" }
""",
        )
    )
    assert "not_imported" not in sys.modules
    assert manifest.extension.routes == (route,)
    assert manifest.extension.events == (event,)
    empty = ActiveExtension("example", "1.0.0", "digest")
    empty_snapshot = ExtensionSnapshot((empty,))
    empty_wire = json.loads(canonical_extension_snapshot(empty_snapshot))["extensions"][0]
    assert not {"routes", "events", "frontend"} & empty_wire.keys()
    active = replace(empty, routes=(route,), events=(event,))
    snapshot = ExtensionSnapshot((active,))
    assert extension_behavior_hash(snapshot) != extension_behavior_hash(empty_snapshot)
    wire = json.loads(canonical_extension_snapshot(snapshot))["extensions"][0]
    assert pack_surfaces_from_wire(wire) == ((route,), (event,))
    assert wire["events"][0]["schemaVersion"] == 1
    assert wire["events"][0]["scope"] == "execution"
    with pytest.raises(ValueError, match="server privilege"):
        ExtensionDeclaration(routes=(route,))
    with pytest.raises(ValueError, match="schema privilege"):
        ExtensionDeclaration(events=(event,))


@pytest.mark.parametrize(
    "payload", [{}, {"count": True}, {"count": 1.0}, {"count": 1, "other": 2}, {"count": None}]
)
def test_typed_pack_event_rejects_malformed_payload(payload: object) -> None:
    from dinkster_api.v1 import JsonField, JsonObjectSchema

    with pytest.raises(ValueError):
        JsonObjectSchema((JsonField("count", "integer"),)).validate(payload)


@pytest.mark.parametrize(
    "route_id", ["../health", "a/b", "a?x=1", "a%2fb", "{path}", "", "a#fragment"]
)
def test_pack_route_ids_cannot_escape_host_namespace(route_id: str) -> None:
    from dinkster_api.v1 import PackRoute

    with pytest.raises(ValueError, match="route id"):
        PackRoute(route_id, "GET", "pack:handler")


@pytest.mark.parametrize(
    "module",
    [
        "../outside.js",
        "./../outside.js",
        "https://example.com/x.js",
        "/absolute.js",
        "./a//x.js",
        "./a\\x.js",
        "./a/%2e%2e/x.js",
    ],
)
def test_frontend_module_paths_are_pack_relative(module: str) -> None:
    from dinkster_api.v1 import FrontendModule

    with pytest.raises(ValueError, match="relative.js"):
        FrontendModule("example.frontend", module, (), ())
