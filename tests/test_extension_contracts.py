"""S0-A declarative extension and snapshot identity contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import dinkster_protocol as protocol
import pytest
from dinkster_protocol import (
    EXTENSION_SCOPES,
    ActiveExtension,
    ExtensionSnapshot,
    KeyedContribution,
    canonical_extension_snapshot,
    extension_behavior_hash,
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


@pytest.mark.parametrize("scope", EXTENSION_SCOPES)
def test_manifest_extension_entry_requires_matching_privilege(tmp_path: Path, scope: str) -> None:
    with pytest.raises(ManifestError, match=f"matching privileges: {scope}"):
        load_manifest(
            _manifest(
                tmp_path,
                f'[pack.extension]\n{scope} = "extension:{scope}"\n',
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
