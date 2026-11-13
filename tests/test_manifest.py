from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import (
    ManifestError,
    load_manifest,
)
from dinkster_workers.doctor import prepare_catalog
from dinkster_workers.host import load_pack
from dinkster_workers.manifest import (
    validate_registration_claims,
)


def test_in_process_wheel_keeps_standard_library_imports_unshadowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    package = tmp_path / "wheel_import_boundary"
    package.mkdir()
    (package / "__init__.py").write_text("NODES = []\n")
    (package / "types.py").write_text("raise RuntimeError('pack types shadows stdlib')\n")
    manifest = package / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "wheel-import-boundary"\n'
        '[pack.entry]\nnodes = "wheel_import_boundary:NODES"\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        load_pack(load_manifest(manifest), import_from_pack_root=True)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from types import MappingProxyType; "
                "from wheel_import_boundary import NODES; assert NODES == []",
            ],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        sys.modules.pop("wheel_import_boundary", None)


def write_manifest(path: Path, handler: object = None) -> Path:
    handler_line = "" if handler is None else f"workgroup_handler = {handler}\n"
    path.write_text(
        '[pack]\nname = "test"\n[pack.entry]\nnodes = "test_pack:NODES"\n' + handler_line
    )
    return path


def test_manifest_ignores_unrecognized_pack_metadata(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "dinkster-pack.toml")
    expected = load_manifest(path)
    with path.open("a") as output:
        output.write('\n[pack.custom_metadata]\nimports = ["example"]\nrevision = "1"\n')
    assert load_manifest(path) == expected


def test_development_pack_declares_its_version_and_engine_range() -> None:
    from dinkster import __version__

    manifest = load_manifest(Path("packages/dinkster-nodes-dev/dinkster-pack.toml"))
    assert manifest.compatibility is not None, manifest.name
    manifest.compatibility.require_engine(manifest.name, __version__)


@pytest.mark.parametrize(
    ("engine", "accepted"),
    [
        ("0.3.9", False),
        ("0.4.0", True),
        ("0.6.3+g123", True),
        ("0.6.4rc1", True),
        ("0.7.0rc1", False),
        ("0.7.0", False),
    ],
)
def test_pack_declares_independent_version_and_engine_compatibility(
    tmp_path: Path, engine: str, accepted: bool
) -> None:
    path = write_manifest(tmp_path / "dinkster-pack.toml")
    path.write_text(
        path.read_text().replace("[pack]\n", '[pack]\nversion = "2.3.1"\nengine = ">=0.4,<0.7"\n')
    )
    manifest = load_manifest(path)
    assert manifest.compatibility is not None
    assert manifest.compatibility.version == "2.3.1"
    if accepted:
        manifest.compatibility.require_engine(manifest.name, engine)
    else:
        with pytest.raises(ValueError) as refusal:
            manifest.compatibility.require_engine(manifest.name, engine)
        message = str(refusal.value)
        assert "test@2.3.1" in message
        assert ">=0.4,<0.7" in message
        assert engine in message


@pytest.mark.parametrize("field", ["namespaces", "type-namespaces"])
def test_registration_requires_pack_prefix_for_non_dinkster_publishers(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "mypack"\npublisher = "someone"\n'
        f'{field} = ["foreign"]\n[pack.entry]\nnodes = "mypack:NODES"\n'
    )
    with pytest.raises(ManifestError, match="must be prefixed by pack name"):
        validate_registration_claims(load_manifest(path))
    path.write_text(path.read_text().replace('"foreign"', '"mypack.types"'))
    validate_registration_claims(load_manifest(path))
    path.write_text(
        path.read_text().replace('"someone"', '"dinkster"').replace('"mypack.types"', '"foreign"')
    )
    validate_registration_claims(load_manifest(path))


@pytest.mark.parametrize(
    ("pack_name", "reserved"),
    [
        ("dinkster", True),
        ("std", True),
        ("comfy", True),
        ("core", True),
        ("dev", False),
        ("std-extra", True),
    ],
)
def test_pack_prefix_reservation_depends_on_publisher(
    tmp_path: Path, pack_name: str, reserved: bool
) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        f'[pack]\nname = "{pack_name}"\npublisher = "someone"\n'
        '[pack.entry]\nnodes = "mypack:NODES"\n'
    )
    if reserved:
        with pytest.raises(ManifestError, match="only publisher 'dinkster'.*reserved"):
            validate_registration_claims(load_manifest(path))
    else:
        validate_registration_claims(load_manifest(path))
    path.write_text(path.read_text().replace('"someone"', '"dinkster"'))
    validate_registration_claims(load_manifest(path))


@pytest.mark.parametrize("with_codec", [False, True])
def test_pack_registration_refuses_duplicate_type_owner(tmp_path: Path, with_codec: bool) -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    for name in ("first-owner", "second-owner"):
        root = tmp_path / name
        root.mkdir()
        module = name.replace("-", "_") + f"_{int(with_codec)}"
        (root / f"{module}.py").write_text(
            'NODES = []\ndef register_types(registry):\n    registry.register("SHARED")\n'
        )
        path = root / "dinkster-pack.toml"
        path.write_text(
            f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
            'type-namespaces = ["SHARED"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\n'
            + (f'types = "{module}:register_types"\n' if with_codec else "")
        )
        if name == "first-owner":
            load_pack(load_manifest(path), registry=registry, import_from_pack_root=True)
            assert registry.owner_of("SHARED") == "first-owner"
        else:
            with pytest.raises(ValueError, match="overlaps"):
                load_pack(load_manifest(path), registry=registry, import_from_pack_root=True)
    assert registry.owner_of("SHARED") == "first-owner"


def test_pack_registration_rejects_bare_claim_before_importing_entries(tmp_path: Path) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "mypack"\npublisher = "someone"\ntype-namespaces = ["THING"]\n'
        '[pack.entry]\nnodes = "missing_module:NODES"\n'
    )
    with pytest.raises(ManifestError, match="only publisher 'dinkster'"):
        load_pack(load_manifest(path))


def test_pack_registration_rejects_bare_node_under_a_prefixed_claim(tmp_path: Path) -> None:
    (tmp_path / "bare_node_pack.py").write_text(
        "from dinkster_api.v1 import Node, NodeSchema\n"
        "class BareNode(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='bare_node')\n"
        "NODES = [BareNode]\n"
    )
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "mypack"\npublisher = "someone"\n'
        '[pack.entry]\nnodes = "bare_node_pack:NODES"\n'
    )
    with pytest.raises(ManifestError, match="node.*bare_node.*pack-name prefix"):
        load_pack(load_manifest(path), import_from_pack_root=True)
    assert not prepare_catalog(path).ok


def test_pack_frontend_assets_and_settings_are_validated_and_immutable(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "badge.png").write_bytes(b"\x89PNG\r\nfixture")
    (frontend / "theme.css").write_bytes(b".pack { color: #123; }\n")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "enabled": {"type": "boolean", "title": "Enabled", "default": True},
            "strength": {
                "type": "number",
                "title": "Strength",
                "description": "Rendering strength",
                "default": 0.5,
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["enabled", "strength"],
    }
    (tmp_path / "settings.schema.json").write_text(json.dumps(schema), encoding="utf-8")
    path = write_manifest(tmp_path / "dinkster-pack.toml")
    path.write_text(
        '[pack]\nname = "test"\n[pack.entry]\nnodes = "test_pack:NODES"\n'
        '[pack.frontend]\nassets = "./frontend/"\n'
        '[pack.settings]\nschema = "./settings.schema.json"\n',
        encoding="utf-8",
    )

    manifest = load_manifest(path)

    assert [(asset.path, asset.media_type) for asset in manifest.frontend_assets] == [
        ("badge.png", "image/png"),
        ("theme.css", "text/css"),
    ]
    assert manifest.settings_schema is not None
    assert manifest.settings_schema.defaults == {"enabled": True, "strength": 0.5}
    (frontend / "theme.css").write_bytes(b"changed")
    assert manifest.frontend_assets[1].data == b".pack { color: #123; }\n"


@pytest.mark.parametrize(
    ("declaration", "message"),
    (
        ('[pack.frontend]\nassets = "./missing/"\n', "is not a directory"),
        ('[pack.frontend]\nassets = "../outside/"\n', "must not escape"),
        ('[pack.settings]\nschema = "./missing.json"\n', "is not a file"),
    ),
)
def test_pack_frontend_and_settings_reject_missing_or_escaping_paths(
    tmp_path: Path, declaration: str, message: str
) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        f'[pack]\nname = "test"\n{declaration}[pack.entry]\nnodes = "test_pack:NODES"\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match=message):
        load_manifest(path)


@pytest.mark.parametrize(
    ("target_inside_assets", "message"),
    (
        (True, "must not contain symlinks"),
        (False, "must not escape the asset directory"),
    ),
    ids=("in-root", "escape"),
)
def test_pack_frontend_assets_reject_symlinks(
    tmp_path: Path, target_inside_assets: bool, message: str
) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    target = (frontend if target_inside_assets else tmp_path) / "target.css"
    target.write_text("body {}", encoding="utf-8")
    try:
        (frontend / "theme.css").symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "test"\n[pack.entry]\nnodes = "test_pack:NODES"\n'
        '[pack.frontend]\nassets = "frontend"\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match=message):
        load_manifest(path)


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junction behavior is Windows-specific")
def test_pack_frontend_assets_reject_junction_escape(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "theme.css").write_text("body {}", encoding="utf-8")
    junction = frontend / "linked"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"junctions unavailable: {completed.stderr.strip()}")
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "test"\n[pack.entry]\nnodes = "test_pack:NODES"\n'
        '[pack.frontend]\nassets = "frontend"\n',
        encoding="utf-8",
    )

    try:
        with pytest.raises(ManifestError, match="must not escape the asset directory"):
            load_manifest(path)
    finally:
        junction.rmdir()
