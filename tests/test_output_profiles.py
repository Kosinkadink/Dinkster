from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import dinkster_inference.output_profiles as output_profiles
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetIntegrityError,
    AssetRef,
    AssetVault,
    LibraryStore,
    digest_file,
    open_verified,
)
from dinkster_compat_comfy import native_arm
from dinkster_inference import FLOAT32, load_model_output_profile, load_safetensors_header
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_nodes_generation import GENERATION_NODES
from dinkster_schema import OutputInterface
from dinkster_server.auth import Principal, StaticBearerAuthenticator, install_auth
from dinkster_server.library import ServerLibrary, add_library_routes
from dinkster_values import Value, ValueMeta
from dinkster_values.model import PyObjPayload

from dinkster.native_policy import NativeDispatchPolicy
from tests.test_inference_assembly import (
    ltxav_checkpoint,
    sdxl_combined_geometries,
    wan21_split_sources,
)
from tests.test_inference_assembly import (
    source as geometry_source,
)


def _safetensors(path: Path, *, key: str = "weight", count: int = 1) -> Path:
    payload_size = count * 4
    header = json.dumps(
        {
            key: {
                "dtype": "F32",
                "shape": [count],
                "data_offsets": [0, payload_size],
            }
        },
        separators=(",", ":"),
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(payload_size))
    return path


def _source(path: Path) -> Any:
    digest = digest_file(path)
    return load_safetensors_header(path, asset_digest=digest, asset_size=path.stat().st_size)


def _component(
    path: Path, name: str = "diffusion", source_key: str = "weight"
) -> ComponentPlan[object]:
    return ComponentPlan(
        component=name,
        path=path,
        config=("test", name),
        keys={"weight": source_key},
        dtypes={"weight": FLOAT32},
        quant={},
    )


def _combined_plan(path: Path) -> object:
    components = (
        _component(path, "diffusion"),
        _component(path, "clip_l"),
        _component(path, "vae"),
    )
    return SimpleNamespace(
        family=SimpleNamespace(id="dinkster.test"),
        identity_components=components,
    )


class _ModelRegistry:
    def __init__(self, path: Path) -> None:
        self.plan = _component(path)
        self.descriptor = SimpleNamespace(
            family_for=lambda _plan: "dinkster.test",
        )

    def detect(self, _source: object, _path: Path) -> tuple[object, ...]:
        return (object(),)

    def select_detected(self, _matches: object, _kind: str) -> tuple[object, str, object]:
        return self.descriptor, "diffusion", self.plan


class _Resolver:
    def __init__(self, digest: str, path: Path) -> None:
        self.digest = digest
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == self.digest else None


def test_profile_probe_is_header_only_and_imports_no_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors")
    monkeypatch.setattr(output_profiles, "plan_native", lambda **_kwargs: _combined_plan(path))

    profile = load_model_output_profile(
        path,
        asset_digest=digest_file(path),
        asset_size=path.stat().st_size,
    )

    assert profile.kind == "checkpoint"
    assert [entry["id"] for entry in cast(list[dict[str, str]], profile.document["entries"])] == [
        "model",
        "clip",
        "vae",
    ]
    assert profile.document["detectorRevision"] == "1"
    assert cast(str, profile.document["shapeDigest"]).startswith("sha256:")
    assert "torch" not in sys.modules


def test_profile_uses_caller_owned_verified_descriptor_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors", count=1)
    replacement = _safetensors(tmp_path / "replacement.safetensors", count=2)
    digest = digest_file(path)
    monkeypatch.setattr(
        output_profiles,
        "plan_native",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("unknown")),
    )
    monkeypatch.setattr(
        output_profiles,
        "builtin_registries",
        lambda: SimpleNamespace(
            components=SimpleNamespace(
                detect=lambda *_args: (),
                select_detected=lambda *_args: (_ for _ in ()).throw(ValueError("unknown")),
            )
        ),
    )

    with open_verified(path, digest) as verified:
        expected = load_model_output_profile(
            path,
            asset_digest=digest,
            asset_size=path.stat().st_size,
            handle=verified,
        )
        original_open = Path.open
        monkeypatch.setattr(
            Path,
            "open",
            lambda candidate, *args, **kwargs: original_open(
                replacement if candidate == path else candidate, *args, **kwargs
            ),
        )
        actual = load_model_output_profile(
            path,
            asset_digest=digest,
            asset_size=os.fstat(verified.fileno()).st_size,
            handle=verified,
        )

    assert actual.document == expected.document
    assert actual.document["assetDigest"] == digest
    assert (
        actual.document["shapeDigest"]
        != load_model_output_profile(
            path,
            asset_digest=digest_file(replacement),
            asset_size=replacement.stat().st_size,
        ).document["shapeDigest"]
    )


def test_model_only_profile_uses_component_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "diffusion.safetensors")
    monkeypatch.setattr(
        output_profiles,
        "plan_native",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("not combined")),
    )
    monkeypatch.setattr(
        output_profiles,
        "builtin_registries",
        lambda: SimpleNamespace(components=_ModelRegistry(path)),
    )

    profile = output_profiles.probe_model_output_profile(_source(path))

    assert profile.kind == "model"
    assert profile.document["entries"] == [{"id": "model", "name": "MODEL", "type": "model"}]
    components = cast(dict[str, object], profile.document["components"])
    assert components["family"] == "dinkster.test"
    plans = cast(list[dict[str, object]], components["plans"])
    assert [plan["component"] for plan in plans] == ["diffusion"]
    assert cast(str, plans[0]["identity"]).startswith("sha256:")
    assert cast(str, plans[0]["mappingDigest"]).startswith("sha256:")
    assert profile.document["diagnostics"] == []


def test_unknown_header_defaults_to_model_with_diagnostic(tmp_path: Path) -> None:
    path = _safetensors(tmp_path / "unknown.safetensors")
    profile = output_profiles.probe_model_output_profile(_source(path))

    assert profile.kind == "model"
    assert profile.document["components"] == {"family": None, "plans": []}
    assert profile.document["diagnostics"]


@pytest.mark.parametrize(
    ("family", "expected_kind"),
    [("dinkster.wan21", "model"), ("dinkster.ltxav", "checkpoint")],
)
def test_standalone_profiles_normalize_real_component_plans(
    family: str, expected_kind: output_profiles.ModelProfileKind
) -> None:
    planned_source = (
        wan21_split_sources("t2v-14b")["diffusion"]
        if family == "dinkster.wan21"
        else ltxav_checkpoint()
    )
    source = SafetensorsSource(
        path=Path("standalone.safetensors"),
        entries={key: planned_source.entry(key) for key in planned_source.keys()},
        extra=planned_source.metadata(),
        asset_digest="blake3:" + "1" * 64,
        asset_size=0,
    )
    profile = output_profiles.probe_model_output_profile(source)
    assert profile.kind == expected_kind
    assert profile.document["diagnostics"] == []
    components = cast(dict[str, object], profile.document["components"])
    assert components["family"] == family
    assert components["plans"]


def test_real_planner_configuration_reads_keep_verified_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors", key="edm_vpred.sigma_max")
    replacement = _safetensors(tmp_path / "replacement.safetensors", key="edm_vpred.sigma_max")
    for target, value in ((path, 42.5), (replacement, 84.5)):
        with target.open("r+b") as handle:
            handle.seek(-4, os.SEEK_END)
            handle.write(struct.pack("<f", value))
    weights = geometry_source(sdxl_combined_geometries(), "model.safetensors")

    def header(handle: BinaryIO, **kwargs: Any) -> SafetensorsSource:
        parsed = load_safetensors_header_from_file(handle, **kwargs)
        # Full SDXL geometry needs no multi-gigabyte zero-weight payload fixture.
        return replace(
            parsed,
            entries={**{key: weights.entry(key) for key in weights.keys()}, **parsed.entries},
        )

    monkeypatch.setattr(output_profiles, "load_safetensors_header_from_file", header)
    digest = digest_file(path)
    with open_verified(path, digest) as verified:
        size = os.fstat(verified.fileno()).st_size
        expected = load_model_output_profile(
            path, asset_digest=digest, asset_size=size, handle=verified
        )
        assert expected.kind == "checkpoint"
        original_open = Path.open
        monkeypatch.setattr(
            Path,
            "open",
            lambda candidate, *args, **kwargs: original_open(
                replacement if candidate == path else candidate, *args, **kwargs
            ),
        )
        actual = load_model_output_profile(
            path, asset_digest=digest, asset_size=size, handle=verified, stored=expected.to_json()
        )
        assert actual.document == expected.document
        assert not verified.closed
        replaced = load_model_output_profile(
            path, asset_digest=digest_file(replacement), asset_size=size
        )
        assert replaced.document["components"] != expected.document["components"]
        assert replaced.document["shapeDigest"] != expected.document["shapeDigest"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda document: document["entries"].reverse(),
        lambda document: document["entries"][0].__setitem__("id", "changed"),
        lambda document: document["entries"][0].__setitem__("name", "Changed"),
        lambda document: document["entries"][0].__setitem__("type", "clip"),
        lambda document: document.__setitem__("assetDigest", "blake3:" + "0" * 64),
        lambda document: document.__setitem__("detectorRevision", "2"),
        lambda document: document.__setitem__("shapeDigest", "sha256:" + "0" * 64),
        lambda document: document.__setitem__("components", {}),
    ),
)
def test_profile_validation_rejects_tampered_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: Any
) -> None:
    path = _safetensors(tmp_path / "combined.safetensors")
    monkeypatch.setattr(output_profiles, "plan_native", lambda **_kwargs: _combined_plan(path))
    source = _source(path)
    profile = output_profiles.probe_model_output_profile(source)
    tampered = deepcopy(profile.document)
    mutation(tampered)

    with pytest.raises(ValueError, match="does not match"):
        output_profiles.load_model_output_profile(
            path,
            asset_digest=source.asset_digest,
            asset_size=cast(int, source.asset_size),
            stored=json.dumps(tampered),
        )


def test_profile_changes_with_asset_and_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _safetensors(tmp_path / "first.safetensors", count=1)
    second = _safetensors(tmp_path / "second.safetensors", count=2)
    monkeypatch.setattr(output_profiles, "plan_native", lambda **_kwargs: _combined_plan(first))

    first_profile = output_profiles.probe_model_output_profile(_source(first)).document
    second_profile = output_profiles.probe_model_output_profile(_source(second)).document

    assert first_profile["assetDigest"] != second_profile["assetDigest"]
    assert first_profile["shapeDigest"] != second_profile["shapeDigest"]


def test_profile_changes_when_canonical_source_mapping_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "mapped.safetensors")
    source = _source(path)
    monkeypatch.setattr(output_profiles, "plan_native", lambda **_kwargs: _combined_plan(path))
    first = output_profiles.probe_model_output_profile(source).document
    remapped = _combined_plan(path)
    cast(Any, remapped).identity_components = (
        _component(path, "diffusion", "alternate"),
        _component(path, "clip_l"),
        _component(path, "vae"),
    )
    monkeypatch.setattr(output_profiles, "plan_native", lambda **_kwargs: remapped)
    second = output_profiles.probe_model_output_profile(source).document

    first_plans = cast(dict[str, Any], first["components"])["plans"]
    second_plans = cast(dict[str, Any], second["components"])["plans"]
    assert first_plans[0]["identity"] == second_plans[0]["identity"]
    assert first_plans[0]["mappingDigest"] != second_plans[0]["mappingDigest"]
    assert first["shapeDigest"] != second["shapeDigest"]
    with pytest.raises(ValueError, match="does not match"):
        output_profiles.load_model_output_profile(
            path,
            asset_digest=cast(str, source.asset_digest),
            asset_size=cast(int, source.asset_size),
            stored=json.dumps(first),
        )


def test_schema_and_worker_registrations_preserve_semantic_ids() -> None:
    schema = next(
        node.schema()
        for node in GENERATION_NODES
        if node.schema().node_type == "dinkster.load_model_profile"
    )
    descriptors = schema.output_descriptors
    assert descriptors is not None
    entries = next(input for input in schema.inputs if input.id == "entries")
    assert entries.display_name == "Outputs"
    assert entries.hidden is False
    assert [(choice.id, choice.type.runtime_type_id()) for choice in descriptors.choices] == [
        ("model", "dinkster.model"),
        ("clip", "dinkster.clip"),
        ("vae", "dinkster.vae"),
    ]
    assert descriptors.fixed_ids is True
    assert descriptors.probe is not None and descriptors.probe.revision == "1"
    assert any(
        node.schema().node_type == "dinkster.load_model_profile"
        for node in native_arm.GENERATION_PROVIDER_NODES
    )
    assert any(
        node.schema().node_type == "dinkster.load_model_profile"
        for node in native_arm.NATIVE_ARM_NODES
    )


def test_worker_projects_same_checkpoint_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "checkpoint.safetensors")
    digest = digest_file(path)
    resolver = _Resolver(digest, path)
    checkpoint = AssetRef(digest, path.name, path.stat().st_size, resolver=resolver)
    handle = object()
    monkeypatch.setattr(
        output_profiles,
        "plan_native",
        lambda **_kwargs: _combined_plan(path),
    )
    profile = output_profiles.probe_model_output_profile(_source(path))
    monkeypatch.setattr(
        native_arm.NativeLoadCheckpoint,
        "execute",
        classmethod(lambda _cls, **_kwargs: {"model": handle, "clip": handle, "vae": handle}),
    )
    schema = native_arm.NativeLoadModelProfile.schema()
    assert schema.output_descriptors is not None

    result = native_arm.NativeLoadModelProfile.execute(
        checkpoint=checkpoint,
        entries=profile.to_json(),
        output_spec=OutputInterface((), schema.output_descriptors.choices),
    )

    assert result["model"] is result["clip"] is result["vae"] is handle


def _value(value: object, *, meta: dict[str, object] | None = None) -> Value:
    return Value(
        type_id="dinkster.asset" if meta else "core.string",
        fingerprint=repr(value),
        meta=ValueMeta(meta or {}),
        payload=PyObjPayload(value),
    )


def test_host_revalidates_file_before_reusing_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors")
    digest = digest_file(path)
    monkeypatch.setattr(
        output_profiles,
        "plan_native",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("unknown")),
    )
    monkeypatch.setattr(
        output_profiles,
        "builtin_registries",
        lambda: SimpleNamespace(
            components=SimpleNamespace(
                detect=lambda *_args: (),
                select_detected=lambda *_args: (_ for _ in ()).throw(ValueError("unknown")),
            )
        ),
    )
    profile = output_profiles.probe_model_output_profile(_source(path))
    policy = NativeDispatchPolicy(
        lambda candidate: path if candidate == digest else None,
        lambda _diagnostic: None,
    )
    inputs = {
        "checkpoint": _value(None, meta={"digest": digest, "name": path.name}),
        "entries": _value(profile.to_json()),
    }
    arms = ("compat", {"compat": "compat-tag", "compat@native": "native-tag"})

    first = asyncio.run(policy.select("dinkster.load_model_profile", inputs, arms))
    assert first is not None
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises(AssetIntegrityError, match="digest_mismatch"):
        asyncio.run(policy.select("dinkster.load_model_profile", inputs, arms))


def test_model_profile_endpoint_is_revisioned_and_not_cached(tmp_path: Path) -> None:
    async def scenario() -> None:
        source_path = _safetensors(tmp_path / "profile.safetensors")
        data = source_path.read_bytes()
        digest = digest_file(source_path)
        vault = AssetVault(tmp_path / "vault")
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        library = ServerLibrary(
            vault,
            LibraryStore(tmp_path / "library.sqlite"),
            model_output_profile=lambda _path, _handle, asset_digest, _size: {
                "entries": [{"id": "model", "name": "MODEL", "type": "model"}],
                "assetDigest": asset_digest,
                "detectorRevision": "1",
                "shapeDigest": "sha256:" + "1" * 64,
                "components": {},
                "diagnostics": [],
            },
        )
        app = web.Application()
        add_library_routes(app, library)
        install_auth(
            app,
            StaticBearerAuthenticator(
                {
                    "reader": Principal("reader", {"local": frozenset({"assets:read"})}),
                    "viewer": Principal("viewer", {"local": frozenset({"jobs:read"})}),
                }
            ),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            url = f"/api/output-profiles/model?digest={digest}&revision=1"
            assert (await client.get(url)).status == 401
            denied = await client.get(url, headers={"Authorization": "Bearer viewer"})
            assert denied.status == 403
            assert (await denied.json())["capability"] == "assets:read"
            client.session.headers["Authorization"] = "Bearer reader"
            malformed = await client.get("/api/output-profiles/model?digest=nope&revision=1")
            assert malformed.status == 400
            wrong_revision = await client.get(
                f"/api/output-profiles/model?digest={digest}&revision=2"
            )
            assert wrong_revision.status == 400
            response = await client.get(f"/api/output-profiles/model?digest={digest}&revision=1")
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-cache"
            assert (await response.json())["assetDigest"] == digest
        finally:
            await client.close()

    asyncio.run(scenario())
