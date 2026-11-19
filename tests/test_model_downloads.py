from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Generator
from contextlib import contextmanager
from http.client import HTTPMessage
from pathlib import Path
from typing import cast

import pytest
from dinkster_assets import AssetWriter, MountSnapshotWriter, SaveTarget, digest_bytes
from dinkster_assets.fetch import HTTPSResponse
from dinkster_assets.huggingface import select_huggingface_file

from tests.platform_support import symlink_or_skip
from tests.test_asset_writer import one_mount_snapshot


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("authenticated", [False, True])
@pytest.mark.parametrize("hidden_first", [False, True])
def test_individual_model_hiding_is_project_scoped_and_revision_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    indexed: bool,
    authenticated: bool,
    hidden_first: bool,
) -> None:
    from aiohttp.test_utils import TestClient, TestServer
    from dinkster_assets import AssetRef, AssetVault, LibraryStore, LocalAssetRef, MountTable
    from dinkster_assets.model import ModelHidden
    from dinkster_assets.provenance import ProvenanceStore
    from dinkster_graph import Graph, GraphNode
    from dinkster_schema import AssetWidget, InputSpec, NodeSchema, TypeExpr
    from dinkster_server import ServerLibrary, create_app
    from dinkster_server.auth import Principal, StaticBearerAuthenticator
    from dinkster_server.preflight import asset_preflight, bind_model_names
    from dinkster_values.asset_folders import AssetFolderContext, use_asset_folders

    from dinkster.model_downloads import ModelDownloads, shared_model_mounts
    from dinkster.model_downloads_api import add_model_download_routes
    from dinkster.mounts_api import MountService, add_mount_routes
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import SCHEMAS, make_engine

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    root = tmp_path / "models"
    root.mkdir()
    path = root / "selected.safetensors"
    payload = b"one model hidden only in first project"
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "duplicate.safetensors").write_bytes(payload)
    path.write_bytes(payload)
    (root / "keep.safetensors").write_bytes(b"different model stays visible")
    (tmp_path / "model-settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "downloadLocation": None,
                "locations": [
                    {
                        "id": "earlier",
                        "path": str(earlier),
                        "kind": "model/checkpoint",
                        "mode": "read",
                        "visibility": "default-on",
                    },
                    {
                        "id": "studio",
                        "path": str(root),
                        "kind": "model/checkpoint",
                        "mode": "read",
                        "visibility": "default-on",
                    },
                ],
            }
        )
    )

    if hidden_first:
        settings_path = tmp_path / "model-settings.json"
        settings = json.loads(settings_path.read_text())
        settings["locations"].reverse()
        settings_path.write_text(json.dumps(settings))

    def start() -> MountService:
        table = MountTable(tmp_path / "snapshot.json", index_root=tmp_path / "indexes")
        for mount in shared_model_mounts():
            table.add(mount, source="shared-models")
            if indexed or mount.id == "earlier-checkpoint":
                table.scan(mount.id)
        return MountService(table, tmp_path / "mounts.toml", allow_changes=True, shared_models=True)

    service = start()
    table = service.table
    ref = table.bind_name(path.name, "model/checkpoint")
    assert isinstance(ref, AssetRef if indexed else LocalAssetRef)
    assert ref is not None
    original = ref.to_wire()
    downloads = ModelDownloads(service, ProvenanceStore(tmp_path / "provenance.json"))
    store = LibraryStore(tmp_path / "library.sqlite")
    library = ServerLibrary(
        AssetVault(tmp_path / "vault"),
        store,
        resolver=table,
        local_resolver=table,
        hidden_model_name=downloads.hidden_name,
        hidden_model=downloads.hidden_digest,
        model_sources_for_name=lambda *_: ["https://example.org/must-not-download"],
    )
    schemas = {
        "loader": NodeSchema(
            "loader",
            inputs=(
                InputSpec(
                    "file",
                    TypeExpr.concrete("ASSET"),
                    widget=AssetWidget(kind="model/checkpoint"),
                ),
            ),
        )
    }

    async def scenario() -> None:
        auth = (
            StaticBearerAuthenticator(
                {
                    "operator": Principal(
                        "owner", {"local": frozenset({"assets:read", "settings:write"})}
                    ),
                    "viewer": Principal("viewer", {"local": frozenset({"assets:read"})}),
                }
            )
            if authenticated
            else None
        )
        app = create_app(make_engine, SCHEMAS, authenticator=auth)
        add_mount_routes(app, service)
        add_model_download_routes(app, downloads)
        client = TestClient(
            TestServer(app), headers={"Authorization": "Bearer operator"} if authenticated else None
        )
        await client.start_server()
        try:
            if authenticated:
                denied = await client.put(
                    "/api/models/hidden/first",
                    json={
                        "asset": original,
                        "kind": "model/checkpoint",
                    },
                    headers={"Authorization": "Bearer viewer"},
                )
                assert denied.status == 403
                denied = await client.delete(
                    "/api/models/hidden/first",
                    json={"identity": "unused"},
                    headers={"Authorization": "Bearer viewer"},
                )
                assert denied.status == 403
            result = await client.put(
                "/api/models/hidden/first",
                json={
                    "asset": original,
                    "kind": "model/checkpoint",
                },
            )
            assert result.status == 200, await result.text()
            preferences = (await result.json())["hiddenModels"]
            identity = preferences[0]["identities"][0]
            assert str(root) not in json.dumps(preferences)
            context = service.project_context("first")
            assert AssetFolderContext.from_wire(context.to_wire()) == context
            with use_asset_folders(context):
                assert table.bind_name(path.name, "model/checkpoint") is None
                assert table.bind_name("keep.safetensors", "model/checkpoint") is not None
                with pytest.raises(ModelHidden):
                    ref.read_bytes()
                for value in (path.name, original):
                    _, missing = bind_model_names(
                        Graph({"load": GraphNode("loader", {"file": value})}), schemas, library
                    )
                    assert missing[0]["hidden"] is True
                    assert missing[0]["sources"] == []
                if indexed:
                    missing = asset_preflight(
                        library,
                        {digest_bytes(payload): path.name},
                        {digest_bytes(payload)},
                        {},
                        project="first",
                    )
                    assert missing[0]["hidden"] is True
                    assert missing[0]["fetchable"] is False
            with use_asset_folders(service.project_context("second")):
                assert ref.read_bytes() == payload
            if not indexed:
                table.scan("studio-checkpoint")
            with use_asset_folders(service.project_context("first")):
                assert table.bind_name(path.name, "model/checkpoint") is None
                enriched = table.ref("mounts/studio-checkpoint/selected.safetensors")
                _, missing = bind_model_names(
                    Graph({"load": GraphNode("loader", {"file": enriched.to_wire()})}),
                    schemas,
                    library,
                )
                assert missing[0]["hidden"] is True
                assert missing[0]["sources"] == []
                if not indexed:
                    duplicate = table.bind_name("duplicate.safetensors", "model/checkpoint")
                    assert duplicate is not None
                    assert duplicate.read_bytes() == payload
                    _, missing = bind_model_names(
                        Graph({"load": GraphNode("loader", {"file": duplicate.to_wire()})}),
                        schemas,
                        library,
                    )
                    assert missing == []
                    assert (
                        asset_preflight(
                            library,
                            {digest_bytes(payload): duplicate.name},
                            set(),
                            {},
                            project="first",
                        )
                        == []
                    )
            for route in ("list", "entries?q=selected"):
                hidden = await client.get(
                    f"/api/mounts/studio-checkpoint/{route}"
                    + ("&" if "?" in route else "?")
                    + "project=first"
                )
                assert (
                    (await hidden.json())["entries"] == []
                    if "?" in route
                    else all(row["name"] != path.name for row in (await hidden.json())["entries"])
                )
                visible = await client.get(
                    f"/api/mounts/studio-checkpoint/{route}"
                    + ("&" if "?" in route else "?")
                    + "project=second"
                )
                assert any(row["name"] == path.name for row in (await visible.json())["entries"])
            planned = await client.post(
                "/api/models/downloads/plan",
                json={
                    "name": path.name,
                    "kind": "model/checkpoint",
                    "project": "first",
                    "scope": "local",
                },
            )
            assert planned.status == 200
            assert (await planned.json())["status"] == "model-hidden"
            assert service.projects.hidden_models("second") == ()
            restarted = start()
            try:
                assert restarted.projects.hidden_models("first") == service.projects.hidden_models(
                    "first"
                )
                with use_asset_folders(restarted.project_context("first")):
                    assert restarted.table.bind_name(path.name, "model/checkpoint") is None
            finally:
                await restarted.close()
            replacement = root / "replacement"
            replacement.write_bytes(b"replacement is not the hidden revision")
            replacement.replace(path)
            table.scan("studio-checkpoint")
            with use_asset_folders(service.project_context("first")):
                changed = table.bind_name(path.name, "model/checkpoint")
                assert changed is not None and changed.read_bytes() == path.read_bytes()
            result = await client.delete("/api/models/hidden/first", json={"identity": identity})
            assert result.status == 200
            assert (await result.json())["hiddenModels"] == []
            assert path.read_bytes() == b"replacement is not the hidden revision"
        finally:
            await client.close()
            await service.close()
            store.close()

    run_scenario(scenario())


@pytest.mark.parametrize("expire_stream", [False, True])
def test_declared_model_plan_pins_metadata_without_downloading_and_respects_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expire_stream: bool
) -> None:
    from aiohttp.test_utils import TestClient, TestServer
    from dinkster_assets import MountTable, huggingface
    from dinkster_assets.provenance import ProvenanceStore
    from dinkster_server import create_app
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    from dinkster.model_downloads import ModelDownloads, shared_model_mounts
    from dinkster.model_downloads_api import add_model_download_routes
    from dinkster.mounts_api import MountService
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import SCHEMAS, make_engine

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    (tmp_path / "model-settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "downloadLocation": "studio",
                "locations": [
                    {
                        "id": "studio",
                        "path": str(tmp_path / "models"),
                        "mode": "readwrite",
                        "visibility": "opt-in",
                    }
                ],
            }
        )
    )
    table = MountTable()
    for mount in shared_model_mounts():
        table.add(mount, source="shared-models")
    mounts = MountService(table, tmp_path / "mounts.toml", shared_models=True)
    downloads = ModelDownloads(mounts, ProvenanceStore(tmp_path / "provenance.json"))
    requests: list[str] = []

    @contextmanager
    def respond(url: str, *, huggingface_token: str | None = None) -> Generator[HTTPSResponse]:
        assert huggingface_token is None
        requests.append(url)
        assert url in {
            "https://huggingface.co/api/models/Comfy-Org/flux1-dev/revision/main?blobs=true",
            "https://huggingface.co/api/models/studio/custom/revision/main?blobs=true",
        }
        body = {
            "sha": "d" * 40,
            "cardData": {"license": "flux-1-dev"},
            "siblings": [
                {
                    "rfilename": filename,
                    "size": 137,
                    "lfs": {"sha256": "e" * 64, "size": 137},
                }
                for filename in ("flux1-dev.safetensors", "custom.safetensors", "ae.safetensors")
            ],
        }
        yield cast("HTTPSResponse", io.BytesIO(json.dumps(body).encode()))

    monkeypatch.setattr(huggingface, "open_model_response", respond)
    grants = frozenset(
        {"assets:read", "assets:write", "settings:write", "jobs:submit", "jobs:read"}
    )
    auth = StaticBearerAuthenticator(
        {
            "operator": Principal("owner", {"local": grants}),
            "viewer": Principal("viewer", {"local": frozenset({"assets:read"})}),
        }
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, authenticator=auth)
        add_model_download_routes(app, downloads)
        client = TestClient(TestServer(app), headers={"Authorization": "Bearer operator"})
        await client.start_server()
        body = {
            "name": "flux1-dev.safetensors",
            "kind": "model/diffusion",
            "project": "work",
            "scope": "local",
        }
        try:
            response = await client.post(
                "/api/models/downloads/plan", json=body, headers={"Authorization": "Bearer viewer"}
            )
            assert response.status == 403
            for invalid in (
                {**body, "name": "../flux1-dev.safetensors"},
                {**body, "source": "https://example.com/secret"},
                {
                    **body,
                    "source": "https://huggingface.co/studio/custom/resolve/main/other.safetensors",
                },
                {
                    **body,
                    "source": "https://huggingface.co/studio/custom/resolve/main/flux1-dev.safetensors?token=private",
                },
                {
                    **body,
                    "source": "https://huggingface.co@evil.example/studio/custom/resolve/main/flux1-dev.safetensors",
                },
                {**body, "source": None},
                {**body, "scope": "other"},
                {**body, "kind": "media/image"},
            ):
                response = await client.post("/api/models/downloads/plan", json=invalid)
                assert response.status in {400, 403}
            for name, kind, status in (
                ("unknown.bin", "model/vae", "picker-required"),
                ("ae.safetensors", "model/vae", "picker-required"),
                (body["name"], body["kind"], "destination-required"),
            ):
                response = await client.post(
                    "/api/models/downloads/plan", json={**body, "name": name, "kind": kind}
                )
                assert response.status == 200
                assert (await response.json())["status"] == status
            assert requests == []
            mounts.update_project("work", [m.id for m in shared_model_mounts()], None)
            response = await client.post("/api/models/downloads/plan", json=body)
            assert response.status == 200
            plan = await response.json()
            assert isinstance(plan.pop("planId"), str)
            assert plan == {
                "project": "work",
                "status": "consent-required",
                "provider": "huggingface",
                "kind": "model/diffusion",
                "name": body["name"],
                "repo": "Comfy-Org/flux1-dev",
                "revision": "d" * 40,
                "file": body["name"],
                "sha256": "e" * 64,
                "size": 137,
                "license": "flux-1-dev",
                "source": "https://huggingface.co/Comfy-Org/flux1-dev/resolve/"
                + "d" * 40
                + "/flux1-dev.safetensors",
                "destination": str(tmp_path / "models/diffusion_models/flux1-dev.safetensors"),
                "mount": "studio-diffusion",
            }
            assert not list((tmp_path / "models").rglob("*.safetensors"))
            assert len(requests) == 1
            local = tmp_path / "models/diffusion_models/flux1-dev.safetensors"
            local.write_bytes(b"unhashed local model")
            response = await client.post("/api/models/downloads/plan", json=body)
            assert (await response.json())["status"] == "local"
            response = await client.post(
                "/api/models/downloads/plan", json={**body, "project": "other"}
            )
            assert await response.json() == {
                "status": "folder-disabled",
                "disabledLocations": ["studio"],
                "detail": "Folder not enabled for this project",
            }
            assert len(requests) == 1
            assert local.read_bytes() == b"unhashed local model"
            from dinkster_assets import AssetVault, LibraryStore
            from dinkster_graph import Graph, GraphNode
            from dinkster_schema import AssetWidget, InputSpec, NodeSchema, TypeExpr
            from dinkster_server import ServerLibrary
            from dinkster_server.preflight import bind_model_names
            from dinkster_values.asset_folders import use_asset_folders

            from dinkster.template_model_sources import TemplateModelSources

            store = LibraryStore(tmp_path / "library.sqlite")
            library = ServerLibrary(
                AssetVault(tmp_path / "vault"),
                store,
                resolver=table,
                model_sources_for_name=TemplateModelSources().for_name,
                disabled_model_name_locations=downloads.disabled_name_locations,
            )
            graph = Graph({"loader": GraphNode("loader", {"file": body["name"]})})
            schemas = {
                "loader": NodeSchema(
                    "loader",
                    inputs=(
                        InputSpec(
                            "file",
                            TypeExpr.concrete("ASSET"),
                            widget=AssetWidget(kind="model/diffusion"),
                        ),
                    ),
                )
            }
            try:
                with use_asset_folders(mounts.project_context("other")):
                    _, missing = bind_model_names(graph, schemas, library)
                assert len(missing) == 1
                assert missing[0]["disabledLocations"] == ["studio"]
                assert missing[0]["sources"] == []
                assert missing[0]["fetchable"] is False
                assert "digest" not in missing[0]
                with use_asset_folders(mounts.project_context("work")):
                    _, missing = bind_model_names(graph, schemas, library)
                assert missing == []
            finally:
                store.close()
            local.unlink()

            for name, kind in (
                ("custom.safetensors", "model/diffusion"),
                ("ae.safetensors", "model/vae"),
            ):
                response = await client.post(
                    "/api/models/downloads/plan", json={**body, "name": name, "kind": kind}
                )
                assert (await response.json())["status"] == "picker-required"
                response = await client.post(
                    "/api/models/downloads/plan",
                    json={
                        **body,
                        "name": name,
                        "kind": kind,
                        "source": f"https://huggingface.co/studio/custom/resolve/main/{name}",
                    },
                )
                assert response.status == 200
                selected = await response.json()
                assert selected["status"] == "consent-required"
                assert selected["repo"] == "studio/custom"
                assert selected["revision"] == "d" * 40
                assert selected["file"] == name
                assert selected["sha256"] == "e" * 64
                assert selected["size"] == 137
                assert selected["destination"] == str(
                    tmp_path
                    / "models"
                    / ("vae" if kind == "model/vae" else "diffusion_models")
                    / name
                )
            assert not list((tmp_path / "models").rglob("*.safetensors"))

            from dinkster_assets import civitai

            metadata = {
                "id": 71,
                "modelId": 19,
                "files": [
                    {
                        "id": 83,
                        "modelVersionId": 71,
                        "name": "custom.safetensors",
                        "sizeKB": 0.13,
                        "hashes": {"SHA256": hashlib.sha256(b"x" * 137).hexdigest().upper()},
                        "downloadUrl": "https://civitai.com/api/download/models/71",
                    }
                ],
            }
            probed: list[io.BytesIO] = []
            civitai_bearer = "synthetic-civitai-planning"
            downloads.credentials.set_token(civitai_bearer, provider="civitai")

            class CivitaiStream(io.BytesIO):
                headers = HTTPMessage()
                headers["Content-Length"] = "137"

                def read(self, size: int | None = -1) -> bytes:
                    if expire_stream:
                        monkeypatch.setattr(
                            provider_credentials, "time", SimpleNamespace(time=lambda: 3)
                        )
                    return super().read(size)

            @contextmanager
            def civitai_response(
                url: str, *, civitai_token: str | None = None
            ) -> Generator[HTTPSResponse]:
                assert civitai_token == civitai_bearer
                if url == "https://civitai.com/api/v1/model-versions/71":
                    yield cast("HTTPSResponse", io.BytesIO(json.dumps(metadata).encode()))
                elif url == "https://civitai.com/api/v1/models/19":
                    yield cast(
                        "HTTPSResponse",
                        io.BytesIO(
                            json.dumps(
                                {"id": 19, "allowNoCredit": False, "allowCommercialUse": "None"}
                            ).encode()
                        ),
                    )
                else:
                    assert url == "https://civitai.com/api/download/models/71?fileId=83"
                    stream = CivitaiStream(b"x" * 137)
                    probed.append(stream)
                    yield cast("HTTPSResponse", stream)

            monkeypatch.setattr(civitai, "open_model_response", civitai_response)
            from types import SimpleNamespace

            from dinkster import provider_credentials

            monkeypatch.setattr(provider_credentials, "time", SimpleNamespace(time=lambda: 0))
            downloads.credentials.set_token("synthetic-expired-hf", expires_at=1)
            monkeypatch.setattr(provider_credentials, "time", SimpleNamespace(time=lambda: 2))
            with pytest.raises(provider_credentials.ProviderCredentialsExpired):
                downloads.credentials.snapshot()
            response = await client.post(
                "/api/models/downloads/plan",
                json={
                    **body,
                    "name": "custom.safetensors",
                    "source": "https://civitai.com/api/download/models/71?fileId=83",
                },
            )
            assert response.status == 200
            chosen = await response.json()
            assert chosen["status"] == "consent-required"
            assert chosen["provider"] == "civitai" and chosen["repo"] == "19"
            assert chosen["revision"] == "71" and chosen["file"] == "custom.safetensors"
            assert chosen["source"] == "https://civitai.com/api/download/models/71?fileId=83"
            assert chosen["terms"] == {
                "source": "https://civitai.com/models/19?modelVersionId=71",
                "allowNoCredit": False,
                "allowCommercialUse": "None",
            }
            assert (
                chosen["sha256"] == hashlib.sha256(b"x" * 137).hexdigest() and chosen["size"] == 137
            )
            assert chosen["destination"] == str(
                tmp_path / "models/diffusion_models/custom.safetensors"
            )
            assert len(probed) == 1 and probed[0].tell() == 0
            assert not list((tmp_path / "models").rglob("*.safetensors"))
            from dinkster_assets.fetch import ModelAccessDenied

            @contextmanager
            def refused(_url: str, *, civitai_token: str | None = None) -> Generator[HTTPSResponse]:
                assert civitai_token == civitai_bearer
                raise ModelAccessDenied(403, provider="civitai")
                yield  # pragma: no cover

            monkeypatch.setattr(civitai, "open_model_response", refused)
            response = await client.post(
                "/api/models/downloads/plan",
                json={
                    **body,
                    "name": "custom.safetensors",
                    "source": "https://civitai.com/api/download/models/71",
                },
            )
            assert response.status == 409
            assert await response.json() == {
                "error": "civitai-access-required",
                "provider": "civitai",
            }

            monkeypatch.setattr(civitai, "open_model_response", civitai_response)
            civitai_bearer = "synthetic-civitai-replacement"
            downloads.credentials.set_token(civitai_bearer, provider="civitai")
            response = await client.post(
                "/api/models/downloads/start",
                json={"planId": chosen["planId"], "consent": True},
            )
            assert response.status == 409
            assert await response.json() == {"error": "provider-credentials-changed"}
            downloads.credentials.set_token(civitai_bearer, provider="civitai", expires_at=3)
            monkeypatch.setattr(provider_credentials, "time", SimpleNamespace(time=lambda: 3))
            for endpoint, payload in (
                ("start", {"planId": chosen["planId"], "consent": True}),
                (
                    "plan",
                    {
                        **body,
                        "name": "custom.safetensors",
                        "source": "https://civitai.com/api/download/models/71?fileId=83",
                    },
                ),
            ):
                response = await client.post(f"/api/models/downloads/{endpoint}", json=payload)
                assert response.status == 409
                requirement = await response.json()
                assert requirement["error"] == "credential-required"
                assert requirement["provider"] == "civitai" and requirement["expired"] is True
                assert isinstance(requirement["requirementId"], str)
            monkeypatch.setattr(provider_credentials, "time", SimpleNamespace(time=lambda: 2))
            downloads.credentials.set_token(
                civitai_bearer, provider="civitai", expires_at=3 if expire_stream else None
            )
            response = await client.post(
                "/api/models/downloads/plan",
                json={
                    **body,
                    "name": "custom.safetensors",
                    "source": "https://civitai.com/api/download/models/71?fileId=83",
                },
            )
            assert response.status == 200
            chosen = await response.json()
            assert civitai_bearer not in json.dumps(chosen)
            response = await client.delete("/api/models/providers/huggingface?scope=local")
            assert response.status == 200
            monkeypatch.setattr(provider_credentials, "time", SimpleNamespace(time=lambda: 0))
            downloads.credentials.set_token("another-expired-hf", expires_at=1)
            monkeypatch.setattr(provider_credentials, "time", SimpleNamespace(time=lambda: 2))
            with pytest.raises(provider_credentials.ProviderCredentialsExpired):
                downloads.credentials.snapshot()
            response = await client.post(
                "/api/models/downloads/start",
                json={
                    "planId": chosen["planId"],
                    "consent": True,
                },
            )
            assert response.status == 202
            job = await response.json()
            import asyncio

            async with asyncio.timeout(5):
                while True:
                    response = await client.get(f"/api/models/downloads/{job['jobId']}")
                    completed = await response.json()
                    if completed["status"] in {"completed", "failed"}:
                        break
                    await asyncio.sleep(0.01)
            if expire_stream:
                assert completed["status"] == "failed"
                assert completed["error"] == "credential-required"
                assert completed["provider"] == "civitai" and completed["expired"] is True
                assert isinstance(completed["requirementId"], str)
                assert civitai_bearer not in json.dumps(completed)
                assert not list((tmp_path / "models").rglob("*.safetensors"))
                assert not (tmp_path / "provenance.json").exists()
                return
            assert completed["status"] == "completed"
            assert (
                tmp_path / "models/diffusion_models/custom.safetensors"
            ).read_bytes() == b"x" * 137
            downloads.credentials.set_token(None)

            def unavailable(_url: str, *, huggingface_token: str | None = None) -> None:
                assert huggingface_token is None
                raise OSError("synthetic-private-provider-response")

            monkeypatch.setattr(huggingface, "open_model_response", unavailable)
            response = await client.post("/api/models/downloads/plan", json=body)
            assert response.status == 409
            assert await response.json() == {"error": "model-source-unavailable"}
        finally:
            await client.close()
            await mounts.close()

    run_scenario(scenario())


@pytest.mark.parametrize(
    "outcome", ["complete", "cancel", "revoke", "sha256", "index", "reuse", "reuse-outside"]
)
def test_consented_model_job_streams_verifies_and_preserves_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    import asyncio
    from threading import Event

    from aiohttp.test_utils import TestClient, TestServer
    from dinkster_assets import MountTable, huggingface
    from dinkster_assets.provenance import ProvenanceStore
    from dinkster_server import create_app
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    from dinkster.model_downloads import ModelDownloads, shared_model_mounts
    from dinkster.model_downloads_api import add_model_download_routes
    from dinkster.mounts_api import MountService
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import SCHEMAS, make_engine

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    if outcome == "reuse-outside":
        monkeypatch.setattr("dinkster_assets.integrity._stat_scheme", lambda: "windows-v1")
    root = tmp_path / "models"
    (tmp_path / "model-settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "downloadLocation": "studio",
                "locations": [
                    {
                        "id": "studio",
                        "path": str(root),
                        "mode": "readwrite",
                    }
                ],
            }
        )
    )
    table = MountTable()
    for mount in shared_model_mounts():
        table.add(mount, source="shared-models")
    mounts = MountService(table, tmp_path / "mounts.toml", shared_models=True)
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    downloads = ModelDownloads(mounts, provenance)
    if outcome == "index":

        def index_failure(_mount: str) -> None:
            raise OSError("synthetic private indexing failure")

        monkeypatch.setattr(table, "refresh_publication", index_failure)
    payload = b"verified-model-123"
    transferred = b"corrupt!-model-123" if outcome == "sha256" else payload
    assert len(payload) == len(transferred)
    sha256 = hashlib.sha256(payload).hexdigest()
    blocked, release = Event(), Event()
    requests: list[str] = []
    name = "flux1-dev.safetensors"
    source = "https://huggingface.co/Comfy-Org/flux1-dev/resolve/" + "b" * 40 + "/" + name

    class Stream(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell() == 0:
                return super().read(3)
            if self.tell() == 3:
                blocked.set()
                assert release.wait(10), "test did not release provider stream"
            return super().read(size)

    @contextmanager
    def respond(url: str, *, huggingface_token: str | None = None) -> Generator[HTTPSResponse]:
        assert huggingface_token is None
        requests.append(url)
        if "/api/models/" in url:
            body = {
                "sha": "b" * 40,
                "siblings": [
                    {
                        "rfilename": name,
                        "size": len(payload),
                        "lfs": {"sha256": sha256, "size": len(payload)},
                    }
                ],
            }
            yield cast("HTTPSResponse", io.BytesIO(json.dumps(body).encode()))
        else:
            assert url == source
            yield cast("HTTPSResponse", Stream(transferred))

    monkeypatch.setattr(huggingface, "open_model_response", respond)
    grants = frozenset(
        {"assets:read", "assets:write", "settings:write", "jobs:read", "jobs:submit", "jobs:cancel"}
    )
    auth = StaticBearerAuthenticator(
        {
            "owner": Principal("owner", {"local": grants}),
            "other": Principal("other", {"local": grants}),
            "no-execute": Principal("owner", {"local": grants - {"jobs:submit"}}),
        }
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, authenticator=auth)
        add_model_download_routes(app, downloads)
        client = TestClient(TestServer(app), headers={"Authorization": "Bearer owner"})
        await client.start_server()
        try:
            response = await client.post(
                "/api/models/downloads/plan",
                json={
                    "name": name,
                    "kind": "model/diffusion",
                    "project": "work",
                    "scope": "local",
                },
            )
            assert response.status == 200
            plan = await response.json()
            assert "digest" not in plan
            start = {"planId": plan["planId"], "consent": True}
            for body, headers, expected in (
                ({**start, "consent": False}, {}, 400),
                ({**start, "source": "https://example.com/untrusted"}, {}, 400),
                (start, {"Authorization": "Bearer other"}, 404),
                (start, {"Authorization": "Bearer no-execute"}, 403),
            ):
                response = await client.post(
                    "/api/models/downloads/start", json=body, headers=headers
                )
                assert response.status == expected
            assert len(requests) == 1
            mounts.update_project("work", [], None)
            response = await client.post("/api/models/downloads/start", json=start)
            assert response.status == 409
            assert (await response.json())["error"] == "download-destination-changed"
            mounts.update_project("work", list(mounts.mount_ids()), None)
            response = await client.post(
                "/api/models/downloads/plan",
                json={
                    "name": name,
                    "kind": "model/diffusion",
                    "project": "work",
                    "scope": "local",
                },
            )
            second_plan = await response.json()
            old = root / "diffusion_models" / name
            old.write_bytes(b"old-checkpoint-do-not-overwrite")
            existing = (
                root / "unconventional" / "different-name.safetensors"
                if outcome == "reuse-outside"
                else old.parent / "nested" / "different-name.safetensors"
            )
            if outcome in {"reuse", "reuse-outside"}:
                existing.parent.mkdir()
                existing.write_bytes(payload)
                (old.parent / "same-size-corrupt.safetensors").write_bytes(b"x" * len(payload))
            response = await client.post("/api/models/downloads/start", json=start)
            assert response.status == 202
            job = await response.json()
            job_id = job["jobId"]
            if outcome in {"reuse", "reuse-outside"}:
                async with asyncio.timeout(5):
                    while True:
                        response = await client.get(f"/api/models/downloads/{job_id}")
                        status = await response.json()
                        if status["status"] in {"completed", "failed"}:
                            break
                        await asyncio.sleep(0.01)
                assert status["status"] == "completed"
                assert status["reused"] is True and status["renamed"] is False
                assert status["path"] == str(existing)
                assert status["asset"]["digest"] == digest_bytes(payload)
                assert table.resolve(digest_bytes(payload)) == existing
                assert existing.read_bytes() == payload
                assert old.read_bytes() == b"old-checkpoint-do-not-overwrite"
                assert not old.with_name("flux1-dev-" + sha256[:8] + ".safetensors").exists()
                assert not [url for url in requests if "/api/models/" not in url]
                record = provenance.get(digest_bytes(payload))
                assert record is not None and record.metadata["sha256"] == sha256
                rebound = table.bind_name(name, "model/diffusion")
                assert rebound is not None and rebound.local_path() == existing
                response = await client.post(
                    "/api/models/downloads/plan",
                    json={
                        "name": name,
                        "kind": "model/diffusion",
                        "project": "work",
                        "scope": "local",
                    },
                )
                assert response.status == 200 and (await response.json())["status"] == "local"
                if outcome == "reuse-outside":
                    from dinkster_assets import AssetError, AssetRef, MountSnapshotResolver
                    from dinkster_values.asset_folders import use_asset_folders

                    sibling = existing.with_name("unrelated.safetensors")
                    sibling.write_bytes(b"unrelated model")
                    snapshot = tmp_path / "restarted-mounts.json"
                    restarted_table = MountTable(snapshot_path=snapshot)
                    restarted = MountService(
                        restarted_table, tmp_path / "mounts.toml", shared_models=True
                    )
                    with use_asset_folders(restarted.project_context("work")):
                        resolver = MountSnapshotResolver(snapshot)
                        bound = restarted_table.bind_name(existing.name, "model/diffusion")
                        assert isinstance(bound, AssetRef)
                        assert bound.local_path() == existing
                        alias = restarted_table.bind_name(name, "model/diffusion")
                        assert isinstance(alias, AssetRef) and alias.local_path() == existing
                        with bound.open() as contents:
                            assert contents.read() == payload
                        assert (
                            resolver.resolve_for_kind(bound.digest, "model/diffusion") == existing
                        )
                        assert resolver.resolve_for_kind(bound.digest, "model/vae") is None
                        assert restarted_table.bind_name(sibling.name, "model/diffusion") is None
                        assert root / "unconventional" not in resolver.roots_for_kind(
                            "model/diffusion"
                        )
                    from dinkster_assets.huggingface import HuggingFaceFile
                    from dinkster_assets.selections import resolve_asset_selection

                    restarted_downloads = ModelDownloads(restarted, provenance)
                    vae_target = restarted_downloads.target("model/vae", "work")
                    assert vae_target is not None
                    vae_ref = restarted_downloads.download_selected(
                        HuggingFaceFile(
                            "Comfy-Org/flux1-dev", "b" * 40, name, sha256, len(payload), ""
                        ),
                        vae_target,
                        "work",
                        lambda _ref, _url: None,
                    )
                    with use_asset_folders(restarted.project_context("work")):
                        loaded = resolve_asset_selection(
                            "model/vae", vae_ref, env={"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}
                        )
                        assert isinstance(loaded, AssetRef)
                        assert loaded.local_path() == existing
                        aliased_vae = resolve_asset_selection(
                            "model/vae", name, env={"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}
                        )
                        assert isinstance(aliased_vae, AssetRef)
                        assert aliased_vae.local_path() == existing
                        with pytest.raises(AssetError, match="not granted"):
                            resolve_asset_selection(
                                "model/vae",
                                sibling.name,
                                env={"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)},
                            )
                    restarted.update_project("disabled", [], None)
                    with use_asset_folders(restarted.project_context("disabled")):
                        assert resolver.resolve_for_kind(bound.digest, "model/diffusion") is None
                    import os

                    stamp = existing.stat()
                    existing.write_bytes(b"x" * len(payload))
                    os.utime(existing, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    with use_asset_folders(restarted.project_context("work")):
                        assert restarted_table.bind_name(existing.name, "model/diffusion") is None
                        assert restarted_table.bind_name(name, "model/diffusion") is None
                        assert resolver.resolve_for_kind(bound.digest, "model/diffusion") is None
                        with pytest.raises(AssetError, match="no longer readable"):
                            resolve_asset_selection(
                                "model/vae", name, env={"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}
                            )
                return
            assert await asyncio.to_thread(blocked.wait, 5)
            response = await client.get(f"/api/models/downloads/{job_id}")
            status = await response.json()
            assert status["status"] == "downloading" and status["received"] == 3
            assert status["size"] == len(payload) and "asset" not in status
            response = await client.post(
                "/api/models/downloads/start",
                json={
                    "planId": second_plan["planId"],
                    "consent": True,
                },
            )
            assert response.status == 409
            assert (await response.json())["error"] == "model-download-busy"
            response = await client.get("/api/models/downloads")
            assert [row["jobId"] for row in (await response.json())["jobs"]] == [job_id]
            response = await client.get(
                "/api/models/downloads", headers={"Authorization": "Bearer other"}
            )
            assert (await response.json())["jobs"] == []
            for method in (client.get, client.delete):
                response = await method(
                    f"/api/models/downloads/{job_id}", headers={"Authorization": "Bearer other"}
                )
                assert response.status == 404
            response = await client.post("/api/models/downloads/start", json=start)
            assert response.status == 200 and (await response.json())["jobId"] == job_id
            if outcome == "cancel":
                response = await client.delete(f"/api/models/downloads/{job_id}")
                assert response.status == 200 and (await response.json())["status"] == "cancelling"
            elif outcome == "revoke":
                mounts.update_project("work", [], None)
            release.set()
            async with asyncio.timeout(5):
                while True:
                    response = await client.get(f"/api/models/downloads/{job_id}")
                    status = await response.json()
                    if status["status"] in {"completed", "cancelled", "failed"}:
                        break
                    await asyncio.sleep(0.01)
            assert old.read_bytes() == b"old-checkpoint-do-not-overwrite"
            assert not list(root.rglob(".dinkster-save-*"))
            if outcome in {"complete", "index"}:
                assert status["status"] == "completed"
                asset = status["asset"]
                expected = old.with_name("flux1-dev-" + sha256[:8] + ".safetensors")
                assert status["path"] == str(expected)
                assert expected.read_bytes() == payload
                assert asset["digest"] == digest_bytes(payload)
                assert status["renamed"] is True
                if outcome == "index":
                    assert status["indexingPending"] is True
                else:
                    assert "indexingPending" not in status
                    assert table.resolve(asset["digest"]) == expected
                record = provenance.get(asset["digest"])
                assert record is not None and record.sources == (source,)
                assert record.metadata["sha256"] == sha256
                response = await client.delete(f"/api/models/downloads/{job_id}")
                assert response.status == 409
            else:
                assert status["status"] == ("cancelled" if outcome == "cancel" else "failed")
                assert "asset" not in status
                assert list(old.parent.glob("*.safetensors")) == [old]
            assert len([url for url in requests if "/api/models/" not in url]) == 1
        finally:
            release.set()
            await client.close()
            await mounts.close()

    run_scenario(scenario())


@pytest.mark.parametrize("change", ["symlink", "digest", "mutate", "revoke", "cancel"])
def test_model_reuse_refuses_unverified_changed_or_revoked_files(
    tmp_path: Path, change: str
) -> None:
    import os

    from dinkster_assets import AssetError
    from dinkster_assets.fetch import FetchAborted

    root = tmp_path / "models"
    root.mkdir()
    snapshot = one_mount_snapshot(tmp_path, root)
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    payload = b"existing-asymmetric-model-123"
    existing = root / "different-name.bin"
    existing.write_bytes(payload)
    expected = digest_bytes(payload)
    if change == "symlink":
        existing.unlink()
        outside = tmp_path / "outside.bin"
        outside.write_bytes(payload)
        symlink_or_skip(existing, outside)
    elif change == "digest":
        expected = digest_bytes(b"different-workflow-expectation")

    def publish(_ref: object) -> None:
        if change == "mutate":
            before = existing.stat()
            existing.write_bytes(b"z" * len(payload))
            os.utime(existing, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif change == "revoke":
            one_mount_snapshot(tmp_path, root, "readonly")

    def progress(_stage: str, received: int) -> None:
        if change == "cancel" and received:
            raise FetchAborted()

    if change in {"symlink", "digest"}:
        result = writer.reuse_model(
            SaveTarget("out", "provider.bin"),
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
            digest=expected,
            before_publish=publish,
        )
        assert result is None
    else:
        with pytest.raises(FetchAborted if change == "cancel" else AssetError):
            writer.reuse_model(
                SaveTarget("out", "provider.bin"),
                sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
                digest=expected,
                before_publish=publish,
                progress=progress,
            )
    assert not (root / "provider.bin").exists()
    assert existing.read_bytes() == (b"z" * len(payload) if change == "mutate" else payload)


@pytest.mark.parametrize("direct_version", [False, True])
def test_civitai_selection_requires_file_choice_and_verifies_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direct_version: bool
) -> None:
    from dinkster_assets import AssetError, civitai

    payload = b"Civitai model content"
    sha256 = hashlib.sha256(payload).hexdigest()
    selected_url = "https://civitai.com/api/download/models/72?fileId=84"
    versions = [
        {
            "id": version,
            "files": [
                {
                    "id": file_id,
                    "modelVersionId": version,
                    "name": "custom.safetensors",
                    "sizeKB": 0.02,
                    "hashes": {"SHA256": sha256.upper()},
                    "downloadUrl": f"https://civitai.com/api/download/models/{version}?fileId={file_id}",
                }
            ],
        }
        for version, file_id in [(71, 83), (72, 84)]
    ]
    metadata = {
        "id": 19,
        "modelVersions": versions,
        "allowNoCredit": False,
        "allowDerivatives": True,
        "allowDifferentLicense": "malformed",
        "allowCommercialUse": ["Image", "RentCivit"] if direct_version else "{Image,RentCivit}",
        "description": "synthetic-private-body-canary",
    }
    headers = HTTPMessage()
    headers["Content-Length"] = str(len(payload))
    streams: list[io.BytesIO] = []

    class Stream(io.BytesIO):
        headers: HTTPMessage

    @contextmanager
    def respond(url: str, *, civitai_token: str | None = None) -> Generator[HTTPSResponse]:
        assert civitai_token == "synthetic-civitai-bearer"
        if url == "https://civitai.com/api/v1/models/19":
            yield cast("HTTPSResponse", io.BytesIO(json.dumps(metadata).encode()))
        elif url == "https://civitai.com/api/v1/model-versions/72":
            yield cast(
                "HTTPSResponse", io.BytesIO(json.dumps({**versions[1], "modelId": 19}).encode())
            )
        else:
            assert url == selected_url
            stream = Stream(payload)
            stream.headers = headers
            streams.append(stream)
            yield cast("HTTPSResponse", stream)

    monkeypatch.setattr(civitai, "open_model_response", respond)
    assert civitai.select_civitai_file(
        "https://civitai.com/models/19/example",
        "custom.safetensors",
        token="synthetic-civitai-bearer",
    ) == (
        "https://civitai.com/api/download/models/71?fileId=83",
        selected_url,
    )
    assert streams == []
    selected = civitai.select_civitai_file(
        selected_url
        if direct_version
        else "https://civitai.com/models/19/example?modelVersionId=72",
        "custom.safetensors",
        token="synthetic-civitai-bearer",
    )
    assert isinstance(selected, civitai.CivitaiFile)
    assert selected.url == selected_url and selected.revision == "72"
    assert selected.sha256 == sha256 and selected.size == len(payload)
    assert selected.license == ""
    assert selected.terms == {
        "source": "https://civitai.com/models/19?modelVersionId=72",
        "allowNoCredit": False,
        "allowDerivatives": True,
        "allowCommercialUse": ["Image", "RentCivit"] if direct_version else "{Image,RentCivit}",
    }
    assert streams[0].tell() == 0
    root = tmp_path / "models"
    root.mkdir()
    writer = AssetWriter(MountSnapshotWriter(one_mount_snapshot(tmp_path, root)))
    ref = selected.download(
        writer, SaveTarget("out", "custom.safetensors"), token="synthetic-civitai-bearer"
    )
    assert ref.size == len(payload)
    assert (root / "custom.safetensors").read_bytes() == payload

    from dinkster_assets import MountTable
    from dinkster_assets.provenance import ProvenanceStore

    from dinkster.model_downloads import ModelDownloads, shared_model_mounts
    from dinkster.mounts_api import MountService

    home = tmp_path / "settings"
    home.mkdir()
    (home / "model-settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "downloadLocation": "isolated",
                "locations": [
                    {"id": "isolated", "path": str(tmp_path / "published"), "mode": "readwrite"}
                ],
            }
        )
    )
    monkeypatch.setenv("DINKSTER_HOME", str(home))
    table = MountTable()
    for mount in shared_model_mounts():
        table.add(mount, source="shared-models")
    store_path = tmp_path / "provenance.json"
    downloads = ModelDownloads(MountService(table), ProvenanceStore(store_path))
    downloads.credentials.set_token("synthetic-civitai-bearer", provider="civitai")
    target = downloads.target("model/checkpoint", "project")
    assert target is not None
    published = downloads.download_selected(selected, target, "project", lambda *_args: None)
    record = ProvenanceStore(store_path).get(published.digest)
    assert record is not None and record.license == ""
    assert record.sources == (selected_url,)
    assert record.metadata["terms"] == selected.terms
    assert record.metadata["revision"] == "72" and record.metadata["provider"] == "civitai"
    assert "synthetic-" not in store_path.read_text()

    payload = b"wrong content of equal length"[: len(payload)].ljust(len(payload), b"x")
    with pytest.raises(AssetError):
        selected.download(
            writer, SaveTarget("out", "corrupt.safetensors"), token="synthetic-civitai-bearer"
        )
    assert not (root / "corrupt.safetensors").exists()
    del headers["Content-Length"]
    with pytest.raises(AssetError, match="exact file size"):
        civitai.select_civitai_file(
            "https://civitai.com/models/19?modelVersionId=72",
            "custom.safetensors",
            token="synthetic-civitai-bearer",
        )
    for source in (
        "https://civitai.com/models/19?token=secret",
        "https://civitai.com/models/19?modelVersionId=71&modelVersionId=72",
        "https://civitai.com@evil.example/models/19",
        "https://civitai.com/api/download/models/71?fileId=83&token=secret",
    ):
        with pytest.raises(AssetError):
            civitai.select_civitai_file(source, "custom.safetensors")


def test_huggingface_selection_pins_metadata_and_verifies_named_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import fetch

    payload = b"provider-checkpoint-12345"
    repo, commit = "acme/checkpoints", "b" * 40
    sha256 = hashlib.sha256(payload).hexdigest()
    metadata = json.dumps(
        {
            "sha": commit,
            "cardData": {"license": "creativeml-openrail-m"},
            "siblings": [
                {
                    "rfilename": "weights/model.safetensors",
                    "size": len(payload),
                    "lfs": {"sha256": sha256, "size": len(payload)},
                },
            ],
        }
    ).encode()
    requests: list[str] = []
    authorizations: list[str | None] = []
    token = "synthetic-hf-transport-canary"

    class Response(io.BytesIO):
        status = 200
        headers = HTTPMessage()

    @contextmanager
    def respond(
        url: str, _addresses: object, headers: dict[str, str], _timeout: object
    ) -> Generator[HTTPSResponse]:
        requests.append(url)
        authorizations.append(headers.get("Authorization"))
        if "/api/models/" in url:
            data = metadata
        elif url == f"https://huggingface.co/{repo}/resolve/{commit}/weights/model.safetensors":
            response = Response()
            response.status = 302
            response.headers = HTTPMessage()
            response.headers["Location"] = "https://cdn.example/model?signature=synthetic"
            yield cast("HTTPSResponse", response)
            return
        else:
            assert url == "https://cdn.example/model?signature=synthetic"
            data = payload
        yield cast("HTTPSResponse", Response(data))

    monkeypatch.setattr(fetch, "_globally_routable_addresses", lambda *_args: ("8.8.8.8",))
    monkeypatch.setattr(fetch, "_open_pinned_https", respond)
    selection = select_huggingface_file(
        f"https://huggingface.co/{repo}/resolve/main/weights/model.safetensors", token=token
    )
    assert selection.revision == commit
    assert selection.sha256 == sha256
    assert selection.license == "creativeml-openrail-m"
    root = tmp_path / "models"
    root.mkdir()
    writer = AssetWriter(MountSnapshotWriter(one_mount_snapshot(tmp_path, root)))
    ref = selection.download(
        writer, SaveTarget("out", "model.safetensors"), digest=digest_bytes(payload), token=token
    )
    assert authorizations == [f"Bearer {token}", f"Bearer {token}", None]
    assert token not in repr(selection)
    assert (root / ref.name).read_bytes() == payload
    assert ref.name == "model.safetensors"
    assert ref.digest == digest_bytes(payload)
    assert requests == [
        f"https://huggingface.co/api/models/{repo}/revision/main?blobs=true",
        f"https://huggingface.co/{repo}/resolve/{commit}/weights/model.safetensors",
        "https://cdn.example/model?signature=synthetic",
    ]


def test_fresh_model_settings_are_shared_without_changing_existing_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import MountTable

    from dinkster.model_downloads import shared_model_mounts
    from dinkster.mounts_api import MountService

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("DINKSTER_HOME", str(home / ".dinkster"))
    first = shared_model_mounts()
    saved = (home / ".dinkster" / "model-settings.json").read_bytes()
    assert {mount.path.name for mount in first} == {
        "checkpoints",
        "loras",
        "vae",
        "clip",
        "text_encoders",
        "diffusion_models",
        "controlnet",
        "upscale_models",
        "embeddings",
        "clip_vision",
    }
    for project in ("first", "second"):
        table = MountTable()
        for mount in shared_model_mounts():
            assert mount.path.parent == home / "Dinkster" / "Models"
            assert mount.path.is_dir() and mount.mode == "readwrite"
            table.add(mount, source="shared-models")
        service = MountService(table, tmp_path / project / "mounts.toml")
        assert set(service.project_context(project).mounts) == {mount.id for mount in first}
    assert (home / ".dinkster" / "model-settings.json").read_bytes() == saved


@pytest.mark.parametrize("invalid", ["mode", "duplicate", "default"])
def test_invalid_model_settings_create_no_model_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    from dinkster_assets import MountsError

    from dinkster.model_downloads import shared_model_mounts

    home = tmp_path / "settings"
    home.mkdir()
    locations = [
        {"id": "valid", "path": str(tmp_path / "valid"), "mode": "readwrite"},
        {
            "id": "valid" if invalid == "duplicate" else "other",
            "path": str(tmp_path / "other"),
            "mode": "invalid" if invalid == "mode" else "readwrite",
        },
    ]
    (home / "model-settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "downloadLocation": "absent" if invalid == "default" else "valid",
                "locations": locations,
            }
        )
    )
    monkeypatch.setenv("DINKSTER_HOME", str(home))
    with pytest.raises(MountsError):
        shared_model_mounts()
    assert not (tmp_path / "valid").exists()
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("metadata_failure", ["", "journal", "provenance"])
def test_shared_model_default_is_used_and_published_before_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata_failure: str
) -> None:
    from dinkster_assets import (
        AssetNeed,
        MountSnapshotResolver,
        MountTable,
        RemoteSource,
        huggingface,
    )
    from dinkster_assets.provenance import ProvenanceStore

    from dinkster.model_downloads import ModelDownloads, shared_model_mounts
    from dinkster.mounts_api import MountService

    home = tmp_path / "settings"
    home.mkdir()
    locations = [
        {"id": name, "path": str(tmp_path / name), "mode": mode}
        for name, mode in (("existing", "read"), ("other", "readwrite"), ("selected", "readwrite"))
    ]
    (home / "model-settings.json").write_text(
        json.dumps({"version": 1, "downloadLocation": "selected", "locations": locations})
    )
    monkeypatch.setenv("DINKSTER_HOME", str(home))
    snapshot = tmp_path / "snapshot.json"
    table = MountTable(snapshot, index_root=tmp_path / "indexes")
    for mount in shared_model_mounts():
        table.add(mount, source="shared-models")
    service = ModelDownloads(MountService(table), ProvenanceStore(tmp_path / "provenance.json"))
    payload = b"fresh text encoder verified before a directory scan"
    sha256 = hashlib.sha256(payload).hexdigest()
    metadata = json.dumps(
        {
            "sha": "b" * 40,
            "cardData": {"license": "apache-2.0"},
            "siblings": [
                {
                    "rfilename": "weights/encoder.safetensors",
                    "size": len(payload),
                    "lfs": {"sha256": sha256, "size": len(payload)},
                }
            ],
        }
    ).encode()

    @contextmanager
    def respond(url: str, *, huggingface_token: str | None = None) -> Generator[HTTPSResponse]:
        assert huggingface_token is None
        yield cast("HTTPSResponse", io.BytesIO(metadata if "/api/models/" in url else payload))

    monkeypatch.setattr(huggingface, "open_model_response", respond)
    need = AssetNeed(
        digest=digest_bytes(payload),
        name="encoder.safetensors",
        kind="model/text-encoder",
        sources=(
            RemoteSource(
                "https://huggingface.co/acme/model/resolve/main/weights/encoder.safetensors"
            ),
        ),
    )
    expected = tmp_path / "selected" / "text_encoders" / "encoder.safetensors"

    def publish(_ref: object, _url: str) -> None:
        if metadata_failure == "journal":
            raise OSError("journal unavailable")

    if metadata_failure == "provenance":
        (tmp_path / "provenance.json").mkdir()
    if metadata_failure:
        from dinkster_assets.vault import AssetPublicationError

        with pytest.raises(AssetPublicationError):
            service.acquire(need, "project", publish)
        assert not expected.exists()
        assert not list(expected.parent.glob(".dinkster-save-*"))
        return
    from dinkster_assets import AssetVault, LibraryStore
    from dinkster_server import ServerLibrary
    from dinkster_server.preflight import asset_preflight

    library = ServerLibrary(
        AssetVault(tmp_path / "internal-storage"),
        LibraryStore(tmp_path / "library.sqlite"),
        resolver=table,
        local_resolver=table,
        upload_writer=AssetWriter(table),
        project_folders_for=service.mounts.project_context,
        provenance=service.provenance,
        acquire_model=service.acquire,
        model_destination=service.destination,
        asset_needs={need.digest: need},
    )
    assert asset_preflight(library, {}, set(), {}, project="project") == []
    referenced = {need.digest: need.name}
    plan = asset_preflight(library, referenced, set(), {}, project="project")
    assert len(plan) == 1
    assert plan[0]["sources"] == [need.sources[0].to_wire()["url"]]
    assert plan[0]["kind"] == "model/text-encoder" and plan[0]["fetchable"] is True
    assert plan[0]["destination"] == str(expected.parent)
    assert not expected.exists()
    assert asset_preflight(library, referenced, {need.digest}, {}, project="project") == []
    assert not library.vault.has(need.digest)
    assert not list((library.vault.root / ".upload-observations").glob("*.json"))
    assert library.locate(need.digest) == expected
    assert expected.read_bytes() == payload
    assert not (tmp_path / "existing").exists()
    assert not list((tmp_path / "other").rglob("*.safetensors"))
    assert MountSnapshotResolver(snapshot).resolve(need.digest) == expected
    assert table.ref("mounts/selected-text-encoder/encoder.safetensors").read_bytes() == payload
    record = ProvenanceStore(tmp_path / "provenance.json").get(need.digest)
    assert record is not None and record.license == "apache-2.0"
    assert record.metadata["sha256"] == sha256

    from dataclasses import replace

    from dinkster_assets import AssetVault
    from dinkster_assets.acquire import acquire_need

    def no_network(_url: str, *, huggingface_token: str | None = None) -> None:
        pytest.fail("already-held bytes must not reach a provider")

    monkeypatch.setattr(huggingface, "open_model_response", no_network)
    same_bytes = replace(need, name="a-different-provider-name.safetensors")
    reused = acquire_need(
        same_bytes,
        AssetVault(tmp_path / "internal-storage"),
        resolver=table,
        remote_acquire=lambda missing: service.acquire(missing, "project", publish),
    )
    assert reused.status == "held" and reused.path == expected
    assert list(expected.parent.glob("*.safetensors")) == [expected]
    for row in table.descriptors():
        if row.get("kind") == "model/text-encoder" and row["mode"] == "readwrite":
            table.remove(str(row["id"]))
    blocked = asset_preflight(library, referenced, set(), {}, project="project")
    assert len(blocked) == 1 and blocked[0]["fetchable"] is False
    assert "destination" not in blocked[0]
    assert expected.read_bytes() == payload
    library.store.close()


@pytest.mark.parametrize("bad_field", ["sha256", "size", "revision"])
def test_huggingface_refuses_unverified_provider_metadata(
    monkeypatch: pytest.MonkeyPatch, bad_field: str
) -> None:
    from dinkster_assets import AssetError, huggingface

    row = {
        "rfilename": "model.safetensors",
        "size": 7,
        "lfs": {"size": 7, "sha256": "c" * 64 if bad_field != "sha256" else "git-blob-id"},
    }
    if bad_field == "size":
        row["size"] = True
    body = json.dumps(
        {"sha": "d" * 40 if bad_field != "revision" else "main", "siblings": [row]}
    ).encode()

    @contextmanager
    def respond(_url: str, *, huggingface_token: str | None = None) -> Generator[HTTPSResponse]:
        assert huggingface_token is None
        yield cast("HTTPSResponse", io.BytesIO(body))

    monkeypatch.setattr(huggingface, "open_model_response", respond)
    with pytest.raises(AssetError):
        select_huggingface_file("https://huggingface.co/acme/model/resolve/main/model.safetensors")


def test_model_folder_modes_persist_project_choices_and_follow_shared_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import AssetError, MountTable
    from dinkster_assets.revision import mount_revision_capability
    from dinkster_values.asset_folders import use_asset_folders

    from dinkster.model_settings import ModelSettings
    from dinkster.mounts_api import MountService
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import make_client

    def capability(root: Path) -> dict[str, object]:
        if not root.exists():
            return {"supported": False, "reason": "directory does not exist"}
        return mount_revision_capability(root)

    monkeypatch.setattr("dinkster_assets.mounts.mount_revision_capability", capability)
    settings = tmp_path / "settings"
    settings.mkdir()
    monkeypatch.setenv("DINKSTER_HOME", str(settings))
    document = {
        "version": 1,
        "downloadLocation": "common",
        "locations": [
            {
                "id": name,
                "path": str(tmp_path / name),
                "mode": "readwrite" if name == "common" else "read",
                "visibility": mode,
            }
            for name, mode in (
                ("required", "always-on"),
                ("common", "default-on"),
                ("private", "opt-in"),
            )
        ],
    }
    path = settings / "model-settings.json"
    path.write_text(json.dumps(document))
    model_settings = ModelSettings.load()
    table = MountTable()
    for mount in model_settings.prepare():
        table.add(mount, source="shared-models")
    private = tmp_path / "private" / "checkpoints"
    private.mkdir(parents=True)
    payload = b"opt-in checkpoint, intentionally not indexed"
    (private / "private.safetensors").write_bytes(payload)
    service = MountService(
        table, tmp_path / "library" / "mounts.toml", allow_changes=True, shared_models=True
    )
    categories = (
        "checkpoint",
        "lora",
        "vae",
        "text-encoder",
        "diffusion",
        "controlnet",
        "upscale",
        "upscaler",
        "embedding",
        "clip-vision",
        "clip",
    )
    required = tuple(f"required-{category}" for category in categories)
    common = tuple(f"common-{category}" for category in categories)
    optional = tuple(f"private-{category}" for category in categories)
    for project in ("first", "second"):
        assert set(service.project_context(project).mounts) == set(required + common)
        with use_asset_folders(service.project_context(project)):
            assert table.bind_name("private.safetensors", "model/checkpoint") is None

    async def scenario() -> None:
        client = await make_client(service)
        try:
            # An always-on location cannot be silently disabled by the old folder editor.
            response = await client.put(
                "/api/mounts/projects/first", json={"mounts": list(common), "outputMount": None}
            )
            assert response.status == 400
            assert "always-on" in (await response.json())["error"]
            response = await client.put(
                "/api/mounts/projects/first",
                json={"mounts": list(required + optional[:1]), "outputMount": None},
            )
            assert response.status == 400
            assert "complete model location" in (await response.json())["error"]
            chosen = list(required + optional)
            response = await client.put(
                "/api/mounts/projects/first", json={"mounts": chosen, "outputMount": None}
            )
            assert response.status == 200
            assert [row["id"] for row in (await response.json())["mounts"]] == chosen
            assert set(service.project_context("second").mounts) == set(required + common)
            with use_asset_folders(service.project_context("first")):
                with monkeypatch.context() as patch:
                    patch.setattr(
                        "dinkster_assets.mounts.mount_revision_capability",
                        lambda _root: {"supported": False, "reason": "untracked filesystem"},
                    )
                    with pytest.raises(AssetError, match="untracked filesystem"):
                        table.bind_name("private.safetensors", "model/checkpoint")
                ref = table.bind_name("private.safetensors", "model/checkpoint")
                assert ref is not None and ref.read_bytes() == payload
        finally:
            await service.close()
            await client.close()

    run_scenario(scenario())
    restarted = MountService(
        table, tmp_path / "library" / "mounts.toml", allow_changes=True, shared_models=True
    )
    assert restarted.project_context("first").mounts == required + optional
    assert set(restarted.project_context("second").mounts) == set(required + common)
    # Shared mode edits take effect in both running project libraries, not only on restart.
    document["locations"][1]["visibility"] = "always-on"
    document["locations"][2]["visibility"] = "default-on"
    path.write_text(json.dumps(document))
    for owner in (service, restarted):
        assert set(owner.project_context("first").mounts) == set(required + common + optional)
        assert set(owner.project_context("second").mounts) == set(required + common + optional)
    assert (private / "private.safetensors").read_bytes() == payload


@pytest.mark.parametrize("invalid", ["visibility", "duplicate-path"])
def test_invalid_model_folder_modes_leave_existing_bytes_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    from dinkster_assets import MountsError

    from dinkster.model_downloads import shared_model_mounts

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.bin").write_bytes(b"do not change")
    rows = [
        {"id": "first", "path": str(existing), "mode": "readwrite", "visibility": "default-on"},
        {
            "id": "second",
            "path": str(existing if invalid == "duplicate-path" else tmp_path / "uncreated"),
            "mode": "readwrite",
            "visibility": "hidden" if invalid == "visibility" else "opt-in",
        },
    ]
    (tmp_path / "model-settings.json").write_text(
        json.dumps({"version": 1, "downloadLocation": "first", "locations": rows})
    )
    with pytest.raises(MountsError):
        shared_model_mounts()
    assert sorted(path.name for path in existing.iterdir()) == ["keep.bin"]
    assert (existing / "keep.bin").read_bytes() == b"do not change"
    assert not (tmp_path / "uncreated").exists()


def test_explicit_digest_requires_enabled_folder_before_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import AssetVault, LibraryStore, MountDef, MountTable
    from dinkster_assets.provenance import ProvenanceStore
    from dinkster_server import ServerLibrary
    from dinkster_server.preflight import asset_preflight

    from dinkster.model_downloads import ModelDownloads, shared_model_mounts
    from dinkster.mounts_api import MountService

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    root = tmp_path / "checkpoints"
    root.mkdir()
    payload = b"model in an opt-in folder"
    (root / "model.safetensors").write_bytes(payload)
    digest = digest_bytes(payload)
    (tmp_path / "model-settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "downloadLocation": None,
                "locations": [
                    {
                        "id": "studio",
                        "path": str(root),
                        "kind": "model/checkpoint",
                        "mode": "read",
                        "visibility": "opt-in",
                    }
                ],
            }
        )
    )
    table = MountTable()
    for mount in shared_model_mounts():
        table.add(mount, source="shared-models", kind="model/checkpoint")
        table.scan(mount.id)
    service = MountService(table, tmp_path / "mounts.toml", shared_models=True)
    downloads = ModelDownloads(service, ProvenanceStore(tmp_path / "provenance.json"))
    store = LibraryStore(tmp_path / "library.sqlite")
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "bytes"),
        store=store,
        local_resolver=table,
        disabled_model_locations=downloads.disabled_locations,
    )
    try:
        expected = [
            {
                "digest": digest,
                "name": "model.safetensors",
                "status": "missing",
                "sources": [],
                "fetchable": False,
                "disabledLocations": ["studio"],
                "detail": "Folder not enabled for this project",
            }
        ]
        for consent in (set(), {digest}):
            assert (
                asset_preflight(library, {digest: "model.safetensors"}, consent, {}, project="work")
                == expected
            )
        service.update_project("work", ["studio-checkpoint"], None)
        assert (
            asset_preflight(library, {digest: "model.safetensors"}, set(), {}, project="work") == []
        )
        assert (
            asset_preflight(library, {digest: "model.safetensors"}, set(), {}, project="other")
            == expected
        )
        duplicate = tmp_path / "enabled-copy"
        duplicate.mkdir()
        (duplicate / "same.safetensors").write_bytes(payload)
        table.add(MountDef("copy", duplicate, "read"))
        table.scan("copy")
        service.update_project("other", ["copy"], None)
        assert (
            asset_preflight(library, {digest: "model.safetensors"}, set(), {}, project="other")
            == []
        )
        (duplicate / "same.safetensors").unlink()
        assert (
            asset_preflight(library, {digest: "model.safetensors"}, set(), {}, project="other")
            == expected
        )
        (duplicate / "same.safetensors").write_bytes(payload)
        table.scan("copy")
        snapshot = table.ready_snapshot

        def rescan_after_snapshot():
            captured = snapshot()
            (duplicate / "same.safetensors").write_bytes(b"different model")
            table.scan("copy")
            return captured

        with monkeypatch.context() as patch:
            patch.setattr(table, "ready_snapshot", rescan_after_snapshot)
            assert downloads.disabled_locations(digest, "other") == ("studio",)
        table.remove("copy")
        with library.vault.writer(digest) as writer:
            writer.write(payload)
            writer.commit()
        assert (
            asset_preflight(library, {digest: "model.safetensors"}, set(), {}, project="other")
            == []
        )
        assert (root / "model.safetensors").read_bytes() == payload
    finally:
        store.close()


@pytest.mark.parametrize("visibility", ["opt-in", "default-on"])
def test_model_layout_edits_preserve_whole_location_project_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visibility: str
) -> None:
    from dinkster_assets import MountTable

    from dinkster.model_settings import ModelSettings
    from dinkster.mounts_api import MountService

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    location = {
        "id": "studio",
        "path": str(tmp_path / "models"),
        "mode": "read",
        "kind": "model/checkpoint",
        "visibility": visibility,
    }
    settings_path = tmp_path / "model-settings.json"

    def configure(kind: str) -> None:
        location["kind"] = kind
        settings_path.write_text(
            json.dumps({"version": 1, "downloadLocation": None, "locations": [location]})
        )

    configure("model/checkpoint")
    service = MountService(MountTable(), tmp_path / "mounts.toml", shared_models=True)
    service.refresh_models()
    service.update_project("enabled", ["studio-checkpoint"], None)
    service.update_project("disabled", [], None)
    for kind in ("", "model/vae", "model/checkpoint", ""):
        configure(kind)
        expected = {mount.id for mount in ModelSettings.load().mounts()}
        assert set(service.project_context("enabled").mounts) == expected
        assert service.project_context("disabled").mounts == ()
        assert set(service.project_context("untouched").mounts) == (
            expected if visibility == "default-on" else set()
        )
        restarted = MountService(MountTable(), tmp_path / "mounts.toml", shared_models=True)
        assert set(restarted.project_context("enabled").mounts) == expected
        assert restarted.project_context("disabled").mounts == ()


def test_model_group_refresh_preserves_similarly_named_config_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import MountDef, MountTable

    from dinkster.mounts_api import MountService

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    settings = {
        "version": 1,
        "downloadLocation": None,
        "locations": [
            {
                "id": "studio",
                "path": str(tmp_path / "models"),
                "kind": "model/checkpoint",
                "mode": "read",
                "visibility": "opt-in",
            }
        ],
    }
    path = tmp_path / "model-settings.json"
    path.write_text(json.dumps(settings))
    table = MountTable()
    table.add(MountDef("studio-vae", tmp_path / "unrelated", "read"))
    service = MountService(table, tmp_path / "mounts.toml", shared_models=True)
    service.refresh_models()
    service.update_project("model-only", ["studio-checkpoint"], None)
    service.update_project("config-only", ["studio-vae"], None)
    settings["downloadLocation"] = "studio"
    path.write_text(
        json.dumps({**settings, "locations": [{**settings["locations"][0], "mode": "readwrite"}]})
    )
    assert service.project_context("model-only").mounts == ("studio-checkpoint",)
    assert service.project_context("config-only").mounts == ("studio-vae",)


@pytest.mark.parametrize("sandboxed", [False, True])
@pytest.mark.parametrize("change", ["remove", "layout", "narrow", "visibility"])
def test_model_settings_preserve_fixed_project_authority_until_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandboxed: bool, change: str
) -> None:
    from dinkster_assets import MountTable
    from dinkster_assets.provenance import ProvenanceStore

    from dinkster.model_downloads import ModelDownloads, shared_model_mounts
    from dinkster.mounts_api import MountService
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import make_client

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    root = tmp_path / "models"
    root.mkdir()
    (root / "keep.safetensors").write_bytes(b"keep the original model")
    location = {
        "id": "studio",
        "path": str(root),
        "kind": "model/checkpoint",
        "mode": "read",
        "visibility": "default-on",
    }
    if change == "narrow":
        del location["kind"]
        (root / "checkpoints").mkdir()
        (root / "checkpoints" / "keep.safetensors").write_bytes(b"keep the original model")
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = {
        "id": "other",
        "path": str(other_root),
        "kind": "model/checkpoint",
        "mode": "read",
        "visibility": "default-on",
    }
    settings = {"version": 1, "downloadLocation": None, "locations": [location, other]}
    (tmp_path / "model-settings.json").write_text(json.dumps(settings))

    def start() -> MountService:
        table = MountTable()
        for mount in shared_model_mounts():
            table.add(mount, source="shared-models")
            if mount.path.is_dir():
                table.scan(mount.id)
        return MountService(
            table,
            tmp_path / "mounts.toml",
            allow_changes=True,
            restart_for_changes=sandboxed,
            shared_models=True,
        )

    service = start()
    service.refresh_models()
    service.update_project(
        "enabled",
        [mount_id for mount_id in service.mount_ids() if mount_id != "other-checkpoint"],
        None,
    )
    service.update_project("disabled", [], None)
    before = {
        project: service.project_context(project).mounts for project in ("enabled", "disabled")
    }
    if change == "remove":
        settings["locations"] = [other]
    elif change == "layout":
        del location["kind"]
    elif change == "narrow":
        location["kind"] = "model/checkpoint"
    else:
        location["visibility"] = "always-on"

    async def scenario() -> None:
        client = await make_client(service)
        try:
            current = await (await client.get("/api/models/settings")).json()
            response = await client.put(
                "/api/models/settings", json={"settings": settings, "revision": current["revision"]}
            )
            assert response.status == 200
            saved = await response.json()
            assert saved["settings"] == settings
            assert saved["restartRequired"] is sandboxed
            if change != "remove":
                assert saved["locations"][0]["active"] is (not sandboxed)
            for project in before:
                active = service.project_context(project).mounts
                if sandboxed:
                    assert active == before[project]
                elif change == "remove":
                    assert active == ()
                elif change == "visibility":
                    assert active == ("studio-checkpoint",)
            if sandboxed:
                for project in before:
                    response = await client.put(
                        f"/api/models/locations/other/projects/{project}", json={"enabled": True}
                    )
                    assert response.status == 409
                    assert "restart" in (await response.json())["error"]
                    response = await client.put(
                        f"/api/mounts/projects/{project}",
                        json={"mounts": ["other-checkpoint"], "outputMount": None},
                    )
                    assert response.status == 409
                    assert service.project_context(project).mounts == before[project]
                downloads = ModelDownloads(service, ProvenanceStore(tmp_path / "provenance.json"))
                digest = digest_bytes(b"keep the original model")
                assert downloads.disabled_locations(digest, "enabled") == ()
                assert downloads.disabled_locations(digest, "disabled") == ("studio",)
            if change != "layout":
                for project, expected in (("enabled", 200), ("disabled", 403)):
                    response = await client.get(
                        f"/api/mounts/studio-checkpoint/entries?project={project}"
                    )
                    assert response.status == (
                        expected
                        if sandboxed or change == "narrow"
                        else 404
                        if change == "remove"
                        else 200
                    )
        finally:
            await service.close()
            await client.close()

    run_scenario(scenario())
    restarted = start()
    if change == "remove":
        assert restarted.project_context("enabled").mounts == ()
    elif change == "visibility":
        assert restarted.project_context("disabled").mounts == ("studio-checkpoint",)
    else:
        enabled = restarted.project_context("enabled").mounts
        assert (len(enabled) > 1) if change == "layout" else enabled == ("studio-checkpoint",)
        assert restarted.project_context("disabled").mounts == ()
    restarted.update_project(
        "enabled", [*restarted.project_context("enabled").mounts, "other-checkpoint"], None
    )
    assert "other-checkpoint" in restarted.project_context("enabled").mounts
    assert (root / "keep.safetensors").read_bytes() == b"keep the original model"


@pytest.mark.parametrize("sandboxed", [False, True])
def test_model_management_shares_locations_without_deleting_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandboxed: bool
) -> None:
    from dinkster_assets import MountTable

    from dinkster.model_settings import ModelSettings
    from dinkster.mounts_api import MountService
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import make_client

    home = tmp_path / "settings"
    monkeypatch.setenv("DINKSTER_HOME", str(home))
    home.mkdir()
    (home / "model-settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "downloadLocation": None,
                "locations": [],
            }
        )
    )
    first = MountService(
        MountTable(),
        tmp_path / "first" / "mounts.toml",
        allow_changes=True,
        restart_for_changes=sandboxed,
        shared_models=True,
    )
    second = MountService(
        MountTable(),
        tmp_path / "second" / "mounts.toml",
        allow_changes=True,
        restart_for_changes=sandboxed,
        shared_models=True,
    )
    models = tmp_path / "existing-comfy" / "checkpoints"
    models.mkdir(parents=True)
    original = models / "existing.safetensors"
    original.write_bytes(b"existing model stays in place")

    async def scenario() -> None:
        a, b = await make_client(first), await make_client(second)
        try:
            empty = await (await a.get("/api/models/settings?project=work")).json()
            settings = {
                "version": 1,
                "downloadLocation": None,
                "locations": [
                    {
                        "id": "existing",
                        "path": str(models),
                        "kind": "model/checkpoint",
                        "mode": "read",
                        "visibility": "opt-in",
                    }
                ],
            }
            response = await a.put(
                "/api/models/settings?project=work",
                json={
                    "settings": settings,
                    "revision": empty["revision"],
                },
            )
            assert response.status == 200
            added = await response.json()
            assert added["settings"] == settings
            assert added["locations"][0]["mounts"] == ["existing-checkpoint"]
            assert added["locations"][0]["enabled"] is False
            assert added["locations"][0]["active"] is (not sandboxed)
            assert added["restartRequired"] is sandboxed
            assert not (models / "checkpoints").exists()
            for client in (a, b):
                response = await client.get("/api/models/settings?project=work")
                assert response.status == 200
                state = await response.json()
                assert state["settings"] == settings
                assert state["locations"][0]["enabled"] is False
            if not sandboxed:
                first.table.scan("existing-checkpoint")
                second.table.scan("existing-checkpoint")
                for route in ("entries?q=existing", "list?path="):
                    response = await a.get(f"/api/mounts/existing-checkpoint/{route}&project=work")
                    assert response.status == 403
                    assert (await response.json())["error"] == "folder-not-enabled"
            response = await a.put(
                "/api/models/locations/existing/projects/work", json={"enabled": True}
            )
            assert response.status == (409 if sandboxed else 200)
            if sandboxed:
                assert "restart" in (await response.json())["error"]
            else:
                assert (await response.json())["locations"][0]["enabled"] is True
            assert (await (await b.get("/api/models/settings?project=work")).json())["locations"][
                0
            ]["enabled"] is False
            if not sandboxed:
                for route in ("entries?q=existing", "list?path="):
                    response = await a.get(f"/api/mounts/existing-checkpoint/{route}&project=work")
                    assert response.status == 200
                    assert [row["name"] for row in (await response.json())["entries"]] == [
                        "existing.safetensors"
                    ]
                    for client, project in ((a, "other"), (b, "work")):
                        response = await client.get(
                            f"/api/mounts/existing-checkpoint/{route}&project={project}"
                        )
                        assert response.status == 403
                response = await a.put(
                    "/api/models/locations/existing/projects/work", json={"enabled": False}
                )
                assert response.status == 200
                response = await a.get("/api/mounts/existing-checkpoint/entries?project=work")
                assert response.status == 403
            # A stale editor must not create its requested folder or overwrite the newer settings.
            stale = {
                "version": 1,
                "downloadLocation": "late",
                "locations": [
                    {
                        "id": "late",
                        "path": str(tmp_path / "must-not-exist"),
                        "mode": "readwrite",
                    }
                ],
            }
            response = await b.put(
                "/api/models/settings", json={"settings": stale, "revision": empty["revision"]}
            )
            assert response.status == 409
            assert not (tmp_path / "must-not-exist").exists()
            assert ModelSettings.load().to_wire() == settings
            response = await a.put(
                "/api/models/settings",
                json={
                    "settings": {"version": 1, "downloadLocation": None, "locations": []},
                    "revision": added["revision"],
                },
            )
            assert response.status == 200
            assert (await response.json())["settings"]["locations"] == []
            assert original.read_bytes() == b"existing model stays in place"
            assert ModelSettings.load().locations == ()
            for client in (a, b):
                response = await client.get("/api/models/settings?project=work")
                assert response.status == 200
                assert (await response.json())["locations"] == []
        finally:
            for service in (first, second):
                await service.close()
            for client in (a, b):
                await client.close()

    run_scenario(scenario())


def test_provider_credentials_are_private_separate_and_forgettable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import stat

    from dinkster.provider_credentials import ProviderCredentials, ProviderCredentialsError

    home = tmp_path / "engine-home"
    credentials = ProviderCredentials(home)
    assert credentials.token() is None
    assert not credentials.path.exists()
    credentials.set_token("synthetic-provider-store-canary")
    reopened = ProviderCredentials(home)
    assert reopened.token() == "synthetic-provider-store-canary"
    assert reopened.token() == "synthetic-provider-store-canary"
    import subprocess
    import sys

    def fresh_process_snapshot() -> None:
        result = subprocess.run(
            [sys.executable, "-", str(home)],
            input=(
                "import sys\n"
                "from pathlib import Path\n"
                "from dinkster.provider_credentials import ProviderCredentials\n"
                "store = ProviderCredentials(Path(sys.argv[1]))\n"
                "assert store.snapshot() == " + repr(reopened.snapshot()) + "\n"
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    fresh_process_snapshot()
    if os.name != "nt":
        assert stat.S_IMODE(credentials.path.stat().st_mode) == 0o600
    assert not list(home.glob(".credentials-*"))
    assert not (home / "model-settings.json").exists()
    for invalid in ("", "with whitespace", "header\r\ninjection", "x" * 8193):
        with pytest.raises(ProviderCredentialsError):
            credentials.set_token(invalid)
    assert reopened.token() == "synthetic-provider-store-canary"
    credentials.set_token(None)
    assert reopened.token() is None
    assert not credentials.path.exists()
    assert all("synthetic-" not in path.read_text() for path in home.glob("*.json"))
    fresh_process_snapshot()
    _, revoked_revision = reopened.snapshot()
    credentials.set_token("synthetic-provider-store-canary")
    credentials.set_token(None)
    assert reopened.snapshot()[0] is None
    assert reopened.snapshot()[1] != revoked_revision
    credentials.set_token("synthetic-replacement")
    before_failure = reopened.revision()
    fresh_process_snapshot()
    unlink = Path.unlink

    def deny_removal(path: Path, missing_ok: bool = False) -> None:
        if path == credentials.path:
            raise PermissionError("removal denied")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", deny_removal)
        with pytest.raises(ProviderCredentialsError, match="Cannot forget"):
            credentials.set_token(None)
    assert credentials.path.exists()
    assert reopened.token() is None
    assert reopened.revision() != before_failure
    fresh_process_snapshot()
    with pytest.raises(ProviderCredentialsError, match="changed"):
        reopened.set_token("synthetic-stale-refresh", expected_revision=before_failure)
    credentials.set_token(None)
    assert not credentials.path.exists()
    fresh_process_snapshot()
    credentials.set_token("synthetic-new-sign-in")
    assert reopened.token() == "synthetic-new-sign-in"
    fresh_process_snapshot()
    credentials.set_token(None)

    def deny_revocation_removal(path: Path, missing_ok: bool = False) -> None:
        if path.name == "credentials-revoked.json":
            raise PermissionError("revocation removal denied")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", deny_revocation_removal)
        with pytest.raises(ProviderCredentialsError, match="Cannot save"):
            credentials.set_token("synthetic-uncommitted-sign-in")
    assert reopened.token() is None
    fresh_process_snapshot()
    credentials.set_token("synthetic-retry-sign-in")
    assert reopened.token() == "synthetic-retry-sign-in"
    fresh_process_snapshot()


@pytest.mark.parametrize("provider", ["huggingface", "civitai"])
@pytest.mark.parametrize("legacy", [False, True])
def test_provider_credentials_isolate_revisions_and_durable_revocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, legacy: bool
) -> None:
    import os
    import subprocess
    import sys

    from dinkster.provider_credentials import ProviderCredentials, ProviderCredentialsError

    credentials = ProviderCredentials(tmp_path)
    credentials.set_token("synthetic-hf")
    if legacy:
        credentials.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "revision": "legacy-hf-revision",
                    "providers": {
                        "huggingface": {"access_token": "synthetic-hf", "method": "token"}
                    },
                }
            )
        )
        assert credentials.revision() == "legacy-hf-revision"
    hf_snapshot = credentials.snapshot()
    empty_civitai = credentials.snapshot("civitai")
    credentials.set_token(
        "synthetic-civitai", provider="civitai", expected_revision=empty_civitai[1]
    )
    assert credentials.snapshot() == hf_snapshot
    other = "civitai" if provider == "huggingface" else "huggingface"
    other_snapshot = credentials.snapshot(other)
    _, before = credentials.snapshot(provider)
    replace = os.replace

    def deny_bearer_replace(source: str, destination: str) -> None:
        if Path(destination) == credentials.path:
            raise PermissionError("bearer replacement denied")
        replace(source, destination)

    with monkeypatch.context() as context:
        context.setattr(os, "replace", deny_bearer_replace)
        with pytest.raises(ProviderCredentialsError, match="Cannot forget"):
            credentials.set_token(None, provider=provider)
    reopened = ProviderCredentials(tmp_path)
    assert reopened.snapshot(provider)[0] is None
    assert reopened.revision(provider) != before
    assert reopened.snapshot(other) == other_snapshot
    assert "synthetic-" not in (tmp_path / "credentials-revoked.json").read_text()
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path)],
        input=(
            "import sys\n"
            "from pathlib import Path\n"
            "from dinkster.provider_credentials import ProviderCredentials\n"
            "store = ProviderCredentials(Path(sys.argv[1]))\n"
            f"assert store.snapshot({provider!r}) == {reopened.snapshot(provider)!r}\n"
            f"assert store.snapshot({other!r}) == {other_snapshot!r}\n"
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(ProviderCredentialsError, match="changed"):
        reopened.set_token("synthetic-stale", provider=provider, expected_revision=before)
    reopened.set_token("synthetic-replacement", provider=provider)
    assert credentials.token(provider) == "synthetic-replacement"
    assert credentials.snapshot(other) == other_snapshot
    credentials.set_token(None, provider=provider)
    assert credentials.token(provider) is None
    assert credentials.snapshot(other) == other_snapshot
    credentials.set_token(None, provider=other)
    assert not credentials.path.exists()
    assert all("synthetic-" not in path.read_text() for path in tmp_path.glob("*.json"))


@pytest.mark.parametrize("provider", ["huggingface", "civitai"])
@pytest.mark.parametrize("restore", [False, True])
def test_shared_credential_revocation_stops_stream_before_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore: bool, provider: str
) -> None:
    from itertools import count
    from types import SimpleNamespace

    from dinkster_assets import MountTable, civitai, huggingface
    from dinkster_assets.civitai import CivitaiFile
    from dinkster_assets.fetch import FetchAborted
    from dinkster_assets.huggingface import HuggingFaceFile
    from dinkster_assets.provenance import ProvenanceStore

    from dinkster import model_downloads
    from dinkster.model_downloads import ModelDownloads, shared_model_mounts
    from dinkster.mounts_api import MountService
    from dinkster.provider_credentials import ProviderCredentials

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "home"))
    table = MountTable()
    for mount in shared_model_mounts():
        table.add(mount, source="shared-models")
    downloads = ModelDownloads(MountService(table), ProvenanceStore(tmp_path / "provenance.json"))
    token = "synthetic-stream-revocation-canary"
    downloads.credentials.set_token(token, provider=provider)
    other = ProviderCredentials(downloads.credentials.path.parent)
    payload = b"bounded-stream" * (3 * 1024 * 1024)
    stream = io.BytesIO(payload)
    selected = HuggingFaceFile(
        repo="acme/model",
        revision="c" * 40,
        file="model.safetensors",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        license="unknown",
    )

    if provider == "civitai":
        selected = CivitaiFile(
            repo="19",
            revision="71",
            file="model.safetensors",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
            license="unknown",
            url="https://civitai.com/api/download/models/71?fileId=83",
        )

    @contextmanager
    def respond(_url: str, **credentials: str | None):
        assert credentials == {f"{provider}_token": token}
        yield stream

    consumed: list[int] = []

    def revoke(_status: str, received: int) -> None:
        if received and not consumed:
            consumed.append(received)
            other.set_token(None, provider=provider)
            if restore:
                other.set_token(token, provider=provider)

    ticks = count(1)
    monkeypatch.setattr(model_downloads, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(
        huggingface if provider == "huggingface" else civitai, "open_model_response", respond
    )
    target = downloads.target("model/checkpoint", "work")
    assert target is not None
    with pytest.raises(FetchAborted):
        downloads.download_selected(selected, target, "work", lambda *_args: None, progress=revoke)
    assert consumed and consumed[0] <= stream.tell() < len(payload)
    assert not list(target.path.rglob("*.safetensors"))
    assert not (tmp_path / "provenance.json").exists()


@pytest.mark.parametrize("granted_scope", ["5", "1", "4", "13", None, 5, "05"])
def test_civitai_refresh_rotates_only_read_only_model_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, granted_scope: object
) -> None:
    from types import SimpleNamespace

    from dinkster import provider_credentials as module
    from dinkster.provider_credentials import ProviderCredentials, ProviderCredentialsError

    now = 100.0
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now))
    credentials = ProviderCredentials(tmp_path)
    credentials.set_token("synthetic-hf")
    credentials.set_token(
        "synthetic-civitai",
        provider="civitai",
        expires_at=110,
        refresh_token="synthetic-refresh",
        client_id="synthetic-public-client",
    )
    before = credentials.revision("civitai")
    now = 110
    exchanges: list[str] = []

    def refresh(endpoint: str, fields: dict[str, str], *, provider: str) -> dict[str, object]:
        assert endpoint == "token" and provider == "civitai"
        assert fields == {
            "grant_type": "refresh_token",
            "refresh_token": "synthetic-refresh",
            "client_id": "synthetic-public-client",
            "scope": "5",
        }
        exchanges.append(provider)
        ProviderCredentials(tmp_path).set_token("synthetic-hf-replacement")
        return {
            "access_token": "synthetic-renewed",
            "token_type": "Bearer",
            "expires_in": 23,
            "refresh_token": "synthetic-rotated",
            "scope": granted_scope,
            "id_token": "synthetic-discarded-id",
        }

    monkeypatch.setattr(module, "post_provider_form", refresh)
    if granted_scope == "5":
        assert credentials.token("civitai") == "synthetic-renewed"
        persisted = json.loads(credentials.path.read_text())["providers"]["civitai"]
        assert persisted["refresh_token"] == "synthetic-rotated"
        assert (
            persisted["client_id"] == "synthetic-public-client" and persisted["expires_at"] == 133
        )
        assert "synthetic-discarded-id" not in credentials.path.read_text()
    else:
        with pytest.raises(ProviderCredentialsError):
            credentials.token("civitai")
        with pytest.raises(ProviderCredentialsError):
            credentials.token("civitai")
        assert "synthetic-renewed" not in credentials.path.read_text()
        assert "synthetic-rotated" not in credentials.path.read_text()
    assert credentials.token() == "synthetic-hf-replacement"
    assert credentials.revision("civitai") != before
    assert exchanges == ["civitai"]


def test_provider_oauth_expiry_refresh_and_revision_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from dinkster import provider_credentials as module
    from dinkster.provider_credentials import (
        ProviderCredentials,
        ProviderCredentialsError,
        ProviderCredentialsExpired,
    )

    now = 1000.0
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now))
    credentials = ProviderCredentials(tmp_path)
    revision = credentials.revision()
    credentials.set_token("synthetic-expiring", expires_at=1010, expected_revision=revision)
    assert credentials.token() == "synthetic-expiring"
    now = 1010
    with pytest.raises(ProviderCredentialsExpired):
        credentials.token()
    with pytest.raises(ProviderCredentialsError):
        credentials.set_token("synthetic-stale", expected_revision=revision)
    credentials.set_token(
        "synthetic-refreshable", expires_at=1020, refresh_token="synthetic-refresh"
    )
    before = credentials.revision()
    now = 1020

    def refresh(
        endpoint: str, fields: dict[str, str], *, provider: str = "huggingface"
    ) -> dict[str, object]:
        assert endpoint == "token" and provider == "huggingface"
        assert fields["refresh_token"] == "synthetic-refresh"
        assert fields["grant_type"] == "refresh_token"
        ProviderCredentials(tmp_path).set_token("synthetic-civitai", provider="civitai")
        return {
            "access_token": "synthetic-renewed",
            "token_type": "Bearer",
            "expires_in": 23,
            "refresh_token": "synthetic-rotated",
            "id_token": "synthetic-discarded-id",
        }

    monkeypatch.setattr(module, "post_provider_form", refresh)
    assert credentials.token() == "synthetic-renewed"
    assert credentials.token("civitai") == "synthetic-civitai"
    assert credentials.revision() != before
    persisted = json.loads(credentials.path.read_text())
    assert persisted["providers"]["huggingface"]["expires_at"] == 1043
    assert "synthetic-discarded-id" not in credentials.path.read_text()
    now = 1043
    monkeypatch.setattr(
        module, "post_provider_form", lambda *_args, **_kwargs: {"error": "invalid_grant"}
    )
    with pytest.raises(ProviderCredentialsExpired):
        credentials.token()
    monkeypatch.setattr(
        module,
        "post_provider_form",
        lambda *_args, **_kwargs: pytest.fail("refused refresh must not repeat"),
    )
    with pytest.raises(ProviderCredentialsExpired):
        credentials.token()


@pytest.mark.parametrize("provider", ["huggingface", "civitai"])
@pytest.mark.parametrize("deny_removal", [False, True])
def test_provider_revoke_wins_while_refresh_response_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deny_removal: bool, provider: str
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace

    from dinkster import provider_credentials as module
    from dinkster.provider_credentials import ProviderCredentials, ProviderCredentialsError

    now = 100.0
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now))
    credentials = ProviderCredentials(tmp_path)
    credentials.set_token(
        "synthetic-expiring",
        provider=provider,
        expires_at=110,
        refresh_token="synthetic-refresh",
        client_id="synthetic-public-client" if provider == "civitai" else None,
    )
    now = 110
    entered, release = Event(), Event()

    def refresh(*_args: object, **_kwargs: object) -> dict[str, object]:
        entered.set()
        assert release.wait(10)
        return {
            "access_token": "synthetic-must-not-restore",
            "token_type": "bearer",
            "expires_in": 100,
            "refresh_token": "synthetic-must-not-persist",
            "scope": "5",
        }

    monkeypatch.setattr(module, "post_provider_form", refresh)
    with ThreadPoolExecutor(max_workers=1) as pool:
        refreshing = pool.submit(credentials.token, provider)
        try:
            assert entered.wait(10)
            before = credentials.revision(provider)
            unlink = Path.unlink

            def remove(path: Path, missing_ok: bool = False) -> None:
                if deny_removal and path == credentials.path:
                    raise PermissionError("removal denied")
                unlink(path, missing_ok=missing_ok)

            with monkeypatch.context() as context:
                context.setattr(Path, "unlink", remove)
                if deny_removal:
                    with pytest.raises(ProviderCredentialsError, match="Cannot forget"):
                        ProviderCredentials(tmp_path).set_token(None, provider=provider)
                else:
                    ProviderCredentials(tmp_path).set_token(None, provider=provider)
            assert credentials.token(provider) is None
            assert credentials.revision(provider) != before
        finally:
            release.set()
        with pytest.raises(ProviderCredentialsError, match="changed"):
            refreshing.result(timeout=10)
    assert credentials.token(provider) is None
    assert credentials.path.exists() == deny_removal


@pytest.mark.parametrize(
    "verification_uri",
    ["https://huggingface.co/oauth/authorize_device", "https://hf.co/oauth/device"],
)
def test_provider_oauth_pkce_device_and_cancellation_are_owner_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verification_uri: str
) -> None:
    import asyncio
    import base64
    from threading import Event
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlsplit

    import aiohttp
    from aiohttp.test_utils import TestClient, TestServer
    from dinkster_server import create_app
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    from dinkster import provider_sign_in as module
    from dinkster.provider_credentials import HUGGINGFACE_CLIENT_ID, ProviderCredentials
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import SCHEMAS, make_engine

    credentials = ProviderCredentials(tmp_path)
    exchanges: list[tuple[str, dict[str, str]]] = []
    changed: list[bool] = []
    now = 10.0
    outcome = "authorization_pending"
    entered, release = Event(), Event()

    def exchange(endpoint: str, fields: dict[str, str]) -> dict[str, object]:
        exchanges.append((endpoint, fields))
        assert fields["client_id"] == HUGGINGFACE_CLIENT_ID
        if endpoint == "device":
            assert fields["scope"] == "openid profile gated-repos"
            return {
                "device_code": "synthetic-private-device",
                "user_code": "ABCD-EFGH",
                "verification_uri": verification_uri,
                "expires_in": 90,
                "interval": 3,
            }
        if "device_code" in fields and outcome == "broken":
            from dinkster_assets import AssetError

            raise AssetError("Hugging Face sign-in response interrupted; sign in again")
        if "device_code" in fields and outcome == "blocked":
            entered.set()
            assert release.wait(10)
        if "device_code" in fields and outcome not in {"success", "blocked"}:
            return {"error": outcome, "error_description": "synthetic-private-error"}
        return {
            "access_token": "synthetic-oauth-access",
            "token_type": "bearer",
            "expires_in": 120,
            "refresh_token": "synthetic-oauth-refresh",
            "id_token": "synthetic-oauth-id",
        }

    monkeypatch.setattr(module, "post_provider_form", exchange)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now))
    auth = StaticBearerAuthenticator(
        {
            name: Principal(name, {"local": frozenset({"settings:write"})})
            for name in ("owner", "other")
        }
    )

    async def scenario() -> None:
        nonlocal now, outcome
        app = create_app(make_engine, SCHEMAS, authenticator=auth)
        module.add_provider_sign_in_routes(app, credentials, lambda: changed.append(True))
        client = TestClient(TestServer(app), headers={"Authorization": "Bearer owner"})
        await client.start_server()
        path = "/api/models/providers/huggingface/sign-in"
        try:
            remote = await client.post(
                path + "?scope=local", json={"method": "pkce"}, headers={"Host": "remote.example"}
            )
            assert remote.status == 421
            invalid = await client.post(path + "?scope=local", json={"method": []})
            assert invalid.status == 400
            started = await client.post(path + "?scope=local", json={"method": "pkce"})
            assert started.status == 201
            flow = await started.json()
            query = parse_qs(urlsplit(flow["authorizationUrl"]).query)
            assert query["scope"] == ["openid profile gated-repos"]
            assert query["code_challenge_method"] == ["S256"]
            poll = path + "/" + flow["flowId"] + "?scope=local"
            foreign = await client.post(poll, headers={"Authorization": "Bearer other"})
            assert foreign.status == 404
            async with aiohttp.ClientSession() as browser:
                wrong = await browser.get(
                    query["redirect_uri"][0], params={"state": "wrong", "code": "synthetic-code"}
                )
                assert wrong.status == 400
                callback = await browser.get(
                    query["redirect_uri"][0],
                    params={"state": query["state"][0], "code": "synthetic-code"},
                )
                assert callback.status == 200
                replay = await browser.get(
                    query["redirect_uri"][0],
                    params={"state": query["state"][0], "code": "synthetic-code"},
                )
                assert replay.status == 400
            assert not credentials.path.exists()
            finished = await client.post(poll)
            assert await finished.json() == {"status": "signed-in", "provider": "huggingface"}
            fields = exchanges[-1][1]
            challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(fields["code_verifier"].encode()).digest())
                .rstrip(b"=")
                .decode()
            )
            assert query["code_challenge"] == [challenge]
            assert fields["redirect_uri"] == query["redirect_uri"][0]
            assert fields["code"] == "synthetic-code"
            assert credentials.token() == "synthetic-oauth-access" and changed == [True]
            assert "synthetic-oauth-id" not in credentials.path.read_text()
            assert (await client.post(poll)).status == 404
            device = await client.post(path + "?scope=local", json={"method": "device"})
            device_flow = await device.json()
            assert device.status == 201 and device_flow["userCode"] == "ABCD-EFGH"
            assert "synthetic-private-device" not in json.dumps(device_flow)
            device_poll = path + "/" + device_flow["flowId"] + "?scope=local"
            count = len(exchanges)
            assert (await (await client.post(device_poll)).json())["status"] == "pending"
            assert len(exchanges) == count
            now += 3
            assert (await (await client.post(device_poll)).json())["interval"] == 3
            outcome = "slow_down"
            now += 3
            assert (await (await client.post(device_poll)).json())["interval"] == 8
            outcome = "success"
            now += 8
            assert (await (await client.post(device_poll)).json())["status"] == "signed-in"
            cancelled = await client.post(path + "?scope=local", json={"method": "device"})
            cancel_poll = path + "/" + (await cancelled.json())["flowId"] + "?scope=local"
            assert (await (await client.delete(cancel_poll)).json())["status"] == "cancelled"
            assert (await client.post(cancel_poll)).status == 404
            stale = await client.post(path + "?scope=local", json={"method": "device"})
            stale_poll = path + "/" + (await stale.json())["flowId"] + "?scope=local"
            credentials.set_token(None)
            assert (await client.post(stale_poll)).status == 409
            assert credentials.token() is None
            assert "synthetic-oauth-access" not in json.dumps(device_flow)
            broken = await client.post(path + "?scope=local", json={"method": "device"})
            broken_poll = path + "/" + (await broken.json())["flowId"] + "?scope=local"
            outcome = "broken"
            now += 3
            refused = await client.post(broken_poll)
            assert refused.status == 409
            assert await refused.json() == {"error": "provider-sign-in-unavailable"}
            count = len(exchanges)
            assert (await client.post(broken_poll)).status == 404
            assert len(exchanges) == count
            blocked = await client.post(path + "?scope=local", json={"method": "device"})
            blocked_poll = path + "/" + (await blocked.json())["flowId"] + "?scope=local"
            now += 3
            outcome = "blocked"
            polling = asyncio.ensure_future(client.post(blocked_poll))
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                assert (await (await client.delete(blocked_poll)).json())["status"] == "cancelled"
            finally:
                release.set()
            assert (await polling).status == 409
            assert credentials.token() is None
        finally:
            await client.close()

    run_scenario(scenario())


@pytest.fixture
def civitai_callback_port(tmp_path_factory: pytest.TempPathFactory) -> Generator[None]:
    from filelock import FileLock

    # Every xdist worker shares the one registered loopback port.
    shared = tmp_path_factory.getbasetemp().parent
    with FileLock(str(shared / "civitai-callback-port.lock"), timeout=60):
        yield


@pytest.mark.usefixtures("civitai_callback_port")
@pytest.mark.parametrize("client_id", ["", "invalid public client", "synthetic-public-client"])
def test_civitai_sign_in_requires_registered_client_and_available_callback(
    tmp_path: Path, client_id: str
) -> None:
    from urllib.parse import parse_qs, urlsplit

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from dinkster_server import create_app
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    from dinkster.provider_credentials import ProviderCredentials
    from dinkster.provider_sign_in import add_provider_sign_in_routes
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import SCHEMAS, make_engine

    async def scenario() -> None:
        credentials = ProviderCredentials(tmp_path)
        auth = StaticBearerAuthenticator(
            {"owner": Principal("owner", {"local": frozenset({"settings:write"})})}
        )
        app = create_app(make_engine, SCHEMAS, authenticator=auth)
        add_provider_sign_in_routes(
            app, credentials, lambda: None, provider="civitai", client_id=client_id
        )
        client = TestClient(TestServer(app), headers={"Authorization": "Bearer owner"})
        occupied = TestServer(web.Application(), host="127.0.0.1", port=15493)
        await client.start_server()
        await occupied.start_server()
        path = "/api/models/providers/civitai/sign-in?scope=local"
        try:
            refused = await client.post(path, json={"method": "pkce"})
            assert refused.status == 409
            assert await refused.json() == {"error": "provider-sign-in-unavailable"}
            await occupied.close()
            retry = await client.post(path, json={"method": "pkce"})
            if client_id == "synthetic-public-client":
                assert retry.status == 201
                query = parse_qs(urlsplit((await retry.json())["authorizationUrl"]).query)
                assert query["redirect_uri"] == ["http://localhost:15493/oauth/callback/civitai"]
            else:
                assert retry.status == 409
                # Invalid configuration must not reserve the callback port.
                probe = TestServer(web.Application(), host="127.0.0.1", port=15493)
                try:
                    await probe.start_server()
                finally:
                    await probe.close()
            assert credentials.token("civitai") is None
        finally:
            await occupied.close()
            await client.close()

    run_scenario(scenario())


@pytest.mark.usefixtures("civitai_callback_port")
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "cancel",
        "replaced",
        "broader-grant",
        "refused",
        "transport-error",
        "provider-error",
        "unknown-provider-error",
    ],
)
def test_civitai_pkce_callback_is_one_use_and_provider_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, outcome: str
) -> None:
    import base64
    from urllib.parse import parse_qs, urlsplit

    import aiohttp
    from aiohttp.test_utils import TestClient, TestServer
    from dinkster_server import create_app
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    from dinkster import provider_sign_in as module
    from dinkster.provider_credentials import ProviderCredentials
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import SCHEMAS, make_engine

    credentials = ProviderCredentials(tmp_path)
    credentials.set_token("synthetic-hf-before")
    caplog.set_level("INFO", logger=module.__name__)
    exchanges: list[dict[str, str]] = []
    changed: list[bool] = []

    def exchange(
        endpoint: str, fields: dict[str, str], *, provider: str = "huggingface"
    ) -> dict[str, object]:
        assert provider == "civitai" and endpoint == "token"
        assert fields["client_id"] == "synthetic-public-client"
        assert fields["grant_type"] == "authorization_code"
        assert "client_secret" not in fields
        exchanges.append(fields)
        if outcome == "transport-error":
            raise OSError("synthetic-private-exception-canary")
        if outcome in {"provider-error", "unknown-provider-error"}:
            return {
                "error": "invalid_grant"
                if outcome == "provider-error"
                else "synthetic-private-error-canary",
                "error_description": "synthetic-private-description-canary",
            }
        return {
            "access_token": "synthetic-civitai-access",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "synthetic-civitai-refresh",
            "scope": "7" if outcome == "broader-grant" else "5",
        }

    monkeypatch.setattr(module, "post_provider_form", exchange)
    auth = StaticBearerAuthenticator(
        {
            name: Principal(name, {"local": frozenset({"settings:write"})})
            for name in ("owner", "other")
        }
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, authenticator=auth)
        module.add_provider_sign_in_routes(app, credentials, lambda: None)
        module.add_provider_sign_in_routes(
            app,
            credentials,
            lambda: changed.append(True),
            provider="civitai",
            client_id="synthetic-public-client",
        )
        client = TestClient(TestServer(app), headers={"Authorization": "Bearer owner"})
        await client.start_server()
        path = "/api/models/providers/civitai/sign-in?scope=local"
        try:
            assert (await client.post(path, json={"method": "device"})).status == 400
            response = await client.post(path, json={"method": "pkce"})
            assert response.status == 201
            flow = await response.json()
            authorize = urlsplit(flow["authorizationUrl"])
            assert authorize.scheme == "https" and authorize.netloc == "auth.civitai.com"
            assert authorize.path == "/api/auth/oauth/authorize"
            query = parse_qs(authorize.query)
            assert query["scope"] == ["4"] and query["code_challenge_method"] == ["S256"]
            assert query["redirect_uri"] == ["http://localhost:15493/oauth/callback/civitai"]
            assert query["client_id"] == ["synthetic-public-client"]
            poll = "/api/models/providers/civitai/sign-in/" + flow["flowId"] + "?scope=local"
            assert (
                await client.post(poll, headers={"Authorization": "Bearer other"})
            ).status == 404
            assert (await (await client.post(poll)).json()) == {"status": "pending"}
            async with aiohttp.ClientSession() as browser:
                callback = query["redirect_uri"][0]
                params = {"state": query["state"][0], "code": "synthetic-code"}
                wrong = await browser.get(callback, params={**params, "state": "wrong"})
                assert wrong.status == 400
                duplicate = await browser.get(
                    callback,
                    params=[
                        ("state", params["state"]),
                        ("state", params["state"]),
                        ("code", params["code"]),
                    ],
                )
                assert duplicate.status == 400
                if outcome == "refused":
                    refused = {"state": params["state"], "error": "access_denied"}
                    for malformed in (
                        {**refused, "state": "wrong"},
                        {**refused, "code": params["code"]},
                        {**refused, "error": ""},
                        [
                            ("state", params["state"]),
                            ("error", "access_denied"),
                            ("error", "other"),
                        ],
                    ):
                        assert (await browser.get(callback, params=malformed)).status == 400
                    assert (await (await client.post(poll)).json()) == {"status": "pending"}
                    response = await browser.get(callback, params=refused)
                    assert response.status == 200
                    assert "access_denied" not in await response.text()
                    # Another owner can use the fixed port before the refused owner polls.
                    other = await client.post(
                        path,
                        json={"method": "pkce"},
                        headers={"Authorization": "Bearer other"},
                    )
                    assert other.status == 201
                    assert (await browser.get(callback, params=params)).status == 400
                    finished = await client.post(poll)
                    assert finished.status == 409
                    assert await finished.json() == {"error": "provider-sign-in-refused"}
                    assert (await client.post(poll)).status == 404
                    assert credentials.token("civitai") is None and not changed and not exchanges
                    assert credentials.token() == "synthetic-hf-before"
                    return
                assert (await browser.get(callback, params=params)).status == 200
                assert (await browser.get(callback, params=params)).status == 400
            assert credentials.token("civitai") is None and not exchanges
            credentials.set_token("synthetic-hf-after")
            if outcome == "cancel":
                assert (await client.delete(poll)).status == 200
                assert (await client.post(poll)).status == 404
            elif outcome == "replaced":
                credentials.set_token(None, provider="civitai")
                assert (await (await client.post(poll)).json()) == {
                    "error": "provider-credentials-changed"
                }
            else:
                finished = await client.post(poll)
                assert len(exchanges) == 1
                fields = exchanges[0]
                challenge = (
                    base64.urlsafe_b64encode(
                        hashlib.sha256(fields["code_verifier"].encode("ascii")).digest()
                    )
                    .rstrip(b"=")
                    .decode("ascii")
                )
                assert query["code_challenge"] == [challenge]
                assert fields["redirect_uri"] == query["redirect_uri"][0]
                assert fields["code"] == "synthetic-code"
                if outcome == "success":
                    assert await finished.json() == {"status": "signed-in", "provider": "civitai"}
                    assert credentials.token("civitai") == "synthetic-civitai-access"
                    assert changed == [True]
                    assert "synthetic-public-client" in credentials.path.read_text()
                else:
                    assert finished.status == 409
                    assert credentials.token("civitai") is None and not changed
                assert (await client.post(poll)).status == 404
                assert len(exchanges) == 1
            assert credentials.token() == "synthetic-hf-after"
            if outcome != "success":
                assert credentials.token("civitai") is None and not changed
        finally:
            await client.close()

    run_scenario(scenario())
    if outcome == "success":
        assert "outcome=signed-in" in caplog.text
    elif outcome == "transport-error":
        assert "stage=token-exchange outcome=failed exception=OSError" in caplog.text
    elif outcome == "broader-grant":
        assert "stage=credential-validation outcome=failed" in caplog.text
    elif outcome in {"provider-error", "unknown-provider-error"}:
        reason = "invalid_grant" if outcome == "provider-error" else "provider-error"
        assert f"stage=token-exchange outcome=refused reason={reason}" in caplog.text
    for private in (
        "synthetic-private",
        "synthetic-code",
        "synthetic-civitai-access",
        "synthetic-civitai-refresh",
    ):
        assert private not in caplog.text
    for fields in exchanges:
        assert fields["code_verifier"] not in caplog.text


@pytest.mark.parametrize("filename", ["credentials.json", "credentials-revoked.json"])
def test_provider_credentials_refuse_symlink_destination(tmp_path: Path, filename: str) -> None:
    from dinkster.provider_credentials import ProviderCredentials, ProviderCredentialsError

    destination = tmp_path / "not-credentials.json"
    destination.write_text("must not be changed")
    credentials = ProviderCredentials(tmp_path)
    symlink_or_skip(tmp_path / filename, destination)
    with pytest.raises(ProviderCredentialsError):
        credentials.set_token("synthetic-unused")
    assert destination.read_text() == "must not be changed"


@pytest.mark.usefixtures("civitai_callback_port")
@pytest.mark.parametrize(
    "client_id", [None, "", "invalid public client", "synthetic-public-client"]
)
def test_provider_credential_routes_keep_secrets_out_of_plans_and_revoke_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_id: str | None
) -> None:
    from aiohttp.test_utils import TestClient, TestServer
    from dinkster_assets import MountTable, fetch
    from dinkster_assets.provenance import ProvenanceStore
    from dinkster_server import create_app
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    from dinkster.model_downloads import ModelDownloads, shared_model_mounts
    from dinkster.model_downloads_api import add_model_download_routes
    from dinkster.mounts_api import MountService
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import SCHEMAS, make_engine

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    if client_id is None:
        monkeypatch.delenv("DINKSTER_CIVITAI_CLIENT_ID", raising=False)
    else:
        monkeypatch.setenv("DINKSTER_CIVITAI_CLIENT_ID", client_id)
    pkce_available = client_id is None or client_id == "synthetic-public-client"
    token = "synthetic-route-provider-canary"
    civitai_token = "synthetic-civitai-route-canary"
    entitled = False
    table = MountTable()
    shared = shared_model_mounts()
    for mount in shared:
        table.add(mount, source="shared-models")
    mounts = MountService(table, tmp_path / "mounts.toml", shared_models=True)
    mounts.update_project("work", [mount.id for mount in shared], None)
    downloads = ModelDownloads(mounts, ProvenanceStore(tmp_path / "provenance.json"))

    class Response(io.BytesIO):
        status = 200
        headers = HTTPMessage()

    @contextmanager
    def provider(url: str, _addresses: object, headers: dict[str, str], _timeout: object):
        authenticated = headers.get("Authorization") == f"Bearer {token}"
        if url.endswith("/api/whoami-v2"):
            assert url == "https://huggingface.co/api/whoami-v2"
            response = Response(b'{"name":"synthetic-user"}')
            response.status = 200 if authenticated else 401
        elif url.endswith("/api/v1/me"):
            assert url == "https://civitai.com/api/v1/me"
            response = Response(b'{"id":137,"username":"synthetic-civitai-user"}')
            response.status = (
                200 if headers.get("Authorization") == f"Bearer {civitai_token}" else 401
            )
        elif authenticated and entitled:
            response = Response(
                json.dumps(
                    {
                        "sha": "d" * 40,
                        "siblings": [
                            {
                                "rfilename": "flux1-dev.safetensors",
                                "size": 7,
                                "lfs": {"sha256": "e" * 64, "size": 7},
                            }
                        ],
                    }
                ).encode()
            )
        else:
            response = Response(b"do-not-echo-private-provider-error")
            response.status = 401
            response.headers = HTTPMessage()
            if authenticated:
                response.headers["X-Error-Code"] = "GatedRepo"
        yield cast("HTTPSResponse", response)

    monkeypatch.setattr(fetch, "_globally_routable_addresses", lambda *_args: ("8.8.8.8",))
    monkeypatch.setattr(fetch, "_open_pinned_https", provider)
    grants = frozenset({"assets:read", "assets:write", "settings:write", "jobs:submit"})
    auth = StaticBearerAuthenticator(
        {
            "operator": Principal("owner", {"local": grants}),
            "viewer": Principal("viewer", {"local": frozenset({"assets:read"})}),
        }
    )

    async def scenario() -> None:
        nonlocal entitled
        app = create_app(make_engine, SCHEMAS, authenticator=auth)
        add_model_download_routes(app, downloads)
        client = TestClient(TestServer(app), headers={"Authorization": "Bearer operator"})
        await client.start_server()
        path = "/api/models/providers/huggingface?scope=local"
        body = {
            "name": "flux1-dev.safetensors",
            "kind": "model/diffusion",
            "project": "work",
            "scope": "local",
        }
        try:
            denied = await client.put(
                path, json={"token": token}, headers={"Authorization": "Bearer viewer"}
            )
            assert denied.status == 403 and not downloads.credentials.path.exists()
            missing = await client.post("/api/models/downloads/plan", json=body)
            assert missing.status == 409
            assert (await missing.json())["error"] == "credential-required"
            invalid = await client.put(path, json={"token": "invalid-synthetic"})
            assert invalid.status == 401 and not downloads.credentials.path.exists()
            saved = await client.put(path, json={"token": token})
            assert saved.status == 200
            assert await saved.json() == {
                "provider": "huggingface",
                "configured": True,
                "pkceAvailable": True,
            }
            assert saved.headers["Cache-Control"] == "no-store"
            refused = await client.post("/api/models/downloads/plan", json=body)
            assert refused.status == 409
            assert (await refused.json())["error"] == "provider-grant-required"
            entitled = True
            ready = await client.post("/api/models/downloads/plan", json=body)
            assert ready.status == 200
            plan = await ready.json()
            assert plan["status"] == "consent-required"
            assert token not in json.dumps(plan)
            assert not list((tmp_path / "models").rglob("*.safetensors"))
            civitai = "/api/models/providers/civitai?scope=local"
            assert await (await client.get(civitai)).json() == {
                "provider": "civitai",
                "configured": False,
                "pkceAvailable": pkce_available,
            }
            started = await client.post(
                f"{civitai.split('?')[0]}/sign-in?scope=local", json={"method": "pkce"}
            )
            if pkce_available:
                from urllib.parse import parse_qs, urlsplit

                assert started.status == 201
                flow = await started.json()
                authorization = urlsplit(flow["authorizationUrl"])
                assert authorization.netloc == "auth.civitai.com"
                query = parse_qs(authorization.query)
                assert query["client_id"] == [client_id or "aa0a7a67-b80d-481d-b165-54f7b254cdaa"]
                assert query["redirect_uri"] == ["http://localhost:15493/oauth/callback/civitai"]
                assert query["scope"] == ["4"] and query["code_challenge_method"] == ["S256"]
                cancelled = await client.delete(
                    f"{civitai.split('?')[0]}/sign-in/{flow['flowId']}?scope=local"
                )
                assert cancelled.status == 200
            else:
                assert started.status == 409
                assert await started.json() == {"error": "provider-sign-in-unavailable"}
            denied = await client.put(
                civitai, json={"token": civitai_token}, headers={"Authorization": "Bearer viewer"}
            )
            assert denied.status == 403
            invalid = await client.put(civitai, json={"token": "invalid-synthetic"})
            assert invalid.status == 401
            assert downloads.credentials.token("civitai") is None
            saved = await client.put(civitai, json={"token": civitai_token})
            assert saved.status == 200
            assert await saved.json() == {
                "provider": "civitai",
                "configured": True,
                "pkceAvailable": pkce_available,
            }
            status = await client.get(civitai)
            assert await status.json() == {
                "provider": "civitai",
                "configured": True,
                "pkceAvailable": pkce_available,
            }
            assert status.headers["Cache-Control"] == "no-store"
            assert downloads.credentials.token("civitai") == civitai_token
            assert downloads.credentials.token() == token
            forgotten = await client.delete(civitai)
            assert forgotten.status == 200
            assert await forgotten.json() == {
                "provider": "civitai",
                "configured": False,
                "pkceAvailable": pkce_available,
            }
            assert downloads.credentials.token("civitai") is None
            assert downloads.credentials.token() == token
            # A second engine process may change the shared private credentials.
            downloads.credentials.set_token(None)
            downloads.credentials.set_token(token)
            changed = await client.post(
                "/api/models/downloads/start?scope=local",
                json={"planId": plan["planId"], "consent": True},
            )
            assert changed.status == 409
            assert await changed.json() == {"error": "provider-credentials-changed"}
            revoked = await client.delete(path)
            assert revoked.status == 200
            assert await revoked.json() == {
                "provider": "huggingface",
                "configured": False,
                "pkceAvailable": True,
            }
            stale = await client.post(
                "/api/models/downloads/start?scope=local",
                json={"planId": plan["planId"], "consent": True},
            )
            assert stale.status == 404
            assert downloads.credentials.token() is None
            assert not downloads.credentials.path.exists()
        finally:
            await client.close()
            await mounts.close()

    run_scenario(scenario())
