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

from tests.test_asset_writer import one_mount_snapshot


def test_declared_model_plan_pins_metadata_without_downloading_and_respects_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    def respond(url: str) -> Generator[HTTPSResponse]:
        requests.append(url)
        assert (
            url == "https://huggingface.co/api/models/Comfy-Org/flux1-dev/revision/main?blobs=true"
        )
        body = {
            "sha": "d" * 40,
            "cardData": {"license": "flux-1-dev"},
            "siblings": [
                {
                    "rfilename": "flux1-dev.safetensors",
                    "size": 137,
                    "lfs": {"sha256": "e" * 64, "size": 137},
                }
            ],
        }
        yield cast("HTTPSResponse", io.BytesIO(json.dumps(body).encode()))

    monkeypatch.setattr(huggingface, "open_model_response", respond)
    grants = frozenset({"assets:read", "settings:write"})
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

            def unavailable(_url: str) -> None:
                raise OSError("synthetic-private-provider-response")

            monkeypatch.setattr(huggingface, "open_model_response", unavailable)
            response = await client.post("/api/models/downloads/plan", json=body)
            assert response.status == 409
            assert await response.json() == {"error": "model-source-unavailable"}
        finally:
            await client.close()
            await mounts.close()

    run_scenario(scenario())


@pytest.mark.parametrize("outcome", ["complete", "cancel", "revoke", "sha256", "index", "reuse"])
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
    def respond(url: str) -> Generator[HTTPSResponse]:
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
            existing = old.parent / "nested" / "different-name.safetensors"
            if outcome == "reuse":
                existing.parent.mkdir()
                existing.write_bytes(payload)
                (old.parent / "same-size-corrupt.safetensors").write_bytes(b"x" * len(payload))
            response = await client.post("/api/models/downloads/start", json=start)
            assert response.status == 202
            job = await response.json()
            job_id = job["jobId"]
            if outcome == "reuse":
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
        existing.symlink_to(outside)
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

    class Response(io.BytesIO):
        status = 200
        headers = HTTPMessage()

    @contextmanager
    def respond(
        url: str, _addresses: object, _headers: object, _timeout: object
    ) -> Generator[HTTPSResponse]:
        requests.append(url)
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
        f"https://huggingface.co/{repo}/resolve/main/weights/model.safetensors"
    )
    assert selection.revision == commit
    assert selection.sha256 == sha256
    assert selection.license == "creativeml-openrail-m"
    root = tmp_path / "models"
    root.mkdir()
    writer = AssetWriter(MountSnapshotWriter(one_mount_snapshot(tmp_path, root)))
    ref = selection.download(
        writer, SaveTarget("out", "model.safetensors"), digest=digest_bytes(payload)
    )
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
    def respond(url: str) -> Generator[HTTPSResponse]:
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

    def no_network(_url: str) -> None:
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
    def respond(_url: str) -> Generator[HTTPSResponse]:
        yield cast("HTTPSResponse", io.BytesIO(body))

    monkeypatch.setattr(huggingface, "open_model_response", respond)
    with pytest.raises(AssetError):
        select_huggingface_file("https://huggingface.co/acme/model/resolve/main/model.safetensors")


def test_model_folder_modes_persist_project_choices_and_follow_shared_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import MountTable
    from dinkster_values.asset_folders import use_asset_folders

    from dinkster.model_settings import ModelSettings
    from dinkster.mounts_api import MountService
    from tests.async_scenarios import run_scenario
    from tests.test_mounts import make_client

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
