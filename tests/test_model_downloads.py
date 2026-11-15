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
