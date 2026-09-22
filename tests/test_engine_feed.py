"""Tests for engine feed documents and mirror downloads."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from dinkster.engine_feed import (
    BASE_ARCHIVE_SUFFIXES,
    Artifact,
    EngineBase,
    EngineFeedError,
    Mirror,
    Wheel,
    parse_channel,
    parse_manifest,
)

BASE_ARCHIVE_BYTES = b"fake base archive payload"
WHEEL_BYTES = b"fake wheel payload"
COMMIT = "a" * 40
BASE_ID = "b" * 64


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_record(path: str, data: bytes) -> dict[str, object]:
    return {"path": path, "sha256": digest(data), "size": len(data)}


def wheel_record(
    name: str = "dinkster",
    version: str = "0.0.1",
    filename: str | None = None,
    environments: list[str] | None = None,
    path: str | None = None,
) -> dict[str, object]:
    return {
        "path": path or f"engine/{COMMIT}/wheels/{name}-{version}-py3-none-any.whl",
        "sha256": digest(WHEEL_BYTES),
        "size": len(WHEEL_BYTES),
        "filename": filename or f"{name}-{version}-py3-none-any.whl",
        "name": name,
        "version": version,
        "environments": environments if environments is not None else ["execution"],
    }


def base_record() -> dict[str, object]:
    return {
        "id": BASE_ID,
        "archive": artifact_record(
            "base/win-cu128/py312-torch2.13-9f2a.tar.gz", BASE_ARCHIVE_BYTES
        ),
        "python": "python/bin/python3",
        "packages": {"torch": "2.13.0", "torchvision": "0.28.0", "filelock": "3.20.0"},
    }


def manifest_record() -> dict[str, object]:
    return {
        "format": "dinkster.engine/1",
        "commit": COMMIT,
        "cell": "win-cu128",
        "base": base_record(),
        "wheels": [
            wheel_record("dinkster", "0.0.1"),
            wheel_record("dinkster-workers", "0.2.0"),
            wheel_record("numpy", "2.1.0", environments=["control", "execution"]),
        ],
    }


def channel_record() -> dict[str, object]:
    return {
        "format": "dinkster.engine-channel/1",
        "channel": "github-live",
        "commit": COMMIT,
        "minimumLauncherVersion": "0.0.1",
        "cells": {
            "win-cu128": artifact_record(f"engine/{COMMIT}/win-cu128.json", b"{}"),
            "linux-cu128": artifact_record(f"engine/{COMMIT}/linux-cu128.json", b"{}"),
        },
    }


def encoded(record: dict[str, object]) -> bytes:
    return json.dumps(record).encode()


# --------------------------------------------------------------------------- #
# Artifact and wheel validation


class TestArtifact:
    def test_valid_artifact(self) -> None:
        artifact = Artifact(path="engine/x/win-cu128.json", sha256=digest(WHEEL_BYTES), size=7)
        assert artifact.to_record() == {
            "path": "engine/x/win-cu128.json",
            "sha256": digest(WHEEL_BYTES),
            "size": 7,
        }

    @pytest.mark.parametrize(
        "bad_path",
        [
            "../escape.tar.gz",
            "a/../b.tar.gz",
            "/absolute.tar.gz",
            "dir/",
            "",
            "a//b.tar.gz",
            "a/./b.tar.gz",
            ".hidden",
            "./a.tar.gz",
            "a\\b.tar.gz",
            "http://evil/x.tar.gz",
            "C:/x.tar.gz",
            "a.tar.gz?query=1",
            "a.tar.gz#fragment",
            "user@host/a.tar.gz",
            "a%2fb.tar.gz",
            "a%5cb.tar.gz",
            "a%2Fb.tar.gz",
            "a\x00b",
            "a\nb",
        ],
    )
    def test_rejects_bad_paths(self, bad_path: str) -> None:
        with pytest.raises(EngineFeedError):
            Artifact(path=bad_path, sha256=digest(WHEEL_BYTES), size=7)

    def test_rejects_uppercase_and_short_digests(self) -> None:
        for bad in (digest(WHEEL_BYTES).upper(), digest(WHEEL_BYTES)[:63], "z" * 64, 123):
            with pytest.raises(EngineFeedError):
                Artifact(path="a/b", sha256=bad, size=7)  # type: ignore

    def test_rejects_bad_sizes(self) -> None:
        for bad in (-1, True, False, 1.5, "7", None):
            with pytest.raises(EngineFeedError):
                Artifact(path="a/b", sha256=digest(WHEEL_BYTES), size=bad)  # type: ignore

    def test_zero_size_allowed(self) -> None:
        artifact = Artifact(path="a/empty", sha256=hashlib.sha256(b"").hexdigest(), size=0)
        assert artifact.size == 0


class TestWheel:
    def test_wheel_is_artifact(self) -> None:
        wheel = Wheel(
            path="engine/wheels/dinkster-0.0.1-py3-none-any.whl",
            sha256=digest(WHEEL_BYTES),
            size=7,
            filename="dinkster-0.0.1-py3-none-any.whl",
            name="dinkster",
            version="0.0.1",
            environments=("execution",),
        )
        assert isinstance(wheel, Artifact)

    def test_environments_canonicalized(self) -> None:
        wheel = Wheel(
            path="w.whl",
            sha256=digest(WHEEL_BYTES),
            size=7,
            filename="w.whl",
            name="dinkster",
            version="0.0.1",
            environments=("execution", "control"),
        )
        assert wheel.environments == ("control", "execution")

    @pytest.mark.parametrize(
        "bad_environments",
        [[], ["gpu"], ["control", "control"], "execution", [1]],
    )
    def test_rejects_bad_environments(self, bad_environments: object) -> None:
        with pytest.raises(EngineFeedError):
            Wheel(
                path="w.whl",
                sha256=digest(WHEEL_BYTES),
                size=7,
                filename="w.whl",
                name="dinkster",
                version="0.0.1",
                environments=bad_environments,  # type: ignore
            )

    @pytest.mark.parametrize(
        "bad_name", ["Dinkster", "dinkster_", "dinkster..x", "-dinkster", "dinkster-", ""]
    )
    def test_rejects_unnormalized_names(self, bad_name: str) -> None:
        with pytest.raises(EngineFeedError):
            Wheel(
                path="w.whl",
                sha256=digest(WHEEL_BYTES),
                size=7,
                filename="w.whl",
                name=bad_name,
                version="0.0.1",
                environments=("execution",),
            )

    @pytest.mark.parametrize(
        "bad_filename", ["../evil.whl", "a/b.whl", "a\\b.whl", ".hidden.whl", "", "w?h.whl"]
    )
    def test_rejects_bad_filenames(self, bad_filename: str) -> None:
        with pytest.raises(EngineFeedError):
            Wheel(
                path="w.whl",
                sha256=digest(WHEEL_BYTES),
                size=7,
                filename=bad_filename,
                name="dinkster",
                version="0.0.1",
                environments=("execution",),
            )

    @pytest.mark.parametrize("bad_version", ["", "1.2\n3", " 1.2", "1 2"])
    def test_rejects_bad_versions(self, bad_version: str) -> None:
        with pytest.raises(EngineFeedError):
            Wheel(
                path="w.whl",
                sha256=digest(WHEEL_BYTES),
                size=7,
                filename="w.whl",
                name="dinkster",
                version=bad_version,
                environments=("execution",),
            )

    def test_to_record_round_trip(self) -> None:
        wheel = Wheel(
            path="engine/wheels/dinkster-0.0.1-py3-none-any.whl",
            sha256=digest(WHEEL_BYTES),
            size=7,
            filename="dinkster-0.0.1-py3-none-any.whl",
            name="dinkster",
            version="0.0.1",
            environments=("control", "execution"),
        )
        rebuilt = parse_manifest(
            encoded(
                {
                    "format": "dinkster.engine/1",
                    "commit": COMMIT,
                    "cell": "win-cu128",
                    "base": {
                        "id": BASE_ID,
                        "archive": artifact_record("base/x.tar.gz", BASE_ARCHIVE_BYTES),
                        "python": "python/bin/python3",
                        "packages": {},
                    },
                    "wheels": [wheel.to_record()],
                }
            )
        )
        assert rebuilt.wheels == (wheel,)


# --------------------------------------------------------------------------- #
# Manifest parsing


class TestManifestParsing:
    def test_valid_manifest(self) -> None:
        manifest = parse_manifest(encoded(manifest_record()))
        assert manifest.format == "dinkster.engine/1"
        assert manifest.commit == COMMIT
        assert manifest.cell == "win-cu128"
        assert manifest.base.id == BASE_ID
        assert manifest.base.archive.path == "base/win-cu128/py312-torch2.13-9f2a.tar.gz"
        assert manifest.base.python == "python/bin/python3"
        assert manifest.base.packages["torch"] == "2.13.0"
        assert [wheel.name for wheel in manifest.wheels] == [
            "dinkster",
            "dinkster-workers",
            "numpy",
        ]

    def test_torch_and_closure_allowed_in_base(self) -> None:
        record = manifest_record()
        record["base"] = {  # type: ignore
            **base_record(),
            "packages": {"torch": "2.13.0", "torchvision": "0.28.0", "sympy": "1.13"},
        }
        manifest = parse_manifest(encoded(record))
        assert manifest.base.packages["sympy"] == "1.13"

    def test_rejects_dinkster_base_packages(self) -> None:
        for bad in ("dinkster-kitchen", "dinkster-aimdo", "dinkster", "dinkster-workers"):
            record = manifest_record()
            record["base"] = {**base_record(), "packages": {bad: "1.0"}}  # type: ignore
            with pytest.raises(EngineFeedError, match="code layer"):
                parse_manifest(encoded(record))

    def test_rejects_wheel_repeating_base_distribution(self) -> None:
        record = manifest_record()
        record["wheels"] = [wheel_record("torch", "2.13.0")]  # type: ignore
        with pytest.raises(EngineFeedError, match="base distribution"):
            parse_manifest(encoded(record))

    def test_rejects_duplicate_wheel_names(self) -> None:
        record = manifest_record()
        record["wheels"] = [  # type: ignore
            wheel_record("dinkster", "0.0.1"),
            wheel_record("dinkster", "0.0.2"),
        ]
        with pytest.raises(EngineFeedError, match="duplicate wheel distribution name"):
            parse_manifest(encoded(record))

    def test_rejects_duplicate_wheel_filenames(self) -> None:
        record = manifest_record()
        record["wheels"] = [  # type: ignore
            wheel_record("dinkster", "0.0.1"),
            wheel_record("other", "1.0", filename="dinkster-0.0.1-py3-none-any.whl"),
        ]
        with pytest.raises(EngineFeedError, match="duplicate wheel filename"):
            parse_manifest(encoded(record))

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda r: r.update(format="dinkster.engine/2"),
            lambda r: r.update(commit="A" * 40),
            lambda r: r.update(commit="a" * 39),
            lambda r: r.update(commit="g" * 40),
            lambda r: r.update(commit=123),
            lambda r: r.update(cell="win-cu256"),
            lambda r: r.update(cell=None),
            lambda r: r.update(unexpected=1),
            lambda r: r.pop("base"),
            lambda r: r.update(wheels={}),
            lambda r: r["base"].update(id="b" * 63),
            lambda r: r["base"].update(python="/etc/passwd"),
            lambda r: r["base"].update(python="../python"),
            lambda r: r["base"].update(unexpected=None),
            lambda r: r["base"].update(packages={"torch": None}),
            lambda r: r["base"].update(packages={"torch": 2.13}),
            lambda r: r["base"]["archive"].pop("size"),
        ],
    )
    def test_rejects_invalid_manifests(self, mutate: object) -> None:
        record = manifest_record()
        mutate(record)  # type: ignore
        with pytest.raises(EngineFeedError):
            parse_manifest(encoded(record))

    def test_rejects_duplicate_json_keys(self) -> None:
        raw = b'{"format": "dinkster.engine/1", "format": "dinkster.engine/1"}'
        with pytest.raises(EngineFeedError, match="duplicate JSON key"):
            parse_manifest(raw)

    def test_rejects_nan_and_infinity(self) -> None:
        with pytest.raises(EngineFeedError):
            parse_manifest(b'{"size": NaN}')
        with pytest.raises(EngineFeedError):
            parse_manifest(b'{"size": Infinity}')

    def test_rejects_non_object_and_invalid_json(self) -> None:
        with pytest.raises(EngineFeedError, match="JSON object"):
            parse_manifest(b"[1, 2]")
        with pytest.raises(EngineFeedError, match="not valid JSON"):
            parse_manifest(b"{nope}")
        with pytest.raises(EngineFeedError, match="UTF-8"):
            parse_manifest(b'{"a": "\xff"}')

    def test_to_record_round_trip(self) -> None:
        manifest = parse_manifest(encoded(manifest_record()))
        reparsed = parse_manifest(json.dumps(manifest.to_record()).encode())
        assert reparsed == manifest


# --------------------------------------------------------------------------- #
# Channel parsing


class TestChannelParsing:
    def test_valid_channel(self) -> None:
        channel = parse_channel(encoded(channel_record()))
        assert channel.format == "dinkster.engine-channel/1"
        assert channel.channel == "github-live"
        assert channel.commit == COMMIT
        assert channel.minimum_launcher_version == "0.0.1"
        assert set(channel.cells) == {"win-cu128", "linux-cu128"}
        assert channel.cells["win-cu128"].path == f"engine/{COMMIT}/win-cu128.json"

    def test_stable_channel_accepted(self) -> None:
        record = channel_record()
        record["channel"] = "stable"  # type: ignore
        assert parse_channel(encoded(record)).channel == "stable"

    def test_newer_minimum_launcher_version_accepted(self) -> None:
        record = channel_record()
        record["minimumLauncherVersion"] = "0.1.0"  # type: ignore
        channel = parse_channel(encoded(record))
        assert channel.minimum_launcher_version == "0.1.0"

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda r: r.update(format="dinkster.engine-channel/2"),
            lambda r: r.update(channel="beta"),
            lambda r: r.update(commit="a" * 40 + "0"),
            lambda r: r.update(minimumLauncherVersion="1.0"),
            lambda r: r.update(minimumLauncherVersion="0.0.1-beta"),
            lambda r: r.update(minimumLauncherVersion="01.0.0"),
            lambda r: r.update(minimumLauncherVersion=1),
            lambda r: r.update(cells={}),
            lambda r: r.update(cells=None),
            lambda r: r["cells"].update(
                {"mac-arm64": {"path": "wrong", "sha256": "b" * 64, "size": 0}}
            ),
            lambda r: r["cells"].update(
                {
                    "no-such-cell": {
                        "path": f"engine/{COMMIT}/no.json",
                        "sha256": "b" * 64,
                        "size": 0,
                    }
                }
            ),
            lambda r: r.update(unexpected=None),
        ],
    )
    def test_rejects_invalid_channels(self, mutate: object) -> None:
        record = channel_record()
        mutate(record)  # type: ignore
        with pytest.raises(EngineFeedError):
            parse_channel(encoded(record))

    def test_cell_artifact_fields_validated(self) -> None:
        record = channel_record()
        record["cells"] = {
            "win-cu128": {"path": f"engine/{COMMIT}/win-cu128.json", "sha256": "z" * 64, "size": 0}
        }  # type: ignore
        with pytest.raises(EngineFeedError, match="hex"):
            parse_channel(encoded(record))

    def test_to_record_round_trip(self) -> None:
        channel = parse_channel(encoded(channel_record()))
        reparsed = parse_channel(json.dumps(channel.to_record()).encode())
        assert reparsed == channel


# --------------------------------------------------------------------------- #
# Mirror URL validation


class TestMirrorUrlValidation:
    def test_https_accepted_without_loopback(self) -> None:
        mirror = Mirror("https://mirror.example.com/dl")
        assert mirror.base_url == "https://mirror.example.com/dl"
        assert mirror.allow_local_http is False

    def test_loopback_http_requires_opt_in(self) -> None:
        with pytest.raises(EngineFeedError, match="HTTPS"):
            Mirror("http://127.0.0.1:9000/dl")
        for host in ("127.0.0.1", "127.9.9.9", "localhost", "::1"):
            mirror = Mirror(
                f"http://[{host}]:9000/dl" if host == "::1" else f"http://{host}:9000/dl",
                allow_local_http=True,
            )
            assert mirror.allow_local_http is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://10.0.0.5:9000/dl",
            "http://192.168.1.10/dl",
            "http://mirror.example.com/dl",
            "ftp://mirror.example.com/dl",
        ],
    )
    def test_plain_http_rejected_off_loopback(self, url: str) -> None:
        with pytest.raises(EngineFeedError, match="HTTPS"):
            Mirror(url, allow_local_http=True)

    @pytest.mark.parametrize(
        "url",
        [
            "https://user:pass@mirror.example.com/dl",
            "https://user@mirror.example.com/dl",
            "https://mirror.example.com/dl?sign=token",
            "https://mirror.example.com/dl#fragment",
            "mirror.example.com/dl",
            "https://mirror.example.com/a/../dl",
            "https://mirror.example.com/dl ",
            "",
        ],
    )
    def test_rejects_bad_base_urls(self, url: str) -> None:
        with pytest.raises(EngineFeedError):
            Mirror(url)


# --------------------------------------------------------------------------- #
# Loopback mirror server


@dataclass
class _ServerConfig:
    content: dict[str, bytes] = field(default_factory=dict)
    redirects: dict[str, str] = field(default_factory=dict)
    ignore_range: bool = False
    bad_range_start: int | None = None
    serve_only: int | None = None
    error_status: int | None = None
    requests: list[tuple[str, str | None]] = field(default_factory=list)

    def ranged_requests(self) -> list[tuple[str, str]]:
        return [(path, rng) for path, rng in self.requests if rng is not None]


class _FeedHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, _FeedServer)
        config = server.config
        config.requests.append((self.path, self.headers.get("Range")))
        location = config.redirects.get(self.path)
        if location is not None:
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if config.error_status is not None:
            self.send_response(config.error_status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = config.content.get(self.path)
        if body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        range_header = self.headers.get("Range")
        start = 0
        status = 200
        if range_header and range_header.startswith("bytes=") and not config.ignore_range:
            start = int(range_header.removeprefix("bytes=").split("-")[0])
            status = 206
        payload = body[start:]
        self.send_response(status)
        if status == 206:
            advertised = start if config.bad_range_start is None else config.bad_range_start
            self.send_header("Content-Range", f"bytes {advertised}-{len(body) - 1}/{len(body)}")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if config.serve_only is not None:
            payload = payload[: config.serve_only]
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _FeedServer(ThreadingHTTPServer):
    def __init__(self, config: _ServerConfig) -> None:
        super().__init__(("127.0.0.1", 0), _FeedHandler)
        self.config = config


@pytest.fixture
def feed_server() -> object:
    server = _FeedServer(_ServerConfig())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def make_mirror(server: _FeedServer) -> Mirror:
    port = server.server_address[1]
    return Mirror(f"http://127.0.0.1:{port}/dl", allow_local_http=True)


# --------------------------------------------------------------------------- #
# get_json


class TestGetJson:
    def test_returns_document_bytes(self, feed_server: _FeedServer) -> None:
        payload = b'{"ok": true}'
        feed_server.config.content["/dl/engine/abc/win-cu128.json"] = payload
        assert make_mirror(feed_server).get_json("engine/abc/win-cu128.json") == payload

    def test_missing_document_raises(self, feed_server: _FeedServer) -> None:
        with pytest.raises(EngineFeedError, match="HTTP 404"):
            make_mirror(feed_server).get_json("engine/abc/win-cu128.json")

    def test_rejects_oversized_documents(self, feed_server: _FeedServer) -> None:
        feed_server.config.content["/dl/big.json"] = b"x" * (4 * 1024 * 1024 + 1)
        with pytest.raises(EngineFeedError, match="exceeds"):
            make_mirror(feed_server).get_json("big.json")

    def test_rejects_traversal_paths_before_network(self, feed_server: _FeedServer) -> None:
        with pytest.raises(EngineFeedError):
            make_mirror(feed_server).get_json("../secrets.json")
        assert feed_server.config.requests == []


# --------------------------------------------------------------------------- #
# download


CONTENT = bytes((index * 37 + 11) % 256 for index in range(50_000))


def feed_artifact(path: str = "store/0001/base.tar.gz", data: bytes = CONTENT) -> Artifact:
    return Artifact(path=path, sha256=digest(data), size=len(data))


class TestDownload:
    def test_downloads_and_verifies(self, feed_server: _FeedServer, tmp_path: Path) -> None:
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = CONTENT
        destination = tmp_path / "nested" / "deeper" / "base.tar.gz"
        result = make_mirror(feed_server).download(feed_artifact(), destination)
        assert result == destination
        assert destination.read_bytes() == CONTENT
        assert not destination.with_name(destination.name + ".part").exists()

    def test_resume_after_interruption(self, feed_server: _FeedServer, tmp_path: Path) -> None:
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = CONTENT
        feed_server.config.serve_only = 300
        destination = tmp_path / "base.tar.gz"
        part = destination.with_name(destination.name + ".part")
        with pytest.raises(EngineFeedError, match="did not complete"):
            make_mirror(feed_server).download(feed_artifact(), destination)
        assert not destination.exists()
        assert part.exists()
        partial_size = part.stat().st_size
        assert 0 < partial_size < len(CONTENT)
        assert part.read_bytes() == CONTENT[:partial_size]
        feed_server.config.serve_only = None
        make_mirror(feed_server).download(feed_artifact(), destination)
        assert destination.read_bytes() == CONTENT
        assert feed_server.config.ranged_requests(), "resume must send Range requests"

    def test_ignored_range_resets(self, feed_server: _FeedServer, tmp_path: Path) -> None:
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = CONTENT
        feed_server.config.ignore_range = True
        destination = tmp_path / "base.tar.gz"
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(b"JUNKJUNKJUNK")
        make_mirror(feed_server).download(feed_artifact(), destination)
        assert destination.read_bytes() == CONTENT
        assert feed_server.config.ranged_requests(), "server must have been asked for a range"

    def test_bad_content_range_offset_resets(
        self, feed_server: _FeedServer, tmp_path: Path
    ) -> None:
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = CONTENT
        feed_server.config.bad_range_start = 0
        destination = tmp_path / "base.tar.gz"
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(CONTENT[:100])
        make_mirror(feed_server).download(feed_artifact(), destination)
        assert destination.read_bytes() == CONTENT

    def test_corrupt_prefix_fails_and_deletes_part(
        self, feed_server: _FeedServer, tmp_path: Path
    ) -> None:
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = CONTENT
        destination = tmp_path / "base.tar.gz"
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(b"JUNKJUNKJUNK")
        with pytest.raises(EngineFeedError, match="sha256"):
            make_mirror(feed_server).download(feed_artifact(), destination)
        assert not part.exists()
        assert not destination.exists()

    def test_corrupt_payload_rejected(self, feed_server: _FeedServer, tmp_path: Path) -> None:
        wrong = CONTENT[:-1] + bytes([CONTENT[-1] ^ 0xFF])
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = wrong
        destination = tmp_path / "base.tar.gz"
        with pytest.raises(EngineFeedError, match="sha256"):
            make_mirror(feed_server).download(feed_artifact(), destination)
        assert not destination.exists()
        assert not destination.with_name(destination.name + ".part").exists()

    def test_truncated_transfer_keeps_partial_bytes(
        self, feed_server: _FeedServer, tmp_path: Path
    ) -> None:
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = CONTENT
        feed_server.config.serve_only = 100
        destination = tmp_path / "base.tar.gz"
        part = destination.with_name(destination.name + ".part")
        with pytest.raises(EngineFeedError, match="did not complete"):
            make_mirror(feed_server).download(feed_artifact(), destination)
        assert not destination.exists()
        assert part.read_bytes() == CONTENT[: part.stat().st_size]

    def test_valid_destination_untouched_even_when_mirror_lies(
        self, feed_server: _FeedServer, tmp_path: Path
    ) -> None:
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = b"garbage from a lying mirror"
        destination = tmp_path / "base.tar.gz"
        destination.write_bytes(CONTENT)
        result = make_mirror(feed_server).download(feed_artifact(), destination)
        assert result == destination
        assert destination.read_bytes() == CONTENT
        assert feed_server.config.requests == [], "verified destination must skip the network"

    def test_corrupt_destination_replaced(self, feed_server: _FeedServer, tmp_path: Path) -> None:
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = CONTENT
        destination = tmp_path / "base.tar.gz"
        destination.write_bytes(b"corrupt complete data")
        make_mirror(feed_server).download(feed_artifact(), destination)
        assert destination.read_bytes() == CONTENT

    def test_complete_part_promoted_without_network(
        self, feed_server: _FeedServer, tmp_path: Path
    ) -> None:
        destination = tmp_path / "base.tar.gz"
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(CONTENT)
        result = make_mirror(feed_server).download(feed_artifact(), destination)
        assert result == destination
        assert destination.read_bytes() == CONTENT
        assert feed_server.config.requests == []
        assert not part.exists()

    def test_server_errors_are_retried(self, feed_server: _FeedServer, tmp_path: Path) -> None:
        feed_server.config.content["/dl/store/0001/base.tar.gz"] = CONTENT
        feed_server.config.error_status = 503
        destination = tmp_path / "base.tar.gz"
        with pytest.raises(EngineFeedError, match="did not complete"):
            make_mirror(feed_server).download(feed_artifact(), destination)
        feed_server.config.error_status = None
        make_mirror(feed_server).download(feed_artifact(), destination)
        assert destination.read_bytes() == CONTENT

    def test_client_errors_are_fatal(self, feed_server: _FeedServer, tmp_path: Path) -> None:
        destination = tmp_path / "base.tar.gz"
        with pytest.raises(EngineFeedError, match="HTTP 404"):
            make_mirror(feed_server).download(feed_artifact(), destination)
        assert not destination.exists()

    def test_directory_destination_rejected(self, feed_server: _FeedServer, tmp_path: Path) -> None:
        destination = tmp_path / "base.tar.gz"
        destination.mkdir()
        with pytest.raises(EngineFeedError, match="directory"):
            make_mirror(feed_server).download(feed_artifact(), destination)


class TestDownloadRedirects:
    PATH = "/dl/store/0001/base.tar.gz"

    def test_same_origin_redirect_under_prefix_followed(
        self, feed_server: _FeedServer, tmp_path: Path
    ) -> None:
        feed_server.config.content["/dl/moved/0001/base.tar.gz"] = CONTENT
        feed_server.config.redirects[self.PATH] = (
            f"http://127.0.0.1:{feed_server.server_address[1]}/dl/moved/0001/base.tar.gz"
        )
        destination = tmp_path / "base.tar.gz"
        make_mirror(feed_server).download(feed_artifact(), destination)
        assert destination.read_bytes() == CONTENT

    def test_same_origin_redirect_outside_prefix_rejected(
        self, feed_server: _FeedServer, tmp_path: Path
    ) -> None:
        feed_server.config.content["/elsewhere/base.tar.gz"] = CONTENT
        feed_server.config.redirects[self.PATH] = (
            f"http://127.0.0.1:{feed_server.server_address[1]}/elsewhere/base.tar.gz"
        )
        destination = tmp_path / "base.tar.gz"
        with pytest.raises(EngineFeedError, match="base prefix"):
            make_mirror(feed_server).download(feed_artifact(), destination)
        assert not destination.exists()

    def test_cross_origin_redirect_rejected(self, feed_server: _FeedServer, tmp_path: Path) -> None:
        other = _FeedServer(_ServerConfig())
        thread = threading.Thread(target=other.serve_forever, daemon=True)
        thread.start()
        try:
            other.config.content["/dl/store/0001/base.tar.gz"] = CONTENT
            feed_server.config.redirects[self.PATH] = (
                f"http://127.0.0.1:{other.server_address[1]}/dl/store/0001/base.tar.gz"
            )
            destination = tmp_path / "base.tar.gz"
            with pytest.raises(EngineFeedError, match="origin"):
                make_mirror(feed_server).download(feed_artifact(), destination)
            assert not destination.exists()
            assert other.config.requests == [], "cross-origin redirect must not be contacted"
        finally:
            other.shutdown()
            other.server_close()

    def test_redirect_with_credentials_rejected(
        self, feed_server: _FeedServer, tmp_path: Path
    ) -> None:
        feed_server.config.redirects[self.PATH] = (
            "http://user:pass@127.0.0.1:1/dl/store/0001/base.tar.gz"
        )
        destination = tmp_path / "base.tar.gz"
        with pytest.raises(EngineFeedError, match="credentials"):
            make_mirror(feed_server).download(feed_artifact(), destination)
        assert not destination.exists()

    def test_get_json_follows_same_origin_redirect(self, feed_server: _FeedServer) -> None:
        payload = b'{"moved": true}'
        feed_server.config.content["/dl/engine/abc/win-cu128.json"] = payload
        feed_server.config.redirects["/dl/engine/old.json"] = (
            f"http://127.0.0.1:{feed_server.server_address[1]}/dl/engine/abc/win-cu128.json"
        )
        assert make_mirror(feed_server).get_json("engine/old.json") == payload


# --------------------------------------------------------------------------- #
# Base archive suffix (parser scope; extraction lives in the installer)


class TestBaseArchiveSuffix:
    def test_manifest_requires_known_base_archive_suffix(self) -> None:
        record = manifest_record()
        base = base_record()
        base["archive"] = artifact_record("base/win-cu128/py312-torch2.13.zip", BASE_ARCHIVE_BYTES)  # type: ignore
        record["base"] = base  # type: ignore
        with pytest.raises(EngineFeedError, match="base archive must be one of"):
            parse_manifest(encoded(record))

    @pytest.mark.parametrize("suffix", BASE_ARCHIVE_SUFFIXES)
    def test_all_documented_suffixes_accepted(self, suffix: str) -> None:
        base = EngineBase(
            id=BASE_ID,
            archive=Artifact(
                path=f"base/win-cu128/py312-torch2.13-9f2a{suffix}",
                sha256=digest(BASE_ARCHIVE_BYTES),
                size=len(BASE_ARCHIVE_BYTES),
            ),
            python="python/bin/python3",
            packages={"torch": "2.13.0"},
        )
        assert base.archive.path.endswith(suffix)

    def test_engine_base_immutability(self) -> None:
        base = EngineBase(
            id=BASE_ID,
            archive=Artifact(
                path="base/x.tar.gz",
                sha256=digest(BASE_ARCHIVE_BYTES),
                size=len(BASE_ARCHIVE_BYTES),
            ),
            python="python/bin/python3",
            packages={"torch": "2.13.0"},
        )
        with pytest.raises(AttributeError):
            base.python = "other"  # type: ignore
        with pytest.raises(TypeError):
            base.packages["torch"] = "9.9.9"  # type: ignore
