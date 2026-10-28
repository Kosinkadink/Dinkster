from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from dinkster_supervisor.environment_downloads import DownloadError, fetch_artifact


@contextmanager
def artifact_server(
    body: bytes,
    *,
    range_mode: str = "honor",
    interrupted: bool = False,
    redirect: str | None = None,
) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if redirect is not None:
                self.send_response(302)
                self.send_header("Location", redirect)
                self.end_headers()
                return
            start = 0
            range_header = self.headers.get("Range")
            if range_mode == "honor" and range_header:
                start = int(range_header.removeprefix("bytes=").removesuffix("-"))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            elif range_mode == "wrong-offset" and range_header:
                self.send_response(206)
                self.send_header("Content-Range", f"bytes 0-{len(body) - 1}/{len(body)}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            response = body[start:]
            self.wfile.write(response[: len(response) // 2] if interrupted else response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_falls_back_after_corrupt_local_mirror(tmp_path: Path) -> None:
    expected = b"correct artifact"
    bad = tmp_path / "bad"
    good = tmp_path / "good"
    (bad / "packs").mkdir(parents=True)
    (good / "packs").mkdir(parents=True)
    (bad / "packs" / "demo.bin").write_bytes(b"corrupt artifact")
    (good / "packs" / "demo.bin").write_bytes(expected)

    result = fetch_artifact(
        (str(bad), str(good)),
        "packs/demo.bin",
        tmp_path / "cache" / "demo.bin",
        digest(expected),
        len(expected),
    )

    assert result.read_bytes() == expected


def test_ignored_range_restarts_safely(tmp_path: Path) -> None:
    data = b"range ignored by server"
    destination = tmp_path / "artifact.bin"
    destination.with_name("artifact.bin.part").write_bytes(data[:5])
    with artifact_server(data, range_mode="ignore") as mirror:
        fetch_artifact((mirror,), "artifact.bin", destination, digest(data), len(data))
    assert destination.read_bytes() == data


def test_resumes_http_partial_from_exact_offset(tmp_path: Path) -> None:
    data = b"resume from this exact byte offset"
    destination = tmp_path / "artifact.bin"
    destination.with_name("artifact.bin.part").write_bytes(data[:9])
    with artifact_server(data) as mirror:
        fetch_artifact((mirror,), "artifact.bin", destination, digest(data), len(data))
    assert destination.read_bytes() == data


def test_corrupt_partial_retries_the_same_healthy_mirror_from_zero(tmp_path: Path) -> None:
    data = b"a valid suffix cannot repair a corrupt saved prefix"
    destination = tmp_path / "artifact.bin"
    destination.with_name("artifact.bin.part").write_bytes(b"wrong prefix")
    with artifact_server(data) as mirror:
        fetch_artifact((mirror,), "artifact.bin", destination, digest(data), len(data))
    assert destination.read_bytes() == data
    assert not destination.with_name("artifact.bin.part").exists()


def test_rejects_wrong_http_resume_offset(tmp_path: Path) -> None:
    data = b"resume needs exact content range"
    destination = tmp_path / "artifact.bin"
    destination.with_name("artifact.bin.part").write_bytes(data[:4])
    with artifact_server(data, range_mode="wrong-offset") as mirror:
        with pytest.raises(DownloadError, match="invalid Content-Range"):
            fetch_artifact((mirror,), "artifact.bin", destination, digest(data), len(data))
    assert destination.with_name("artifact.bin.part").read_bytes() == data[:4]


def test_short_local_response_falls_back_and_cached_corruption_is_replaced(tmp_path: Path) -> None:
    data = b"complete artifact"
    short = tmp_path / "short"
    good = tmp_path / "good"
    for mirror in (short, good):
        (mirror / "artifact").mkdir(parents=True)
    (short / "artifact" / "demo.bin").write_bytes(data[:4])
    (good / "artifact" / "demo.bin").write_bytes(data)
    destination = tmp_path / "cache" / "demo.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"not cached correctly")

    fetch_artifact(
        (str(short), str(good)), "artifact/demo.bin", destination, digest(data), len(data)
    )

    assert destination.read_bytes() == data


def test_interrupted_http_response_retries_next_mirror(tmp_path: Path) -> None:
    data = b"an interrupted transfer still retries the next mirror"
    destination = tmp_path / "artifact.bin"
    with artifact_server(data, interrupted=True) as interrupted, artifact_server(data) as good:
        fetch_artifact((interrupted, good), "artifact.bin", destination, digest(data), len(data))
    assert destination.read_bytes() == data


@pytest.mark.parametrize(
    "relative_path",
    ("../artifact.bin", "/artifact.bin", "a/../../artifact.bin", "a/./file", "a//file", "a/\nfile"),
)
def test_rejects_traversal_paths(tmp_path: Path, relative_path: str) -> None:
    with pytest.raises(DownloadError, match="unsafe artifact path"):
        fetch_artifact((str(tmp_path),), relative_path, tmp_path / "artifact.bin", "0" * 64, 0)


def test_local_paths_with_spaces_and_complete_partials_work_without_source(tmp_path: Path) -> None:
    mirror = tmp_path / "local mirror"
    mirror.mkdir()
    data = b"local directories are not remote URLs"
    (mirror / "artifact.bin").write_bytes(data)
    destination = tmp_path / "artifact.bin"
    fetch_artifact((str(mirror),), "artifact.bin", destination, digest(data), len(data))
    assert destination.read_bytes() == data
    destination.rename(destination.with_name("artifact.bin.part"))
    (mirror / "artifact.bin").unlink()
    fetch_artifact((str(mirror),), "artifact.bin", destination, digest(data), len(data))
    assert destination.read_bytes() == data


def test_rejects_external_plain_http_without_exposing_credentials(tmp_path: Path) -> None:
    with pytest.raises(DownloadError) as failure:
        fetch_artifact(
            ("http://private-user:private-password@example.org",),
            "artifact.bin",
            tmp_path / "artifact.bin",
            digest(b"data"),
            4,
        )
    assert "unsafe mirror URL" in str(failure.value)
    assert "private-password" not in str(failure.value)
    assert "private-user" not in str(failure.value)
    assert not (tmp_path / "artifact.bin").exists()


def test_rejects_redirect_to_non_tls_external_origin(tmp_path: Path) -> None:
    with artifact_server(b"", redirect="http://example.org/artifact.bin") as mirror:
        with pytest.raises(DownloadError, match="HTTPS or loopback HTTP"):
            fetch_artifact((mirror,), "artifact.bin", tmp_path / "artifact.bin", digest(b"data"), 4)
    assert not (tmp_path / "artifact.bin").exists()


def test_accepts_integrity_checked_signed_query_redirect(tmp_path: Path) -> None:
    data = b"artifact bytes remain authoritative after a signed redirect"
    destination = tmp_path / "artifact.bin"
    with artifact_server(data) as target:
        with artifact_server(b"", redirect=f"{target}/artifact.bin?signature=fixture") as mirror:
            fetch_artifact((mirror,), "artifact.bin", destination, digest(data), len(data))
    assert destination.read_bytes() == data
