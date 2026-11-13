"""Public development sources use verified HTTPS independently of system git policy."""

from __future__ import annotations

import os
import select
import shutil
import socket
import ssl
import subprocess
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pygit2
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


@pytest.fixture(scope="session")
def git_server(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("git-server")
    monkeypatch = pytest.MonkeyPatch()
    git = shutil.which("git")
    assert git is not None, "fixture server needs git http-backend; the client must not use git"
    repository = pygit2.init_repository(tmp_path / "source.git", bare=True)
    signature = pygit2.Signature("Fixture", "fixture@example.invalid", 1700000000, 0)
    commits = {}
    for ref, text, mode in (
        ("main", b"result = 17\n", pygit2.GIT_FILEMODE_BLOB),
        ("changed", b"result = 43\n", pygit2.GIT_FILEMODE_BLOB),
        (
            "lfs",
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 8\n",
            pygit2.GIT_FILEMODE_BLOB,
        ),
        ("submodule", None, pygit2.GIT_FILEMODE_COMMIT),
    ):
        package = repository.TreeBuilder()
        package.insert("node.py", repository.create_blob(text) if text else commits["main"], mode)
        tree = repository.TreeBuilder()
        tree.insert("pack", package.write(), pygit2.GIT_FILEMODE_TREE)
        commits[ref] = repository.create_commit(
            f"refs/heads/{ref}", signature, signature, ref, tree.write(), []
        )
    repository.set_head("refs/heads/main")
    repository.create_tag(
        "release", commits["main"], pygit2.enums.ObjectType.COMMIT, signature, "release"
    )
    repository.references.create("refs/pull/13/head", commits["changed"])
    repository.free()

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(days=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert = tmp_path / "certificate.pem"
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    private = tmp_path / "private.pem"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    requests = []
    tunnels = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            requests.append(self.path)
            if self.path.startswith("/private.git"):
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="private"')
                self.end_headers()
                return
            parsed = urlsplit(self.path)
            environment = {
                **os.environ,
                "GIT_PROJECT_ROOT": str(tmp_path),
                "GIT_HTTP_EXPORT_ALL": "1",
                "PATH_INFO": parsed.path,
                "QUERY_STRING": parsed.query,
                "REQUEST_METHOD": self.command,
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                "REMOTE_USER": "",
            }
            response = subprocess.run(
                [git, "http-backend"],
                input=self.rfile.read(int(environment["CONTENT_LENGTH"])),
                capture_output=True,
                env=environment,
                check=True,
            ).stdout
            headers, _, body = response.partition(b"\r\n\r\n")
            self.send_response(200)
            for header in headers.decode().splitlines():
                key, value = header.split(":", 1)
                self.send_header(key, value.strip())
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_CONNECT(self):
            tunnels.append(self.path)
            with socket.create_connection(
                ("localhost", server.server_port), timeout=10
            ) as upstream:
                self.send_response(200)
                self.end_headers()
                peers = [self.connection, upstream]
                while True:
                    ready, _, _ = select.select(peers, [], [], 10)
                    if not ready:
                        return
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (upstream if source is self.connection else self.connection).sendall(data)

    server = ThreadingHTTPServer(("localhost", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, private)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    proxy = ThreadingHTTPServer(("localhost", 0), Handler)
    threads = [threading.Thread(target=item.serve_forever, daemon=True) for item in (server, proxy)]
    for thread in threads:
        thread.start()
    config = tmp_path / "config"
    config.mkdir()
    levels = (
        pygit2.enums.ConfigLevel.GLOBAL,
        pygit2.enums.ConfigLevel.XDG,
        pygit2.enums.ConfigLevel.SYSTEM,
    )
    previous = {level: pygit2.settings.search_path[level] for level in levels}
    for level in levels:
        pygit2.settings.search_path[level] = str(config)
    monkeypatch.setattr(pygit2.settings, "ssl_cert_file", str(cert))
    for variable in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(variable, raising=False)
    try:
        yield {
            "url": f"https://localhost:{server.server_port}/source.git",
            "root": tmp_path,
            "proxy": f"http://localhost:{proxy.server_port}",
            "config": config / ".gitconfig",
            "commits": commits,
            "requests": requests,
            "tunnels": tunnels,
        }
    finally:
        for level, path in previous.items():
            pygit2.settings.search_path[level] = path
        for item in (server, proxy):
            item.shutdown()
            item.server_close()
        for thread in threads:
            thread.join()
        monkeypatch.undo()
