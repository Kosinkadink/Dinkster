"""Measure the native trust lifecycle of one generated Windows fixture certificate."""

from __future__ import annotations

import ctypes
import os
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pygit2
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def exercise_https(certificate: x509.Certificate, key: rsa.RSAPrivateKey) -> None:
    git = shutil.which("git")
    assert git is not None
    with tempfile.TemporaryDirectory(prefix="dinkster-native-trust-") as temporary:
        root = Path(temporary)
        cert = root / "certificate.pem"
        cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        private = root / "private.pem"
        private.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        repository = pygit2.init_repository(root / "source.git", bare=True)
        signature = pygit2.Signature("Fixture", "fixture@example.invalid")
        expected = repository.create_commit(
            "refs/heads/main",
            signature,
            signature,
            "fixture",
            repository.TreeBuilder().write(),
            [],
        )
        repository.set_head("refs/heads/main")
        repository.free()
        requests: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                requests.append(self.path)
                parsed = urlsplit(self.path)
                response = subprocess.run(
                    [git, "http-backend"],
                    input=self.rfile.read(int(self.headers.get("Content-Length", "0"))),
                    capture_output=True,
                    check=True,
                    env={
                        **os.environ,
                        "GIT_PROJECT_ROOT": str(root),
                        "GIT_HTTP_EXPORT_ALL": "1",
                        "PATH_INFO": parsed.path,
                        "QUERY_STRING": parsed.query,
                        "REQUEST_METHOD": self.command,
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                        "REMOTE_USER": "",
                    },
                ).stdout
                headers, separator, body = response.partition(b"\r\n\r\n")
                assert separator
                self.send_response(200)
                for header in headers.decode().splitlines():
                    name, value = header.split(":", 1)
                    self.send_header(name, value.strip())
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_POST = do_GET

        server = ThreadingHTTPServer(("localhost", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, private)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            started = time.monotonic()
            clone = pygit2.clone_repository(
                f"https://localhost:{server.server_port}/source.git", root / "valid"
            )
            assert clone.head.target == expected
            clone.free()
            assert requests
            print("Verified valid HTTPS clone; seconds:", time.monotonic() - started, flush=True)
            before = len(requests)
            try:
                unexpected = pygit2.clone_repository(
                    f"https://127.0.0.1:{server.server_port}/source.git", root / "invalid-host"
                )
            except pygit2.GitError as error:
                assert any(
                    word in str(error).lower() for word in ("ssl", "certificate", "hostname")
                )
                assert len(requests) == before
                print("Invalid-host TLS refused before HTTP:", str(error), flush=True)
            else:
                unexpected.free()
                raise AssertionError("Invalid-host clone unexpectedly succeeded")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def main() -> None:
    if (
        os.name != "nt"
        or os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
    ):
        raise SystemExit("This probe requires a disposable GitHub Windows runner")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, f"Dinkster fixture probe {uuid.uuid4()}")]
    )
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    encoded = certificate.public_bytes(serialization.Encoding.DER)
    Path("probe-public-certificate.der").write_bytes(encoded)
    print("Generated certificate SHA1:", certificate.fingerprint(hashes.SHA1()).hex(), flush=True)
    assert not any(data == encoded for data, _, _ in ssl.enum_certificates("ROOT"))

    api = ctypes.WinDLL("crypt32", use_last_error=True)
    api.CertOpenStore.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_wchar_p,
    ]
    api.CertOpenStore.restype = ctypes.c_void_p
    api.CertAddEncodedCertificateToStore.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    api.CertAddEncodedCertificateToStore.restype = ctypes.c_int
    api.CertDeleteCertificateFromStore.argtypes = [ctypes.c_void_p]
    api.CertDeleteCertificateFromStore.restype = ctypes.c_int
    api.CertCloseStore.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    api.CertCloseStore.restype = ctypes.c_int

    # SYSTEM_W, LOCAL_MACHINE and OPEN_EXISTING select the disposable runner's store.
    store = api.CertOpenStore(ctypes.c_void_p(10), 0, None, 0x20000 | 0x4000, "ROOT")
    if not store:
        raise ctypes.WinError(ctypes.get_last_error())
    context = ctypes.c_void_p()
    started = time.monotonic()
    try:
        print("Adding unique certificate to LocalMachine Root through CryptoAPI", flush=True)
        buffer = ctypes.create_string_buffer(encoded)
        # ADD_NEW refuses a match instead of replacing any existing certificate.
        if not api.CertAddEncodedCertificateToStore(
            store, 1, buffer, len(encoded), 1, ctypes.byref(context)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        print("Add returned after seconds:", time.monotonic() - started, flush=True)
        assert any(data == encoded for data, _, _ in ssl.enum_certificates("ROOT"))
        print("Generated certificate is present", flush=True)
        exercise_https(certificate, key)
    finally:
        try:
            if context.value:
                print("Deleting only returned certificate context", flush=True)
                if not api.CertDeleteCertificateFromStore(context):
                    raise ctypes.WinError(ctypes.get_last_error())
        finally:
            if not api.CertCloseStore(store, 0):
                raise ctypes.WinError(ctypes.get_last_error())
    assert not any(data == encoded for data, _, _ in ssl.enum_certificates("ROOT"))
    print(
        "Generated certificate is absent; lifecycle seconds:",
        time.monotonic() - started,
        flush=True,
    )


if __name__ == "__main__":
    main()
