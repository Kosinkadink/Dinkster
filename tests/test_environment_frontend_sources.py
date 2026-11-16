"""Explicit Git frontend selection verifies CI identity and retains built assets."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from http.client import HTTPMessage
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler

import pytest
from dinkster_supervisor import environment_frontend_sources as frontend
from dinkster_supervisor.environment_state import EnvironmentError


def _checkout(root: Path) -> Path:
    root.mkdir()
    (root / "package.json").write_text('{"version":"0.2.0"}')
    (root / "frontend-release.json").write_text('{"engineApi":{"min":1,"max":2}}')
    return root


def _artifact(root: Path, commit: str, extra: str = "assets/app.js") -> dict:
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, data in {
            "index.html": b"<main>selected frontend</main>",
            extra: b"marker=73",
        }.items():
            entry = zipfile.ZipInfo(name)
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    payload = bundle.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    path = root / "fixture.zip"
    with zipfile.ZipFile(path, "w") as transport:
        transport.writestr("bundle.zip", payload)
        transport.writestr(
            "manifest.json",
            json.dumps(
                {"commit": commit, "version": "0.2.0", "sha256": digest, "size": len(payload)}
            ),
        )
    return {
        "id": 11,
        "name": "dinkster-frontend-" + digest,
        "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_in_bytes": path.stat().st_size,
        "run": 12,
        "path": path,
    }


def test_ci_assets_need_no_node_and_wrong_commit_is_not_a_build_fallback(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path / "checkout")
    origin = {"commit": "a" * 40}
    artifact = _artifact(tmp_path, origin["commit"])

    def fetch(roots, name, destination, sha, size):
        assert sha == artifact["digest"][7:] and size == artifact["size_in_bytes"]
        destination.write_bytes(artifact["path"].read_bytes())

    monkeypatch.setattr(frontend, "fetch_artifact", fetch)
    monkeypatch.setattr(frontend.shutil, "which", lambda name: None)
    wheel, build = frontend.prepare_frontend_wheel(checkout, origin, tmp_path / "result", artifact)
    assert build["kind"] == "ci" and build["artifact"] == 11
    with zipfile.ZipFile(wheel) as archive:
        assert (
            archive.read("dinkster_frontend/static/index.html") == b"<main>selected frontend</main>"
        )
        assert archive.read("dinkster_frontend/static/assets/app.js") == b"marker=73"
        metadata = json.loads(archive.read("dinkster_frontend/manifest.json"))
        assert metadata["commit"] == origin["commit"] and metadata["engineApi"] == {
            "min": 1,
            "max": 2,
        }
    with pytest.raises(EnvironmentError, match="identity differs"):
        frontend.prepare_frontend_wheel(
            checkout, {"commit": "b" * 40}, tmp_path / "wrong", artifact
        )


@pytest.mark.parametrize(
    "entry", ["../outside", "assets/../app.js", "assets//app.js", "a\\\\b", "C:escape"]
)
def test_verified_ci_bundle_still_refuses_unsafe_entries(tmp_path, monkeypatch, entry):
    checkout = _checkout(tmp_path / "checkout")
    origin = {"commit": "a" * 40}
    artifact = _artifact(tmp_path, origin["commit"], entry)

    def fetch(roots, name, destination, sha, size):
        destination.write_bytes(artifact["path"].read_bytes())

    monkeypatch.setattr(frontend, "fetch_artifact", fetch)
    with pytest.raises(EnvironmentError, match="unsafe frontend bundle entry"):
        frontend.prepare_frontend_wheel(checkout, origin, tmp_path / "result", artifact)
    assert not list((tmp_path / "result").glob("*.whl"))


@pytest.mark.parametrize("missing", [("node",), ("pnpm",), ("node", "pnpm")])
def test_local_fallback_names_missing_tools_without_running_build(tmp_path, monkeypatch, missing):
    checkout = _checkout(tmp_path / "checkout")
    monkeypatch.setattr(
        frontend.shutil, "which", lambda name: None if name in missing else "/tools/" + name
    )
    with pytest.raises(EnvironmentError, match="local build requires " + " and ".join(missing)):
        frontend.prepare_frontend_wheel(checkout, {"commit": "a" * 40}, tmp_path / "result", None)


@pytest.mark.parametrize(
    "mismatch", ["head_sha", "conclusion", "path", "event", "repository", "expired", "valid"]
)
def test_ci_selection_does_not_use_another_commit_or_untrusted_run(monkeypatch, mismatch):
    origin = {"url": "https://github.com/Kosinkadink/Dinkster-Frontend", "commit": "a" * 40}
    run = {
        "id": 12,
        "head_sha": origin["commit"],
        "conclusion": "success",
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_repository": {"full_name": "Kosinkadink/Dinkster-Frontend"},
    }
    artifact = {
        "id": 11,
        "expired": False,
        "name": "dinkster-frontend-" + "b" * 64,
        "digest": "sha256:" + "c" * 64,
    }
    if mismatch == "repository":
        run["head_repository"] = {"full_name": "another/fork"}
    elif mismatch == "expired":
        artifact["expired"] = True
    elif mismatch != "valid":
        run[mismatch] = "wrong"
    monkeypatch.setattr(
        frontend,
        "_github_json",
        lambda path: (
            {"artifacts": [artifact]} if "/artifacts?" in path else {"workflow_runs": [run]}
        ),
    )
    selected = frontend.frontend_ci_artifact(origin)
    if mismatch == "valid":
        assert selected == {**artifact, "run": 12}
    else:
        assert selected is None


@pytest.mark.parametrize("token", [None, "", "fixture-job-token"])
def test_ci_metadata_authentication_stays_on_github_requests(monkeypatch, token):
    if token is None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GITHUB_TOKEN", token)
    commit = "a" * 40
    run = {
        "id": 73,
        "head_sha": commit,
        "conclusion": "success",
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_repository": {"full_name": "Kosinkadink/Dinkster-Frontend"},
    }
    artifact = {
        "id": 97,
        "expired": False,
        "name": "dinkster-frontend-" + "b" * 64,
        "digest": "sha256:" + "c" * 64,
    }
    requests = []

    def read(request, *, timeout):
        requests.append(request)
        payload = (
            {"artifacts": [artifact]}
            if "/artifacts?" in request.full_url
            else {"workflow_runs": [run]}
        )
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(frontend, "urlopen", read)
    assert frontend.frontend_ci_artifact(
        {"url": "https://github.com/Kosinkadink/Dinkster-Frontend.git", "commit": commit}
    ) == {**artifact, "run": 73}
    assert len(requests) == 2
    for request in requests:
        assert request.get_header("Authorization") == (f"Bearer {token}" if token else None)
        redirected = HTTPRedirectHandler().redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            HTTPMessage(),
            "https://artifact.example/bundle.zip",
        )
        assert redirected is not None and redirected.get_header("Authorization") is None


@pytest.mark.parametrize("endpoint", ["runs", "artifacts"])
def test_ci_lookup_failure_identifies_request_without_disclosing_token(monkeypatch, endpoint):
    monkeypatch.setenv("GITHUB_TOKEN", "fixture-private-token")
    commit = "d" * 40
    base = "https://api.github.com/repos/Kosinkadink/Dinkster-Frontend/actions/runs"
    url = (
        f"{base}?head_sha={commit}&per_page=100&page=1"
        if endpoint == "runs"
        else f"{base}/73/artifacts?per_page=100&page=1"
    )

    def read(request, *, timeout):
        if request.full_url == url:
            raise HTTPError(url, 403, "rate limit exceeded", HTTPMessage(), None)
        return io.BytesIO(
            json.dumps(
                {
                    "workflow_runs": [
                        {
                            "id": 73,
                            "head_sha": commit,
                            "conclusion": "success",
                            "path": ".github/workflows/ci.yml",
                            "event": "push",
                            "head_repository": {"full_name": "Kosinkadink/Dinkster-Frontend"},
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(frontend, "urlopen", read)
    with pytest.raises(EnvironmentError) as failure:
        frontend.frontend_ci_artifact(
            {"url": "https://github.com/Kosinkadink/Dinkster-Frontend.git", "commit": commit}
        )
    assert url in str(failure.value) and "HTTP 403" in str(failure.value)
    assert "fixture-private-token" not in str(failure.value)


def test_local_fallback_packages_built_files(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path / "checkout")
    monkeypatch.setattr(frontend.shutil, "which", lambda name: "/tools/" + name)

    def build(command, *, cwd):
        if command[-1] == "build":
            dist = cwd / "packages/app/dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text("<main>local ref</main>")

    monkeypatch.setattr(frontend, "run_local_command", build)
    wheel, selection = frontend.prepare_frontend_wheel(
        checkout, {"commit": "a" * 40}, tmp_path / "result", None
    )
    assert selection == {"kind": "local"}
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read("dinkster_frontend/static/index.html") == b"<main>local ref</main>"
