"""Explicit public sources retain git identity and refuse unsafe acquisition."""

import shutil
from urllib.parse import urlsplit

import pytest
from dinkster_supervisor.environment_state import EnvironmentError
from dinkster_supervisor.git_sources import fetched_git_source, git_refs


def test_fetch_ignores_hostile_git_policy_without_system_git(git_server, monkeypatch):
    url = git_server["url"]
    git_server["config"].write_text(
        '[url "https://invalid.invalid/"]\n insteadOf = https://localhost:\n'
        "[http]\n sslVerify = false\n"
    )
    monkeypatch.setenv("PATH", "")
    assert shutil.which("git") is None
    for ref, expected in (
        ("main", 17),
        ("release", 17),
        ("refs/pull/13/head", 43),
        (str(git_server["commits"]["changed"]), 43),
    ):
        with fetched_git_source(f"{url}#ref={ref}&subdirectory=pack") as (folder, origin):
            assert (folder / "node.py").read_text() == f"result = {expected}\n"
            assert origin["commit"] == str(
                git_server["commits"]["main" if expected == 17 else "changed"]
            )
            assert origin["requestedRef"] == ref
            assert origin["subdirectory"] == "pack"
            assert origin["url"] == url
    assert git_server["requests"]


@pytest.mark.parametrize("suffix", ["", ".git"])
def test_explicit_mirror_fetches_locally_and_keeps_original_identity(
    git_server, monkeypatch, suffix
):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("DINKSTER_GIT_MIRROR", git_server["proxy"])
    git_server["config"].write_text(f"[http]\n proxy = {git_server['proxy']}\n")
    before = len(git_server["requests"])
    tunnels = len(git_server["tunnels"])
    url = f"https://github.com/Kosinkadink/source{suffix}"
    with fetched_git_source(f"{url}#ref=changed&subdirectory=pack") as (folder, origin):
        assert (folder / "node.py").read_text() == "result = 43\n"
        assert origin["url"] == url
        assert origin["commit"] == str(git_server["commits"]["changed"])
        assert origin["requestedRef"] == "changed"
    refs = git_refs(url)
    assert refs["refs/heads/main"] == str(git_server["commits"]["main"])
    assert refs["refs/pull/13/head"] == str(git_server["commits"]["changed"])
    assert "refs/tags/release" in refs
    assert len(git_server["requests"]) > before
    assert all(
        urlsplit(path).path.startswith("/Kosinkadink/source.git/")
        for path in git_server["requests"][before:]
    )
    assert len(git_server["tunnels"]) == tunnels


def test_source_refs_cli_uses_git_protocol_without_creating_project(
    git_server, monkeypatch, tmp_path, capsys
):
    import json

    from dinkster.cli import main

    root = tmp_path / "prospective"
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("DINKSTER_GIT_MIRROR", "https://invalid.invalid/mirror")
    assert (
        main(
            [
                "source",
                "--root",
                str(root),
                "refs",
                "https://github.com/Kosinkadink/source",
                "--git-mirror",
                git_server["proxy"],
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "clean"
    assert result["refs"]["refs/heads/changed"] == str(git_server["commits"]["changed"])
    assert result["refs"]["refs/tags/release^{}"] == str(git_server["commits"]["main"])
    assert not root.exists()


def test_failed_mirror_does_not_fall_back_to_github(git_server):
    before = len(git_server["requests"])
    tunnels = len(git_server["tunnels"])
    git_server["config"].write_text(f"[http]\n proxy = {git_server['proxy']}\n")
    with pytest.raises(EnvironmentError, match="cannot fetch"):
        with fetched_git_source(
            "https://github.com/Kosinkadink/source", mirror=git_server["proxy"] + "/missing"
        ):
            pytest.fail("missing mirror repository was accepted")
    assert len(git_server["requests"]) > before
    assert all(
        urlsplit(path).path.startswith("/missing/Kosinkadink/source.git/")
        for path in git_server["requests"][before:]
    )
    assert len(git_server["tunnels"]) == tunnels


def test_mirror_does_not_rewrite_unrelated_public_sources(git_server, monkeypatch):
    monkeypatch.setenv("DINKSTER_GIT_MIRROR", "https://invalid.invalid/mirror")
    with fetched_git_source(git_server["url"]) as (folder, origin):
        assert (folder / "pack/node.py").read_text() == "result = 17\n"
        assert origin["url"] == git_server["url"]


@pytest.mark.parametrize(
    "mirror",
    [
        "http://example.invalid",
        "http://127.attacker.invalid",
        "http://localhost.attacker.invalid",
        "https://user:secret@example.invalid",
        "https://example.invalid?token=secret",
        "https://example.invalid#secret",
        "https://example.invalid/../escape",
        "https://example.invalid/%2e%2e/escape",
    ],
)
def test_mirror_refuses_unsafe_configuration_without_disclosing_credentials(mirror):
    with pytest.raises(EnvironmentError, match="git mirror requires") as error:
        with fetched_git_source("https://github.com/Kosinkadink/source", mirror=mirror):
            pytest.fail("unsafe mirror was accepted")
    assert "secret" not in str(error.value)


def test_fetch_honors_global_http_proxy(git_server):
    git_server["config"].write_text(f"[http]\n proxy = {git_server['proxy']}\n")
    with fetched_git_source(git_server["url"]) as (folder, _origin):
        assert (folder / "pack/node.py").read_text() == "result = 17\n"
    assert git_server["tunnels"]


def test_fetch_refuses_invalid_tls_despite_global_disable(git_server):
    git_server["config"].write_text("[http]\n sslVerify = false\n")
    with pytest.raises(EnvironmentError, match="cannot fetch"):
        with fetched_git_source(git_server["url"].replace("localhost", "127.0.0.1")):
            pytest.fail("certificate for a different host was accepted")


@pytest.mark.parametrize(
    "ref,message",
    [
        ("absent", "cannot fetch"),
        ("lfs", "LFS pointers"),
        ("submodule", "submodules"),
        ("symlink", "symlinks"),
    ],
)
def test_fetch_refuses_unavailable_or_incomplete_sources(git_server, ref, message):
    with pytest.raises(EnvironmentError, match=message):
        with fetched_git_source(f"{git_server['url']}#ref={ref}&subdirectory=pack"):
            pytest.fail("incomplete source was accepted")


def test_fetch_refuses_repository_requiring_credentials(git_server):
    with pytest.raises(EnvironmentError, match="credentials are not supported"):
        with fetched_git_source(git_server["url"].replace("source.git", "private.git")):
            pytest.fail("private repository was accepted")


@pytest.mark.parametrize(
    "value",
    [
        "http://example.invalid/repo",
        "ssh://example.invalid/repo",
        "file:///repo",
        "https://user:secret@example.invalid/repo",
        "https://token@example.invalid/repo",
        "https://example.invalid/repo?token=secret",
        "https://example.invalid/repo#token=secret",
        "https://example.invalid/repo#subdirectory=../escape",
    ],
)
def test_fetch_refuses_unsafe_source_without_disclosing_credentials(value):
    message = (
        "subdirectory must stay within"
        if "subdirectory=" in value
        else "fragment accepts"
        if "#" in value
        else "public HTTPS URL without credentials or query"
    )
    with pytest.raises(EnvironmentError, match=message) as error:
        with fetched_git_source(value):
            pytest.fail("unsafe source was accepted")
    assert "secret" not in str(error.value)
