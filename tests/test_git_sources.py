"""Explicit public sources retain git identity and refuse unsafe acquisition."""

import shutil

import pytest
from dinkster_supervisor.environment_state import EnvironmentError
from dinkster_supervisor.git_sources import fetched_git_source


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
    [("absent", "cannot fetch"), ("lfs", "LFS pointers"), ("submodule", "submodules")],
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
