"""Acquire explicit public git sources without system git or ambient git policy."""

from __future__ import annotations

import ipaddress
import os
import re
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit, urlunsplit

import pygit2

from .environment_state import EnvironmentError

_CONFIG_LOCK = threading.RLock()


def parse_git_source(value: str) -> tuple[str, str | None, str]:
    """HTTPS URL fragments select a ref and a repository-relative subdirectory."""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or "%" in parsed.netloc
    ):
        raise EnvironmentError(
            "git sources require a public HTTPS URL without credentials or query"
        )
    try:
        options = parse_qs(parsed.fragment, strict_parsing=True, keep_blank_values=True)
    except ValueError:
        raise EnvironmentError("git source fragment must use ref=REF&subdirectory=PATH") from None
    if any(
        key not in {"ref", "subdirectory"} or len(values) != 1 for key, values in options.items()
    ):
        raise EnvironmentError("git source fragment accepts one ref and one subdirectory")
    ref = options.get("ref", [None])[0]
    subdirectory = options.get("subdirectory", ["."])[0]
    path = PurePosixPath(subdirectory)
    if (
        not subdirectory
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in subdirectory
        or ":" in subdirectory
        or ".git" in path.parts
    ):
        raise EnvironmentError("git source subdirectory must stay within the repository")
    if ref is not None and (not ref or any(ord(char) < 32 for char in ref)):
        raise EnvironmentError("git source ref must not be empty or contain control characters")
    return urlunsplit(parsed._replace(fragment="")), ref, path.as_posix()


@contextmanager
def _isolated_config() -> Generator[str | None]:
    # libgit2 search paths are process-global; restore them before releasing the lock.
    with _CONFIG_LOCK, TemporaryDirectory(prefix="dinkster-git-config-") as empty:
        proxy = None
        try:
            proxy = pygit2.Config.get_global_config()["http.proxy"]
        except (KeyError, OSError, pygit2.GitError):
            pass
        levels = (
            pygit2.enums.ConfigLevel.PROGRAMDATA,
            pygit2.enums.ConfigLevel.SYSTEM,
            pygit2.enums.ConfigLevel.XDG,
            pygit2.enums.ConfigLevel.GLOBAL,
        )
        previous = {level: pygit2.settings.search_path[level] for level in levels}
        try:
            for level in levels:
                pygit2.settings.search_path[level] = empty
            yield proxy
        finally:
            for level, path in previous.items():
                pygit2.settings.search_path[level] = path


def _check_tree(tree: pygit2.Tree) -> None:
    for entry in tree:
        if entry.filemode == pygit2.GIT_FILEMODE_LINK:
            raise EnvironmentError("git development sources with symlinks are not supported")
        if entry.filemode == pygit2.GIT_FILEMODE_COMMIT:
            raise EnvironmentError("git development sources with submodules are not supported")
        if isinstance(entry, pygit2.Tree):
            _check_tree(entry)
        elif isinstance(entry, pygit2.Blob):
            if entry.name == ".gitmodules":
                raise EnvironmentError("git development sources with submodules are not supported")
            if entry.data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise EnvironmentError(
                    "git development sources with LFS pointers are not supported"
                )


def _fetch_url(url: str, mirror: str | None) -> str:
    mirror = os.environ.get("DINKSTER_GIT_MIRROR") if mirror is None else mirror
    if not mirror:
        return url
    parsed = urlsplit(mirror)
    try:
        loopback = (
            parsed.hostname == "localhost"
            or ipaddress.ip_address(parsed.hostname or "").is_loopback
        )
    except ValueError:
        loopback = False
    if (
        parsed.scheme != "https"
        and not (parsed.scheme == "http" and loopback)
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "%" in parsed.netloc
        or any(part in {".", ".."} for part in parsed.path.split("/"))
        or "%" in parsed.path
        or "\\" in parsed.path
    ):
        raise EnvironmentError("git mirror requires credential-free HTTPS or loopback HTTP")
    original = urlsplit(url)
    parts = original.path.strip("/").split("/")
    if (
        original.hostname != "github.com"
        or original.port not in {None, 443}
        or len(parts) != 2
        or parts[0].lower() != "kosinkadink"
    ):
        return url
    repository = parts[1].removesuffix(".git")
    if re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", repository) is None or repository in {
        ".",
        "..",
    }:
        raise EnvironmentError("git mirror requires a valid GitHub repository name")
    return mirror.rstrip("/") + "/Kosinkadink/" + repository + ".git"


def git_refs(value: str, *, mirror: str | None = None) -> dict[str, str]:
    """List advertised refs through git, without checking out or executing source code."""
    url, _ref, _subdirectory = parse_git_source(value)
    with TemporaryDirectory(prefix="dinkster-git-refs-") as directory, _isolated_config() as proxy:
        repository = pygit2.init_repository(directory, bare=True)
        try:
            remote = repository.remotes.create("origin", _fetch_url(url, mirror))
            repository.config["http.followRedirects"] = "false"
            callbacks = pygit2.RemoteCallbacks(certificate_check=lambda _cert, valid, _host: valid)
            return {
                row.name: str(row.oid)
                for row in remote.list_heads(callbacks=callbacks, proxy=proxy or True)
                if row.name is not None
            }
        except (pygit2.GitError, KeyError, ValueError):
            raise EnvironmentError(
                "cannot list public git refs (credentials are not supported)"
            ) from None
        finally:
            repository.free()


@contextmanager
def fetched_git_source(
    value: str, *, mirror: str | None = None, include_tags: bool = False
) -> Generator[tuple[Path, dict[str, object]]]:
    """Yield a checked-out selected folder and its credential-free provenance."""
    url, ref, subdirectory = parse_git_source(value)
    with TemporaryDirectory(prefix="dinkster-git-") as directory:
        checkout = Path(directory) / "checkout"
        with _isolated_config() as proxy:
            callbacks = pygit2.RemoteCallbacks(certificate_check=lambda _cert, valid, _host: valid)
            repository = None
            try:
                repository = pygit2.init_repository(checkout)
                remote = repository.remotes.create("origin", _fetch_url(url, mirror))
                repository.config["http.followRedirects"] = "false"
                if proxy is not None:
                    repository.config["http.proxy"] = proxy
                advertised = {
                    row.name: row.oid
                    for row in remote.list_heads(callbacks=callbacks, proxy=proxy or True)
                    if row.name is not None
                }
                if ref is None:
                    selected_ref = "HEAD"
                elif re.fullmatch(r"[0-9a-fA-F]{40}", ref):
                    selected_ref = ref.lower()
                else:
                    selected_ref = next(
                        name
                        for name in (f"refs/heads/{ref}", f"refs/tags/{ref}", ref)
                        if name in advertised
                    )
                refspecs = (
                    [f"+{selected_ref}:refs/dinkster/selected"]
                    if selected_ref in advertised
                    else ["+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*"]
                )
                if include_tags and "+refs/tags/*:refs/tags/*" not in refspecs:
                    refspecs.append("+refs/tags/*:refs/tags/*")
                remote.fetch(refspecs, callbacks=callbacks, proxy=proxy or True)
                target = advertised.get(selected_ref, selected_ref)
                if target not in repository:
                    remote.fetch(
                        [f"+{selected_ref}:refs/dinkster/selected"],
                        callbacks=callbacks,
                        proxy=proxy or True,
                    )
                commit = repository[target].peel(pygit2.Commit)
                tree = commit.tree
                if subdirectory != ".":
                    tree = tree[subdirectory]
                if not isinstance(tree, pygit2.Tree):
                    raise EnvironmentError("git source subdirectory is not a folder")
                _check_tree(tree)
                if os.name == "nt":
                    repository.config["core.symlinks"] = False
                repository.set_head(commit.id)
                repository.reset(commit.id, pygit2.enums.ResetMode.HARD)
            except (pygit2.GitError, KeyError, ValueError, StopIteration) as error:
                if repository is not None:
                    repository.free()
                if isinstance(error, EnvironmentError):
                    raise
                raise EnvironmentError(
                    "cannot fetch public git source or resolve its ref/subdirectory; "
                    "check the public HTTPS URL and ref (credentials are not supported)"
                ) from None
        try:
            yield (
                checkout / subdirectory,
                {
                    "url": url,
                    "requestedRef": ref,
                    "commit": str(commit.id),
                    "subdirectory": subdirectory,
                },
            )
        finally:
            repository.free()
