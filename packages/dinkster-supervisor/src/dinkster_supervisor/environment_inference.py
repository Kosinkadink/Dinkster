"""Resolve explicit inference requests using published wheels and source ancestry."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlsplit

import pygit2
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .environment_artifacts import WheelLinks
from .environment_downloads import read_metadata
from .environment_state import EnvironmentError
from .git_sources import fetched_git_source

INFERENCE_REPOSITORY = "https://github.com/Kosinkadink/dinkster-inference.git"


def resolve_inference(request: str, package_index: str | None = None) -> dict[str, Any]:
    """An unavailable prerelease needs a retained tag; version order is not ancestry."""
    commit_request = re.fullmatch(r"[0-9a-fA-F]{40}", request) is not None
    requested_version = None
    if not commit_request:
        try:
            requested_version = Version(request)
        except InvalidVersion as error:
            raise EnvironmentError(
                "inference request must be an exact version or full commit"
            ) from error
    links = WheelLinks()
    links.feed(
        read_metadata(package_index or "https://pypi.org/simple", "dinkster-inference").decode(
            "utf-8"
        )
    )
    published: set[Version] = set()
    for link in links.links:
        if link in links.yanked:
            continue
        filename = unquote(urlsplit(link).path.rsplit("/", 1)[-1])
        try:
            name, version, _, _ = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        if name == "dinkster-inference":
            published.add(version)
    with fetched_git_source(INFERENCE_REPOSITORY, include_tags=True) as (folder, _):
        repository = pygit2.Repository(folder / ".git")
        try:
            releases: list[tuple[Version, pygit2.Commit]] = []
            for reference in repository.references:
                if not reference.startswith("refs/tags/v"):
                    continue
                try:
                    version = Version(reference.removeprefix("refs/tags/v"))
                except InvalidVersion:
                    continue
                target = repository[repository.references[reference].target]
                commit = target.peel(pygit2.Commit)
                releases.append((version, commit))
            if commit_request:
                try:
                    requested_commit = repository[request.lower()].peel(pygit2.Commit)
                except (KeyError, ValueError, pygit2.GitError) as error:
                    raise EnvironmentError(
                        "inference commit is unknown; no containing release can be proved"
                    ) from error
            else:
                matching = [row for row in releases if row[0] == requested_version]
                if not matching:
                    raise EnvironmentError(
                        "inference version has no source tag; no containing release can be proved"
                    )
                requested_commit = matching[0][1]
            selected = None
            if requested_version in published:
                selected = next(row for row in releases if row[0] == requested_version)
            elif commit_request:
                exact = [
                    row
                    for row in releases
                    if row[0].is_prerelease
                    and row[0] in published
                    and row[1].id == requested_commit.id
                ]
                if exact:
                    selected = max(exact, key=lambda row: row[0])
            fallback = selected is None
            if selected is None:
                if requested_version is not None and not requested_version.is_prerelease:
                    raise EnvironmentError(
                        f"inference release {request} is not available from the package index"
                    )
                containing = [
                    row
                    for row in releases
                    if not row[0].is_prerelease
                    and not row[0].is_devrelease
                    and row[0] in published
                    and (
                        row[1].id == requested_commit.id
                        or repository.descendant_of(row[1].id, requested_commit.id)
                    )
                ]
                if not containing:
                    raise EnvironmentError(
                        f"inference {request} is unavailable and no published tagged release "
                        "contains its changes"
                    )
                selected = min(containing, key=lambda row: row[0])
            version, commit = selected
            notice = (
                f"inference {request} has no available prerelease; "
                f"using oldest containing tagged release {version}"
                if fallback
                else f"inference {request} selects published version {version}"
            )
            return {
                "requested": request,
                "requestedCommit": str(requested_commit.id),
                "version": str(version),
                "sourceCommit": str(commit.id),
                "repository": INFERENCE_REPOSITORY,
                "fallback": fallback,
                "notice": notice,
            }
        finally:
            repository.free()
