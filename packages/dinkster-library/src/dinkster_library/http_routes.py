"""The HTTP shape of the library routes the editor and the projects page read.

The engine and the always-on Dinkster process both answer ``/api/projects``,
a project's workflow list (``GET /api/library``) and its latest outputs
through the library helper; the process does so while no engine is open. Both
translate requests and helper errors here, so the two answer alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

__all__ = ["ERROR_STATUS", "library_list_call", "outputs_call", "project_call"]

ERROR_STATUS: Mapping[str, int] = {"stale": 409, "missing": 404, "invalid": 400}
"""HTTP status for a helper error code; any other code is 503."""


def project_call(
    method: str, identifier: str | None, query: Mapping[str, str], body: Mapping[str, object]
) -> tuple[str, dict[str, object]]:
    """The helper operation and arguments for one ``/api/projects`` request.

    Raises ValueError, answered as 400, for a request the helper must not see.
    """
    arguments: dict[str, object] = {} if identifier is None else {"id": identifier}
    if method == "GET":
        if identifier is not None:
            return "projects.get", arguments
        arguments.update(query)
        if "limit" in arguments:
            arguments["limit"] = int(cast(str, arguments["limit"]))
        if "includeDeleted" in arguments:
            if arguments["includeDeleted"] not in ("true", "false"):
                raise ValueError("includeDeleted must be true or false")
            arguments["includeDeleted"] = arguments["includeDeleted"] == "true"
        return "projects.list", arguments
    patch = body.get("patch")
    if isinstance(patch, dict) and "accountProject" in patch:
        raise ValueError("use-account-project-link-route")
    if method == "POST":
        return "projects.register", dict(body)
    operation = "projects.delete" if method == "DELETE" else "projects.patch"
    return operation, {**body, "id": identifier}


def library_list_call(query: Mapping[str, str]) -> dict[str, object]:
    """``records.list`` arguments for ``GET /api/library``; ValueError answers 400."""
    scope = query.get("scope", "").strip()
    if not scope:
        raise ValueError("scope is required (single-user: 'local')")
    try:
        limit = min(200, max(1, int(query.get("limit", "50"))))
    except ValueError:
        raise ValueError("limit must be an integer") from None
    arguments: dict[str, object] = {
        "scope": scope,
        "q": query.get("q", ""),
        "label": query.get("label", ""),
        "limit": limit,
    }
    for key in ("project", "mediaType", "cursor"):
        if key in query:
            arguments[key] = query[key]
    return arguments


def outputs_call(identifier: str, query: Mapping[str, str]) -> dict[str, object]:
    """``assets.list`` arguments for ``GET /api/projects/{id}/outputs``: what the
    project's runs produced, newest first."""
    try:
        limit = min(200, max(1, int(query.get("limit", "24"))))
    except ValueError:
        raise ValueError("limit must be an integer") from None
    arguments: dict[str, object] = {
        "scope": "local",
        "project": identifier,
        "origin": "produced",
        "limit": limit,
    }
    if "cursor" in query:
        arguments["cursor"] = query["cursor"]
    return arguments
