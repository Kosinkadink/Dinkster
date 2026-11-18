"""Civitai file selections verified against provider metadata and transfer size."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO, cast
from urllib.parse import parse_qs, urlencode, urlsplit

from .fetch import open_model_response
from .identity import AssetError
from .model import AssetRef
from .save_target import SaveTarget
from .writer import AssetWriter


@dataclass(frozen=True)
class CivitaiFile:
    repo: str
    revision: str
    file: str
    sha256: str
    size: int
    license: str
    url: str

    def download(
        self,
        writer: AssetWriter,
        target: SaveTarget,
        *,
        digest: str | None = None,
        before_publish: Callable[[AssetRef], None] | None = None,
        progress: Callable[[str, int], None] | None = None,
        token: str | None = None,
    ) -> AssetRef:
        with open_model_response(self.url) as response:
            return writer.save_model_stream(
                target,
                cast("BinaryIO", response),
                sha256=self.sha256,
                size=self.size,
                digest=digest,
                before_publish=before_publish,
                progress=progress,
            )


def _source(url: str) -> tuple[str, str, dict[str, str]]:
    parsed = urlsplit(url)
    match = re.fullmatch(r"/(models|api/download/models)/([1-9][0-9]*)(?:/[^/]*)?", parsed.path)
    try:
        values = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        raise AssetError("invalid Civitai source query") from None
    allowed = (
        {"modelVersionId"}
        if match and match[1] == "models"
        else {"fileId", "type", "format", "size", "fp", "quantType"}
    )
    if (
        parsed.scheme != "https"
        or parsed.netloc != "civitai.com"
        or parsed.fragment
        or match is None
        or any(ord(char) <= 32 or ord(char) == 127 for char in url)
        or any(
            key not in allowed
            or len(items) != 1
            or re.fullmatch(r"[A-Za-z0-9 ._-]{1,64}", items[0]) is None
            for key, items in values.items()
        )
        or any(
            key in values and re.fullmatch(r"[1-9][0-9]*", values[key][0]) is None
            for key in ("modelVersionId", "fileId")
        )
    ):
        raise AssetError("model source must be a Civitai model or download URL without credentials")
    return match[1], match[2], {key: items[0] for key, items in values.items()}


def select_civitai_file(url: str, name: str) -> CivitaiFile | tuple[str, ...]:
    """Match a captured filename, requiring a version/file choice when ambiguous."""
    route, identifier, choice = _source(url)
    endpoint = "models" if route == "models" else "model-versions"
    with open_model_response(f"https://civitai.com/api/v1/{endpoint}/{identifier}") as response:
        payload = response.read(2 * 1024 * 1024 + 1)
    try:
        raw: object = json.loads(payload) if len(payload) <= 2 * 1024 * 1024 else None
    except (ValueError, UnicodeError):
        raw = None
    if not isinstance(raw, dict):
        raise AssetError("Civitai returned invalid model metadata")
    data = cast("dict[str, object]", raw)
    if type(data.get("id")) is not int or str(data["id"]) != identifier:
        raise AssetError("Civitai returned invalid model identity")
    versions = data.get("modelVersions") if route == "models" else [data]
    if not isinstance(versions, list):
        raise AssetError("Civitai did not list model versions")
    candidates: list[tuple[str, str, str, str]] = []
    for version in cast("list[object]", versions):
        if not isinstance(version, dict):
            continue
        version = cast("dict[str, object]", version)
        if type(version.get("id")) is not int:
            continue
        revision = str(version["id"])
        if "modelVersionId" in choice and revision != choice["modelVersionId"]:
            continue
        model_id = identifier if route == "models" else str(version.get("modelId", ""))
        files = version.get("files")
        if not isinstance(files, list) or re.fullmatch(r"[1-9][0-9]*", model_id) is None:
            continue
        for item in cast("list[object]", files):
            if not isinstance(item, dict):
                continue
            row = cast("dict[str, object]", item)
            if row.get("name") != name:
                continue
            hashes = row.get("hashes")
            sha256 = (
                cast("dict[str, object]", hashes).get("SHA256")
                if isinstance(hashes, dict)
                else None
            )
            download = row.get("downloadUrl")
            if (
                not isinstance(sha256, str)
                or re.fullmatch(r"[a-fA-F0-9]{64}", sha256) is None
                or not isinstance(download, str)
            ):
                continue
            download_route, download_version, query = _source(download)
            if download_route != "api/download/models" or download_version != revision:
                raise AssetError("Civitai returned an inconsistent file source")
            if "fileId" in choice and str(row.get("id")) != choice["fileId"]:
                continue
            metadata = row.get("metadata")
            if any(
                key not in {"fileId", "modelVersionId"}
                and value
                != (
                    row.get("type")
                    if key == "type"
                    else cast("dict[str, object]", metadata).get(key)
                    if isinstance(metadata, dict)
                    else None
                )
                for key, value in choice.items()
            ):
                continue
            # Linked component files retain the provider's host-version access gate.
            if row.get("modelVersionId") == version["id"] and type(row.get("id")) is int:
                query = {"fileId": str(row["id"])}
            canonical = f"https://civitai.com/api/download/models/{revision}"
            if query:
                canonical += "?" + urlencode(query)
            candidates.append((model_id, revision, sha256.lower(), canonical))
    if not candidates:
        raise AssetError("Civitai has no matching file with a verified SHA-256")
    if len(candidates) != 1:
        return tuple(dict.fromkeys(row[3] for row in candidates))
    model_id, revision, sha256, canonical = candidates[0]
    with open_model_response(canonical) as response:
        lengths = response.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or re.fullmatch(r"[0-9]+", lengths[0]) is None:
            raise AssetError("Civitai did not provide an exact file size")
        size = int(lengths[0])
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise AssetError("Civitai returned an encoded model stream")
    return CivitaiFile(model_id, revision, name, sha256, size, "", canonical)
