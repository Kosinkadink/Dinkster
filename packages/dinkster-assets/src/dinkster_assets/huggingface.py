"""Immutable public Hugging Face model selections and verified named downloads."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import BinaryIO, cast
from urllib.parse import quote, unquote, urlsplit

from .fetch import open_model_response
from .identity import AssetError
from .model import AssetRef
from .save_target import SaveTarget
from .writer import AssetWriter


@dataclass(frozen=True)
class HuggingFaceFile:
    repo: str
    revision: str
    file: str
    sha256: str
    size: int
    license: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{self.repo}/resolve/{self.revision}/"
            f"{quote(self.file, safe='/')}"
        )

    def download(
        self,
        writer: AssetWriter,
        target: SaveTarget,
        *,
        digest: str,
        before_publish: Callable[[AssetRef], None] | None = None,
    ) -> AssetRef:
        with open_model_response(self.url) as response:
            return writer.save_model_stream(
                target,
                cast("BinaryIO", response),
                sha256=self.sha256,
                size=self.size,
                digest=digest,
                before_publish=before_publish,
            )


def select_huggingface_file(url: str) -> HuggingFaceFile:
    """Resolve a declared Hub resolve URL to provider-verified immutable metadata."""
    parsed = urlsplit(url)
    parts = unquote(parsed.path).strip("/").split("/")
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or parsed.query
        or parsed.fragment
        or len(parts) < 5
        or parts[2] != "resolve"
        or any(not part or part in {".", ".."} or "\\" in part for part in parts)
    ):
        raise AssetError("model source must be a Hugging Face resolve URL")
    repo, revision, file = "/".join(parts[:2]), parts[3], "/".join(parts[4:])
    metadata_url = (
        f"https://huggingface.co/api/models/{quote(repo, safe='/')}/revision/"
        f"{quote(revision, safe='')}?blobs=true"
    )
    with open_model_response(metadata_url) as response:
        payload = response.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise AssetError("model provider metadata exceeds its limit")
    try:
        raw: object = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise AssetError("model provider returned invalid metadata") from exc
    if not isinstance(raw, dict):
        raise AssetError("model provider metadata must be an object")
    data = cast("dict[str, object]", raw)
    commit = data.get("sha")
    siblings = data.get("siblings")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise AssetError("model provider did not identify an immutable revision")
    if not isinstance(siblings, list):
        raise AssetError("model provider did not list files")
    for item in cast("list[object]", siblings):
        if not isinstance(item, dict):
            continue
        row = cast("dict[str, object]", item)
        if row.get("rfilename") != file:
            continue
        lfs = row.get("lfs")
        if not isinstance(lfs, dict):
            break
        info = cast("dict[str, object]", lfs)
        sha256, size = info.get("sha256"), row.get("size")
        if (
            not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or type(size) is not int
            or size < 0
            or info.get("size") != size
        ):
            break
        card = data.get("cardData")
        license = (
            cast("dict[str, object]", card).get("license", "") if isinstance(card, dict) else ""
        )
        return HuggingFaceFile(
            repo, commit, file, sha256, size, license if isinstance(license, str) else ""
        )
    raise AssetError(f"model provider has no verified size/SHA-256 for {PurePosixPath(file).name}")
