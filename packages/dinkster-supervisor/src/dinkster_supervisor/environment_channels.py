"""Build channel documents from the publishers' verified, unchanged release inputs."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .environment_artifacts import (
    artifact_rows,
    engine_constraints,
    parse_metadata,
    validate_dependency_base,
    validate_manifest,
)
from .environment_constraints import effective_constraints
from .environment_state import (
    Channel,
    EnvironmentError,
    content_digest,
    relative_file,
    write_document,
)


def engine_publication_names(
    engine: dict[str, object], identity: str, channel: str
) -> tuple[str, str]:
    """Derive external names only after the engine manifest bytes are final."""
    label = engine.get("label")
    if label is None:
        return identity[7:], relative_file(engine.get("tag"))
    if (
        not isinstance(label, str)
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9][A-Za-z0-9._+\-]*", label) is None
    ):
        raise EnvironmentError("engine label must be a portable dated version")
    name = f"{label}-{identity[7:15]}"
    tag = f"v{engine['version']}" if channel == "stable" else f"main-{name}"
    return name, tag


def write_channel(
    release: Path,
    bases: Sequence[Path],
    *,
    channel: str,
    output_root: Path,
    minimum_desktop_version: str,
) -> dict[str, object]:
    """Validate all inputs before atomically replacing one local channel document."""
    if channel not in {"stable", "main"}:
        raise EnvironmentError("channel must be stable or main")
    data = (release / "release-manifest.json").read_bytes()
    engine_id = "sha256:" + hashlib.sha256(data).hexdigest()
    engine = validate_manifest(data, engine_id, None)
    if engine.get("repository") != "Kosinkadink/Dinkster":
        raise EnvironmentError("channel engine must come from Kosinkadink/Dinkster")
    engine_name, tag = engine_publication_names(engine, engine_id, channel)
    for row in artifact_rows(engine["artifacts"], "name"):
        artifact = release / cast("str", row["name"])
        if artifact.stat().st_size != row["size"] or content_digest(artifact) != "sha256:" + cast(
            "str", row["sha256"]
        ):
            raise EnvironmentError(f"engine artifact {artifact.name} differs from release manifest")
    constraints = (release / "constraints.txt").read_bytes()
    selected: dict[str, object] = {}
    for path in bases:
        base_data = path.read_bytes()
        raw = parse_metadata(base_data)
        variant, base_id = raw.get("variant"), raw.get("id")
        if not isinstance(variant, str) or not isinstance(base_id, str):
            raise EnvironmentError("base manifest must name its variant and id")
        base = validate_manifest(base_data, base_id, variant)
        if variant in selected:
            raise EnvironmentError(f"duplicate selected base variant: {variant}")
        if base["workerProtocol"] != engine["workerProtocol"]:
            raise EnvironmentError(f"workerProtocol mismatch for selected base {variant}")
        if "dependencies" in engine:
            selection = cast("dict[str, object]", engine["dependencies"]).get(variant)
            validate_dependency_base(selection, base)
            effective_constraints(engine_constraints(engine, variant), base, strict=True)
        else:
            effective_constraints(constraints.decode("utf-8"), base)
        base_name = base.get("label", base_id[7:])
        selected[variant] = {
            "id": base_id,
            "locations": [{"type": "mirror", "path": f"base/{variant}/{base_name}/manifest.json"}],
        }
    document: dict[str, object] = {
        "format": "dinkster.channel/1",
        "engine": {
            "version": engine["version"],
            "release": engine_id,
            "tag": tag,
            "locations": [
                {"type": "mirror", "path": f"engine/{engine_name}/release-manifest.json"},
                {
                    "type": "github",
                    "repository": "Kosinkadink/Dinkster",
                    "tag": tag,
                    "asset": "release-manifest.json",
                },
            ],
        },
        "base": selected,
        "minimumDesktopVersion": minimum_desktop_version,
        "signature": None,
    }
    Channel.parse(document)
    write_document(output_root / "channels" / f"{channel}.json", document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, action="append", required=True)
    parser.add_argument("--channel", choices=("stable", "main"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-desktop-version", required=True)
    args = parser.parse_args()
    try:
        write_channel(
            args.release,
            cast("list[Path]", args.base_manifest),
            channel=args.channel,
            output_root=args.output_root,
            minimum_desktop_version=args.minimum_desktop_version,
        )
    except (OSError, ValueError) as exc:
        print(f"channel refused: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
