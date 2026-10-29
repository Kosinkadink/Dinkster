"""Record verified OpenTimelineIO binary identities for release acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import parse_wheel_filename


def record(source: Path, wheels: Path, output: Path, target: str) -> None:
    pin = json.loads(Path(__file__).with_name("opentimelineio_source.json").read_text())
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != pin["commit"]:
        raise ValueError("OpenTimelineIO source differs from the pinned commit")
    submodules = subprocess.check_output(
        ["git", "-C", str(source), "submodule", "status", "--recursive"], text=True
    ).splitlines()
    if not submodules or any(not line.startswith(" ") for line in submodules):
        raise ValueError("OpenTimelineIO submodules differ from the pinned source")
    artifacts = []
    paths = sorted(wheels.glob("*.whl"))
    if len(paths) != 1:
        raise ValueError("expected exactly one OpenTimelineIO wheel per target")
    for path in paths:
        name, version, _, tags = parse_wheel_filename(path.name)
        valid = all(
            tag.interpreter == "cp314"
            and tag.abi == "cp314"
            and (
                tag.platform == "win_amd64"
                if target == "windows"
                else tag.platform.startswith("manylinux") and tag.platform.endswith("_x86_64")
            )
            for tag in tags
        )
        if name != "opentimelineio" or str(version) != pin["version"] or not valid:
            raise ValueError(f"unexpected OpenTimelineIO wheel identity: {path.name}")
        with zipfile.ZipFile(path) as wheel:
            names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError("wheel must contain exactly one METADATA file")
            metadata = BytesParser().parsebytes(wheel.read(names[0]))
            if metadata["Name"].lower() != "opentimelineio" or metadata["Version"] != str(version):
                raise ValueError("wheel metadata disagrees with its filename")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        artifacts.append({"filename": path.name, "size": path.stat().st_size, "sha256": digest})
    output.mkdir(parents=True, exist_ok=True)
    provenance = {**pin, "submodules": submodules, "target": target, "wheels": artifacts}
    (output / f"opentimelineio-{target}.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (output / f"opentimelineio-{target}.md").write_text(
        f"Upstream: https://github.com/{pin['repository']}/commit/{commit}\n\n"
        f"Upstream Python 3.14 CI: {pin['upstream_ci']}\n\n"
        + "\n".join(
            f"- `{item['filename']}`: {item['size']} bytes, SHA256 `{item['sha256']}`"
            for item in artifacts
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("linux", "windows"), required=True)
    args = parser.parse_args()
    record(args.source, args.wheels, args.output, args.target)
