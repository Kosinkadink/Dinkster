"""Select independently published browser assets against the engine schema API."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from .environment_state import EnvironmentError, interpreter


def frontend_reference(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EnvironmentError("frontend selection must be an object")
    row = cast("dict[str, object]", value)
    version, sha, size = row.get("version"), row.get("sha256"), row.get("size")
    if (
        not isinstance(version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None
        or not isinstance(sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha) is None
        or type(size) is not int
        or size <= 0
    ):
        raise EnvironmentError("frontend selection requires a version, SHA256 and positive size")
    filename = f"dinkster_frontend-{version}-py3-none-any.whl"
    url = row.get("url")
    if not isinstance(url, str):
        raise EnvironmentError("frontend selection requires its permanent PyPI wheel URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rsplit("/", 1)[-1] != filename
    ):
        raise EnvironmentError("frontend selection requires its permanent PyPI wheel URL")
    api = row.get("engineApi")
    if not isinstance(api, dict):
        raise EnvironmentError("frontend selection requires its engine API range")
    api = cast("dict[str, object]", api)
    minimum, maximum = api.get("min"), api.get("max")
    if type(minimum) is not int or type(maximum) is not int or not 1 <= minimum <= maximum:
        raise EnvironmentError("frontend engine API range must be inclusive positive integers")
    return row


def resolve_frontend(version: str) -> dict[str, object]:
    """Select a published wheel and read its API declaration from verified bytes."""
    from .environment_downloads import fetch_artifact, read_metadata

    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise EnvironmentError("frontend version must be an exact published release version")
    document = json.loads(
        read_metadata("https://pypi.org/pypi/dinkster-frontend", version + "/json")
    )
    filename = f"dinkster_frontend-{version}-py3-none-any.whl"
    candidates = [
        row for row in document["urls"] if row["filename"] == filename and not row.get("yanked")
    ]
    if len(candidates) != 1:
        raise EnvironmentError(f"frontend {version} has no unique non-yanked published wheel")
    candidate = candidates[0]
    row: dict[str, object] = {
        "version": version,
        "url": candidate["url"],
        "sha256": candidate["digests"]["sha256"],
        "size": candidate["size"],
    }
    url = urlsplit(cast("str", row["url"]))
    if (
        url.scheme != "https"
        or url.netloc != "files.pythonhosted.org"
        or url.query
        or url.fragment
        or url.path.rsplit("/", 1)[-1] != filename
    ):
        raise EnvironmentError("frontend selection requires its permanent PyPI wheel URL")
    with tempfile.TemporaryDirectory(prefix="dinkster-frontend-release-") as directory:
        wheel = Path(directory) / filename
        fetch_artifact(
            (cast("str", row["url"]).rsplit("/", 1)[0],),
            filename,
            wheel,
            cast("str", row["sha256"]),
            cast("int", row["size"]),
        )
        with zipfile.ZipFile(wheel) as archive:
            manifest = json.loads(archive.read("dinkster_frontend/manifest.json"))
        row["engineApi"] = manifest.get("engineApi")
        frontend_reference(row)
        verify_frontend_wheel(wheel, row)
    return row


def select_frontend(
    engine: dict[str, object], channel: dict[str, object] | None = None
) -> dict[str, object] | None:
    selected = (channel or {}).get("frontend", engine.get("frontend"))
    if selected is None and "frontend" not in engine and "frontend" not in (channel or {}):
        return None
    row = frontend_reference(selected)
    api = engine.get("schemaApi")
    supported = cast("dict[str, int]", row["engineApi"])
    if type(api) is not int or not supported["min"] <= api <= supported["max"]:
        raise EnvironmentError(
            f"frontend {row['version']} supports engine schema API "
            f"{supported['min']} through {supported['max']}; selected engine declares {api}. "
            "Select a compatible engine or frontend release."
        )
    return row


def frontend_requirements(
    requirements: str, engine: dict[str, object], channel: dict[str, object] | None = None
) -> str:
    row = select_frontend(engine, channel)
    if row is None:
        return requirements
    return requirements + f"\ndinkster-frontend=={row['version']} --hash=sha256:{row['sha256']}\n"


def frontend_filename(row: dict[str, object]) -> str:
    return f"dinkster_frontend-{row['version']}-py3-none-any.whl"


def validate_installed_frontend(generation: Path) -> None:
    result = subprocess.run(
        [
            str(interpreter(generation)),
            "-I",
            "-B",
            "-c",
            "import importlib.util, importlib.resources, json; "
            "found = importlib.util.find_spec('dinkster_frontend'); "
            "manifest = json.loads(importlib.resources.files('dinkster_frontend')"
            ".joinpath('manifest.json').read_text()) if found else None; "
            "api = __import__('dinkster_schema.wire', fromlist=['SCHEMA_WIRE_VERSION'])"
            ".SCHEMA_WIRE_VERSION if found else None; "
            "print(json.dumps({'frontend': manifest, 'api': api}))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise EnvironmentError(
            f"cannot inspect installed frontend compatibility: {result.stderr.strip()}"
        )
    try:
        pair = json.loads(result.stdout)
        frontend = pair["frontend"]
        if frontend is None:
            return
        supported = frontend["engineApi"]
        api = pair["api"]
        compatible = (
            type(api) is int
            and type(supported["min"]) is int
            and type(supported["max"]) is int
            and 1 <= supported["min"] <= api <= supported["max"]
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise EnvironmentError("invalid installed frontend API declaration") from exc
    if not compatible:
        raise EnvironmentError(
            f"installed frontend {frontend['version']} does not support engine schema API {api}"
        )


def verify_frontend_wheel(path: Path, row: dict[str, object]) -> None:
    """Check producer metadata after the transport verified the complete wheel bytes."""
    with zipfile.ZipFile(path) as wheel:
        manifest = json.loads(wheel.read("dinkster_frontend/manifest.json"))
        if (
            manifest.get("format") != "dinkster.frontend/1"
            or manifest.get("repository") != "Kosinkadink/Dinkster-Frontend"
            or manifest.get("version") != row["version"]
            or manifest.get("engineApi") != row["engineApi"]
            or "dinkster_frontend/static/index.html" not in wheel.namelist()
        ):
            raise EnvironmentError("frontend wheel metadata differs from the selected release")
