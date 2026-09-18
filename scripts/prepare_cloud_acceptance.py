"""Build and import-check a commit-pinned cloud acceptance source archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def prepare(root: Path, revision: str, output: Path, *, uv: str = "uv") -> dict[str, Any]:
    root = root.resolve()
    commit = _run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=root)
    tree = _run(["git", "rev-parse", f"{commit}^{{tree}}"], cwd=root)
    with tempfile.TemporaryDirectory(prefix="dinkster-cloud-acceptance-") as directory:
        scratch = Path(directory)
        candidate = scratch / "dinkster-source.tar.gz"
        subprocess.run(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.eol=lf",
                "archive",
                "--format=tar.gz",
                f"--output={candidate}",
                commit,
            ],
            cwd=root,
            check=True,
        )
        source = scratch / "source"
        source.mkdir()
        with tarfile.open(candidate, "r:gz") as archive:
            archive.extractall(source, filter="data")
        if not (source / "packages/dinkster-acceptance/pyproject.toml").is_file():
            raise ValueError(f"selected commit {commit} has no installable acceptance package")
        environment = os.environ.copy()
        environment["UV_PROJECT_ENVIRONMENT"] = str(scratch / "environment")
        subprocess.run(
            [
                uv,
                "sync",
                "--project",
                str(source),
                "--package",
                "dinkster-acceptance",
                "--extra",
                "cloud",
                "--no-dev",
                "--locked",
            ],
            cwd=source,
            env=environment,
            check=True,
        )
        scripts = scratch / "environment" / ("Scripts" if os.name == "nt" else "bin")
        imports = json.loads(_run([str(scripts / "dinkster-acceptance-import-check")], cwd=source))
        if imports.get("status") != "ok":
            raise ValueError("acceptance import check did not report success")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(candidate, output)
    receipt = {
        "schema": "dinkster-cloud-acceptance-source/1",
        "commit": commit,
        "tree": tree,
        "archive": output.name,
        "archive_sha256": digest,
        "created_at": datetime.now(UTC).isoformat(),
        "import_check": imports,
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="Git revision to resolve and archive")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    if shutil.which(args.uv) is None:
        parser.error(f"uv executable not found: {args.uv}")
    root = Path(__file__).resolve().parent.parent
    print(json.dumps(prepare(root, args.commit, args.output, uv=args.uv), sort_keys=True))


if __name__ == "__main__":
    main()
