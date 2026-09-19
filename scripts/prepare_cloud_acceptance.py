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


def prepare(
    root: Path,
    revision: str,
    output: Path,
    *,
    evidence_root: Path,
    evidence_revision: str,
    uv: str = "uv",
) -> dict[str, Any]:
    root = root.resolve()
    commit = _run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=root)
    tree = _run(["git", "rev-parse", f"{commit}^{{tree}}"], cwd=root)
    evidence_root = evidence_root.resolve()
    evidence_commit = _run(
        ["git", "rev-parse", "--verify", f"{evidence_revision}^{{commit}}"], cwd=evidence_root
    )
    evidence_tree = _run(["git", "rev-parse", f"{evidence_commit}^{{tree}}"], cwd=evidence_root)
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
        acceptance_archive = scratch / "acceptance.tar"
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={acceptance_archive}",
                evidence_commit,
                "packages/dinkster-acceptance",
            ],
            cwd=evidence_root,
            check=True,
        )
        with tarfile.open(acceptance_archive) as archive:
            archive.extractall(source, filter="data")
        if not (source / "packages/dinkster-acceptance/pyproject.toml").is_file():
            raise ValueError(
                f"evidence commit {evidence_commit} has no installable acceptance package"
            )
        environment = os.environ.copy()
        environment["UV_PROJECT_ENVIRONMENT"] = str(scratch / "environment")
        # Restore the external package only in the disposable archive workspace.
        # Existing dependency pins remain locked when adding its workspace entry.
        subprocess.run(
            [uv, "lock", "--project", str(source)],
            cwd=source,
            env=environment,
            check=True,
        )
        with tarfile.open(candidate, "w:gz") as archive:
            for entry in sorted(source.iterdir()):
                archive.add(entry, arcname=entry.name)
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
        "evidence_commit": evidence_commit,
        "evidence_tree": evidence_tree,
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
    parser.add_argument("--evidence-commit", required=True, help="Evidence revision to resolve")
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path(
            os.environ.get(
                "DINKSTER_EVIDENCE_ROOT", Path(__file__).resolve().parents[2] / "dinkster-evidence"
            )
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    if shutil.which(args.uv) is None:
        parser.error(f"uv executable not found: {args.uv}")
    root = Path(__file__).resolve().parent.parent
    print(
        json.dumps(
            prepare(
                root,
                args.commit,
                args.output,
                evidence_root=args.evidence_root,
                evidence_revision=args.evidence_commit,
                uv=args.uv,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
