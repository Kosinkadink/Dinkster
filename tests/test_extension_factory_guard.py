from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_extension_factories.py"


def run_guard(root: Path, allowlist: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--allowlist",
            str(allowlist),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_extension_factory_guard_rejects_site_and_ceiling_drift(tmp_path: Path) -> None:
    source = tmp_path / "packages/example/src/example"
    source.mkdir(parents=True)
    module = source / "runtime.py"
    allowlist = tmp_path / "allowlist.json"
    module.write_text(
        "registry = builtin_family_registry()\nfamilies = builtin_families()\n"
        "assemblies = build_builtin_assembly_registry(registry)\n",
        encoding="utf-8",
    )
    assert run_guard(tmp_path, allowlist, "--write").returncode == 0
    assert run_guard(tmp_path, allowlist).returncode == 0

    module.write_text(
        "registry = builtin_family_registry()\nfamilies = builtin_families()\n"
        "assemblies = build_builtin_assembly_registry(registry)\n"
        "other = builtin_sampler_registry()\n",
        encoding="utf-8",
    )
    added = run_guard(tmp_path, allowlist)
    assert added.returncode == 1
    assert "unlisted" in added.stderr

    module.write_text(
        "registry = builtin_family_registry()\n"
        "assemblies = build_builtin_assembly_registry(registry)\n",
        encoding="utf-8",
    )
    removed_catalog = run_guard(tmp_path, allowlist)
    assert removed_catalog.returncode == 1
    assert '"call": "builtin_families"' in removed_catalog.stderr

    module.write_text(
        "registry = builtin_family_registry()\nfamilies = builtin_families()\n",
        encoding="utf-8",
    )
    removed_assembly = run_guard(tmp_path, allowlist)
    assert removed_assembly.returncode == 1
    assert '"call": "build_builtin_assembly_registry"' in removed_assembly.stderr

    module.write_text(
        "registry = object()\nfamilies = builtin_families()\n"
        "assemblies = build_builtin_assembly_registry(registry)\n",
        encoding="utf-8",
    )
    removed = run_guard(tmp_path, allowlist)
    assert removed.returncode == 1
    assert "stale" in removed.stderr

    recorded = json.loads(allowlist.read_text(encoding="utf-8"))
    recorded["ceilings"]["registryFactory"] += 1
    allowlist.write_text(json.dumps(recorded), encoding="utf-8")
    raised = run_guard(tmp_path, allowlist)
    assert raised.returncode == 1
    assert "registryFactory: current=0, allowlisted=1, ceiling=2" in raised.stderr
