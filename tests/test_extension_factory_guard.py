from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_extension_factories.py"
SITE_KINDS = ("assemblyBuilder", "attentionFactory", "descriptorCatalog", "registryFactory")


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


def write_allowlist(path: Path, sites: list[dict[str, object]]) -> None:
    ceilings = {kind: sum(site["kind"] == kind for site in sites) for kind in SITE_KINDS}
    path.write_text(
        json.dumps(
            {"ceilings": ceilings, "slack": dict.fromkeys(SITE_KINDS, 0), "sites": sites},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_extension_factory_guard_tracks_each_descriptor_catalog(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    module = source / "runtime.py"
    allowlist = tmp_path / "allowlist.json"

    for call in (
        "builtin_families",
        "builtin_sampler_snapshot",
        "builtin_samplers",
        "builtin_schedulers",
    ):
        module.write_text(f"catalog = {call}()\n", encoding="utf-8")
        expected = [
            {
                "kind": "descriptorCatalog",
                "call": call,
                "path": "src/runtime.py",
                "line": 1,
                "column": 11,
                "issue": 120,
            }
        ]
        write_allowlist(allowlist, expected)
        assert run_guard(tmp_path, allowlist, "--write").returncode == 0
        recorded = json.loads(allowlist.read_text(encoding="utf-8"))
        assert recorded["sites"] == expected


def test_core_only_attention_contributions_cannot_bypass_composition(tmp_path: Path) -> None:
    source = tmp_path / "packages/engine/src/engine"
    source.mkdir(parents=True)
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist, [])
    for name in (
        "AttentionRegistry",
        "AttentionQKVDescriptor",
        "AttentionWrapperDescriptor",
        "AttentionOutputDescriptor",
        "AttentionBackendDescriptor",
        "BlockInjectionDescriptor",
    ):
        (source / "attention.py").write_text(
            f'if model.family == "private.family":\n    hook = {name}()\n', encoding="utf-8"
        )
        result = run_guard(tmp_path, allowlist)
        assert result.returncode == 1
        assert name in result.stderr
        assert '"kind": "attentionFactory"' in result.stderr
        assert "explicit owning issues" in result.stderr


def test_extension_factory_guard_write_refuses_to_raise_ceiling(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    module = source / "runtime.py"
    allowlist = tmp_path / "allowlist.json"
    module.write_text("registry = builtin_family_registry()\n", encoding="utf-8")
    write_allowlist(
        allowlist,
        [
            {
                "kind": "registryFactory",
                "call": "builtin_family_registry",
                "path": "src/runtime.py",
                "line": 1,
                "column": 12,
                "issue": 120,
            }
        ],
    )
    assert run_guard(tmp_path, allowlist, "--write").returncode == 0
    recorded = allowlist.read_text(encoding="utf-8")

    module.write_text(
        "registry = builtin_family_registry()\nother = builtin_preview_registry()\n",
        encoding="utf-8",
    )
    plain = run_guard(tmp_path, allowlist)
    assert plain.returncode == 1
    assert "explicit owning issues" in plain.stderr
    raised = run_guard(tmp_path, allowlist, "--write")
    assert raised.returncode == 1
    assert "builtin_preview_registry" in raised.stderr
    assert "positive issue" in raised.stderr
    assert allowlist.read_text(encoding="utf-8") == recorded

    explicit = json.loads(recorded)
    explicit["sites"].append(
        {
            "kind": "registryFactory",
            "call": "builtin_preview_registry",
            "path": "src/runtime.py",
            "line": 2,
            "column": 9,
            "issue": 120,
        }
    )
    allowlist.write_text(json.dumps(explicit, indent=2) + "\n", encoding="utf-8")
    explicit_recorded = allowlist.read_text(encoding="utf-8")
    raised = run_guard(tmp_path, allowlist, "--write")
    assert raised.returncode == 1
    assert "registryFactory: current=2, ceiling=1" in raised.stderr
    assert allowlist.read_text(encoding="utf-8") == explicit_recorded


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
    write_allowlist(
        allowlist,
        [
            {
                "kind": "assemblyBuilder",
                "call": "build_builtin_assembly_registry",
                "path": "packages/example/src/example/runtime.py",
                "line": 3,
                "column": 14,
                "issue": 120,
            },
            {
                "kind": "descriptorCatalog",
                "call": "builtin_families",
                "path": "packages/example/src/example/runtime.py",
                "line": 2,
                "column": 12,
                "issue": 120,
            },
            {
                "kind": "registryFactory",
                "call": "builtin_family_registry",
                "path": "packages/example/src/example/runtime.py",
                "line": 1,
                "column": 12,
                "issue": 120,
            },
        ],
    )
    assert run_guard(tmp_path, allowlist, "--write").returncode == 0
    assert run_guard(tmp_path, allowlist).returncode == 0

    module.write_text(
        "registry = builtin_family_registry()\nfamilies = builtin_families()\n"
        "assemblies = build_builtin_assembly_registry(registry)\n"
        "other = builtin_preview_registry()\n",
        encoding="utf-8",
    )
    added = run_guard(tmp_path, allowlist)
    assert added.returncode == 1
    assert "unlisted" in added.stderr
    assert "scripts/check_extension_factories.py --write" in added.stderr

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


def test_extension_factory_guard_ratchets_against_baseline(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    module = source / "runtime.py"
    allowlist = tmp_path / "allowlist.json"
    baseline = tmp_path / "baseline.json"
    first = {
        "kind": "registryFactory",
        "call": "builtin_family_registry",
        "path": "src/runtime.py",
        "line": 1,
        "column": 12,
        "issue": 120,
    }
    module.write_text("registry = builtin_family_registry()\n", encoding="utf-8")
    write_allowlist(allowlist, [first])
    baseline.write_bytes(allowlist.read_bytes())

    second = {
        "kind": "registryFactory",
        "call": "builtin_preview_registry",
        "path": "src/runtime.py",
        "line": 2,
        "column": 9,
        "issue": 305,
    }
    module.write_text(
        "registry = builtin_family_registry()\nother = builtin_preview_registry()\n",
        encoding="utf-8",
    )
    write_allowlist(allowlist, [first, second])
    raised = run_guard(tmp_path, allowlist, "--baseline-allowlist", str(baseline))
    assert raised.returncode == 1
    assert "baseline=1, proposed=2" in raised.stderr

    module.write_text("", encoding="utf-8")
    stale = json.loads(baseline.read_text(encoding="utf-8"))
    stale["sites"] = []
    allowlist.write_text(json.dumps(stale), encoding="utf-8")
    not_lowered = run_guard(tmp_path, allowlist)
    assert not_lowered.returncode == 1
    assert "registryFactory: current=0, allowlisted=0, ceiling=1" in not_lowered.stderr
