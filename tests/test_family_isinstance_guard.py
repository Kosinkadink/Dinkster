from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_family_isinstance_gates.py"


def run_guard(root: Path, allowlist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--allowlist", str(allowlist)],
        capture_output=True,
        text=True,
        check=False,
    )


def write_fixture(root: Path, body: str) -> Path:
    source = root / "packages/dinkster-inference/src/dinkster_inference/runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text(body, encoding="utf-8")
    return source


def test_family_isinstance_guard_allows_classified_boundary_check(tmp_path: Path) -> None:
    write_fixture(
        tmp_path,
        "def load(plan):\n"
        "    if not isinstance(plan, FluxAssemblyPlan):\n"
        "        raise TypeError('wrong plan')\n"
        "    return plan.family\n",
    )
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "ceiling": 1,
                "sites": [
                    {
                        "path": "packages/dinkster-inference/src/dinkster_inference/runtime.py",
                        "line": 2,
                        "column": 12,
                        "type": "FluxAssemblyPlan",
                        "classification": "boundary",
                        "reason": "Reject the wrong plan before reading family-specific fields.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert run_guard(tmp_path, allowlist).returncode == 0


def test_family_isinstance_guard_rejects_family_branching(tmp_path: Path) -> None:
    write_fixture(
        tmp_path,
        "def load(plan):\n"
        "    if isinstance(plan, FluxAssemblyPlan):\n"
        "        return special_flux_behavior(plan)\n"
        "    return generic_behavior(plan)\n",
    )
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps({"ceiling": 0, "sites": []}), encoding="utf-8")
    result = run_guard(tmp_path, allowlist)
    assert result.returncode == 1
    assert "prohibited family branching" in result.stderr
    assert "FluxAssemblyPlan" in result.stderr
