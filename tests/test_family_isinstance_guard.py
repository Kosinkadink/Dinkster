from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.check_family_isinstance_gates import write_allowlist as write_canonical_allowlist

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_family_isinstance_gates.py"
SOURCE_PATH = "packages/dinkster-inference/src/dinkster_inference/runtime.py"


def run_guard(
    root: Path, allowlist: Path, *, write: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--allowlist",
            str(allowlist),
            "--write" if write else "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def semantic_site(
    *,
    scope: str,
    subject: str,
    checked_type: str,
    classification: str,
    occurrence: int = 1,
    path: str = SOURCE_PATH,
) -> dict[str, object]:
    return {
        "path": path,
        "scope": scope,
        "subject": subject,
        "type": checked_type,
        "classification": classification,
        "occurrence": occurrence,
    }


def write_allowlist(
    path: Path,
    *,
    sites: list[dict[str, object]] | None = None,
    ceiling: int | None = None,
    value_sites: list[dict[str, object]] | None = None,
    value_ceiling: int | None = None,
) -> None:
    classic = sites or []
    values = value_sites or []
    path.write_text(
        json.dumps(
            {
                "ceiling": len(classic) if ceiling is None else ceiling,
                "sites": classic,
                "value_type_allowlist": {
                    "ceiling": len(values) if value_ceiling is None else value_ceiling,
                    "sites": values,
                },
            }
        ),
        encoding="utf-8",
    )


def boundary_site(
    *, scope: str = "load", checked_type: str = "FluxAssemblyPlan"
) -> dict[str, object]:
    return {
        **semantic_site(
            scope=scope,
            subject="plan",
            checked_type=checked_type,
            classification="boundary",
        ),
        "reason": "Reject the wrong plan before reading family-specific fields.",
    }


def write_fixture(root: Path, body: str) -> Path:
    source = root / SOURCE_PATH
    source.parent.mkdir(parents=True)
    source.write_text(body, encoding="utf-8")
    return source


def write_native_fixture(root: Path, body: str) -> Path:
    source = root / "packages/dinkster-native/src/dinkster_native/nodes_runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text(body, encoding="utf-8")
    return source


def write_worker_environment_fixture(root: Path, body: str) -> Path:
    source = root / "packages/dinkster-workers/src/dinkster_workers/backend_env.py"
    source.parent.mkdir(parents=True)
    source.write_text(body, encoding="utf-8")
    return source


def fixture_source(*, function: str = "load", checked_type: str = "FamilyRuntime") -> str:
    return (
        f"def {function}(plan, value):\n"
        "    if not isinstance(plan, FluxAssemblyPlan):\n"
        "        raise TypeError('wrong plan')\n"
        f"    if isinstance(value, {checked_type}):\n"
        "        return special(value)\n"
        "    return generic(value)\n"
    )


def test_family_isinstance_guard_allows_classified_boundary_check(tmp_path: Path) -> None:
    write_fixture(
        tmp_path,
        "def load(plan):\n"
        "    if not isinstance(plan, FluxAssemblyPlan):\n"
        "        raise TypeError('wrong plan')\n"
        "    return plan.family\n",
    )
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist, sites=[boundary_site()])

    assert run_guard(tmp_path, allowlist).returncode == 0


def test_write_is_complete_deterministic_and_stable_under_line_motion(tmp_path: Path) -> None:
    source = write_fixture(tmp_path, fixture_source())
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist, sites=[boundary_site()], value_ceiling=1)

    first_write = run_guard(tmp_path, allowlist, write=True)
    assert first_write.returncode == 0, first_write.stderr
    first = allowlist.read_bytes()
    written = json.loads(first)
    assert written["ceiling"] == 1
    assert written["value_type_allowlist"]["ceiling"] == 1
    assert written["value_type_allowlist"]["sites"] == [
        semantic_site(
            scope="load",
            subject="value",
            checked_type="FamilyRuntime",
            classification="value-type",
        )
    ]
    assert "line" not in first.decode()
    assert "column" not in first.decode()

    source.write_text("\n# unrelated source movement\n" + fixture_source(), encoding="utf-8")
    second_write = run_guard(tmp_path, allowlist, write=True)

    assert second_write.returncode == 0, second_write.stderr
    assert allowlist.read_bytes() == first
    assert run_guard(tmp_path, allowlist).returncode == 0


def test_write_discovers_the_repository_outside_its_working_directory(tmp_path: Path) -> None:
    canonical = SCRIPT.with_name("family-isinstance-allowlist.json")
    generated = tmp_path / "allowlist.json"
    generated.write_bytes(canonical.read_bytes())

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--allowlist", str(generated), "--write"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert generated.read_bytes() == canonical.read_bytes()


def test_canonical_writer_requests_lf_newlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowlist = tmp_path / "allowlist.json"
    original_open = Path.open
    newlines: list[object] = []

    def recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        mode = args[0] if args else kwargs.get("mode")
        if self == allowlist and mode == "w":
            newlines.append(kwargs.get("newline"))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    write_canonical_allowlist(
        allowlist,
        ceiling=0,
        allowed=[],
        value_type_ceiling=0,
        value_type_sites=[],
    )

    assert newlines == ["\n"]
    assert b"\r\n" not in allowlist.read_bytes()


@pytest.mark.parametrize(
    ("changed", "new_identity", "old_identity"),
    [
        (
            fixture_source(function="load_renamed"),
            '"scope": "load_renamed"',
            '"scope": "load"',
        ),
        (
            fixture_source(checked_type="OtherRuntime"),
            '"type": "OtherRuntime"',
            '"type": "FamilyRuntime"',
        ),
    ],
)
def test_semantic_identity_changes_make_the_canonical_file_stale(
    tmp_path: Path, changed: str, new_identity: str, old_identity: str
) -> None:
    source = write_fixture(tmp_path, fixture_source())
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist, sites=[boundary_site()], value_ceiling=1)
    assert run_guard(tmp_path, allowlist, write=True).returncode == 0

    source.write_text(changed, encoding="utf-8")
    result = run_guard(tmp_path, allowlist)

    assert result.returncode == 1
    assert "unlisted value-type" in result.stderr
    assert "stale value-type" in result.stderr
    assert new_identity in result.stderr
    assert old_identity in result.stderr


def test_adding_and_removing_sites_make_the_canonical_file_stale(tmp_path: Path) -> None:
    source = write_fixture(tmp_path, fixture_source())
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist, sites=[boundary_site()], value_ceiling=1)
    assert run_guard(tmp_path, allowlist, write=True).returncode == 0

    source.write_text(
        fixture_source().replace(
            "    return generic(value)\n",
            "    if isinstance(other, OtherCarrier):\n"
            "        return special(other)\n"
            "    return generic(value)\n",
        ),
        encoding="utf-8",
    )
    added = run_guard(tmp_path, allowlist)
    assert added.returncode == 1
    assert '"type": "OtherCarrier"' in added.stderr
    assert "value-types=2/1" in added.stderr

    source.write_text(
        "def load(plan, value):\n"
        "    if not isinstance(plan, FluxAssemblyPlan):\n"
        "        raise TypeError('wrong plan')\n"
        "    return generic(value)\n",
        encoding="utf-8",
    )
    removed = run_guard(tmp_path, allowlist)
    assert removed.returncode == 1
    assert '"type": "FamilyRuntime"' in removed.stderr
    assert "value-types=0/1" in removed.stderr


def test_duplicate_semantic_sites_use_stable_occurrence_counts(tmp_path: Path) -> None:
    source = write_fixture(
        tmp_path,
        "def dispatch(value):\n"
        "    if isinstance(value, FamilyRuntime):\n"
        "        first(value)\n"
        "    if isinstance(value, FamilyRuntime):\n"
        "        second(value)\n",
    )
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist, value_ceiling=2)
    assert run_guard(tmp_path, allowlist, write=True).returncode == 0
    first = json.loads(allowlist.read_text(encoding="utf-8"))
    assert [site["occurrence"] for site in first["value_type_allowlist"]["sites"]] == [1, 2]

    source.write_text("\n\n" + source.read_text(encoding="utf-8"), encoding="utf-8")
    assert run_guard(tmp_path, allowlist, write=True).returncode == 0
    second = json.loads(allowlist.read_text(encoding="utf-8"))
    assert second == first


def test_family_isinstance_guard_rejects_family_branching(tmp_path: Path) -> None:
    write_fixture(
        tmp_path,
        "def load(plan):\n"
        "    if isinstance(plan, FluxAssemblyPlan):\n"
        "        return special_flux_behavior(plan)\n"
        "    return generic_behavior(plan)\n",
    )
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist)

    result = run_guard(tmp_path, allowlist)

    assert result.returncode == 1
    assert "prohibited family branching" in result.stderr
    assert "FluxAssemblyPlan" in result.stderr


def test_family_isinstance_guard_rejects_unlisted_family_value_type_gates(
    tmp_path: Path,
) -> None:
    write_fixture(
        tmp_path,
        "def dispatch(value):\n"
        "    if isinstance(value, (FamilyRuntime, module.OtherRuntime)):\n"
        "        return special(value)\n"
        "    if type(value) is FamilyCarrier:\n"
        "        return special(value)\n"
        "    if type(value) is FamilyLatent:\n"
        "        return special(value)\n"
        "    if isinstance(value, FamilyConditioning):\n"
        "        return special(value)\n"
        "    return generic(value)\n",
    )
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist)

    result = run_guard(tmp_path, allowlist)

    assert result.returncode == 1
    assert result.stderr.count("unlisted value-type") == 5
    for name in (
        "FamilyRuntime",
        "module.OtherRuntime",
        "FamilyCarrier",
        "FamilyLatent",
        "FamilyConditioning",
    ):
        assert name in result.stderr


def test_family_isinstance_guard_rejects_shared_family_comparisons(tmp_path: Path) -> None:
    write_native_fixture(
        tmp_path,
        "def load(handle, family, family_id):\n"
        "    if handle.recipe.family_id == TARGET.id:\n"
        "        return special(handle)\n"
        "    if family == 'minimax_h3':\n"
        "        return special(handle)\n"
        "    if family_id in FAMILY_IDS:\n"
        "        return special(handle)\n"
        "    return generic(handle)\n",
    )
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist)

    result = run_guard(tmp_path, allowlist)

    assert result.returncode == 1
    assert result.stderr.count("prohibited family branching") == 3
    assert "handle.recipe.family_id ==" in result.stderr
    assert "family ==" in result.stderr
    assert "family_id in" in result.stderr


def test_family_isinstance_guard_rejects_worker_residency_family_comparisons(
    tmp_path: Path,
) -> None:
    write_worker_environment_fixture(
        tmp_path,
        "def _residency_problems(family):\n"
        "    if family == 'minimax_h3':\n"
        "        return require_accelerator_residency()\n"
        "    return generic_residency()\n",
    )
    allowlist = tmp_path / "allowlist.json"
    write_allowlist(allowlist)

    result = run_guard(tmp_path, allowlist)

    assert result.returncode == 1
    assert "prohibited family branching" in result.stderr
    assert "family ==" in result.stderr


def test_canonical_allowlist_contains_only_semantic_site_fields() -> None:
    allowlist = SCRIPT.with_name("family-isinstance-allowlist.json")
    raw: dict[str, Any] = json.loads(allowlist.read_text(encoding="utf-8"))
    all_sites = raw["sites"] + raw["value_type_allowlist"]["sites"]

    assert raw["ceiling"] == 8
    assert raw["value_type_allowlist"]["ceiling"] == 264
    assert len(raw["sites"]) == 8
    assert len(raw["value_type_allowlist"]["sites"]) == 264
    assert all("line" not in site and "column" not in site for site in all_sites)
