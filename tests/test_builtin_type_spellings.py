from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_builtin_type_spellings.py"


def test_builtin_type_guard_rejects_literals_but_not_protocol_tags(tmp_path: Path) -> None:
    spellings = tmp_path / "packages/dinkster-values/src/dinkster_values/builtin_types.py"
    spellings.parent.mkdir(parents=True)
    spellings.write_text('CORE_INT = "INT"\n', encoding="utf-8")
    source = tmp_path / "packages/example/src/example.py"
    source.parent.mkdir(parents=True)
    source.write_text('socket = TypeExpr.concrete("INT")\n', encoding="utf-8")

    def check() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )

    refused = check()
    assert refused.returncode == 1
    assert "example.py:1: import the type constant" in refused.stdout
    source.write_text('registry.wrap("INT", 3)\n', encoding="utf-8")
    assert check().returncode == 1
    source.write_text('MY_TYPE = "INT"\n', encoding="utf-8")
    assert check().returncode == 1
    source.write_text(
        "from dinkster_values import CORE_INT\n"
        "socket = TypeExpr.concrete(CORE_INT)\n"
        'widget = {"type": "INT"}\n'
        'COMBO_IO_TYPE = "INT"\n',
        encoding="utf-8",
    )
    assert check().returncode == 0
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "literal.py").write_text('socket = TypeExpr.concrete("INT")\n', encoding="utf-8")
    assert check().returncode == 0
    nested = tmp_path / "packages/example/preview/src/preview.py"
    nested.parent.mkdir(parents=True)
    nested.write_text('socket = TypeExpr.concrete("INT")\n', encoding="utf-8")
    assert check().returncode == 1
