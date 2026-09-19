"""The umbrella `dinkster` CLI (DESIGN 5 doctor exposure: `dinkster doctor` as
the main-CLI spelling of the publish gate, standalone scripts as aliases).

The dispatcher's whole contract: resolve a subcommand name to the
standalone script's existing main() lazily and add no behavior - same
flags, same exit codes, same output through either door. Proven here by
driving the real doctor through `dinkster doctor` against a real pack, and
threading sys.argv-style commands (`dinkster pack --help`) through their
own argparse.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dinkster.cli import main

HEALTHY_MANIFEST = """\
[pack]
name = "healthy-pack"
namespaces = ["healthy"]
requires = ["numpy>=1.26"]

[pack.entry]
nodes = "healthy_nodes:NODES"
"""

HEALTHY_NODES = """\
from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr


class Doubler(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="healthy.doubler",
            inputs=(InputSpec("value", TypeExpr.concrete("core.int")),),
            outputs=(OutputSpec("doubled", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls, *, value):
        return cls.outputs(doubled=value * 2)


NODES = [Doubler]
"""


def write_pack(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dinkster-pack.toml").write_text(HEALTHY_MANIFEST)
    (root / "healthy_nodes.py").write_text(HEALTHY_NODES)
    return root


def test_no_command_is_usage_on_stderr_exit_2(capsys) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage: dinkster" in captured.err
    assert "doctor" in captured.err


def test_help_lists_every_command_exit_0(capsys) -> None:
    assert main(["--help"]) == 0
    listing = capsys.readouterr().out
    for name in (
        "doctor",
        "serve",
        "pack",
        "installs",
        "port",
        "p2p-diagnostics",
        "demo",
        "isolated-demo",
    ):
        assert name in listing
    assert "run/bootstrap a registry service" not in listing


def test_unknown_command_is_loud_exit_2(capsys) -> None:
    assert main(["frobnicate"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'frobnicate'" in err
    assert "usage: dinkster" in err


def test_doctor_subcommand_is_the_real_publish_gate(tmp_path: Path, capsys) -> None:
    """`dinkster doctor` runs the identical doctor main: same JSON report,
    same exit codes, against a real pack."""
    pack = write_pack(tmp_path / "healthy")
    assert main(["doctor", str(pack), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["reportVersion"] == 1
    assert payload["nodeTypes"] == ["healthy.doubler"]

    (pack / "healthy_nodes.py").write_text("raise Exception()\n")
    assert main(["doctor", str(pack)]) == 1
    assert "entry.unresolvable" in capsys.readouterr().out


def test_sysargv_commands_get_threaded_args_and_argv_restored(capsys) -> None:
    """sys.argv-style mains (here: `pack`, hitting dinkster.manager's own
    argparse) receive the tail through a rebound sys.argv that is
    restored afterwards even when the command exits."""
    before = list(sys.argv)
    with pytest.raises(SystemExit) as excinfo:
        main(["pack", "--help"])
    assert excinfo.value.code == 0
    assert "install root" in capsys.readouterr().out
    assert sys.argv == before


def test_dispatcher_imports_lazily_and_module_form_works() -> None:
    """`python -m dinkster --help` answers without importing any subcommand
    module - `dinkster doctor` never pays for the server stack."""
    probe = (
        "import sys\n"
        "import dinkster.cli\n"
        "rc = dinkster.cli.main(['--help'])\n"
        "assert rc == 0, rc\n"
        "heavy = [m for m in ('dinkster.serve', 'dinkster.demo', 'dinkster.manager',"
        " 'dinkster_workers.doctor') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True, timeout=60)
    result = subprocess.run(
        [sys.executable, "-m", "dinkster", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "usage: dinkster" in result.stdout


def test_demo_does_not_import_isolated_default_packs(monkeypatch) -> None:
    import dinkster.demo as demo

    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", "")
    real_load_pack = demo.load_pack

    def checked_load_pack(manifest, **kwargs):
        assert manifest.name != "dinkster-nodes-remote"
        return real_load_pack(manifest, **kwargs)

    monkeypatch.setattr(demo, "load_pack", checked_load_pack)
    asyncio.run(demo.run_demo())
