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

import json
from pathlib import Path

import pytest

from dinkster.cli import main

HEALTHY_MANIFEST = """\
[pack]
name = "healthy"
namespaces = ["healthy"]
requires = ["numpy>=1.26"]

[pack.entry]
nodes = "healthy:NODES"
"""

HEALTHY_NODES = """\
from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr


class Doubler(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="healthy.doubler",
            inputs=(InputSpec("value", TypeExpr.concrete("INT")),),
            outputs=(OutputSpec("doubled", TypeExpr.concrete("INT")),),
        )

    @classmethod
    def execute(cls, *, value):
        return cls.outputs(doubled=value * 2)


NODES = [Doubler]
"""


def write_pack(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dinkster-pack.toml").write_text(HEALTHY_MANIFEST)
    (root / "healthy").mkdir(exist_ok=True)
    (root / "healthy/__init__.py").write_text(HEALTHY_NODES)
    return root


@pytest.mark.parametrize("command", [[], ["serve"], ["stop"], ["update"], ["rollback"]])
def test_lifecycle_help_lists_project_and_operation_options(command: list[str], capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([*command, "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: dinkster" in help_text
    assert "--root" in help_text
    assert "--plan" in help_text and "--yes" in help_text
    if command in ([], ["serve"]):
        assert "--port" in help_text and "--gpu" in help_text
        assert "--no-browser" in help_text
        assert "--frontend-root" in help_text and "--frontend-dev" in help_text
    elif command == ["update"]:
        assert "--channel" in help_text and "--variant" in help_text
    elif command == ["rollback"]:
        assert "--approve" in help_text


def test_unknown_command_is_loud_exit_2(capsys) -> None:
    assert main(["frobnicate"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'frobnicate'" in err
    assert "usage: dinkster" in err


@pytest.mark.parametrize("command", ["doctor", "prepare-catalogs"])
@pytest.mark.parametrize("endpoint", [None, "", "http://127.0.0.1:8123"])
def test_default_pack_helpers_do_not_enable_hosted_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, endpoint: str | None
) -> None:
    from dinkster_server import PackInfo

    from dinkster import compose

    for name in ("DINKSTER_REMOTE_CATALOG_BASE", "DINKSTER_REMOTE_GATEWAY_BASE"):
        monkeypatch.delenv(name, raising=False)
    pack = write_pack(tmp_path / "healthy")
    expected = endpoint or ""
    module = pack / "healthy/__init__.py"
    module.write_text(
        "import os\n"
        f"assert os.environ.get('DINKSTER_REMOTE_CATALOG_BASE') == {expected!r}\n"
        f"assert os.environ.get('DINKSTER_REMOTE_GATEWAY_BASE') == {expected!r}\n" + HEALTHY_NODES
    )
    spec = compose.PackSpec(
        pack / "dinkster-pack.toml",
        packs={"dinkster-nodes-remote": PackInfo("Remote")},
        env={"PYTHONPATH": str(pack)},
        in_process=True,
    )
    monkeypatch.setattr(compose, "default_pack_specs", lambda: (spec,))
    arguments = ["pack", command, "--defaults"]
    if endpoint is not None:
        arguments += ["--remote-catalog-base", endpoint]
    assert main(arguments) == 0


def test_doctor_subcommand_is_the_real_publish_gate(tmp_path: Path, capsys) -> None:
    """`dinkster doctor` runs the identical doctor main: same JSON report,
    same exit codes, against a real pack."""
    pack = write_pack(tmp_path / "healthy")
    assert main(["doctor", str(pack), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["reportVersion"] == 1
    assert payload["nodeTypes"] == ["healthy.doubler"]

    (pack / "healthy/__init__.py").write_text("raise Exception()\n")
    assert main(["doctor", str(pack)]) == 1
    assert "entry.unresolvable" in capsys.readouterr().out
