"""dinkster doctor: the pack linter (DESIGN 3.9).

Doctor's promise is behavioral: a well-authored pack comes back healthy,
and each documented bad practice produces its specific finding code with a
fix attached - contract violations as errors, perf drift as warnings. The
probe runs pack imports in a subprocess, so a noisy or broken import is a
finding here, never pollution of the test process.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from dinkster_protocol import EXTENSION_SCOPES
from dinkster_workers import detect_bubblewrap, load_manifest
from dinkster_workers.doctor import (
    diagnose,
    diagnose_static,
    main,
    render_text,
)

HEALTHY_MANIFEST = """\
[pack]
name = "healthy-nodes"
namespaces = ["healthy"]
type-namespaces = ["healthy"]
publisher = "dinkster"
requires = ["numpy>=1.26"]

[pack.sandbox]

[pack.entry]
nodes = "healthy_nodes:NODES"
types = "healthy_nodes:register_types"
"""

HEALTHY_NODES = """\
import json

from dinkster_api.v1 import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
)


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


class Tagger(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="healthy.tagger",
            inputs=(InputSpec("text", TypeExpr.concrete("STRING")),),
            outputs=(OutputSpec("tag", TypeExpr.concrete("healthy.tag")),),
        )

    @classmethod
    async def execute(cls, *, text):
        return cls.outputs(tag={"label": text})


def register_types(registry: TypeRegistry) -> None:
    registry.register(
        "healthy.tag",
        encode=lambda obj: json.dumps(obj).encode(),
        decode=lambda data: json.loads(data),
    )


NODES = [Doubler, Tagger]
"""


def write_pack(root: Path, manifest: str, module_name: str, source: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "dinkster-pack.toml"
    manifest_path.write_text(manifest)
    package = root / module_name
    package.mkdir()
    (package / "__init__.py").write_text(source)
    return manifest_path


def codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_healthy_pack_is_healthy(tmp_path: Path) -> None:
    manifest = write_pack(tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES)
    report = diagnose(manifest)
    assert report.ok, render_text(report)
    assert report.pack_name == "healthy-nodes"
    assert report.entry_path == str((manifest.parent / "healthy_nodes/__init__.py").resolve())
    assert report.interpreter == str(Path(sys.executable).absolute())
    assert report.node_types == ("healthy.doubler", "healthy.tagger")
    payload = json.loads(report.to_json())
    assert payload["entryPath"] == report.entry_path
    assert payload["interpreter"] == report.interpreter
    text = render_text(report)
    assert f"  entry: {report.entry_path}" in text
    assert f"  interpreter: {report.interpreter}" in text
    # Elapsed import time depends on host load, not just pack behavior.
    assert codes(report) <= {"import.slow"}, render_text(report)


def test_pack_nested_in_distribution_does_not_add_undeclared_source_root(tmp_path: Path) -> None:
    distribution = tmp_path / "distribution"
    manifest = distribution / "healthy_pack" / "dinkster-pack.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(HEALTHY_MANIFEST)
    (distribution / "pyproject.toml").write_text("[project]\nname = 'healthy'\nversion = '1'\n")
    source = distribution / "src"
    source.mkdir()
    (source / "healthy_nodes.py").write_text(HEALTHY_NODES)

    report = diagnose(manifest)

    assert not report.ok
    assert "entry.unresolvable" in codes(report)
    assert report.entry_path == ""


def test_doctor_python_runs_probe_under_selected_interpreter(tmp_path: Path) -> None:
    manifest = write_pack(tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES)
    marker = tmp_path / "selected-python.txt"
    if os.name == "nt":
        wrapper = tmp_path / "python-wrapper.cmd"
        wrapper.write_text(
            f'@echo off\r\n<nul set /p="selected" > "{marker}"\r\n"{sys.executable}" %*\r\n'
        )
    else:
        wrapper = tmp_path / "python-wrapper"
        wrapper.write_text(f'#!/bin/sh\nprintf selected > {marker}\nexec "{sys.executable}" "$@"\n')
        wrapper.chmod(0o755)

    assert main(["--python", str(wrapper), str(manifest)]) == 0
    assert marker.read_text() == "selected"


def test_invalid_asset_is_a_publish_gate_error(tmp_path: Path) -> None:
    """The loader warn-and-drops bad [[pack.assets]] entries; doctor is the
    publish gate, so the same validator verdicts surface as error findings
    with fixes attached. Valid declarations add no finding."""
    from dinkster_assets import digest_bytes

    model_bytes = b"doctor aux model" * 8
    digest = digest_bytes(model_bytes)
    manifest = HEALTHY_MANIFEST + (
        "\n[[pack.assets]]\n"
        f'id = "aux"\nname = "Aux Model"\ndigest = "{digest}"\n'
        'file = "models/aux.bin"\n'
    )
    manifest_path = write_pack(tmp_path / "assetpack", manifest, "healthy_nodes", HEALTHY_NODES)
    models = tmp_path / "assetpack" / "models"
    models.mkdir()
    (models / "aux.bin").write_bytes(model_bytes)
    report = diagnose_static(manifest_path.parent, load_manifest(manifest_path))
    assert not any(f.severity == "error" for f in report.findings), report
    assert "assets.invalid" not in codes(report)

    # Missing packaged file: same validator verdict as the loader, as error.
    (models / "aux.bin").unlink()
    report = diagnose_static(manifest_path.parent, load_manifest(manifest_path))
    assert any(f.severity == "error" for f in report.findings)
    finding = next(f for f in report.findings if f.code == "assets.invalid")
    assert finding.severity == "error"
    assert "does not exist" in finding.message
    assert finding.fix

    # Duplicate ids are their own code - the loader would keep the first
    # declaration and drop this one.
    (models / "aux.bin").write_bytes(model_bytes)
    duplicated = manifest + (
        "\n[[pack.assets]]\n"
        f'id = "aux"\nname = "Again"\ndigest = "{digest}"\n'
        'urls = ["https://hub.example/aux.bin"]\n'
    )
    manifest_path.write_text(duplicated)
    report = diagnose_static(manifest_path.parent, load_manifest(manifest_path))
    assert any(f.severity == "error" for f in report.findings)
    assert "assets.duplicate-id" in codes(report)


def test_uncovered_node_type_is_an_error(tmp_path: Path) -> None:
    """Every probed node type must fall under a declared namespace claim
    (DESIGN M8): the default claim is the pack name, and healthy does
    not cover healthy.* - the registry and composition both refuse this,
    so doctor says it first, naming the node type and the claims."""
    manifest = HEALTHY_MANIFEST.replace(
        'name = "healthy-nodes"\nnamespaces = ["healthy"]\n',
        'name = "healthy-pack"\n',
    )
    manifest_path = write_pack(tmp_path / "uncovered", manifest, "healthy_nodes", HEALTHY_NODES)
    report = diagnose(manifest_path)
    assert not report.ok
    uncovered = [f for f in report.findings if f.code == "namespace.uncovered"]
    assert len(uncovered) == 2  # both healthy.* node types
    assert uncovered[0].severity == "error"
    assert "healthy.doubler" in uncovered[0].message
    assert "healthy-pack" in uncovered[0].message  # the claims it missed
    assert "[pack] namespaces" in uncovered[0].fix


def test_invalid_requirement_syntax_is_an_error(tmp_path: Path) -> None:
    """A requirement uv would refuse at provision time is caught here,
    before any user hits it - as an error naming the offending list."""
    manifest = write_pack(
        tmp_path / "badreq",
        HEALTHY_MANIFEST.replace('"numpy>=1.26"', '"numpy >=>< 1.0"'),
        "healthy_nodes",
        HEALTHY_NODES,
    )
    report = diagnose_static(manifest.parent, load_manifest(manifest))
    assert any(f.severity == "error" for f in report.findings)
    bad = [f for f in report.findings if f.code == "manifest.invalid-requirement"]
    assert len(bad) == 1
    assert "[pack] requires" in bad[0].message
    assert bad[0].fix


def test_private_and_host_imports_are_errors(tmp_path: Path) -> None:
    source = (
        "from dinkster_workers._doctor_probe import probe\n"
        "from dinkster_engine import Engine\n"
        "from dinkster_engine.engine import Engine\n"
        "from dinkster_engine.testing import _engine\n"
        # The execution boundary is host machinery too: packs author
        # through dinkster_api, never against Invocation/Worker directly.
        "from dinkster_protocol import Invocation\n"
        "import dinkster_schema\n" + HEALTHY_NODES
    )
    manifest = write_pack(tmp_path / "trespasser", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose_static(manifest.parent, load_manifest(manifest))
    assert any(f.severity == "error" for f in report.findings)
    by_code = {f.code: f for f in report.findings}
    assert by_code["imports.private-module"].severity == "error"
    assert by_code["imports.host-machinery"].severity == "error"
    machinery = [f for f in report.findings if f.code == "imports.host-machinery"]
    named = {f.message for f in machinery}
    assert any("dinkster_engine" in m for m in named)
    assert any("dinkster_protocol" in m for m in named)
    assert by_code["imports.internal-module"].severity == "warning"
    assert "dinkster_api.v1" in by_code["imports.internal-module"].fix
    # findings point at the offending file:line
    assert any(
        finding.code == "imports.private-module" and finding.location.endswith(":1")
        for finding in report.findings
    )


def test_cross_pack_implementation_import_is_an_error_but_own_module_is_allowed(
    tmp_path: Path,
) -> None:
    manifest_text = HEALTHY_MANIFEST.replace(
        'nodes = "healthy_nodes:NODES"', 'nodes = "dinkster_nodes_healthy:NODES"'
    ).replace(
        'types = "healthy_nodes:register_types"',
        'types = "dinkster_nodes_healthy:register_types"',
    )
    source = (
        "from typing import TYPE_CHECKING\n"
        "import dinkster_nodes_healthy\n"
        "import dinkster_nodes_healthy_helper\n"
        "if TYPE_CHECKING:\n"
        "    import os, dinkster_nodes_foreign\n" + HEALTHY_NODES
    )
    root = tmp_path / "cross-pack"
    manifest = write_pack(
        root,
        manifest_text,
        "dinkster_nodes_healthy",
        source,
    )
    (root / "dinkster_nodes_healthy_helper.py").write_text("VALUE = 1\n")

    report = diagnose_static(manifest.parent, load_manifest(manifest))

    cross_pack = [
        finding for finding in report.findings if finding.code == "imports.pack-implementation"
    ]
    assert len(cross_pack) == 1
    assert cross_pack[0].severity == "error"
    assert "dinkster_nodes_foreign" in cross_pack[0].message
    assert "registered ids" in cross_pack[0].fix
    assert not any("dinkster_nodes_healthy" in finding.message for finding in cross_pack)


def test_broken_entry_is_a_finding_not_a_crash(tmp_path: Path) -> None:
    manifest = write_pack(
        tmp_path / "broken",
        HEALTHY_MANIFEST,
        "healthy_nodes",
        'raise RuntimeError("kaboom at import")\n',
    )
    report = diagnose(manifest)
    assert not report.ok
    unresolvable = [f for f in report.findings if f.code == "entry.unresolvable"]
    assert len(unresolvable) == 1
    assert "kaboom" in unresolvable[0].message


@pytest.mark.parametrize(
    ("types_entry", "source"),
    [
        ("healthy_nodes:not_callable", HEALTHY_NODES + "\nnot_callable = 42\n"),
        (
            "healthy_nodes:register_broken_types",
            HEALTHY_NODES
            + "\ndef register_broken_types(registry):\n    raise RuntimeError('broken types')\n",
        ),
    ],
)
def test_types_entry_failures_are_actionable(tmp_path: Path, types_entry: str, source: str) -> None:
    manifest_text = HEALTHY_MANIFEST.replace(
        'types = "healthy_nodes:register_types"', f'types = "{types_entry}"'
    )
    manifest = write_pack(tmp_path / "broken-types", manifest_text, "healthy_nodes", source)

    report = diagnose(manifest)

    failed = [finding for finding in report.findings if finding.code == "entry.types-failed"]
    assert len(failed) == 1
    assert failed[0].severity == "error"
    assert failed[0].fix


@pytest.mark.parametrize("scope", EXTENSION_SCOPES)
def test_every_extension_entry_scope_must_resolve(tmp_path: Path, scope: str) -> None:
    valid_manifest_text = HEALTHY_MANIFEST + (
        f'\n[pack.extension]\n{scope} = "healthy_nodes:Doubler"\nprivileges = ["{scope}"]\n'
    )
    valid_manifest = write_pack(
        tmp_path / "valid" / scope,
        valid_manifest_text,
        "healthy_nodes",
        HEALTHY_NODES,
    )
    valid_report = diagnose(valid_manifest)
    assert all(item.code != "extension.entry-unresolvable" for item in valid_report.findings), (
        render_text(valid_report)
    )

    invalid_manifest_text = HEALTHY_MANIFEST + (
        f'\n[pack.extension]\n{scope} = "healthy_nodes:missing_{scope}"\nprivileges = ["{scope}"]\n'
    )
    invalid_manifest = write_pack(
        tmp_path / "invalid" / scope,
        invalid_manifest_text,
        "healthy_nodes",
        HEALTHY_NODES,
    )

    report = diagnose(invalid_manifest)

    findings = [item for item in report.findings if item.code == "extension.entry-unresolvable"]
    assert len(findings) == 1, render_text(report)
    assert findings[0].severity == "error"
    assert f"[pack.extension] {scope}" in findings[0].message
    assert f"missing_{scope}" in findings[0].message
    assert findings[0].fix


def test_schema_problems_and_duplicates_are_errors(tmp_path: Path) -> None:
    source = HEALTHY_NODES + (
        "\n\nclass Impostor(Doubler):\n"
        "    pass\n"
        "\n\nclass Broken(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        '        raise ValueError("no schema today")\n'
        "\n\nNODES = [Doubler, Tagger, Impostor, Broken, 42]\n"
    )
    manifest = write_pack(tmp_path / "schemas", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    assert not report.ok
    found = codes(report)
    assert "schema.duplicate-node-type" in found
    assert "schema.invalid" in found
    assert "entry.not-node-classes" in found


def test_schema_exception_detail_is_bounded_and_redacts_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "doctor-secret-value-that-must-not-leak"
    monkeypatch.setenv("DINKSTER_DOCTOR_TEST_SECRET", secret)
    source = HEALTHY_NODES + (
        "\n\nclass Broken(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        import os\n"
        '        raise ValueError(os.environ["DINKSTER_DOCTOR_TEST_SECRET"] + "x" * 10000)\n'
        "\n\nNODES = [Broken]\n"
    )
    manifest = write_pack(tmp_path / "secret", HEALTHY_MANIFEST, "healthy_nodes", source)

    report = diagnose(manifest)

    finding = next(item for item in report.findings if item.code == "schema.invalid")
    assert "ValueError" in finding.message
    assert secret not in finding.message
    assert "<redacted>" in finding.message
    assert len(finding.message) < 2200


def test_probe_crash_detail_redacts_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "doctor-probe-secret-that-must-not-leak"
    monkeypatch.setenv("DINKSTER_DOCTOR_TEST_SECRET", secret)
    source = (
        "import os\n"
        "os.write(2, b'\\xff')\n"
        "class Hostile:\n"
        "    def __repr__(self):\n"
        '        raise RuntimeError(os.environ["DINKSTER_DOCTOR_TEST_SECRET"])\n'
        "NODES = [Hostile()]\n"
    )
    manifest = write_pack(
        tmp_path / "hostile",
        HEALTHY_MANIFEST.replace('types = "healthy_nodes:register_types"\n', ""),
        "healthy_nodes",
        source,
    )

    report = diagnose(manifest)

    finding = next(item for item in report.findings if item.code == "doctor.probe-failed")
    assert secret not in finding.message
    assert "<redacted>" in finding.message
    assert len(finding.message) < 550


def test_cli_renders_and_exits_by_health(tmp_path: Path, capsys) -> None:
    manifest = write_pack(tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES)
    # pack directory (not manifest path) is accepted, --json is machine-readable
    assert main([str(manifest.parent), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["nodeTypes"] == ["healthy.doubler", "healthy.tagger"]
    # the machine interface names its own version and the contract it
    # validated against (DESIGN 5: doctor exposure plan)
    assert payload["reportVersion"] == 1
    assert payload["apiVersion"] == "v1"
    assert isinstance(payload["doctorVersion"], str) and payload["doctorVersion"]

    broken = write_pack(
        tmp_path / "broken", HEALTHY_MANIFEST, "healthy_nodes", "raise Exception()\n"
    )
    assert main([str(broken)]) == 1
    rendered = capsys.readouterr().out
    assert "entry.unresolvable" in rendered
    assert "unhealthy" in rendered

    assert main([]) == 2


# -- sandboxed probe (DESIGN M8: publish-probe isolation) ----------------------
#
# The jail changes what a hostile import can DO, never what the report
# means: findings are identical jailed or unjailed. These tests prove the
# wiring (the probe argv - doctor's only subprocess - is the thing that
# gets wrapped) and, where this machine can build a jail, the isolation
# itself.


def test_cli_sandbox_refusal_is_loud_and_leaves_stdout_clean(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_workers import SandboxUnavailable

    manifest = write_pack(tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES)

    def refuse() -> object:
        raise SandboxUnavailable("no bwrap on this machine")

    monkeypatch.setattr("dinkster_workers.doctor.detect_probe_jail", refuse)
    assert main([str(manifest), "--sandbox", "--json"]) == 2
    captured = capsys.readouterr()
    assert "refused" in captured.err and "no bwrap on this machine" in captured.err
    assert captured.out == ""  # --json consumers never see refusal noise


_JAIL_CAPABILITY = detect_bubblewrap() if sys.platform == "linux" else None


@pytest.mark.skipif(
    _JAIL_CAPABILITY is None or not _JAIL_CAPABILITY.available,
    reason=(
        "bubblewrap jail cannot be built here: "
        + (_JAIL_CAPABILITY.detail if _JAIL_CAPABILITY is not None else "not Linux")
    ),
)
class TestLiveJailedProbe:
    def test_healthy_pack_diagnoses_clean_inside_the_jail(self, tmp_path: Path) -> None:
        from dinkster_workers import detect_probe_jail

        manifest = write_pack(
            tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES
        )
        report = diagnose(manifest, probe_jail=detect_probe_jail())
        assert report.ok, render_text(report)
        assert report.node_types == ("healthy.doubler", "healthy.tagger")

    def test_explicit_environment_is_the_only_probe_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dinkster_workers import detect_probe_jail

        monkeypatch.setenv("DINKSTER_UNRELATED_HOST_SENTINEL", "must-not-leak")
        source = (
            HEALTHY_NODES
            + "\nimport os\n"
            + "expected = {'DINKSTER_EXPLICIT_SENTINEL': 'delivered'}\n"
            + "assert dict(os.environ) == expected, repr(dict(os.environ))\n"
        )
        manifest = write_pack(tmp_path / "explicit", HEALTHY_MANIFEST, "healthy_nodes", source)
        report = diagnose(
            manifest,
            probe_jail=detect_probe_jail(),
            environment={"DINKSTER_EXPLICIT_SENTINEL": "delivered"},
        )
        assert report.ok, render_text(report)

    def test_explicit_pythonpath_cannot_shadow_the_probe_harness(self, tmp_path: Path) -> None:
        from dinkster_workers import detect_probe_jail

        root = tmp_path / "shadowed-harness"
        pythonpath = root / "dependency"
        shadow = pythonpath / "dinkster_workers"
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").write_text("")
        (shadow / "_doctor_probe.py").write_text("raise SystemExit(23)\n")
        manifest = write_pack(root, HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES)
        report = diagnose(
            manifest,
            probe_jail=detect_probe_jail(),
            environment={"PYTHONPATH": str(pythonpath)},
        )
        assert report.ok, render_text(report)

    def test_hostile_import_cannot_write_outside_the_jail(self, tmp_path: Path) -> None:
        """A pack whose import drops a file next to its own root: unjailed
        that lands on the host; jailed it lands in the private tmpfs (or
        fails on the read-only bind) and the host never sees it."""
        from dinkster_workers import detect_probe_jail

        marker = tmp_path / "escaped.txt"
        source = HEALTHY_NODES + f"\n\nopen({str(marker)!r}, 'w').write('escaped')\n"
        manifest = write_pack(tmp_path / "hostile", HEALTHY_MANIFEST, "healthy_nodes", source)
        diagnose(manifest, probe_jail=detect_probe_jail())
        assert not marker.exists()
        # sanity: the same pack DOES escape without the jail
        diagnose(manifest)
        assert marker.exists()
