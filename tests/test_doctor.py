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
import subprocess
import sys
from pathlib import Path

import pytest
from dinkster_workers import detect_bubblewrap, load_manifest
from dinkster_workers.doctor import (
    DOCTOR_REPORT_VERSION,
    diagnose,
    diagnose_static,
    main,
    render_text,
)

HEALTHY_MANIFEST = """\
[pack]
name = "healthy-pack"
namespaces = ["healthy"]
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
            inputs=(InputSpec("value", TypeExpr.concrete("core.int")),),
            outputs=(OutputSpec("doubled", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls, *, value):
        return cls.outputs(doubled=value * 2)


class Tagger(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="healthy.tagger",
            inputs=(InputSpec("text", TypeExpr.concrete("core.string")),),
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
    (root / f"{module_name}.py").write_text(source)
    return manifest_path


def codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_healthy_pack_is_healthy(tmp_path: Path) -> None:
    manifest = write_pack(tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES)
    report = diagnose(manifest)
    assert report.ok, render_text(report)
    assert report.pack_name == "healthy-pack"
    assert report.node_types == ("healthy.doubler", "healthy.tagger")
    # Elapsed import time depends on host load, not just pack behavior.
    assert codes(report) <= {"import.slow"}, render_text(report)


def test_blocking_import_emits_slow_warning(tmp_path: Path) -> None:
    source = "import time\ntime.sleep(2.1)\n" + HEALTHY_NODES
    manifest = write_pack(tmp_path / "blocking", HEALTHY_MANIFEST, "healthy_nodes", source)

    report = diagnose(manifest)

    assert report.ok, render_text(report)
    assert report.node_types == ("healthy.doubler", "healthy.tagger")
    slow = [finding for finding in report.findings if finding.code == "import.slow"]
    assert len(slow) == 1, render_text(report)
    assert slow[0].severity == "warning"


@pytest.mark.parametrize("elapsed_ms, warns", [(1999.0, False), (2000.0, False), (2001.0, True)])
def test_slow_import_warning_boundary(tmp_path: Path, elapsed_ms: float, warns: bool) -> None:
    from dinkster_workers.doctor import _probe_findings, _run_probe

    path = write_pack(tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES)
    manifest = load_manifest(path)
    probed = _run_probe(manifest, None)
    assert isinstance(probed, dict), probed
    probed["import_ms"] = elapsed_ms
    findings = _probe_findings(probed, manifest.name)
    slow = [finding for finding in findings if finding.code == "import.slow"]
    assert bool(slow) is warns
    if warns:
        assert len(slow) == 1
        assert slow[0].severity == "warning"
        assert "2001ms" in slow[0].message


def test_diagnose_static_matches_full_doctor_static_prefix(tmp_path: Path) -> None:
    marker = tmp_path / "pack-code-ran"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('yes')\n" + HEALTHY_NODES
    manifest_path = write_pack(
        tmp_path / "static",
        HEALTHY_MANIFEST.replace("[pack.sandbox]\n\n", ""),
        "healthy_nodes",
        source,
    )
    manifest = load_manifest(manifest_path)

    static = diagnose_static(manifest_path.parent, manifest)
    assert static.report_version == DOCTOR_REPORT_VERSION
    assert [finding.code for finding in static.findings] == ["sandbox.needs-undeclared"]
    assert not marker.exists()

    full = diagnose(manifest_path)
    assert marker.read_text() == "yes"
    assert full.findings[: len(static.findings)] == static.findings


def test_diagnose_static_rejects_a_different_pack_root(tmp_path: Path) -> None:
    manifest_path = write_pack(
        tmp_path / "static", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES
    )

    with pytest.raises(ValueError, match="pack_root"):
        diagnose_static(tmp_path / "other", load_manifest(manifest_path))


def test_doctor_warns_until_sandbox_needs_are_declared(tmp_path: Path) -> None:
    undeclared = HEALTHY_MANIFEST.replace("[pack.sandbox]\n\n", "")
    manifest = write_pack(tmp_path / "undeclared", undeclared, "healthy_nodes", HEALTHY_NODES)

    report = diagnose(manifest)

    finding = next(item for item in report.findings if item.code == "sandbox.needs-undeclared")
    assert finding.severity == "warning"
    assert "[pack.sandbox]" in finding.fix


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


def test_missing_manifest_is_one_clear_error(tmp_path: Path) -> None:
    report = diagnose(tmp_path / "nowhere" / "dinkster-pack.toml")
    assert not report.ok
    assert codes(report) == {"manifest.invalid"}
    assert report.findings[0].fix  # diagnostics carry the fix


def test_invalid_icon_is_a_publish_gate_error(tmp_path: Path) -> None:
    """The loader warn-and-drops a bad [pack.presentation] icon; doctor is
    the publish gate, so the same validator verdict surfaces as an error
    finding with the fix attached. A valid icon adds no finding."""
    from icon_bytes import png_bytes

    manifest = HEALTHY_MANIFEST + '\n[pack.presentation]\nicon = "badge.png"\n'
    manifest_path = write_pack(tmp_path / "iconpack", manifest, "healthy_nodes", HEALTHY_NODES)

    (tmp_path / "iconpack" / "badge.png").write_bytes(png_bytes())
    report = diagnose(manifest_path)
    assert report.ok, render_text(report)
    assert "presentation.icon-invalid" not in codes(report)

    (tmp_path / "iconpack" / "badge.png").write_bytes(png_bytes(32, 32))
    report = diagnose(manifest_path)
    assert not report.ok
    finding = next(f for f in report.findings if f.code == "presentation.icon-invalid")
    assert finding.severity == "error"
    assert "64x64" in finding.message
    assert finding.fix


def test_doctor_reports_docs_coverage_unknown_nodes_and_long_descriptions(tmp_path: Path) -> None:
    nodes = HEALTHY_NODES.replace(
        'node_type="healthy.doubler",',
        'node_type="healthy.doubler",\n            description="' + ("x" * 241) + '",',
    )
    manifest = HEALTHY_MANIFEST + '\n[pack.docs]\ndir = "docs"\ndefault_locale = "en"\n'
    manifest_path = write_pack(tmp_path / "docpack", manifest, "healthy_nodes", nodes)
    for node_type in ("healthy.doubler", "healthy.unknown"):
        directory = tmp_path / "docpack" / "docs" / "nodes" / node_type
        directory.mkdir(parents=True)
        (directory / "en.md").write_text(
            '+++\ntitle = "Node"\nsummary = "Node help."\nschema_version = 1\n+++\n'
        )

    report = diagnose(manifest_path)
    assert report.docs_coverage is not None
    assert report.docs_coverage.total_nodes == 2
    assert dict(report.docs_coverage.locale_counts) == {"en": 1, "zh": 0}
    assert codes(report) >= {
        "docs.description-too-long",
        "docs.unknown-node",
    }
    text = render_text(report)
    assert "docs coverage:" in text
    assert "en      1/2" in text
    assert "zh      0/2" in text

    plain_manifest = write_pack(tmp_path / "plain-pack", HEALTHY_MANIFEST, "healthy_nodes", nodes)
    assert "docs.description-too-long" in codes(diagnose(plain_manifest))


@pytest.mark.parametrize(
    ("declaration", "message"),
    (
        ('dir = ["docs"]', "must be a non-empty string"),
        ('dir = "missing"', "is not a directory"),
        ('dir = "../outside"', "must be relative"),
    ),
)
def test_doctor_reports_invalid_docs_declarations(
    tmp_path: Path, declaration: str, message: str
) -> None:
    manifest = HEALTHY_MANIFEST + f"\n[pack.docs]\n{declaration}\n"
    manifest_path = write_pack(tmp_path / "docpack", manifest, "healthy_nodes", HEALTHY_NODES)

    report = diagnose(manifest_path)

    assert not report.ok
    finding = next(item for item in report.findings if item.code == "docs.invalid")
    assert finding.severity == "error"
    assert message in finding.message
    assert finding.fix


def test_doctor_reports_symlinked_docs_tree(tmp_path: Path) -> None:
    manifest = HEALTHY_MANIFEST + '\n[pack.docs]\ndir = "docs"\n'
    manifest_path = write_pack(tmp_path / "docpack", manifest, "healthy_nodes", HEALTHY_NODES)
    docs = manifest_path.parent / "docs"
    docs.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n")
    try:
        (docs / "linked.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    report = diagnose(manifest_path)

    assert not report.ok
    finding = next(item for item in report.findings if item.code == "docs.invalid")
    assert finding.severity == "error"
    assert "symlink" in finding.message


@pytest.mark.parametrize(
    ("body", "message"),
    (
        ('blueprint = "other.starter"\n', "references unknown blueprint 'other.starter'"),
        ('caption = "No target"\n', "must declare exactly one blueprint or template"),
        (
            'blueprint = "starter"\ntemplate = "workflow"\n',
            "must declare exactly one blueprint or template",
        ),
        ('blueprint = "starter"\ncaption = true\n', "caption must be a string"),
    ),
)
def test_doctor_reports_invalid_doc_example_blocks(tmp_path: Path, body: str, message: str) -> None:
    manifest = HEALTHY_MANIFEST + (
        '\n[pack.docs]\ndir = "docs"\n'
        '[[pack.blueprints]]\nid = "starter"\nname = "Starter"\nfile = "starter.json"\n'
        '[[pack.templates]]\nid = "workflow"\nname = "Workflow"\nfile = "workflow.json"\n'
    )
    manifest_path = write_pack(tmp_path / "docpack", manifest, "healthy_nodes", HEALTHY_NODES)
    (manifest_path.parent / "starter.json").write_text("{}")
    (manifest_path.parent / "workflow.json").write_text("{}")
    guide = manifest_path.parent / "docs" / "guides" / "examples"
    guide.mkdir(parents=True)
    (guide / "en.md").write_text(
        '+++\ntitle = "Examples"\nsummary = "Example workflows."\n+++\n'
        f"```dinkster-example\n{body}```\n"
    )

    loaded = load_manifest(manifest_path)
    assert loaded.docs is not None
    assert loaded.docs.pages == ()
    assert len(loaded.docs.validation_problems) == 1
    report = diagnose(manifest_path)

    assert not report.ok
    finding = next(item for item in report.findings if item.code == "docs.invalid")
    assert finding.severity == "error"
    assert str(Path("docs") / "guides" / "examples" / "en.md") in finding.message
    assert message in finding.message
    assert finding.fix


@pytest.mark.parametrize(
    ("data", "message"),
    (
        ('{"nodes": []}\n', "nodes must be an object"),
        ('{"nodes": {"healthy.missing": {"displayName": "Missing"}}}\n', "healthy.missing"),
    ),
)
def test_doctor_reports_invalid_locale_catalogs(tmp_path: Path, data: str, message: str) -> None:
    manifest_path = write_pack(
        tmp_path / "catalog-pack", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES
    )
    locales = manifest_path.parent / "locales"
    locales.mkdir()
    (locales / "pt-br.json").write_text(data)

    report = diagnose(manifest_path)

    assert not report.ok
    finding = next(item for item in report.findings if item.code == "docs.invalid")
    assert finding.severity == "error"
    assert message in finding.message
    assert finding.fix


def test_doctor_accepts_valid_locale_catalog(tmp_path: Path) -> None:
    manifest_path = write_pack(
        tmp_path / "catalog-pack", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES
    )
    locales = manifest_path.parent / "locales"
    locales.mkdir()
    (locales / "en.json").write_text(
        '{"nodes":{"healthy.doubler":{"displayName":"Doubler"}},'
        '"searchTerms":{"healthy.doubler":["double"]}}\n'
    )

    report = diagnose(manifest_path)

    assert report.ok, render_text(report)
    assert "docs.invalid" not in codes(report)


def test_doctor_reports_locale_catalog_file_failures(tmp_path: Path) -> None:
    from dinkster_workers.manifest import LOCALE_CATALOG_MAX_BYTES

    manifest_path = write_pack(
        tmp_path / "catalog-pack", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES
    )
    locales = manifest_path.parent / "locales"
    locales.mkdir()
    (locales / "PT-br.json").write_text("{}")
    (locales / "bad.json").write_text("{")
    (locales / "zh.json").write_bytes(b" " * (LOCALE_CATALOG_MAX_BYTES + 1))
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    try:
        (locales / "linked.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    report = diagnose(manifest_path)

    findings = [finding for finding in report.findings if finding.code == "docs.invalid"]
    assert not report.ok
    assert len(findings) == 4
    assert all(finding.severity == "error" for finding in findings)
    assert all(finding.fix for finding in findings)
    messages = "\n".join(finding.message for finding in findings)
    assert "canonical lowercase" in messages
    assert "invalid JSON" in messages
    assert "byte cap" in messages
    assert "symlink" in messages


def test_doctor_reports_malformed_catalog_pack_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dinkster_workers.manifest as manifest_module

    manifest_path = write_pack(
        tmp_path / "catalog-pack", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES
    )
    locales = manifest_path.parent / "locales"
    locales.mkdir()
    (locales / "en.json").write_text("{")
    (locales / "fr.json").write_text("{")
    monkeypatch.setattr(manifest_module, "LOCALE_CATALOG_PACK_MAX_BYTES", 1)

    report = diagnose(manifest_path)

    findings = [finding for finding in report.findings if finding.code == "docs.invalid"]
    assert not report.ok
    assert len(findings) == 2
    assert all(finding.severity == "error" for finding in findings)
    assert all(finding.fix for finding in findings)
    assert any("invalid JSON" in finding.message for finding in findings)
    assert any("pack budget" in finding.message for finding in findings)


def test_doctor_reports_unknown_doc_node_reference(tmp_path: Path) -> None:
    manifest = HEALTHY_MANIFEST + '\n[pack.docs]\ndir = "docs"\n'
    manifest_path = write_pack(tmp_path / "docpack", manifest, "healthy_nodes", HEALTHY_NODES)
    guide = manifest_path.parent / "docs" / "guides" / "node-help"
    guide.mkdir(parents=True)
    (guide / "en.md").write_text(
        '+++\ntitle = "Node help"\nsummary = "References nodes."\n+++\n'
        '```dinkster-node\nnode = "healthy.doubler"\n```\n'
        '```dinkster-node\nnode = "healthy.missing"\n```\n'
    )

    report = diagnose(manifest_path)

    finding = next(item for item in report.findings if item.code == "docs.unknown-node-reference")
    assert finding.severity == "warning"
    assert "healthy.missing" in finding.message
    assert "healthy.doubler" not in finding.message
    assert finding.fix


def test_invalid_blueprint_is_a_publish_gate_error(tmp_path: Path) -> None:
    """The loader warn-and-drops bad [[pack.blueprints]] entries; doctor is
    the publish gate, so the same validator verdicts surface as error
    findings with fixes attached. Valid blueprints add no finding."""
    manifest = HEALTHY_MANIFEST + (
        '\n[[pack.blueprints]]\nid = "starter"\nname = "Starter"\nfile = "starter.json"\n'
    )
    manifest_path = write_pack(tmp_path / "bppack", manifest, "healthy_nodes", HEALTHY_NODES)

    (tmp_path / "bppack" / "starter.json").write_text('{"graphs": {}}')
    report = diagnose(manifest_path)
    assert report.ok, render_text(report)
    assert "blueprints.invalid" not in codes(report)

    # Malformed JSON body: same validator verdict as the loader, as error.
    (tmp_path / "bppack" / "starter.json").write_text("{nope")
    report = diagnose(manifest_path)
    assert not report.ok
    finding = next(f for f in report.findings if f.code == "blueprints.invalid")
    assert finding.severity == "error"
    assert "JSON" in finding.message
    assert finding.fix

    # Duplicate ids are their own code - the loader would keep the first
    # declaration and drop this one.
    (tmp_path / "bppack" / "starter.json").write_text('{"graphs": {}}')
    duplicated = manifest + (
        '\n[[pack.blueprints]]\nid = "starter"\nname = "Again"\nfile = "starter.json"\n'
    )
    manifest_path.write_text(duplicated)
    report = diagnose(manifest_path)
    assert not report.ok
    assert "blueprints.duplicate-id" in codes(report)


def test_invalid_template_is_a_publish_gate_error(tmp_path: Path) -> None:
    """The loader warn-and-drops bad [[pack.templates]] entries; doctor is
    the publish gate, so the same validator verdicts surface as error
    findings with fixes attached - including the referential rule: every
    referenced asset id must exist among the pack's [[pack.assets]]
    declarations. Valid templates add no finding."""
    from dinkster_assets import digest_bytes

    weights_digest = digest_bytes(b"weights")
    manifest = HEALTHY_MANIFEST + (
        "\n[[pack.assets]]\n"
        f'id = "model"\nname = "Model"\ndigest = "{weights_digest}"\n'
        'file = "model.bin"\n'
        "[[pack.templates]]\n"
        'id = "starter"\nname = "Starter"\nfile = "starter.json"\n'
        'assets = ["model"]\n'
    )
    manifest_path = write_pack(tmp_path / "tppack", manifest, "healthy_nodes", HEALTHY_NODES)

    (tmp_path / "tppack" / "model.bin").write_bytes(b"weights")
    (tmp_path / "tppack" / "starter.json").write_text('{"graphs": {}}')
    report = diagnose(manifest_path)
    assert report.ok, render_text(report)
    assert "templates.invalid" not in codes(report)

    # Malformed JSON body: same validator verdict as the loader, as error.
    (tmp_path / "tppack" / "starter.json").write_text("{nope")
    report = diagnose(manifest_path)
    assert not report.ok
    finding = next(f for f in report.findings if f.code == "templates.invalid")
    assert finding.severity == "error"
    assert "JSON" in finding.message
    assert finding.fix

    # A dangling asset reference is its own code - the loader would drop
    # the template because its acquisition plan cannot be constructed.
    (tmp_path / "tppack" / "starter.json").write_text('{"graphs": {}}')
    dangling = manifest.replace('assets = ["model"]', 'assets = ["no-such-asset"]')
    manifest_path.write_text(dangling)
    report = diagnose(manifest_path)
    assert not report.ok
    finding = next(f for f in report.findings if f.code == "templates.dangling-asset")
    assert finding.severity == "error"
    assert "no-such-asset" in finding.message
    assert finding.fix

    # Duplicate ids are their own code - the loader would keep the first
    # declaration and drop this one.
    duplicated = manifest + (
        '\n[[pack.templates]]\nid = "starter"\nname = "Again"\nfile = "starter.json"\n'
    )
    manifest_path.write_text(duplicated)
    report = diagnose(manifest_path)
    assert not report.ok
    assert "templates.duplicate-id" in codes(report)


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
    report = diagnose(manifest_path)
    assert report.ok, render_text(report)
    assert "assets.invalid" not in codes(report)

    # Missing packaged file: same validator verdict as the loader, as error.
    (models / "aux.bin").unlink()
    report = diagnose(manifest_path)
    assert not report.ok
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
    report = diagnose(manifest_path)
    assert not report.ok
    assert "assets.duplicate-id" in codes(report)


def test_uncovered_node_type_is_an_error(tmp_path: Path) -> None:
    """Every probed node type must fall under a declared namespace claim
    (DESIGN M8): the default claim is the pack name, and healthy-pack does
    not cover healthy.* - the registry and composition both refuse this,
    so doctor says it first, naming the node type and the claims."""
    manifest = HEALTHY_MANIFEST.replace('namespaces = ["healthy"]\n', "")
    manifest_path = write_pack(tmp_path / "uncovered", manifest, "healthy_nodes", HEALTHY_NODES)
    report = diagnose(manifest_path)
    assert not report.ok
    uncovered = [f for f in report.findings if f.code == "namespace.uncovered"]
    assert len(uncovered) == 2  # both healthy.* node types
    assert uncovered[0].severity == "error"
    assert "healthy.doubler" in uncovered[0].message
    assert "healthy-pack" in uncovered[0].message  # the claims it missed
    assert "[pack] namespaces" in uncovered[0].fix


def test_executed_node_types_need_no_namespace_claim(tmp_path: Path) -> None:
    """Node types in [pack] executes are covered by their owning pack's
    claim, exactly as composition treats them - a pure executor that
    claims nothing is healthy."""
    manifest = HEALTHY_MANIFEST.replace(
        'namespaces = ["healthy"]\n',
        'namespaces = []\nexecutes = ["healthy.doubler", "healthy.tagger"]\n',
    )
    manifest_path = write_pack(tmp_path / "executor", manifest, "healthy_nodes", HEALTHY_NODES)
    report = diagnose(manifest_path)
    assert "namespace.uncovered" not in codes(report)


def test_reserved_namespace_claim_warns(tmp_path: Path) -> None:
    """A claim under a reserved root (std/comfy/core/dinkster) is legal shape
    but composes only under host trust and is never granted by the
    registry - a warning naming the root, not an error (doctor cannot know
    it is not looking at a first-party pack)."""
    manifest = HEALTHY_MANIFEST.replace(
        'namespaces = ["healthy"]', 'namespaces = ["healthy", "std.extras"]'
    )
    manifest_path = write_pack(tmp_path / "reserved", manifest, "healthy_nodes", HEALTHY_NODES)
    report = diagnose(manifest_path)
    assert report.ok  # warning, not error
    reserved = [f for f in report.findings if f.code == "namespace.reserved"]
    assert len(reserved) == 1
    assert "'std.extras'" in reserved[0].message
    assert "'std'" in reserved[0].message


def test_overlapping_namespace_claims_are_manifest_invalid(tmp_path: Path) -> None:
    """Self-overlapping claims are a load-time ManifestError, so doctor
    surfaces them through its manifest.invalid finding."""
    manifest = HEALTHY_MANIFEST.replace(
        'namespaces = ["healthy"]', 'namespaces = ["healthy", "healthy.extra"]'
    )
    manifest_path = write_pack(tmp_path / "overlap", manifest, "healthy_nodes", HEALTHY_NODES)
    report = diagnose(manifest_path)
    assert not report.ok
    assert codes(report) == {"manifest.invalid"}
    assert "overlap" in report.findings[0].message


def test_unpinned_dependency_warns_with_fix(tmp_path: Path) -> None:
    manifest = write_pack(
        tmp_path / "unpinned",
        HEALTHY_MANIFEST.replace('"numpy>=1.26"', '"numpy"'),
        "healthy_nodes",
        HEALTHY_NODES,
    )
    report = diagnose(manifest)
    assert report.ok  # a warning, not an error
    unpinned = [f for f in report.findings if f.code == "manifest.unpinned-dep"]
    assert len(unpinned) == 1
    assert "numpy" in unpinned[0].message
    assert unpinned[0].fix


def test_invalid_requirement_syntax_is_an_error(tmp_path: Path) -> None:
    """A requirement uv would refuse at provision time is caught here,
    before any user hits it - as an error naming the offending list."""
    manifest = write_pack(
        tmp_path / "badreq",
        HEALTHY_MANIFEST.replace('"numpy>=1.26"', '"numpy >=>< 1.0"'),
        "healthy_nodes",
        HEALTHY_NODES,
    )
    report = diagnose(manifest)
    assert not report.ok
    bad = [f for f in report.findings if f.code == "manifest.invalid-requirement"]
    assert len(bad) == 1
    assert "[pack] requires" in bad[0].message
    assert bad[0].fix


def test_environment_checks_are_advisory_and_gpu_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Platform/accelerator fit is context, not a defect: an excluded
    host platform and an uncovered host accelerator are info findings
    that never fail the report, computed from driver-file detection -
    no torch import, no GPU context. Extra-requires lists get the same
    PEP 508 validation as base requires."""
    import dinkster_workers.doctor as doctor_module

    foreign = "win32" if sys.platform != "win32" else "linux"
    manifest = write_pack(
        tmp_path / "envpack",
        HEALTHY_MANIFEST.replace(
            'requires = ["numpy>=1.26"]',
            f'requires = ["numpy>=1.26"]\nplatforms = ["{foreign}"]\n'
            f'[pack.extra-requires]\ncuda = ["torch>=2.4"]\nrocm = ["torch =bad= 1"]',
        ),
        "healthy_nodes",
        HEALTHY_NODES,
    )
    monkeypatch.setattr(doctor_module, "detect_accelerator", lambda: "cpu")
    report = diagnose(manifest)
    found = codes(report)
    assert "environment.platform-excluded" in found
    assert "environment.accelerator-not-covered" in found
    infos = [f for f in report.findings if f.code.startswith("environment.")]
    assert all(f.severity == "info" for f in infos)
    bad = [f for f in report.findings if f.code == "manifest.invalid-requirement"]
    assert len(bad) == 1 and "[pack.extra-requires] rocm" in bad[0].message
    assert not report.ok  # the syntax error gates; the infos never would


def test_private_and_host_imports_are_errors(tmp_path: Path) -> None:
    source = (
        "from dinkster_workers._doctor_probe import probe\n"
        "from dinkster_engine import Engine\n"
        # The execution boundary is host machinery too: packs author
        # through dinkster_api, never against Invocation/Worker directly.
        "from dinkster_protocol import Invocation\n"
        "import dinkster_schema\n" + HEALTHY_NODES
    )
    manifest = write_pack(tmp_path / "trespasser", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    assert not report.ok
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
    assert by_code["imports.private-module"].location.endswith(":1")


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

    report = diagnose(manifest)

    cross_pack = [
        finding for finding in report.findings if finding.code == "imports.pack-implementation"
    ]
    assert len(cross_pack) == 1
    assert cross_pack[0].severity == "error"
    assert "dinkster_nodes_foreign" in cross_pack[0].message
    assert "registered ids" in cross_pack[0].fix
    assert not any("dinkster_nodes_healthy" in finding.message for finding in cross_pack)


def test_import_side_effects_and_thread_spawns_warn(tmp_path: Path) -> None:
    source = (
        "import threading\n"
        'print("loading my pack!!")\n'
        "_t = threading.Thread(target=lambda: threading.Event().wait(), "
        "daemon=True)\n"
        "_t.start()\n" + HEALTHY_NODES
    )
    manifest = write_pack(tmp_path / "noisy", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    found = codes(report)
    assert "import.side-effect-output" in found
    assert "import.side-effect-threads" in found
    output = next(f for f in report.findings if f.code == "import.side-effect-output")
    assert "loading my pack" in output.message


def test_cuda_init_at_import_warns(tmp_path: Path) -> None:
    """A pack whose import initializes a CUDA context gets flagged: an idle
    context holds ~450 MiB of VRAM per worker process for its whole
    lifetime (measured, RTX 4090) before any work runs, so the first GPU
    op belongs in execute(), never at import. The probe reads
    sys.modules["torch"].cuda.is_initialized(), so a stub torch exercises
    the seam without a GPU."""
    root = tmp_path / "hungry"
    root.mkdir(parents=True)
    (root / "torch.py").write_text(
        "class cuda:\n    @staticmethod\n    def is_initialized():\n        return True\n"
    )
    source = "import torch  # stub: pretends import created a context\n" + (HEALTHY_NODES)
    manifest = write_pack(root, HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    assert "import.side-effect-cuda" in codes(report)
    finding = next(f for f in report.findings if f.code == "import.side-effect-cuda")
    assert finding.severity == "warning"
    assert "VRAM" in finding.fix


def test_import_time_pack_logging_is_structured_not_raw_output(
    tmp_path: Path,
) -> None:
    source = (
        "from dinkster_api.v1 import pack_logger\n"
        '_log = pack_logger("healthy-pack")\n'
        '_log.info("registering %d nodes", 2)\n' + HEALTHY_NODES
    )
    manifest = write_pack(tmp_path / "logging", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    found = codes(report)
    # Proper logging is an informational finding, never a raw-output warning.
    assert "import.log" in found
    assert "import.side-effect-output" not in found
    assert report.ok, render_text(report)
    log_finding = next(f for f in report.findings if f.code == "import.log")
    assert log_finding.severity == "info"
    assert "dinkster.pack.healthy-pack" in log_finding.message
    assert "registering 2 nodes" in log_finding.message


def test_logging_under_another_packs_origin_warns(tmp_path: Path) -> None:
    source = (
        "from dinkster_api.v1 import pack_logger\n"
        '_log = pack_logger("somebody-else")\n'
        '_log.info("hello")\n' + HEALTHY_NODES
    )
    manifest = write_pack(tmp_path / "spoof", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    assert "import.log-foreign-origin" in codes(report)
    finding = next(f for f in report.findings if f.code == "import.log-foreign-origin")
    assert finding.severity == "warning"
    assert "dinkster.pack.somebody-else" in finding.message
    assert "healthy-pack" in finding.fix


def test_raw_prints_still_warn_alongside_structured_logs(tmp_path: Path) -> None:
    source = (
        "from dinkster_api.v1 import pack_logger\n"
        'pack_logger("healthy-pack").info("fine")\n'
        'print("not fine")\n' + HEALTHY_NODES
    )
    manifest = write_pack(tmp_path / "mixed", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    found = codes(report)
    assert "import.log" in found
    assert "import.side-effect-output" in found
    output = next(f for f in report.findings if f.code == "import.side-effect-output")
    # The log record must not leak into the captured raw output.
    assert "fine" in output.message
    assert "dinkster.pack" not in output.message


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


def test_fallback_codec_on_crossing_type_warns(tmp_path: Path) -> None:
    source = HEALTHY_NODES.replace(
        "registry.register(\n"
        '        "healthy.tag",\n'
        "        encode=lambda obj: json.dumps(obj).encode(),\n"
        "        decode=lambda data: json.loads(data),\n"
        "    )",
        'registry.register("healthy.tag")',
    )
    manifest = write_pack(tmp_path / "fallback", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    assert report.ok  # perf finding: warning, not error (DESIGN 3.9)
    fallback = [f for f in report.findings if f.code == "types.fallback-codec"]
    assert len(fallback) == 1
    assert "healthy.tag" in fallback[0].message
    assert "encode" in fallback[0].fix


def test_unregistered_schema_type_warns(tmp_path: Path) -> None:
    source = HEALTHY_NODES.replace(
        'registry.register(\n        "healthy.tag",',
        'registry.register(\n        "healthy.other",',
    )
    manifest = write_pack(tmp_path / "unregistered", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    unregistered = [f for f in report.findings if f.code == "types.unregistered"]
    assert len(unregistered) == 1
    assert "healthy.tag" in unregistered[0].message


@pytest.mark.parametrize("open_slot", [False, True])
def test_unregistered_nested_slot_type_warns(tmp_path: Path, open_slot: bool) -> None:
    from dinkster_workers.catalog import read_catalog

    slot_type = "TypeExpr.list_of(TypeExpr.concrete('healthy.slot_value'))"
    declaration = (
        f"slot_type={slot_type}" if open_slot else f"variants=(SlotVariant('value', {slot_type}),)"
    )
    source = (
        HEALTHY_NODES
        + f"""
from dinkster_api.v1 import DynamicSlotSpec, InputFamilySpec, SlotVariant

class Slot(Node):
    @classmethod
    def define_schema(cls):
        slot = DynamicSlotSpec('socket', {declaration})
        return NodeSchema(
            node_type='healthy.slot',
            input_families=(InputFamilySpec('items', (slot,)),),
        )

    @classmethod
    def execute(cls, **inputs):
        return cls.outputs()

NODES = [Slot]
"""
    )
    manifest = write_pack(tmp_path / "slot", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    assert report.ok, render_text(report)
    unregistered = [finding for finding in report.findings if finding.code == "types.unregistered"]
    assert len(unregistered) == 1
    assert "healthy.slot_value" in unregistered[0].message
    assert read_catalog(load_manifest(manifest)) is not None


def test_empty_pack_warns(tmp_path: Path) -> None:
    source = HEALTHY_NODES + "\n\nNODES = []\n"
    manifest = write_pack(tmp_path / "empty", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest)
    assert "entry.no-nodes" in codes(report)


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


def test_diagnose_wraps_the_probe_argv_in_the_jail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import ProbeJail

    manifest = write_pack(tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES)
    plain = [sys.executable, "-m", "dinkster_workers._doctor_probe", str(manifest)]
    jail = ProbeJail(bwrap="/fake/bwrap", user_namespaces=True)
    seen: list[list[str]] = []
    seen_kwargs: list[dict[str, object]] = []
    seen_environment_files: list[dict[str, str]] = []
    real_run = subprocess.run

    def spy(argv: list[str], **kwargs: object) -> object:
        seen.append(list(argv))
        seen_kwargs.append(kwargs)
        inherited = kwargs["pass_fds"]
        assert isinstance(inherited, tuple) and len(inherited) == 1
        with os.fdopen(os.dup(inherited[0]), "rb") as environment_file:
            seen_environment_files.append(json.load(environment_file))
        runnable = dict(kwargs)
        runnable.pop("pass_fds", None)
        runnable["env"] = None
        return real_run(plain, **runnable)  # type: ignore[arg-type]

    monkeypatch.setattr("dinkster_workers.doctor.preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr("dinkster_workers.doctor.subprocess.run", spy)
    explicit_environment = {"DINKSTER_EXPLICIT_SENTINEL": "delivered-by-fd"}
    report = diagnose(manifest, probe_jail=jail, environment=explicit_environment)
    assert report.ok, render_text(report)
    # the import probe is doctor's only subprocess, and it went through bwrap
    assert len(seen) == 1
    wrapped = seen[0]
    assert seen_environment_files == [explicit_environment]
    assert seen_kwargs[0]["env"] == {}
    assert len(seen_kwargs[0]["pass_fds"]) == 1  # type: ignore[arg-type]
    assert wrapped[0] == "/fake/bwrap"
    assert "--unshare-net" in wrapped
    assert "--setenv" not in wrapped
    assert "delivered-by-fd" not in wrapped
    # the original probe invocation survives at the tail, past the rlimit stub
    assert wrapped[-3:] == plain[1:]
    # the pack root rides in read-only
    root_binds = [wrapped[i + 1] for i, a in enumerate(wrapped) if a == "--ro-bind"]
    assert str(manifest.parent) in root_binds


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


def test_cli_sandbox_passes_the_detected_jail_to_diagnose(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_workers import ProbeJail

    manifest = write_pack(tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", HEALTHY_NODES)
    jail = ProbeJail(bwrap="/fake/bwrap", user_namespaces=True)
    monkeypatch.setattr("dinkster_workers.doctor.detect_probe_jail", lambda: jail)
    received: list[object] = []

    def fake_diagnose(path: Path, *, probe_jail: object = None) -> object:
        received.append(probe_jail)
        return diagnose(path)  # real, unjailed - the wiring is under test

    monkeypatch.setattr("dinkster_workers.doctor.diagnose", fake_diagnose)
    assert main([str(manifest), "--sandbox", "--json"]) == 0
    assert received == [jail]
    assert json.loads(capsys.readouterr().out)["ok"] is True


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

    def test_explicit_pythonpath_is_available_to_pack_imports(self, tmp_path: Path) -> None:
        from dinkster_workers import detect_probe_jail

        root = tmp_path / "explicit-pythonpath"
        dependency = root / "dependency"
        dependency.mkdir(parents=True)
        (dependency / "probe_dependency.py").write_text("VALUE = 'available'\n")
        source = (
            HEALTHY_NODES
            + "\nfrom probe_dependency import VALUE\n"
            + "assert VALUE == 'available'\n"
        )
        manifest = write_pack(root, HEALTHY_MANIFEST, "healthy_nodes", source)
        report = diagnose(
            manifest,
            probe_jail=detect_probe_jail(),
            environment={"PYTHONPATH": str(dependency)},
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
