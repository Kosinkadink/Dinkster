"""``dinkster port`` codegen (DESIGN 3.8): a legacy v1 pack in, a doctor-clean
native pack skeleton out.

These tests need no ComfyUI: packs are synthesized on disk and probed
in-process through the same loader/translator path the subprocess probe
runs (``probe_pack``), then fed to the pure generator (``generate_pack``).
The contract under test:

- the probe carries real wire schemas plus the porting facts translation
  erases (FUNCTION source, combo choices, hidden inputs, IS_LIST),
- generation is deterministic byte-for-byte,
- the generated pack imports, its schemas match the translated ones, its
  execute() stubs refuse loudly, and ``dinkster doctor`` calls it healthy,
- refusals (unknown/skipped node selection, unloadable pack) are loud.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from dinkster_api.v1 import BooleanWidget, ComboWidget, Node, TypeExpr, TypeRegistry
from dinkster_compat_comfy.port_probe import probe_pack
from dinkster_workers.doctor import diagnose, render_text

from dinkster.port import PortError, generate_pack, main

# --- pack scaffolding ----------------------------------------------------


def write_pack(root: Path, name: str, init_source: str) -> Path:
    pack = root / name
    pack.mkdir()
    (pack / "__init__.py").write_text(init_source, encoding="utf-8")
    return pack


PORT_PACK = '''
class Adder:
    """Add two ints."""

    CATEGORY = "math"
    DESCRIPTION = "Adds a and b."
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total",)
    FUNCTION = "add"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("INT", {"default": 1}), "b": ("INT", {"default": 2})}}

    def add(self, a, b):
        return (a + b,)


class Sampler:
    CATEGORY = "sampling"
    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "image")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["fast", "slow", "exact"],),
                "add_noise": (["enable", "disable"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, model, mode, add_noise, unique_id=None):
        return (model, model)


class Batcher:
    CATEGORY = "math"
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"xs": ("INT",)}}

    def run(self, xs):
        return (sum(xs),)


class Broken:
    RETURN_TYPES = ("INT",)
    FUNCTION = "missing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT",)}}


NODE_CLASS_MAPPINGS = {
    "Adder": Adder,
    "Sampler (Fancy)": Sampler,
    "Batcher": Batcher,
    "Broken": Broken,
}
NODE_DISPLAY_NAME_MAPPINGS = {"Adder": "Add Two Ints"}
'''

SCALAR_PACK = """
class Shouter:
    CATEGORY = "text"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shouted",)
    FUNCTION = "shout"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "hi"})}}

    def shout(self, text):
        return (text.upper(),)


NODE_CLASS_MAPPINGS = {"Shouter": Shouter}
"""

# Duck-typed V3 fixtures: the probe resolves comfy_entrypoint the way
# ComfyUI's loader does and translate_v3 reads schemas structurally, so a
# pack needs no comfy_api import to exercise the real V3 path.
V3_HELPERS = """
class _Input:
    def __init__(self, id, io_type, default=None, optional=False, tooltip=None,
                 options=None, template=None):
        self.id = id
        self._io_type = io_type
        self.default = default
        self.optional = optional
        self.tooltip = tooltip
        self.options = options
        self.template = template

    def get_io_type(self):
        return self._io_type


class _Output:
    def __init__(self, io_type, id=None):
        self.id = id
        self._io_type = io_type
        self.display_name = id
        self.tooltip = None
        self.is_output_list = False

    def get_io_type(self):
        return self._io_type


class _AutogrowTemplate:
    def __init__(self, input, min=0, max=None):
        self.input = input
        self.min = min
        self.max = max
        self.names = None


class _Schema:
    def __init__(self, node_id, inputs, outputs, **extra):
        self.node_id = node_id
        self.display_name = extra.get("display_name", "")
        self.category = extra.get("category", "")
        self.description = extra.get("description", "")
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = extra.get("hidden", [])
        self.is_input_list = False
        self.is_output_node = False
        self.not_idempotent = False
        self.is_api_node = False
        self.is_deprecated = False
        self.is_dev_only = False
        self.search_aliases = extra.get("search_aliases", [])


class Scaler:
    @classmethod
    def define_schema(cls):
        return _Schema(
            "Scaler",
            [_Input("value", "FLOAT", default=1.0), _Input("factor", "FLOAT", default=2.0)],
            [_Output("FLOAT", id="scaled")],
            display_name="Scale a Float",
            category="math",
            search_aliases=["Multiplier"],
        )

    @classmethod
    def execute(cls, value, factor):
        return (value * factor,)


class Stacker:
    @classmethod
    def define_schema(cls):
        return _Schema(
            "Stacker",
            [_Input("xs", "COMFY_AUTOGROW_V3",
                    template=_AutogrowTemplate(_Input("x", "INT"), min=1, max=5))],
            [_Output("INT", id="total")],
            category="math",
        )

    @classmethod
    def execute(cls, xs):
        return (sum(xs.values()),)


class BrokenSchema:
    @classmethod
    def define_schema(cls):
        raise RuntimeError("bad schema")
"""

V3_PACK = (
    V3_HELPERS
    + """

class _Extension:
    async def on_load(self):
        self.loaded = True

    async def get_node_list(self):
        return [Scaler, Stacker, BrokenSchema]


async def comfy_entrypoint():
    return _Extension()
"""
)

MIXED_PACK = (
    """
class Adder:
    CATEGORY = "math"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total",)
    FUNCTION = "add"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("INT", {"default": 1}), "b": ("INT", {"default": 2})}}

    def add(self, a, b):
        return (a + b,)


NODE_CLASS_MAPPINGS = {"Adder": Adder}
"""
    + V3_HELPERS
    + """

class _Extension:
    def get_node_list(self):
        return [Scaler]


def comfy_entrypoint():
    return _Extension()
"""
)

BROKEN_ENTRYPOINT_PACK = """
def comfy_entrypoint():
    raise RuntimeError("no extension for you")
"""


@pytest.fixture
def port_probe_report(tmp_path: Path) -> dict[str, object]:
    pack = write_pack(tmp_path, "fancy_pack", PORT_PACK)
    return probe_pack(pack)


def load_pack_module(pack_dir: Path, module_file: str) -> ModuleType:
    """Import a generated pack module off its directory, as the worker
    host resolves manifest entries - under a unique name so parallel
    generated packs never collide in sys.modules."""
    path = pack_dir / module_file
    module_name = f"generated_{pack_dir.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


# --- probe ----------------------------------------------------------------


def test_probe_carries_schemas_and_port_extras(
    port_probe_report: dict[str, object],
) -> None:
    report = cast("dict[str, object]", port_probe_report["report"])
    assert report["status"] == "loaded"
    nodes = cast("list[dict[str, object]]", port_probe_report["nodes"])
    by_name = {cast("str", n["sourceName"]): n for n in nodes}
    assert all(n["sourceApi"] == "v1" for n in nodes)
    assert set(by_name) == {"Adder", "Sampler (Fancy)", "Batcher"}
    assert cast("dict[str, str]", report["nodes_skipped"]).keys() == {"Broken"}

    adder = by_name["Adder"]
    schema = cast("dict[str, object]", adder["schema"])
    assert schema["displayName"] == "Add Two Ints"
    assert adder["function"] == "add"
    assert "def add(self, a, b):" in cast("str", adder["source"])

    sampler = by_name["Sampler (Fancy)"]
    assert sampler["hidden"] == ["unique_id"]
    assert cast("dict[str, object]", sampler["choices"])["mode"] == [
        "fast",
        "slow",
        "exact",
    ]

    batcher = by_name["Batcher"]
    assert batcher["inputIsList"] is True

    assert port_probe_report["opaqueTypes"] == ["comfy.IMAGE", "comfy.MODEL"]


def test_probe_resolves_pure_v3_entrypoint(tmp_path: Path) -> None:
    """A pure V3 pack (async entrypoint/on_load/get_node_list) probes:
    schemas translate, per-node faults skip with reasons, porting facts
    (execute source, search aliases) ride along."""
    probe = probe_pack(write_pack(tmp_path, "v3_pack", V3_PACK))
    report = cast("dict[str, object]", probe["report"])
    assert report["status"] == "v3-entrypoint"
    nodes = cast("list[dict[str, object]]", probe["nodes"])
    by_name = {cast("str", n["sourceName"]): n for n in nodes}
    assert set(by_name) == {"Scaler", "Stacker"}
    assert all(n["sourceApi"] == "v3" for n in nodes)

    scaler = by_name["Scaler"]
    schema = cast("dict[str, object]", scaler["schema"])
    assert schema["displayName"] == "Scale a Float"
    assert scaler["function"] == "execute"
    assert "def execute(cls, value, factor):" in cast("str", scaler["source"])
    assert scaler["searchAliases"] == ["Multiplier"]

    # The autogrow input landed as a family on the wire schema.
    stacker_schema = cast("dict[str, object]", by_name["Stacker"]["schema"])
    interface = cast("list[dict[str, object]]", stacker_schema["interface"])
    roles = {cast("str", e.get("role")) for e in interface}
    assert "inputFamily" in roles

    # BrokenSchema skipped individually, siblings survived.
    assert probe["v3Skipped"] == {"BrokenSchema": "RuntimeError: bad schema"}


def test_probe_mixed_pack_carries_both_halves(tmp_path: Path) -> None:
    probe = probe_pack(write_pack(tmp_path, "mixed_pack", MIXED_PACK))
    report = cast("dict[str, object]", probe["report"])
    assert report["status"] == "loaded"
    assert report["v3_entrypoint_ignored"] is True
    nodes = cast("list[dict[str, object]]", probe["nodes"])
    apis = {cast("str", n["sourceName"]): n["sourceApi"] for n in nodes}
    assert apis == {"Adder": "v1", "Scaler": "v3"}


def test_probe_broken_entrypoint_reports_v3_error(tmp_path: Path) -> None:
    pack = write_pack(tmp_path, "broken_v3", BROKEN_ENTRYPOINT_PACK)
    probe = probe_pack(pack)
    report = cast("dict[str, object]", probe["report"])
    assert report["status"] == "v3-entrypoint"
    assert probe["nodes"] == []
    assert "no extension for you" in cast("str", probe["v3Error"])
    with pytest.raises(PortError, match="V3 entrypoint failed"):
        generate_pack(probe, "broken-v3", tmp_path / "out")


# --- generation -----------------------------------------------------------


def test_generated_pack_imports_and_schemas_survive(
    port_probe_report: dict[str, object], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = generate_pack(cast("dict[str, object]", port_probe_report), "fancy-pack", out)
    assert [n.node_type for n in result.ported] == [
        "fancy-pack.adder",
        "fancy-pack.sampler_fancy",
        "fancy-pack.batcher",
    ]
    assert result.skipped == {"Broken": "Broken: FUNCTION does not name a method"}
    assert result.opaque_types == ("comfy.IMAGE", "comfy.MODEL")

    module = load_pack_module(out, "fancy_pack_nodes.py")
    nodes = cast("list[type[Node]]", module.NODES)
    schemas = {cls.define_schema().node_type: cls.define_schema() for cls in nodes}
    assert set(schemas) == {
        "fancy-pack.adder",
        "fancy-pack.sampler_fancy",
        "fancy-pack.batcher",
    }

    adder = schemas["fancy-pack.adder"]
    assert adder.display_name == "Add Two Ints"
    assert adder.category == "math"
    assert adder.description == "Adds a and b."
    assert adder.aliases == ("Adder",)
    assert [(spec.id, spec.default) for spec in adder.inputs] == [("a", 1), ("b", 2)]
    assert adder.outputs[0].id == "total"

    sampler = schemas["fancy-pack.sampler_fancy"]
    assert sampler.output_node is True
    assert sampler.idempotent is False
    assert sampler.occupies == ("gpu",)
    # Hidden v1 inputs never enter the schema; the combo is core.combo.
    assert [spec.id for spec in sampler.inputs] == ["model", "mode", "add_noise"]
    # Translated widgets survive codegen: the combo keeps its dropdown,
    # the disguised boolean becomes a labeled toggle.
    mode_spec, noise_spec = sampler.inputs[1], sampler.inputs[2]
    assert mode_spec.type == TypeExpr.concrete("core.combo")
    assert mode_spec.widget == ComboWidget(options=("fast", "slow", "exact"))
    assert noise_spec.type == TypeExpr.concrete("core.boolean")
    assert noise_spec.widget == BooleanWidget(label_on="enable", label_off="disable")

    batcher = schemas["fancy-pack.batcher"]
    assert batcher.inputs[0].type.kind == "list"

    # Stubs refuse loudly instead of pretending fidelity.
    adder_cls = next(cls for cls in nodes if cls.define_schema().node_type == "fancy-pack.adder")
    with pytest.raises(NotImplementedError, match="fancy-pack.adder"):
        adder_cls.execute(a=1, b=2)

    # The v1 reference source rides along as comments.
    text = (out / "fancy_pack_nodes.py").read_text(encoding="utf-8")
    assert "def add(self, a, b):" in text
    assert "TODO(port)" in text
    assert "mode: str" in text  # core.combo payloads stay string-shaped
    # Choices that became widgets need no prose reminder.
    assert "source combo choices" not in text

    # register_types registers exactly the opaque types the schemas use.
    registry = TypeRegistry()
    module.register_types(registry)
    assert registry.spec("comfy.MODEL") is not None
    assert registry.spec("comfy.IMAGE") is not None


def test_type_expr_codegen_reads_wire15_variable_spelling() -> None:
    # Wire-15 spells the variable constraint set "allowed" (co-pinned
    # contract; the probe emits through schema_to_wire). The generated
    # source must reconstruct the same TypeExpr, including a real tuple
    # for a single-atom allowed set.
    from dinkster.port import _collect_atoms, _type_expr_code

    cases = {
        "TypeExpr.variable('T')": {"kind": "variable", "templateId": "T"},
        "TypeExpr.variable('T', ('comfy.IMAGE',))": {
            "kind": "variable",
            "templateId": "T",
            "allowed": ["comfy.IMAGE"],
        },
        "TypeExpr.variable('T', ('comfy.IMAGE', 'comfy.MASK'))": {
            "kind": "variable",
            "templateId": "T",
            "allowed": ["comfy.IMAGE", "comfy.MASK"],
        },
    }
    for expected_code, wire in cases.items():
        assert _type_expr_code(wire) == expected_code
        rebuilt = eval(expected_code, {"TypeExpr": TypeExpr})  # noqa: S307
        assert isinstance(rebuilt, TypeExpr)
        assert rebuilt.template_id == "T"
        assert rebuilt.types == tuple(cast("list[str]", wire.get("allowed", [])))
    atoms: set[str] = set()
    _collect_atoms(
        {
            "kind": "list",
            "element": {
                "kind": "variable",
                "templateId": "T",
                "allowed": ["comfy.IMAGE", "comfy.MASK"],
            },
        },
        atoms,
    )
    assert atoms == {"comfy.IMAGE", "comfy.MASK"}


def test_generation_is_deterministic(port_probe_report: dict[str, object], tmp_path: Path) -> None:
    probe = cast("dict[str, object]", port_probe_report)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    generate_pack(probe, "fancy-pack", out_a)
    generate_pack(probe, "fancy-pack", out_b)
    files_a = sorted(p.relative_to(out_a) for p in out_a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(out_b) for p in out_b.rglob("*") if p.is_file())
    assert files_a == files_b
    for relative in files_a:
        assert (out_a / relative).read_bytes() == (out_b / relative).read_bytes()


def test_node_selection_and_refusals(port_probe_report: dict[str, object], tmp_path: Path) -> None:
    probe = cast("dict[str, object]", port_probe_report)
    result = generate_pack(probe, "fancy-pack", tmp_path / "sel", only=["Adder"])
    assert [n.source_name for n in result.ported] == ["Adder"]
    module = load_pack_module(tmp_path / "sel", "fancy_pack_nodes.py")
    assert len(cast("list[type]", module.NODES)) == 1
    # Only Adder's atoms get registered: no opaque types at all.
    assert result.opaque_types == ()

    with pytest.raises(PortError, match="unknown source node name"):
        generate_pack(probe, "fancy-pack", tmp_path / "x", only=["Nope"])
    with pytest.raises(PortError, match="skipped by the translator"):
        generate_pack(probe, "fancy-pack", tmp_path / "y", only=["Broken"])


def test_unloadable_pack_refuses(tmp_path: Path) -> None:
    pack = write_pack(tmp_path, "broken_pack", "raise RuntimeError('boom')\n")
    probe = probe_pack(pack)
    with pytest.raises(PortError, match="did not load"):
        generate_pack(probe, "broken-pack", tmp_path / "out")


# --- doctor ----------------------------------------------------------------


def test_generated_scalar_pack_is_doctor_clean_with_no_warnings(
    tmp_path: Path,
) -> None:
    pack = write_pack(tmp_path, "scalar_pack", SCALAR_PACK)
    out = tmp_path / "out"
    generate_pack(probe_pack(pack), "shout-port", out)
    report = diagnose(out / "dinkster-pack.toml")
    assert report.ok, render_text(report)
    assert report.findings == (), render_text(report)
    assert report.node_types == ("shout-port.shouter",)


def test_generated_opaque_pack_is_doctor_clean_with_honest_warnings(
    port_probe_report: dict[str, object], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    generate_pack(cast("dict[str, object]", port_probe_report), "fancy-pack", out)
    report = diagnose(out / "dinkster-pack.toml")
    assert report.ok, render_text(report)
    # The only findings are the honest ones: opaque comfy.* types still
    # need declared codecs - exactly what the README checklist says.
    assert {f.code for f in report.findings} == {"types.fallback-codec"}


def test_generated_v3_pack_is_doctor_clean(tmp_path: Path) -> None:
    """A pure V3 probe generates a native skeleton that imports, keeps the
    input family, and passes doctor with no findings (all-core types)."""
    probe = probe_pack(write_pack(tmp_path, "v3_pack", V3_PACK))
    out = tmp_path / "out"
    result = generate_pack(probe, "v3-port", out)
    assert {(n.source_name, n.source_api) for n in result.ported} == {
        ("Scaler", "v3"),
        ("Stacker", "v3"),
    }
    report = diagnose(out / "dinkster-pack.toml")
    assert report.ok, render_text(report)
    assert report.findings == (), render_text(report)
    assert set(report.node_types) == {"v3-port.scaler", "v3-port.stacker"}

    module = load_pack_module(out, "v3_port_nodes.py")
    schemas = {
        cls.define_schema().node_type: cls.define_schema()
        for cls in cast("list[type[Node]]", module.NODES)
    }
    stacker = schemas["v3-port.stacker"]
    (family,) = stacker.input_families
    assert family.id == "xs"
    assert family.min_members == 1
    assert family.max_members == 5
    # The skipped V3 node made the README, not the module.
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "BrokenSchema" in readme
    assert "RuntimeError: bad schema" in readme


def test_generated_mixed_pack_ports_both_apis(tmp_path: Path) -> None:
    probe = probe_pack(write_pack(tmp_path, "mixed_pack", MIXED_PACK))
    out = tmp_path / "out"
    result = generate_pack(probe, "mixed-port", out)
    assert {(n.source_name, n.source_api) for n in result.ported} == {
        ("Adder", "v1"),
        ("Scaler", "v3"),
    }
    report = diagnose(out / "dinkster-pack.toml")
    assert report.ok, render_text(report)
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "| `mixed-port.adder` | `Adder` | v1 |" in readme
    assert "| `mixed-port.scaler` | `Scaler` | v3 |" in readme


# --- CLI validation ---------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("DINKSTER_COMFYUI_ROOT", ""),
    reason="DINKSTER_COMFYUI_ROOT not set (live ComfyUI test)",
)
def test_cli_ports_a_pack_through_the_real_probe_subprocess(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The full CLI path: probe subprocess under the ComfyUI install's
    interpreter, JSON transport, generation, doctor-clean output."""
    pack = write_pack(tmp_path, "live_cli_pack", SCALAR_PACK)
    out_dir = tmp_path / "ported"

    assert main([str(pack), "--name", "live-port", "--out", str(out_dir)]) == 0
    stdout = capsys.readouterr().out
    assert "live-port.shouter" in stdout

    report = diagnose(out_dir / "dinkster-pack.toml")
    render_text(report)
    assert not report.findings


def test_cli_refuses_bad_names_and_dirty_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = write_pack(tmp_path, "cli_pack", SCALAR_PACK)

    assert main([str(pack), "--name", "Bad_Name"]) == 1
    assert "lowercase" in capsys.readouterr().err

    assert main([str(pack), "--name", "comfy.mine"]) == 1
    assert "reserved root" in capsys.readouterr().err

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("x", encoding="utf-8")
    assert main([str(pack), "--name", "ok-pack", "--out", str(occupied)]) == 1
    assert "--force" in capsys.readouterr().err

    assert main([str(tmp_path / "missing"), "--name", "ok-pack"]) == 1
    assert "does not exist" in capsys.readouterr().err
