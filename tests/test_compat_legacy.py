"""Legacy custom-pack quarantine (DESIGN 3.8): unmodified ComfyUI packs
load in the compat child, translate through the same v1 rules as core,
and every failure is a classified diagnostic, never a mystery.

These tests need no ComfyUI: packs are synthesized on disk and loaded with
the same import mechanics the real loader uses. The live end-to-end run
against a real install is in test_compat_live.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from dinkster_compat_comfy import (
    CompatError,
    CompatTranslation,
    load_legacy_pack,
    translate_mappings,
)
from dinkster_compat_comfy.legacy import _import_pack
from dinkster_schema import TypeExpr
from dinkster_values import CORE_INT

# --- pack scaffolding ----------------------------------------------------


def write_pack(root: Path, name: str, init_source: str) -> Path:
    pack = root / name
    pack.mkdir()
    (pack / "__init__.py").write_text(init_source, encoding="utf-8")
    return pack


GOOD_PACK = '''
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


class Batcher:
    """Uses the list-batch calling convention: translates as list<T> sockets."""

    CATEGORY = "math"
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"xs": ("INT",)}}

    def run(self, xs):
        return (sum(xs),)


class NoFunction:
    """FUNCTION names a method that does not exist: untranslatable."""

    RETURN_TYPES = ("INT",)
    FUNCTION = "missing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT",)}}


NODE_CLASS_MAPPINGS = {"Adder": Adder, "Batcher": Batcher, "NoFunction": NoFunction}
NODE_DISPLAY_NAME_MAPPINGS = {"Adder": "Add Two Ints"}
WEB_DIRECTORY = "./web"
'''


# --- translate-layer rules the loader depends on -------------------------


class V1ListInput:
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"x": ("INT",)}}

    def run(self, x):  # noqa: ANN001, ANN201
        return (x,)


class V1ListOutput:
    RETURN_TYPES = ("INT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"x": ("INT",)}}

    def run(self, x):  # noqa: ANN001, ANN201
        return ([x],)


class V1Plain:
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"x": ("INT", {"default": 0})}}

    def run(self, x):  # noqa: ANN001, ANN201
        return (x,)


def test_list_conventions_translate_to_list_sockets() -> None:
    translation = translate_mappings({"In": V1ListInput, "Out": V1ListOutput}, only=["In", "Out"])
    schemas = {
        cls.define_schema().node_type: cls.define_schema() for cls in translation.node_classes
    }
    int_list = TypeExpr.list_of(TypeExpr.concrete(CORE_INT))
    # INPUT_IS_LIST is class-wide: every input wraps; outputs stay scalar.
    assert schemas["comfy.In"].inputs[0].type == int_list
    assert schemas["comfy.In"].outputs[0].type == TypeExpr.concrete(CORE_INT)
    # OUTPUT_IS_LIST is per-output: only the flagged position wraps.
    assert schemas["comfy.Out"].inputs[0].type == TypeExpr.concrete(CORE_INT)
    assert schemas["comfy.Out"].outputs[0].type == int_list


class V1NoSuchFunction:
    RETURN_TYPES = ("INT",)
    FUNCTION = "missing"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"x": ("INT",)}}


def test_untranslatable_nodes_skip_with_reason_without_only() -> None:
    translation = translate_mappings({"Bad": V1NoSuchFunction, "Good": V1Plain})
    assert len(translation.node_classes) == 1
    assert "Bad" in translation.skipped
    assert "FUNCTION" in translation.skipped["Bad"]


class V1DuplicateInput:
    """Same name in required and optional: v1 tolerates it, so must we.

    Real example: Comfyroll's "CR ControlNet Input Switch"."""

    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {
            "required": {"x": ("INT", {"default": 7})},
            "optional": {"x": ("STRING", {"default": "shadowed"})},
        }

    def run(self, x):  # noqa: ANN001, ANN201
        return (x,)


class V1SchemaCrash:
    """INPUT_TYPES() raising arbitrary exceptions (version skew, plain bugs)
    must skip the node, not kill the pack sweep.

    Real example: KJNodes' ideogram node referencing io.BoundingBox that its
    installed ComfyUI does not have (AttributeError, not CompatError)."""

    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        raise AttributeError("module has no attribute 'BoundingBox'")

    def run(self, x):  # noqa: ANN001, ANN201
        return (x,)


class _AnyType(str):
    """The ecosystem's wildcard hack (AnyType/AlwaysEqualProxy): a str
    subclass that compares equal to everything and - because it defines
    __eq__ without __hash__ - is unhashable."""

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = None  # type: ignore[assignment]


class V1WildcardProxy:
    """Wildcard-proxy inputs/outputs must translate, not crash on hashing.

    Real examples: ComfyUI_LayerStyle's AnyType (24 nodes), Easy-Use's
    AlwaysEqualProxy (7 nodes)."""

    RETURN_TYPES = (_AnyType("*"),)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"anything": (_AnyType("*"),)}}

    def run(self, anything):  # noqa: ANN001, ANN201
        return (anything,)


def test_wildcard_str_subclass_translates_as_wildcard() -> None:
    translation = translate_mappings({"Any": V1WildcardProxy})
    assert not translation.skipped
    (cls,) = translation.node_classes
    schema = cls.define_schema()
    assert schema.inputs[0].type.kind == "wildcard"
    assert schema.outputs[0].type.kind == "wildcard"


class V1SloppyReturnNames:
    """Duplicate names and wrong arity in RETURN_NAMES: v1 treats them as
    positional display labels, so packs ship this and never notice.

    Real examples: Comfyroll's "CR Aspect Ratio SDXL" (duplicates), KJNodes'
    StartRecordCUDAMemoryHistory (2 names for 1 type)."""

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "width")  # duplicate, and one name short
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {}}

    def run(self):  # noqa: ANN201
        return (1, 2, 3.0)


def test_sloppy_return_names_get_tolerant_positional_ids() -> None:
    translation = translate_mappings({"Sloppy": V1SloppyReturnNames})
    assert not translation.skipped
    (cls,) = translation.node_classes
    ids = [spec.id for spec in cls.define_schema().outputs]
    assert ids == ["width", "width_2", "float"]


def test_duplicate_input_ids_first_declaration_wins() -> None:
    translation = translate_mappings({"Dup": V1DuplicateInput})
    assert not translation.skipped
    (cls,) = translation.node_classes
    schema = cls.define_schema()
    (spec,) = schema.inputs
    assert spec.id == "x"
    assert spec.default == 7


def test_arbitrary_schema_exceptions_skip_without_only_raise_with_only() -> None:
    translation = translate_mappings({"Crash": V1SchemaCrash, "Good": V1Plain})
    assert len(translation.node_classes) == 1
    assert translation.skipped["Crash"].startswith("AttributeError: ")
    with pytest.raises(AttributeError):
        translate_mappings({"Crash": V1SchemaCrash}, only=["Crash"])


def test_namespace_scopes_node_ids_and_accumulates() -> None:
    translation = CompatTranslation()
    translate_mappings({"Same": V1Plain}, namespace="pack_a", translation=translation)
    translate_mappings({"Same": V1Plain}, namespace="pack_b", translation=translation)
    ids = {cls.define_schema().node_type for cls in translation.node_classes}
    assert ids == {"comfy.pack_a.Same", "comfy.pack_b.Same"}


# --- pack loading --------------------------------------------------------


def test_good_pack_loads_and_reports_skips(tmp_path: Path) -> None:
    pack = write_pack(tmp_path, "good_pack", GOOD_PACK)
    translation = CompatTranslation()
    report = load_legacy_pack(pack, translation, server_instance=None)
    assert report.status == "loaded"
    assert report.pack_id == "good_pack"
    assert report.nodes_translated == 2
    assert "NoFunction" in report.nodes_skipped
    assert report.web_directory == "./web"
    schemas = {
        cls.define_schema().node_type: cls.define_schema() for cls in translation.node_classes
    }
    adder = schemas["comfy.good_pack.Adder"]
    assert adder.display_name == "Add Two Ints"
    # The list-batch node loads with an honest list<core.int> socket.
    batcher = schemas["comfy.good_pack.Batcher"]
    assert batcher.inputs[0].type == TypeExpr.list_of(TypeExpr.concrete(CORE_INT))


def test_missing_dependency_is_classified(tmp_path: Path) -> None:
    pack = write_pack(
        tmp_path, "needs_dep", "import definitely_not_installed_xyz\nNODE_CLASS_MAPPINGS = {}\n"
    )
    report = load_legacy_pack(pack, CompatTranslation(), server_instance=None)
    assert report.status == "missing-dependency"
    assert report.missing_module == "definitely_not_installed_xyz"


def test_import_error_is_classified(tmp_path: Path) -> None:
    pack = write_pack(tmp_path, "broken", "raise RuntimeError('boom at import')\n")
    report = load_legacy_pack(pack, CompatTranslation(), server_instance=None)
    assert report.status == "import-error"
    assert "boom at import" in report.error


def test_no_mappings_and_v3_are_distinguished(tmp_path: Path) -> None:
    empty = write_pack(tmp_path, "empty_pack", "x = 1\n")
    v3 = write_pack(tmp_path, "v3_pack", "def comfy_entrypoint():\n    return None\n")
    empty_report = load_legacy_pack(empty, CompatTranslation(), server_instance=None)
    v3_report = load_legacy_pack(v3, CompatTranslation(), server_instance=None)
    assert empty_report.status == "no-mappings"
    assert v3_report.status == "v3-entrypoint"
    assert not v3_report.v3_entrypoint_ignored


def test_mixed_pack_loads_v1_and_flags_ignored_v3(tmp_path: Path) -> None:
    """v1 mappings plus a V3 entrypoint: the runtime loads the v1 half
    (ComfyUI's own precedence) but says so instead of staying silent."""
    mixed = write_pack(
        tmp_path,
        "mixed_pack",
        GOOD_PACK + "\ndef comfy_entrypoint():\n    return None\n",
    )
    report = load_legacy_pack(mixed, CompatTranslation(), server_instance=None)
    assert report.status == "loaded"
    assert report.v3_entrypoint_ignored
    assert report.nodes_translated > 0
    # A plain v1 pack never trips the flag.
    plain = write_pack(tmp_path, "plain_pack", GOOD_PACK)
    plain_report = load_legacy_pack(plain, CompatTranslation(), server_instance=None)
    assert plain_report.status == "loaded"
    assert not plain_report.v3_entrypoint_ignored


def test_pack_relative_imports_resolve(tmp_path: Path) -> None:
    """ComfyUI packs are packages: `from .py.thing import X` must work."""
    pack = tmp_path / "relative_pack"
    (pack / "py").mkdir(parents=True)
    (pack / "py" / "__init__.py").write_text("", encoding="utf-8")
    (pack / "py" / "node.py").write_text(
        "class Echo:\n"
        "    RETURN_TYPES = ('STRING',)\n"
        "    FUNCTION = 'run'\n"
        "    @classmethod\n"
        "    def INPUT_TYPES(cls):\n"
        "        return {'required': {'text': ('STRING', {'default': ''})}}\n"
        "    def run(self, text):\n"
        "        return (text,)\n",
        encoding="utf-8",
    )
    (pack / "__init__.py").write_text(
        "from .py.node import Echo\nNODE_CLASS_MAPPINGS = {'Echo': Echo}\n",
        encoding="utf-8",
    )
    report = load_legacy_pack(pack, CompatTranslation(), server_instance=None)
    assert report.status == "loaded"
    assert report.nodes_translated == 1


def test_single_file_pack_loads(tmp_path: Path) -> None:
    single = tmp_path / "solo_node.py"
    single.write_text(
        "class One:\n"
        "    RETURN_TYPES = ('INT',)\n"
        "    FUNCTION = 'run'\n"
        "    @classmethod\n"
        "    def INPUT_TYPES(cls):\n"
        "        return {'required': {}}\n"
        "    def run(self):\n"
        "        return (1,)\n"
        "NODE_CLASS_MAPPINGS = {'One': One}\n",
        encoding="utf-8",
    )
    translation = CompatTranslation()
    report = load_legacy_pack(single, translation, server_instance=None)
    assert report.status == "loaded"
    assert report.pack_id == "solo_node"
    assert translation.node_classes[0].define_schema().node_type == "comfy.solo_node.One"


def test_pack_without_init_is_rejected(tmp_path: Path) -> None:
    bare = tmp_path / "not_a_pack"
    bare.mkdir()
    with pytest.raises(CompatError, match="__init__.py"):
        _import_pack(bare)
