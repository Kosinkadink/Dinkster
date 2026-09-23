"""Translated filesystem-list determinism. Identity-proven listings from
allowlisted model categories become digest-backed AssetWidget inputs;
unrecognized categories remain remote ComboWidgets, while true enums,
copies, and postprocessed lists stay frozen.

Translation stays pure - the probe is injected, so these tests need no
ComfyUI. The probe implementation itself (filename_listing_probe) is
exercised against a fake ``folder_paths`` module in sys.modules.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetError, AssetRef, MountDef, MountTable, digest_bytes
from dinkster_compat_comfy import CompatTranslation, bootstrap, translate_mappings
from dinkster_compat_comfy.native import NATIVE_NODES
from dinkster_compat_comfy.translate import (
    MODEL_FILE_CATEGORIES,
    MODEL_FILE_SELECTORS,
    CompatError,
    InputTypesProbe,
    ListingObservation,
    ModelFileSelector,
    listing_choice_id,
    translate_node,
)
from dinkster_engine import Engine, EventListener
from dinkster_nodes_generation import GENERATION_NODES
from dinkster_schema import AssetWidget, ComboWidget, InputSpec, TypeExpr
from dinkster_server import create_app
from dinkster_values import TypeRegistry
from dinkster_workers import ManifestError
from dinkster_workers.host import load_choices
from dinkster_workers.manifest import load_manifest

from dinkster.compose import ServingComposer
from tests.platform_support import symlink_or_skip

# --- helpers ------------------------------------------------------------


def _combo_node(options: list[str], *, input_is_list: bool = False) -> type:
    """A v1 node whose one combo input holds ``options`` - the object
    itself, verbatim, so observation identity can bind to it."""

    class V1Combo:
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206 - v1 shape, deliberately untyped
            return {"required": {"ckpt_name": (options,)}}

        def load(self, ckpt_name):  # noqa: ANN001, ANN201
            return (ckpt_name,)

    if input_is_list:
        V1Combo.INPUT_IS_LIST = True  # type: ignore[attr-defined]
    return V1Combo


def _selector_node(selector: ModelFileSelector, options: list[str]) -> type:
    input_id = selector.input_id

    class V1Selector:
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206 - v1 shape, deliberately untyped
            return {"required": {input_id: (options,)}}

        def load(self, **kwargs):  # noqa: ANN003, ANN201
            return (kwargs[input_id],)

    V1Selector.__module__ = selector.module
    V1Selector.__name__ = selector.class_name
    return V1Selector


def _fake_comfy_root(tmp_path: Path, name: str = "ComfyUI") -> Path:
    root = tmp_path / name
    (root / "comfy_extras").mkdir(parents=True)
    (root / "folder_paths.py").write_text("# marker\n")
    (root / "nodes.py").write_text("# marker\n")
    return root


def _selector_source(root: Path, selector: ModelFileSelector) -> Path:
    source = root.joinpath(*selector.module.split(".")).with_suffix(".py")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# exact built-in selector source\n")
    return source


def _register_selector_class(
    monkeypatch: pytest.MonkeyPatch,
    node_class: type,
    source: Path,
) -> None:
    module = ModuleType(node_class.__module__)
    module.__file__ = str(source)
    setattr(module, node_class.__name__, node_class)
    monkeypatch.setitem(sys.modules, node_class.__module__, module)


_CHECKPOINT_SELECTOR = MODEL_FILE_SELECTORS[0]


def _listing(category: str, source: list[str]) -> ListingObservation:
    return ListingObservation(category, tuple(source), source)


def _probe_with(*observations: ListingObservation) -> InputTypesProbe:
    """A probe reporting fixed observations for every INPUT_TYPES() call."""

    def probe(invoke):  # noqa: ANN001, ANN202
        return invoke(), observations

    return probe


def _only_widget(node: type) -> ComboWidget | None:
    (spec,) = node.schema().inputs
    widget = spec.widget
    assert widget is None or isinstance(widget, ComboWidget)
    return widget


def _only_asset_widget(node: type) -> AssetWidget:
    (spec,) = node.schema().inputs
    assert isinstance(spec.widget, AssetWidget)
    return spec.widget


# --- choice-id grammar ----------------------------------------------------


def test_listing_choice_id_grammar() -> None:
    assert listing_choice_id("checkpoints") == "comfy.files.checkpoints"
    assert listing_choice_id("text_encoders") == "comfy.files.text_encoders"
    # Ungrammatical categories (uppercase, spaces) get no provider id:
    # their combos stay frozen instead of minting an unservable route.
    assert listing_choice_id("Checkpoints") is None
    assert listing_choice_id("my models") is None
    assert listing_choice_id("") is None


# --- translation-level behavior -------------------------------------------


def test_exact_model_selector_becomes_digest_backed_asset() -> None:
    listed = ["a.safetensors", "b.safetensors"]
    translation = CompatTranslation()
    node = translate_node(
        "LoadCkpt",
        _selector_node(_CHECKPOINT_SELECTOR, listed),
        translation,
        probe=_probe_with(_listing("checkpoints", listed)),
    )
    (spec,) = node.schema().inputs
    assert spec.type == TypeExpr.concrete("dinkster.asset")
    assert spec.required and spec.default is None
    assert _only_asset_widget(node) == AssetWidget(
        accept=("application/octet-stream",), kind="model/checkpoint"
    )
    # Asset pickers browse the digest catalog, not a startup filename list.
    assert translation.listing_snapshots == {}
    registry = TypeRegistry()
    translation.register_types(registry)
    assert "dinkster.asset" in registry


def test_custom_node_using_model_category_remains_remote_combo() -> None:
    listed = ["a.safetensors", "b.safetensors"]
    translation = CompatTranslation()
    node = translate_node(
        "LoadCustom",
        _combo_node(listed),
        translation,
        probe=_probe_with(_listing("checkpoints", listed)),
    )
    widget = _only_widget(node)
    assert widget == ComboWidget(
        options=("a.safetensors", "b.safetensors"),
        remote_route="/api/choices/comfy.files.checkpoints",
        refresh_button=True,
    )
    registry = TypeRegistry()
    translation.register_types(registry)
    assert "dinkster.asset" not in registry


def test_unrecognized_listing_category_keeps_remote_combo() -> None:
    listed = ["a.bin", "b.bin"]
    translation = CompatTranslation()
    node = translate_node(
        "LoadCustom",
        _combo_node(listed),
        translation,
        probe=_probe_with(_listing("custom_models", listed)),
    )
    widget = _only_widget(node)
    assert widget is not None
    assert widget.options == ("a.bin", "b.bin")
    assert widget.remote_route == "/api/choices/comfy.files.custom_models"
    assert widget.refresh_button
    assert translation.listing_snapshots == {"comfy.files.custom_models": ("a.bin", "b.bin")}
    registry = TypeRegistry()
    translation.register_types(registry)
    assert "dinkster.asset" not in registry


def test_true_enum_stays_frozen_under_probe() -> None:
    translation = CompatTranslation()
    node = translate_node(
        "Blend",
        _combo_node(["normal", "multiply", "screen"]),
        translation,
        probe=_probe_with(_listing("checkpoints", ["a.safetensors"])),
    )
    widget = _only_widget(node)
    assert widget is not None
    assert widget.options == ("normal", "multiply", "screen")
    assert not widget.remote_route and not widget.refresh_button
    assert translation.listing_snapshots == {}


def test_equal_values_without_identity_stay_frozen() -> None:
    # A static enum (or a copied/no-op-sorted listing) that merely EQUALS
    # a recorded listing is not the listing: identity is the provenance
    # test, so it stays frozen.
    listed = ["a.safetensors", "b.safetensors"]
    lookalike = sorted(listed)  # new object, same values
    assert lookalike == listed
    translation = CompatTranslation()
    node = translate_node(
        "LoadCopy",
        _combo_node(lookalike),
        translation,
        probe=_probe_with(_listing("checkpoints", listed)),
    )
    widget = _only_widget(node)
    assert widget is not None
    assert not widget.remote_route
    assert translation.listing_snapshots == {}


def test_mutated_listing_stays_frozen() -> None:
    # In-place postprocessing (the ``insert(0, "None")`` shape): the
    # object IS the listing but its values no longer are, so no route.
    listed = ["a.safetensors"]
    observation = _listing("loras", listed)
    listed.insert(0, "None")
    translation = CompatTranslation()
    node = translate_node(
        "LoadOptional",
        _combo_node(listed),
        translation,
        probe=_probe_with(observation),
    )
    widget = _only_widget(node)
    assert widget is not None
    assert widget.options == ("None", "a.safetensors")
    assert not widget.remote_route
    assert translation.listing_snapshots == {}


def test_listing_shared_by_two_supported_categories_refuses() -> None:
    # A custom cache returning the SAME object for two model categories:
    # there is no category-safe asset-to-name conversion, so translation
    # refuses instead of guessing either root.
    listed = ["x.safetensors"]
    translation = CompatTranslation()
    with pytest.raises(CompatError, match="multiple categories"):
        translate_node(
            "LoadEither",
            _selector_node(_CHECKPOINT_SELECTOR, listed),
            translation,
            probe=_probe_with(_listing("checkpoints", listed), _listing("loras", listed)),
        )
    assert translation.listing_snapshots == {}
    registry = TypeRegistry()
    translation.register_types(registry)
    assert "dinkster.asset" not in registry


def test_identity_disambiguates_equal_value_categories() -> None:
    # Two categories with identical VALUES but distinct objects: identity
    # names the one this combo actually holds.
    checkpoints = ["x.safetensors"]
    loras = ["x.safetensors"]
    selector = next(selector for selector in MODEL_FILE_SELECTORS if selector.category == "loras")
    translation = CompatTranslation()
    node = translate_node(
        "LoadCkpt",
        _selector_node(selector, loras),
        translation,
        probe=_probe_with(_listing("checkpoints", checkpoints), _listing("loras", loras)),
    )
    assert _only_asset_widget(node) == AssetWidget(
        accept=("application/octet-stream",), kind="model/lora"
    )
    assert translation.listing_snapshots == {}


def test_model_listing_that_spells_a_boolean_pair_stays_an_asset() -> None:
    # A verbatim listing is never a disguised boolean, even when its two
    # values spell a recognized truthy/falsy pair.
    listed = ["enable", "disable"]
    translation = CompatTranslation()
    node = translate_node(
        "LoadWeird",
        _selector_node(_CHECKPOINT_SELECTOR, listed),
        translation,
        probe=_probe_with(_listing("checkpoints", listed)),
    )
    assert _only_asset_widget(node) == AssetWidget(
        accept=("application/octet-stream",), kind="model/checkpoint"
    )


def test_exact_translated_model_selector_inventory_is_closed() -> None:
    assert len(MODEL_FILE_SELECTORS) == 38
    assert len(set(MODEL_FILE_SELECTORS)) == 38
    payload = json.dumps(
        [list(selector) for selector in MODEL_FILE_SELECTORS],
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == (
        "3f7f84364b0a502be947c79639d5492f66677eaee614e29bcf53ca15abdf2533"
    )
    for selector in MODEL_FILE_SELECTORS:
        assert selector.category in MODEL_FILE_CATEGORIES
        if selector.provenance == "stable-sorted-source":
            continue
        listed = ["model.safetensors"]
        translation = CompatTranslation()
        node = translate_node(
            selector.class_name,
            _selector_node(selector, listed),
            translation,
            probe=_probe_with(_listing(selector.category, listed)),
        )
        assert _only_asset_widget(node) == AssetWidget(
            accept=("application/octet-stream",),
            kind=MODEL_FILE_CATEGORIES[selector.category].kind,
        )


_BACKGROUND_SCHEMA_SOURCE = (
    "@classmethod\n"
    "def define_schema(cls):\n"
    '    files = folder_paths.get_filename_list("background_removal")\n'
    "    return IO.Schema(\n"
    '        node_id="LoadBackgroundRemovalModel",\n'
    '        display_name="Load Background Removal Model",\n'
    '        category="model/loaders",\n'
    "        inputs=[\n"
    '            IO.Combo.Input("bg_removal_name", options=sorted(files), '
    'tooltip="The model used to remove backgrounds from images"),\n'
    "        ],\n"
    "        outputs=[\n"
    '            IO.BackgroundRemoval.Output("bg_model")\n'
    "        ]\n"
    "    )\n"
)


def test_path_loaded_builtin_extra_inventory_and_complete_catalog_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_comfy_root(tmp_path)
    extras = [
        selector for selector in MODEL_FILE_SELECTORS if selector.module.startswith("comfy_extras.")
    ]
    core = [selector for selector in MODEL_FILE_SELECTORS if selector.module == "nodes"]
    assert len(extras) == 29
    assert len(core) == 9
    for selector in extras:
        _selector_source(root, selector)
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(root))
    monkeypatch.setattr(
        "dinkster_compat_comfy.translate.inspect.getsource",
        lambda _method: _BACKGROUND_SCHEMA_SOURCE,
    )

    translated_assets: list[tuple[str, str, str]] = []
    for selector in MODEL_FILE_SELECTORS:
        listed = ["model.safetensors"]
        node_class = _selector_node(selector, listed)
        if selector in extras:
            source = root.joinpath(*selector.module.split(".")).with_suffix(".py")
            node_class.__module__ = str(source.with_suffix(""))
            _register_selector_class(monkeypatch, node_class, source)
        if selector.provenance == "stable-sorted-source":
            node_class.define_schema = classmethod(lambda cls: None)  # type: ignore[attr-defined]
        node = translate_node(
            selector.class_name,
            node_class,
            CompatTranslation(),
            probe=_probe_with(_listing(selector.category, listed)),
        )
        widget = _only_asset_widget(node)
        assert widget == AssetWidget(
            accept=("application/octet-stream",),
            kind=MODEL_FILE_CATEGORIES[selector.category].kind,
        )
        translated_assets.append((selector.class_name, selector.input_id, widget.kind))

    native_assets = [
        (schema.node_type, input_spec.id, input_spec.widget.kind)
        for node_class in (*NATIVE_NODES, *GENERATION_NODES)
        for schema in (node_class.schema(),)
        for input_spec in (
            *schema.inputs,
            *(item for family in schema.input_families for item in family.template),
        )
        if isinstance(input_spec, InputSpec)
        and isinstance(input_spec.widget, AssetWidget)
        and input_spec.widget.kind.startswith("model/")
    ]
    assert len(translated_assets) == 38
    assert ("dinkster.load_controlnet", "control_net_name", "model/controlnet") in native_assets
    assert len(native_assets) == 26
    assert len(translated_assets) + len(native_assets) == 64
    assert (
        "dinkster.load_dual_clip",
        "text_encoder1",
        "model/text-encoder",
    ) in native_assets
    assert (
        "dinkster.load_dual_clip",
        "text_encoder2",
        "model/text-encoder",
    ) in native_assets


def test_path_loaded_builtin_selector_provenance_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = next(
        selector
        for selector in MODEL_FILE_SELECTORS
        if selector.module == "comfy_extras.nodes_audio_encoder"
    )
    listed = ["model.safetensors"]
    root = _fake_comfy_root(tmp_path)
    expected_source = _selector_source(root, selector)
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(root))

    def assert_combo(
        module_name: str,
        *,
        source_name: str | None = None,
        class_name: str = selector.class_name,
        input_id: str = selector.input_id,
        category: str = selector.category,
    ) -> None:
        candidate = ModelFileSelector(
            selector.module,
            class_name,
            input_id,
            selector.category,
        )
        node_class = _selector_node(candidate, listed)
        node_class.__module__ = module_name
        _register_selector_class(
            monkeypatch,
            node_class,
            Path(source_name or f"{module_name}.py"),
        )
        node = translate_node(
            "UntrustedSelector",
            node_class,
            CompatTranslation(),
            probe=_probe_with(_listing(category, listed)),
        )
        (input_spec,) = node.schema().inputs
        assert input_spec.type == TypeExpr.concrete("core.combo")
        assert isinstance(input_spec.widget, ComboWidget)

    third_party = tmp_path / "custom_nodes" / expected_source.name
    third_party.parent.mkdir()
    third_party.write_text("# same basename, third party\n")
    assert_combo(str(third_party.with_suffix("")))

    sibling_root = _fake_comfy_root(tmp_path, "OtherComfyUI")
    sibling_source = _selector_source(sibling_root, selector)
    assert_combo(str(sibling_source.with_suffix("")))

    alias_root = tmp_path / "ComfyUIAlias"
    symlink_or_skip(alias_root, root, target_is_directory=True)
    alias_source = alias_root.joinpath(*selector.module.split(".")).with_suffix(".py")
    assert_combo(str(alias_source.with_suffix("")), source_name=str(alias_source))

    assert_combo(str(expected_source), source_name=str(expected_source))
    assert_combo("comfy_extras/nodes_audio_encoder")
    assert_combo(f"{root}/comfy_extras/../comfy_extras/nodes_audio_encoder")
    assert_combo(str(expected_source.with_suffix("")), class_name="OtherLoader")
    assert_combo(str(expected_source.with_suffix("")), input_id="other_name")
    assert_combo(str(expected_source.with_suffix("")), category="checkpoints")

    other_source = root / "comfy_extras" / "other.py"
    other_source.write_text("# wrong class source\n")
    assert_combo(str(expected_source.with_suffix("")), source_name=str(other_source))

    missing_root = tmp_path / "missing-root"
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(missing_root))
    assert_combo(str(expected_source.with_suffix("")))

    escaped_root = _fake_comfy_root(tmp_path, "SymlinkComfyUI")
    escaped_source = escaped_root.joinpath(*selector.module.split(".")).with_suffix(".py")
    outside_source = tmp_path / "outside.py"
    outside_source.write_text("# outside root\n")
    symlink_or_skip(escaped_source, outside_source)
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(escaped_root))
    assert_combo(str(escaped_source.with_suffix("")))

    missing_file_root = _fake_comfy_root(tmp_path, "MissingFileComfyUI")
    missing_source = missing_file_root.joinpath(*selector.module.split(".")).with_suffix(".py")
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(missing_file_root))
    assert_combo(str(missing_source.with_suffix("")))


def test_path_loaded_selector_uses_comfy_realpath_root_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = next(
        selector
        for selector in MODEL_FILE_SELECTORS
        if selector.module == "comfy_extras.nodes_audio_encoder"
    )
    root = _fake_comfy_root(tmp_path)
    source = _selector_source(root, selector)
    configured_root = tmp_path / "ConfiguredComfyUI"
    symlink_or_skip(configured_root, root, target_is_directory=True)
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(configured_root))
    listed = ["model.safetensors"]
    node_class = _selector_node(selector, listed)
    node_class.__module__ = str(source.with_suffix(""))
    _register_selector_class(monkeypatch, node_class, source)

    node = translate_node(
        selector.class_name,
        node_class,
        CompatTranslation(),
        probe=_probe_with(_listing(selector.category, listed)),
    )
    assert _only_asset_widget(node) == AssetWidget(
        accept=("application/octet-stream",),
        kind="model/audio-encoder",
    )


def test_background_removal_exact_stable_sort_is_the_only_copy_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = next(
        selector
        for selector in MODEL_FILE_SELECTORS
        if selector.provenance == "stable-sorted-source"
    )
    listed = ["z.safetensors", "a.safetensors"]
    node_class = _selector_node(selector, sorted(listed))
    node_class.define_schema = classmethod(lambda cls: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "dinkster_compat_comfy.translate.inspect.getsource",
        lambda _method: _BACKGROUND_SCHEMA_SOURCE,
    )
    translation = CompatTranslation()
    node = translate_node(
        selector.class_name,
        node_class,
        translation,
        probe=_probe_with(_listing(selector.category, listed)),
    )
    assert _only_asset_widget(node) == AssetWidget(
        accept=("application/octet-stream",),
        kind="model/background-removal",
    )

    monkeypatch.setattr(
        "dinkster_compat_comfy.translate.inspect.getsource",
        lambda _method: _BACKGROUND_SCHEMA_SOURCE.replace("sorted(files)", "list(files)"),
    )
    refused = translate_node(
        selector.class_name,
        node_class,
        CompatTranslation(),
        probe=_probe_with(_listing(selector.category, listed)),
    )
    widget = _only_widget(refused)
    assert widget is not None
    assert not widget.remote_route


def test_ungrammatical_category_stays_frozen() -> None:
    listed = ["m.ckpt"]
    translation = CompatTranslation()
    node = translate_node(
        "LoadCustom",
        _combo_node(listed),
        translation,
        probe=_probe_with(_listing("My Models", listed)),
    )
    widget = _only_widget(node)
    assert widget is not None
    assert not widget.remote_route
    assert translation.listing_snapshots == {}


def test_input_is_list_combo_carries_no_widget_or_snapshot() -> None:
    # Under INPUT_IS_LIST the socket is list<core.combo>; a widget on a
    # list socket is undefined in the wire contract, so nothing to
    # remote-ify and no snapshot is recorded.
    listed = ["a.safetensors"]
    translation = CompatTranslation()
    node = translate_node(
        "BatchLoad",
        _combo_node(listed, input_is_list=True),
        translation,
        probe=_probe_with(_listing("checkpoints", listed)),
    )
    (spec,) = node.schema().inputs
    assert spec.widget is None
    assert translation.listing_snapshots == {}


def test_first_committed_snapshot_wins_conflicts_stay_frozen() -> None:
    # Two valid nodes observing DIFFERENT values for one category (the
    # filesystem changed mid-translation): the first committed snapshot
    # is canonical, the later combo stays frozen instead of overwriting
    # the served provider.
    first = ["a.safetensors"]
    second = ["a.safetensors", "b.safetensors"]
    translation = CompatTranslation()
    node_one = translate_node(
        "LoadOne",
        _combo_node(first),
        translation,
        probe=_probe_with(_listing("custom_models", first)),
    )
    node_two = translate_node(
        "LoadTwo",
        _combo_node(second),
        translation,
        probe=_probe_with(_listing("custom_models", second)),
    )
    widget_one = _only_widget(node_one)
    widget_two = _only_widget(node_two)
    assert widget_one is not None and widget_one.remote_route
    assert widget_two is not None and not widget_two.remote_route
    assert translation.listing_snapshots == {"comfy.files.custom_models": ("a.safetensors",)}


def test_failed_node_commits_no_snapshot() -> None:
    # A node that stages a listing snapshot but then fails translation
    # (bad RETURN_TYPES surfaces after the input loop already staged it)
    # must leave no snapshot behind - and a conflicting observation from
    # a failing node cannot clobber a valid node's provider either.
    good = ["a.safetensors"]
    stale = ["a.safetensors", "old.safetensors"]
    loras = ["l.safetensors"]

    class V1Broken:
        RETURN_TYPES = "MODEL"  # not a tuple: raises after input loop
        FUNCTION = "load"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "ckpt_name": (stale,),
                    "lora_name": (loras,),
                }
            }

    translation = translate_mappings(
        {"LoadGood": _combo_node(good), "Broken": V1Broken},
        probe=_probe_with(
            _listing("custom_models", good),
            _listing("custom_models", stale),
            _listing("custom_loras", loras),
        ),
    )
    assert set(translation.skipped) == {"Broken"}
    # Broken staged comfy.files.loras before failing: discarded, never
    # committed; its conflicting checkpoints observation never displaced
    # the valid node's snapshot.
    assert translation.listing_snapshots == {"comfy.files.custom_models": ("a.safetensors",)}


def test_translate_mappings_threads_probe() -> None:
    listed = ["a.safetensors"]
    translation = translate_mappings(
        {"LoadCkpt": _combo_node(listed)},
        probe=_probe_with(_listing("custom_models", listed)),
    )
    assert translation.listing_snapshots == {"comfy.files.custom_models": ("a.safetensors",)}
    (node,) = translation.node_classes
    widget = _only_widget(node)
    assert widget is not None
    assert widget.remote_route == "/api/choices/comfy.files.custom_models"


class _OnePathResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        del digest
        return self.path


def _asset_at(path: Path, payload: bytes = b"model bytes") -> AssetRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return AssetRef(
        digest_bytes(payload),
        path.name,
        len(payload),
        resolver=_OnePathResolver(path),
    )


def _install_category_paths(
    monkeypatch: pytest.MonkeyPatch,
    roots: dict[str, list[Path]],
) -> None:
    module = ModuleType("folder_paths")

    def get_folder_paths(category: str) -> list[str]:
        return [str(root) for root in roots[category]]

    def get_full_path(category: str, relative: str) -> str | None:
        for root in roots[category]:
            candidate = root / relative
            if candidate.is_file():
                return str(candidate)
        return None

    module.get_folder_paths = get_folder_paths  # type: ignore[attr-defined]
    module.get_full_path = get_full_path  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", module)


def test_translated_model_asset_converts_to_exact_v1_relative_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models" / "checkpoints"
    ref = _asset_at(root / "sd15" / "model.safetensors")
    _install_category_paths(monkeypatch, {"checkpoints": [root]})
    listed = ["sd15/model.safetensors"]
    translation = CompatTranslation()
    node = translate_node(
        "LoadCkpt",
        _selector_node(_CHECKPOINT_SELECTOR, listed),
        translation,
        probe=_probe_with(_listing("checkpoints", listed)),
    )

    assert node.execute(ckpt_name=ref) == {"model": "sd15/model.safetensors"}
    registry = TypeRegistry()
    translation.register_types(registry)
    assert "dinkster.asset" in registry


def test_translated_model_asset_resolves_only_from_its_semantic_mount_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root = tmp_path / "models" / "checkpoints"
    lora_root = tmp_path / "models" / "loras"
    payload = b"duplicate model bytes"
    checkpoint = checkpoint_root / "model.safetensors"
    wrong_copy = lora_root / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    wrong_copy.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    wrong_copy.write_bytes(payload)
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot)
    table.add(
        MountDef(id="checkpoint", path=checkpoint_root),
        kind="model/checkpoint",
    )
    table.add(MountDef(id="lora", path=lora_root), kind="model/lora")
    table.scan("checkpoint")
    table.scan("lora")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    _install_category_paths(monkeypatch, {"checkpoints": [checkpoint_root]})

    ref = AssetRef(
        digest=digest_bytes(payload),
        name="model.safetensors",
        size=len(payload),
        resolver=_OnePathResolver(wrong_copy),
    )
    listed = ["model.safetensors"]
    node = translate_node(
        "LoadCkpt",
        _selector_node(_CHECKPOINT_SELECTOR, listed),
        CompatTranslation(),
        probe=_probe_with(_listing("checkpoints", listed)),
    )
    assert node.execute(ckpt_name=ref) == {"model": "model.safetensors"}


def test_translated_model_asset_refuses_wrong_shadowed_and_ambiguous_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_one = tmp_path / "one"
    root_two = tmp_path / "two"
    listed = ["model.safetensors"]
    node = translate_node(
        "LoadCkpt",
        _selector_node(_CHECKPOINT_SELECTOR, listed),
        CompatTranslation(),
        probe=_probe_with(_listing("checkpoints", listed)),
    )

    outside = _asset_at(tmp_path / "loras" / "model.safetensors")
    _install_category_paths(monkeypatch, {"checkpoints": [root_one]})
    with pytest.raises(AssetError, match="not under a configured"):
        node.execute(ckpt_name=outside)

    _asset_at(root_one / "model.safetensors")
    shadowed = _asset_at(root_two / "model.safetensors")
    _install_category_paths(monkeypatch, {"checkpoints": [root_one, root_two]})
    with pytest.raises(AssetError, match="shadowed or unresolved"):
        node.execute(ckpt_name=shadowed)

    nested = root_one / "nested"
    ambiguous = _asset_at(nested / "model.safetensors")
    _install_category_paths(monkeypatch, {"checkpoints": [root_one, nested]})
    with pytest.raises(AssetError, match="ambiguous across"):
        node.execute(ckpt_name=ambiguous)


# --- filename_listing_probe against a fake folder_paths -------------------


def _install_folder_paths(
    monkeypatch: pytest.MonkeyPatch, listings: dict[str, object]
) -> ModuleType:
    module = ModuleType("folder_paths")

    def get_filename_list(category):  # noqa: ANN001, ANN202
        return listings[category]

    module.get_filename_list = get_filename_list  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", module)
    return module


def test_probe_records_listing_and_restores_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listed = ["a.safetensors", "b.safetensors"]
    module = _install_folder_paths(monkeypatch, {"checkpoints": listed})
    original = module.get_filename_list
    probe = bootstrap.filename_listing_probe()
    assert probe is not None

    def input_types() -> dict[str, object]:
        got = module.get_filename_list("checkpoints")
        return {"required": {"ckpt_name": (got,)}}

    result, observed = probe(input_types)
    assert isinstance(result, dict)
    (observation,) = observed
    assert observation.category == "checkpoints"
    assert observation.values == ("a.safetensors", "b.safetensors")
    assert observation.source is listed  # identity: the very object
    assert module.get_filename_list is original


def test_probe_restores_patch_when_input_types_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _install_folder_paths(monkeypatch, {"checkpoints": ["a.safetensors"]})
    original = module.get_filename_list
    probe = bootstrap.filename_listing_probe()
    assert probe is not None

    def raising() -> dict[str, object]:
        module.get_filename_list("checkpoints")
        raise RuntimeError("INPUT_TYPES failed")

    with pytest.raises(RuntimeError, match="INPUT_TYPES failed"):
        probe(raising)
    assert module.get_filename_list is original


def test_probe_passes_exotic_values_through_unrecorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _install_folder_paths(monkeypatch, {"steps": [8, 16], "weird": "not-a-list"})
    probe = bootstrap.filename_listing_probe()
    assert probe is not None

    def input_types() -> tuple[object, object]:
        # A custom get_filename_list override returning non-string-list
        # shapes: values pass through verbatim, nothing is recorded.
        return (
            module.get_filename_list("steps"),
            module.get_filename_list("weird"),
        )

    result, observed = probe(input_types)
    assert result == ([8, 16], "not-a-list")
    assert observed == ()


def test_worker_model_root_snapshot_covers_every_approved_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    calls: list[str] = []

    def get_folder_paths(category: str) -> list[str]:
        calls.append(category)
        category_root = root / category
        return [str(category_root), str(category_root)]

    monkeypatch.setattr(
        bootstrap,
        "initialize_comfy_paths",
        lambda: SimpleNamespace(get_folder_paths=get_folder_paths),
    )
    result = bootstrap.comfy_model_roots()
    assert calls == list(MODEL_FILE_CATEGORIES)
    assert set(result) == set(MODEL_FILE_CATEGORIES)
    for category, descriptor in MODEL_FILE_CATEGORIES.items():
        assert result[category] == {
            "kind": descriptor.kind,
            "roots": (str((root / category).resolve()),),
        }


def test_worker_model_root_snapshot_skips_categories_absent_from_older_comfy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints = tmp_path / "checkpoints"

    def get_folder_paths(category: str) -> list[str]:
        if category != "checkpoints":
            raise KeyError(category)
        return [str(checkpoints)]

    monkeypatch.setattr(
        bootstrap,
        "initialize_comfy_paths",
        lambda: SimpleNamespace(get_folder_paths=get_folder_paths),
    )
    assert bootstrap.comfy_model_roots() == {
        "checkpoints": {
            "kind": "model/checkpoint",
            "roots": (str(checkpoints.resolve()),),
        }
    }


def test_comfy_path_initialization_matches_upstream_order_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_config = tmp_path / "extra_model_paths.yaml"
    default_config.write_text("default: {}\n", "utf-8")
    output = tmp_path / "custom-output"
    input_root = tmp_path / "custom-input"
    user_root = tmp_path / "custom-user"
    loaded_configs: list[str] = []
    added: list[tuple[str, str]] = []
    assigned: dict[str, str] = {}
    folder_paths = SimpleNamespace(
        set_output_directory=lambda path: assigned.__setitem__("output", path),
        get_output_directory=lambda: str(output),
        add_model_folder_path=lambda category, path: added.append((category, path)),
        set_input_directory=lambda path: assigned.__setitem__("input", path),
        set_user_directory=lambda path: assigned.__setitem__("user", path),
    )
    extra_config = SimpleNamespace(load_extra_path_config=lambda path: loaded_configs.append(path))
    cli_args = SimpleNamespace(
        args=SimpleNamespace(
            extra_model_paths_config=(("first.yaml", "second.yaml"), ("third.yaml",)),
            output_directory=str(output),
            input_directory=str(input_root),
            user_directory=str(user_root),
        )
    )
    monkeypatch.setattr(bootstrap, "_comfy_paths_initialized", False)
    monkeypatch.setattr(bootstrap, "_comfy_root_on_path", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "initialize_comfy_args", lambda: cli_args)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: extra_config if name == "utils.extra_config" else folder_paths,
    )

    assert bootstrap.initialize_comfy_paths() is folder_paths
    assert bootstrap.initialize_comfy_paths() is folder_paths
    assert loaded_configs == [
        str(default_config),
        "first.yaml",
        "second.yaml",
        "third.yaml",
    ]
    assert assigned == {
        "output": str(output),
        "input": str(input_root),
        "user": str(user_root),
    }
    assert added == [
        ("checkpoints", str(output / "checkpoints")),
        ("clip", str(output / "clip")),
        ("vae", str(output / "vae")),
        ("diffusion_models", str(output / "diffusion_models")),
        ("loras", str(output / "loras")),
    ]


def test_probe_is_none_without_folder_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "folder_paths", raising=False)
    assert bootstrap.filename_listing_probe() is None
    # Present but without the function: same graceful None.
    monkeypatch.setitem(sys.modules, "folder_paths", ModuleType("folder_paths"))
    assert bootstrap.filename_listing_probe() is None


# --- entry-level exposure ---------------------------------------------------


def test_combo_choices_serves_listing_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # entry has child-only import side effects, so neutralize its bootstrap
    # with a translation that already carries a recorded snapshot.
    def fake_load(*, required: object = ()) -> CompatTranslation:
        del required
        translation = CompatTranslation()
        translation.listing_snapshots["comfy.files.checkpoints"] = (
            "a.safetensors",
            "b.safetensors",
        )
        return translation

    monkeypatch.setattr(bootstrap, "load_comfyui_nodes", fake_load)
    sys.modules.pop("dinkster_compat_comfy.entry", None)
    entry = importlib.import_module("dinkster_compat_comfy.entry")
    real_import = entry.importlib.import_module
    folder_paths = SimpleNamespace(
        get_filename_list=lambda category: {
            "embeddings": ["style.pt"],
            "loras": ["detail.safetensors"],
        }[category]
    )
    monkeypatch.setattr(
        entry.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["normal"]))
            if name == "comfy.samplers"
            else folder_paths
            if name == "folder_paths"
            else real_import(name)
        ),
    )
    assert entry.combo_choices() == {
        "comfy.samplers": ("euler",),
        "comfy.schedulers": ("normal",),
        "comfy.files.checkpoints": ("a.safetensors", "b.safetensors"),
        "comfy.files.embeddings": ["style.pt"],
        "comfy.files.loras": ["detail.safetensors"],
    }


@pytest.mark.parametrize(
    ("listings", "expected"),
    [
        ({"embeddings": [], "loras": []}, {"embeddings": [], "loras": []}),
        (
            {
                "embeddings": ["z.pt", "nested/a.safetensors"],
                "loras": ["second.safetensors", "first.safetensors"],
            },
            {
                "embeddings": ["z.pt", "nested/a.safetensors"],
                "loras": ["second.safetensors", "first.safetensors"],
            },
        ),
    ],
)
def test_prompt_inventory_choices_compose_without_matching_schemas(
    monkeypatch: pytest.MonkeyPatch,
    listings: dict[str, list[str]],
    expected: dict[str, list[str]],
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "load_comfyui_nodes",
        lambda *, required=(): CompatTranslation(),
    )
    sys.modules.pop("dinkster_compat_comfy.entry", None)
    entry = importlib.import_module("dinkster_compat_comfy.entry")
    real_import = entry.importlib.import_module
    monkeypatch.setattr(
        entry.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=(), SCHEDULERS=()))
            if name == "comfy.samplers"
            else SimpleNamespace(get_filename_list=lambda category: listings[category])
            if name == "folder_paths"
            else real_import(name)
        ),
    )

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            composition = composer.composition

            def make_engine(on_event: EventListener) -> Engine:
                return composition.make_engine(on_event)

            choices = {**composition.choices, **entry.combo_choices()}
            app = create_app(make_engine, {}, choices=choices)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                for category in ("embeddings", "loras"):
                    response = await client.get(f"/api/choices/comfy.files.{category}")
                    assert response.status == 200
                    assert response.headers["Cache-Control"] == "no-store"
                    assert await response.json() == expected[category]
            finally:
                await client.close()
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([""], "non-string or empty"),
        (["x" * 4097], "maximum is 4096"),
        ("model.safetensors", "sequence of strings"),
        ({"model.safetensors": True}, "sequence of strings"),
        ({"model.safetensors"}, "sequence of strings"),
    ],
)
def test_prompt_inventory_choices_use_shared_validation(
    monkeypatch: pytest.MonkeyPatch,
    values: object,
    match: str,
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "load_comfyui_nodes",
        lambda *, required=(): CompatTranslation(),
    )
    sys.modules.pop("dinkster_compat_comfy.entry", None)
    entry = importlib.import_module("dinkster_compat_comfy.entry")
    real_import = entry.importlib.import_module
    monkeypatch.setattr(
        entry.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=(), SCHEDULERS=()))
            if name == "comfy.samplers"
            else SimpleNamespace(
                get_filename_list=lambda category: values if category == "embeddings" else ()
            )
            if name == "folder_paths"
            else real_import(name)
        ),
    )
    manifest = load_manifest(Path("packages/dinkster-compat-comfy/dinkster-pack.toml"))
    with pytest.raises(ManifestError, match=match):
        load_choices(manifest)
