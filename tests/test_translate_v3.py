"""ComfyUI V3 schema translation (DESIGN 3.8): a V3 ``Schema`` object in,
an honest Dinkster ``NodeSchema`` out, for ``dinkster port``.

translate_v3.py reads schemas duck-typed - it never imports ComfyUI - so
these tests drive it with plain fakes shaped exactly like comfy_api's
``Schema``/``Input``/``Output`` objects (verified against
ComfyUI/comfy_api/latest/_io.py). The contract under test:

- io_type strings translate by the same rules as v1 type strings
  (primitives, COMBO -> core.combo, ``*`` -> wildcard, opaque comfy.*),
- MultiType comma-joins become unions, MatchType templates become type
  variables, Autogrow inputs become input families,
- lifecycle flags land on the Dinkster fields that mean the same thing,
- unknown structural (COMFY_*_V3) port kinds refuse loudly instead of
  minting meaningless opaque types.
"""

from __future__ import annotations

import dataclasses
from enum import Enum, StrEnum

import pytest
from dinkster_compat_comfy import CompatError, CompatTranslation, translate_v3_schema
from dinkster_compat_comfy.translate_v3 import translate_v3_type
from dinkster_schema import (
    AssetWidget,
    BooleanWidget,
    ColorWidget,
    ComboWidget,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputSpec,
    MultiComboWidget,
    NumberWidget,
    StringWidget,
    TypeExpr,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import CORE_BOOLEAN, CORE_COMBO, CORE_FLOAT, CORE_INT, CORE_STRING

# --- fakes shaped like comfy_api.latest._io ---------------------------------

_MISSING = object()


class FakeInput:
    def __init__(
        self,
        id: str,
        io_type: str,
        *,
        optional: bool = False,
        default: object = None,
        tooltip: str | None = None,
        options: list[object] | None = None,
        multiselect: bool = False,
        template: object = None,
        label_on: str | None = None,
        label_off: str | None = None,
        min: int | float | None = None,
        max: int | float | None = None,
        step: int | float | None = None,
        round: object = None,
        display_mode: object = None,
        multiline: object = _MISSING,
        placeholder: object = None,
        multi_select: object = _MISSING,
        dynamic_prompts: object = None,
        control_after_generate: object = None,
        remote: object = None,
        upload: object = None,
        image_folder: object = None,
    ) -> None:
        self.id = id
        self._io_type = io_type
        self.optional = optional
        self.default = default
        self.tooltip = tooltip
        self.options: object = options
        self.multiselect = multiselect
        self.template = template
        self.label_on = label_on
        self.label_off = label_off
        self.min = min
        self.max = max
        self.step = step
        self.round = round
        self.display_mode = display_mode
        if multiline is not _MISSING:
            self.multiline = multiline
        self.placeholder = placeholder
        self.chip: object = None
        if multi_select is not _MISSING:
            self.multi_select = multi_select
        self.dynamic_prompts = dynamic_prompts
        self.control_after_generate = control_after_generate
        self.remote = remote
        self.upload = upload
        self.image_folder = image_folder
        self.slot: FakeInput | None = None
        self.inputs: list[FakeInput] = []
        self.force_input = False
        self.lazy: bool | None = None
        self.rawLink: bool | None = None

    def get_io_type(self) -> str:
        return self._io_type


class FakeNumberDisplay:
    def __init__(self, value: str) -> None:
        self.value = value


class UploadType(StrEnum):
    image = "image_upload"
    audio = "audio_upload"
    video = "video_upload"
    model = "file_upload"


class FolderType(StrEnum):
    input = "input"
    output = "output"
    temp = "temp"


INCOMPLETE_UPLOAD = Enum("UploadType", {"image": "image_upload"}).image


class FakeRemoteOptions:
    def __init__(
        self,
        route: str,
        refresh_button: object,
        control_after_refresh: object = "first",
        timeout: object = None,
        max_retries: object = None,
        refresh: object = None,
    ) -> None:
        self.route = route
        self.refresh_button = refresh_button
        self.control_after_refresh = control_after_refresh
        self.timeout = timeout
        self.max_retries = max_retries
        self.refresh = refresh


class FakeOutput:
    def __init__(
        self,
        io_type: str,
        *,
        id: str | None = None,
        display_name: str | None = None,
        tooltip: str | None = None,
        is_output_list: bool = False,
        template: object = None,
    ) -> None:
        self.id = id
        self._io_type = io_type
        self.display_name = display_name
        self.tooltip = tooltip
        self.is_output_list = is_output_list
        self.template = template

    def get_io_type(self) -> str:
        return self._io_type


class FakeMatchTemplate:
    def __init__(self, template_id: str, allowed_types: list[object]) -> None:
        self.template_id = template_id
        self.allowed_types = allowed_types


class FakeComfyType:
    def __init__(self, io_type: str) -> None:
        self.io_type = io_type


class FakeAutogrowTemplate:
    def __init__(
        self,
        input: FakeInput,
        *,
        min: int = 0,
        max: int | None = None,
        names: list[str] | None = None,
        prefix: str | None = None,
    ) -> None:
        self.input = input
        self.min = min
        self.max = max
        self.names = (
            [f"{prefix}{index}" for index in range(max)]
            if names is None and prefix is not None and max is not None
            else names
        )
        self.prefix = prefix


class FakeDynamicOption:
    def __init__(self, key: str, inputs: list[FakeInput]) -> None:
        self.key = key
        self.inputs = inputs


class FakeSchema:
    def __init__(
        self,
        node_id: str = "TestNode",
        *,
        display_name: str = "",
        category: str = "",
        description: str = "",
        inputs: list[FakeInput] | None = None,
        outputs: list[FakeOutput] | None = None,
        is_input_list: bool = False,
        is_output_node: bool = False,
        not_idempotent: bool = False,
        is_api_node: bool = False,
        is_deprecated: bool = False,
        is_dev_only: bool = False,
        accept_all_inputs: bool = False,
        enable_expand: bool = False,
    ) -> None:
        self.node_id = node_id
        self.display_name = display_name
        self.category = category
        self.description = description
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.is_input_list = is_input_list
        self.is_output_node = is_output_node
        self.not_idempotent = not_idempotent
        self.is_api_node = is_api_node
        self.is_deprecated = is_deprecated
        self.is_dev_only = is_dev_only
        self.accept_all_inputs = accept_all_inputs
        self.enable_expand = enable_expand


def translate(schema: FakeSchema, translation: CompatTranslation | None = None):
    return translate_v3_schema(schema, translation or CompatTranslation(), namespace="pack")


# --- type translation --------------------------------------------------------


def test_primitive_io_types_translate_to_core_types() -> None:
    translation = CompatTranslation()
    assert translate_v3_type("INT", translation) == TypeExpr.concrete(CORE_INT)
    assert translate_v3_type("FLOAT", translation) == TypeExpr.concrete(CORE_FLOAT)
    assert translate_v3_type("STRING", translation) == TypeExpr.concrete(CORE_STRING)
    assert translate_v3_type("BOOLEAN", translation) == TypeExpr.concrete(CORE_BOOLEAN)
    assert translate_v3_type("COMBO", translation) == TypeExpr.concrete(CORE_COMBO)
    assert translation.opaque_types == set()


def test_v3_upload_type_and_folder_translate_to_source_filename_assets() -> None:
    scalar, audio = translate_v3_schema(
        FakeSchema(
            inputs=[
                FakeInput(
                    "image",
                    "COMBO",
                    options=["existing.png"],
                    upload=UploadType.image,
                    image_folder=FolderType.output,
                ),
                FakeInput(
                    "audio",
                    "COMBO",
                    options=["one.wav", "two.wav"],
                    upload=UploadType.audio,
                ),
            ]
        ),
        CompatTranslation(),
    ).inputs
    assert scalar.type == TypeExpr.concrete("dinkster.asset")
    assert scalar.source_filename is not None
    assert (scalar.source_filename.kind, scalar.source_filename.category) == (
        "media/image",
        "output",
    )
    assert scalar.widget == AssetWidget(
        accept=("image/png", "image/jpeg", "image/webp"),
        kind="media/image",
        allow_upload=True,
    )
    assert audio.type == TypeExpr.concrete("dinkster.asset")
    assert audio.source_filename is not None
    assert (audio.source_filename.kind, audio.source_filename.category) == (
        "media/audio",
        "input",
    )


@pytest.mark.parametrize(
    ("input_obj", "is_input_list", "message"),
    (
        (FakeInput("x", "COMBO", upload="image_upload"), False, "exact UploadType"),
        (FakeInput("x", "COMBO", upload=INCOMPLETE_UPLOAD), False, "exact UploadType"),
        (FakeInput("x", "COMBO", upload=UploadType.model), False, "file_upload"),
        (
            FakeInput("x", "COMBO", upload=UploadType.image, image_folder="output"),
            False,
            "exact FolderType",
        ),
        (FakeInput("x", "COMBO", image_folder=FolderType.input), False, "no UploadType"),
        (FakeInput("x", "COMBO", upload=UploadType.video), True, "is_input_list"),
        (
            FakeInput("x", "COMBO", upload=UploadType.video, multiselect=True),
            False,
            "multiselect",
        ),
        (FakeInput("x", "STRING", upload=UploadType.image), False, "exact COMBO"),
        (
            FakeInput("x", "COMBO", upload=UploadType.image, default="ambient.png"),
            False,
            "ambient default",
        ),
    ),
)
def test_v3_source_filename_declarations_refuse_lookalikes_and_mismatches(
    input_obj: FakeInput,
    is_input_list: bool,
    message: str,
) -> None:
    with pytest.raises(CompatError, match=message):
        translate(FakeSchema(inputs=[input_obj], is_input_list=is_input_list))


def test_top_level_ordinary_ids_keep_wire_14_permissiveness() -> None:
    result = translate(
        FakeSchema(
            "PermissiveIds",
            inputs=[FakeInput("input_blocks.0.", "INT")],
            outputs=[FakeOutput("STRING", id="Audio VAE")],
        )
    )
    assert result.inputs[0].id == "input_blocks.0."
    assert result.outputs[0].id == "Audio VAE"


def test_opaque_io_types_register_comfy_types() -> None:
    translation = CompatTranslation()
    assert translate_v3_type("LATENT", translation) == TypeExpr.concrete("comfy.LATENT")
    assert translation.opaque_types == {"comfy.LATENT"}


def test_wildcard_and_multitype_unions() -> None:
    translation = CompatTranslation()
    assert translate_v3_type("*", translation) == TypeExpr.wildcard()
    # MultiType serializes its io_type as a comma-join.
    assert translate_v3_type("INT,FLOAT", translation) == TypeExpr.union(CORE_INT, CORE_FLOAT)
    # A * member means the whole thing is as permissive as a wildcard.
    assert translate_v3_type("INT,*", translation) == TypeExpr.wildcard()
    # COMBO and STRING keep distinct concrete members.
    assert translate_v3_type("COMBO,STRING", translation) == TypeExpr.union(CORE_COMBO, CORE_STRING)


def test_empty_io_type_refuses() -> None:
    with pytest.raises(CompatError, match="empty V3 io_type"):
        translate_v3_type("", CompatTranslation())


# --- schema shape ------------------------------------------------------------


def test_basic_schema_translates() -> None:
    schema = FakeSchema(
        "Adder",
        display_name="Add Two Ints",
        category="math",
        description="Adds a and b.",
        inputs=[
            FakeInput("a", "INT", default=1, tooltip="first addend"),
            FakeInput("b", "INT"),
            FakeInput("mode", "COMBO", options=["fast", "slow"]),
        ],
        outputs=[FakeOutput("INT", id="total", tooltip="the sum")],
    )
    result = translate(schema)
    assert result.node_type == "comfy.pack.Adder"
    assert result.display_name == "Add Two Ints"
    assert result.category == "comfy/math"
    assert result.description == "Adds a and b."
    a, b, mode = result.inputs
    assert a.type == TypeExpr.concrete(CORE_INT)
    assert a.default == 1
    assert not a.required  # has a default
    assert a.doc == "first addend"
    assert b.required
    # Combo identity is core.combo; the first option is the implicit default,
    # and the choice list rides along as a static dropdown widget.
    assert mode.type == TypeExpr.concrete(CORE_COMBO)
    assert mode.default == "fast"
    assert isinstance(mode.widget, ComboWidget)
    assert mode.widget.options == ("fast", "slow")
    assert a.widget is None and b.widget is None
    (total,) = result.outputs
    assert total.id == "total"
    assert total.doc == "the sum"
    # The V3 node_id stays resolvable for API submissions.
    assert result.aliases == ("Adder",)


def test_numeric_inputs_preserve_v3_display_and_constraints() -> None:
    schema = FakeSchema(
        "Numbers",
        inputs=[
            FakeInput(
                "count",
                "INT",
                default=1,
                min=0,
                max=10,
                step=1,
                display_mode=FakeNumberDisplay("slider"),
            ),
            FakeInput(
                "strength",
                "FLOAT",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.01,
                round=0.001,
                display_mode=FakeNumberDisplay("number"),
            ),
            FakeInput("bounded", "FLOAT", min=-1.0, max=1.0),
            FakeInput("unknown", "INT", display_mode=FakeNumberDisplay("dial")),
        ],
        outputs=[FakeOutput("FLOAT")],
    )
    count, strength, bounded, unknown = translate(schema).inputs
    assert count.widget == NumberWidget(min=0, max=10, step=1, display="slider")
    assert strength.widget == NumberWidget(
        min=0.0, max=1.0, step=0.01, round=0.001, display="number"
    )
    assert bounded.widget == NumberWidget(min=-1.0, max=1.0)
    assert unknown.widget is None


def test_v3_widget_v19_declarations_are_preserved_exactly() -> None:
    class FakeControl(str, Enum):  # noqa: UP042 - matches pinned ComfyUI API
        decrement = "decrement"

    schema = FakeSchema(
        "WidgetFacts",
        inputs=[
            FakeInput(
                "prompt",
                "STRING",
                multiline=True,
                placeholder="Describe an image",
                dynamic_prompts=True,
            ),
            FakeInput("single_line", "STRING", multiline=False),
            FakeInput("plain", "STRING"),
            FakeInput("invalid_mode", "STRING", multiline="yes"),
            FakeInput(
                "mixed",
                "STRING",
                multiline=False,
                placeholder=1,
                dynamic_prompts=False,
            ),
            FakeInput(
                "choice",
                "COMBO",
                options=["a", "b"],
                control_after_generate=FakeControl.decrement,
            ),
            FakeInput("color", "COLOR", default="#abcdef"),
            FakeInput("colors", "COLORS"),
        ],
        outputs=[FakeOutput("STRING")],
    )
    translation = CompatTranslation()
    prompt, single_line, plain, invalid_mode, mixed, choice, color, colors = translate_v3_schema(
        schema, translation
    ).inputs
    assert prompt.widget == StringWidget(
        multiline=True,
        placeholder="Describe an image",
        dynamic_prompts=True,
    )
    assert single_line.widget == StringWidget(multiline=False)
    assert plain.widget is None
    assert invalid_mode.widget is None
    assert mixed.widget == StringWidget(multiline=False, dynamic_prompts=False)
    assert choice.widget == ComboWidget(options=("a", "b"), control_after_generate="decrement")
    assert color.type == TypeExpr.concrete(CORE_STRING)
    assert color.widget == ColorWidget()
    assert colors.type == TypeExpr.concrete("comfy.COLORS")
    assert "comfy.COLOR" not in translation.opaque_types
    assert "comfy.COLORS" in translation.opaque_types

    output_translation = CompatTranslation()
    color_output_schema = FakeSchema("ColorOutput", outputs=[FakeOutput("COLOR")])
    output = translate_v3_schema(color_output_schema, output_translation).outputs[0]
    assert output.type == TypeExpr.concrete(CORE_STRING)
    assert output.type == color.type
    assert "comfy.COLOR" not in output_translation.opaque_types


def test_v3_remote_options_require_trusted_choice_and_preserve_exact_policy() -> None:
    translation = CompatTranslation()
    translation.listing_snapshots["comfy.files.checkpoints"] = ("a.safetensors",)
    schema = FakeSchema(
        "RemoteCombo",
        inputs=[
            FakeInput(
                "model",
                "COMBO",
                options=["a.safetensors"],
                remote=FakeRemoteOptions(
                    "/api/choices/comfy.files.checkpoints",
                    True,
                    "last",
                    4096,
                    2,
                    0,
                ),
            )
        ],
    )
    (model,) = translate_v3_schema(schema, translation).inputs
    assert model.widget == ComboWidget(
        options=("a.safetensors",),
        remote_route="/api/choices/comfy.files.checkpoints",
        refresh_button=True,
        control_after_refresh="last",
        remote_timeout_ms=4096,
        remote_max_retries=2,
        remote_refresh_ms=0,
    )

    for remote, message in (
        (FakeRemoteOptions("https://example.invalid/choices", True), "trusted registered"),
        (FakeRemoteOptions("/api/choices/comfy.files.unknown", True), "trusted registered"),
        (FakeRemoteOptions("/api/choices/comfy.files.checkpoints", 1), "refresh_button"),
        (FakeRemoteOptions("/api/choices/comfy.files.checkpoints", True, timeout=True), "timeout"),
    ):
        bad = FakeSchema(
            "BadRemote",
            inputs=[FakeInput("model", "COMBO", options=["a"], remote=remote)],
        )
        with pytest.raises(CompatError, match=message):
            translate_v3_schema(bad, translation)


def test_boolean_combos_and_labels_translate_to_boolean_widgets() -> None:
    """The v1 disguised-boolean rule applies to V3 combos too, and V3
    Boolean inputs keep their label_on/label_off as a BooleanWidget -
    the porting fact lands in the schema, not the discard pile."""
    schema = FakeSchema(
        "Toggles",
        inputs=[
            FakeInput("add_noise", "COMBO", options=["enable", "disable"]),
            FakeInput("mode", "COMBO", options=["fast", "slow"]),
            FakeInput("mute", "BOOLEAN", default=False, label_on="muted", label_off="audible"),
        ],
        outputs=[FakeOutput("INT")],
    )
    translation = CompatTranslation()
    result = translate_v3_schema(schema, translation)
    add_noise, mode, mute = result.inputs
    assert add_noise.type == TypeExpr.concrete(CORE_BOOLEAN)
    assert add_noise.widget == BooleanWidget(label_on="enable", label_off="disable")
    assert add_noise.default is True  # first option "enable" is truthy
    assert isinstance(mode.widget, ComboWidget)  # not a boolean pair
    assert mute.type == TypeExpr.concrete(CORE_BOOLEAN)
    assert mute.widget == BooleanWidget(label_on="muted", label_off="audible")
    assert mute.default is False


def test_missing_node_id_refuses() -> None:
    with pytest.raises(CompatError, match="lacks a node_id"):
        translate(FakeSchema(""))


def test_missing_input_id_refuses() -> None:
    schema = FakeSchema("N", inputs=[FakeInput("", "INT")])
    with pytest.raises(CompatError, match="input lacks an id"):
        translate(schema)


def test_optional_inputs_are_not_required() -> None:
    schema = FakeSchema("N", inputs=[FakeInput("x", "INT", optional=True)])
    result = translate(schema)
    assert not result.inputs[0].required


def test_output_id_fallbacks_and_dedupe() -> None:
    schema = FakeSchema(
        "N",
        outputs=[
            FakeOutput("INT", display_name="count"),
            FakeOutput("LATENT"),
            FakeOutput("LATENT"),
        ],
    )
    result = translate(schema)
    assert [o.id for o in result.outputs] == ["count", "latent", "latent_2"]


def test_is_input_list_wraps_sockets_and_defaults() -> None:
    schema = FakeSchema(
        "Batcher",
        inputs=[FakeInput("xs", "INT", default=1)],
        outputs=[FakeOutput("INT", id="out")],
        is_input_list=True,
    )
    result = translate(schema)
    assert result.inputs[0].type == TypeExpr.list_of(TypeExpr.concrete(CORE_INT))
    assert result.inputs[0].default == [1]
    # is_input_list is input-side only; outputs stay scalar.
    assert result.outputs[0].type == TypeExpr.concrete(CORE_INT)


def test_is_input_list_admits_only_declared_whole_list_lazy_inputs() -> None:
    lazy = FakeInput("values", "INT")
    lazy.lazy = True
    result = translate(FakeSchema("LazyBatch", inputs=[lazy], is_input_list=True))
    assert result.inputs == (
        InputSpec(
            "values",
            TypeExpr.list_of(TypeExpr.concrete(CORE_INT)),
            lazy=True,
        ),
    )

    scalar_lazy = FakeInput("value", "INT")
    scalar_lazy.lazy = True
    with pytest.raises(CompatError, match="unsupported lazy semantics"):
        translate(FakeSchema("ScalarLazy", inputs=[scalar_lazy]))

    nested_lazy = FakeInput("value", "INT")
    nested_lazy.lazy = True
    dynamic = FakeInput("mode", "COMFY_DYNAMICCOMBO_V3")
    dynamic.options = [FakeDynamicOption("choice", [nested_lazy])]
    with pytest.raises(CompatError, match="unsupported lazy semantics"):
        translate(FakeSchema("DynamicLazy", inputs=[dynamic], is_input_list=True))


def test_is_input_list_admits_lazy_matchtype_as_list_variable() -> None:
    template = FakeMatchTemplate("T", [FakeComfyType("INT"), FakeComfyType("STRING")])
    lazy = FakeInput("values", "COMFY_MATCHTYPE_V3", template=template)
    lazy.lazy = True

    result = translate(FakeSchema("LazyMatchBatch", inputs=[lazy], is_input_list=True))

    assert result.inputs == (
        InputSpec(
            "values",
            TypeExpr.list_of(TypeExpr.variable("T", (CORE_INT, CORE_STRING))),
            lazy=True,
        ),
    )


def test_is_output_list_wraps_only_that_output() -> None:
    schema = FakeSchema(
        "Splitter",
        outputs=[
            FakeOutput("INT", id="parts", is_output_list=True),
            FakeOutput("INT", id="count"),
        ],
    )
    result = translate(schema)
    assert result.outputs[0].type == TypeExpr.list_of(TypeExpr.concrete(CORE_INT))
    assert result.outputs[1].type == TypeExpr.concrete(CORE_INT)


def test_combo_outputs_and_list_modes_keep_combo_identity() -> None:
    scalar = translate(
        FakeSchema(
            "ComboOut",
            inputs=[FakeInput("choice", "COMBO", options=["a", "b"])],
            outputs=[FakeOutput("COMBO", id="choice")],
        )
    )
    assert scalar.inputs[0].type == TypeExpr.concrete(CORE_COMBO)
    assert scalar.outputs[0].type == TypeExpr.concrete(CORE_COMBO)

    listed_input = translate(
        FakeSchema(
            "ComboInputList",
            inputs=[FakeInput("choices", "COMBO", options=["a", "b"])],
            is_input_list=True,
        )
    )
    assert listed_input.inputs[0].type == TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO))
    assert listed_input.inputs[0].widget is None

    listed_output = translate(
        FakeSchema(
            "ComboOutputList",
            outputs=[FakeOutput("COMBO", id="choices", is_output_list=True)],
        )
    )
    assert listed_output.outputs[0].type == TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO))


# --- MatchType / Autogrow ----------------------------------------------------


def test_matchtype_translates_to_type_variable() -> None:
    template = FakeMatchTemplate(
        "T",
        [
            FakeComfyType("INT"),
            FakeComfyType("COMBO"),
            FakeComfyType("STRING"),
        ],
    )
    schema = FakeSchema(
        "Passthrough",
        inputs=[FakeInput("value", "COMFY_MATCHTYPE_V3", template=template)],
        outputs=[FakeOutput("COMFY_MATCHTYPE_V3", id="out", template=template)],
    )
    result = translate(schema)
    expected = TypeExpr.variable("T", (CORE_INT, CORE_COMBO, CORE_STRING))
    assert result.inputs[0].type == expected
    assert result.outputs[0].type == expected


def test_matchtype_anytype_member_drops_the_allow_list() -> None:
    template = FakeMatchTemplate("T", [FakeComfyType("*")])
    schema = FakeSchema(
        "AnyPass", inputs=[FakeInput("value", "COMFY_MATCHTYPE_V3", template=template)]
    )
    result = translate(schema)
    assert result.inputs[0].type == TypeExpr.variable("T")


def test_matchtype_composite_structural_marker_refuses_without_opaque_registration() -> None:
    template = FakeMatchTemplate("T", [FakeComfyType("INT,COMFY_DYNAMICSLOT_V3")])
    translation = CompatTranslation()
    with pytest.raises(CompatError, match="dynamic type marker"):
        translate(
            FakeSchema(
                "BadMatch",
                inputs=[FakeInput("value", "COMFY_MATCHTYPE_V3", template=template)],
            ),
            translation,
        )
    assert translation.opaque_types == set()


def test_matchtype_without_template_refuses() -> None:
    schema = FakeSchema("N", inputs=[FakeInput("value", "COMFY_MATCHTYPE_V3")])
    with pytest.raises(CompatError, match="lacks a template_id"):
        translate(schema)


def test_autogrow_translates_to_input_family() -> None:
    template = FakeAutogrowTemplate(
        FakeInput("image", "IMAGE", tooltip="one frame"),
        min=1,
        max=10,
        prefix="image",
    )
    schema = FakeSchema(
        "Stacker",
        inputs=[FakeInput("images", "COMFY_AUTOGROW_V3", template=template)],
        outputs=[FakeOutput("IMAGE", id="stack")],
    )
    translation = CompatTranslation()
    result = translate(schema, translation)
    assert result.inputs == ()
    (family,) = result.input_families
    assert family.id == "images"
    assert family.type == TypeExpr.concrete("comfy.IMAGE")
    assert family.min_members == 1
    assert family.max_members == 10
    assert family.member_prefix == "image"
    assert family.member_names is None
    assert family.doc == "one frame"
    assert "comfy.IMAGE" in translation.opaque_types

    listed = translate(
        FakeSchema(
            "ListedStacker",
            inputs=[FakeInput("images", "COMFY_AUTOGROW_V3", template=template)],
            is_input_list=True,
        )
    )
    assert listed.input_families[0].type == TypeExpr.list_of(TypeExpr.concrete("comfy.IMAGE"))


def test_autogrow_names_template_caps_members() -> None:
    template = FakeAutogrowTemplate(FakeInput("x", "INT"), min=0, names=["first", "second"])
    schema = FakeSchema("Named", inputs=[FakeInput("xs", "COMFY_AUTOGROW_V3", template=template)])
    result = translate(schema)
    assert result.input_families[0].max_members == 2


def test_autogrow_without_template_refuses() -> None:
    schema = FakeSchema("N", inputs=[FakeInput("xs", "COMFY_AUTOGROW_V3")])
    with pytest.raises(CompatError, match="lacks a template input"):
        translate(schema)


def test_unknown_structural_port_kinds_refuse() -> None:
    # Future COMFY_*_V3 kinds and dynamic outputs must not fall through to
    # the opaque branch as fake type ids.
    schema = FakeSchema("Dyn", inputs=[FakeInput("choice", "COMFY_FUTURE_V3")])
    with pytest.raises(CompatError, match="dynamic input kind"):
        translate(schema)
    schema = FakeSchema("Dyn", outputs=[FakeOutput("COMFY_DYNAMICSLOT_V3", id="slot")])
    with pytest.raises(CompatError, match="dynamic output kind"):
        translate(schema)


def test_dynamic_combo_and_slot_translate_recursively_from_direct_v3_objects() -> None:
    nested = FakeInput("sub", "COMFY_DYNAMICCOMBO_V3")
    nested.options = [FakeDynamicOption("inner", [FakeInput("value", "INT")])]
    combo = FakeInput("mode", "COMFY_DYNAMICCOMBO_V3", default="outer")
    combo.options = [FakeDynamicOption("outer", [nested])]

    slot_socket = FakeInput("source", "IMAGE")
    slot = FakeInput("source", "COMFY_DYNAMICSLOT_V3", optional=True)
    slot.slot = slot_socket
    slot.inputs = [FakeInput("weight", "FLOAT")]
    slot.force_input = True

    result = translate(FakeSchema("Dynamic", inputs=[combo, slot]))
    translated_combo = result.combos[0]
    assert isinstance(translated_combo, DynamicComboSpec)
    assert translated_combo.default == "outer"
    nested_combo = translated_combo.options[0].inputs[0]
    assert isinstance(nested_combo, DynamicComboSpec)
    nested_input = nested_combo.options[0].inputs[0]
    assert isinstance(nested_input, InputSpec)
    assert nested_input.type == TypeExpr.concrete(CORE_INT)

    translated_slot = result.slots[0]
    assert isinstance(translated_slot, DynamicSlotSpec)
    assert translated_slot.slot_type == TypeExpr.concrete("comfy.IMAGE")
    slot_input = translated_slot.inputs[0]
    assert isinstance(slot_input, InputSpec)
    assert slot_input.type == TypeExpr.concrete(CORE_FLOAT)
    assert translated_slot.required is False
    assert translated_slot.force_input is True

    listed = translate(FakeSchema("Listed", inputs=[slot], is_input_list=True))
    assert listed.slots[0].slot_type == TypeExpr.list_of(TypeExpr.concrete("comfy.IMAGE"))
    listed_dependent = listed.slots[0].inputs[0]
    assert isinstance(listed_dependent, InputSpec)
    assert listed_dependent.type == TypeExpr.list_of(TypeExpr.concrete(CORE_FLOAT))


def test_direct_required_zero_option_combo_refuses_but_optional_is_inactive() -> None:
    required = FakeInput("mode", "COMFY_DYNAMICCOMBO_V3")
    required.options = []
    with pytest.raises(CompatError, match="required but has zero options"):
        translate(FakeSchema("Required", inputs=[required]))

    optional = FakeInput("mode", "COMFY_DYNAMICCOMBO_V3", optional=True)
    optional.options = []
    result = translate(FakeSchema("Optional", inputs=[optional]))
    assert result.combos[0].options == ()
    assert result.combos[0].required is False


def test_direct_accept_all_inputs_lazy_and_rawlink_stay_classified() -> None:
    with pytest.raises(CompatError, match="accept_all_inputs"):
        translate(FakeSchema("CatchAll", accept_all_inputs=True))

    lazy = FakeInput("value", "INT")
    lazy.lazy = True
    with pytest.raises(CompatError, match="unsupported lazy semantics"):
        translate(FakeSchema("Lazy", inputs=[lazy]))

    template = FakeMatchTemplate("T", [FakeComfyType("*")])
    switch = FakeInput("switch", "BOOLEAN")
    off = FakeInput("on_false", "COMFY_MATCHTYPE_V3", template=template, optional=True)
    on = FakeInput("on_true", "COMFY_MATCHTYPE_V3", template=template, optional=True)
    off.lazy = on.lazy = True
    switch_schema = FakeSchema(
        "ComfySwitchNode",
        inputs=[switch, off, on],
        outputs=[FakeOutput("COMFY_MATCHTYPE_V3", template=template)],
    )
    translated_switch = translate(switch_schema)
    assert translated_switch.selector is not None
    assert [input_spec.required for input_spec in translated_switch.inputs] == [True, False, False]
    on.lazy = False
    with pytest.raises(CompatError, match="unsupported lazy semantics"):
        translate(switch_schema)

    raw = FakeInput("value", "INT")
    raw.rawLink = True
    with pytest.raises(CompatError, match="unsupported rawLink semantics"):
        translate(FakeSchema("Raw", inputs=[raw]))


def test_direct_custom_combo_exact_shape_becomes_closed_family() -> None:
    schema = FakeSchema(
        "CustomCombo",
        inputs=[FakeInput("choice", "COMBO", options=[])],
        outputs=[
            FakeOutput("STRING", display_name="STRING"),
            FakeOutput("INT", display_name="INDEX"),
        ],
        accept_all_inputs=True,
    )
    result = translate_v3_schema(schema, CompatTranslation())
    assert result.node_type == "comfy.CustomCombo"
    assert [(item.id, item.type, item.default) for item in result.inputs] == [
        ("choice", TypeExpr.concrete(CORE_COMBO), None),
        ("index", TypeExpr.concrete(CORE_INT), 0),
    ]
    assert [item.id for item in result.outputs] == ["STRING", "INDEX"]
    (family,) = result.input_families
    assert family.id == "options"
    assert family.type == TypeExpr.concrete(CORE_STRING)
    assert family.min_members == 0
    assert family.max_members == 100
    assert family.member_names == tuple(f"option{index}" for index in range(1, 101))

    finalized = FakeSchema(
        "CustomCombo",
        inputs=[FakeInput("choice", "COMBO", options=[])],
        outputs=[
            FakeOutput("STRING", id="_0_STRING_", display_name="STRING"),
            FakeOutput("INT", id="_1_INT_", display_name="INDEX"),
        ],
        accept_all_inputs=True,
    )
    finalized_result = translate_v3_schema(finalized, CompatTranslation())
    assert [item.id for item in finalized_result.outputs] == ["STRING", "INDEX"]


@pytest.mark.parametrize(
    ("schema", "namespace"),
    [
        (FakeSchema("CatchAll", accept_all_inputs=True), ""),
        (
            FakeSchema(
                "CustomCombo",
                inputs=[FakeInput("choice", "COMBO", options=["fixed"])],
                outputs=[FakeOutput("STRING"), FakeOutput("INT")],
                accept_all_inputs=True,
            ),
            "",
        ),
        (
            FakeSchema(
                "CustomCombo",
                inputs=[FakeInput("choice", "COMBO", options=[])],
                outputs=[FakeOutput("STRING"), FakeOutput("INT")],
                accept_all_inputs=True,
            ),
            "custom_pack",
        ),
    ],
)
def test_direct_custom_combo_lookalikes_keep_accept_all_refusal(
    schema: FakeSchema, namespace: str
) -> None:
    with pytest.raises(CompatError, match="accept_all_inputs"):
        translate_v3_schema(schema, CompatTranslation(), namespace=namespace)


def test_direct_custom_combo_requires_canonical_boolean_fields() -> None:
    def schema() -> FakeSchema:
        return FakeSchema(
            "CustomCombo",
            inputs=[FakeInput("choice", "COMBO", options=[])],
            outputs=[
                FakeOutput("STRING", display_name="STRING"),
                FakeOutput("INT", display_name="INDEX"),
            ],
            accept_all_inputs=True,
        )

    missing_multiselect = schema()
    del missing_multiselect.inputs[0].multiselect
    non_boolean_accept = schema()
    non_boolean_accept.accept_all_inputs = 1  # type: ignore[reportAttributeAccessIssue]
    falsey_output_flag = schema()
    falsey_output_flag.outputs[1].is_output_list = 0  # type: ignore[reportAttributeAccessIssue]
    falsey_input_list = schema()
    falsey_input_list.is_input_list = 0  # type: ignore[reportAttributeAccessIssue]

    class EqualEmptyList:
        def __eq__(self, other: object) -> bool:
            return other == []

    equality_options = schema()
    equality_options.inputs[0].options = EqualEmptyList()
    for lookalike in (
        missing_multiselect,
        non_boolean_accept,
        falsey_output_flag,
        falsey_input_list,
        equality_options,
    ):
        with pytest.raises(CompatError, match="accept_all_inputs"):
            translate_v3_schema(lookalike, CompatTranslation())


def test_v3_multicombo_translates_exact_list_contract_and_refuses_conflicts() -> None:
    declared_default = ["beta", "alpha", "beta"]
    input_obj = FakeInput(
        "providers",
        "COMBO",
        options=["beta", "alpha", "beta"],
        multiselect=True,
        default=declared_default,
        placeholder="Pick providers",
    )
    input_obj.chip = False
    result = translate(FakeSchema("MultiCombo", inputs=[input_obj]))
    (providers,) = result.inputs
    assert providers.type == TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO))
    assert providers.default == ["beta", "alpha", "beta"]
    assert providers.widget == MultiComboWidget(
        options=("beta", "alpha", "beta"),
        placeholder="Pick providers",
        chip=False,
    )
    wire_before_mutation = schema_to_wire(result)
    signature_before_mutation = schema_signature(result)
    declared_default.append("pack-mutated")
    assert providers.default == ["beta", "alpha", "beta"]
    assert schema_to_wire(result) == wire_before_mutation
    assert schema_signature(result) == signature_before_mutation

    for mutate, match in (
        (lambda item: setattr(item, "default", "beta"), "default"),
        (lambda item: setattr(item, "options", ["beta", 1]), "options"),
        (lambda item: setattr(item, "options", []), "static options or a remote source"),
        (lambda item: setattr(item, "control_after_generate", True), "control_after_generate"),
        (lambda item: setattr(item, "chip", "yes"), "chip"),
    ):
        malformed = FakeInput("providers", "COMBO", options=["beta"], multiselect=True)
        malformed.chip = None
        mutate(malformed)
        with pytest.raises(CompatError, match=match):
            translate(FakeSchema("BadMultiCombo", inputs=[malformed]))

    with pytest.raises(CompatError, match="is_input_list"):
        translate(
            FakeSchema(
                "DoubleList",
                inputs=[FakeInput("providers", "COMBO", options=["beta"], multiselect=True)],
                is_input_list=True,
            )
        )

    scalar = FakeInput("choice", "COMBO", options=["a", "b"])
    scalar.multi_select = {"chip": True}
    del scalar.multiselect
    with pytest.raises(CompatError, match="multiselect"):
        translate(FakeSchema("MissingFlag", inputs=[scalar]))

    for malformed_flag in (0, 1, "yes"):
        malformed_flag_input = FakeInput("choice", "COMBO", options=["a", "b"])
        malformed_flag_input.multiselect = malformed_flag  # type: ignore[assignment]
        with pytest.raises(CompatError, match="multiselect must be a bool"):
            translate(FakeSchema("MalformedFlag", inputs=[malformed_flag_input]))


# --- lifecycle flags ---------------------------------------------------------


def test_output_node_and_idempotence_flags() -> None:
    result = translate(FakeSchema("Save", is_output_node=True))
    assert result.output_node
    assert not result.idempotent
    result = translate(FakeSchema("Rand", not_idempotent=True))
    assert not result.idempotent
    assert not result.output_node


def test_translated_v3_schema_declares_may_expand_graph() -> None:
    """V3 classification is exact: only a schema declaring enable_expand is
    flagged (ComfyUI itself refuses NodeOutput.expand without it), and the
    flag stays capability metadata outside the schema signature. At the
    reference revision the shipped StartLoop node is the one enable_expand
    expander; EndLoop never declares it."""
    start_loop = translate(FakeSchema("StartLoop", enable_expand=True))
    assert start_loop.may_expand_graph is True
    wire = schema_to_wire(start_loop)
    assert wire["mayExpandGraph"] is True
    assert schema_signature(start_loop) == schema_signature(
        dataclasses.replace(start_loop, may_expand_graph=False)
    )
    end_loop = translate(FakeSchema("EndLoop"))
    assert end_loop.may_expand_graph is False
    assert "mayExpandGraph" not in schema_to_wire(end_loop)


def test_api_node_maps_to_io_bound_and_suppresses_gpu() -> None:
    # A MODEL input normally marks the node gpu-occupying, but io_bound
    # and occupies are mutually exclusive - waiting on a service wins.
    schema = FakeSchema("Remote", inputs=[FakeInput("model", "MODEL")], is_api_node=True)
    result = translate(schema)
    assert result.io_bound
    assert result.occupies == ()


def test_resident_input_marks_gpu_occupancy() -> None:
    schema = FakeSchema("Sampler", inputs=[FakeInput("model", "MODEL")])
    result = translate(schema)
    assert result.occupies == ("gpu",)


def test_nested_resident_input_marks_gpu_occupancy() -> None:
    combo = FakeInput("mode", "COMFY_DYNAMICCOMBO_V3")
    combo.options = [FakeDynamicOption("load", [FakeInput("model", "MODEL")])]
    result = translate(FakeSchema("NestedSampler", inputs=[combo]))
    assert result.occupies == ("gpu",)

    slot = FakeInput("source", "COMFY_DYNAMICSLOT_V3", optional=True)
    slot.slot = FakeInput("source", "MODEL")
    slot.inputs = [FakeInput("clip", "CLIP")]
    slot_result = translate(FakeSchema("SlotSampler", inputs=[slot]))
    assert slot_result.occupies == ("gpu",)


def test_failed_direct_translation_rolls_back_opaque_types() -> None:
    opaque = FakeInput("opaque", "FAILED_OPAQUE")
    empty = FakeInput("mode", "COMFY_DYNAMICCOMBO_V3")
    empty.options = []
    translation = CompatTranslation()
    translation.opaque_types.add("comfy.KEEP")
    with pytest.raises(CompatError, match="required but has zero options"):
        translate(FakeSchema("Fails", inputs=[opaque, empty]), translation)
    assert translation.opaque_types == {"comfy.KEEP"}


def test_composite_structural_markers_refuse_without_opaque_registration() -> None:
    translation = CompatTranslation()
    with pytest.raises(CompatError, match="dynamic type marker"):
        translate_v3_type("INT,COMFY_DYNAMICSLOT_V3", translation)
    assert translation.opaque_types == set()


def test_deprecation_flag_carries_a_placeholder_message() -> None:
    result = translate(FakeSchema("Old", is_deprecated=True))
    assert result.deprecation is not None
    assert "is_deprecated" in result.deprecation.message
    assert result.deprecation.replacement == ""


def test_dev_only_maps_to_hidden_search_visibility() -> None:
    assert translate(FakeSchema("Dev", is_dev_only=True)).search_visibility == "hidden"
    assert translate(FakeSchema("Normal")).search_visibility == "normal"
