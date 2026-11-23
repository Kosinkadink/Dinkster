"""Nodes whose buttons exercise every declared step, and the gallery's button examples.

Translate, Seed, LoRA preset and App workflow follow examples 24, 23, 22 and 7
of the node example gallery. Translate calls a small local service that needs
no key and answers the same way every time.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    AddMember,
    AddNode,
    Append,
    Ask,
    AssetWidget,
    Button,
    Call,
    CallContext,
    Choice,
    ClearMembers,
    ComboWidget,
    Copy,
    Described,
    Disconnect,
    Download,
    Formula,
    Group,
    Input,
    Interrupt,
    Lookup,
    Node,
    NodeSchema,
    Numbered,
    NumberWidget,
    OpenEditor,
    OpenLink,
    OpenPicker,
    Option,
    Output,
    PasteImage,
    Pick,
    Queue,
    Randomize,
    RemoveMember,
    Reset,
    Row,
    Set,
    SetEach,
    SetMode,
    State,
    StringWidget,
    Swap,
    Toggle,
    TypeExpr,
    lookup,
    pack_function,
)

from .image import IMAGE

STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
COMBO = TypeExpr.concrete(CORE_COMBO)
ASSET = TypeExpr.concrete(ASSET_TYPE)
ANY = TypeExpr.variable("A")


# --- the local translation service ---------------------------------------------

LANGUAGES = ("en", "es", "fr")
_WORDS: Mapping[tuple[str, str], Mapping[str, str]] = {
    ("es", "en"): {
        "hola": "hello",
        "un": "a",
        "gato": "cat",
        "en": "on",
        "la": "the",
        "playa": "beach",
    },
    ("fr", "en"): {
        "bonjour": "hello",
        "un": "a",
        "chat": "cat",
        "sur": "on",
        "la": "the",
        "plage": "beach",
    },
    ("en", "es"): {
        "hello": "hola",
        "a": "un",
        "cat": "gato",
        "on": "en",
        "the": "la",
        "beach": "playa",
    },
}


def _detect(words: list[str]) -> str:
    scores = {source: sum(word in table for word in words) for (source, _), table in _WORDS.items()}
    return max(scores, key=lambda source: scores[source])


@pack_function
def translate_now(
    context: CallContext, text: str, source: str, target: str, delay: float
) -> dict[str, object]:
    """Translate word by word from a fixed table; a word it does not know is kept and noted."""
    words = text.replace(",", " ,").split()
    if source == "auto":
        source = _detect([word.lower() for word in words])
    if source == target:
        return {"text": text}
    table = _WORDS.get((source, target))
    if table is None:
        raise ValueError(f"No translation from {source} to {target} in the test service.")
    translated: list[str] = []
    unknown: list[str] = []
    for index, word in enumerate(words):
        if context.cancelled:
            return {}
        context.progress(index / max(len(words), 1), "Translating")
        if delay:
            time.sleep(delay / max(len(words), 1))
        found = table.get(word.lower())
        if found is None and word != ",":
            unknown.append(word)
        translated.append(found or word)
    if unknown:
        context.note(f"Kept as written: {', '.join(unknown)}.")
    return {"text": " ".join(translated).replace(" ,", ",")}


class ButtonTranslate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.buttons.translate",
            display_name="Translate text",
            category="dev/buttons",
            description="Press Translate to translate the text in place with a local test service.",
            body=(
                Input(
                    "from_translate",
                    COMBO,
                    default="auto",
                    widget=ComboWidget(options=("auto", *LANGUAGES)),
                ),
                Input("to_translate", COMBO, default="en", widget=ComboWidget(options=LANGUAGES)),
                Input("manual_translate", BOOLEAN, default=True),
                Button(
                    "translate",
                    "Translate",
                    icon="languages",
                    enabled=Formula("manual_translate"),
                    disabled_reason=(
                        "Manual translate is off. The translation happens when you press Run."
                    ),
                    steps=(
                        Call(
                            translate_now,
                            args=("text", "from_translate", "to_translate", "service_delay"),
                            writes=("text",),
                        ),
                    ),
                ),
                Input(
                    "text",
                    STRING,
                    default="hola, un gato en la playa",
                    widget=StringWidget(multiline=True, placeholder="Input prompt"),
                ),
                State("service_delay", FLOAT, 0.0, part_of_run=False),
                Output("text", STRING),
            ),
        )

    @classmethod
    def execute(
        cls, *, from_translate: str, to_translate: str, manual_translate: bool, text: str
    ) -> Mapping[str, object]:
        del from_translate, to_translate, manual_translate
        return cls.outputs(text=text)


# --- example 23: seed with reuse ------------------------------------------------


class ButtonSeed(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.buttons.seed",
            display_name="Seed",
            category="dev/buttons",
            description=(
                "A seed with its after-run controller and a button that picks a new fixed "
                "random seed."
            ),
            body=(
                Input(
                    "seed",
                    INT,
                    default=0,
                    widget=NumberWidget(min=-1125899906842624, max=1125899906842624),
                    after_run="randomize",
                ),
                Row(
                    rows=(
                        Button("each_time", "Each time", steps=(SetMode("seed", "randomize"),)),
                        Button(
                            "new_fixed_random",
                            "New",
                            steps=(Randomize("seed"), SetMode("seed", "fixed")),
                        ),
                    )
                ),
                Output("SEED", INT),
            ),
        )

    @classmethod
    def execute(cls, *, seed: int) -> Mapping[str, object]:
        return cls.outputs(SEED=seed)


# --- example 22: LoRA preset picker ---------------------------------------------

_LORAS = ("a.safetensors", "sd/x.safetensors", "sdxl/y.safetensors")
_PRESETS = {
    "SD-NONE": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
    "SD-ALL": "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
    "SD-INS": "1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0",
    "SD-MIDD": "1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0",
    "SD-OUTD": "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1",
}


@lookup
def preset_names() -> list[str]:
    return list(_PRESETS)


@lookup
def preset_vector(name: str) -> str:
    return _PRESETS[name]


class ButtonLoraPreset(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.buttons.lora_preset",
            display_name="Load LoRA (block weight)",
            category="dev/buttons",
            description="Pick a block weight preset and apply its vector with a button.",
            body=(
                Input(
                    "category_filter",
                    COMBO,
                    default="All",
                    widget=ComboWidget(options=("All", "sd", "sdxl")),
                ),
                Input("lora_name", COMBO, default=_LORAS[0], widget=ComboWidget(options=_LORAS)),
                Row(
                    rows=(
                        Input(
                            "preset",
                            COMBO,
                            default="SD-ALL",
                            widget=ComboWidget(options=tuple(_PRESETS)),
                        ),
                        Button(
                            "apply_preset",
                            "Apply",
                            enabled=Formula("known(preset) and preset != None"),
                            steps=(Set("block_vector", Lookup(preset_vector, args=("preset",))),),
                        ),
                    )
                ),
                Button(
                    "pick_preset",
                    "Pick a preset",
                    icon="list",
                    steps=(
                        Pick("chosen", Lookup(preset_names)),
                        Set("preset", Formula("chosen")),
                        Set("block_vector", Lookup(preset_vector, args=(Formula("chosen"),))),
                    ),
                ),
                Input(
                    "block_vector",
                    STRING,
                    default="1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1",
                    widget=StringWidget(multiline=True, placeholder="block weight vectors"),
                ),
                Output("populated_vector", STRING),
            ),
        )

    @classmethod
    def execute(
        cls, *, category_filter: str, lora_name: str, preset: str, block_vector: str
    ) -> Mapping[str, object]:
        del category_filter, lora_name, preset
        return cls.outputs(populated_vector=block_vector)


# --- example 7: app workflow ----------------------------------------------------

_APPS: Mapping[str, list[dict[str, object]]] = {
    "portrait.app.json": [
        {"id": "steps", "label": "Steps", "kind": "number", "default": 20, "min": 1, "max": 150},
        {"id": "cfg", "label": "CFG", "kind": "decimal", "default": 7.5, "min": 0.0, "max": 30.0},
        {"id": "prompt", "label": "Prompt", "kind": "text", "default": "a portrait"},
        {"id": "image", "label": "Image", "kind": "result"},
    ],
    "upscale.app.json": [
        {
            "id": "scale",
            "label": "Scale",
            "kind": "decimal",
            "default": 2.0,
            "min": 1.0,
            "max": 4.0,
        },
        {"id": "image", "label": "Image", "kind": "result"},
    ],
}


@pack_function
def exposure(app: str) -> dict[str, object]:
    """The saved app workflow's exposed inputs and results, as described entries."""
    entries = _APPS.get(app)
    if entries is None:
        raise ValueError(f"No saved app workflow named {app}.")
    return {"exposed": json.dumps({"entries": entries}), "overrides": "{}"}


class ButtonAppWorkflow(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.buttons.app_workflow",
            display_name="App workflow",
            category="dev/buttons",
            description="Load the exposed inputs of a saved app workflow with a button.",
            body=(
                Input(
                    "app",
                    COMBO,
                    default="portrait.app.json",
                    widget=ComboWidget(options=tuple(sorted(_APPS))),
                ),
                Input("overrides", STRING, default="{}", widget=StringWidget(multiline=True)),
                Choice(
                    "source",
                    leader=Formula('"wired" if linked(app) else "picked"'),
                    options=(
                        Option(
                            "picked",
                            rows=(
                                Button(
                                    "load_inputs",
                                    "Load inputs",
                                    icon="download",
                                    steps=(
                                        Call(
                                            exposure, args=("app",), writes=("exposed", "overrides")
                                        ),
                                    ),
                                ),
                                Described(
                                    "exposed",
                                    max_entries=32,
                                    kinds=(
                                        Input("number", INT),
                                        Input("decimal", FLOAT),
                                        Input("toggle", BOOLEAN),
                                        Input("text", STRING),
                                        Output("result", ANY),
                                    ),
                                ),
                            ),
                        ),
                        Option(
                            "wired",
                            rows=(Input("generic_in", ANY, required=False),),
                        ),
                    ),
                ),
                Output("exposed_info", STRING),
            ),
        )

    @classmethod
    def execute(cls, **values: object) -> Mapping[str, object]:
        return cls.outputs(exposed_info=json.dumps(sorted(values), default=str))


# --- every step, for tests and the gallery -------------------------------------


class ButtonFieldSteps(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.buttons.field_steps",
            display_name="Buttons: field steps",
            category="dev/buttons",
            description="One button per step that sets fields.",
            body=(
                Input("title", STRING, default="draft"),
                Input("width", INT, default=512, widget=NumberWidget(min=64, max=2048, step=64)),
                Input("height", INT, default=768, widget=NumberWidget(min=64, max=2048, step=64)),
                Input(
                    "scale", FLOAT, default=0.5, widget=NumberWidget(min=0.0, max=1.0, step=0.25)
                ),
                Input("flag", BOOLEAN, default=False),
                Input(
                    "mode",
                    COMBO,
                    default="fast",
                    widget=ComboWidget(options=("fast", "balanced", "best")),
                ),
                Input(
                    "seed", INT, default=7, widget=NumberWidget(min=0, max=1000), after_run="fixed"
                ),
                Input("image", IMAGE, required=False),
                Row(
                    rows=(
                        Button(
                            "double",
                            "Double",
                            steps=(
                                Set("width", Formula("width * 2")),
                                Set("height", Formula("width + 64")),
                            ),
                        ),
                        Button(
                            "swap",
                            "Swap",
                            icon="arrow-left-right",
                            steps=(Swap("width", "height"),),
                        ),
                        Button("toggle", "Toggle", steps=(Toggle("flag"),)),
                        Button(
                            "reset",
                            "Reset",
                            icon="rotate-ccw",
                            steps=(Reset("width"), Reset("title")),
                        ),
                    )
                ),
                Row(
                    rows=(
                        Button("append", "Append", steps=(Append("title", Formula("mode")),)),
                        Button(
                            "random", "Random", icon="dice-5", steps=(Randomize("seed", 10, 20),)
                        ),
                        Button("increment", "Increment", steps=(SetMode("seed", "increment"),)),
                        Button(
                            "best",
                            "Best",
                            steps=(Set("mode", "best"), Set("scale", Formula("scale + 0.25"))),
                        ),
                    )
                ),
                Row(
                    rows=(
                        Button(
                            "too_wide", "Too wide", steps=(Set("title", "wide"), Set("width", 4096))
                        ),
                        Button("off_step", "Off step", steps=(Set("width", 100),)),
                        Button("bad_option", "Bad option", steps=(Set("mode", "slow"),)),
                        Button("quarter", "Quarter", steps=(Set("scale", 0.3),)),
                    )
                ),
                Row(
                    rows=(
                        Button(
                            "needs_image",
                            "Needs image",
                            steps=(Set("title", Formula('"image " + str(linked(image))')),),
                        ),
                        Button("disconnect", "Disconnect", icon="x", steps=(Disconnect("image"),)),
                        Button(
                            "add_source",
                            "Add source",
                            icon="plus",
                            steps=(
                                AddNode("dev.image.gradient", to="image", values={"width": 64}),
                            ),
                        ),
                    )
                ),
            ),
            menu=(
                Button("menu_reset", "Reset size", steps=(Reset("width"), Reset("height"))),
                Button(
                    "menu_copy",
                    "Copy size",
                    icon="copy",
                    steps=(Copy(Formula('str(width) + "x" + str(height)')),),
                ),
            ),
        )

    @classmethod
    def execute(cls, **values: object) -> Mapping[str, object]:
        del values
        return {}


class ButtonMemberSteps(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.buttons.member_steps",
            display_name="Buttons: member steps",
            category="dev/buttons",
            description="Buttons that add, remove and fill the members of a group.",
            body=(
                Group(
                    "loras",
                    keys=Numbered("lora_"),
                    grows="button",
                    min_members=1,
                    max_members=4,
                    member=(
                        Input("name", STRING, default="none"),
                        Input(
                            "strength", FLOAT, default=1.0, widget=NumberWidget(min=-2.0, max=2.0)
                        ),
                        Input("clip", IMAGE, required=False),
                    ),
                ),
                Row(
                    rows=(
                        Button(
                            "add",
                            "Add",
                            icon="plus",
                            steps=(
                                AddMember(
                                    "loras",
                                    values={
                                        "name": "added",
                                        "strength": Formula("len(loras) * 0.5"),
                                    },
                                ),
                            ),
                        ),
                        Button(
                            "remove_first", "Remove first", steps=(RemoveMember("loras", "first"),)
                        ),
                        Button(
                            "remove_last", "Remove last", steps=(RemoveMember("loras", "last"),)
                        ),
                        Button(
                            "remove_second",
                            "Remove 2nd",
                            steps=(RemoveMember("loras", Formula("1")),),
                        ),
                    )
                ),
                Row(
                    rows=(
                        Button("clear", "Clear", icon="trash-2", steps=(ClearMembers("loras"),)),
                        Button(
                            "copy_first",
                            "Copy first",
                            steps=(SetEach("loras", "strength", Formula("loras.strength[0]")),),
                        ),
                        Button(
                            "halve_each",
                            "Halve each",
                            steps=(SetEach("loras", "strength", Formula("strength / 2")),),
                        ),
                    )
                ),
            ),
        )

    @classmethod
    def execute(cls, **values: object) -> Mapping[str, object]:
        del values
        return {}


_FILL = 'join(", ", [fixed(start + (end - start) * i / (count - 1), 2) for i in range(count)])'


class ButtonAskSteps(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.buttons.ask_steps",
            display_name="Buttons: lists, forms and outside steps",
            category="dev/buttons",
            description="Buttons that ask first, open host editors, and act outside the workflow.",
            body=(
                Input(
                    "values",
                    STRING,
                    default="0.0, 0.25, 0.5, 0.75, 1.0",
                    widget=StringWidget(multiline=True),
                ),
                Input("size", STRING, default="medium"),
                Input("site", STRING, default="example.com"),
                Input(
                    "picture",
                    ASSET,
                    required=False,
                    widget=AssetWidget(accept=("image/png", "image/jpeg"), kind="media/image"),
                ),
                Row(
                    rows=(
                        Button(
                            "fill",
                            "Fill",
                            steps=(
                                Ask(
                                    "Fill with a range",
                                    rows=(
                                        Input("start", FLOAT, default=0.0),
                                        Input("end", FLOAT, default=1.0),
                                        Input(
                                            "count",
                                            INT,
                                            default=5,
                                            widget=NumberWidget(min=2, max=64),
                                        ),
                                    ),
                                ),
                                Set(
                                    "values",
                                    Formula(_FILL),
                                ),
                            ),
                        ),
                        Button(
                            "size",
                            "Size",
                            icon="list",
                            steps=(
                                Pick("chosen", ("small", "medium", "large")),
                                Set("size", Formula("chosen")),
                            ),
                        ),
                    )
                ),
                Row(
                    rows=(
                        Button(
                            "pick_picture",
                            "Choose",
                            icon="folder-open",
                            steps=(OpenPicker("picture"),),
                        ),
                        Button(
                            "paste", "Paste", icon="clipboard-paste", steps=(PasteImage("picture"),)
                        ),
                        Button("mask", "Mask", icon="pencil", steps=(OpenEditor("picture"),)),
                    )
                ),
                Row(
                    rows=(
                        Button("run", "Run", icon="play", steps=(Queue(),)),
                        Button("run_here", "Run to here", steps=(Queue(to_here=True),)),
                        Button("stop", "Stop", icon="square", steps=(Interrupt(),)),
                    )
                ),
                Row(
                    rows=(
                        Button("copy", "Copy", icon="copy", steps=(Copy(Formula("values")),)),
                        Button(
                            "save",
                            "Save",
                            icon="save",
                            steps=(Download(Formula("values"), "values.txt"),),
                        ),
                        Button(
                            "open",
                            "Open site",
                            icon="external-link",
                            steps=(OpenLink(Formula('"https://" + site')),),
                        ),
                    )
                ),
                Output("values", STRING),
            ),
        )

    @classmethod
    def execute(cls, *, values: str, **rest: object) -> Mapping[str, object]:
        del rest
        return cls.outputs(values=values)


# --- call outcomes, for tests ----------------------------------------------------

CALL_MODES = ("answer", "partial", "raise", "extra", "wrong_type", "slow", "file", "large_file")


@pack_function(time_limit=2.0)
def call_probe(context: CallContext, mode: str, delay: float) -> object:
    """Answer each way a pack function can, for tests of the button runner."""
    deadline = time.monotonic() + delay
    while time.monotonic() < deadline and not context.cancelled:
        context.progress(min(1.0, 1 - (deadline - time.monotonic()) / max(delay, 1e-9)), "Waiting")
        time.sleep(0.02)
    if mode == "answer":
        return {"first": "one", "second": 2}
    if mode == "partial":
        context.note("The second field was left out.")
        return {"first": "only"}
    if mode == "raise":
        raise ValueError("No sheet named video in this file.")
    if mode == "extra":
        return {"first": "one", "elsewhere": 3}
    if mode == "wrong_type":
        return {"second": "two"}
    if mode == "slow":
        time.sleep(3.0)
        return {"first": "late"}
    if mode == "file":
        return b"pack file\n"
    return b"\0" * (64 * 1024 * 1024 + 1)


class ButtonCallProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.buttons.call_probe",
            display_name="Buttons: call outcomes",
            category="dev/buttons",
            description="A button whose pack function answers in each way the runner handles.",
            body=(
                Input("mode", COMBO, default="answer", widget=ComboWidget(options=CALL_MODES)),
                Input("delay", FLOAT, default=0.0, widget=NumberWidget(min=0.0, max=10.0)),
                Input("first", STRING, default=""),
                Input("second", INT, default=0, widget=NumberWidget(min=0, max=10)),
                Row(
                    rows=(
                        Button(
                            "call",
                            "Call",
                            steps=(
                                Set("first", "staged"),
                                Call(
                                    call_probe, args=("mode", "delay"), writes=("first", "second")
                                ),
                            ),
                        ),
                        Button(
                            "download",
                            "Download",
                            icon="download",
                            steps=(Call(call_probe, args=("mode", "delay"), download="probe.txt"),),
                        ),
                    )
                ),
            ),
        )

    @classmethod
    def execute(cls, **values: object) -> Mapping[str, object]:
        del values
        return {}


BUTTON_NODES = [
    ButtonTranslate,
    ButtonSeed,
    ButtonLoraPreset,
    ButtonAppWorkflow,
    ButtonFieldSteps,
    ButtonMemberSteps,
    ButtonAskSteps,
    ButtonCallProbe,
]
