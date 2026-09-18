"""The widget/socket gallery: dev nodes that exercise every construct the
native schema wire (v11) can express, so the frontend can verify rendering
and decode against a real server instead of only its local fixtures.

Coverage surface, not semantics: every node is a passthrough or trivial
producer. What matters is the schemas - primitive widgets with defaults,
numeric min/max/step and control-after-generate, single-line vs multiline
strings, per-input display names, static/remote/empty-remote combos, bare
and labeled booleans, asset and save-target widgets, required/optional
inputs, optional outputs, unions of 2/3/4+ members, wildcards, lists
(optional, of-union, nested), match templates, and searchTerms. The
companion template document (gallery_template.json) instantiates all of
it, connected and not.

Known NOT expressible on the native wire today (do not invent fields here;
these are tracked contract gaps, see ROADMAP):
- a COLOR widget
- union-of-list types (list<A>|list<B>): union entries are atoms by
  design; list-of-union (list<A|B>) IS expressible and shown instead
- union/wildcard-typed OUTPUTS execute only when a variable solves them:
  dev.gallery.exotic_out exists for rendering, executing it is a
  deliberate contract error
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Mapping, Sequence

import numpy as np
from dinkster_api.v1 import (
    ABSENT,
    ASSET_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    SAVE_TARGET_TYPE,
    AssetWidget,
    BooleanWidget,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SaveTargetWidget,
    StringWidget,
    TypeExpr,
    TypeRegistry,
    register_asset_type,
    register_save_target_type,
    resolver_from_env,
)

from .image import DEV_IMAGE

# Name-only marker types (DESIGN 3.2's lazy path: a registration can be
# nothing but a name). They exist so union sockets have distinct members
# to render, the way IMAGE/MASK/LATENT/AUDIO differ in ComfyUI.
GALLERY_MASK = "dev.gallery.mask"
GALLERY_LATENT = "dev.gallery.latent"
GALLERY_AUDIO = "dev.gallery.audio"

IMAGE = TypeExpr.concrete(DEV_IMAGE)
MASK = TypeExpr.concrete(GALLERY_MASK)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
COMBO = TypeExpr.concrete(CORE_COMBO)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
ASSET = TypeExpr.concrete(ASSET_TYPE)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)

SAMPLERS_CHOICE_ID = "dev.gallery.samplers"
EMPTY_CHOICE_ID = "dev.gallery.empty"

SAMPLERS = ("euler", "euler_ancestral", "heun", "dpmpp_2m", "ddim", "uni_pc")

# The shipped starter document instantiating the whole gallery. Metadata
# here is the single importable source; the [[pack.templates]] entry in
# dinkster-pack.toml mirrors it for the manifest-composed path (a test pins
# the two together).
GALLERY_TEMPLATE_ID = "dev-gallery"
GALLERY_TEMPLATE_NAME = "Widget & Socket Gallery"
GALLERY_TEMPLATE_DESCRIPTION = (
    "Every native wire construct on one canvas: widgets, socket variants, "
    "lists, match templates - connected and unconnected."
)
GALLERY_TEMPLATE_TAGS = ("dev", "gallery")


def gallery_template_bytes() -> bytes:
    """The template document bytes, verbatim from package data - whoever
    serves the template serves exactly these bytes under their digest."""
    resource = importlib.resources.files("dinkster_nodes_dev") / "gallery_template.json"
    return resource.read_bytes()


def combo_choices() -> Mapping[str, Sequence[str]]:
    """Choice lists behind /api/choices/{id}: one populated (the remote
    combo's authoritative source) and one deliberately empty (the
    legal-transient "remote answered with nothing" state the frontend
    renders as a loading/empty dropdown)."""
    return {SAMPLERS_CHOICE_ID: SAMPLERS, EMPTY_CHOICE_ID: ()}


def register_gallery_types(registry: TypeRegistry) -> None:
    """Marker types plus the shared asset/save-target types the widget
    nodes speak. Guarded like the image module's save-target registration:
    the shared types are one definition wherever they come from."""
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if SAVE_TARGET_TYPE not in registry:
        register_save_target_type(registry)
    for type_id in (GALLERY_MASK, GALLERY_LATENT, GALLERY_AUDIO):
        registry.register(type_id)


class GalleryWidgets(Node):
    """Every native input-widget descriptor on one node: primitive
    defaults, bounded and unbounded numbers, a seed with a
    control-after-generate controller, single-line and multiline strings,
    a per-input display name, static/remote/empty-remote combos, bare and
    labeled booleans. COLOR remains off the wire - never invented here."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.gallery.widgets",
            display_name="Gallery: Widgets",
            category="dev/gallery",
            description=(
                "Rendering surface for every native widget descriptor; "
                "executes as an echo of its primitive inputs."
            ),
            search_terms=("gallery", "widget zoo", "kitchen sink"),
            inputs=(
                InputSpec(
                    "count",
                    INT,
                    default=50,
                    display_name="Item Count",
                    widget=NumberWidget(min=1, max=100, step=1),
                    doc="bounded int + a per-input display name",
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, control_after_generate="randomize"),
                    doc="seed-style int with a control-after-generate controller",
                ),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.05),
                    doc="bounded float with a fractional step",
                ),
                InputSpec("bare_number", INT, default=7, doc="plain int, no widget: unbounded"),
                InputSpec("line", STRING, default="a single-line string"),
                InputSpec(
                    "prose",
                    STRING,
                    default="first line\nsecond line\nthird line",
                    widget=StringWidget(multiline=True),
                    doc="multiline string editor",
                ),
                InputSpec(
                    "combo_static",
                    COMBO,
                    default="alpha",
                    widget=ComboWidget(
                        options=(
                            "alpha",
                            "beta",
                            "gamma",
                            "a distinctly longer option label",
                        )
                    ),
                ),
                InputSpec(
                    "combo_remote",
                    COMBO,
                    default=SAMPLERS[0],
                    widget=ComboWidget(
                        options=SAMPLERS,
                        remote_route=f"/api/choices/{SAMPLERS_CHOICE_ID}",
                        refresh_button=True,
                    ),
                    doc="static snapshot + authoritative remote route",
                ),
                InputSpec(
                    "combo_remote_empty",
                    COMBO,
                    default="",
                    widget=ComboWidget(
                        remote_route=f"/api/choices/{EMPTY_CHOICE_ID}",
                        refresh_button=True,
                    ),
                    doc="no static options; the route answers [] - legal transient",
                ),
                InputSpec("flag", BOOLEAN, default=False, doc="bare toggle, no widget"),
                InputSpec(
                    "flag_labeled",
                    BOOLEAN,
                    default=True,
                    widget=BooleanWidget(label_on="enable", label_off="disable"),
                ),
            ),
            outputs=(
                OutputSpec("count", INT),
                OutputSpec("text", STRING),
                OutputSpec("flag", BOOLEAN),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        count: int,
        seed: int,
        strength: float,
        bare_number: int,
        line: str,
        prose: str,
        combo_static: str,
        combo_remote: str,
        combo_remote_empty: str,
        flag: bool,
        flag_labeled: bool,
    ) -> Mapping[str, object]:
        del seed, strength, bare_number, prose
        del combo_remote, combo_remote_empty, flag_labeled
        return cls.outputs(count=count, text=f"{line}/{combo_static}", flag=flag)


class GalleryAssets(Node):
    """The asset picker and save-target widgets. The asset input is
    optional so the gallery template renders the widget without shipping
    real asset bytes; execution just names what it got."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.gallery.assets",
            display_name="Gallery: Assets",
            category="dev/gallery",
            inputs=(
                InputSpec(
                    "picture",
                    ASSET,
                    required=False,
                    widget=AssetWidget(accept=("image/png", "image/jpeg"), kind="media/image"),
                ),
                InputSpec(
                    "destination",
                    SAVE_TARGET,
                    required=False,
                    widget=SaveTargetWidget(suffix=".png"),
                ),
            ),
            outputs=(OutputSpec("summary", STRING),),
        )

    @classmethod
    def execute(cls, *, picture: object = None, destination: object = None) -> Mapping[str, object]:
        return cls.outputs(summary=f"picture={picture!r} destination={destination!r}")


class GallerySource(Node):
    """Producer for the connected half of the socket gallery: concrete,
    optional (deliberately ABSENT), list, and nested-list outputs, all
    executable. Union/wildcard outputs live on dev.gallery.exotic_out."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.gallery.source",
            display_name="Gallery: Source",
            category="dev/gallery",
            inputs=(InputSpec("size", INT, default=4),),
            outputs=(
                OutputSpec("image", IMAGE),
                OutputSpec("mask", MASK),
                OutputSpec("maybe_image", IMAGE, optional=True, doc="always ABSENT here"),
                OutputSpec("image_list", TypeExpr.list_of(IMAGE)),
                OutputSpec("mask_list", TypeExpr.list_of(MASK)),
                OutputSpec(
                    "image_grid",
                    TypeExpr.list_of(TypeExpr.list_of(IMAGE)),
                    doc="nested list<list<dev.image>>",
                ),
            ),
        )

    @classmethod
    def execute(cls, *, size: int) -> Mapping[str, object]:
        side = max(1, min(int(size), 64))
        image = np.zeros((side, side), dtype=np.float32)
        mask = np.ones((side, side), dtype=np.float32)
        return cls.outputs(
            image=image,
            mask=mask,
            maybe_image=ABSENT,
            image_list=[image, image * 0.5],
            mask_list=[mask],
            image_grid=[[image], [image, image]],
        )


class GallerySockets(Node):
    """Scalar input socket variants: required/optional concrete, unions of
    2/3/4 members, wildcard, optional wildcard."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.gallery.sockets",
            display_name="Gallery: Sockets",
            category="dev/gallery",
            inputs=(
                InputSpec("req_image", IMAGE),
                InputSpec("opt_image", IMAGE, required=False),
                InputSpec("union2", TypeExpr.union(DEV_IMAGE, GALLERY_MASK)),
                InputSpec(
                    "union3",
                    TypeExpr.union(DEV_IMAGE, GALLERY_MASK, GALLERY_LATENT),
                ),
                InputSpec(
                    "union4",
                    TypeExpr.union(
                        DEV_IMAGE,
                        GALLERY_MASK,
                        GALLERY_LATENT,
                        GALLERY_AUDIO,
                        CORE_STRING,
                    ),
                    doc="5 members - past the frontend's 3-slice pie truncation",
                ),
                InputSpec(
                    "opt_union3",
                    TypeExpr.union(DEV_IMAGE, GALLERY_MASK, GALLERY_LATENT),
                    required=False,
                ),
                InputSpec("any_in", TypeExpr.wildcard()),
                InputSpec("opt_any", TypeExpr.wildcard(), required=False),
            ),
            outputs=(OutputSpec("image", IMAGE),),
        )

    @classmethod
    def execute(
        cls,
        *,
        req_image: object,
        union2: object,
        any_in: object,
        opt_image: object = None,
        union3: object = None,
        union4: object = None,
        opt_union3: object = None,
        opt_any: object = None,
    ) -> Mapping[str, object]:
        del union2, union3, union4, opt_image, opt_union3, any_in, opt_any
        return cls.outputs(image=req_image)


class GalleryLists(Node):
    """List socket variants: required/optional list<T>, list-of-union
    (the expressible cousin of the inexpressible union-of-lists), and
    nested list<list<T>>."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.gallery.lists",
            display_name="Gallery: Lists",
            category="dev/gallery",
            inputs=(
                InputSpec("list_req", TypeExpr.list_of(IMAGE)),
                InputSpec("list_opt", TypeExpr.list_of(IMAGE), required=False),
                InputSpec(
                    "list_of_union",
                    TypeExpr.list_of(TypeExpr.union(DEV_IMAGE, GALLERY_MASK)),
                    doc="list<dev.image|dev.gallery.mask> - union-of-lists is not expressible",
                ),
                InputSpec(
                    "list_nested",
                    TypeExpr.list_of(TypeExpr.list_of(IMAGE)),
                    required=False,
                ),
            ),
            outputs=(OutputSpec("list_out", TypeExpr.list_of(IMAGE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        list_req: Sequence[object],
        list_of_union: Sequence[object],
        list_opt: Sequence[object] | None = None,
        list_nested: Sequence[Sequence[object]] | None = None,
    ) -> Mapping[str, object]:
        del list_of_union, list_opt, list_nested
        return cls.outputs(list_out=list(list_req))


class GalleryMatch(Node):
    """One match template T through input and output: the template ships
    two instances, one solved by a connection and one left open."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.gallery.match",
            display_name="Gallery: Match",
            category="dev/gallery",
            inputs=(InputSpec("var_in", TypeExpr.variable("T")),),
            outputs=(OutputSpec("var_out", TypeExpr.variable("T")),),
        )

    @classmethod
    def execute(cls, *, var_in: object) -> Mapping[str, object]:
        return cls.outputs(var_out=var_in)


class GalleryExoticOut(Node):
    """Union- and wildcard-typed OUTPUT sockets, for rendering only. The
    wire expresses them, so the gallery must show them - but a worker can
    only wrap an output whose type resolves to one concrete runtime id,
    so executing this node is a deliberate, documented contract error."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.gallery.exotic_out",
            display_name="Gallery: Exotic Outputs",
            category="dev/gallery",
            description=(
                "Render-only: union/wildcard outputs cannot be wrapped at "
                "execution (no concrete runtime type) - running this node "
                "is a contract error by design."
            ),
            inputs=(InputSpec("trigger", STRING, default=""),),
            outputs=(
                OutputSpec("union_out", TypeExpr.union(DEV_IMAGE, GALLERY_MASK)),
                OutputSpec("any_out", TypeExpr.wildcard()),
                OutputSpec("maybe_any", TypeExpr.wildcard(), optional=True),
            ),
        )

    @classmethod
    def execute(cls, *, trigger: str) -> Mapping[str, object]:
        del trigger
        # Reaching the worker's wrap step with these outputs is the
        # documented contract error; returning ABSENT for the optional one
        # keeps the error message pointed at the union output.
        return cls.outputs(union_out=None, any_out=None, maybe_any=ABSENT)


GALLERY_NODES: list[type[Node]] = [
    GalleryWidgets,
    GalleryAssets,
    GallerySource,
    GallerySockets,
    GalleryLists,
    GalleryMatch,
    GalleryExoticOut,
]
