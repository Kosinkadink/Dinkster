"""Save canonical SVG streams through the mounted batch writer."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping
from io import BytesIO

from dinkster_api.v1 import (
    CORE_STRING,
    SVG_TYPE,
    AssetError,
    AssetWriter,
    InputSpec,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    attach_saved_artifact,
    current_export_snapshot,
    materialize_svg,
)
from dinkster_values.limits import ENCODED_MEDIA_LIMIT_BYTES


class SaveSVGNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveSVGNode",
            display_name="Save SVG",
            description="Save SVG files on disk.",
            category="image",
            search_terms=("export vector", "save vector graphics"),
            inputs=(
                InputSpec("svg", TypeExpr.concrete(SVG_TYPE)),
                InputSpec(
                    "filename_prefix",
                    TypeExpr.concrete(CORE_STRING),
                    default="svg/ComfyUI",
                    widget=StringWidget(multiline=False),
                    doc=(
                        "The prefix for the file to save. This may include formatting information "
                        "such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values "
                        "from nodes."
                    ),
                ),
                InputSpec(
                    "mount",
                    TypeExpr.concrete(CORE_STRING),
                    required=False,
                    default=None,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
                ),
            ),
            outputs=(OutputSpec("svg", TypeExpr.concrete(SVG_TYPE)),),
            output_node=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls, *, svg: object, filename_prefix: str = "svg/ComfyUI", mount: str | None = None
    ) -> Mapping[str, object]:
        snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
        if not snapshot:
            raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
        writer = AssetWriter(MountSnapshotWriter(snapshot))
        submitted = current_export_snapshot()
        metadata: dict[str, object] = {}
        if submitted is not None:
            if submitted.prompt is not None:
                metadata["prompt"] = dict(submitted.prompt)
            if submitted.extra_pnginfo is not None:
                metadata.update(submitted.extra_pnginfo)
        metadata_json = json.dumps(metadata, indent=2) if metadata else None
        element = (
            "  <metadata>\n                <![CDATA[\n            "
            f"{metadata_json}\n                ]]>\n            </metadata>\n            "
        )

        def sources() -> Iterator[BytesIO]:
            for stream in materialize_svg(svg):
                with stream:
                    content = stream.read().decode("utf-8")
                if metadata_json:
                    content = re.sub(
                        r"(<svg[^>]*>)",
                        lambda match: match.group(1) + "\n" + element,
                        content,
                        flags=re.UNICODE,
                    )
                with BytesIO(content.encode("utf-8")) as source:
                    yield source

        refs = writer.save_stream_batch(
            writer.output_target(filename_prefix, mount=mount),
            sources(),
            suffix="_.svg",
            media_type="image/svg+xml",
            limit=ENCODED_MEDIA_LIMIT_BYTES,
        )
        for ref in refs:
            attach_saved_artifact(ref.to_wire())
        return cls.outputs(svg=svg)
