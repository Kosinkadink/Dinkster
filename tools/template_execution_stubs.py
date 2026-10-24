"""CPU-only implementations for the non-control nodes in the switch template."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr, TypeRegistry

_SPECS = {
    "LoraLoaderModelOnly": ((("model", "MODEL"),), (("MODEL", "MODEL"),)),
    "SaveImage": ((("images", "IMAGE"),), ()),
    "UNETLoader": ((), (("MODEL", "MODEL"),)),
    "MarkdownNote": ((), ()),
    "0f79b7f9-d214-4587-b475-bc429519b976": (
        (("model", "MODEL"), ("text", "STRING")),
        (("IMAGE", "IMAGE"),),
    ),
    "PreviewAny": ((("source", "IMAGE"),), (("STRING", "STRING"),)),
    "CLIPLoader": ((("clip_name", "COMBO"),), (("CLIP", "CLIP"),)),
    "VAELoader": ((("vae_name", "COMBO"),), (("VAE", "VAE"),)),
    "KSampler": (
        (
            ("model", "MODEL"),
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
            ("latent_image", "LATENT"),
        ),
        (("LATENT", "LATENT"),),
    ),
    "ModelSamplingAuraFlow": ((("model", "MODEL"),), (("MODEL", "MODEL"),)),
}


def _node(
    node_type: str, inputs: tuple[tuple[str, str], ...], outputs: tuple[tuple[str, str], ...]
) -> type[Node]:
    def define_schema(cls: type[Node]) -> NodeSchema:
        return NodeSchema(
            node_type=node_type,
            inputs=tuple(InputSpec(name, TypeExpr.concrete(type_id)) for name, type_id in inputs),
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id)) for name, type_id in outputs
            ),
        )

    def execute(cls: type[Node], **values: object) -> Mapping[str, object]:
        result = {}
        for name, _type_id in outputs:
            result[name] = values.get(name) if name in values else None
        return cls.outputs(**result)

    return type(
        node_type.replace("-", "_"),
        (Node,),
        {"define_schema": classmethod(define_schema), "execute": classmethod(execute)},
    )


TEMPLATE_STUB_NODES = [
    _node(node_type, inputs, outputs) for node_type, (inputs, outputs) in _SPECS.items()
]


def register_template_stub_types(registry: TypeRegistry) -> None:
    for type_id in {type_id for spec in _SPECS.values() for port in spec for _, type_id in port}:
        if type_id not in registry:
            registry.register(type_id)
