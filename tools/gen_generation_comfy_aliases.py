"""Generate the generation pack's maintained ComfyUI alias registry.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies:

    COMFYUI_ROOT=/path/to/legacy/ComfyUI \
      CURRENT_COMFYUI_ROOT=/path/to/current/ComfyUI \
      PYTHONPATH=<all local package src paths> \
      /path/to/python tools/gen_generation_comfy_aliases.py

Both ComfyUI checkouts must be clean and pinned to the commits below. The
registry carries the conditioning-utility, loader, latent tensor-math,
latent batch/metadata, latent operation, and current SeedVR2 workflow aliases.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from dinkster_compat_comfy import CompatTranslation, translate_node, translate_v3_schema
from dinkster_schema import (
    AssetWidget,
    ComboWidget,
    InputFamilyMapping,
    InputFamilyMember,
    InputSpec,
    MappingSource,
    ReplacementCase,
    ReplacementLink,
    ReplacementNode,
    ReplacementPredicate,
    ReplacementRule,
    TypeExpr,
    ValueTransform,
)
from dinkster_schema.model import NodeSchema
from dinkster_schema.replace import rule_to_wire
from dinkster_schema.wire import schema_to_wire

COMFY_BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
CURRENT_COMFY_BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
CHROMA_RADIANCE_BASELINE = "8a33128f2f8c5585c57486c07de481241e70a39c"
TRELLIS2_BASELINE = "8a33128f"
KJ_BASELINE = "827fe6ee0ed7348d8daa988ed852bedf1272380c"
ESSENTIALS_BASELINE = "9d9f4bedfc9f0321c19faf71855e228c93bd0dc9"
LOADER_KJ_BASELINE = "3f20054214fec9f9234fd3841ae6f1e4287948f6"
RGTHREE_BASELINE = "35c9f1e186603ba312d3b15350e89aa50b860ee6"
EASY_USE_BASELINE = "58e077a7435631301cf7443412515cf958e7f3d1"
WAS_BASELINE = "44de705818d4663fefefde57ffe0ea5a9ea39df4"
PYSSSSS_BASELINE = "609f3afaa74b2f88ef9ce8d939626065e3247469"
EFFICIENCY_BASELINE = "4579b7d6076b2870998a08f5d37883fbc8261ff2"
INSPIRE_BASELINE = "6b2ca017a168bcdba5f22c258b3b86c5c76470ca"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages" / "dinkster-nodes-generation" / "comfy-aliases.json"

LATENT_PARITY_EVIDENCE = [
    "packages/dinkster-inference-torch/tests/test_latent_op_nodes.py::test_latent_ops_match_comfy_goldens"
]

LATENT_BATCH_PARITY_EVIDENCE = [
    "packages/dinkster-inference-torch/tests/test_latent_op_nodes.py::test_latent_batch_ops_match_comfy_goldens"
]

LATENT_OPERATION_PARITY_EVIDENCE = [
    "packages/dinkster-inference-torch/tests/test_latent_op_nodes.py::test_latent_operation_ops_match_comfy_goldens"
]

LATENT_APPLY_OPERATION_CFG_PARITY_EVIDENCE = [
    "packages/dinkster-inference-torch/tests/test_latent_op_nodes.py::test_latent_apply_operation_cfg_matches_comfy_goldens"
]

LOADER_STACK_EVIDENCE = [
    "tests/test_generation_comfy_aliases.py::test_loader_stack_aliases_are_conservative_and_ordered"
]

GENERATE_NOISE_PARITY_EVIDENCE = [
    "packages/dinkster-inference-torch/tests/test_noise_nodes.py::test_generate_noise_matches_kj_goldens"
]

INJECT_NOISE_PARITY_EVIDENCE = [
    "packages/dinkster-inference-torch/tests/test_noise_nodes.py::test_inject_noise_matches_kj_goldens"
]

TRELLIS2_WORKFLOW_EVIDENCE = [
    "tests/test_generation_comfy_aliases.py::test_trellis2_official_workflow_surface_is_maintained"
]

SEEDVR2_WORKFLOW_EVIDENCE = [
    "tests/test_generation_comfy_aliases.py::test_seedvr2_workflow_aliases_cover_current_core_surface",
    "tests/test_generation_nodes.py::test_seedvr2_comfy_aliases_translate_outputs_and_manual_chunking",
]


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _record(
    *,
    node_class: str,
    carrier: str,
    rule: ReplacementRule,
    evidence: list[str],
    source_pack: str = "comfy-core",
    revision: str = "b78cec87",
    tier: str = "exact",
) -> dict[str, object]:
    return {
        "id": f"comfy_alias:{source_pack}/{node_class}",
        "mappingKind": "op",
        "carrier": carrier,
        "source": {
            "pack": source_pack,
            "nodeClass": node_class,
            "nodeType": rule.from_type,
            "revision": revision,
        },
        "replacement": rule_to_wire(rule),
        "confidence": {"tier": tier, "evidence": evidence},
    }


def _copy_rule(
    from_type: str,
    carrier: str,
    *input_ids: str,
    output_id: str = "conditioning",
    source_output_id: str | None = None,
) -> ReplacementRule:
    return ReplacementRule(
        from_type=from_type,
        cases=(
            ReplacementCase.build(
                carrier,
                inputs={input_id: MappingSource.copy(input_id) for input_id in input_ids},
                outputs={output_id: source_output_id or output_id},
            ),
        ),
    )


def _combo_rule(
    *,
    from_type: str,
    carrier: str,
    slot: str,
    variant: str,
    inputs: dict[str, str],
) -> ReplacementRule:
    return ReplacementRule(
        from_type=from_type,
        cases=(
            ReplacementCase.build(
                carrier,
                slot_variants={slot: variant},
                inputs={target: MappingSource.copy(source) for target, source in inputs.items()},
                outputs={"conditioning": "conditioning"},
            ),
        ),
    )


def _core_v3_schema(node_class: type[Any]) -> NodeSchema:
    return translate_v3_schema(node_class.GET_SCHEMA(), CompatTranslation())


def _static_schema(
    node_class: str,
    *,
    namespace: str,
    category: str | None = None,
    description: str = "",
    required: dict[str, object],
    optional: dict[str, object] | None = None,
    returns: tuple[str, ...] = ("CONDITIONING",),
    return_names: tuple[str, ...] | None = None,
) -> NodeSchema:
    def input_types(_cls: type[Any]) -> dict[str, object]:
        value: dict[str, object] = {"required": required}
        if optional:
            value["optional"] = optional
        return value

    def execute(_self: object) -> None:
        raise RuntimeError("static schema shim")

    attributes: dict[str, object] = {
        "INPUT_TYPES": classmethod(input_types),
        "RETURN_TYPES": returns,
        "FUNCTION": "execute",
        "CATEGORY": namespace if category is None else category,
        "DESCRIPTION": description,
        "execute": execute,
    }
    if return_names is not None:
        attributes["RETURN_NAMES"] = return_names
    shim = type(f"_{node_class.replace('+', 'Plus')}", (), attributes)
    return translate_node(
        node_class,
        shim,
        CompatTranslation(),
        namespace=namespace,
    ).schema()


def _with_asset_inputs(
    schema: NodeSchema,
    assets: dict[str, tuple[str, bool]],
) -> NodeSchema:
    inputs = []
    for input_spec in schema.inputs:
        asset = assets.get(input_spec.id)
        if asset is None:
            inputs.append(input_spec)
            continue
        kind, required = asset
        inputs.append(
            replace(
                input_spec,
                type=TypeExpr.concrete("dinkster.asset"),
                required=required,
                default=None,
                widget=AssetWidget(
                    accept=("application/octet-stream",),
                    kind=kind,
                ),
            )
        )
    return replace(schema, inputs=tuple(inputs))


def _with_model3d_ports(
    schema: NodeSchema,
    *,
    inputs: frozenset[str] = frozenset(),
    outputs: frozenset[str] = frozenset(),
) -> NodeSchema:
    model3d = TypeExpr.concrete("dinkster.model3d")
    return replace(
        schema,
        inputs=tuple(
            replace(input_spec, type=model3d) if input_spec.id in inputs else input_spec
            for input_spec in schema.inputs
        ),
        outputs=tuple(
            replace(output_spec, type=model3d) if output_spec.id in outputs else output_spec
            for output_spec in schema.outputs
        ),
    )


def _present(input_id: str) -> ReplacementPredicate:
    return ReplacementPredicate.any_of(
        ReplacementPredicate.input_connected(input_id),
        ReplacementPredicate.value_present(input_id),
    )


def _absent(input_id: str) -> ReplacementPredicate:
    return ReplacementPredicate.not_(_present(input_id))


def _flatten_combo(schema: NodeSchema, combo_id: str, default: str) -> NodeSchema:
    combo = next(item for item in schema.combos if item.id == combo_id)
    child_inputs: dict[str, InputSpec] = {}
    for option in combo.options:
        for item in option.inputs:
            if isinstance(item, InputSpec):
                child_inputs.setdefault(item.id, replace(item, required=False))
    selection = InputSpec(
        combo_id,
        TypeExpr.concrete("core.combo"),
        required=False,
        default=default,
        widget=ComboWidget(options=tuple(option.key for option in combo.options)),
    )
    return replace(
        schema,
        inputs=(*schema.inputs, selection, *child_inputs.values()),
        combos=tuple(item for item in schema.combos if item.id != combo_id),
    )


def _seedvr2_workflow_records() -> list[dict[str, object]]:
    sampler_names = (
        "euler",
        "euler_cfg_pp",
        "euler_ancestral",
        "euler_ancestral_cfg_pp",
        "heun",
        "heunpp2",
        "exp_heun_2_x0",
        "exp_heun_2_x0_sde",
        "dpm_2",
        "dpm_2_ancestral",
        "lms",
        "dpm_fast",
        "dpm_adaptive",
        "dpmpp_2s_ancestral",
        "dpmpp_2s_ancestral_cfg_pp",
        "dpmpp_sde",
        "dpmpp_sde_gpu",
        "dpmpp_2m",
        "dpmpp_2m_cfg_pp",
        "dpmpp_2m_sde",
        "dpmpp_2m_sde_gpu",
        "dpmpp_2m_sde_heun",
        "dpmpp_2m_sde_heun_gpu",
        "dpmpp_3m_sde",
        "dpmpp_3m_sde_gpu",
        "ddpm",
        "lcm",
        "ipndm",
        "ipndm_v",
        "deis",
        "res_multistep",
        "res_multistep_cfg_pp",
        "res_multistep_ancestral",
        "res_multistep_ancestral_cfg_pp",
        "gradient_estimation",
        "gradient_estimation_cfg_pp",
        "er_sde",
        "seeds_2",
        "seeds_3",
        "sa_solver",
        "sa_solver_pece",
        "ddim",
        "uni_pc",
        "uni_pc_bh2",
    )
    schedulers = (
        "simple",
        "sgm_uniform",
        "karras",
        "exponential",
        "ddim_uniform",
        "beta",
        "normal",
        "linear_quadratic",
        "kl_optimal",
    )
    revision = "b78cec87"
    return [
        _record(
            node_class="KSampler",
            carrier="dinkster.ksampler",
            revision=revision,
            rule=ReplacementRule(
                from_type="comfy.KSampler",
                cases=(
                    ReplacementCase.build(
                        "dinkster.ksampler",
                        inputs={
                            **{
                                name: MappingSource.copy(name)
                                for name in (
                                    "model",
                                    "seed",
                                    "steps",
                                    "cfg",
                                    "positive",
                                    "negative",
                                    "latent_image",
                                    "denoise",
                                )
                            },
                            "sampler_name": MappingSource.from_value(
                                "sampler_name",
                                ValueTransform.enum_rename(
                                    {name: f"dinkster.{name}" for name in sampler_names}
                                ),
                            ),
                            "scheduler": MappingSource.from_value(
                                "scheduler",
                                ValueTransform.enum_rename(
                                    {name: f"dinkster.{name}" for name in schedulers}
                                ),
                            ),
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
                note=(
                    "Current native sampler and scheduler ids are translated explicitly; "
                    "unknown future ids fail closed."
                ),
            ),
            evidence=[*SEEDVR2_WORKFLOW_EVIDENCE, *TRELLIS2_WORKFLOW_EVIDENCE],
        ),
        _record(
            node_class="UNETLoader",
            carrier="dinkster.load_diffusion_model",
            revision=revision,
            rule=ReplacementRule(
                from_type="comfy.UNETLoader",
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_diffusion_model",
                        inputs={
                            "diffusion_model": MappingSource.copy("unet_name"),
                            "weight_dtype": MappingSource.copy("weight_dtype"),
                        },
                        outputs={"model": "model"},
                    ),
                ),
            ),
            evidence=[*SEEDVR2_WORKFLOW_EVIDENCE, *TRELLIS2_WORKFLOW_EVIDENCE],
        ),
        *(
            _record(
                node_class=node_class,
                carrier=carrier,
                revision=revision,
                rule=_copy_rule(
                    f"comfy.{node_class}",
                    carrier,
                    *inputs,
                    output_id=target_output,
                    source_output_id=source_output,
                ),
                evidence=SEEDVR2_WORKFLOW_EVIDENCE,
            )
            for node_class, carrier, inputs, target_output, source_output in (
                (
                    "VAEEncodeTiled",
                    "dinkster.vae_encode_tiled",
                    ("pixels", "vae", "tile_size", "overlap", "temporal_size", "temporal_overlap"),
                    "latent",
                    "latent",
                ),
                (
                    "VAEDecodeTiled",
                    "dinkster.vae_decode_tiled",
                    ("samples", "vae", "tile_size", "overlap", "temporal_size", "temporal_overlap"),
                    "image",
                    "image",
                ),
                (
                    "SeedVR2Preprocess",
                    "dinkster.seedvr2_preprocess",
                    ("resized_images",),
                    "images",
                    "images",
                ),
                (
                    "SeedVR2PostProcessing",
                    "dinkster.seedvr2_postprocess",
                    ("images", "original_resized_images", "color_correction_method"),
                    "images",
                    "_0_IMAGE_",
                ),
                (
                    "SeedVR2TemporalMerge",
                    "dinkster.seedvr2_temporal_merge",
                    ("latents", "temporal_overlap"),
                    "latent",
                    "_0_LATENT_",
                ),
            )
        ),
        _record(
            node_class="SeedVR2Conditioning",
            carrier="dinkster.seedvr2_conditioning",
            revision=revision,
            rule=ReplacementRule(
                from_type="comfy.SeedVR2Conditioning",
                cases=(
                    ReplacementCase.build(
                        "dinkster.seedvr2_conditioning",
                        inputs={
                            "model": MappingSource.copy("model"),
                            "vae_conditioning": MappingSource.copy("vae_conditioning"),
                        },
                        outputs={
                            "positive": "_0_CONDITIONING_",
                            "negative": "_1_CONDITIONING_",
                        },
                    ),
                ),
            ),
            evidence=SEEDVR2_WORKFLOW_EVIDENCE,
        ),
        _record(
            node_class="SeedVR2TemporalChunk",
            carrier="dinkster.seedvr2_temporal_chunk",
            revision=revision,
            rule=ReplacementRule(
                from_type="comfy.SeedVR2TemporalChunk",
                cases=(
                    ReplacementCase.build(
                        "dinkster.seedvr2_temporal_chunk",
                        when=ReplacementPredicate.value_equals("chunking_mode", "manual"),
                        slot_variants={"chunking_mode": "manual"},
                        inputs={
                            "latent": MappingSource.copy("latent"),
                            "temporal_overlap": MappingSource.copy("temporal_overlap"),
                            "chunking_mode.frames_per_chunk": MappingSource.copy(
                                "frames_per_chunk"
                            ),
                        },
                        outputs={
                            "latents": "_0_LATENT_",
                            "temporal_overlap": "_1_INT_",
                        },
                    ),
                    ReplacementCase.build(
                        "dinkster.seedvr2_temporal_chunk",
                        slot_variants={"chunking_mode": "auto"},
                        inputs={
                            "latent": MappingSource.copy("latent"),
                            "temporal_overlap": MappingSource.copy("temporal_overlap"),
                        },
                        outputs={
                            "latents": "_0_LATENT_",
                            "temporal_overlap": "_1_INT_",
                        },
                    ),
                ),
            ),
            evidence=SEEDVR2_WORKFLOW_EVIDENCE,
        ),
    ]


def _build_current_seedvr2_registry(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != CURRENT_COMFY_BASELINE:
        raise RuntimeError(f"Current ComfyUI must be pinned to {CURRENT_COMFY_BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("Current ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as _comfy_args  # pyright: ignore[reportMissingImports]

    _comfy_args.cpu = True
    from comfy_extras import nodes_seedvr  # pyright: ignore[reportMissingImports]
    from nodes import (  # pyright: ignore[reportMissingImports]
        KSampler,
        UNETLoader,
        VAEDecodeTiled,
        VAEEncodeTiled,
    )

    def core_schema(node_class: type[Any], name: str) -> Any:
        return translate_node(name, node_class, CompatTranslation()).schema()

    source_schemas = [
        core_schema(KSampler, "KSampler"),
        _with_asset_inputs(
            core_schema(UNETLoader, "UNETLoader"),
            {"unet_name": ("model/diffusion", True)},
        ),
        core_schema(VAEEncodeTiled, "VAEEncodeTiled"),
        core_schema(VAEDecodeTiled, "VAEDecodeTiled"),
        _core_v3_schema(nodes_seedvr.SeedVR2Preprocess),
        _core_v3_schema(nodes_seedvr.SeedVR2PostProcessing),
        _core_v3_schema(nodes_seedvr.SeedVR2Conditioning),
        _flatten_combo(
            _core_v3_schema(nodes_seedvr.SeedVR2TemporalChunk),
            "chunking_mode",
            "auto",
        ),
        _core_v3_schema(nodes_seedvr.SeedVR2TemporalMerge),
    ]
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(schema) for schema in source_schemas],
        "records": _seedvr2_workflow_records(),
    }


def _merge_slot(local_id: str = "") -> str:
    return f"{local_id}:mode" if local_id else "mode"


def _merge_input(index: int, local_id: str = "") -> str:
    path = f"mode.inputs.conditioning_{index}"
    return f"{local_id}:{path}" if local_id else path


def _multi_combine_case(count: int, operation: str, *, fallback: bool = False) -> ReplacementCase:
    nodes = {"count": ReplacementNode.build("dinkster.int", values={"value": count})}
    inputs: dict[str, MappingSource] = {}
    links: list[ReplacementLink] = []
    slot_variants = {"mode": operation}

    if operation == "combine":
        if count <= 8:
            inputs.update(
                {
                    _merge_input(index): MappingSource.copy(f"conditioning_{index}")
                    for index in range(1, count + 1)
                }
            )
        else:
            helper_index = 1
            local_id = f"merge_{helper_index}"
            nodes[local_id] = ReplacementNode.build("dinkster.conditioning_merge")
            slot_variants[_merge_slot(local_id)] = "combine"
            for target_index, source_index in enumerate(range(1, 9), start=1):
                inputs[_merge_input(target_index, local_id)] = MappingSource.copy(
                    f"conditioning_{source_index}"
                )
            consumed = 8
            previous = local_id
            while count - consumed > 7:
                helper_index += 1
                local_id = f"merge_{helper_index}"
                nodes[local_id] = ReplacementNode.build("dinkster.conditioning_merge")
                slot_variants[_merge_slot(local_id)] = "combine"
                links.append(ReplacementLink(f"{previous}:conditioning", _merge_input(1, local_id)))
                for offset in range(1, 8):
                    inputs[_merge_input(offset + 1, local_id)] = MappingSource.copy(
                        f"conditioning_{consumed + offset}"
                    )
                consumed += 7
                previous = local_id
            links.append(ReplacementLink(f"{previous}:conditioning", _merge_input(1)))
            for offset, source_index in enumerate(range(consumed + 1, count + 1), start=2):
                inputs[_merge_input(offset)] = MappingSource.copy(f"conditioning_{source_index}")
    else:
        if count == 2:
            inputs["mode.conditioning_to"] = MappingSource.copy("conditioning_1")
            inputs["mode.conditioning_from"] = MappingSource.copy("conditioning_2")
        else:
            previous = "merge_1"
            nodes[previous] = ReplacementNode.build("dinkster.conditioning_merge")
            slot_variants[_merge_slot(previous)] = "concat"
            inputs[f"{previous}:mode.conditioning_to"] = MappingSource.copy("conditioning_1")
            inputs[f"{previous}:mode.conditioning_from"] = MappingSource.copy("conditioning_2")
            for source_index in range(3, count):
                local_id = f"merge_{source_index - 1}"
                nodes[local_id] = ReplacementNode.build("dinkster.conditioning_merge")
                slot_variants[_merge_slot(local_id)] = "concat"
                links.append(
                    ReplacementLink(f"{previous}:conditioning", f"{local_id}:mode.conditioning_to")
                )
                inputs[f"{local_id}:mode.conditioning_from"] = MappingSource.copy(
                    f"conditioning_{source_index}"
                )
                previous = local_id
            links.append(ReplacementLink(f"{previous}:conditioning", "mode.conditioning_to"))
            inputs["mode.conditioning_from"] = MappingSource.copy(f"conditioning_{count}")

    when = None
    if not fallback:
        when = ReplacementPredicate.all_of(
            ReplacementPredicate.value_equals("inputcount", count),
            ReplacementPredicate.value_equals("operation", operation),
        )
    return ReplacementCase.build(
        "dinkster.conditioning_merge",
        when=when,
        nodes=nodes,
        slot_variants=slot_variants,
        inputs=inputs,
        links=links,
        outputs={"conditioning": "combined", "count:value": "inputcount"},
    )


def _multi_combine_rule() -> ReplacementRule:
    cases = [
        _multi_combine_case(count, operation)
        for count in range(2, 21)
        for operation in ("combine", "concat")
        if (count, operation) != (20, "concat")
    ]
    cases.append(_multi_combine_case(20, "concat", fallback=True))
    return ReplacementRule(
        from_type="comfy.comfyui-kjnodes.ConditioningMultiCombine",
        cases=tuple(cases),
        note=(
            "Counts 2-20 preserve ordered combine or iterative token concatenation "
            "and the count output."
        ),
    )


def _set_mask_and_combine_rule(count: int) -> ReplacementRule:
    nodes: dict[str, ReplacementNode] = {
        "negative_merge": ReplacementNode.build("dinkster.conditioning_merge"),
        "set_cond_area_route": ReplacementNode.build(
            "dinkster.route.gate", values={"condition": True}
        ),
    }
    slot_variants = {
        "mode": "combine",
        "negative_merge:mode": "combine",
    }
    inputs: dict[str, MappingSource] = {
        "set_cond_area_route:value": MappingSource.copy("set_cond_area")
    }
    links: list[ReplacementLink] = []
    for index in range(1, count + 1):
        mask_route = f"mask_{index}_route"
        strength_route = f"mask_{index}_strength_route"
        nodes[mask_route] = ReplacementNode.build("dinkster.route.gate", values={"condition": True})
        nodes[strength_route] = ReplacementNode.build(
            "dinkster.route.gate", values={"condition": True}
        )
        inputs[f"{mask_route}:value"] = MappingSource.copy(f"mask_{index}")
        inputs[f"{strength_route}:value"] = MappingSource.copy(f"mask_{index}_strength")
        for polarity in ("positive", "negative"):
            local_id = f"{polarity}_{index}"
            nodes[local_id] = ReplacementNode.build("dinkster.conditioning_set_mask")
            inputs[f"{local_id}:conditioning"] = MappingSource.copy(f"{polarity}_{index}")
            links.extend(
                (
                    ReplacementLink(f"{mask_route}:value", f"{local_id}:mask"),
                    ReplacementLink(f"{strength_route}:value", f"{local_id}:strength"),
                    ReplacementLink("set_cond_area_route:value", f"{local_id}:set_cond_area"),
                )
            )
            target = _merge_input(index, "negative_merge" if polarity == "negative" else "")
            links.append(ReplacementLink(f"{local_id}:conditioning", target))
    suffix = "" if count == 2 else str(count)
    return ReplacementRule(
        from_type=f"comfy.comfyui-kjnodes.ConditioningSetMaskAndCombine{suffix}",
        cases=(
            ReplacementCase.build(
                "dinkster.conditioning_merge",
                nodes=nodes,
                slot_variants=slot_variants,
                inputs=inputs,
                links=links,
                outputs={
                    "conditioning": "combined_positive",
                    "negative_merge:conditioning": "combined_negative",
                },
            ),
        ),
    )


def _essentials_combine_rule() -> ReplacementRule:
    cases: list[ReplacementCase] = []
    for members in (
        (3, 4, 5),
        (3, 4),
        (3, 5),
        (4, 5),
        (3,),
        (4,),
        (5,),
        (),
    ):
        source_inputs = (1, 2, *members)
        cases.append(
            ReplacementCase.build(
                "dinkster.conditioning_merge",
                when=(
                    ReplacementPredicate.all_of(
                        *(_present(f"conditioning_{index}") for index in members)
                    )
                    if members
                    else None
                ),
                slot_variants={"mode": "combine"},
                inputs={
                    _merge_input(target_index): MappingSource.copy(f"conditioning_{source_index}")
                    for target_index, source_index in enumerate(source_inputs, start=1)
                },
                outputs={"conditioning": "conditioning"},
            )
        )
    return ReplacementRule(
        from_type="comfy.comfyui_essentials.ConditioningCombineMultiple+",
        cases=tuple(cases),
        note="Optional inputs are retained in conditioning_3-through-conditioning_5 order.",
    )


def _sd3_negative_rule() -> ReplacementRule:
    source_type = "comfy.comfyui_essentials.SD3NegativeConditioning+"
    return ReplacementRule(
        from_type=source_type,
        cases=(
            ReplacementCase.build(
                "dinkster.conditioning_zero_out",
                when=ReplacementPredicate.value_equals("end", 0),
                inputs={"conditioning": MappingSource.copy("conditioning")},
                outputs={"conditioning": "conditioning"},
            ),
            ReplacementCase.build(
                "dinkster.conditioning_merge",
                nodes={
                    "zero": ReplacementNode.build("dinkster.conditioning_zero_out"),
                    "head": ReplacementNode.build("dinkster.conditioning_set_timestep_range"),
                    "tail": ReplacementNode.build("dinkster.conditioning_set_timestep_range"),
                    "conditioning_route": ReplacementNode.build(
                        "dinkster.route.gate", values={"condition": True}
                    ),
                    "end_route": ReplacementNode.build(
                        "dinkster.route.gate", values={"condition": True}
                    ),
                },
                slot_variants={"mode": "combine"},
                inputs={
                    "conditioning_route:value": MappingSource.copy("conditioning"),
                    "end_route:value": MappingSource.copy("end"),
                    "head:start": MappingSource.constant(0.0),
                    "tail:end": MappingSource.constant(1.0),
                },
                links=(
                    ReplacementLink("conditioning_route:value", "zero:conditioning"),
                    ReplacementLink("conditioning_route:value", "head:conditioning"),
                    ReplacementLink("zero:conditioning", "tail:conditioning"),
                    ReplacementLink("end_route:value", "head:end"),
                    ReplacementLink("end_route:value", "tail:start"),
                    ReplacementLink("tail:conditioning", _merge_input(1)),
                    ReplacementLink("head:conditioning", _merge_input(2)),
                ),
                outputs={"conditioning": "conditioning"},
            ),
        ),
    )


MEGAPACK_EVIDENCE = [
    "tests/test_generation_comfy_aliases.py::test_conditioning_megapack_aliases_preserve_grouped_graph_shapes"
]


def _megapack_records() -> list[dict[str, object]]:
    records = [
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ConditioningMultiCombine",
            revision=KJ_BASELINE,
            carrier="dinkster.conditioning_merge",
            rule=_multi_combine_rule(),
            tier="grouped",
            evidence=MEGAPACK_EVIDENCE,
        )
    ]
    for count in range(2, 6):
        suffix = "" if count == 2 else str(count)
        records.append(
            _record(
                source_pack="comfyui-kjnodes",
                node_class=f"ConditioningSetMaskAndCombine{suffix}",
                revision=KJ_BASELINE,
                carrier="dinkster.conditioning_merge",
                rule=_set_mask_and_combine_rule(count),
                tier="grouped",
                evidence=MEGAPACK_EVIDENCE,
            )
        )
    records.extend(
        (
            _record(
                source_pack="comfyui_essentials",
                node_class="ConditioningCombineMultiple+",
                revision=ESSENTIALS_BASELINE,
                carrier="dinkster.conditioning_merge",
                rule=_essentials_combine_rule(),
                tier="grouped",
                evidence=MEGAPACK_EVIDENCE,
            ),
            _record(
                source_pack="comfyui_essentials",
                node_class="SD3NegativeConditioning+",
                revision=ESSENTIALS_BASELINE,
                carrier="dinkster.conditioning_merge",
                rule=_sd3_negative_rule(),
                tier="grouped",
                evidence=MEGAPACK_EVIDENCE,
            ),
        )
    )
    return records


def _loader_source_schemas() -> list[NodeSchema]:
    asset = ["model.safetensors"]
    optional_lora = (["None", "lora.safetensors"], {"default": "None"})
    weight_dtypes = [
        "default",
        "fp8_e4m3fn",
        "fp8_e4m3fn_fast",
        "fp8_e5m2",
        "fp16",
        "bf16",
        "fp32",
    ]
    compute_dtypes = ["default", "fp16", "bf16", "fp32"]
    sage_modes = [
        "disabled",
        "auto",
        "sageattn_qk_int8_pv_fp16_cuda",
        "sageattn_qk_int8_pv_fp16_triton",
        "sageattn_qk_int8_pv_fp8_cuda",
        "sageattn_qk_int8_pv_fp8_cuda++",
        "sageattn3",
        "sageattn3_per_block_mean",
    ]

    schemas = [
        _with_asset_inputs(
            _static_schema(
                "CheckpointLoaderKJ",
                namespace="comfyui-kjnodes",
                required={
                    "ckpt_name": (asset,),
                    "weight_dtype": (weight_dtypes,),
                    "compute_dtype": (compute_dtypes, {"default": "default"}),
                    "patch_cublaslinear": ("BOOLEAN", {"default": False}),
                    "sage_attention": (sage_modes, {"default": False}),
                    "enable_fp16_accumulation": ("BOOLEAN", {"default": False}),
                },
                returns=("MODEL", "CLIP", "VAE"),
            ),
            {"ckpt_name": ("model/checkpoint", True)},
        ),
        _with_asset_inputs(
            _static_schema(
                "DiffusionModelLoaderKJ",
                namespace="comfyui-kjnodes",
                required={
                    "model_name": (asset,),
                    "weight_dtype": (weight_dtypes,),
                    "compute_dtype": (compute_dtypes, {"default": "default"}),
                    "patch_cublaslinear": ("BOOLEAN", {"default": False}),
                    "sage_attention": (sage_modes, {"default": False}),
                    "enable_fp16_accumulation": ("BOOLEAN", {"default": False}),
                },
                optional={
                    "extra_state_dict": ("STRING", {"forceInput": True}),
                },
                returns=("MODEL",),
            ),
            {"model_name": ("model/diffusion", True)},
        ),
    ]

    rgthree_required: dict[str, object] = {
        "model": ("MODEL",),
        "clip": ("CLIP",),
    }
    for index in range(1, 5):
        rgthree_required[f"lora_{index:02}"] = optional_lora
        rgthree_required[f"strength_{index:02}"] = (
            "FLOAT",
            {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
        )
    schemas.append(
        _with_asset_inputs(
            _static_schema(
                "Lora Loader Stack (rgthree)",
                namespace="rgthree-comfy",
                required=rgthree_required,
                returns=("MODEL", "CLIP"),
            ),
            {f"lora_{index:02}": ("model/lora", False) for index in range(1, 5)},
        )
    )

    full_required: dict[str, object] = {
        "ckpt_name": (asset,),
        "config_name": (["Default"], {"default": "Default"}),
        "vae_name": (["Baked VAE"], {"default": "Baked VAE"}),
        "clip_skip": ("INT", {"default": -2, "min": -24, "max": -1}),
        "lora_name": optional_lora,
        "lora_model_strength": (
            "FLOAT",
            {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
        ),
        "lora_clip_strength": (
            "FLOAT",
            {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
        ),
        "resolution": ("STRING", {"default": "512 x 512"}),
        "empty_latent_width": ("INT", {"default": 512, "min": 64, "step": 64}),
        "empty_latent_height": ("INT", {"default": 512, "min": 64, "step": 64}),
        "positive": ("STRING", {"default": "", "multiline": True}),
        "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
        "positive_weight_interpretation": (
            ["comfy", "A1111", "comfy++", "compel", "fixed attention"],
        ),
        "negative": ("STRING", {"default": "", "multiline": True}),
        "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
        "negative_weight_interpretation": (
            ["comfy", "A1111", "comfy++", "compel", "fixed attention"],
        ),
        "batch_size": ("INT", {"default": 1, "min": 1}),
    }
    full_optional = {
        "model_override": ("MODEL",),
        "clip_override": ("CLIP",),
        "vae_override": ("VAE",),
        "optional_lora_stack": ("LORA_STACK",),
        "optional_controlnet_stack": ("CONTROL_NET_STACK",),
        "a1111_prompt_style": ("BOOLEAN", {"default": False}),
    }
    schemas.append(
        _with_asset_inputs(
            _static_schema(
                "easy fullLoader",
                namespace="comfyui-easy-use",
                required=full_required,
                optional=full_optional,
                returns=(
                    "PIPE_LINE",
                    "MODEL",
                    "VAE",
                    "CLIP",
                    "CONDITIONING",
                    "CONDITIONING",
                    "LATENT",
                ),
                return_names=("pipe", "model", "vae", "clip", "positive", "negative", "latent"),
            ),
            {
                "ckpt_name": ("model/checkpoint", True),
                "lora_name": ("model/lora", False),
            },
        )
    )

    standard_required: dict[str, object] = {
        "ckpt_name": (asset,),
        "vae_name": (["Baked VAE"], {"default": "Baked VAE"}),
        "clip_skip": ("INT", {"default": -2, "min": -24, "max": -1}),
        "lora_name": optional_lora,
        "lora_model_strength": (
            "FLOAT",
            {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
        ),
        "lora_clip_strength": (
            "FLOAT",
            {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
        ),
        "resolution": ("STRING", {"default": "512 x 512"}),
        "empty_latent_width": ("INT", {"default": 512, "min": 64, "step": 64}),
        "empty_latent_height": ("INT", {"default": 512, "min": 64, "step": 64}),
        "positive": ("STRING", {"default": "", "multiline": True}),
        "negative": ("STRING", {"default": "", "multiline": True}),
        "batch_size": ("INT", {"default": 1, "min": 1}),
    }
    for node_class, extra_optional in (
        (
            "easy a1111Loader",
            {"a1111_prompt_style": ("BOOLEAN", {"default": False})},
        ),
        ("easy comfyLoader", {}),
    ):
        schemas.append(
            _with_asset_inputs(
                _static_schema(
                    node_class,
                    namespace="comfyui-easy-use",
                    required=standard_required,
                    optional={
                        "optional_lora_stack": ("LORA_STACK",),
                        "optional_controlnet_stack": ("CONTROL_NET_STACK",),
                        **extra_optional,
                    },
                    returns=("PIPE_LINE", "MODEL", "VAE"),
                    return_names=("pipe", "model", "vae"),
                ),
                {
                    "ckpt_name": ("model/checkpoint", True),
                    "lora_name": ("model/lora", False),
                },
            )
        )

    for node_class, has_negative, has_overrides in (
        ("easy fluxLoader", False, True),
        ("easy hunyuanDiTLoader", True, False),
    ):
        required: dict[str, object] = {
            "ckpt_name": (asset,),
            "vae_name": (["Baked VAE"], {"default": "Baked VAE"}),
            "lora_name": optional_lora,
            "lora_model_strength": (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
            ),
            "lora_clip_strength": (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
            ),
            "resolution": ("STRING", {"default": "1024 x 1024"}),
            "empty_latent_width": ("INT", {"default": 1024, "min": 64, "step": 64}),
            "empty_latent_height": ("INT", {"default": 1024, "min": 64, "step": 64}),
            "positive": ("STRING", {"default": "", "multiline": True}),
        }
        if has_negative:
            required["negative"] = ("STRING", {"default": "", "multiline": True})
        required["batch_size"] = ("INT", {"default": 1, "min": 1})
        optional: dict[str, object] = {
            "optional_lora_stack": ("LORA_STACK",),
            "optional_controlnet_stack": ("CONTROL_NET_STACK",),
        }
        if has_overrides:
            optional = {
                "model_override": ("MODEL",),
                "clip_override": ("CLIP",),
                "vae_override": ("VAE",),
                **optional,
            }
        schemas.append(
            _with_asset_inputs(
                _static_schema(
                    node_class,
                    namespace="comfyui-easy-use",
                    required=required,
                    optional=optional,
                    returns=("PIPE_LINE", "MODEL", "VAE"),
                    return_names=("pipe", "model", "vae"),
                ),
                {
                    "ckpt_name": ("model/checkpoint", True),
                    "lora_name": ("model/lora", False),
                },
            )
        )

    schemas.extend(
        (
            _with_asset_inputs(
                _static_schema(
                    "Checkpoint Loader (Simple)",
                    namespace="was-node-suite-comfyui",
                    required={"ckpt_name": (asset,)},
                    returns=("MODEL", "CLIP", "VAE", "STRING"),
                    return_names=("MODEL", "CLIP", "VAE", "NAME_STRING"),
                ),
                {"ckpt_name": ("model/checkpoint", True)},
            ),
            *(
                _with_asset_inputs(
                    _static_schema(
                        node_class,
                        namespace="was-node-suite-comfyui",
                        required={
                            "model": ("MODEL",),
                            "clip": ("CLIP",),
                            "lora_name": optional_lora,
                            "strength_model": (
                                "FLOAT",
                                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                            ),
                            "strength_clip": (
                                "FLOAT",
                                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                            ),
                        },
                        returns=("MODEL", "CLIP", "STRING"),
                        return_names=("MODEL", "CLIP", "NAME_STRING"),
                    ),
                    {"lora_name": ("model/lora", False)},
                )
                for node_class in ("Load Lora", "Lora Loader")
            ),
            _with_asset_inputs(
                _static_schema(
                    "LoraLoader|pysssss",
                    namespace="comfyui-custom-scripts",
                    required={
                        "model": ("MODEL",),
                        "clip": ("CLIP",),
                        "lora_name": (asset,),
                        "strength_model": (
                            "FLOAT",
                            {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                        ),
                        "strength_clip": (
                            "FLOAT",
                            {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                        ),
                    },
                    optional={"prompt": ("STRING", {"default": "", "multiline": True})},
                    returns=("MODEL", "CLIP", "STRING"),
                    return_names=("MODEL", "CLIP", "example"),
                ),
                {"lora_name": ("model/lora", True)},
            ),
            _with_asset_inputs(
                _static_schema(
                    "CheckpointLoader|pysssss",
                    namespace="comfyui-custom-scripts",
                    required={"ckpt_name": (asset,)},
                    optional={"prompt": ("STRING", {"default": "", "multiline": True})},
                    returns=("MODEL", "CLIP", "VAE", "STRING"),
                    return_names=("MODEL", "CLIP", "VAE", "example"),
                ),
                {"ckpt_name": ("model/checkpoint", True)},
            ),
            _with_asset_inputs(
                _static_schema(
                    "Efficient Loader",
                    namespace="efficiency-nodes-comfyui",
                    required={
                        "ckpt_name": (asset,),
                        "vae_name": (["Baked VAE"], {"default": "Baked VAE"}),
                        "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1}),
                        "lora_name": optional_lora,
                        "lora_model_strength": (
                            "FLOAT",
                            {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                        ),
                        "lora_clip_strength": (
                            "FLOAT",
                            {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                        ),
                        "positive": (
                            "STRING",
                            {"default": "CLIP_POSITIVE", "multiline": True},
                        ),
                        "negative": (
                            "STRING",
                            {"default": "CLIP_NEGATIVE", "multiline": True},
                        ),
                        "token_normalization": (["none", "mean", "length", "length+mean"],),
                        "weight_interpretation": (
                            ["comfy", "A1111", "compel", "comfy++", "down_weight"],
                        ),
                        "empty_latent_width": ("INT", {"default": 512, "min": 64, "step": 64}),
                        "empty_latent_height": (
                            "INT",
                            {"default": 512, "min": 64, "step": 64},
                        ),
                        "batch_size": ("INT", {"default": 1, "min": 1}),
                    },
                    optional={
                        "lora_stack": ("LORA_STACK",),
                        "cnet_stack": ("CONTROL_NET_STACK",),
                    },
                    returns=(
                        "MODEL",
                        "CONDITIONING",
                        "CONDITIONING",
                        "LATENT",
                        "VAE",
                        "CLIP",
                        "DEPENDENCIES",
                    ),
                    return_names=(
                        "MODEL",
                        "CONDITIONING+",
                        "CONDITIONING-",
                        "LATENT",
                        "VAE",
                        "CLIP",
                        "DEPENDENCIES",
                    ),
                ),
                {
                    "ckpt_name": ("model/checkpoint", True),
                    "lora_name": ("model/lora", False),
                },
            ),
        )
    )
    return schemas


def _checkpoint_stack_rule(
    source_type: str,
    guards: tuple[ReplacementPredicate, ...],
    outputs: dict[str, str],
    *,
    clip_skip: bool,
) -> ReplacementRule:
    inputs = {"checkpoint": MappingSource.copy("ckpt_name")}
    if clip_skip:
        inputs["stop_at_clip_layer"] = MappingSource.copy("clip_skip")
    no_lora = ReplacementCase.build(
        "dinkster.load_checkpoint_stack",
        when=ReplacementPredicate.all_of(
            *guards,
            ReplacementPredicate.value_equals("lora_name", "None"),
        ),
        inputs=inputs,
        outputs=outputs,
    )
    with_lora = ReplacementCase.build(
        "dinkster.load_checkpoint_stack",
        when=ReplacementPredicate.all_of(
            *guards,
            _present("lora_name"),
            ReplacementPredicate.not_(ReplacementPredicate.value_equals("lora_name", "None")),
        ),
        inputs=inputs,
        input_families={
            "loras": InputFamilyMapping.from_members(
                InputFamilyMember.build(
                    "inline",
                    inputs={
                        "lora": MappingSource.copy("lora_name"),
                        "strength_model": MappingSource.copy("lora_model_strength"),
                        "strength_clip": MappingSource.copy("lora_clip_strength"),
                    },
                )
            )
        },
        outputs=outputs,
    )
    return ReplacementRule(
        from_type=source_type,
        cases=(
            no_lora,
            with_lora,
            _refusal_case(
                "dinkster.load_checkpoint_stack",
                target_input="execution_mode",
                source_input="vae_name",
            ),
        ),
        note=(
            "Only the checkpoint-backed VAE path with no overrides, prior LoRA stack, "
            "ControlNet stack, or prompt-embedded LoRA is automatic."
        ),
    )


def _refusal_case(
    target_type: str,
    *,
    target_input: str,
    source_input: str,
) -> ReplacementCase:
    """Build a case whose enum transform always rejects before mutation."""
    return ReplacementCase.build(
        target_type,
        inputs={
            target_input: MappingSource.from_value(
                source_input,
                ValueTransform.enum_rename({}),
            )
        },
    )


def _rgthree_refusal_case(
    when: ReplacementPredicate | None = None,
) -> ReplacementCase:
    return ReplacementCase.build(
        "dinkster.apply_lora_stack",
        when=when,
        input_families={
            "loras": InputFamilyMapping.from_members(
                InputFamilyMember.build(
                    "blocked",
                    inputs={
                        "lora": MappingSource.from_value(
                            "lora_01",
                            ValueTransform.enum_rename({}),
                        )
                    },
                )
            )
        },
    )


def _rgthree_slot_active(index: int) -> ReplacementPredicate:
    lora = f"lora_{index:02}"
    strength = f"strength_{index:02}"
    return ReplacementPredicate.all_of(
        _present(lora),
        ReplacementPredicate.not_(ReplacementPredicate.value_equals(lora, "None")),
        ReplacementPredicate.not_(ReplacementPredicate.input_connected(strength)),
        ReplacementPredicate.not_(ReplacementPredicate.value_equals(strength, 0.0)),
    )


def _rgthree_slot_inactive(index: int) -> ReplacementPredicate:
    lora = f"lora_{index:02}"
    strength = f"strength_{index:02}"
    return ReplacementPredicate.any_of(
        ReplacementPredicate.value_equals(lora, "None"),
        ReplacementPredicate.all_of(
            ReplacementPredicate.not_(ReplacementPredicate.input_connected(strength)),
            ReplacementPredicate.value_equals(strength, 0.0),
        ),
    )


def _loader_records() -> list[dict[str, object]]:
    records = [
        _record(
            source_pack="comfyui-kjnodes",
            node_class="CheckpointLoaderKJ",
            revision=LOADER_KJ_BASELINE,
            carrier="dinkster.load_checkpoint",
            tier="grouped",
            evidence=LOADER_STACK_EVIDENCE,
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.CheckpointLoaderKJ",
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_checkpoint",
                        when=ReplacementPredicate.all_of(
                            ReplacementPredicate.value_equals("weight_dtype", "default"),
                            ReplacementPredicate.value_equals("compute_dtype", "default"),
                            ReplacementPredicate.value_equals("patch_cublaslinear", False),
                            ReplacementPredicate.value_equals("sage_attention", "disabled"),
                            ReplacementPredicate.value_equals("enable_fp16_accumulation", False),
                        ),
                        inputs={"checkpoint": MappingSource.copy("ckpt_name")},
                        outputs={"model": "model", "clip": "clip", "vae": "vae"},
                    ),
                    _refusal_case(
                        "dinkster.load_checkpoint",
                        target_input="checkpoint",
                        source_input="ckpt_name",
                    ),
                ),
            ),
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="DiffusionModelLoaderKJ",
            revision=LOADER_KJ_BASELINE,
            carrier="dinkster.load_diffusion_model",
            tier="grouped",
            evidence=LOADER_STACK_EVIDENCE,
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.DiffusionModelLoaderKJ",
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_diffusion_model",
                        when=ReplacementPredicate.all_of(
                            ReplacementPredicate.value_equals("weight_dtype", "default"),
                            ReplacementPredicate.value_equals("compute_dtype", "default"),
                            ReplacementPredicate.value_equals("patch_cublaslinear", False),
                            ReplacementPredicate.value_equals("sage_attention", "disabled"),
                            ReplacementPredicate.value_equals("enable_fp16_accumulation", False),
                            _absent("extra_state_dict"),
                        ),
                        inputs={"diffusion_model": MappingSource.copy("model_name")},
                        outputs={"model": "model"},
                    ),
                    _refusal_case(
                        "dinkster.load_diffusion_model",
                        target_input="weight_dtype",
                        source_input="compute_dtype",
                    ),
                ),
            ),
        ),
    ]

    rgthree_cases = [
        _rgthree_refusal_case(
            ReplacementPredicate.all_of(*(_rgthree_slot_inactive(index) for index in range(1, 5)))
        )
    ]
    for mask in range(1, 16):
        active = tuple(index for index in range(1, 5) if mask & (1 << (index - 1)))
        rgthree_cases.append(
            ReplacementCase.build(
                "dinkster.apply_lora_stack",
                when=ReplacementPredicate.all_of(
                    *(
                        _rgthree_slot_active(index)
                        if index in active
                        else _rgthree_slot_inactive(index)
                        for index in range(1, 5)
                    )
                ),
                inputs={
                    "model": MappingSource.copy("model"),
                    "clip": MappingSource.copy("clip"),
                },
                input_families={
                    "loras": InputFamilyMapping.from_members(
                        *(
                            InputFamilyMember.build(
                                f"lora_{index:02}",
                                inputs={
                                    "lora": MappingSource.copy(f"lora_{index:02}"),
                                    "strength_model": MappingSource.from_value(
                                        f"strength_{index:02}"
                                    ),
                                    "strength_clip": MappingSource.from_value(
                                        f"strength_{index:02}"
                                    ),
                                },
                            )
                            for index in active
                        )
                    )
                },
                outputs={"model": "model", "clip": "clip"},
            )
        )
    rgthree_cases.append(_rgthree_refusal_case())
    records.append(
        _record(
            source_pack="rgthree-comfy",
            node_class="Lora Loader Stack (rgthree)",
            revision=RGTHREE_BASELINE,
            carrier="dinkster.apply_lora_stack",
            tier="grouped",
            evidence=LOADER_STACK_EVIDENCE,
            rule=ReplacementRule(
                from_type="comfy.rgthree-comfy.Lora Loader Stack (rgthree)",
                cases=tuple(rgthree_cases),
                note="The all-None no-op stays unchanged; populated slots preserve source order.",
            ),
        )
    )

    common_easy_guards = (
        ReplacementPredicate.value_equals("vae_name", "Baked VAE"),
        _absent("optional_lora_stack"),
        _absent("optional_controlnet_stack"),
    )
    easy_rules = (
        (
            "easy fullLoader",
            (
                *common_easy_guards,
                ReplacementPredicate.not_(ReplacementPredicate.value_equals("ckpt_name", "None")),
                ReplacementPredicate.value_equals("config_name", "Default"),
                ReplacementPredicate.value_equals("positive", ""),
                ReplacementPredicate.value_equals("negative", ""),
                _absent("model_override"),
                _absent("clip_override"),
                _absent("vae_override"),
            ),
            {"model": "model", "clip": "clip", "vae": "vae"},
            True,
        ),
        *(
            (
                node_class,
                (
                    *common_easy_guards,
                    ReplacementPredicate.value_equals("positive", ""),
                    ReplacementPredicate.value_equals("negative", ""),
                ),
                {"model": "model", "vae": "vae"},
                False,
            )
            for node_class in ("easy a1111Loader", "easy comfyLoader")
        ),
        (
            "easy fluxLoader",
            (
                *common_easy_guards,
                ReplacementPredicate.not_(ReplacementPredicate.value_equals("ckpt_name", "None")),
                ReplacementPredicate.value_equals("positive", ""),
                _absent("model_override"),
                _absent("clip_override"),
                _absent("vae_override"),
            ),
            {"model": "model", "vae": "vae"},
            False,
        ),
        (
            "easy hunyuanDiTLoader",
            (
                *common_easy_guards,
                ReplacementPredicate.value_equals("positive", ""),
                ReplacementPredicate.value_equals("negative", ""),
            ),
            {"model": "model", "vae": "vae"},
            False,
        ),
    )
    for node_class, guards, outputs, clip_skip in easy_rules:
        records.append(
            _record(
                source_pack="comfyui-easy-use",
                node_class=node_class,
                revision=EASY_USE_BASELINE,
                carrier="dinkster.load_checkpoint_stack",
                tier="grouped",
                evidence=LOADER_STACK_EVIDENCE,
                rule=_checkpoint_stack_rule(
                    f"comfy.comfyui-easy-use.{node_class}",
                    guards,
                    outputs,
                    clip_skip=clip_skip,
                ),
            )
        )

    records.append(
        _record(
            source_pack="was-node-suite-comfyui",
            node_class="Checkpoint Loader (Simple)",
            revision=WAS_BASELINE,
            carrier="dinkster.load_checkpoint",
            tier="grouped",
            evidence=LOADER_STACK_EVIDENCE,
            rule=ReplacementRule(
                from_type="comfy.was-node-suite-comfyui.Checkpoint Loader (Simple)",
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_checkpoint",
                        inputs={"checkpoint": MappingSource.copy("ckpt_name")},
                        outputs={"model": "MODEL", "clip": "CLIP", "vae": "VAE"},
                    ),
                ),
                note="The checkpoint basename output remains review-required when connected.",
            ),
        )
    )
    for node_class in ("Load Lora", "Lora Loader"):
        records.append(
            _record(
                source_pack="was-node-suite-comfyui",
                node_class=node_class,
                revision=WAS_BASELINE,
                carrier="dinkster.load_lora",
                tier="grouped",
                evidence=LOADER_STACK_EVIDENCE,
                rule=ReplacementRule(
                    from_type=f"comfy.was-node-suite-comfyui.{node_class}",
                    cases=(
                        ReplacementCase.build(
                            "dinkster.load_lora",
                            when=ReplacementPredicate.all_of(
                                _present("lora_name"),
                                ReplacementPredicate.not_(
                                    ReplacementPredicate.value_equals("lora_name", "None")
                                ),
                            ),
                            inputs={
                                "model": MappingSource.copy("model"),
                                "clip": MappingSource.copy("clip"),
                                "lora": MappingSource.copy("lora_name"),
                                "strength_model": MappingSource.copy("strength_model"),
                                "strength_clip": MappingSource.copy("strength_clip"),
                            },
                            outputs={"model": "MODEL", "clip": "CLIP"},
                        ),
                        _refusal_case(
                            "dinkster.load_lora",
                            target_input="lora",
                            source_input="lora_name",
                        ),
                    ),
                    note=(
                        "The source None sentinel is not executable; the name output is "
                        "review-required."
                    ),
                ),
            )
        )

    for node_class, carrier, source_outputs, target_inputs in (
        (
            "LoraLoader|pysssss",
            "dinkster.load_lora",
            {"model": "MODEL", "clip": "CLIP", "example:value": "example"},
            {
                "model": "model",
                "clip": "clip",
                "lora": "lora_name",
                "strength_model": "strength_model",
                "strength_clip": "strength_clip",
            },
        ),
        (
            "CheckpointLoader|pysssss",
            "dinkster.load_checkpoint",
            {
                "model": "MODEL",
                "clip": "CLIP",
                "vae": "VAE",
                "example:value": "example",
            },
            {"checkpoint": "ckpt_name"},
        ),
    ):
        records.append(
            _record(
                source_pack="comfyui-custom-scripts",
                node_class=node_class,
                revision=PYSSSSS_BASELINE,
                carrier=carrier,
                tier="grouped",
                evidence=LOADER_STACK_EVIDENCE,
                rule=ReplacementRule(
                    from_type=f"comfy.comfyui-custom-scripts.{node_class}",
                    cases=(
                        ReplacementCase.build(
                            carrier,
                            nodes={"example": ReplacementNode.build("dinkster.string")},
                            inputs={
                                **{
                                    target: MappingSource.copy(source)
                                    for target, source in target_inputs.items()
                                },
                                "example:value": MappingSource.copy("prompt"),
                            },
                            outputs=source_outputs,
                        ),
                    ),
                ),
            )
        )

    records.append(
        _record(
            source_pack="efficiency-nodes-comfyui",
            node_class="Efficient Loader",
            revision=EFFICIENCY_BASELINE,
            carrier="dinkster.load_checkpoint_stack",
            tier="grouped",
            evidence=LOADER_STACK_EVIDENCE,
            rule=_checkpoint_stack_rule(
                "comfy.efficiency-nodes-comfyui.Efficient Loader",
                (
                    ReplacementPredicate.value_equals("vae_name", "Baked VAE"),
                    ReplacementPredicate.value_equals("positive", "CLIP_POSITIVE"),
                    ReplacementPredicate.value_equals("negative", "CLIP_NEGATIVE"),
                    _absent("lora_stack"),
                    _absent("cnet_stack"),
                ),
                {"model": "MODEL", "vae": "VAE", "clip": "CLIP"},
                clip_skip=True,
            ),
        )
    )
    return records


def _noise_records() -> list[dict[str, object]]:
    return [
        _record(
            source_pack="comfyui-kjnodes",
            node_class="GenerateNoise",
            revision=LOADER_KJ_BASELINE,
            carrier="dinkster.latent.generate_noise",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.GenerateNoise",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.generate_noise",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in (
                                "width",
                                "height",
                                "batch_size",
                                "seed",
                                "multiplier",
                                "constant_batch_noise",
                                "normalize",
                                "model",
                                "sigmas",
                                "latent_channels",
                                "shape",
                            )
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=GENERATE_NOISE_PARITY_EVIDENCE,
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="InjectNoiseToLatent",
            revision=LOADER_KJ_BASELINE,
            carrier="dinkster.latent.inject_noise",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.InjectNoiseToLatent",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.inject_noise",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in (
                                "latents",
                                "strength",
                                "noise",
                                "normalize",
                                "average",
                                "mask",
                                "mix_randn_amount",
                                "seed",
                            )
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=INJECT_NOISE_PARITY_EVIDENCE,
        ),
    ]


def _latent_records() -> list[dict[str, object]]:
    return [
        _record(
            node_class="LatentAdd",
            carrier="dinkster.latent.combine",
            rule=ReplacementRule(
                from_type="comfy.LatentAdd",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.combine",
                        inputs={
                            "samples1": MappingSource.copy("samples1"),
                            "samples2": MappingSource.copy("samples2"),
                            "operation": MappingSource.constant("add"),
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentSubtract",
            carrier="dinkster.latent.combine",
            rule=ReplacementRule(
                from_type="comfy.LatentSubtract",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.combine",
                        inputs={
                            "samples1": MappingSource.copy("samples1"),
                            "samples2": MappingSource.copy("samples2"),
                            "operation": MappingSource.constant("subtract"),
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentInterpolate",
            carrier="dinkster.latent.mix",
            rule=ReplacementRule(
                from_type="comfy.LatentInterpolate",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.mix",
                        inputs={
                            "samples1": MappingSource.copy("samples1"),
                            "samples2": MappingSource.copy("samples2"),
                            "operation": MappingSource.constant("interpolate"),
                            "factor": MappingSource.copy("ratio"),
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentBlend",
            carrier="dinkster.latent.mix",
            rule=ReplacementRule(
                from_type="comfy.LatentBlend",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.mix",
                        inputs={
                            "samples1": MappingSource.copy("samples1"),
                            "samples2": MappingSource.copy("samples2"),
                            "operation": MappingSource.constant("blend"),
                            "factor": MappingSource.copy("blend_factor"),
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentMultiply",
            carrier="dinkster.latent.multiply",
            rule=ReplacementRule(
                from_type="comfy.LatentMultiply",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.multiply",
                        inputs={
                            "samples": MappingSource.copy("samples"),
                            "multiplier": MappingSource.copy("multiplier"),
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentRotate",
            carrier="dinkster.latent.rotate",
            rule=ReplacementRule(
                from_type="comfy.LatentRotate",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.rotate",
                        inputs={
                            "samples": MappingSource.copy("samples"),
                            "angle": MappingSource.from_value(
                                "rotation",
                                ValueTransform.enum_rename(
                                    {
                                        "none": "none",
                                        "90 degrees": "90",
                                        "180 degrees": "180",
                                        "270 degrees": "270",
                                    }
                                ),
                            ),
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentFlip",
            carrier="dinkster.latent.flip",
            rule=ReplacementRule(
                from_type="comfy.LatentFlip",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.flip",
                        inputs={
                            "samples": MappingSource.copy("samples"),
                            "axis": MappingSource.from_value(
                                "flip_method",
                                ValueTransform.enum_rename(
                                    {
                                        "x-axis: vertically": "vertical",
                                        "y-axis: horizontally": "horizontal",
                                    }
                                ),
                            ),
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentCrop",
            carrier="dinkster.latent.crop",
            rule=ReplacementRule(
                from_type="comfy.LatentCrop",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.crop",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in ("samples", "width", "height", "x", "y")
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentUpscale",
            carrier="dinkster.latent.resize",
            rule=ReplacementRule(
                from_type="comfy.LatentUpscale",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.resize",
                        inputs={
                            "samples": MappingSource.copy("samples"),
                            "method": MappingSource.copy("upscale_method"),
                            "width": MappingSource.copy("width"),
                            "height": MappingSource.copy("height"),
                            "crop": MappingSource.copy("crop"),
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentUpscaleBy",
            carrier="dinkster.latent.resize_by",
            rule=ReplacementRule(
                from_type="comfy.LatentUpscaleBy",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.resize_by",
                        inputs={
                            "samples": MappingSource.copy("samples"),
                            "method": MappingSource.copy("upscale_method"),
                            "scale_by": MappingSource.copy("scale_by"),
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentComposite",
            carrier="dinkster.latent.composite",
            rule=ReplacementRule(
                from_type="comfy.LatentComposite",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.composite",
                        inputs={
                            "destination": MappingSource.copy("samples_to"),
                            "source": MappingSource.copy("samples_from"),
                            "x": MappingSource.copy("x"),
                            "y": MappingSource.copy("y"),
                            "feather": MappingSource.copy("feather"),
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentCompositeMasked",
            carrier="dinkster.latent.composite_masked",
            rule=ReplacementRule(
                from_type="comfy.LatentCompositeMasked",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.composite_masked",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in (
                                "destination",
                                "source",
                                "x",
                                "y",
                                "resize_source",
                                "mask",
                            )
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentConcat",
            carrier="dinkster.latent.concat",
            rule=ReplacementRule(
                from_type="comfy.LatentConcat",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.concat",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in ("samples1", "samples2", "dim")
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentCut",
            carrier="dinkster.latent.cut",
            rule=ReplacementRule(
                from_type="comfy.LatentCut",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.cut",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in ("samples", "dim", "index", "amount")
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentCutToBatch",
            carrier="dinkster.latent.cut_to_batch",
            rule=ReplacementRule(
                from_type="comfy.LatentCutToBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.cut_to_batch",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in ("samples", "dim", "slice_size")
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentFromBatch",
            carrier="dinkster.latent.from_batch",
            rule=ReplacementRule(
                from_type="comfy.LatentFromBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.from_batch",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in ("samples", "batch_index", "length")
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_BATCH_PARITY_EVIDENCE,
        ),
        _record(
            node_class="RepeatLatentBatch",
            carrier="dinkster.latent.repeat",
            rule=ReplacementRule(
                from_type="comfy.RepeatLatentBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.repeat",
                        inputs={
                            "samples": MappingSource.copy("samples"),
                            "amount": MappingSource.copy("amount"),
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_BATCH_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentBatchSeedBehavior",
            carrier="dinkster.latent.seed_behavior",
            rule=ReplacementRule(
                from_type="comfy.LatentBatchSeedBehavior",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.seed_behavior",
                        inputs={
                            "samples": MappingSource.copy("samples"),
                            "behavior": MappingSource.copy("seed_behavior"),
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_BATCH_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentBatch",
            carrier="dinkster.latent.batch",
            rule=ReplacementRule(
                from_type="comfy.LatentBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.batch",
                        input_families={
                            "latents": InputFamilyMapping.from_members(
                                InputFamilyMember.build(
                                    "1",
                                    inputs={"value": MappingSource.copy("samples1")},
                                ),
                                InputFamilyMember.build(
                                    "2",
                                    inputs={"value": MappingSource.copy("samples2")},
                                ),
                            )
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_BATCH_PARITY_EVIDENCE,
        ),
        _record(
            node_class="BatchLatentsNode",
            carrier="dinkster.latent.batch",
            rule=ReplacementRule(
                from_type="comfy.BatchLatentsNode",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.batch",
                        input_families={
                            "latents": InputFamilyMapping.copy(
                                "latents", inputs={"value": MappingSource.copy("latent")}
                            )
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_BATCH_PARITY_EVIDENCE,
        ),
        _record(
            node_class="RebatchLatents",
            carrier="dinkster.latent.rebatch",
            rule=ReplacementRule(
                from_type="comfy.RebatchLatents",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.rebatch",
                        nodes={
                            "batch_size": ReplacementNode.build(
                                "std.list.element", values={"index": 0}
                            )
                        },
                        inputs={
                            "latents": MappingSource.copy("latents"),
                            "batch_size:list": MappingSource.copy("batch_size"),
                        },
                        links=(ReplacementLink("batch_size:item", "batch_size"),),
                        outputs={"latents": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_BATCH_PARITY_EVIDENCE,
        ),
        _record(
            node_class="SetLatentNoiseMask",
            carrier="dinkster.latent.set_noise_mask",
            rule=ReplacementRule(
                from_type="comfy.SetLatentNoiseMask",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.set_noise_mask",
                        inputs={
                            "samples": MappingSource.copy("samples"),
                            "mask": MappingSource.copy("mask"),
                        },
                        outputs={"latent": "latent"},
                    ),
                ),
            ),
            evidence=LATENT_BATCH_PARITY_EVIDENCE,
        ),
        _record(
            node_class="ReplaceVideoLatentFrames",
            carrier="dinkster.latent.replace_frames",
            rule=ReplacementRule(
                from_type="comfy.ReplaceVideoLatentFrames",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.replace_frames",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in ("destination", "source", "index")
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_BATCH_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentApplyOperation",
            carrier="dinkster.latent.apply_operation",
            rule=ReplacementRule(
                from_type="comfy.LatentApplyOperation",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.apply_operation",
                        inputs={
                            "samples": MappingSource.copy("samples"),
                            "operation": MappingSource.copy("operation"),
                        },
                        outputs={"latent": "_0_LATENT_"},
                    ),
                ),
            ),
            evidence=LATENT_OPERATION_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentOperationTonemapReinhard",
            carrier="dinkster.latent.operation_tonemap_reinhard",
            rule=ReplacementRule(
                from_type="comfy.LatentOperationTonemapReinhard",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.operation_tonemap_reinhard",
                        inputs={"multiplier": MappingSource.copy("multiplier")},
                        outputs={"operation": "_0_LATENT_OPERATION_"},
                    ),
                ),
            ),
            evidence=LATENT_OPERATION_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentOperationSharpen",
            carrier="dinkster.latent.operation_sharpen",
            rule=ReplacementRule(
                from_type="comfy.LatentOperationSharpen",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.operation_sharpen",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in ("sharpen_radius", "sigma", "alpha")
                        },
                        outputs={"operation": "_0_LATENT_OPERATION_"},
                    ),
                ),
            ),
            evidence=LATENT_OPERATION_PARITY_EVIDENCE,
        ),
        _record(
            node_class="LatentApplyOperationCFG",
            carrier="dinkster.latent.apply_operation_cfg",
            rule=ReplacementRule(
                from_type="comfy.LatentApplyOperationCFG",
                cases=(
                    ReplacementCase.build(
                        "dinkster.latent.apply_operation_cfg",
                        inputs={
                            "model": MappingSource.copy("model"),
                            "operation": MappingSource.copy("operation"),
                        },
                        outputs={"model": "_0_MODEL_"},
                    ),
                ),
            ),
            evidence=LATENT_APPLY_OPERATION_CFG_PARITY_EVIDENCE,
        ),
    ]


def _trellis2_source_schemas() -> list[NodeSchema]:
    model_file = ["model.safetensors"]

    schemas = [
        _static_schema(
            "CFGOverride",
            namespace="",
            category="model/sampling/guiders",
            required={
                "model": ("MODEL",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            returns=("MODEL",),
            return_names=("_0_MODEL_",),
        ),
        _static_schema(
            "RescaleCFG",
            namespace="",
            category="model/patch",
            required={
                "model": ("MODEL",),
                "multiplier": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
            },
            returns=("MODEL",),
            return_names=("model",),
        ),
        _static_schema(
            "ModelSamplingSD3",
            namespace="",
            category="model/patch/stable diffusion",
            required={
                "model": ("MODEL",),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0}),
            },
            returns=("MODEL",),
            return_names=("model",),
        ),
        _static_schema(
            "EmptyTrellis2LatentStructure",
            namespace="",
            category="model/latent/trellis",
            required={"batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})},
            returns=("LATENT",),
            return_names=("_0_LATENT_",),
        ),
        _static_schema(
            "Trellis2Conditioning",
            namespace="",
            category="model/conditioning/trellis2",
            required={"clip_vision_model": ("CLIP_VISION",), "image": ("IMAGE",)},
            returns=("CONDITIONING", "CONDITIONING"),
            return_names=("_0_CONDITIONING_", "_1_CONDITIONING_"),
        ),
        _static_schema(
            "Pixal3DConditioning",
            namespace="",
            category="model/conditioning/trellis2",
            required={
                "clip_vision_model": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "camera_angle_x": (
                    "FLOAT",
                    {"default": 49.13, "min": 1.0, "max": 170.0, "step": 0.01},
                ),
            },
            returns=("CONDITIONING", "CONDITIONING"),
            return_names=("_0_CONDITIONING_", "_1_CONDITIONING_"),
        ),
        _static_schema(
            "VaeDecodeStructureTrellis2",
            namespace="",
            category="model/latent/trellis",
            required={
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "resolution": (["32", "64"], {"default": "32"}),
            },
            returns=("VOXEL",),
            return_names=("voxel",),
        ),
        _static_schema(
            "Trellis2ShapeStage",
            namespace="",
            category="model/conditioning/trellis2",
            required={
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "voxel": ("VOXEL",),
            },
            returns=("CONDITIONING", "CONDITIONING", "LATENT"),
            return_names=("_0_CONDITIONING_", "_1_CONDITIONING_", "_2_LATENT_"),
        ),
        _static_schema(
            "Trellis2UpsampleStage",
            namespace="",
            category="model/conditioning/trellis2",
            required={
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "shape_latent": ("LATENT",),
                "vae": ("VAE",),
                "target_resolution": (
                    "INT",
                    {"default": 1024, "min": 1024, "max": 2048, "step": 128},
                ),
            },
            returns=("CONDITIONING", "CONDITIONING", "LATENT"),
            return_names=("_0_CONDITIONING_", "_1_CONDITIONING_", "_2_LATENT_"),
        ),
        _static_schema(
            "VaeDecodeShapeTrellis",
            namespace="",
            category="model/latent/trellis",
            required={"samples": ("LATENT",), "vae": ("VAE",)},
            returns=("MESH", "SHAPE_SUBDIVIDES"),
            return_names=("mesh", "_1_SHAPE_SUBDIVIDES_"),
        ),
        _static_schema(
            "Trellis2TextureStage",
            namespace="",
            category="model/conditioning/trellis2",
            required={
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "shape_latent": ("LATENT",),
            },
            returns=("CONDITIONING", "CONDITIONING", "LATENT"),
            return_names=("_0_CONDITIONING_", "_1_CONDITIONING_", "_2_LATENT_"),
        ),
        _static_schema(
            "VaeDecodeTextureTrellis",
            namespace="",
            category="model/latent/trellis",
            required={
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "shape_subdivides": ("SHAPE_SUBDIVIDES",),
            },
            returns=("VOXEL",),
            return_names=("voxel_colors",),
        ),
        _with_asset_inputs(
            _static_schema(
                "LoadMoGeModel",
                namespace="",
                category="model/loaders",
                required={"model_name": (model_file,)},
                returns=("MOGE_MODEL",),
                return_names=("_0_MOGE_MODEL_",),
            ),
            {"model_name": ("model/geometry-estimation", True)},
        ),
        _static_schema(
            "MoGeInference",
            namespace="",
            category="image/geometry estimation",
            required={
                "moge_model": ("MOGE_MODEL",),
                "image": ("IMAGE",),
                "resolution_level": ("INT", {"default": 9, "min": 0, "max": 9}),
                "fov_x_degrees": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 170.0}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64}),
                "force_projection": ("BOOLEAN", {"default": True}),
                "apply_mask": ("BOOLEAN", {"default": True}),
            },
            returns=("MOGE_GEOMETRY",),
            return_names=("_0_MOGE_GEOMETRY_",),
        ),
        _static_schema(
            "MoGeGeometryToFOV",
            namespace="",
            category="image/geometry estimation",
            required={
                "moge_geometry": ("MOGE_GEOMETRY",),
                "axis": (["vertical", "horizontal", "diagonal"], {"default": "vertical"}),
                "unit": (["degrees", "radians"], {"default": "degrees"}),
            },
            returns=("FLOAT", "FLOAT"),
            return_names=("_0_FLOAT_", "_1_FLOAT_"),
        ),
        _with_asset_inputs(
            _static_schema(
                "LoadBackgroundRemovalModel",
                namespace="",
                category="model/loaders",
                required={"bg_removal_name": (model_file,)},
                returns=("BACKGROUND_REMOVAL",),
                return_names=("bg_model",),
            ),
            {"bg_removal_name": ("model/background-removal", True)},
        ),
        _static_schema(
            "RemoveBackground",
            namespace="",
            category="image/background removal",
            required={"bg_removal_model": ("BACKGROUND_REMOVAL",), "image": ("IMAGE",)},
            returns=("MASK",),
            return_names=("mask",),
        ),
        _static_schema(
            "ImageCropToMask",
            namespace="",
            category="image/transform",
            required={
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "pad_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0}),
                "grow_mask": ("INT", {"default": 0, "min": -32, "max": 32}),
                "background": ("STRING", {"default": "#000000", "multiline": False}),
            },
            returns=("IMAGE",),
            return_names=("_0_IMAGE_",),
        ),
        _static_schema(
            "MaskPreview",
            namespace="",
            category="image/mask",
            required={"mask": ("MASK",)},
            returns=("MASK",),
            return_names=("_0_MASK_",),
        ),
        _static_schema(
            "VoxelToMesh",
            namespace="",
            category="3d",
            required={
                "voxel": ("VOXEL",),
                "algorithm": (["surface net", "basic"], {"default": "surface net"}),
                "threshold": ("FLOAT", {"default": 0.6, "min": -1.0, "max": 1.0}),
            },
            returns=("MESH",),
            return_names=("_0_MESH_",),
        ),
        _static_schema(
            "GetMeshInfo",
            namespace="",
            category="3d/mesh",
            required={"mesh": ("MESH",)},
            returns=("MESH", "STRING"),
            return_names=("_0_MESH_", "_1_STRING_"),
        ),
        _static_schema(
            "RemeshMesh",
            namespace="",
            category="3d/mesh",
            required={
                "mesh": ("MESH",),
                "resolution": ("INT", {"default": 512, "min": 32, "max": 2048}),
                "sign_mode": (["udf", "sdf"], {"default": "udf"}),
                "qef": ("BOOLEAN", {"default": False}),
                "drop_inverted_components": ("BOOLEAN", {"default": False}),
                "drop_enclosed_components": ("BOOLEAN", {"default": False}),
                "band": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0}),
                "project_back": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "fix_poles": ("BOOLEAN", {"default": False}),
                "smooth_iters": ("INT", {"default": 0, "min": 0, "max": 20}),
                "drop_small_components": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0, "max": 0.5},
                ),
                "precluster_max_verts": (
                    "INT",
                    {"default": 20_000_000, "min": 0, "max": 100_000_000},
                ),
            },
            returns=("MESH",),
            return_names=("mesh",),
        ),
        _static_schema(
            "DecimateMesh",
            namespace="",
            category="3d/mesh",
            required={
                "mesh": ("MESH",),
                "target_face_count": (
                    "INT",
                    {"default": 200_000, "min": 0, "max": 50_000_000},
                ),
                "placement_mode": (["midpoint", "qem"], {"default": "midpoint"}),
            },
            returns=("MESH",),
            return_names=("mesh",),
        ),
        _static_schema(
            "MeshSmoothNormals",
            namespace="",
            category="3d/mesh",
            required={
                "mesh": ("MESH",),
                "crease_angle": ("FLOAT", {"default": 180.0, "min": 0.0, "max": 180.0}),
            },
            returns=("MESH",),
            return_names=("mesh",),
        ),
        _static_schema(
            "UnwrapMesh",
            namespace="",
            category="3d/texturing",
            required={
                "mesh": ("MESH",),
                "segmenter": (["pec", "adaptive"], {"default": "pec"}),
                "resolution": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "padding": ("INT", {"default": 1, "min": 0, "max": 16}),
                "weld_distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            },
            returns=("MESH",),
            return_names=("mesh",),
        ),
        _static_schema(
            "PaintMesh",
            namespace="",
            category="3d/texturing",
            required={"mesh": ("MESH",), "voxel_colors": ("VOXEL",)},
            returns=("MESH",),
            return_names=("mesh",),
        ),
        _static_schema(
            "BakeTextureFromVoxel",
            namespace="",
            category="3d/texturing",
            required={
                "mesh": ("MESH",),
                "voxel_colors": ("VOXEL",),
                "texture_size": ("INT", {"default": 2048, "min": 64, "max": 8192}),
            },
            optional={"reference_mesh": ("MESH",)},
            returns=("IMAGE", "IMAGE", "IMAGE"),
            return_names=("_0_IMAGE_", "_1_IMAGE_", "_2_IMAGE_"),
        ),
        _static_schema(
            "BakeNormalMapFromMesh",
            namespace="",
            category="3d/texturing",
            required={
                "low_poly": ("MESH",),
                "high_poly": ("MESH",),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "cage_distance": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5}),
                "ignore_backfaces": ("BOOLEAN", {"default": True}),
            },
            returns=("IMAGE",),
            return_names=("_0_IMAGE_",),
        ),
        _static_schema(
            "BakeAmbientOcclusion",
            namespace="",
            category="3d/texturing",
            required={
                "low_poly": ("MESH",),
                "high_poly": ("MESH",),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "samples": ("INT", {"default": 64, "min": 4, "max": 1024}),
                "max_distance": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 2.0}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "bias": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.2}),
            },
            returns=("IMAGE",),
            return_names=("_0_IMAGE_",),
        ),
        _static_schema(
            "RenderUVAtlas",
            namespace="",
            category="3d/texturing",
            required={
                "mesh": ("MESH",),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 4096}),
            },
            returns=("IMAGE",),
            return_names=("image",),
        ),
        _static_schema(
            "ApplyTextureToMesh",
            namespace="",
            category="3d/texturing",
            required={"mesh": ("MESH",), "base_color": ("IMAGE",)},
            optional={
                "metallic": ("IMAGE",),
                "roughness": ("IMAGE",),
                "occlusion": ("IMAGE",),
                "normal_map": ("IMAGE",),
            },
            returns=("MESH",),
            return_names=("mesh",),
        ),
        _with_model3d_ports(
            _static_schema(
                "MeshToFile3D",
                namespace="",
                category="3d",
                required={"mesh": ("MESH",)},
                returns=("FILE_3D_GLB",),
                return_names=("_0_FILE_3D_GLB_",),
            ),
            outputs=frozenset({"_0_FILE_3D_GLB_"}),
        ),
    ]
    return schemas


def _trellis2_rule(
    source: str,
    carrier: str,
    *,
    inputs: dict[str, MappingSource],
    outputs: dict[str, str],
    nodes: dict[str, ReplacementNode] | None = None,
    links: tuple[ReplacementLink, ...] = (),
) -> ReplacementRule:
    return ReplacementRule(
        from_type=f"comfy.{source}",
        cases=(
            ReplacementCase.build(
                carrier,
                inputs=inputs,
                outputs=outputs,
                nodes=nodes or {},
                links=links,
            ),
        ),
    )


def _trellis2_records() -> list[dict[str, object]]:
    direct: list[tuple[str, str, tuple[str, ...], dict[str, str]]] = [
        (
            "CFGOverride",
            "dinkster.cfg_override",
            ("model", "cfg", "start_percent", "end_percent"),
            {"model": "_0_MODEL_"},
        ),
        ("RescaleCFG", "dinkster.rescale_cfg", ("model", "multiplier"), {"model": "model"}),
        ("ModelSamplingSD3", "dinkster.model_sampling_sd3", ("model", "shift"), {"model": "model"}),
        (
            "EmptyTrellis2LatentStructure",
            "dinkster.empty_trellis2_latent_structure",
            ("batch_size",),
            {"latent": "_0_LATENT_"},
        ),
        (
            "Trellis2Conditioning",
            "dinkster.trellis2_conditioning",
            ("clip_vision_model", "image"),
            {"positive": "_0_CONDITIONING_", "negative": "_1_CONDITIONING_"},
        ),
        (
            "Pixal3DConditioning",
            "dinkster.pixal3d_conditioning",
            ("clip_vision_model", "image", "camera_angle_x"),
            {"positive": "_0_CONDITIONING_", "negative": "_1_CONDITIONING_"},
        ),
        (
            "VaeDecodeStructureTrellis2",
            "dinkster.vae_decode_structure_trellis2",
            ("samples", "vae", "resolution"),
            {"voxel": "voxel"},
        ),
        (
            "Trellis2ShapeStage",
            "dinkster.trellis2_shape_stage",
            ("positive", "negative", "voxel"),
            {
                "positive": "_0_CONDITIONING_",
                "negative": "_1_CONDITIONING_",
                "latent": "_2_LATENT_",
            },
        ),
        (
            "Trellis2UpsampleStage",
            "dinkster.trellis2_upsample_stage",
            ("positive", "negative", "shape_latent", "vae", "target_resolution"),
            {
                "positive": "_0_CONDITIONING_",
                "negative": "_1_CONDITIONING_",
                "latent": "_2_LATENT_",
            },
        ),
        (
            "VaeDecodeShapeTrellis",
            "dinkster.vae_decode_shape_trellis",
            ("samples", "vae"),
            {"mesh": "mesh", "shape_subdivides": "_1_SHAPE_SUBDIVIDES_"},
        ),
        (
            "Trellis2TextureStage",
            "dinkster.trellis2_texture_stage",
            ("positive", "negative", "shape_latent"),
            {
                "positive": "_0_CONDITIONING_",
                "negative": "_1_CONDITIONING_",
                "latent": "_2_LATENT_",
            },
        ),
        (
            "VaeDecodeTextureTrellis",
            "dinkster.vae_decode_texture_trellis",
            ("samples", "vae", "shape_subdivides"),
            {"voxel_colors": "voxel_colors"},
        ),
        (
            "ImageCropToMask",
            "dinkster.image_crop_to_mask",
            ("images", "masks", "width", "height", "pad_factor", "grow_mask", "background"),
            {"images": "_0_IMAGE_"},
        ),
        ("MaskPreview", "dinkster.preview_mask", ("mask",), {"mask": "_0_MASK_"}),
        (
            "VoxelToMesh",
            "dinkster.voxel_to_mesh",
            ("voxel", "algorithm", "threshold"),
            {"mesh": "_0_MESH_"},
        ),
        (
            "GetMeshInfo",
            "dinkster.get_mesh_info",
            ("mesh",),
            {"mesh": "_0_MESH_", "info": "_1_STRING_"},
        ),
        (
            "DecimateMesh",
            "dinkster.decimate_mesh",
            ("mesh", "target_face_count", "placement_mode"),
            {"mesh": "mesh"},
        ),
        (
            "MeshSmoothNormals",
            "dinkster.smooth_mesh_normals",
            ("mesh", "crease_angle"),
            {"mesh": "mesh"},
        ),
        (
            "UnwrapMesh",
            "dinkster.unwrap_mesh",
            ("mesh", "segmenter", "resolution", "padding", "weld_distance"),
            {"mesh": "mesh"},
        ),
        ("PaintMesh", "dinkster.paint_mesh", ("mesh", "voxel_colors"), {"mesh": "mesh"}),
        (
            "BakeTextureFromVoxel",
            "dinkster.bake_texture_from_voxel",
            ("mesh", "voxel_colors", "texture_size", "reference_mesh"),
            {"base_color": "_0_IMAGE_", "metallic": "_1_IMAGE_", "roughness": "_2_IMAGE_"},
        ),
        (
            "BakeNormalMapFromMesh",
            "dinkster.bake_normal_map_from_mesh",
            ("low_poly", "high_poly", "resolution", "cage_distance", "ignore_backfaces"),
            {"normal_map": "_0_IMAGE_"},
        ),
        (
            "BakeAmbientOcclusion",
            "dinkster.bake_ambient_occlusion",
            ("low_poly", "high_poly", "resolution", "samples", "max_distance", "strength", "bias"),
            {"occlusion": "_0_IMAGE_"},
        ),
        ("RenderUVAtlas", "dinkster.render_uv_atlas", ("mesh", "resolution"), {"image": "image"}),
        (
            "ApplyTextureToMesh",
            "dinkster.apply_texture_to_mesh",
            ("mesh", "base_color", "metallic", "roughness", "occlusion", "normal_map"),
            {"mesh": "mesh"},
        ),
        ("MeshToFile3D", "dinkster.mesh_to_model3d", ("mesh",), {"model": "_0_FILE_3D_GLB_"}),
    ]
    records = [
        _record(
            node_class=node_class,
            carrier=carrier,
            rule=_trellis2_rule(
                node_class,
                carrier,
                inputs={input_id: MappingSource.copy(input_id) for input_id in input_ids},
                outputs=outputs,
            ),
            revision=TRELLIS2_BASELINE,
            evidence=TRELLIS2_WORKFLOW_EVIDENCE,
        )
        for node_class, carrier, input_ids, outputs in direct
    ]
    records.extend(
        (
            _record(
                node_class="LoadMoGeModel",
                carrier="dinkster.load_geometry_model",
                rule=_trellis2_rule(
                    "LoadMoGeModel",
                    "dinkster.load_geometry_model",
                    inputs={"model": MappingSource.copy("model_name")},
                    outputs={"model": "_0_MOGE_MODEL_"},
                ),
                revision=TRELLIS2_BASELINE,
                evidence=TRELLIS2_WORKFLOW_EVIDENCE,
            ),
            _record(
                node_class="MoGeInference",
                carrier="dinkster.estimate_geometry",
                rule=_trellis2_rule(
                    "MoGeInference",
                    "dinkster.estimate_geometry",
                    inputs={
                        "model": MappingSource.copy("moge_model"),
                        **{
                            input_id: MappingSource.copy(input_id)
                            for input_id in (
                                "image",
                                "resolution_level",
                                "fov_x_degrees",
                                "batch_size",
                                "force_projection",
                                "apply_mask",
                            )
                        },
                    },
                    outputs={"geometry": "_0_MOGE_GEOMETRY_"},
                ),
                revision=TRELLIS2_BASELINE,
                evidence=TRELLIS2_WORKFLOW_EVIDENCE,
            ),
            _record(
                node_class="MoGeGeometryToFOV",
                carrier="dinkster.geometry_to_fov",
                rule=_trellis2_rule(
                    "MoGeGeometryToFOV",
                    "dinkster.geometry_to_fov",
                    inputs={
                        "geometry": MappingSource.copy("moge_geometry"),
                        "axis": MappingSource.copy("axis"),
                        "unit": MappingSource.copy("unit"),
                    },
                    outputs={"fov": "_0_FLOAT_", "focal_pixels": "_1_FLOAT_"},
                ),
                revision=TRELLIS2_BASELINE,
                evidence=TRELLIS2_WORKFLOW_EVIDENCE,
            ),
            _record(
                node_class="LoadBackgroundRemovalModel",
                carrier="dinkster.load_background_removal",
                rule=_trellis2_rule(
                    "LoadBackgroundRemovalModel",
                    "dinkster.load_background_removal",
                    inputs={"model": MappingSource.copy("bg_removal_name")},
                    outputs={"model": "bg_model"},
                ),
                revision=TRELLIS2_BASELINE,
                evidence=TRELLIS2_WORKFLOW_EVIDENCE,
            ),
            _record(
                node_class="RemoveBackground",
                carrier="dinkster.remove_background",
                rule=_trellis2_rule(
                    "RemoveBackground",
                    "dinkster.remove_background",
                    inputs={
                        "model": MappingSource.copy("bg_removal_model"),
                        "image": MappingSource.copy("image"),
                    },
                    outputs={"mask": "mask"},
                ),
                revision=TRELLIS2_BASELINE,
                evidence=TRELLIS2_WORKFLOW_EVIDENCE,
            ),
            _record(
                node_class="RemeshMesh",
                carrier="dinkster.remesh_mesh",
                rule=_trellis2_rule(
                    "RemeshMesh",
                    "dinkster.remesh_mesh",
                    inputs={
                        input_id: MappingSource.copy(input_id)
                        for input_id in (
                            "mesh",
                            "resolution",
                            "sign_mode",
                            "qef",
                            "drop_inverted_components",
                            "drop_enclosed_components",
                            "band",
                            "project_back",
                            "fix_poles",
                            "smooth_iters",
                            "drop_small_components",
                            "precluster_max_verts",
                        )
                    },
                    outputs={"mesh": "mesh"},
                ),
                revision=TRELLIS2_BASELINE,
                evidence=TRELLIS2_WORKFLOW_EVIDENCE,
            ),
        )
    )
    return records


def build_registry(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != COMFY_BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {COMFY_BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as _comfy_args  # pyright: ignore[reportMissingImports]

    # Schemas are inert interface data; forcing CPU keeps the import working
    # on hosts whose torch build has no CUDA support.
    _comfy_args.cpu = True

    # nodes_post_processing must load before nodes_latent: the two modules
    # form an import cycle that only resolves in this order.
    import comfy_extras.nodes_post_processing  # noqa: F401  # pyright: ignore[reportMissingImports]
    from comfy_extras import (  # pyright: ignore[reportMissingImports]
        nodes_controlnet,
        nodes_latent,
        nodes_mask,
        nodes_post_processing,
        nodes_rebatch,
    )
    from comfy_extras.nodes_cond import (  # pyright: ignore[reportMissingImports]
        CLIPTextEncodeControlnet,
        T5TokenizerOptions,
    )
    from comfy_extras.nodes_video_model import (  # pyright: ignore[reportMissingImports]
        ConditioningSetAreaPercentageVideo,
    )
    from nodes import (  # pyright: ignore[reportMissingImports]
        CLIPSetLastLayer,
        ConditioningAverage,
        ConditioningCombine,
        ConditioningConcat,
        ConditioningMultiply,
        ConditioningSetArea,
        ConditioningSetAreaPercentage,
        ConditioningSetMask,
        ConditioningSetTimestepRange,
        ConditioningZeroOut,
        ControlNetApply,
        ControlNetApplyAdvanced,
        ControlNetLoader,
        LatentBlend,
        LatentComposite,
        LatentCrop,
        LatentFlip,
        LatentFromBatch,
        LatentRotate,
        LatentUpscale,
        LatentUpscaleBy,
        RepeatLatentBatch,
        SetLatentNoiseMask,
    )

    def core_schema(node_class: type[Any], name: str) -> Any:
        return translate_node(name, node_class, CompatTranslation()).schema()

    source_schemas = [
        core_schema(CLIPSetLastLayer, "CLIPSetLastLayer"),
        core_schema(T5TokenizerOptions, "T5TokenizerOptions"),
        core_schema(CLIPTextEncodeControlnet, "CLIPTextEncodeControlnet"),
        _with_asset_inputs(
            core_schema(ControlNetLoader, "ControlNetLoader"),
            {"control_net_name": ("model/controlnet", True)},
        ),
        core_schema(ControlNetApply, "ControlNetApply"),
        core_schema(ControlNetApplyAdvanced, "ControlNetApplyAdvanced"),
        _core_v3_schema(nodes_controlnet.SetUnionControlNetType),
        core_schema(ConditioningCombine, "ConditioningCombine"),
        core_schema(ConditioningAverage, "ConditioningAverage"),
        core_schema(ConditioningConcat, "ConditioningConcat"),
        core_schema(ConditioningMultiply, "ConditioningMultiply"),
        core_schema(ConditioningSetArea, "ConditioningSetArea"),
        core_schema(ConditioningSetAreaPercentage, "ConditioningSetAreaPercentage"),
        core_schema(ConditioningSetAreaPercentageVideo, "ConditioningSetAreaPercentageVideo"),
        core_schema(ConditioningSetMask, "ConditioningSetMask"),
        core_schema(ConditioningSetTimestepRange, "ConditioningSetTimestepRange"),
        core_schema(ConditioningZeroOut, "ConditioningZeroOut"),
        _core_v3_schema(nodes_latent.LatentAdd),
        _core_v3_schema(nodes_latent.LatentSubtract),
        _core_v3_schema(nodes_latent.LatentInterpolate),
        core_schema(LatentBlend, "LatentBlend"),
        _core_v3_schema(nodes_latent.LatentMultiply),
        core_schema(LatentRotate, "LatentRotate"),
        core_schema(LatentFlip, "LatentFlip"),
        core_schema(LatentCrop, "LatentCrop"),
        core_schema(LatentUpscale, "LatentUpscale"),
        core_schema(LatentUpscaleBy, "LatentUpscaleBy"),
        core_schema(LatentComposite, "LatentComposite"),
        _core_v3_schema(nodes_mask.LatentCompositeMasked),
        _core_v3_schema(nodes_latent.LatentConcat),
        _core_v3_schema(nodes_latent.LatentCut),
        _core_v3_schema(nodes_latent.LatentCutToBatch),
        core_schema(LatentFromBatch, "LatentFromBatch"),
        core_schema(RepeatLatentBatch, "RepeatLatentBatch"),
        _core_v3_schema(nodes_latent.LatentBatchSeedBehavior),
        _core_v3_schema(nodes_latent.LatentBatch),
        _core_v3_schema(nodes_post_processing.BatchLatentsNode),
        _core_v3_schema(nodes_rebatch.LatentRebatch),
        core_schema(SetLatentNoiseMask, "SetLatentNoiseMask"),
        _core_v3_schema(nodes_latent.ReplaceVideoLatentFrames),
        _core_v3_schema(nodes_latent.LatentApplyOperation),
        _core_v3_schema(nodes_latent.LatentOperationTonemapReinhard),
        _core_v3_schema(nodes_latent.LatentOperationSharpen),
        _core_v3_schema(nodes_latent.LatentApplyOperationCFG),
        _static_schema(
            "ModelSamplingAuraFlow",
            namespace="",
            category="model/patch",
            required={
                "model": ("MODEL",),
                "shift": (
                    "FLOAT",
                    {"default": 1.73, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
            },
            returns=("MODEL",),
        ),
        _static_schema(
            "EmptySD3LatentImage",
            namespace="",
            category="model/latent/stable diffusion",
            required={
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 16384, "step": 16},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            returns=("LATENT",),
        ),
        _static_schema(
            "ChromaRadianceOptions",
            namespace="",
            category="model/patch/chroma radiance",
            description="Allows setting advanced options for the Chroma Radiance model.",
            required={
                "model": ("MODEL",),
                "preserve_wrapper": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When enabled, will delegate to an existing model function wrapper "
                            "if it exists. Generally should be left enabled."
                        ),
                    },
                ),
                "start_sigma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "First sigma that these options will be in effect.",
                        "advanced": True,
                    },
                ),
                "end_sigma": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Last sigma that these options will be in effect.",
                        "advanced": True,
                    },
                ),
                "nerf_tile_size": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "tooltip": (
                            "Allows overriding the default NeRF tile size. -1 means use the "
                            "default (32). 0 means use non-tiling mode (may require a lot of VRAM)."
                        ),
                        "advanced": True,
                    },
                ),
                "force_sequential_txt_ids": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Force usage of sequential text token IDs instead of zeroes. Should "
                            "be used for checkpoints from 2026-05-22 to 2026-06-01 that are "
                            "trained in this way but do not contain the __sequential__ key in "
                            "the state dict."
                        ),
                        "advanced": True,
                    },
                ),
            },
            returns=("MODEL",),
            return_names=("_0_MODEL_",),
        ),
        _static_schema(
            "EmptyChromaRadianceLatentImage",
            namespace="",
            category="model/latent/chroma radiance",
            required={
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 16384, "step": 16},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            returns=("LATENT",),
            return_names=("_0_LATENT_",),
        ),
        _static_schema(
            "ConditioningMultiCombine",
            namespace="comfyui-kjnodes",
            required={
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 20, "step": 1}),
                "operation": (["combine", "concat"], {"default": "combine"}),
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
            },
            optional={f"conditioning_{index}": ("CONDITIONING",) for index in range(3, 21)},
            returns=("CONDITIONING", "INT"),
            return_names=("combined", "inputcount"),
        ),
        *(
            _static_schema(
                f"ConditioningSetMaskAndCombine{'' if count == 2 else count}",
                namespace="comfyui-kjnodes",
                required={
                    **{
                        f"{polarity}_{index}": ("CONDITIONING",)
                        for index in range(1, count + 1)
                        for polarity in ("positive", "negative")
                    },
                    **{f"mask_{index}": ("MASK",) for index in range(1, count + 1)},
                    **{
                        f"mask_{index}_strength": (
                            "FLOAT",
                            {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                        )
                        for index in range(1, count + 1)
                    },
                    "set_cond_area": (["default", "mask bounds"],),
                },
                returns=("CONDITIONING", "CONDITIONING"),
                return_names=("combined_positive", "combined_negative"),
            )
            for count in range(2, 6)
        ),
        _static_schema(
            "ConditioningCombineMultiple+",
            namespace="comfyui_essentials",
            required={
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
            },
            optional={f"conditioning_{index}": ("CONDITIONING",) for index in range(3, 6)},
        ),
        _static_schema(
            "SD3NegativeConditioning+",
            namespace="comfyui_essentials",
            required={
                "conditioning": ("CONDITIONING",),
                "end": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        ),
        _static_schema(
            "GenerateNoise",
            namespace="comfyui-kjnodes",
            required={
                "width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4096, "step": 0.01}),
                "constant_batch_noise": ("BOOLEAN", {"default": False}),
                "normalize": ("BOOLEAN", {"default": False}),
            },
            optional={
                "model": ("MODEL",),
                "sigmas": ("SIGMAS",),
                "latent_channels": (["4", "16"],),
                "shape": (["BCHW", "BCTHW", "BTCHW"],),
            },
            returns=("LATENT",),
        ),
        _static_schema(
            "InjectNoiseToLatent",
            namespace="comfyui-kjnodes",
            required={
                "latents": ("LATENT",),
                "strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 200.0, "step": 0.0001}),
                "noise": ("LATENT",),
                "normalize": ("BOOLEAN", {"default": False}),
                "average": ("BOOLEAN", {"default": False}),
            },
            optional={
                "mask": ("MASK",),
                "mix_randn_amount": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001},
                ),
                "seed": ("INT", {"default": 123, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
            },
            returns=("LATENT",),
        ),
        _static_schema(
            "ScheduledCFGGuider //Inspire",
            namespace="comfyui-inspire-pack",
            category="sampling/custom_sampling/guiders",
            required={
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sigmas": ("SIGMAS",),
                "from_cfg": (
                    "FLOAT",
                    {"default": 6.5, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "to_cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "schedule": (["linear", "log", "exp", "cos"], {"default": "log"}),
            },
            returns=("GUIDER", "SIGMAS"),
        ),
        *_loader_source_schemas(),
        *_trellis2_source_schemas(),
    ]

    records = [
        _record(
            node_class="CLIPSetLastLayer",
            carrier="dinkster.clip_set_last_layer",
            rule=_copy_rule(
                "comfy.CLIPSetLastLayer",
                "dinkster.clip_set_last_layer",
                "clip",
                "stop_at_clip_layer",
                output_id="clip",
            ),
            evidence=[
                "tests/test_native_arm.py::test_generation_clip_options_chain_without_mutating_sources"
            ],
        ),
        _record(
            node_class="T5TokenizerOptions",
            carrier="dinkster.t5_tokenizer_options",
            rule=_copy_rule(
                "comfy.T5TokenizerOptions",
                "dinkster.t5_tokenizer_options",
                "clip",
                "min_padding",
                "min_length",
                output_id="clip",
                source_output_id="CLIP",
            ),
            evidence=[
                "tests/test_native_arm.py::test_generation_clip_options_chain_without_mutating_sources"
            ],
        ),
        _record(
            node_class="CLIPTextEncodeControlnet",
            carrier="dinkster.clip_text_encode_controlnet",
            rule=_copy_rule(
                "comfy.CLIPTextEncodeControlnet",
                "dinkster.clip_text_encode_controlnet",
                "clip",
                "conditioning",
                "text",
                source_output_id="CONDITIONING",
            ),
            evidence=[
                "tests/test_native_arm.py::test_generation_clip_text_encode_controlnet_stamps_control_payloads"
            ],
        ),
        _record(
            node_class="ControlNetLoader",
            carrier="dinkster.load_controlnet",
            rule=_copy_rule(
                "comfy.ControlNetLoader",
                "dinkster.load_controlnet",
                "control_net_name",
                output_id="control_net",
            ),
            evidence=[
                "tests/test_generation_comfy_aliases.py::test_controlnet_aliases_preserve_schema_wiring"
            ],
        ),
        _record(
            node_class="ControlNetApply",
            carrier="dinkster.apply_controlnet",
            rule=_copy_rule(
                "comfy.ControlNetApply",
                "dinkster.apply_controlnet",
                "conditioning",
                "control_net",
                "image",
                "strength",
            ),
            evidence=[
                "tests/test_generation_comfy_aliases.py::test_controlnet_aliases_preserve_schema_wiring"
            ],
        ),
        _record(
            node_class="ControlNetApplyAdvanced",
            carrier="dinkster.apply_controlnet_advanced",
            rule=ReplacementRule(
                from_type="comfy.ControlNetApplyAdvanced",
                cases=(
                    ReplacementCase.build(
                        "dinkster.apply_controlnet_advanced",
                        inputs={
                            input_id: MappingSource.copy(input_id)
                            for input_id in (
                                "positive",
                                "negative",
                                "control_net",
                                "image",
                                "strength",
                                "start_percent",
                                "end_percent",
                                "vae",
                            )
                        },
                        outputs={"positive": "positive", "negative": "negative"},
                    ),
                ),
            ),
            evidence=[
                "tests/test_generation_comfy_aliases.py::test_controlnet_aliases_preserve_schema_wiring"
            ],
        ),
        _record(
            node_class="SetUnionControlNetType",
            carrier="dinkster.set_controlnet_union_type",
            rule=_copy_rule(
                "comfy.SetUnionControlNetType",
                "dinkster.set_controlnet_union_type",
                "control_net",
                "type",
                output_id="control_net",
                source_output_id="_0_CONTROL_NET_",
            ),
            evidence=[
                "tests/test_generation_comfy_aliases.py::test_controlnet_aliases_preserve_schema_wiring"
            ],
        ),
        _record(
            node_class="ConditioningCombine",
            carrier="dinkster.conditioning_merge",
            rule=_combo_rule(
                from_type="comfy.ConditioningCombine",
                carrier="dinkster.conditioning_merge",
                slot="mode",
                variant="combine",
                inputs={
                    "mode.inputs.conditioning_1": "conditioning_1",
                    "mode.inputs.conditioning_2": "conditioning_2",
                },
            ),
            evidence=[
                "tests/test_generation_comfy_aliases.py::test_core_conditioning_combo_aliases_select_native_variants"
            ],
        ),
        _record(
            node_class="ConditioningAverage",
            carrier="dinkster.conditioning_merge",
            rule=_combo_rule(
                from_type="comfy.ConditioningAverage",
                carrier="dinkster.conditioning_merge",
                slot="mode",
                variant="average",
                inputs={
                    "mode.conditioning_to": "conditioning_to",
                    "mode.conditioning_from": "conditioning_from",
                    "mode.conditioning_to_strength": "conditioning_to_strength",
                },
            ),
            evidence=[
                "tests/test_generation_comfy_aliases.py::test_core_conditioning_combo_aliases_select_native_variants"
            ],
        ),
        _record(
            node_class="ConditioningConcat",
            carrier="dinkster.conditioning_merge",
            rule=_combo_rule(
                from_type="comfy.ConditioningConcat",
                carrier="dinkster.conditioning_merge",
                slot="mode",
                variant="concat",
                inputs={
                    "mode.conditioning_to": "conditioning_to",
                    "mode.conditioning_from": "conditioning_from",
                },
            ),
            evidence=[
                "tests/test_generation_comfy_aliases.py::test_core_conditioning_combo_aliases_select_native_variants"
            ],
        ),
        _record(
            node_class="ConditioningMultiply",
            carrier="dinkster.conditioning_scale",
            rule=_copy_rule(
                "comfy.ConditioningMultiply",
                "dinkster.conditioning_scale",
                "conditioning",
                "multiplier",
            ),
            evidence=[
                "tests/test_native_arm.py::test_generation_conditioning_scale_replays_pinned_comfyui_goldens"
            ],
        ),
        _record(
            node_class="ConditioningSetArea",
            carrier="dinkster.conditioning_set_area",
            rule=_combo_rule(
                from_type="comfy.ConditioningSetArea",
                carrier="dinkster.conditioning_set_area",
                slot="units",
                variant="pixels",
                inputs={
                    "conditioning": "conditioning",
                    "strength": "strength",
                    "units.width": "width",
                    "units.height": "height",
                    "units.x": "x",
                    "units.y": "y",
                },
            ),
            evidence=[
                "tests/test_generation_comfy_aliases.py::test_core_conditioning_combo_aliases_select_native_variants"
            ],
        ),
        _record(
            node_class="ConditioningSetAreaPercentage",
            carrier="dinkster.conditioning_set_area",
            rule=_combo_rule(
                from_type="comfy.ConditioningSetAreaPercentage",
                carrier="dinkster.conditioning_set_area",
                slot="units",
                variant="percent",
                inputs={
                    "conditioning": "conditioning",
                    "strength": "strength",
                    "units.width": "width",
                    "units.height": "height",
                    "units.x": "x",
                    "units.y": "y",
                },
            ),
            evidence=[
                "tests/test_generation_comfy_aliases.py::test_core_conditioning_combo_aliases_select_native_variants"
            ],
        ),
        _record(
            node_class="ConditioningSetAreaPercentageVideo",
            carrier="dinkster.conditioning_set_area",
            rule=_combo_rule(
                from_type="comfy.ConditioningSetAreaPercentageVideo",
                carrier="dinkster.conditioning_set_area",
                slot="units",
                variant="percent-video",
                inputs={
                    "conditioning": "conditioning",
                    "strength": "strength",
                    "units.width": "width",
                    "units.height": "height",
                    "units.temporal": "temporal",
                    "units.x": "x",
                    "units.y": "y",
                    "units.z": "z",
                },
            ),
            evidence=[
                "tests/test_native_arm.py::test_generation_conditioning_set_area_pixels_percent_and_video_match_pin"
            ],
        ),
        _record(
            node_class="ConditioningSetMask",
            carrier="dinkster.conditioning_set_mask",
            rule=_copy_rule(
                "comfy.ConditioningSetMask",
                "dinkster.conditioning_set_mask",
                "conditioning",
                "mask",
                "strength",
                "set_cond_area",
            ),
            evidence=[
                "tests/test_native_arm.py::test_generation_conditioning_set_mask_binds_mask_payload"
            ],
        ),
        _record(
            node_class="ConditioningSetTimestepRange",
            carrier="dinkster.conditioning_set_timestep_range",
            rule=_copy_rule(
                "comfy.ConditioningSetTimestepRange",
                "dinkster.conditioning_set_timestep_range",
                "conditioning",
                "start",
                "end",
            ),
            evidence=[
                "tests/test_native_arm.py::test_generation_conditioning_set_timestep_range_encodes_schedule"
            ],
        ),
        _record(
            node_class="ConditioningZeroOut",
            carrier="dinkster.conditioning_zero_out",
            rule=_copy_rule(
                "comfy.ConditioningZeroOut",
                "dinkster.conditioning_zero_out",
                "conditioning",
            ),
            evidence=[
                "tests/test_native_arm.py::test_generation_conditioning_zero_out_replays_pinned_comfyui_goldens"
            ],
        ),
        _record(
            node_class="ModelSamplingAuraFlow",
            carrier="dinkster.chroma_model_sampling",
            rule=_copy_rule(
                "comfy.ModelSamplingAuraFlow",
                "dinkster.chroma_model_sampling",
                "model",
                "shift",
                output_id="model",
            ),
            evidence=[
                "tests/test_native_arm.py::test_chroma_model_sampling_accepts_only_chroma_models",
                "tests/test_generation_comfy_aliases.py::test_chroma_workflow_aliases_preserve_sampling_and_latent_contracts",
            ],
        ),
        _record(
            node_class="EmptySD3LatentImage",
            carrier="dinkster.empty_sd3_latent_image",
            rule=_copy_rule(
                "comfy.EmptySD3LatentImage",
                "dinkster.empty_sd3_latent_image",
                "width",
                "height",
                "batch_size",
                output_id="latent",
            ),
            evidence=[
                "tests/test_native_arm.py::test_empty_sd3_latent_matches_reference_shape_and_metadata",
                "tests/test_generation_comfy_aliases.py::test_chroma_workflow_aliases_preserve_sampling_and_latent_contracts",
            ],
        ),
        _record(
            node_class="ChromaRadianceOptions",
            carrier="dinkster.chroma_radiance_options",
            rule=_copy_rule(
                "comfy.ChromaRadianceOptions",
                "dinkster.chroma_radiance_options",
                "model",
                "preserve_wrapper",
                "start_sigma",
                "end_sigma",
                "nerf_tile_size",
                "force_sequential_txt_ids",
                output_id="model",
                source_output_id="_0_MODEL_",
            ),
            evidence=[
                "tests/test_native_arm.py::test_chroma_radiance_options_noop_chaining_and_replacement",
                "tests/test_generation_comfy_aliases.py::test_chroma_workflow_aliases_preserve_sampling_and_latent_contracts",
            ],
            revision=CHROMA_RADIANCE_BASELINE[:8],
        ),
        _record(
            node_class="EmptyChromaRadianceLatentImage",
            carrier="dinkster.empty_chroma_radiance_latent_image",
            rule=_copy_rule(
                "comfy.EmptyChromaRadianceLatentImage",
                "dinkster.empty_chroma_radiance_latent_image",
                "width",
                "height",
                "batch_size",
                output_id="latent",
                source_output_id="_0_LATENT_",
            ),
            evidence=[
                "tests/test_native_arm.py::test_empty_chroma_radiance_latent_has_native_rgb_shape",
                "tests/test_generation_comfy_aliases.py::test_chroma_workflow_aliases_preserve_sampling_and_latent_contracts",
            ],
            revision=CHROMA_RADIANCE_BASELINE[:8],
        ),
        _record(
            node_class="ScheduledCFGGuider //Inspire",
            carrier="dinkster.scheduled_cfg_guider",
            rule=ReplacementRule(
                from_type="comfy.comfyui-inspire-pack.ScheduledCFGGuider //Inspire",
                cases=(
                    ReplacementCase.build(
                        "dinkster.scheduled_cfg_guider",
                        inputs={
                            input_id: MappingSource.copy(input_id)
                            for input_id in (
                                "model",
                                "positive",
                                "negative",
                                "sigmas",
                                "from_cfg",
                                "to_cfg",
                                "schedule",
                            )
                        },
                        outputs={"guider": "guider", "sigmas": "sigmas"},
                    ),
                ),
            ),
            source_pack="comfyui-inspire-pack",
            revision=INSPIRE_BASELINE,
            evidence=[
                "packages/dinkster-inference-torch/tests/test_scheduled_cfg.py::test_scheduled_cfg_matches_inspire_for_sd_and_flow",
            ],
        ),
        *_latent_records(),
        *_noise_records(),
        *_megapack_records(),
        *_loader_records(),
        *_trellis2_records(),
    ]
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(schema) for schema in source_schemas],
        "records": records,
    }


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--current-seedvr2":
        print(json.dumps(_build_current_seedvr2_registry(Path(sys.argv[2]).resolve())))
        return
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    current_configured = os.environ.get("CURRENT_COMFYUI_ROOT")
    if not current_configured:
        raise RuntimeError("CURRENT_COMFYUI_ROOT must point to the pinned current ComfyUI checkout")
    registry = build_registry(comfy_root.resolve())
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--current-seedvr2",
            str(Path(current_configured).resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    current = json.loads(completed.stdout)
    registry["sourceSchemas"] = [*registry["sourceSchemas"], *current["sourceSchemas"]]
    registry["records"] = [*registry["records"], *current["records"]]
    content = (json.dumps(registry, indent=2) + "\n").encode()
    OUT.write_bytes(content)
    print(f"{OUT}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
