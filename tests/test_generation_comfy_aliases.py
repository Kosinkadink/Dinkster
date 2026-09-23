"""The generation pack's maintained ComfyUI alias registry stays canonical."""

from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_nodes_generation import GENERATION_NODES
from dinkster_schema import (
    AssetWidget,
    ComboWidget,
    comfy_alias_registry_from_wire,
    schema_from_wire,
    schema_to_wire,
    validate_replacement_references,
)
from dinkster_schema.replace import rule_from_wire, rule_to_wire

ROOT = Path(__file__).parent.parent
ALIAS_PATH = ROOT / "packages" / "dinkster-nodes-generation" / "comfy-aliases.json"

CLIP_CARRIERS = frozenset(
    {
        "dinkster.clip_set_last_layer",
        "dinkster.t5_tokenizer_options",
        "dinkster.clip_text_encode_controlnet",
    }
)

CONDITIONING_CARRIERS = frozenset(
    {
        "dinkster.conditioning_merge",
        "dinkster.conditioning_scale",
        "dinkster.conditioning_set_area",
        "dinkster.conditioning_set_mask",
        "dinkster.conditioning_set_timestep_range",
        "dinkster.conditioning_zero_out",
    }
)

LATENT_CARRIERS = frozenset(
    {
        "dinkster.latent.combine",
        "dinkster.latent.mix",
        "dinkster.latent.multiply",
        "dinkster.latent.rotate",
        "dinkster.latent.flip",
        "dinkster.latent.crop",
        "dinkster.latent.resize",
        "dinkster.latent.resize_by",
        "dinkster.latent.composite",
        "dinkster.latent.composite_masked",
        "dinkster.latent.concat",
        "dinkster.latent.cut",
        "dinkster.latent.cut_to_batch",
        "dinkster.latent.from_batch",
        "dinkster.latent.repeat",
        "dinkster.latent.seed_behavior",
        "dinkster.latent.batch",
        "dinkster.latent.rebatch",
        "dinkster.latent.set_noise_mask",
        "dinkster.latent.replace_frames",
        "dinkster.latent.apply_operation",
        "dinkster.latent.operation_tonemap_reinhard",
        "dinkster.latent.operation_sharpen",
        "dinkster.latent.apply_operation_cfg",
        "dinkster.latent.generate_noise",
        "dinkster.latent.inject_noise",
    }
)

LOADER_CARRIERS = frozenset(
    {
        "dinkster.apply_lora_stack",
        "dinkster.load_checkpoint",
        "dinkster.load_checkpoint_stack",
        "dinkster.load_diffusion_model",
        "dinkster.load_lora",
    }
)

CONTROLNET_CARRIERS = frozenset(
    {
        "dinkster.load_controlnet",
        "dinkster.apply_controlnet",
        "dinkster.apply_controlnet_advanced",
        "dinkster.set_controlnet_union_type",
    }
)

SEEDVR2_WORKFLOW_CARRIERS = frozenset(
    {
        "dinkster.ksampler",
        "dinkster.vae_decode_tiled",
        "dinkster.vae_encode_tiled",
        "dinkster.seedvr2_preprocess",
        "dinkster.seedvr2_postprocess",
        "dinkster.seedvr2_conditioning",
        "dinkster.seedvr2_temporal_chunk",
        "dinkster.seedvr2_temporal_merge",
    }
)

CHROMA_WORKFLOW_CARRIERS = frozenset(
    {
        "dinkster.chroma_model_sampling",
        "dinkster.chroma_radiance_options",
        "dinkster.empty_chroma_radiance_latent_image",
        "dinkster.empty_sd3_latent_image",
    }
)

GUIDER_CARRIERS = frozenset({"dinkster.scheduled_cfg_guider"})

CUSTOM_SAMPLING_CARRIERS = frozenset(
    {
        "dinkster.sampler_custom",
        "dinkster.sampler_custom_advanced",
    }
)

TRELLIS2_WORKFLOW_CARRIERS = frozenset(
    {
        "dinkster.apply_texture_to_mesh",
        "dinkster.bake_ambient_occlusion",
        "dinkster.bake_normal_map_from_mesh",
        "dinkster.bake_texture_from_voxel",
        "dinkster.cfg_override",
        "dinkster.decimate_mesh",
        "dinkster.empty_trellis2_latent_structure",
        "dinkster.estimate_geometry",
        "dinkster.geometry_to_fov",
        "dinkster.get_mesh_info",
        "dinkster.image_crop_to_mask",
        "dinkster.ksampler",
        "dinkster.load_background_removal",
        "dinkster.load_geometry_model",
        "dinkster.file3d_to_mesh",
        "dinkster.mesh_to_model3d",
        "dinkster.model_sampling_sd3",
        "dinkster.paint_mesh",
        "dinkster.pixal3d_conditioning",
        "dinkster.preview_mask",
        "dinkster.remesh_mesh",
        "dinkster.remove_background",
        "dinkster.render_uv_atlas",
        "dinkster.rescale_cfg",
        "dinkster.smooth_mesh_normals",
        "dinkster.trellis2_conditioning",
        "dinkster.trellis2_shape_stage",
        "dinkster.trellis2_texture_stage",
        "dinkster.trellis2_upsample_stage",
        "dinkster.unwrap_mesh",
        "dinkster.vae_decode_shape_trellis",
        "dinkster.vae_decode_structure_trellis2",
        "dinkster.vae_decode_texture_trellis",
        "dinkster.voxel_to_mesh",
    }
)


def _registry() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(ALIAS_PATH.read_text(encoding="utf-8")))


def test_generation_comfy_aliases_use_the_canonical_wire_contract() -> None:
    registry = _registry()
    assert registry["format"] == "dinkster-comfy-alias/1"
    assert set(registry) == {"format", "sourceSchemas", "records"}
    comfy_alias_registry_from_wire(registry)

    source_schemas = [schema_from_wire(wire) for wire in registry["sourceSchemas"]]
    assert len(source_schemas) == len({schema.node_type for schema in source_schemas})
    assert all(not schema.replacements for schema in source_schemas)
    assert [schema_to_wire(schema) for schema in source_schemas] == registry["sourceSchemas"]

    records = registry["records"]
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))
    assert all(alias_id.startswith("comfy_alias:") for alias_id in ids)
    source_types = {schema.node_type for schema in source_schemas}
    assert {record["source"]["nodeType"] for record in records} == source_types

    for record in records:
        assert record["mappingKind"] == "op"
        assert record["source"]["nodeType"] == record["replacement"]["from"]
        rule = rule_from_wire(record["replacement"])
        assert rule_to_wire(rule) == record["replacement"]
        assert record["carrier"] in {case.to for case in rule.cases}


def test_generation_comfy_alias_replacements_validate_against_native_schemas() -> None:
    registry = _registry()
    native_schemas = {
        node.schema().node_type: node.schema()
        for node in (
            *FOUNDATION_NODES,
            *GENERATION_NODES,
        )
    }
    records_by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in registry["records"]:
        records_by_carrier[record["carrier"]].append(record)
    assert set(records_by_carrier) == (
        CLIP_CARRIERS
        | CONDITIONING_CARRIERS
        | LATENT_CARRIERS
        | LOADER_CARRIERS
        | CONTROLNET_CARRIERS
        | SEEDVR2_WORKFLOW_CARRIERS
        | CHROMA_WORKFLOW_CARRIERS
        | GUIDER_CARRIERS
        | CUSTOM_SAMPLING_CARRIERS
        | TRELLIS2_WORKFLOW_CARRIERS
    )

    source_schemas = [schema_from_wire(wire) for wire in registry["sourceSchemas"]]
    schemas = {**native_schemas, **{schema.node_type: schema for schema in source_schemas}}
    for carrier, records in records_by_carrier.items():
        rules = tuple(rule_from_wire(record["replacement"]) for record in records)
        schemas[carrier] = replace(native_schemas[carrier], replacements=rules)
    assert validate_replacement_references(schemas) == ()
    assert all(
        not alias.startswith("comfy_alias:")
        for schema in native_schemas.values()
        for alias in schema.aliases
    )


def test_generation_combo_aliases_select_materialized_target_inputs() -> None:
    registry = _registry()
    carriers = {record["carrier"] for record in registry["records"]}
    assert "dinkster.conditioning_merge" in carriers
    assert "dinkster.conditioning_set_area" in carriers
    native_schemas = {node.schema().node_type: node.schema() for node in GENERATION_NODES}
    assert native_schemas["dinkster.conditioning_merge"].combos
    assert native_schemas["dinkster.conditioning_set_area"].combos
    combo_records = [
        record
        for record in registry["records"]
        if record["carrier"] in {"dinkster.conditioning_merge", "dinkster.conditioning_set_area"}
    ]
    assert all(
        "slotVariants" in case
        for record in combo_records
        for case in record["replacement"]["cases"]
        if case["to"] in {"dinkster.conditioning_merge", "dinkster.conditioning_set_area"}
    )


def test_seedvr2_workflow_aliases_cover_current_core_surface() -> None:
    records = {record["source"]["nodeClass"]: record for record in _registry()["records"]}
    expected = {
        "KSampler": "dinkster.ksampler",
        "UNETLoader": "dinkster.load_diffusion_model",
        "VAEEncodeTiled": "dinkster.vae_encode_tiled",
        "VAEDecodeTiled": "dinkster.vae_decode_tiled",
        "SeedVR2Preprocess": "dinkster.seedvr2_preprocess",
        "SeedVR2PostProcessing": "dinkster.seedvr2_postprocess",
        "SeedVR2Conditioning": "dinkster.seedvr2_conditioning",
        "SeedVR2TemporalChunk": "dinkster.seedvr2_temporal_chunk",
        "SeedVR2TemporalMerge": "dinkster.seedvr2_temporal_merge",
    }
    assert {name: records[name]["carrier"] for name in expected} == expected
    assert {records[name]["source"]["revision"] for name in expected} == {"b78cec87"}

    sampler = records["KSampler"]["replacement"]["cases"][0]
    sampler_map = sampler["inputs"]["sampler_name"]["transform"]["map"]
    scheduler_map = sampler["inputs"]["scheduler"]["transform"]["map"]
    assert sampler_map["euler"] == "dinkster.euler"
    assert sampler_map["cfgpp_ud10_ab"] == "dinkster.cfgpp_ud10_ab"
    assert scheduler_map["simple"] == "dinkster.simple"

    loader = records["UNETLoader"]["replacement"]["cases"][0]
    assert loader["inputs"]["diffusion_model"] == {"kind": "copy", "input": "unet_name"}
    assert loader["inputs"]["weight_dtype"] == {"kind": "copy", "input": "weight_dtype"}

    chunk_cases = records["SeedVR2TemporalChunk"]["replacement"]["cases"]
    assert chunk_cases[0]["when"] == {
        "kind": "valueEquals",
        "input": "chunking_mode",
        "value": "manual",
    }
    assert chunk_cases[0]["slotVariants"] == {"chunking_mode": "manual"}
    assert chunk_cases[1]["slotVariants"] == {"chunking_mode": "auto"}


def test_custom_sampling_aliases_copy_every_shared_input() -> None:
    records = {record["source"]["nodeClass"]: record for record in _registry()["records"]}
    expected = {
        "SamplerCustom": (
            "dinkster.sampler_custom",
            (
                "model",
                "add_noise",
                "noise_seed",
                "cfg",
                "positive",
                "negative",
                "sampler",
                "sigmas",
                "latent_image",
            ),
        ),
        "SamplerCustomAdvanced": (
            "dinkster.sampler_custom_advanced",
            ("noise", "guider", "sampler", "sigmas", "latent_image"),
        ),
    }
    source_schemas = {
        schema.node_type: schema for schema in map(schema_from_wire, _registry()["sourceSchemas"])
    }
    for node_class, (carrier, input_ids) in expected.items():
        record = records[node_class]
        assert record["carrier"] == carrier
        assert record["source"] == {
            "pack": "comfy-core",
            "nodeClass": node_class,
            "nodeType": f"comfy.{node_class}",
            "revision": "b78cec87",
        }
        case = record["replacement"]["cases"][0]
        assert case["to"] == carrier
        assert case["inputs"] == {
            input_id: {"kind": "copy", "input": input_id} for input_id in input_ids
        }
        assert case["outputs"] == {"output": "output", "denoised_output": "denoised_output"}
        # The pinned source schema must expose exactly the ports the alias
        # maps, so no Comfy input can be silently left unmapped.
        source = source_schemas[f"comfy.{node_class}"]
        assert tuple(item.id for item in source.inputs) == input_ids
        assert tuple(item.id for item in source.outputs) == (
            "output",
            "denoised_output",
        )


def test_controlnet_native_owner_schemas_match_source_contracts() -> None:
    schemas = {node.schema().node_type: node.schema() for node in GENERATION_NODES}
    assert CONTROLNET_CARRIERS <= schemas.keys()

    loader = schemas["dinkster.load_controlnet"]
    assert loader.aliases == ("ControlNetLoader",)
    assert [(item.id, item.type.types) for item in loader.inputs] == [
        ("control_net_name", ("dinkster.asset",))
    ]
    assert isinstance(loader.inputs[0].widget, AssetWidget)
    assert loader.inputs[0].widget.kind == "model/controlnet"
    assert [(item.id, item.type.types) for item in loader.outputs] == [
        ("control_net", ("comfy.CONTROL_NET",))
    ]

    apply = schemas["dinkster.apply_controlnet"]
    assert apply.aliases == ("ControlNetApply",)
    assert [(item.id, item.type.types) for item in apply.inputs] == [
        ("conditioning", ("dinkster.conditioning",)),
        ("control_net", ("comfy.CONTROL_NET",)),
        ("image", ("dinkster.image",)),
        ("strength", ("core.float",)),
    ]
    assert apply.inputs[-1].default == 1.0
    assert [(item.id, item.type.types) for item in apply.outputs] == [
        ("conditioning", ("dinkster.conditioning",))
    ]

    advanced = schemas["dinkster.apply_controlnet_advanced"]
    assert advanced.aliases == ("ControlNetApplyAdvanced",)
    assert [item.id for item in advanced.inputs] == [
        "positive",
        "negative",
        "control_net",
        "image",
        "strength",
        "start_percent",
        "end_percent",
        "vae",
    ]
    assert [item.default for item in advanced.inputs[4:7]] == [1.0, 0.0, 1.0]
    assert advanced.inputs[-1].required is False
    assert advanced.inputs[-1].type.types == ("dinkster.vae",)
    assert [item.id for item in advanced.outputs] == ["positive", "negative"]

    union = schemas["dinkster.set_controlnet_union_type"]
    assert union.aliases == ("SetUnionControlNetType",)
    assert union.inputs[0].type.types == ("comfy.CONTROL_NET",)
    assert union.inputs[1].default == "auto"
    assert isinstance(union.inputs[1].widget, ComboWidget)
    assert tuple(union.inputs[1].widget.options) == (
        "auto",
        "openpose",
        "depth",
        "hed/pidi/scribble/ted",
        "canny/lineart/anime_lineart/mlsd",
        "normal",
        "segment",
        "tile",
        "repaint",
    )
    assert union.outputs[0].type.types == ("comfy.CONTROL_NET",)


def test_controlnet_aliases_preserve_schema_wiring() -> None:
    registry = _registry()
    source_schemas = {
        schema.node_type: schema for schema in map(schema_from_wire, registry["sourceSchemas"])
    }
    records = {record["source"]["nodeClass"]: record for record in registry["records"]}
    expected = {
        "ControlNetLoader": (
            "dinkster.load_controlnet",
            ("control_net_name",),
            {"control_net": "control_net"},
        ),
        "ControlNetApply": (
            "dinkster.apply_controlnet",
            ("conditioning", "control_net", "image", "strength"),
            {"conditioning": "conditioning"},
        ),
        "ControlNetApplyAdvanced": (
            "dinkster.apply_controlnet_advanced",
            (
                "positive",
                "negative",
                "control_net",
                "image",
                "strength",
                "start_percent",
                "end_percent",
                "vae",
            ),
            {"positive": "positive", "negative": "negative"},
        ),
        "SetUnionControlNetType": (
            "dinkster.set_controlnet_union_type",
            ("control_net", "type"),
            {"control_net": "_0_CONTROL_NET_"},
        ),
    }
    for node_class, (carrier, input_ids, outputs) in expected.items():
        record = records[node_class]
        assert record["carrier"] == carrier
        case = record["replacement"]["cases"][0]
        assert case["to"] == carrier
        assert case["inputs"] == {
            input_id: {"kind": "copy", "input": input_id} for input_id in input_ids
        }
        assert case["outputs"] == outputs

    assert source_schemas["comfy.ControlNetApply"].inputs[-1].default == 1.0
    advanced = source_schemas["comfy.ControlNetApplyAdvanced"]
    assert [item.default for item in advanced.inputs[4:7]] == [1.0, 0.0, 1.0]
    union = source_schemas["comfy.SetUnionControlNetType"]
    assert union.inputs[1].default == "auto"


def test_core_conditioning_combo_aliases_select_native_variants() -> None:
    records = {record["id"]: record for record in _registry()["records"]}
    expected = {
        "ConditioningCombine": (
            "dinkster.conditioning_merge",
            {"mode": "combine"},
            {
                "mode.inputs.conditioning_1": "conditioning_1",
                "mode.inputs.conditioning_2": "conditioning_2",
            },
        ),
        "ConditioningAverage": (
            "dinkster.conditioning_merge",
            {"mode": "average"},
            {
                "mode.conditioning_to": "conditioning_to",
                "mode.conditioning_from": "conditioning_from",
                "mode.conditioning_to_strength": "conditioning_to_strength",
            },
        ),
        "ConditioningConcat": (
            "dinkster.conditioning_merge",
            {"mode": "concat"},
            {
                "mode.conditioning_to": "conditioning_to",
                "mode.conditioning_from": "conditioning_from",
            },
        ),
        "ConditioningSetArea": (
            "dinkster.conditioning_set_area",
            {"units": "pixels"},
            {
                "conditioning": "conditioning",
                "strength": "strength",
                "units.width": "width",
                "units.height": "height",
                "units.x": "x",
                "units.y": "y",
            },
        ),
        "ConditioningSetAreaPercentage": (
            "dinkster.conditioning_set_area",
            {"units": "percent"},
            {
                "conditioning": "conditioning",
                "strength": "strength",
                "units.width": "width",
                "units.height": "height",
                "units.x": "x",
                "units.y": "y",
            },
        ),
        "ConditioningSetAreaPercentageVideo": (
            "dinkster.conditioning_set_area",
            {"units": "percent-video"},
            {
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
    }
    for node_class, (carrier, slot_variants, mappings) in expected.items():
        record = records[f"comfy_alias:comfy-core/{node_class}"]
        assert record["carrier"] == carrier
        case = record["replacement"]["cases"][0]
        assert case["to"] == carrier
        assert case["slotVariants"] == slot_variants
        assert case["outputs"] == {"conditioning": "conditioning"}
        assert case["inputs"] == {
            target: {"kind": "copy", "input": source} for target, source in mappings.items()
        }
        assert "inputFamilies" not in case


def test_trellis2_official_workflow_surface_is_maintained() -> None:
    execution_nodes = {
        "ApplyTextureToMesh",
        "BakeAmbientOcclusion",
        "BakeNormalMapFromMesh",
        "BakeTextureFromVoxel",
        "CFGOverride",
        "CLIPVisionLoader",
        "ComfySwitchNode",
        "DecimateMesh",
        "EmptyTrellis2LatentStructure",
        "GetMeshInfo",
        "ImageCropToMask",
        "KSampler",
        "LoadBackgroundRemovalModel",
        "LoadImage",
        "LoadMoGeModel",
        "MaskPreview",
        "MeshSmoothNormals",
        "MeshToFile3D",
        "ModelSamplingSD3",
        "MoGeGeometryToFOV",
        "MoGeInference",
        "PaintMesh",
        "Pixal3DConditioning",
        "Preview3DAdvanced",
        "PreviewImage",
        "PrimitiveBoolean",
        "PrimitiveInt",
        "RemoveBackground",
        "RemeshMesh",
        "RenderUVAtlas",
        "RescaleCFG",
        "Save3DAdvanced",
        "Trellis2Conditioning",
        "Trellis2ShapeStage",
        "Trellis2TextureStage",
        "Trellis2UpsampleStage",
        "UNETLoader",
        "UnwrapMesh",
        "VAELoader",
        "VaeDecodeShapeTrellis",
        "VaeDecodeStructureTrellis2",
        "VaeDecodeTextureTrellis",
        "VoxelToMesh",
    }
    maintained = {
        record["source"]["nodeClass"]
        for path in ROOT.glob("packages/*/comfy-aliases.json")
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]
        if record["source"]["pack"] == "comfy-core"
    }
    assert execution_nodes <= maintained


def test_generation_comfy_alias_confidence_has_pinned_evidence() -> None:
    registry = _registry()
    expected = {
        "comfy-core": ({"b78cec87", "8a33128f", "95539f56"}, "exact"),
        "comfyui-kjnodes": (
            {
                "827fe6ee0ed7348d8daa988ed852bedf1272380c",
                "3f20054214fec9f9234fd3841ae6f1e4287948f6",
            },
            "grouped",
        ),
        "comfyui_essentials": ({"9d9f4bedfc9f0321c19faf71855e228c93bd0dc9"}, "grouped"),
        "rgthree-comfy": ({"35c9f1e186603ba312d3b15350e89aa50b860ee6"}, "grouped"),
        "comfyui-easy-use": ({"58e077a7435631301cf7443412515cf958e7f3d1"}, "grouped"),
        "was-node-suite-comfyui": (
            {"44de705818d4663fefefde57ffe0ea5a9ea39df4"},
            "grouped",
        ),
        "comfyui-custom-scripts": (
            {"609f3afaa74b2f88ef9ce8d939626065e3247469"},
            "grouped",
        ),
        "efficiency-nodes-comfyui": (
            {"4579b7d6076b2870998a08f5d37883fbc8261ff2"},
            "grouped",
        ),
        "comfyui-inspire-pack": (
            {"6b2ca017a168bcdba5f22c258b3b86c5c76470ca"},
            "exact",
        ),
    }
    # Single-node replacements backed by bit-exact goldens carry "exact"
    # even when the rest of their pack's records are grouped graph rewrites.
    exact_tier_ids = {
        "comfy_alias:comfyui-kjnodes/GenerateNoise",
        "comfy_alias:comfyui-kjnodes/InjectNoiseToLatent",
    }
    for record in registry["records"]:
        source = record["source"]
        revisions, tier = expected[source["pack"]]
        if record["id"] in exact_tier_ids:
            tier = "exact"
        assert source["revision"] in revisions
        confidence = record["confidence"]
        assert confidence["tier"] == tier
        assert confidence["evidence"]
        assert "tolerances" not in confidence
        for selector in confidence["evidence"]:
            path_text, _, test_name = selector.partition("::")
            path = ROOT / path_text
            assert path.is_file()
            assert test_name and f"def {test_name}(" in path.read_text(encoding="utf-8")


def test_chroma_workflow_aliases_preserve_sampling_and_latent_contracts() -> None:
    records = {record["id"]: record for record in _registry()["records"]}
    sampling = records["comfy_alias:comfy-core/ModelSamplingAuraFlow"]
    assert sampling["source"]["revision"] == "b78cec87"
    assert sampling["replacement"]["cases"] == [
        {
            "to": "dinkster.chroma_model_sampling",
            "inputs": {name: {"kind": "copy", "input": name} for name in ("model", "shift")},
            "outputs": {"model": "model"},
        }
    ]

    empty = records["comfy_alias:comfy-core/EmptySD3LatentImage"]
    assert empty["source"]["revision"] == "b78cec87"
    assert empty["replacement"]["cases"] == [
        {
            "to": "dinkster.empty_sd3_latent_image",
            "inputs": {
                name: {"kind": "copy", "input": name} for name in ("width", "height", "batch_size")
            },
            "outputs": {"latent": "latent"},
        }
    ]

    radiance_options = records["comfy_alias:comfy-core/ChromaRadianceOptions"]
    assert radiance_options["source"]["revision"] == "8a33128f"
    assert radiance_options["replacement"]["cases"] == [
        {
            "to": "dinkster.chroma_radiance_options",
            "inputs": {
                name: {"kind": "copy", "input": name}
                for name in (
                    "model",
                    "preserve_wrapper",
                    "start_sigma",
                    "end_sigma",
                    "nerf_tile_size",
                    "force_sequential_txt_ids",
                )
            },
            "outputs": {"model": "_0_MODEL_"},
        }
    ]

    radiance_latent = records["comfy_alias:comfy-core/EmptyChromaRadianceLatentImage"]
    assert radiance_latent["source"]["revision"] == "8a33128f"
    assert radiance_latent["replacement"]["cases"] == [
        {
            "to": "dinkster.empty_chroma_radiance_latent_image",
            "inputs": {
                name: {"kind": "copy", "input": name} for name in ("width", "height", "batch_size")
            },
            "outputs": {"latent": "_0_LATENT_"},
        }
    ]


def test_conditioning_megapack_aliases_preserve_grouped_graph_shapes() -> None:
    records = {record["id"]: record for record in _registry()["records"]}
    megapack_ids = {
        alias_id
        for alias_id, record in records.items()
        if record["source"]["pack"] != "comfy-core"
        and (
            record["source"]["nodeClass"].startswith("Conditioning")
            or record["source"]["nodeClass"] == "SD3NegativeConditioning+"
        )
    }
    assert megapack_ids == {
        "comfy_alias:comfyui-kjnodes/ConditioningMultiCombine",
        "comfy_alias:comfyui-kjnodes/ConditioningSetMaskAndCombine",
        "comfy_alias:comfyui-kjnodes/ConditioningSetMaskAndCombine3",
        "comfy_alias:comfyui-kjnodes/ConditioningSetMaskAndCombine4",
        "comfy_alias:comfyui-kjnodes/ConditioningSetMaskAndCombine5",
        "comfy_alias:comfyui_essentials/ConditioningCombineMultiple+",
        "comfy_alias:comfyui_essentials/SD3NegativeConditioning+",
    }

    multi_cases = records["comfy_alias:comfyui-kjnodes/ConditioningMultiCombine"]["replacement"][
        "cases"
    ]
    assert len(multi_cases) == 38
    assert sum("when" not in case for case in multi_cases) == 1
    for case in multi_cases:
        count = case["nodes"]["count"]["values"]["value"]
        operation = case["slotVariants"]["mode"]
        assert 2 <= count <= 20
        assert operation in {"combine", "concat"}
        assert case["outputs"] == {
            "conditioning": "combined",
            "count:value": "inputcount",
        }
        copied_conditioning = [
            mapping["input"]
            for mapping in case["inputs"].values()
            if mapping.get("kind") == "copy" and mapping["input"].startswith("conditioning_")
        ]
        assert sorted(copied_conditioning) == sorted(
            f"conditioning_{index}" for index in range(1, count + 1)
        )
        assert all(choice == operation for choice in case["slotVariants"].values())
        merge_helpers = {
            local_id
            for local_id, node in case["nodes"].items()
            if node["type"] == "dinkster.conditioning_merge"
        }
        assert all(f"{local_id}:mode" in case["slotVariants"] for local_id in merge_helpers)
        if operation == "combine":
            assert len(merge_helpers) == max(0, (count - 2) // 7)
            assert case.get("links", []) == [
                *(
                    {
                        "from": f"merge_{index}:conditioning",
                        "to": f"merge_{index + 1}:mode.inputs.conditioning_1",
                    }
                    for index in range(1, len(merge_helpers))
                ),
                *(
                    [
                        {
                            "from": f"merge_{len(merge_helpers)}:conditioning",
                            "to": "mode.inputs.conditioning_1",
                        }
                    ]
                    if merge_helpers
                    else []
                ),
            ]
        else:
            assert len(merge_helpers) == max(0, count - 2)
            assert case.get("links", []) == [
                *(
                    {
                        "from": f"merge_{index}:conditioning",
                        "to": f"merge_{index + 1}:mode.conditioning_to",
                    }
                    for index in range(1, len(merge_helpers))
                ),
                *(
                    [
                        {
                            "from": f"merge_{len(merge_helpers)}:conditioning",
                            "to": "mode.conditioning_to",
                        }
                    ]
                    if merge_helpers
                    else []
                ),
            ]

    for count in range(2, 6):
        suffix = "" if count == 2 else str(count)
        case = records[f"comfy_alias:comfyui-kjnodes/ConditioningSetMaskAndCombine{suffix}"][
            "replacement"
        ]["cases"][0]
        assert case["outputs"] == {
            "conditioning": "combined_positive",
            "negative_merge:conditioning": "combined_negative",
        }
        assert case["slotVariants"] == {
            "mode": "combine",
            "negative_merge:mode": "combine",
        }
        assert case["links"] == [
            link
            for index in range(1, count + 1)
            for polarity in ("positive", "negative")
            for link in (
                {
                    "from": f"mask_{index}_route:value",
                    "to": f"{polarity}_{index}:mask",
                },
                {
                    "from": f"mask_{index}_strength_route:value",
                    "to": f"{polarity}_{index}:strength",
                },
                {
                    "from": "set_cond_area_route:value",
                    "to": f"{polarity}_{index}:set_cond_area",
                },
                {
                    "from": f"{polarity}_{index}:conditioning",
                    "to": (
                        f"negative_merge:mode.inputs.conditioning_{index}"
                        if polarity == "negative"
                        else f"mode.inputs.conditioning_{index}"
                    ),
                },
            )
        ]
        route_sources = {
            "set_cond_area_route": "set_cond_area",
            **{
                f"mask_{index}_{suffix}": f"mask_{index}{source_suffix}"
                for index in range(1, count + 1)
                for suffix, source_suffix in (("route", ""), ("strength_route", "_strength"))
            },
        }
        for local_id, source_id in route_sources.items():
            assert case["nodes"][local_id] == {
                "type": "dinkster.route.gate",
                "values": {"condition": True},
            }
            assert case["inputs"][f"{local_id}:value"] == {
                "kind": "copy",
                "input": source_id,
            }
        for index in range(1, count + 1):
            for polarity in ("positive", "negative"):
                local_id = f"{polarity}_{index}"
                assert case["nodes"][local_id]["type"] == "dinkster.conditioning_set_mask"
                assert case["inputs"][f"{local_id}:conditioning"] == {
                    "kind": "copy",
                    "input": local_id,
                }

    essentials = records["comfy_alias:comfyui_essentials/ConditioningCombineMultiple+"][
        "replacement"
    ]["cases"]
    expected_members = (
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4),
        (1, 2, 3, 5),
        (1, 2, 4, 5),
        (1, 2, 3),
        (1, 2, 4),
        (1, 2, 5),
        (1, 2),
    )
    for case_index, (case, expected) in enumerate(zip(essentials, expected_members, strict=True)):
        assert case["slotVariants"] == {"mode": "combine"}
        assert tuple(mapping["input"] for mapping in case["inputs"].values()) == tuple(
            f"conditioning_{index}" for index in expected
        )
        if case_index == len(essentials) - 1:
            assert "when" not in case
            continue
        predicates = case["when"]["of"]
        assert tuple(predicate["of"][0]["input"] for predicate in predicates) == tuple(
            f"conditioning_{index}" for index in expected[2:]
        )
        assert all(
            predicate["of"]
            == [
                {"kind": "inputConnected", "input": predicate["of"][0]["input"]},
                {"kind": "valuePresent", "input": predicate["of"][0]["input"]},
            ]
            for predicate in predicates
        )

    sd3_cases = records["comfy_alias:comfyui_essentials/SD3NegativeConditioning+"]["replacement"][
        "cases"
    ]
    assert sd3_cases[0]["to"] == "dinkster.conditioning_zero_out"
    assert sd3_cases[0]["when"] == {"kind": "valueEquals", "input": "end", "value": 0}
    assert sd3_cases[1]["nodes"] == {
        "zero": {"type": "dinkster.conditioning_zero_out"},
        "head": {"type": "dinkster.conditioning_set_timestep_range"},
        "tail": {"type": "dinkster.conditioning_set_timestep_range"},
        "conditioning_route": {
            "type": "dinkster.route.gate",
            "values": {"condition": True},
        },
        "end_route": {
            "type": "dinkster.route.gate",
            "values": {"condition": True},
        },
    }
    assert sd3_cases[1]["slotVariants"] == {"mode": "combine"}
    assert sd3_cases[1]["inputs"] == {
        "conditioning_route:value": {"kind": "copy", "input": "conditioning"},
        "end_route:value": {"kind": "copy", "input": "end"},
        "head:start": {"kind": "constant", "value": 0.0},
        "tail:end": {"kind": "constant", "value": 1.0},
    }
    assert sd3_cases[1]["links"] == [
        {"from": "conditioning_route:value", "to": "zero:conditioning"},
        {"from": "conditioning_route:value", "to": "head:conditioning"},
        {"from": "zero:conditioning", "to": "tail:conditioning"},
        {"from": "end_route:value", "to": "head:end"},
        {"from": "end_route:value", "to": "tail:start"},
        {"from": "tail:conditioning", "to": "mode.inputs.conditioning_1"},
        {"from": "head:conditioning", "to": "mode.inputs.conditioning_2"},
    ]


def test_loader_stack_aliases_are_conservative_and_ordered() -> None:
    registry = _registry()
    records = {record["id"]: record for record in registry["records"]}
    loader_ids = {
        alias_id for alias_id, record in records.items() if record["carrier"] in LOADER_CARRIERS
    }
    assert loader_ids == {
        "comfy_alias:comfy-core/UNETLoader",
        "comfy_alias:comfyui-kjnodes/CheckpointLoaderKJ",
        "comfy_alias:comfyui-kjnodes/DiffusionModelLoaderKJ",
        "comfy_alias:rgthree-comfy/Lora Loader Stack (rgthree)",
        "comfy_alias:comfyui-easy-use/easy fullLoader",
        "comfy_alias:comfyui-easy-use/easy a1111Loader",
        "comfy_alias:comfyui-easy-use/easy comfyLoader",
        "comfy_alias:comfyui-easy-use/easy fluxLoader",
        "comfy_alias:comfyui-easy-use/easy hunyuanDiTLoader",
        "comfy_alias:was-node-suite-comfyui/Checkpoint Loader (Simple)",
        "comfy_alias:was-node-suite-comfyui/Load Lora",
        "comfy_alias:was-node-suite-comfyui/Lora Loader",
        "comfy_alias:comfyui-custom-scripts/LoraLoader|pysssss",
        "comfy_alias:comfyui-custom-scripts/CheckpointLoader|pysssss",
        "comfy_alias:efficiency-nodes-comfyui/Efficient Loader",
    }

    def is_refusal(case: dict[str, Any]) -> bool:
        mappings = list(case.get("inputs", {}).values())
        for family in case.get("inputFamilies", {}).values():
            for member in family.get("members", []):
                mappings.extend(member["inputs"].values())
        return any(
            mapping.get("transform") == {"kind": "enumRename", "map": {}} for mapping in mappings
        )

    for alias_id in (
        "comfy_alias:comfyui-kjnodes/CheckpointLoaderKJ",
        "comfy_alias:comfyui-kjnodes/DiffusionModelLoaderKJ",
        "comfy_alias:was-node-suite-comfyui/Load Lora",
        "comfy_alias:was-node-suite-comfyui/Lora Loader",
    ):
        cases = records[alias_id]["replacement"]["cases"]
        assert "when" in cases[0]
        assert "when" not in cases[-1]
        assert is_refusal(cases[-1])

    kj_checkpoint = records["comfy_alias:comfyui-kjnodes/CheckpointLoaderKJ"]["replacement"][
        "cases"
    ][0]
    assert {
        predicate["input"]: predicate.get("value") for predicate in kj_checkpoint["when"]["of"]
    } == {
        "weight_dtype": "default",
        "compute_dtype": "default",
        "patch_cublaslinear": False,
        "sage_attention": "disabled",
        "enable_fp16_accumulation": False,
    }

    for node_class in (
        "easy fullLoader",
        "easy a1111Loader",
        "easy comfyLoader",
        "easy fluxLoader",
        "easy hunyuanDiTLoader",
        "Efficient Loader",
    ):
        pack = (
            "efficiency-nodes-comfyui" if node_class == "Efficient Loader" else "comfyui-easy-use"
        )
        cases = records[f"comfy_alias:{pack}/{node_class}"]["replacement"]["cases"]
        assert len(cases) == 3
        assert "inputFamilies" not in cases[0]
        assert len(cases[1]["inputFamilies"]["loras"]["members"]) == 1
        assert is_refusal(cases[2]) and "when" not in cases[2]
        assert all("when" in case for case in cases[:2])

    rgthree = records["comfy_alias:rgthree-comfy/Lora Loader Stack (rgthree)"]["replacement"][
        "cases"
    ]
    assert len(rgthree) == 17
    assert is_refusal(rgthree[0]) and "when" in rgthree[0]
    assert is_refusal(rgthree[-1]) and "when" not in rgthree[-1]

    def slot_condition(index: int, active: bool) -> dict[str, Any]:
        lora = f"lora_{index:02}"
        strength = f"strength_{index:02}"
        if active:
            return {
                "kind": "all",
                "of": [
                    {
                        "kind": "any",
                        "of": [
                            {"kind": "inputConnected", "input": lora},
                            {"kind": "valuePresent", "input": lora},
                        ],
                    },
                    {
                        "kind": "not",
                        "of": {"kind": "valueEquals", "input": lora, "value": "None"},
                    },
                    {
                        "kind": "not",
                        "of": {"kind": "inputConnected", "input": strength},
                    },
                    {
                        "kind": "not",
                        "of": {"kind": "valueEquals", "input": strength, "value": 0.0},
                    },
                ],
            }
        return {
            "kind": "any",
            "of": [
                {"kind": "valueEquals", "input": lora, "value": "None"},
                {
                    "kind": "all",
                    "of": [
                        {
                            "kind": "not",
                            "of": {"kind": "inputConnected", "input": strength},
                        },
                        {"kind": "valueEquals", "input": strength, "value": 0.0},
                    ],
                },
            ],
        }

    for mask, case in enumerate(rgthree[:-1]):
        expected_suffixes = [
            f"lora_{index:02}" for index in range(1, 5) if mask & (1 << (index - 1))
        ]
        assert case["when"] == {
            "kind": "all",
            "of": [slot_condition(index, bool(mask & (1 << (index - 1)))) for index in range(1, 5)],
        }
        if mask == 0:
            continue
        members = case["inputFamilies"]["loras"]["members"]
        assert [member["suffix"] for member in members] == expected_suffixes
        predicate = json.dumps(case["when"], sort_keys=True)
        for suffix, member in zip(expected_suffixes, members, strict=True):
            assert f"strength_{suffix[-2:]}" in predicate
            assert member["inputs"]["lora"] == {"kind": "copy", "input": suffix}
            for strength in ("strength_model", "strength_clip"):
                assert member["inputs"][strength] == {
                    "kind": "value",
                    "input": f"strength_{suffix[-2:]}",
                }

    was_simple = records["comfy_alias:was-node-suite-comfyui/Checkpoint Loader (Simple)"][
        "replacement"
    ]["cases"][0]
    assert was_simple["outputs"] == {"model": "MODEL", "clip": "CLIP", "vae": "VAE"}

    for node_class, carrier_outputs in (
        ("LoraLoader|pysssss", {"model": "MODEL", "clip": "CLIP"}),
        (
            "CheckpointLoader|pysssss",
            {"model": "MODEL", "clip": "CLIP", "vae": "VAE"},
        ),
    ):
        case = records[f"comfy_alias:comfyui-custom-scripts/{node_class}"]["replacement"]["cases"][
            0
        ]
        assert case["nodes"] == {"example": {"type": "dinkster.string"}}
        assert case["inputs"]["example:value"] == {"kind": "copy", "input": "prompt"}
        assert case["outputs"] == {**carrier_outputs, "example:value": "example"}


def test_latent_aliases_map_arithmetic_and_mix_operations() -> None:
    records = {record["id"]: record for record in _registry()["records"]}

    for alias_id, operation in (
        ("comfy_alias:comfy-core/LatentAdd", "add"),
        ("comfy_alias:comfy-core/LatentSubtract", "subtract"),
    ):
        case = records[alias_id]["replacement"]["cases"][0]
        assert case["to"] == "dinkster.latent.combine"
        assert case["inputs"]["operation"] == {"kind": "constant", "value": operation}
        assert case["inputs"]["samples1"] == {"kind": "copy", "input": "samples1"}
        assert case["inputs"]["samples2"] == {"kind": "copy", "input": "samples2"}

    interpolate = records["comfy_alias:comfy-core/LatentInterpolate"]["replacement"]["cases"][0]
    assert interpolate["to"] == "dinkster.latent.mix"
    assert interpolate["inputs"]["operation"] == {"kind": "constant", "value": "interpolate"}
    assert interpolate["inputs"]["factor"] == {"kind": "copy", "input": "ratio"}

    blend = records["comfy_alias:comfy-core/LatentBlend"]["replacement"]["cases"][0]
    assert blend["to"] == "dinkster.latent.mix"
    assert blend["inputs"]["operation"] == {"kind": "constant", "value": "blend"}
    assert blend["inputs"]["factor"] == {"kind": "copy", "input": "blend_factor"}


def test_latent_aliases_rename_geometry_enums_and_composite_inputs() -> None:
    records = {record["id"]: record for record in _registry()["records"]}

    rotate = records["comfy_alias:comfy-core/LatentRotate"]["replacement"]["cases"][0]
    assert rotate["inputs"]["angle"] == {
        "kind": "value",
        "input": "rotation",
        "transform": {
            "kind": "enumRename",
            "map": {
                "none": "none",
                "90 degrees": "90",
                "180 degrees": "180",
                "270 degrees": "270",
            },
        },
    }

    flip = records["comfy_alias:comfy-core/LatentFlip"]["replacement"]["cases"][0]
    assert flip["inputs"]["axis"] == {
        "kind": "value",
        "input": "flip_method",
        "transform": {
            "kind": "enumRename",
            "map": {
                "x-axis: vertically": "vertical",
                "y-axis: horizontally": "horizontal",
            },
        },
    }

    composite = records["comfy_alias:comfy-core/LatentComposite"]["replacement"]["cases"][0]
    assert composite["inputs"]["destination"] == {"kind": "copy", "input": "samples_to"}
    assert composite["inputs"]["source"] == {"kind": "copy", "input": "samples_from"}
    assert composite["inputs"]["feather"] == {"kind": "copy", "input": "feather"}

    masked = records["comfy_alias:comfy-core/LatentCompositeMasked"]["replacement"]["cases"][0]
    assert masked["inputs"]["mask"] == {"kind": "copy", "input": "mask"}
    assert masked["inputs"]["resize_source"] == {"kind": "copy", "input": "resize_source"}


def test_latent_batch_aliases_map_families_lists_and_renamed_widgets() -> None:
    records = {record["id"]: record for record in _registry()["records"]}

    batch = records["comfy_alias:comfy-core/LatentBatch"]["replacement"]["cases"][0]
    assert batch["to"] == "dinkster.latent.batch"
    assert batch["inputFamilies"]["latents"] == {
        "kind": "members",
        "members": [
            {"suffix": "1", "inputs": {"value": {"kind": "copy", "input": "samples1"}}},
            {"suffix": "2", "inputs": {"value": {"kind": "copy", "input": "samples2"}}},
        ],
    }
    assert batch["outputs"] == {"latent": "_0_LATENT_"}

    multi = records["comfy_alias:comfy-core/BatchLatentsNode"]["replacement"]["cases"][0]
    assert multi["to"] == "dinkster.latent.batch"
    assert multi["inputFamilies"]["latents"] == {
        "kind": "copy",
        "sourceFamily": "latents",
        "inputs": {"value": {"kind": "copy", "input": "latent"}},
    }

    rebatch = records["comfy_alias:comfy-core/RebatchLatents"]["replacement"]["cases"][0]
    assert rebatch["to"] == "dinkster.latent.rebatch"
    assert rebatch["nodes"] == {"batch_size": {"type": "std.list.element", "values": {"index": 0}}}
    assert rebatch["inputs"]["latents"] == {"kind": "copy", "input": "latents"}
    assert rebatch["inputs"]["batch_size:list"] == {"kind": "copy", "input": "batch_size"}
    assert rebatch["links"] == [{"from": "batch_size:item", "to": "batch_size"}]
    assert rebatch["outputs"] == {"latents": "_0_LATENT_"}

    seed_behavior = records["comfy_alias:comfy-core/LatentBatchSeedBehavior"]["replacement"][
        "cases"
    ][0]
    assert seed_behavior["to"] == "dinkster.latent.seed_behavior"
    assert seed_behavior["inputs"]["behavior"] == {"kind": "copy", "input": "seed_behavior"}


def test_latent_operation_aliases_map_plain_copies() -> None:
    records = {record["id"]: record for record in _registry()["records"]}

    apply = records["comfy_alias:comfy-core/LatentApplyOperation"]["replacement"]["cases"][0]
    assert apply["to"] == "dinkster.latent.apply_operation"
    assert apply["inputs"]["samples"] == {"kind": "copy", "input": "samples"}
    assert apply["inputs"]["operation"] == {"kind": "copy", "input": "operation"}
    assert apply["outputs"] == {"latent": "_0_LATENT_"}

    tonemap = records["comfy_alias:comfy-core/LatentOperationTonemapReinhard"]["replacement"][
        "cases"
    ][0]
    assert tonemap["to"] == "dinkster.latent.operation_tonemap_reinhard"
    assert tonemap["inputs"] == {"multiplier": {"kind": "copy", "input": "multiplier"}}
    assert tonemap["outputs"] == {"operation": "_0_LATENT_OPERATION_"}

    sharpen = records["comfy_alias:comfy-core/LatentOperationSharpen"]["replacement"]["cases"][0]
    assert sharpen["to"] == "dinkster.latent.operation_sharpen"
    assert sharpen["inputs"] == {
        name: {"kind": "copy", "input": name} for name in ("sharpen_radius", "sigma", "alpha")
    }
    assert sharpen["outputs"] == {"operation": "_0_LATENT_OPERATION_"}


def test_every_latent_node_carries_an_executable_comfy_alias() -> None:
    registry = _registry()
    native_types = {node.schema().node_type for node in GENERATION_NODES}
    assert LATENT_CARRIERS <= native_types
    latent_carriers = {
        record["carrier"] for record in registry["records"] if record["carrier"] in LATENT_CARRIERS
    }
    assert latent_carriers == LATENT_CARRIERS


def test_generation_pack_bundles_alias_registry_and_declares_helper_dependency() -> None:
    pack_root = ROOT / "packages" / "dinkster-nodes-generation"
    configuration = cast(
        "dict[str, Any]",
        tomllib.loads((pack_root / "pyproject.toml").read_text(encoding="utf-8")),
    )
    force_include = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include["comfy-aliases.json"] == (
        "dinkster_nodes_generation_pack/comfy-aliases.json"
    )
    manifest = cast(
        "dict[str, Any]",
        tomllib.loads((pack_root / "dinkster-pack.toml").read_text(encoding="utf-8")),
    )
    assert manifest["pack"]["dependencies"]["dinkster-nodes-foundation"] == ">=0.0.1,<1"
