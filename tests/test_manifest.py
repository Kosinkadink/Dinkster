from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    ComfyAliasConfidence,
    ComfyAliasRecord,
    ComfyAliasRegistry,
    ComfyAliasSource,
    ComfyAliasSourceSchema,
    MappingSource,
    NodeSchema,
    ReplacementCase,
    ReplacementRule,
    comfy_alias_registry_to_wire,
    comfy_group_registry_to_wire,
)
from dinkster_workers import (
    PACK_AUTHOR_API_CONTRACT,
    PACK_HOST_CONTRACT,
    PACK_INFERENCE_CONTRACT,
    GenerationProvider,
    ManifestError,
    VisionProvider,
    load_manifest,
)
from dinkster_workers.manifest import (
    COMFY_ALIASES_MAX_BYTES,
    COMFY_ALIASES_MAX_ITEMS,
    generation_provider_to_wire,
    generation_providers_from_wire,
    vision_provider_to_wire,
    vision_providers_from_wire,
)


def write_manifest(path: Path, handler: object = None) -> Path:
    handler_line = "" if handler is None else f"workgroup_handler = {handler}\n"
    path.write_text(
        '[pack]\nname = "test"\n[pack.entry]\nnodes = "test_pack:NODES"\n' + handler_line
    )
    return path


def write_alias_registry(path: Path, *, carrier: str = "test.native") -> None:
    source = ComfyAliasSource("comfy-core", "TestNode", "comfy.TestNode", "b78cec87")
    registry = ComfyAliasRegistry(
        source_schemas=(ComfyAliasSourceSchema(NodeSchema(source.node_type), SCHEMA_WIRE_VERSION),),
        records=(
            ComfyAliasRecord(
                id="comfy_alias:comfy-core/TestNode",
                mapping_kind="op",
                carrier=carrier,
                source=source,
                replacement=ReplacementRule(
                    from_type=source.node_type,
                    cases=(
                        ReplacementCase.build(
                            carrier,
                            inputs={"value": MappingSource.constant(1)},
                        ),
                    ),
                ),
                confidence=ComfyAliasConfidence("exact", ("tests/test_alias.py::test",)),
            ),
        ),
    )
    path.write_text(json.dumps(comfy_alias_registry_to_wire(registry)), encoding="utf-8")


def write_group_registry(path: Path) -> None:
    from test_comfy_group_registry import registry

    path.write_text(json.dumps(comfy_group_registry_to_wire(registry())), encoding="utf-8")


def test_manifest_loads_strict_adjacent_comfy_alias_registry_without_importing_code(
    tmp_path: Path,
) -> None:
    sys.modules.pop("manifest_purity_probe", None)
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "test"\n[pack.entry]\nnodes = "manifest_purity_probe:NODES"\n'
    )
    write_alias_registry(tmp_path / "comfy-aliases.json")

    manifest = load_manifest(manifest_path)

    assert manifest.comfy_aliases is not None
    assert manifest.comfy_aliases.records[0].id == "comfy_alias:comfy-core/TestNode"
    assert "manifest_purity_probe" not in sys.modules
    (tmp_path / "comfy-aliases.json").unlink()
    assert load_manifest(manifest_path).comfy_aliases is None


def test_manifest_loads_strict_adjacent_comfy_group_registry_without_importing_code(
    tmp_path: Path,
) -> None:
    sys.modules.pop("manifest_purity_probe", None)
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "test"\n[pack.entry]\nnodes = "manifest_purity_probe:NODES"\n'
    )
    write_group_registry(tmp_path / "comfy-groups.json")

    manifest = load_manifest(manifest_path)

    assert manifest.comfy_groups is not None
    assert manifest.comfy_groups.records[0].id == (
        "comfy_group:comfyui-controlnet-aux/canny-resize"
    )
    assert "manifest_purity_probe" not in sys.modules
    (tmp_path / "comfy-groups.json").unlink()
    assert load_manifest(manifest_path).comfy_groups is None


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            '{"format":"dinkster-comfy-alias/1","sourceSchemas":[],"records":[],"records":[]}',
            "duplicate JSON object key",
        ),
        (
            '{"format":"dinkster-comfy-alias/1","sourceSchemas":[],"records":[],"bad":NaN}',
            "non-finite JSON number",
        ),
        (
            '{"format":"dinkster-comfy-alias/1","sourceSchemas":[],"records":[],"bad":1e999}',
            "non-finite JSON number",
        ),
        ("[" * 70 + "]" * 70, "nesting depth"),
    ],
)
def test_manifest_rejects_hostile_comfy_alias_json(tmp_path: Path, body: str, message: str) -> None:
    manifest_path = write_manifest(tmp_path / "dinkster-pack.toml")
    (tmp_path / "comfy-aliases.json").write_text(body, encoding="utf-8")
    with pytest.raises(ManifestError, match=message):
        load_manifest(manifest_path)


def test_manifest_rejects_hostile_comfy_group_json(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "dinkster-pack.toml")
    (tmp_path / "comfy-groups.json").write_text(
        '{"format":"dinkster-comfy-group/1","sourceSchemas":[],"groupSchemas":[],'
        '"records":[],"records":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="duplicate JSON object key"):
        load_manifest(manifest_path)


def test_manifest_rejects_oversized_and_escaping_comfy_alias_registry(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "dinkster-pack.toml")
    alias_path = tmp_path / "comfy-aliases.json"
    alias_path.write_bytes(b" " * (COMFY_ALIASES_MAX_BYTES + 1))
    with pytest.raises(ManifestError, match="cap"):
        load_manifest(manifest_path)

    alias_path.unlink()
    outside = tmp_path.parent / "outside-comfy-aliases.json"
    outside.write_text('{"format":"dinkster-comfy-alias/1","sourceSchemas":[],"records":[]}')
    try:
        alias_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ManifestError, match="escape"):
        load_manifest(manifest_path)


def test_manifest_rejects_comfy_alias_json_over_the_item_budget(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "dinkster-pack.toml")
    (tmp_path / "comfy-aliases.json").write_text(
        "[" + ",".join("0" for _ in range(COMFY_ALIASES_MAX_ITEMS)) + "]",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="item count"):
        load_manifest(manifest_path)


def test_workgroup_handler_entry_is_optional_and_loading_is_pure(tmp_path: Path) -> None:
    sys.modules.pop("manifest_purity_probe", None)
    plain = load_manifest(write_manifest(tmp_path / "plain.toml"))
    declared = load_manifest(
        write_manifest(tmp_path / "declared.toml", '"manifest_purity_probe:create_handler"')
    )

    assert plain.workgroup_handler_entry is None
    assert declared.workgroup_handler_entry == "manifest_purity_probe:create_handler"
    assert "manifest_purity_probe" not in sys.modules


@pytest.mark.parametrize(
    "declared",
    [
        "17",
        '""',
        '":factory"',
        '"module:"',
        '"module:factory:extra"',
        '"module :factory"',
        '"module: factory"',
    ],
)
def test_workgroup_handler_entry_requires_exact_nonempty_module_attr(
    tmp_path: Path, declared: str
) -> None:
    with pytest.raises(ManifestError, match="workgroup_handler.*module:attr"):
        load_manifest(write_manifest(tmp_path / "invalid.toml", declared))


def test_pack_contracts_dependencies_and_registry_requirements_are_data_only(
    tmp_path: Path,
) -> None:
    sys.modules.pop("manifest_purity_probe", None)
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "consumer"\n'
        '[pack.contracts]\nhost = "dinkster-pack-host/1"\n'
        'api = "dinkster-api/v1"\ninference = "dinkster-inference/1"\n'
        '[pack.dependencies]\nprovider = ">=1.2,<2"\n'
        '[pack.requirements.registry]\n"dinkster.model-families" = ["dinkster.wan21"]\n'
        '[pack.requirements.capabilities]\n"dinkster.video-generation" = ">=2,<3"\n'
        '[pack.provides.registry]\n"dinkster.samplers" = ["consumer.sampler"]\n'
        '[pack.capabilities]\n"consumer.graph-import" = "1.0.0"\n'
        '[pack.entry]\nnodes = "manifest_purity_probe:NODES"\n',
        encoding="utf-8",
    )

    manifest = load_manifest(path)

    assert manifest.contracts is not None
    assert manifest.contracts.host == PACK_HOST_CONTRACT
    assert manifest.contracts.api == PACK_AUTHOR_API_CONTRACT
    assert manifest.contracts.inference == PACK_INFERENCE_CONTRACT
    assert [(item.pack, item.version) for item in manifest.dependencies] == [
        ("provider", "<2,>=1.2")
    ]
    assert [(item.registry, item.id) for item in manifest.requirements.registry] == [
        ("dinkster.model-families", "dinkster.wan21")
    ]
    assert [(item.id, item.version) for item in manifest.requirements.capabilities] == [
        ("dinkster.video-generation", "<3,>=2")
    ]
    assert [(item.registry, item.id) for item in manifest.provides.registry] == [
        ("dinkster.samplers", "consumer.sampler")
    ]
    assert [(item.id, item.version) for item in manifest.capabilities] == [
        ("consumer.graph-import", "1.0.0")
    ]
    assert "manifest_purity_probe" not in sys.modules


def test_pack_sandbox_needs_are_strict_data_only(tmp_path: Path) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "sandboxed"\n'
        "[pack.sandbox]\ngpu = true\nnetwork = true\nwritable-mounts = true\n"
        '[pack.entry]\nnodes = "manifest_purity_probe:NODES"\n',
        encoding="utf-8",
    )

    manifest = load_manifest(path)

    assert manifest.sandbox_declared
    assert manifest.sandbox.gpu
    assert manifest.sandbox.network
    assert manifest.sandbox.writable_mounts
    assert "manifest_purity_probe" not in sys.modules


def test_model_backed_vision_provider_requirements_are_data_only(tmp_path: Path) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "depth-provider"\nnamespaces = []\n'
        'executes = ["dinkster.preprocess.depth"]\n'
        '[[pack.vision-providers]]\nchoice = "dinkster.vision.depth"\n'
        'node = "dinkster.preprocess.depth"\ndevices = ["cpu", "cuda"]\n'
        'dtypes = ["float16", "float32"]\nbatching = "batch"\n'
        'model = "depth-anything-v3"\n'
        'artifacts = ["depth-model"]\n'
        '[[pack.assets]]\nid = "depth-model"\nname = "Depth Model"\n'
        f'digest = "blake3:{"a" * 64}"\n'
        'urls = ["https://models.example/depth.safetensors"]\n'
        '[pack.entry]\nnodes = "manifest_purity_probe:NODES"\n',
        encoding="utf-8",
    )

    manifest = load_manifest(path)

    assert manifest.vision_providers == (
        VisionProvider(
            choice="dinkster.vision.depth",
            node="dinkster.preprocess.depth",
            devices=("cpu", "cuda"),
            dtypes=("float16", "float32"),
            batching="batch",
            artifacts=("depth-model",),
            model="depth-anything-v3",
        ),
    )
    assert "manifest_purity_probe" not in sys.modules


def test_vision_provider_without_declared_assets_allows_empty_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "upscale-provider"\nnamespaces = []\n'
        'executes = ["dinkster.image.upscale_model"]\n'
        '[[pack.vision-providers]]\nchoice = "dinkster.image.upscale_model.providers"\n'
        'node = "dinkster.image.upscale_model"\ndevices = ["cpu"]\n'
        'dtypes = ["float32"]\nbatching = "batch"\n'
        "artifacts = []\n"
        '[pack.entry]\nnodes = "manifest_purity_probe:NODES"\n',
        encoding="utf-8",
    )

    manifest = load_manifest(path)

    assert manifest.vision_providers == (
        VisionProvider(
            choice="dinkster.image.upscale_model.providers",
            node="dinkster.image.upscale_model",
            devices=("cpu",),
            dtypes=("float32",),
            batching="batch",
            artifacts=(),
        ),
    )
    assert "manifest_purity_probe" not in sys.modules


def test_external_generation_provider_declaration_is_data_only(tmp_path: Path) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "openai-provider"\nnamespaces = []\n'
        'executes = ["dinkster.text_generate", "dinkster.prompt_enhance"]\n'
        '[[pack.generation-providers]]\nchoice = "dinkster.generation.providers"\n'
        'node = "dinkster.text_generate"\nlabel = "Configured service"\n'
        '[[pack.generation-providers]]\nchoice = "dinkster.generation.providers"\n'
        'node = "dinkster.prompt_enhance"\n'
        '[pack.entry]\nnodes = "manifest_purity_probe:NODES"\n',
        encoding="utf-8",
    )

    manifest = load_manifest(path)

    assert manifest.generation_providers == (
        GenerationProvider(
            "dinkster.generation.providers",
            "dinkster.text_generate",
            "Configured service",
        ),
        GenerationProvider("dinkster.generation.providers", "dinkster.prompt_enhance"),
    )
    assert "manifest_purity_probe" not in sys.modules


def test_provider_declarations_round_trip_over_worker_hello_wire() -> None:
    vision = VisionProvider(
        choice="dinkster.vision.depth",
        node="dinkster.preprocess.depth",
        devices=("cpu", "cuda"),
        dtypes=("float16", "float32"),
        batching="batch",
        artifacts=("depth-model",),
        model="depth-anything-v3",
    )
    generation = GenerationProvider(
        "dinkster.generation.providers",
        "dinkster.text_generate",
        "Configured service",
    )

    assert vision_providers_from_wire([vision_provider_to_wire(vision)]) == (vision,)
    assert generation_providers_from_wire([generation_provider_to_wire(generation)]) == (
        generation,
    )
    assert vision_providers_from_wire([]) == ()
    assert generation_providers_from_wire([]) == ()


@pytest.mark.parametrize(
    ("decode", "wire", "message"),
    [
        (vision_providers_from_wire, {}, "must be a list"),
        (
            vision_providers_from_wire,
            [
                {
                    "choice": "dinkster.vision.depth",
                    "node": "dinkster.preprocess.depth",
                    "devices": ["quantum"],
                    "dtypes": ["float32"],
                    "batching": "batch",
                    "artifacts": [],
                }
            ],
            "unknown values",
        ),
        (
            generation_providers_from_wire,
            [
                {
                    "choice": "dinkster.generation.providers",
                    "node": "dinkster.text_generate",
                    "unexpected": True,
                }
            ],
            "malformed fields",
        ),
    ],
)
def test_provider_declaration_wire_rejects_malformed_evidence(
    decode: Callable[[object], object], wire: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        decode(wire)


def test_generation_provider_node_must_be_executed_by_its_pack(tmp_path: Path) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "openai-provider"\nnamespaces = []\n'
        'executes = ["dinkster.text_generate"]\n'
        '[[pack.generation-providers]]\nchoice = "dinkster.generation.providers"\n'
        'node = "dinkster.prompt_enhance"\n'
        '[pack.entry]\nnodes = "manifest_purity_probe:NODES"\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="must appear in.*executes"):
        load_manifest(path)


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (
            'choice = "dinkster.vision.depth"\nnode = "dinkster.preprocess.other"\n'
            'devices = ["cpu"]\ndtypes = ["float32"]\nbatching = "batch"\n'
            'artifacts = ["depth-model"]\n',
            "must appear in.*executes",
        ),
        (
            'choice = "dinkster.vision.depth"\nnode = "dinkster.preprocess.depth"\n'
            'devices = ["tpu"]\ndtypes = ["float32"]\nbatching = "batch"\n'
            'artifacts = ["depth-model"]\n',
            "devices contains unknown values",
        ),
        (
            'choice = "dinkster.vision.depth"\nnode = "dinkster.preprocess.depth"\n'
            'devices = ["cpu"]\ndtypes = ["float32"]\nbatching = "stream"\n'
            'artifacts = ["depth-model"]\n',
            "batching must be one of",
        ),
        (
            'choice = "dinkster.vision.depth"\nnode = "dinkster.preprocess.depth"\n'
            'devices = ["cpu"]\ndtypes = ["float32"]\nbatching = "batch"\n'
            'artifacts = ["absent"]\n',
            "unknown.*asset.*absent",
        ),
        (
            'choice = "dinkster.vision.depth"\nnode = "dinkster.preprocess.depth"\n'
            'devices = ["cpu"]\ndtypes = ["float32"]\nbatching = "batch"\n'
            'model = ""\nartifacts = ["depth-model"]\n',
            "model must be a non-empty string",
        ),
    ],
)
def test_vision_provider_requirements_fail_closed(
    tmp_path: Path, provider: str, message: str
) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "depth-provider"\nnamespaces = []\n'
        'executes = ["dinkster.preprocess.depth"]\n'
        f"[[pack.vision-providers]]\n{provider}"
        '[[pack.assets]]\nid = "depth-model"\nname = "Depth Model"\n'
        f'digest = "blake3:{"a" * 64}"\n'
        'urls = ["https://models.example/depth.safetensors"]\n'
        '[pack.entry]\nnodes = "manifest_purity_probe:NODES"\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match=message):
        load_manifest(path)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("sandbox = true\n", "must be a table"),
        ('[pack.sandbox]\ngpu = "yes"\n', "gpu must be a boolean"),
        ("[pack.sandbox]\negress = true\n", "unknown fields: egress"),
    ],
)
def test_pack_sandbox_needs_reject_ambiguous_shapes(
    tmp_path: Path, body: str, message: str
) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        f'[pack]\nname = "sandboxed"\n{body}[pack.entry]\nnodes = "test_pack:NODES"\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match=message):
        load_manifest(path)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[pack.contracts]\nhost = "dinkster-pack-host/0"\napi = "dinkster-api/v1"\n', "host"),
        ('[pack.dependencies]\nprovider = ""\n', "non-empty"),
        (
            '[pack.requirements.registry]\n"families" = ["dinkster.wan21"]\n',
            "must be namespaced",
        ),
        (
            '[pack.requirements.capabilities]\n"video-generation" = ">=1"\n',
            "must be namespaced",
        ),
        (
            '[pack.provides.registry]\n"families" = ["consumer.family"]\n',
            "must be namespaced",
        ),
        ('[pack.capabilities]\n"consumer.video" = "1.0"\n', "major.minor.patch"),
    ],
)
def test_pack_contract_metadata_rejects_ambiguous_shapes(
    tmp_path: Path, body: str, message: str
) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        f'[pack]\nname = "consumer"\n{body}[pack.entry]\nnodes = "test_pack:NODES"\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match=message):
        load_manifest(path)


def test_same_session_arms_may_implement_cross_pack_executes_claims(tmp_path: Path) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "provider"\nexecutes = ["owner.generate"]\n'
        '[pack.arms]\nnative = ["owner.generate"]\n'
        '[pack.entry]\nnodes = "provider:NODES"\narm_nodes = "provider:ARM_NODES"\n',
        encoding="utf-8",
    )

    manifest = load_manifest(path)

    assert manifest.executes == ("owner.generate",)
    assert manifest.arms == (("native", ("owner.generate",)),)


def test_same_session_arms_reject_unclaimed_external_schemas(tmp_path: Path) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "provider"\n'
        '[pack.arms]\nnative = ["owner.generate"]\n'
        '[pack.entry]\nnodes = "provider:NODES"\narm_nodes = "provider:ARM_NODES"\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="neither owned nor listed"):
        load_manifest(path)


def test_schema_only_claims_must_be_owned_and_body_free(tmp_path: Path) -> None:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "owner"\nschema-only = ["owner.generate"]\n'
        '[pack.entry]\nnodes = "owner:NODES"\n',
        encoding="utf-8",
    )

    assert load_manifest(path).schema_only == ("owner.generate",)

    path.write_text(
        '[pack]\nname = "owner"\nschema-only = ["other.generate"]\n'
        '[pack.entry]\nnodes = "owner:NODES"\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="is not owned"):
        load_manifest(path)

    path.write_text(
        '[pack]\nname = "owner"\nschema-only = ["owner.generate"]\n'
        '[pack.arms]\nnative = ["owner.generate"]\n'
        '[pack.entry]\nnodes = "owner:NODES"\narm_nodes = "owner:ARM_NODES"\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="cannot declare.*bodies"):
        load_manifest(path)
