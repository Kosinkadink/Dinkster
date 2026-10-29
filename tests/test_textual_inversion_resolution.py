from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_assets import AssetError, EmbeddingNameIndex, MountsError, digest_file
from dinkster_inference import FLOAT32, build_runtime_identity_from_facts


def _snapshot(tmp_path: Path, rows: dict[str, dict[str, object]]) -> Path:
    root = tmp_path / "embeddings"
    root.mkdir(exist_ok=True)
    index = tmp_path / "index.json"
    index.write_text(json.dumps(rows), "utf-8")
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "embeddings",
                        "root": str(root),
                        "index": str(index),
                        "kind": "model/embedding",
                    }
                ]
            }
        ),
        "utf-8",
    )
    return snapshot


def _row(digest: str = "blake3:" + "1" * 64) -> dict[str, object]:
    return {"digest": digest, "size": 4, "mtimeNs": 1}


def test_embedding_index_aliases_refusals_and_binding(tmp_path: Path) -> None:
    index = EmbeddingNameIndex(_snapshot(tmp_path, {"styles/cat.safetensors": _row()}))
    assert index.resolve("styles/cat") == index.resolve("styles/cat.safetensors")
    assert index.binding_digest is not None and len(index.binding_digest) == 64
    for unsafe in ("", "../cat", "/cat", "C:/cat", "styles\\cat", "cat\n"):
        with pytest.raises(AssetError, match="unsafe"):
            index.resolve(unsafe)
    assert index.resolve("cat.pt") is None


def test_mixed_case_safetensors_preserves_authored_aliases(tmp_path: Path) -> None:
    index = EmbeddingNameIndex(_snapshot(tmp_path, {"Styles/Cat.SafeTensors": _row()}))
    assert index.resolve("Styles/Cat.SafeTensors") is not None
    assert index.resolve("Styles/Cat") is not None
    assert index.resolve("styles/cat") is None


def test_embedding_binding_rotates_runtime_identity() -> None:
    first = build_runtime_identity_from_facts(
        family_id="dinkster.sd15",
        component_identity=("family=dinkster.sd15",),
        diffusion_dtype=FLOAT32.name,
        text_dtype=FLOAT32.name,
        vae_dtype=FLOAT32.name,
        fp8_matmul=False,
        embedding_binding_digest="1" * 64,
    )
    second = build_runtime_identity_from_facts(
        family_id="dinkster.sd15",
        component_identity=("family=dinkster.sd15",),
        diffusion_dtype=FLOAT32.name,
        text_dtype=FLOAT32.name,
        vae_dtype=FLOAT32.name,
        fp8_matmul=False,
        embedding_binding_digest="2" * 64,
    )
    assert first != second


@pytest.mark.parametrize(
    "row",
    [
        None,
        {},
        {"digest": "blake3:" + "1" * 64, "size": 4},
        {"digest": "1" * 64, "size": 4, "mtimeNs": 1},
        {"digest": "blake3:" + "1" * 64, "size": True, "mtimeNs": 1},
        {"digest": "blake3:" + "1" * 64, "size": 4, "mtimeNs": -1},
        {"digest": "blake3:" + "1" * 64, "size": 4, "mtimeNs": 1, "extra": 0},
    ],
)
def test_embedding_index_rejects_every_malformed_row(tmp_path: Path, row: object) -> None:
    snapshot = _snapshot(tmp_path, {"bad.safetensors": row})  # type: ignore[dict-item]
    with pytest.raises(MountsError, match="malformed"):
        EmbeddingNameIndex(snapshot)


@pytest.mark.parametrize(
    "relative",
    ["../bad.safetensors", "/bad.safetensors", "C:/bad.safetensors", "a\\b.safetensors"],
)
def test_embedding_index_rejects_unsafe_index_paths(tmp_path: Path, relative: str) -> None:
    with pytest.raises(MountsError, match="unsafe"):
        EmbeddingNameIndex(_snapshot(tmp_path, {relative: _row()}))


def test_unsupported_row_rotates_binding_but_does_not_poison_safe_alias(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path, {"cat.safetensors": _row()})
    first = EmbeddingNameIndex(snapshot)
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps({"cat.safetensors": _row(), "legacy.pt": _row("blake3:" + "2" * 64)}),
        "utf-8",
    )
    second = EmbeddingNameIndex(snapshot)
    assert second.resolve("cat") is not None
    assert first.binding_digest != second.binding_digest
    with pytest.raises(AssetError, match="unsupported"):
        second.resolve("legacy.pt")
    with pytest.raises(AssetError, match="unsupported"):
        second.resolve("legacy")


@pytest.mark.parametrize("extension", (".pt", ".bin", ".PT", ".Bin"))
def test_legacy_exact_and_extensionless_names_refuse_case_insensitively(
    tmp_path: Path, extension: str
) -> None:
    index = EmbeddingNameIndex(_snapshot(tmp_path, {f"legacy{extension}": _row()}))
    with pytest.raises(AssetError, match="unsupported"):
        index.resolve(f"legacy{extension}")
    with pytest.raises(AssetError, match="unsupported"):
        index.resolve("legacy")


@pytest.mark.parametrize("extension", (".zip", ".pickle", ".xyz", ""))
def test_every_indexed_non_safetensors_exact_name_refuses(tmp_path: Path, extension: str) -> None:
    name = f"legacy{extension}"
    index = EmbeddingNameIndex(_snapshot(tmp_path, {name: _row()}))
    with pytest.raises(AssetError, match="unsupported"):
        index.resolve(name)


def test_multiple_legacy_candidates_and_safe_alias_never_select_a_shadow(
    tmp_path: Path,
) -> None:
    index = EmbeddingNameIndex(
        _snapshot(
            tmp_path,
            {
                "legacy.pt": _row(),
                "legacy.BIN": _row("blake3:" + "2" * 64),
                "legacy.safetensors": _row("blake3:" + "3" * 64),
            },
        )
    )
    with pytest.raises(AssetError, match="ambiguous|unsupported"):
        index.resolve("legacy")


def test_unsupported_exact_name_cannot_be_shadowed_by_safe_alias(
    tmp_path: Path,
) -> None:
    index = EmbeddingNameIndex(
        _snapshot(
            tmp_path,
            {"cat.pt": _row(), "cat.pt.safetensors": _row("blake3:" + "2" * 64)},
        )
    )
    with pytest.raises(AssetError, match="ambiguous|unsupported"):
        index.resolve("cat.pt")


def test_equal_digest_in_two_mounts_is_still_ambiguous(tmp_path: Path) -> None:
    digest = "blake3:" + "3" * 64
    mounts = []
    for mount_id in ("a", "b"):
        root = tmp_path / mount_id
        root.mkdir()
        index = tmp_path / f"{mount_id}.json"
        index.write_text(json.dumps({"same.safetensors": _row(digest)}), "utf-8")
        mounts.append(
            {"id": mount_id, "root": str(root), "index": str(index), "kind": "model/embedding"}
        )
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(json.dumps({"mounts": mounts}), "utf-8")
    with pytest.raises(AssetError, match="ambiguous"):
        EmbeddingNameIndex(snapshot).resolve("same")


def test_frozen_index_does_not_follow_later_snapshot_change(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, {"old.safetensors": _row()})
    frozen = EmbeddingNameIndex(snapshot)
    (tmp_path / "index.json").write_text(
        json.dumps({"new.safetensors": _row("blake3:" + "4" * 64)}), "utf-8"
    )
    assert frozen.resolve("old") is not None
    assert frozen.resolve("new") is None
    assert EmbeddingNameIndex(snapshot).resolve("new") is not None


def test_embedding_ref_local_path_verifies_real_file(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    root.mkdir()
    tensor = root / "cat.safetensors"
    tensor.write_bytes(b"safe")
    stat = tensor.stat()
    snapshot = _snapshot(
        tmp_path,
        {
            "cat.safetensors": {
                "digest": digest_file(tensor),
                "size": stat.st_size,
                "mtimeNs": stat.st_mtime_ns,
            }
        },
    )
    assert EmbeddingNameIndex(snapshot).resolve("cat").local_path() == tensor  # type: ignore[union-attr]


def test_embedding_ref_local_path_rejects_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    root.mkdir()
    tensor = root / "cat.safetensors"
    tensor.write_bytes(b"safe")
    stat = tensor.stat()
    snapshot = _snapshot(
        tmp_path,
        {
            "cat.safetensors": {
                "digest": "blake3:" + "1" * 64,
                "size": stat.st_size,
                "mtimeNs": stat.st_mtime_ns,
            }
        },
    )
    ref = EmbeddingNameIndex(snapshot).resolve("cat")
    with pytest.raises(AssetError, match="digest|integrity"):
        ref.local_path()  # type: ignore[union-attr]


def _write_safetensors(path: Path, shapes: dict[str, tuple[int, ...]]) -> Path:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for value, (key, shape) in enumerate(shapes.items()):
        size = 4
        for dimension in shape:
            size *= dimension
        header[key] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        payload.extend(struct.pack("<f", float(value)) * (size // 4))
        offset += size
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def _tensor_snapshot(tmp_path: Path, shapes: dict[str, tuple[int, ...]]) -> Path:
    root = tmp_path / "embeddings"
    root.mkdir(exist_ok=True)
    tensor = _write_safetensors(root / "bundle.safetensors", shapes)
    stat = tensor.stat()
    return _snapshot(
        tmp_path,
        {
            tensor.name: {
                "digest": digest_file(tensor),
                "size": stat.st_size,
                "mtimeNs": stat.st_mtime_ns,
            }
        },
    )


def test_native_embedding_resource_selects_direct_component_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    import dinkster_inference_torch
    from dinkster_compat_comfy import native_arm

    snapshot = _tensor_snapshot(
        tmp_path,
        {"clip_l": (2, 3), "clip_g": (2, 4), "t5xxl": (2, 5)},
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    index, lookups = native_arm._embedding_resource(dinkster_inference_torch)
    assert index is not None and index.binding_digest is not None
    assert lookups is not None
    assert set(lookups) == {"clip_l", "clip_g", "t5xxl"}
    assert lookups["clip_l"]("bundle").shape == (2, 3)
    assert lookups["clip_g"]("bundle").shape == (2, 4)
    assert lookups["t5xxl"]("bundle").shape == (2, 5)


@pytest.mark.parametrize("missing_profile", (False, True))
def test_recipe_declared_roles_build_real_mounted_embedding_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_profile: bool
) -> None:
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from dinkster_assets import AssetRef
    from dinkster_inference import LUMINA2, ComponentPlan, component_catalog, text_recipes
    from dinkster_inference.clip_text import CLIP_L_TEXT_CONFIG, ClipTextConfig
    from dinkster_inference.component_registry import ComponentDescriptor, ComponentRegistry
    from dinkster_inference.prompt_tokens import CLIP_L_PROFILE
    from dinkster_inference.registry import Registry
    from dinkster_inference.text_recipes import (
        DetectedTextSources,
        TextEncodingProfile,
        TextRecipeBinding,
        TextRecipeComponent,
        TextRecipeDescriptor,
    )
    from dinkster_inference.weights import AssetIdentifiedSource, WeightSource
    from dinkster_inference_torch.clip_text import ClipTextModel
    from dinkster_workers import ExecutionContext
    from dinkster_workers.execution import use_execution_context

    arm = importlib.import_module("dinkster_compat_comfy.native_arm")
    save_file = importlib.import_module("safetensors.torch").save_file
    roles = ("llama_like", "clip_l")
    snapshot = _tensor_snapshot(
        tmp_path,
        {role: (2, 8) for role in roles},
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "off")
    config = replace(
        CLIP_L_TEXT_CONFIG,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
    )
    models = {role: ClipTextModel(config) for role in roles}
    generator = torch.Generator().manual_seed(1261)
    with torch.no_grad():
        for model in models.values():
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.03)
    path = tmp_path / "encoders.safetensors"
    save_file(
        {
            f"{role}.{key}": value
            for role, model in models.items()
            for key, value in model.state_dict().items()
        },
        path,
    )

    class EncoderResolver:
        def resolve(self, digest: str) -> Path:
            assert digest == asset.digest
            return path

    asset = AssetRef(
        digest=digest_file(path),
        name=path.name,
        size=path.stat().st_size,
        resolver=EncoderResolver(),
    )

    def detect(
        source: WeightSource, path: Path
    ) -> tuple[tuple[str, ComponentPlan[ClipTextConfig]], ...]:
        assert isinstance(source, AssetIdentifiedSource)
        return tuple(
            (
                role,
                ComponentPlan(
                    role,
                    path,
                    config,
                    {key: f"{role}.{key}" for key in model.state_dict()},
                    {key: FLOAT32 for key in model.state_dict()},
                    {},
                    identity_facts=(
                        f"asset_digest={source.asset_digest}",
                        f"asset_size={source.asset_size}",
                    ),
                ),
            )
            for role, model in models.items()
        )

    components = ComponentRegistry()
    components.register(
        ComponentDescriptor(
            replace(LUMINA2, id="test.custom_clip", display_name="Custom CLIP"),
            detect,
            roles,
            roles,
            (),
            "unused",
            "unused",
            requires_text_recipe=True,
        )
    )
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: components)
    profile = None if missing_profile else TextEncodingProfile(CLIP_L_PROFILE)

    def bind(sources: DetectedTextSources) -> TextRecipeBinding:
        assert len(sources) == len(sources[0]) == 1
        return TextRecipeBinding(
            "test.custom_clip",
            "test.custom_clip",
            tuple(
                TextRecipeComponent(0, role, plan, profile)
                for role, plan in sources[0][0].components
            ),
            "dinkster_inference_torch.text_recipes:assemble_text_recipe",
            "dinkster_inference_torch.text_recipes:TextRecipeRuntime",
            "dinkster_inference_torch.clip_text:compose_sdxl_conditioning",
            roles,
        )

    recipes: Registry[TextRecipeDescriptor] = Registry()
    recipes.register(TextRecipeDescriptor("test.custom_clip", bind))
    monkeypatch.setattr(text_recipes, "default_text_recipe_registry", lambda: recipes)
    inference = importlib.import_module("dinkster_inference")
    source = inference.load_safetensors_header(
        path, asset_digest=asset.digest, asset_size=asset.size
    )
    binding = text_recipes.resolve_text_recipe(
        (components.detect(source, path),), "test.custom_clip"
    )
    recipe = binding.recipe(
        (arm._weight_source_ref(inference, asset),),
        "float32",
        embedding_binding_digest=EmbeddingNameIndex(snapshot).binding_digest,
    )
    with use_execution_context(
        ExecutionContext(
            "native",
            recipe.runtime_identity,
            diffusion_dtype="float32",
            text_dtype="float32",
            vae_dtype="float32",
        )
    ):
        if missing_profile:
            with pytest.raises(
                ValueError, match="^CLIP/T5 text runtime requires a tokenizer packing profile$"
            ):
                arm.NativeLoadClip.execute(
                    text_encoder=asset, type="test.custom_clip", device="cpu"
                )
            return
        handle = arm.NativeLoadClip.execute(
            text_encoder=asset, type="test.custom_clip", device="cpu"
        )["clip"]
    captured = {}
    hooks = [
        handle.module[role].register_forward_pre_hook(
            lambda _module, args, role=role: captured.__setitem__(role, args[0].clone())
        )
        for role in roles
    ]
    try:
        assert handle.recipe == recipe
        arm.GenerationClipTextEncode.execute(text="embedding:bundle", clip=handle)
        assert tuple(captured) == roles
        assert torch.equal(captured["llama_like"][0, 1:3], torch.zeros((2, 8)))
        assert torch.equal(captured["clip_l"][0, 1:3], torch.ones((2, 8)))
    finally:
        for hook in hooks:
            hook.remove()
        handle.terminal_release()

    index, lookups = arm._freeze_embedding_resource(
        component_roles=tuple(part.role for part in binding.components)
    )

    assert index is not None and index.binding_digest is not None
    assert lookups is not None and tuple(lookups) == ("llama_like", "clip_l")
    llama = lookups["llama_like"]("bundle")
    clip = lookups["clip_l"]("bundle")
    assert llama is lookups["llama_like"]("bundle")
    assert clip is lookups["clip_l"]("bundle")
    assert llama.data_ptr() != clip.data_ptr()
    assert torch.equal(llama, torch.zeros((2, 8)))
    assert torch.equal(clip, torch.ones((2, 8)))


def test_native_embedding_resource_selects_bundled_component_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    import dinkster_inference_torch
    from dinkster_compat_comfy import native_arm

    snapshot = _tensor_snapshot(
        tmp_path,
        {
            "bundle_emb.a.clip_l": (1, 3),
            "bundle_emb.b.clip_l": (1, 3),
            "bundle_emb.a.clip_g": (1, 4),
        },
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    _, lookups = native_arm._embedding_resource(dinkster_inference_torch)
    assert lookups is not None
    assert lookups["clip_l"]("bundle").shape == (2, 3)
    assert lookups["clip_g"]("bundle").shape == (1, 4)


def test_native_bundle_rows_preserve_safetensors_header_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    import dinkster_inference_torch
    from dinkster_compat_comfy import native_arm

    snapshot = _tensor_snapshot(
        tmp_path,
        {
            "bundle_emb.z.clip_l": (1, 3),
            "bundle_emb.a.clip_l": (1, 3),
        },
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    _, lookups = native_arm._embedding_resource(dinkster_inference_torch)
    assert lookups is not None
    selected = lookups["clip_l"]("bundle")
    assert torch.equal(selected[0], torch.zeros(3))
    assert torch.equal(selected[1], torch.ones(3))


def test_native_embedding_resource_refuses_unqualified_multi_tensor_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    import dinkster_inference_torch
    from dinkster_compat_comfy import native_arm

    snapshot = _tensor_snapshot(
        tmp_path,
        {"first": (1, 3), "second": (1, 3)},
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    _, lookups = native_arm._embedding_resource(dinkster_inference_torch)
    assert lookups is not None
    with pytest.raises(ValueError, match="no unambiguous component"):
        lookups["clip_l"]("bundle")


def test_torch_runtime_requires_paired_embedding_authority() -> None:
    pytest.importorskip("torch")
    from dinkster_inference_torch import WiringError, load_runtime

    lookup = {"clip_l": lambda _name: None}
    with pytest.raises(WiringError, match="provided together"):
        load_runtime(embedding_lookups=lookup)
    with pytest.raises(WiringError, match="provided together"):
        load_runtime(embedding_binding_digest="1" * 64)
    with pytest.raises(WiringError, match="unknown embedding lookup"):
        load_runtime(
            embedding_lookups={"unknown": lambda _name: None},  # type: ignore[dict-item]
            embedding_binding_digest="1" * 64,
        )


def test_empty_snapshot_environment_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy import native_arm

    from dinkster import native_policy

    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", "")
    assert native_policy._embedding_binding_digest() is None
    assert native_arm._freeze_embedding_resource() == (None, None)


def test_unset_snapshot_environment_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy import native_arm

    from dinkster import native_policy

    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    assert native_policy._embedding_binding_digest() is None
    assert native_arm._freeze_embedding_resource() == (None, None)


def test_native_materializer_refuses_worker_recipe_binding_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy import native_arm

    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace() if name == "dinkster_inference_torch" else real_import(name),
    )
    recipe = SimpleNamespace(
        sources=(),
        knobs=SimpleNamespace(embedding_binding_digest="1" * 64),
    )
    worker_index = SimpleNamespace(binding_digest="2" * 64)
    with pytest.raises(RuntimeError, match="binding does not match"):
        native_arm._materialize_recipe_handle(
            recipe,
            {},
            object(),
            (worker_index, {"clip_l": lambda _name: None}),  # type: ignore[arg-type]
        )
