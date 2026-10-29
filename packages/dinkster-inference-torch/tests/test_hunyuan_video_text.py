"""Source-grounded Hunyuan policy and real ordered loading/reconstruction."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import dinkster_inference as inference
import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    Conditioning,
    ConditioningCarrier,
    PayloadReference,
    llama3_text,
)
from dinkster_inference import clip_text as clip_text_contract
from dinkster_inference.clip_text import CLIP_L_TEXT_CONFIG
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.llama3_text import HUNYUAN_LLAMA3_CONFIG
from dinkster_inference.prompt_tokens import EmbeddingSlot, PackedToken
from dinkster_inference.recipe import WeightSourceRef
from dinkster_inference.sources import load_safetensors_header
from dinkster_inference.text_recipes import resolve_text_recipe
from dinkster_inference_torch.clip_text import ClipEncodePolicy, ClipTextEncoder, ClipTextModel
from dinkster_inference_torch.hunyuan_video_text import (
    ATTENTION_MASK_METADATA,
    HunyuanLlamaEncoder,
    LlamaTextEncoding,
    compose_hunyuan_video_conditioning,
    hunyuan_crop,
)
from dinkster_inference_torch.payloads import payload_binding_to_tensor
from dinkster_inference_torch.qwen_text import QwenTextModel
from dinkster_native import native_arm
from dinkster_workers import ExecutionContext
from dinkster_workers.execution import use_execution_context
from safetensors.torch import save_file


def build_models() -> tuple[QwenTextModel, ClipTextModel]:
    llama = QwenTextModel(
        replace(
            HUNYUAN_LLAMA3_CONFIG,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    clip = ClipTextModel(
        replace(
            CLIP_L_TEXT_CONFIG,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
    )
    generator = torch.Generator().manual_seed(1261)
    with torch.no_grad():
        for model in (llama, clip):
            for name, parameter in model.named_parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
                if "norm.weight" in name:
                    parameter.add_(1)
    return llama, clip


def tensors(carrier: ConditioningCarrier) -> dict[str, torch.Tensor]:
    values = {item.reference_id: payload_binding_to_tensor(item) for item in carrier.bindings}
    record = carrier.conditioning.records[0]
    result = {channel.value: values[payload.reference.id] for channel, payload in record.channels}
    for key, value in record.extension_metadata:
        if key == ATTENTION_MASK_METADATA:
            assert isinstance(value, PayloadReference)
            result["attention_mask"] = values[value.id]
    return result


def test_source_token_ids_template_and_structural_crop() -> None:
    llama, _ = build_models()
    encoder = HunyuanLlamaEncoder(llama)
    tokens = encoder.tokenize("A cat.")
    ids = [token.unit for token in tokens]
    # Executed pinned ComfyUI tokenizer/template, not output from a second Dinkster tokenizer.
    assert ids[0] == 128000
    assert ids[91:95] == [128006, 882, 128007, 271]
    assert ids[95:] == [32, 8415, 13, 128009]
    assert len(tokens) == 99
    assert hunyuan_crop(tokens) == slice(95, 99)
    assert encoder.bpe.encode("<image><pad>", add_special_tokens=False).ids == [128257, 128258]
    changed = tokens[:4] + (PackedToken(EmbeddingSlot("style", 0), 1.0, 1),) * 3 + tokens[4:]
    assert hunyuan_crop(changed) == slice(98, 102)
    padded = encoder.tokenize("A cat.", min_length=256, min_padding=2)
    assert len(padded) == 256
    assert all(token.unit == 128258 for token in padded[99:])
    assert hunyuan_crop(padded) == slice(95, 99)
    # A user-supplied structural header defines the active user boundary.
    nested = encoder.tokenize("<|start_header_id|>user<|end_header_id|>\n\nA cat.")
    assert [token.unit for token in nested[hunyuan_crop(nested)]] == [32, 8415, 13, 128009]


def test_llama_hidden_mask_weighting_and_embedding_rows() -> None:
    llama, _ = build_models()
    vectors = torch.randn(3, 8, generator=torch.Generator().manual_seed(16))
    encoder = HunyuanLlamaEncoder(llama, lambda name: vectors if name == "style" else None)
    captures: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        captures.append(cast(torch.Tensor, output).clone())

    def inputs(_module: torch.nn.Module, args: Any) -> None:
        masks.append(args[1].clone())

    layer_hook = llama.layers[-3].register_forward_hook(capture)
    mask_hook = llama.register_forward_pre_hook(inputs)
    text = "A <pad> embedding:style cat."
    with torch.no_grad():
        result = encoder.encode(text, min_length=150)
    layer_hook.remove()
    mask_hook.remove()
    tokens = encoder.tokenize(text, min_length=150)
    crop = hunyuan_crop(tokens)
    torch.testing.assert_close(result.embeddings, captures[0][:1, crop], rtol=0, atol=0)
    torch.testing.assert_close(result.attention_mask, masks[0][:1, crop], rtol=0, atol=0)
    assert 0 in result.attention_mask
    assert not torch.equal(result.embeddings, llama.norm(result.embeddings))
    slots = [i for i, token in enumerate(tokens) if isinstance(token.unit, EmbeddingSlot)]
    assert len(slots) == 3
    assert all(masks[0][0, index] == 1 for index in slots)
    captures.clear()
    layer_hook = llama.layers[-3].register_forward_hook(capture)
    with torch.no_grad():
        weighted = encoder.encode("(cat:1.5)")
    layer_hook.remove()
    weighted_tokens = encoder.tokenize("(cat:1.5)")
    expected = captures[0][:1].clone()
    for index, token in enumerate(weighted_tokens):
        if token.weight != 1.0:
            empty = captures[0][-1, index]
            expected[:, index] = (expected[:, index] - empty) * token.weight + empty
    torch.testing.assert_close(
        weighted.embeddings, expected[:, hunyuan_crop(weighted_tokens)], rtol=0, atol=0
    )
    with torch.no_grad():
        plain = encoder.encode("cat")
        overridden = encoder.encode("cat", hidden_layer=-1)
        restored = encoder.encode("cat")
    assert not torch.equal(weighted.embeddings, plain.embeddings)
    assert not torch.equal(overridden.embeddings, plain.embeddings)
    torch.testing.assert_close(restored.embeddings, plain.embeddings, rtol=0, atol=0)
    assert llama.config.output_hidden_layer == -3


def test_carrier_mask_reference_elision_and_raw_pooled() -> None:
    sequence = torch.randn(1, 4, 8)
    pooled = torch.randn(1, 8)
    for mask in (torch.ones(1, 4, dtype=torch.long), torch.tensor([[1, 1, 0, 0]])):
        result = compose_hunyuan_video_conditioning(
            LlamaTextEncoding(sequence, mask), Conditioning(torch.zeros(1, 77, 8), pooled)
        )
        values = tensors(result)
        torch.testing.assert_close(values["text"], sequence, rtol=0, atol=0)
        torch.testing.assert_close(values["pooled"], pooled, rtol=0, atol=0)
        metadata = dict(result.conditioning.records[0].extension_metadata)
        if bool(mask.all()):
            assert not metadata and "attention_mask" not in values
        else:
            assert set(metadata) == {ATTENTION_MASK_METADATA}
            assert isinstance(metadata[ATTENTION_MASK_METADATA], PayloadReference)
            torch.testing.assert_close(values["attention_mask"], mask, rtol=0, atol=0)


@dataclass(frozen=True)
class Resolver:
    path: Path

    def resolve(self, digest: str) -> Path:
        return self.path


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("quantized_role", (None, "llama", "clip_l"))
def test_public_ordered_load_handle_rebuild_and_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reverse: bool, quantized_role: str | None
) -> None:
    llama, clip = build_models()
    monkeypatch.setattr(llama3_text, "HUNYUAN_LLAMA3_CONFIG", llama.config)
    monkeypatch.setattr(clip_text_contract, "KNOWN_CLIP_TEXT_CONFIGS", (clip.config,))
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "off")

    def no_embeddings(**_kwargs: object) -> tuple[None, None]:
        return None, None

    monkeypatch.setattr(native_arm, "_freeze_embedding_resource", no_embeddings)
    models = {"llama": llama, "clip_l": clip}
    quant_layers = {
        "llama": "layers.0.self_attn.q_proj",
        "clip_l": "text_model.encoder.layers.0.self_attn.q_proj",
    }

    assets: list[AssetRef] = []
    for role, model in models.items():
        prefix = "llama.model." if role == "llama" else "clip_l."
        state = {prefix + key: value for key, value in model.state_dict().items()}
        metadata = {}
        if role == quantized_role:
            key = prefix + quant_layers[role]
            state[key + ".weight"] = state[key + ".weight"].to(torch.float8_e4m3fn)
            state[key + ".weight_scale"] = torch.tensor(1.0)
            metadata["_quantization_metadata"] = (
                '{"layers":{"' + key + '":{"format":"float8_e4m3fn"}}}'
            )
        path = tmp_path / f"{role}.safetensors"
        save_file(state, path, metadata=metadata)
        assets.append(
            AssetRef(
                digest=digest_file(path),
                name=path.name,
                size=path.stat().st_size,
                resolver=Resolver(path),
            )
        )
    if reverse:
        assets.reverse()
    sources = tuple(
        load_safetensors_header(
            asset.local_path(), asset_digest=asset.digest, asset_size=asset.size
        )
        for asset in assets
    )
    binding = resolve_text_recipe(
        tuple(default_component_registry().detect(source, source.path) for source in sources),
        "hunyuan_video",
    )
    if quantized_role is not None:
        part = next(part for part in binding.components if part.role == quantized_role)
        assert part.plan.quant[quant_layers[quantized_role]].format == "float8_e4m3fn"
    refs = tuple(WeightSourceRef(asset.digest, asset.name, asset.size) for asset in assets)
    recipe = binding.recipe(refs, "float32")
    context = ExecutionContext(
        "native",
        recipe.runtime_identity,
        diffusion_dtype="unloaded",
        text_dtype="float32",
        vae_dtype="unloaded",
    )
    arm: Any = native_arm
    with use_execution_context(context):
        handle: Any = arm.NativeLoadDualClip.execute(
            text_encoder1=assets[0],
            text_encoder2=assets[1],
            type="hunyuan_video",
            device="cpu",
        )["clip"]
    try:
        assert handle.recipe == recipe
        with handle.stage(), torch.inference_mode():
            result = handle.runtime.encode_text("A cat.")
            expected = ClipTextEncoder(
                cast(ClipTextModel, handle.module["clip_l"]),
                policy=ClipEncodePolicy(projected_pooled=False),
            ).encode(handle.runtime.clip_tokenizer.tokenize("A cat."))
        assert expected.pooled is not None
        values = tensors(result)
        assert values["text"].dtype == torch.float32
        assert torch.isfinite(values["text"]).all()
        assert torch.isfinite(values["pooled"]).all()
        torch.testing.assert_close(values["pooled"], expected.pooled, rtol=0, atol=0)
        emission = arm.GenerationClipTextEncode.execute(text="A cat.", clip=handle)
        node_carrier, node_binding = inference.split_component_conditioning(
            emission["conditioning"]
        )
        assert node_binding == inference.ComponentBinding(
            "text", "dinkster.hunyuan_video", recipe.runtime_identity
        )
        node_values = tensors(node_carrier)
        assert node_values.keys() == values.keys()
        for channel, value in values.items():
            torch.testing.assert_close(node_values[channel], value, rtol=0, atol=0)
        rebuilt = handle.rebuild()
        try:
            assert rebuilt.recipe == recipe
            with rebuilt.stage(), torch.inference_mode():
                again = rebuilt.runtime.encode_text("A cat.")
            assert result == again
        finally:
            rebuilt.terminal_release()
        original = tensors(result)
        for role, target, width, changed_channel in (
            ("llama", "layers.0.input_layernorm.weight", 8, "text"),
            (
                "clip_l",
                "text_model.encoder.layers.0.mlp.fc1.bias",
                16,
                "pooled",
            ),
        ):
            patch_path = tmp_path / f"{role}-overlay.safetensors"
            save_file({"delta": torch.full((width,), 0.125)}, patch_path)
            patch_asset = AssetRef(
                digest=digest_file(patch_path),
                name=patch_path.name,
                size=patch_path.stat().st_size,
                resolver=Resolver(patch_path),
            )
            overlay = inference.PatchOverlay.from_decoded(
                source=WeightSourceRef(patch_asset.digest, patch_asset.name, patch_asset.size),
                dialect="none",
                key_map="native.dinkster.test.v1",
                strength_model=0.0,
                strength_clip=1.0,
                patches=(
                    inference.OverlayPatch(
                        role,
                        inference.PatchTarget(target),
                        inference.DiffPatchRef("delta"),
                    ),
                ),
            )
            patched = handle.clone(
                (overlay,), source_resolvers={patch_asset.digest: patch_asset.resolver}
            )
            try:
                assert patched.resource_identity != handle.resource_identity
                with patched.stage(), torch.inference_mode():
                    changed = patched.runtime.encode_text("A cat.")
                assert not torch.equal(original[changed_channel], tensors(changed)[changed_channel])
            finally:
                patched.terminal_release()
        with handle.stage(), torch.inference_mode():
            assert handle.runtime.encode_text("A cat.") == result
    finally:
        handle.terminal_release()


def test_llama_embedding_lookup_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama, clip = build_models()
    monkeypatch.setattr(llama3_text, "HUNYUAN_LLAMA3_CONFIG", llama.config)
    monkeypatch.setattr(clip_text_contract, "KNOWN_CLIP_TEXT_CONFIGS", (clip.config,))
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "off")
    assets: list[AssetRef] = []
    for role, model in (("llama", llama), ("clip_l", clip)):
        prefix = "llama.model." if role == "llama" else "clip_l."
        path = tmp_path / f"{role}.safetensors"
        save_file({prefix + key: value for key, value in model.state_dict().items()}, path)
        assets.append(
            AssetRef(
                digest=digest_file(path),
                name=path.name,
                size=path.stat().st_size,
                resolver=Resolver(path),
            )
        )
    sources = tuple(
        load_safetensors_header(
            asset.local_path(), asset_digest=asset.digest, asset_size=asset.size
        )
        for asset in assets
    )
    binding = resolve_text_recipe(
        tuple(default_component_registry().detect(source, source.path) for source in sources),
        "hunyuan_video",
    )
    vectors = torch.ones(2, 8)

    def lookup(name: str) -> torch.Tensor | None:
        return vectors if name == "style" else None

    digest = "e" * 64
    resource = (SimpleNamespace(binding_digest=digest), {"llama": lookup})
    refs = tuple(WeightSourceRef(asset.digest, asset.name, asset.size) for asset in assets)
    recipe = binding.recipe(refs, "float32", embedding_binding_digest=digest)
    arm: Any = native_arm
    handle: Any = arm.build_text_recipe_handle(
        tuple(assets),
        "hunyuan_video",
        recipe.runtime_identity,
        compute_dtype="float32",
        load_device="cpu",
        embedding_resource=resource,
    )
    try:
        with handle.stage(), torch.inference_mode():
            embedded = handle.runtime.encode_text("embedding:style cat")
            plain = handle.runtime.encode_text("cat")
        assert not torch.equal(tensors(embedded)["text"], tensors(plain)["text"])
        assert handle.recipe.knobs.embedding_binding_digest == digest
    finally:
        handle.terminal_release()
