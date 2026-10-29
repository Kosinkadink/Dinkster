"""Synthetic tiny weights, real tokenizers/assembly/encoding/native reconstruction.

Production full-header detection is covered by test_inference_ltx_text_recipe;
the detector here supplies tiny configs, not official-artifact parity evidence.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import dinkster_inference as inference
import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_compat_comfy import native_arm
from dinkster_inference import FLOAT32, LTXAV, ComponentPlan, component_catalog
from dinkster_inference.component_registry import ComponentDescriptor, ComponentRegistry
from dinkster_inference.gemma_text import (
    GEMMA3_LTX_12B_CONFIG,
    GEMMA4_LTX_12B_CONFIG,
    LTX_TEXT_CONNECTOR_CONFIG,
    tokenize_ltx_gemma_prompt,
)
from dinkster_inference.ltx_text_recipe import LTX_TEXT_RECIPE
from dinkster_inference.text_recipes import (
    TextRecipeBinding,
    TextRecipeComponent,
    resolve_text_recipe,
)
from dinkster_inference.weights import AssetIdentifiedSource, WeightSource
from dinkster_inference_torch import ltx_text_recipe as recipe_module
from dinkster_inference_torch.gemma_text import (
    GemmaTextModel,
    LtxDualTextProjection,
    LtxGemmaTextEncoder,
)
from dinkster_inference_torch.gemma_tokenizer import GemmaJsonTokenizer, GemmaSentencePieceTokenizer
from dinkster_inference_torch.ltx_connector import LtxTextConnectors
from dinkster_inference_torch.ltxav_runtime import (
    _materialize_conditioning,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.quant_linear import Fp8Linear, Int8Linear, Nvfp4Linear
from dinkster_workers import ExecutionContext
from dinkster_workers.execution import use_execution_context
from safetensors.torch import save_file


@pytest.fixture(scope="module")
def tokenizer_models() -> dict[str, bytes]:
    spm = importlib.import_module("sentencepiece")
    corpus = ["a bird flies over a tree", "raw prompt (bird:1.2)"]
    options = dict(model_type="char", pad_id=0, eos_id=1, bos_id=2, unk_id=3, minloglevel=2)
    first = BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus), model_writer=first, vocab_size=64, **options
    )
    count = spm.SentencePieceProcessor(model_proto=first.getvalue()).get_piece_size()
    padded = BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus),
        model_writer=padded,
        vocab_size=262144,
        user_defined_symbols=[f"<{i:x}>" for i in range(262144 - count)],
        **options,
    )
    tokenizers = importlib.import_module("tokenizers")
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "a", "bird"]
    vocab = {word: index for index, word in enumerate(words)}
    vocab.update({f"unused_{i}": i for i in range(len(words), 262144)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    return {"spiece_model": padded.getvalue(), "tokenizer_json": tokenizer.to_str().encode()}


@dataclass(frozen=True)
class Resolver:
    path: Path

    def resolve(self, digest: str) -> Path:
        return self.path


def test_assembly_loads_bound_sources_and_pins_quantized_gemma_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        GEMMA3_LTX_12B_CONFIG,
        hidden_size=8,
        num_hidden_layers=2,
    )
    gemma_plan = ComponentPlan("gemma3_12b", Path("gemma"), config, {}, {}, {})
    projection_plan = ComponentPlan(
        "text_projection",
        Path("projection"),
        "dual_linear",
        {
            "video_aggregate_embed.weight": "video",
            "audio_aggregate_embed.weight": "audio",
        },
        {
            "video_aggregate_embed.weight": FLOAT32,
            "audio_aggregate_embed.weight": FLOAT32,
        },
        {},
    )
    binding = TextRecipeBinding(
        LTX_TEXT_RECIPE.id,
        "dinkster.ltxav",
        (
            TextRecipeComponent(0, "gemma3_12b", gemma_plan, None),
            TextRecipeComponent(1, "text_projection", projection_plan, None),
        ),
        "unused:loader",
        "unused:runtime",
        "unused:composer",
        ("gemma3_12b", "text_projection"),
    )
    tokenizer_reads: list[tuple[object, str, int]] = []

    class Source:
        def read_uint8_configuration_from_file(
            self, handle: BinaryIO, key: str, *, limit: int
        ) -> bytes:
            tokenizer_reads.append((handle, key, limit))
            return b"tokenizer"

        def entry(self, _key: str) -> SimpleNamespace:
            return SimpleNamespace(geometry=SimpleNamespace(shape=(8, 24)))

    gemma_source = Source()
    projection_source = Source()
    gemma = torch.nn.ModuleDict(
        {
            "fp8": Fp8Linear(16, 16, bias=False, compute_dtype=torch.bfloat16),
            "int8": Int8Linear(
                16,
                16,
                bias=False,
                compute_dtype=torch.bfloat16,
                convrot=False,
                convrot_groupsize=0,
            ),
            "nvfp4": Nvfp4Linear(16, 16, bias=False, compute_dtype=torch.bfloat16),
        }
    )
    projection = LtxDualTextProjection(24, 8, 4)
    load_calls: list[tuple[str, object, object]] = []

    def load_component(
        plan: ComponentPlan[object], _builder: object, **kwargs: object
    ) -> torch.nn.Module:
        load_calls.append((plan.component, kwargs["source"], kwargs["source_file"]))
        return gemma if plan is gemma_plan else projection

    monkeypatch.setattr(recipe_module, "_load_component", load_component)

    def resolve_attention(*_args: object) -> SimpleNamespace:
        return SimpleNamespace(kernel=None, status="sdpa")

    monkeypatch.setattr(
        recipe_module,
        "resolve_role_attention",
        resolve_attention,
    )
    gemma_file = BytesIO()
    projection_file = BytesIO()
    loaded = recipe_module.assemble_ltx_text_recipe(
        binding,
        compute_dtype=torch.bfloat16,
        sources=cast("Any", (gemma_source, projection_source)),
        source_files=(gemma_file, projection_file),
        attention_backends=(("gemma3_12b", "qwen"), ("connectors", "flux")),
    )

    assert tokenizer_reads == [(gemma_file, "spiece_model", 8 * 1024 * 1024)]
    assert load_calls == [
        ("gemma3_12b", gemma_source, gemma_file),
        ("text_projection", projection_source, projection_file),
    ]
    assert loaded.attention_status == ("sdpa",)
    for layer in gemma.values():
        assert isinstance(layer, Fp8Linear | Int8Linear | Nvfp4Linear)
        assert layer.compute_dtype is torch.float32
        assert layer.full_precision_matmul


@pytest.mark.parametrize("kind", ("single_linear", "dual_linear", "dual_linear_gemma4"))
@pytest.mark.parametrize(("reverse", "combined"), ((False, False), (True, False), (False, True)))
def test_tiny_native_ordered_ltx_load_encode_rebuild_and_projection_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_models: dict[str, bytes],
    kind: str,
    reverse: bool,
    combined: bool,
) -> None:
    gemma4 = kind == "dual_linear_gemma4"
    role = "gemma4_12b" if gemma4 else "gemma3_12b"
    tokenizer_key = "tokenizer_json" if gemma4 else "spiece_model"
    config = replace(
        GEMMA4_LTX_12B_CONFIG if gemma4 else GEMMA3_LTX_12B_CONFIG,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        global_head_dim=4,
        num_global_key_value_heads=1,
        min_tokens=32,
        sliding_pattern=(True, False),
    )
    connector_config = replace(
        LTX_TEXT_CONNECTOR_CONFIG,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_learnable_registers=8,
    )
    models: dict[str, torch.nn.Module] = {
        role: GemmaTextModel(config),
        "text_projection": (
            torch.nn.Linear(24, 8, bias=False)
            if kind == "single_linear"
            else LtxDualTextProjection(24, 8, 4)
        ),
    }
    if kind == "single_linear":
        models["connectors"] = LtxTextConnectors(connector_config)
    configs: dict[str, object] = {
        role: config,
        "text_projection": kind,
        "connectors": connector_config,
    }
    generator = torch.Generator().manual_seed(1261)
    for model in models.values():
        with torch.no_grad():
            for tensor in model.state_dict().values():
                tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.03)

    def detect(source: WeightSource, path: Path) -> tuple[tuple[str, ComponentPlan[Any]], ...]:
        assert isinstance(source, AssetIdentifiedSource)
        return tuple(
            (
                name,
                ComponentPlan(
                    name,
                    path,
                    configs[name],
                    {key: f"{name}.{key}" for key in model.state_dict()},
                    {key: FLOAT32 for key in model.state_dict()},
                    {},
                    identity_facts=(
                        f"asset_digest={source.asset_digest}",
                        f"asset_size={source.asset_size}",
                    ),
                ),
            )
            for name, model in models.items()
            if any(key.startswith(f"{name}.") for key in source.keys())
        )

    registry = ComponentRegistry()
    registry.register(
        ComponentDescriptor(
            family=LTXAV,
            detector=detect,
            roles=tuple(models),
            text_encoder_roles=tuple(models),
            codec_roles=(),
            loader=LTX_TEXT_RECIPE.id + ":unused",
            runtime_class="unused:Runtime",
            requires_text_recipe=True,
        )
    )
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "off")
    assets: list[AssetRef] = []
    groups = (
        (tuple(models),) if combined else ((role,), tuple(name for name in models if name != role))
    )
    for index, group in enumerate(groups):
        state = {
            f"{name}.{key}": value
            for name in group
            for key, value in models[name].state_dict().items()
        }
        if role in group:
            state[tokenizer_key] = torch.frombuffer(
                bytearray(tokenizer_models[tokenizer_key]), dtype=torch.uint8
            ).clone()
        path = tmp_path / f"encoder-{index}.safetensors"
        save_file(state, path)
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
    binding = resolve_text_recipe(
        tuple(
            registry.detect(
                inference.load_safetensors_header(
                    asset.local_path(), asset_digest=asset.digest, asset_size=asset.size
                ),
                asset.local_path(),
            )
            for asset in assets
        ),
        "ltxv",
    )
    arm: Any = native_arm
    recipe = binding.recipe(tuple(arm._weight_source_ref(inference, a) for a in assets), "float32")
    context = ExecutionContext(
        "native",
        recipe.runtime_identity,
        diffusion_dtype="unloaded",
        text_dtype="float32",
        vae_dtype="unloaded",
    )
    with use_execution_context(context):
        handle = (
            arm.NativeLoadClip.execute(text_encoder=assets[0], type="ltxv", device="cpu")["clip"]
            if combined
            else arm.NativeLoadDualClip.execute(
                text_encoder1=assets[0],
                text_encoder2=assets[1],
                type="ltxv",
                device="cpu",
            )["clip"]
        )
    try:
        assert tuple(item.source.digest for item in handle.recipe.sources) == tuple(
            asset.digest for asset in assets
        )
        tokenizer = (GemmaJsonTokenizer if gemma4 else GemmaSentencePieceTokenizer)(
            tokenizer_models[tokenizer_key]
        )
        gemma = models[role]
        projection = models["text_projection"]
        connectors = models.get("connectors")
        assert isinstance(gemma, GemmaTextModel)
        assert isinstance(projection, torch.nn.Linear | LtxDualTextProjection)
        assert connectors is None or isinstance(connectors, LtxTextConnectors)
        composer = LtxGemmaTextEncoder(gemma, projection, tokenizer, connectors=connectors)
        tokens = tokenize_ltx_gemma_prompt("a bird", encode=tokenizer.encode, config=config)
        with torch.inference_mode():
            expected = composer.encode_tokens(tokens).embeddings
        output = arm.GenerationClipTextEncode.execute(text="a bird", clip=handle)
        carrier, bound = inference.split_component_conditioning(output["conditioning"])
        assert bound == inference.ComponentBinding(
            "text", "dinkster.ltxav", recipe.runtime_identity
        )
        actual = _materialize_conditioning(
            carrier,
            text_dim=expected.shape[-1],
            family_id="dinkster.ltxav",
            expected_text_stream=role,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
        for option in ("hidden_layer", "min_padding", "min_length"):
            with pytest.raises(ValueError, match="encoding overrides"):
                handle.runtime.encode_text("a bird", **{option: 1})
        rebuilt = handle.rebuild()
        try:
            assert rebuilt.recipe == recipe
            with rebuilt.stage(), torch.inference_mode():
                torch.testing.assert_close(
                    rebuilt.runtime.encode_text("a bird").embeddings, expected, rtol=0, atol=0
                )
        finally:
            rebuilt.terminal_release()
        target = "weight" if kind == "single_linear" else "video_aggregate_embed.bias"
        delta = torch.full_like(models["text_projection"].state_dict()[target], 0.125)
        path = tmp_path / "overlay.safetensors"
        save_file({"delta": delta}, path)
        asset = AssetRef(
            digest=digest_file(path),
            name=path.name,
            size=path.stat().st_size,
            resolver=Resolver(path),
        )
        overlay = inference.PatchOverlay.from_decoded(
            source=arm._weight_source_ref(inference, asset),
            dialect="none",
            key_map="native.dinkster.test.v1",
            strength_model=0.0,
            strength_clip=1.0,
            patches=(
                inference.OverlayPatch(
                    "text_projection",
                    inference.PatchTarget(target),
                    inference.DiffPatchRef("delta"),
                ),
            ),
        )
        patched = handle.clone((overlay,), source_resolvers={asset.digest: asset.resolver})
        try:
            assert patched.resource_identity != handle.resource_identity
            with patched.stage(), torch.inference_mode():
                changed = patched.runtime.encode_text("a bird").embeddings
            assert not torch.equal(changed, expected)
        finally:
            patched.terminal_release()
        with handle.stage(), torch.inference_mode():
            torch.testing.assert_close(
                handle.runtime.encode_text("a bird").embeddings, expected, rtol=0, atol=0
            )
    finally:
        handle.terminal_release()
