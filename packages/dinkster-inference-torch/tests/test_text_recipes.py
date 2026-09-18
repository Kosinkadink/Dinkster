"""Real tiny encoders exercise recipe assembly, source order, and text options."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from dinkster_inference import FLOAT32, FLUX_DEV, ComponentPlan
from dinkster_inference.clip_bpe import load_clip_bpe
from dinkster_inference.clip_text import CLIP_L_TEXT_CONFIG
from dinkster_inference.component_registry import ComponentDescriptor, DetectedComponents
from dinkster_inference.prompt_tokens import CLIP_G_PROFILE, PromptTokenizer
from dinkster_inference.sources import load_safetensors_header
from dinkster_inference.t5_spm import load_t5_spm
from dinkster_inference.t5_text import T5_XXL_CONFIG
from dinkster_inference.text_recipes import (
    TextRecipeBinding,
    TextRecipeComponent,
    resolve_text_recipe,
)
from dinkster_inference.weights import WeightSource
from dinkster_inference_torch.clip_text import (
    SDXL_CLIP_POLICY,
    ClipTextEncoder,
    ClipTextModel,
    compose_sdxl_conditioning,
)
from dinkster_inference_torch.t5_text import T5TextEncoder, T5TextModel, compose_flux_conditioning
from dinkster_inference_torch.text_recipes import (
    LoadedTextRecipe,
    TextRecipeRuntime,
    assemble_text_recipe,
)
from safetensors.torch import save_file


@pytest.mark.parametrize("role", ("clip_l", "t5xxl"))
def test_default_runtime_requires_explicit_packing_profile(role: str) -> None:
    with torch.device("meta"):
        model = (
            ClipTextModel(CLIP_L_TEXT_CONFIG) if role == "clip_l" else T5TextModel(T5_XXL_CONFIG)
        )
    component = TextRecipeComponent(
        0, role, ComponentPlan(role, Path("unused.safetensors"), model.config, {}, {}, {}), None
    )
    binding = TextRecipeBinding(
        "test.no_profile", "test.no_profile", (component,), "unused", "unused", None, (role,)
    )
    loaded = LoadedTextRecipe(binding, torch.nn.ModuleDict({role: model}), ())
    with pytest.raises(
        ValueError, match="^CLIP/T5 text runtime requires a tokenizer packing profile$"
    ):
        TextRecipeRuntime(loaded)


@pytest.mark.parametrize("kind", ("sdxl", "flux", "stable_diffusion"))
@pytest.mark.parametrize("reverse", (False, True))
def test_ordered_files_load_and_encode_with_the_declared_recipe(
    tmp_path: Path, kind: str, reverse: bool
) -> None:
    clip_config = replace(
        CLIP_L_TEXT_CONFIG,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
    )
    clip_l = ClipTextModel(clip_config)
    second = (
        ClipTextModel(replace(clip_config, hidden_act="gelu"))
        if kind == "sdxl"
        else T5TextModel(
            replace(T5_XXL_CONFIG, d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=2)
        )
    )
    models = (("clip_l", clip_l), ("clip_g" if kind == "sdxl" else "t5xxl", second))
    if kind == "stable_diffusion":
        models = models[:1]
    generator = torch.Generator().manual_seed(917)
    for _, model in models:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.03)
    if reverse:
        models = tuple(reversed(models))

    def detect(_source: WeightSource, _path: Path) -> tuple[()]:
        return ()

    descriptor = ComponentDescriptor(
        FLUX_DEV,
        detect,
        tuple(role for role, _ in models),
        tuple(role for role, _ in models),
        (),
        "unused",
        "unused",
    )
    detected = []
    paths = []
    for role, model in models:
        path = tmp_path / f"{role}.safetensors"
        state = model.state_dict()
        save_file({f"physical.{role}.{k}": value for k, value in state.items()}, path)
        plan = ComponentPlan(
            component=role,
            path=path,
            config=model.config,
            keys={key: f"physical.{role}.{key}" for key in state},
            dtypes={key: FLOAT32 for key in state},
            quant={},
        )
        detected.append((DetectedComponents(descriptor, ((role, plan),)),))
        paths.append(path)
    binding = resolve_text_recipe(tuple(detected), kind)
    with ExitStack() as stack:
        loaded = assemble_text_recipe(
            binding,
            compute_dtype=torch.float32,
            sources=tuple(load_safetensors_header(path) for path in paths),
            source_files=tuple(stack.enter_context(path.open("rb")) for path in paths),
        )
    for role, model in models:
        for key, expected in model.state_dict().items():
            assert torch.equal(loaded.module[role].state_dict()[key], expected)
    runtime = TextRecipeRuntime(loaded)
    text = "a (bright:1.2) bird"
    clip_spans = PromptTokenizer(encode_word=load_clip_bpe().encode).tokenize(text)
    for hidden_layer in (None, -1):
        actual = runtime.encode_text(text, hidden_layer=hidden_layer, min_length=32)
        if kind == "sdxl":
            assert isinstance(second, ClipTextModel)
            expected = compose_sdxl_conditioning(
                ClipTextEncoder(clip_l, policy=SDXL_CLIP_POLICY).encode(
                    clip_spans, hidden_layer=hidden_layer
                ),
                ClipTextEncoder(second, profile=CLIP_G_PROFILE, policy=SDXL_CLIP_POLICY).encode(
                    clip_spans, hidden_layer=hidden_layer
                ),
            )
        elif kind == "stable_diffusion":
            expected = ClipTextEncoder(clip_l).encode(clip_spans, hidden_layer=hidden_layer)
        else:
            assert isinstance(second, T5TextModel)
            t5_spans = PromptTokenizer(encode_word=load_t5_spm().encode).tokenize(text)
            expected = compose_flux_conditioning(
                T5TextEncoder(second).encode(t5_spans, min_length=32),
                ClipTextEncoder(clip_l).encode(clip_spans, hidden_layer=hidden_layer),
            )
        assert torch.equal(actual.embeddings, expected.embeddings)
        assert actual.pooled is not None and expected.pooled is not None
        assert torch.equal(actual.pooled, expected.pooled)
        assert runtime.text_conditioning_carrier(actual)
