from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from dinkster_model_yue2 import YUE2_MODEL_NODE_IDS, YUE2_MODEL_NODES, register_inference
from dinkster_model_yue2.codec import AudioOobleckVAE
from dinkster_model_yue2.declarations import (
    FAMILY_ID,
    FRAMES_PER_SECOND,
    YUE2_FAMILY,
    YuE2Detector,
)
from dinkster_model_yue2.provider import execute_empty_latent
from dinkster_model_yue2.runtime import (
    YuE2Conditioning,
    materialize_yue2_conditioning,
    yue2_conditioning_to_carrier,
)
from dinkster_model_yue2.text import (
    ABC_END,
    CODEC_OFFSET,
    _rope_positions,  # pyright: ignore[reportPrivateUsage]
    chunk_ranges,
    distribution,
)
from dinkster_schema import build_schemas
from dinkster_workers import load_manifest

PACKAGE = Path(__file__).resolve().parents[1]


class _YuE2Header:
    def __init__(self, position_shape: tuple[int, ...] = (24_576, 2_048)) -> None:
        self._shapes = {
            "model.diffusion_model.latent_pos_embed.pe": position_shape,
            "model.diffusion_model.vae2llm.weight": (2_048, 64),
            "model.diffusion_model.llm2vae.weight": (64, 2_048),
            "text_encoders.model.embed_tokens.weight": (184_704, 2_048),
            "text_encoders.model.lm_head.weight": (184_704, 2_048),
            "text_encoders.yue2_tokenizer_json": (1_000,),
            "vae.decoder.layers.6.layers.1.weight_v": (64, 64, 7),
        }

    def keys(self) -> tuple[str, ...]:
        return tuple(self._shapes)

    def entry(self, key: str) -> object:
        return SimpleNamespace(geometry=SimpleNamespace(shape=self._shapes[key]))


def test_manifest_declares_pack_owned_family_registration() -> None:
    manifest = load_manifest(PACKAGE / "dinkster-pack.toml")

    assert manifest.name == "dinkster-model-yue2"
    assert manifest.extension.entries.inference == "dinkster_model_yue2:register_inference"
    assert manifest.extension.capabilities == ("model-family-registration",)
    assert tuple((item.registry, item.id) for item in manifest.provides.registry) == (
        ("dinkster.model-families", FAMILY_ID),
    )


def test_inference_contribution_is_complete_and_pack_owned() -> None:
    contribution = register_inference()

    assert contribution.families == (YUE2_FAMILY,)
    assert tuple(item.id for item in contribution.components) == (FAMILY_ID,)
    assert tuple(item.id for item in contribution.assemblies) == (FAMILY_ID,)
    assert contribution.components[0].roles == ("diffusion", "text", "vae")


def test_detector_requires_the_exact_combined_yue2_signature() -> None:
    detected = YuE2Detector().detect(_YuE2Header())

    assert detected is not None
    assert detected.family_id == FAMILY_ID
    assert detected.fields == {
        "context": 24_576,
        "hidden": 2_048,
        "latent_channels": 64,
        "sample_rate": 48_000,
    }
    assert YuE2Detector().detect(_YuE2Header((8_192, 2_048))) is None


def test_nodes_match_upstream_yue2_workflow_boundaries() -> None:
    schemas = build_schemas(YUE2_MODEL_NODES)

    assert (
        tuple(schemas)
        == YUE2_MODEL_NODE_IDS
        == (
            "dinkster.yue2_generate_abc",
            "dinkster.yue2_generate_music",
            "dinkster.empty_yue2_latent_audio",
            "dinkster.yue2_decode_audio",
        )
    )
    music = schemas["dinkster.yue2_generate_music"]
    assert {item.id for item in music.inputs} == {
        "clip",
        "style",
        "lyrics",
        "abc",
        "seed",
        "mode",
        "max_duration",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "cfg_scale",
    }
    assert tuple(item.id for item in music.outputs) == ("conditioning", "seconds")
    cfg = next(item for item in music.inputs if item.id == "cfg_scale")
    assert cfg.required is False
    assert cfg.default is None
    decode = schemas["dinkster.yue2_decode_audio"]
    assert tuple(item.id for item in decode.inputs) == ("samples", "vae")
    assert tuple(item.id for item in decode.outputs) == ("audio",)


def test_empty_latent_uses_generated_audio_geometry() -> None:
    result = execute_empty_latent(seconds=1.24, batch_size=2)
    latent = result["latent"]

    assert isinstance(latent, dict)
    assert latent["samples"].shape == (2, 64, round(1.24 * FRAMES_PER_SECOND))
    assert latent["type"] == "audio"
    assert latent["downscale_ratio_temporal"] == 1920


def test_distribution_enforces_each_phase_vocabulary_and_terminal_budget() -> None:
    logits = torch.zeros((1, 184_704))

    abc = distribution(
        logits,
        [],
        0,
        "abc",
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        penalty_window=1,
        min_tokens=1,
    )
    semantic = distribution(
        logits,
        [],
        1,
        "semantic",
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        penalty_window=1,
        min_tokens=1,
    )

    assert torch.isneginf(abc[0, ABC_END])
    assert torch.isfinite(abc[0, 42])
    assert torch.isneginf(abc[0, CODEC_OFFSET])
    assert torch.isfinite(semantic[0, CODEC_OFFSET])
    assert torch.isneginf(semantic[0, 42])


def test_chunking_and_conditioning_metadata_round_trip() -> None:
    chunks = chunk_ranges(30, 24, context=47)
    assert chunks == ((0, 10), (10, 20), (20, 30))
    condition = YuE2Conditioning(
        torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
        None,
        ((0, 3, 0, 3),),
        3,
    )

    restored = materialize_yue2_conditioning(yue2_conditioning_to_carrier(condition), device="cpu")

    assert torch.equal(restored.embeddings, condition.embeddings)
    assert restored.chunks == condition.chunks
    assert restored.frames == condition.frames


def test_cfg_rope_positions_do_not_advance_left_padding() -> None:
    positions = torch.tensor([[0, 1, 2], [0, 0, 1]])
    cosine, sine, _negative_sine = _rope_positions(8, positions, 1_000_000.0)

    assert cosine.shape == (2, 1, 3, 8)
    assert torch.equal(cosine[0, :, 0], cosine[1, :, 0])
    assert torch.equal(cosine[0, :, 1], cosine[1, :, 2])
    assert torch.equal(sine[0, :, 1], sine[1, :, 2])


def test_yue2_codec_matches_official_1920x_state_layout() -> None:
    state = AudioOobleckVAE().state_dict()

    assert state["encoder.layers.0.parametrizations.weight.original1"].shape == (64, 2, 7)
    assert state["encoder.layers.6.layers.4.parametrizations.weight.original1"].shape == (
        2048,
        1024,
        12,
    )
    assert state["decoder.layers.0.parametrizations.weight.original1"].shape == (2048, 64, 7)
    assert len(state) == 435
