from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dinkster_native.families.minimax_h3 import (
    NativeEmptyMiniMaxH3AV,
    NativeMiniMaxH3FL2VAConditioning,
    NativeMiniMaxH3ImageToVideo,
    NativeMiniMaxH3REF2VAConditioning,
    NativeMiniMaxH3ReferenceToVideo,
    NativeMiniMaxH3T2VAConditioning,
)
from dinkster_native.native import (
    MiniMaxH3AudioReferenceValue,
    MiniMaxH3ImageReferenceValue,
    MiniMaxH3VideoReferenceValue,
    ResolutionSelector,
)
from dinkster_schema import comfy_alias_registry_from_wire

ROOT = Path(__file__).parents[1]
ALIAS_PATH = ROOT / "packages/dinkster-compat-comfy/comfy-aliases.json"
COMFYUI_REVISION = "b5cc8830279eae909a59de030af1e50761c36751"


def _registry() -> dict[str, Any]:
    return json.loads(ALIAS_PATH.read_text(encoding="utf-8"))


def test_native_h3_aliases_are_canonical_and_pinned() -> None:
    registry = _registry()
    assert comfy_alias_registry_from_wire(registry)
    h3_records = [
        record for record in registry["records"] if record["source"]["revision"] == COMFYUI_REVISION
    ]
    assert {record["source"]["nodeClass"] for record in h3_records} == {
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3AddGuide",
        "ResolutionSelector",
    }


def test_native_h3_aliases_preserve_source_inputs_and_outputs() -> None:
    records = {record["source"]["nodeClass"]: record for record in _registry()["records"]}
    image = records["MiniMaxH3ImageToVideo"]["replacement"]["cases"][0]
    assert set(image["inputs"]) == {
        "clip",
        "vae",
        "prompt",
        "width",
        "height",
        "length",
        "first_frame",
        "last_frame",
    }
    assert image["outputs"] == {"positive": "positive", "latent": "LATENT"}

    reference = records["MiniMaxH3ReferenceToVideo"]["replacement"]["cases"][0]
    assert set(reference["inputFamilies"]) == {
        "ref_images",
        "ref_videos",
        "ref_video_audios",
        "ref_audios",
    }
    assert all(
        family["kind"] == "copy"
        and family["inputs"] == {"value": {"kind": "copy", "input": "value"}}
        for family in reference["inputFamilies"].values()
    )

    guide = records["MiniMaxH3AddGuide"]["replacement"]["cases"][0]
    assert set(guide["inputs"]) == {
        "positive",
        "latent",
        "frame_idx",
        "vae",
        "audio_vae",
        "image",
        "audio",
    }
    assert guide["outputs"] == {"positive": "positive"}


def test_resolution_selector_matches_comfyui_rounding() -> None:
    assert ResolutionSelector.execute(
        aspect_ratio="16:9 (Widescreen)", megapixels=0.7, multiple=32
    ) == {"width": 1152, "height": 640}
    assert ResolutionSelector.execute(
        aspect_ratio="2:3 (Portrait Photo)", megapixels=1.3, multiple=8
    ) == {"width": 952, "height": 1432}


def test_image_to_video_composes_empty_target_and_selects_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        NativeEmptyMiniMaxH3AV,
        "execute",
        classmethod(lambda cls, **values: calls.append(("empty", values)) or {"latent": "av"}),
    )
    monkeypatch.setattr(
        NativeMiniMaxH3T2VAConditioning,
        "execute",
        classmethod(lambda cls, **values: calls.append(("t2va", values)) or {"conditioning": "t2"}),
    )
    monkeypatch.setattr(
        NativeMiniMaxH3FL2VAConditioning,
        "execute",
        classmethod(
            lambda cls, **values: calls.append(("fl2va", values)) or {"conditioning": "fl"}
        ),
    )

    assert NativeMiniMaxH3ImageToVideo.execute(
        clip="clip",
        vae="vae",
        prompt="prompt",
        width=1344,
        height=768,
        length=124,
    ) == {"positive": "t2", "latent": "av"}
    assert calls == [
        ("empty", {"width": 1344, "height": 768, "frame_count": 124}),
        ("t2va", {"clip": "clip", "target": "av", "prompt": "prompt"}),
    ]

    calls.clear()
    assert NativeMiniMaxH3ImageToVideo.execute(
        clip="clip",
        vae="vae",
        prompt="prompt",
        width=768,
        height=1344,
        length=90,
        first_frame="first",
    ) == {"positive": "fl", "latent": "av"}
    assert calls[-1] == (
        "fl2va",
        {
            "clip": "clip",
            "video_vae": "vae",
            "target": "av",
            "prompt": "prompt",
            "first_image": "first",
            "last_image": None,
        },
    )


def test_reference_to_video_preserves_reference_order_and_video_audio_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        NativeEmptyMiniMaxH3AV,
        "execute",
        classmethod(lambda cls, **values: {"latent": "av"}),
    )

    def condition(cls: type[object], **values: object) -> dict[str, object]:
        captured.update(values)
        return {"conditioning": "ref"}

    monkeypatch.setattr(
        NativeMiniMaxH3REF2VAConditioning,
        "execute",
        classmethod(condition),
    )
    video_audio = {"waveform": "video-wave", "sample_rate": 32_000}
    standalone_audio = {"waveform": "standalone-wave", "sample_rate": 44_100}

    assert NativeMiniMaxH3ReferenceToVideo.execute(
        clip="clip",
        vae="video-vae",
        audio_vae="audio-vae",
        prompt="prompt",
        width=1344,
        height=768,
        length=124,
        ref_image_size="match",
        ref_images={"ref_image_0": "image"},
        ref_videos={"ref_video_2": "frames"},
        ref_video_audios={"ref_video_audio_2": video_audio},
        ref_audios={"ref_audio_1": standalone_audio},
    ) == {"positive": "ref", "latent": "av"}

    references = captured["references"]
    assert isinstance(references, list)
    assert references == [
        MiniMaxH3ImageReferenceValue("image"),
        MiniMaxH3VideoReferenceValue("frames", MiniMaxH3AudioReferenceValue("video-wave", 32_000)),
        MiniMaxH3AudioReferenceValue("standalone-wave", 44_100),
    ]
