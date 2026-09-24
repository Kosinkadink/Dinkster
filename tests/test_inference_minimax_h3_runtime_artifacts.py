from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from tools.inference_parity.minimax_h3_runtime_receipts import (  # pyright: ignore[reportMissingImports]
    ArtifactReceiptError,
    load_manifest,
    validate_manifest,
)

from tools.evidence_paths import EVIDENCE_ROOT

MANIFEST = EVIDENCE_ROOT / "tools/inference_parity/minimax_h3_runtime_artifacts.json"


def test_runtime_authority_pins_int8_graph_and_bf16_dit_alternates() -> None:
    manifest = load_manifest(MANIFEST)
    validate_manifest(manifest)
    assert manifest["provider"]["revision"] == "3f57e8291d2ef846f9a074b1b76d2767db434abe"
    assert manifest["license"]["acceptance_claimed"] is False
    artifacts = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert set(artifacts) == {
        "fl2va-dit",
        "ref2va-dit",
        "qwen3vl-32b-conditioner",
        "video-vae",
        "audio-vae",
    }
    assert artifacts["fl2va-dit"]["bytes"] == 34038892334
    assert artifacts["ref2va-dit"]["sha256"] == (
        "9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9"
    )
    assert artifacts["qwen3vl-32b-conditioner"]["sha256"] == (
        "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923"
    )
    alternates = {artifact["role"]: artifact for artifact in manifest["alternate_dit_artifacts"]}
    assert alternates == {
        "fl2va-dit": {
            "bytes": 66280487368,
            "provider_path": "diffusion_models/minimax_h3_fl2va_bf16.safetensors",
            "role": "fl2va-dit",
            "sha256": "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182",
            "source_url": (
                "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/"
                "3f57e8291d2ef846f9a074b1b76d2767db434abe/"
                "diffusion_models/minimax_h3_fl2va_bf16.safetensors"
            ),
        },
        "ref2va-dit": {
            "bytes": 66280487368,
            "provider_path": "diffusion_models/minimax_h3_ref2va_bf16.safetensors",
            "role": "ref2va-dit",
            "sha256": "e32c54c1a7b4f5f397f195cea267ccb18806303bb665678c4bee60953bdf3026",
            "source_url": (
                "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/"
                "3f57e8291d2ef846f9a074b1b76d2767db434abe/"
                "diffusion_models/minimax_h3_ref2va_bf16.safetensors"
            ),
        },
    }


def test_runtime_authority_refuses_mutable_incomplete_or_false_claims() -> None:
    original = load_manifest(MANIFEST)
    cases = []
    for path, value in (
        (("schema",), True),
        (("provider", "revision"), "main"),
        (("provider", "gated"), 0),
        (("license", "acceptance_claimed"), True),
        (("license", "territory_statement"), "worldwide"),
        (("artifacts", 0, "bytes"), 0),
        (("artifacts", 0, "sha256"), "0" * 64),
        (("artifacts", 1, "role"), "fl2va-dit"),
        (("artifacts", 2, "source_url"), "https://example.invalid/model"),
        (("alternate_dit_artifacts", 0, "bytes"), 0),
        (("alternate_dit_artifacts", 1, "sha256"), "0" * 64),
        (("alternate_dit_artifacts", 1, "role"), "fl2va-dit"),
    ):
        broken = deepcopy(original)
        target: Any = broken
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        cases.append(broken)
    missing = deepcopy(original)
    missing["artifacts"].pop()
    cases.append(missing)
    missing_alternate = deepcopy(original)
    missing_alternate["alternate_dit_artifacts"].pop()
    cases.append(missing_alternate)
    extra = deepcopy(original)
    extra["note"] = "unverified"
    cases.append(extra)
    for broken in cases:
        with pytest.raises(ArtifactReceiptError):
            validate_manifest(broken)
