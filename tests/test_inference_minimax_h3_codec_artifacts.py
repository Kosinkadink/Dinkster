from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from dinkster_inference import MINIMAX_H3_CONFIG, builtin_families
from tools.inference_parity import minimax_h3_codec_receipts
from tools.inference_parity.minimax_h3_codec_receipts import (
    ArtifactReceiptError,
    load_manifest,
    validate_manifest,
)
from tools.inference_parity.qwen_image_receipts import validate_manifest as validate_qwen

from tools.evidence_paths import EVIDENCE_ROOT

MANIFEST_PATH = EVIDENCE_ROOT / "tools/inference_parity/minimax_h3_codec_artifacts.json"
MODULE_PATH = EVIDENCE_ROOT / "tools/inference_parity/minimax_h3_codec_receipts.py"
QWEN_MANIFEST_PATH = EVIDENCE_ROOT / "tools/inference_parity/qwen_image_artifacts.json"


def _set_path(value: dict[str, Any], path: tuple[object, ...], replacement: object) -> None:
    target: Any = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement


def test_minimax_h3_codec_authority_pins_exact_official_sources_and_license() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    validate_manifest(manifest)
    assert set(manifest) == {"artifacts", "license", "provider", "schema", "scope"}
    assert manifest["schema"] == 1
    assert manifest["scope"] == ["MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE"]
    assert manifest["provider"] == {
        "api_url": (
            "https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/"
            "revision/014cd40f7e177756c6b2473c0d93b1c89a790dd2?blobs=true"
        ),
        "model_card_url": (
            "https://huggingface.co/Comfy-Org/MiniMax-H3/raw/"
            "014cd40f7e177756c6b2473c0d93b1c89a790dd2/README.md"
        ),
        "repository": "Comfy-Org/MiniMax-H3",
        "revision": "014cd40f7e177756c6b2473c0d93b1c89a790dd2",
    }
    assert manifest["license"] == {
        "acceptance_claimed": False,
        "bytes": 17604,
        "date": "2026-08-02",
        "name": "MiniMax H3 COMMUNITY LICENSE AGREEMENT",
        "repository": "MiniMaxAI/MiniMax-H3",
        "revision": "9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6",
        "sha256": "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44",
        "source_url": (
            "https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/"
            "9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6/LICENSE"
        ),
        "territory_statement": (
            "Applicable Territory is worldwide excluding the Excluded Territories; "
            "Excluded Territories are the European Union, United Kingdom, Republic of "
            "Korea, and United States of America."
        ),
    }
    artifacts = {item["role"]: item for item in manifest["artifacts"]}
    assert artifacts == {
        "video-vae": {
            "bytes": 5207808496,
            "component_id": "MiniMaxH3VideoVAE",
            "digest_authority": "provider_api.lfs.sha256",
            "provider_path": "vae/minimax_h3_video_vae_fp16.safetensors",
            "role": "video-vae",
            "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
            "source_url": (
                "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/"
                "014cd40f7e177756c6b2473c0d93b1c89a790dd2/"
                "vae/minimax_h3_video_vae_fp16.safetensors"
            ),
        },
        "audio-vae": {
            "bytes": 605254808,
            "component_id": "MiniMaxH3AudioVAE",
            "digest_authority": "provider_api.lfs.sha256",
            "provider_path": "vae/minimax_h3_audio_vae_fp32.safetensors",
            "role": "audio-vae",
            "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
            "source_url": (
                "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/"
                "014cd40f7e177756c6b2473c0d93b1c89a790dd2/"
                "vae/minimax_h3_audio_vae_fp32.safetensors"
            ),
        },
    }


def test_authority_refuses_every_mutable_or_wrong_exact_fact() -> None:
    original = load_manifest(MANIFEST_PATH)
    mutations = (
        (("schema",), True),
        (("scope",), ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"]),
        (("provider", "repository"), "MiniMaxAI/MiniMax-H3"),
        (("provider", "revision"), "main"),
        (
            ("provider", "api_url"),
            "https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/"
            "revision/014cd40f7e177756c6b2473c0d93b1c89a790dd2",
        ),
        (("provider", "api_url"), "https://huggingface.co/api/models/Comfy-Org/MiniMax-H3"),
        (("provider", "model_card_url"), "https://huggingface.co/Comfy-Org/MiniMax-H3"),
        (("license", "acceptance_claimed"), True),
        (("license", "bytes"), 17604.0),
        (("license", "date"), "2026-08-03"),
        (("license", "name"), "MiniMax H3 License"),
        (("license", "repository"), "Comfy-Org/MiniMax-H3"),
        (("license", "revision"), "main"),
        (("license", "sha256"), "0" * 64),
        (("license", "source_url"), "https://huggingface.co/MiniMaxAI/MiniMax-H3/LICENSE"),
        (("license", "territory_statement"), "worldwide"),
        (("artifacts", 0, "role"), "audio-vae"),
        (("artifacts", 0, "component_id"), "MiniMaxH3AudioVAE"),
        (("artifacts", 0, "provider_path"), "minimax_h3_video_vae_fp16.safetensors"),
        (("artifacts", 0, "source_url"), "https://huggingface.co/Comfy-Org/MiniMax-H3"),
        (("artifacts", 0, "bytes"), True),
        (("artifacts", 0, "bytes"), 5207808495),
        (("artifacts", 0, "sha256"), "0" * 64),
        (("artifacts", 0, "digest_authority"), "sha256"),
        (("artifacts", 1, "role"), "video-vae"),
        (("artifacts", 1, "component_id"), "MiniMaxH3VideoVAE"),
        (("artifacts", 1, "provider_path"), "vae/audio.safetensors"),
        (("artifacts", 1, "source_url"), "https://example.invalid/audio"),
        (("artifacts", 1, "bytes"), 0),
        (("artifacts", 1, "sha256"), "f" * 64),
        (("artifacts", 1, "digest_authority"), "provider_card.sha256"),
    )
    for path, replacement in mutations:
        broken = deepcopy(original)
        _set_path(broken, path, replacement)
        with pytest.raises(ArtifactReceiptError):
            validate_manifest(broken)


def test_authority_refuses_missing_extra_duplicate_and_malformed_shapes() -> None:
    original = load_manifest(MANIFEST_PATH)
    cases: list[object] = []
    for field in tuple(original):
        broken = deepcopy(original)
        del broken[field]
        cases.append(broken)
    extra_top = deepcopy(original)
    extra_top["note"] = "not authority"
    cases.append(extra_top)
    for section in ("provider", "license"):
        for field in tuple(original[section]):
            broken = deepcopy(original)
            del broken[section][field]
            cases.append(broken)
        broken = deepcopy(original)
        broken[section]["extra"] = "not authority"
        cases.append(broken)
    for index in range(2):
        for field in tuple(original["artifacts"][index]):
            broken = deepcopy(original)
            del broken["artifacts"][index][field]
            cases.append(broken)
        broken = deepcopy(original)
        broken["artifacts"][index]["extra"] = "not authority"
        cases.append(broken)
    one_role = deepcopy(original)
    one_role["artifacts"].pop()
    cases.append(one_role)
    extra_role = deepcopy(original)
    extra_role["artifacts"].append(deepcopy(extra_role["artifacts"][0]))
    cases.append(extra_role)

    class ArtifactList(list[object]):
        pass

    list_subclass = deepcopy(original)
    list_subclass["artifacts"] = ArtifactList(list_subclass["artifacts"])
    cases.append(list_subclass)
    cases.extend(([], {"schema": 1}, {**deepcopy(original), "artifacts": {}}))
    for broken in cases:
        with pytest.raises(ArtifactReceiptError):
            validate_manifest(broken)  # type: ignore[arg-type]


@pytest.mark.parametrize("digest", ("A" * 64, "0" * 63, " " + "0" * 63, 0, None))
def test_authority_refuses_malformed_digests(digest: object) -> None:
    for path in (("license", "sha256"), ("artifacts", 0, "sha256")):
        broken = load_manifest(MANIFEST_PATH)
        _set_path(broken, path, digest)
        with pytest.raises(ArtifactReceiptError, match="lowercase SHA256"):
            validate_manifest(broken)


def test_shared_loader_refuses_duplicate_json_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":1,"schema":1}', encoding="ascii")
    with pytest.raises(ArtifactReceiptError, match="duplicate JSON object key"):
        load_manifest(duplicate)


def test_component_ids_match_h3_generic_family() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest["scope"] == [
        MINIMAX_H3_CONFIG.video_codec_id,
        MINIMAX_H3_CONFIG.audio_codec_id,
    ]
    assert {item["component_id"] for item in manifest["artifacts"]} == {
        MINIMAX_H3_CONFIG.video_codec_id,
        MINIMAX_H3_CONFIG.audio_codec_id,
    }
    assert MINIMAX_H3_CONFIG.family_id in {family.id for family in builtin_families()}
    assert not hasattr(MINIMAX_H3_CONFIG, "runtime")


def test_authority_module_is_static_source_only() -> None:
    source = MODULE_PATH.read_text(encoding="ascii")
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
    imports = {
        node.module: tuple(alias.name for alias in node.names)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert imports == {
        "__future__": ("annotations",),
        "typing": ("Any",),
        "dinkster_inference.minimax_h3": ("MINIMAX_H3_CONFIG",),
        "tools.inference_parity.qwen_image_receipts": (
            "ArtifactReceiptError",
            "load_manifest",
        ),
    }
    assert called_names <= {
        "ArtifactReceiptError",
        "_exact_json",
        "_expected_artifacts",
        "_require_digest",
        "all",
        "any",
        "frozenset",
        "isinstance",
        "len",
        "set",
        "type",
        "zip",
    }
    assert called_attributes <= {"add", "get", "items", "keys"}
    lowered = source.lower()
    for forbidden in (
        "artifact_root",
        "local_path",
        "acquisition",
        "storage_receipt",
        "blake3",
        "header_",
        "cuda",
        "native_arm",
        "register(",
        "supported =",
    ):
        assert forbidden not in lowered


def test_qwen_receipt_validator_behavior_is_unchanged() -> None:
    qwen_manifest = load_manifest(QWEN_MANIFEST_PATH)
    validate_qwen(qwen_manifest)
    assert minimax_h3_codec_receipts.load_manifest is load_manifest
