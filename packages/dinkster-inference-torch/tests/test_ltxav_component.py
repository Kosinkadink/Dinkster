from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    LTXAV_22B_V25_VAE_CONFIG,
    LTXAV_BWE_VOCODER_CONFIG,
    LTXAVAudioCodecAssemblyError,
    LTXAVComponentAssemblyError,
)
from dinkster_inference_torch import ltxav_component as component
from dinkster_inference_torch.quant_linear import Fp8Linear, Int8Embedding, Int8Linear, Nvfp4Linear


class _FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


class _MissingResolver:
    def resolve(self, digest: str) -> None:
        del digest
        return None


def _asset(path: Path) -> AssetRef:
    return AssetRef(
        digest_file(path), path.name, path.stat().st_size, resolver=_FixedResolver(path)
    )


def _component_path(tmp_path: Path) -> Path:
    path = tmp_path / "audio-codec.safetensors"
    path.write_bytes(b"audio-codec")
    return path


def _patch_planning(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_identity: str,
    seen: dict[str, object],
    bwe: bool = False,
) -> SimpleNamespace:
    source = SimpleNamespace()
    planned = SimpleNamespace(
        audio_vae=object(),
        vocoder=SimpleNamespace(config=LTXAV_BWE_VOCODER_CONFIG if bwe else object()),
    )

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path, header_source=source)
        return source

    def plan(candidate: object, *, path: Path) -> object:
        seen.update(planner_source=candidate, planner_path=path)
        return planned

    def identity(candidate: object) -> str:
        seen.update(identity_plan=candidate)
        return expected_identity

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_ltxav_split_audio_codec", plan)
    monkeypatch.setattr(component, "ltxav_audio_codec_runtime_identity", identity)
    return planned


@pytest.mark.parametrize(
    ("role", "config", "builder_name"),
    (
        ("diffusion", object(), "LTXAVModel"),
        ("gemma3_12b", object(), "GemmaTextModel"),
        ("gemma4_12b", object(), "GemmaTextModel"),
        ("text_projection", "single_linear", "build_projection"),
        ("text_projection", "dual_linear", "build_projection"),
        ("text_projection", "dual_linear_gemma4", "build_projection"),
        ("connectors", object(), "LtxTextConnectors"),
        ("latent_upscaler", object(), "LTXLatentUpsampler"),
        ("vae", object(), "LTXVideoVAE"),
        ("vae", LTXAV_22B_V25_VAE_CONFIG, "LTXDiffusionVideoVAE"),
    ),
)
def test_component_loader_preserves_role_identity_and_one_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    config: object,
    builder_name: str,
) -> None:
    path = tmp_path / f"{role}.safetensors"
    path.write_bytes(role.encode())
    asset = _asset(path)
    expected_identity = "native:dinkster.ltxav:" + "1" * 64
    source = SimpleNamespace()
    planned = SimpleNamespace(
        role=role,
        component=SimpleNamespace(
            config=config,
            quant=(
                {
                    "transformer_blocks.2.attn1.to_q": SimpleNamespace(
                        format="float8_e4m3fn",
                        full_precision_matmul=False,
                    )
                }
                if role == "diffusion"
                else {}
            ),
        ),
        tokenizer_source_key=(
            "spiece_model"
            if role == "gemma3_12b"
            else "tokenizer_json"
            if role == "gemma4_12b"
            else ""
        ),
    )
    seen: dict[str, object] = {}
    if role == "gemma4_12b":
        module = torch.nn.ModuleDict(
            {
                "embedding": Int8Embedding(
                    2,
                    16,
                    compute_dtype=torch.bfloat16,
                    convrot=False,
                    convrot_groupsize=0,
                ),
                "linear": Int8Linear(
                    16,
                    2,
                    bias=False,
                    compute_dtype=torch.bfloat16,
                    convrot=False,
                    convrot_groupsize=0,
                ),
            }
        )
    elif role == "gemma3_12b":
        module = torch.nn.ModuleDict(
            {
                "fp8": Fp8Linear(
                    16,
                    16,
                    bias=False,
                    compute_dtype=torch.bfloat16,
                ),
                "nvfp4": Nvfp4Linear(
                    16,
                    16,
                    bias=False,
                    compute_dtype=torch.bfloat16,
                ),
            }
        )
    else:
        module = torch.nn.Identity()
    attention_kernel = object()
    attention_token = cast("Any", object())
    attention_calls: list[tuple[str, object, object]] = []

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path, header_source=source)
        return source

    def plan(candidate: object, *, role: str, path: Path) -> object:
        seen.update(planner_source=candidate, planner_role=role, planner_path=path)
        return planned

    def identity(candidate: object, dtype: object, **kwargs: object) -> str:
        seen.update(identity_plan=candidate, identity_dtype=dtype, identity_kwargs=kwargs)
        return expected_identity

    def resolve_attention(attention_role: str, policy: object, token: object) -> SimpleNamespace:
        attention_calls.append((attention_role, policy, token))
        return SimpleNamespace(kernel=attention_kernel)

    def load(candidate: object, builder: object, **kwargs: object) -> torch.nn.Module:
        pinned = cast(SimpleNamespace, seen["planner_source"])
        seen.update(load_candidate=candidate, builder=builder, load_kwargs=kwargs)
        assert pinned.file is seen["header_handle"]
        assert not pinned.file.closed
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        return module

    def load_tokenizer(
        handle: BinaryIO, candidate: object, keys: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        assert handle is seen["header_handle"]
        assert candidate is source
        tokenizer_key = cast("str", planned.tokenizer_source_key)
        assert keys == (tokenizer_key,)
        return {tokenizer_key: torch.tensor([1, 2, 3], dtype=torch.uint8)}

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_ltxav_split_component", plan)
    monkeypatch.setattr(component, "ltxav_component_runtime_identity", identity)
    monkeypatch.setattr(component, "resolve_role_attention", resolve_attention)
    monkeypatch.setattr(component, "_load_component", load)
    monkeypatch.setattr(component, "load_tensors_from_file", load_tokenizer)

    loaded = component.load_ltxav_component(
        path,
        asset=asset,
        expected_role=cast("Any", role),
        expected_identity=expected_identity,
        compute_dtype=torch.bfloat16,
        attention_policy="sdpa",
        attention_route_token=attention_token,
        attention_backend=cast(
            "Any",
            {
                "diffusion": "flux",
                "gemma3_12b": "qwen",
                "gemma4_12b": "qwen",
                "connectors": "flux",
            }.get(role),
        ),
    )

    assert loaded.role == role
    assert loaded.plan is planned.component
    assert loaded.runtime_identity == expected_identity
    assert seen["header_path"] == path
    assert seen["planner_role"] == role
    assert seen["planner_path"] == path
    assert seen["identity_plan"] is planned
    builder = seen["builder"]
    if role in ("diffusion", "gemma3_12b", "gemma4_12b", "connectors"):
        assert isinstance(builder, partial)
        assert builder.func.__name__ == builder_name
        assert builder.keywords == {"attention_kernel": attention_kernel}
        expected_attention_role = "qwen" if role in ("gemma3_12b", "gemma4_12b") else "flux"
        assert attention_calls == [(expected_attention_role, "sdpa", attention_token)]
    else:
        assert getattr(builder, "__name__", None) == builder_name
        assert attention_calls == []
        if role == "text_projection":
            if config == "single_linear":
                projection = torch.nn.Linear(2, 3, bias=False)

                def linear(*_args: object, **_kwargs: object) -> torch.nn.Module:
                    return projection

                operations = SimpleNamespace(linear=linear)
                assert cast("Any", builder)(config, operations=operations) is projection
            else:

                def linear(*_args: object, **_kwargs: object) -> torch.nn.Module:
                    return torch.nn.Identity()

                operations = SimpleNamespace(linear=linear)
                projection = cast("Any", builder)(config, operations=operations)
                assert type(projection) is component.LtxDualTextProjection
    assert seen["identity_kwargs"] == {
        "attention_policy": "sdpa",
        "attention_route_token": attention_token,
    }
    load_kwargs = cast("dict[str, object]", seen["load_kwargs"])
    assert load_kwargs["source_file"] is seen["header_handle"]
    assert load_kwargs["source"] is source
    assert load_kwargs["compute_dtype"] is torch.bfloat16
    assert load_kwargs["fp8_matmul"] is (role == "diffusion")
    if role in ("gemma3_12b", "gemma4_12b"):
        assert type(loaded.module) is component.LTXAVGemmaComponent
        gemma = loaded.module
        assert gemma.model is module
        assert gemma.tokenizer_model == b"\x01\x02\x03"
        assert loaded.tokenizer_model == b"\x01\x02\x03"
        if role == "gemma4_12b":
            packed = cast("torch.nn.ModuleDict", module)
            assert cast("Int8Embedding", packed["embedding"]).compute_dtype is torch.bfloat16
            linear = cast("Int8Linear", packed["linear"])
            assert linear.compute_dtype is torch.float32
            assert linear.full_precision_matmul
        else:
            packed = cast("torch.nn.ModuleDict", module)
            for name in ("fp8", "nvfp4"):
                linear = cast("Fp8Linear | Nvfp4Linear", packed[name])
                assert linear.compute_dtype is torch.float32
                assert linear.full_precision_matmul
    else:
        assert loaded.module is module
        assert loaded.tokenizer_model is None


def test_component_loader_refuses_identity_mismatch_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "diffusion.safetensors"
    path.write_bytes(b"diffusion")
    planned = SimpleNamespace(
        role="diffusion",
        component=SimpleNamespace(config=object()),
        tokenizer_source_key="",
    )

    def load_header(_handle: BinaryIO, *, path: Path) -> object:
        return SimpleNamespace(path=path)

    def plan(*_args: object, **_kwargs: object) -> object:
        return planned

    def identity(*_args: object, **_kwargs: object) -> str:
        return "native:dinkster.ltxav:" + "2" * 64

    def load(*_args: object, **_kwargs: object) -> object:
        pytest.fail("payload loading must not run")

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_ltxav_split_component", plan)
    monkeypatch.setattr(component, "ltxav_component_runtime_identity", identity)
    monkeypatch.setattr(component, "_load_component", load)

    with pytest.raises(LTXAVComponentAssemblyError, match="expected component identity"):
        component.load_ltxav_component(
            path,
            asset=_asset(path),
            expected_role="diffusion",
            expected_identity="native:dinkster.ltxav:" + "3" * 64,
            compute_dtype=torch.bfloat16,
            attention_backend="flux",
        )


@pytest.mark.parametrize(
    ("bwe", "vocoder_builder"),
    ((False, "LTXVocoder"), (True, "LTXVocoderWithBWE")),
)
def test_audio_codec_loader_preserves_identity_and_one_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bwe: bool,
    vocoder_builder: str,
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    expected_identity = "native:dinkster.ltxav:" + "a" * 64
    seen: dict[str, object] = {}
    planned = _patch_planning(
        monkeypatch,
        expected_identity=expected_identity,
        seen=seen,
        bwe=bwe,
    )
    audio_vae = torch.nn.Identity()
    vocoder = torch.nn.Identity()
    loads: list[tuple[object, object, dict[str, object]]] = []

    def load(candidate: object, builder: object, **kwargs: object) -> torch.nn.Module:
        pinned = cast(SimpleNamespace, seen["planner_source"])
        loads.append((candidate, builder, kwargs))
        assert kwargs["source_file"] is seen["header_handle"]
        assert kwargs["source"] is seen["header_source"]
        assert pinned.file is kwargs["source_file"]
        assert not pinned.file.closed
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        return audio_vae if candidate is planned.audio_vae else vocoder

    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_ltxav_audio_codec(
        path,
        asset=asset,
        expected_identity=expected_identity,
    )

    assert loaded.module.audio_vae is audio_vae
    assert loaded.module.vocoder is vocoder
    assert loaded.plan is planned
    assert loaded.runtime_identity == expected_identity
    assert seen["header_path"] == path
    assert seen["identity_plan"] is planned
    assert [candidate for candidate, _, _ in loads] == [planned.audio_vae, planned.vocoder]
    assert [getattr(builder, "__name__", None) for _, builder, _ in loads] == [
        "LTXAudioVAE",
        vocoder_builder,
    ]
    for _, _, kwargs in loads:
        assert kwargs["compute_dtype"] is torch.float32
        assert kwargs["fp8_matmul"] is False
        assert kwargs["storage_dtype_follows_compute"] is True


def test_audio_codec_loader_refuses_identity_mismatch_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    seen: dict[str, object] = {}
    _patch_planning(
        monkeypatch,
        expected_identity="native:dinkster.ltxav:" + "b" * 64,
        seen=seen,
    )

    def fail_load(*_args: object, **_kwargs: object) -> torch.nn.Module:
        pytest.fail("payload loading must not run")

    monkeypatch.setattr(component, "_load_component", fail_load)

    with pytest.raises(LTXAVAudioCodecAssemblyError, match="expected component identity"):
        component.load_ltxav_audio_codec(
            path,
            asset=_asset(path),
            expected_identity="native:dinkster.ltxav:" + "c" * 64,
        )


def test_audio_codec_loader_refuses_size_and_unavailable_artifact(tmp_path: Path) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    identity = "native:dinkster.ltxav:" + "d" * 64
    mismatched = AssetRef(
        asset.digest,
        asset.name,
        asset.size + 1,
        resolver=_FixedResolver(path),
    )
    with pytest.raises(LTXAVAudioCodecAssemblyError, match="byte size differs"):
        component.load_ltxav_audio_codec(
            path,
            asset=mismatched,
            expected_identity=identity,
        )
    unavailable = AssetRef(
        asset.digest,
        asset.name,
        asset.size,
        resolver=_MissingResolver(),
    )
    with pytest.raises(LTXAVAudioCodecAssemblyError, match="artifact is unavailable"):
        component.load_ltxav_audio_codec(
            path,
            asset=unavailable,
            expected_identity=identity,
        )
