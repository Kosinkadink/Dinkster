from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import BFLOAT16, Lumina2ComponentAssemblyError
from dinkster_inference_torch import assemble as assembly
from dinkster_inference_torch import lumina2_component as component
from dinkster_inference_torch import lumina2_runtime, wiring

from tests.test_inference_lumina2 import (  # pyright: ignore[reportMissingImports]
    _checkpoint_geometries,
)
from tests.test_inference_lumina2 import (  # pyright: ignore[reportMissingImports]
    _Source as CheckpointSource,
)


def test_checkpoint_wiring_plans_all_components_and_binds_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = CheckpointSource(tmp_path / "checkpoint.safetensors", _checkpoint_geometries())
    seen: list[tuple[str, object, object]] = []

    def load_header(path: Path) -> object:
        assert path == source.path
        return source

    def load(plan: Any, builder: Any, **kwargs: Any) -> torch.nn.Module:
        seen.append((plan.component, builder.func, kwargs["compute_dtype"]))
        assert callable(builder.keywords["attention_kernel"])
        model = torch.nn.Identity()
        model.__dict__["config"] = plan.config
        return model

    def tokenizer(model: bytes, *, expected_vocab_size: int) -> object:
        assert model == b"tokenizer"
        assert expected_vocab_size == 256000
        return object()

    monkeypatch.setattr(assembly, "load_safetensors_header", load_header)
    monkeypatch.setattr(assembly, "_load_component", load)
    monkeypatch.setattr(lumina2_runtime, "GemmaSentencePieceTokenizer", tokenizer)
    runtime = wiring.load_runtime(source)
    assert type(runtime) is lumina2_runtime.Lumina2Runtime
    assert runtime.family.id == "dinkster.lumina2"
    assert seen == [
        ("diffusion", component.ZImage, torch.bfloat16),
        ("gemma2_2b", component.GemmaTextModel, torch.float32),
        ("vae", component.AutoencoderKL, torch.bfloat16),
    ]
    assembled = runtime.assembled
    assert isinstance(assembled, assembly.AssembledLumina2)
    assert assembled.gemma2_2b.__dict__[component.LUMINA2_TOKENIZER_ATTRIBUTE] == b"tokenizer"
    assert runtime.codec.encoder is assembled.vae
    assert runtime.codec.decoder is assembled.vae
    assert runtime.codec.compute_dtype is torch.bfloat16
    with pytest.raises(wiring.WiringError, match="expected runtime identity"):
        wiring.load_runtime(source, expected_identity="wrong")
    same_codec = wiring.load_runtime(source, vae=source)
    assert same_codec.runtime_identity == runtime.runtime_identity
    wrong_role = CheckpointSource(
        source.path,
        {
            key: geometry
            for key, geometry in _checkpoint_geometries().items()
            if key.startswith("text_encoders.")
        },
    )
    with pytest.raises(wiring.WiringError, match="requires role 'vae'"):
        wiring.load_runtime(source, vae=wrong_role)


@pytest.mark.parametrize(
    "combined,tokenizer_key",
    [
        (True, "text_encoders.spiece_model"),
        (False, "text_encoders.spiece_model"),
        (False, "spiece_model"),
    ],
)
def test_checkpoint_admission_and_loading_share_the_selected_tokenizer_source(
    combined: bool,
    tokenizer_key: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dinkster_inference import load_safetensors_header, plan_native, probe_native
    from dinkster_inference.component_checkpoint import (
        ComponentCheckpointPlan,
        component_source_claims,
    )
    from dinkster_inference.lumina2_component import lumina2_checkpoint_assembly

    geometries = _checkpoint_geometries()
    selected = (
        geometries
        if combined
        else {
            key: geometry
            for key, geometry in geometries.items()
            if key.startswith("text_encoders.")
        }
    )
    if tokenizer_key == "spiece_model":
        selected = {
            (
                "spiece_model"
                if key == "text_encoders.spiece_model"
                else "logit_scale"
                if key.endswith(".logit_scale")
                else "model." + key.removeprefix("text_encoders.gemma2_2b.transformer.model.")
            ): geometry
            for key, geometry in selected.items()
        }
    offsets: dict[str, int] = {}
    header: dict[str, object] = {}
    size = 0
    for key, geometry in selected.items():
        offsets[key] = size
        header[key] = {
            "dtype": {"bfloat16": "BF16", "float32": "F32", "uint8": "U8"}[geometry.dtype.name],
            "shape": list(geometry.shape),
            "data_offsets": [size, size + geometry.nbytes],
        }
        size += geometry.nbytes
    encoded = json.dumps(header).encode()
    path = tmp_path / "checkpoint.safetensors"
    payload = bytes(range(selected[tokenizer_key].nbytes))
    with path.open("w+b") as file:
        total_size = 8 + len(encoded) + size
        if os.name == "nt":
            from dinkster_assets.p2p_windows import make_sparse

            make_sparse(file.fileno(), total_size)
        else:
            file.truncate(total_size)
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        file.seek(8 + len(encoded) + offsets[tokenizer_key])
        file.write(payload)
    sources: dict[str, Any] = {"checkpoint": load_safetensors_header(path)}
    if not combined:
        for role, prefix in (("diffusion", "model.diffusion_model."), ("vae", "vae.")):
            sources[role] = CheckpointSource(
                tmp_path / f"{role}.safetensors",
                {key: geometry for key, geometry in geometries.items() if key.startswith(prefix)},
            )
    assert probe_native(**sources).native
    plan = plan_native(**sources)
    assert isinstance(plan, ComponentCheckpointPlan)
    text = dict(plan.role_plans)["gemma2_2b"]
    assert (path, tokenizer_key) in component_source_claims(text)
    assert lumina2_checkpoint_assembly(plan).tokenizer_source_key == tokenizer_key

    def load(part: Any, _builder: Any, **_kwargs: Any) -> torch.nn.Module:
        model = torch.nn.Identity()
        model.__dict__["config"] = part.config
        return model

    def tokenizer(model: bytes, *, expected_vocab_size: int) -> object:
        assert model == payload
        assert expected_vocab_size == 256000
        return object()

    monkeypatch.setattr(assembly, "_load_component", load)
    monkeypatch.setattr(lumina2_runtime, "GemmaSentencePieceTokenizer", tokenizer)
    runtime = wiring.load_runtime(**sources)
    assert isinstance(runtime, lumina2_runtime.Lumina2Runtime)
    assert isinstance(runtime.assembled, assembly.AssembledLumina2)
    assert runtime.assembled.gemma2_2b.__dict__[component.LUMINA2_TOKENIZER_ATTRIBUTE] == payload


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


class _Source:
    def __init__(self, checkpoint: bool) -> None:
        self._keys = (
            (
                "model.diffusion_model.x_embedder.weight",
                "text_encoders.gemma2_2b.model.embed_tokens.weight",
                "text_encoders.spiece_model",
                "vae.encoder.conv_in.weight",
            )
            if checkpoint
            else ("model.embed_tokens.weight", "spiece_model")
        )

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def read_uint8_configuration_from_file(
        self, handle: BinaryIO, key: str, *, limit: int
    ) -> bytes:
        assert not handle.closed
        assert key in ("spiece_model", "text_encoders.spiece_model")
        assert limit == component.LUMINA2_TOKENIZER_BYTE_CAP
        return b"tokenizer"


def asset(path: Path) -> AssetRef:
    return AssetRef(
        digest_file(path), path.name, path.stat().st_size, resolver=_FixedResolver(path)
    )


def component_path(tmp_path: Path) -> Path:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    return path


def patch_planning(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_identity: str,
    role: str,
    checkpoint: bool,
    seen: dict[str, object],
) -> SimpleNamespace:
    source = _Source(checkpoint)
    planned = SimpleNamespace(component=role)

    def load_header(handle: BinaryIO, *, path: Path) -> _Source:
        seen.update(header_handle=handle, header_path=path, source=source)
        return source

    def artifact_plan(candidate: object, *, path: Path) -> tuple[tuple[str, object], ...]:
        seen.update(planner="artifact", planner_source=candidate, planner_path=path)
        if not checkpoint:
            return ((role, planned),)
        return tuple(
            (
                candidate_role,
                planned if role == candidate_role else SimpleNamespace(component=candidate_role),
            )
            for candidate_role in ("diffusion", "gemma2_2b", "vae")
        )

    def identity(candidate: object, identity_role: str, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_role=identity_role, identity_dtype=dtype)
        return expected_identity

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_lumina2_artifact_components", artifact_plan)
    monkeypatch.setattr(component, "lumina2_component_runtime_identity", identity)
    return planned


@pytest.mark.parametrize(
    ("role", "builder", "checkpoint"),
    (
        ("diffusion", component.ZImage, False),
        ("gemma2_2b", component.GemmaTextModel, False),
        ("vae", component.AutoencoderKL, False),
        ("diffusion", component.ZImage, True),
        ("gemma2_2b", component.GemmaTextModel, True),
        ("vae", component.AutoencoderKL, True),
    ),
)
def test_component_loader_preserves_identity_and_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    builder: object,
    checkpoint: bool,
) -> None:
    path = component_path(tmp_path)
    ref = asset(path)
    identity = "native:dinkster.lumina2:" + "a" * 64
    seen: dict[str, object] = {}
    planned = patch_planning(
        monkeypatch,
        expected_identity=identity,
        role=role,
        checkpoint=checkpoint,
        seen=seen,
    )
    module = torch.nn.Identity()

    def load(candidate: object, load_builder: object, **kwargs: object) -> torch.nn.Module:
        pinned = cast("Any", seen["planner_source"])
        seen.update(load_plan=candidate, load_builder=load_builder, load_kwargs=kwargs)
        assert kwargs["source_file"] is seen["header_handle"]
        assert pinned.file is kwargs["source_file"]
        assert pinned.asset_digest == ref.digest
        assert pinned.asset_size == ref.size
        return module

    monkeypatch.setattr(component, "_load_component", load)
    loaded = component.load_lumina2_component(
        path,
        asset=ref,
        expected_role=cast("Any", role),
        expected_identity=identity,
        compute_dtype=torch.bfloat16,
    )
    assert loaded.role == role
    assert loaded.module is module
    assert loaded.plan is planned
    assert loaded.runtime_identity == identity
    assert seen["planner"] == "artifact"
    assert seen["identity_plan"] is planned
    assert seen["identity_role"] == role
    assert seen["identity_dtype"] is BFLOAT16
    assert seen["load_plan"] is planned
    assert seen["load_builder"] is builder
    kwargs = cast("dict[str, object]", seen["load_kwargs"])
    assert kwargs["compute_dtype"] is torch.bfloat16
    assert kwargs["fp8_matmul"] is False
    if role == "gemma2_2b":
        assert module.__dict__[component.LUMINA2_TOKENIZER_ATTRIBUTE] == b"tokenizer"


def test_identity_mismatch_refuses_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = component_path(tmp_path)
    seen: dict[str, object] = {}
    patch_planning(
        monkeypatch,
        expected_identity="native:dinkster.lumina2:" + "b" * 64,
        role="diffusion",
        checkpoint=False,
        seen=seen,
    )

    def fail_load(*_args: object, **_kwargs: object) -> None:
        pytest.fail("payload loading must not run")

    monkeypatch.setattr(component, "_load_component", fail_load)
    with pytest.raises(Lumina2ComponentAssemblyError, match="expected component identity"):
        component.load_lumina2_component(
            path,
            asset=asset(path),
            expected_role="diffusion",
            expected_identity="native:dinkster.lumina2:" + "c" * 64,
            compute_dtype=torch.bfloat16,
        )


def test_component_loader_refuses_size_unavailable_and_invalid_arguments(
    tmp_path: Path,
) -> None:
    path = component_path(tmp_path)
    ref = asset(path)
    identity = "native:dinkster.lumina2:" + "d" * 64
    mismatched = AssetRef(ref.digest, ref.name, ref.size + 1, resolver=_FixedResolver(path))
    with pytest.raises(Lumina2ComponentAssemblyError, match="byte size differs"):
        component.load_lumina2_component(
            path,
            asset=mismatched,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    unavailable = AssetRef(ref.digest, ref.name, ref.size, resolver=_MissingResolver())
    with pytest.raises(Lumina2ComponentAssemblyError, match="artifact is unavailable"):
        component.load_lumina2_component(
            path,
            asset=unavailable,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(TypeError, match="must be an AssetRef"):
        component.load_lumina2_component(
            path,
            asset=cast("Any", object()),
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="requires an expected identity"):
        component.load_lumina2_component(
            path,
            asset=ref,
            expected_role="diffusion",
            expected_identity="",
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(TypeError, match="compute dtype must be"):
        component.load_lumina2_component(
            path,
            asset=ref,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.float64,
        )
