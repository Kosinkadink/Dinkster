from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import dinkster_inference_torch
import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import BFLOAT16, FLOAT8_E4M3, FLOAT32, QWEN_IMAGE
from dinkster_inference.qwen_image_layout import qwen_image_dit_layout
from dinkster_inference.qwen_image_text import qwen_image_text_layout
from dinkster_inference.weights import TensorGeometry, WeightEntry
from dinkster_inference_torch import qwen_image_assembly as assembly
from dinkster_inference_torch.assemble import AssembleError
from dinkster_inference_torch.operations import INITLESS
from dinkster_inference_torch.qwen_image_assembly import (
    QwenImageSplitArtifactReceipt,
    QwenImageSplitAssemblyError,
    assemble_qwen_image_split,
    plan_qwen_image_split_assembly,
)
from dinkster_inference_torch.qwen_image_text import QwenImageTextModel
from dinkster_inference_torch.wan21_vae import WanVAE


class HeaderSource:
    def __init__(
        self,
        path: Path,
        geometries: Mapping[str, TensorGeometry],
        *,
        keys: Sequence[str] | None = None,
    ) -> None:
        self.path = path
        self.geometries = dict(geometries)
        self._keys = tuple(self.geometries if keys is None else keys)

    def keys(self) -> Sequence[str]:
        return self._keys

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


class FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _module_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def _linear_weight_keys() -> tuple[str, ...]:
    with torch.device("meta"):
        model = QwenImageTextModel()
    return tuple(
        sorted(
            f"{name}.weight"
            for name, child in model.named_modules()
            if isinstance(child, torch.nn.Linear)
        )
    )


def _sources(tmp_path: Path) -> tuple[HeaderSource, HeaderSource, HeaderSource]:
    diffusion = HeaderSource(
        tmp_path / "dit.safetensors",
        {
            key: TensorGeometry(shape, BFLOAT16)
            for key, shape in qwen_image_dit_layout().keys.items()
        },
    )
    fp8_weights = frozenset(_linear_weight_keys()[:358])
    text_geometries: dict[str, TensorGeometry] = {}
    for key, shape in qwen_image_text_layout().items():
        dtype = FLOAT8_E4M3 if key in fp8_weights else BFLOAT16
        text_geometries[key] = TensorGeometry(shape, dtype)
        if dtype is FLOAT8_E4M3:
            stem = key.removesuffix(".weight")
            text_geometries[f"{stem}.scale_input"] = TensorGeometry((), FLOAT32)
            text_geometries[f"{stem}.scale_weight"] = TensorGeometry((), FLOAT32)
    text_geometries["lm_head.weight"] = TensorGeometry((152064, 3584), BFLOAT16)
    text_geometries["scaled_fp8"] = TensorGeometry((0,), FLOAT8_E4M3)
    text = HeaderSource(tmp_path / "text.safetensors", text_geometries)
    with torch.device("meta"):
        vae_shapes = _module_shapes(WanVAE())
    vae = HeaderSource(
        tmp_path / "vae.safetensors",
        {key: TensorGeometry(shape, BFLOAT16) for key, shape in vae_shapes.items()},
    )
    return diffusion, text, vae


def _receipt(
    diffusion: HeaderSource, text: HeaderSource, vae: HeaderSource
) -> QwenImageSplitArtifactReceipt:
    return QwenImageSplitArtifactReceipt(
        provider_revision="46839d338df81ce625d5fae27d7e370314c0fbc9",
        paths={
            "qwen-image-dit": diffusion.path,
            "qwen2.5-vl-7b-text": text.path,
            "wan21-vae": vae.path,
        },
        header_sha256={
            "qwen-image-dit": "9356eb06d3b193fa894c2921ad8f61b19bb87823d54b0f57e22bdd76bc3a5b9f",
            "qwen2.5-vl-7b-text": (
                "c4e6e0abbd46c2216857d21eaef3e85ed56553e9ddd3103dbbeff243b8a38d73"
            ),
            "wan21-vae": "5fcff35e07ec3899a69d23ddec25bc5ec29c092d2902a66c0135bf36f3ec7cc5",
        },
    )


def _plan(tmp_path: Path) -> assembly.QwenImageSplitAssemblyPlan:
    diffusion, text, vae = _sources(tmp_path)
    return plan_qwen_image_split_assembly(
        diffusion=diffusion,
        text=text,
        vae=vae,
        receipt=_receipt(diffusion, text, vae),
    )


def _patch_current_headers(
    monkeypatch: pytest.MonkeyPatch,
    sources: tuple[HeaderSource, HeaderSource, HeaderSource],
    events: list[str],
) -> None:
    by_path = {source.path: source for source in sources}

    def load(path: Path) -> HeaderSource:
        events.append(f"header:{path.name}")
        return by_path[path]

    monkeypatch.setattr(assembly, "load_safetensors_header", load)


def test_plan_claims_exact_official_split_layout_and_quantization(tmp_path: Path) -> None:
    diffusion, text, vae = _sources(tmp_path)
    original_keys = (tuple(diffusion.keys()), tuple(text.keys()), tuple(vae.keys()))
    plan = plan_qwen_image_split_assembly(
        diffusion=diffusion,
        text=text,
        vae=vae,
        receipt=_receipt(diffusion, text, vae),
    )

    assert len(plan.diffusion.keys) == 1933
    assert len(plan.text.keys) == 728
    assert len(plan.text.quant) == 358
    assert len(plan.vae.keys) == 194
    assert plan.text.ignored == ("lm_head.weight", "scaled_fp8")
    assert set(plan.claims) == {
        "qwen-image-dit",
        "qwen2.5-vl-7b-text",
        "wan21-vae",
    }
    assert plan.claims["qwen-image-dit"] == tuple(sorted(diffusion.keys()))
    assert plan.claims["qwen2.5-vl-7b-text"] == tuple(sorted(text.keys()))
    assert plan.claims["wan21-vae"] == tuple(sorted(vae.keys()))
    assert all(item.format == "float8_e4m3fn" for item in plan.text.quant.values())
    assert all(
        item.input_scale == f"{layer}.scale_input" and item.weight_scale == f"{layer}.scale_weight"
        for layer, item in plan.text.quant.items()
    )
    assert (
        assembly._canonical_header_digest(  # pyright: ignore[reportPrivateUsage]
            diffusion
        )
        == plan.receipt.header_sha256["qwen-image-dit"]
    )
    assert (
        assembly._canonical_header_digest(  # pyright: ignore[reportPrivateUsage]
            text
        )
        == plan.receipt.header_sha256["qwen2.5-vl-7b-text"]
    )
    assert (
        assembly._canonical_header_digest(  # pyright: ignore[reportPrivateUsage]
            vae
        )
        == plan.receipt.header_sha256["wan21-vae"]
    )
    assert original_keys == (tuple(diffusion.keys()), tuple(text.keys()), tuple(vae.keys()))
    with pytest.raises(TypeError):
        plan.claims["foreign"] = ()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        plan.receipt = _receipt(diffusion, text, vae)  # type: ignore[misc]


def test_plan_constructor_refuses_incompatible_component_replacement(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    text_dtypes = dict(plan.text.dtypes)
    fp8_key = next(key for key, dtype in text_dtypes.items() if dtype is FLOAT8_E4M3)
    text_dtypes[fp8_key] = BFLOAT16
    with pytest.raises(QwenImageSplitAssemblyError, match="ComponentPlan"):
        replace(plan, text=replace(plan.text, dtypes=text_dtypes))
    with pytest.raises(QwenImageSplitAssemblyError, match="ComponentPlan"):
        replace(plan, vae=replace(plan.vae, ignored=("foreign",)))


@pytest.mark.parametrize("role", ("diffusion", "text", "vae"))
def test_plan_refuses_missing_foreign_duplicate_and_wrong_geometry(
    tmp_path: Path, role: str
) -> None:
    sources = list(_sources(tmp_path))
    index = {"diffusion": 0, "text": 1, "vae": 2}[role]
    source = sources[index]
    first = next(iter(source.geometries))
    missing = HeaderSource(source.path, source.geometries, keys=tuple(source.keys())[1:])
    sources[index] = missing
    with pytest.raises(QwenImageSplitAssemblyError, match="missing"):
        plan_qwen_image_split_assembly(
            diffusion=sources[0],
            text=sources[1],
            vae=sources[2],
            receipt=_receipt(sources[0], sources[1], sources[2]),
        )

    sources = list(_sources(tmp_path))
    source = sources[index]
    foreign_geometries = dict(source.geometries)
    foreign_geometries["foreign.weight"] = TensorGeometry((), BFLOAT16)
    sources[index] = HeaderSource(source.path, foreign_geometries)
    with pytest.raises(QwenImageSplitAssemblyError, match="foreign"):
        plan_qwen_image_split_assembly(
            diffusion=sources[0],
            text=sources[1],
            vae=sources[2],
            receipt=_receipt(sources[0], sources[1], sources[2]),
        )

    sources = list(_sources(tmp_path))
    source = sources[index]
    sources[index] = HeaderSource(source.path, source.geometries, keys=(*source.keys(), first))
    with pytest.raises(QwenImageSplitAssemblyError, match="duplicate"):
        plan_qwen_image_split_assembly(
            diffusion=sources[0],
            text=sources[1],
            vae=sources[2],
            receipt=_receipt(sources[0], sources[1], sources[2]),
        )

    sources = list(_sources(tmp_path))
    source = sources[index]
    wrong = dict(source.geometries)
    wrong[first] = TensorGeometry((*wrong[first].shape, 1), wrong[first].dtype)
    sources[index] = HeaderSource(source.path, wrong)
    with pytest.raises(QwenImageSplitAssemblyError, match="shape|geometry"):
        plan_qwen_image_split_assembly(
            diffusion=sources[0],
            text=sources[1],
            vae=sources[2],
            receipt=_receipt(sources[0], sources[1], sources[2]),
        )


def test_plan_refuses_mixed_prefix_dtype_quant_and_receipt_mismatches(tmp_path: Path) -> None:
    diffusion, text, vae = _sources(tmp_path)
    dit_geometries = dict(diffusion.geometries)
    key = next(iter(dit_geometries))
    dit_geometries["model.diffusion_model." + key] = dit_geometries.pop(key)
    mixed = HeaderSource(diffusion.path, dit_geometries)
    with pytest.raises(QwenImageSplitAssemblyError, match="prefix|missing"):
        plan_qwen_image_split_assembly(
            diffusion=mixed, text=text, vae=vae, receipt=_receipt(mixed, text, vae)
        )

    text_geometries = dict(text.geometries)
    scale = next(key for key in text_geometries if key.endswith(".scale_weight"))
    text_geometries[scale] = TensorGeometry((1,), FLOAT32)
    bad_scale = HeaderSource(text.path, text_geometries)
    with pytest.raises(QwenImageSplitAssemblyError, match="scale.*scalar float32"):
        plan_qwen_image_split_assembly(
            diffusion=diffusion,
            text=bad_scale,
            vae=vae,
            receipt=_receipt(diffusion, bad_scale, vae),
        )

    text_geometries = dict(text.geometries)
    del text_geometries[scale]
    missing_scale = HeaderSource(text.path, text_geometries)
    with pytest.raises(QwenImageSplitAssemblyError, match="scale"):
        plan_qwen_image_split_assembly(
            diffusion=diffusion,
            text=missing_scale,
            vae=vae,
            receipt=_receipt(diffusion, missing_scale, vae),
        )

    text_geometries = dict(text.geometries)
    fp8_weight = next(key for key, value in text_geometries.items() if value.dtype is FLOAT8_E4M3)
    text_geometries[fp8_weight] = TensorGeometry(text_geometries[fp8_weight].shape, BFLOAT16)
    wrong_dtype = HeaderSource(text.path, text_geometries)
    with pytest.raises(QwenImageSplitAssemblyError, match="358|dtype"):
        plan_qwen_image_split_assembly(
            diffusion=diffusion,
            text=wrong_dtype,
            vae=vae,
            receipt=_receipt(diffusion, wrong_dtype, vae),
        )

    receipt = _receipt(diffusion, text, vae)
    with pytest.raises(QwenImageSplitAssemblyError, match="receipt path"):
        plan_qwen_image_split_assembly(
            diffusion=diffusion,
            text=text,
            vae=vae,
            receipt=replace(
                receipt,
                paths={**receipt.paths, "wan21-vae": tmp_path / "other.safetensors"},
            ),
        )
    with pytest.raises(ValueError, match="header digest"):
        replace(
            receipt,
            header_sha256={**receipt.header_sha256, "wan21-vae": "0" * 64},
        )

    text_geometries = dict(text.geometries)
    fp8_weight = next(
        key
        for key, value in text_geometries.items()
        if key.endswith(".weight") and value.dtype is FLOAT8_E4M3
    )
    bf16_linear = next(
        key for key in _linear_weight_keys() if text_geometries[key].dtype is BFLOAT16
    )
    text_geometries[fp8_weight] = TensorGeometry(text_geometries[fp8_weight].shape, BFLOAT16)
    text_geometries[bf16_linear] = TensorGeometry(text_geometries[bf16_linear].shape, FLOAT8_E4M3)
    for suffix in ("scale_input", "scale_weight"):
        old_scale = f"{fp8_weight.removesuffix('.weight')}.{suffix}"
        new_scale = f"{bf16_linear.removesuffix('.weight')}.{suffix}"
        text_geometries[new_scale] = text_geometries.pop(old_scale)
    structurally_valid_foreign_header = HeaderSource(text.path, text_geometries)
    with pytest.raises(QwenImageSplitAssemblyError, match="header digest"):
        plan_qwen_image_split_assembly(
            diffusion=diffusion,
            text=structurally_valid_foreign_header,
            vae=vae,
            receipt=receipt,
        )


def test_executor_reuses_strict_component_loader_and_propagates_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path)
    plan = plan_qwen_image_split_assembly(
        diffusion=sources[0],
        text=sources[1],
        vae=sources[2],
        receipt=_receipt(*sources),
    )
    calls: list[tuple[str, torch.dtype, bool]] = []
    events: list[str] = []
    _patch_current_headers(monkeypatch, sources, events)

    class Marker(torch.nn.Module):
        pass

    def load(
        component: object,
        _build: Callable[..., torch.nn.Module],
        *,
        compute_dtype: torch.dtype,
        fp8_matmul: bool,
    ) -> torch.nn.Module:
        name = cast(assembly.ComponentPlan[object], component).component
        events.append(f"payload:{name}")
        calls.append((name, compute_dtype, fp8_matmul))
        if name == "text":
            with torch.device("meta"):
                assert isinstance(
                    _build(
                        cast(assembly.ComponentPlan[object], component).config,
                        operations=INITLESS,
                    ),
                    QwenImageTextModel,
                )
        return Marker()

    monkeypatch.setattr(assembly, "_load_component", load)
    assembled = assemble_qwen_image_split(plan, fp8_matmul=True)
    assert calls == [
        ("diffusion", torch.bfloat16, True),
        ("text", torch.bfloat16, True),
        ("vae", torch.bfloat16, True),
    ]
    assert events[:3] == [
        "header:dit.safetensors",
        "header:text.safetensors",
        "header:vae.safetensors",
    ]
    assert events[3:] == ["payload:diffusion", "payload:text", "payload:vae"]
    assert isinstance(assembled.diffusion, Marker)
    assert isinstance(assembled.text, Marker)
    assert isinstance(assembled.vae, Marker)
    assert assembled.family is QWEN_IMAGE
    assert assembled._component_compute_dtypes == {  # pyright: ignore[reportPrivateUsage]
        "diffusion": torch.bfloat16,
        "text": torch.bfloat16,
        "vae": torch.bfloat16,
    }
    assert not assembled._storage_dtype_follows_compute  # pyright: ignore[reportPrivateUsage]

    calls.clear()
    events.clear()
    assembled = assemble_qwen_image_split(plan, vae_dtype=torch.float16)
    assert calls[-1] == ("vae", torch.float16, False)
    assert assembled._component_compute_dtypes["vae"] is torch.float16  # pyright: ignore[reportPrivateUsage]

    calls.clear()
    events.clear()

    def fail_second(
        component: object,
        _build: object,
        *,
        compute_dtype: torch.dtype,
        fp8_matmul: bool,
    ) -> torch.nn.Module:
        del compute_dtype, fp8_matmul
        name = cast(assembly.ComponentPlan[object], component).component
        calls.append((name, torch.bfloat16, False))
        if name == "text":
            raise AssembleError("text failed")
        return Marker()

    monkeypatch.setattr(assembly, "_load_component", fail_second)
    with pytest.raises(AssembleError, match="text failed"):
        assemble_qwen_image_split(plan)
    assert [name for name, _, _ in calls] == ["diffusion", "text"]


def test_executor_revalidates_every_current_header_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path)
    plan = plan_qwen_image_split_assembly(
        diffusion=sources[0],
        text=sources[1],
        vae=sources[2],
        receipt=_receipt(*sources),
    )
    bad_vae_geometries = dict(sources[2].geometries)
    first = next(iter(bad_vae_geometries))
    geometry = bad_vae_geometries[first]
    bad_vae_geometries[first] = TensorGeometry((*geometry.shape, 1), geometry.dtype)
    current_sources = (
        sources[0],
        sources[1],
        HeaderSource(sources[2].path, bad_vae_geometries),
    )
    events: list[str] = []
    _patch_current_headers(monkeypatch, current_sources, events)

    def payload_load(
        _component: object,
        _build: object,
        *,
        compute_dtype: torch.dtype,
        fp8_matmul: bool,
    ) -> None:
        del compute_dtype, fp8_matmul
        events.append("payload")

    monkeypatch.setattr(assembly, "_load_component", payload_load)

    with pytest.raises(QwenImageSplitAssemblyError, match="header digest"):
        assemble_qwen_image_split(plan)
    assert events == [
        "header:dit.safetensors",
        "header:text.safetensors",
        "header:vae.safetensors",
    ]


def test_assembly_is_direct_import_only_and_refuses_unsupported_compute_dtype(
    tmp_path: Path,
) -> None:
    assert not hasattr(dinkster_inference_torch, "assemble_qwen_image_split")
    with pytest.raises(TypeError, match="compute dtype"):
        assemble_qwen_image_split(_plan(tmp_path), text_dtype=torch.int32)
    with pytest.raises(TypeError, match="VAE compute dtype"):
        assemble_qwen_image_split(_plan(tmp_path), vae_dtype=torch.int32)


def test_component_loader_preserves_dispatch_identity_and_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    asset = AssetRef(
        digest_file(path), path.name, path.stat().st_size, resolver=FixedResolver(path)
    )
    expected_identity = "native:qwen-image:" + "a" * 64
    source = SimpleNamespace(path=path)
    plan = SimpleNamespace(component="diffusion", config=object())
    module = torch.nn.Identity()
    seen: dict[str, object] = {}

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path)
        return source

    def plan_component(candidate: object, *, role: str, path: Path) -> object:
        seen.update(planner_source=candidate, planner_role=role, planner_path=path)
        return plan

    def component_identity(candidate: object, role: str, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_role=role, identity_dtype=dtype)
        return expected_identity

    monkeypatch.setattr(
        assembly,
        "load_safetensors_header_from_file",
        load_header,
    )
    monkeypatch.setattr(
        assembly,
        "plan_qwen_image_official_component",
        plan_component,
    )
    monkeypatch.setattr(
        assembly,
        "qwen_image_component_runtime_identity",
        component_identity,
    )

    def load(
        candidate: object,
        _builder: object,
        **kwargs: object,
    ) -> torch.nn.Module:
        pinned = cast(SimpleNamespace, kwargs["source"])
        seen.update(load_plan=candidate, load_kwargs=kwargs)
        assert kwargs["source_file"] is seen["header_handle"]
        assert pinned.file is kwargs["source_file"]
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        return module

    monkeypatch.setattr(assembly, "_load_component", load)

    loaded = assembly.load_qwen_image_component(
        path,
        asset=asset,
        expected_role="diffusion",
        expected_identity=expected_identity,
        compute_dtype=torch.bfloat16,
    )

    assert loaded.role == "diffusion"
    assert loaded.module is module
    assert loaded.plan is plan
    assert loaded.runtime_identity == expected_identity
    assert seen["planner_role"] == seen["identity_role"] == "diffusion"
    assert seen["identity_dtype"] is BFLOAT16


def test_pinned_source_serves_real_tensor_reads(tmp_path: Path) -> None:
    """The pinned wrapper must satisfy every attribute the tensor reader
    uses on a SafetensorsSource (path, keys, entries, entry)."""
    from dinkster_inference.sources import load_safetensors_header_from_file
    from dinkster_inference_torch.sources import load_tensors_from_file
    from test_assemble import write_checkpoint

    tensors = {
        "alpha": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "beta": torch.ones(4, dtype=torch.bfloat16),
    }
    path = write_checkpoint(tmp_path / "pinned.safetensors", tensors)
    with path.open("rb") as handle:
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = assembly._DescriptorPinnedSource(  # pyright: ignore[reportPrivateUsage]
            source, handle, "blake3:" + "0" * 64, path.stat().st_size
        )
        loaded = load_tensors_from_file(handle, cast("Any", pinned), ("alpha", "beta"))
    assert torch.equal(loaded["alpha"], tensors["alpha"])
    assert torch.equal(loaded["beta"], tensors["beta"])


def test_component_loader_rejects_wrong_structure_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    asset = AssetRef(
        digest_file(path), path.name, path.stat().st_size, resolver=FixedResolver(path)
    )

    def load_header(_handle: BinaryIO, *, path: Path) -> object:
        return SimpleNamespace(path=path)

    def reject_structure(_source: object, *, role: str, path: Path) -> object:
        del role, path
        raise QwenImageSplitAssemblyError("geometry mismatch")

    def fail_tensor_load(*_args: object, **_kwargs: object) -> None:
        pytest.fail("tensor loading must not run")

    monkeypatch.setattr(
        assembly,
        "load_safetensors_header_from_file",
        load_header,
    )
    monkeypatch.setattr(
        assembly,
        "plan_qwen_image_official_component",
        reject_structure,
    )
    monkeypatch.setattr(
        assembly,
        "_load_component",
        fail_tensor_load,
    )

    with pytest.raises(QwenImageSplitAssemblyError, match="geometry mismatch"):
        assembly.load_qwen_image_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity="native:qwen-image:" + "a" * 64,
            compute_dtype=torch.bfloat16,
        )
