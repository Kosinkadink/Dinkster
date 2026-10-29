from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    BFLOAT16,
    FLUX2_DEV,
    KLEIN_QWEN3_8B_CONFIG,
    MISTRAL3_24B_PRUNED_CONFIG,
    Conditioning,
    Flux2ComponentAssemblyError,
    QwenTextConfig,
    TekkenBpe,
    load_flux2_tekken_bpe,
    tokenize_flux2_dev_prompt,
)
from dinkster_inference.tekken_bpe import TEKKEN_FLUX2_SHA256, TEKKEN_FLUX2_VENDORED
from dinkster_inference.vendored import read_vendored
from dinkster_inference_torch import (
    FLUX2_TEKKEN_ATTRIBUTE,
    INITLESS,
    Flux2DevTextEncoder,
    Flux2DiffusionRuntime,
    Flux2KleinTextEncoder,
    Flux2RuntimeError,
    Flux2TextRuntime,
    ModuleStateStore,
)
from dinkster_inference_torch import flux2_assembly as assembly
from dinkster_inference_torch import sampling_execution as sampling_execution_mod
from dinkster_inference_torch.conditioning_adapters import basic_conditioning_to_carrier


class FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


class MissingResolver:
    def resolve(self, digest: str) -> None:
        del digest
        return None


def _asset(path: Path) -> AssetRef:
    return AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=FixedResolver(path))


def _component_path(tmp_path: Path) -> Path:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    return path


def _patch_planner_seam(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_identity: str,
    seen: dict[str, object],
    source_keys: tuple[str, ...] = (),
    read_uint8: object | None = None,
) -> SimpleNamespace:
    """Stub header/plan/identity so loader tests exercise only the seam."""

    source = SimpleNamespace(
        keys=lambda: source_keys,
        read_uint8_configuration_from_file=read_uint8,
    )
    planned = SimpleNamespace(plan=SimpleNamespace(), family_id="dinkster.flux2_dev")

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path)
        return source

    def plan_component(candidate: object, *, role: str, path: Path) -> object:
        seen.update(planner_source=candidate, planner_role=role, planner_path=path)
        return planned

    def component_identity(candidate: object, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_dtype=dtype)
        return expected_identity

    monkeypatch.setattr(assembly, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(assembly, "plan_flux2_split_component", plan_component)
    monkeypatch.setattr(assembly, "flux2_component_runtime_identity", component_identity)
    return planned


@pytest.mark.parametrize(
    ("role", "builder"),
    [
        ("diffusion", assembly.Flux),
        ("mistral3_24b", assembly.QwenTextModel),
        ("qwen3_8b", assembly.QwenTextModel),
        ("qwen3_4b", assembly.QwenTextModel),
        ("vae", assembly.AutoencoderKL),
    ],
)
def test_component_loader_preserves_dispatch_identity_and_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, builder: object
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    expected_identity = "native:dinkster.flux2_dev:" + "a" * 64
    seen: dict[str, object] = {}
    planned = _patch_planner_seam(monkeypatch, expected_identity=expected_identity, seen=seen)
    module = torch.nn.Identity()

    def load(candidate: object, load_builder: object, **kwargs: object) -> torch.nn.Module:
        pinned = cast(SimpleNamespace, kwargs["source"])
        seen.update(load_plan=candidate, load_builder=load_builder, load_kwargs=kwargs)
        assert kwargs["source_file"] is seen["header_handle"]
        assert pinned.file is kwargs["source_file"]
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        return module

    monkeypatch.setattr(assembly, "_load_component", load)

    loaded = assembly.load_flux2_component(
        path,
        asset=asset,
        expected_role=cast("Any", role),
        expected_identity=expected_identity,
        compute_dtype=torch.bfloat16,
    )

    assert loaded.role == role
    assert loaded.family_id == "dinkster.flux2_dev"
    assert loaded.module is module
    assert loaded.plan is planned.plan
    assert loaded.runtime_identity == expected_identity
    assert seen["header_path"] == path
    assert seen["planner_role"] == role
    assert seen["planner_path"] == path
    assert cast(SimpleNamespace, seen["planner_source"]).file is seen["header_handle"]
    assert seen["identity_plan"] is planned
    assert seen["identity_dtype"] is BFLOAT16
    assert seen["load_plan"] is planned.plan
    assert seen["load_builder"] is builder
    kwargs = cast("dict[str, object]", seen["load_kwargs"])
    assert kwargs["compute_dtype"] is torch.bfloat16
    assert kwargs["fp8_matmul"] is False


def test_component_loader_declares_routed_materialization_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    expected_identity = "native:dinkster.flux2_klein_4b:" + "a" * 64
    seen: dict[str, object] = {}
    _patch_planner_seam(monkeypatch, expected_identity=expected_identity, seen=seen)
    module = INITLESS.linear(2, 2, bias=False)
    module.load_state_dict(
        {"weight": torch.ones(2, 2, dtype=torch.bfloat16)},
        strict=True,
        assign=True,
    )

    def load_component(*_args: object, **_kwargs: object) -> torch.nn.Module:
        return module

    monkeypatch.setattr(assembly, "_load_component", load_component)

    loaded = assembly.load_flux2_component(
        path,
        asset=asset,
        expected_role="qwen3_4b",
        expected_identity=expected_identity,
        compute_dtype=torch.bfloat16,
    )

    store = ModuleStateStore(loaded.module)
    assert store.max_materialized_itemsize("weight") == torch.bfloat16.itemsize


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
        pinned = assembly._PinnedSource(  # pyright: ignore[reportPrivateUsage]
            source, handle, "blake3:" + "0" * 64, path.stat().st_size
        )
        loaded = load_tensors_from_file(handle, cast("Any", pinned), ("alpha", "beta"))
    assert torch.equal(loaded["alpha"], tensors["alpha"])
    assert torch.equal(loaded["beta"], tensors["beta"])


@pytest.mark.parametrize("embedded", [True, False])
def test_component_loader_attaches_dev_tekken_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedded: bool
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    expected_identity = "native:dinkster.flux2_dev:" + "b" * 64
    seen: dict[str, object] = {}
    reads: list[tuple[object, str, int]] = []
    tekken_bytes = read_vendored(TEKKEN_FLUX2_VENDORED, TEKKEN_FLUX2_SHA256)

    def read_uint8(handle: object, key: str, *, limit: int) -> bytes:
        reads.append((handle, key, limit))
        return tekken_bytes

    _patch_planner_seam(
        monkeypatch,
        expected_identity=expected_identity,
        seen=seen,
        source_keys=("tekken_model",) if embedded else (),
        read_uint8=read_uint8,
    )

    def load(*_args: object, **_kwargs: object) -> torch.nn.Module:
        return torch.nn.Identity()

    monkeypatch.setattr(assembly, "_load_component", load)

    loaded = assembly.load_flux2_component(
        path,
        asset=asset,
        expected_role="mistral3_24b",
        expected_identity=expected_identity,
        compute_dtype=torch.bfloat16,
    )

    tokenizer = loaded.module.__dict__[FLUX2_TEKKEN_ATTRIBUTE]
    assert isinstance(tokenizer, TekkenBpe)
    if embedded:
        # The published checkpoint embeds a ~19 MB tokenizer model, so
        # the read must carry the tokenizer byte cap, not the default
        # 65536-byte configuration cap.
        assert reads == [(seen["header_handle"], "tekken_model", assembly.TEKKEN_MODEL_BYTE_CAP)]
        assert assembly.TEKKEN_MODEL_BYTE_CAP > len(tekken_bytes)
    else:
        assert reads == []
        assert tokenizer is load_flux2_tekken_bpe()


def test_component_loader_refuses_identity_mismatch_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    seen: dict[str, object] = {}
    _patch_planner_seam(
        monkeypatch, expected_identity="native:dinkster.flux2_dev:" + "c" * 64, seen=seen
    )

    def fail_load(*_args: object, **_kwargs: object) -> torch.nn.Module:
        pytest.fail("payload loading must not run on identity mismatch")

    monkeypatch.setattr(assembly, "_load_component", fail_load)

    with pytest.raises(Flux2ComponentAssemblyError, match="expected component identity"):
        assembly.load_flux2_component(
            path,
            asset=_asset(path),
            expected_role="diffusion",
            expected_identity="native:dinkster.flux2_dev:" + "d" * 64,
            compute_dtype=torch.bfloat16,
        )


def test_component_loader_refuses_byte_size_disagreeing_with_asset_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    asset = AssetRef(
        digest_file(path), path.name, path.stat().st_size + 1, resolver=FixedResolver(path)
    )

    def fail_header(*_args: object, **_kwargs: object) -> object:
        pytest.fail("header planning must not run on size mismatch")

    monkeypatch.setattr(assembly, "load_safetensors_header_from_file", fail_header)

    with pytest.raises(Flux2ComponentAssemblyError, match="byte size differs"):
        assembly.load_flux2_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity="native:dinkster.flux2_dev:" + "e" * 64,
            compute_dtype=torch.bfloat16,
        )


def test_component_loader_reports_unavailable_artifact(tmp_path: Path) -> None:
    path = _component_path(tmp_path)
    asset = AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=MissingResolver())
    with pytest.raises(Flux2ComponentAssemblyError, match="artifact is unavailable"):
        assembly.load_flux2_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity="native:dinkster.flux2_dev:" + "f" * 64,
            compute_dtype=torch.bfloat16,
        )


def test_component_loader_validates_arguments_before_touching_the_artifact(
    tmp_path: Path,
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    identity = "native:dinkster.flux2_dev:" + "0" * 64
    with pytest.raises(TypeError, match="must be an AssetRef"):
        assembly.load_flux2_component(
            path,
            asset=cast("Any", SimpleNamespace(digest="x", size=1)),
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="requires an expected identity"):
        assembly.load_flux2_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity="",
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(TypeError, match="compute dtype must be"):
        assembly.load_flux2_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.float64,
        )


class _StackTower:
    """Fake text tower returning a (batch, captures, tokens, hidden) stack."""

    hidden = 2

    def __init__(self, config: QwenTextConfig) -> None:
        self.config = config
        self.embed_tokens = SimpleNamespace(weight=torch.empty(0))
        self.ids: torch.Tensor | None = None

    def __call__(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.ids = ids
        captures = len(self.config.output_hidden_layers or ())
        count = captures * ids.shape[1] * self.hidden
        return torch.arange(count, dtype=torch.float32).reshape(
            1, captures, ids.shape[1], self.hidden
        )


def test_flux2_text_runtime_encodes_dev_prompts_with_the_component_tokenizer() -> None:
    tower = _StackTower(MISTRAL3_24B_PRUNED_CONFIG)
    tower.__dict__[FLUX2_TEKKEN_ATTRIBUTE] = load_flux2_tekken_bpe()
    runtime = Flux2TextRuntime(cast("Any", tower))
    encoder = runtime._encoder  # pyright: ignore[reportPrivateUsage]
    assert isinstance(encoder, Flux2DevTextEncoder)
    assert runtime.text is cast("Any", tower)
    conditioning = runtime.encode_text("Hello, world!")
    expected = tokenize_flux2_dev_prompt("Hello, world!", tokenizer=load_flux2_tekken_bpe())
    assert tower.ids is not None
    assert tower.ids.tolist() == [list(expected.ids)]
    assert conditioning.pooled is None


def test_flux2_text_runtime_refuses_dev_component_without_tekken_tokenizer() -> None:
    tower = _StackTower(MISTRAL3_24B_PRUNED_CONFIG)
    with pytest.raises(Flux2RuntimeError, match="carries no tekken tokenizer"):
        Flux2TextRuntime(cast("Any", tower))


def test_flux2_text_runtime_selects_klein_encoder_and_refuses_foreign_profiles() -> None:
    tower = _StackTower(KLEIN_QWEN3_8B_CONFIG)
    runtime = Flux2TextRuntime(cast("Any", tower))
    encoder = runtime._encoder  # pyright: ignore[reportPrivateUsage]
    assert isinstance(encoder, Flux2KleinTextEncoder)
    foreign = SimpleNamespace(config=SimpleNamespace(architecture="ovis_qwen3_2b"))
    with pytest.raises(Flux2RuntimeError, match="not a Flux2 text profile"):
        Flux2TextRuntime(cast("Any", foreign))


class _ParamFlux(torch.nn.Module):
    def __init__(self, value: float = 0.0, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.value = value
        self.guidance_in = object()
        self.vector_in = None
        self.config = SimpleNamespace(
            vec_in_dim=None,
            context_in_dim=8,
            hidden_size=16,
            depth=2,
            depth_single_blocks=4,
            guidance_embed=True,
        )
        self.calls: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, object, torch.Tensor | None]
        ] = []

    def forward(
        self,
        xc: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: object,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:
        self.calls.append((xc, timesteps, context, y, guidance))
        return torch.full_like(xc, self.value)


def test_flux2_diffusion_runtime_exposes_identity_and_module_dtype() -> None:
    model = _ParamFlux(dtype=torch.bfloat16)
    runtime = Flux2DiffusionRuntime(cast("Any", model), FLUX2_DEV, runtime_identity="native:test")
    assert runtime.family is FLUX2_DEV
    assert runtime.runtime_identity == "native:test"
    assert runtime.conditioning_identity == (
        "dinkster.flux2.conditioning:v1:dinkster.flux2_dev:8:16:2:4:True"
    )
    assert runtime.assembled.compute_dtype("diffusion") is torch.bfloat16
    assert runtime.assembled.compute_dtype("vae") is None


def test_flux2_diffusion_runtime_samples_end_to_end_with_shift_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ParamFlux(value=0.0)
    runtime = Flux2DiffusionRuntime(cast("Any", model), FLUX2_DEV, runtime_identity="native:test")
    shifts: list[float] = []
    original = sampling_execution_mod.build_sampling_schedule

    def capture(*args: Any, **kwargs: Any) -> Any:
        shifts.append(args[1].shift)
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling_execution_mod, "build_sampling_schedule", capture)
    latent = torch.zeros((1, 128, 2, 2))
    cond = Conditioning(torch.zeros((1, 3, 8)), None)
    result = runtime.sample(
        latent,
        cond=cond,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        seed=4,
        sampling_shift=1.5,
    )
    assert result.shape == latent.shape
    assert model.calls
    assert model.calls[0][4] is not None
    assert model.calls[0][4].tolist() == [3.5]
    assert shifts == [1.5]


def test_flux2_diffusion_runtime_registers_module_dtype_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference import CustomSamplingResult

    model = _ParamFlux(dtype=torch.bfloat16)
    runtime = Flux2DiffusionRuntime(cast("Any", model), FLUX2_DEV, runtime_identity="native:test")
    seen: dict[str, object] = {}

    def fake_sample(
        self: object, latent: torch.Tensor, **kwargs: object
    ) -> CustomSamplingResult[torch.Tensor]:
        seen.update(kwargs)
        seen.update(runtime=self, latent=latent)
        return CustomSamplingResult(latent, None)

    monkeypatch.setattr(Flux2DiffusionRuntime, "sample_custom", fake_sample)
    latent = torch.zeros((1, 128, 2, 2))
    cond = Conditioning(torch.zeros((1, 3, 8)), None)
    runtime.sample(
        latent,
        cond=cond,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        sampling_shift=2.5,
    )
    assert seen["runtime"] is runtime
    assert seen["latent"] is latent
    assert "compute_dtype" not in seen
    assert "device" not in seen
    assert seen["sampling_shift"] == 2.5
    registration = runtime.sampling_execution_registration
    assert registration.compute_dtype(runtime) is torch.bfloat16
    assert registration.device(runtime) == next(model.parameters()).device


def test_flux2_diffusion_runtime_materializes_basic_conditioning_on_module_device() -> None:
    model = _ParamFlux()
    runtime = Flux2DiffusionRuntime(cast("Any", model), FLUX2_DEV, runtime_identity="native:test")
    embeddings = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    carrier = basic_conditioning_to_carrier(Conditioning(embeddings, None))
    out = runtime.prepare_single_stream_conditioning(carrier)
    assert torch.equal(out.embeddings, embeddings)
    assert out.embeddings.device == next(model.parameters()).device
    assert out.pooled is None
