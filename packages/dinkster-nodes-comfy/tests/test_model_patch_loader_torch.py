"""CPU behavior coverage for ModelPatchLoader checkpoint dispatch."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DINKSTER_ACCELERATOR"] = "cpu"

import dinkster_nodes_comfy.model_patch_loader as loader
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    MODEL_PATCH_TYPE,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_assets import AssetVault, MountDef, MountTable
from dinkster_assets.identity import digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.model_patch_loader import (
    ModelPatchLoader,
    _multi_talk_patch,
    _qwen_blockwise_controlnet,
    _siglip_multi_feat_projector,
    _z_image_convert,
)

from dinkster.compose import PackSpec, ServingComposer


class LocalResolver:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.revoked = False

    def resolve_local(self, binding: str) -> tuple[Path, tuple[int, int, int, int, int]] | None:
        if self.revoked:
            return None
        stat = self.path.stat()
        return self.path, (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_dev,
            stat.st_ino,
        )

    def reconcile_local(self, binding: str) -> None:
        return None


def test_z_image_conversion_preserves_qkv_order_for_asymmetric_weights() -> None:
    state = {
        "x.attention.to_k.weight": torch.tensor([[10.0], [20.0]]),
        "x.attention.to_q.weight": torch.tensor([[1.0], [2.0]]),
        "x.attention.to_v.weight": torch.tensor([[100.0], [200.0]]),
        "x.attention.to_out.0.weight": torch.tensor([[7.0]]),
    }
    converted = _z_image_convert(state, torch)
    assert torch.equal(
        converted["x.attention.qkv.weight"],
        torch.tensor([[1.0], [2.0], [10.0], [20.0], [100.0], [200.0]]),
    )
    assert torch.equal(converted["x.attention.out.weight"], torch.tensor([[7.0]]))


def test_loader_rejects_unbound_asset() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        ModelPatchLoader.execute(name=object())


def test_local_architecture_helpers_construct_without_production_weights() -> None:
    class SmallOperations:
        @staticmethod
        def Linear(in_features: int, out_features: int, **kwargs: Any) -> torch.nn.Linear:
            return torch.nn.Linear(in_features, out_features, **kwargs)

        @staticmethod
        def RMSNorm(dim: int, **kwargs: Any) -> torch.nn.RMSNorm:
            return torch.nn.RMSNorm(dim, **kwargs)

        @staticmethod
        def LayerNorm(dim: int, **kwargs: Any) -> torch.nn.LayerNorm:
            return torch.nn.LayerNorm(dim, **kwargs)

    qwen = _qwen_blockwise_controlnet(SmallOperations)(
        additional_in_dim=3, num_layers=2, in_dim=4, dim=4, device="cpu", dtype=torch.float32
    )
    assert qwen.img_in.in_features == 7
    assert len(qwen.controlnet_blocks) == 2
    siglip = _siglip_multi_feat_projector(SmallOperations)(device="meta", dtype=torch.float32)
    assert siglip.high_projection.in_features == 1152
    assert siglip.low_projection.out_features == 3072

    class MultiTalk:
        @staticmethod
        def MultiTalkAudioProjModel(**kwargs: Any) -> torch.nn.Linear:
            assert kwargs["seq_len"] == 5
            assert kwargs["context_tokens"] == 32
            return torch.nn.Linear(2, 3, device=kwargs["device"], dtype=kwargs.get("dtype"))

        @staticmethod
        def WanMultiTalkAttentionBlock(*args: Any, **kwargs: Any) -> torch.nn.Linear:
            assert args == (5, 7)
            return torch.nn.Linear(3, 2, device=kwargs["device"], dtype=kwargs.get("dtype"))

    multitalk = _multi_talk_patch(MultiTalk, SmallOperations)(
        in_dim=5, intermediate_dim=6, out_dim=7, device="cpu", dtype=torch.bfloat16
    )
    assert len(multitalk.blocks) == 40
    assert {parameter.dtype for parameter in multitalk.parameters()} == {torch.bfloat16}


@pytest.mark.parametrize("vault_backed", [False, True])
def test_loader_builds_duration_head_from_safe_asset(tmp_path: Path, vault_backed: bool) -> None:
    from dinkster_inference.ldm.lightricks.duration_head import DurationHead
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint = tmp_path / "duration.safetensors"
    original = DurationHead()
    state = {
        f"model.diffusion_model.duration_head.{key}": torch.full_like(value, 0.001)
        for key, value in original.state_dict().items()
    }
    state["model.diffusion_model.duration_head.mlp_out.weight"].zero_()
    state["model.diffusion_model.duration_head.mlp_out.bias"].fill_(0.5)
    save_file(state, str(checkpoint), metadata={"format": "pt", "source": "duration-test"})
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        assert handle.metadata() == {"format": "pt", "source": "duration-test"}
    resolver = LocalResolver(checkpoint)
    asset = LocalAssetRef(
        "duration",
        checkpoint.name,
        checkpoint.stat().st_size,
        checkpoint.name,
        resolver=resolver,
    )
    loaded_asset: AssetRef | LocalAssetRef = asset
    if vault_backed:
        vault = AssetVault(tmp_path / "vault")
        digest = digest_file(checkpoint)
        with vault.writer(digest) as writer:
            writer.write(checkpoint.read_bytes())
            writer.commit()
        loaded_asset = AssetRef(digest, checkpoint.name, checkpoint.stat().st_size, resolver=vault)
        assert not loaded_asset.local_path().suffix
    patcher = ModelPatchLoader.execute(name=loaded_asset)["MODEL_PATCH"]
    assert type(patcher).__name__ == "ModelPatcher"
    assert type(patcher.model).__name__ == "DurationHead"
    assert patcher.model.attention_pooler.query_tokens.dtype is torch.float32
    assert torch.equal(
        patcher.model.attention_pooler.query_tokens,
        state["model.diffusion_model.duration_head.attention_pooler.query_tokens"].float(),
    )
    reference = DurationHead()
    reference.load_state_dict(
        {
            key.removeprefix("model.diffusion_model.duration_head."): value.float()
            for key, value in state.items()
        }
    )
    video = torch.full((1, 1, 4096), 0.25)
    assert torch.equal(patcher.model(video_tokens=video), reference(video_tokens=video))
    assert torch.equal(patcher.model(video_tokens=video), torch.tensor([0.5]).exp())
    verified = AssetRef(
        digest_file(checkpoint),
        checkpoint.name,
        checkpoint.stat().st_size,
        resolver=SimpleNamespace(resolve=lambda _digest: checkpoint),
    )
    verified_patcher = ModelPatchLoader.execute(name=verified)["MODEL_PATCH"]
    assert type(verified_patcher.model).__name__ == "DurationHead"
    resolver.revoked = True
    with pytest.raises(Exception, match="revoked"):
        ModelPatchLoader.execute(name=asset)


class _RecordedModel:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.loaded: dict[str, Any] | None = None
        self.assign: bool | None = None
        self.denoise_encoder_sd: dict[str, Any] | None = None
        self.frozen = False

    def load_state_dict(self, state_dict: dict[str, Any], *, assign: bool) -> None:
        self.loaded = state_dict
        self.assign = assign

    def requires_grad_(self, value: bool) -> None:
        self.frozen = not value


class _Patcher:
    def __init__(self, model: _RecordedModel, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs

    def is_dynamic(self) -> bool:
        return False


class _BranchRuntime:
    """Controlled constructor seam for fixed-size production architectures."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        state: dict[str, Any],
        metadata: dict[str, str] | None,
    ) -> None:
        self.state = state
        self.metadata = metadata
        self.models: list[_RecordedModel] = []
        self.quantization: object | None = None
        self.orig_import = loader.importlib.import_module
        self.monkeypatch = monkeypatch
        monkeypatch.setattr(loader, "initialize_inference_process", lambda: None)
        monkeypatch.setattr(loader, "_qwen_blockwise_controlnet", self._factory)
        monkeypatch.setattr(loader, "_siglip_multi_feat_projector", self._factory)
        monkeypatch.setattr(
            loader, "_multi_talk_patch", lambda _multitalk, _operations: self._factory()
        )
        monkeypatch.setattr(loader.importlib, "import_module", self._import)

    def _factory(self, *_args: Any, **_kwargs: Any) -> Any:
        def construct(**kwargs: Any) -> _RecordedModel:
            model = _RecordedModel(**kwargs)
            self.models.append(model)
            return model

        return construct

    def _anima(
        self, state_dict: dict[str, Any], metadata: dict[str, str] | None, **kwargs: Any
    ) -> _RecordedModel:
        model = _RecordedModel(**kwargs)
        model.loaded = state_dict
        model.metadata = metadata
        self.models.append(model)
        return model

    def _import(self, name: str) -> Any:
        if name == "torch" or name == "torch.nn":
            return self.orig_import(name)
        if name == "dinkster_inference.utils":
            return SimpleNamespace(
                load_torch_file=lambda *_args, **_kwargs: (self.state, self.metadata),
                weight_dtype=lambda _state: torch.float16,
                detect_layer_quantization=lambda _state, _prefix: self.quantization,
                state_dict_prefix_replace=self._prefix_replace,
            )
        if name == "dinkster_inference.model_management":
            return SimpleNamespace(
                unet_offload_device=lambda: "offload",
                get_torch_device=lambda: "cpu",
                unet_dtype=lambda **_kwargs: torch.float32,
                unet_manual_cast=lambda *_args, **_kwargs: None,
            )
        if name == "dinkster_inference.ops":
            return SimpleNamespace(
                manual_cast=object(),
                mixed_precision_ops=lambda quant, dtype: ("mixed", quant, dtype),
                pick_operations=lambda dtype, manual, **_kwargs: ("picked", dtype, manual),
            )
        if name == "dinkster_inference.storage":
            return SimpleNamespace(state_dict_fast_disk=lambda _state: "fast-disk")
        if name == "dinkster_inference.model_patcher":
            return SimpleNamespace(CoreModelPatcher=_Patcher)
        if name == "dinkster_inference.ldm.anima.lllite":
            return SimpleNamespace(AnimaLLLite=self._anima)
        if name == "dinkster_inference.ldm.lumina.controlnet":
            return SimpleNamespace(ZImage_Control=self._factory())
        if name == "dinkster_inference.ldm.qwen_image21.model":
            return SimpleNamespace(QwenImage21FunControl=self._factory())
        if name == "dinkster_inference.ldm.minimax.controlnet":
            return SimpleNamespace(
                is_minimax_h3_fun_state_dict=lambda state: "minimax" in state,
                MiniMaxH3FunControl=self._factory(),
            )
        if name == "dinkster_inference.ldm.wan.uni3c":
            return SimpleNamespace(WanUni3CControlnet=self._factory())
        if name == "dinkster_inference.ldm.wan.model_multitalk":
            return SimpleNamespace()
        if name == "dinkster_inference.ldm.supir.supir_modules":
            return SimpleNamespace(SUPIR=self._factory())
        return self.orig_import(name)

    @staticmethod
    def _prefix_replace(
        state: dict[str, Any], prefixes: dict[str, str], *, filter_keys: bool
    ) -> dict[str, Any]:
        converted: dict[str, Any] = {}
        for key, value in state.items():
            for source, destination in prefixes.items():
                if key.startswith(source):
                    converted[destination + key[len(source) :]] = value
                    break
            else:
                if not filter_keys:
                    converted[key] = value
        return converted

    def execute(self, tmp_path: Path) -> _RecordedModel:
        tmp_path.mkdir(parents=True, exist_ok=True)
        checkpoint = tmp_path / "patch.safetensors"
        checkpoint.touch()
        asset = LocalAssetRef(
            "patch", checkpoint.name, 0, checkpoint.name, resolver=LocalResolver(checkpoint)
        )
        patcher = ModelPatchLoader.execute(name=asset)["MODEL_PATCH"]
        assert isinstance(patcher, _Patcher)
        assert patcher.kwargs["fast_disk"] == "fast-disk"
        return patcher.model


def test_loader_dispatches_anima_qwen_siglip_and_multitalk_with_controlled_constructors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = (
        (
            {"lllite_conditioning1.conv1.weight": torch.tensor([11.0])},
            {"source": "anima"},
            {"device": "offload", "dtype": torch.float16},
        ),
        (
            {
                "controlnet_blocks.0.y_rms.weight": torch.tensor([1.0]),
                "img_in.weight": torch.empty(2, 71),
            },
            None,
            {"additional_in_dim": 7},
        ),
        (
            {"feature_embedder.mid_layer_norm.bias": torch.tensor([2.0])},
            None,
            {},
        ),
        (
            {
                "audio_proj.proj1.weight": torch.empty(13, 1),
                "audio_proj.norm.weight": torch.empty(17),
                "blocks.0.audio_cross_attn.proj.weight": torch.empty(19, 1),
            },
            None,
            {"in_dim": 19, "intermediate_dim": 13, "out_dim": 17, "dtype": torch.float16},
        ),
    )
    for index, (state, metadata, expected) in enumerate(cases):
        with monkeypatch.context() as patch:
            runtime = _BranchRuntime(patch, state, metadata)
            model = runtime.execute(tmp_path / str(index))
            assert runtime.models == [model]
            assert all(model.kwargs[key] == value for key, value in expected.items())
            if index == 0:
                assert model.metadata == metadata
            if index == 2:
                assert model.loaded == {
                    "mid_layer_norm.bias": state["feature_embedder.mid_layer_norm.bias"]
                }


@pytest.mark.parametrize(
    ("layers", "broken", "expected"),
    [
        (
            (),
            False,
            {"n_control_layers": 3, "additional_in_dim": 17, "refiner_control": True},
        ),
        ((4,), False, {}),
        (
            (14,),
            True,
            {
                "n_control_layers": 15,
                "additional_in_dim": 17,
                "refiner_control": True,
                "broken": True,
            },
        ),
        ((14,), False, {"n_control_layers": 15, "additional_in_dim": 17, "refiner_control": True}),
    ],
)
def test_loader_configures_zimage_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    layers: tuple[int, ...],
    broken: bool,
    expected: dict[str, object],
) -> None:
    state: dict[str, Any] = {
        "control_all_x_embedder.2-1.weight": torch.tensor([3.0]),
        "x.attention.to_q.weight": torch.tensor([[1.0]]),
        "x.attention.to_k.weight": torch.tensor([[2.0]]),
        "x.attention.to_v.weight": torch.tensor([[3.0]]),
    }
    for layer in layers:
        state[f"control_layers.{layer}.adaLN_modulation.0.weight"] = torch.tensor([4.0])
    if layers == (14,):
        state["control_noise_refiner.0.after_proj.weight"] = torch.tensor([0.0 if broken else 5.0])
    runtime = _BranchRuntime(monkeypatch, state, None)
    model = runtime.execute(tmp_path)
    assert {key: model.kwargs[key] for key in expected} == expected
    assert torch.equal(model.loaded["x.attention.qkv.weight"], torch.tensor([[1.0], [2.0], [3.0]]))


@pytest.mark.parametrize("fused", [False, True])
def test_loader_configures_qwen21_block_count_and_quantization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fused: bool
) -> None:
    state: dict[str, Any] = {
        "control_img_in.weight": torch.empty(8, 129),
        "control_blocks.0.img_mlp.out.weight": torch.empty(1),
        "control_blocks.0.img_mlp.proj.weight": torch.empty(24, 8),
        "control_blocks.0.attn.norm_q.weight": torch.empty(4),
        "control_blocks.0.after_proj.weight": torch.empty(1),
        "control_blocks.1.after_proj.weight": torch.empty(1),
    }
    if fused:
        state["control_blocks.0.img_mlp.gate_up.weight"] = torch.empty(32, 8)
    runtime = _BranchRuntime(monkeypatch, state, None)
    runtime.quantization = "int8"
    model = runtime.execute(tmp_path)
    assert model.kwargs["num_blocks"] == 2
    assert model.kwargs["fused_mlp"] is fused
    assert model.kwargs["mlp_ratio"] == (2 if fused else 3)
    assert model.kwargs["dtype"] is torch.bfloat16
    assert model.kwargs["operations"] == ("mixed", "int8", torch.bfloat16)


def test_loader_configures_minimax_metadata_and_rejects_placement_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = {
        "minimax": torch.tensor(1),
        "control_blocks.0.after_proj.weight": torch.empty(1),
        "control_blocks.1.after_proj.weight": torch.empty(1),
        "control_proj_in.weight": torch.empty(12, 49),
        "control_blocks.0.attn.qkv_proj.weight": torch.empty(36, 1),
        "control_blocks.0.attn.q_norm.weight": torch.empty(3),
        "control_blocks.0.mlp.fc1.weight": torch.empty(20, 1),
    }
    metadata = {
        "control_blocks_places": "[4, 31]",
        "minimax_h3_fun_controlnet": "adaln_basis",
        "inpaint_masked_pixel_mode": "post_norm",
    }
    runtime = _BranchRuntime(monkeypatch, state, metadata)
    runtime.quantization = "nf4"
    model = runtime.execute(tmp_path / "valid")
    assert model.kwargs["injection_layers"] == (4, 31)
    assert model.kwargs["time_embed_dim"] == 8
    assert model.kwargs["use_adaln_curves"] is True
    assert model.kwargs["inpaint_post_norm"] is True
    assert model.kwargs["num_attention_heads"] == 4
    assert model.frozen is True

    runtime = _BranchRuntime(monkeypatch, state, {"control_blocks_places": "[4]"})
    with pytest.raises(ValueError, match="does not match"):
        runtime.execute(tmp_path / "invalid")


def test_loader_rewrites_wan_keys_and_uses_conv_dimension_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = {
        "controlnet_patch_embedding.weight": torch.empty(7, 5, 1, 1),
        "controlnet_blocks.0.ffn.0.bias": torch.empty(13),
        "controlnet_blocks.0.norm1.linear.weight": torch.empty(1, 17),
        "controlnet_blocks.0.self_attn.to_q.weight": torch.tensor([23.0]),
        "controlnet_blocks.0.self_attn.to_k.weight": torch.tensor([29.0]),
        "controlnet_blocks.0.self_attn.to_v.weight": torch.tensor([31.0]),
        "controlnet_blocks.0.self_attn.to_out.0.weight": torch.tensor([37.0]),
        "proj_out.0.weight": torch.empty(19, 1),
        "proj_out.1.weight": torch.empty(19, 1),
        "controlnet_mask_embedding.mask_proj.0.weight": torch.empty(11, 3, 1, 1),
    }
    runtime = _BranchRuntime(monkeypatch, state, None)
    model = runtime.execute(tmp_path)
    assert model.kwargs["dim"] == 7
    assert model.kwargs["num_layers"] == 2
    assert model.loaded["controlnet_blocks.0.self_attn.q.weight"].item() == 23.0
    assert model.loaded["controlnet_blocks.0.self_attn.o.weight"].item() == 37.0


@pytest.mark.parametrize("prefixed", [False, True])
def test_loader_filters_supir_state_and_retains_denoise_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prefixed: bool
) -> None:
    prefix = "model.control_model." if prefixed else "control_model."
    project_prefix = "model.diffusion_model.project_modules." if prefixed else "project_modules."
    state = {
        f"{prefix}input_hint_block.0.weight": torch.tensor([41.0]),
        f"{prefix}mask_LQ": torch.tensor([43.0]),
        f"{project_prefix}project.weight": torch.tensor([47.0]),
        "first_stage_model.denoise_encoder.weight": torch.tensor([53.0]),
        "discard.weight": torch.tensor([59.0]),
    }
    runtime = _BranchRuntime(monkeypatch, state, None)
    model = runtime.execute(tmp_path)
    assert set(model.loaded) == {
        "control_model.input_hint_block.0.weight",
        "project_modules.project.weight",
    }
    assert model.denoise_encoder_sd == {"weight": state["first_stage_model.denoise_encoder.weight"]}


class ModelPatchInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="model-patch-fixture.inspect",
            inputs=(InputSpec("patch", TypeExpr.concrete(MODEL_PATCH_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, patch: Any) -> dict[str, object]:
        from dinkster_inference.ldm.lightricks.duration_head import DurationHead
        from dinkster_inference.model_patcher import ModelPatcher

        assert isinstance(patch, ModelPatcher)
        assert isinstance(patch.model, DurationHead)
        weights = patch.model.state_dict()
        for key, value in weights.items():
            assert value.dtype == torch.float32
            expected = 0.0 if key == "mlp_out.weight" else 0.5 if key == "mlp_out.bias" else 0.001
            assert torch.equal(value, torch.full_like(value, expected))
        output = patch.model(video_tokens=torch.full((1, 2, 4096), 0.25))
        assert torch.equal(output, torch.tensor([0.5]).exp())
        return cls.outputs(summary="duration checkpoint retained")


MODEL_PATCH_FIXTURES = (ModelPatchInspect,)


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("verified", (False, True))
def test_model_patch_loader_both_ordinary_hosts_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, verified: bool
) -> None:
    from dinkster_inference.ldm.lightricks.duration_head import DurationHead
    from safetensors.torch import save_file

    checkpoint = tmp_path / "duration.safetensors"
    state = {
        f"duration_head.{key}": torch.full_like(value, 0.001)
        for key, value in DurationHead().state_dict().items()
    }
    state["duration_head.mlp_out.weight"].zero_()
    state["duration_head.mlp_out.bias"].fill_(0.5)
    save_file(
        state,
        str(checkpoint),
        metadata={"format": "pt"},
    )
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    table.add(MountDef(id="patches", path=tmp_path), kind="model/patch")
    if verified:
        table.scan("patches")
    asset = table.bind_name(checkpoint.name, "model/patch")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = (
            root / "dinkster-pack.toml",
            root / "tests/fixtures/model_patch/dinkster-pack.toml",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src"))),
            }
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="model-patch-fixture",
                        group_manifests=manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "load": GraphNode("ModelPatchLoader", {"name": asset.to_wire()}),
                    "inspect": GraphNode(
                        "model-patch-fixture.inspect", {"patch": Link("load", "MODEL_PATCH")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve() == "duration checkpoint retained"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
