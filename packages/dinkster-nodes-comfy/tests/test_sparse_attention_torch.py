"""CPU seams for BlockSparseAttention model patching."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.sparse_attention import (
    BlockSparseAttention,
    SparseAttnPatch,
    h3_sparse_attention,
    install_override,
    make_attention_override,
    make_h3_block_patch,
)

from dinkster.compose import PackSpec, ServingComposer


class _Sampling:
    def percent_to_sigma(self, percent: float) -> float:
        return 1.0 - percent


class _Model:
    def __init__(self) -> None:
        self.model_options = {"transformer_options": {}}
        self.callbacks: dict[tuple[Any, str], Any] = {}

    def clone(self) -> _Model:
        result = _Model()
        result.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        return result

    def get_model_object(self, name: str) -> Any:
        return _Sampling() if name == "model_sampling" else object()

    def add_callback_with_key(self, _kind: Any, key: str, callback: Any) -> None:
        self.callbacks[_kind, key] = callback


class ManifestModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.previous_calls = 0
        self.model_options["transformer_options"]["optimized_attention_override"] = self.previous

    def clone(self) -> ManifestModel:
        clone = ManifestModel()
        clone.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        return clone

    def previous(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.previous_calls += 1
        return func(*args, **kwargs)


class ManifestSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sparse-attention-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(MODEL=ManifestModel())


class ManifestReader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sparse-attention-fixture.reader",
            inputs=(
                InputSpec("source", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("patched", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("verified", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, source: ManifestModel, patched: ManifestModel) -> dict[str, object]:
        source_hook = source.model_options["transformer_options"]["optimized_attention_override"]
        assert getattr(source_hook, "__self__", None) is source
        assert not source.callbacks
        prepare, cleanup = patched.callbacks.values()
        replacement_calls: list[str] = []

        def replacement(func: Any, *args: Any, **kwargs: Any) -> Any:
            replacement_calls.append("replacement")
            return func(*args, **kwargs)

        options = {"transformer_options": {"optimized_attention_override": replacement}}
        prepare(patched, 0, options)
        installed = options["transformer_options"]["optimized_attention_override"]
        prepare(patched, 0, options)
        assert options["transformer_options"]["optimized_attention_override"] is installed
        q = torch.zeros((1, 5, 128), dtype=torch.float32)
        result = installed(lambda *args, **kwargs: args[0] + args[1] + args[2], q, q, q, 1)
        assert torch.equal(result, torch.zeros_like(q))
        assert replacement_calls == ["replacement"]
        patch = next(
            value.cell_contents
            for value in cleanup.__closure__ or ()
            if isinstance(value.cell_contents, SparseAttnPatch)
        )
        patch.pooled[(1,)] = (q, q)
        patch.vsa_plans[(1,)] = {}
        cleanup(patched)
        assert not patch.pooled and not patch.vsa_plans and patch.vsa_rope is None
        return cls.outputs(verified=1.0)


MANIFEST_NODES = (ManifestSource, ManifestReader)


def install_runtime_fakes() -> None:
    package = types.ModuleType("dinkster_inference")
    package.__path__ = []
    cli_args = types.ModuleType("dinkster_inference.cli_args")
    cli_args.args = SimpleNamespace(cpu=False)
    management = types.ModuleType("dinkster_inference.model_management")
    extension = types.ModuleType("dinkster_inference.patcher_extension")
    extension.CallbacksMP = SimpleNamespace(ON_PREPARE_STATE="prepare", ON_CLEANUP="cleanup")
    minimax = types.ModuleType("dinkster_inference.ldm.minimax.model")

    class MiniMaxH3Model:
        pass

    minimax.MiniMaxH3Model = MiniMaxH3Model
    modules = {
        "dinkster_inference": package,
        "dinkster_inference.cli_args": cli_args,
        "dinkster_inference.model_management": management,
        "dinkster_inference.patcher_extension": extension,
        "dinkster_inference.ldm": types.ModuleType("dinkster_inference.ldm"),
        "dinkster_inference.ldm.minimax": types.ModuleType("dinkster_inference.ldm.minimax"),
        "dinkster_inference.ldm.minimax.model": minimax,
    }
    sys.modules.update(modules)


def _patch(*, minimum: int = 0) -> SparseAttnPatch:
    return SparseAttnPatch(1.3, 0.0, False, 1.0, 0.0, minimum, {2}, "off", 0, False)


def test_schema_has_source_identity_required_inputs_and_stable_methods() -> None:
    schema = BlockSparseAttention.define_schema()
    assert BlockSparseAttention in COMFY_NODES
    assert schema.node_type == "BlockSparseAttention"
    assert not schema.aliases
    assert [item.id for item in schema.inputs] == [
        "model",
        "start_percent",
        "end_percent",
        "dense_blocks",
        "min_tokens",
        "extra_tokens",
        "sink_conditioning",
        "verbose",
    ]
    assert all(item.required for item in schema.inputs)
    assert [option.key for option in schema.combos[0].options] == ["sol-attn", "sla", "vsa"]


def test_dense_fallback_preserves_previous_override_and_all_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dinkster_nodes_comfy.sparse_attention._runtime", lambda: (torch, object()))
    patch = _patch(minimum=8)
    calls: list[str] = []

    def base(*_args: Any, **_kwargs: Any) -> str:
        calls.append("base")
        return "base"

    def previous(func: Any, *args: Any, **kwargs: Any) -> str:
        calls.append("previous")
        return func(*args, **kwargs)

    override = make_attention_override(patch, previous)
    q = torch.zeros((1, 4, 128))
    for options in ({}, {"block_index": 2}, {"sigmas": torch.tensor([2.0])}):
        assert override(base, q, q, q, 1, transformer_options=options) == "base"
    assert override(base, q, q, q, 1, mask=torch.zeros((1, 1))) == "base"
    assert calls == ["previous", "base"] * 4
    assert patch.dense_reason({"sigmas": torch.tensor([1.0])}, 8, 0) is None
    assert patch.dense_reason({"sigmas": torch.tensor([0.0])}, 8, 0) is None


def test_vsa_plan_inverse_rope_cache_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dinkster_nodes_comfy.sparse_attention._runtime", lambda: (torch, object()))
    patch = _patch()
    layout = SimpleNamespace(
        signature=(0, 2, 4, 4, 0),
        segments=((0, 3, "text"), (3, 11, "video")),
        seq_len=11,
    )
    plan = patch.vsa_plan(layout, torch.device("cpu"))
    assert plan["src"][plan["inv"]].tolist() == list(range(layout.seq_len))
    freqs = torch.randn((1, layout.seq_len, 2))
    assert patch.vsa_rope_freqs(freqs, plan) is patch.vsa_rope_freqs(freqs, plan)
    patch.reset()
    assert not patch.vsa_plans and patch.vsa_rope is None


def test_override_installation_is_idempotent_and_chains_reinstall() -> None:
    patch = _patch()
    options: dict[str, Any] = {"optimized_attention_override": lambda *_args, **_kwargs: None}
    original = options["optimized_attention_override"]
    install_override(patch, options)
    installed = options["optimized_attention_override"]
    install_override(patch, options)
    assert options["optimized_attention_override"] is installed
    options["optimized_attention_override"] = original
    install_override(patch, options)
    assert options["optimized_attention_override"] is not original


def test_h3_chunked_producer_uses_kernel_seam_and_block_patch_keeps_cpu_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Kitchen:
        calls: list[dict[str, Any]] = []

        @staticmethod
        def sol_attn_chunked(
            producer: Any, tokens: int, heads: int, freqs: Any, weights: Any, **kwargs: Any
        ) -> Any:
            chunks = list(producer())
            Kitchen.calls.append(
                {"tokens": tokens, "heads": heads, "freqs": freqs, "weights": weights, **kwargs}
            )
            assert len(chunks) == 1 and chunks[0].shape == (5, 3)
            return (
                torch.ones((tokens, heads, 128)),
                torch.full((heads, 128), 2.0),
                torch.full((heads, 128), 3.0),
            )

    management = types.SimpleNamespace(cast_to=lambda weight, **_kwargs: weight)
    prefetch = types.SimpleNamespace(pause_malloc_graph=contextlib.nullcontext)
    monkeypatch.setattr("dinkster_nodes_comfy.sparse_attention._runtime", lambda: (torch, Kitchen))
    monkeypatch.setitem(sys.modules, "dinkster_inference.model_management", management)
    monkeypatch.setitem(sys.modules, "dinkster_inference.model_prefetch", prefetch)
    attn = SimpleNamespace(
        heads=1,
        head_dim=128,
        q_norm=SimpleNamespace(weight=torch.ones(128), eps=1e-5),
        k_norm=SimpleNamespace(weight=torch.ones(128)),
        qkv_proj=lambda value: torch.cat((value, value, value), dim=1),
        out_proj=lambda value: value,
        to_gate_compress=None,
    )
    patch = _patch()
    x = torch.arange(5 * 1, dtype=torch.bfloat16).reshape(5, 1)
    freqs = torch.zeros((1, 5, 2))
    options = {"uuids": ("conditioning",)}
    assert h3_sparse_attention(attn, x, freqs, options, patch, 3).shape == (5, 128)
    assert h3_sparse_attention(attn, x, freqs, options, patch, 3).shape == (5, 128)
    assert (
        Kitchen.calls[0]["kmean"] is None
        and Kitchen.calls[1]["kmean"] is patch.pooled[(3, 5, ("conditioning",))][0]
    )
    original_calls: list[dict[str, Any]] = []
    block = SimpleNamespace(attn=attn)
    block_patch = make_h3_block_patch(block, 3, patch)
    args = {"img": x, "rope_freqs": freqs, "transformer_options": options}
    assert (
        block_patch(args, {"original_block": lambda value: original_calls.append(value) or value})
        is args
    )
    assert original_calls == [args]


@pytest.mark.parametrize("in_process", (True, False))
def test_complete_manifest_runs_sparse_attention_callbacks_twice_each_host(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        source_root = root / "src"
        install_runtime_fakes()
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(source_root))
                )
            }
        )
        try:
            manifest = root / "dinkster-pack.toml"
            (tmp_path / "sparse_attention_fixture_sources.py").write_text(
                "from test_sparse_attention_torch import MANIFEST_NODES, install_runtime_fakes\n"
                "install_runtime_fakes()\n"
                "from dinkster_api.v1 import register_inference_types\n"
                "from dinkster_nodes_comfy.sparse_attention import BlockSparseAttention\n"
                "NODES = (BlockSparseAttention, *MANIFEST_NODES)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            contents = re.sub(
                r"namespaces = \[.*?\]",
                'namespaces = ["BlockSparseAttention", "sparse-attention-fixture"]',
                manifest.read_text(encoding="ascii"),
                count=1,
                flags=re.DOTALL,
            )
            contents = contents.replace(
                "dinkster_nodes_comfy:COMFY_NODES", "sparse_attention_fixture_sources:NODES"
            ).replace(
                "dinkster_nodes_comfy:register_inference_types",
                "sparse_attention_fixture_sources:register_inference_types",
            )
            fixture_manifest.write_text(contents, encoding="ascii")
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("sparse-attention-fixture.source"),
                    "patch": GraphNode(
                        "BlockSparseAttention",
                        {
                            "model": Link("source", "MODEL"),
                            "selection.tau": 1.3,
                            "start_percent": 0.2,
                            "end_percent": 1.0,
                            "dense_blocks": "2",
                            "min_tokens": 8,
                            "extra_tokens": 256,
                            "sink_conditioning": "off",
                            "verbose": False,
                        },
                        slot_variants={"selection": "sol-attn"},
                    ),
                    "reader": GraphNode(
                        "sparse-attention-fixture.reader",
                        {"source": Link("source", "MODEL"), "patched": Link("patch", "MODEL")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["reader"])
                assert result.outputs["reader"]["verified"].resolve() == 1.0
        finally:
            await composer.close()

    asyncio.run(scenario())
