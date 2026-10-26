"""Block-sparse attention model patching."""

from __future__ import annotations

import importlib
import logging
import re
import weakref
from collections.abc import Callable, Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    MODEL_TYPE,
    ComboOption,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
)

HEAD_DIM = 128
BLOCK_SIZE = 64
PRODUCER_CHUNK = 4096
VSA_CUBE = (4, 4, 4)
VSA_PLAN_CACHE = 4

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_STRING = TypeExpr.concrete(CORE_STRING)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


def _runtime() -> tuple[Any, Any]:
    initialize_inference_process()
    return importlib.import_module("torch"), importlib.import_module("comfy_kitchen")


def parse_block_list(text: str) -> set[int]:
    blocks: set[int] = set()
    for part in re.findall(r"\d+\s*-\s*\d+|\d+", text or ""):
        if "-" in part:
            start, end = (int(value) for value in part.split("-"))
            blocks.update(range(min(start, end), max(start, end) + 1))
        else:
            blocks.add(int(part))
    return blocks


class SparseAttnPatch:
    def __init__(
        self,
        tau: float,
        topk_ratio: float,
        vsa: bool,
        sigma_start: float,
        sigma_end: float,
        min_tokens: int,
        dense_blocks: set[int],
        sink_conditioning: str,
        extra_tokens: int,
        verbose: bool,
    ) -> None:
        self.tau = tau
        self.topk_ratio = topk_ratio
        self.extra_tokens = extra_tokens
        self.vsa = vsa
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.min_tokens = min_tokens
        self.dense_blocks = dense_blocks
        self.sink_conditioning = sink_conditioning
        self.verbose = verbose
        self.installed: set[Callable[..., Any]] = set()
        self.reset()

    def reset(self) -> None:
        self.pooled: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self.vsa_plans: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.vsa_rope: tuple[Any, dict[str, Any], Any] | None = None
        self._logged: set[Any] = set()

    def log_once(self, key: Any, message: str) -> None:
        if self.verbose and key not in self._logged:
            self._logged.add(key)
            logging.info("BlockSparseAttention: %s", message)

    def dense_reason(
        self, transformer_options: Mapping[str, Any], tokens: int, block_index: int | None
    ) -> str | None:
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None:
            sigma = float(sigmas[0])
            if sigma > self.sigma_start or sigma < self.sigma_end:
                return f"sigma {sigma:.3g} outside the start/end window"
        if tokens < self.min_tokens:
            return f"{tokens} tokens < min_tokens {self.min_tokens}"
        if self.dense_blocks:
            if block_index is None:
                self.log_once(
                    "no_block_index",
                    "this model does not report block indices; dense_blocks ignored",
                )
            elif block_index in self.dense_blocks:
                return f"block {block_index} in dense_blocks"
        return None

    def sinks(
        self, transformer_options: Mapping[str, Any], tokens: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        layout = transformer_options.get("minimax_h3_layout")
        if self.sink_conditioning == "off" or layout is None or layout.seq_len != tokens:
            return (0, 0), (0, 0)
        video = next(
            ((start, end) for start, end, kind in layout.segments if kind == "video"), None
        )
        if video is None or video[0] <= 0:
            return (0, 0), (0, 0)
        blocks = (0, (video[0] + BLOCK_SIZE - 1) // BLOCK_SIZE)
        if self.sink_conditioning != "exact_kv_and_rows":
            return blocks, (0, 0)
        audio = next(
            ((start, end) for start, end, kind in layout.segments if kind == "audio"), None
        )
        return (blocks, blocks if audio is None else (audio[0] // BLOCK_SIZE, blocks[1]))

    def vsa_plan(self, layout: Any, device: Any) -> dict[str, Any]:
        torch, _ = _runtime()
        key = (tuple(layout.signature), tuple(layout.segments), str(device))
        plan = self.vsa_plans.get(key)
        if plan is not None:
            return plan
        _, latent_t, latent_h, latent_w, _ = layout.signature
        grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
        tiles: list[Any] = []
        prefix_tiles = 0
        for start, end, kind in layout.segments:
            rows = end - start
            if kind != "video":
                count = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
                segment = torch.full((count * BLOCK_SIZE,), -1, dtype=torch.int64, device=device)
                segment[:rows] = torch.arange(start, end, device=device)
                tiles.append(segment.view(count, BLOCK_SIZE))
                prefix_tiles += count
                continue
            if grid[0] * grid[1] * grid[2] != rows:
                raise RuntimeError(
                    f"VSA: video segment of {rows} rows does not match latent grid {grid}"
                )
            cube_t, cube_h, cube_w = VSA_CUBE
            pad_t, pad_h, pad_w = (
                (size + cube - 1) // cube * cube for size, cube in zip(grid, VSA_CUBE, strict=True)
            )
            padded = torch.full((pad_t, pad_h, pad_w), -1, dtype=torch.int64, device=device)
            padded[: grid[0], : grid[1], : grid[2]] = torch.arange(start, end, device=device).view(
                *grid
            )
            cubes = (
                padded.view(
                    pad_t // cube_t, cube_t, pad_h // cube_h, cube_h, pad_w // cube_w, cube_w
                )
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(-1, BLOCK_SIZE)
            )
            order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
            tiles.append(torch.gather(cubes, 1, order))
        source = torch.cat(tiles).reshape(-1)
        live = source >= 0
        inverse = torch.empty(layout.seq_len, dtype=torch.int64, device=device)
        inverse[source[live]] = torch.nonzero(live).flatten()
        plan = {
            "n": int(source.numel()),
            "n_prefix": prefix_tiles,
            "src": source,
            "inv": inverse,
            "block_len": (torch.cat(tiles) >= 0).sum(1).to(torch.int32),
        }
        while len(self.vsa_plans) >= VSA_PLAN_CACHE:
            del self.vsa_plans[next(iter(self.vsa_plans))]
        self.vsa_plans[key] = plan
        return plan

    def vsa_rope_freqs(self, rope_freqs: Any, plan: dict[str, Any]) -> Any:
        hit = self.vsa_rope
        if hit is not None and hit[0]() is rope_freqs and hit[1] is plan:
            return hit[2]
        padded = rope_freqs.new_zeros((1, plan["n"]) + tuple(rope_freqs.shape[2:]))
        padded[0, plan["inv"]] = rope_freqs[0]
        self.vsa_rope = (weakref.ref(rope_freqs), plan, padded)
        return padded


def _ineligible(q: Any, k: Any, v: Any, dim_head: int) -> str | None:
    torch, kitchen = _runtime()
    if q.device.type != "cuda":
        return "not on CUDA"
    available = getattr(kitchen, "sol_attn_is_available", None)
    if not callable(available) or not available(q.device):
        return "no compiled sol_attn kernel for this GPU"
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {q.dtype} (kernel takes bf16/fp16)"
    if dim_head != HEAD_DIM:
        return f"head_dim {dim_head} != {HEAD_DIM}"
    if q.shape != k.shape or q.shape != v.shape:
        return "cross-attention or GQA (kept dense)"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return f"mixed dtypes {q.dtype}/{k.dtype}/{v.dtype}"
    return None


def make_attention_override(
    patch: SparseAttnPatch, previous: Callable[..., Any] | None
) -> Callable[..., Any]:
    def override(
        func: Callable[..., Any],
        q: Any,
        k: Any,
        v: Any,
        heads: int,
        mask: Any = None,
        attn_precision: Any = None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
        **kwargs: Any,
    ) -> Any:
        transformer_options = kwargs.get("transformer_options")
        if not isinstance(transformer_options, Mapping):
            transformer_options = {}
        transformer_options = cast(Mapping[str, Any], transformer_options)

        def dense() -> Any:
            options = dict(
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
            return (
                func(q, k, v, heads, **options)
                if previous is None
                else previous(func, q, k, v, heads, **options)
            )

        if mask is not None or patch.vsa:
            return dense()
        tokens = q.shape[2] if skip_reshape else q.shape[1]
        reason = patch.dense_reason(
            transformer_options, tokens, transformer_options.get("block_index")
        )
        if reason is not None:
            patch.log_once(("dense", tokens, reason), f"dense ({tokens} tokens): {reason}")
            return dense()
        if skip_reshape:
            batch, _, _, dim_head = q.shape
            queries, keys, values = (tensor.transpose(1, 2) for tensor in (q, k, v))
        else:
            batch, _, dim_head = q.shape
            dim_head //= heads
            queries, keys, values = (
                tensor.view(batch, -1, heads, dim_head) for tensor in (q, k, v)
            )
        reason = _ineligible(queries, keys, values, dim_head)
        if reason is not None:
            patch.log_once(
                ("ineligible", tuple(queries.shape), reason),
                f"dense {tuple(queries.shape)}: {reason}",
            )
            return dense()
        torch, kitchen = _runtime()
        sinks, sink_queries = patch.sinks(transformer_options, tokens)
        if q.dtype == torch.float32:
            queries, keys, values = (
                tensor.to(torch.bfloat16) for tensor in (queries, keys, values)
            )
        output = kitchen.sol_attn(
            queries,
            keys,
            values,
            tau=patch.tau,
            scale=kwargs.get("scale"),
            sink_blocks=list(sinks),
            sink_q=list(sink_queries),
            topk_ratio=patch.topk_ratio,
            token_aug=patch.extra_tokens,
        ).to(q.dtype)
        patch.log_once(
            ("sparse", tuple(queries.shape)),
            f"sparse {tuple(queries.shape)}, sinks {sinks}/{sink_queries}",
        )
        return (
            output.transpose(1, 2)
            if skip_output_reshape
            else output.reshape(batch, -1, heads * dim_head)
        )

    return override


def install_override(patch: SparseAttnPatch, transformer_options: dict[str, Any]) -> None:
    current = transformer_options.get("optimized_attention_override")
    if current in patch.installed:
        return
    override = make_attention_override(patch, current)
    patch.installed.add(override)
    transformer_options["optimized_attention_override"] = override


def h3_eligible(
    attn: Any,
    x: Any,
    rope_freqs: Any,
    transformer_options: Mapping[str, Any],
    patch: SparseAttnPatch,
    block_index: int,
) -> bool:
    _, kitchen = _runtime()
    tokens = x.shape[0]
    if (
        rope_freqs is None
        or x.dtype != _runtime()[0].bfloat16
        or x.device.type != "cuda"
        or attn.head_dim != HEAD_DIM
    ):
        return False
    reason = patch.dense_reason(transformer_options, tokens, block_index)
    available = getattr(kitchen, "sol_attn_is_available", None)
    if reason is None and (not callable(available) or not available(x.device)):
        reason = "no compiled sol_attn kernel for this GPU"
    if reason is not None:
        patch.log_once(("dense", tokens, reason), f"dense ({tokens} tokens): {reason}")
        return False
    if patch.vsa:
        layout = transformer_options.get("minimax_h3_layout")
        if layout is None or layout.seq_len != tokens:
            patch.log_once("no_layout", "no H3 layout for this call; running dense")
            return False
    return True


def h3_sparse_attention(
    attn: Any,
    x: Any,
    rope_freqs: Any,
    transformer_options: Mapping[str, Any],
    patch: SparseAttnPatch,
    block_index: int,
) -> Any:
    torch, kitchen = _runtime()
    tokens, heads, head_dim = x.shape[0], attn.heads, attn.head_dim
    management = importlib.import_module("dinkster_inference.model_management")
    prefetch = importlib.import_module("dinkster_inference.model_prefetch")
    q_weight = management.cast_to(attn.q_norm.weight, device=x.device)
    k_weight = management.cast_to(attn.k_norm.weight, device=x.device)
    extra: dict[str, Any] = {}
    plan = gate = None
    padded_tokens, freqs = tokens, rope_freqs
    with prefetch.pause_malloc_graph():
        if patch.vsa:
            plan = patch.vsa_plan(transformer_options["minimax_h3_layout"], x.device)
            padded_tokens = plan["n"]
            freqs = patch.vsa_rope_freqs(rope_freqs, plan)
        key = (block_index, padded_tokens, tuple(transformer_options.get("uuids", ())))
        pooled = patch.pooled.get(key)
        first = pooled is None
        if first:
            pooled = (
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
            )
    if plan is not None:
        sinks = sink_queries = (0, plan["n_prefix"])
        extra = {"tail": False, "block_len": plan["block_len"]}
        gate = attn.to_gate_compress
        if gate is not None:
            extra["coarse_gate"] = x.new_empty(padded_tokens, heads * head_dim).view(
                1, padded_tokens, heads, head_dim
            )
    else:
        sinks, sink_queries = patch.sinks(transformer_options, tokens)

    def chunks() -> Any:
        for start in range(0, padded_tokens, PRODUCER_CHUNK):
            if plan is None:
                yield attn.qkv_proj(x[start : start + PRODUCER_CHUNK])
                continue
            source = plan["src"][start : start + PRODUCER_CHUNK]
            chunk = x[source.clamp_min(0)] * (source >= 0).unsqueeze(1).to(x.dtype)
            if gate is not None:
                extra["coarse_gate"].view(padded_tokens, heads * head_dim)[
                    start : start + chunk.shape[0]
                ] = gate(chunk)
            yield attn.qkv_proj(chunk)

    output, kmean, vscale = kitchen.sol_attn_chunked(
        chunks,
        padded_tokens,
        heads,
        freqs,
        (q_weight, k_weight),
        kmean=None if first else pooled[0],
        vscale=None if first else pooled[1],
        tau=patch.tau,
        topk_ratio=patch.topk_ratio,
        token_aug=patch.extra_tokens,
        sink_blocks=list(sinks),
        sink_q=list(sink_queries),
        rope_eps=attn.q_norm.eps,
        **extra,
    )
    pooled[0].copy_(kmean)
    pooled[1].copy_(vscale)
    patch.pooled[key] = pooled
    mode = (
        f"VSA tiles ({padded_tokens} padded rows, {sinks[1]} prefix tiles)"
        if plan is not None
        else f"sinks {sinks}/{sink_queries}"
    )
    patch.log_once(("producer", padded_tokens), f"sparse producer path: {tokens} tokens, {mode}")
    output = output.view(padded_tokens, heads * head_dim)
    if plan is not None:
        output = output[plan["inv"]]
    return attn.out_proj(output)


def make_h3_block_patch(block: Any, block_index: int, patch: SparseAttnPatch) -> Callable[..., Any]:
    def attention(
        hidden: Any, rope_freqs: Any = None, transformer_options: dict[str, Any] | None = None
    ) -> Any:
        return h3_sparse_attention(
            block.attn,
            hidden,
            rope_freqs,
            {} if transformer_options is None else transformer_options,
            patch,
            block_index,
        )

    def block_patch(args: dict[str, Any], extra: Mapping[str, Any]) -> Any:
        if h3_eligible(
            block.attn,
            args["img"],
            args["rope_freqs"],
            args["transformer_options"],
            patch,
            block_index,
        ):
            args = {**args, "attention": attention}
        return extra["original_block"](args)

    return block_patch


def apply_block_sparse_attention(
    model: Any,
    *,
    tau: float,
    topk_ratio: float,
    vsa: bool,
    start_percent: float,
    end_percent: float,
    min_tokens: int,
    dense_blocks: set[int],
    sink_conditioning: str,
    extra_tokens: int,
    verbose: bool,
) -> Any:
    initialize_inference_process()
    sampling = model.get_model_object("model_sampling")
    if vsa and extra_tokens:
        logging.info("VSA: extra_tokens ignored (the trained sparse pattern is the target)")
        extra_tokens = 0
    patch = SparseAttnPatch(
        tau,
        topk_ratio,
        vsa,
        float(sampling.percent_to_sigma(start_percent)),
        float(sampling.percent_to_sigma(end_percent)),
        min_tokens,
        dense_blocks,
        sink_conditioning,
        extra_tokens,
        verbose,
    )
    result = model.clone()
    install_override(patch, result.model_options["transformer_options"])
    extension = importlib.import_module("dinkster_inference.patcher_extension")

    def prepare_callback(_patcher: Any, _timestep: Any, options: Mapping[str, Any]) -> None:
        transformer_options = options["transformer_options"]
        if not isinstance(transformer_options, dict):
            raise TypeError("transformer_options must be a dictionary")
        install_override(patch, cast(dict[str, Any], transformer_options))

    def cleanup_callback(_patcher: Any) -> None:
        patch.reset()

    result.add_callback_with_key(
        extension.CallbacksMP.ON_PREPARE_STATE,
        "block_sparse_attention",
        prepare_callback,
    )
    result.add_callback_with_key(
        extension.CallbacksMP.ON_CLEANUP, "block_sparse_attention", cleanup_callback
    )
    diffusion = model.get_model_object("diffusion_model")
    minimax = importlib.import_module("dinkster_inference.ldm.minimax.model")
    if isinstance(diffusion, minimax.MiniMaxH3Model):
        for index, block in enumerate(diffusion.blocks):
            result.set_model_patch_replace(
                make_h3_block_patch(block, index, patch), "dit", "double_block", index
            )
        if vsa and diffusion.blocks[0].attn.to_gate_compress is None:
            logging.warning("VSA: no to_gate_compress layers; coarse path unavailable")
    elif vsa:
        raise ValueError("VSA selection needs a MiniMax-H3 model")
    return result


def _number(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
    *,
    advanced: bool = False,
) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=default,
        advanced=advanced,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _method_input(name: str, default: float, doc: str) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=default,
        doc=doc,
        widget=NumberWidget(
            min=0.0 if name == "tau" else 0.5,
            max=4.0 if name == "tau" else 95.0,
            step=0.05 if name == "tau" else 0.5,
        ),
    )


class BlockSparseAttention(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BlockSparseAttention",
            display_name="Model Sparse Attention",
            category="model/patch",
            description=(
                "Applies block-sparse attention to eligible layers. "
                "The dense model attention backend is used outside the schedule, "
                "for dense blocks, and below min_tokens."
            ),
            search_terms=("Block Sparse Attention", "sol-attn", "SLA", "FastVideo"),
            inputs=(
                InputSpec("model", _MODEL, display_name="model", doc="The model to patch."),
                _number("start_percent", 0.2, 0.0, 1.0, 0.01),
                _number("end_percent", 1.0, 0.0, 1.0, 0.01),
                InputSpec(
                    "dense_blocks",
                    _STRING,
                    default="",
                    advanced=True,
                    widget=StringWidget(multiline=False),
                    doc="Transformer blocks that always run dense, e.g. '0, 1, 47-49'.",
                ),
                InputSpec(
                    "min_tokens",
                    _INT,
                    default=12288,
                    advanced=True,
                    doc="Sequences shorter than this stay dense.",
                    widget=NumberWidget(min=0, max=1_048_576, step=512),
                ),
                InputSpec(
                    "extra_tokens",
                    _INT,
                    default=256,
                    advanced=True,
                    doc="Extra top-scoring tokens beyond selected blocks. Ignored for VSA.",
                    widget=NumberWidget(min=0, max=256, step=64),
                ),
                InputSpec(
                    "sink_conditioning",
                    _COMBO,
                    default="exact_kv_and_rows",
                    advanced=True,
                    doc="MiniMax-H3 conditioning rows kept exact.",
                    widget=ComboWidget(
                        tuple(
                            ComboOption(value) for value in ("exact_kv", "exact_kv_and_rows", "off")
                        )
                    ),
                ),
                InputSpec(
                    "verbose",
                    _BOOLEAN,
                    default=False,
                    advanced=True,
                    doc="Log sparse or dense routing.",
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "selection",
                    display_name="method",
                    doc="Method used to choose key blocks for exact token-level attention.",
                    options=(
                        DynamicComboOption(
                            "sol-attn",
                            (
                                _method_input(
                                    "tau", 1.3, "Adaptive threshold in score-distribution sigmas."
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "sla",
                            (
                                _method_input(
                                    "keep_percent", 10.0, "Percent of key blocks to keep exactly."
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "vsa",
                            (
                                _method_input(
                                    "keep_percent", 10.0, "Percent of video cubes to keep exactly."
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            input_order=(
                "model",
                "selection",
                "start_percent",
                "end_percent",
                "dense_blocks",
                "min_tokens",
                "extra_tokens",
                "sink_conditioning",
                "verbose",
            ),
            outputs=(OutputSpec("MODEL", _MODEL, display_name="model"),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        selection: str | Mapping[str, Any],
        start_percent: float = 0.2,
        end_percent: float = 1.0,
        dense_blocks: str = "",
        min_tokens: int = 12288,
        extra_tokens: int = 0,
        sink_conditioning: str = "exact_kv_and_rows",
        verbose: bool = False,
        **values: Any,
    ) -> Mapping[str, object]:
        if isinstance(selection, Mapping):
            selected = selection
        else:
            selected = {
                "selection": selection,
                **{
                    input_id.removeprefix("selection."): value for input_id, value in values.items()
                },
            }
        mode = selected["selection"]
        return cls.outputs(
            MODEL=apply_block_sparse_attention(
                model,
                tau=float(selected.get("tau", 1.3)),
                topk_ratio=0.0 if mode == "sol-attn" else float(selected["keep_percent"]) / 100.0,
                vsa=mode == "vsa",
                start_percent=start_percent,
                end_percent=end_percent,
                min_tokens=min_tokens,
                dense_blocks=parse_block_list(dense_blocks),
                sink_conditioning=sink_conditioning,
                extra_tokens=extra_tokens,
                verbose=verbose,
            )
        )


SPARSE_ATTENTION_NODES = (BlockSparseAttention,)
