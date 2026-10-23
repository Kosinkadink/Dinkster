"""Distributed execution adapters for the fork sampling engine."""

from __future__ import annotations

import importlib
from typing import Any, cast

from . import attention

torch = attention.torch


def _first_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value
    nested = importlib.import_module("dinkster_inference.nested_tensor").NestedTensor
    if type(value) is nested:
        return value.unbind()[0]
    raise TypeError("distributed output must be a tensor or NestedTensor")


def _all_reduce(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        torch.distributed.all_reduce(value)
        return
    for tensor in value.unbind():
        torch.distributed.all_reduce(tensor)


def _raise_group_failure(error: BaseException | None, device: Any) -> None:
    failed = torch.tensor(int(error is not None), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if error is not None:
        raise error
    if bool(failed.item()):
        raise RuntimeError("peer distributed execution rank failed")


def _guidance_wrapper(
    executor: Any, model: Any, conds: list[Any], x: Any, timestep: Any, options: dict[str, Any]
) -> list[Any]:
    config = attention._ensure_process_group()  # pyright: ignore[reportPrivateUsage]
    active = [index for index, conditioning in enumerate(conds) if conditioning]
    owners = {index: lane % config.world_size for lane, index in enumerate(active)}
    local_conds = [
        conditioning if owners.get(index) == config.rank else None
        for index, conditioning in enumerate(conds)
    ]
    error: BaseException | None = None
    outputs: list[Any] = []
    try:
        outputs = executor(model, local_conds, x, timestep, options)
    except BaseException as exc:
        error = exc
    _raise_group_failure(error, _first_tensor(x).device)
    for output in outputs:
        _all_reduce(output)
    return outputs


def _local_modulation_segments(
    segments: list[tuple[int, int, Any]], start: int, stop: int
) -> list[tuple[int, int, Any]]:
    local: list[tuple[int, int, Any]] = []
    for segment_start, segment_stop, row in segments:
        overlap_start = max(segment_start, start)
        overlap_stop = min(segment_stop, stop)
        if overlap_start >= overlap_stop:
            continue
        if isinstance(row, torch.Tensor):
            row = row[overlap_start - segment_start : overlap_stop - segment_start]
        local.append((overlap_start - start, overlap_stop - start, row))
    return local


class _SequenceBlockPatch:
    def __init__(self, index: int, count: int) -> None:
        self.index = index
        self.count = count

    def __call__(self, args: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        config = attention._ensure_process_group()  # pyright: ignore[reportPrivateUsage]
        sequence = int(args["rope_freqs"].shape[1])
        base_width, wider_ranks = divmod(sequence, config.world_size)
        sequence_widths = tuple(
            base_width + int(rank < wider_ranks) for rank in range(config.world_size)
        )
        width = sequence_widths[config.rank]
        start = sum(sequence_widths[: config.rank])
        stop = start + width
        local_args = dict(args)
        if self.index == 0:
            local_args["img"] = args["img"][start:stop]
        elif int(args["img"].shape[0]) != width:
            raise RuntimeError("H3 sequence shard changed width between DiT blocks")
        local_args["rope_freqs"] = args["rope_freqs"][:, start:stop]
        local_args["mod_segments"] = _local_modulation_segments(args["mod_segments"], start, stop)
        transformer_options = args["transformer_options"]
        transformer_options["dinkster_sequence_sharded"] = True
        transformer_options["dinkster_sequence_widths"] = sequence_widths
        try:
            output = extra["original_block"](local_args)["img"]
        finally:
            transformer_options.pop("dinkster_sequence_sharded", None)
            transformer_options.pop("dinkster_sequence_widths", None)
        if self.index + 1 == self.count:
            gathered = [
                output.new_empty((rank_width, output.shape[1])) for rank_width in sequence_widths
            ]
            torch.distributed.all_gather(
                gathered,
                output.contiguous(),
                group=attention._sequence_group(config),  # pyright: ignore[reportPrivateUsage]
            )
            output = torch.cat(gathered, dim=0)
        return {"img": output}


def configure_distributed_model(model: Any) -> None:
    config = attention._distributed_config()  # pyright: ignore[reportPrivateUsage]
    patcher_extension = cast("Any", importlib.import_module("dinkster_inference.patcher_extension"))
    if config.mode == "guidance":
        model.add_wrapper_with_key(
            patcher_extension.WrappersMP.CALC_COND_BATCH,
            "dinkster_guidance",
            _guidance_wrapper,
        )
        return
    diffusion_model = model.model.diffusion_model
    blocks = diffusion_model.blocks
    if not isinstance(blocks, torch.nn.ModuleList) or not blocks:
        raise RuntimeError("Ulysses sequence mode requires an H3 DiT block list")
    for index in range(len(blocks)):
        model.set_model_patch_replace(
            _SequenceBlockPatch(index, len(blocks)), "dit", "double_block", index
        )


__all__ = ["configure_distributed_model"]
