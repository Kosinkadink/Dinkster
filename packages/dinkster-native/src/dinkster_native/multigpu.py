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


def _empty_like(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value)
    nested = importlib.import_module("dinkster_inference.nested_tensor").NestedTensor
    if type(value) is nested:
        return nested(tuple(torch.empty_like(tensor) for tensor in value.unbind()))
    raise TypeError("distributed output must be a tensor or NestedTensor")


def _all_reduce(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        torch.distributed.all_reduce(value)
        return
    for tensor in value.unbind():
        torch.distributed.all_reduce(tensor)


def _broadcast(value: Any, owner: int) -> None:
    if isinstance(value, torch.Tensor):
        torch.distributed.broadcast(value, owner)
        return
    for tensor in value.unbind():
        torch.distributed.broadcast(tensor, owner)


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


def _window_owner(index: int, count: int, world_size: int) -> int:
    width, remainder = divmod(count, world_size)
    boundary = (width + 1) * remainder
    if index < boundary:
        return index // (width + 1)
    if width == 0:
        return index
    return remainder + (index - boundary) // width


def _window_wrapper(
    executor: Any,
    evaluate: Any,
    model: Any,
    conds: list[Any],
    template: Any,
    timestep: Any,
    options: dict[str, Any],
    packed: bool,
) -> list[list[Any]]:
    config = attention._ensure_process_group()  # pyright: ignore[reportPrivateUsage]
    window_executor = executor.class_obj
    count = len(window_executor.plan.joint_windows)
    local: dict[int, list[Any]] = {}
    error: BaseException | None = None
    try:
        for index in range(count):
            if _window_owner(index, count, config.world_size) == config.rank:
                local[index] = window_executor.evaluate_window(
                    index, evaluate, model, conds, template, timestep, options, packed
                )
    except BaseException as exc:
        error = exc
    _raise_group_failure(error, _first_tensor(template).device)

    outputs: list[list[Any]] = [[None] * count for _ in conds]
    for index in range(count):
        owner = _window_owner(index, count, config.world_size)
        expected = window_executor.window_latent(index, template)
        for condition_index in range(len(conds)):
            value = local[index][condition_index] if owner == config.rank else _empty_like(expected)
            _broadcast(value, owner)
            outputs[condition_index][index] = value
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
        if config.world_size != 2:
            raise RuntimeError("Ulysses sequence mode supports exactly two ranks")
        sequence = int(args["rope_freqs"].shape[1])
        width = (sequence + config.world_size - 1) // config.world_size
        padded_sequence = width * config.world_size
        start = config.rank * width
        stop = start + width
        local_args = dict(args)
        if self.index == 0:
            hidden = args["img"]
            if padded_sequence != sequence:
                hidden = torch.cat(
                    (hidden, hidden.new_zeros((padded_sequence - sequence, hidden.shape[1]))),
                    dim=0,
                )
            local_args["img"] = hidden[start:stop]
        elif int(args["img"].shape[0]) != width:
            raise RuntimeError("H3 sequence shard changed width between DiT blocks")
        rope = args["rope_freqs"]
        if padded_sequence != sequence:
            rope = torch.cat(
                (
                    rope,
                    rope.new_zeros((rope.shape[0], padded_sequence - sequence, *rope.shape[2:])),
                ),
                dim=1,
            )
        local_args["rope_freqs"] = rope[:, start:stop]
        local_args["mod_segments"] = _local_modulation_segments(args["mod_segments"], start, stop)
        transformer_options = args["transformer_options"]
        transformer_options["dinkster_sequence_sharded"] = True
        transformer_options["dinkster_sequence_valid"] = sequence
        try:
            output = extra["original_block"](local_args)["img"]
        finally:
            transformer_options.pop("dinkster_sequence_sharded", None)
            transformer_options.pop("dinkster_sequence_valid", None)
        if self.index + 1 == self.count:
            gathered = [torch.empty_like(output) for _ in range(config.world_size)]
            torch.distributed.all_gather(gathered, output.contiguous())
            output = torch.cat(gathered, dim=0)[:sequence]
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
    if config.mode == "window":
        model.add_wrapper_with_key(
            patcher_extension.WrappersMP.WINDOW_EXECUTE,
            "dinkster_window",
            _window_wrapper,
        )
        return
    if config.world_size != 2:
        raise RuntimeError("Ulysses sequence mode supports exactly two ranks")
    diffusion_model = model.model.diffusion_model
    blocks = diffusion_model.blocks
    if not isinstance(blocks, torch.nn.ModuleList) or not blocks:
        raise RuntimeError("Ulysses sequence mode requires an H3 DiT block list")
    for index in range(len(blocks)):
        model.set_model_patch_replace(
            _SequenceBlockPatch(index, len(blocks)), "dit", "double_block", index
        )


__all__ = ["configure_distributed_model"]
