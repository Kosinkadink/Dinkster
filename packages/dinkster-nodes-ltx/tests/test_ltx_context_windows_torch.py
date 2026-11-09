"""LTX frame-to-latent context windows preserve model state."""

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_ltx.context_windows import LTXVContextWindows


@pytest.mark.parametrize(
    ("length", "overlap", "retain", "expected"),
    ((145, 40, True, (19, 5, [0])), (8, 7, False, (1, 0, []))),
)
def test_ltx_context_windows_clone_and_configure_model_patcher(
    length: int, overlap: int, retain: bool, expected: tuple[int, int, list[int]]
) -> None:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    source = ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))
    source.model_options["transformer_options"]["preserved"] = {"value": 17}
    result = LTXVContextWindows.execute(
        model=source,
        context_length=length,
        context_overlap=overlap,
        context_schedule="looped_uniform",
        context_stride=2,
        closed_loop=True,
        fuse_method="flat",
        freenoise=True,
        retain_first_frame=retain,
        split_conds_to_windows=True,
    )["MODEL"]
    handler = result.model_options["context_handler"]
    assert result is not source
    assert "context_handler" not in source.model_options
    assert result.model_options["transformer_options"] == {"preserved": {"value": 17}}
    assert (
        handler.context_length,
        handler.context_overlap,
        handler.cond_retain_index_list,
    ) == expected
    assert handler.latent_retain_index_list == expected[2]
    assert handler.dim == 2
    assert handler.freenoise is True
    assert handler.split_conds_to_windows is True
    assert handler.causal_window_fix is True
