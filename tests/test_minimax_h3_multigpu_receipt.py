from __future__ import annotations

from typing import Any, cast

import pytest

torch = cast("Any", pytest.importorskip("torch"))

from tools.minimax_h3_multigpu_receipt import _sparse_quality_oracle  # noqa: E402


def test_sparse_quality_oracle_reports_dense_relative_error_by_role(tmp_path) -> None:
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(
        {
            "audio": torch.tensor([3.0, 4.0]),
            "video": torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
        },
        reference_path,
    )
    torch.save(
        {
            "audio": torch.tensor([0.0, 4.0]),
            "video": torch.tensor([[2.0, -2.0], [3.0, -6.0]]),
        },
        candidate_path,
    )

    oracle = _sparse_quality_oracle(reference_path, candidate_path)

    assert oracle["baseline"].startswith("same-session dense SDPA")
    audio = oracle["metrics_by_role"]["audio"]
    assert audio["shape"] == [2]
    assert audio["max_abs"] == 3.0
    assert audio["rmse"] == pytest.approx(3.0 / 2.0**0.5)
    assert audio["relative_rmse"] == pytest.approx(0.6)
    video = oracle["metrics_by_role"]["video"]
    assert video["shape"] == [2, 2]
    assert video["max_abs"] == 2.0
    assert video["rmse"] == pytest.approx(5.0**0.5 / 2.0)
    assert video["cosine_similarity"] == pytest.approx(39.0 / (30.0 * 53.0) ** 0.5)
