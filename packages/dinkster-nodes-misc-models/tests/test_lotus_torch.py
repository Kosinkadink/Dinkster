from __future__ import annotations

import hashlib
from typing import Any

import torch
from dinkster_api.v1 import (
    materialize_latent,
)
from dinkster_misc_models.lotus import LotusConditioning

_EMBEDDING_HASH = "a7ba15b03c89931a7764a7d4b398201b322e00524d03efaac172b48dc637397d"


def _check(conditioning: Any) -> str:
    assert len(conditioning) == 1 and conditioning[0][1] == {}
    embedding = materialize_latent(conditioning[0][0])
    assert embedding.dtype == torch.float32 and embedding.shape == (1, 2, 1024)
    assert embedding.device.type == "cpu"
    digest = hashlib.sha256(embedding.numpy().tobytes()).hexdigest()
    assert digest == _EMBEDDING_HASH
    return digest


def test_lotus_returns_exact_frozen_embedding_with_independent_storage() -> None:
    first = LotusConditioning.execute()["conditioning"]
    assert _check(first) == _EMBEDDING_HASH
    with torch.inference_mode():
        first[0][0].fill_(12)
    first[0][1]["changed"] = True
    second = LotusConditioning.execute()["conditioning"]
    assert _check(second) == _EMBEDDING_HASH
    assert first[0][0].data_ptr() != second[0][0].data_ptr()
    assert first[0][1] is not second[0][1]
