from __future__ import annotations

import pytest
from dinkster_acceptance.model_sampling_flux import run_acceptance


def test_cpu_acceptance_executes_all_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINKSTER_ACCEPTANCE_COMMIT", "1" * 40)
    monkeypatch.setenv("DINKSTER_ACCEPTANCE_DEVICE", "cpu")
    report = run_acceptance()
    assert report["commit"] == "1" * 40
    assert report["device"] == "cpu"
    assert report["verdicts"] == {
        "overlay_shift_matches_comfyui": True,
        "schedule_matches_comfyui": True,
        "pre_offset_schedule_matches_comfyui": True,
        "offset_schedule_matches_comfyui": True,
        "brownian_bounds_match_comfyui": True,
        "brownian_draws_match_comfyui": True,
        "patched_matches_control": True,
        "patched_is_deterministic": True,
        "control_schedule_matches": True,
        "finite": True,
    }
