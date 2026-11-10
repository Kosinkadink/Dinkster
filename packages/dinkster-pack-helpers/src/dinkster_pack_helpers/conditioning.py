"""Conditioning metadata transforms shared by node packs."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any


def conditioning_set_values(
    conditioning: Sequence[Sequence[Any]], values: Mapping[str, object], *, append: bool = False
) -> list[list[Any]]:
    result: list[list[Any]] = []
    for entry in conditioning:
        metadata = entry[1].copy()
        for key, value in values.items():
            old_value = metadata.get(key)
            metadata[key] = old_value + value if append and old_value is not None else value
        result.append([entry[0], metadata])
    return result


def conditioning_set_values_with_timestep_range(
    conditioning: Sequence[Sequence[Any]],
    values: Mapping[str, object],
    start_percent: float = 0.0,
    end_percent: float = 1.0,
) -> Sequence[Sequence[Any]]:
    if start_percent > end_percent:
        logging.warning(
            "start_percent (%s) must be <= end_percent (%s)", start_percent, end_percent
        )
        return conditioning
    result: list[Sequence[Any]] = []
    # The sampler uses strict boundaries; each timestep must activate only one entry.
    epsilon = 1e-5
    for entry in conditioning:
        cond_start = entry[1].get("start_percent", 0.0)
        cond_end = entry[1].get("end_percent", 1.0)
        intersect_start = max(start_percent, cond_start)
        intersect_end = min(end_percent, cond_end)
        if intersect_start >= intersect_end:
            result.append(entry)
            continue
        if intersect_start > cond_start:
            result.extend(
                conditioning_set_values(
                    [entry], {"start_percent": cond_start, "end_percent": intersect_start - epsilon}
                )
            )
        result.extend(
            conditioning_set_values(
                [entry], {**values, "start_percent": intersect_start, "end_percent": intersect_end}
            )
        )
        if intersect_end < cond_end:
            result.extend(
                conditioning_set_values(
                    [entry], {"start_percent": intersect_end + epsilon, "end_percent": cond_end}
                )
            )
    return result
