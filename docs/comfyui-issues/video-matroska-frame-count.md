# Matroska frame count stops after one frame

Status: found; not reported upstream.

Reference: ComfyUI commit
[`15eb748b3ec5`](https://github.com/Comfy-Org/ComfyUI/commit/15eb748b3ec5f8a0a2d470b7fb280e2d7579f916),
`comfy_api/latest/_input_impl/video_types.py`, `VideoFromFile.get_frame_count`,
lines 367-422.

An untrimmed ten-frame FFV1 MKV and six-frame variable-rate H.264 MKV both
return 1 from the upstream getter. These streams expose `frames == 0` and
`duration is None`, although the container has a duration. The metadata
estimate requires stream duration and does not use container duration.
The fallback starts its count at 1 and calculates `end_pts == start_pts`
for the untrimmed duration-0 sentinel. It stops at the next frame.

Reproduce with the `alpha` and `vfr_mkv` fixtures and the real upstream
getter in `tests/test_video_vhs_live.py`. Set `DINKSTER_COMFYUI_ROOT`,
`DINKSTER_EXECUTION_PYTHON`, and `DINKSTER_VHS_ROOT` to the pinned live environment
described in that test, then run it with pytest. The receipt includes
upstream, wrapper, and VHS source getters independently of decoded counts.

An upstream fix should use container duration as a fallback for metadata
estimation and represent duration 0 as an unbounded end in decode counting.
An exact fallback must count decoded frames, not packets. Container duration
times frame rate is only an estimate for variable-rate sources.

Dinkster uses container duration when stream duration is unavailable and marks
the resulting count as estimated. Its compatibility wrapper reports these
probe facts without inheriting the faulty upstream fallback. Disassemble
reports actual decoded frame counts. ComfyUI is not the alpha reference;
the fixture's RGBA pixels are checked independently.
