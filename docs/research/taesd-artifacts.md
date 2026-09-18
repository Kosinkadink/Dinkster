# TAESD artifact provenance

Official upstream weights were fetched from madebyollin/taesd on 2026-07-29.
Each URL is `https://raw.githubusercontent.com/madebyollin/taesd/main/<filename>`.

| Filename | sha256 |
| --- | --- |
| `taesd_decoder.pth` | `02873377c3f4659cd9f9adb2f718dcb434ba4c7bba3af3ee7bb95cbdabe2d3cf` |
| `taesd_encoder.pth` | `15bc6128f0ac51c673d3427216082ebfa62402ffc89763af74edfe259b32d49d` |
| `taesdxl_decoder.pth` | `a3956b8a7a763f251c7357aad6375dacdb6d971121c73e97a3069fc33a94fe1e` |
| `taesdxl_encoder.pth` | `a5648de089aba6b641e505c0f132b2dc7ff70ab6070c31be5e624e807e711c5e` |

Architecture and facade semantics are pinned to ComfyUI commit
`f4b99bc62389af315013dda85f24f2bbd262b686` (`comfy/taesd/taesd.py` and
`comfy/sd.py`). Executed-reference goldens are generated separately from
ComfyUI commit `947c2749dd04c51ef0e21b069544d8b0b4f9b411`, following Dinkster's
numerical-parity pin. The SD1.5/SDXL TAESD architecture and facade math are
identical between those commits; the later checkout adds only out-of-scope
Flux2 TAE support and type annotations in `comfy/taesd/taesd.py`.
