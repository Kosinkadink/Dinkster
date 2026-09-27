## Dtypes and quantization

SD1.5 and MiniMax H3 use the storage, compute dtype, quantization, attention,
and residency behavior supplied by the pinned `dinkster_comfy` runtime.
Dinkster does not maintain a second implementation of those mechanisms.

Other model-family dtype and quantization combinations are unsupported.
