# SD1.5 full T2I Adapter comparison

Reference: ComfyUI origin/master commit `de6b062f`,
`comfy/t2i_adapter/adapter.py`, full non-XL `Adapter`.

Dinkster matches PixelUnshuffle 8, channels 320/640/1280/1280, two residual
blocks per level, checkpoint-derived kernel size 1, skip connections, and
average-pool downsampling. Its four stage outputs map to SD1.5 down residual
sites 2, 5, 8, and 11; all other down residuals and middle are typed zeros.

The real-load fixture is TencentARC/T2I-Adapter revision
`003997adaae088a8311e25f6d395e821b5398a11`, file
`t2iadapter_canny_sd15v2.pth`, 308015219 bytes, SHA-256
`a5ec162813b3997a925d39dc546ff696c42f877fec55ef2e6afa3a4aa642018e`,
and BLAKE3 `0c483d0094b18c9f542ab7586869e52c6d410c48941513bcd0b156d63c5ae5ac`.
