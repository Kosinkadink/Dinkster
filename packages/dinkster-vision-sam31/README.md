# dinkster-vision-sam31

`dinkster-vision-sam31` is the separately installable SAM 3.1 provider for
`dinkster.detection.detect`, `dinkster.detection.segment`,
`dinkster.detection.segment_text`, and `dinkster.detection.track`.
It runs in an isolated CPU pack worker. Detection finds the objects named by a
comma-separated or literal text prompt and returns scored boxes with
source-sized binary masks. `max_results` limits each frame independently;
default count mode uses -1 for unlimited results, while slice-stop mode applies
Python slice-stop semantics for source-compatible imports.
Text-prompted segmentation exposes those detections and their masks in one
operation, and an empty prompt returns empty outputs. Box segmentation uses
each detection box as a prompt and preserves the detection metadata while
attaching a source-sized binary mask. Tracking uses detections or masks on the
first frame and returns one source-sized BHW mask for each object across the
complete frame batch plus their combined union.

The combined checkpoint is read only through `dinkster_api.v1.declared_asset`;
execution never downloads it. Dinkster preflight presents the exact artifact need
and acquires it only with user consent. The model is distributed under the SAM
License. Its pinned 1,745,546,848-byte artifact has SHA-256
`9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`
and BLAKE3
`1c8d5762dbaf238bc9a2f10de07e0c476d1b68e75453566feb8dc9bcf2cb41c5`.
Use and redistribution remain subject to the [bundled SAM License](SAM_LICENSE),
including its trade-control restrictions. The bundled agreement is copied
verbatim from the model repository's [immutable artifact revision](https://huggingface.co/Comfy-Org/sam3.1/blob/f38cd62b71494b53ac2b56ca36e24f3c8d565581/LICENSE).
Text detection uses the OpenAI CLIP byte-pair
[vocabulary](https://raw.githubusercontent.com/openai/CLIP/3bee28119e6b28e75b82b811b87b56935314e6a5/clip/bpe_simple_vocab_16e6.txt.gz)
from immutable commit `3bee28119e6b28e75b82b811b87b56935314e6a5`.
The vocabulary and its [bundled MIT license](CLIP_LICENSE), sourced from the
same immutable commit, are packaged with the provider, so execution does not
require network access.

Images are resized to 1008 by 1008 with bilinear interpolation. The provider
runs SAM 3.1's text encoder and detector once per prompt phrase, then refines
each coarse detector mask with the interactive decoder. Box-prompt segmentation
performs the same second decoder refinement as ComfyUI, restores logits to the
source size, and thresholds them at zero. Video frames use SAM 3.1's bicubic
tracking preprocessing, 16-object multiplex decoder, seven-frame memory, and
predicted-IoU mask selection. Invalid first-frame boxes produce zero masks.

Architecture, preprocessing, refinement, and output conversion match ComfyUI
commit `8dc3f3f2094121c0a013e21d89136ebc331d2974`, which includes the native SAM
3.1 implementation and its large-input fix. Exact-vector provenance is recorded
by `tools/gen_sam31_detect_golden.py`, `tools/gen_sam31_golden.py`, and
`tools/gen_sam31_track_golden.py`. Exact vector parity uses one torch intra-op
thread. The detection golden SHA-256 is
`3a686eed763d64873e81a3aaacc101a3ae3f2e68564d1626bdffaf40eb8a2345`; the
segmentation golden SHA-256 is
`67610cf077bdbe65d7fff9aace38a572b4e280990ce43f547c5a98887c44ac4a`; the
tracking golden SHA-256 is
`1d7ea415a504476475d634cc7a308a4bef0a11f4997fad082c16703b6c2e5521`.
Ordinary provider execution retains the worker process's thread settings.
