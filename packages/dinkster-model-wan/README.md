# dinkster-model-wan

First-party native nodes for Wan model operations. The pack owns Wan 2.1 Uni3C
model-patch loading and application plus Wan 2.1 Animate2, Wan 2.1
SCAIL/SCAIL2, Wan 2.1 HuMo, Wan 2.1 InfiniteTalk, and Wan 2.2 Animate
conditioning while requiring the universal generation schema capability
without importing its owning package.

The conditioning nodes consume distinct public `dinkster.model` and `dinkster.vae`
resources, canonical `dinkster.conditioning`, and media-owned inputs to produce a
single-stream `dinkster.latent`. They derive reference or pose vision conditioning
when the loaded runtime includes CLIP vision and stage codec and vision work
through their public resource leases. Animate2 also supports a dedicated pose
prompt, pose application window, pose/reference strengths, and one-frame
continuation.
SCAIL/SCAIL2 supports batched reference images, half-resolution pose video,
colored identity masks, animation and replacement layouts, pose scheduling,
and SCAIL2 continuation anchors. It uses the existing `dinkster.image` mask
surface; colored-mask production remains independent of model conditioning.

`Wan HuMo Image to Video` prepares the exact 17B HuMo conditioning contract
from positive and negative Wan text lanes, an optional reference image, and
optional Whisper Large v3 audio features. It groups all 33 Whisper encoder
layers into the published 25 Hz eight-frame windows, stages reference encoding
through the Wan VAE lease, and emits a canonical target latent. `Load Wan Audio
Encoder` and `Encode Wan Audio` load and execute HuMo Whisper, InfiniteTalk
Chinese base Wav2Vec2, or Wan 2.2 S2V Wav2Vec2 according to exact artifact
metadata.

`Wan InfiniteTalk Image to Video` applies the official MultiTalk patch to the
exact base Wan 2.1 I2V 14B model. It converts Wav2Vec2 Chinese base features to
25 Hz audio context, supports one or two speakers with spatial masks, and can
continue from a previous video while preserving the motion overlap.

`Load Wan 2.1 Uni3C` loads the exact published 20-block Uni3C patch. `Apply Wan
2.1 Uni3C` accepts an RGB render video and a Wan 2.1 VAE, resizes and encodes the
render for the sampling latent, and applies one shared control lane across CFG
lanes. It supports only exact base Wan 2.1 14B T2V and I2V models and refuses
other interventions, including Phantom conditioning.
