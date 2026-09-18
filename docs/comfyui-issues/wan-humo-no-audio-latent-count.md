# Wan HuMo no-audio conditioning has one extra frame

Status: found 2026-08-26; not reported upstream; fixed in Dinkster.

Baseline: ComfyUI `b78cec879b9460d5cb25228a83a942fb78d2cd24`.

## Symptom

`WanHuMoImageToVideo` creates one more no-audio embedding frame than the
target video latent. Because the node always supplies a reference latent, the
HuMo model then appends another zero audio frame for that reference. The audio
sequence is therefore one frame longer than the combined target and reference
video sequence.

## Root cause and reproduction

For the default length of 97, the target latent has 25 frames:

```python
latent_t = ((97 - 1) // 4) + 1
assert latent_t == 25
```

The no-audio path creates `[batch, latent_t + 1, 8, 5, 1280]`, while
`HumoWanModel.forward_orig` independently appends one frame whenever a
reference latent is present. The resulting audio sequence has 27 frames for
26 video frames.

## Suggested upstream fix

Create no-audio conditioning with exactly `latent_t` frames. Leave reference
padding to `HumoWanModel.forward_orig`, as the real-audio path already does.

Dinkster uses exactly the target latent count and appends reference-audio zeros
inside the model.
