# SD1.5 ControlNet native parity

Status: understood cross-process fp16 kernel variance on NVIDIA GeForce RTX
5060 Ti. No sampling tolerance or final-hash allowlist is used.

The comparison uses ComfyUI commit
`947c2749dd04c51ef0e21b069544d8b0b4f9b411` and Dinkster native with the same
SD1.5 checkpoint, `control_v11p_sd15_canny` weights, 256x256 Canny hint, seed
239, Euler sampler, normal scheduler, 16 steps, CFG 7, and constant strength.
The artifact URLs, sizes, hashes, prompts, and output hashes are recorded in
`packages/dinkster-inference-torch/tests/goldens/sd15_controlnet_acceptance.json`.

One Dinkster execution using the saved ComfyUI conditioning matched the ComfyUI
latent `cc8004f4a1b1ec8f7882d34f82f1ef4da69e80a0ee9204b90f88963a964c89a8`
bit for bit. Pinned ComfyUI also produced a different result in one identical
cross-process execution: 1 of 6 reference executions was anomalous, with
maximum/mean latent differences of 0.5831151/0.04189986 from `cc8004f4`.
Conditioning, the first ControlNet input, and all first-step residuals were bit
exact; divergence began after the first model evaluation. Two instrumented
ComfyUI regenerations then matched at all 16 current-latent and denoised-latent
seams and reproduced `cc8004f4`.

Later Dinkster processes reproduced the alternate kernel path. ControlNet inputs
were bit exact. The deeper fp16 residuals differed by at most 0.015625 with a
maximum observed mean absolute difference of 0.0014066696166992188. The
maximum is one fp16 ulp at magnitude 16-32, and the mean is ulp-scale near
magnitude 1. Diffusion amplifies this pre-sampling reduction-order difference,
so final hashes are not a stable cross-process comparison on this GPU.
The reference outlier is classified as transient rather than an alternate code
path because it was bit exact through the first evaluation and did not recur
in either instrumented regeneration.

A bounded rerun with `torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` did not collapse the two kernel outcomes:
Dinkster produced `197982109d4fef8b642049ff45cd8c74fff72e5b787e64c3f07a8c15b0d76976`
and ComfyUI produced `cc8004f4`. The deterministic acceptance assertions are
therefore intra-run: gain zero equals no control bit for bit, a constant gain
schedule equals scalar strength bit for bit, and a non-uniform keyframed
schedule visibly and numerically changes the image.
