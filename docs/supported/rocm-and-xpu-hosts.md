# AMD ROCm and Intel XPU hosts

Dinkster retains accelerator selection and isolated worker support for ROCm and
XPU environments. Model-family sampling on these backends is currently
unsupported: the previous evidence covered the retired inference runtime, not
the `dinkster_inference` runtime now used by Dinkster.

ROCm and XPU must use separate Python environments because they install
different Torch builds. Mixed-vendor execution in one worker is unsupported.
