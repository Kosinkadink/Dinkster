# Qwen3-0.6B generation comparison

This benchmark compares raw-prompt greedy generation on one pinned
Qwen3-0.6B checkpoint. Model load time is excluded. Each backend runs in a
separate process so weights do not overlap in VRAM.

Pinned inputs:

- Safetensors source: [circlestone-labs/Anima at e26179e](https://huggingface.co/circlestone-labs/Anima/resolve/e26179e4b23bcb3a9e91b4ad2961a76ab9644d43/split_files/text_encoders/qwen_3_06b_base.safetensors)
- Safetensors bytes: `1192135096`
- Safetensors SHA-256:
  `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`
- ComfyUI commit: `3ac5d7941dfa2504555260512132d1cb5664648d`
- Sampler: greedy
- Default generated-token limit: 128

Run Dinkster and ComfyUI with the same CUDA interpreter and prompt:

```bash
python tools/benchmark_qwen_generation.py dinkster \
  --model /path/to/qwen_3_06b_base.safetensors \
  --output benchmarks/qwen-generation/dinkster.json

python tools/benchmark_qwen_generation.py comfyui \
  --model /path/to/qwen_3_06b_base.safetensors \
  --comfyui-root /path/to/ComfyUI-at-3ac5d794 \
  --output benchmarks/qwen-generation/comfyui.json
```

External runtimes require a local conversion of the same model. Start the
server with that model loaded, then run the same prompt and token limit through
both a direct client and the Dinkster OpenAI-compatible provider. For LM Studio:

```bash
python tools/benchmark_qwen_generation.py lm-studio \
  --external-model publisher/qwen3-0.6b-conversion \
  --external-artifact /path/to/qwen3-0.6b-conversion.gguf \
  --output /tmp/lm-studio-direct.json

python tools/benchmark_qwen_generation.py lm-studio-provider \
  --external-model publisher/qwen3-0.6b-conversion \
  --external-artifact /path/to/qwen3-0.6b-conversion.gguf \
  --compare-with /tmp/lm-studio-direct.json \
  --output /tmp/lm-studio-provider.json
```

For another OpenAI-compatible runtime such as llama.cpp, use the `openai` and
`openai-provider` backends and identify the exact server revision:

```bash
python tools/benchmark_qwen_generation.py openai \
  --external-runtime llama.cpp@REVISION \
  --external-model qwen3-0.6b-conversion \
  --external-artifact /path/to/qwen3-0.6b-conversion.gguf \
  --output /tmp/llama-cpp-direct.json

python tools/benchmark_qwen_generation.py openai-provider \
  --external-runtime llama.cpp@REVISION \
  --external-compatibility llama.cpp \
  --external-model qwen3-0.6b-conversion \
  --external-artifact /path/to/qwen3-0.6b-conversion.gguf \
  --compare-with /tmp/llama-cpp-direct.json \
  --output /tmp/llama-cpp-provider.json
```

The JSON records every run, exact workload, implementation revision, source
and converted artifact hashes, output hashes, median latency, and generated
tokens per second. `--compare-with` refuses mismatched workloads, artifacts,
hosts, runtimes, runtime models, Python versions, output hashes, token counts,
or repeat counts before recording provider/direct latency and throughput
ratios. Compare accelerator backends only when reports were produced on the
same host and accelerator. A converted model is a performance reference, not
a numerical-parity reference, because its storage format may change values.

## Explicit layer placement

Compare one- and two-GPU execution of the pinned Dinkster model in separate
processes. The default two-GPU split accounts for the embedding on the first
device rather than assigning equal layer counts.

```bash
python tools/benchmark_qwen_layer_placement.py one-gpu \
  --model /path/to/qwen_3_06b_base.safetensors \
  --output /tmp/qwen-one-gpu.json

python tools/benchmark_qwen_layer_placement.py two-gpu \
  --model /path/to/qwen_3_06b_base.safetensors \
  --compare-with /tmp/qwen-one-gpu.json \
  --require-memory-reduction \
  --output /tmp/qwen-two-gpu.json

cp /tmp/qwen-one-gpu.json \
  benchmarks/qwen-generation/qwen-layer-placement-one-gpu.json
cp /tmp/qwen-two-gpu.json \
  benchmarks/qwen-generation/qwen-layer-placement-two-gpu.json
```

The reports bind exact output hashes, placement ranges, implementation and
artifact revisions, throughput, and per-device allocated and peak memory.
