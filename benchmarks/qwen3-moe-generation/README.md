# Qwen3-30B-A3B generation benchmark

This benchmark measures native direct and continuous generation from the
original Qwen3-30B-A3B BF16 checkpoint. It records model-load time, GPU and
host memory, time to first token, prefill and decode throughput, inter-token
p50/p95, aggregate throughput, completion spread, errors, and exact output
hashes. Output hashes must be stable within each fixed workload and execution
mode. The report records whether serialized and continuous hashes also match,
but does not require cross-batch equality: BF16 CUDA kernels can differ by a
few ULPs when their batch shapes differ. Numerical comparisons to another
runtime must use the same batch shape.

The tool accepts only the 16 original safetensors shards from
`Qwen/Qwen3-30B-A3B` at immutable revision
`d47d535f78ec44bd57128f8e8aeba17eeb0285ea`. Every shard byte size and SHA-256
is pinned in the tool and verified before loading.

Run on a CUDA device with at least 64 GB free memory:

```bash
python tools/benchmark_qwen3_moe_generation.py \
  --model-directory /path/to/Qwen3-30B-A3B \
  --device cuda:0 \
  --concurrency 4 \
  --output /tmp/qwen3-moe-generation.json
```

Use identical prompt, token limit, dtype, device, warmup, repeat, and
concurrency settings for external runtime comparisons. Record the external
runtime revision and converted artifact hashes beside its report. A converted
or quantized model is a performance reference, not a numerical-parity
reference. Report unavailable LM Studio, llama.cpp, vLLM, or FreeToken routes
as unavailable rather than comparing unmatched workloads.
