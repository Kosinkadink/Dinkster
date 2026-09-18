## Text generation

- Native raw-prompt text generation from the Anima Qwen3-0.6B component,
  exposed as Generate Text (`TextGenerate`) and Enhance Prompt
  (`TextGenerateLTX2Prompt`). Both nodes support greedy generation or an
  ordered seeded sampler chain with temperature, top-k, min-p, top-p,
  repetition penalty, and presence penalty. Generation is cancellable and
  uses the component's resident placement.
- Native provider callers can opt into fixed-capacity continuous generation
  with chunked prompt prefill, bounded decode priority, round-robin decode
  cohorts, and same-position decode batching. Sessions, cancellation,
  deterministic sampling, and commit-or-rollback behavior are unchanged
  under scheduling.
- Native provider callers can explicitly place contiguous Qwen transformer
  layer ranges on indexed CUDA devices. Weights and paged KV remain under
  per-device residency mechanisms; direct and continuous generation transfer
  activations only at range and tied-output boundaries. Automatic placement is
  not supported.
- Enhance Prompt applies the LTX-2 text-to-video instruction and removes
  reasoning and channel markers from the result.
- Generate Text and Enhance Prompt can instead use a configured external
  OpenAI-compatible service. Configure one with `dinkster-serve`
  `--openai-base-url` and `--openai-model`, then choose the human-labeled
  service in the node's advanced section. Omitting Service keeps native
  execution. API key, `openai` or `llama.cpp` compatibility,
  SSE or JSON responses, and timeout are configurable. External execution
  does not load or transfer a connected native text encoder.
- Sandboxed external generation supports public HTTPS endpoints through the
  per-pack egress grant. Local HTTP endpoints are supported without pack
  sandboxing.
- Image, video, audio, thinking mode, and model-default templates are refused
  rather than ignored. Chat generation and multimodal generation are not
  supported by this provider.
- The torch-free OpenAI-compatible provider API supports text completion and
  chat endpoints, SSE and non-streaming JSON responses, cancellation, bounded
  timeouts, and usage reporting. Standard OpenAI mode supports greedy,
  temperature, and top-p sampling. llama.cpp mode additionally preserves the
  explicit order of top-k, min-p, and typical-p stages. Stateful sessions,
  token-ID stops, tool calls, reasoning content, and penalty stages whose
  complete-history semantics cannot be guaranteed are refused.
- Configuring an external generation model on `dinkster-serve` also enables the
  native `/api/generation` HTTP surface and OpenAI-compatible `/v1/models`,
  `/v1/completions`, `/v1/chat/completions`, and `/v1/responses` routes. JSON
  and SSE responses, disconnect cancellation, lazy model load and unload, and
  server-owned continuation identities are supported. Stateful continuation
  is available only when the selected provider advertises session support.
