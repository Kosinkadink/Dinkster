# dinkster-nodes-generation-openai

Isolated execution provider for `dinkster.text_generate` and
`dinkster.prompt_enhance`. It forwards requests to one host-configured
OpenAI-compatible endpoint. Endpoint, model, dialect, response mode, and
timeout rotate execution identity; the API key is transport authority only
and is excluded from identity and diagnostics.

The pack requests network access. Sandboxed serving grants only the configured
public HTTPS origin through Dinkster's per-worker egress proxy. Local HTTP servers
can be used when pack sandboxing is disabled.

Configure the pack through `dinkster-serve`:

```text
--openai-base-url URL
--openai-model MODEL
[--openai-api-key KEY]
[--openai-compatibility openai|llama.cpp]
[--openai-response-mode stream|json]
[--openai-timeout SECONDS]
```

The same values can come from `DINKSTER_OPENAI_BASE_URL`,
`DINKSTER_OPENAI_MODEL`, `DINKSTER_OPENAI_API_KEY`,
`DINKSTER_OPENAI_COMPATIBILITY`, `DINKSTER_OPENAI_STREAM`, and
`DINKSTER_OPENAI_TIMEOUT`. The parent removes these variables before launching
other workers. Standard OpenAI mode refuses non-portable sampler stages;
llama.cpp mode preserves its supported ordered sampler chain.
