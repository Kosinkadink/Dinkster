## Text generation

The torch-free external-provider API supports text completion and chat,
streaming and non-streaming responses, cancellation, bounded timeouts, and
usage reporting. Configuring an external provider enables `/api/generation`
and the OpenAI-compatible `/v1` routes.

Dinkster does not currently ship an in-process text-generation model runtime.
