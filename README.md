# Solwyn Python SDK

AI Agent Control Plane SDK — hard spending caps, automatic provider failover, per-agent cost attribution.

## Install

```bash
pip install solwyn
```

For accurate token counting with OpenAI models:

```bash
pip install solwyn[openai]
```

## Quickstart

```python
from openai import OpenAI
from solwyn import Solwyn

client = Solwyn(OpenAI())  # Wraps your existing client
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

That's it. Solwyn wraps your existing provider client and automatically tracks token usage, enforces budget limits, and handles provider failover — without ever seeing your prompts.

## Architecture

```
Your Application                          Solwyn Cloud
┌──────────────────────────┐             ┌──────────────────┐
│  your code               │  metadata   │  PricingService   │
│    ↓                     │  (tokens,   │  Budget engine    │
│  Solwyn(OpenAI())        │  latency)   │  Cost dashboard   │
│    ↓                     │ ──────────→ │  Alerts           │
│  LLM provider  ←─────── │ direct call │                    │
└──────────────────────────┘             └──────────────────┘
```

**Key principle:** The SDK is a wrapper, not a proxy. LLM calls go directly from your application to the provider. Only metadata (token counts, latency, model name) is sent to Solwyn's servers.

## What data does the SDK send?

The SDK sends a `MetadataEvent` after each LLM call containing **only**:

| Field | Type | Description |
|-------|------|-------------|
| `project_id` | string | Your project identifier |
| `model` | string | LLM model name (e.g. `gpt-4o`) |
| `provider` | string | Provider name (`openai`, `anthropic`, `google`) |
| `input_tokens` | int | Input token count |
| `output_tokens` | int | Output token count |
| `token_details` | object | Detailed token breakdown (cached, reasoning, audio, etc.) |
| `latency_ms` | float | Call latency in milliseconds |
| `status` | string | Call outcome (`success`, `error`, `budget_denied`) |
| `is_failover` | bool | Whether a fallback provider was used |
| `sdk_instance_id` | string | Per-process UUID for deduplication (not user tracking) |
| `timestamp` | datetime | When the call completed |

**The SDK never captures, logs, or transmits prompts or responses.** This is enforced by structural tests in the codebase. See [`src/solwyn/_privacy.py`](src/solwyn/_privacy.py) for the implementation.

## Supported Providers

- **OpenAI** — GPT-4o, GPT-4, GPT-3.5, o1, o3, and all chat completion models
- **Anthropic** — Claude 4, Claude 3.5, and all messages API models
- **Google/Gemini** — Gemini 2.0, Gemini 1.5, and all generate_content models

## Links

- [Documentation](https://docs.solwyn.ai)
- [Solwyn Cloud](https://solwyn.ai) — Dashboard, alerts, and analytics
- [MPI.sh](https://mpi.sh) — Free LLM API pricing comparison

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
