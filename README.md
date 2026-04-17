# Solwyn Python SDK

Budget enforcement, circuit breaking, and usage tracking for OpenAI, Anthropic, and Google LLM clients.

[![CI](https://github.com/solwyn-ai/solwyn-python/actions/workflows/ci.yml/badge.svg)](https://github.com/solwyn-ai/solwyn-python/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/solwyn)](https://pypi.org/project/solwyn/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/solwyn)](https://pypi.org/project/solwyn/)
[![License](https://img.shields.io/github/license/solwyn-ai/solwyn-python)](LICENSE)

Solwyn wraps your existing LLM client. Calls go directly to the provider — the SDK only reports metadata (token counts, latency, model name) to the Solwyn API. **Prompts and responses never leave your application.**

## Installation

```sh
pip install solwyn
```

For improved token estimation with OpenAI models:

```sh
pip install solwyn[openai]
```

## Quick Start

```python
from openai import OpenAI
from solwyn import Solwyn

client = Solwyn(
    OpenAI(),
    api_key="sk_solwyn_...",
    project_id="proj_abc12345",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

client.close()
```

Or use as a context manager:

```python
with Solwyn(OpenAI(), api_key="sk_solwyn_...", project_id="proj_abc12345") as client:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

## Providers

### OpenAI

```python
from openai import OpenAI
from solwyn import Solwyn

client = Solwyn(OpenAI(), api_key="sk_solwyn_...", project_id="proj_abc12345")
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

### Anthropic

```python
from anthropic import Anthropic
from solwyn import Solwyn

client = Solwyn(Anthropic(), api_key="sk_solwyn_...", project_id="proj_abc12345")
response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[...])
```

### Google Gemini

```python
from google import genai
from solwyn import Solwyn

client = Solwyn(genai.Client(api_key="..."), api_key="sk_solwyn_...", project_id="proj_abc12345")
response = client.models.generate_content(model="gemini-2.0-flash", contents="Hello!")
```

## Async

```python
from openai import AsyncOpenAI
from solwyn import AsyncSolwyn

async with AsyncSolwyn(
    AsyncOpenAI(),
    api_key="sk_solwyn_...",
    project_id="proj_abc12345",
) as client:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

## Streaming

Pass `stream=True` as you normally would. Solwyn wraps the stream transparently and reports usage when it completes:

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## Budget Enforcement

Set `budget_mode` to control spending:

```python
client = Solwyn(
    OpenAI(),
    api_key="sk_solwyn_...",
    project_id="proj_abc12345",
    budget_mode="hard_deny",
)
```

| Mode | Behavior |
|------|----------|
| `alert_only` | Log a warning when budget is exceeded (default) |
| `hard_deny` | Raise `BudgetExceededError` and block the call |

```python
from solwyn import BudgetExceededError

try:
    response = client.chat.completions.create(model="gpt-4o", messages=[...])
except BudgetExceededError as e:
    print(f"Budget limit: ${e.budget_limit}, usage: ${e.current_usage}")
```

## Configuration

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| `api_key` | `SOLWYN_API_KEY` | *required* | Solwyn API key |
| `project_id` | `SOLWYN_PROJECT_ID` | *required* | Project identifier |
| `api_url` | `SOLWYN_API_URL` | `https://api.solwyn.ai` | Solwyn API endpoint |
| `fail_open` | `SOLWYN_FAIL_OPEN` | `True` | Allow LLM calls when Solwyn API is unreachable |
| `budget_mode` | `SOLWYN_BUDGET_MODE` | `alert_only` | Budget enforcement mode |
| `fallback_model` | `SOLWYN_FALLBACK_MODEL` | `None` | Model name to retry with when the primary call fails (same provider, same client) |

Use env vars to avoid passing credentials in code:

```sh
export SOLWYN_API_KEY="sk_solwyn_..."
export SOLWYN_PROJECT_ID="proj_abc12345"
```

```python
client = Solwyn(OpenAI())  # picks up from environment
```

## Error Handling

All SDK errors inherit from `SolwynError`:

| Exception | Raised when |
|-----------|-------------|
| `BudgetExceededError` | Budget exceeded in `hard_deny` mode |
| `ProviderUnavailableError` | Circuit breaker is open |
| `ConfigurationError` | Invalid API key or project ID format |

Provider errors (e.g., `openai.RateLimitError`) pass through unmodified.

## Data Transparency

The SDK sends a `MetadataEvent` after each LLM call. This is everything it transmits:

| Field | Type | Description |
|-------|------|-------------|
| `project_id` | `str` | Project identifier |
| `model` | `str` | Model name (e.g., `gpt-4o`) |
| `provider` | `str` | `openai`, `anthropic`, or `google` |
| `input_tokens` | `int` | Input token count |
| `output_tokens` | `int` | Output token count |
| `token_details` | `object` | Breakdown: cached, reasoning, audio tokens |
| `latency_ms` | `float` | Call duration in milliseconds |
| `status` | `str` | `success`, `error`, or `budget_denied` |
| `is_failover` | `bool` | Whether the call used `fallback_model` after the primary model failed |
| `sdk_instance_id` | `str` | Per-process UUID for deduplication |
| `timestamp` | `datetime` | When the call completed (UTC) |

**The SDK never captures, logs, or transmits prompts or responses.** This is enforced by [structural tests](tests/unit/test_privacy_firewall.py) and the [privacy module](src/solwyn/_privacy.py).

## Requirements

Python 3.11+

## Contributing

```sh
make install          # install in dev mode
make install-hooks    # install pre-commit hook
make check            # lint + format + typecheck
make test             # run unit tests
```

## Links

- [Documentation](https://docs.solwyn.ai)
- [Solwyn Cloud](https://solwyn.ai) — Dashboard, alerts, and analytics
- [MPI.sh](https://mpi.sh) — LLM API pricing comparison

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
