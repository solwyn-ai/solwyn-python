"""Anthropic provider adapter — token extraction only, no pricing.

Anthropic usage normalization:

    input_tokens (normalized) = base_input + cache_read + cache_creation

Anthropic reports cache_read_input_tokens and cache_creation_input_tokens as
SEPARATE additive fields, not subsets of input_tokens. Our normalized
input_tokens is the sum of all three.

reasoning_tokens stays 0 — Anthropic folds extended thinking tokens into
output_tokens and does not report them separately (documented blind spot).

Cache fields may be absent on older API responses — all missing fields default
to 0 via getattr guards.
"""

from __future__ import annotations

from typing import Any

from solwyn._token_details import TokenDetails


class AnthropicAdapter:
    """Extracts normalized TokenDetails from Anthropic API responses."""

    @property
    def name(self) -> str:
        return "anthropic"

    def detect_client(self, client: Any) -> bool:
        """Return True if the client's module path contains 'anthropic'."""
        return "anthropic" in getattr(type(client), "__module__", "")

    def detect_model(self, model: str) -> bool:
        """Return True for claude-* model prefix."""
        return model.startswith("claude-")

    def extract_usage(self, response: Any) -> TokenDetails:
        """Extract token usage from an Anthropic messages.create() response.

        Returns TokenDetails() with all zeros when usage is unavailable.
        Never raises — returns zeros for any unexpected response shape.

        Normalization:
        - input_tokens = base + cache_read + cache_creation (all additive)
        - cached_input_tokens = cache_read_input_tokens
        - cache_creation_tokens = cache_creation_input_tokens
        - reasoning_tokens = 0 (folded into output_tokens by Anthropic)
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenDetails()

        base_input = getattr(usage, "input_tokens", 0) or 0
        output = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        return TokenDetails(
            input_tokens=base_input + cache_read + cache_creation,
            output_tokens=output,
            cached_input_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            # reasoning_tokens intentionally 0 — Anthropic doesn't report separately
        )

    def prepare_streaming(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Anthropic streams include usage events by default — no changes needed."""
        return dict(kwargs)

    def create_stream_accumulator(self) -> AnthropicStreamAccumulator:
        return AnthropicStreamAccumulator()


class AnthropicStreamAccumulator:
    """Accumulates usage from Anthropic streaming events.

    Input tokens arrive in message_start.message.usage.
    Output tokens arrive in message_delta.usage.
    Cache fields are on message_start and may be absent on older APIs.
    """

    def __init__(self) -> None:
        self._base_input: int = 0
        self._cache_read: int = 0
        self._cache_creation: int = 0
        self._output: int = 0

    def observe(self, chunk: Any) -> None:
        event_type = getattr(chunk, "type", None)

        if event_type == "message_start":
            usage = getattr(getattr(chunk, "message", None), "usage", None)
            if usage is not None:
                self._base_input = getattr(usage, "input_tokens", 0) or 0
                self._cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                self._cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        elif event_type == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._output = getattr(usage, "output_tokens", 0) or 0

    def finalize(self) -> TokenDetails:
        return TokenDetails(
            input_tokens=self._base_input + self._cache_read + self._cache_creation,
            output_tokens=self._output,
            cached_input_tokens=self._cache_read,
            cache_creation_tokens=self._cache_creation,
        )
