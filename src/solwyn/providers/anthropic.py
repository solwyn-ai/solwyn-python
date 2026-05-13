"""Anthropic provider adapter — token extraction only, no pricing.

Anthropic usage normalization:

    input_tokens (normalized) = base_input + cache_read + cache_5m + cache_1h

Anthropic reports cache_read_input_tokens as a separate additive field (not a
subset of input_tokens). Cache writes are broken out by TTL tier in the
usage.cache_creation sub-object:

    usage.cache_creation.ephemeral_5m_input_tokens  — 5-minute TTL (1.25× rate)
    usage.cache_creation.ephemeral_1h_input_tokens  — 1-hour TTL  (2×    rate)

The cache_creation sub-object is absent when no cache write occurred or on
older API responses — both 5m and 1h default to 0 via getattr guards.

reasoning_tokens stays 0 — Anthropic folds extended thinking tokens into
output_tokens and does not report them separately (documented blind spot).
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
        - input_tokens = base + cache_read + (5m + 1h cache writes)  (all additive)
        - cached_input_tokens = cache_read_input_tokens
        - cache_creation_5m_tokens = usage.cache_creation.ephemeral_5m_input_tokens
        - cache_creation_1h_tokens = usage.cache_creation.ephemeral_1h_input_tokens
        - reasoning_tokens = 0 (folded into output_tokens by Anthropic)

        The cache_creation sub-object is absent on responses where no cache write
        occurred; both 5m and 1h default to 0 via getattr guards.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenDetails()

        base_input = getattr(usage, "input_tokens", 0) or 0
        output = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        cache_creation_detail = getattr(usage, "cache_creation", None)
        cache_5m = getattr(cache_creation_detail, "ephemeral_5m_input_tokens", 0) or 0
        cache_1h = getattr(cache_creation_detail, "ephemeral_1h_input_tokens", 0) or 0

        return TokenDetails(
            input_tokens=base_input + cache_read + cache_5m + cache_1h,
            output_tokens=output,
            cached_input_tokens=cache_read,
            cache_creation_5m_tokens=cache_5m,
            cache_creation_1h_tokens=cache_1h,
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
        self._cache_5m: int = 0
        self._cache_1h: int = 0
        self._output: int = 0

    def observe(self, chunk: Any) -> None:
        event_type = getattr(chunk, "type", None)

        if event_type == "message_start":
            usage = getattr(getattr(chunk, "message", None), "usage", None)
            if usage is not None:
                self._base_input = getattr(usage, "input_tokens", 0) or 0
                self._cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                detail = getattr(usage, "cache_creation", None)
                self._cache_5m = getattr(detail, "ephemeral_5m_input_tokens", 0) or 0
                self._cache_1h = getattr(detail, "ephemeral_1h_input_tokens", 0) or 0

        elif event_type == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._output = getattr(usage, "output_tokens", 0) or 0

    def finalize(self) -> TokenDetails:
        return TokenDetails(
            input_tokens=self._base_input + self._cache_read + self._cache_5m + self._cache_1h,
            output_tokens=self._output,
            cached_input_tokens=self._cache_read,
            cache_creation_5m_tokens=self._cache_5m,
            cache_creation_1h_tokens=self._cache_1h,
        )
