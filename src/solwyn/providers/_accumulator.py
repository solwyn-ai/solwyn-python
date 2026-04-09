"""Protocol for accumulating token usage from streaming chunks.

Each provider's streaming format is different:
- OpenAI: usage only in the final chunk (requires stream_options)
- Anthropic: input in message_start, output in message_delta
- Google: usage_metadata on each chunk, most complete on last

The accumulator observes every chunk and produces TokenDetails on finalize.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from solwyn._token_details import TokenDetails


@runtime_checkable
class StreamUsageAccumulator(Protocol):
    """Accumulates token usage from streaming response chunks."""

    def observe(self, chunk: Any) -> None:
        """Observe a single streaming chunk. Called for every chunk yielded."""
        ...

    def finalize(self) -> TokenDetails:
        """Return accumulated TokenDetails after the stream is exhausted.

        Must return TokenDetails() (all zeros) if no usage data was observed.
        Must never raise.
        """
        ...
