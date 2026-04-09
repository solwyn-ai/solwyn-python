# sdks/python/src/solwyn/stream.py
"""Stream wrappers for intercepting streaming LLM responses.

These wrappers yield every chunk unchanged to the user while a
StreamUsageAccumulator silently observes usage data. After the stream
completes (or errors), a callback fires to settle the budget reservation
and report metadata.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, Self, cast

from solwyn._token_details import TokenDetails
from solwyn.providers._accumulator import StreamUsageAccumulator

logger = logging.getLogger(__name__)


class SyncStreamWrapper:
    """Wraps a synchronous streaming response for token accumulation.

    Yields every chunk unchanged. After the iterator is exhausted,
    calls on_complete(token_details, elapsed_ms). If iteration raises,
    calls on_error(exception) and re-raises.

    If the caller breaks early or abandons the iterator, call close()
    explicitly or use the context manager (``with stream:``) to settle
    the budget reservation with whatever usage data was observed.

    Thread safety: _settle() and _settle_error() are protected by a
    threading.Lock so that concurrent calls (e.g. close() racing with
    iterator exhaustion) fire on_complete/on_error exactly once.
    """

    def __init__(
        self,
        stream: Any,
        accumulator: StreamUsageAccumulator,
        on_complete: Callable[[TokenDetails, float], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self._stream = stream
        self._accumulator = accumulator
        self._on_complete = on_complete
        self._on_error = on_error
        self._start_time = time.monotonic()
        self._settled = False
        self._lock = threading.Lock()

    def _settle(self) -> None:
        """Fire on_complete exactly once with accumulated data."""
        with self._lock:
            if self._settled:
                return
            self._settled = True
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        token_details = self._accumulator.finalize()
        self._on_complete(token_details, elapsed_ms)

    def _settle_error(self, exc: Exception) -> None:
        """Fire on_error exactly once. Mirrors _settle() for the error path."""
        with self._lock:
            if self._settled:
                return
            self._settled = True
        self._on_error(exc)

    def __iter__(self) -> Iterator[Any]:
        try:
            for chunk in self._stream:
                self._accumulator.observe(chunk)
                yield chunk
        except Exception as exc:
            self._settle_error(exc)
            raise
        else:
            self._settle()

    def close(self) -> None:
        """Settle the budget reservation with whatever data we have, then
        forward cleanup to the inner stream's close() method.

        Safe to call multiple times — only the first call fires on_complete.
        Forwarding close() to the inner stream is also safe if called multiple
        times; well-behaved stream implementations are idempotent.
        Called automatically by __exit__ if the stream wasn't fully consumed.
        """
        self._settle()
        if hasattr(self._stream, "close"):
            self._stream.close()

    def __enter__(self) -> Self:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: object) -> bool | None:
        try:
            # self.close() now settles AND forwards to inner close().
            # Broad except is intentional: callback/provider-close failures
            # must not mask the application exception propagating through
            # __exit__ (if any).
            self.close()
        except Exception:
            logger.warning("on_complete raised during __exit__; suppressing", exc_info=True)
        if hasattr(self._stream, "__exit__"):
            return cast("bool | None", self._stream.__exit__(*args))
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AsyncStreamWrapper:
    """Wraps an asynchronous streaming response for token accumulation.

    Async version of SyncStreamWrapper. on_complete and on_error are
    coroutines (awaited after stream completes or errors).

    Call close() or use ``async with stream:`` to settle on early abort.

    No lock needed: async wrappers run in a single-threaded event loop,
    so concurrent settlement is not possible.
    """

    def __init__(
        self,
        stream: Any,
        accumulator: StreamUsageAccumulator,
        on_complete: Callable[[TokenDetails, float], Awaitable[None]],
        on_error: Callable[[Exception], Awaitable[None]],
    ) -> None:
        self._stream = stream
        self._accumulator = accumulator
        self._on_complete = on_complete
        self._on_error = on_error
        self._start_time = time.monotonic()
        self._settled = False

    async def _settle(self) -> None:
        """Fire on_complete exactly once with accumulated data."""
        if self._settled:
            return
        self._settled = True
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        token_details = self._accumulator.finalize()
        await self._on_complete(token_details, elapsed_ms)

    async def _settle_error(self, exc: Exception) -> None:
        """Fire on_error exactly once. Mirrors _settle() for the error path."""
        if self._settled:
            return
        self._settled = True
        await self._on_error(exc)

    async def __aiter__(self) -> AsyncIterator[Any]:
        try:
            async for chunk in self._stream:
                self._accumulator.observe(chunk)
                yield chunk
        except Exception as exc:
            await self._settle_error(exc)
            raise
        else:
            await self._settle()

    async def close(self) -> None:
        """Settle the budget reservation with whatever data we have, then
        forward cleanup to the inner stream's aclose() or close() method.

        Safe to call multiple times — only the first call fires on_complete.
        Forwarding to the inner stream is also safe if called multiple times;
        well-behaved stream implementations are idempotent.
        """
        await self._settle()
        if hasattr(self._stream, "aclose"):
            await self._stream.aclose()
        elif hasattr(self._stream, "close"):
            self._stream.close()

    async def __aenter__(self) -> Self:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        try:
            # self.close() now settles AND forwards to inner aclose()/close().
            # Broad except is intentional: callback/provider-close failures
            # must not mask the application exception propagating through
            # __aexit__ (if any).
            await self.close()
        except Exception:
            logger.warning("on_complete raised during __aexit__; suppressing", exc_info=True)
        if hasattr(self._stream, "__aexit__"):
            return cast("bool | None", await self._stream.__aexit__(*args))
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
