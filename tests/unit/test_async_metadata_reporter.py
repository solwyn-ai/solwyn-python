"""Tests for AsyncMetadataReporter lifecycle — start, flush, close, context manager."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._types import MetadataEvent, ProviderName
from solwyn.reporter import AsyncMetadataReporter


def _make_event(**overrides) -> MetadataEvent:
    """Create a MetadataEvent with sensible test defaults."""
    defaults = {
        "project_id": "proj_abc12345",
        "model": "gpt-4o",
        "provider": ProviderName.OPENAI,
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 200.0,
        "status": "success",
        "is_model_fallback": False,
        "sdk_instance_id": "test-instance-001",
        "timestamp": datetime.now(UTC),
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)


# ---------------------------------------------------------------------------
# Lifecycle: start -> flush -> close
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterLifecycle:
    """Test the full start/report/flush/close lifecycle."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_creates_flush_task(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        reporter.start()

        assert reporter._flush_task is not None
        assert reporter._shutdown_event is not None
        assert not reporter._shutdown_event.is_set()

        # Clean up
        reporter._shutdown_event.set()
        await reporter._flush_task
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_flushes_remaining_events(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            batch_size=10,
            flush_interval=60.0,
        )
        reporter.start()

        for _ in range(3):
            reporter.report(_make_event())

        assert len(reporter._queue) == 3

        mock_response = AsyncMock()
        with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
            await reporter.close()

        # Should have flushed all events
        assert len(reporter._queue) == 0
        assert mock_post.call_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_sets_shutdown_event(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        reporter.start()

        with patch.object(reporter._http, "post", new_callable=AsyncMock):
            await reporter.close()

        assert reporter._shutdown_event.is_set()


# ---------------------------------------------------------------------------
# Batch sending
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterSendBatch:
    """_send_batch posts events to the ingest endpoint."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_posts_to_ingest(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        batch = [_make_event() for _ in range(3)]
        mock_response = AsyncMock()

        with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
            await reporter._send_batch(batch)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "metadata/ingest" in call_kwargs[0][0]
        assert len(call_kwargs[1]["json"]) == 3
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_swallows_errors(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        batch = [_make_event()]

        with patch.object(reporter._http, "post", side_effect=RuntimeError("fail")):
            # Should not raise
            await reporter._send_batch(batch)

        await reporter._http.aclose()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterContextManager:
    """async with AsyncMetadataReporter starts and closes correctly."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager_starts_and_closes(self) -> None:
        with patch.object(AsyncMetadataReporter, "_send_batch", new_callable=AsyncMock):
            async with AsyncMetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
                flush_interval=60.0,
            ) as reporter:
                reporter.report(_make_event())
                assert reporter._flush_task is not None

        # After exit, shutdown should be set
        assert reporter._shutdown_event.is_set()


# ---------------------------------------------------------------------------
# Batch size flush
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterBatchFlush:
    """Events are flushed in correct batch sizes."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_flush_remaining_batches_correctly(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            batch_size=3,
        )

        for _ in range(5):
            reporter.report(_make_event())

        mock_response = AsyncMock()
        with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
            await reporter._flush_remaining()

        # 5 events / batch_size 3 = 2 batches (3 + 2)
        assert mock_post.call_count == 2
        assert len(reporter._queue) == 0
        await reporter._http.aclose()
