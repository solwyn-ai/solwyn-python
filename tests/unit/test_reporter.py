"""Tests for metadata reporter."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._types import MetadataEvent, ProviderName
from solwyn.reporter import (
    AsyncMetadataReporter,
    MetadataReporter,
    _ReporterBase,
)


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
        "is_failover": False,
        "sdk_instance_id": "test-instance-001",
        "timestamp": datetime.now(UTC),
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)


# ---------------------------------------------------------------------------
# Base class (sans-I/O) tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReporterBase:
    """Tests for _ReporterBase sans-I/O logic."""

    def test_enqueue_adds_event(self) -> None:
        base = _ReporterBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        event = _make_event()
        base._enqueue(event)
        assert len(base._queue) == 1

    def test_drain_batch_returns_up_to_batch_size(self) -> None:
        base = _ReporterBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            batch_size=3,
        )
        for _ in range(5):
            base._enqueue(_make_event())

        batch = base._drain_batch()
        assert len(batch) == 3
        assert len(base._queue) == 2

    def test_drain_batch_returns_all_when_less_than_batch_size(self) -> None:
        base = _ReporterBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            batch_size=10,
        )
        for _ in range(3):
            base._enqueue(_make_event())

        batch = base._drain_batch()
        assert len(batch) == 3
        assert len(base._queue) == 0

    def test_queue_overflow_drops_oldest(self) -> None:
        base = _ReporterBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            max_queue_size=3,
        )

        events = []
        for i in range(5):
            event = _make_event(input_tokens=i)
            events.append(event)
            base._enqueue(event)

        # Queue should contain only the 3 most recent
        assert len(base._queue) == 3
        assert base._queue[0].input_tokens == 2
        assert base._queue[1].input_tokens == 3
        assert base._queue[2].input_tokens == 4


# ---------------------------------------------------------------------------
# Sync reporter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetadataReporter:
    """Tests for the synchronous MetadataReporter."""

    def test_report_enqueues_event(self) -> None:
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
            )
            # Stop the background thread to avoid timing issues
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            event = _make_event()
            reporter.report(event)
            assert len(reporter._queue) == 1
            reporter._http.close()

    def test_batch_flush_triggers_at_batch_size(self) -> None:
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
                batch_size=3,
            )
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            # Enqueue 5 events
            for _ in range(5):
                reporter.report(_make_event())

            # Mock the HTTP call
            mock_response = MagicMock()
            with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
                reporter._flush_remaining()

            # Should have sent 2 batches (3 + 2)
            assert mock_post.call_count == 2
            reporter._http.close()

    def test_graceful_shutdown_flushes_remaining(self) -> None:
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
                batch_size=10,
            )
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            for _ in range(3):
                reporter.report(_make_event())

            assert len(reporter._queue) == 3

            mock_response = MagicMock()
            with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
                reporter.close()

            # Should have flushed all remaining events
            assert mock_post.call_count == 1
            assert len(reporter._queue) == 0

    def test_context_manager(self) -> None:
        with (
            patch("solwyn.reporter.MetadataReporter._flush_loop"),
            MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
            ) as reporter,
        ):
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)
            reporter.report(_make_event())
        # After context exit, reporter should be closed


# ---------------------------------------------------------------------------
# Async reporter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncMetadataReporter:
    """Tests for the async reporter base logic (no event loop needed)."""

    def test_report_enqueues_event(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )
        event = _make_event()
        reporter.report(event)
        assert len(reporter._queue) == 1

    def test_queue_overflow(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            max_queue_size=2,
        )
        reporter.report(_make_event(input_tokens=1))
        reporter.report(_make_event(input_tokens=2))
        reporter.report(_make_event(input_tokens=3))

        assert len(reporter._queue) == 2
        assert reporter._queue[0].input_tokens == 2
        assert reporter._queue[1].input_tokens == 3
