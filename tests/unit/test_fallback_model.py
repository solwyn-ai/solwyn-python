"""Tests for same-provider model fallback in Solwyn and AsyncSolwyn."""

from __future__ import annotations

from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._base import _SolwynBase
from solwyn._types import CircuitState
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.config import SolwynConfig


def _cfg(**overrides):
    defaults = {
        "api_key": VALID_API_KEY,
    }
    defaults.update(overrides)
    return SolwynConfig(**defaults)


@pytest.mark.unit
class TestShouldRetryWithFallback:
    def test_returns_true_when_fallback_model_differs(self) -> None:
        base = _SolwynBase(_cfg(fallback_model="gpt-4o-mini"))
        assert base._should_retry_with_fallback("gpt-4o") is True

    def test_returns_false_when_no_fallback_configured(self) -> None:
        base = _SolwynBase(_cfg())
        assert base._should_retry_with_fallback("gpt-4o") is False

    def test_returns_false_when_already_on_fallback(self) -> None:
        base = _SolwynBase(_cfg(fallback_model="gpt-4o-mini"))
        assert base._should_retry_with_fallback("gpt-4o-mini") is False


@pytest.mark.unit
class TestPrepareFallbackKwargs:
    def test_swaps_model_key(self) -> None:
        base = _SolwynBase(_cfg(fallback_model="gpt-4o-mini"))
        swapped = base._prepare_fallback_kwargs(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert swapped["model"] == "gpt-4o-mini"
        assert swapped["messages"] == [{"role": "user", "content": "hi"}]

    def test_does_not_mutate_input(self) -> None:
        base = _SolwynBase(_cfg(fallback_model="gpt-4o-mini"))
        original = {"model": "gpt-4o", "temperature": 0.7}
        swapped = base._prepare_fallback_kwargs(original)
        assert original["model"] == "gpt-4o"
        assert swapped is not original

    def test_raises_runtime_error_when_no_fallback(self) -> None:
        base = _SolwynBase(_cfg())
        with pytest.raises(RuntimeError, match="fallback_model is not configured"):
            base._prepare_fallback_kwargs({"model": "gpt-4o"})


# ---------------------------------------------------------------------------
# Sync retry (via Solwyn._intercepted_call)
# ---------------------------------------------------------------------------


def _mock_openai_client_with_failure_then_success():
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"

    success_response = MagicMock()
    success_response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)

    client.chat.completions.create.side_effect = [
        RuntimeError("primary boom"),
        success_response,
    ]
    return client, success_response


def _mock_openai_client_always_fail():
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.chat.completions.create.side_effect = [
        RuntimeError("primary boom"),
        RuntimeError("fallback boom"),
    ]
    return client


def _make_solwyn(client, **overrides):
    defaults = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    return solwyn


def _allow_budget_mock():
    return MagicMock(
        allowed=True,
        reservation_id=None,
        mode=MagicMock(value="alert_only"),
    )


def _monotonic_sequence(*values: float):
    iterator = iter(values)
    last = values[-1]

    def _next() -> float:
        nonlocal last
        with suppress(StopIteration):
            last = next(iterator)
        return last

    return _next


@pytest.mark.unit
class TestSyncFallbackModel:
    """Sync client falls back to fallback_model on primary failure."""

    def test_retry_success_sets_is_model_fallback(self) -> None:
        client, resp = _mock_openai_client_with_failure_then_success()
        solwyn = _make_solwyn(client, fallback_model="gpt-4o-mini")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report") as report_mock,
        ):
            result = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result is resp
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"
        reported = [c.args[0] for c in report_mock.call_args_list]
        assert any(e.is_model_fallback and e.status.value == "success" for e in reported)

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_retry_both_fail_raises_primary(self) -> None:
        client = _mock_openai_client_always_fail()
        solwyn = _make_solwyn(client, fallback_model="gpt-4o-mini")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report"),
            pytest.raises(RuntimeError, match="primary boom"),
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert client.chat.completions.create.call_count == 2

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_no_retry_when_fallback_not_configured(self) -> None:
        client = _mock_openai_client_always_fail()
        solwyn = _make_solwyn(client)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report"),
            pytest.raises(RuntimeError, match="primary boom"),
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert client.chat.completions.create.call_count == 1

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_no_retry_when_model_equals_fallback_model(self) -> None:
        """Calling with model=fallback_model must not retry (loop guard)."""
        client = _mock_openai_client_always_fail()
        solwyn = _make_solwyn(client, fallback_model="gpt-4o")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report"),
            pytest.raises(RuntimeError, match="primary boom"),
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert client.chat.completions.create.call_count == 1

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


# ---------------------------------------------------------------------------
# Async retry (via AsyncSolwyn._intercepted_call)
# ---------------------------------------------------------------------------


def _mock_async_openai_client_fail_then_success():
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "AsyncOpenAI"

    success_response = MagicMock()
    success_response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)

    client.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("primary boom"), success_response]
    )
    return client, success_response


def _mock_async_openai_client_always_fail():
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "AsyncOpenAI"
    client.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("primary boom"), RuntimeError("fallback boom")]
    )
    return client


def _make_async_solwyn(client, **overrides):
    defaults = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    solwyn = AsyncSolwyn(client, **defaults)
    solwyn._budget.check_budget = AsyncMock(return_value=_allow_budget_mock())
    solwyn._reporter.report = MagicMock()
    return solwyn


@pytest.mark.unit
class TestAsyncFallbackModel:
    """Async client falls back to fallback_model on primary failure."""

    @pytest.mark.asyncio
    async def test_async_retry_success_sets_is_model_fallback(self) -> None:
        client, resp = _mock_async_openai_client_fail_then_success()
        solwyn = _make_async_solwyn(client, fallback_model="gpt-4o-mini")

        result = await solwyn.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is resp
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"
        reported = [c.args[0] for c in solwyn._reporter.report.call_args_list]
        assert any(e.is_model_fallback and e.status.value == "success" for e in reported)

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_async_retry_both_fail_raises_primary(self) -> None:
        client = _mock_async_openai_client_always_fail()
        solwyn = _make_async_solwyn(client, fallback_model="gpt-4o-mini")

        with pytest.raises(RuntimeError, match="primary boom"):
            await solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert client.chat.completions.create.call_count == 2

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_async_no_retry_when_fallback_not_configured(self) -> None:
        client = _mock_async_openai_client_always_fail()
        solwyn = _make_async_solwyn(client)

        with pytest.raises(RuntimeError, match="primary boom"):
            await solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert client.chat.completions.create.call_count == 1

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_async_no_retry_when_model_equals_fallback_model(self) -> None:
        client = _mock_async_openai_client_always_fail()
        solwyn = _make_async_solwyn(client, fallback_model="gpt-4o")

        with pytest.raises(RuntimeError, match="primary boom"):
            await solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert client.chat.completions.create.call_count == 1

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


# ---------------------------------------------------------------------------
# Streaming retry (stream=True, _force_stream=True)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncFallbackModelStreaming:
    """Retry works when stream=True and when _force_stream=True (Google)."""

    def test_force_stream_google_retry(self) -> None:
        """_force_stream=True is Google-only. Primary fails, retry via same code path."""
        client = MagicMock()
        client.__class__.__module__ = "google.genai._client"
        client.__class__.__name__ = "Client"

        fake_stream = iter([])
        client.models.generate_content_stream.side_effect = [
            RuntimeError("primary boom"),
            fake_stream,
        ]

        solwyn = _make_solwyn(client, fallback_model="gemini-2.0-flash-lite")
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report"),
        ):
            wrapper = solwyn.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents="hi",
            )

            list(wrapper)

        assert client.models.generate_content_stream.call_count == 2
        assert (
            client.models.generate_content_stream.call_args_list[1].kwargs["model"]
            == "gemini-2.0-flash-lite"
        )

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


@pytest.mark.unit
class TestAsyncFallbackModelStreaming:
    @pytest.mark.asyncio
    async def test_async_stream_true_retry_returns_wrapped_stream(self) -> None:
        client = MagicMock()
        client.__class__.__module__ = "openai._client"
        client.__class__.__name__ = "AsyncOpenAI"

        async def _empty_async_iter():
            return
            yield  # pragma: no cover

        client.chat.completions.create = AsyncMock(
            side_effect=[RuntimeError("primary boom"), _empty_async_iter()]
        )

        solwyn = _make_async_solwyn(client, fallback_model="gpt-4o-mini")

        wrapper = await solwyn.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for _ in wrapper:
            pass

        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


# ---------------------------------------------------------------------------
# Anthropic provider retry (covers _sync_dispatch's messages.create branch)
# ---------------------------------------------------------------------------


def _mock_anthropic_client_fail_then_success():
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    client.__class__.__name__ = "Anthropic"

    success_response = MagicMock()
    success_response.usage = SimpleNamespace(input_tokens=10, output_tokens=5)

    client.messages.create.side_effect = [
        RuntimeError("primary boom"),
        success_response,
    ]
    return client, success_response


@pytest.mark.unit
class TestAnthropicFallbackModel:
    """Anthropic's messages.create path retries with fallback_model."""

    def test_anthropic_retry_success_sets_is_model_fallback(self) -> None:
        client, resp = _mock_anthropic_client_fail_then_success()
        solwyn = _make_solwyn(client, fallback_model="claude-3-haiku-20240307")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report") as report_mock,
        ):
            result = solwyn.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result is resp
        assert client.messages.create.call_count == 2
        assert client.messages.create.call_args_list[1].kwargs["model"] == "claude-3-haiku-20240307"
        reported = [c.args[0] for c in report_mock.call_args_list]
        assert any(e.is_model_fallback and e.status.value == "success" for e in reported)

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


# ---------------------------------------------------------------------------
# Circuit breaker state after rescued retry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCircuitBreakerStateAfterRescuedRetry:
    """Rescued retries must not open the circuit in CLOSED state."""

    def test_breaker_stays_closed_after_five_rescued_retries(self) -> None:
        """Primary fails, fallback succeeds, repeated 5x: breaker stays CLOSED.

        record_success() resets failure_count to 0 in CLOSED state
        (circuit_breaker.py:82-84), so each rescued retry nets out.
        """
        client = MagicMock()
        client.__class__.__module__ = "openai._client"
        client.__class__.__name__ = "OpenAI"

        success_response = MagicMock()
        success_response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)

        # 5 rescued retries = 10 calls alternating fail/success
        client.chat.completions.create.side_effect = [
            RuntimeError("primary boom"),
            success_response,
        ] * 5

        solwyn = _make_solwyn(client, fallback_model="gpt-4o-mini")
        cb = solwyn._get_circuit_breaker("openai")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report"),
        ):
            for _ in range(5):
                solwyn.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                )

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert client.chat.completions.create.call_count == 10

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_half_open_probe_that_recovers_closes_breaker(self) -> None:
        client, _ = _mock_openai_client_with_failure_then_success()
        solwyn = _make_solwyn(client, fallback_model="gpt-4o-mini")
        cb = solwyn._get_circuit_breaker("openai")
        cb.failure_threshold = 1
        cb.success_threshold = 1
        cb.recovery_timeout = 0
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report"),
        ):
            result = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result is not None
        assert cb.state == CircuitState.CLOSED

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    @pytest.mark.asyncio
    async def test_async_half_open_probe_that_recovers_closes_breaker(self) -> None:
        client, _ = _mock_async_openai_client_fail_then_success()
        solwyn = _make_async_solwyn(client, fallback_model="gpt-4o-mini")
        cb = solwyn._get_circuit_breaker("openai")
        cb.failure_threshold = 1
        cb.success_threshold = 1
        cb.recovery_timeout = 0
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        result = await solwyn.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is not None
        assert cb.state == CircuitState.CLOSED

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


# ---------------------------------------------------------------------------
# Streaming on_complete fires with is_model_fallback=True
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamingOnCompleteFailoverEvent:
    """When stream retry succeeds, on_complete emits a success event with is_model_fallback=True."""

    def test_stream_true_on_complete_reports_failover(self) -> None:
        client = MagicMock()
        client.__class__.__module__ = "openai._client"
        client.__class__.__name__ = "OpenAI"

        fake_stream = iter([])
        client.chat.completions.create.side_effect = [
            RuntimeError("primary boom"),
            fake_stream,
        ]

        solwyn = _make_solwyn(client, fallback_model="gpt-4o-mini")
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report") as report_mock,
        ):
            wrapper = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            list(wrapper)

        reported = [c.args[0] for c in report_mock.call_args_list]
        success_events = [e for e in reported if e.status.value == "success"]
        assert success_events, "expected at least one success event from on_complete"
        assert all(e.is_model_fallback for e in success_events)
        assert all(e.model == "gpt-4o-mini" for e in success_events)

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_stream_true_on_complete_reports_full_failover_latency(self) -> None:
        client = MagicMock()
        client.__class__.__module__ = "openai._client"
        client.__class__.__name__ = "OpenAI"

        fake_stream = iter([])
        client.chat.completions.create.side_effect = [
            RuntimeError("primary boom"),
            fake_stream,
        ]

        solwyn = _make_solwyn(client, fallback_model="gpt-4o-mini")
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report") as report_mock,
            patch(
                "time.monotonic",
                side_effect=_monotonic_sequence(
                    100.000,
                    100.060,
                    100.060,
                    100.060,
                    100.060,
                    100.061,
                    100.090,
                ),
            ),
        ):
            wrapper = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            list(wrapper)

        reported = [c.args[0] for c in report_mock.call_args_list]
        success_events = [e for e in reported if e.status.value == "success"]
        assert success_events, "expected at least one success event from on_complete"
        assert success_events[-1].latency_ms == pytest.approx(90.0)

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    @pytest.mark.asyncio
    async def test_async_stream_true_on_complete_reports_full_failover_latency(self) -> None:
        client = MagicMock()
        client.__class__.__module__ = "openai._client"
        client.__class__.__name__ = "AsyncOpenAI"

        async def _empty_async_iter():
            return
            yield  # pragma: no cover

        client.chat.completions.create = AsyncMock(
            side_effect=[RuntimeError("primary boom"), _empty_async_iter()]
        )

        solwyn = _make_async_solwyn(client, fallback_model="gpt-4o-mini")
        with patch(
            "time.monotonic",
            side_effect=_monotonic_sequence(
                200.000,
                200.060,
                200.060,
                200.060,
                200.060,
                200.061,
                200.090,
            ),
        ):
            wrapper = await solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for _ in wrapper:
                pass

        reported = [c.args[0] for c in solwyn._reporter.report.call_args_list]
        success_events = [e for e in reported if e.status.value == "success"]
        assert success_events, "expected at least one success event from on_complete"
        assert success_events[-1].latency_ms == pytest.approx(90.0)

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


# ---------------------------------------------------------------------------
# Primary exception carries a PEP 678 note when fallback also fails
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFallbackFailureNote:
    """raise primary_exc attaches a note describing the fallback failure."""

    def test_primary_exc_has_fallback_note(self) -> None:
        client = _mock_openai_client_always_fail()
        solwyn = _make_solwyn(client, fallback_model="gpt-4o-mini")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_mock()),
            patch.object(solwyn._reporter, "report"),
            pytest.raises(RuntimeError, match="primary boom") as exc_info,
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        notes = getattr(exc_info.value, "__notes__", [])
        assert any("gpt-4o-mini" in note and "RuntimeError" in note for note in notes)

        solwyn._reporter._http.close()
        solwyn._budget._http.close()
