"""Tests for same-provider model fallback in Solwyn and AsyncSolwyn."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._base import _SolwynBase
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.config import SolwynConfig


def _cfg(**overrides):
    defaults = {
        "api_key": VALID_API_KEY,
        "project_id": VALID_PROJECT_ID,
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
    defaults = {"api_key": VALID_API_KEY, "project_id": VALID_PROJECT_ID}
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


@pytest.mark.unit
class TestSyncFallbackModel:
    """Sync client falls back to fallback_model on primary failure."""

    def test_retry_success_sets_is_failover(self) -> None:
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
        assert any(e.is_failover and e.status.value == "success" for e in reported)

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
    defaults = {"api_key": VALID_API_KEY, "project_id": VALID_PROJECT_ID}
    defaults.update(overrides)
    solwyn = AsyncSolwyn(client, **defaults)
    solwyn._budget.check_budget = AsyncMock(return_value=_allow_budget_mock())
    solwyn._reporter.report = MagicMock()
    return solwyn


@pytest.mark.unit
class TestAsyncFallbackModel:
    """Async client falls back to fallback_model on primary failure."""

    @pytest.mark.asyncio
    async def test_async_retry_success_sets_is_failover(self) -> None:
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
        assert any(e.is_failover and e.status.value == "success" for e in reported)

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

    def test_stream_true_retry_returns_wrapped_stream(self) -> None:
        """stream=True: primary raises on .create, retry returns a stream iterable."""
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
            patch.object(solwyn._reporter, "report"),
        ):
            wrapper = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )

            list(wrapper)

        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"
        assert client.chat.completions.create.call_args_list[1].kwargs["stream"] is True

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

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
