"""Tests for same-provider model fallback in Solwyn and AsyncSolwyn."""

from __future__ import annotations

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._base import _SolwynBase
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


from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from solwyn.client import Solwyn  # noqa: E402


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
