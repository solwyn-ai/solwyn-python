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
