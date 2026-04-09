"""Tests for token counting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from solwyn.tokenizer import TokenizerManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_client(token_count: int = 42) -> MagicMock:
    """Build a mock Anthropic client whose count_tokens returns *token_count*."""
    client = MagicMock()
    response = MagicMock()
    response.input_tokens = token_count
    client.messages.count_tokens.return_value = response
    return client


# ---------------------------------------------------------------------------
# Tiktoken path (OpenAI models)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTiktokenPath:
    """When tiktoken is installed, OpenAI estimates use real encoding."""

    def test_returns_nonzero_for_text(self) -> None:
        tm = TokenizerManager()
        result = tm.estimate_tokens("Hello, world!", model="gpt-4o", provider="openai")
        assert result > 0

    def test_consistent_results(self) -> None:
        tm = TokenizerManager()
        a = tm.estimate_tokens("deterministic", model="gpt-4o", provider="openai")
        b = tm.estimate_tokens("deterministic", model="gpt-4o", provider="openai")
        assert a == b

    def test_longer_text_more_tokens(self) -> None:
        tm = TokenizerManager()
        short = tm.estimate_tokens("hi", model="gpt-4o", provider="openai")
        long = tm.estimate_tokens("hi " * 200, model="gpt-4o", provider="openai")
        assert long > short


# ---------------------------------------------------------------------------
# Heuristic fallback when tiktoken is unavailable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHeuristicFallback:
    """When tiktoken is None, the manager falls back to char / 4."""

    def test_openai_without_tiktoken(self) -> None:
        tm = TokenizerManager()
        # Patch tiktoken to None at module level
        with patch("solwyn.tokenizer.tiktoken", None):
            result = tm.estimate_tokens("a" * 100, model="gpt-4o", provider="openai")
        assert result == 25  # 100 / 4

    def test_unknown_provider_fallback(self) -> None:
        tm = TokenizerManager()
        result = tm.estimate_tokens("a" * 80, model="some-unknown-model", provider="unknown")
        assert result == 20  # 80 / 4


# ---------------------------------------------------------------------------
# Anthropic heuristic estimation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnthropicHeuristic:
    """Anthropic models use model-specific character ratios."""

    def test_haiku_ratio(self) -> None:
        tm = TokenizerManager()
        result = tm.estimate_tokens("a" * 450, model="claude-haiku-3-5", provider="anthropic")
        assert result == int(450 / 4.5)

    def test_sonnet_ratio(self) -> None:
        tm = TokenizerManager()
        result = tm.estimate_tokens("a" * 420, model="claude-sonnet-4-5", provider="anthropic")
        assert result == int(420 / 4.2)

    def test_opus_ratio(self) -> None:
        tm = TokenizerManager()
        result = tm.estimate_tokens("a" * 380, model="claude-opus-4-5", provider="anthropic")
        assert result == int(380 / 3.8)


# ---------------------------------------------------------------------------
# Anthropic exact counting via mocked client
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnthropicExactCounting:
    """count_tokens() with an Anthropic client returns exact counts."""

    def test_exact_count(self) -> None:
        client = _make_anthropic_client(token_count=99)
        tm = TokenizerManager(anthropic_client=client)

        messages = [{"role": "user", "content": "hello"}]
        result = tm.count_tokens(messages, model="claude-sonnet-4-5", provider="anthropic")

        assert result == 99
        client.messages.count_tokens.assert_called_once_with(
            model="claude-sonnet-4-5",
            messages=messages,
        )

    def test_exact_count_with_system(self) -> None:
        client = _make_anthropic_client(token_count=120)
        tm = TokenizerManager(anthropic_client=client)

        messages = [{"role": "user", "content": "hello"}]
        result = tm.count_tokens(
            messages,
            model="claude-sonnet-4-5",
            provider="anthropic",
            system="Be helpful",
        )

        assert result == 120
        client.messages.count_tokens.assert_called_once_with(
            model="claude-sonnet-4-5",
            messages=messages,
            system="Be helpful",
        )

    def test_falls_back_on_exception(self) -> None:
        client = MagicMock()
        client.messages.count_tokens.side_effect = RuntimeError("API down")
        tm = TokenizerManager(anthropic_client=client)

        messages = [{"role": "user", "content": "a" * 420}]
        result = tm.count_tokens(messages, model="claude-sonnet-4-5", provider="anthropic")

        # Falls back to length-only heuristic (estimate_tokens_from_length with
        # provider="anthropic" ratio 3.8) — not the model-specific ratio.
        # The SDK's public contract is "approximate pre-flight estimate."
        assert result == pytest.approx(int(420 / 3.8), rel=0.1)

    def test_returns_none_without_client(self) -> None:
        tm = TokenizerManager()
        messages = [{"role": "user", "content": "hello"}]
        result = tm.count_tokens(messages, model="claude-sonnet-4-5", provider="anthropic")
        assert result is None
