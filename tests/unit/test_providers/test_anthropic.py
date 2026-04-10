"""Tests for Anthropic provider adapter — token extraction only, no pricing.

Key subtlety: Anthropic's input_tokens, cache_read_input_tokens, and
cache_creation_input_tokens are SEPARATE additive fields, not subsets.
Our normalized input_tokens is the sum of all three.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from solwyn._token_details import TokenDetails
from solwyn.providers.anthropic import AnthropicAdapter

# ---------------------------------------------------------------------------
# Helpers — build fake Anthropic response objects
# ---------------------------------------------------------------------------


def _anthropic_response(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    include_cache_fields: bool = True,
) -> Any:
    """Build a fake Anthropic messages.create() response.

    By default includes cache fields (even as 0). Set include_cache_fields=False
    to simulate older responses that have no cache fields at all.
    """
    if include_cache_fields:
        usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        )
    else:
        usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    return SimpleNamespace(usage=usage)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnthropicAdapterProtocol:
    def test_satisfies_provider_adapter_protocol(self) -> None:
        from solwyn.providers._protocol import ProviderAdapter

        assert isinstance(AnthropicAdapter(), ProviderAdapter)

    def test_name(self) -> None:
        assert AnthropicAdapter().name == "anthropic"


@pytest.mark.unit
class TestAnthropicAdapterDetect:
    def test_detect_model_claude_3(self) -> None:
        assert AnthropicAdapter().detect_model("claude-3-5-sonnet") is True

    def test_detect_model_claude_opus(self) -> None:
        assert AnthropicAdapter().detect_model("claude-opus-4-6") is True

    def test_detect_model_claude_haiku(self) -> None:
        assert AnthropicAdapter().detect_model("claude-haiku-3-5") is True

    def test_detect_model_does_not_match_gpt(self) -> None:
        assert AnthropicAdapter().detect_model("gpt-4o") is False

    def test_detect_model_does_not_match_gemini(self) -> None:
        assert AnthropicAdapter().detect_model("gemini-2.5-flash") is False

    def test_detect_client_anthropic_module(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "anthropic.resources"
        assert AnthropicAdapter().detect_client(FakeClient()) is True

    def test_detect_client_non_anthropic(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "openai"
        assert AnthropicAdapter().detect_client(FakeClient()) is False


@pytest.mark.unit
class TestAnthropicAdapterExtractUsage:
    def test_basic_tokens(self) -> None:
        response = _anthropic_response(input_tokens=1000, output_tokens=500)
        result = AnthropicAdapter().extract_usage(response)
        assert result.output_tokens == 500

    def test_input_tokens_normalized_to_sum_of_all_three(self) -> None:
        """input_tokens = base + cache_read + cache_creation (all additive)."""
        response = _anthropic_response(
            input_tokens=1000,
            cache_read_input_tokens=300,
            cache_creation_input_tokens=200,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.input_tokens == 1500  # 1000 + 300 + 200

    def test_base_input_only_no_cache(self) -> None:
        """With no cache, input_tokens == base."""
        response = _anthropic_response(input_tokens=800)
        result = AnthropicAdapter().extract_usage(response)
        assert result.input_tokens == 800

    def test_cache_read_mapped_to_cached_input_tokens(self) -> None:
        response = _anthropic_response(
            input_tokens=1000,
            cache_read_input_tokens=400,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.cached_input_tokens == 400

    def test_cache_creation_mapped_to_cache_creation_tokens(self) -> None:
        response = _anthropic_response(
            input_tokens=1000,
            cache_creation_input_tokens=250,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.cache_creation_tokens == 250

    def test_reasoning_tokens_always_zero(self) -> None:
        """Anthropic folds thinking tokens into output_tokens — documented blind spot."""
        response = _anthropic_response(input_tokens=1000, output_tokens=600)
        result = AnthropicAdapter().extract_usage(response)
        assert result.reasoning_tokens == 0

    def test_full_cache_response(self) -> None:
        """All cache fields present: normalized input = sum of all three."""
        response = _anthropic_response(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=400,
            cache_creation_input_tokens=100,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.input_tokens == 1500  # 1000 + 400 + 100
        assert result.output_tokens == 500
        assert result.cached_input_tokens == 400
        assert result.cache_creation_tokens == 100

    def test_missing_cache_fields_graceful(self) -> None:
        """Older Anthropic responses without cache fields return zeros for those."""
        response = _anthropic_response(
            input_tokens=500,
            output_tokens=200,
            include_cache_fields=False,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.input_tokens == 500
        assert result.output_tokens == 200
        assert result.cached_input_tokens == 0
        assert result.cache_creation_tokens == 0

    def test_returns_token_details_instance(self) -> None:
        response = _anthropic_response(input_tokens=10, output_tokens=5)
        result = AnthropicAdapter().extract_usage(response)
        assert isinstance(result, TokenDetails)

    def test_audio_tokens_always_zero(self) -> None:
        """Anthropic doesn't have audio token fields."""
        response = _anthropic_response(input_tokens=100, output_tokens=50)
        result = AnthropicAdapter().extract_usage(response)
        assert result.audio_input_tokens == 0
        assert result.audio_output_tokens == 0

    def test_prediction_tokens_always_zero(self) -> None:
        """Anthropic doesn't have predicted output token fields."""
        response = _anthropic_response(input_tokens=100, output_tokens=50)
        result = AnthropicAdapter().extract_usage(response)
        assert result.accepted_prediction_tokens == 0
        assert result.rejected_prediction_tokens == 0

    def test_tool_use_input_tokens_always_zero(self) -> None:
        """Anthropic doesn't report tool_use_input_tokens."""
        response = _anthropic_response(input_tokens=100, output_tokens=50)
        result = AnthropicAdapter().extract_usage(response)
        assert result.tool_use_input_tokens == 0


@pytest.mark.unit
class TestAnthropicAdapterNoneHandling:
    def test_none_usage_returns_zeros(self) -> None:
        """When response.usage is None, return all-zero TokenDetails."""
        response = SimpleNamespace(usage=None)
        result = AnthropicAdapter().extract_usage(response)
        assert result == TokenDetails()

    def test_no_usage_attr_returns_zeros(self) -> None:
        """When response has no usage attribute, return all-zero TokenDetails."""
        result = AnthropicAdapter().extract_usage(SimpleNamespace())
        assert result == TokenDetails()
