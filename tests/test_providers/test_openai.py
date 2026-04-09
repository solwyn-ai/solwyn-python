"""Tests for OpenAI provider adapter — token extraction only, no pricing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from solwyn._token_details import TokenDetails
from solwyn.providers.openai import OpenAIAdapter

# ---------------------------------------------------------------------------
# Helpers — build fake OpenAI response objects
# ---------------------------------------------------------------------------


def _chat_response(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    audio_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    audio_output_tokens: int = 0,
    accepted_prediction_tokens: int = 0,
    rejected_prediction_tokens: int = 0,
    include_details: bool = True,
) -> Any:
    """Build a fake Chat Completions API response (prompt_tokens naming)."""
    if include_details:
        prompt_details = SimpleNamespace(
            cached_tokens=cached_tokens,
            audio_tokens=audio_input_tokens,
        )
        completion_details = SimpleNamespace(
            reasoning_tokens=reasoning_tokens,
            audio_tokens=audio_output_tokens,
            accepted_prediction_tokens=accepted_prediction_tokens,
            rejected_prediction_tokens=rejected_prediction_tokens,
        )
    else:
        prompt_details = None
        completion_details = None

    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=prompt_details,
            completion_tokens_details=completion_details,
        )
    )


def _responses_api_response(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> Any:
    """Build a fake Responses API response (input_tokens naming)."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenAIAdapterProtocol:
    def test_satisfies_provider_adapter_protocol(self) -> None:
        from solwyn.providers._protocol import ProviderAdapter

        assert isinstance(OpenAIAdapter(), ProviderAdapter)

    def test_name(self) -> None:
        assert OpenAIAdapter().name == "openai"


@pytest.mark.unit
class TestOpenAIAdapterDetect:
    def test_detect_model_gpt(self) -> None:
        assert OpenAIAdapter().detect_model("gpt-4o") is True

    def test_detect_model_gpt_4(self) -> None:
        assert OpenAIAdapter().detect_model("gpt-4") is True

    def test_detect_model_o3(self) -> None:
        assert OpenAIAdapter().detect_model("o3-mini") is True

    def test_detect_model_o4(self) -> None:
        assert OpenAIAdapter().detect_model("o4-mini") is True

    def test_detect_model_does_not_match_claude(self) -> None:
        assert OpenAIAdapter().detect_model("claude-3-5-sonnet") is False

    def test_detect_model_does_not_match_gemini(self) -> None:
        assert OpenAIAdapter().detect_model("gemini-2.5-flash") is False

    def test_detect_client_openai_module(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "openai.resources"
        assert OpenAIAdapter().detect_client(FakeClient()) is True

    def test_detect_client_non_openai(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "anthropic"
        assert OpenAIAdapter().detect_client(FakeClient()) is False


@pytest.mark.unit
class TestOpenAIAdapterExtractUsageChatCompletions:
    """Chat Completions API uses prompt_tokens / completion_tokens naming."""

    def test_basic_tokens(self) -> None:
        response = _chat_response(prompt_tokens=1000, completion_tokens=500)
        result = OpenAIAdapter().extract_usage(response)
        assert result.input_tokens == 1000
        assert result.output_tokens == 500

    def test_cached_tokens(self) -> None:
        response = _chat_response(
            prompt_tokens=1000,
            completion_tokens=500,
            cached_tokens=400,
        )
        result = OpenAIAdapter().extract_usage(response)
        assert result.cached_input_tokens == 400

    def test_audio_input_tokens(self) -> None:
        response = _chat_response(prompt_tokens=100, audio_input_tokens=50)
        result = OpenAIAdapter().extract_usage(response)
        assert result.audio_input_tokens == 50

    def test_reasoning_tokens(self) -> None:
        response = _chat_response(completion_tokens=300, reasoning_tokens=100)
        result = OpenAIAdapter().extract_usage(response)
        assert result.reasoning_tokens == 100

    def test_audio_output_tokens(self) -> None:
        response = _chat_response(completion_tokens=200, audio_output_tokens=80)
        result = OpenAIAdapter().extract_usage(response)
        assert result.audio_output_tokens == 80

    def test_accepted_prediction_tokens(self) -> None:
        response = _chat_response(accepted_prediction_tokens=150)
        result = OpenAIAdapter().extract_usage(response)
        assert result.accepted_prediction_tokens == 150

    def test_rejected_prediction_tokens(self) -> None:
        response = _chat_response(rejected_prediction_tokens=25)
        result = OpenAIAdapter().extract_usage(response)
        assert result.rejected_prediction_tokens == 25

    def test_missing_detail_sub_objects_returns_zeros(self) -> None:
        """When prompt_tokens_details / completion_tokens_details are None, use zeros."""
        response = _chat_response(
            prompt_tokens=500,
            completion_tokens=200,
            include_details=False,
        )
        result = OpenAIAdapter().extract_usage(response)
        assert result.input_tokens == 500
        assert result.output_tokens == 200
        assert result.cached_input_tokens == 0
        assert result.reasoning_tokens == 0
        assert result.audio_input_tokens == 0
        assert result.audio_output_tokens == 0
        assert result.accepted_prediction_tokens == 0
        assert result.rejected_prediction_tokens == 0

    def test_returns_token_details_instance(self) -> None:
        response = _chat_response(prompt_tokens=10, completion_tokens=5)
        result = OpenAIAdapter().extract_usage(response)
        assert isinstance(result, TokenDetails)

    def test_cache_creation_tokens_always_zero(self) -> None:
        """OpenAI doesn't have cache creation tokens — field stays 0."""
        response = _chat_response(prompt_tokens=100, cached_tokens=50)
        result = OpenAIAdapter().extract_usage(response)
        assert result.cache_creation_tokens == 0

    def test_tool_use_input_tokens_always_zero(self) -> None:
        """OpenAI doesn't report tool_use_input_tokens — field stays 0."""
        response = _chat_response(prompt_tokens=100)
        result = OpenAIAdapter().extract_usage(response)
        assert result.tool_use_input_tokens == 0


@pytest.mark.unit
class TestOpenAIAdapterExtractUsageResponsesAPI:
    """Responses API uses input_tokens / output_tokens naming."""

    def test_basic_tokens(self) -> None:
        response = _responses_api_response(input_tokens=800, output_tokens=300)
        result = OpenAIAdapter().extract_usage(response)
        assert result.input_tokens == 800
        assert result.output_tokens == 300

    def test_cached_tokens(self) -> None:
        response = _responses_api_response(input_tokens=800, cached_tokens=200)
        result = OpenAIAdapter().extract_usage(response)
        assert result.cached_input_tokens == 200

    def test_reasoning_tokens(self) -> None:
        response = _responses_api_response(output_tokens=500, reasoning_tokens=150)
        result = OpenAIAdapter().extract_usage(response)
        assert result.reasoning_tokens == 150


@pytest.mark.unit
class TestOpenAIAdapterExtractUsageNoneHandling:
    def test_none_usage_returns_zeros(self) -> None:
        """When response.usage is None, return all-zero TokenDetails."""
        response = SimpleNamespace(usage=None)
        result = OpenAIAdapter().extract_usage(response)
        assert result == TokenDetails()

    def test_no_usage_attr_returns_zeros(self) -> None:
        """When response has no usage attribute, return all-zero TokenDetails."""
        result = OpenAIAdapter().extract_usage(SimpleNamespace())
        assert result == TokenDetails()
