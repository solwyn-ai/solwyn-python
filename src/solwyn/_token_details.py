"""TokenDetails — normalized token usage breakdown.

Normalized token usage breakdown for one LLM call.
"""

from pydantic import BaseModel, ConfigDict, Field


class TokenDetails(BaseModel):
    """Normalized token usage breakdown for one LLM call.

    Provider adapters populate whichever fields their API exposes; the rest
    stay at 0.  The API uses this struct to compute exact costs rather than
    trusting SDK-side estimates.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0, description="Total input tokens (normalized)")
    output_tokens: int = Field(default=0, ge=0, description="Total output tokens (normalized)")
    cached_input_tokens: int = Field(
        default=0, ge=0, description="Input tokens served from prompt cache"
    )
    cache_creation_tokens: int = Field(
        default=0, ge=0, description="Input tokens written to prompt cache (Anthropic)"
    )
    reasoning_tokens: int = Field(
        default=0, ge=0, description="Tokens used for chain-of-thought / thinking"
    )
    audio_input_tokens: int = Field(default=0, ge=0, description="Audio input tokens (OpenAI)")
    audio_output_tokens: int = Field(default=0, ge=0, description="Audio output tokens (OpenAI)")
    accepted_prediction_tokens: int = Field(
        default=0, ge=0, description="Predicted output tokens accepted (OpenAI)"
    )
    rejected_prediction_tokens: int = Field(
        default=0, ge=0, description="Predicted output tokens rejected (OpenAI)"
    )
    tool_use_input_tokens: int = Field(
        default=0, ge=0, description="Tokens used for tool/function definitions (Google)"
    )

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens.  Excluded from serialization."""
        return self.input_tokens + self.output_tokens
