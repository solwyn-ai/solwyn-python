"""Vendored enums and wire-format models for SDK <-> API contracts.

Pydantic models for API request/response contracts.
Excludes API-internal types: ProjectConfig, ProviderHealth,
NotificationEventType, Environment, BudgetPeriod.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from solwyn._token_details import TokenDetails

# ── Enums ────────────────────────────────────────────────────────────────


class BudgetMode(StrEnum):
    """How the SDK reacts when a budget limit is reached."""

    ALERT_ONLY = "alert_only"
    HARD_DENY = "hard_deny"


class CircuitState(StrEnum):
    """Circuit breaker states for provider health tracking."""

    CLOSED = "closed"  # Normal operation — requests flow through
    OPEN = "open"  # Failing — reject requests, try fallback
    HALF_OPEN = "half_open"  # Testing recovery — allow probe requests


class ProviderName(StrEnum):
    """Supported LLM provider identifiers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class CallStatus(StrEnum):
    """Outcome status for LLM call metadata events."""

    SUCCESS = "success"
    ERROR = "error"
    BUDGET_DENIED = "budget_denied"


# ── Wire-format models ──────────────────────────────────────────────────


class MetadataEvent(BaseModel):
    """Telemetry event sent from SDK to API after each LLM call.

    Contains token/latency metadata only — never prompts, responses, or
    SDK-computed costs.
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(..., description="Project identifier (proj_...)")
    model: str = Field(..., max_length=100, description="LLM model name (e.g. gpt-4o)")
    provider: ProviderName = Field(..., description="LLM provider")
    input_tokens: int = Field(..., ge=0, description="Input token count")
    output_tokens: int = Field(..., ge=0, description="Output token count")
    token_details: TokenDetails | None = Field(
        None, description="Full token breakdown from provider adapter"
    )
    latency_ms: float = Field(..., description="End-to-end call latency in ms")
    status: CallStatus = Field(..., description="Call outcome")
    is_failover: bool = Field(..., description="Whether this call used a fallback provider")
    sdk_instance_id: str = Field(..., description="Unique SDK instance identifier")
    timestamp: datetime = Field(..., description="When the LLM call completed (UTC)")


class BudgetCheckRequest(BaseModel):
    """Pre-flight budget check sent before an LLM call."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(..., description="Project identifier (proj_...)")
    estimated_input_tokens: int = Field(
        ..., ge=0, description="Estimated input token count for the pending call"
    )
    model: str = Field(..., max_length=100, description="LLM model name")
    provider: ProviderName = Field(..., description="Target provider")


class BudgetCheckResponse(BaseModel):
    """API response to a budget check request."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool = Field(..., description="Whether the call is within budget")
    remaining_budget: float = Field(..., description="Remaining budget in USD for current period")
    reservation_id: str | None = Field(
        None, description="Budget reservation ID (for cost reconciliation)"
    )
    mode: BudgetMode = Field(..., description="Current budget enforcement mode")
    budget_limit: float = Field(..., description="Total budget limit for current period in USD")
    current_usage: float = Field(..., description="Current spend in USD for this period")
    denied_by_period: str | None = Field(
        ..., description="Which budget period triggered denial (e.g. 'daily')"
    )


class BudgetConfirmRequest(BaseModel):
    """Post-call budget confirmation sent after an LLM call completes."""

    model_config = ConfigDict(extra="forbid")

    reservation_id: str = Field(
        ..., description="Budget reservation ID returned by BudgetCheckResponse"
    )
    model: str = Field(..., max_length=100, description="LLM model name used for the call")
    token_details: TokenDetails = Field(
        ..., description="Actual token breakdown from the provider adapter"
    )
