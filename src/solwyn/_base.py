"""Shared sans-I/O logic for Solwyn clients.

Contains _SolwynBase with config, budget logic, metadata formatting,
and pricing calculations. No I/O -- sync and async clients inherit
from this and add their own HTTP layer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from solwyn._token_details import TokenDetails
from solwyn._types import CallStatus, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.config import SolwynConfig
from solwyn.exceptions import ProviderUnavailableError
from solwyn.tokenizer import TokenizerManager


class _AttemptContext(BaseModel):
    """Per-attempt state for an intercepted LLM call.

    Immutable. On retry-success, use ``model_copy(update=...)`` to atomically
    replace all fields that change — prevents the silent-drift bug where a
    future contributor forgets to update one of several loose locals.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    model: str
    kwargs: dict[str, object]
    start_time: float
    is_model_fallback: bool


class _SolwynBase:
    """Shared sans-I/O base class for Solwyn sync and async clients.

    Provides:
    - Token estimation and cost calculation
    - Metadata event construction
    - Budget request construction
    - Circuit breaker management and provider selection
    - SDK instance identity
    """

    def __init__(self, config: SolwynConfig) -> None:
        self._config = config
        self._sdk_instance_id = str(uuid.uuid4())
        self._tokenizer = TokenizerManager()

        # One circuit breaker per configured provider. Additional providers
        # get lazily-created breakers via _get_circuit_breaker.
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            config.primary_provider.value: CircuitBreaker(
                failure_threshold=config.circuit_breaker_failure_threshold,
                recovery_timeout=config.circuit_breaker_recovery_timeout,
                success_threshold=config.circuit_breaker_success_threshold,
            )
        }

    def _build_metadata_event(
        self,
        *,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        token_details: TokenDetails | None,
        latency_ms: float,
        status: CallStatus,
        is_model_fallback: bool,
        sdk_instance_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> MetadataEvent:
        """Build a MetadataEvent for reporting to the cloud API."""
        return MetadataEvent(
            model=model,
            provider=ProviderName(provider),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_details=token_details,
            latency_ms=latency_ms,
            status=status,
            is_model_fallback=is_model_fallback,
            sdk_instance_id=sdk_instance_id or self._sdk_instance_id,
            timestamp=timestamp or datetime.now(UTC),
        )

    def _build_error_event(
        self,
        *,
        model: str,
        provider: str,
        latency_ms: float,
        is_model_fallback: bool,
    ) -> MetadataEvent:
        """Build an error-status MetadataEvent with zeroed token counts.

        Convenience wrapper for the retry/dispatch-failure paths where
        token_details is unavailable and status is always ERROR.
        """
        return self._build_metadata_event(
            model=model,
            provider=provider,
            input_tokens=0,
            output_tokens=0,
            token_details=None,
            latency_ms=latency_ms,
            status=CallStatus.ERROR,
            is_model_fallback=is_model_fallback,
        )

    def _get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get the circuit breaker for a provider.

        Lazily creates a circuit breaker if one doesn't exist for this provider.
        """
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = CircuitBreaker(
                failure_threshold=self._config.circuit_breaker_failure_threshold,
                recovery_timeout=self._config.circuit_breaker_recovery_timeout,
                success_threshold=self._config.circuit_breaker_success_threshold,
            )
        return self._circuit_breakers[provider]

    def _select_provider(self) -> str:
        """Select the primary provider via its circuit breaker.

        Returns the primary provider name if its circuit is CLOSED or HALF_OPEN.
        Raises ProviderUnavailableError if the circuit is OPEN. Same-provider
        model fallback is handled at dispatch time in client.py, not here.

        Returns:
            The primary provider name (e.g. "openai" or "anthropic").

        Raises:
            ProviderUnavailableError: If the primary provider's circuit is open.
        """
        primary = self._config.primary_provider.value
        primary_cb = self._get_circuit_breaker(primary)

        if primary_cb.can_proceed():
            return primary

        raise ProviderUnavailableError(
            f"Provider {primary} is unavailable (circuit open)",
            provider=primary,
            circuit_state=primary_cb.state.value,
        )

    def _should_retry_with_fallback(self, model: str) -> bool:
        """Return True if the primary call should be retried with fallback_model.

        Guards against infinite retry loops by refusing to retry when the
        primary call is already targeting the fallback model.
        """
        fm = self._config.fallback_model
        return fm is not None and model != fm

    def _prepare_fallback_kwargs(self, kwargs: dict[str, object]) -> dict[str, object]:
        """Return a shallow copy of kwargs with model swapped to fallback_model.

        Does not mutate the input. Caller must have verified
        _should_retry_with_fallback first.
        """
        if self._config.fallback_model is None:
            raise RuntimeError(
                "fallback_model is not configured — caller should have checked "
                "_should_retry_with_fallback() first"
            )
        swapped = dict(kwargs)
        swapped["model"] = self._config.fallback_model
        return swapped
