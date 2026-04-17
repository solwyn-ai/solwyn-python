"""Shared sans-I/O logic for Solwyn clients.

Contains _SolwynBase with config, budget logic, metadata formatting,
and pricing calculations. No I/O -- sync and async clients inherit
from this and add their own HTTP layer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from solwyn._token_details import TokenDetails
from solwyn._types import CallStatus, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.config import SolwynConfig
from solwyn.exceptions import ProviderUnavailableError
from solwyn.tokenizer import TokenizerManager


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
        project_id: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        token_details: TokenDetails | None,
        latency_ms: float,
        status: CallStatus,
        is_failover: bool,
        sdk_instance_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> MetadataEvent:
        """Build a MetadataEvent for reporting to the cloud API."""
        return MetadataEvent(
            project_id=project_id,
            model=model,
            provider=ProviderName(provider),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_details=token_details,
            latency_ms=latency_ms,
            status=status,
            is_failover=is_failover,
            sdk_instance_id=sdk_instance_id or self._sdk_instance_id,
            timestamp=timestamp or datetime.now(UTC),
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
