"""Solwyn SDK exceptions.

BudgetExceededError -- raised when hard-deny mode blocks a request.
ProviderUnavailableError -- raised when all providers are circuit-broken.
ConfigurationError -- raised when configuration is invalid.
"""

from __future__ import annotations


class SolwynError(Exception):
    """Base exception for all Solwyn SDK errors.

    Users can ``except solwyn.SolwynError:`` to catch any error
    produced by the SDK (budget, provider, configuration).
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.args!r})"


class BudgetExceededError(SolwynError):
    """Raised when a request would exceed the configured budget limit.

    Only raised in ``BudgetMode.HARD_DENY`` mode.  In ``ALERT_ONLY`` mode
    the SDK logs a warning instead.

    Attributes:
        budget_limit: The configured spending cap (dollars).
        current_usage: Spending already consumed in the current period.
        estimated_cost: Estimated cost of the blocked request.
        budget_period: The budget window (daily / weekly / monthly).
        mode: The active budget mode when the error was raised.
    """

    def __init__(
        self,
        message: str,
        *,
        budget_limit: float,
        current_usage: float,
        estimated_cost: float,
        budget_period: str,
        mode: str,
    ) -> None:
        super().__init__(message)
        self.budget_limit = budget_limit
        self.current_usage = current_usage
        self.estimated_cost = estimated_cost
        self.budget_period = budget_period
        self.mode = mode

    def __repr__(self) -> str:
        return (
            f"BudgetExceededError("
            f"budget_limit={self.budget_limit!r}, "
            f"current_usage={self.current_usage!r})"
        )


class ProviderUnavailableError(SolwynError):
    """Raised when a provider's circuit breaker is open.

    Attributes:
        provider: Name of the unavailable provider (e.g. ``"openai"``).
        circuit_state: Current circuit breaker state string.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        circuit_state: str,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.circuit_state = circuit_state


class ConfigurationError(SolwynError):
    """Raised when SDK configuration is invalid or incomplete.

    Attributes:
        field: The configuration field that failed validation (may be ``None``).
        message: Human-readable description of the problem.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.message = message
