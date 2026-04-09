"""Tests for circuit breaker state machine."""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from solwyn._types import CircuitState
from solwyn.circuit_breaker import CircuitBreaker, CircuitBreakerState


@pytest.mark.unit
class TestClosedToOpen:
    """CLOSED -> OPEN after N consecutive failures."""

    def test_opens_after_failure_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # 2 < 3

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # resets streak
        assert cb.failure_count == 0

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # only 2 since reset


@pytest.mark.unit
class TestOpenToHalfOpen:
    """OPEN -> HALF_OPEN after recovery_timeout elapses."""

    def test_transitions_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Simulate time passing beyond recovery_timeout
        cb.last_failure_time = time.monotonic() - 15
        assert cb.can_proceed() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_stays_open_before_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_proceed() is False
        assert cb.state == CircuitState.OPEN


@pytest.mark.unit
class TestHalfOpenToClosed:
    """HALF_OPEN -> CLOSED after N successes."""

    def test_closes_after_success_threshold(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0,
            success_threshold=2,
        )
        cb.record_failure()  # -> OPEN
        assert cb.state == CircuitState.OPEN

        cb.can_proceed()  # -> HALF_OPEN (recovery_timeout=0)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # 1 < 2

        cb.record_success()
        assert cb.state == CircuitState.CLOSED


@pytest.mark.unit
class TestHalfOpenFailure:
    """HALF_OPEN -> OPEN on any failure."""

    def test_reopens_on_failure(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0,
            success_threshold=3,
        )
        cb.record_failure()  # -> OPEN
        cb.can_proceed()  # -> HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()  # -> back to OPEN
        assert cb.state == CircuitState.OPEN


@pytest.mark.unit
class TestCanProceed:
    """can_proceed() returns correct value per state."""

    def test_returns_true_when_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.can_proceed() is True

    def test_returns_false_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.record_failure()
        assert cb.can_proceed() is False

    def test_returns_true_when_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        cb.can_proceed()  # triggers HALF_OPEN transition
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_proceed() is True


@pytest.mark.unit
class TestGetState:
    """get_state() returns a well-formed dict."""

    def test_returns_correct_dataclass(self) -> None:
        cb = CircuitBreaker()
        state = cb.get_state()

        assert state.state == CircuitState.CLOSED
        assert state.failure_count == 0
        assert state.success_count == 0
        assert state.last_failure_time is None
        assert isinstance(state.last_state_change, float)

    def test_reflects_mutations(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()

        state = cb.get_state()
        assert state.state == CircuitState.OPEN
        assert state.failure_count == 1
        assert state.last_failure_time is not None


def test_circuit_breaker_state_is_pydantic_model() -> None:
    """CircuitBreakerState must be a Pydantic BaseModel, not a dataclass."""
    assert issubclass(CircuitBreakerState, BaseModel)
