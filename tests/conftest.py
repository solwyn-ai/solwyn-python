"""Shared test fixtures for Solwyn SDK tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from solwyn.circuit_breaker import CircuitBreaker
from solwyn.config import SolwynConfig

# Valid credentials that pass format validation
VALID_API_KEY = "sk_solwyn_" + "a" * 32
VALID_PROJECT_ID = "proj_abc12345"

# Standard allow response for budget mock patching
ALLOW_BUDGET_RESPONSE = {
    "allowed": True,
    "remaining_budget": 80.0,
    "reservation_id": "res_123",
    "mode": "alert_only",
    "budget_limit": 100.0,
    "current_usage": 20.0,
    "denied_by_period": None,
}


@pytest.fixture
def mock_httpx_client():
    """Return a mocked httpx.Client."""
    import httpx

    return MagicMock(spec=httpx.Client)


@pytest.fixture
def mock_async_httpx_client():
    """Return a mocked httpx.AsyncClient."""
    import httpx

    return MagicMock(spec=httpx.AsyncClient)


@pytest.fixture
def solwyn_config():
    """Return a SolwynConfig with test defaults."""
    return SolwynConfig(
        api_key="sk_solwyn_test1234567890abcdef1234567890abcdef",
        project_id="proj_test12345678",
    )


@pytest.fixture
def circuit_breaker():
    """Return a CircuitBreaker with default thresholds."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=60,
        success_threshold=2,
    )
