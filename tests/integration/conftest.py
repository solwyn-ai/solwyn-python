"""Shared fixtures for integration tests.

Bootstraps test credentials by signing up a throwaway user and creating
a project via the Solwyn API.  Falls back to env vars when pre-provisioned
credentials are available (CI).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import httpx
import pytest

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter


@dataclass(frozen=True)
class Credentials:
    """Credentials bootstrapped for an integration test session."""

    api_url: str
    api_key: str
    project_id: str


@pytest.fixture(scope="session")
def api_url() -> str:
    """Solwyn API base URL."""
    return os.environ.get("SOLWYN_TEST_API_URL", "http://127.0.0.1:8080")


@pytest.fixture(scope="session", autouse=True)
def require_api(api_url: str) -> None:
    """Skip the entire session if the Solwyn API is unreachable."""
    try:
        r = httpx.get(f"{api_url}/health", timeout=3)
        r.raise_for_status()
    except (httpx.HTTPError, OSError):
        pytest.skip("Solwyn API not available")


def _bootstrap_credentials(api_url: str) -> Credentials:
    """Sign up a throwaway user, create a project, return credentials."""
    session_id = uuid.uuid4().hex[:12]
    email = f"sdk-test-{session_id}@example.com"
    password = f"TestPass!{session_id}"

    with httpx.Client(base_url=api_url, timeout=10) as http:
        # Sign up (fall back to login if user already exists)
        r = http.post(
            "/api/v1/auth/signup",
            json={"email": email, "password": password},
        )
        if r.status_code == 409:
            r = http.post(
                "/api/v1/auth/login",
                json={"email": email, "password": password},
            )
        r.raise_for_status()
        token = r.json()["access_token"]

        # Create project (auto-generates first API key)
        auth = {"Authorization": f"Bearer {token}"}
        r = http.post(
            "/api/v1/projects",
            json={
                "name": f"sdk-integ-{session_id}",
                "budget_limit": 100.0,
                "budget_period": "monthly",
                "budget_mode": "alert_only",
            },
            headers=auth,
        )
        r.raise_for_status()
        project = r.json()

        return Credentials(
            api_url=api_url,
            api_key=project["api_key"],
            project_id=project["id"],
        )


@pytest.fixture(scope="session")
def test_credentials(api_url: str) -> Credentials:
    """Session-scoped test credentials.

    Uses env vars if both SOLWYN_TEST_API_KEY and SOLWYN_TEST_PROJECT_ID
    are set, otherwise bootstraps via the API.
    """
    env_key = os.environ.get("SOLWYN_TEST_API_KEY")
    env_project = os.environ.get("SOLWYN_TEST_PROJECT_ID")

    if env_key and env_project:
        return Credentials(
            api_url=api_url,
            api_key=env_key,
            project_id=env_project,
        )

    return _bootstrap_credentials(api_url)


# ---------------------------------------------------------------------------
# SDK component fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def budget_enforcer(test_credentials: Credentials) -> BudgetEnforcer:
    """Sync BudgetEnforcer pointed at the live API."""
    enforcer = BudgetEnforcer(
        project_id=test_credentials.project_id,
        api_url=test_credentials.api_url,
        api_key=test_credentials.api_key,
        budget_mode=BudgetMode.ALERT_ONLY,
        fail_open=True,
    )
    yield enforcer
    enforcer.close()


@pytest.fixture
async def async_budget_enforcer(test_credentials: Credentials) -> AsyncBudgetEnforcer:
    """Async BudgetEnforcer pointed at the live API."""
    enforcer = AsyncBudgetEnforcer(
        project_id=test_credentials.project_id,
        api_url=test_credentials.api_url,
        api_key=test_credentials.api_key,
        budget_mode=BudgetMode.ALERT_ONLY,
        fail_open=True,
    )
    yield enforcer
    await enforcer.close()


@pytest.fixture
def metadata_reporter(test_credentials: Credentials) -> MetadataReporter:
    """Sync MetadataReporter pointed at the live API."""
    reporter = MetadataReporter(
        api_url=test_credentials.api_url,
        api_key=test_credentials.api_key,
        flush_interval=1.0,
    )
    yield reporter
    reporter.close()


@pytest.fixture
async def async_metadata_reporter(test_credentials: Credentials) -> AsyncMetadataReporter:
    """Async MetadataReporter pointed at the live API."""
    reporter = AsyncMetadataReporter(
        api_url=test_credentials.api_url,
        api_key=test_credentials.api_key,
        flush_interval=1.0,
    )
    await reporter.start()
    yield reporter
    await reporter.close()


# Reusable token details for confirm calls
SAMPLE_TOKEN_DETAILS = TokenDetails(input_tokens=100, output_tokens=50)
