"""Tests for env-var config loading and credential validation via Solwyn constructors.

Covers two doc-pattern gaps identified during the SDK docs audit:
1. Zero-config construction via SOLWYN_* env vars (Quickstart pattern)
2. ConfigurationError from malformed credentials (Installation pattern)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID
from pydantic import ValidationError

from solwyn._types import BudgetMode
from solwyn.client import Solwyn
from solwyn.exceptions import ConfigurationError, SolwynError


@pytest.fixture(autouse=True)
def _clean_solwyn_env(monkeypatch):
    """Remove all SOLWYN_* env vars to prevent test pollution from host environment."""
    for key in list(os.environ):
        if key.startswith("SOLWYN_"):
            monkeypatch.delenv(key)


def _mock_openai_client():
    """Create a mock that looks like openai.OpenAI() for provider detection."""
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    return client


def _make_solwyn(client, **config_kwargs):
    """Create a Solwyn wrapper with mocked reporter thread."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **config_kwargs)
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    return solwyn


# ---------------------------------------------------------------------------
# Test 1: Environment-variable-only construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvVarConstruction:
    """Solwyn(client) loads config from SOLWYN_* env vars when no kwargs given."""

    def test_env_vars_populate_config(self, monkeypatch) -> None:
        """Both SOLWYN_API_KEY and SOLWYN_PROJECT_ID in env -> successful construction."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)
        monkeypatch.setenv("SOLWYN_PROJECT_ID", VALID_PROJECT_ID)

        client = _mock_openai_client()
        solwyn = _make_solwyn(client)

        assert solwyn._config.api_key == VALID_API_KEY
        assert solwyn._config.project_id == VALID_PROJECT_ID

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_kwargs_override_env_vars(self, monkeypatch) -> None:
        """Constructor kwargs take precedence over env vars."""
        env_key = "sk_solwyn_" + "b" * 32
        env_project = "proj_" + "b" * 8
        monkeypatch.setenv("SOLWYN_API_KEY", env_key)
        monkeypatch.setenv("SOLWYN_PROJECT_ID", env_project)

        client = _mock_openai_client()
        solwyn = _make_solwyn(client, api_key=VALID_API_KEY, project_id=VALID_PROJECT_ID)

        assert solwyn._config.api_key == VALID_API_KEY
        assert solwyn._config.project_id == VALID_PROJECT_ID

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_missing_api_key_env_var_raises(self, monkeypatch) -> None:
        """No SOLWYN_API_KEY in env and no kwarg -> ValidationError."""
        monkeypatch.setenv("SOLWYN_PROJECT_ID", VALID_PROJECT_ID)

        client = _mock_openai_client()
        with pytest.raises(ValidationError):
            _make_solwyn(client)

    def test_missing_project_id_env_var_raises(self, monkeypatch) -> None:
        """No SOLWYN_PROJECT_ID in env and no kwarg -> ValidationError."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)

        client = _mock_openai_client()
        with pytest.raises(ValidationError):
            _make_solwyn(client)

    def test_api_url_loads_from_env(self, monkeypatch) -> None:
        """SOLWYN_API_URL env var overrides the default api_url."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)
        monkeypatch.setenv("SOLWYN_PROJECT_ID", VALID_PROJECT_ID)
        monkeypatch.setenv("SOLWYN_API_URL", "https://custom.solwyn.ai")

        client = _mock_openai_client()
        solwyn = _make_solwyn(client)

        assert solwyn._config.api_url == "https://custom.solwyn.ai"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_budget_mode_loads_from_env(self, monkeypatch) -> None:
        """SOLWYN_BUDGET_MODE env var overrides the default budget_mode."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)
        monkeypatch.setenv("SOLWYN_PROJECT_ID", VALID_PROJECT_ID)
        monkeypatch.setenv("SOLWYN_BUDGET_MODE", "hard_deny")

        client = _mock_openai_client()
        solwyn = _make_solwyn(client)

        assert solwyn._config.budget_mode == BudgetMode.HARD_DENY

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    @pytest.mark.parametrize(
        ("env_val", "expected"),
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_fail_open_boolean_coercion(self, monkeypatch, env_val: str, expected: bool) -> None:
        """SOLWYN_FAIL_OPEN coerces string values to booleans."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)
        monkeypatch.setenv("SOLWYN_PROJECT_ID", VALID_PROJECT_ID)
        monkeypatch.setenv("SOLWYN_FAIL_OPEN", env_val)

        client = _mock_openai_client()
        solwyn = _make_solwyn(client)

        assert solwyn._config.fail_open is expected

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


# ---------------------------------------------------------------------------
# Test 2: ConfigurationError from malformed credentials
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigurationErrorFromBadCredentials:
    """Malformed api_key or project_id raises ConfigurationError with correct attributes."""

    def test_bad_api_key_prefix_raises_configuration_error(self) -> None:
        """api_key without sk_solwyn_ prefix -> ConfigurationError(field='api_key')."""
        client = _mock_openai_client()
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key="bad_key", project_id=VALID_PROJECT_ID)
        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_api_key_too_short_raises_configuration_error(self) -> None:
        """api_key with fewer than 32 chars after prefix -> ConfigurationError."""
        client = _mock_openai_client()
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key="sk_solwyn_short", project_id=VALID_PROJECT_ID)
        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_api_key_too_long_raises_configuration_error(self) -> None:
        """api_key with more than 64 chars after prefix -> ConfigurationError."""
        client = _mock_openai_client()
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key="sk_solwyn_" + "a" * 65, project_id=VALID_PROJECT_ID)
        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_empty_api_key_raises_configuration_error(self) -> None:
        """Empty string api_key -> ConfigurationError(field='api_key')."""
        client = _mock_openai_client()
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key="", project_id=VALID_PROJECT_ID)
        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_bad_project_id_too_short_raises_configuration_error(self) -> None:
        """project_id with fewer than 8 chars after prefix -> ConfigurationError."""
        client = _mock_openai_client()
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key=VALID_API_KEY, project_id="proj_" + "a" * 7)
        assert exc_info.value.field == "project_id"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_empty_project_id_raises_configuration_error(self) -> None:
        """Empty string project_id -> ConfigurationError(field='project_id')."""
        client = _mock_openai_client()
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key=VALID_API_KEY, project_id="")
        assert exc_info.value.field == "project_id"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_unicode_homograph_api_key_raises_configuration_error(self) -> None:
        """Unicode homograph in api_key (Cyrillic a) -> ConfigurationError."""
        client = _mock_openai_client()
        bad_key = "sk_solwyn_" + "\u0430" * 32  # Cyrillic 'а' (U+0430)
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key=bad_key, project_id=VALID_PROJECT_ID)
        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_path_traversal_project_id_raises_configuration_error(self) -> None:
        """Path traversal in project_id -> ConfigurationError(field='project_id')."""
        client = _mock_openai_client()
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key=VALID_API_KEY, project_id="proj_../etc")
        assert exc_info.value.field == "project_id"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_configuration_error_catchable_as_solwyn_error(self) -> None:
        """ConfigurationError can be caught as SolwynError."""
        client = _mock_openai_client()
        with pytest.raises(SolwynError):
            Solwyn(client, api_key="bad_key", project_id=VALID_PROJECT_ID)
