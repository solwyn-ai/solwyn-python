"""Solwyn -- AI Agent Control Plane SDK.

Drop-in wrapper for ``openai.OpenAI`` and ``anthropic.Anthropic`` clients
that adds hard spending caps, automatic provider failover, and per-agent
cost attribution -- without ever seeing customer prompts.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("solwyn")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.config import SolwynConfig
from solwyn.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    ProviderUnavailableError,
    SolwynError,
)

__all__ = [
    "__version__",
    "Solwyn",
    "AsyncSolwyn",
    "SolwynConfig",
    "SolwynError",
    "BudgetExceededError",
    "ProviderUnavailableError",
    "ConfigurationError",
]
