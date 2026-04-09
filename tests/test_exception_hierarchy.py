"""SDK exceptions must share a base class and be importable from the package root."""

from __future__ import annotations

from solwyn import (
    BudgetExceededError,
    ConfigurationError,
    ProviderUnavailableError,
    SolwynError,
)


def test_solwyn_error_is_importable_from_package_root() -> None:
    import solwyn

    assert hasattr(solwyn, "SolwynError"), "solwyn.SolwynError must be exported"


def test_all_sdk_exceptions_inherit_from_solwyn_error() -> None:
    assert issubclass(BudgetExceededError, SolwynError)
    assert issubclass(ProviderUnavailableError, SolwynError)
    assert issubclass(ConfigurationError, SolwynError)


def test_solwyn_error_catches_all_families() -> None:
    cases: list[tuple[type[SolwynError], dict[str, object]]] = [
        (
            BudgetExceededError,
            {
                "budget_limit": 100.0,
                "current_usage": 120.0,
                "estimated_cost": 5.0,
                "budget_period": "daily",
                "mode": "hard_deny",
            },
        ),
        (ProviderUnavailableError, {"provider": "openai", "circuit_state": "open"}),
        (ConfigurationError, {"field": "api_key"}),
    ]
    for exc_class, kwargs in cases:
        try:
            raise exc_class("test", **kwargs)  # type: ignore[arg-type]
        except SolwynError:
            pass  # expected
        except Exception as err:
            raise AssertionError(f"{exc_class.__name__} did not match SolwynError") from err


def test_exceptions_have_useful_repr() -> None:
    exc = BudgetExceededError(
        "over budget",
        budget_limit=100.0,
        current_usage=120.0,
        estimated_cost=5.0,
        budget_period="daily",
        mode="hard_deny",
    )
    rep = repr(exc)
    assert "BudgetExceededError" in rep
    assert "100" in rep
