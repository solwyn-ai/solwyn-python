# Tests

## Structure

```
tests/
  unit/                   # All unit tests — mock all external deps
    conftest.py           # Shared fixtures + constants (VALID_API_KEY, ALLOW_BUDGET_RESPONSE)
    test_providers/       # Provider adapter tests
  integration/            # Real HTTP against the Solwyn API
    conftest.py           # Auto-bootstraps test user + project + API key
```

No `__init__.py` files anywhere. Import shared constants with `from conftest import ...` (absolute, not relative).

## Markers

Every test must have a category marker. Marker order on methods:

```python
@pytest.mark.unit       # 1. category
@pytest.mark.asyncio    # 2. execution mode
@pytest.mark.parametrize(...)  # 3. parametrize
```

Registered markers: `unit`, `integration`, `chaos`, `performance`, `stress`.

CI runs only `unit`. Integration is opt-in (PR label `run-integration` or `workflow_dispatch`).

## Running Integration Tests

```bash
# In ../core:
make db-setup && make dev-api

# In this repo:
uv run pytest tests/ -m integration -v
```

The integration conftest auto-creates a test user and project via the API. Set `SOLWYN_TEST_API_URL` (default `http://127.0.0.1:8080`), or `SOLWYN_TEST_API_KEY` + `SOLWYN_TEST_PROJECT_ID` to skip bootstrap.

## Conventions

- Arrange-Act-Assert pattern with comments
- Mock at service boundaries, never internal methods
- All mocks must use `spec=` (catches renamed methods)
- Async mocking: `AsyncMock` for the async method, `MagicMock` for the response (httpx Response.json() is sync)
- Shared test constants in `conftest.py` — import, don't redefine
- `_mock_anthropic_client()` exists in two test files with DIFFERENT signatures — do NOT consolidate
- Mock response dicts must include all fields the API returns — no relying on Pydantic defaults
