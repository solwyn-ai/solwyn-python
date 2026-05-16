"""Agent-run scope: ``with solwyn.run("name"):``.

Binds an active ``(agent_run_id, agent_run_name)`` pair to a ``ContextVar``
for the duration of a scope. Cost events emitted inside the scope are
tagged with these values in the wire payload; events emitted outside are
tagged with ``None`` and the API synthesizes a per-day fallback id.

The contextvar is the integration seam — no parameter is threaded through
the LLM call path. ``contextvars`` propagation guarantees correct scoping
across async tasks and threads spawned via ``threading.Thread`` (which
copies the calling context).

This module never touches prompt or response content.
"""

from __future__ import annotations

import secrets
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar, Token
from types import TracebackType

# Wire cap on agent_run_name (mirrors MetadataEvent.agent_run_name max_length).
_NAME_MAX_LENGTH = 255

# Single contextvar holding either None or an immutable (id, name) tuple.
# Storing both together makes the swap atomic — no async task can ever
# observe an id without its matching name, even mid-transition.
_active_run: ContextVar[tuple[str, str] | None] = ContextVar("solwyn_active_run", default=None)


def new_run_id() -> str:
    """Generate a fresh agent_run_id.

    Format: ``run_`` + 16 url-safe base64 chars (~96 bits of entropy).
    Comfortably under the 255-char wire cap.
    """
    return "run_" + secrets.token_urlsafe(12)


def current_run() -> tuple[str | None, str | None]:
    """Return the active ``(agent_run_id, agent_run_name)`` or ``(None, None)``.

    Read at metadata-event build time. Returning a tuple (rather than the
    raw ``ContextVar`` value) lets callers unpack without a None-check.
    """
    active = _active_run.get()
    if active is None:
        return (None, None)
    return active


class _RunScope(AbstractContextManager[str], AbstractAsyncContextManager[str]):
    """Context manager returned by ``solwyn.run(name)``.

    Supports both ``with`` and ``async with``. Nesting replaces the outer
    scope for the inner's duration — matches OpenTelemetry span semantics.
    The outer scope is restored automatically on exit via ``ContextVar.reset``.
    """

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("solwyn.run(name) requires a non-empty name")
        if len(name) > _NAME_MAX_LENGTH:
            raise ValueError(
                f"solwyn.run(name) exceeds max length {_NAME_MAX_LENGTH} (got {len(name)})"
            )
        self._name = name
        self._run_id: str | None = None
        self._token: Token[tuple[str, str] | None] | None = None

    def _enter(self) -> str:
        self._run_id = new_run_id()
        self._token = _active_run.set((self._run_id, self._name))
        return self._run_id

    def _exit(self) -> None:
        if self._token is not None:
            _active_run.reset(self._token)
            self._token = None

    def __enter__(self) -> str:
        return self._enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit()

    async def __aenter__(self) -> str:
        return self._enter()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit()


def run(name: str) -> _RunScope:
    """Open an agent-run scope.

    Cost events emitted inside the scope are tagged with a fresh stable
    ``agent_run_id`` and the provided ``name``. Use as a sync or async
    context manager::

        with solwyn.run("nightly-batch") as run_id:
            client.chat.completions.create(...)

        async with solwyn.run("ingest-job") as run_id:
            await aclient.chat.completions.create(...)

    Nesting replaces the outer scope: the inner ``run()`` gets its own id,
    and the outer is restored after the inner ``__exit__``. Sequential
    scopes always get distinct ids — the API aggregates by id, not name.
    """
    return _RunScope(name)
