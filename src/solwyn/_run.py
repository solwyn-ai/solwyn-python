"""Agent-run scope: ``with solwyn.run("name"):``.

Binds an active ``(agent_run_id, agent_run_name)`` pair to a ``ContextVar``
for the duration of a scope. Cost events emitted inside the scope are
tagged with these values in the wire payload; events emitted outside are
tagged with ``None`` and the API synthesizes a per-day fallback id.

The contextvar is the integration seam — no parameter is threaded through
the LLM call path. ``contextvars`` propagation guarantees correct scoping
across asyncio tasks. Threads and ``ThreadPoolExecutor`` workers do not
inherit the active run reliably; use ``run_in_executor(...)`` or
``contextvars.copy_context().run(...)`` when submitting threaded work.

This module never touches prompt or response content.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from concurrent.futures import Executor, Future
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar, Token, copy_context
from dataclasses import dataclass
from types import TracebackType
from typing import ParamSpec, TypeVar

from solwyn._constants import AGENT_RUN_NAME_MAX_LENGTH

_P = ParamSpec("_P")
_T = TypeVar("_T")

# Single contextvar holding either None or an immutable (id, name) tuple.
# Storing both together makes the swap atomic — no async task can ever
# observe an id without its matching name, even mid-transition.
_active_run: ContextVar[tuple[str, str] | None] = ContextVar("solwyn_active_run", default=None)


@dataclass(frozen=True)
class _RunFrame:
    run_id: str
    token: Token[tuple[str, str] | None]


def _new_run_id() -> str:
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
        if not name.strip():
            raise ValueError("solwyn.run(name) requires a non-empty name")
        if any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ValueError("solwyn.run(name) must not contain control characters")
        if len(name) > AGENT_RUN_NAME_MAX_LENGTH:
            raise ValueError(
                f"solwyn.run(name) exceeds max length {AGENT_RUN_NAME_MAX_LENGTH} (got {len(name)})"
            )
        self._name = name
        self._frames: ContextVar[tuple[_RunFrame, ...]] = ContextVar(
            f"solwyn_run_scope_frames_{id(self)}",
            default=(),
        )

    def _enter(self) -> str:
        run_id = _new_run_id()
        token = _active_run.set((run_id, self._name))
        frames = self._frames.get()
        self._frames.set((*frames, _RunFrame(run_id=run_id, token=token)))
        return run_id

    def _exit(self) -> None:
        frames = self._frames.get()
        if not frames:
            return
        frame = frames[-1]
        try:
            _active_run.reset(frame.token)
        except ValueError:
            # Async generator finalizers can run in a different Context than
            # __aenter__. A token cannot be reset from there, so avoid
            # surfacing a cleanup-time exception.
            pass
        finally:
            self._frames.set(frames[:-1])

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


def run_in_executor(
    executor: Executor,
    fn: Callable[_P, _T],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> Future[_T]:
    """Submit ``fn`` to an executor with the active context propagated.

    ``ThreadPoolExecutor`` does not copy ``ContextVar`` values into worker
    threads. This helper preserves the active ``solwyn.run(...)`` tag for
    threaded work by wrapping the callable in ``copy_context().run``.
    """
    ctx = copy_context()

    def run_with_context() -> _T:
        return ctx.run(fn, *args, **kwargs)

    return executor.submit(run_with_context)


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
