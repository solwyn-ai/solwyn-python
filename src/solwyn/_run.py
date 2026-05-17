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
Do not open a run scope inside an async generator; async generator yields
share the consumer's context and would leak the generator's run into the
consumer body.

This module never touches prompt or response content.
"""

from __future__ import annotations

import inspect
import sys
import unicodedata
import uuid
from collections.abc import Callable
from concurrent.futures import Executor, Future
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar, Token, copy_context
from dataclasses import dataclass
from types import FrameType, TracebackType
from typing import ParamSpec, TypeVar

from solwyn._constants import AGENT_RUN_NAME_MAX_LENGTH

_P = ParamSpec("_P")
_T = TypeVar("_T")

# Single contextvar holding either None or an immutable (id, name) tuple.
# Storing both together makes the swap atomic — no async task can ever
# observe an id without its matching name, even mid-transition.
_active_run: ContextVar[tuple[str, str] | None] = ContextVar("solwyn_active_run", default=None)

_DISALLOWED_NAME_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


@dataclass(frozen=True)
class _RunFrame:
    scope_id: int
    token: Token[tuple[str, str] | None]
    prior_active: tuple[str, str] | None


_run_frames: ContextVar[tuple[_RunFrame, ...]] = ContextVar("solwyn_run_frames", default=())


def _new_run_id() -> str:
    """Generate a fresh agent_run_id.

    Format: ``run_`` + UUID4 canonical string.
    Comfortably under the 255-char wire cap.
    """
    return f"run_{uuid.uuid4()}"


def current_run() -> tuple[str | None, str | None]:
    """Return the active ``(agent_run_id, agent_run_name)`` or ``(None, None)``.

    Read at metadata-event build time. Returning a tuple (rather than the
    raw ``ContextVar`` value) lets callers unpack without a None-check.
    """
    active = _active_run.get()
    if active is None:
        return (None, None)
    return active


def _name_has_disallowed_char(name: str) -> bool:
    return any(unicodedata.category(char) in _DISALLOWED_NAME_CATEGORIES for char in name)


def _called_from_async_generator() -> bool:
    """Return True when ``run()`` is being entered inside an async generator."""
    frame: FrameType | None = sys._getframe(2)
    try:
        while frame is not None:
            if frame.f_code.co_flags & inspect.CO_ASYNC_GENERATOR:
                return True
            frame = frame.f_back
        return False
    finally:
        del frame


class _RunScope(AbstractContextManager[str], AbstractAsyncContextManager[str]):
    """Context manager returned by ``solwyn.run(name)``.

    Supports both ``with`` and ``async with``. Nesting replaces the outer
    scope for the inner's duration — matches OpenTelemetry span semantics.
    The outer scope is restored automatically on exit via ``ContextVar.reset``.
    """

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"solwyn.run(name) requires str, got {type(name).__name__}")
        if not name.strip():
            raise ValueError("solwyn.run(name) requires a non-empty name")
        if _name_has_disallowed_char(name):
            raise ValueError(
                "solwyn.run(name) must not contain control characters, "
                "format, or line-separator characters"
            )
        if len(name) > AGENT_RUN_NAME_MAX_LENGTH:
            raise ValueError(
                f"solwyn.run(name) exceeds max length {AGENT_RUN_NAME_MAX_LENGTH} (got {len(name)})"
            )
        self._name = name
        self._scope_id = id(self)

    def _enter(self) -> str:
        if _called_from_async_generator():
            raise TypeError(
                "solwyn.run(...) inside async generators is not supported; "
                "open the scope in the consumer or await the generator inside an outer scope"
            )
        run_id = _new_run_id()
        prior_active = _active_run.get()
        token = _active_run.set((run_id, self._name))
        frames = _run_frames.get()
        _run_frames.set(
            (*frames, _RunFrame(scope_id=self._scope_id, token=token, prior_active=prior_active))
        )
        return run_id

    def _exit(self) -> None:
        frames = _run_frames.get()
        if not frames:
            return
        frame = frames[-1]
        if frame.scope_id != self._scope_id:
            raise RuntimeError("solwyn.run scopes must exit in LIFO order")
        try:
            _active_run.reset(frame.token)
        except ValueError:
            # Async generator finalizers can run in a different Context than
            # __aenter__. A token cannot be reset from there, so avoid
            # surfacing a cleanup-time exception. Restore the prior value in
            # the finalizer context as a best effort; entering inside async
            # generators is rejected to prevent contaminating the consumer.
            _active_run.set(frame.prior_active)
        finally:
            _run_frames.set(frames[:-1])

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
    """Submit ``fn`` to an executor with the active ``solwyn.run(...)`` tag preserved.

    ``ThreadPoolExecutor`` does not copy ``ContextVar`` values into worker
    threads. This helper wraps ``fn`` in ``copy_context().run`` so the
    active run id propagates to the worker.

    Returns:
        The executor's ``concurrent.futures.Future[T]``, not an awaitable.
        Call ``.result()`` to wait for the value. In asyncio code, bridge it
        with ``asyncio.wrap_future(future)``.

    Args:
        executor: Any ``concurrent.futures.Executor``.
        fn: Callable to run in a worker.
        *args, **kwargs: Forwarded to ``fn`` through the copied context.

    Note:
        If ``executor.shutdown(cancel_futures=True)`` cancels work before it
        starts, ``future.result()`` raises ``CancelledError``.
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

    Tasks created with ``asyncio.create_task(...)`` inside a scope capture
    that task-local context. Calls made by those tasks after the scope exits
    are still attributed to the captured run id. Use ``asyncio.TaskGroup`` or
    await spawned tasks before exiting when attribution must be bounded.

    ``solwyn.run(...)`` is not supported inside async generators. A scope
    opened before ``yield`` would remain active in the consumer's ``async for``
    body, so the SDK raises at scope entry instead.
    """
    return _RunScope(name)
