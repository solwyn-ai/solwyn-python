"""Tests for ``solwyn.run("name")`` context manager.

The context manager binds an active agent_run_id + agent_run_name to a
contextvar so the metadata-event builder can tag cost events with the
current run. Outside the scope, both fields must be None so the API's
server-side auto-fallback engages.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context

import pytest

import solwyn
from solwyn import _run
from solwyn._run import current_run


@pytest.mark.unit
class TestRunIdGenerator:
    """The id generator must produce stable, prefixed, unique ids."""

    def test_private_generator_exists(self) -> None:
        assert hasattr(_run, "_new_run_id")

    def test_starts_with_run_prefix(self) -> None:
        assert _run._new_run_id().startswith("run_")

    def test_ids_are_unique(self) -> None:
        ids = {_run._new_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_id_fits_wire_max_length(self) -> None:
        # Wire field cap is 255 chars; the id must comfortably fit.
        assert len(_run._new_run_id()) <= 255


@pytest.mark.unit
class TestRunContextManagerSync:
    """Synchronous ``with solwyn.run("name")`` behavior."""

    def test_yields_stable_run_id(self) -> None:
        with solwyn.run("foo") as run_id:
            # Same id throughout the scope.
            assert run_id.startswith("run_")
            assert current_run() == (run_id, "foo")

    def test_outside_scope_returns_none(self) -> None:
        # Before any scope is entered, no run is active.
        assert current_run() == (None, None)

    def test_exit_clears_active_run(self) -> None:
        with solwyn.run("foo"):
            pass
        assert current_run() == (None, None)

    def test_sequential_scopes_have_different_ids(self) -> None:
        # Each entry generates a fresh id — even when the name repeats.
        with solwyn.run("foo") as a:
            pass
        with solwyn.run("foo") as b:
            pass
        assert a != b

    def test_nested_inner_replaces_outer(self) -> None:
        # Documented behavior: inner replaces outer for its duration.
        # Matches OpenTelemetry span semantics.
        with solwyn.run("outer") as outer_id:
            assert current_run() == (outer_id, "outer")
            with solwyn.run("inner") as inner_id:
                assert inner_id != outer_id
                assert current_run() == (inner_id, "inner")
            # Outer is restored after inner exits.
            assert current_run() == (outer_id, "outer")
        assert current_run() == (None, None)

    def test_exception_propagates_and_resets_state(self) -> None:
        with pytest.raises(RuntimeError, match="boom"), solwyn.run("foo"):
            raise RuntimeError("boom")
        # State must be reset even on exception — no leaked run.
        assert current_run() == (None, None)

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"), solwyn.run(""):
            pass

    def test_whitespace_only_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"), solwyn.run("   "):
            pass

    def test_name_max_length_enforced(self) -> None:
        # Wire field cap is 255; reject longer names eagerly so callers
        # find out at scope entry rather than via wire validation later.
        with pytest.raises(ValueError, match="max length"), solwyn.run("x" * 256):
            pass

    @pytest.mark.parametrize("name", ["nightly\nbatch", "nightly\x00batch", "nightly\x7fbatch"])
    def test_control_chars_in_name_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="control characters"), solwyn.run(name):
            pass


@pytest.mark.unit
class TestRunContextManagerAsync:
    """Async ``async with solwyn.run("name")`` behavior.

    The contextvar must isolate concurrent asyncio tasks — each task sees
    only its own run, never the other's.
    """

    @pytest.mark.asyncio
    async def test_async_with_yields_run_id(self) -> None:
        async with solwyn.run("foo") as run_id:
            assert run_id.startswith("run_")
            assert current_run() == (run_id, "foo")
        assert current_run() == (None, None)

    @pytest.mark.asyncio
    async def test_concurrent_tasks_have_independent_runs(self) -> None:
        # Two tasks enter their own scopes simultaneously. The contextvar
        # must isolate them — task A must never see task B's run id and
        # vice versa.
        a_seen: list[tuple[str | None, str | None]] = []
        b_seen: list[tuple[str | None, str | None]] = []

        async def task_a() -> str:
            async with solwyn.run("task-a") as run_id:
                a_seen.append(current_run())
                # Yield to the scheduler so task_b runs in between.
                await asyncio.sleep(0)
                a_seen.append(current_run())
                return run_id

        async def task_b() -> str:
            async with solwyn.run("task-b") as run_id:
                b_seen.append(current_run())
                await asyncio.sleep(0)
                b_seen.append(current_run())
                return run_id

        a_id, b_id = await asyncio.gather(task_a(), task_b())

        # Each task observed only its own run across the await boundary.
        assert a_id != b_id
        assert all(seen == (a_id, "task-a") for seen in a_seen)
        assert all(seen == (b_id, "task-b") for seen in b_seen)


@pytest.mark.unit
class TestRunScopeFailureModes:
    """Adversarial scenarios for scope reuse and abandoned async cleanup."""

    @pytest.mark.asyncio
    async def test_shared_scope_across_tasks_isolated(self) -> None:
        scope = solwyn.run("shared")
        seen: list[tuple[str, tuple[str | None, str | None]]] = []

        async def task(label: str) -> str:
            async with scope as run_id:
                await asyncio.sleep(0)
                seen.append((label, current_run()))
                return run_id

        first_id, second_id = await asyncio.gather(task("first"), task("second"))

        assert first_id != second_id
        assert ("first", (first_id, "shared")) in seen
        assert ("second", (second_id, "shared")) in seen
        assert current_run() == (None, None)

    def test_double_enter_balances_outer_state(self) -> None:
        scope = solwyn.run("x")

        first_id = scope.__enter__()
        second_id = scope.__enter__()
        assert current_run() == (second_id, "x")

        scope.__exit__(None, None, None)
        assert current_run() == (first_id, "x")

        scope.__exit__(None, None, None)
        assert current_run() == (None, None)

    def test_run_in_executor_propagates_context(self) -> None:
        with solwyn.run("batch") as run_id, ThreadPoolExecutor(max_workers=1) as executor:
            future = solwyn.run_in_executor(executor, current_run)

        assert future.result() == (run_id, "batch")

    @pytest.mark.asyncio
    async def test_abandoned_async_generator_close_from_different_context_does_not_raise(
        self,
    ) -> None:
        async def producer():
            async with solwyn.run("gen"):
                for item in range(10):
                    yield item

        try:
            gen = producer()
            async for item in gen:
                if item == 2:
                    break
            close_task = asyncio.create_task(gen.aclose(), context=Context())
            await close_task
        finally:
            _run._active_run.set(None)

        assert close_task.done()


@pytest.mark.unit
class TestRunPublicSurface:
    """The context manager must be exported at the package top level."""

    def test_run_is_exported(self) -> None:
        assert hasattr(solwyn, "run")
        assert callable(solwyn.run)

    def test_run_is_in_dunder_all(self) -> None:
        assert "run" in solwyn.__all__

    def test_run_in_executor_is_exported(self) -> None:
        assert hasattr(solwyn, "run_in_executor")
        assert callable(solwyn.run_in_executor)

    def test_run_in_executor_is_in_dunder_all(self) -> None:
        assert "run_in_executor" in solwyn.__all__
