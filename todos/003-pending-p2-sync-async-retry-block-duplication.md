---
status: pending
priority: p2
issue_id: "003"
tags: [code-review, architecture, maintainability, pr-6]
dependencies: []
---

# ~45-line retry block duplicated between sync and async `_intercepted_call`

The retry logic is structurally identical in sync and async clients. Any future change to the retry policy requires dual maintenance; drift will cause silently-divergent sync/async behavior.

## Problem Statement

`client.py:261-308` (sync) and `client.py:568-614` (async) implement the same state machine:

1. Record primary failure on breaker
2. Build + report primary-error metadata event
3. Gate on `_should_retry_with_fallback(model)`
4. Prepare fallback kwargs
5. Attempt retry in try/except
6. On retry failure: record, report, re-raise primary
7. On retry success: rebind `model`, `kwargs`, `is_failover`, `start_time` and fall through

Only differences: `await` on dispatch, `_sync_dispatch` vs `_async_dispatch`. The repeated ~50 lines violates the CLAUDE.md rule "all business logic in `_base.py` (sans-I/O); client classes are thin I/O wrappers."

## Findings

- Duplication: `src/solwyn/client.py:261-308` vs `src/solwyn/client.py:568-614`.
- Flagged by two reviewers: kieran-python-reviewer #1 (blocking), architecture-strategist #2 (high severity).
- Simplicity reviewer #3 noted that a shared helper is hard because it calls an async dispatcher; recommended partial hoist (just the metadata-event builder).

## Proposed Solutions

### Option 1: Extract `_build_error_event(...)` helper to `_base.py` (partial hoist)

**Approach:** The most-duplicated fragment is the two identical `_build_metadata_event(...)` calls with `input_tokens=0`, `output_tokens=0`, `token_details=None`, `status=CallStatus.ERROR`. Factor into a one-liner on `_SolwynBase`:
```python
def _build_error_event(self, *, model, provider, elapsed_ms, is_failover) -> MetadataEvent:
    return self._build_metadata_event(
        project_id=self._config.project_id,
        model=model, provider=provider,
        input_tokens=0, output_tokens=0, token_details=None,
        latency_ms=elapsed_ms, status=CallStatus.ERROR,
        is_failover=is_failover,
    )
```

**Pros:**
- Cuts each retry block by ~20 lines without restructuring control flow.
- Minimal diff, low risk.

**Cons:**
- Retry control flow still duplicated.

**Effort:** 1 hour

**Risk:** Low

---

### Option 2: Sans-I/O retry-driver generator in `_base.py`

**Approach:** Implement retry as a pure generator on `_SolwynBase` that yields dispatch intents and consumes dispatch outcomes. Both sync and async clients drive it with ~10 lines:
```python
driver = self._retry_driver(model, kwargs)
intent = next(driver)
while intent.op == "dispatch":
    try:
        response = self._sync_dispatch(intent.kwargs, _force_stream=_force_stream)
        intent = driver.send(("ok", response))
    except Exception as exc:
        intent = driver.send(("fail", exc))
```
The driver emits `("breaker", "failure"/"success")`, `("report", event)`, and terminates with either `("result", response, model, kwargs, is_failover, start_time)` or `("raise", exc)`.

**Pros:**
- Eliminates duplication at the control-flow level.
- Drops cleanly into the existing sans-I/O split.
- Extends naturally to a second retry dimension (provider failover, structured backoff).

**Cons:**
- Generator-driver pattern is unfamiliar; adds indirection.
- Bigger change, more tests to write.

**Effort:** 4-6 hours

**Risk:** Medium

---

### Option 3: Accept the duplication

**Approach:** Do nothing. Rely on code review to catch sync/async drift.

**Pros:**
- Zero effort.

**Cons:**
- Drift is the default outcome — every future change to retry needs to be applied twice.
- Violates the stated CLAUDE.md rule.

**Effort:** 0

**Risk:** Medium-High (accumulated over time)

## Recommended Action

**To be filled during triage.** Preferred: Option 1 now, Option 2 when a second retry dimension lands.

## Technical Details

**Affected files:**
- `src/solwyn/client.py:261-308` — sync retry
- `src/solwyn/client.py:568-614` — async retry
- `src/solwyn/_base.py:47-75` — `_build_metadata_event` (helper lives near here)

## Acceptance Criteria

- [ ] Post-fix diff: sync and async retry blocks share the error-event construction.
- [ ] All 17 existing fallback tests pass.
- [ ] `mypy src/solwyn/` clean.

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Python reviewer:** kieran-python-reviewer #1
- **Architecture reviewer:** architecture-strategist #2
- **Simplicity reviewer:** code-simplicity-reviewer #3

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)

**Actions:**
- Compared sync/async blocks line-by-line; confirmed 95% structural identity.

## Notes

- Consider pairing with todo 005 (`_AttemptContext` dataclass) — the state-bag refactor would cooperate well with either option.
