---
status: pending
priority: p2
issue_id: "005"
tags: [code-review, architecture, maintainability, pr-6]
dependencies: ["003"]
---

# Manual state rebinding on retry success is fragile — introduce `_AttemptContext`

The retry-success path rebinds four separate locals (`model`, `kwargs`, `is_failover`, `start_time`) before falling through to post-processing. Any new per-attempt variable added later must remember to update here too — bug-prone by construction.

## Problem Statement

`client.py:305-308` (sync) and `client.py:611-614` (async):

```python
# Retry succeeded — post-processing below uses these variables.
model = fallback_model
kwargs = fallback_kwargs
is_failover = True
start_time = retry_start
```

Downstream code (streaming `on_complete`, non-streaming metadata event, breaker success) reads these locals. If anyone adds a new per-attempt variable in the future (e.g., `request_id`, `attempt_count`, `selected_provider` if cross-provider fallback ever lands), they must update all four sites (two rebind sites + two metadata-event sites). Miss one and the reported metadata silently diverges from the actual call.

## Findings

- Flagged by architecture-strategist #5 (medium severity).
- Coupled to todo 003 (sync/async duplication): whatever refactor fixes #003 likely improves this too.

## Proposed Solutions

### Option 1: `_AttemptContext` dataclass

**Approach:** Bundle per-attempt state in a `@dataclass(frozen=True, slots=True)` on `_base.py`:

```python
@dataclass(frozen=True, slots=True)
class _AttemptContext:
    model: str
    kwargs: dict[str, object]
    start_time: float
    is_failover: bool
```

At line 258/565, `ctx = _AttemptContext(model=model, kwargs=kwargs, start_time=time.monotonic(), is_failover=is_failover)`. On retry success, replace the whole context atomically: `ctx = replace(ctx, model=fallback_model, kwargs=fallback_kwargs, start_time=retry_start, is_failover=True)`.

**Pros:**
- Atomic state replacement — no forgetting a field.
- Type-checker catches missed fields at the construction site.
- Fits cleanly with sans-I/O style.

**Cons:**
- One more type to understand.
- Minor verbosity at call sites (`ctx.model` vs `model`).

**Effort:** 2-3 hours

**Risk:** Low

---

### Option 2: Inline dict

**Approach:** Use a plain `dict` instead of a dataclass.

**Pros:**
- Zero new type.

**Cons:**
- No type checking; defeats the purpose.

**Effort:** 1 hour

**Risk:** Medium (no compile-time safety)

---

### Option 3: Do nothing

**Approach:** Accept the fragility; rely on code review.

**Pros:**
- Zero effort.

**Cons:**
- The *next* per-attempt variable WILL get missed somewhere.

**Effort:** 0

**Risk:** Medium

## Recommended Action

**To be filled during triage.** Pair with todo 003.

## Technical Details

**Affected files:**
- `src/solwyn/_base.py` — new dataclass
- `src/solwyn/client.py:202, 203, 204, 258, 305-308, 514, 515, 516, 565, 611-614` — construction + rebind + downstream reads

## Acceptance Criteria

- [ ] Rebind is a single `ctx = replace(ctx, ...)` call.
- [ ] Mypy-safe — any new field in `_AttemptContext` forces an explicit decision at each rebind site.
- [ ] Existing tests pass.

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Architecture reviewer:** architecture-strategist #5

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)
