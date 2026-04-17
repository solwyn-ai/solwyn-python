---
status: pending
priority: p3
issue_id: "016"
tags: [code-review, simplicity, pr-6]
dependencies: []
---

# Consider inlining `_prepare_fallback_kwargs` and dropping defensive RuntimeError

The helper is 15 lines (with docstring + RuntimeError guard + FUTURE comment) for a dict copy + key set. Its `RuntimeError` branch is defending against its only caller being rewritten.

## Problem Statement

`src/solwyn/_base.py:124-139`:
```python
def _prepare_fallback_kwargs(self, kwargs: dict[str, object]) -> dict[str, object]:
    if self._config.fallback_model is None:
        raise RuntimeError(
            "fallback_model is not configured — caller should have checked "
            "_should_retry_with_fallback() first"
        )
    # FUTURE: If we ever gate retry by HTTP status code...
    swapped = dict(kwargs)
    swapped["model"] = self._config.fallback_model
    return swapped
```

Call sites already gate on `_should_retry_with_fallback(model)` immediately before, which guarantees `fallback_model is not None`. The `RuntimeError` is defending against a future refactor — YAGNI.

## Findings

- Flagged by code-simplicity-reviewer #1.
- `src/solwyn/client.py:282, 589` — only two call sites; both correctly gated.
- `tests/unit/test_fallback_model.py:40-60` — 21 lines of tests for this helper.

## Proposed Solutions

### Option 1: Inline at call sites

**Approach:** Replace the helper with:
```python
fallback_model = self._config.fallback_model
assert fallback_model is not None  # guarded by _should_retry_with_fallback above
fallback_kwargs = {**kwargs, "model": fallback_model}
```
Note: `assert` will be stripped by `-O`. If the runtime invariant is important, use `if fallback_model is None: raise RuntimeError(...)` per CLAUDE.md's "no asserts" rule — but at that point you're back to the helper.

**Pros:**
- Removes one method + 21 lines of tests.

**Cons:**
- Inline path still needs a runtime check (can't use `assert` per CLAUDE.md). Net LOC change is smaller than it looks.
- Loses the "input not mutated" unit test (but `{**kwargs, ...}` makes that obvious).

**Effort:** 30 min

**Risk:** Low

---

### Option 2: Keep helper, remove defensive `RuntimeError`

**Approach:** Keep the function; drop the early-return/RuntimeError and trust the call-site gate.

**Pros:**
- One helper per sans-I/O concern, centralized.
- Drops the redundant guard.

**Cons:**
- CLAUDE.md says runtime invariants use `raise RuntimeError`, not `assert`. Removing the guard eliminates the invariant; bug would silently set `model=None` in kwargs.

**Effort:** 15 min

**Risk:** Medium (removes a runtime check)

---

### Option 3: Keep as-is

**Approach:** 15 lines is cheap insurance.

**Pros:**
- Zero risk.
- CLAUDE.md-aligned.

**Cons:**
- None.

**Effort:** 0

**Risk:** None

## Recommended Action

**To be filled during triage.** Kept because CLAUDE.md explicitly endorses runtime invariants via RuntimeError. Consider closing this todo as "won't fix."

## Technical Details

**Affected files:**
- `src/solwyn/_base.py:124-139`
- `src/solwyn/client.py:282, 589` (if inlining)
- `tests/unit/test_fallback_model.py:40-60` (if inlining)

## Acceptance Criteria

- [ ] Runtime invariant (non-None fallback_model at retry time) preserved.
- [ ] LOC reduced OR decision documented.

## Resources

- **Simplicity reviewer:** code-simplicity-reviewer #1

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)

## Notes

- This one has a real tension: simplicity vs CLAUDE.md's "no `assert`" rule. The RuntimeError guard IS the correct pattern per house rules; the helper is small; don't inline just to inline.
