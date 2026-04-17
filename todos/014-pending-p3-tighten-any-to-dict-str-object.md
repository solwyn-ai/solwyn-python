---
status: pending
priority: p3
issue_id: "014"
tags: [code-review, types, quality, pr-6]
dependencies: []
---

# Finish tightening `Any` → `dict[str, object]` along the retry path

Per the feedback-avoid-any memory, prefer `object`, protocols, or concrete types over `typing.Any`. The new dispatch helpers correctly use `dict[str, object]`, but the surrounding `_intercepted_call` is still `**kwargs: Any`.

## Problem Statement

`_sync_dispatch`/`_async_dispatch` signatures use `dict[str, object]` — good. But:
- `_intercepted_call(*, _force_stream: bool = False, **kwargs: Any)` at `client.py:190, 512`
- `_intercepted_call` constructs `kwargs: dict[str, Any]` implicitly
- The `Any` types propagate through `_prepare_fallback_kwargs`'s argument

Finishing the tightening end-to-end catches typos at the mypy layer and enforces the convention.

## Findings

- `src/solwyn/client.py:190, 512` — `**kwargs: Any`.
- `src/solwyn/_base.py:124` — `kwargs: dict[str, object]` (already good).
- Flagged by kieran-python-reviewer #6.

## Proposed Solutions

### Option 1: Mechanical swap `Any` → `object`

**Approach:** `**kwargs: Any` becomes `**kwargs: object` where possible. May require cast at the specific sites that pass through to the provider SDK.

**Pros:**
- Matches memory-stated convention.
- Mypy catches more accidental misuses.

**Cons:**
- `object` is stricter; forces casts at SDK-interface sites.

**Effort:** 1-2 hours

**Risk:** Low (mypy-guided)

---

### Option 2: Leave as-is

**Approach:** `Any` is pragmatic for SDK-wrapper surface; tightening gains little at the API boundary where the caller's kwargs genuinely have unbounded shape.

**Pros:**
- Zero effort.

**Cons:**
- Memory says to avoid `Any`.

**Effort:** 0

**Risk:** Low

## Recommended Action

**To be filled during triage.**

## Technical Details

**Affected files:**
- `src/solwyn/client.py:190, 512` and the bodies that consume `kwargs`
- Potentially `src/solwyn/_proxies.py` if proxy signatures propagate `Any`

## Acceptance Criteria

- [ ] `mypy --strict src/solwyn/` still clean (or same level as before).
- [ ] No `Any` introduced in any path touched by this PR.

## Resources

- **Python reviewer:** kieran-python-reviewer #6
- **Memory:** `feedback_avoid_any_type.md`

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)
