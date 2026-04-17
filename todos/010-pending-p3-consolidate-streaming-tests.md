---
status: pending
priority: p3
issue_id: "010"
tags: [code-review, tests, simplicity, pr-6]
dependencies: []
---

# Consolidate or drop redundant streaming fallback tests

Three streaming retry tests exercise the same retry branch; two of them add little coverage beyond the first. Consider cutting ~60 lines.

## Problem Statement

`tests/unit/test_fallback_model.py` has:

- `test_stream_true_retry_returns_wrapped_stream` (line 313-343, 30 lines)
- `test_force_stream_google_retry` (line 345-376, 32 lines)
- `test_async_stream_true_retry_returns_wrapped_stream` (line 382-409, 28 lines)

The retry logic is identical; differences are which SDK method is hit (`chat.completions.create` vs `generate_content_stream`) and sync vs async dispatch, which are already covered by non-streaming tests over all three providers.

## Findings

- Flagged by code-simplicity-reviewer #6.
- `_sync_dispatch` is already covered by non-streaming tests for all three providers' call paths.
- Streaming-retry behavior is *only* unique in that it wraps the response — that's covered by `test_stream_true_retry_returns_wrapped_stream` alone.

Note: this todo depends on todo 006 (add missing coverage). If the streaming test consolidation removes streaming-plus-is_failover coverage that todo 006 adds, keep the richer version.

## Proposed Solutions

### Option 1: Drop `test_force_stream_google_retry` + `test_async_stream_true_retry_returns_wrapped_stream`

**Approach:** Keep only `test_stream_true_retry_returns_wrapped_stream`. Assumes todo 006's streaming `on_complete` assertion is added to this test.

**Pros:**
- ~60 LOC reduction.

**Cons:**
- Loses explicit Google + async stream coverage.

**Effort:** 15 min

**Risk:** Low

---

### Option 2: Parametrize to cover all three stream shapes in one test

**Approach:** One parametrized test over `{sync/stream=True, sync/_force_stream, async/stream=True}`.

**Pros:**
- Preserves coverage; removes duplication.

**Cons:**
- Parametrize over async+sync is awkward.

**Effort:** 1 hour

**Risk:** Low

---

### Option 3: Keep all three

**Approach:** Accept the duplication. Streaming is intricate enough that explicit per-variant tests have documentation value.

**Pros:**
- Zero effort; explicit per-variant coverage.

**Cons:**
- 60 lines of near-duplicate test code.

**Effort:** 0

**Risk:** None

## Recommended Action

**To be filled during triage.** Simplicity reviewer recommends Option 1 paired with the on_complete assertion in todo 006.

## Technical Details

**Affected files:**
- `tests/unit/test_fallback_model.py:309-409`

## Acceptance Criteria

- [ ] Streaming retry coverage preserved for at least one variant.
- [ ] LOC reduction measurable.

## Resources

- **Simplicity reviewer:** code-simplicity-reviewer #6
- **Related:** todo 006 (missing `on_complete` assertion)

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)

## Notes

- Resolve after 006; doing before may remove a test that 006 wants to enrich.
