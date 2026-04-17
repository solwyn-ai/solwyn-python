---
status: pending
priority: p2
issue_id: "007"
tags: [code-review, metrics, observability, pr-6]
dependencies: []
---

# `start_time = retry_start` loses primary-attempt latency from the success event

On rescued retry, the reported success event reports only the *retry* latency. The user actually waited for both the primary failure and the retry — that combined latency is what they paid and what the dashboard should show.

## Problem Statement

`src/solwyn/client.py:308` (sync) and `:614` (async) rebind `start_time = retry_start`. All downstream code that computes `elapsed_ms = (time.monotonic() - start_time) * 1000` — specifically the streaming `on_complete` at line 314/619 and non-streaming success at line 358/659 — now reports only the retry duration.

Effect: the reported `latency_ms` for a rescued call understates the real wall-clock wait by the duration of the primary attempt. On a timeout-rescued retry (primary 30s timeout → 1s fallback success), the dashboard shows 1s latency when the user waited 31s.

## Findings

- `src/solwyn/client.py:308` — `start_time = retry_start` — discards primary cost.
- Same issue at `:614` (async).
- Flagged by performance-oracle #7 (LOW severity, observability-only).
- The metadata event for the *failed primary attempt* does get its own `latency_ms` (line 273/580), so the data exists — just isn't correlated with the success event.

## Proposed Solutions

### Option 1: Keep the outer `start_time` for the success event

**Approach:** Don't rebind. Keep the original `start_time` so `elapsed_ms` in the success event reflects user-perceived latency. Add a separate field `retry_latency_ms` on `MetadataEvent` (or leave the primary-failure event to carry the primary latency).

**Pros:**
- `latency_ms` matches what the user waited.
- Dashboards don't need to join events across retries.

**Cons:**
- New `MetadataEvent` field is a schema change.
- Or: loses the ability to see retry-only latency.

**Effort:** 1-2 hours

**Risk:** Low

---

### Option 2: Emit both events (primary failure + success) and compute total server-side

**Approach:** Keep current `start_time = retry_start` rebind. Document that dashboards correlate the two events by `sdk_instance_id + timestamp` to compute total latency.

**Pros:**
- Zero SDK change.
- Server owns the correlation logic.

**Cons:**
- Every dashboard query has to do the join.
- Error-prone.

**Effort:** 30 min (docs only)

**Risk:** Low

---

### Option 3: Report combined latency in success event, drop primary-failure event when rescued

**Approach:** On rescued retry, suppress the primary-failure metadata event; emit only the success event with `latency_ms = retry_start - start_time + retry_elapsed`.

**Pros:**
- Single event per user-facing call.
- Matches the "one call = one event" mental model.

**Cons:**
- Hides that a failover happened at the event level (though `is_failover=True` still flags it).
- Complicates debugging provider failures.

**Effort:** 2-3 hours

**Risk:** Medium

## Recommended Action

**To be filled during triage.**

## Technical Details

**Affected files:**
- `src/solwyn/client.py:308` — sync rebind
- `src/solwyn/client.py:614` — async rebind
- `src/solwyn/_types.py:73` — `latency_ms` field (Option 1 adds a second field)
- Tests that assert `latency_ms`

## Acceptance Criteria

- [ ] Rescued-retry success event reports an accurate user-perceived latency OR clear primary/retry split.
- [ ] Documentation aligned with the chosen option.
- [ ] Existing tests adjusted.

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Performance reviewer:** performance-oracle #7

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)
