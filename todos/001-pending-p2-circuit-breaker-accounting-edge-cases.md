---
status: pending
priority: p2
issue_id: "001"
tags: [code-review, quality, circuit-breaker, pr-6]
dependencies: []
---

# Circuit breaker accounting: edge cases under retry (REVISED after validation)

Two real but mild accounting issues under the retry path. **This todo was originally rated P1 but validation against `circuit_breaker.py` showed the main "bricks the client after 3 rescued retries" claim was false.**

## Problem Statement

On validation the originally-flagged bug — "rescued retries never offset the primary's record_failure, so 3 rescued retries trip the breaker" — does **not** apply. `CircuitBreaker.record_success()` in CLOSED state resets `failure_count = 0` (`circuit_breaker.py:82-84`, test confirmed at `test_circuit_breaker.py:29-38`). A rescued retry calls `record_success()` at `client.py:360` (non-streaming) or `client.py:316` (streaming `on_complete`), cleanly offsetting the primary failure.

However, two narrower issues remain:

### A. Fully-failed rescue records 2 failures per user-facing call
When both primary and retry fail, `cb.record_failure()` fires at both line 264 (primary) and line 289 (retry). With default `failure_threshold=3`, the breaker transitions to OPEN after ~2 consecutive fully-failed user-facing calls instead of the intuitive 3. Arguably correct (each retry *is* a real provider call), but it's a behavior change worth a test + doc note.

### B. HALF_OPEN rescue loses the success signal
If state is HALF_OPEN (probing recovery), the primary-failure `record_failure()` at line 264 transitions state → OPEN (`circuit_breaker.py:94-96`). The retry then dispatches without a `can_proceed()` check (`client.py:282-286`). If the retry succeeds, the downstream `record_success()` at line 360/316 runs against state=OPEN — the early-return branches only check HALF_OPEN and CLOSED, so success is silently discarded. Breaker stays OPEN until the next `recovery_timeout` window.

Effect: a user whose fallback model successfully rescued the probe call still has to wait another `recovery_timeout` (60s default) before any traffic flows. Not "client bricked" but unnecessary downtime.

## Findings

- `src/solwyn/circuit_breaker.py:82-84` — CLOSED success resets failure_count. Original blocker-claim invalidated.
- `test_circuit_breaker.py:29-38` — `test_success_resets_failure_count` confirms behavior.
- `src/solwyn/client.py:264, 289` — two record_failure calls per fully-failed call. **Real issue A.**
- `src/solwyn/client.py:282-286` — retry dispatches without `cb.can_proceed()` check. **Enabler of issue B.**
- `src/solwyn/circuit_breaker.py:76-84` — `record_success()` is a no-op when state is OPEN. **Cause of issue B.**

## Proposed Solutions

### Option 1: Gate retry dispatch on `cb.can_proceed()`

**Approach:** Before `_sync_dispatch` retry at line 286 / 593, check `cb.can_proceed()`. If False (state=OPEN after primary pushed it over threshold), skip retry and raise primary. Also fixes issue B partially — retry during HALF_OPEN probing will be gated.

**Pros:**
- Preserves breaker intent (don't hammer a provider in OPEN state).
- Minimal diff.

**Cons:**
- Still doesn't rescue issue B entirely — if retry is allowed in HALF_OPEN, success should count.

**Effort:** 1-2 hours

**Risk:** Low

---

### Option 2: Record one outcome per user-facing call

**Approach:** Defer the primary `record_failure()` until after the retry outcome is known. If retry succeeds, record one `record_success()`. If retry fails, record one `record_failure()`. If no fallback, record `record_failure()`.

**Pros:**
- Clean semantics: one breaker event per user-facing call.
- Fixes both issue A and the HALF_OPEN edge case.

**Cons:**
- Larger diff; reorders metadata-event emission relative to breaker.

**Effort:** 2-3 hours

**Risk:** Low-Medium

---

### Option 3: Accept current behavior, add tests + doc

**Approach:** Document that a fully-failed rescue counts 2 breaker failures and that HALF_OPEN probes of same-provider fallback may be lost. Add tests to pin the behavior.

**Pros:**
- Zero behavior change.

**Cons:**
- Unusual semantics stay unusual.

**Effort:** 30 min

**Risk:** None

## Recommended Action

**To be filled during triage.** Option 1 is the smallest fix that addresses the real issue.

## Technical Details

**Affected files:**
- `src/solwyn/client.py:264, 282-286, 289, 571, 589-593, 596` — retry blocks
- `src/solwyn/circuit_breaker.py` — semantics, no change needed
- `tests/unit/test_fallback_model.py` — breaker-state assertions (see todo 006)

## Acceptance Criteria

- [ ] Test: two consecutive fully-failed calls open the breaker (documents issue A).
- [ ] Test: HALF_OPEN → primary fails → retry succeeds → state is CLOSED (if Option 1 or 2).
- [ ] Test: rescued retry in CLOSED state keeps failure_count=0 (already works; pin it).
- [ ] Existing 17 fallback tests still pass.

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Original reviewer findings:** kieran-python-reviewer #2 (main claim invalidated on review); performance-oracle #3 (issue A valid)
- **CircuitBreaker source:** `src/solwyn/circuit_breaker.py:76-97`

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)

### 2026-04-16 - Validation against codebase

**By:** Claude Code

**Actions:**
- Read `circuit_breaker.py` end-to-end.
- Verified `record_success()` in CLOSED resets failure_count to 0.
- Confirmed with `test_success_resets_failure_count` test.
- Downgraded from P1 to P2; rewrote problem statement to describe the actual (narrower) issues.

**Learnings:**
- Two independent reviewers can converge on a false positive when each reasons locally about `record_failure()` sites without checking `record_success()` semantics.
- "Double record_failure" is not automatically equivalent to "opens in half the calls" once a success path is in the loop.

## Notes

- Previously P1. Downgraded to P2 after validation. Does NOT block merge.
- Consider whether the retry-hammering-OPEN-provider concern (Option 1) is itself worth the change, even in isolation.
