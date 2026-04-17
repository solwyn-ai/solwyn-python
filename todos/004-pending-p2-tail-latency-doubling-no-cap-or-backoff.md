---
status: pending
priority: p2
issue_id: "004"
tags: [code-review, performance, reliability, pr-6]
dependencies: []
---

# Fallback retry doubles tail latency with no cap or exception-class gating

Under sustained primary failure, every failing call pays `primary_rtt + fallback_rtt` with no backoff, no jitter, and no budget cap. For streaming LLM calls where primary_rtt may be 30-60s before a terminal error, callers face unbounded worst-case latency.

## Problem Statement

The current retry policy is: "on any `Exception`, try once more with the fallback model, immediately." No retry budget, no exception-class whitelist, no jitter.

Failure modes this creates:
1. **Timeout amplification**: `httpx.ReadTimeout` on a 30s streaming primary → another 30s on the fallback → 60s caller wait.
2. **Non-retriable errors retried anyway**: 401/403 (auth), 422 (schema), 429 (rate limit) — the fallback will almost certainly hit the same error but the caller pays 2x RTT to find out.
3. **Caller-side retries compound**: A user with their own retry loop (e.g., `tenacity`) now sees their effective backoff doubled.

## Findings

- `src/solwyn/client.py:258-286` — primary `.create()` call, no pre-check of exception class.
- `src/solwyn/client.py:282-286` — retry issued immediately, no delay, no cap.
- Performance reviewer flagged as HIGH severity (#1). Security reviewer noted as P2 amplification risk (#6).
- No `retry_budget_ms`, `retry_on_status`, or similar config fields exist on `SolwynConfig`.

## Proposed Solutions

### Option 1: Exception-class whitelist (simple)

**Approach:** Gate retry on exception *type*. Skip retry for auth/schema/rate-limit classes; only retry on network-class errors.

```python
NON_RETRIABLE = (
    httpx.HTTPStatusError,  # 4xx (gated further by status code)
    AuthenticationError,     # provider SDK-specific
)
# Check `getattr(exc, "status_code", None)` in (401, 403, 422, 429)
```

**Pros:**
- Targets the worst cases without touching latency path.
- Composable with later additions.

**Cons:**
- Each provider SDK has its own exception taxonomy — requires per-adapter knowledge.
- Adds coupling to provider SDKs in the base layer.

**Effort:** 2-3 hours

**Risk:** Low-Medium

---

### Option 2: Retry budget cap + time-based skip

**Approach:** Add `retry_budget_ms: int = 20_000` to `SolwynConfig`. Skip retry if `(time.monotonic() - start_time) * 1000 > retry_budget_ms`.

**Pros:**
- Simple to implement and reason about.
- Caps the worst case.

**Cons:**
- Doesn't help when primary fails fast (e.g., 401) — still double-costs the call.

**Effort:** 1-2 hours

**Risk:** Low

---

### Option 3: Both — exception whitelist AND retry budget

**Approach:** Combine options 1 and 2.

**Pros:**
- Covers both fail-slow (timeout) and fail-fast (auth) cases.

**Cons:**
- Two config fields, two code paths, two documentation sections.

**Effort:** 4-5 hours

**Risk:** Low-Medium

---

### Option 4: Accept current behavior, document clearly

**Approach:** Document the latency contract in the README: "configuring `fallback_model` up to 2x your primary timeout budget." No code change.

**Pros:**
- Zero implementation risk.
- Users tune their own timeouts.

**Cons:**
- Pushes the complexity onto every user.
- Doesn't help the "retry a guaranteed-failing 401" waste.

**Effort:** 30 min

**Risk:** Low (but punts the problem)

## Recommended Action

**To be filled during triage.**

## Technical Details

**Affected files:**
- `src/solwyn/config.py` — new config field(s)
- `src/solwyn/_base.py:115-122` — `_should_retry_with_fallback` gains a `primary_exc` parameter or an `elapsed_ms` parameter
- `src/solwyn/client.py:279, 586` — retry gate
- `README.md` — document retry policy

**Related components:**
- `providers/*` adapters — may need to expose a "classify this exception" helper

## Acceptance Criteria

- [ ] Unit test: primary raises AuthenticationError-equivalent → no retry, raises immediately.
- [ ] Unit test: primary raises timeout → retry occurs unless elapsed > budget.
- [ ] README documents latency cap.
- [ ] Existing rescue/re-raise tests still pass.

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Performance reviewer finding:** performance-oracle #1
- **Security reviewer finding:** security-sentinel #6
- **`_base.py:135-136` FUTURE comment** — speaks to Option 1 but is structurally misplaced (see todo 009)

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)

## Notes

- Consider user expectations: many callers wrap the SDK in their own retry loop; compounding with internal retry is often undesirable.
