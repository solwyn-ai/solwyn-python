---
status: pending
priority: p3
issue_id: "012"
tags: [code-review, reliability, providers, pr-6]
dependencies: ["004"]
---

# Skip fallback retry on non-retriable exception classes (401/403/422)

Some primary failures (auth, schema validation) will almost certainly reproduce on the fallback model. Retrying them wastes time and (for per-call quotas) budget.

## Problem Statement

Currently `except Exception as primary_exc` at `client.py:261/568` catches everything uniformly. For specific failure classes, the retry is guaranteed-useless:
- **401 Unauthorized** — bad API key; fallback will fail identically.
- **403 Forbidden** — same.
- **422 Unprocessable Entity** — malformed request; same.
- **Schema validation errors** raised client-side before hitting the wire — same.

The `src/solwyn/_base.py:135-136` FUTURE comment explicitly calls this out ("Option B in the plan"). This todo is the actual implementation.

## Findings

- `src/solwyn/client.py:261, 568` — catches all exceptions.
- `_base.py:135-136` — FUTURE signpost.
- Flagged by security-sentinel #6 and performance-oracle #1.
- Each provider SDK has its own exception taxonomy (openai.AuthenticationError, anthropic.APIStatusError, google.api_core exceptions).

## Proposed Solutions

### Option 1: Exception-class whitelist in `_should_retry_with_fallback`

**Approach:** Pass `primary_exc` to `_should_retry_with_fallback`. Return False if the exception matches a known non-retriable class.

**Pros:**
- Direct fix at the gate site.
- Composable with the existing `model != fallback_model` guard.

**Cons:**
- Exception-class checks coupled to provider SDKs at `_base.py` level.

**Effort:** 2-3 hours

**Risk:** Low-Medium

---

### Option 2: Adapter-level `is_retriable(exc)` helper

**Approach:** Each provider adapter exposes `is_retriable(exc: Exception) -> bool`. Base calls it before deciding to retry.

**Pros:**
- Keeps provider-specific knowledge encapsulated.
- Adapters evolve independently as provider SDKs add new error classes.

**Cons:**
- More indirection; new method on every adapter.

**Effort:** 4-5 hours

**Risk:** Low

---

### Option 3: HTTP-status-based (most providers raise HTTPStatusError-like)

**Approach:** Check `getattr(exc, "status_code", None) in (401, 403, 422)`.

**Pros:**
- Provider-agnostic.
- Simple.

**Cons:**
- Relies on duck-typing; some SDKs wrap differently.
- Misses non-HTTP errors (schema validation raised locally).

**Effort:** 1-2 hours

**Risk:** Medium (fragile)

## Recommended Action

**To be filled during triage.** Pair with todo 004. Option 2 is cleanest but Option 1 is cheaper.

## Technical Details

**Affected files:**
- `src/solwyn/_base.py:115-122` — `_should_retry_with_fallback` signature
- `src/solwyn/providers/*.py` — potential new `is_retriable` method
- `src/solwyn/client.py:279, 586` — pass exc to the gate

## Acceptance Criteria

- [ ] Primary raises AuthenticationError → no retry; primary exc propagates.
- [ ] Primary raises network timeout → retry occurs.
- [ ] Primary raises schema error → no retry.
- [ ] Existing rescue/re-raise tests still pass.

## Resources

- **Security reviewer:** security-sentinel #6
- **Performance reviewer:** performance-oracle #1
- **Design hint:** `_base.py:135-136` FUTURE comment

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)
