---
status: pending
priority: p2
issue_id: "008"
tags: [code-review, budget, correctness, pr-6]
dependencies: []
---

# Budget reservation made against primary model, confirmed against fallback model

If the fallback model has materially different pricing from the primary, the reservation amount booked pre-call and the actual amount confirmed post-call will mismatch. Budget accounting drifts.

## Problem Statement

The flow is:
1. `_budget.check_budget(model=primary_model)` — server may create a reservation for estimated tokens priced against *primary* model.
2. Primary fails, retry runs with `fallback_model`.
3. `_budget.confirm_cost(reservation_id, model=fallback_model, token_details)` — server is told the actual cost is for *fallback* model.

If primary = gpt-4o ($5/MTok input) and fallback = gpt-4o-mini ($0.15/MTok input), the reservation was 30x too large. The Solwyn API's reconciliation logic has to either reconcile by reservation_id alone (trusting whichever model is in confirm) or reject the mismatch.

Additionally, on the *fully-failed* path (both primary and retry fail), neither confirm nor release is called — the reservation sits until the server's TTL cleans it up. Depending on TTL length, repeated failures could pile up stale reservations.

## Findings

- `src/solwyn/client.py:215-219` — budget check uses `model` (primary).
- `src/solwyn/client.py:317-322` (streaming) / `:362-363` (non-streaming) — confirm uses the (possibly rebound) `model`.
- `src/solwyn/client.py:525-529, 623-625, 663-664` — async equivalents.
- No reservation-release call exists on the fully-failed path.
- Flagged by kieran-python-reviewer #3.

## Proposed Solutions

### Option 1: Accept drift, document, rely on server-side TTL

**Approach:** Document that rescued-retry reservations may be over/under-estimated and that the confirm call is authoritative. Rely on server to reconcile against the confirmed model, not the reservation model. Server TTL handles fully-failed cases.

**Pros:**
- Zero SDK change.
- Matches the "API owns pricing" rule from CLAUDE.md.

**Cons:**
- Assumes the server correctly handles model-mismatch in reconciliation. Verify before accepting.

**Effort:** 30 min (docs) + server audit

**Risk:** Depends on server behavior

---

### Option 2: Re-check budget against fallback model before retry

**Approach:** On retry, call `_budget.check_budget(model=fallback_model)` again and release the primary reservation. Only proceed with the retry if the fallback check allows.

**Pros:**
- Strictly correct accounting.
- Respects hard-deny semantics on the fallback model too.

**Cons:**
- Adds a synchronous HTTP round-trip on the retry path — more tail latency (worsens todo 004).
- Requires a `release_reservation` API (may not exist today).

**Effort:** 4-6 hours (including server coordination)

**Risk:** Medium

---

### Option 3: Pass both primary and fallback models to the initial check

**Approach:** `check_budget(model=primary, fallback_model=fallback)` — server reserves the max of the two.

**Pros:**
- One round-trip.
- No mismatch.

**Cons:**
- API contract change; over-reserves when fallback isn't used.

**Effort:** 3-4 hours

**Risk:** Low-Medium

## Recommended Action

**To be filled during triage.** Option 1 is likely sufficient pre-launch if server handles mismatches; verify before merge. Add reconciliation/release behavior in a later iteration.

## Technical Details

**Affected files:**
- `src/solwyn/client.py:215-219, 317-322, 362-363` — sync
- `src/solwyn/client.py:525-529, 623-625, 663-664` — async
- `src/solwyn/budget.py` — reservation lifecycle

**Server side:**
- Audit: how does `confirm_cost` handle model changing between `check` and `confirm`?
- Feature needed: reservation release on failure (may already exist via TTL).

## Acceptance Criteria

- [ ] Behavior documented in `budget.py` or inline.
- [ ] Server-side reconciliation confirmed safe against model mismatch.
- [ ] Fully-failed-path reservation release verified (TTL or explicit).

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Python reviewer:** kieran-python-reviewer #3

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)

## Notes

- The CLAUDE.md invariant "API owns pricing" means the SDK can stay simple if the server is robust here. Worth auditing.
