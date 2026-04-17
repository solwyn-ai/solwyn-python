---
status: pending
priority: p2
issue_id: "002"
tags: [code-review, architecture, contract, pre-launch, pr-6]
dependencies: []
---

# `is_failover` MetadataEvent field semantics changed — rename pre-launch

The field name `is_failover` now means "used `fallback_model` after the primary failed" but the name still reads as provider-level failover. Rename before launch while there are no prod consumers.

## Problem Statement

`MetadataEvent.is_failover` was originally intended to flag provider-level failover (e.g., OpenAI → Anthropic). That feature was never implemented — the removed `fallback_provider` field attested to the original intent — and this PR repurposes the boolean to mean *model* failover.

Same field, silently different meaning. Dashboards or analytics built against the original semantics would now be incorrectly grouping per-model fallbacks with the (never-existed) provider failovers. Per the launch-status memory, there are no prod consumers server-side yet — this is the last opportunity to fix the contract without a version bump.

## Findings

- `src/solwyn/_types.py:75-77` — docstring updated to "used fallback_model after the primary failed". Field name unchanged.
- `README.md:193` — same description updated.
- The name `is_failover` is semantically overloaded if provider-level failover is ever added later.
- Flagged by architecture-strategist #6.

## Proposed Solutions

### Option 1: Rename to `is_model_fallback`

**Approach:** Rename the field in `MetadataEvent`, update all constructions in `client.py` and `_base.py`, update README.

**Pros:**
- Unambiguous; leaves `is_provider_fallback` available if that feature is ever added.
- No prod consumers to break (per MEMORY.md).

**Cons:**
- Requires coordination with the Solwyn API server (this field is reported in the metadata payload).

**Effort:** 1 hour (plus API coordination)

**Risk:** Low

---

### Option 2: Replace with `fallback_kind: FallbackKind | None` enum

**Approach:** Replace boolean with `None | "model" | "provider"` enum. Today only "model" is emitted.

**Pros:**
- Fully generalizes. Adding provider failover later is purely additive.
- Clear intent at the call site.

**Cons:**
- Larger API contract change.
- Serialization change (bool → string).

**Effort:** 2-3 hours

**Risk:** Low-Medium

---

### Option 3: Keep `is_failover`, add `MetadataEvent.schema_version: int = 1`

**Approach:** No rename. Add a schema-version field so future repurposing can bump the version.

**Pros:**
- Zero behavior change now.

**Cons:**
- Kicks the problem down the road; `is_failover` name remains misleading.

**Effort:** 30 min

**Risk:** Low

## Recommended Action

**To be filled during triage.** Option 1 or 2 strongly preferred pre-launch per MEMORY.md.

## Technical Details

**Affected files:**
- `src/solwyn/_types.py:75-77` — field definition
- `src/solwyn/_base.py:47-75` — `_build_metadata_event` signature
- `src/solwyn/client.py:275, 299, 333, 350, 374, 582, 606, 635, 652, 675` — all sites constructing the event
- `README.md:193` — documentation
- `tests/unit/test_fallback_model.py:133, 253` — assertions referencing `e.is_failover`

**Related components:**
- Solwyn cloud API — consumes this field in the metadata payload

## Acceptance Criteria

- [ ] Field rename applied consistently across SDK.
- [ ] API-side coordination confirmed (server accepts the new field name).
- [ ] All existing tests updated and passing.
- [ ] README and type docstrings match.

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Architecture reviewer finding:** architecture-strategist #6
- **Launch status memory:** confirms no prod consumers server-side yet

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)

**Actions:**
- Identified contract ambiguity during architecture review.
- Cross-referenced with launch-status memory showing pre-launch state.

## Notes

- Pre-launch timing is ideal; post-launch this becomes a breaking change.
