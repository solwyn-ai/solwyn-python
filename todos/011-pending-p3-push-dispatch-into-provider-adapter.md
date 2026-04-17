---
status: pending
priority: p3
issue_id: "011"
tags: [code-review, architecture, providers, pr-6]
dependencies: []
---

# Push `_sync_dispatch` / `_async_dispatch` provider routing into adapters

The dispatch helpers on `Solwyn` / `AsyncSolwyn` do a 3-way provider `if` plus a `_force_stream`-is-Google-only runtime check. That invariant belongs in the Google adapter, not the client.

## Problem Statement

`client.py:176-188` (sync) and `498-510` (async) route to `client.chat.completions.create`, `client.messages.create`, `client.models.generate_content`, or `client.models.generate_content_stream` based on `self._detected_provider` and `_force_stream`. Plus a `RuntimeError` if `_force_stream` is used with a non-Google provider.

This is provider-routing logic scattered across the two client classes when the codebase already has a provider adapter layer (`providers/openai.py`, `providers/anthropic.py`, `providers/google.py`) designed exactly for this.

## Findings

- Flagged by architecture-strategist #3 (low severity).
- The `_force_stream`-is-Google-only invariant at `client.py:182-186` / `504-508` is a leaky abstraction — lives where the Google adapter lives would be more cohesive.

## Proposed Solutions

### Option 1: Add `adapter.call_sync(client, kwargs, force_stream)` / `adapter.call_async(...)`

**Approach:** Push dispatch into the provider adapter. Each adapter knows its own call shape.

**Pros:**
- Encapsulates provider knowledge where it belongs.
- Eliminates the runtime `_force_stream` check at the client level.
- `Solwyn._sync_dispatch` becomes `return self._adapter.call_sync(self._client, kwargs, force_stream=_force_stream)`.

**Cons:**
- Each adapter needs a sync + async dispatch method (6 new methods across 3 adapters).
- More plumbing.

**Effort:** 3-4 hours

**Risk:** Low-Medium

---

### Option 2: Leave as-is

**Approach:** The 3-way branch is small and clear.

**Pros:**
- Zero risk.

**Cons:**
- When a 4th provider ever lands, both clients need an update (duplicated).

**Effort:** 0

**Risk:** Low

## Recommended Action

**To be filled during triage.**

## Technical Details

**Affected files:**
- `src/solwyn/client.py:176-188, 498-510`
- `src/solwyn/providers/openai.py`, `anthropic.py`, `google.py` — add dispatch methods

## Acceptance Criteria

- [ ] `Solwyn._sync_dispatch` shrinks to one adapter call.
- [ ] Each adapter has its own sync/async call method.
- [ ] `_force_stream` check lives in Google adapter only.
- [ ] All existing tests pass without modification (interface boundary unchanged).

## Resources

- **Architecture reviewer:** architecture-strategist #3

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)
