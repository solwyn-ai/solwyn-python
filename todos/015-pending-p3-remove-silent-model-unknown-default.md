---
status: pending
priority: p3
issue_id: "015"
tags: [code-review, quality, simplicity, pr-6]
dependencies: []
---

# Remove silent `"unknown"` default for `model` kwarg

`model = kwargs.get("model", "unknown")` silently tolerates a missing required kwarg. The underlying SDK will raise anyway; the fallback just muddles the metadata event if it ever gets reported.

## Problem Statement

`src/solwyn/client.py:202` (sync) and `:514` (async): `model = kwargs.get("model", "unknown")`. All three SDK methods (`chat.completions.create`, `messages.create`, `generate_content`) require `model`. Missing-model is a caller bug that will raise at dispatch. The `"unknown"` default has two effects:
1. Metadata events may briefly report `model="unknown"` before the downstream raise cancels them.
2. Any logic downstream that branches on model name sees "unknown" instead of erroring.

## Findings

- Flagged by kieran-python-reviewer #5.
- No legitimate path where `model` is absent.

## Proposed Solutions

### Option 1: Use `kwargs["model"]` — let KeyError raise

**Approach:** Replace `.get("model", "unknown")` with `kwargs["model"]`. Wrap with a clearer error:
```python
try:
    model = kwargs["model"]
except KeyError:
    raise TypeError("model is a required kwarg") from None
```

**Pros:**
- Explicit; matches caller expectation.
- No sentinel magic in metadata events.

**Cons:**
- None.

**Effort:** 15 min

**Risk:** Low

---

### Option 2: Leave as-is

**Approach:** The underlying SDK raises a clearer error anyway; the default is harmless.

**Pros:**
- Zero risk.

**Cons:**
- Sentinel value can leak into logs if the underlying SDK doesn't raise synchronously.

**Effort:** 0

**Risk:** Low

## Recommended Action

**To be filled during triage.**

## Technical Details

**Affected files:**
- `src/solwyn/client.py:202, 514`

## Acceptance Criteria

- [ ] Missing-model call raises a typed error before any metadata is emitted.
- [ ] Existing tests pass (they all pass `model=...`).

## Resources

- **Python reviewer:** kieran-python-reviewer #5

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)
