---
status: pending
priority: p3
issue_id: "009"
tags: [code-review, simplicity, comments, pr-6]
dependencies: []
---

# Remove redundant narration and misplaced FUTURE comments

Two specific comments added in this PR are either narrating obvious code or documenting a speculative future feature. Per CLAUDE.md / global rules: comments should explain non-obvious WHY.

## Problem Statement

- `src/solwyn/client.py:304` — `# Retry succeeded — post-processing below uses these variables.` narrates what the next four assignments obviously do.
- `src/solwyn/_base.py:135-136` — `# FUTURE: If we ever gate retry by HTTP status code (Option B in the plan), that check goes in _should_retry_with_fallback, not here.` is a signpost for a speculative feature placed inside the *wrong* function (belongs in `_should_retry_with_fallback`'s docstring if anywhere).

## Findings

- Flagged by code-simplicity-reviewer #5.
- The "# FUTURE:" comment points to `_should_retry_with_fallback` but lives in `_prepare_fallback_kwargs` — it misleads future readers about where the logic belongs.
- The "Retry succeeded" comment precedes `model = fallback_model; kwargs = fallback_kwargs; ...` — assignments that are self-evident.

## Proposed Solutions

### Option 1: Delete both

**Approach:** Remove lines; let code speak.

**Pros:**
- Matches CLAUDE.md style.

**Cons:**
- None.

**Effort:** 5 min

**Risk:** None

---

### Option 2: Move FUTURE comment to correct function docstring, delete narration

**Approach:** Move the FUTURE content into `_should_retry_with_fallback`'s docstring as a paragraph about extensibility. Delete the "Retry succeeded" comment.

**Pros:**
- Preserves the design hint.

**Cons:**
- Still speculative; YAGNI.

**Effort:** 10 min

**Risk:** None

## Recommended Action

**To be filled during triage.** Simplicity reviewer recommended deletion.

## Technical Details

**Affected files:**
- `src/solwyn/client.py:304`
- `src/solwyn/_base.py:135-136`

## Acceptance Criteria

- [ ] Comments either deleted or moved to correct location.
- [ ] No loss of factual information.

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Simplicity reviewer:** code-simplicity-reviewer #5

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)
