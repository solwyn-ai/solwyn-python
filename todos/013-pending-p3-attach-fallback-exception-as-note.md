---
status: pending
priority: p3
issue_id: "013"
tags: [code-review, ergonomics, debugging, pr-6]
dependencies: []
---

# Attach fallback exception to primary via PEP 678 `__notes__` for debugging

On fully-failed retry, `raise primary_exc from None` is correct (callers tuned their error handling for the primary model) but discards the fallback exception's info from the caller's view. Attach it as a note so debugging users can still see both.

## Problem Statement

`src/solwyn/client.py:302` and `:609` do `raise primary_exc from None`, deliberately suppressing the exception chain. This is right — the caller expects their primary-model error handling to fire — but all information about what the *fallback* did is now only in the metadata event, invisible from the caller's perspective.

Users debugging locally (no access to cloud metadata dashboard yet, per MEMORY.md launch status) can't see why the fallback also failed.

## Findings

- `src/solwyn/client.py:302, 609` — `raise primary_exc from None`.
- Flagged by kieran-python-reviewer #4.
- PEP 678 (`add_note`) shipped in Python 3.11; the project's `from __future__ import annotations` suggests 3.11+ compatibility. Verify minimum supported Python.

## Proposed Solutions

### Option 1: Add fallback error as a note

**Approach:** Before `raise primary_exc from None`, call:
```python
primary_exc.add_note(f"fallback_model={fallback_model} also failed: {retry_exc!r}")
```

**Pros:**
- Zero API change.
- `traceback.print_exc()` and most debuggers display notes.
- One line per site.

**Cons:**
- Mutates the exception object in place; if the caller re-raises with additional notes, theirs are appended (fine).
- PEP 678 requires Python 3.11+.

**Effort:** 15 min

**Risk:** Low

---

### Option 2: Log the fallback exception at DEBUG level

**Approach:** `logger.debug("fallback retry also failed", exc_info=retry_exc)`.

**Pros:**
- Invisible in normal operation; available for opted-in debug.

**Cons:**
- Users often don't have debug logging on.

**Effort:** 15 min

**Risk:** Low (careful: never log prompt content; this logs the exception only, which may include provider-side messages — usually safe but verify)

## Recommended Action

**To be filled during triage.** Option 1 if Python 3.11+ is guaranteed.

## Technical Details

**Affected files:**
- `src/solwyn/client.py:302` — sync re-raise
- `src/solwyn/client.py:609` — async re-raise
- `pyproject.toml` — confirm min Python version

## Acceptance Criteria

- [ ] Rescued-retry test: after both fail, primary exception's notes include fallback exception info.
- [ ] Privacy firewall remains green — note content must not include prompt/message text.

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Python reviewer:** kieran-python-reviewer #4
- **PEP 678:** https://peps.python.org/pep-0678/

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)

## Notes

- Careful: the privacy firewall enforces no-prompt-content. An exception note that includes `repr(retry_exc)` should be reviewed — upstream SDK exceptions sometimes include request body fragments.
