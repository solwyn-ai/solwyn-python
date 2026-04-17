---
status: pending
priority: p2
issue_id: "006"
tags: [code-review, tests, quality, pr-6]
dependencies: []
---

# Missing test coverage: circuit breaker state, Anthropic retry, streaming `on_complete` assertion, `spec=` mocks

The 409-line new test file has solid happy/sad-path coverage but misses three specific scenarios the PR's own changes affect, and doesn't follow the project's `spec=` convention.

## Problem Statement

Missing cases directly correspond to actual behavior in the PR:

1. **No test for circuit breaker state after a rescued retry.** This coverage gap is what hid todo 001 (breaker double-counting).
2. **No Anthropic-provider retry test.** `_sync_dispatch` branches on provider (`openai.chat.completions.create` vs `anthropic.messages.create` vs `google.models.generate_content`). Only OpenAI and Google are exercised in the fallback tests.
3. **Streaming tests iterate `list(wrapper)` but don't assert `on_complete` fires the reported metadata event with `is_failover=True`.** The whole point of streaming retry is that the end-of-stream event is tagged as failover — untested.
4. **Mocks don't use `spec=`.** Per `tests/CLAUDE.md`: "All mocks must use `spec=` (catches renamed methods)." `_mock_openai_client_with_failure_then_success` and friends use bare `MagicMock()`.

## Findings

- `tests/unit/test_fallback_model.py:69-91` — bare `MagicMock()` without `spec=`.
- `tests/unit/test_fallback_model.py:68-194, 202-301` — OpenAI-only sync/async retry tests.
- `tests/unit/test_fallback_model.py:313-343, 382-409` — streaming tests iterate the wrapper but don't assert on the reported event payload.
- No test for `_mock_anthropic_client` retry. `providers/anthropic.py` extraction path diverges from OpenAI meaningfully.
- No test asserting `cb.state` or `cb.failure_count` after rescued retry, rescued-twice retries, or fully-failed retries.
- Flagged by kieran-python-reviewer #7.

## Proposed Solutions

### Option 1: Add missing tests, one per gap

**Approach:** Four new tests:

1. `test_breaker_state_after_rescued_retry` — assert `cb.state == CircuitState.CLOSED` after 5 rescued retries; blocks the 001 regression from re-emerging.
2. `test_anthropic_retry_success` — parametrize or copy the OpenAI test with Anthropic client shape and `messages.create`.
3. `test_stream_retry_reports_failover_event` — iterate the wrapper, capture `report` calls, assert the SUCCESS event has `is_failover=True` and `model == fallback_model`.
4. Retrofit existing mocks with `spec=MagicMock(spec=openai.OpenAI)` or equivalent.

**Pros:**
- Targets exact coverage gaps.
- Low complexity.

**Cons:**
- More tests to maintain.

**Effort:** 2-3 hours

**Risk:** Low

---

### Option 2: Parametrize existing tests over providers

**Approach:** Replace OpenAI-specific tests with parametrized versions that run the same assertions against `{OpenAI, Anthropic, Google}` client mocks.

**Pros:**
- Guarantees parity across all three providers; prevents provider-specific regressions.
- Cleaner than 3x duplicated classes.

**Cons:**
- Larger refactor of the test file.

**Effort:** 4-6 hours

**Risk:** Low

## Recommended Action

**To be filled during triage.** Recommend Option 1 at minimum; Option 2 if the test file is going to grow further.

## Technical Details

**Affected files:**
- `tests/unit/test_fallback_model.py` — all new tests
- `tests/unit/conftest.py` — possibly new shared fixtures for Anthropic client mock
- `src/solwyn/circuit_breaker.py` — needed to assert state from tests

## Acceptance Criteria

- [ ] Breaker state test: CLOSED after 5 rescued retries (catches todo 001 regression).
- [ ] Anthropic retry test passes (both sync and async).
- [ ] Streaming retry test asserts reported event has `is_failover=True` and correct model.
- [ ] All mocks use `spec=`.
- [ ] `make test` green, `make check` green.

## Resources

- **PR:** https://github.com/solwyn-ai/solwyn-python/pull/6
- **Python reviewer:** kieran-python-reviewer #7
- **Project convention:** `tests/CLAUDE.md` — "All mocks must use `spec=`"

## Work Log

### 2026-04-16 - Initial Discovery

**By:** Claude Code (`/compound-engineering:ce:review`)
