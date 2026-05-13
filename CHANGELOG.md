# Changelog

## 0.2.0 — 2026-05-12

### Fixed
- **OpenAI Responses-API extractor** now surfaces all 8 token sub-fields, matching
  the Chat Completions extractor. Previously, `audio_input_tokens`,
  `audio_output_tokens`, `accepted_prediction_tokens`, and
  `rejected_prediction_tokens` were silently dropped on Responses API responses.

### Changed (breaking)
- **`TokenDetails.cache_creation_tokens` removed**, replaced by two duration-specific
  fields: `cache_creation_5m_tokens` (Anthropic `ephemeral_5m_input_tokens`) and
  `cache_creation_1h_tokens` (Anthropic `ephemeral_1h_input_tokens`). The Solwyn API
  prices these at the correct rates (1.25× base input for 5m, 2× base for 1h);
  collapsing them into a single bucket lost billing accuracy.
- **Anthropic adapter** now reads cache writes from the structured
  `usage.cache_creation` sub-object (`.ephemeral_5m_input_tokens` and
  `.ephemeral_1h_input_tokens`) instead of the aggregate
  `cache_creation_input_tokens` field.

### Added
- **OpenAI `service_tier`** is now extracted from API responses and forwarded on
  `MetadataEvent`. The Solwyn API stores this on `cost_events` for future per-tier
  repricing; for now, all tiers are priced as standard. Anthropic and Google
  responses send `service_tier=None` (only OpenAI has the concept).

### Compatibility
- Pre-launch release; no preexisting customer SDKs. Pairs with the API in
  `core@feature/token-usage` (PR `solwyn-ai/core#71`) — recommend same-day deploy
  of both PRs.
