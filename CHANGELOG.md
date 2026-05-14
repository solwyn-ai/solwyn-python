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
  `.ephemeral_1h_input_tokens`). When a non-beta/older response only includes
  the aggregate `cache_creation_input_tokens` field, the SDK attributes it to
  the 5-minute cache bucket.

### Added
- **OpenAI `service_tier`** is now extracted from API responses and forwarded on
  `MetadataEvent`. The Solwyn API stores this on `cost_events` for future per-tier
  repricing; for now, all tiers are priced as standard. Anthropic and Google
  responses send `service_tier=None` (only OpenAI has the concept).

### Compatibility
- Pre-launch release; no preexisting customer SDKs.
- **Deploy order matters.** SDK 0.2.0 events will be rejected with 422 by Solwyn
  API versions older than `core#71`. Deploy the API first, then publish the SDK.
  The reverse direction, old SDK 0.1.x against the new API, continues to work
  because new fields are optional and old SDKs never send them.
- The reporter swallows failed telemetry sends by design, so a mismatched deploy
  produces succeeding LLM calls with missing telemetry.
