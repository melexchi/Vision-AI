---
name: test-integration
description: Write and run integration tests for third-party services
---
Write integration tests for third-party service connections.

1. Read docs/ARCHITECTURE.md to find all third-party integrations
2. For each integration (LiveKit, HuggingFace, Chatterbox TTS, etc.), write tests that:
   - Mock the external API with realistic responses
   - Verify your code correctly handles success responses
   - Verify your code correctly handles error responses (timeout, 500, rate limit, invalid API key)
   - Verify webhook handlers process real payload shapes correctly
   - Verify webhook signature verification rejects invalid signatures
   - Verify retry logic works (if applicable)
   - Verify data is correctly saved after processing
3. Run tests and report results

Format:
[SERVICE] — [N scenarios tested, M passed, K failed]
For failures: [scenario] — [what went wrong]
