---
name: test-business
description: Write and run business logic unit tests
---
Write business logic tests for: $ARGUMENTS

If no argument given, identify the top 5 most critical business logic modules by reading docs/ARCHITECTURE.md.

1. Read the source code of the target module(s)
2. Identify all business rules, calculations, state transitions, and permission checks
3. For each business rule, write tests covering:
   - Happy path (normal expected input → expected output)
   - Boundary values (min, max, zero, empty, one-off)
   - Invalid input (wrong type, missing required fields, out-of-range values)
   - State transitions (what happens when status changes from A to B)
   - Permission checks (can user X perform action Y — both allowed and denied)
4. Run tests and report results

Format per module:
[MODULE] — [N rules tested, M passed, K failed]
For failures: [rule] — expected [X], got [Y]
