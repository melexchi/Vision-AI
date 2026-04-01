---
name: fix-qa-bug
description: Fix a bug found by human QA and add regression test
---
A bug was found during manual QA testing: $ARGUMENTS

1. Understand the bug from the description
2. Find the root cause in the codebase
3. Fix the bug
4. Write an automated regression test that would have caught this bug
5. Run the regression test to confirm it passes with the fix and would have FAILED without it
6. Add a lesson to tasks/lessons.md: what caused this bug and the rule to prevent it
7. Update the QA sheet (docs/QA-SHEET.md) — mark the bug as fixed

This ensures every human-found bug gets permanently covered by automation.
