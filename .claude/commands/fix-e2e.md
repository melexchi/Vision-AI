---
name: fix-e2e
description: Fix a failing Playwright E2E test
---
A Playwright E2E test is failing: $ARGUMENTS

1. Run the specific failing test: npx playwright test [test-file] --headed
2. Read the error output and failure screenshot
3. Determine root cause: test bug (selector, timing, assertion) or app bug (feature broken, regression)?
4. If TEST BUG: Fix the test code, re-run, confirm green
5. If APP BUG: Fix the application code, re-run the test, add a lesson to tasks/lessons.md, update CHANGELOG.md
6. Run the full E2E suite to check for regressions
