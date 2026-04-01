---
name: generate-qa-sheet
description: Generate manual test cases for human QA tester
---
Generate a manual QA test sheet for human testers. This covers what automated tests CANNOT check — UX, visual, cross-device, and subjective quality.

1. Read docs/ARCHITECTURE.md for system overview
2. Read CHANGELOG.md for recent changes (focus on ## [Unreleased])
3. Read tasks/todo.md for recently completed tasks
4. Check which automated tests already exist (run the test suite, note what's covered)

Then generate a test document saved to docs/QA-SHEET.md with this EXACT format:

# QA Test Sheet
Generated: [today's date]
Scope: [what changed recently — from changelog]
Automated Coverage: [summary of what automated tests already cover — DO NOT re-test these manually]

## Critical User Flows (Priority 1 — Test First)
| # | Flow | Steps | Expected Result | Status | Tester |
|---|------|-------|-----------------|--------|--------|
| 1 | [flow] | 1. [step] 2. [step] 3. [step] | [expected] | ⬜ | |

## Feature-Specific Tests (Priority 2)
| # | Feature | Steps | Expected Result | Status | Tester |
|---|---------|-------|-----------------|--------|--------|
(tests for recently changed/added features)

## Cross-Device Tests (Priority 3)
| # | Page/Flow | Device | Check | Status | Tester |
|---|-----------|--------|-------|--------|--------|
(key pages on mobile, tablet, desktop)

## Edge Cases (Priority 4)
| # | Scenario | Steps | Expected Result | Status | Tester |
|---|----------|-------|-----------------|--------|--------|
(unusual inputs, race conditions, network failures, empty states)

## Bug Report Section
When you find a bug, fill this in:
| # | Summary | Steps to Reproduce | Expected | Actual | Severity | Screenshot |
|---|---------|-------------------|----------|--------|----------|------------|
| | | | | | | |

After filling bug reports, give this file back to the developer to:
1. Fix the bugs
2. Add regression tests (so automated tests catch it next time)
3. Add lessons to tasks/lessons.md
