---
name: test-live
description: Run tests against live server
---
STEP 0 — SSH PRE-CHECK:
Run: ssh [host-alias] "echo connected"
If fails → STOP. Report: "SSH not configured. See docs/SSH_CONFIG.md."

1. Verify application is running (hit health check endpoint)
2. Run checks against the live API:
   a. Hit every endpoint from docs/ARCHITECTURE.md
   b. Verify response shapes match docs/PATTERNS.md
   c. Test auth flow end-to-end
   d. Test file upload if applicable
   e. Measure response times — flag anything >2 seconds
3. Check server resources: disk, memory, CPU, DB connections, PM2/Docker status, Redis, recent error logs, SSL certificate expiry
4. Report: API health, server health, issues found (sorted by severity), recommendations

Non-destructive. No data created, modified, or deleted.
Run after every deployment and weekly as routine monitoring.
