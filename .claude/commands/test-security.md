---
name: test-security
description: Write and run automated security tests
---
Write automated security tests for this project.

1. Read CLAUDE.md security rules
2. Write tests for each of the following:
   AUTH:
   - Expired token returns 401, not data
   - Invalid token returns 401
   - Missing token returns 401 on protected routes
   - Admin routes return 403 for non-admin users
   - Rate limiting blocks after N failed login attempts

   DATA:
   - SQL injection payloads in input fields return 400, not 500
   - XSS payloads in text fields are escaped in the response
   - File upload with executable MIME type is rejected
   - File upload exceeding size limit is rejected

   TENANT (if multi-tenant):
   - User A cannot access User B's data by guessing IDs
   - API endpoints don't leak tenant data in error messages

   API:
   - CORS preflight rejects unauthorized origins
   - Error responses never contain stack traces, file paths, or SQL
   - Rate limiting returns 429 after threshold

3. Run tests and report results

Group by category. Mark CRITICAL for any failing security test.
