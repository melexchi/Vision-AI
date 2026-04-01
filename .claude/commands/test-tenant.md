---
name: test-tenant
description: Write and run tenant isolation tests
---
Write tenant isolation tests for this project. This is CRITICAL for multi-tenant applications.

1. Read docs/ARCHITECTURE.md to understand the database schema and tenant model
2. Read docs/PATTERNS.md to understand how queries are written
3. Use subagents to search ALL database queries in the codebase
4. For each query, verify it filters by tenant_id (or equivalent)
5. Write tests that:
   - Create two test tenants with separate data
   - For every API endpoint that returns data, call it as Tenant A and verify Tenant B's data is NEVER included
   - For every database query, verify the WHERE clause includes tenant_id
   - Test that creating/updating/deleting data for Tenant A never affects Tenant B
   - Test that direct database query manipulation (missing tenant_id) is caught by middleware/ORM hooks
6. Run the tests and report results

If this is NOT a multi-tenant project, report that and skip.

Format: [ENDPOINT/QUERY] → [PASS: tenant isolated] or [FAIL: leaks data — file:line]
