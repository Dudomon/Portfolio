# ADR 0003 — Tenant isolation with PostgreSQL Row-Level Security

- Status: Accepted
- Date: 2026-10-09

## Context

All tenants share one database. A missing `WHERE tenant_id = ...` in a single query would leak another
company's invoices — the most damaging bug this product can have.

## Decision

Two layers:

1. **Application:** `tenant_id` is taken only from the verified JWT (`tid` claim), never from request input.
   Repositories always filter by it.
2. **Database:** every tenant-owned table has an RLS policy
   `USING (tenant_id = current_setting('app.tenant_id')::uuid)`. The request-scoped session runs
   `SET LOCAL app.tenant_id = :tid` at the start of each transaction. The application role is not the
   table owner and has no `BYPASSRLS`.

Internal endpoints used by the worker receive `tenant_id` in the payload, authenticated by a service token,
and set the same session variable.

## Consequences

- A forgotten filter fails closed (returns nothing) instead of leaking data.
- `SET LOCAL` is transaction-scoped, so pooled connections cannot carry a tenant across requests.
- Migrations and the seed script run as the owner role.
